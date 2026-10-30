"""Turn a Qwen Edit Studio draft into a ComfyUI API prompt.

The graph mirrors ComfyUI's "image_qwen_image_2_1_image_edit" template, with its subgraph flattened:
refs -> TextEncodeQwenImage21 (+ optional Qwen 3.5 prompt enhancer) -> KSampler -> VAEDecode -> SaveImage.
"""
import math
import os
import random
import re
import time

MAX_REFS = 10
MAX_COUNT = 4
RES_MULTIPLE = 32
# references (image 2+) larger than limit² pixels are shrunk to that budget; 0 = no limit
REF_LIMITS = (1024, 1216, 1536, 2048, 0)
RESIZE_METHOD = "lanczos"  # same filter TextEncodeQwenImage21 uses
OUTPUT_PREFIX = "QwenEdit/QwenEdit"
UPLOAD_SUBFOLDER = "qwenedit"

DEFAULT_UNET = "qwen_image_2.1_int8_convrot.safetensors"
DEFAULT_CLIP = "qwen3vl_8b_int8_convrot.safetensors"
DEFAULT_VAE = "qwen_image_2.1_vae_bf16.safetensors"
DEFAULT_PE = "qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors"
CLIP_TYPE = "qwen_image"

# Official prompt-enhancer sampling values (see the template's "Prompt Enhancer" note)
PE_SAMPLING = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0, "repetition_penalty": 1.0,
               "presence_penalty": 0.0}
_PE_PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pe_system_prompt.md")

# Node ids in the built prompt (the page uses them for progress text)
NODE_UNET, NODE_CACHE, NODE_CLIP, NODE_VAE = "unet", "cache", "clip", "vae"
NODE_PE_CLIP, NODE_PE_BATCH, NODE_PE, NODE_PE_TEXT = "pe_clip", "pe_batch", "pe", "pe_text"
NODE_ENCODE, NODE_LATENT, NODE_SAMPLER, NODE_DECODE, NODE_SAVE = "encode", "latent", "sampler", "decode", "save"
NODE_TITLES = {
    NODE_UNET: "Load model", NODE_CACHE: "Qwen cache", NODE_CLIP: "Load text encoder", NODE_VAE: "Load VAE",
    NODE_PE_CLIP: "Load prompt enhancer", NODE_PE_BATCH: "Batch refs", NODE_PE: "Prompt enhancer",
    NODE_PE_TEXT: "Enhanced prompt", NODE_ENCODE: "Encode prompt + refs", NODE_LATENT: "Empty latent",
    NODE_SAMPLER: "KSampler", NODE_DECODE: "VAE decode", NODE_SAVE: "Save image",
}

DEFAULT_DRAFT = {
    "prompt": "",
    "negative": "",
    "refs": [],
    "resolution": 0,          # total pixel budget for the refs (side of a square); 0 = keep each ref's size
    "refLimit": 1024,         # references are shrunk to about refLimit² pixels when larger; 0 = no limit
    "customSize": False,      # False: canvas follows image 1; True: width x height below
    "width": 1024,
    "height": 1024,
    "steps": 25,
    "cfg": 1.0,
    "sampler": "euler",
    "scheduler": "simple",
    "seed": 0,
    "seedRandom": True,
    "count": 1,               # images per click, one job each with its own seed
    "unetName": "",           # "" = the DEFAULT_* file above
    "clipName": "",
    "vaeName": "",
    "refine": False,          # run the Qwen 3.5 prompt enhancer first
    "thinking": True,
    "maxLength": 4096,
    "peName": "",
    "cacheDevice": "auto",
    "cacheDtype": "default",
}

_AT_TAG = re.compile(r"@image_?(\d+)", re.IGNORECASE)
_IMAGE_TAG = re.compile(r"<image\s*(\d+)>", re.IGNORECASE)


class DraftError(ValueError):
    pass


def random_seed():
    return random.randrange(2 ** 48)


def snap(v):
    return max(RES_MULTIPLE, round(int(v) / RES_MULTIPLE) * RES_MULTIPLE)


def budget_size(w, h, side):
    """Size for about side x side pixels at w:h, in multiples of 32 (TextEncodeQwenImage21's own formula)."""
    ratio = w / h
    return (max(RES_MULTIPLE, round(math.sqrt(side * side * ratio) / RES_MULTIPLE) * RES_MULTIPLE),
            max(RES_MULTIPLE, round(math.sqrt(side * side / ratio) / RES_MULTIPLE) * RES_MULTIPLE))


def target_size(w, h, resolution, ref_limit=0):
    """Size an image is fed to the model at. resolution 0 keeps it (rounded to 32); ref_limit caps references."""
    size = budget_size(w, h, resolution) if resolution else (max(RES_MULTIPLE, round(w / RES_MULTIPLE) * RES_MULTIPLE),
                                                             max(RES_MULTIPLE, round(h / RES_MULTIPLE) * RES_MULTIPLE))
    if ref_limit and size[0] * size[1] > ref_limit * ref_limit:
        size = budget_size(w, h, ref_limit)
    return size


def prompt_for_model(text):
    """The model understands <image1>; @image_1 is accepted as shorthand."""
    return _AT_TAG.sub(lambda m: f"<image{m.group(1)}>", text)


def ref_path(ref):
    """LoadImage path for a ref. Output images are used in place through ComfyUI's "[output]" annotation."""
    path = f"{ref['subfolder']}/{ref['name']}" if ref.get("subfolder") else ref["name"]
    return f"{path} [output]" if ref.get("type") == "output" else path


def _check_range(draft, key, lo, hi, label):
    try:
        v = float(draft.get(key))
    except (TypeError, ValueError):
        raise DraftError(f"{label} không hợp lệ.")
    if not lo <= v <= hi:
        raise DraftError(f"{label} phải trong khoảng {lo}–{hi}.")


def _check_choice(value, allowed, label):
    if allowed and value not in allowed:
        raise DraftError(f"{label} không hợp lệ: {value}")


def validate(draft, choices):
    """choices: {"samplers", "schedulers", "cacheDevices", "cacheDtypes"} read from the installed nodes."""
    if not str(draft.get("prompt", "")).strip():
        raise DraftError("Prompt đang trống.")
    refs = draft.get("refs") or []
    if not refs:
        raise DraftError("Cần ít nhất 1 ảnh (ảnh 1 là ảnh được sửa).")
    if len(refs) > MAX_REFS:
        raise DraftError(f"Tối đa {MAX_REFS} ảnh tham chiếu.")
    if any(r.get("uploading") or not r.get("name") for r in refs):
        raise DraftError("Có ảnh chưa tải lên xong.")
    used = {int(n) for n in _IMAGE_TAG.findall(prompt_for_model(draft["prompt"]))}
    bad = sorted(n for n in used if n < 1 or n > len(refs))
    if bad:
        raise DraftError("Prompt nhắc tới ảnh không tồn tại: " + ", ".join(f"<image{n}>" for n in bad))
    _check_range(draft, "resolution", 0, 4096, "Độ phân giải ref")
    _check_choice(int(draft.get("refLimit", 0)), REF_LIMITS, "Giới hạn ảnh tham chiếu")
    _check_range(draft, "steps", 1, 200, "Steps")
    _check_range(draft, "cfg", 0, 20, "CFG")
    _check_range(draft, "count", 1, MAX_COUNT, "Số ảnh")
    _check_range(draft, "maxLength", 256, 32768, "Max length")
    if draft.get("customSize"):
        _check_range(draft, "width", 256, 4096, "Chiều rộng")
        _check_range(draft, "height", 256, 4096, "Chiều cao")
    _check_choice(draft.get("sampler"), choices.get("samplers"), "Sampler")
    _check_choice(draft.get("scheduler"), choices.get("schedulers"), "Scheduler")
    _check_choice(draft.get("cacheDevice"), choices.get("cacheDevices"), "Cache device")
    _check_choice(draft.get("cacheDtype"), choices.get("cacheDtypes"), "Cache dtype")


def _pe_system_prompt():
    with open(_PE_PROMPT_FILE, encoding="utf-8") as f:
        return f.read()


def _node(class_type, inputs, node_id):
    return {"class_type": class_type, "inputs": inputs, "_meta": {"title": NODE_TITLES.get(node_id, node_id)}}


def _add_enhancer(p, draft, images, prompt, seed):
    """Rewrite the instruction with the Qwen 3.5 prompt enhancer. Returns the link to the rewritten text."""
    p[NODE_PE_CLIP] = _node("CLIPLoader", {"clip_name": draft.get("peName") or DEFAULT_PE, "type": CLIP_TYPE,
                                           "device": "default"}, NODE_PE_CLIP)
    p[NODE_PE_BATCH] = _node("BatchImagesNode", {f"images.image{i}": link for i, link in enumerate(images)},
                             NODE_PE_BATCH)
    sampling = {f"sampling_mode.{k}": v for k, v in PE_SAMPLING.items()}
    p[NODE_PE] = _node("TextGenerate", {
        "clip": [NODE_PE_CLIP, 0], "image": [NODE_PE_BATCH, 0], "prompt": prompt,
        "system_prompt": _pe_system_prompt(), "max_length": int(draft["maxLength"]),
        "sampling_mode": "on", **sampling, "sampling_mode.seed": int(seed),
        "thinking": bool(draft.get("thinking")), "use_default_template": True, "mtp": "auto",
    }, NODE_PE)
    # PreviewAny is an output node: its text lands in the job's history, so the page can show the rewrite
    p[NODE_PE_TEXT] = _node("PreviewAny", {"source": [NODE_PE, 0]}, NODE_PE_TEXT)
    return [NODE_PE_TEXT, 0]


def _add_ref(p, i, ref, size, resolution, ref_limit):
    """LoadImage for ref i (0-based), plus an ImageScale when its model size differs. Returns the image link."""
    rid = f"ref{i + 1}"
    p[rid] = {"class_type": "LoadImage", "inputs": {"image": ref_path(ref)}, "_meta": {"title": f"Image {i + 1}"}}
    w, h = target_size(*size, resolution, ref_limit if i > 0 else 0)
    if (w, h) == tuple(size):
        return [rid, 0]
    p[f"{rid}_scale"] = {"class_type": "ImageScale", "_meta": {"title": f"Resize image {i + 1}"},
                         "inputs": {"image": [rid, 0], "upscale_method": RESIZE_METHOD, "width": w, "height": h,
                                    "crop": "disabled"}}
    return [f"{rid}_scale", 0]


def build(draft, seed, choices, sizes):
    """sizes: (width, height) of each ref file, in draft order."""
    validate(draft, choices)
    if len(sizes) != len(draft["refs"]):
        raise DraftError("Không đọc được kích thước ảnh.")
    prompt = prompt_for_model(draft["prompt"])
    p = {
        NODE_UNET: _node("UNETLoader", {"unet_name": draft.get("unetName") or DEFAULT_UNET,
                                        "weight_dtype": "default"}, NODE_UNET),
        NODE_CACHE: _node("QwenImage21Cache", {"model": [NODE_UNET, 0], "device": draft["cacheDevice"],
                                               "dtype": draft["cacheDtype"]}, NODE_CACHE),
        NODE_CLIP: _node("CLIPLoader", {"clip_name": draft.get("clipName") or DEFAULT_CLIP, "type": CLIP_TYPE,
                                        "device": "default"}, NODE_CLIP),
        NODE_VAE: _node("VAELoader", {"vae_name": draft.get("vaeName") or DEFAULT_VAE}, NODE_VAE),
    }
    resolution = snap(draft["resolution"]) if int(draft["resolution"]) else 0
    ref_limit = int(draft.get("refLimit", 0))
    images = [_add_ref(p, i, r, sizes[i], resolution, ref_limit) for i, r in enumerate(draft["refs"])]
    text = _add_enhancer(p, draft, images, prompt, seed) if draft.get("refine") else prompt
    # every image arrives already sized (image 1 by the resolution setting, references also capped by refLimit),
    # so the encoder's own resize is off: 0 only rounds to multiples of 32, which they already are
    p[NODE_ENCODE] = _node("TextEncodeQwenImage21", {
        "clip": [NODE_CLIP, 0], "vae": [NODE_VAE, 0], "prompt": text, "negative_prompt": draft.get("negative", ""),
        "resolution": 0, **{f"images.image_{i + 1}": link for i, link in enumerate(images)},
    }, NODE_ENCODE)
    if draft.get("customSize"):
        p[NODE_LATENT] = _node("EmptyLatentImage", {"width": snap(draft["width"]), "height": snap(draft["height"]),
                                                    "batch_size": 1}, NODE_LATENT)
        latent = [NODE_LATENT, 0]
    else:
        latent = [NODE_ENCODE, 2]
    p[NODE_SAMPLER] = _node("KSampler", {
        "model": [NODE_CACHE, 0], "positive": [NODE_ENCODE, 0], "negative": [NODE_ENCODE, 1], "latent_image": latent,
        "seed": int(seed), "steps": int(draft["steps"]), "cfg": float(draft["cfg"]),
        "sampler_name": draft["sampler"], "scheduler": draft["scheduler"], "denoise": 1.0,
    }, NODE_SAMPLER)
    p[NODE_DECODE] = _node("VAEDecode", {"samples": [NODE_SAMPLER, 0], "vae": [NODE_VAE, 0]}, NODE_DECODE)
    # SaveImage numbers files from the highest one on disk, so deleting results reuses old names and the browser
    # shows its cached copy of the old image: a timestamp in the prefix keeps every name new
    prefix = f"{OUTPUT_PREFIX}_{time.strftime('%Y%m%d-%H%M%S')}"
    p[NODE_SAVE] = _node("SaveImage", {"images": [NODE_DECODE, 0], "filename_prefix": prefix}, NODE_SAVE)
    return p


def summary(draft, seed, sizes):
    resolution = snap(draft["resolution"]) if int(draft["resolution"]) else 0
    ref_limit = int(draft.get("refLimit", 0))
    out = {
        "inputSize": list(target_size(*sizes[0], resolution)),
        "refLimit": ref_limit,
        "scaledRefs": sum(1 for s in sizes[1:] if target_size(*s, resolution, ref_limit) != target_size(*s, resolution)),
        "seed": seed, "steps": int(draft["steps"]), "cfg": float(draft["cfg"]), "sampler": draft["sampler"],
        "scheduler": draft["scheduler"], "resolution": int(draft["resolution"]), "refine": bool(draft.get("refine")),
        "refs": [{k: r[k] for k in ("name", "subfolder", "type", "label") if k in r} for r in draft["refs"]],
    }
    if draft.get("customSize"):
        out["size"] = [snap(draft["width"]), snap(draft["height"])]
    for key in ("unetName", "clipName", "vaeName"):
        if draft.get(key):
            out[key] = draft[key]
    if draft.get("negative") and float(draft["cfg"]) > 1:
        out["negative"] = draft["negative"]
    return out
