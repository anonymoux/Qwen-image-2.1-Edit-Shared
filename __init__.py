"""Qwen Edit Studio: a simple web page for the Qwen Image 2.1 image edit workflow, served at /qwenedit.

Same design as H3 Studio: the draft (prompt, images, settings) and job list live on the server in
user/qwenedit_state.json, and every change is pushed to open pages over ComfyUI's websocket as a
"qwenedit.state" event.

API
  GET    /qwenedit/options            -> model lists, sampler/scheduler/cache choices, missing models
  GET    /qwenedit/state              -> {rev, draft, jobs}  (job statuses refreshed)
  PUT    /qwenedit/state              body {draft?: {...partial}, source?}  -> state
  POST   /qwenedit/queue              body {source?}  queue the current draft (draft.count jobs) -> {jobs}
  POST   /qwenedit/jobs/{id}/cancel   interrupt / dequeue a job
  DELETE /qwenedit/jobs/{id}          hide a job from the list (file is kept)
  POST   /qwenedit/jobs/{id}/delete   remove a finished job and delete its image files from disk
"""
import asyncio
import json
import logging
import os
import time

import aiohttp
from aiohttp import web
from PIL import Image

import folder_paths
import nodes
from server import PromptServer

from . import builder

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_HERE = os.path.dirname(os.path.abspath(__file__))
_INDEX = os.path.join(_HERE, "web", "index.html")
_STATE_FILE = os.path.join(folder_paths.get_user_directory(), "qwenedit_state.json")
_MAX_JOBS = 80
_ACTIVE = ("queued", "running")
_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")
_EVENT = "qwenedit.state"
_HF = "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main"
# (folder, file, size) of the models the default draft needs
_REQUIRED_MODELS = (
    ("diffusion_models", builder.DEFAULT_UNET, "6.8 GB"),
    ("text_encoders", builder.DEFAULT_CLIP, "8.7 GB"),
    ("vae", builder.DEFAULT_VAE, "0.6 GB"),
    ("text_encoders", builder.DEFAULT_PE, "8.8 GB"),
)

log = logging.getLogger("QwenEditStudio")
routes = PromptServer.instance.routes
_lock = asyncio.Lock()


# ---------- state store ----------
def _load_state():
    try:
        with open(_STATE_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    except (OSError, ValueError) as e:
        log.error("Cannot read %s: %s", _STATE_FILE, e)
        data = {}
    draft = {**builder.DEFAULT_DRAFT, **(data.get("draft") or {})}
    if not draft.get("seed"):
        draft["seed"] = builder.random_seed()
    return {"rev": int(data.get("rev", 0)), "draft": draft, "jobs": list(data.get("jobs") or [])}


def _save_state(state):
    tmp = _STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    os.replace(tmp, _STATE_FILE)


_state = _load_state()


def _commit(source):
    """Persist the in-memory state and notify every open page. Caller holds _lock."""
    global _state
    _state = {**_state, "rev": _state["rev"] + 1, "jobs": _state["jobs"][:_MAX_JOBS]}
    _save_state(_state)
    PromptServer.instance.send_sync(_EVENT, {"rev": _state["rev"], "source": source})


def _replace_job(job_id, patch):
    global _state
    _state = {**_state, "jobs": [{**j, **patch} if j.get("id") == job_id else j for j in _state["jobs"]]}


# ---------- job status ----------
def _images(outputs):
    found = []
    for out in (outputs or {}).values():
        for f in out.get("images") or []:
            if isinstance(f, dict) and str(f.get("filename", "")).lower().endswith(_IMAGE_EXT):
                found.append({"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")})
    return found


def _enhanced_prompt(outputs):
    text = ((outputs or {}).get(builder.NODE_PE_TEXT) or {}).get("text") or []
    return text[0] if text and isinstance(text[0], str) else None


def _status_from_history(entry):
    status = entry.get("status") or {}
    messages = status.get("messages") or []
    # in-memory history keeps (event, data) tuples; the JSON API turns them into lists
    kinds = {m[0]: m[1] for m in messages if isinstance(m, (list, tuple)) and len(m) == 2}
    # ComfyUI stamps each execution message with epoch milliseconds
    end = next((kinds[k] for k in ("execution_success", "execution_error", "execution_interrupted") if k in kinds), {})
    times = {"startedAt": (kinds.get("execution_start") or {}).get("timestamp"), "finishedAt": end.get("timestamp")}
    times = {k: v for k, v in times.items() if v}
    if "execution_interrupted" in kinds:
        return {"status": "cancelled", **times}
    if status.get("status_str") == "error" or "execution_error" in kinds:
        err = kinds.get("execution_error") or {}
        return {"status": "error", **times,
                "error": f"{err.get('node_type', '')}: {err.get('exception_message', 'Lỗi khi chạy workflow.')}".strip()}
    outputs = entry.get("outputs")
    enhanced = _enhanced_prompt(outputs)
    return {"status": "done", "images": _images(outputs), **times, **({"enhanced": enhanced} if enhanced else {})}


def _refresh_jobs():
    """Sync active jobs with ComfyUI's queue/history. Returns True if anything changed. Caller holds _lock."""
    active = [j for j in _state["jobs"] if j.get("status") in _ACTIVE]
    if not active:
        return False
    queue = PromptServer.instance.prompt_queue
    running, pending = queue.get_current_queue_volatile()
    running_ids = {item[1] for item in running}
    pending_ids = {item[1] for item in pending}
    changed = False
    for job in active:
        jid = job["id"]
        if jid in running_ids:
            patch = {"status": "running"}
        elif jid in pending_ids:
            patch = {"status": "queued"}
        else:
            entry = queue.get_history(prompt_id=jid).get(jid)
            patch = _status_from_history(entry) if entry else {
                "status": "error", "error": "Không tìm thấy job trong ComfyUI (có thể ComfyUI đã khởi động lại)."}
        if any(job.get(k) != v for k, v in patch.items()):
            _replace_job(jid, patch)
            changed = True
    return changed


# ---------- helpers ----------
def _model_list(folder, contains=""):
    try:
        names = folder_paths.get_filename_list(folder)
    except KeyError:
        return []
    return sorted(n for n in names if contains in n.lower())


def _combo_options(class_type, name):
    """Options of a combo input, for both classic ([...],) and V3 ("COMBO", {"options": [...]}) node schemas."""
    cls = nodes.NODE_CLASS_MAPPINGS.get(class_type)
    try:
        spec = cls.INPUT_TYPES()["required"][name]
        return list(spec[1]["options"]) if isinstance(spec[0], str) else list(spec[0])
    except Exception:  # node missing or its schema changed: skip validation rather than fail the page
        return []


def _choices():
    return {
        "samplers": _combo_options("KSampler", "sampler_name"),
        "schedulers": _combo_options("KSampler", "scheduler"),
        "cacheDevices": _combo_options("QwenImage21Cache", "device"),
        "cacheDtypes": _combo_options("QwenImage21Cache", "dtype"),
    }


def _missing_models():
    return [{"folder": folder, "file": name, "size": size, "url": f"{_HF}/{folder}/{name}"}
            for folder, name, size in _REQUIRED_MODELS if name not in _model_list(folder)]


def _check_models(draft):
    """Name the model files this draft needs but are not installed, instead of ComfyUI's long validation dump."""
    needed = [("diffusion_models", draft.get("unetName") or builder.DEFAULT_UNET),
              ("text_encoders", draft.get("clipName") or builder.DEFAULT_CLIP),
              ("vae", draft.get("vaeName") or builder.DEFAULT_VAE)]
    if draft.get("refine"):
        needed.append(("text_encoders", draft.get("peName") or builder.DEFAULT_PE))
    missing = [f"{folder}/{name}" for folder, name in needed if name not in _model_list(folder)]
    if missing:
        raise builder.DraftError("Chưa có model (tải vào ComfyUI/models/ rồi khởi động lại ComfyUI):\n" + "\n".join(missing))


_EXIF_ORIENTATION = 0x0112
_ROTATED = (5, 6, 7, 8)  # EXIF orientations that LoadImage turns sideways, swapping width and height


def _image_sizes(refs):
    """(width, height) of each ref as LoadImage will see it. Reads headers only."""
    sizes = []
    for r in refs:
        try:
            with Image.open(folder_paths.get_annotated_filepath(builder.ref_path(r))) as im:
                w, h = im.size
                if im.getexif().get(_EXIF_ORIENTATION) in _ROTATED:
                    w, h = h, w
        except (OSError, ValueError) as e:
            raise builder.DraftError(f"Không đọc được ảnh {r.get('label') or r.get('name')}: {e}")
        sizes.append((w, h))
    return sizes


def _local_base_url():
    port = getattr(PromptServer.instance, "port", 8188)
    return f"http://127.0.0.1:{port}"


async def _json_body(request):
    try:
        body = await request.json()
    except (ValueError, aiohttp.ContentTypeError):
        raise web.HTTPBadRequest(text=json.dumps({"error": "Body phải là JSON."}), content_type="application/json")
    if not isinstance(body, dict):
        raise web.HTTPBadRequest(text=json.dumps({"error": "Body phải là object JSON."}), content_type="application/json")
    return body


def _error(message, status=400):
    return web.json_response({"error": message}, status=status)


_DRAFT_KEYS = set(builder.DEFAULT_DRAFT)
_REF_KEYS = ("id", "name", "subfolder", "type", "label")


def _clean_draft_patch(patch):
    unknown = set(patch) - _DRAFT_KEYS
    if unknown:
        raise builder.DraftError("Trường không hợp lệ: " + ", ".join(sorted(unknown)))
    if "refs" in patch:
        refs = patch["refs"]
        if not isinstance(refs, list) or len(refs) > builder.MAX_REFS:
            raise builder.DraftError(f"refs phải là danh sách tối đa {builder.MAX_REFS} ảnh.")
        clean = [{k: r[k] for k in _REF_KEYS if k in r} for r in refs if isinstance(r, dict) and r.get("name")]
        if any(r.get("type", "input") not in ("input", "output") for r in clean):
            raise builder.DraftError("Ảnh chỉ được lấy từ thư mục input hoặc output.")
        patch = {**patch, "refs": clean}
    return patch


# ---------- routes ----------
@routes.get("/qwenedit")
async def qwenedit_page(request):
    return web.FileResponse(_INDEX, headers={"Cache-Control": "no-store"})


@routes.get("/qwenedit/options")
async def qwenedit_options(request):
    return web.json_response({
        # Qwen Image 2.1 files only: the older Qwen-Image VAE and encoders don't fit this model
        "unets": _model_list("diffusion_models", "qwen_image_2"),
        "clips": _model_list("text_encoders", "qwen3vl_8b"),
        "peModels": [n for n in _model_list("text_encoders", "qwen") if "_pe_" in n],
        "vaes": _model_list("vae", "qwen_image_2"),
        "defaults": {"unetName": builder.DEFAULT_UNET, "clipName": builder.DEFAULT_CLIP,
                     "vaeName": builder.DEFAULT_VAE, "peName": builder.DEFAULT_PE},
        "missing": _missing_models(),
        "hfBase": _HF,
        "titles": builder.NODE_TITLES,
        "maxRefs": builder.MAX_REFS,
        "maxCount": builder.MAX_COUNT,
        "refLimits": list(builder.REF_LIMITS),
        "uploadSubfolder": builder.UPLOAD_SUBFOLDER,
        **_choices(),
    })


@routes.get("/qwenedit/state")
async def qwenedit_get_state(request):
    async with _lock:
        if _refresh_jobs():
            _commit("server")
        return web.json_response(_state)


@routes.put("/qwenedit/state")
async def qwenedit_put_state(request):
    global _state
    body = await _json_body(request)
    async with _lock:
        try:
            patch = _clean_draft_patch(body.get("draft") or {})
        except builder.DraftError as e:
            return _error(str(e))
        _state = {**_state, "draft": {**_state["draft"], **patch}}
        _commit(str(body.get("source") or "api"))
        return web.json_response(_state)


@routes.post("/qwenedit/queue")
async def qwenedit_queue(request):
    global _state
    body = await _json_body(request) if request.can_read_body else {}
    source = str(body.get("source") or "api")
    async with _lock:
        draft = _state["draft"]
        count = max(1, min(builder.MAX_COUNT, int(draft.get("count") or 1)))
        first = builder.random_seed() if draft.get("seedRandom") else int(draft.get("seed") or 0)
        # a fixed seed with several images steps the seed, so each image differs but stays reproducible
        seeds = [first] + [builder.random_seed() if draft.get("seedRandom") else first + i for i in range(1, count)]
        try:
            builder.validate(draft, _choices())
            _check_models(draft)
            sizes = _image_sizes(draft["refs"])
        except builder.DraftError as e:
            return _error(str(e))
        new_jobs = []
        for seed in seeds:
            try:
                prompt = builder.build(draft, seed, _choices(), sizes)
            except builder.DraftError as e:
                return _error(str(e))
            except (OSError, ValueError, KeyError) as e:
                return _error(f"Không dựng được workflow: {e}", 500)
            result = await _submit_prompt(prompt)
            if isinstance(result, web.Response):
                if not new_jobs:
                    return result
                break  # keep the jobs ComfyUI already accepted
            new_jobs.append({"id": result["prompt_id"], "created": int(time.time() * 1000), "status": "queued",
                             "summary": builder.summary(draft, seed, sizes), "prompt": draft["prompt"]})
        _state = {**_state, "draft": {**draft, "seed": first}, "jobs": list(reversed(new_jobs)) + _state["jobs"]}
        _commit(source)
        return web.json_response({"jobs": new_jobs})


async def _submit_prompt(prompt):
    """POST a prompt to this ComfyUI. Returns its JSON result, or an error response to send back as-is."""
    # no client_id: ComfyUI then broadcasts progress events to every open page
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{_local_base_url()}/prompt", json={"prompt": prompt}) as resp:
                result = await resp.json(content_type=None)
                if resp.status != 200:
                    return web.json_response({"error": "ComfyUI từ chối workflow.", "details": result}, status=resp.status)
                return result
    except aiohttp.ClientError as e:
        return _error(f"Không gửi được tới ComfyUI: {e}", 502)


@routes.post("/qwenedit/jobs/{job_id}/cancel")
async def qwenedit_cancel(request):
    job_id = request.match_info["job_id"]
    queue = PromptServer.instance.prompt_queue
    async with _lock:
        if not any(j.get("id") == job_id for j in _state["jobs"]):
            return _error("Không có job này.", 404)
        if queue.delete_queue_item(lambda item: item[1] == job_id):
            _replace_job(job_id, {"status": "cancelled"})
            _commit("server")
            return web.json_response({"cancelled": "pending"})
    running, _ = queue.get_current_queue_volatile()
    if any(item[1] == job_id for item in running):
        nodes.interrupt_processing()
        return web.json_response({"cancelled": "interrupt-sent"})
    return _error("Job không còn trong hàng đợi.", 409)


def _output_path(f):
    """Absolute path of a job's output file. Refuses anything outside ComfyUI/output."""
    root = os.path.realpath(folder_paths.get_output_directory())
    path = os.path.realpath(os.path.join(root, f.get("subfolder") or "", f.get("filename") or ""))
    if f.get("type", "output") != "output" or os.path.commonpath([root, path]) != root or path == root:
        raise ValueError(f"không phải file trong thư mục output: {f.get('filename')}")
    return path


def _delete_files(files):
    """Delete output files. Returns (deleted paths, error messages); files already gone are skipped."""
    deleted, errors = [], []
    for f in files:
        try:
            path = _output_path(f)
            if os.path.exists(path):
                os.remove(path)
                deleted.append(path)
        except (OSError, ValueError) as e:
            errors.append(f"{f.get('filename')}: {e}")
    return deleted, errors


@routes.post("/qwenedit/jobs/{job_id}/delete")
async def qwenedit_delete_job(request):
    """Remove a finished job from the list and delete its image files from disk for good."""
    global _state
    job_id = request.match_info["job_id"]
    async with _lock:
        job = next((j for j in _state["jobs"] if j.get("id") == job_id), None)
        if not job:
            return _error("Không có job này.", 404)
        if job.get("status") in _ACTIVE:
            return _error("Job đang chạy hoặc đang chờ, hãy huỷ trước khi xoá.", 409)
        images = job.get("images") or []
        deleted, errors = _delete_files(images)
        if errors:
            # keep the job so the user still sees what could not be deleted
            return _error("Không xoá được: " + "; ".join(errors), 500)
        # the draft may use a deleted result as an input ("Sửa tiếp"): drop those images too
        gone = {(i.get("subfolder") or "", i["filename"]) for i in images}
        refs = [r for r in _state["draft"]["refs"]
                if not (r.get("type") == "output" and ((r.get("subfolder") or ""), r.get("name")) in gone)]
        refs_removed = len(_state["draft"]["refs"]) - len(refs)
        _state = {**_state, "draft": {**_state["draft"], "refs": refs},
                  "jobs": [j for j in _state["jobs"] if j.get("id") != job_id]}
        _commit(str(request.query.get("source") or "api"))
        return web.json_response({"deleted": job_id, "files": deleted, "refsRemoved": refs_removed})


@routes.delete("/qwenedit/jobs/{job_id}")
async def qwenedit_hide_job(request):
    global _state
    job_id = request.match_info["job_id"]
    async with _lock:
        jobs = [j for j in _state["jobs"] if j.get("id") != job_id]
        if len(jobs) == len(_state["jobs"]):
            return _error("Không có job này.", 404)
        _state = {**_state, "jobs": jobs}
        _commit(str(request.query.get("source") or "api"))
        return web.json_response({"hidden": job_id})


print("[Qwen Edit Studio] UI available at /qwenedit")
