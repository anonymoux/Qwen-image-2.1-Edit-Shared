# Qwen Edit Studio

A compact web UI for ComfyUI's **Qwen Image 2.1 image edit** workflow (template `image_qwen_image_2_1_image_edit`), served by ComfyUI itself at `http://127.0.0.1:8188/qwenedit`.

- Prompt editor with `<image1>` / `@image_1` tags that renumber when images are reordered
- Up to 10 images: image 1 is the one being edited, the rest are references. Drag-and-drop upload, paste, reorder, enlarged preview on hover
- Image sizes are set per image: image 1 follows the size setting, and references larger than the reference limit (default ~1 MP; ~1.5, ~2.4, ~4 MP or no limit) are shrunk to it (an `ImageScale` node per image; the encoder's own resize is then off). The page shows each image's file size → model size, the output size, and warns when image 1 goes past 2K
- Settings: image 1 size budget (keep / 1 MP / 2.4 MP / 2K), output canvas (follow image 1 or custom width × height), steps, CFG, sampler, scheduler, seed, 1–4 images per click
- Optional Qwen 3.5 prompt enhancer (rewrites the instruction before the edit model sees it); the rewritten prompt is shown in the job details
- Model pickers for the diffusion model, text encoder, VAE and enhancer, plus the Qwen Image 2.1 prefix cache settings. Picking a diffusion model also picks the installed text encoder and VAE of the same precision (`int8_convrot`, `bf16`…); Qwen Image 2.1 ships one bf16 VAE, used with every model
- Results panel with live progress, before/after compare slider, details dialog (timings, inputs, prompts), and one-click "edit this result again" (the output is used in place from `ComfyUI/output`, no copy). Results can be hidden from the list, or deleted for good together with their files (after a confirmation)
- Header link to [H3 Studio](https://github.com/anonymoux/H3_Studio) (`/h3studio`), the MiniMax H3 video companion app on the same ComfyUI
- A banner lists the model files the current settings need but are not installed, with download links
- Draft and job list are stored on the server (`user/qwenedit_state.json`) and pushed to open pages over the ComfyUI websocket, so scripts and the page share the same state

## Install

Copy this folder into `ComfyUI/custom_nodes/` and restart ComfyUI. Needs a ComfyUI version that has the `TextEncodeQwenImage21`, `QwenImage21Cache` and `TextGenerate` nodes (0.43 or newer).

The workflow graph is built in `builder.py` (the template's subgraph flattened). `pe_system_prompt.md` is the prompt enhancer's system prompt, copied from the template.

## Models

Paths are relative to `ComfyUI/models/`. Source: [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1).

### Required (~16 GB)

| Path | File | Size |
|---|---|---|
| `diffusion_models/` | [qwen_image_2.1_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/qwen_image_2.1_int8_convrot.safetensors) | 6.8 GB |
| `text_encoders/` | [qwen3vl_8b_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors) | 8.7 GB |
| `vae/` | [qwen_image_2.1_vae_bf16.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors) | 0.6 GB |

### Optional

| Feature | Path | File | Size |
|---|---|---|---|
| Prompt enhancer | `text_encoders/` | [qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors) | 8.8 GB |
| Full-precision model | `diffusion_models/` | [qwen_image_2.1_bf16.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/qwen_image_2.1_bf16.safetensors) | 13.3 GB |
| Full-precision text encoder | `text_encoders/` | [qwen3vl_8b_bf16.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_bf16.safetensors) | 16.3 GB |

## API

```
GET    /qwenedit/options            model lists, sampler/scheduler/cache choices, missing models
GET    /qwenedit/state              {rev, draft, jobs}
PUT    /qwenedit/state              {draft: {...partial}}
POST   /qwenedit/queue              queue the current draft (draft.count jobs)
POST   /qwenedit/jobs/{id}/cancel
DELETE /qwenedit/jobs/{id}          hide from the list (file is kept)
```

## Tips (from the template)

- Keep CFG at 1 for the official path; the negative prompt only matters above 1.
- The official pipeline uses about 40–50 steps with euler; the default here is 25.
- Turn on `thinking` together with the enhancer: the enhancer was trained with a reasoning block.
