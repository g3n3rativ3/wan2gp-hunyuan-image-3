# HunyuanImage 3.0 Instruct-Distil for WanGP

A model plugin for [WanGP](https://github.com/deepbeepmeep/Wan2GP) that adds Tencent's **HunyuanImage-3.0 Instruct-Distil**.

It is an 80-billion-parameter *mixture-of-experts* model, with 13 billion parameters active per token, used here with **W4A8** weights. The model does three things:

- **text to image** in 8 steps;
- **image editing** ("turn this photo into a watercolor");
- **multi-image fusion**, with up to 3 Reference Images.

This plugin is a port of the ComfyUI pack [PedroMarinhoDev/ComfyUI-HunyuanImage3](https://github.com/PedroMarinhoDev/ComfyUI-HunyuanImage3). Only the **Instruct-Distil (8 steps)** version is ported. Prompt rewriting is not included.

## Requirements

| | |
|---|---|
| GPU | NVIDIA, 12 GB of VRAM or more |
| System RAM | about **50 GB free**: the weights stay in RAM and are streamed to the GPU layer by layer at every step |
| Disk | about 50 GB (44 GB transformer, plus the VAE and the vision tower) |

The model is bound by weight transfers, not by compute. The amount of RAM and the PCIe bus speed matter more than the size of the GPU.

## Installation

1. Unzip the archive into WanGP's `plugins/` folder. You should get `plugins/wan2gp-hunyuan-image-3/plugin_info.json`.
2. Start WanGP, open the **Plugins** tab, tick **HunyuanImage 3.0 Instruct-Distil**, then click *Save Settings*.
3. **Restart WanGP.** The model appears in the **HunyuanImage 3.0** family.

Nothing needs to be installed with pip: the plugin only uses packages WanGP already provides.

## Automatic model download

On the **first generation**, WanGP downloads the files from [PedroMarinhoDev/HunyuanImage-3.0-Instruct-Distil-ComfyUI](https://huggingface.co/PedroMarinhoDev/HunyuanImage-3.0-Instruct-Distil-ComfyUI). They go into the models folder set in **Configuration**:

| File | Location |
|---|---|
| `hunyuan_image_3_instruct_distil_w4a8.safetensors` (≈ 44 GB) | `<models folder>/` |
| `vae/hunyuan_image_3_vae_fp16.safetensors` | `<models folder>/hunyuan_image_3/vae/` |
| `clip_vision/hunyuan_image_3_instruct_distil_siglip2_so400m_naflex.safetensors` | `<models folder>/hunyuan_image_3/clip_vision/` |

Tencent's `config.json` and `tokenizer.json` ship with the plugin, in `models/tencent/`.

If you already have these files (for ComfyUI, for instance), copy or link them to these locations to avoid downloading them again.

## Usage

### Text to image

Write a natural-language prompt, in English or Chinese, and generate. The default settings are the ones the model was distilled with: **8 steps** and **Embedded Guidance 2.5**.

**Image size.** The model was trained around 1 megapixel. The *Resolution* list is split into three categories:

- **720p**: the model's 37 native sizes (about 1 Mpx, from 2048×512 to 512×2048). Best quality.
- **1080p**: larger sizes, up to 1440×1440 or 1920×1088. The rope is rescaled automatically, but prompt adherence weakens as the size moves away from 1 Mpx, and each step is slower (up to about twice the image tokens).
- **540p**: smaller sizes (about 0.5 Mpx) for faster drafts. They are outside the training range, so quality may drop.

Custom sizes up to about 1536×1536 work. Beyond that the image degrades: generate at 1024, then upscale with WanGP's post-processing.

### Editing and fusion with Reference Images

In the **Reference Images** selector, choose:

- **First one is the image to edit** (`KI`): the first image is the one to modify, and the output size follows its aspect ratio. Use this mode for editing.
- **Free composition** (`I`): the images act as references, and the output size is the one chosen in *Resolution*.

Add **1 to 3 images** in the reference window. In the prompt, refer to them by order: "Put the cat from the first image on the sofa in the second image". Also state what must not change.

Each image is resized to about 1 Mpx, as Tencent's pipeline does. Each extra image makes every step slower.

**Tip:** if an edit looks oversaturated or "overbaked", change the seed. Reusing the seed that produced the input image pushes the result too far.

### Extra options (below the prompt)

The first three options only show when Reference Images are active.

| Option | Effect |
|---|---|
| Reference Vision Strength | weight of the SigLIP2 vision tower tokens (1.0 = reference behaviour) |
| Reference Latent Strength | weight of the references' VAE latents (1.0 = reference) |
| Vision Tokens Padding | pads each image to 1024 vision tokens, as the reference processor does (default). "Image patches only" reproduces the ComfyUI pack's earlier behaviour |
| Custom System Prompt | replaces the model's built-in system prompt. Leave it empty in most cases |

### Advanced Mode

- **General**: seed, number of steps, *Embedded Guidance Scale*. This model's guidance is an input token, not CFG: there is no negative pass, hence no negative prompt. Keep 8 steps, as other values drift from the distilled trajectory.
- **LoRAs**: folder `loras/hunyuan_image_3/`. LoRAs apply to the dense layers (attention and shared expert), not to the quantized routed experts.
- **Post Processing / Misc.**: all of WanGP's standard options (upscaling, film grain, etc.).
- **Presets**: *Instruct-Distil Reference Settings* restores the original settings.

The preview during denoising is an approximate projection (3 principal components of the latent). The colors are not real, but the composition is visible.

## Memory profiles

Use a profile that keeps the transformer in RAM and loads it layer by layer (profiles 2, 4 or 5 when RAM is tight, profile 3 if you have plenty of RAM). If WanGP reports that reserved RAM is short, raising `perc_reserved_mem_max` in the configuration pins more blocks and speeds up the steps.

For the VAE, WanGP picks tiling automatically from the VRAM size (the *VAE Tiling* option in the configuration). When memory runs out, decoding automatically falls back to 512-pixel tiles, then 256-pixel tiles.

## Technical details

This information is useful for maintenance.

**W4A8 weights.** The format is `comfy_kitchen`'s (`AsymW4A8Int8Layout`): 4-bit Lloyd-Max codes, fp8 scales per group of 16, fp32 scales per row, ConvRot rotation (Hadamard 256).
- The dense projections go through WanGP's native W4A8 handler (`shared/qtypes/asym_w4a8_int8.py`).
- The 64 experts of each layer are stored as 3-D banks. They are handled by `models/hy3_experts.py`, which reuses the same kernels (Triton decode and int8 GEMM) one expert at a time.
- The file stores the fp8 scales as `uint8`. The plugin reinterprets them at load time, in `preprocess_w4a8_state_dict`.

**Sampling.** Flow matching with shift 3, Euler in fp32, `<timestep_r>` (meanflow) set to the next sigma, guidance of 2.5 × 1000 passed as an embedding.

**Sequence.** The full sequence is recomputed at every step, with no KV cache. The result is identical to the reference pipeline, because everything ahead of the image block is causal.

**Deliberate difference from the ComfyUI pack.** In the VAE attention block, the `b c f h w -> b 1 (f h w) c` flattening follows the `rearrange` of Tencent's reference (channels permuted). The ComfyUI pack uses a plain `reshape`.

**Code derived from ComfyUI-HunyuanImage3.** `models/hy3_tokenizer.py` and `models/hy3_system_prompt.py` are taken as is. `hy3_model.py` and `hy3_vae.py` are adapted from the same pack's `model.py`, `pipeline.py` and `vae.py`.

## Troubleshooting

- **Noisy or incoherent images**: go back to 8 steps and an Embedded Guidance of 2.5.
- **Out of memory on the RAM side**: close other applications or pick a memory profile that pins less RAM.
- **"at most 3 Reference Images"**: the model does not accept more than 3 references.
- **"needs N positions … handles 22800"**: the sequence is too long. Reduce the resolution, the number of references or the prompt length.
- **The model does not appear**: check that the plugin is enabled and that WanGP was restarted. The console should show `[PluginManager] Loaded model plugin: HunyuanImage 3.0 Instruct-Distil`.

## Licenses

- **Plugin code**: GPL-3.0 (see `LICENSE`), like the ComfyUI pack it is derived from (© 2026 PedroMarinhoDev, modified for WanGP). Parts of the model, VAE and tokenizer come from Tencent's reference implementation and remain subject to the Tencent Hunyuan Community License.
- **Weights, `config.json` and `tokenizer.json`**: [Tencent Hunyuan Community License](LICENSE-TENCENT-HUNYUAN). **This license does not apply in the European Union, the United Kingdom or South Korea**: it grants no usage rights in those territories. It also includes an Acceptable Use Policy. See `NOTICE`.

## Credits

- [Tencent Hunyuan](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0) for HunyuanImage-3.0.
- [PedroMarinhoDev](https://github.com/PedroMarinhoDev/ComfyUI-HunyuanImage3) for the ComfyUI port, the W4A8 weights and the converted files.
- [Comfy-Org comfy_kitchen](https://github.com/Comfy-Org) for the W4A8 ConvRot format.
- [DeepBeepMeep / WanGP](https://github.com/deepbeepmeep/Wan2GP) and mmgp.
