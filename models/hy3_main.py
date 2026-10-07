# HunyuanImage-3.0 Instruct-Distil pipeline for WanGP (text-to-image, editing, multi-image fusion).
#
# Sampling contract of the Instruct-Distil checkpoint (ComfyUI-HunyuanImage3 `pipeline.py`, reference
# `FlowMatchDiscreteScheduler`): flow matching, shift 3.0, `sigmas = shift*t / (1 + (shift-1)*t)` over
# `linspace(1, 0, steps + 1)`, Euler in fp32, 8 steps, guidance as a token embedding (2.5 x 1000) instead of
# CFG, meanflow `<timestep_r>` = the next sigma. No negative pass.
import gc
import json
import os

import numpy as np
import torch
from accelerate import init_empty_weights
from PIL import Image

from mmgp import offload
from shared.utils import files_locator as fl
from shared.utils.phase_progress import generation_progress

from .hy3_experts import preprocess_w4a8_state_dict
from .hy3_model import (NUM_TRAIN_TIMESTEPS, HunyuanImage3Transformer, build_attention_mask, build_rope,
                        params_from_config, sequence_from_ids)
from .hy3_system_prompt import SYSTEM_PROMPTS
from .hy3_tokenizer import COND_PATCH_LIMIT, MAX_COND_IMAGES, ResolutionGroup, build_sequence
from .hy3_vae import HunyuanImage3VAEWrapper
from .hy3_vision import HunyuanImage3VisionTower

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_HERE, "tencent", "config.json")
TOKENIZER_PATH = os.path.join(_HERE, "tencent", "tokenizer.json")

HF_REPO = "PedroMarinhoDev/HunyuanImage-3.0-Instruct-Distil-ComfyUI"
ASSETS_FOLDER = "hunyuan_image_3"
VAE_FILE = "hunyuan_image_3_vae_fp16.safetensors"
VISION_FILE = "hunyuan_image_3_instruct_distil_siglip2_so400m_naflex.safetensors"
VAE_PATH = os.path.join(ASSETS_FOLDER, "vae", VAE_FILE)
VISION_PATH = os.path.join(ASSETS_FOLDER, "clip_vision", VISION_FILE)

SHIFT = 3.0
DEFAULT_STEPS = 8
DEFAULT_GUIDANCE = 2.5
BASE_SIZE = 1024
LATENT_CHANNELS = 32
SPATIAL_FACTOR = 16


def flow_sigmas(steps, shift=SHIFT):
    sigmas = torch.linspace(1, 0, steps + 1, dtype=torch.float32, device="cpu")
    return (shift * sigmas) / (1 + (shift - 1) * sigmas)


def resize_and_crop(image, width, height):
    """Reference `resize_and_crop`: cover `width`x`height` keeping the ratio (Lanczos), center crop."""
    image_width, image_height = image.size
    if image_height / image_width < height / width:
        resize_height, resize_width = height, int(round(height / image_height * image_width))
    else:
        resize_height, resize_width = int(round(width / image_width * image_height)), width
    image = image.resize((resize_width, resize_height), resample=Image.Resampling.LANCZOS)
    top, left = int(round((resize_height - height) / 2.0)), int(round((resize_width - width) / 2.0))
    return image.crop((left, top, left + width, top + height))


def _pil_to_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def _to_pil(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if torch.is_tensor(image):
        from shared.utils.utils import convert_tensor_to_image
        return convert_tensor_to_image(image).convert("RGB")
    return Image.fromarray(np.asarray(image)).convert("RGB")


def _setting(custom_settings, key, default):
    if isinstance(custom_settings, dict) and custom_settings.get(key, None) not in (None, ""):
        return custom_settings[key]
    return default


def _load_transformer(model_filename):
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    # the sampling contract belongs to the Instruct-Distil weights, whatever config.json says
    config.update({"cfg_distilled": True, "use_meanflow": True, "sequence_template": "instruct", "model_type_hy3": "instruct_distil"})
    params = params_from_config(config)
    with init_empty_weights(include_buffers=True):
        transformer = HunyuanImage3Transformer(params)
    offload.load_model_data(transformer, model_filename, writable_tensors=False, preprocess_sd=preprocess_w4a8_state_dict,
                            default_dtype=torch.bfloat16, ignore_unused_weights=True)
    transformer.eval().requires_grad_(False)
    transformer._model_dtype = torch.bfloat16
    return transformer, params


def _load_vae(dtype):
    with init_empty_weights(include_buffers=True):
        vae = HunyuanImage3VAEWrapper(in_channels=3, out_channels=3, latent_channels=LATENT_CHANNELS,
                                      block_out_channels=(128, 256, 512, 1024, 1024), layers_per_block=2,
                                      ffactor_spatial=16, ffactor_temporal=4, scaling_factor=1.0,
                                      downsample_match_channel=True, upsample_match_channel=True)
    offload.load_model_data(vae, fl.locate_file(VAE_PATH), writable_tensors=False, default_dtype=dtype, ignore_unused_weights=True)
    vae.eval().requires_grad_(False)
    return vae


def _load_vision():
    with init_empty_weights(include_buffers=True):
        vision = HunyuanImage3VisionTower()
    offload.load_model_data(vision, fl.locate_file(VISION_PATH), writable_tensors=False, default_dtype=torch.bfloat16, ignore_unused_weights=True)
    vision.eval().requires_grad_(False)
    return vision


class model_factory:
    def __init__(self, checkpoint_dir=None, model_filename=None, model_type=None, model_def=None, base_model_type=None,
                 dtype=torch.bfloat16, VAE_dtype=torch.float16, **kwargs):
        from tokenizers import Tokenizer

        filename = model_filename[0] if isinstance(model_filename, (list, tuple)) else model_filename
        self.model_def = model_def or {}
        self.base_model_type = base_model_type
        self.dtype = torch.bfloat16
        self.transformer, self.params = _load_transformer(filename)
        self.vae = _load_vae(torch.float32 if VAE_dtype == torch.float32 else torch.float16)
        self.vision_encoder = _load_vision()
        self.tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        self._interrupt_flag = False

    @property
    def _interrupt(self):
        return self._interrupt_flag

    @_interrupt.setter
    def _interrupt(self, value):
        self._interrupt_flag = value
        if hasattr(self, "transformer"):
            self.transformer._interrupt = value

    # ------------------------------------------------------------------------------- conditioning

    def _condition_on_image(self, image, device, vit_strength, latent_strength, reference_padding, tile_px):
        """One conditioning image: model-space latent, aligned tower tokens, tower grid, block size."""
        target_width, target_height = ResolutionGroup(self.params.image_base_size).get_target_size(image.width, image.height)
        pixels = _pil_to_tensor(resize_and_crop(image, target_width, target_height)).to(device)
        cond_latent = self.vae.encode_pixels(pixels, tile_px=tile_px) * latent_strength
        tokens, grid = self.vision_encoder.encode_image(image, max_num_patches=COND_PATCH_LIMIT, reference_padding=reference_padding)
        cond_vit = self.transformer.vision_aligner(tokens.to(device=device, dtype=self.dtype)) * vit_strength
        return {"latent": cond_latent, "vit": cond_vit, "grid": grid, "size": ((target_height, target_width), grid)}

    # ------------------------------------------------------------------------------- generation

    @generation_progress
    def generate(self, seed=None, input_prompt="", n_prompt=None, sampling_steps=DEFAULT_STEPS, width=1024, height=1024,
                 embedded_guidance_scale=DEFAULT_GUIDANCE, batch_size=1, callback=None, VAE_tile_size=None,
                 loras_slists=None, input_ref_images=None, original_input_ref_images=None, video_prompt_type="",
                 custom_settings=None, set_progress_status=None, **kwargs):
        from shared.utils.loras_mutipliers import update_loras_slists

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        steps = max(1, int(sampling_steps or DEFAULT_STEPS))
        guidance_scale = float(DEFAULT_GUIDANCE if embedded_guidance_scale is None else embedded_guidance_scale)
        width = int(width) // SPATIAL_FACTOR * SPATIAL_FACTOR
        height = int(height) // SPATIAL_FACTOR * SPATIAL_FACTOR
        batch_size = max(1, int(batch_size or 1))
        seed = int(seed) if seed is not None and int(seed) >= 0 else int(torch.seed() % (2 ** 31))
        tile_px = 0
        if isinstance(VAE_tile_size, (tuple, list)) and len(VAE_tile_size) and VAE_tile_size[0]:
            tile_px = int(VAE_tile_size[1]) if len(VAE_tile_size) > 1 else 512
        elif isinstance(VAE_tile_size, int) and VAE_tile_size > 0:
            tile_px = VAE_tile_size

        vit_strength = float(_setting(custom_settings, "vit_strength", 1.0))
        latent_strength = float(_setting(custom_settings, "latent_strength", 1.0))
        reference_padding = int(_setting(custom_settings, "reference_vit_padding", 1)) != 0
        system_prompt = str(_setting(custom_settings, "system_prompt", "")).strip() or SYSTEM_PROMPTS["instruct_distil"]

        # ---- reference images (originals when available: the model resizes them itself)
        references = []
        if "I" in (video_prompt_type or ""):
            source = original_input_ref_images if original_input_ref_images and input_ref_images and len(original_input_ref_images) == len(input_ref_images) else input_ref_images
            references = [_to_pil(image) for image in (source or []) if image is not None]
        if len(references) > MAX_COND_IMAGES:
            raise ValueError(f"HunyuanImage 3.0 accepts at most {MAX_COND_IMAGES} reference images ({len(references)} given)")

        conds = []
        if references:
            if callable(set_progress_status):
                set_progress_status("Encoding Reference Images")
            for image in references:
                if self._interrupt:
                    return None
                conds.append(self._condition_on_image(image, device, vit_strength, latent_strength, reference_padding, tile_px))

        # ---- token sequence and its geometry (shared by every image of the batch)
        sequence = build_sequence(self.tokenizer, input_prompt or "", f"{width}x{height}", system_prompt,
                                  base_size=self.params.image_base_size,
                                  max_position_embeddings=self.params.max_position_embeddings,
                                  cfg_distilled=True, use_meanflow=True, sequence_template="instruct",
                                  cond_images=[cond["size"] for cond in conds], uncond=False, extra_rows=True,
                                  reference_vit_padding=reference_padding)
        token_height, token_width = height // SPATIAL_FACTOR, width // SPATIAL_FACTOR
        cond_latents = [cond["latent"] for cond in conds]
        cond_vits = [cond["vit"] for cond in conds]
        step_geometry = sequence_from_ids(sequence["ids"].to(device), token_height, token_width, self.params,
                                          cond_latents=cond_latents, cond_grids=[cond["grid"] for cond in conds])
        seq_len = step_geometry["ids"].shape[0]
        mask = build_attention_mask(step_geometry, seq_len, self.dtype, device)
        cos, sin = build_rope(seq_len, self.params.attention_head_dim, step_geometry["rope_image_info"],
                              self.params.rope_theta, device)
        print(f"[HunyuanImage 3] {width}x{height}, {len(conds)} reference image(s), {seq_len} tokens, {steps} steps")

        sigmas = flow_sigmas(steps)
        guidance = torch.full((1,), 1000.0 * guidance_scale, dtype=torch.bfloat16, device=device)
        if loras_slists is not None:
            update_loras_slists(self.transformer, loras_slists, steps)
        total_steps = steps * batch_size
        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=total_steps)

        images = []
        for image_no in range(batch_size):
            generator = torch.Generator(device="cpu").manual_seed(seed + image_no)
            # WanGP sets torch's default device to cuda: the CPU generator (same noise as ComfyUI for a seed)
            # needs an explicit CPU tensor
            latents = torch.randn((1, LATENT_CHANNELS, token_height, token_width), generator=generator,
                                  dtype=torch.float32, device="cpu").to(device)
            for i in range(steps):
                if self._interrupt:
                    return None
                offload.set_step_no_for_lora(self.transformer, i)
                sigma, sigma_next = float(sigmas[i]), float(sigmas[i + 1])
                inputs = {
                    "latents": latents,
                    "timestep": torch.full((1,), sigma * NUM_TRAIN_TIMESTEPS, dtype=torch.float32, device=device),
                    "timestep_r": torch.full((1,), sigma_next * NUM_TRAIN_TIMESTEPS, dtype=torch.float32, device=device),
                    "guidance": guidance,
                    "step": step_geometry, "cos": cos, "sin": sin, "mask": mask,
                    "cond_latents": cond_latents, "cond_vits": cond_vits,
                }
                velocity = self.transformer(inputs)
                inputs = None
                if velocity is None or self._interrupt:
                    return None
                velocity = velocity.float()
                denoised = latents - velocity * sigma
                latents = latents + velocity * (sigma_next - sigma)   # fp32 Euler, as the reference
                del velocity
                if callback is not None:
                    callback(image_no * steps + i, denoised[0].unsqueeze(1), False)
                del denoised
            if callable(set_progress_status):
                set_progress_status("VAE Decoding")
            decoded = self.vae.decode_latents(latents, tile_px=tile_px)
            images.append(decoded[0].float().cpu())
            del latents, decoded
            gc.collect()

        del mask, cos, sin
        return torch.stack(images, dim=1)  # (3, B, H, W) in [-1, 1]
