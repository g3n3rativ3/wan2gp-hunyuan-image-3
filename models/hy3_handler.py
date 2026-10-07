# WanGP family handler for HunyuanImage-3.0 Instruct-Distil (W4A8).
import torch
from PIL import Image

from .hy3_tokenizer import MAX_COND_IMAGES, ResolutionGroup

MODEL_TYPE = "hunyuan_image_3_instruct_distil"
FAMILY = "hunyuan_image_3"
HF_REPO = "PedroMarinhoDev/HunyuanImage-3.0-Instruct-Distil-ComfyUI"
ASSETS_FOLDER = "hunyuan_image_3"
VAE_FILE = "hunyuan_image_3_vae_fp16.safetensors"
VISION_FILE = "hunyuan_image_3_instruct_distil_siglip2_so400m_naflex.safetensors"


def _native_resolutions():
    group = ResolutionGroup(1024)
    choices, seen = [], set()
    for entry in sorted(group.data, key=lambda r: r.width / r.height, reverse=True):
        value = f"{entry.width}x{entry.height}"
        if value in seen:
            continue
        seen.add(value)
        divisor = _gcd(entry.width, entry.height)
        choices.append((f"{value} ({entry.width // divisor}:{entry.height // divisor})", value))
    return choices


# Extra sizes outside the model's native ~1 Mpx table. WanGP files resolutions in categories by pixel
# count: 1080p is (1024x1024, 1920x1088], 540p is (832x624, 960x544]. The 1080p sizes stay below
# ~1536x1536, where the rope rescaling still holds; the 540p ones are faster but outside training.
EXTRA_RESOLUTIONS_1080P = [
    ("1440x1440 (1:1)", "1440x1440"),
    ("1280x1280 (1:1)", "1280x1280"),
    ("1920x1088 (16:9)", "1920x1088"),
    ("1088x1920 (9:16)", "1088x1920"),
    ("1664x1248 (4:3)", "1664x1248"),
    ("1248x1664 (3:4)", "1248x1664"),
    ("1536x1024 (3:2)", "1536x1024"),
    ("1024x1536 (2:3)", "1024x1536"),
    ("1920x832 (21:9)", "1920x832"),
    ("832x1920 (9:21)", "832x1920"),
]

EXTRA_RESOLUTIONS_540P = [
    ("960x544 (16:9)", "960x544"),
    ("544x960 (9:16)", "544x960"),
    ("928x560 (5:3)", "928x560"),
    ("560x928 (3:5)", "560x928"),
    ("880x592 (3:2)", "880x592"),
    ("592x880 (2:3)", "592x880"),
    ("816x640 (5:4)", "816x640"),
    ("640x816 (4:5)", "640x816"),
    ("1088x480 (9:4)", "1088x480"),
    ("480x1088 (4:9)", "480x1088"),
]


def _all_resolutions():
    choices, seen = [], set()
    for label, value in EXTRA_RESOLUTIONS_1080P + _native_resolutions() + EXTRA_RESOLUTIONS_540P:
        if value not in seen:
            seen.add(value)
            choices.append((label, value))
    return choices


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


_CUSTOM_SETTINGS = [
    {
        "id": "vit_strength",
        "label": "Reference Vision Strength (SigLIP2 tokens, 1.0 = reference)",
        "name": "Vision Strength",
        "type": "float", "min": 0.0, "max": 2.0, "inc": 0.05, "default": 1.0,
        "video_prompt_type": "I",
    },
    {
        "id": "latent_strength",
        "label": "Reference Latent Strength (VAE latents, 1.0 = reference)",
        "name": "Latent Strength",
        "type": "float", "min": 0.0, "max": 2.0, "inc": 0.05, "default": 1.0,
        "video_prompt_type": "I",
    },
    {
        "id": "reference_vit_padding",
        "label": "Vision Tokens Padding",
        "name": "Vision Padding",
        "type": "dropdown",
        "choices": [("Pad to 1024 tokens (as the reference processor)", 1), ("Image patches only", 0)],
        "default": 1,
        "video_prompt_type": "I",
    },
    {
        "id": "system_prompt",
        "label": "Custom System Prompt (empty = the model's own unified prompt)",
        "name": "System Prompt",
        "type": "text",
        "default": "",
    },
]


class family_handler:
    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {
            "image_outputs": True,
            "inference_steps": True,
            "embedded_guidance": True,
            "guidance_max_phases": 0,
            "no_negative_prompt": True,
            "no_background_removal": True,
            "fit_into_canvas_image_refs": 0,
            "vae_block_size": 16,
            "image_batch_size_max": 8,
            "resolutions": [[label, value] for label, value in _all_resolutions()],
            "image_ref_choices": {
                "choices": [
                    ("None (Text to Image)", ""),
                    ("Reference Images: first one is the image to edit, output size follows it", "KI"),
                    ("Reference Images: free composition, output size from Resolution", "I"),
                ],
                "letters_filter": "KI",
                "default": "",
                "label": f"Reference Images (edit / fusion, up to {MAX_COND_IMAGES})",
            },
            "custom_settings": [one.copy() for one in _CUSTOM_SETTINGS],
            "preset_profiles_dir": ["hunyuan_image_3_presets"],
            "infos": "HunyuanImage-3.0 Instruct-Distil (Tencent), 80B mixture-of-experts (13B active), W4A8 weights. "
                     "Text to image in 8 steps, image editing and multi-image fusion with up to 3 Reference Images. "
                     "The guidance is embedded in the model (no CFG pass): keep 8 steps and an Embedded Guidance around 2.5.",
            "prompt_infos": "Write in natural language (English or Chinese). For editing, give a direct instruction and say what must "
                            "stay unchanged, e.g. \"Turn this photo into a watercolor painting, keep the composition.\" With several "
                            "references, name them by order: \"Put the cat from the first image on the sofa of the second image.\" "
                            "Put literal text to render between double quotes.",
            "deepy_infos": "HunyuanImage-3.0 Instruct-Distil: text-to-image, and editing / fusion from 1 to 3 ordered `image_refs` "
                           "(`KI`: the first reference is the image to edit). 8 steps, embedded guidance 2.5.",
            "deepy_prompt_infos": "Natural-language description, or an edit instruction that names references by order (first image, second image) and states what must not change.",
        }

    @staticmethod
    def query_supported_types():
        return [MODEL_TYPE]

    @staticmethod
    def query_family_maps():
        return {}, {MODEL_TYPE: [MODEL_TYPE]}

    @staticmethod
    def query_model_family():
        return FAMILY

    @staticmethod
    def query_family_infos():
        return {FAMILY: (1160, "HunyuanImage 3.0")}

    @staticmethod
    def get_lora_dir(base_model_type):
        return "hunyuan_image_3"

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [{
            "repoId": HF_REPO,
            "sourceFolderList": ["vae", "clip_vision"],
            "fileList": [[VAE_FILE], [VISION_FILE]],
            "targetFolderList": [ASSETS_FOLDER, ASSETS_FOLDER],
        }]

    @staticmethod
    def load_model(model_filename, model_type=None, base_model_type=None, model_def=None, quantizeTransformer=False,
                   text_encoder_quantization=None, dtype=torch.bfloat16, VAE_dtype=torch.float16,
                   mixed_precision_transformer=False, save_quantized=False, submodel_no_list=None,
                   text_encoder_filename=None, **kwargs):
        from .hy3_main import model_factory

        pipe_processor = model_factory(model_filename=model_filename, model_type=model_type, model_def=model_def,
                                       base_model_type=base_model_type, dtype=dtype, VAE_dtype=VAE_dtype)
        pipe = {
            "transformer": pipe_processor.transformer,
            "vae": pipe_processor.vae,
            "vision_encoder": pipe_processor.vision_encoder,
        }
        return pipe_processor, pipe

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "image_mode": 1,
            "batch_size": 1,
            "num_inference_steps": 8,
            "embedded_guidance_scale": 2.5,
            "guidance_scale": 1.0,
            "resolution": "1024x1024",
            "video_prompt_type": "",
            "custom_settings": {"vit_strength": 1.0, "latent_strength": 1.0, "reference_vit_padding": 1, "system_prompt": ""},
        })

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        ui_defaults.setdefault("image_mode", 1)
        ui_defaults.setdefault("embedded_guidance_scale", 2.5)

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        if "I" in (inputs.get("video_prompt_type") or ""):
            refs = inputs.get("image_refs") or []
            if len(refs) > MAX_COND_IMAGES:
                return f"HunyuanImage 3.0 accepts at most {MAX_COND_IMAGES} Reference Images."
        try:
            width, height = (int(v) for v in str(inputs.get("resolution", "1024x1024")).lower().split("x"))
        except (TypeError, ValueError):
            return None
        if width * height > 1600 * 1600:
            return "HunyuanImage 3.0 was trained around 1 megapixel: sizes beyond ~1536x1536 come out garbled. Generate at 1024 and upscale."
        return None

    @staticmethod
    def get_rgb_factors(base_model_type):
        return None, None

    @staticmethod
    def preview_latents(base_model_type, latents, meta):
        """No RGB projection has been fitted for this VAE: show the 3 principal components of the
        denoised-latent estimate, which is enough to follow the composition."""
        if not torch.is_tensor(latents) or latents.dim() != 4:
            return None
        x = latents[:, 0].detach().float().cpu()                 # (C, h, w)
        channels, height, width = x.shape
        flat = x.reshape(channels, -1).t()
        flat = flat - flat.mean(0, keepdim=True)
        try:
            with torch.device("cpu"):
                _, _, v = torch.pca_lowrank(flat, q=3, center=False)
            rgb = flat @ v[:, :3]
        except Exception:
            rgb = flat[:, :3]
        low, high = rgb.quantile(0.01, dim=0), rgb.quantile(0.99, dim=0)
        rgb = ((rgb - low) / (high - low).clamp(min=1e-6)).clamp(0, 1)
        image = (rgb.t().reshape(3, height, width) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
        preview = Image.fromarray(image)
        scale = 200 / max(1, preview.height)
        return preview.resize((max(1, int(round(preview.width * scale))), 200), resample=Image.Resampling.BILINEAR)
