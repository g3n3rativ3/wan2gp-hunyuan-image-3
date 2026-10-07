# SigLIP2 so400m NaFlex vision tower used by HunyuanImage-3.0 for its conditioning images.
#
# Pure torch port of the HF `Siglip2VisionTransformer` (Apache-2.0) restricted to what HunyuanImage-3.0
# reads: `last_hidden_state` (after `post_layernorm`). The attention-pooling head is never used, so it is
# not built (its keys are simply ignored at load time). Key names follow the checkpoint
# (`vision_model.embeddings.patch_embedding.weight`, `vision_model.encoder.layers.N...`).
#
# The preprocessing reproduces the reference processor (`Siglip2ImageProcessor`): resize so that the
# image fits `max_num_patches` patches of 16 px, bilinear, rescale to [0, 1], normalize with mean/std 0.5,
# patchify channels-last, and pad the run to `max_num_patches` with zero patches whose keys are masked.
import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

VISION_CONFIG = {
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "layer_norm_eps": 1e-6,
    "num_attention_heads": 16,
    "num_hidden_layers": 27,
    "num_patches": 256,
    "patch_size": 16,
    "num_channels": 3,
}


@lru_cache(maxsize=256)
def get_image_size_for_max_num_patches(image_height, image_width, patch_size, max_num_patches, eps=1e-5):
    def scaled(scale, size):
        size = math.ceil(size * scale / patch_size) * patch_size
        return int(max(patch_size, size))

    scale_min, scale_max = eps / 10, 100.0
    while (scale_max - scale_min) >= eps:
        scale = (scale_min + scale_max) / 2
        num_patches = (scaled(scale, image_height) / patch_size) * (scaled(scale, image_width) / patch_size)
        if num_patches <= max_num_patches:
            scale_min = scale
        else:
            scale_max = scale
    return scaled(scale_min, image_height), scaled(scale_min, image_width)


def preprocess_image(image: Image.Image, patch_size=16, max_num_patches=1024):
    """PIL image -> (patches (N, p*p*3) float32, (patch_height, patch_width))."""
    image = image.convert("RGB")
    target_height, target_width = get_image_size_for_max_num_patches(image.height, image.width, patch_size, max_num_patches)
    image = image.resize((target_width, target_height), resample=Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    ph, pw = target_height // patch_size, target_width // patch_size
    patches = array.reshape(ph, patch_size, pw, patch_size, 3).transpose(0, 2, 1, 3, 4).reshape(ph * pw, -1)
    return torch.from_numpy(np.ascontiguousarray(patches)), (ph, pw)


class Siglip2Embeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_size = cfg["patch_size"]
        self.embed_dim = cfg["hidden_size"]
        self.patch_embedding = nn.Linear(cfg["num_channels"] * self.patch_size * self.patch_size, self.embed_dim)
        self.position_embedding_size = int(cfg["num_patches"] ** 0.5)
        self.position_embedding = nn.Embedding(cfg["num_patches"], self.embed_dim)

    def forward(self, pixel_values, grid):
        patch_embeds = self.patch_embedding(pixel_values.to(self.patch_embedding.weight.dtype))
        height, width = grid
        table = self.position_embedding.weight.reshape(self.position_embedding_size, self.position_embedding_size, -1)
        table = table.permute(2, 0, 1).unsqueeze(0).float()
        resized = F.interpolate(table, size=(height, width), mode="bilinear", align_corners=False, antialias=True)
        resized = resized.reshape(self.embed_dim, height * width).transpose(0, 1).to(patch_embeds.dtype)
        positions = resized.new_empty((pixel_values.shape[1], self.embed_dim))
        positions[: height * width] = resized
        positions[height * width:] = resized[0]
        return patch_embeds + positions.unsqueeze(0)


class Siglip2Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = cfg["hidden_size"]
        self.num_heads = cfg["num_attention_heads"]
        self.head_dim = self.embed_dim // self.num_heads
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x, attention_mask=None):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        return self.out_proj(out.transpose(1, 2).reshape(b, s, self.embed_dim))


class Siglip2MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"])
        self.fc2 = nn.Linear(cfg["intermediate_size"], cfg["hidden_size"])

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class Siglip2EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(cfg["hidden_size"], eps=cfg["layer_norm_eps"])
        self.self_attn = Siglip2Attention(cfg)
        self.layer_norm2 = nn.LayerNorm(cfg["hidden_size"], eps=cfg["layer_norm_eps"])
        self.mlp = Siglip2MLP(cfg)

    def forward(self, x, attention_mask=None):
        x = x + self.self_attn(self.layer_norm1(x), attention_mask)
        return x + self.mlp(self.layer_norm2(x))


class Siglip2Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([Siglip2EncoderLayer(cfg) for _ in range(cfg["num_hidden_layers"])])

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


class Siglip2VisionTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embeddings = Siglip2Embeddings(cfg)
        self.encoder = Siglip2Encoder(cfg)
        self.post_layernorm = nn.LayerNorm(cfg["hidden_size"], eps=cfg["layer_norm_eps"])

    def forward(self, pixel_values, grid, valid_tokens=None):
        hidden = self.embeddings(pixel_values, grid)
        mask = None
        if valid_tokens is not None and valid_tokens < hidden.shape[1]:
            # keys-only padding mask, as the reference's `_prepare_4d_attention_mask`
            mask = torch.zeros(1, 1, 1, hidden.shape[1], dtype=hidden.dtype, device=hidden.device)
            mask[..., valid_tokens:] = torch.finfo(hidden.dtype).min
        hidden = self.encoder(hidden, mask)
        return self.post_layernorm(hidden)


class HunyuanImage3VisionTower(nn.Module):
    """Root module whose keys match `hunyuan_image_3_*_siglip2_so400m_naflex.safetensors`."""

    def __init__(self, cfg=None):
        super().__init__()
        self.config = dict(VISION_CONFIG if cfg is None else cfg)
        self.vision_model = Siglip2VisionTransformer(self.config)

    @torch.no_grad()
    def encode_image(self, image: Image.Image, max_num_patches=1024, reference_padding=True):
        """Returns ((1, tokens, 1152) last_hidden_state, (patch_height, patch_width)).

        With `reference_padding` the run is padded to `max_num_patches` zero patches whose keys are
        masked, exactly as the reference processor feeds the tower."""
        weight = self.vision_model.embeddings.patch_embedding.weight
        patches, grid = preprocess_image(image, self.config["patch_size"], max_num_patches)
        valid = patches.shape[0]
        if reference_padding and valid < max_num_patches:
            patches = torch.cat([patches, patches.new_zeros(max_num_patches - valid, patches.shape[1])], dim=0)
        # mmgp may keep the weights in RAM until the tower runs: target the compute device, not the weight's
        device = torch.device("cuda") if torch.cuda.is_available() else weight.device
        pixel_values = patches.unsqueeze(0).to(device=device, dtype=weight.dtype)
        tokens = self.vision_model(pixel_values, grid, valid_tokens=valid)
        return tokens, grid
