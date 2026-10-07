# HunyuanImage-3.0 (Instruct-Distil) diffusion transformer for WanGP.
#
# Ported from ComfyUI-HunyuanImage3 (GPL-3.0, PedroMarinhoDev, https://github.com/PedroMarinhoDev/ComfyUI-HunyuanImage3),
# itself adapted from Tencent's HunyuanImage-3.0 reference implementation (Tencent Hunyuan Community
# License). Modified for WanGP: ComfyUI operations / attention / rope / streaming replaced with torch.nn,
# torch SDPA and mmgp block offloading; the routed experts use WanGP's W4A8 kernels (hy3_experts.py).
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hy3_experts import W4A8ExpertBank

NUM_TRAIN_TIMESTEPS = 1000
# The image grid of a 1024x1024 request (base 1024 / 16): above it the rope base is rescaled.
_TRAINED_GRID = 64


@dataclass
class HunyuanImage3Params:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    attention_head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    attention_bias: bool
    mlp_bias: bool
    moe_intermediate_size: int
    num_experts: int
    moe_topk: int
    num_shared_expert: int
    cfg_distilled: bool
    use_meanflow: bool
    model_type: str
    sequence_template: str
    pad_token_id: int
    image_token_id: int
    patch_size: int
    patch_embed_hidden_dim: int
    image_base_size: int
    vae_latent_channels: int
    vit_aligner: dict


def _uniform(value, name):
    if isinstance(value, (list, tuple)):
        if len(set(value)) != 1:
            raise ValueError(f"{name} must be uniform across layers")
        return value[0]
    return value


def params_from_config(config):
    if config.get("moe_layer_num_skipped", 0) != 0 or config.get("use_mixed_mlp_moe") is not True or config.get("hidden_act") != "silu":
        raise ValueError("unsupported HunyuanImage-3.0 configuration")
    return HunyuanImage3Params(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        attention_head_dim=config["attention_head_dim"],
        rms_norm_eps=config["rms_norm_eps"],
        rope_theta=config["rope_theta"],
        max_position_embeddings=config["max_position_embeddings"],
        attention_bias=config["attention_bias"],
        mlp_bias=config["mlp_bias"],
        moe_intermediate_size=_uniform(config["moe_intermediate_size"], "moe_intermediate_size"),
        num_experts=_uniform(config["num_experts"], "num_experts"),
        moe_topk=_uniform(config["moe_topk"], "moe_topk"),
        num_shared_expert=_uniform(config["num_shared_expert"], "num_shared_expert"),
        cfg_distilled=config.get("cfg_distilled", True),
        use_meanflow=config.get("use_meanflow", True),
        model_type=config.get("model_type_hy3", "instruct_distil"),
        sequence_template=config.get("sequence_template", "instruct"),
        pad_token_id=config["pad_token_id"],
        image_token_id=config["image_token_id"],
        patch_size=config.get("patch_size", 1),
        patch_embed_hidden_dim=config.get("patch_embed_hidden_dim", 1024),
        image_base_size=config.get("image_base_size", 1024),
        vae_latent_channels=config["vae"]["latent_channels"],
        vit_aligner=config["vit_aligner"],
    )


# ----------------------------------------------------------------------------------------- geometry

def _as_list(value):
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def sequence_from_ids(ids, token_height, token_width, config, cond_latents=None, cond_grids=None):
    """Per-request sequence geometry, derived from the token ids (ComfyUI-HunyuanImage3 `_sequence_from_ids`).

    Each conditioning image contributes two runs of `<img>` (its VAE latent, then its tower patches),
    the generated image one more run at the end."""
    ids = ids.to(torch.long)
    is_image = (ids == config.image_token_id).tolist()
    runs, run_start = [], None
    for index, flag in enumerate(is_image):
        if flag and run_start is None:
            run_start = index
        elif not flag and run_start is not None:
            runs.append(slice(run_start, index))
            run_start = None
    if run_start is not None:
        runs.append(slice(run_start, len(is_image)))

    cond_latents, cond_grids = _as_list(cond_latents), _as_list(cond_grids)
    expected = 2 * len(cond_latents) + 1
    if len(runs) != expected:
        raise ValueError(f"the token sequence carries {len(runs)} image runs, {expected} expected")

    step = {"ids": ids, "cond_blocks": [], "full_attention_slices": [], "rope_image_info": []}
    for index, (latent, grid) in enumerate(zip(cond_latents, cond_grids)):
        vae, vit = runs[2 * index], runs[2 * index + 1]
        height, width = latent.shape[-2], latent.shape[-1]
        patch_height, patch_width = (int(value) for value in grid)
        vae = slice(vae.start, vae.start + height * width)
        step["cond_blocks"].append({"vae_slice": vae, "vit_slice": vit, "timestep_position": vae.start - 1,
                                    "token_height": height, "token_width": width})
        step["full_attention_slices"].append(slice(vae.start, vit.stop))
        step["rope_image_info"] += [(vae, (height, width)), (vit, (patch_height, patch_width))]

    gen = runs[-1]
    position = gen.start - 1 - (1 if config.cfg_distilled else 0) - (1 if config.use_meanflow else 0)
    step.update({
        "image_slice": slice(gen.start, gen.start + token_height * token_width),
        "token_height": token_height,
        "token_width": token_width,
        "timestep_position": position,
        "guidance_position": position + 1 if config.cfg_distilled else None,
        "timestep_r_position": position + 1 + int(config.cfg_distilled) if config.use_meanflow else None,
    })
    step["full_attention_slices"].append(gen)
    step["rope_image_info"].append((gen, (token_height, token_width)))
    return step


def build_rope(seq_len, head_dim, rope_image_info, base, device):
    """2D rope as (cos, sin), each (seq_len, head_dim // 2), in fp32 (reference `build_2d_rope`)."""
    pairs = head_dim // 2
    positions = torch.zeros(seq_len, 2, dtype=torch.float32, device=device)
    text_positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    grid_scale = max(1.0, max((math.sqrt(h * w) / _TRAINED_GRID for _, (h, w) in rope_image_info), default=1.0))
    last_pos = 0
    for section, (height, width) in rope_image_info:
        start = section.start
        if last_pos < start:
            positions[last_pos:start, 0] = text_positions[last_pos:start]
            positions[last_pos:start, 1] = text_positions[last_pos:start]
        beta_y = start + (width * height - height) / 2
        beta_x = start + (width * height - width) / 2
        index = torch.arange(height * width, dtype=torch.float32, device=device)
        positions[start:start + height * width, 0] = (beta_y + torch.div(index, width, rounding_mode="floor")).trunc()
        positions[start:start + height * width, 1] = (beta_x + index % width).trunc()
        last_pos = start + height * width
    positions[last_pos:, 0] = text_positions[last_pos:]
    positions[last_pos:, 1] = text_positions[last_pos:]
    if grid_scale > 1.0:
        base = base * grid_scale ** (head_dim / (head_dim - 2))
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    angles = positions[:, torch.arange(pairs, device=device) % 2] * theta
    return torch.cos(angles), torch.sin(angles)


def build_attention_mask(step, seq_len, dtype, device):
    """Additive mask (1, 1, S, S): causal, plus bidirectional inside every image block.

    The key axis is allocated padded to a multiple of 16 so the SDPA memory-efficient kernel can use
    the mask in place (stride alignment); the returned tensor is the [..., :S] view."""
    padded = (seq_len + 15) // 16 * 16
    keep = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril_(diagonal=0)
    for full in step["full_attention_slices"]:
        keep[full, full] = True
    mask = torch.full((1, 1, seq_len, padded), float("-inf"), dtype=dtype, device=device)
    mask[0, 0, :, :seq_len].masked_fill_(keep, 0.0)
    del keep
    return mask[..., :seq_len]


def apply_rope(x, cos, sin):
    """Split-half rotation, computed in fp32 (comfy_kitchen `apply_rope_split_half`)."""
    half = x.shape[-1] // 2
    xf = x.float()
    x1, x2 = xf[..., :half], xf[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).type_as(x)


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ----------------------------------------------------------------------------------------- modules

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * xf.to(dtype)


class LightProjector(nn.Module):
    """vision_aligner: Linear -> GELU -> Linear (mlp_gelu, depth 2)."""

    def __init__(self, config):
        super().__init__()
        if config["projector_type"] != "mlp_gelu":
            raise ValueError(f"unsupported vit_aligner projector_type {config['projector_type']!r}")
        layers = [nn.Linear(config["input_dim"], config["n_embed"])]
        for _ in range(1, config["depth"]):
            layers.append(nn.GELU())
            layers.append(nn.Linear(config["n_embed"], config["n_embed"]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size, bias=True), nn.GELU(),
                                 nn.Linear(hidden_size, hidden_size, bias=True))

    def forward(self, t):
        # sinusoids in fp32, then the weight dtype, as the reference's `.type(self.mlp[0].weight.dtype)`
        t_freq = timestep_embedding(t, self.frequency_embedding_size).to(self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


class ResBlock(nn.Module):
    def __init__(self, in_channels, emb_channels, out_channels):
        super().__init__()
        self.in_layers = nn.Sequential(nn.GroupNorm(32, in_channels), nn.SiLU(),
                                       nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * out_channels))
        self.out_layers = nn.Sequential(nn.GroupNorm(32, out_channels), nn.SiLU(), nn.Identity(),
                                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.skip_connection = nn.Identity() if out_channels == in_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        while emb_out.dim() < h.dim():
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_layers[0](h) * (1.0 + scale) + shift
        h = self.out_layers[1](h)
        h = self.out_layers[3](h)
        return self.skip_connection(x) + h


class UNetDown(nn.Module):
    def __init__(self, in_channels, emb_channels, hidden_channels, out_channels):
        super().__init__()
        self.model = nn.ModuleList([nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
                                    ResBlock(hidden_channels, emb_channels, out_channels)])

    def forward(self, x, t):
        x = self.model[0](x)
        x = self.model[1](x, t)
        token_h, token_w = x.shape[-2:]
        return x.flatten(2).transpose(1, 2), token_h, token_w


class UNetUp(nn.Module):
    def __init__(self, in_channels, emb_channels, hidden_channels, out_channels):
        super().__init__()
        self.model = nn.ModuleList([
            ResBlock(in_channels, emb_channels, hidden_channels),
            nn.Sequential(nn.GroupNorm(32, hidden_channels), nn.SiLU(),
                          nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)),
        ])

    def forward(self, x, t, token_h, token_w):
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], token_h, token_w)
        x = self.model[0](x, t)
        return self.model[1](x)


def _swiglu(gate_up):
    x1, x2 = gate_up.chunk(2, dim=-1)
    return x1 * F.silu(x2)


def _attention(query, key, value, mask, chunk=4096):
    """SDPA with the additive mask; queries are chunked on long sequences to bound peak memory."""
    seq_len = query.shape[-2]
    if seq_len <= 2 * chunk:
        return F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    out = torch.empty_like(query)
    for start in range(0, seq_len, chunk):
        end = min(start + chunk, seq_len)
        out[:, :, start:end] = F.scaled_dot_product_attention(query[:, :, start:end], key, value,
                                                              attn_mask=mask[:, :, start:end])
    return out


class HunyuanImage3Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.attention_head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        q_size = self.head_dim * self.num_heads
        kv_size = self.head_dim * self.num_key_value_heads
        self.qkv_proj = nn.Linear(config.hidden_size, q_size + 2 * kv_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(q_size, config.hidden_size, bias=config.attention_bias)
        self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, cos, sin, mask):
        bsz, q_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        # fused layout per key/value head: [q * groups, k, v]
        qkv = qkv.reshape(bsz, q_len, self.num_key_value_heads, self.num_key_value_groups + 2, self.head_dim)
        query, key, value = torch.split(qkv, [self.num_key_value_groups, 1, 1], dim=3)
        query = query.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        del qkv
        # rope goes in before the qk norm in this model
        query = self.query_layernorm(apply_rope(query, cos, sin))
        key = self.key_layernorm(apply_rope(key, cos, sin))
        key = key.repeat_interleave(self.num_key_value_groups, dim=1)
        value = value.repeat_interleave(self.num_key_value_groups, dim=1)
        out = _attention(query, key, value, mask)
        del query, key, value
        return self.o_proj(out.transpose(1, 2).reshape(bsz, q_len, -1))


class HunyuanImage3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, bias=False):
        super().__init__()
        self.gate_and_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x):
        return self.down_proj(_swiglu(self.gate_and_up_proj(x)))


class HunyuanImage3MoEGate(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.wg = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states):
        probabilities = self.wg(hidden_states).float().softmax(dim=-1)
        top_k_weights, top_k_index = torch.topk(probabilities, self.top_k, dim=-1)
        return top_k_weights / top_k_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8), top_k_index


class HunyuanImage3MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.moe_topk
        self.shared_mlp = HunyuanImage3MLP(config.hidden_size, config.moe_intermediate_size * config.num_shared_expert, bias=config.mlp_bias)
        self.gate = HunyuanImage3MoEGate(config.hidden_size, self.num_experts, self.top_k)
        self.experts_gate_up_proj = W4A8ExpertBank(self.num_experts, config.hidden_size, 2 * config.moe_intermediate_size)
        self.experts_down_proj = W4A8ExpertBank(self.num_experts, config.moe_intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        bsz, seq_len, hidden_size = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_size)
        top_k_weights, top_k_index = self.gate(flat)
        top_k_weights = top_k_weights.to(hidden_states.dtype).reshape(-1)
        routes = top_k_index.reshape(-1)                       # flat position = token * top_k + slot
        order = torch.argsort(routes, stable=True)
        counts = torch.bincount(routes, minlength=self.num_experts).tolist()
        combined = torch.zeros((routes.shape[0], hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
        start = 0
        for expert, count in enumerate(counts):
            if count == 0:
                continue
            positions = order[start:start + count]
            start += count
            tokens = flat[positions // self.top_k]
            expert_out = self.experts_down_proj(_swiglu(self.experts_gate_up_proj(tokens, expert)), expert)
            combined[positions] = (expert_out * top_k_weights[positions, None]).to(combined.dtype)
            del tokens, expert_out
        # (N * top_k, hidden) summed rather than index_add_ in bf16: more accurate accumulation
        routed = combined.view(bsz, seq_len, self.top_k, hidden_size).sum(dim=2)
        del combined
        return self.shared_mlp(hidden_states) + routed


class HunyuanImage3DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = HunyuanImage3Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = HunyuanImage3MoE(config)

    def forward(self, hidden_states, cos, sin, mask):
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states), cos, sin, mask)
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class HunyuanImage3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([HunyuanImage3DecoderLayer(config) for _ in range(config.num_hidden_layers)])


class HunyuanImage3Transformer(nn.Module):
    def __init__(self, config: HunyuanImage3Params):
        super().__init__()
        if config.patch_size != 1:
            raise ValueError("only patch size 1 is supported")
        self.config = config
        self._interrupt = False
        self.model = HunyuanImage3Model(config)
        self.vision_aligner = LightProjector(config.vit_aligner)
        self.patch_embed = UNetDown(config.vae_latent_channels, config.hidden_size, config.patch_embed_hidden_dim, config.hidden_size)
        self.final_layer = UNetUp(config.hidden_size, config.hidden_size, config.patch_embed_hidden_dim, config.vae_latent_channels)
        self.time_embed = TimestepEmbedder(config.hidden_size)
        self.time_embed_2 = TimestepEmbedder(config.hidden_size)
        self.timestep_emb = TimestepEmbedder(config.hidden_size)
        if config.cfg_distilled:
            self.guidance_emb = TimestepEmbedder(config.hidden_size)
        if config.use_meanflow:
            self.timestep_r_emb = TimestepEmbedder(config.hidden_size)

    def align_vision(self, tokens):
        return self.vision_aligner(tokens)

    def forward(self, inputs):
        """One denoising step: rebuild the input embeddings and return the velocity (1, C, h, w).

        `inputs` is a dict (latents, timestep, timestep_r, guidance, step, cos, sin, mask, cond_latents,
        cond_vits): mmgp casts fp32 tensors passed directly to a root forward to bf16, which would round
        the timesteps and the rope tables; tensors inside a dict are left alone.

        The whole sequence is recomputed every step (no KV cache): every token ahead of the generated
        block is causal or inside its own image block, so recomputing it reproduces the cached values."""
        latents, timestep, timestep_r, guidance = inputs["latents"], inputs["timestep"], inputs["timestep_r"], inputs["guidance"]
        step, cos, sin, mask = inputs["step"], inputs["cos"], inputs["sin"], inputs["mask"]
        cond_latents, cond_vits = inputs.get("cond_latents", ()), inputs.get("cond_vits", ())
        ids = step["ids"]
        embeds = self.model.wte(ids.unsqueeze(0))
        dtype = embeds.dtype
        image_emb, token_h, token_w = self.patch_embed(latents.to(dtype), self.time_embed(timestep))
        embeds[0, step["image_slice"]] = image_emb[0].to(dtype)
        embeds[0, step["timestep_position"]] = self.timestep_emb(timestep)[0].to(dtype)
        if step["guidance_position"] is not None:
            embeds[0, step["guidance_position"]] = self.guidance_emb(guidance)[0].to(dtype)
        if step["timestep_r_position"] is not None:
            embeds[0, step["timestep_r_position"]] = self.timestep_r_emb(timestep_r)[0].to(dtype)
        del image_emb
        blocks = step["cond_blocks"]
        if blocks:
            # conditioning images are clean: t = 0 (reference `vae_encode`)
            clean = torch.zeros(1, dtype=torch.float32, device=ids.device)
            clean_time, clean_timestep = self.time_embed(clean), self.timestep_emb(clean)[0].to(dtype)
            for block, latent, vit in zip(blocks, cond_latents, cond_vits):
                cond_emb, _, _ = self.patch_embed(latent.to(device=ids.device, dtype=dtype), clean_time)
                embeds[0, block["vae_slice"]] = cond_emb[0].to(dtype)
                embeds[0, block["timestep_position"]] = clean_timestep
                embeds[0, block["vit_slice"]] = vit.reshape(-1, embeds.shape[-1]).to(device=ids.device, dtype=dtype)
                del cond_emb

        hidden = embeds
        del embeds
        for layer in self.model.layers:
            if self._interrupt:
                return None
            hidden = layer(hidden, cos, sin, mask)
        image_hidden = hidden[0, step["image_slice"]]
        del hidden
        return self.final_layer(image_hidden.unsqueeze(0), self.time_embed_2(timestep), token_h, token_w)
