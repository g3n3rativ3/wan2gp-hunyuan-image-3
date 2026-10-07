# Routed-expert banks of HunyuanImage-3.0 stored in comfy_kitchen's grouped W4A8 format
# (`AsymW4A8Int8Layout`: ConvRot-rotated weights, 4-bit Lloyd-Max codes, fp8 group scales, fp32
# channel scales), as written by ComfyUI-HunyuanImage3's `tools/convert_w4a8.py`.
#
# One bank stacks the 64 experts of one projection of one layer:
#   qdata     int8          (E, N, K/2)   two 4-bit codes per byte, even column = low nibble
#   s_rel     float8_e4m3fn (E, N, K/16)  per-group relative scale (stored as uint8 in the file)
#   s_channel float32       (E, N)        per-row scale
#   codebook  float32       (E, 16) or (16,)
#
# WanGP already ships the matching 2-D kernels (`shared/qtypes/asym_w4a8_int8.py`: Triton decode to the
# int8 grid + int8 GEMM, with a portable torch fallback). The banks are 3-D, which that handler does not
# claim, so this module slices one expert at a time and hands it to the same kernel.
import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP_SIZE = 16
CONVROT_GROUP_SIZE = 256
NUM_EXPERTS = 64

try:  # WanGP's W4A8 linear (Triton decode + int8 GEMM)
    from shared.qtypes.asym_w4a8_int8 import _w4a8_linear as _wangp_w4a8_linear
except Exception:  # pragma: no cover - outside WanGP
    _wangp_w4a8_linear = None

_HADAMARD = {}


def _hadamard(size, device, dtype):
    key = (size, str(device), dtype)
    h = _HADAMARD.get(key)
    if h is None:
        h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32, device="cpu")
        h = h4
        while h.shape[0] < size:
            h = torch.kron(h, h4)
        h = (h / size ** 0.5).to(device=device, dtype=dtype)
        _HADAMARD[key] = h
    return h


def _eager_w4a8_linear(x, qdata, s_rel, s_channel, codebook):
    """Portable fallback: decode to the rotated basis and run the GEMM on the rotated activation.

    W = W_rot @ H^T per 256-group and H is symmetric orthonormal, so x @ W^T = (x @ H) @ W_rot^T."""
    n, k_half = qdata.shape
    k = k_half * 2
    packed = qdata.view(torch.uint8)
    codes = torch.empty((n, k), dtype=torch.long, device=qdata.device)
    codes[:, 0::2] = (packed & 0xF).long()
    codes[:, 1::2] = (packed >> 4).long()
    values = codebook.to(device=qdata.device, dtype=torch.float32)[codes] if codebook is not None else codes.float() - 8
    values = (values.view(n, -1, GROUP_SIZE) * s_rel.float().unsqueeze(-1)).round_().clamp_(-127, 127)
    weight_rot = (values.view(n, k) * s_channel.float().unsqueeze(1)).to(x.dtype)
    h = _hadamard(CONVROT_GROUP_SIZE, x.device, x.dtype)
    shape = x.shape
    x_rot = torch.matmul(x.reshape(-1, k // CONVROT_GROUP_SIZE, CONVROT_GROUP_SIZE), h).reshape(shape)
    return F.linear(x_rot, weight_rot)


class _ExpertWeightView:
    """The attribute set `shared.qtypes.asym_w4a8_int8._w4a8_linear` reads from a weight."""

    __slots__ = ("_data", "_s_rel", "_s_channel", "_codebook", "_correction", "_group_size",
                 "_convrot_group_size", "shape")

    def __init__(self, qdata, s_rel, s_channel, codebook):
        self._data = qdata
        self._s_rel = s_rel
        self._s_channel = s_channel
        self._codebook = codebook
        self._correction = None
        self._group_size = GROUP_SIZE
        self._convrot_group_size = CONVROT_GROUP_SIZE
        self.shape = (qdata.shape[0], qdata.shape[1] * 2)


class W4A8ExpertBank(nn.Module):
    # mmgp: keep the int8 / fp8 / fp32 tensors in their own dtypes (no float conversion, no dtype check)
    _lock_dtype = None

    def __init__(self, num_experts, in_features, out_features):
        super().__init__()
        if in_features % CONVROT_GROUP_SIZE or in_features % GROUP_SIZE:
            raise ValueError(f"W4A8 bank needs in_features divisible by {CONVROT_GROUP_SIZE}, got {in_features}")
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.qdata = nn.Parameter(torch.empty((num_experts, out_features, in_features // 2), dtype=torch.int8), requires_grad=False)
        self.s_rel = nn.Parameter(torch.empty((num_experts, out_features, in_features // GROUP_SIZE), dtype=torch.float8_e4m3fn), requires_grad=False)
        self.s_channel = nn.Parameter(torch.empty((num_experts, out_features), dtype=torch.float32), requires_grad=False)
        self.codebook = nn.Parameter(torch.empty((num_experts, 16), dtype=torch.float32), requires_grad=False)

    def forward(self, x, expert):
        qdata = self.qdata[expert]
        if qdata.device != x.device:  # should not happen under mmgp, but stay correct if it does
            qdata = qdata.to(x.device)
        # fp8 -> fp16 is exact and keeps the Triton decode independent of fp8 support on the GPU
        s_rel = self.s_rel[expert].to(device=x.device).to(torch.float16)
        s_channel = self.s_channel[expert].to(device=x.device, dtype=torch.float32)
        codebook = self.codebook[expert] if self.codebook.ndim == 2 else self.codebook
        codebook = codebook.to(device=x.device, dtype=torch.float32).contiguous()
        if _wangp_w4a8_linear is not None:
            return _wangp_w4a8_linear(x, _ExpertWeightView(qdata.contiguous(), s_rel.contiguous(), s_channel.contiguous(), codebook))
        return _eager_w4a8_linear(x, qdata, s_rel, s_channel, codebook)


def preprocess_w4a8_state_dict(state_dict, quantization_map=None, tied_weights_map=None):
    """Adapt a ComfyUI-HunyuanImage3 W4A8 checkpoint to this model before mmgp's quantization detection.

    * drops ComfyUI's per-module `comfy_quant` descriptors and the text head (prompt rewriting only);
    * expert banks: renamed to this module's parameters, `s_rel` viewed back to fp8 (the file stores the
      bytes as uint8), a per-expert codebook table is kept as is;
    * dense projections: `weight_s_rel` converted to fp16 (exact) so WanGP's 2-D W4A8 handler reads real
      scales (it would otherwise read the raw bytes as numbers).
    """
    out = {}
    for key, tensor in state_dict.items():
        if key.endswith(".comfy_quant") or key.startswith(("lm_head.", "model.ln_f.", "vae.", "vision_model.")):
            continue
        if ".mlp.experts_" in key:
            base, _, field = key.rpartition(".")
            if field == "weight":
                out[base + ".qdata"] = tensor
            elif field == "weight_s_rel":
                out[base + ".s_rel"] = tensor.view(torch.float8_e4m3fn) if tensor.dtype == torch.uint8 else tensor.to(torch.float8_e4m3fn)
            elif field == "weight_s_channel":
                out[base + ".s_channel"] = tensor
            elif field == "weight_codebook":
                out[base + ".codebook"] = tensor if tensor.ndim == 2 else tensor.reshape(1, -1).expand(NUM_EXPERTS, -1).contiguous()
            else:
                out[key] = tensor
            continue
        if key.endswith(".weight_s_rel"):
            if tensor.dtype == torch.uint8:
                tensor = tensor.view(torch.float8_e4m3fn)
            out[key] = tensor.to(torch.float16)
            continue
        out[key] = tensor
    return out, quantization_map, tied_weights_map
