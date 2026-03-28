"""Reference implementation for RMS normalization."""

import torch


def rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_f32 = x.float()
    rms = torch.sqrt(torch.mean(x_f32 ** 2, dim=-1, keepdim=True) + eps)
    out = x_f32 / rms * weight.float()
    return out.to(x.dtype)
