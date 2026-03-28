"""Reference implementation for QKV partial rotary position embedding."""

import torch


def qkv_part_rope_ref(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_heads: int = 10,
    kv_heads: int = 1,
    nope_dim: int = 192,
    negative_sin: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Pure PyTorch RoPE: split rope/nope dims, rotate Q/K, pass V through."""
    if qkv.dim() == 3:
        qkv = qkv.unsqueeze(0)

    batch, seq_len, num_heads, head_dim = qkv.shape
    rope_dim = head_dim - nope_dim
    half_rope = rope_dim // 2
    nqk_heads = q_heads + kv_heads

    out = qkv.clone()

    for h in range(nqk_heads):
        x_rope = qkv[:, :, h, nope_dim:].float()
        x0 = x_rope[..., :half_rope]
        x1 = x_rope[..., half_rope:]

        cos_exp = cos.unsqueeze(0).float()
        sin_exp = sin.unsqueeze(0).float()
        if negative_sin:
            sin_exp = -sin_exp

        out_0 = x0 * cos_exp - x1 * sin_exp
        out_1 = x0 * sin_exp + x1 * cos_exp

        out[:, :, h, nope_dim:nope_dim + half_rope] = out_0.to(qkv.dtype)
        out[:, :, h, nope_dim + half_rope:] = out_1.to(qkv.dtype)

    return out
