"""Reference implementation for SwiGLU + input FP8 quantization."""

import torch


def swiglu_input_quant_ref(x: torch.Tensor, eps: float = 1e-15) -> tuple:
    m, n2 = x.shape
    n = n2 // 2
    x0, x1 = x[:, :n], x[:, n:]

    x1_f32 = x1.float()
    out = (x0 * (x1_f32 * torch.sigmoid(x1_f32))).to(x.dtype)

    block_size = 128
    n_blocks = n2 // block_size
    x_fp8 = torch.empty(m, n2, dtype=torch.float8_e4m3fn, device=x.device)
    x_scale = torch.empty(n_blocks, m, dtype=torch.float32, device=x.device)

    for j in range(n_blocks):
        col_start = j * block_size
        block = x[:, col_start:col_start + block_size]
        row_max = block.float().abs().amax(dim=1).clamp(min=eps)
        scale = row_max / 448.0
        quantized = (block * (1.0 / scale[:, None]).to(block.dtype))
        x_fp8[:, col_start:col_start + block_size] = quantized.to(torch.float8_e4m3fn)
        x_scale[j] = scale

    return out, x_fp8, x_scale
