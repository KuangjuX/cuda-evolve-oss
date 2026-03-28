"""Reference implementation for matrix multiplication."""

import torch


def matmul_ref(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.mm(A, B)
