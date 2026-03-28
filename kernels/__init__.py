"""Kernel library for cuda-evolve.

Each kernel module should export:
  - KERNEL_TYPE: str           -- kernel identifier (must match a config in kernel_configs/)
  - kernel_fn(**inputs) -> torch.Tensor (or tuple)
  - get_inputs() -> dict
  - get_flops() -> int  (optional, for roofline)
  - get_bytes() -> int  (optional, for roofline)

Usage:
  cp kernels/<kernel_name>.py kernel.py   # select a kernel to optimize
  uv run tools/bench.py                   # benchmark it

To add a kernel, create a .py file in this directory implementing the interface
above, then add a matching config pair in kernel_configs/ (<name>.toml + <name>.py)
and a reference implementation in references/.
"""

AVAILABLE_KERNELS: list[str] = [
    # Add your kernel names here, e.g.:
    # "rms_norm",
    # "matmul",
]
