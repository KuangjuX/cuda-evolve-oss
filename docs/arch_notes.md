# GPU Architecture Quick Reference

Key specifications and quirks for optimization, organized by architecture.

---

## Hopper (SM 9.0) — H100, H800, H200

**Compute**:
- 132 SMs
- FP16/BF16 tensor cores: 989.5 TFLOPS (H100 SXM)
- FP32: ~60 TFLOPS (CUDA cores only)
- TMA (Tensor Memory Accelerator): hardware-managed async copy with address generation

**Memory**:
- HBM3: 80 GB, 3352 GB/s (H100 SXM) / 4800 GB/s (H200)
- L2 cache: 50 MB
- L1/shared memory: 256 KB per SM (configurable split)
- Register file: 65536 32-bit registers per SM

**Key features**:
- **TMA**: `tl.make_block_ptr` (Triton) or `cp.async.bulk` (CUDA). Handles multi-dimensional addressing and boundary checks in hardware. Reduces register pressure from pointer arithmetic.
- **Thread Block Clusters**: group blocks across SMs for distributed shared memory
- **Warp specialization**: different warp groups can be assigned different roles with independent register budgets
- **Asynchronous barriers**: `cp.async` with `arrive`/`wait` pattern for producer-consumer overlap

**Quirks**:
- `tl.make_tensor_descriptor` requires specific memory layouts (e.g., column-major B for GEMM)
- `flatten=True` only works with `tl.make_tensor_descriptor`, NOT with `tl.make_block_ptr`
- L2 partition camping can occur with certain grid launch patterns — use swizzled tile ordering
- Persistent kernels benefit from TMA but need careful barrier management

---

## Ada Lovelace (SM 8.9) — RTX 4090, L40S

**Compute**:
- 128 SMs (4090), 142 SMs (L40S)
- FP16 tensor cores: 330 TFLOPS (4090)
- Third-gen RT cores

**Memory**:
- GDDR6X: 24 GB, 1008 GB/s (4090) / 48 GB, 864 GB/s (L40S)
- L2 cache: 72 MB (4090) / 48 MB (L40S)

**Quirks**:
- Large L2 makes tile ordering less critical than on HBM GPUs
- GDDR6X has different latency characteristics than HBM
- Consumer cards: no NVLink, no MIG

---

## Ampere (SM 8.0) — A100

**Compute**:
- 108 SMs
- FP16/BF16 tensor cores: 312 TFLOPS (A100 SXM)
- TF32 tensor cores: 156 TFLOPS

**Memory**:
- HBM2e: 80 GB, 2039 GB/s (A100 SXM)
- L2 cache: 40 MB
- L1/shared memory: 192 KB per SM (configurable)

**Key differences from Hopper**:
- No TMA — must use `cp.async` manually
- No Thread Block Clusters
- `cp.async` supports up to 16 bytes per thread (float4)
- Async copy is SM-initiated, not hardware-addressed like TMA

**Quirks**:
- MIG (Multi-Instance GPU): can partition into 7 instances
- `__shfl_sync` works on full warp only (no sub-warp shuffle)
- A100-80GB PCIe has lower bandwidth (1935 GB/s) than SXM variant

---

## Blackwell (SM 10.0) — B200, B100

**Compute**:
- 2250 TFLOPS FP16 (B200)
- Fifth-gen tensor cores with FP4 support
- 2048 warp registers per SM (fixed allocation across warp groups)

**Memory**:
- HBM3e: 192 GB, 8000 GB/s (B200)
- L2 cache: 64 MB

**Key features**:
- **Fixed warp register budget**: 2048 registers per SM divided among warp groups. Must be explicitly balanced (e.g., 184/88/56 across 3 groups).
- **FP4 tensor cores**: 2x throughput over FP8
- **Second-gen TMA**: enhanced async copy capabilities

**Quirks**:
- Register spilling is catastrophic: local memory latency is very high relative to the fast tensor cores
- Register rebalancing between warp groups is a key optimization lever (AVO: +2.1%)
- The 2048-register budget is NOT per-warp — it's shared across all warp groups in the SM

---

## Common Specifications Lookup

| GPU | Arch | SMs | FP16 TFLOPS | BW (GB/s) | VRAM | L2 (MB) | Ridge Point (FLOPs/Byte) |
|-----|------|-----|-------------|-----------|------|---------|--------------------------|
| B200 | Blackwell | - | 2250 | 8000 | 192 GB | 64 | ~281 |
| H200 | Hopper | 132 | 989.5 | 4800 | 141 GB | 50 | ~206 |
| H100 SXM | Hopper | 132 | 989.5 | 3352 | 80 GB | 50 | ~295 |
| H800 | Hopper | 132 | 989.5 | 3352 | 80 GB | 50 | ~295 |
| A100 SXM | Ampere | 108 | 312 | 2039 | 80 GB | 40 | ~153 |
| 4090 | Ada | 128 | 330 | 1008 | 24 GB | 72 | ~327 |
| L40S | Ada | 142 | 362 | 864 | 48 GB | 48 | ~419 |
| L4 | Ada | 58 | 121 | 300 | 24 GB | 48 | ~403 |

Ridge point = FP16 TFLOPS * 1e12 / (BW * 1e9) = arithmetic intensity threshold.
Below ridge point: memory-bound. Above ridge point: compute-bound.
