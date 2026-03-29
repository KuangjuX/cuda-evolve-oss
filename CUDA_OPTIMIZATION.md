# CUDA Kernel Optimization Guide

This document is **maintained by the optimization agent**. As kernels are optimized, the agent summarizes effective optimization strategies here, organized by kernel type and by cross-kernel pattern. This serves as a growing knowledge base for future optimization runs.

When investigating a specific bottleneck, read the relevant files in `docs/` directly (e.g. `docs/stall_reasons.md`, `docs/memory_optimization.md`, `docs/compute_optimization.md`, `docs/arch_notes.md`).

---

## rms_norm (Per-Row RMSNorm)

### Characteristics
- Bottleneck: memory-bound (arithmetic intensity ~3, well below ridge point)
- Data access: streaming reads (input), streaming writes (output), broadcast read (gamma)
- Per-row reduction (sum of squares) followed by element-wise normalization
- Typical sizes: M=2048-4096 rows, N=1024-5120 columns, bf16/fp16

### Effective Optimizations

1. **Maximize occupancy via grid sizing** (40% latency reduction): `[occupancy]` `[launch-config]`
   - Baseline launched 132 blocks (1/SM) with persistent thread loop → 6.25% occupancy
   - Increasing grid to rows (one block per row) → 65% occupancy
   - Key: check NCU "Block Limit Registers" to know max blocks/SM, then size grid accordingly
   - Expected speedup: 1.4-1.8x

2. **Inline helper functions to reduce register pressure** (registers: 96 → 39): `[register-pressure]` `[occupancy]`
   - The `_do_rms_norm` helper function inflated register count due to call overhead
   - Inlining and removing unnecessary type conversions (`hidden.to(gamma.dtype).to(tl.float32)`) cut registers by 60%
   - Lower registers → more blocks/SM → higher theoretical occupancy (31% → 75%)
   - Expected impact: enables other occupancy optimizations

3. **Row-per-block launch instead of persistent threads** (~5% improvement): `[launch-config]` `[occupancy]`
   - For M ≥ 1024, `grid=(M,)` outperforms persistent kernel with `tl.range` loop
   - Eliminates loop overhead, branch resolution stalls, and software pipelining register cost
   - Hardware wave dispatch is efficient for these grid sizes
   - Expected speedup: 1.03-1.05x over persistent with same occupancy

4. **L2 eviction policy hints** (~1.5% improvement): `[cache]` `[memory-access]`
   - `evict_last` for input loads (keep briefly for coalescing across warps)
   - `evict_first` for output stores (streaming write, never re-read)
   - Expected speedup: 1.01-1.02x

5. **Triton autotune for num_warps** (~1% improvement): `[launch-config]`
   - Optimal num_warps varies by column width (N)
   - num_warps=8 optimal for N=4096 on H800
   - Search space: num_warps=[4,8,16,32], num_stages=[1,2]

### Anti-patterns (things that didn't work)

- **num_warps=16 or 32 for N=4096**: Too many threads per block → fewer blocks/SM → lower occupancy
- **Persistent threads with low register inlined kernel**: Loop overhead + software pipelining register cost negated occupancy gains (32 us vs 29 us)
- **2 rows per block**: Branch overhead from bounds checking outweighed dispatch savings
- **evict_first for both load and store**: Input data benefits from brief L2 residency
- **int32 offsets**: Triton already handles offset optimization internally; explicit int32 can generate worse code
- **Division → multiplication by reciprocal**: Compiler already optimizes this for constexpr divisors

---

## qkv_part_rope (QKV Partial Rotary Position Embedding)

### Characteristics
- Bottleneck: memory-bound (arithmetic intensity ~0.34, deeply below ridge point of ~295)
- Data access: streaming read+write of packed QKV tensor, broadcast read of cos/sin tables
- 77% of data is pure copy (nope + V heads), 23% has fp32 rope computation
- CUDA kernel with persistent thread model
- Typical sizes: batch=2, seq=4096, q_heads=10, kv_heads=1, head_dim=256, nope_dim=192

### Effective Optimizations

1. **Increase SeqTile from 2 to 4** (19% latency reduction): `[tile-size]` `[register-pressure]`
   - Doubles work per tile → halves number of tiles (scheduler iterations)
   - Better amortization of per-tile overhead: scheduler coord computation, cos/sin loads
   - ElemPerThread goes from 1 to 2, improving instruction-level parallelism
   - Registers increase from 31 to 60/thread (still fits 1 block of 1024/SM)
   - Expected speedup: 1.15-1.20x
   - NOTE: SeqTile=8 causes register spilling (regression). SeqTile=4 is the sweet spot.

2. **Float4 (16-byte) nope copy** (~1.3% additional improvement): `[memory-coalescing]` `[vectorized-loads]`
   - Widen nope copy from float2 (8B) to float4 (16B) loads/stores
   - Halves instruction count for the dominant 77%-of-traffic nope copy path
   - Small but real gain when combined with SeqTile=4
   - Standalone (without SeqTile increase): negligible improvement

### Anti-patterns (things that didn't work)

- **Doubling grid size (2 blocks/SM)**: With 31 regs/thread, 2 blocks of 1024 fit, giving 64 warps/SM. But performance WORSENED by 7%. 32 warps/SM already provides sufficient memory latency hiding for this access pattern. More warps increase L2 contention.
- **ld.global.lu (last-use) for nope loads**: Streaming hint evicts data that IS reused by adjacent warps in the same block (processing different heads at the same seq position). Caused 7% regression.
- **__launch_bounds__(_, 2) to force register reduction**: Compiler spills to local memory when constrained to fit 2 blocks/SM. Spilling overhead > occupancy benefit (6% regression).
- **SeqTile=8**: Too much register pressure. Registers exceed comfortable range, causing spilling and 20% regression vs SeqTile=4.

---

## swiglu_input_quant (SwiGLU + FP8 Blockwise Quantization)

### Characteristics
- Bottleneck: memory-bound (arithmetic intensity ~0.17, deeply below ridge point ~295)
- Data access: read BF16 input [M, 2N], write BF16 SwiGLU [M, N], write FP8 [M, 2N], write FP32 scales [2N/128, M]
- Multi-output kernel: 3 output tensors with mixed data types (BF16, FP8, FP32)
- Per-block-of-128-columns row-wise absmax scaling for FP8 quantization
- Scale stored in transposed layout (`block_idx_n * m + row_idx`) for downstream matmul
- Typical sizes: M=4096, N=7168, bf16

### Effective Optimizations

1. **Reduce tile size to cut register pressure** (27% total improvement): `[register-pressure]` `[occupancy]` `[tile-size]`
   - Baseline block_size_m=128 → 126 regs/thread → 6.25% occupancy
   - block_size_m=32 with num_warps=8 → 40 regs/thread → 70.9% occupancy
   - This is the single most impactful optimization: 11x occupancy increase
   - Expected speedup: 1.25-1.35x

2. **Non-persistent grid outperforms persistent for high tile count** (8% improvement): `[launch-config]` `[occupancy]`
   - With 7168 tiles across 132 SMs (~54 tiles/SM), hardware wave dispatch is efficient
   - L1 hit rate improved 0% → 32% from better spatial locality
   - Eliminates loop overhead, branch resolution stalls
   - Expected speedup: 1.05-1.10x over persistent with equivalent occupancy

3. **Increase grid to match SM block capacity** (17% improvement): `[occupancy]` `[launch-config]`
   - For persistent kernels: compute max blocks/SM from register + shared mem limits
   - Baseline: 132 blocks (1/SM). Optimal: 3 blocks/SM for this register profile
   - Expected speedup: 1.15-1.20x

4. **L2 eviction hints** (~0.7% improvement): `[cache]` `[memory-access]`
   - `evict_last` on input loads, `evict_first` on all stores
   - Smaller benefit than rms_norm since this kernel has 5 store streams polluting L2
   - Expected speedup: 1.005-1.01x

### Anti-patterns (things that didn't work)

- **num_warps=8 alone (without block_size_m reduction)**: With block_size_m=64, doubling warps from 4→8 didn't change occupancy (same register budget, same blocks/SM). No improvement.
- **Swizzled tile ordering (GROUP_SIZE_M=8)**: Integer arithmetic overhead for the swizzle mapping exceeded L2 locality benefit. 1% regression.
- **num_warps=16**: Too many threads per block reduced blocks/SM, negating the benefit. ~3% regression.
- **block_size_m=128 with any num_warps**: Always register-limited to ≤4 blocks/SM. Fundamental tile size problem.

---

## persistent_matmul (GEMM — C = A @ B)

### Characteristics
- Bottleneck: compute-bound (arithmetic intensity ~341, well above ridge point ~295)
- Data access: tiled reads of A and B matrices, tiled write of C
- Tensor core dominated: 89.7% HMMA utilization at optimal
- Typical sizes: M=2048-8192, N=2048-11008, K=512-8192, bf16

### Effective Optimizations

1. **Remove .to(tl.float32) before tl.dot** (7.7x speedup): `[tensor-core]` `[data-type]`
   - Baseline cast BF16 loads to FP32 before dot product, forcing scalar FP32 FMA
   - Removing cast enables native BF16 tensor cores with FP32 accumulation
   - Expected speedup: 5-10x (single most impactful change for any GEMM kernel)

2. **Expand autotune to proper GEMM tile sizes** (2x speedup on top of #1): `[tile-size]` `[tensor-core]`
   - Baseline had BLOCK_SIZE_N={16,32} — far too narrow for tensor cores
   - Optimal: BLOCK_SIZE_M=128, BLOCK_SIZE_N=128-256, BLOCK_SIZE_K=64-128
   - Expected: 128x128 or 128x256 tiles with 4-8 warps

3. **Use tl.make_block_ptr for TMA loads** (~7% speedup): `[memory-access]` `[register-pressure]`
   - Enables Hopper TMA (Tensor Memory Accelerator) for async memory operations
   - Reduces register pressure from manual pointer arithmetic
   - Better latency hiding via hardware-managed prefetch
   - Expected speedup: 1.05-1.10x

4. **Non-persistent grid for moderate tile counts** (~3% speedup): `[launch-config]`
   - Hardware scheduler outperforms persistent loop when tiles ≤ 4× num_SMs
   - Autotune selects smaller tiles (128x128 vs 128x256) for better load balance
   - Expected speedup: 1.02-1.05x

### Anti-patterns (things that didn't work)

- **Device-side TMA descriptors with B transpose**: `tl.make_tensor_descriptor` requires column-major B input; runtime `.T.contiguous()` copy costs more than TMA saves (2x regression)
- **flatten=True with block pointers on Hopper**: The `flatten` optimization is designed for `tl.make_tensor_descriptor`, not `tl.make_block_ptr` (regression)
- **Epilogue dependency breaking with block pointers**: Adding separate `tile_id_c` for epilogue PID computation adds overhead without benefit for block pointer stores
- **num_stages > 3 with 128x128 tiles**: 4-stage pipeline exceeds shared memory for 2 blocks/SM, forcing single-block occupancy

---

## dsa_forward (Dynamic Sparse Attention)

### Characteristics
- Bottleneck: compute-bound (attention with sparse block indices)
- Data access: Q (per-query-tile), K/V (per-block random access via block_indices), block_indices (streaming)
- Per-token baseline wastes tensor cores: matmul dims [16,128]x[128,64] too small
- GQA-aware: separate code paths for GQA (n_heads_block != n_heads_kv) vs non-GQA
- Typical sizes: batch=4, seq=2048, n_heads=32, n_heads_kv=8, head_dim=128, blk_siz=64

### Effective Optimizations

1. **FlashAttention-style Q-tiling** (8.4x speedup — most impactful): `[tile-size]` `[tensor-core]` `[algorithmic]`
   - Baseline: 1 query token per block → tiny matmuls ([16,128]x[128,64]) waste tensor cores
   - Tiled: BLOCK_Q=64 tokens per block → large matmuls ([64,128]x[128,64]) with K/V reuse
   - Grid changes from (seq_len, n_heads_block) to (num_q_tiles, n_heads)
   - Expected speedup: 5-10x (depending on matmul dimension ratio)

2. **Remove redundant tl.where on attention weights** (12% speedup): `[algorithmic]` `[warp-divergence]`
   - With b_m initialized to `-1e30` (not `-inf`), `exp(masked_score - b_m)` naturally produces ~0
   - `tl.where(mask, b_p, 0)` is mathematically redundant and wastes instructions
   - Insight: exploit numerical properties of online softmax

3. **Simplify causal mask** (3% speedup): `[algorithmic]` `[warp-divergence]`
   - Fold `q_valid` into `sr` (set sr=-1 for invalid queries) so the causal mask doesn't need a separate q_valid branch
   - Skip `sl` check entirely for non-sliding-window attention
   - Fewer mask ops = fewer instructions in the hot inner loop

4. **Pre-computed loop bound** (7% speedup): `[algorithmic]` `[warp-divergence]`
   - Replace `if blk_st <= tile_max_sr` branch inside the K/V block loop with pre-computed `n_valid = min(blk_cnt, max_sr // blk_siz + 1)`
   - Eliminates per-iteration branch and enables better instruction scheduling

### Anti-patterns (things that didn't work)

- **Remove early skip for pipelining**: Unconditionally processing all K/V blocks (including fully-masked ones) for "better pipelining" caused 59% regression. Causal early-exit is essential.
- **GQA-grouped sequential heads**: Grid=(num_q_tiles, n_heads_kv) with inner loop over GQA group to share K/V loads. Reduced parallelism hurts more than K/V reuse helps on H800 with enough heads. 15% regression.
- **V preload + fused rescale**: Loading V right after K to overlap with QK dot. Increased register pressure reduced occupancy. Slight regression.
- **Register pressure is the ceiling**: At ~46% peak compute, further gains require reducing register count per thread — a fundamental Triton compiler limitation for complex attention kernels.

---

## Cross-Kernel Optimization Patterns

Patterns below are indexed by **bottleneck type** rather than kernel. When the agent encounters a specific bottleneck on any kernel, check this section first for transferable techniques. Tags like `[register-pressure]` on per-kernel entries above map to the categories below.

### `[register-pressure]` — Reducing Registers per Thread

| Technique | Observed impact | Source kernel | Applicability |
|-----------|----------------|---------------|---------------|
| Inline helper functions | 60% register reduction (96→39) | rms_norm | Universal: any Triton/CUDA kernel using helper functions |
| Reduce tile size | 69% register reduction (126→40) | swiglu_input_quant | Any kernel where large tiles inflate register usage |
| Use `tl.make_block_ptr` instead of manual pointer arithmetic | ~7% speedup via register savings | persistent_matmul | Hopper+ with TMA support |
| Find the sweet spot tile size (not too small, not too large) | SeqTile=4 optimal, SeqTile=8 spills | qkv_part_rope | Memory-bound kernels with tunable tile dimensions |

**Diagnostic**: NCU `launch__registers_per_thread` > 64 → likely occupancy-limited. Check `launch__occupancy_limit_registers`.

### `[occupancy]` — Increasing Active Warps per SM

| Technique | Observed impact | Source kernel | Applicability |
|-----------|----------------|---------------|---------------|
| Match grid size to SM block capacity | 40% latency reduction | rms_norm | Any persistent kernel with low block count |
| Non-persistent grid for high tile counts | 8% improvement | swiglu_input_quant | When tiles >> SMs (>10x), hardware dispatch wins |
| Reduce tile size to lower registers | 11x occupancy increase | swiglu_input_quant | Register-limited kernels |

**Diagnostic**: NCU `sm__warps_active.avg.pct_of_peak_sustained_active` < 50% → investigate register and shared memory limits.

**Warning**: Increasing occupancy beyond ~50% rarely helps. At >75%, L1 contention can negate benefits (qkv_part_rope: 2 blocks/SM was 7% slower).

### `[memory-coalescing]` / `[vectorized-loads]` — Efficient Memory Access

| Technique | Observed impact | Source kernel | Applicability |
|-----------|----------------|---------------|---------------|
| Float4 (16-byte) vectorized copy | 1.3% improvement | qkv_part_rope | Any kernel with significant data copy paths |
| Vectorized loads via larger BLOCK_SIZE | Implicit in tile sizing | persistent_matmul | All Triton kernels |

**Diagnostic**: NCU coalescing ratio (`memory_l2_theoretical_sectors_global` / `ideal`) > 1.5 → access pattern needs fixing.

### `[cache]` — L1/L2 Cache Optimization

| Technique | Observed impact | Source kernel | Applicability |
|-----------|----------------|---------------|---------------|
| `evict_last` for inputs, `evict_first` for stores | 0.7-1.5% | rms_norm, swiglu | Streaming kernels (read once, write once) |
| Non-persistent grid improves L1 hit rate | 0%→32% L1 hit rate | swiglu_input_quant | When tiles have spatial locality |

**Warning**: `evict_first` on loads can backfire if data is reused by adjacent warps in the same block (qkv_part_rope: 7% regression).

### `[launch-config]` — Persistent vs Non-Persistent Grid

| When to use persistent | When to use non-persistent |
|------------------------|---------------------------|
| Tiles < 4x num_SMs | Tiles > 10x num_SMs |
| Complex scheduling logic | Simple tile-to-block mapping |
| Need cross-tile state | Stateless tiles |

**Empirical rule**: For tile counts > ~500 on H100 (132 SMs), non-persistent tends to win due to lower overhead and better L1 locality.

### `[tensor-core]` — Maximizing Tensor Core Utilization

| Technique | Observed impact | Source kernel | Applicability |
|-----------|----------------|---------------|---------------|
| Remove FP32 cast before `tl.dot` | 7.7x speedup | persistent_matmul | CRITICAL: any GEMM kernel |
| Increase matmul dimensions (Q-tiling) | 8.4x speedup | dsa_forward | Attention kernels with small per-token matmuls |
| Tile sizes ≥ 128 for M/N dimensions | 2x speedup | persistent_matmul | All GEMM-dominated kernels |

**Diagnostic**: NCU `sm__inst_executed_pipe_tensor.sum` = 0 → tensor cores not being used at all.

### `[warp-divergence]` / `[algorithmic]` — Branch Elimination

| Technique | Observed impact | Source kernel | Applicability |
|-----------|----------------|---------------|---------------|
| Remove redundant conditional operations | 12% speedup | dsa_forward | Any kernel with `tl.where` in hot loops |
| Pre-compute loop bounds | 7% speedup | dsa_forward | Loops with data-dependent exit conditions |
| Fold validity checks into data (set invalid=-1) | 3% speedup | dsa_forward | Kernels with per-element validity masks |
| Branchless rescaling (AVO technique) | 8.1% speedup | FlashAttention (AVO paper) | Online softmax, any conditional rescale |

**Diagnostic**: NCU `smsp__warps_issue_stalled_membar` high + branches in kernel → branchless conversion may help.

### Anti-patterns That Transfer Across Kernels

These failed consistently across multiple kernel types:

1. **Over-subscribing warps per SM**: Adding more blocks/SM beyond the latency-hiding sweet spot increases L2 contention. Failed on: qkv_part_rope, rms_norm.
2. **`__launch_bounds__` to force more blocks**: Compiler spills registers to local memory. Failed on: qkv_part_rope.
3. **num_warps too high**: Reduces blocks/SM without enough benefit. Failed on: rms_norm (num_warps=16/32), swiglu (num_warps=16).
4. **Swizzled tile ordering on small kernels**: Integer overhead exceeds L2 benefit. Failed on: swiglu_input_quant.
5. **Explicit int32 offsets in Triton**: Compiler already optimizes; manual downcasts can generate worse code. Failed on: rms_norm.
