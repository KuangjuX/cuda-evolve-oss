# Warp Stall Reasons Reference

Quick-reference for NCU warp stall types, root causes, and mitigations.

---

## long_scoreboard

**What**: Warp waiting for a data dependency from a long-latency operation (global memory load, texture fetch, or L2 access).

**Root causes**:
- Insufficient memory-level parallelism (not enough outstanding loads to hide latency)
- Sequential dependent loads (pointer chasing)
- Low occupancy providing insufficient warps to hide latency

**Mitigations**:
- Increase `num_stages` (Triton) or add software pipelining (`cp.async` in CUDA)
- Increase occupancy (reduce registers, reduce shared memory)
- Prefetch data to shared memory before compute phase
- Use `tl.load(..., eviction_policy=...)` hints
- Increase tile size to do more compute per load

---

## short_scoreboard

**What**: Warp waiting for result from shared memory or L1 cache operation.

**Root causes**:
- Bank conflicts in shared memory (multiple threads access same bank)
- High shared memory access frequency relative to compute
- L1 cache misses

**Mitigations**:
- Pad shared memory to avoid bank conflicts (`__shared__ float smem[N + 1]`)
- Restructure access patterns: sequential within warp, strided across warps
- Use `tl.load` with `cache_modifier` hints
- Reduce shared memory traffic by computing more per load

---

## wait

**What**: Warp waiting on `__syncthreads()` or other barrier.

**Root causes**:
- Frequent synchronization points in the kernel
- Workload imbalance: some warps finish early and wait at barrier
- Using barriers inside loops

**Mitigations**:
- Reduce number of `__syncthreads()` / `tl.debug_barrier()`
- Restructure to use warp-level primitives (`__shfl_sync`) instead of block-level sync
- Balance workload across warps
- Pipeline: overlap compute of iteration N with loads for iteration N+1

---

## mio_throttle

**What**: Memory instruction queue is full; too many outstanding memory operations.

**Root causes**:
- Kernel issues memory instructions faster than the memory subsystem can handle
- Many small loads instead of few large loads
- High memory instruction count per compute instruction

**Mitigations**:
- Use vectorized loads (`float4`, `tl.load` with larger block sizes)
- Coalesce memory accesses (contiguous threads access contiguous memory)
- Reduce total memory instructions: fuse operations, avoid redundant loads
- Increase arithmetic intensity (more compute per byte loaded)

---

## math_pipe_throttle

**What**: Compute pipeline is saturated; instructions are waiting for execution slots.

**Root causes**:
- Kernel is genuinely compute-bound (good sign if near peak)
- Instruction mix not optimal for the hardware (e.g., too many non-tensor-core ops)

**Mitigations**:
- Already near peak: look for algorithmic improvements to reduce FLOPs
- Use tensor cores for matrix ops (`tl.dot` with native BF16/FP16)
- Strength reduction: replace expensive ops (div, exp, log) with approximations
- If mixed-precision: ensure accumulator doesn't force scalar FP32 path

---

## barrier

**What**: Warp waiting at a named barrier (different from `__syncthreads`).

**Root causes**:
- Explicit barrier instructions in the kernel
- Warp-specialized kernels with producer/consumer barriers

**Mitigations**:
- Restructure producer/consumer pipeline for better overlap
- Reduce the number of barrier-protected critical sections
- Use async copy (`cp.async`) with arrive/wait patterns

---

## membar

**What**: Warp stalled on a memory fence instruction (`__threadfence`, etc.).

**Root causes**:
- Explicit memory ordering fences
- Blocking fences used where non-blocking would suffice
- Fences inside frequently-executed code paths

**Mitigations**:
- Replace `__threadfence()` with `__threadfence_block()` if cross-block ordering not needed
- Use non-blocking fence variants (acquire/release semantics)
- Move fence out of inner loops
- Eliminate branches so compiler can use lighter fence variants (AVO branchless rescaling technique)

---

## not_selected

**What**: Warp is eligible to issue but scheduler chose a different warp.

**Root causes**:
- More eligible warps than issue slots (normal at high occupancy)
- Not a problem unless very high — indicates good latency hiding

**Mitigations**:
- Generally not actionable; indicates healthy occupancy
- If very high: may be over-subscribed; consider reducing occupancy slightly

---

## sleeping

**What**: Warp explicitly put to sleep (rare, usually from nanosleep instruction).

**Root causes**:
- Spin-wait loops with explicit sleep
- Rarely seen in normal kernels

**Mitigations**:
- Replace spin-waits with proper synchronization primitives

---

## tex_throttle

**What**: Texture/L1 pipeline throttled.

**Root causes**:
- Too many texture fetch instructions outstanding
- L1 cache pressure from texture operations

**Mitigations**:
- Reduce texture fetch frequency
- Use direct global loads if texture filtering not needed
- Improve L1 cache locality

---

## no_instruction

**What**: No valid instruction to execute (instruction cache miss or end of warp).

**Root causes**:
- Instruction cache miss (very large kernels)
- Warp has exited but SM hasn't replaced it yet

**Mitigations**:
- Reduce kernel code size
- Avoid excessive loop unrolling
- Usually not actionable
