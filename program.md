# cuda-evolve Program

You are an autonomous GPU kernel optimization agent. Follow this protocol strictly.

## Available Kernels

The `kernels/` directory contains **baseline (read-only)** kernels. **Never modify files in `kernels/`** — they serve as the unmodified reference.

Optimized kernels are saved to `kernels_optimized/`, which mirrors the structure of `kernels/`.

**Directory layout:**
```
kernels/                        # Baseline (READ-ONLY, never modify)
└── <your_kernel>.py            # Add your kernels here

kernels_optimized/              # Optimized versions (agent writes here)
└── <your_kernel>.py
```

Each kernel module must export:
- `KERNEL_TYPE: str` -- identifier matching a key in `bench.py` `KERNEL_CONFIGS`
- `kernel_fn(**inputs) -> torch.Tensor` (or tuple)
- `get_inputs() -> dict`
- `get_flops() -> int` (for roofline analysis)
- `get_bytes() -> int` (for roofline analysis)

## Setup Phase

1. Run `uv run prepare.py` to validate the environment (CUDA, GPU, dependencies).
2. Read `CUDA_OPTIMIZATION.md` to review optimization strategies discovered in previous runs. (This file is maintained by you — the agent — and may be empty on the first run.)
3. Read `MEMORY.md` for the global optimization summary across all kernels.
4. **Select a kernel** to optimize. Copy from the baseline `kernels/` directory (or from `kernels_optimized/` if a previous optimized version exists):
   ```bash
   # First run: start from baseline
   cp kernels/<your_kernel>.py kernel.py

   # Resuming: start from last optimized version (if it exists)
   cp kernels_optimized/<your_kernel>.py kernel.py
   ```
5. Read the per-kernel log in `memory/<kernel_type>.md` if it exists, to review past experiments for this specific kernel.
6. Read `kernel.py` to understand the current kernel implementation.
7. Read `reference.py` to understand the correctness specification.

## Experiment Loop

Repeat the following cycle:

### Step 1: Benchmark (baseline or after change)

Run the benchmark harness. It auto-detects `KERNEL_TYPE` from `kernel.py`:

```bash
uv run bench.py > run.log 2>&1
```

For quick iteration (skip numerical stability, determinism, edge cases):

```bash
uv run bench.py --quick > run.log 2>&1
```

Read `run.log` and extract the key metrics:

```bash
grep "correctness\|throughput_tflops\|speedup_vs_pytorch\|pct_peak_compute\|pct_peak_bandwidth\|bottleneck\|peak_vram_mb" run.log
```

The benchmark reports:
- **correctness**: 5-stage verification (smoke, shape sweep, numerical stability, determinism, edge cases)
- **throughput_tflops**: Achieved throughput
- **bandwidth_gb_s**: Achieved memory bandwidth
- **pct_peak_compute**: % of GPU's theoretical compute peak
- **pct_peak_bandwidth**: % of GPU's theoretical bandwidth peak
- **bottleneck**: `compute_bound` or `memory_bound` (from roofline analysis)
- **speedup_vs_pytorch**: Speedup vs PyTorch reference implementation

### Step 2: Macro Performance Analysis

Analyze the benchmark results to understand the kernel's **macro-level** performance characteristics:

1. **Compute throughput**: How close is `pct_peak_compute` to the GPU's theoretical peak?
2. **Memory bandwidth**: How close is `pct_peak_bandwidth` to the GPU's theoretical bandwidth?
3. **Bottleneck classification**: Is the kernel `compute_bound` or `memory_bound`?
4. **Roofline position**: Where does the kernel sit on the roofline? How far from the ridge point?

This gives you the **direction** of optimization (memory vs. compute), but not the **specific** cause.

### Step 3: NCU Deep Analysis

After understanding the macro picture, use NCU + ncu-cli to identify the **specific** bottleneck:

```bash
uv run ncu_profile.py > ncu.log 2>&1
```

Extract the key findings:

```bash
grep "ncu_bottleneck\|ncu_top_stall\|ncu_finding\|ncu_action\|ncu_occupancy\|ncu_l1_hit_rate\|ncu_l2_hit_rate" ncu.log
```

For targeted analysis (e.g., memory access patterns, warp stalls):

```bash
uv run ncu_profile.py --skills roofline,memory,warp_stall > ncu.log 2>&1
```

To compare before/after an optimization:

```bash
uv run ncu_profile.py --diff before.csv after.csv > ncu_diff.log 2>&1
```

**NCU analysis tells you the *specific* cause:**

- **Memory-bound kernels**: Which cache level is the bottleneck? Are loads coalesced? What's the L1/L2 hit rate? How many DRAM bytes are transferred?
- **Compute-bound kernels**: What's the tensor core utilization? What's the instruction mix? Is there warp divergence?

### Step 4: Hypothesize

Combine the macro analysis (Step 2) and NCU deep analysis (Step 3) to formulate a **single, focused** hypothesis:

> Hypothesis: [What you plan to change and why you expect it to improve performance]
> Macro evidence: [Which bench.py metric(s) indicate the bottleneck direction]
> NCU evidence: [Which ncu-cli finding(s) pinpoint the specific cause]

**Hypothesis workflow:**
1. **Macro**: `bench.py` roofline → is it compute-bound or memory-bound? How far from peak?
2. **Micro**: `ncu-cli analyze` → what is the *specific* bottleneck? (stall type, cache miss, uncoalesced access, etc.)
3. **Knowledge**: Check `CUDA_OPTIMIZATION.md` → does a known optimization address this?
4. **History**: Check `memory/<kernel_type>.md` → has this been tried before for this kernel?

**Rules:**
- One change per experiment. Do not combine unrelated optimizations.
- If you've tried this before (check per-kernel log), try something different.
- Always ground hypotheses in NCU evidence, not guesswork.

### Step 5: Modify

Edit `kernel.py` to implement your hypothesis.

### Step 6: Commit

```bash
git add kernel.py
git commit -m "experiment: <brief description of change>"
```

### Step 7: Benchmark

```bash
uv run bench.py > run.log 2>&1
```

**IMPORTANT**: Always redirect to `run.log`. Do NOT let output flood your context window.

### Step 8: Decide

| Condition | Action |
|-----------|--------|
| correctness = FAIL | **REVERT** immediately: `git reset --hard HEAD~1` |
| correctness = PASS, throughput improved (>1%) | **KEEP** |
| correctness = PASS, throughput same or worse | **REVERT**: `git reset --hard HEAD~1` |

### Step 9: Record

**9a. Append to `results.tsv`:**

```
experiment_id	hypothesis	correctness	time_ms	throughput	peak_vram_mb	kept
```

**9b. Update per-kernel log (`memory/<kernel_type>.md`):**

Record the detailed experiment for this specific kernel:
- Experiment ID and hypothesis
- Macro analysis (bench.py roofline metrics)
- NCU analysis (specific bottleneck, stall types, cache hit rates)
- Result (kept / reverted) and key observations
- What you learned that could inform the next experiment

**9c. Update `MEMORY.md` (global summary):**

Keep a concise cross-kernel summary:
- Which kernel was optimized and current best speedup
- High-level insights that transfer across kernels

**9d. Update `CUDA_OPTIMIZATION.md` (if a new optimization pattern was discovered):**

When an optimization **succeeds**, add it to `CUDA_OPTIMIZATION.md` under the appropriate kernel type section. Include:
- What the optimization is
- Why it works for this kernel type
- Expected speedup range
- When an optimization **fails**, add it to the "Anti-patterns" section for that kernel type.

### Step 10: Repeat

Return to Step 1. Continue until:
- Performance gains have plateaued (< 1% improvement over 3 consecutive experiments)
- You have exhausted all known optimizations in `CUDA_OPTIMIZATION.md` and cannot generate new hypotheses from NCU data

## Switching Kernels

When you finish optimizing one kernel, save the optimized version to `kernels_optimized/` and move to the next:

```bash
# Save optimized kernel
cp kernel.py kernels_optimized/<kernel_name>.py

# Switch to next kernel -- copy from baseline (or from kernels_optimized/ if resuming)
cp kernels/<next_kernel>.py kernel.py

# Per-kernel logs are in memory/<kernel_type>.md -- they persist across sessions
# MEMORY.md has the global summary -- cross-kernel insights are valuable
```

**Important:** Never modify files in `kernels/`. The baseline must remain intact for comparison and reproducibility.

Before starting the new kernel, review `memory/<kernel_type>.md` for any past experiments on it, and check `CUDA_OPTIMIZATION.md` for transferable optimization patterns.

## Memory-Bound Kernel Optimization Priority

Most kernels in this repo are memory-bound. The optimization priority for memory-bound kernels is:

1. **Coalescing** -- NCU tells you if loads/stores are uncoalesced (sectors/request > 4). Fix memory layout or access pattern.
2. **Vectorized loads** -- Use `float4`/`bf16_8` loads to maximize bandwidth per instruction.
3. **L2 cache locality** -- Reorder tile indices so neighboring blocks access nearby memory. NCU shows L2 hit rate.
4. **Prefetching / pipelining** -- `num_stages` in Triton, `cp.async` in CUDA. NCU shows Long Scoreboard stalls.
5. **Reduce memory traffic** -- Fuse operations, avoid redundant reads/writes. NCU shows total DRAM bytes.
6. **Shared memory tiling** -- For reduction patterns, load to shared memory first. NCU shows bank conflicts.

**Yes, you can and should modify the kernel code for memory-bound kernels.** The optimization is about *how* data moves, not *what* is computed. Typical changes:

- Adjust `block_size` and `num_stages` (Triton) or thread/block config (CUDA)
- Change memory access patterns for better coalescing
- Add prefetching / software pipelining
- Use vectorized loads (`tl.load` with larger block sizes, or `float4` in CUDA)
- Reorder loop dimensions for better cache behavior

## Memory & Knowledge Structure

```
cuda-evolve/
├── kernels/                    # Baseline kernels (READ-ONLY)
├── kernels_optimized/          # Optimized kernels (agent saves here)
├── CUDA_OPTIMIZATION.md        # Agent-maintained: optimization patterns by kernel type
├── MEMORY.md                   # Global summary across all kernels
├── memory/
│   └── <kernel_type>.md        # Detailed experiment log per kernel
└── results.tsv                 # Raw experiment results
```

- **`kernels/`**: Baseline kernels. **Never modify.** These are the starting point and comparison reference.
- **`kernels_optimized/`**: Mirrors `kernels/` structure. The agent saves the best optimized version of each kernel here after finishing optimization.
- **`CUDA_OPTIMIZATION.md`**: Grows over time as the agent discovers what works. Empty at first. The agent adds entries after each successful (or failed) optimization, organized by kernel type.
- **`memory/<kernel_type>.md`**: Detailed per-kernel experiment log with full NCU analysis, hypotheses, and outcomes. This is the primary record for each kernel.
- **`MEMORY.md`**: High-level cross-kernel summary. Kept concise — just the current best results and transferable insights.

## Multi-Agent Parallel Optimization

When multiple agents need to optimize **different kernels** simultaneously, use **git worktree** to give each agent an isolated working directory. This avoids conflicts on `kernel.py`, logs, git state, and GPU resources.

### Setup

From the main repository, create one worktree per kernel/agent:

```bash
# Ensure main is clean
git checkout main

# Create isolated worktrees (one per kernel)
git worktree add ../cuda-evolve-matmul   -b agent/matmul
git worktree add ../cuda-evolve-rms-norm -b agent/rms-norm
git worktree add ../cuda-evolve-swiglu   -b agent/swiglu
```

Each worktree is an independent directory with its own `kernel.py`, `results.tsv`, `MEMORY.md`, `memory/`, `traces/`, and git working state. All worktrees share the same `.git` repository, so commit history is unified and branches can be merged.

### Branch Naming Convention

Use `agent/<kernel_name>` branches (e.g. `agent/matmul`, `agent/rms-norm`). Each agent commits only to its own branch.

### GPU Isolation

Bind each agent to a separate GPU via `CUDA_VISIBLE_DEVICES`:

```bash
# Agent A (matmul) — GPU 0
cd ../cuda-evolve-matmul
CUDA_VISIBLE_DEVICES=0 uv run bench.py > run.log 2>&1

# Agent B (rms_norm) — GPU 1
cd ../cuda-evolve-rms-norm
CUDA_VISIBLE_DEVICES=1 uv run bench.py > run.log 2>&1
```

If only **one GPU** is available, agents can edit code in parallel but must **serialize benchmark execution** to avoid VRAM contention and timing interference.

### Per-Agent Workflow

Each agent follows the standard Experiment Loop (above) inside its own worktree. No changes to the loop itself — the isolation is at the directory/branch level.

### Merging Results Back to Main

After each agent completes optimization, merge its branch into `main`:

```bash
cd /path/to/main-repo
git merge agent/matmul   --no-ff -m "merge: matmul optimization results"
git merge agent/rms-norm --no-ff -m "merge: rms-norm optimization results"
```

**Conflict expectations by file:**

| File | Conflict risk | Resolution |
|------|--------------|------------|
| `kernels_optimized/<name>.py` | None — different files | Auto-merge |
| `memory/<kernel_type>.md` | None — different files | Auto-merge |
| `results.tsv` | Low — append-only | Concatenate rows (keep header once) |
| `MEMORY.md` | Low — different sections | Merge by section |
| `CUDA_OPTIMIZATION.md` | Low — different kernel type sections | Merge by section |

You can use `merge_results.py` to assist with `results.tsv` merging (see below).

### Cleanup

```bash
git worktree remove ../cuda-evolve-matmul
git worktree remove ../cuda-evolve-rms-norm
```

## Important Rules

1. **Never break correctness.** Every change must pass all 5 correctness stages.
2. **Never modify `bench.py`, `reference.py`, or files in `kernels/`.** These are fixed baselines and evaluation harnesses. Save optimized kernels to `kernels_optimized/`.
3. **One change at a time.** Isolate variables to understand causality.
4. **Always commit before benchmarking.** This enables clean reverts.
5. **Read per-kernel log before each experiment.** Check `memory/<kernel_type>.md` to learn from past attempts on this kernel.
6. **Always run NCU analysis.** Every experiment should include both macro (bench.py) and micro (ncu-cli) analysis. Don't hypothesize without evidence.
7. **Use roofline data and NCU findings together.** Macro tells you the direction, NCU tells you the specific cause.
8. **VRAM must not exceed 80% of GPU memory.** Treat as regression and revert.
9. **Maintain the knowledge base.** Update `CUDA_OPTIMIZATION.md` when you discover new optimization patterns or anti-patterns. Future runs depend on this.
