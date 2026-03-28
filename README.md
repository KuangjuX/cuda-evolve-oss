# cuda-evolve

Autonomous GPU kernel optimization system driven by AI agents.

## Overview

cuda-evolve lets AI agents (Claude, Codex, etc.) autonomously profile, analyze, and optimize GPU kernels through iterative experimentation. Given a kernel, the agent:

1. **Profiles** the kernel to understand performance characteristics (compute-bound vs memory-bound)
2. **Proposes** an optimization hypothesis based on the CUDA optimization guide
3. **Modifies** the kernel code
4. **Benchmarks** the modified kernel against the reference implementation
5. **Decides** whether to keep or revert the change
6. **Logs** the result to `workspace/MEMORY.md` and `workspace/results.tsv`
7. **Repeats** until satisfactory performance is achieved

## Project Structure

```
cuda-evolve/
├── program.md              # Agent workflow protocol
├── CUDA_OPTIMIZATION.md    # Agent-maintained optimization knowledge base
├── workspace/              # Runtime outputs and shared logs
│   ├── MEMORY.md           # Global optimization log (shared across sessions)
│   ├── results.tsv         # Experiment results tracking
│   └── ncu_reports/        # NCU profiling reports
├── tools/
│   ├── bench.py            # Benchmark harness & correctness checking
│   ├── history.py          # Experiment history utilities
│   ├── merge_results.py    # Merge benchmark / result files
│   ├── ncu_profile.py      # Nsight Compute profiling
│   ├── prepare.py          # Environment preparation & validation
│   ├── profile.py          # Kernel profiling (roofline analysis)
│   ├── retrieve_docs.py    # Documentation retrieval helpers
│   ├── run_loop.py         # Agent / optimization loop driver
│   ├── summarize.py        # Result summarization
│   └── supervisor.py       # Process supervision
├── kernel.py               # The kernel being optimized (editable by agent)
├── references/             # Reference implementations (per-kernel modules)
├── kernels/                # Baseline kernels (READ-ONLY, bring your own)
├── kernels_optimized/      # Agent-optimized kernels (output)
├── memory/                 # Per-kernel experiment logs
└── pyproject.toml
```

## Quick Start

```bash
# Install dependencies
uv sync

# Prepare the environment
uv run tools/prepare.py

# Add your kernel to kernels/ (see "Adding Your Own Kernels" below)
# Then select it for optimization:
cp kernels/your_kernel.py kernel.py

# Run a benchmark
uv run tools/bench.py

# Profile the current kernel
uv run tools/profile.py

# Or kick off the agent loop (via your AI agent):
# "Read program.md and start optimizing the kernel."
```

## How It Works

The agent reads `program.md` which defines the experimental protocol. Each iteration:

1. The agent examines profiling data to understand the bottleneck
2. Consults `CUDA_OPTIMIZATION.md` for optimization strategies
3. Makes a focused change to `kernel.py`
4. Commits and runs `tools/bench.py`
5. If performance improves, keeps the change; otherwise reverts
6. Records the outcome in `workspace/MEMORY.md` and `workspace/results.tsv`

## Adding Your Own Kernels

To add a kernel for the agent to optimize:

1. **Create the kernel module** at `kernels/your_kernel.py` exporting:
   - `KERNEL_TYPE: str` -- identifier (e.g. `"rms_norm"`)
   - `kernel_fn(**inputs) -> torch.Tensor` -- the kernel to optimize
   - `get_inputs() -> dict` -- generates sample inputs
   - `get_flops() -> int` -- total FLOPs for roofline analysis
   - `get_bytes() -> int` -- total bytes accessed for roofline analysis

2. **Add a reference implementation** under `references/` (pure PyTorch, used for correctness checking)

3. **Add a benchmark config** in `tools/bench.py` under `KERNEL_CONFIGS` with test sizes, tolerances, input generator, and reference function

4. **Copy to `kernel.py`** and start optimizing:
   ```bash
   cp kernels/your_kernel.py kernel.py
   uv run tools/bench.py
   ```

## Requirements

- Python >= 3.10
- CUDA-capable GPU
- CUDA Toolkit
- [uv](https://github.com/astral-sh/uv) package manager

## License

MIT
