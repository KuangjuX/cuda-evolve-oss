"""Environment preparation and validation for cuda-evolve."""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_FILE = ROOT / "results.tsv"
MEMORY_FILE = ROOT / "MEMORY.md"


def check_python():
    v = sys.version_info
    print(f"[✓] Python {v.major}.{v.minor}.{v.micro}")
    if v < (3, 10):
        print("[✗] Python >= 3.10 required")
        sys.exit(1)


def check_cuda():
    try:
        import torch

        if not torch.cuda.is_available():
            print("[✗] CUDA is not available. A CUDA-capable GPU is required.")
            sys.exit(1)

        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        vram_mb = torch.cuda.get_device_properties(0).total_mem / (1024**2)
        print(f"[✓] CUDA available: {device_name} (SM {capability[0]}{capability[1]}, {vram_mb:.0f} MB)")
    except ImportError:
        print("[✗] PyTorch not installed. Run: uv sync")
        sys.exit(1)


def check_triton():
    try:
        import triton

        print(f"[✓] Triton {triton.__version__}")
    except ImportError:
        print("[!] Triton not installed — Triton kernels will not be available")


def check_tools():
    for tool, desc in [("nvcc", "CUDA Compiler"), ("ncu", "Nsight Compute"), ("nsys", "Nsight Systems")]:
        path = shutil.which(tool)
        if path:
            print(f"[✓] {desc}: {path}")
        else:
            print(f"[!] {desc} ({tool}) not found in PATH — profiling features may be limited")


def check_git():
    try:
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT)
        if result.returncode == 0:
            print("[✓] Git repository OK")
        else:
            print("[✗] Not a git repository")
            sys.exit(1)
    except FileNotFoundError:
        print("[✗] git not found")
        sys.exit(1)


def init_results():
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text("experiment_id\thypothesis\tcorrectness\ttime_ms\tthroughput\tpeak_vram_mb\tkept\n")
        print(f"[✓] Created {RESULTS_FILE.name}")
    else:
        lines = RESULTS_FILE.read_text().strip().split("\n")
        print(f"[✓] {RESULTS_FILE.name} exists ({len(lines) - 1} experiments recorded)")


def init_memory():
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text(
            "# Optimization Log\n\n"
            "This file records the history of optimization experiments.\n\n---\n\n"
            "<!-- New entries should be added below this line, in reverse chronological order. -->\n"
        )
        print(f"[✓] Created {MEMORY_FILE.name}")
    else:
        print(f"[✓] {MEMORY_FILE.name} exists")


def check_kernel_files():
    kernel_py = ROOT / "kernel.py"
    reference_py = ROOT / "reference.py"

    if kernel_py.exists():
        print(f"[✓] {kernel_py.name} exists")
    else:
        print(f"[!] {kernel_py.name} not found — create it or copy a kernel from kernels/")

    if reference_py.exists():
        print(f"[✓] {reference_py.name} exists")
    else:
        print(f"[!] {reference_py.name} not found — create it before running experiments")


def main():
    print("=" * 60)
    print("  cuda-evolve Environment Check")
    print("=" * 60)
    print()

    check_python()
    check_cuda()
    check_triton()
    print()
    check_tools()
    print()
    check_git()
    init_results()
    init_memory()
    print()
    check_kernel_files()

    print()
    print("=" * 60)
    print("  Environment ready. Read program.md to begin.")
    print("=" * 60)


if __name__ == "__main__":
    main()
