"""Kernel profiling for cuda-evolve.

Profiles the current kernel to determine performance characteristics:
- Compute-bound vs memory-bound (roofline analysis)
- Execution time breakdown
- Memory bandwidth utilization
- Occupancy and throughput

Usage:
    uv run profile.py [--kernel kernel.py] [--use-ncu]
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch


def load_kernel_module(path: Path):
    spec = importlib.util.spec_from_file_location("kernel_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_gpu_peak_specs(device: int = 0) -> dict:
    """Get theoretical peak performance for the current GPU."""
    props = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)

    sm_count = props.multi_processor_count
    clock_ghz = props.clock_rate / 1e6  # Convert KHz to GHz

    mem_clock_ghz = props.memory_clock_rate / 1e6
    mem_bus_width = props.memory_bus_width
    peak_bandwidth_gbs = 2 * mem_clock_ghz * mem_bus_width / 8  # GB/s

    # FP32 cores per SM varies by architecture
    fp32_per_sm = {7: 64, 8: 64, 9: 128}.get(capability[0], 64)
    peak_tflops_fp32 = sm_count * fp32_per_sm * clock_ghz * 2 / 1000  # TFLOPS

    fp16_multiplier = {7: 2, 8: 2, 9: 4}.get(capability[0], 2)
    peak_tflops_fp16 = peak_tflops_fp32 * fp16_multiplier

    return {
        "device_name": props.name,
        "sm_count": sm_count,
        "capability": f"{capability[0]}.{capability[1]}",
        "peak_bandwidth_gbs": peak_bandwidth_gbs,
        "peak_tflops_fp32": peak_tflops_fp32,
        "peak_tflops_fp16": peak_tflops_fp16,
        "balance_point_fp32": peak_tflops_fp32 * 1000 / peak_bandwidth_gbs,  # FLOPs/Byte
        "balance_point_fp16": peak_tflops_fp16 * 1000 / peak_bandwidth_gbs,
    }


def profile_with_torch(kernel_fn, inputs: dict, warmup: int = 10, repeat: int = 100):
    """Profile kernel using CUDA events for accurate timing."""
    for _ in range(warmup):
        kernel_fn(**inputs)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(repeat):
        start_event.record()
        kernel_fn(**inputs)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    times.sort()
    trimmed = times[len(times) // 10 : -len(times) // 10] if len(times) > 20 else times

    return {
        "mean_ms": sum(trimmed) / len(trimmed),
        "min_ms": min(times),
        "max_ms": max(times),
        "median_ms": times[len(times) // 2],
    }


def profile_memory(kernel_fn, inputs: dict):
    """Measure peak memory usage."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    kernel_fn(**inputs)
    torch.cuda.synchronize()

    return {
        "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "current_vram_mb": torch.cuda.memory_allocated() / (1024**2),
    }


def roofline_analysis(time_ms: float, flops: int, bytes_accessed: int, gpu_specs: dict, dtype: str = "fp16"):
    """Perform roofline analysis to determine bottleneck type."""
    achieved_tflops = flops / (time_ms / 1000) / 1e12
    achieved_bw_gbs = bytes_accessed / (time_ms / 1000) / 1e9

    arithmetic_intensity = flops / bytes_accessed if bytes_accessed > 0 else float("inf")

    peak_key = f"peak_tflops_{dtype}"
    balance_key = f"balance_point_{dtype}"
    peak_tflops = gpu_specs.get(peak_key, gpu_specs["peak_tflops_fp32"])
    balance_point = gpu_specs.get(balance_key, gpu_specs["balance_point_fp32"])

    if arithmetic_intensity < balance_point:
        bottleneck = "MEMORY-BOUND"
        efficiency = achieved_bw_gbs / gpu_specs["peak_bandwidth_gbs"] * 100
    else:
        bottleneck = "COMPUTE-BOUND"
        efficiency = achieved_tflops / peak_tflops * 100

    return {
        "bottleneck": bottleneck,
        "arithmetic_intensity": arithmetic_intensity,
        "achieved_tflops": achieved_tflops,
        "achieved_bw_gbs": achieved_bw_gbs,
        "efficiency_pct": efficiency,
        "balance_point": balance_point,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile a CUDA/Triton kernel")
    parser.add_argument("--kernel", default="kernel.py", help="Path to kernel file")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=100, help="Measurement iterations")
    args = parser.parse_args()

    kernel_path = Path(args.kernel)
    if not kernel_path.exists():
        print(f"Error: {kernel_path} not found")
        sys.exit(1)

    print(f"Loading kernel from {kernel_path}...")
    mod = load_kernel_module(kernel_path)

    if not hasattr(mod, "kernel_fn") or not hasattr(mod, "get_inputs"):
        print("Error: kernel module must define 'kernel_fn' and 'get_inputs'")
        sys.exit(1)

    gpu_specs = get_gpu_peak_specs()
    print(f"\nGPU: {gpu_specs['device_name']} (SM {gpu_specs['capability']})")
    print(f"  Peak FP32: {gpu_specs['peak_tflops_fp32']:.1f} TFLOPS")
    print(f"  Peak FP16: {gpu_specs['peak_tflops_fp16']:.1f} TFLOPS")
    print(f"  Peak Bandwidth: {gpu_specs['peak_bandwidth_gbs']:.0f} GB/s")

    inputs = mod.get_inputs()
    print(f"\n--- Timing ({args.warmup} warmup, {args.repeat} iterations) ---")
    timing = profile_with_torch(mod.kernel_fn, inputs, warmup=args.warmup, repeat=args.repeat)
    print(f"  Mean:   {timing['mean_ms']:.4f} ms")
    print(f"  Median: {timing['median_ms']:.4f} ms")
    print(f"  Min:    {timing['min_ms']:.4f} ms")
    print(f"  Max:    {timing['max_ms']:.4f} ms")

    print("\n--- Memory ---")
    mem = profile_memory(mod.kernel_fn, inputs)
    print(f"  Peak VRAM: {mem['peak_vram_mb']:.1f} MB")

    if hasattr(mod, "get_flops") and hasattr(mod, "get_bytes"):
        flops = mod.get_flops()
        nbytes = mod.get_bytes()
        print(f"\n--- Roofline Analysis ---")
        analysis = roofline_analysis(timing["mean_ms"], flops, nbytes, gpu_specs)
        print(f"  Arithmetic Intensity: {analysis['arithmetic_intensity']:.1f} FLOPs/Byte")
        print(f"  Balance Point:        {analysis['balance_point']:.1f} FLOPs/Byte")
        print(f"  Bottleneck:           {analysis['bottleneck']}")
        print(f"  Achieved TFLOPS:      {analysis['achieved_tflops']:.2f}")
        print(f"  Achieved Bandwidth:   {analysis['achieved_bw_gbs']:.1f} GB/s")
        print(f"  Efficiency:           {analysis['efficiency_pct']:.1f}%")
    else:
        print("\n[!] Define get_flops() and get_bytes() in kernel module for roofline analysis")


if __name__ == "__main__":
    main()
