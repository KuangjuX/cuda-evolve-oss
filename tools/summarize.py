#!/usr/bin/env python3
"""
summarize.py -- Compact context injection for the cuda-evolve agent loop.

Reads run.log and optionally ncu.log, and produces a compact summary with
all key metrics in a minimal format suitable for direct injection into the
agent's context window.

Usage:
  uv run tools/summarize.py                     # summarize run.log
  uv run tools/summarize.py --ncu               # include NCU data from ncu.log
  uv run tools/summarize.py --log other.log     # use a different log file
  uv run tools/summarize.py --json              # output as JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def parse_bench_log(log_path: Path) -> dict[str, str]:
    """Extract key metrics from bench.py output."""
    if not log_path.exists():
        return {"error": f"{log_path} not found"}

    content = log_path.read_text(encoding="utf-8")
    metrics: dict[str, str] = {}

    key_metrics = [
        "kernel_type", "correctness", "throughput_tflops", "bandwidth_gb_s",
        "pct_peak_compute", "pct_peak_bandwidth", "bottleneck",
        "speedup_vs_pytorch", "peak_vram_mb", "bench_time_seconds",
        "latency_us", "latency_ms", "arithmetic_intensity", "ridge_point",
        "pytorch_latency_us", "pytorch_tflops", "kernel_tflops",
        "gpu_name",
    ]

    for line in content.split("\n"):
        line = line.strip()

        for key in key_metrics:
            if line.startswith(f"{key}:"):
                val = line.split(":", 1)[1].strip()
                metrics[key] = val
                break

    correctness_stages = [
        "smoke_test", "shape_sweep", "numerical_stability",
        "determinism", "edge_cases",
    ]
    for stage in correctness_stages:
        for line in content.split("\n"):
            if line.strip().startswith(f"{stage}:"):
                val = line.strip().split(":", 1)[1].strip()
                if "FAIL" in val:
                    metrics[f"stage_{stage}"] = val

    if "WARNING" in content:
        for line in content.split("\n"):
            if "WARNING" in line:
                metrics.setdefault("warnings", "")
                metrics["warnings"] += line.strip() + "; "

    return metrics


def parse_ncu_log(log_path: Path) -> dict[str, str]:
    """Extract key metrics from ncu_profile.py output."""
    if not log_path.exists():
        return {}

    content = log_path.read_text(encoding="utf-8")
    metrics: dict[str, str] = {}

    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("ncu_") and ":" in line:
            key, val = line.split(":", 1)
            metrics[key.strip()] = val.strip()

    return metrics


def format_text(bench: dict[str, str], ncu: dict[str, str]) -> str:
    """Format as compact text summary."""
    lines = ["=== BENCH SUMMARY ==="]

    primary = [
        ("correctness", bench.get("correctness")),
        ("throughput_tflops", bench.get("throughput_tflops")),
        ("speedup_vs_pytorch", bench.get("speedup_vs_pytorch")),
        ("bottleneck", bench.get("bottleneck")),
        ("pct_peak_compute", bench.get("pct_peak_compute")),
        ("pct_peak_bandwidth", bench.get("pct_peak_bandwidth")),
        ("peak_vram_mb", bench.get("peak_vram_mb")),
    ]

    for key, val in primary:
        if val:
            lines.append(f"{key}: {val}")

    failed_stages = [k for k in bench if k.startswith("stage_")]
    if failed_stages:
        lines.append("failed_stages: " + ", ".join(
            f"{k.replace('stage_', '')}={bench[k]}" for k in failed_stages
        ))

    if bench.get("warnings"):
        lines.append(f"warnings: {bench['warnings']}")

    if ncu:
        lines.append("--- NCU ---")
        ncu_primary = [
            "ncu_bottleneck", "ncu_top_stall", "ncu_occupancy",
            "ncu_registers_per_thread", "ncu_l1_hit_rate", "ncu_l2_hit_rate",
            "ncu_coalescing_efficiency", "ncu_tensor_core_pct",
        ]
        for key in ncu_primary:
            if key in ncu:
                lines.append(f"{key}: {ncu[key]}")

        findings = sorted(k for k in ncu if k.startswith("ncu_finding"))
        actions = sorted(k for k in ncu if k.startswith("ncu_action"))
        for k in findings:
            lines.append(f"finding: {ncu[k]}")
        for k in actions:
            lines.append(f"action: {ncu[k]}")

    lines.append("=== END SUMMARY ===")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compact summary of bench.py + ncu_profile.py output"
    )
    parser.add_argument(
        "--log",
        type=str,
        default="run.log",
        help="Path to bench.py log file (default: run.log)",
    )
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="Also summarize NCU data from ncu.log",
    )
    parser.add_argument(
        "--ncu-log",
        type=str,
        default="ncu.log",
        help="Path to ncu_profile.py log file (default: ncu.log)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    args = parser.parse_args()

    bench_metrics = parse_bench_log(Path(args.log))

    ncu_metrics: dict[str, str] = {}
    if args.ncu:
        ncu_metrics = parse_ncu_log(Path(args.ncu_log))

    if args.json:
        combined = {"bench": bench_metrics}
        if ncu_metrics:
            combined["ncu"] = ncu_metrics
        print(json.dumps(combined, indent=2))
    else:
        print(format_text(bench_metrics, ncu_metrics))


if __name__ == "__main__":
    main()
