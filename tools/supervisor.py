#!/usr/bin/env python3
"""
supervisor.py -- Stagnation detection and strategy pivot for cuda-evolve.

Analyzes the experiment trajectory and outputs a structured directive telling
the agent whether it is making progress, stagnating, or stuck, along with
concrete suggestions for the next optimization direction.

Usage:
  uv run tools/supervisor.py                          # analyze current kernel
  uv run tools/supervisor.py --kernel-type rms_norm   # analyze specific kernel
  uv run tools/supervisor.py --window 5               # look at last 5 experiments
  uv run tools/supervisor.py --threshold 1.0          # stagnation threshold (%)
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "workspace" / "results.tsv"
MEMORY_DIR = ROOT / "memory"


def load_results(kernel_type: str | None = None) -> list[dict[str, str]]:
    if not RESULTS_FILE.exists():
        return []

    lines = RESULTS_FILE.read_text(encoding="utf-8").strip().split("\n")
    if len(lines) < 2:
        return []

    headers = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        cols = line.split("\t")
        row = {}
        for i, h in enumerate(headers):
            row[h] = cols[i] if i < len(cols) else ""
        rows.append(row)

    if kernel_type:
        rows = [
            r for r in rows
            if kernel_type.lower() in r.get("experiment_id", "").lower()
            or kernel_type.lower() in r.get("hypothesis", "").lower()
        ]

    return rows


def load_per_kernel_log(kernel_type: str) -> str:
    log_path = MEMORY_DIR / f"{kernel_type}.md"
    if log_path.exists():
        return log_path.read_text(encoding="utf-8")
    return ""


def _safe_float(val: str) -> float | None:
    if not val:
        return None
    val = val.strip().replace(",", "").replace("%", "")
    try:
        return float(val)
    except ValueError:
        return None


def classify_hypothesis(hypothesis: str) -> str:
    """Classify a hypothesis into a broad optimization category."""
    h = hypothesis.lower()
    categories = {
        "tile_size": ["tile", "block_size", "block_m", "block_n", "block_k", "block_size_m"],
        "num_warps": ["num_warps", "warp", "warps_per_block"],
        "num_stages": ["num_stages", "pipeline", "prefetch", "pipelining"],
        "memory_access": ["coalescing", "coalesce", "vectorize", "float4", "float2", "ld.global"],
        "register_pressure": ["register", "inline", "spill", "regs_per_thread"],
        "occupancy": ["occupancy", "grid_size", "grid", "blocks_per_sm", "waves"],
        "cache": ["l1", "l2", "evict", "cache", "smem", "shared_mem"],
        "algorithmic": ["fuse", "fusion", "reduce", "eliminate", "simplify", "remove"],
        "launch_config": ["persistent", "non-persistent", "grid_size", "launch_bounds"],
        "data_type": ["fp8", "bf16", "fp16", "fp32", "quantiz", "cast", "dtype"],
    }
    for cat, keywords in categories.items():
        if any(kw in h for kw in keywords):
            return cat
    return "other"


def detect_patterns(rows: list[dict[str, str]], window: int) -> dict:
    """Analyze the last `window` experiments for trajectory patterns."""
    recent = rows[-window:] if len(rows) >= window else rows

    throughputs = []
    kept_count = 0
    reverted_count = 0
    failed_count = 0
    categories: list[str] = []
    hypotheses: list[str] = []

    for r in recent:
        tp = _safe_float(r.get("throughput", ""))
        if tp is not None:
            throughputs.append(tp)

        kept = r.get("kept", "").strip().lower()
        corr = r.get("correctness", "").strip().upper()

        if corr == "FAIL":
            failed_count += 1
        elif kept in ("yes", "true", "1", "kept"):
            kept_count += 1
        else:
            reverted_count += 1

        hyp = r.get("hypothesis", "")
        hypotheses.append(hyp)
        categories.append(classify_hypothesis(hyp))

    max_improvement = 0.0
    if len(throughputs) >= 2:
        for i in range(1, len(throughputs)):
            if throughputs[i - 1] > 0:
                delta = (throughputs[i] - throughputs[i - 1]) / throughputs[i - 1] * 100
                max_improvement = max(max_improvement, delta)

    cat_counts = Counter(categories)
    repeated_category = cat_counts.most_common(1)[0] if cat_counts else ("none", 0)

    return {
        "total_experiments": len(rows),
        "window_size": len(recent),
        "throughputs": throughputs,
        "kept_count": kept_count,
        "reverted_count": reverted_count,
        "failed_count": failed_count,
        "max_improvement_pct": max_improvement,
        "categories": categories,
        "category_counts": dict(cat_counts),
        "most_tried_category": repeated_category,
        "hypotheses": hypotheses,
    }


def determine_status(
    patterns: dict, threshold: float
) -> tuple[str, list[str], list[str]]:
    """Determine agent status and generate suggestions.

    Returns (status, findings, suggestions).
    """
    findings: list[str] = []
    suggestions: list[str] = []
    window = patterns["window_size"]

    if window == 0:
        return "no_data", ["No experiments found"], ["Run initial benchmark and start experimenting"]

    # --- Correctness failures ---
    if patterns["failed_count"] > window * 0.5:
        findings.append(
            f"High failure rate: {patterns['failed_count']}/{window} experiments failed correctness"
        )
        suggestions.append("Focus on correctness: simplify the kernel, revert to last known-good state")
        suggestions.append("Check for race conditions, numerical overflow, or out-of-bounds access")
        return "stuck", findings, suggestions

    # --- All reverted ---
    if patterns["kept_count"] == 0 and window >= 3:
        findings.append(f"No improvements kept in last {window} experiments")
        suggestions.append("Current optimization direction is exhausted")

    # --- Throughput plateau ---
    tps = patterns["throughputs"]
    if len(tps) >= 3:
        recent_3 = tps[-3:]
        if recent_3[0] > 0:
            total_change = abs(recent_3[-1] - recent_3[0]) / recent_3[0] * 100
            if total_change < threshold:
                findings.append(
                    f"Throughput plateau: {total_change:.2f}% change over last 3 experiments "
                    f"(threshold: {threshold}%)"
                )

    # --- Repeated category ---
    cat, count = patterns["most_tried_category"]
    if count >= 3:
        findings.append(
            f"Repeated optimization category: '{cat}' tried {count}/{window} times"
        )
        all_cats = set(
            ["tile_size", "num_warps", "num_stages", "memory_access",
             "register_pressure", "occupancy", "cache", "algorithmic",
             "launch_config", "data_type"]
        )
        tried_cats = set(patterns["categories"])
        untried = all_cats - tried_cats
        if untried:
            suggestions.append(
                f"Untried optimization categories: {', '.join(sorted(untried))}"
            )

    # --- Oscillation detection ---
    if len(tps) >= 4:
        deltas = [tps[i] - tps[i - 1] for i in range(1, len(tps))]
        sign_changes = sum(
            1 for i in range(1, len(deltas))
            if (deltas[i] > 0) != (deltas[i - 1] > 0)
        )
        if sign_changes >= len(deltas) - 1 and len(deltas) >= 3:
            findings.append(
                "Performance oscillation detected: throughput alternating up/down"
            )
            suggestions.append(
                "The kernel may be near a local optimum; try a qualitatively different approach"
            )

    # --- Strategy suggestions based on bottleneck ---
    recent_bottleneck = ""
    for r in reversed(patterns.get("_raw_rows", [])):
        b = r.get("bottleneck", "").strip()
        if b:
            recent_bottleneck = b
            break

    if recent_bottleneck and not suggestions:
        if "memory" in recent_bottleneck:
            suggestions.append(
                "Memory-bound: try coalescing, vectorized loads, L2 tiling, shared memory, prefetching"
            )
        elif "compute" in recent_bottleneck:
            suggestions.append(
                "Compute-bound: try tensor core utilization, algorithmic simplification, "
                "reduce instruction count"
            )

    # --- Determine status ---
    if not findings:
        status = "progressing"
    elif patterns["kept_count"] == 0 and window >= 3:
        status = "stuck"
    elif any("plateau" in f.lower() for f in findings):
        status = "stagnating"
    elif any("repeated" in f.lower() for f in findings):
        status = "stagnating"
    else:
        status = "progressing"

    if not suggestions and status != "progressing":
        suggestions.append("Review NCU profiler data for new bottlenecks")
        suggestions.append("Check CUDA_OPTIMIZATION.md for transferable patterns from other kernels")

    return status, findings, suggestions


def print_directive(
    status: str,
    findings: list[str],
    suggestions: list[str],
    patterns: dict,
    kernel_type: str | None,
) -> None:
    """Print the supervisor directive in greppable format."""
    print("\n=== SUPERVISOR DIRECTIVE ===")
    print(f"supervisor_status: {status}")
    print(f"supervisor_total_experiments: {patterns['total_experiments']}")
    print(f"supervisor_window: {patterns['window_size']}")
    print(f"supervisor_kept: {patterns['kept_count']}")
    print(f"supervisor_reverted: {patterns['reverted_count']}")
    print(f"supervisor_failed: {patterns['failed_count']}")

    if patterns["throughputs"]:
        print(f"supervisor_latest_throughput: {patterns['throughputs'][-1]:.3f}")
        if len(patterns["throughputs"]) >= 2:
            best = max(patterns["throughputs"])
            print(f"supervisor_best_throughput: {best:.3f}")

    print(f"supervisor_max_improvement_pct: {patterns['max_improvement_pct']:.2f}")

    cat, count = patterns["most_tried_category"]
    print(f"supervisor_most_tried_category: {cat} ({count} times)")

    if kernel_type:
        print(f"supervisor_kernel_type: {kernel_type}")

    for i, f in enumerate(findings):
        print(f"supervisor_finding_{i+1}: {f}")

    for i, s in enumerate(suggestions):
        print(f"supervisor_suggestion_{i+1}: {s}")

    avoided = [
        h for h, k in zip(patterns["hypotheses"], [r.get("kept", "") for r in [{}] * len(patterns["hypotheses"])])
        if k not in ("yes", "true", "1", "kept")
    ]
    if avoided:
        print(f"supervisor_avoided_approaches: {'; '.join(avoided[-5:])}")

    print("=== END SUPERVISOR DIRECTIVE ===")


def main():
    parser = argparse.ArgumentParser(
        description="Supervisor: stagnation detection and strategy pivot for cuda-evolve"
    )
    parser.add_argument(
        "--kernel-type",
        type=str,
        default=None,
        help="Filter experiments by kernel type (default: all)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Number of recent experiments to analyze (default: 5)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Stagnation threshold in %% improvement (default: 1.0)",
    )
    args = parser.parse_args()

    rows = load_results(args.kernel_type)

    if not rows:
        print("\n=== SUPERVISOR DIRECTIVE ===")
        print("supervisor_status: no_data")
        print("supervisor_finding_1: No experiments found in results.tsv")
        print("supervisor_suggestion_1: Run initial benchmark and start experimenting")
        print("=== END SUPERVISOR DIRECTIVE ===")
        return

    patterns = detect_patterns(rows, args.window)
    patterns["_raw_rows"] = rows[-args.window:]

    status, findings, suggestions = determine_status(patterns, args.threshold)

    print_directive(status, findings, suggestions, patterns, args.kernel_type)

    if args.kernel_type:
        log = load_per_kernel_log(args.kernel_type)
        if log:
            failed_patterns = re.findall(r"(?:REVERTED|FAIL|reverted).*?:\s*(.*?)(?:\n|$)", log)
            if failed_patterns:
                print(f"\nsupervisor_historical_failures: {len(failed_patterns)} found in memory/{args.kernel_type}.md")


if __name__ == "__main__":
    main()
