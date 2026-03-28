#!/usr/bin/env python3
"""
history.py -- Compact experiment history for cuda-evolve agent context.

Produces a compact summary of recent experiments from results.tsv and
optionally from memory/<kernel_type>.md, suitable for injection into the
agent's context window without reading the full files.

Usage:
  uv run tools/history.py                          # last 10 experiments
  uv run tools/history.py --last 5                 # last 5 experiments
  uv run tools/history.py --kernel-type rms_norm   # filter by kernel type
  uv run tools/history.py --kept-only              # show only kept experiments
  uv run tools/history.py --trajectory             # show throughput trajectory
"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "workspace" / "results.tsv"
MEMORY_DIR = ROOT / "memory"


def load_results() -> tuple[list[str], list[list[str]]]:
    """Load results.tsv and return (headers, rows)."""
    if not RESULTS_FILE.exists():
        return [], []

    lines = RESULTS_FILE.read_text(encoding="utf-8").strip().split("\n")
    if len(lines) < 2:
        return lines[0].split("\t") if lines else [], []

    headers = lines[0].split("\t")
    rows = [line.split("\t") for line in lines[1:] if line.strip()]
    return headers, rows


def _col(headers: list[str], row: list[str], name: str, default: str = "") -> str:
    try:
        idx = headers.index(name)
        return row[idx] if idx < len(row) else default
    except ValueError:
        return default


def _safe_float(val: str) -> float | None:
    try:
        return float(val.strip().replace("%", "").replace("x", ""))
    except (ValueError, AttributeError):
        return None


def format_compact_table(
    headers: list[str],
    rows: list[list[str]],
    last_n: int,
    kernel_type: str | None,
    kept_only: bool,
) -> str:
    """Format experiments as a compact table."""
    filtered = rows
    if kernel_type:
        kt = kernel_type.lower()
        filtered = [
            r for r in filtered
            if kt in _col(headers, r, "experiment_id", "").lower()
            or kt in _col(headers, r, "hypothesis", "").lower()
        ]

    if kept_only:
        filtered = [
            r for r in filtered
            if _col(headers, r, "kept", "").lower() in ("yes", "true", "1", "kept")
        ]

    recent = filtered[-last_n:]

    lines = [f"=== EXPERIMENT HISTORY ({len(recent)}/{len(filtered)} shown) ==="]
    lines.append("")
    lines.append(
        f"{'ID':<10} {'Kept':<5} {'Correct':<8} {'TFLOPS':>8} {'%Peak':>7} "
        f"{'Bottleneck':<14} {'Hypothesis'}"
    )
    lines.append("-" * 90)

    for row in recent:
        exp_id = _col(headers, row, "experiment_id", "?")[:10]
        kept = _col(headers, row, "kept", "?")[:5]
        corr = _col(headers, row, "correctness", "?")[:8]
        tp = _col(headers, row, "throughput", "0")
        pct = _col(headers, row, "pct_peak_compute", "")
        bn = _col(headers, row, "bottleneck", "")[:14]
        hyp = _col(headers, row, "hypothesis", "")[:50]

        tp_str = f"{float(tp):.3f}" if _safe_float(tp) is not None else tp
        pct_str = f"{pct}%" if pct and "%" not in pct else pct

        lines.append(
            f"{exp_id:<10} {kept:<5} {corr:<8} {tp_str:>8} {pct_str:>7} "
            f"{bn:<14} {hyp}"
        )

    best_tp = 0.0
    best_id = ""
    for row in filtered:
        tp = _safe_float(_col(headers, row, "throughput", "0"))
        if tp is not None and tp > best_tp:
            kept = _col(headers, row, "kept", "").lower()
            if kept in ("yes", "true", "1", "kept"):
                best_tp = tp
                best_id = _col(headers, row, "experiment_id", "")

    if best_id:
        lines.append("")
        lines.append(f"best_kept: {best_id} ({best_tp:.3f} TFLOPS)")

    lines.append("=== END HISTORY ===")
    return "\n".join(lines)


def format_trajectory(headers: list[str], rows: list[list[str]], last_n: int) -> str:
    """Show throughput trajectory as a sparkline-like visualization."""
    recent = rows[-last_n:]

    tps = []
    for row in recent:
        tp = _safe_float(_col(headers, row, "throughput", "0"))
        kept = _col(headers, row, "kept", "").lower() in ("yes", "true", "1", "kept")
        exp_id = _col(headers, row, "experiment_id", "?")
        tps.append((exp_id, tp or 0.0, kept))

    if not tps:
        return "No experiments to show."

    max_tp = max(t[1] for t in tps) or 1.0
    bar_width = 40

    lines = ["=== THROUGHPUT TRAJECTORY ===", ""]

    for exp_id, tp, kept in tps:
        bar_len = int(tp / max_tp * bar_width)
        bar = "#" * bar_len + "." * (bar_width - bar_len)
        marker = "+" if kept else "x"
        lines.append(f"  {exp_id:<10} [{bar}] {tp:.3f} {marker}")

    lines.append("")
    kept_tps = [t[1] for t in tps if t[2]]
    if kept_tps:
        lines.append(f"  best_kept: {max(kept_tps):.3f} TFLOPS")
        lines.append(f"  latest_kept: {kept_tps[-1]:.3f} TFLOPS")
        if len(kept_tps) >= 2:
            delta = (kept_tps[-1] - kept_tps[-2]) / kept_tps[-2] * 100 if kept_tps[-2] > 0 else 0
            lines.append(f"  kept_delta: {delta:+.2f}%")

    lines.append("=== END TRAJECTORY ===")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compact experiment history for cuda-evolve agent context"
    )
    parser.add_argument(
        "--last",
        type=int,
        default=10,
        help="Number of recent experiments to show (default: 10)",
    )
    parser.add_argument(
        "--kernel-type",
        type=str,
        default=None,
        help="Filter by kernel type",
    )
    parser.add_argument(
        "--kept-only",
        action="store_true",
        help="Show only kept experiments",
    )
    parser.add_argument(
        "--trajectory",
        action="store_true",
        help="Show throughput trajectory visualization",
    )
    args = parser.parse_args()

    headers, rows = load_results()

    if not rows:
        print("No experiments found in results.tsv")
        return

    if args.trajectory:
        print(format_trajectory(headers, rows, args.last))
    else:
        print(format_compact_table(headers, rows, args.last, args.kernel_type, args.kept_only))


if __name__ == "__main__":
    main()
