#!/usr/bin/env python3
"""
ncu_profile.py -- Nsight Compute profiling wrapper for cuda-evolve.

Wraps `ncu` CLI to collect micro-architectural metrics and outputs them in a
greppable format that the agent loop can parse.

Usage:
  uv run tools/ncu_profile.py                                    # full analysis
  uv run tools/ncu_profile.py --skills roofline,memory           # targeted skills
  uv run tools/ncu_profile.py --skills warp_stall,occupancy      # stall analysis
  uv run tools/ncu_profile.py --diff before.csv after.csv        # compare two profiles
  uv run tools/ncu_profile.py --save baseline                    # save CSV as baseline.csv
  uv run tools/ncu_profile.py --kernel-file my_kernel.py         # profile a specific file
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys

# ---------------------------------------------------------------------------
# NCU metric sets per "skill"
# ---------------------------------------------------------------------------

SKILL_METRICS: dict[str, list[str]] = {
    "roofline": [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained",
        "sm__sass_thread_inst_executed_op_hfma_pred_on.sum.peak_sustained",
    ],
    "memory": [
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
        "l1tex__t_sector_hit_rate.pct",
        "lts__t_sector_hit_rate.pct",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
        "memory_l2_theoretical_sectors_global",
        "memory_l2_theoretical_sectors_global_ideal",
    ],
    "warp_stall": [
        "smsp__warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
        "smsp__warps_issue_stalled_wait_per_issue_active.ratio",
        "smsp__warps_issue_stalled_mio_throttle_per_issue_active.ratio",
        "smsp__warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
        "smsp__warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
        "smsp__warps_issue_stalled_barrier_per_issue_active.ratio",
        "smsp__warps_issue_stalled_membar_per_issue_active.ratio",
        "smsp__warps_issue_stalled_not_selected_per_issue_active.ratio",
        "smsp__warps_issue_stalled_sleeping_per_issue_active.ratio",
        "smsp__warps_issue_stalled_tex_throttle_per_issue_active.ratio",
        "smsp__warps_issue_stalled_no_instruction_per_issue_active.ratio",
    ],
    "occupancy": [
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__warps_active.avg.per_cycle_active",
        "launch__registers_per_thread",
        "launch__shared_mem_per_block_static",
        "launch__shared_mem_per_block_dynamic",
        "launch__block_size",
        "launch__grid_size",
        "launch__occupancy_limit_registers",
        "launch__occupancy_limit_shared_mem",
        "launch__occupancy_limit_warps",
        "launch__occupancy_limit_blocks",
        "launch__waves_per_multiprocessor",
    ],
    "instruction": [
        "sm__inst_executed.sum",
        "sm__inst_executed_pipe_tensor.sum",
        "sm__inst_executed_pipe_fp16.sum",
        "sm__inst_executed_pipe_fp32.sum",
        "sm__inst_executed_pipe_fp64.sum",
        "sm__inst_executed_pipe_lsu.sum",
        "smsp__inst_executed.avg.per_cycle_active",
    ],
}

ALL_SKILLS = list(SKILL_METRICS.keys())

STALL_METRIC_PREFIX = "smsp__warps_issue_stalled_"


def _find_ncu() -> str | None:
    return shutil.which("ncu")


def _get_kernel_launch_cmd(kernel_file: str) -> list[str]:
    """Build the python command that launches the kernel once for profiling."""
    return [
        sys.executable, "-c",
        f"""
import importlib.util, sys, os
os.chdir({os.getcwd()!r})
if {os.getcwd()!r} not in sys.path:
    sys.path.insert(0, {os.getcwd()!r})
spec = importlib.util.spec_from_file_location("kernel_mod", {kernel_file!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
inputs = mod.get_inputs()
import torch; torch.cuda.synchronize()
for _ in range(3):
    mod.kernel_fn(**inputs)
torch.cuda.synchronize()
"""
    ]


def collect_metrics(skills: list[str]) -> list[str]:
    metrics = []
    for s in skills:
        if s in SKILL_METRICS:
            metrics.extend(SKILL_METRICS[s])
    return sorted(set(metrics))


def run_ncu(
    kernel_file: str,
    metrics: list[str],
    output_csv: str,
    output_rep: str | None = None,
    extra_args: list[str] | None = None,
) -> str:
    ncu = _find_ncu()
    if ncu is None:
        print("ERROR: ncu (Nsight Compute) not found in PATH.")
        print("Install NVIDIA Nsight Compute or add it to your PATH.")
        sys.exit(1)

    cmd = [ncu]
    if metrics:
        cmd += ["--metrics", ",".join(metrics)]
    cmd += ["--csv", "--log-file", output_csv]
    if output_rep:
        cmd += ["-o", output_rep]
    cmd += ["--target-processes", "all"]
    cmd += ["--kernel-name-base", "demangled"]
    if extra_args:
        cmd += extra_args
    cmd += _get_kernel_launch_cmd(kernel_file)

    print(f"ncu_command: {' '.join(cmd[:6])} ... (full command logged below)")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        print(f"ncu_error: ncu exited with code {result.returncode}")
        if result.stderr:
            for line in result.stderr.strip().split("\n")[:20]:
                print(f"  {line}")
        sys.exit(1)

    return output_csv


def parse_ncu_csv(csv_path: str) -> list[dict[str, str]]:
    """Parse NCU CSV output into a list of metric dicts per kernel launch."""
    rows: list[dict[str, str]] = []
    try:
        with open(csv_path, encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"ncu_error: CSV file not found: {csv_path}")
        return rows

    lines = content.strip().split("\n")
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('"ID"') or line.startswith("ID"):
            header_idx = i
            break

    if header_idx is None:
        for i, line in enumerate(lines):
            if "Metric Name" in line or "metric_name" in line.lower():
                header_idx = i
                break

    if header_idx is None:
        print("ncu_warning: Could not find CSV header in NCU output")
        return rows

    reader = csv.DictReader(lines[header_idx:])
    for row in reader:
        rows.append(dict(row))

    return rows


def _safe_float(val: str) -> float | None:
    if not val:
        return None
    val = val.strip().replace(",", "").replace("%", "")
    try:
        return float(val)
    except ValueError:
        return None


def analyze_and_print(rows: list[dict[str, str]]) -> dict[str, str]:
    """Analyze parsed NCU data and print greppable output."""
    results: dict[str, str] = {}

    metric_vals: dict[str, float] = {}
    for row in rows:
        name = row.get("Metric Name", row.get("metric_name", ""))
        val_str = row.get("Metric Value", row.get("metric_value", row.get("Average", "")))
        val = _safe_float(val_str)
        if name and val is not None:
            if name not in metric_vals:
                metric_vals[name] = val
            else:
                metric_vals[name] = max(metric_vals[name], val)

    if not metric_vals:
        print("ncu_warning: no metrics parsed from NCU output")
        return results

    # --- Bottleneck classification ---
    sm_pct = metric_vals.get("sm__throughput.avg.pct_of_peak_sustained_elapsed")
    mem_pct = metric_vals.get("dram__throughput.avg.pct_of_peak_sustained_elapsed")
    if sm_pct is not None and mem_pct is not None:
        if mem_pct > sm_pct:
            results["ncu_bottleneck"] = f"memory_bound (sm={sm_pct:.1f}%, dram={mem_pct:.1f}%)"
        else:
            results["ncu_bottleneck"] = f"compute_bound (sm={sm_pct:.1f}%, dram={mem_pct:.1f}%)"
    elif sm_pct is not None:
        results["ncu_bottleneck"] = f"sm_throughput={sm_pct:.1f}%"
    elif mem_pct is not None:
        results["ncu_bottleneck"] = f"dram_throughput={mem_pct:.1f}%"

    # --- Warp stall analysis ---
    stalls: list[tuple[str, float]] = []
    for name, val in metric_vals.items():
        if STALL_METRIC_PREFIX in name:
            short = name.replace(STALL_METRIC_PREFIX, "").replace("_per_issue_active.ratio", "")
            stalls.append((short, val))

    if stalls:
        stalls.sort(key=lambda x: x[1], reverse=True)
        top = stalls[0]
        results["ncu_top_stall"] = f"{top[0]} ({top[1]:.2f})"
        stall_summary = ", ".join(f"{s[0]}={s[1]:.2f}" for s in stalls[:5])
        results["ncu_stall_breakdown"] = stall_summary

    # --- Occupancy ---
    occ = metric_vals.get("sm__warps_active.avg.pct_of_peak_sustained_active")
    if occ is not None:
        results["ncu_occupancy"] = f"{occ:.1f}%"

    regs = metric_vals.get("launch__registers_per_thread")
    if regs is not None:
        results["ncu_registers_per_thread"] = f"{int(regs)}"

    block_size = metric_vals.get("launch__block_size")
    if block_size is not None:
        results["ncu_block_size"] = f"{int(block_size)}"

    grid_size = metric_vals.get("launch__grid_size")
    if grid_size is not None:
        results["ncu_grid_size"] = f"{int(grid_size)}"

    waves = metric_vals.get("launch__waves_per_multiprocessor")
    if waves is not None:
        results["ncu_waves_per_sm"] = f"{waves:.2f}"

    occ_limit_regs = metric_vals.get("launch__occupancy_limit_registers")
    if occ_limit_regs is not None:
        results["ncu_occupancy_limit_registers"] = f"{occ_limit_regs:.1f}%"

    # --- Cache hit rates ---
    l1_hit = metric_vals.get("l1tex__t_sector_hit_rate.pct")
    if l1_hit is not None:
        results["ncu_l1_hit_rate"] = f"{l1_hit:.1f}%"

    l2_hit = metric_vals.get("lts__t_sector_hit_rate.pct")
    if l2_hit is not None:
        results["ncu_l2_hit_rate"] = f"{l2_hit:.1f}%"

    # --- DRAM traffic ---
    dram_read = metric_vals.get("dram__bytes_read.sum")
    dram_write = metric_vals.get("dram__bytes_write.sum")
    if dram_read is not None and dram_write is not None:
        total_gb = (dram_read + dram_write) / 1e9
        results["ncu_dram_traffic_gb"] = f"{total_gb:.3f}"

    # --- Shared memory bank conflicts ---
    ld_conflicts = metric_vals.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum")
    st_conflicts = metric_vals.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum")
    if ld_conflicts is not None or st_conflicts is not None:
        total = (ld_conflicts or 0) + (st_conflicts or 0)
        results["ncu_smem_bank_conflicts"] = f"{int(total)}"

    # --- Coalescing efficiency ---
    actual = metric_vals.get("memory_l2_theoretical_sectors_global")
    ideal = metric_vals.get("memory_l2_theoretical_sectors_global_ideal")
    if actual is not None and ideal is not None and ideal > 0:
        eff = ideal / actual * 100
        results["ncu_coalescing_efficiency"] = f"{eff:.1f}%"

    # --- Tensor core utilization ---
    tc_inst = metric_vals.get("sm__inst_executed_pipe_tensor.sum")
    total_inst = metric_vals.get("sm__inst_executed.sum")
    if tc_inst is not None and total_inst is not None and total_inst > 0:
        tc_pct = tc_inst / total_inst * 100
        results["ncu_tensor_core_pct"] = f"{tc_pct:.1f}%"

    # --- IPC ---
    ipc = metric_vals.get("smsp__inst_executed.avg.per_cycle_active")
    if ipc is not None:
        results["ncu_ipc"] = f"{ipc:.2f}"

    # --- Generate findings and actions ---
    findings: list[str] = []
    actions: list[str] = []

    if stalls and stalls[0][1] > 0.3:
        top_stall = stalls[0][0]
        if top_stall == "long_scoreboard":
            findings.append("High long scoreboard stalls: memory latency dominates")
            actions.append("Add prefetching/pipelining (num_stages), reduce memory accesses, improve L2 locality")
        elif top_stall == "wait":
            findings.append("High wait stalls: barrier synchronization overhead")
            actions.append("Reduce __syncthreads frequency, restructure shared memory access patterns")
        elif top_stall == "mio_throttle":
            findings.append("High MIO throttle: memory instruction queue full")
            actions.append("Reduce number of outstanding memory ops, increase compute-to-memory ratio")
        elif top_stall == "math_pipe_throttle":
            findings.append("High math pipe throttle: compute pipeline saturated")
            actions.append("Already near compute peak; look for algorithmic complexity reduction")
        elif top_stall == "short_scoreboard":
            findings.append("High short scoreboard stalls: shared memory / L1 latency")
            actions.append("Check bank conflicts, reduce shared memory access frequency")
        elif top_stall == "barrier":
            findings.append("High barrier stalls: workload imbalance across warps")
            actions.append("Balance work across warps, reduce barrier-protected sections")
        elif top_stall == "membar":
            findings.append("High memory barrier stalls: fence instructions blocking pipeline")
            actions.append("Replace blocking fences with non-blocking where possible, eliminate unnecessary barriers")

    if occ is not None and occ < 50:
        findings.append(f"Low occupancy ({occ:.0f}%)")
        if regs is not None and regs > 64:
            actions.append(f"Reduce register pressure ({int(regs)} regs/thread): simplify kernel, split into phases")
        else:
            actions.append("Increase block count or reduce shared memory per block")

    if l1_hit is not None and l1_hit < 30:
        findings.append(f"Low L1 hit rate ({l1_hit:.0f}%)")
        actions.append("Improve spatial locality, use shared memory tiling")

    if l2_hit is not None and l2_hit < 50:
        findings.append(f"Low L2 hit rate ({l2_hit:.0f}%)")
        actions.append("Improve tile ordering for L2 reuse, use eviction hints")

    coalescing_str = results.get("ncu_coalescing_efficiency")
    if coalescing_str:
        coal_val = _safe_float(coalescing_str.replace("%", ""))
        if coal_val is not None and coal_val < 80:
            findings.append(f"Poor coalescing ({coal_val:.0f}%)")
            actions.append("Fix memory access patterns for contiguous loads/stores")

    if findings:
        for i, f in enumerate(findings):
            results[f"ncu_finding_{i+1}"] = f
    if actions:
        for i, a in enumerate(actions):
            results[f"ncu_action_{i+1}"] = a

    # --- Print all results in greppable format ---
    print("\n=== NCU ANALYSIS ===")
    for key, val in sorted(results.items()):
        print(f"{key}: {val}")
    print("=== END NCU ANALYSIS ===")

    return results


def diff_profiles(before_csv: str, after_csv: str) -> None:
    """Compare two NCU CSV profiles and show deltas."""
    before = parse_ncu_csv(before_csv)
    after = parse_ncu_csv(after_csv)

    before_metrics: dict[str, float] = {}
    after_metrics: dict[str, float] = {}

    for row in before:
        name = row.get("Metric Name", row.get("metric_name", ""))
        val = _safe_float(row.get("Metric Value", row.get("metric_value", row.get("Average", ""))))
        if name and val is not None:
            before_metrics[name] = val

    for row in after:
        name = row.get("Metric Name", row.get("metric_name", ""))
        val = _safe_float(row.get("Metric Value", row.get("metric_value", row.get("Average", ""))))
        if name and val is not None:
            after_metrics[name] = val

    all_keys = sorted(set(before_metrics.keys()) | set(after_metrics.keys()))

    print("\n=== NCU DIFF ===")
    print(f"{'Metric':<70} {'Before':>12} {'After':>12} {'Delta':>12} {'Change':>8}")
    print("-" * 116)

    significant: list[tuple[str, float, float, float]] = []

    for key in all_keys:
        bv = before_metrics.get(key)
        av = after_metrics.get(key)
        if bv is not None and av is not None:
            delta = av - bv
            pct = (delta / bv * 100) if bv != 0 else 0
            if abs(pct) > 1 or abs(delta) > 0.01:
                significant.append((key, bv, av, pct))

    significant.sort(key=lambda x: abs(x[3]), reverse=True)

    for key, bv, av, pct in significant[:30]:
        delta = av - bv
        direction = "+" if delta > 0 else ""
        short_key = key
        if len(short_key) > 68:
            short_key = short_key[:65] + "..."
        print(f"{short_key:<70} {bv:>12.2f} {av:>12.2f} {direction}{delta:>11.2f} {direction}{pct:>6.1f}%")

    print("=== END NCU DIFF ===")


def main():
    parser = argparse.ArgumentParser(
        description="NCU profiling wrapper for cuda-evolve agent loop"
    )
    parser.add_argument(
        "--skills",
        type=str,
        default=",".join(ALL_SKILLS),
        help=f"Comma-separated skills to profile: {','.join(ALL_SKILLS)} (default: all)",
    )
    parser.add_argument(
        "--diff",
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        help="Compare two NCU CSV profiles",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save CSV output with this name (e.g. --save baseline -> baseline.csv)",
    )
    parser.add_argument(
        "--kernel-file",
        type=str,
        default="kernel.py",
        help="Path to the kernel file to profile (default: kernel.py)",
    )
    parser.add_argument(
        "--ncu-args",
        type=str,
        default="",
        help="Additional arguments to pass to ncu (quoted string)",
    )
    args = parser.parse_args()

    if args.diff:
        diff_profiles(args.diff[0], args.diff[1])
        return

    kernel_file = os.path.abspath(args.kernel_file)
    if not os.path.exists(kernel_file):
        print(f"ERROR: kernel file not found: {kernel_file}")
        sys.exit(1)

    skills = [s.strip() for s in args.skills.split(",")]
    invalid = [s for s in skills if s not in SKILL_METRICS]
    if invalid:
        print(f"WARNING: unknown skills ignored: {', '.join(invalid)}")
        skills = [s for s in skills if s in SKILL_METRICS]

    if not skills:
        skills = ALL_SKILLS

    print("=== NCU PROFILING ===")
    print(f"kernel_file: {args.kernel_file}")
    print(f"skills: {','.join(skills)}")

    metrics = collect_metrics(skills)
    print(f"metrics_count: {len(metrics)}")

    trace_dir = os.environ.get("CUDA_EVOLVE_TRACE_DIR", "./traces")
    os.makedirs(trace_dir, exist_ok=True)

    csv_name = args.save if args.save else "ncu_profile"
    csv_path = os.path.join(trace_dir, f"{csv_name}.csv")
    rep_path = os.path.join(trace_dir, f"{csv_name}.ncu-rep")

    extra = args.ncu_args.split() if args.ncu_args else None
    run_ncu(kernel_file, metrics, csv_path, rep_path, extra)

    print(f"ncu_csv: {csv_path}")
    print(f"ncu_rep: {rep_path}")

    rows = parse_ncu_csv(csv_path)
    print(f"ncu_parsed_rows: {len(rows)}")

    analyze_and_print(rows)


if __name__ == "__main__":
    main()
