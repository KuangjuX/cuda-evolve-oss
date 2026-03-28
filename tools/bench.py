#!/usr/bin/env python3
"""
bench.py -- cuda-evolve benchmark harness (FIXED -- the agent NEVER modifies this file).

Handles:
  1. GPU hardware detection and roofline modelling
  2. Correctness verification (5 stages)
  3. Performance benchmarking (Triton do_bench)
  4. Structured, greppable output for the agent loop

Usage:
  uv run tools/bench.py                        # benchmark kernel.py using its KERNEL_TYPE
  uv run tools/bench.py --kernel matmul        # force kernel type
  uv run tools/bench.py --quick                # skip stages 3-5, bench only large size
  uv run tools/bench.py --profile              # emit torch profiler trace
  uv run tools/bench.py --sizes large          # benchmark only 'large' size
"""

from __future__ import annotations

import argparse
import importlib
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import torch


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------

class BenchTimeoutError(Exception):
    pass


class _Timeout:
    def __init__(self, seconds: int):
        self.seconds = seconds
        self._use_signal = hasattr(signal, "SIGALRM")

    def _handler(self, signum, frame):
        raise BenchTimeoutError(f"Timed out after {self.seconds}s")

    def __enter__(self):
        if self._use_signal:
            self._old = signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)
        else:
            import threading
            self._timer = threading.Timer(self.seconds, self._timeout_thread)
            self._timer.daemon = True
            self._timed_out = False
            self._timer.start()
        return self

    def __exit__(self, *exc):
        if self._use_signal:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._old)
        else:
            self._timer.cancel()
        return False

    def _timeout_thread(self):
        self._timed_out = True
        import _thread
        _thread.interrupt_main()


# =========================================================================
# 1. GPU HARDWARE DETECTION
# =========================================================================

@dataclass
class GPUSpec:
    name: str = "Unknown"
    sm_count: int = 0
    memory_gb: float = 0.0
    peak_tflops_fp16: float = 0.0
    peak_tflops_bf16: float = 0.0
    peak_tflops_fp32: float = 0.0
    peak_bandwidth_gb_s: float = 0.0
    l2_cache_mb: float = 0.0
    compute_capability: Tuple[int, int] = (0, 0)


_KNOWN_GPUS: Dict[str, Tuple[float, float, float]] = {
    "H800":       (989.5,  3352.0, 50.0),
    "H100 SXM":   (989.5,  3352.0, 50.0),
    "H100 PCIe":  (756.0,  2039.0, 50.0),
    "H100":       (756.0,  2039.0, 50.0),
    "H200":       (989.5,  4800.0, 50.0),
    "A100-SXM":   (312.0,  2039.0, 40.0),
    "A100-PCIE":  (312.0,  1935.0, 40.0),
    "A100":       (312.0,  2039.0, 40.0),
    "L40S":       (362.05, 864.0,  48.0),
    "L4":         (121.0,  300.0,  48.0),
    "A10":        (125.0,  600.0,  6.0),
    "4090":       (330.0,  1008.0, 72.0),
    "4080":       (305.0,  716.8,  64.0),
    "3090":       (142.0,  936.2,  6.0),
    "3080":       (119.5,  760.3,  5.0),
    "B200":       (2250.0, 8000.0, 64.0),
    "B100":       (1800.0, 8000.0, 64.0),
}


def detect_gpu() -> GPUSpec:
    if not torch.cuda.is_available():
        print("WARNING: No CUDA GPU detected, using dummy spec")
        return GPUSpec()

    props = torch.cuda.get_device_properties(0)
    name = props.name
    sm_count = props.multi_processor_count
    memory_gb = round(props.total_memory / (1024 ** 3), 1)
    cc = (props.major, props.minor)

    matched = None
    for fragment, specs in _KNOWN_GPUS.items():
        if fragment in name:
            matched = specs
            break

    if matched is not None:
        peak_fp16, peak_bw, l2 = matched
    else:
        ops_per_clock_per_sm = 256 if cc[0] >= 8 else 128
        clock_ghz = props.clock_rate / 1e6
        peak_fp16 = sm_count * ops_per_clock_per_sm * clock_ghz * 2 / 1e3
        peak_bw = max(props.clock_rate / 1e6 * 256 / 8 * 2, 500.0)
        l2 = props.L2_cache_size / (1024 * 1024) if hasattr(props, "L2_cache_size") else 0.0

    peak_bf16 = peak_fp16
    peak_fp32 = peak_fp16 / 2.0

    return GPUSpec(
        name=name,
        sm_count=sm_count,
        memory_gb=memory_gb,
        peak_tflops_fp16=peak_fp16,
        peak_tflops_bf16=peak_bf16,
        peak_tflops_fp32=peak_fp32,
        peak_bandwidth_gb_s=peak_bw,
        l2_cache_mb=l2,
        compute_capability=cc,
    )


# =========================================================================
# 2. INPUT GENERATORS
# =========================================================================

def _dtype_bytes(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def gen_matmul_inputs(size: dict, dtype: torch.dtype, device: str, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    M, N, K = size["M"], size["N"], size["K"]
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)
    return {"a": a, "b": b}


def gen_rms_norm_inputs(size: dict, dtype: torch.dtype, device: str, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    M, N = size["M"], size["N"]
    x = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype)
    return {"x": x, "weight": weight}


def gen_swiglu_input_quant_inputs(size: dict, dtype: torch.dtype, device: str, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    M, N = size["M"], size["N"]
    x = torch.randn(M, N * 2, dtype=dtype, device=device)
    return {"x": x}


def _ref_swiglu_input_quant(inputs: dict):
    import references
    return references.swiglu_input_quant_ref(inputs["x"])


def gen_qkv_part_rope_inputs(size: dict, dtype: torch.dtype, device: str, seed: int = 42) -> dict:  # noqa: ARG001
    torch.manual_seed(seed)
    batch = size["batch"]
    seq_len = size["seq_len"]
    q_heads = size["q_heads"]
    kv_heads = size["kv_heads"]
    head_dim = size["head_dim"]
    nope_dim = size["nope_dim"]
    num_heads = q_heads + 2 * kv_heads
    rope_dim = head_dim - nope_dim

    qkv = torch.randn(batch, seq_len, num_heads, head_dim,
                       dtype=torch.bfloat16, device=device)
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = torch.cos(freqs).to(torch.float32)
    sin = torch.sin(freqs).to(torch.float32)

    return {
        "qkv": qkv, "cos": cos, "sin": sin,
        "q_heads": q_heads, "kv_heads": kv_heads, "nope_dim": nope_dim,
    }


def gen_dsa_forward_inputs(size: dict, dtype: torch.dtype, device: str, seed: int = 42) -> dict:
    import math as _math
    torch.manual_seed(seed)
    batch = size["batch"]
    seq_len_q = size["seq_len_q"]
    seq_len_kv = size["seq_len_kv"]
    n_heads = size["n_heads"]
    n_heads_kv = size["n_heads_kv"]
    head_dim = size["head_dim"]
    block_size = size["block_size"]

    total_q = batch * seq_len_q
    total_kv = batch * seq_len_kv
    n_heads_block = n_heads_kv

    q = torch.randn(total_q, n_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_kv, n_heads_kv, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_kv, n_heads_kv, head_dim, dtype=dtype, device=device)

    cu_seqlens_q = torch.tensor(
        [i * seq_len_q for i in range(batch + 1)], dtype=torch.int32, device=device
    )
    cu_seqlens_k = torch.tensor(
        [i * seq_len_kv for i in range(batch + 1)], dtype=torch.int32, device=device
    )
    token2batch_q = torch.repeat_interleave(
        torch.diff(cu_seqlens_q), output_size=total_q
    )

    num_kv_blocks = seq_len_kv // block_size
    block_indices = torch.arange(num_kv_blocks, device=device, dtype=torch.int32)
    block_indices = block_indices.unsqueeze(0).unsqueeze(0).expand(
        total_q, n_heads_block, -1
    ).contiguous()

    scale = 1.0 / _math.sqrt(head_dim)

    return {
        "q": q, "k": k, "v": v,
        "block_indices": block_indices,
        "indices_blk_siz": block_size,
        "scale": scale,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "token2batch_q": token2batch_q,
    }


# =========================================================================
# 3. REFERENCE WRAPPERS
# =========================================================================

def _ref_matmul(inputs: dict) -> torch.Tensor:
    import references
    return references.matmul_ref(inputs["a"], inputs["b"])


def _ref_rms_norm(inputs: dict) -> torch.Tensor:
    import references
    return references.rms_norm_ref(inputs["x"], inputs["weight"])


def _ref_qkv_part_rope(inputs: dict) -> torch.Tensor:
    import references
    return references.qkv_part_rope_ref(**inputs)


def _ref_dsa_forward(inputs: dict):
    import references
    return references.dsa_forward_ref(**inputs)


# =========================================================================
# 4. KERNEL CONFIGS
# =========================================================================

KERNEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # -----------------------------------------------------------------
    # MATMUL
    # -----------------------------------------------------------------
    "matmul": {
        "test_sizes": [
            ("tiny",    {"M": 128,  "N": 128,  "K": 128}),
            ("small",   {"M": 512,  "N": 512,  "K": 512}),
            ("medium",  {"M": 1024, "N": 1024, "K": 1024}),
            ("large",   {"M": 2048, "N": 2048, "K": 2048}),
            ("xlarge",  {"M": 4096, "N": 4096, "K": 4096}),
            ("tall",    {"M": 8192, "N": 1024, "K": 1024}),
            ("wide",    {"M": 1024, "N": 8192, "K": 1024}),
            ("deep_k",  {"M": 1024, "N": 1024, "K": 8192}),
            ("llm_qkv", {"M": 4096, "N": 4096, "K": 512}),
            ("llm_mlp", {"M": 4096, "N": 11008, "K": 4096}),
        ],
        "test_dtypes": [torch.float16, torch.bfloat16, torch.float32],
        "tolerances": {
            torch.float16:  {"atol": 1e-2, "rtol": 1e-2},
            torch.bfloat16: {"atol": 2e-2, "rtol": 2e-2},
            torch.float32:  {"atol": 1e-4, "rtol": 1e-4},
        },
        "flops_fn": lambda s: 2 * s["M"] * s["N"] * s["K"],
        "bytes_fn": lambda s, dt: (s["M"] * s["K"] + s["K"] * s["N"] + s["M"] * s["N"]) * _dtype_bytes(dt),
        "input_generator": gen_matmul_inputs,
        "reference_fn": _ref_matmul,
        "edge_sizes": [
            ("edge_1023",  {"M": 1023, "N": 1023, "K": 1023}),
            ("edge_4097",  {"M": 4097, "N": 4097, "K": 512}),
        ],
    },

    # -----------------------------------------------------------------
    # RMS NORM
    # -----------------------------------------------------------------
    "rms_norm": {
        "test_sizes": [
            ("tiny",    {"M": 32,    "N": 128}),
            ("small",   {"M": 256,   "N": 768}),
            ("medium",  {"M": 1024,  "N": 1024}),
            ("large",   {"M": 4096,  "N": 4096}),
            ("llm_7b",  {"M": 2048,  "N": 4096}),
            ("llm_13b", {"M": 2048,  "N": 5120}),
        ],
        "test_dtypes": [torch.bfloat16, torch.float16],
        "tolerances": {
            torch.float16:  {"atol": 1e-2, "rtol": 1e-2},
            torch.bfloat16: {"atol": 1e-1, "rtol": 5e-2},
        },
        "flops_fn": lambda s: 6 * s["M"] * s["N"],
        "bytes_fn": lambda s, dt: (2 * s["M"] * s["N"] + s["N"]) * _dtype_bytes(dt),
        "input_generator": gen_rms_norm_inputs,
        "reference_fn": _ref_rms_norm,
        "edge_sizes": [
            ("edge_1023", {"M": 1023, "N": 768}),
            ("edge_4097", {"M": 4097, "N": 1024}),
        ],
    },

    # -----------------------------------------------------------------
    # SWIGLU + INPUT FP8 QUANTIZATION
    # -----------------------------------------------------------------
    "swiglu_input_quant": {
        "multi_output": True,
        "test_sizes": [
            ("small",   {"M": 256,  "N": 1024}),
            ("medium",  {"M": 1024, "N": 3584}),
            ("large",   {"M": 4096, "N": 7168}),
        ],
        "test_dtypes": [torch.bfloat16],
        "tolerances": {
            torch.bfloat16: {"atol": 5e-1, "rtol": 5e-1},
        },
        "flops_fn": lambda s: 13 * s["M"] * s["N"],
        "bytes_fn": lambda s, dt: (
            s["M"] * s["N"] * 2 * _dtype_bytes(dt)
            + s["M"] * s["N"] * _dtype_bytes(dt)
            + s["M"] * s["N"] * 2 * 1
            + (s["N"] * 2 // 128) * s["M"] * 4
        ),
        "input_generator": gen_swiglu_input_quant_inputs,
        "reference_fn": _ref_swiglu_input_quant,
        "edge_sizes": [],
    },

    # -----------------------------------------------------------------
    # QKV PART ROPE
    # -----------------------------------------------------------------
    "qkv_part_rope": {
        "test_sizes": [
            ("small",   {"batch": 1, "seq_len": 2048, "q_heads": 10, "kv_heads": 1, "head_dim": 256, "nope_dim": 192}),
            ("medium",  {"batch": 2, "seq_len": 2048, "q_heads": 10, "kv_heads": 1, "head_dim": 256, "nope_dim": 192}),
            ("large",   {"batch": 2, "seq_len": 4096, "q_heads": 10, "kv_heads": 1, "head_dim": 256, "nope_dim": 192}),
            ("xlarge",  {"batch": 4, "seq_len": 4096, "q_heads": 10, "kv_heads": 1, "head_dim": 256, "nope_dim": 192}),
            ("batch50", {"batch": 50, "seq_len": 2048, "q_heads": 10, "kv_heads": 1, "head_dim": 256, "nope_dim": 192}),
        ],
        "test_dtypes": [torch.bfloat16],
        "tolerances": {
            torch.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
        },
        "flops_fn": lambda s: s["batch"] * s["seq_len"] * (s["q_heads"] + s["kv_heads"]) * (s["head_dim"] - s["nope_dim"]) * 6,
        "bytes_fn": lambda s, dt: (
            s["batch"] * s["seq_len"] * (s["q_heads"] + 2 * s["kv_heads"]) * s["head_dim"] * 2 * 2
            + s["seq_len"] * ((s["head_dim"] - s["nope_dim"]) // 2) * 4 * 2
        ),
        "input_generator": gen_qkv_part_rope_inputs,
        "reference_fn": _ref_qkv_part_rope,
        "edge_sizes": [],
    },

    # -----------------------------------------------------------------
    # DSA FORWARD (Dynamic Sparse Attention)
    # -----------------------------------------------------------------
    "dsa_forward": {
        "multi_output": True,
        "test_sizes": [
            ("tiny",   {"batch": 1, "seq_len_q": 128,  "seq_len_kv": 128,  "n_heads": 32, "n_heads_kv": 8, "head_dim": 128, "block_size": 64}),
            ("small",  {"batch": 2, "seq_len_q": 512,  "seq_len_kv": 512,  "n_heads": 32, "n_heads_kv": 8, "head_dim": 128, "block_size": 64}),
            ("medium", {"batch": 4, "seq_len_q": 1024, "seq_len_kv": 1024, "n_heads": 32, "n_heads_kv": 8, "head_dim": 128, "block_size": 64}),
            ("large",  {"batch": 4, "seq_len_q": 2048, "seq_len_kv": 2048, "n_heads": 32, "n_heads_kv": 8, "head_dim": 128, "block_size": 64}),
        ],
        "test_dtypes": [torch.bfloat16],
        "tolerances": {
            torch.bfloat16: {"atol": 5e-2, "rtol": 5e-2},
        },
        "flops_fn": lambda s: s["batch"] * s["seq_len_q"] * s["n_heads"] * (
            2 * s["seq_len_kv"] * s["head_dim"] + 2 * s["seq_len_kv"] * s["head_dim"]
        ),
        "bytes_fn": lambda s, dt: (
            s["batch"] * s["seq_len_q"] * s["n_heads"] * s["head_dim"] * _dtype_bytes(dt)
            + s["batch"] * s["seq_len_kv"] * s["n_heads_kv"] * s["head_dim"] * _dtype_bytes(dt) * 2
            + s["batch"] * s["seq_len_q"] * s["n_heads"] * s["head_dim"] * _dtype_bytes(dt)
            + s["batch"] * s["seq_len_q"] * s["n_heads"] * 4
        ),
        "input_generator": gen_dsa_forward_inputs,
        "reference_fn": _ref_dsa_forward,
        "edge_sizes": [],
    },
}


# =========================================================================
# 5. CORRECTNESS TESTING (5 stages)
# =========================================================================

def _compare(output: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> dict:
    if output.shape != expected.shape:
        return {
            "match": False,
            "reason": f"shape mismatch: {output.shape} vs {expected.shape}",
            "max_abs_error": float("inf"),
            "mean_abs_error": float("inf"),
            "pct_within_tol": 0.0,
        }

    out_f = output.float()
    exp_f = expected.float()
    abs_diff = (out_f - exp_f).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    within = (abs_diff <= atol + rtol * exp_f.abs()).float().mean().item() * 100.0
    match = torch.allclose(out_f, exp_f, atol=atol, rtol=rtol)

    return {
        "match": match,
        "reason": "" if match else f"max_abs_error={max_abs:.6e} exceeds tol(atol={atol}, rtol={rtol})",
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "pct_within_tol": within,
    }


def _compare_multi(output, expected, atol: float, rtol: float) -> dict:
    """Compare multi-output kernels (e.g. weight_quant returns tuple)."""
    if not isinstance(output, (tuple, list)):
        output = (output,)
    if not isinstance(expected, (tuple, list)):
        expected = (expected,)

    if len(output) != len(expected):
        return {
            "match": False,
            "reason": f"output count mismatch: {len(output)} vs {len(expected)}",
            "max_abs_error": float("inf"),
            "mean_abs_error": float("inf"),
            "pct_within_tol": 0.0,
        }

    worst_error = 0.0
    for i, (o, e) in enumerate(zip(output, expected)):
        r = _compare(o, e, atol, rtol)
        if not r["match"]:
            r["reason"] = f"output[{i}]: {r['reason']}"
            return r
        worst_error = max(worst_error, r["max_abs_error"])

    return {
        "match": True,
        "reason": "",
        "max_abs_error": worst_error,
        "mean_abs_error": 0.0,
        "pct_within_tol": 100.0,
    }


def _has_nan_inf(t) -> bool:
    if isinstance(t, (tuple, list)):
        return any(_has_nan_inf(x) for x in t)
    try:
        return bool(torch.isnan(t).any().item() or torch.isinf(t).any().item())
    except (RuntimeError, NotImplementedError):
        return bool(torch.isnan(t.float()).any().item() or torch.isinf(t.float()).any().item())


def _do_compare(output, expected, atol, rtol, multi_output):
    if multi_output:
        return _compare_multi(output, expected, atol, rtol)
    return _compare(output, expected, atol, rtol)


def run_correctness(kernel_fn: Callable, config: dict, quick: bool = False) -> dict:
    device = "cuda"
    multi_output = config.get("multi_output", False)
    results = {
        "smoke_test": "SKIP",
        "shape_sweep": "SKIP",
        "numerical_stability": "SKIP",
        "determinism": "SKIP",
        "edge_cases": "SKIP",
        "correctness": "FAIL",
    }
    details = []
    all_pass = True

    gen_fn = config["input_generator"]
    ref_fn = config["reference_fn"]
    sizes = config["test_sizes"]
    dtypes = config["test_dtypes"]
    tols = config["tolerances"]

    # ------------------------------------------------------------------
    # Stage 1: SMOKE TEST
    # ------------------------------------------------------------------
    print("\n--- Stage 1: Smoke Test ---")
    try:
        _, tiny_size = sizes[0]
        dtype0 = dtypes[0]
        inputs = gen_fn(tiny_size, dtype0, device, seed=42)
        expected = ref_fn(inputs)
        with _Timeout(30):
            output = kernel_fn(**inputs)

        if _has_nan_inf(output):
            results["smoke_test"] = "FAIL"
            details.append("  smoke: NaN/Inf in output")
            all_pass = False
            print("  FAIL: NaN/Inf in output")
        else:
            tol = tols.get(dtype0, {"atol": 1e-2, "rtol": 1e-2})
            cmp = _do_compare(output, expected, **tol, multi_output=multi_output)
            if cmp["match"]:
                results["smoke_test"] = "PASS"
                print(f"  PASS (max_abs_error={cmp['max_abs_error']:.6e})")
            else:
                results["smoke_test"] = "FAIL"
                details.append(f"  smoke: {cmp['reason']}")
                all_pass = False
                print(f"  FAIL: {cmp['reason']}")
    except BenchTimeoutError:
        results["smoke_test"] = "FAIL"
        details.append("  smoke: TIMEOUT")
        all_pass = False
        print("  FAIL: TIMEOUT")
    except torch.cuda.OutOfMemoryError:
        results["smoke_test"] = "FAIL"
        details.append("  smoke: OOM")
        all_pass = False
        print("  FAIL: OOM on tiny input")
    except Exception as e:
        results["smoke_test"] = "FAIL"
        details.append(f"  smoke: CRASH ({type(e).__name__}: {e})")
        all_pass = False
        print(f"  FAIL: CRASH ({type(e).__name__}: {e})")

    if results["smoke_test"] == "FAIL":
        results["correctness"] = "FAIL"
        results["details"] = details
        print(f"\ncorrectness: FAIL (smoke test failed, aborting remaining stages)")
        return results

    # ------------------------------------------------------------------
    # Stage 2: SHAPE SWEEP
    # ------------------------------------------------------------------
    print("\n--- Stage 2: Shape Sweep ---")
    sweep_pass = True
    sweep_count = 0
    sweep_fail_count = 0
    worst_error = 0.0
    worst_case = ""

    for label, sz in sizes:
        for dtype in dtypes:
            sweep_count += 1
            try:
                inputs = gen_fn(sz, dtype, device, seed=42)
                expected = ref_fn(inputs)
                with _Timeout(30):
                    output = kernel_fn(**inputs)

                if _has_nan_inf(output):
                    sweep_pass = False
                    sweep_fail_count += 1
                    details.append(f"  sweep {label}/{dtype}: NaN/Inf")
                    print(f"  FAIL: {label} {dtype} -> NaN/Inf")
                    continue

                tol = tols.get(dtype, {"atol": 1e-2, "rtol": 1e-2})
                cmp = _do_compare(output, expected, **tol, multi_output=multi_output)

                if cmp["max_abs_error"] > worst_error:
                    worst_error = cmp["max_abs_error"]
                    worst_case = f"{label}/{dtype}"

                if not cmp["match"]:
                    sweep_pass = False
                    sweep_fail_count += 1
                    details.append(f"  sweep {label}/{dtype}: {cmp['reason']}")
                    print(f"  FAIL: {label} {dtype} -> {cmp['reason']}")
                else:
                    print(f"  PASS: {label} {dtype} (max_err={cmp['max_abs_error']:.2e}, within_tol={cmp['pct_within_tol']:.1f}%)")

            except torch.cuda.OutOfMemoryError:
                print(f"  SKIP: {label} {dtype} -> OOM")
                torch.cuda.empty_cache()
                continue
            except BenchTimeoutError:
                sweep_pass = False
                sweep_fail_count += 1
                details.append(f"  sweep {label}/{dtype}: TIMEOUT")
                print(f"  FAIL: {label} {dtype} -> TIMEOUT")
            except Exception as e:
                sweep_pass = False
                sweep_fail_count += 1
                details.append(f"  sweep {label}/{dtype}: {type(e).__name__}: {e}")
                print(f"  FAIL: {label} {dtype} -> {type(e).__name__}: {e}")
            finally:
                torch.cuda.empty_cache()

    if sweep_pass:
        results["shape_sweep"] = f"PASS ({sweep_count} configs, worst_err={worst_error:.2e} at {worst_case})"
        print(f"  shape_sweep: PASS ({sweep_count} configs, worst_err={worst_error:.2e})")
    else:
        results["shape_sweep"] = f"FAIL ({sweep_fail_count}/{sweep_count} failed)"
        all_pass = False
        print(f"  shape_sweep: FAIL ({sweep_fail_count}/{sweep_count} failed)")

    # ------------------------------------------------------------------
    # Stages 3-5: Skip in --quick mode
    # ------------------------------------------------------------------
    if quick:
        results["numerical_stability"] = "SKIP (quick mode)"
        results["determinism"] = "SKIP (quick mode)"
        results["edge_cases"] = "SKIP (quick mode)"
        results["correctness"] = "PASS" if all_pass else "FAIL"
        results["details"] = details
        print(f"\ncorrectness: {results['correctness']} (quick mode: stages 3-5 skipped)")
        return results

    # ------------------------------------------------------------------
    # Stage 3: NUMERICAL STABILITY
    # ------------------------------------------------------------------
    print("\n--- Stage 3: Numerical Stability ---")
    stability_pass = True
    stab_size = None
    for label, sz in sizes:
        if label == "small":
            stab_size = sz
            break
    if stab_size is None:
        stab_size = sizes[min(1, len(sizes) - 1)][1]
    stab_dtype = dtypes[0]

    adversarial_cases = [
        ("near_max", lambda t: t * 60000.0 if t.dtype == torch.float16 else t * 1e30),
        ("near_zero", lambda t: t * 1e-6),
        ("mixed_scale", lambda t: t * torch.where(
            torch.rand_like(t.float()).to(t.dtype) > 0.5,
            torch.tensor(1e3, device=t.device, dtype=t.dtype),
            torch.tensor(1e-3, device=t.device, dtype=t.dtype),
        )),
        ("all_zeros", lambda t: torch.zeros_like(t)),
        ("all_same", lambda t: torch.ones_like(t) * 0.5),
    ]

    for case_name, transform_fn in adversarial_cases:
        try:
            inputs = gen_fn(stab_size, stab_dtype, device, seed=42)
            transformed = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    transformed[k] = transform_fn(v)
                else:
                    transformed[k] = v

            expected = ref_fn(transformed)
            with _Timeout(30):
                output = kernel_fn(**transformed)

            if _has_nan_inf(output) and not _has_nan_inf(expected):
                stability_pass = False
                details.append(f"  stability {case_name}: NaN/Inf (reference is clean)")
                print(f"  FAIL: {case_name} -> NaN/Inf (reference is clean)")
            elif _has_nan_inf(output) and _has_nan_inf(expected):
                print(f"  PASS: {case_name} -> both have NaN/Inf (expected overflow)")
            else:
                tol = tols.get(stab_dtype, {"atol": 1e-2, "rtol": 1e-2})
                relaxed_atol = tol["atol"] * 10
                relaxed_rtol = tol["rtol"] * 10
                cmp = _do_compare(output, expected, atol=relaxed_atol, rtol=relaxed_rtol, multi_output=multi_output)
                if cmp["match"]:
                    print(f"  PASS: {case_name} (max_err={cmp['max_abs_error']:.2e})")
                else:
                    stability_pass = False
                    details.append(f"  stability {case_name}: {cmp['reason']}")
                    print(f"  FAIL: {case_name} -> {cmp['reason']}")

        except torch.cuda.OutOfMemoryError:
            print(f"  SKIP: {case_name} -> OOM")
            torch.cuda.empty_cache()
        except BenchTimeoutError:
            stability_pass = False
            details.append(f"  stability {case_name}: TIMEOUT")
            print(f"  FAIL: {case_name} -> TIMEOUT")
        except Exception as e:
            stability_pass = False
            details.append(f"  stability {case_name}: {type(e).__name__}: {e}")
            print(f"  FAIL: {case_name} -> {type(e).__name__}: {e}")
        finally:
            torch.cuda.empty_cache()

    results["numerical_stability"] = "PASS" if stability_pass else "FAIL"
    if not stability_pass:
        all_pass = False
    print(f"  numerical_stability: {results['numerical_stability']}")

    # ------------------------------------------------------------------
    # Stage 4: DETERMINISM
    # ------------------------------------------------------------------
    print("\n--- Stage 4: Determinism ---")
    determinism_pass = True
    try:
        det_size = stab_size
        det_dtype = dtypes[0]

        def _flatten(x):
            if isinstance(x, (tuple, list)):
                return torch.cat([t.flatten().float() for t in x])
            return x.flatten().float()

        outputs = []
        for _ in range(3):
            inputs_i = gen_fn(det_size, det_dtype, device, seed=42)
            with _Timeout(30):
                out_i = kernel_fn(**inputs_i)
            outputs.append(_flatten(out_i))

        for i in range(1, 3):
            if not torch.equal(outputs[0], outputs[i]):
                determinism_pass = False
                diff = (outputs[0] - outputs[i]).abs()
                details.append(f"  determinism: run 0 vs run {i} differ (max_diff={diff.max().item():.6e})")
                print(f"  FAIL: run 0 vs run {i} differ (max_diff={diff.max().item():.6e})")

        if determinism_pass:
            print("  PASS: 3 runs are bitwise identical")
        results["determinism"] = "PASS" if determinism_pass else "FAIL"
    except Exception as e:
        results["determinism"] = f"FAIL ({type(e).__name__})"
        all_pass = False
        details.append(f"  determinism: {type(e).__name__}: {e}")
        print(f"  FAIL: {type(e).__name__}: {e}")
    finally:
        torch.cuda.empty_cache()

    if not determinism_pass:
        all_pass = False

    # ------------------------------------------------------------------
    # Stage 5: EDGE CASES
    # ------------------------------------------------------------------
    print("\n--- Stage 5: Edge Cases ---")
    edge_pass = True
    edge_sizes = config.get("edge_sizes", [])
    if not edge_sizes:
        results["edge_cases"] = "SKIP (no edge sizes defined)"
        print("  SKIP: no edge sizes defined")
    else:
        for label, sz in edge_sizes:
            for dtype in dtypes[:1]:
                try:
                    inputs = gen_fn(sz, dtype, device, seed=42)
                    expected = ref_fn(inputs)
                    with _Timeout(30):
                        output = kernel_fn(**inputs)

                    if _has_nan_inf(output) and not _has_nan_inf(expected):
                        edge_pass = False
                        details.append(f"  edge {label}: NaN/Inf")
                        print(f"  FAIL: {label} -> NaN/Inf")
                    else:
                        tol = tols.get(dtype, {"atol": 1e-2, "rtol": 1e-2})
                        cmp = _do_compare(output, expected, **tol, multi_output=multi_output)
                        if cmp["match"]:
                            print(f"  PASS: {label} (max_err={cmp['max_abs_error']:.2e})")
                        else:
                            edge_pass = False
                            details.append(f"  edge {label}: {cmp['reason']}")
                            print(f"  FAIL: {label} -> {cmp['reason']}")

                except torch.cuda.OutOfMemoryError:
                    print(f"  SKIP: {label} -> OOM")
                    torch.cuda.empty_cache()
                except BenchTimeoutError:
                    edge_pass = False
                    details.append(f"  edge {label}: TIMEOUT")
                    print(f"  FAIL: {label} -> TIMEOUT")
                except Exception as e:
                    edge_pass = False
                    details.append(f"  edge {label}: {type(e).__name__}: {e}")
                    print(f"  FAIL: {label} -> {type(e).__name__}: {e}")
                finally:
                    torch.cuda.empty_cache()

        results["edge_cases"] = "PASS" if edge_pass else "FAIL"
        if not edge_pass:
            all_pass = False
        print(f"  edge_cases: {results['edge_cases']}")

    results["correctness"] = "PASS" if all_pass else "FAIL"
    results["details"] = details
    print(f"\ncorrectness: {results['correctness']}")
    return results


# =========================================================================
# 6. PERFORMANCE BENCHMARKING
# =========================================================================

def _do_bench(fn: Callable, warmup: int = 25, rep: int = 100) -> float:
    """Benchmark a function and return median time in milliseconds."""
    try:
        from triton.testing import do_bench
        ms = do_bench(fn, warmup=warmup, rep=rep)
        return ms
    except ImportError:
        pass

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


def run_performance(kernel_fn: Callable, config: dict, gpu: GPUSpec,
                    sizes_filter: str = "all") -> dict:
    device = "cuda"
    gen_fn = config["input_generator"]
    ref_fn = config["reference_fn"]
    flops_fn = config["flops_fn"]
    bytes_fn = config["bytes_fn"]
    dtypes = config["test_dtypes"]

    sizes = config["test_sizes"]
    bench_sizes = []
    if sizes_filter == "all":
        bench_sizes = sizes
    else:
        for label, sz in sizes:
            if label == sizes_filter:
                bench_sizes = [(label, sz)]
                break
        if not bench_sizes:
            for label, sz in sizes:
                if label == "large":
                    bench_sizes = [(label, sz)]
                    break
            if not bench_sizes:
                bench_sizes = [sizes[-1]]

    primary_label = None
    primary_size = None
    for label, sz in sizes:
        if label == "large":
            primary_label = label
            primary_size = sz
            break
    if primary_size is None:
        primary_label, primary_size = sizes[-1]

    dtype = dtypes[0]
    all_results = []
    primary_result = None

    for label, sz in bench_sizes:
        print(f"\n  Benchmarking: {label} ...")
        try:
            flops = flops_fn(sz)
            nbytes = bytes_fn(sz, dtype)

            inputs = gen_fn(sz, dtype, device, seed=42)

            _k_inputs = inputs
            with _Timeout(30):
                kernel_ms = _do_bench(lambda _i=_k_inputs: kernel_fn(**_i), warmup=25, rep=100)

            with _Timeout(30):
                ref_ms = _do_bench(lambda _i=_k_inputs: ref_fn(_i), warmup=25, rep=100)

            kernel_us = kernel_ms * 1000.0
            ref_us = ref_ms * 1000.0
            throughput_tflops = flops / (kernel_ms / 1000.0) / 1e12 if kernel_ms > 0 else 0.0
            bandwidth_gb_s = nbytes / (kernel_ms / 1000.0) / 1e9 if kernel_ms > 0 else 0.0
            ref_throughput_tflops = flops / (ref_ms / 1000.0) / 1e12 if ref_ms > 0 else 0.0

            arithmetic_intensity = flops / nbytes if nbytes > 0 else 0.0
            ridge_point = (gpu.peak_tflops_fp16 * 1e12) / (gpu.peak_bandwidth_gb_s * 1e9) if gpu.peak_bandwidth_gb_s > 0 else 0.0

            if arithmetic_intensity < ridge_point:
                bottleneck = "memory_bound"
                pct_peak_bandwidth = (bandwidth_gb_s / gpu.peak_bandwidth_gb_s * 100.0) if gpu.peak_bandwidth_gb_s > 0 else 0.0
                pct_peak_compute = (throughput_tflops / gpu.peak_tflops_fp16 * 100.0) if gpu.peak_tflops_fp16 > 0 else 0.0
            else:
                bottleneck = "compute_bound"
                pct_peak_compute = (throughput_tflops / gpu.peak_tflops_fp16 * 100.0) if gpu.peak_tflops_fp16 > 0 else 0.0
                pct_peak_bandwidth = (bandwidth_gb_s / gpu.peak_bandwidth_gb_s * 100.0) if gpu.peak_bandwidth_gb_s > 0 else 0.0

            speedup = ref_ms / kernel_ms if kernel_ms > 0 else 0.0

            entry = {
                "label": label,
                "size": sz,
                "dtype": str(dtype),
                "flops": flops,
                "bytes": nbytes,
                "kernel_latency_us": kernel_us,
                "pytorch_latency_us": ref_us,
                "throughput_tflops": throughput_tflops,
                "bandwidth_gb_s": bandwidth_gb_s,
                "ref_throughput_tflops": ref_throughput_tflops,
                "pct_peak_compute": pct_peak_compute,
                "pct_peak_bandwidth": pct_peak_bandwidth,
                "arithmetic_intensity": arithmetic_intensity,
                "ridge_point": ridge_point,
                "bottleneck": bottleneck,
                "speedup_vs_pytorch": speedup,
            }
            all_results.append(entry)

            if label == primary_label:
                primary_result = entry

            print(f"    kernel: {kernel_us:.2f} us | pytorch: {ref_us:.2f} us | "
                  f"speedup: {speedup:.3f}x | {throughput_tflops:.3f} TFLOPS | "
                  f"{pct_peak_compute:.1f}% peak")

        except torch.cuda.OutOfMemoryError:
            print(f"    SKIP: {label} -> OOM")
            torch.cuda.empty_cache()
        except BenchTimeoutError:
            print(f"    SKIP: {label} -> TIMEOUT")
        except Exception as e:
            print(f"    ERROR: {label} -> {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()

    if primary_result is None and all_results:
        primary_result = all_results[-1]

    return {
        "primary": primary_result,
        "all": all_results,
    }


# =========================================================================
# 7. PROFILER (optional)
# =========================================================================

def run_profile(kernel_fn: Callable, config: dict):
    device = "cuda"
    gen_fn = config["input_generator"]
    sizes = config["test_sizes"]

    prof_size = None
    for label, sz in sizes:
        if label == "medium":
            prof_size = sz
            break
    if prof_size is None:
        prof_size = sizes[0][1]

    dtype = config["test_dtypes"][0]
    inputs = gen_fn(prof_size, dtype, device, seed=42)

    trace_dir = os.environ.get("CUDA_EVOLVE_TRACE_DIR", "./traces")
    os.makedirs(trace_dir, exist_ok=True)

    print("\n=== PROFILING ===")
    print(f"Profiling with size: {prof_size}, dtype: {dtype}")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(5):
            kernel_fn(**inputs)
        torch.cuda.synchronize()
        for _ in range(10):
            kernel_fn(**inputs)
        torch.cuda.synchronize()

    trace_path = os.path.join(trace_dir, "kernel_trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"profile_trace: {trace_path}")

    try:
        print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=20))
    except Exception:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


# =========================================================================
# 8. MAIN
# =========================================================================

def main():
    t_start = time.time()

    parser = argparse.ArgumentParser(description="cuda-evolve benchmark harness")
    parser.add_argument("--kernel", type=str, default=None,
                        help="Kernel type to benchmark (default: read from kernel.py)")
    parser.add_argument("--sizes", type=str, default="all",
                        help="Which sizes to benchmark: small|medium|large|all (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: skip correctness stages 3-5, bench only large size")
    parser.add_argument("--profile", action="store_true",
                        help="Enable torch profiler trace")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Import the kernel module
    # ------------------------------------------------------------------
    print("=" * 60)
    print("cuda-evolve Benchmark Harness")
    print("=" * 60)

    kernel_module = None
    kernel_fn = None
    kernel_type = args.kernel

    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())

        kernel_module = importlib.import_module("kernel")
        kernel_fn = kernel_module.kernel_fn

        if kernel_type is None:
            kernel_type = getattr(kernel_module, "KERNEL_TYPE", None)
            if kernel_type is None:
                print("ERROR: kernel.py has no KERNEL_TYPE attribute and --kernel not specified")
                sys.exit(1)

        print(f"kernel_type: {kernel_type}")
        print(f"kernel_module: kernel.py loaded successfully")

    except SyntaxError as e:
        print(f"\nERROR: kernel.py has a syntax error:")
        print(f"  {e}")
        traceback.print_exc()
        print(f"\ncorrectness: FAIL")
        print(f"throughput_tflops: 0.000")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Failed to import kernel.py:")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()
        print(f"\ncorrectness: FAIL")
        print(f"throughput_tflops: 0.000")
        sys.exit(1)

    if kernel_type not in KERNEL_CONFIGS:
        print(f"\nERROR: Unknown kernel type '{kernel_type}'")
        print(f"  Available: {', '.join(KERNEL_CONFIGS.keys())}")
        print(f"\ncorrectness: FAIL")
        print(f"throughput_tflops: 0.000")
        sys.exit(1)

    config = KERNEL_CONFIGS[kernel_type]

    # ------------------------------------------------------------------
    # GPU Detection
    # ------------------------------------------------------------------
    gpu = detect_gpu()

    print(f"\n=== GPU INFO ===")
    print(f"gpu_name: {gpu.name}")
    print(f"gpu_sm_count: {gpu.sm_count}")
    print(f"gpu_memory_gb: {gpu.memory_gb}")
    print(f"gpu_peak_tflops_fp16: {gpu.peak_tflops_fp16}")
    print(f"gpu_peak_tflops_bf16: {gpu.peak_tflops_bf16}")
    print(f"gpu_peak_tflops_fp32: {gpu.peak_tflops_fp32}")
    print(f"gpu_peak_bandwidth_gb_s: {gpu.peak_bandwidth_gb_s}")
    print(f"gpu_l2_cache_mb: {gpu.l2_cache_mb}")
    print(f"gpu_compute_capability: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")

    # ------------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------------
    print(f"\n=== CORRECTNESS ===")
    try:
        correctness_results = run_correctness(kernel_fn, config, quick=args.quick)
    except Exception as e:
        print(f"\nFATAL: Correctness testing crashed: {type(e).__name__}: {e}")
        traceback.print_exc()
        correctness_results = {"correctness": "FAIL", "smoke_test": "CRASH", "shape_sweep": "CRASH",
                               "numerical_stability": "CRASH", "determinism": "CRASH", "edge_cases": "CRASH"}

    print(f"\n--- Correctness Summary ---")
    print(f"smoke_test: {correctness_results.get('smoke_test', 'N/A')}")
    print(f"shape_sweep: {correctness_results.get('shape_sweep', 'N/A')}")
    print(f"numerical_stability: {correctness_results.get('numerical_stability', 'N/A')}")
    print(f"determinism: {correctness_results.get('determinism', 'N/A')}")
    print(f"edge_cases: {correctness_results.get('edge_cases', 'N/A')}")
    print(f"correctness: {correctness_results['correctness']}")

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------
    _perf_sizes = config["test_sizes"]
    _perf_primary_label = None
    _perf_primary_size = None
    for _pl, _ps in _perf_sizes:
        if _pl == "large":
            _perf_primary_label = _pl
            _perf_primary_size = _ps
            break
    if _perf_primary_size is None:
        _perf_primary_label, _perf_primary_size = _perf_sizes[-1]
    _perf_dtype = config["test_dtypes"][0]
    _size_params = ", ".join(f"{k}={v}" for k, v in _perf_primary_size.items())
    print(f"\n=== PERFORMANCE ({_perf_primary_label}: {_size_params}, dtype={_perf_dtype}) ===")

    perf_results = {"primary": None, "all": []}
    peak_vram_mb = 0.0
    try:
        sizes_filter = args.sizes
        if args.quick:
            sizes_filter = "large"
        torch.cuda.reset_peak_memory_stats()
        perf_results = run_performance(kernel_fn, config, gpu, sizes_filter=sizes_filter)
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    except Exception as e:
        print(f"\nFATAL: Performance benchmarking crashed: {type(e).__name__}: {e}")
        traceback.print_exc()

    primary = perf_results.get("primary")
    if primary is not None:
        print(f"\n--- Performance Summary (primary: {primary['label']}) ---")
        print(f"latency_us: {primary['kernel_latency_us']:.2f}")
        print(f"latency_ms: {primary['kernel_latency_us'] / 1000.0:.4f}")
        print(f"throughput_tflops: {primary['throughput_tflops']:.3f}")
        print(f"bandwidth_gb_s: {primary['bandwidth_gb_s']:.1f}")
        print(f"pct_peak_compute: {primary['pct_peak_compute']:.1f}%")
        print(f"pct_peak_bandwidth: {primary['pct_peak_bandwidth']:.1f}%")
        print(f"arithmetic_intensity: {primary['arithmetic_intensity']:.2f}")
        print(f"ridge_point: {primary['ridge_point']:.2f}")
        print(f"bottleneck: {primary['bottleneck']}")
        print(f"flops: {primary['flops']}")
        print(f"bytes: {primary['bytes']}")
        print(f"peak_vram_mb: {peak_vram_mb:.1f}")

        print(f"\n=== COMPARISON VS PYTORCH ===")
        print(f"pytorch_latency_us: {primary['pytorch_latency_us']:.2f}")
        print(f"pytorch_latency_ms: {primary['pytorch_latency_us'] / 1000.0:.4f}")
        print(f"kernel_latency_us: {primary['kernel_latency_us']:.2f}")
        print(f"kernel_latency_ms: {primary['kernel_latency_us'] / 1000.0:.4f}")
        print(f"speedup_vs_pytorch: {primary['speedup_vs_pytorch']:.3f}x")
        print(f"pytorch_tflops: {primary['ref_throughput_tflops']:.3f}")
        print(f"kernel_tflops: {primary['throughput_tflops']:.3f}")
    else:
        print(f"\nlatency_us: 0.00")
        print(f"latency_ms: 0.0000")
        print(f"throughput_tflops: 0.000")
        print(f"bandwidth_gb_s: 0.0")
        print(f"pct_peak_compute: 0.0%")
        print(f"pct_peak_bandwidth: 0.0%")
        print(f"peak_vram_mb: {peak_vram_mb:.1f}")

        print(f"\n=== COMPARISON VS PYTORCH ===")
        print(f"pytorch_latency_us: 0.00")
        print(f"pytorch_latency_ms: 0.0000")
        print(f"kernel_latency_us: 0.00")
        print(f"kernel_latency_ms: 0.0000")
        print(f"speedup_vs_pytorch: 0.000x")

    # ------------------------------------------------------------------
    # Size sweep table
    # ------------------------------------------------------------------
    all_perf = perf_results.get("all", [])
    if len(all_perf) > 1:
        print(f"\n=== SIZE SWEEP ===")
        print(f"{'size':<12} {'kernel_us':>12} {'pytorch_us':>12} {'speedup':>10} {'tflops':>10} {'%peak':>8}")
        print("-" * 66)
        for entry in all_perf:
            print(f"{entry['label']:<12} {entry['kernel_latency_us']:>12.2f} "
                  f"{entry['pytorch_latency_us']:>12.2f} {entry['speedup_vs_pytorch']:>9.3f}x "
                  f"{entry['throughput_tflops']:>10.3f} {entry['pct_peak_compute']:>7.1f}%")

    # ------------------------------------------------------------------
    # Profiling (optional)
    # ------------------------------------------------------------------
    if args.profile:
        try:
            run_profile(kernel_fn, config)
        except Exception as e:
            print(f"\nWARNING: Profiling failed: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Final summary (greppable)
    # ------------------------------------------------------------------
    t_elapsed = time.time() - t_start
    throughput = primary["throughput_tflops"] if primary else 0.0

    print(f"\n=== FINAL ===")
    print(f"kernel_type: {kernel_type}")
    print(f"correctness: {correctness_results['correctness']}")
    print(f"throughput_tflops: {throughput:.3f}")
    if primary:
        print(f"speedup_vs_pytorch: {primary['speedup_vs_pytorch']:.3f}x")
        print(f"pct_peak_compute: {primary['pct_peak_compute']:.1f}%")
        print(f"pct_peak_bandwidth: {primary['pct_peak_bandwidth']:.1f}%")
        print(f"bottleneck: {primary['bottleneck']}")
    else:
        print(f"speedup_vs_pytorch: 0.000x")
        print(f"pct_peak_compute: 0.0%")
        print(f"pct_peak_bandwidth: 0.0%")
    print(f"bench_time_seconds: {t_elapsed:.1f}")

    if t_elapsed > 90:
        print(f"WARNING: bench.py took {t_elapsed:.1f}s (budget: 90s)")


if __name__ == "__main__":
    main()
