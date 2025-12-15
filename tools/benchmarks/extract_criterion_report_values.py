#!/usr/bin/env python3
"""Extract key benchmark numbers from Criterion outputs for the benchmark report.

Usage:
  python3 tools/benchmarks/extract_criterion_report_values.py

Reads mean point estimates from target/criterion/**/new/estimates.json and prints a JSON
object with formatted strings suitable for updating docs/benchmarks/2025-12-14-benchmark-report.md.

This script is intentionally narrow: it only extracts the benchmarks referenced by the report.
"""

from __future__ import annotations

import json
from pathlib import Path


CRITERION_ROOT = Path("target/criterion")


def mean_ns(estimates_path: Path) -> float:
    data = json.loads(estimates_path.read_text())
    return float(data["mean"]["point_estimate"])


def fmt_time_from_ns(ns: float) -> str:
    # Match the report's formatting conventions.
    us = ns / 1e3
    if us < 1e3:
        s = f"{us:.4f}".rstrip("0").rstrip(".")
        return f"{s} µs"
    ms = us / 1e3
    if ms < 1e3:
        if ms < 10:
            return f"{ms:.4f} ms"
        return f"{ms:.3f} ms"
    s = ms / 1e3
    if s < 10:
        return f"{s:.3f} s"
    return f"{s:.2f} s"


def fmt_s(seconds: float) -> str:
    if seconds < 10:
        return f"{seconds:.3f} s"
    return f"{seconds:.2f} s"


def fmt_k(rows_per_s: float) -> str:
    return f"{rows_per_s/1e3:.3f} Kelem/s"


def fmt_m(rows_per_s: float, digits: int = 4) -> str:
    return f"{rows_per_s/1e6:.{digits}f} Melem/s"


def rows_per_s(rows: int, ns: float) -> float:
    return rows / (ns / 1e9)


def elems_per_s(elems: int, ns: float) -> float:
    return elems / (ns / 1e9)


def estimates(rel: str) -> float:
    path = CRITERION_ROOT / rel / "new" / "estimates.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return mean_ns(path)


def main() -> None:
    out: dict[str, str] = {}

    # Training cross-library (medium)
    ns_boosters_xgb = estimates("compare_train_xgboost_regression/boosters_cold_full/medium")
    ns_xgb = estimates("compare_train_xgboost_regression/xgboost_cold_dmatrix/medium")
    out["train_vs_xgboost_boosters"] = fmt_s(ns_boosters_xgb / 1e9)
    out["train_vs_xgboost_xgboost"] = fmt_s(ns_xgb / 1e9)

    ns_boosters_lgb = estimates("compare_train_lightgbm_regression/boosters_cold_full/medium")
    ns_lgb = estimates("compare_train_lightgbm_regression/lightgbm_cold_full/medium")
    out["train_vs_lightgbm_boosters"] = fmt_s(ns_boosters_lgb / 1e9)
    out["train_vs_lightgbm_lightgbm"] = fmt_s(ns_lgb / 1e9)

    # Growth strategy training cost (50k x 100 => 5e6 elements)
    elems = 50_000 * 100
    ns_depth = estimates("component_train_gbdt_growth_strategy/depthwise/50000x100")
    ns_leaf = estimates("component_train_gbdt_growth_strategy/leafwise/50000x100")
    out["growth_depth_time"] = fmt_s(ns_depth / 1e9)
    out["growth_depth_thrpt"] = fmt_m(elems_per_s(elems, ns_depth), digits=4)
    out["growth_leaf_time"] = fmt_s(ns_leaf / 1e9)
    out["growth_leaf_thrpt"] = fmt_m(elems_per_s(elems, ns_leaf), digits=4)

    # Core prediction scaling (rows/sec)
    for bs in [1, 10, 100, 1_000, 10_000]:
        ns = estimates(f"component_predict_batch_size/medium/{bs}")
        out[f"pred_bs_{bs}_time"] = fmt_time_from_ns(ns)
        out[f"pred_bs_{bs}_thrpt"] = fmt_k(rows_per_s(bs, ns)) if bs < 1_000 else fmt_k(rows_per_s(bs, ns))

    ns_single = estimates("component_predict_single_row_medium")
    out["pred_single_time"] = fmt_time_from_ns(ns_single)

    # Traversal strategy comparison
    out["trav_std_1000"] = fmt_time_from_ns(estimates("component_predict_traversal_medium/std_no_block/1000"))
    out["trav_std_10000"] = fmt_time_from_ns(estimates("component_predict_traversal_medium/std_no_block/10000"))
    out["trav_unroll_1000"] = fmt_time_from_ns(estimates("component_predict_traversal_medium/unroll_block64/1000"))
    out["trav_unroll_10000"] = fmt_time_from_ns(estimates("component_predict_traversal_medium/unroll_block64/10000"))

    # Parallel prediction scaling (10k rows)
    rows10k = 10_000
    for th in [1, 2, 4, 8]:
        ns = estimates(f"component_predict_thread_scaling_medium/par_predict/{th}")
        out[f"par_{th}_time"] = fmt_time_from_ns(ns)
        out[f"par_{th}_thrpt"] = fmt_m(rows_per_s(rows10k, ns), digits=4)

    ns_base = estimates("component_predict_thread_scaling_medium/predict/1")
    out["base_pred_time"] = fmt_time_from_ns(ns_base)
    out["base_pred_thrpt"] = fmt_k(rows_per_s(rows10k, ns_base))

    # Cross-library inference comparisons (10k)
    out["xgb_pred_boosters_10k"] = fmt_time_from_ns(estimates("compare_predict_xgboost_batch_size_medium/boosters/10000"))
    out["xgb_pred_xgboost_10k"] = fmt_time_from_ns(estimates("compare_predict_xgboost_batch_size_medium/xgboost_cold_dmatrix/10000"))

    out["lgb_pred_boosters_10k"] = fmt_time_from_ns(estimates("compare_predict_lightgbm_batch_size_medium/boosters/10000"))
    out["lgb_pred_lightgbm_10k"] = fmt_time_from_ns(estimates("compare_predict_lightgbm_batch_size_medium/lightgbm/10000"))

    out["lgb_single_boosters"] = fmt_time_from_ns(estimates("compare_predict_lightgbm_single_row_medium/boosters"))
    out["lgb_single_lightgbm"] = fmt_time_from_ns(estimates("compare_predict_lightgbm_single_row_medium/lightgbm"))

    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
