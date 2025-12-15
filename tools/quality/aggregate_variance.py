#!/usr/bin/env python3
"""Aggregate multi-seed `quality_eval --out-json` runs.

Usage:
  python3 tools/quality/aggregate_variance.py run1.json run2.json ...
  python3 tools/quality/aggregate_variance.py --out summary.json run*.json

It prints a compact text summary to stdout and can also write a machine-readable
summary JSON.

Notes:
- This assumes each input JSON was produced by `quality_eval --out-json`.
- CI is a normal-approx 95% CI: mean ± 1.96 * (std / sqrt(n)).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def sample_std(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var)


def ci95_half_width(std: float, n: int) -> float:
    if n <= 1:
        return 0.0
    return 1.96 * (std / math.sqrt(n))


@dataclass(frozen=True)
class GroupKey:
    task: str
    growth: str
    depth: int
    leaves: int
    rows_train: int
    rows_valid: int
    cols: int
    trees: int


def extract_metrics(obj: Any) -> Dict[str, float]:
    if not isinstance(obj, dict):
        return {}
    metrics = obj.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        if is_number(v):
            out[k] = float(v)
    return out


def req_int(obj: Dict[str, Any], key: str) -> int:
    v = obj.get(key)
    if v is None:
        raise KeyError(f"missing required key: {key}")
    return int(v)


def req_str(obj: Dict[str, Any], key: str) -> str:
    v = obj.get(key)
    if v is None:
        raise KeyError(f"missing required key: {key}")
    return str(v)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="Write summary JSON to this path")
    ap.add_argument("inputs", nargs="+", help="quality_eval JSON files")
    args = ap.parse_args(argv)

    runs: List[Dict[str, Any]] = []
    for p in args.inputs:
        with open(p, "r", encoding="utf-8") as f:
            runs.append(json.load(f))

    grouped: Dict[GroupKey, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        if not isinstance(r, dict):
            raise TypeError("each input JSON must be an object")
        key = GroupKey(
            task=req_str(r, "task"),
            growth=req_str(r, "growth"),
            depth=req_int(r, "depth"),
            leaves=req_int(r, "leaves"),
            rows_train=req_int(r, "rows_train"),
            rows_valid=req_int(r, "rows_valid"),
            cols=req_int(r, "cols"),
            trees=req_int(r, "trees"),
        )
        grouped[key].append(r)

    summary: Dict[str, Any] = {"groups": []}

    for key, items in sorted(grouped.items(), key=lambda kv: (kv[0].task, kv[0].growth, kv[0].depth, kv[0].leaves)):
        n = len(items)

        # Collect metrics per library.
        per_lib: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for r in items:
            for lib in ("booste_rs", "xgboost", "lightgbm"):
                v = r.get(lib)
                if v is None:
                    continue
                metrics = extract_metrics(v)
                for mk, mv in metrics.items():
                    per_lib[lib][mk].append(mv)

        group_out: Dict[str, Any] = {
            "task": key.task,
            "growth": key.growth,
            "depth": key.depth,
            "leaves": key.leaves,
            "rows_train": key.rows_train,
            "rows_valid": key.rows_valid,
            "cols": key.cols,
            "trees": key.trees,
            "n": n,
            "libs": {},
        }

        print(f"=== {key.task} | {key.growth} depth={key.depth} leaves={key.leaves} trees={key.trees} (n={n}) ===")

        for lib, metrics_map in per_lib.items():
            lib_out: Dict[str, Any] = {}
            print(f"- {lib}:")
            for mk, xs in sorted(metrics_map.items()):
                if not xs:
                    continue
                m = mean(xs)
                s = sample_std(xs)
                hw = ci95_half_width(s, len(xs))
                lib_out[mk] = {
                    "mean": m,
                    "std": s,
                    "ci95_half_width": hw,
                    "n": len(xs),
                    "values": xs,
                }
                print(f"  - {mk}: mean={m:.6g} std={s:.6g} ci95=±{hw:.6g} (n={len(xs)})")
            group_out["libs"][lib] = lib_out

        summary["groups"].append(group_out)
        print("")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
