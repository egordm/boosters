# Benchmark Results

This directory contains versioned benchmark results for tracking performance over time.

## Files

| File | Description |
|------|-------------|
| [2024-11-27-baseline.md](2024-11-27-baseline.md) | Initial baseline before block traversal |
| [2024-11-28-m352-unrolled.md](2024-11-28-m352-unrolled.md) | Predictor refactor with unrolled layout |
| [2024-11-28-simd-analysis.md](2024-11-28-simd-analysis.md) | SIMD investigation (concluded: not beneficial) |
| [2024-11-28-m37-thread-parallelism.md](2024-11-28-m37-thread-parallelism.md) | Thread parallelism with Rayon |
| [2024-11-29-m38-performance-validation.md](2024-11-29-m38-performance-validation.md) | Final performance validation |

## Naming Convention

```text
YYYY-MM-DD-<description>.md
```

## Running Benchmarks

```bash
# Basic benchmarks
cargo bench

# With XGBoost comparison
cargo bench --features bench-xgboost

# HTML reports
open target/criterion/report/index.html
```
