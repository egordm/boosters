# Benchmark Results

This directory contains versioned benchmark results for tracking performance over time.

## Files

| File | Description |
|------|-------------|
| [2024-11-27-baseline.md](2024-11-27-baseline.md) | Initial baseline before M3.4 Block Traversal |

## Naming Convention

```text
YYYY-MM-DD-<description>.md
```

Examples:

- `2024-11-27-baseline.md` - Initial baseline
- `2024-12-15-block-traversal.md` - After M3.4 implementation
- `2024-12-20-simd.md` - After SIMD optimization

## Running Benchmarks

```bash
# Basic benchmarks
cargo bench

# With XGBoost comparison
cargo bench --features bench-xgboost

# HTML reports
open target/criterion/report/index.html
```
