# Benchmark Results

This directory contains versioned benchmark results for tracking performance over time.

## Benchmark Index

### Inference Optimization (Nov 2024)

| Date | File | Description |
|------|------|-------------|
| 2024-11-27 | [baseline](2024-11-27-baseline.md) | Initial baseline before block traversal |
| 2024-11-28 | [m352-unrolled](2024-11-28-m352-unrolled.md) | Predictor refactor with unrolled layout |
| 2024-11-28 | [simd-analysis](2024-11-28-simd-analysis.md) | SIMD investigation (concluded: not beneficial) |
| 2024-11-28 | [m37-thread-parallelism](2024-11-28-m37-thread-parallelism.md) | Thread parallelism with Rayon |
| 2024-11-29 | [m38-performance-validation](2024-11-29-m38-performance-validation.md) | Final performance validation |

### Training Optimization (Nov 2024)

| Date | File | Description |
|------|------|-------------|
| 2024-11-29 | [gradient-batch](2025-11-29-gradient-batch.md) | Batch gradient computation optimization |
| 2024-11-29 | [gradient-soa](2025-11-29-gradient-soa.md) | SoA vs AoS gradient storage layout |
| 2024-11-29 | [matrix-layout-training](2025-11-29-matrix-layout-training.md) | ColMajor vs CSC for training |

## Naming Convention

```text
YYYY-MM-DD-<description>.md
```

## Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench prediction
cargo bench --bench linear_training

# With XGBoost comparison (requires xgboost feature)
cargo bench --features bench-xgboost

# View HTML reports
open target/criterion/report/index.html
```

## Adding New Benchmarks

1. Create a new file following the naming convention
2. Use [TEMPLATE.md](TEMPLATE.md) as a starting point
3. Update this README with an entry in the appropriate section
4. Commit both the benchmark results and README update together
