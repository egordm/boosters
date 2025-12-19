# Benchmarks

Performance benchmark results for boosters.

## Latest Results

See [benchmark-report](./2025-12-14-benchmark-report.md) for the current consolidated report (training + inference + quality).

## Reference

- [TEMPLATE.md](./TEMPLATE.md) - Template for new benchmark reports

## Running Benchmarks

```bash
# Inference benchmarks
cargo bench --bench prediction_core
cargo bench --bench prediction_strategies
cargo bench --bench prediction_parallel

# Internal training benchmarks
cargo bench --bench training_gbdt

# XGBoost comparison (requires libxgboost)
cargo bench --bench training_xgboost --features bench-xgboost

# LightGBM comparison (requires liblightgbm)
cargo bench --bench training_lightgbm --features bench-lightgbm

# Quality harness
cargo run --release --bin quality_eval --features bench-xgboost,bench-lightgbm -- --help
```
