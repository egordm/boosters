# Benchmarks

Performance benchmark results for booste-rs.

## Latest Results

See [PERFORMANCE_COMPARISON.md](../PERFORMANCE_COMPARISON.md) for the current state (December 2025).

## Historical Benchmarks

| Date | Benchmark | Description |
|------|-----------|-------------|
| 2025-12-14 | [training-and-quality](./2025-12-14-training-and-quality.md) | Training performance and quality validation |
| 2025-12-14 | [prediction-benchmarks](./2025-12-14-prediction-benchmarks.md) | Inference performance |
| 2025-11-30 | [gbtree-vs-xgboost](./2025-11-30-gbtree-vs-xgboost.md) | Initial XGBoost training comparison |
| 2025-11-30 | [lightgbm-vs-booste-rs](./2025-11-30-lightgbm-vs-booste-rs.md) | LightGBM comparison |

## Reference

- [growth_strategy_comparison.md](./growth_strategy_comparison.md) - Level-wise vs leaf-wise analysis
- [TEMPLATE.md](./TEMPLATE.md) - Template for new benchmark reports

## Running Benchmarks

```bash
# Internal training benchmarks
cargo bench --bench training_gbdt

# XGBoost comparison (requires libxgboost)
cargo bench --bench training_xgboost --features bench-xgboost

# LightGBM comparison (requires liblightgbm)
cargo bench --bench training_lightgbm --features bench-lightgbm
```
