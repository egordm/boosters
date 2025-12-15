# 2025-12-14: Benchmark Report (Training + Inference + Quality)

## Goal

- Establish reproducible performance baselines for **training** and **inference**.
- Compare performance vs **XGBoost** and **LightGBM** where supported.
- Track model-quality deltas vs XGBoost/LightGBM on:
  - synthetic datasets (fixed generator + seed)
  - in-repo “real” fixture datasets
  - growth-strategy comparisons (depth-wise vs leaf-wise)

## Summary

- **Training (medium, single-thread)**: booste-rs is **faster than XGBoost** on this harness; LightGBM is roughly on-par/slightly faster.
- **Inference (medium model)**: booste-rs batch prediction is ~**1.15 Melem/s** at 10k rows, and is ~**4.6× faster than LightGBM** at 10k rows. LightGBM still wins on single-row latency.
- **Quality**: booste-rs remains behind XGBoost/LightGBM on the synthetic tasks used here; on the in-repo fixture datasets the gap is mixed (multiclass is close, regression is far behind).

## Environment

| Property | Value |
|----------|-------|
| CPU | Apple M1 Pro |
| OS | macOS 26.1 (25B78) |
| Rust | 1.91.1 |
| Commit | `4abc126` (dirty working tree) |

## Results

### Training performance

#### Cross-library training performance (Criterion)

Regression, histogram bins=256, single-thread. (See benchmark sources for the exact LR used per suite.)

| Dataset | booste-rs warm | booste-rs cold | XGBoost warm | XGBoost cold | LightGBM cold |
|---|---:|---:|---:|---:|---:|
| medium (50k×100, 50 trees, depth=6) | **1.505 s** | 1.515 s | 1.670 s | 2.136 s | 1.535 s |

Notes:
- warm = pre-built dataset structure outside timed region (BinnedDataset / DMatrix)
- cold = build data structures inside timed region

#### Growth strategy training cost (Criterion, booste-rs)

Regression, 50k×100, 50 trees, bins=256, `learning_rate=0.1`, single-thread.

| Growth strategy | Time | Throughput |
|---|---:|---:|
| DepthWise (`max_depth=6`) | **1.434 s** | 3.49 Melem/s |
| LeafWise (`max_leaves=64`) | 1.945 s | 2.57 Melem/s |

### Inference performance

#### Core prediction scaling (Criterion, booste-rs)

Medium model; throughput counts rows/sec (not features).

| Batch size | Time | Throughput |
|---:|---:|---:|
| 1 | 8.818 µs | 113.41 Kelem/s |
| 10 | 13.158 µs | 759.97 Kelem/s |
| 100 | 89.132 µs | **1.1219 Melem/s** |
| 1,000 | 930.35 µs | **1.0749 Melem/s** |
| 10,000 | 8.695 ms | **1.1500 Melem/s** |

Single-row (medium): 8.873 µs.

#### Traversal strategy comparison (Criterion, booste-rs)

Medium model.

| Strategy | Batch 1,000 | Batch 10,000 |
|---|---:|---:|
| Standard (no block) | 2.727 ms | 26.817 ms |
| Unrolled + block64 | **870.08 µs** | **8.745 ms** |

#### Parallel prediction scaling (Criterion, booste-rs)

Medium model, 10,000 rows.

| Threads | `par_predict` time | Throughput |
|---:|---:|---:|
| 1 | 1.391 ms | **7.1905 Melem/s** |
| 2 | 4.599 ms | 2.1742 Melem/s |
| 4 | 2.424 ms | 4.1260 Melem/s |
| 8 | 1.530 ms | 6.5346 Melem/s |

Baseline (non-parallel `predict`, 1 thread): 8.717 ms (1.1472 Melem/s).

#### Cross-library inference comparison (Criterion)

XGBoost comparison uses both:
- warm: reused booster + dmatrix (may include caching effects)
- cold_dmatrix: re-creates DMatrix each iteration

Medium model, 10,000 rows:

| Library / mode | Time |
|---|---:|
| booste-rs | **8.953 ms** |
| XGBoost warm | 32.75 µs *(cached; not comparable)* |
| XGBoost cold_dmatrix | 9.380 ms |

LightGBM comparison (medium model, 10,000 rows):

| Library | Time |
|---|---:|
| booste-rs | **9.143 ms** |
| LightGBM | 42.174 ms |

Single-row (medium model):

| Library | Time |
|---|---:|
| booste-rs | 8.505 µs |
| LightGBM | **3.700 µs** |

### Model quality

Quality harness: `cargo run --bin quality_eval --release --features bench-xgboost,bench-lightgbm`

#### Synthetic quality (20k×50, split 80/20, seed=42)

Regression:

| Library | RMSE (↓) | MAE (↓) |
|---|---:|---:|
| booste-rs | 2.239070 | 1.766966 |
| XGBoost | **1.316839** | **1.049416** |
| LightGBM | 1.334718 | 1.067370 |

Binary classification:

| Library | LogLoss (↓) | Accuracy (↑) |
|---|---:|---:|
| booste-rs | 0.488960 | 0.8083 |
| XGBoost | **0.419187** | **0.8438** |
| LightGBM | 0.421672 | 0.8397 |

Multiclass (5 classes):

| Library | MLogLoss (↓) | Accuracy (↑) |
|---|---:|---:|
| booste-rs | 1.053854 | 0.6205 |
| XGBoost | 0.937139 | 0.7117 |
| LightGBM | **0.816711** | **0.7355** |

#### In-repo fixture datasets (from XGBoost compat test cases)

These are loaded from `tests/test-cases/xgboost/gbtree/training/` via `--xgb-gbtree-case`.

Regression (`regression_simple`, 80 train / 20 test, 5 features; 20 trees, depth=3):

| Library | RMSE (↓) | MAE (↓) |
|---|---:|---:|
| booste-rs | 88.366695 | 76.759535 |
| XGBoost | **38.586405** | **29.293216** |
| LightGBM | 48.044723 | 39.625950 |

Binary (`binary_classification`, 160 train / 40 test, 10 features; 20 trees, depth=3):

| Library | LogLoss (↓) | Accuracy (↑) |
|---|---:|---:|
| booste-rs | 0.761035 | 0.5250 |
| XGBoost | 0.743262 | 0.5000 |
| LightGBM | **0.684498** | **0.5750** |

Multiclass (`multiclass`, 240 train / 60 test, 10 features; 20 trees, depth=3):

| Library | MLogLoss (↓) | Accuracy (↑) |
|---|---:|---:|
| booste-rs | **0.398740** | **0.8500** |
| XGBoost | 0.469370 | **0.8500** |
| LightGBM | 0.418020 | **0.8500** |

### Growth strategy comparison (depth-wise vs leaf-wise)

This section re-runs the comparisons that previously lived in a standalone note.

#### Synthetic (50k×100, split 80/20, seed=42, trees=200)

Depth-wise (`max_depth=6`, leaf budget=64):

| Task | booste-rs | XGBoost | LightGBM |
|---|---:|---:|---:|
| Regression (RMSE) | 1.975692 | 1.498711 | **1.493007** |
| Regression (MAE) | 1.574243 | 1.195885 | **1.192656** |
| Binary (LogLoss) | 0.525598 | **0.325241** | 0.326189 |
| Binary (Accuracy) | 0.7807 | **0.8813** | 0.8804 |
| Multiclass (mLogLoss, K=3) | 0.582110 | 0.471822 | **0.421798** |
| Multiclass (Accuracy, K=3) | 0.7876 | 0.8535 | **0.8640** |

Leaf-wise (`max_leaves=64`):

| Task | booste-rs | XGBoost | LightGBM |
|---|---:|---:|---:|
| Regression (RMSE) | 1.871939 | 1.432027 | **1.428714** |
| Regression (MAE) | 1.495256 | 1.140904 | **1.138415** |
| Binary (LogLoss) | 0.514145 | 0.311620 | **0.311223** |
| Binary (Accuracy) | 0.7978 | **0.8858** | 0.8856 |
| Multiclass (mLogLoss, K=3) | 0.670337 | 0.449122 | **0.397221** |
| Multiclass (Accuracy, K=3) | 0.7860 | 0.8559 | **0.8676** |

#### Small real-dataset sweep (external datasets; seed=42, trees=200)

These datasets are *not stored in this repo*, but are available in this workspace.

Binary: XGBoost demo agaricus (libsvm)

| Growth | booste-rs LogLoss | XGBoost LogLoss | LightGBM LogLoss |
|---|---:|---:|---:|
| depthwise (depth=6) | 0.000716 | 0.000717 | **0.000124** |
| leafwise (leaves=64) | 0.000716 | 0.000717 | **0.000121** |

Regression: UCI machine

| Growth | booste-rs RMSE | XGBoost RMSE | LightGBM RMSE |
|---|---:|---:|---:|
| depthwise (depth=6) | **78.9300** | 83.5060 | 85.8998 |
| leafwise (leaves=64) | **64.0093** | 86.4909 | 86.4276 |

Multiclass: LightGBM example dataset (K=5)

| Growth | booste-rs mLogLoss | XGBoost mLogLoss | LightGBM mLogLoss |
|---|---:|---:|---:|
| depthwise (depth=6) | **1.200947** | 1.293726 | 1.270620 |
| leafwise (leaves=64) | 1.245070 | 1.123613 | **1.106219** |

## Analysis

- Cross-library training: the medium regression benchmark currently shows booste-rs competitive with (and slightly faster than) XGBoost in this harness; LightGBM remains very strong.
- Inference: unrolled traversal + blocking is the main win for batch prediction; LightGBM retains an advantage in single-row latency.
- Quality: the synthetic quality tables and the growth-strategy synthetic sweep both show a consistent gap vs XGBoost/LightGBM that likely merits investigation in training/split logic and/or loss/leaf-weight math.

## Conclusions

- Keep the performance benchmarks as regression tests for future optimizations.
- Treat the quality gap as a top-priority correctness issue; the harness provides stable, reproducible reproductions.
- Leaf-wise growth is currently **more expensive** than depth-wise for training (on the benchmarked config), and does not close the synthetic quality gap.

## Reproducing

```bash
# Training performance (booste-rs internal)
cargo bench --bench training_gbdt

# Training performance vs baselines
cargo bench --features bench-xgboost --bench training_xgboost -- --noplot medium
cargo bench --features bench-lightgbm --bench training_lightgbm -- --noplot medium

# Inference performance (booste-rs internal)
cargo bench --bench prediction_core
cargo bench --bench prediction_strategies -- --noplot medium
cargo bench --bench prediction_parallel -- --noplot medium

# Inference comparisons
cargo bench --features bench-xgboost --bench prediction_xgboost
cargo bench --features bench-lightgbm --bench prediction_lightgbm

# Quality (synthetic)
cargo run --bin quality_eval --release --features bench-xgboost,bench-lightgbm -- \
  --task regression --synthetic 20000 50 --trees 50 --depth 6

# Quality (in-repo fixtures)
cargo run --bin quality_eval --release --features bench-xgboost,bench-lightgbm -- \
  --task regression --xgb-gbtree-case regression_simple --trees 20 --depth 3

# Growth strategy synthetic sweep (example)
cargo run --bin quality_eval --release --features bench-xgboost,bench-lightgbm -- \
  --task multiclass --classes 3 --synthetic 50000 100 --trees 200 --growth leafwise --leaves 64 --seed 42
```
