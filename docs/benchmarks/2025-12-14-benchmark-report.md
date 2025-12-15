# 2025-12-14: Benchmark Report (Training + Inference + Quality)

## Goal

- Establish reproducible performance baselines for **training** and **inference**.
- Compare performance vs **XGBoost** and **LightGBM** where supported.
- Track model-quality deltas vs XGBoost/LightGBM on:
  - synthetic datasets (fixed generator + seed)
  - in-repo “real” fixture datasets
  - growth-strategy comparisons (depth-wise vs leaf-wise)

## Summary

- **Training (medium, single-thread)**: booste-rs is **slower than XGBoost** and **slower than LightGBM** on this harness.
- **Inference (medium model)**: booste-rs batch prediction is ~**1.20 Melem/s** at 10k rows, and is ~**4.5× faster than LightGBM** at 10k rows. LightGBM still wins on single-row latency.
- **Quality**: on the synthetic tasks used here, booste-rs is competitive (often best). On the in-repo fixture datasets the gap is mixed (fixture regression remains behind; multiclass is close).

## Environment

| Property | Value |
|----------|-------|
| CPU | Apple M1 Pro |
| OS | macOS 26.1 (25B78) |
| Rust | 1.91.1 |
| Commit | `957982e` (dirty working tree) |

## Results

### Training performance

#### Cross-library training performance (Criterion)

Regression, histogram bins=256, single-thread. (See benchmark sources for the exact LR used per suite.)

| Dataset | booste-rs warm | booste-rs cold | XGBoost warm | XGBoost cold | LightGBM cold |
|---|---:|---:|---:|---:|---:|
| medium (50k×100, 50 trees, depth=6) | 2.024 s | 2.081 s | **1.637 s** | 2.147 s | 1.476 s |

Notes:

- warm = pre-built dataset structure outside timed region (BinnedDataset / DMatrix)
- cold = build data structures inside timed region

#### Growth strategy training cost (Criterion, booste-rs)

Regression, 50k×100, 50 trees, bins=256, `learning_rate=0.1`, single-thread.

| Growth strategy | Time | Throughput |
|---|---:|---:|
| DepthWise (`max_depth=6`) | **2.027 s** | 2.47 Melem/s |
| LeafWise (`max_leaves=64`) | 2.098 s | 2.38 Melem/s |

### Inference performance

#### Core prediction scaling (Criterion, booste-rs)

Medium model; throughput counts rows/sec (not features).

| Batch size | Time | Throughput |
|---:|---:|---:|
| 1 | 8.877 µs | 112.65 Kelem/s |
| 10 | 12.518 µs | 798.83 Kelem/s |
| 100 | 84.922 µs | **1.1776 Melem/s** |
| 1,000 | 840.29 µs | **1.1901 Melem/s** |
| 10,000 | 8.369 ms | **1.1950 Melem/s** |

Single-row (medium): 8.834 µs.

#### Traversal strategy comparison (Criterion, booste-rs)

Medium model.

| Strategy | Batch 1,000 | Batch 10,000 |
|---|---:|---:|
| Standard (no block) | 2.665 ms | 25.981 ms |
| Unrolled + block64 | **856.23 µs** | **8.600 ms** |

#### Parallel prediction scaling (Criterion, booste-rs)

Medium model, 10,000 rows.

| Threads | `par_predict` time | Throughput |
|---:|---:|---:|
| 1 | 1.311 ms | **7.6269 Melem/s** |
| 2 | 4.545 ms | 2.2003 Melem/s |
| 4 | 2.491 ms | 4.0140 Melem/s |
| 8 | 1.482 ms | 6.7458 Melem/s |

Baseline (non-parallel `predict`, 1 thread): 8.600 ms (1.1628 Melem/s).

#### Cross-library inference comparison (Criterion)

XGBoost comparison uses both:

- warm: reused booster + dmatrix (may include caching effects)
- cold_dmatrix: re-creates DMatrix each iteration

Medium model, 10,000 rows:

| Library / mode | Time |
|---|---:|
| booste-rs | 9.187 ms |
| XGBoost warm | 32.067 µs *(cached; not comparable)* |
| XGBoost cold_dmatrix | **9.103 ms** |

LightGBM comparison (medium model, 10,000 rows):

| Library | Time |
|---|---:|
| booste-rs | **9.248 ms** |
| LightGBM | 41.456 ms |

Single-row (medium model):

| Library | Time |
|---|---:|
| booste-rs | 8.405 µs |
| LightGBM | **3.665 µs** |

### Model quality

Quality harness: `cargo run --bin quality_eval --release --features bench-xgboost,bench-lightgbm`

#### Synthetic quality (20k×50, split 80/20, seed=42)

Regression:

| Library | RMSE (↓) | MAE (↓) |
|---|---:|---:|
| booste-rs | 1.317053 | **1.045699** |
| XGBoost | **1.316839** | 1.049416 |
| LightGBM | 1.334718 | 1.067370 |

Binary classification:

| Library | LogLoss (↓) | Accuracy (↑) |
|---|---:|---:|
| booste-rs | **0.417135** | **0.8438** |
| XGBoost | 0.419187 | **0.8438** |
| LightGBM | 0.421672 | 0.8397 |

Multiclass (5 classes):

| Library | MLogLoss (↓) | Accuracy (↑) |
|---|---:|---:|
| booste-rs | **0.746448** | **0.7505** |
| XGBoost | 0.937139 | 0.7117 |
| LightGBM | 0.816711 | 0.7355 |

#### In-repo fixture datasets (from XGBoost compat test cases)

These are loaded from `tests/test-cases/xgboost/gbtree/training/` via `--xgb-gbtree-case`.

Regression (`regression_simple`, 80 train / 20 test, 5 features; 20 trees, depth=3):

| Library | RMSE (↓) | MAE (↓) |
|---|---:|---:|
| booste-rs | 44.924254 | 36.194095 |
| XGBoost | **38.586405** | **29.293216** |
| LightGBM | 48.044723 | 39.625950 |

Binary (`binary_classification`, 160 train / 40 test, 10 features; 20 trees, depth=3):

| Library | LogLoss (↓) | Accuracy (↑) |
|---|---:|---:|
| booste-rs | 0.761148 | 0.5250 |
| XGBoost | 0.743262 | 0.5000 |
| LightGBM | **0.684498** | **0.5750** |

Multiclass (`multiclass`, 240 train / 60 test, 10 features; 20 trees, depth=3):

| Library | MLogLoss (↓) | Accuracy (↑) |
|---|---:|---:|
| booste-rs | **0.399065** | **0.8500** |
| XGBoost | 0.469370 | **0.8500** |
| LightGBM | 0.418020 | **0.8500** |

### Growth strategy comparison (depth-wise vs leaf-wise)

This section re-runs the comparisons that previously lived in a standalone note.

#### Synthetic (50k×100, split 80/20, seed=42, trees=200)

Depth-wise (`max_depth=6`, leaf budget=64):

| Task | booste-rs | XGBoost | LightGBM |
|---|---:|---:|---:|
| Regression (RMSE) | **1.483721** | 1.498711 | 1.493007 |
| Regression (MAE) | **1.186293** | 1.195885 | 1.192656 |
| Binary (LogLoss) | 0.326684 | **0.325241** | 0.326189 |
| Binary (Accuracy) | 0.8780 | **0.8813** | 0.8804 |
| Multiclass (mLogLoss, K=3) | **0.359661** | 0.471822 | 0.421798 |
| Multiclass (Accuracy, K=3) | **0.8699** | 0.8535 | 0.8640 |

Leaf-wise (`max_leaves=64`):

| Task | booste-rs | XGBoost | LightGBM |
|---|---:|---:|---:|
| Regression (RMSE) | **1.420528** | 1.432027 | 1.428714 |
| Regression (MAE) | **1.129851** | 1.140904 | 1.138415 |
| Binary (LogLoss) | 0.311565 | 0.311620 | **0.311223** |
| Binary (Accuracy) | 0.8840 | **0.8858** | 0.8856 |
| Multiclass (mLogLoss, K=3) | **0.338635** | 0.449122 | 0.397221 |
| Multiclass (Accuracy, K=3) | **0.8768** | 0.8559 | 0.8676 |

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
| depthwise (depth=6) | **77.8829** | 83.5060 | 85.8998 |
| leafwise (leaves=64) | **80.8976** | 86.4909 | 86.4276 |

Multiclass: LightGBM example dataset (K=5)

| Growth | booste-rs mLogLoss | XGBoost mLogLoss | LightGBM mLogLoss |
|---|---:|---:|---:|
| depthwise (depth=6) | **1.208078** | 1.293726 | 1.270620 |
| leafwise (leaves=64) | 1.129189 | 1.123613 | **1.106219** |

## Analysis

- Cross-library training: on this harness, XGBoost and LightGBM are faster than booste-rs for the medium regression benchmark.
- Inference: unrolled traversal + blocking is the main win for batch prediction; LightGBM retains an advantage in single-row latency.
- Quality: synthetic tasks are generally at parity or better than the baselines here; fixture regression remains behind, and the external sweep is mixed.

## Conclusions

- Keep the performance benchmarks as regression tests for future optimizations.
- Keep the fixture regression gap on the radar; the harness provides stable, reproducible reproductions.
- Leaf-wise growth is slightly **more expensive** than depth-wise for training (on the benchmarked config), and does not consistently improve quality on these tasks.

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
  --task regression --synthetic 20000 50 --trees 50 --depth 6 --seed 42

# Quality (in-repo fixtures)
cargo run --bin quality_eval --release --features bench-xgboost,bench-lightgbm -- \
  --task regression --xgb-gbtree-case regression_simple --trees 20 --depth 3 --seed 42

# Growth strategy synthetic sweep (example)
cargo run --bin quality_eval --release --features bench-xgboost,bench-lightgbm -- \
  --task multiclass --classes 3 --synthetic 50000 100 --trees 200 --growth leafwise --leaves 64 --seed 42

# Quality as tests (booste-rs only)
cargo test --test quality_smoke
BOOSTERS_RUN_QUALITY=1 cargo test --test quality_smoke
```
