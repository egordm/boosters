# 2025-12-15: Benchmark Report (Training + Inference + Quality)

## Goal

- Establish reproducible performance baselines for **training** and **inference**.
- Compare performance vs **XGBoost** and **LightGBM** where supported.
- Track model-quality deltas vs XGBoost/LightGBM on:
  - synthetic datasets (fixed generator + seed)
  - in-repo “real” fixture datasets
  - growth-strategy comparisons (depth-wise vs leaf-wise)

## Summary

- **Training (medium, single-thread)**: booste-rs is **~on par with XGBoost** (cold DMatrix) and **slower than LightGBM**.
- **Inference (medium model)**: booste-rs batch prediction is ~**1.18 Melem/s** at 10k rows, and is ~**4.5× faster than LightGBM** at 10k rows. LightGBM still wins on single-row latency.
- **Quality**: on the synthetic tasks used here, booste-rs is competitive (often best). On the in-repo fixture datasets the gap is mixed (fixture regression remains behind; multiclass is close).

## Environment

| Property | Value |
|----------|-------|
| CPU | Apple M1 Pro, 10 cores |
| Rust | 1.91.1 |
| Commit | `a27a582` (dirty working tree) |

## Results

### Training performance

#### Cross-library training performance (Criterion)

Regression, histogram bins=256, single-thread. (See benchmark sources for the exact LR used per suite.)

All comparisons below are **cold-only** (build data structures inside the timed region). This avoids misleading “warm-cache” effects.

| Suite | Dataset | booste-rs cold_full | XGBoost cold_dmatrix | LightGBM cold_full |
|---|---|---:|---:|---:|
| vs XGBoost | medium (50k×100, 50 trees, depth=6) | 2.122 s | **2.083 s** | - |
| vs LightGBM | medium (50k×100, 50 trees, depth=6) | 2.281 s | - | **1.466 s** |

Notes:

- booste-rs `cold_full`: build matrix + bin + train inside the timed region
- XGBoost `cold_dmatrix`: recreate DMatrix inside the timed region
- LightGBM `cold_full`: train end-to-end inside the timed region

#### Growth strategy training cost (Criterion, booste-rs)

Regression, 50k×100, 50 trees, bins=256, `learning_rate=0.1`, single-thread.

| Growth strategy | Time | Throughput |
|---|---:|---:|
| DepthWise (`max_depth=6`) | **2.073 s** | 2.4121 Melem/s |
| LeafWise (`max_leaves=64`) | 2.145 s | 2.3314 Melem/s |

### Inference performance

#### Core prediction scaling (Criterion, booste-rs)

Medium model; throughput counts rows/sec (not features).

| Batch size | Time | Throughput |
|---:|---:|---:|
| 1 | 9.1476 µs | 109.318 Kelem/s |
| 10 | 12.7336 µs | 785.323 Kelem/s |
| 100 | 86.9201 µs | 1150.482 Kelem/s |
| 1,000 | 847.5624 µs | 1179.854 Kelem/s |
| 10,000 | 8.5002 ms | 1176.446 Kelem/s |

Single-row (medium): 9.3817 µs.

#### Traversal strategy comparison (Criterion, booste-rs)

Medium model.

| Strategy | Batch 1,000 | Batch 10,000 |
|---|---:|---:|
| Standard (no block) | 2.6702 ms | 26.167 ms |
| Unrolled + block64 | **863.7457 µs** | **8.6171 ms** |

#### Parallel prediction scaling (Criterion, booste-rs)

Medium model, 10,000 rows.

| Threads | `par_predict` time | Throughput |
|---:|---:|---:|
| 1 | 1.5321 ms | 6.5272 Melem/s |
| 2 | 4.5264 ms | 2.2093 Melem/s |
| 4 | 2.3786 ms | 4.2041 Melem/s |
| 8 | 1.5647 ms | 6.3910 Melem/s |

Baseline (non-parallel `predict`, 1 thread): 8.6292 ms (1158.854 Kelem/s).

#### Cross-library inference comparison (Criterion)

XGBoost comparison is **cold-only** (re-create DMatrix each iteration).

Medium model, 10,000 rows:

| Library / mode | Time |
|---|---:|
| booste-rs | **9.0760 ms** |
| XGBoost cold_dmatrix | 9.1366 ms |

LightGBM comparison (medium model, 10,000 rows):

| Library | Time |
|---|---:|
| booste-rs | **9.0787 ms** |
| LightGBM | 41.020 ms |

Single-row (medium model):

| Library | Time |
|---|---:|
| booste-rs | 8.3532 µs |
| LightGBM | **3.5517 µs** |

### Model quality

Quality harness: `cargo run --bin quality_eval --release --features bench-xgboost,bench-lightgbm`

#### Variance / confidence

All quality numbers below are **single-seed snapshots** (seed=42). When comparing close results, treat deltas as inconclusive unless they exceed run-to-run variance.

Recommended practice for transparency:

- Run $N$ seeds (e.g. 10–20) and report **mean ± std** and an approximate **95% CI**: $\mu \pm 1.96\,\sigma/\sqrt{N}$.
- Use `--out-json` + the helper script at tools/quality/aggregate_variance.py to aggregate metrics.

Example (synthetic regression, 10 seeds):

`mkdir -p docs/benchmarks/artifacts/quality/synth_regression && for s in 42 43 44 45 46 47 48 49 50 51; do cargo run --bin quality_eval --release --features bench-xgboost,bench-lightgbm -- --task regression --synthetic 20000 50 --trees 200 --depth 6 --seed $s --out-json docs/benchmarks/artifacts/quality/synth_regression/seed-$s.json; done`

`python3 tools/quality/aggregate_variance.py --out docs/benchmarks/artifacts/quality/synth_regression/summary.json docs/benchmarks/artifacts/quality/synth_regression/seed-*.json`

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

- Cross-library training: on this harness, booste-rs is ~on par with XGBoost (cold DMatrix) and slower than LightGBM.
- Inference: unrolled traversal + blocking is the main win for batch prediction; LightGBM retains an advantage in single-row latency. LightGBM prediction may also perform internal one-time setup/caching, so treat these as steady-state timings.
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

# External datasets (examples; paths are in this workspace)
cargo run --bin quality_eval --release --features bench-xgboost,bench-lightgbm -- \
  --task binary --libsvm ../xgboost/demo/data/agaricus.txt.train --trees 200 --growth depthwise --depth 6 --seed 42

cargo run --bin quality_eval --release --features bench-xgboost,bench-lightgbm -- \
  --task regression --uci-machine ../xgboost/demo/data/regression/machine.data --trees 200 --growth depthwise --depth 6 --seed 42

cargo run --bin quality_eval --release --features bench-xgboost,bench-lightgbm -- \
  --task multiclass --classes 5 --label0 ../LightGBM/examples/multiclass_classification/multiclass.train --trees 200 --growth depthwise --depth 6 --seed 42

# Quality as tests (booste-rs only)
cargo test --test quality_smoke
BOOSTERS_RUN_QUALITY=1 cargo test --test quality_smoke
```
