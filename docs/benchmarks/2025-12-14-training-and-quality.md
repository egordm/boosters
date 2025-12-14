# 2025-12-14 — Training Performance + Quality Harness

This report captures the state of **training performance benchmarking** (GBDT + GBLinear) and the new **quality evaluation harness**.

## Environment

- Host: macOS 26.1 (25B78)
- CPU: Apple M1 Pro
- Commit: `5a235fb`

## What was added

### Training performance benchmarks

- Component: [benches/suites/component/train_gbdt.rs](../../benches/suites/component/train_gbdt.rs)
  - Quantization throughput (`ColMatrix -> BinnedDataset`)
  - GBDT training throughput for:
    - regression (`SquaredLoss`)
    - binary (`LogisticLoss`)
    - multiclass (`SoftmaxLoss`)
  - Thread scaling (`n_threads`)
- End-to-end: [benches/suites/e2e/train_predict_gbdt.rs](../../benches/suites/e2e/train_predict_gbdt.rs)
  - Train + predict in one benchmark iteration
- Existing component benchmark retained:
  - [benches/suites/component/train_gblinear.rs](../../benches/suites/component/train_gblinear.rs)

### Quality evaluation harness

- New CLI: [src/bin/quality_eval.rs](../../src/bin/quality_eval.rs)
  - Loads data (synthetic by default; Arrow IPC / Parquet optionally)
  - Trains booste-rs GBDT for regression/binary/multiclass
  - Computes basic, consistent metrics on a fixed split
  - Optionally runs XGBoost / LightGBM if features are enabled

## Bug fixes discovered while benchmarking

These were uncovered by running training benchmarks and fixed immediately:

- Histogram subtraction trick could panic when the **parent histogram was not cached** under small histogram cache sizes.
  - Fixed by adding a safe fallback: if the parent histogram isn’t present, build both child histograms directly.
  - Also added slot pinning to prevent eviction during the subtract path.
- Depth-wise growth had an off-by-one condition (`current_depth <= max_depth`) that could create an extra level.
  - Fixed semantics: depth-wise expansion continues while `current_depth < max_depth`.

## How to run

### Training benchmarks

- GBDT component suite:
  - `cargo bench --bench training_gbdt`
- GBDT end-to-end (train + predict):
  - `cargo bench --bench e2e_train_predict_gbdt`
- GBLinear component suite:
  - `cargo bench --features bench-training --bench training_core`

Criterion output:

- `target/criterion/`

### Quality harness

Synthetic data (no extra features):

- Regression:
  - `cargo run --bin quality_eval --release -- --task regression --synthetic 20000 50 --trees 50 --depth 6`
- Binary:
  - `cargo run --bin quality_eval --release -- --task binary --synthetic 20000 50 --trees 50 --depth 6`
- Multiclass:
  - `cargo run --bin quality_eval --release -- --task multiclass --classes 5 --synthetic 20000 50 --trees 50 --depth 6`

Arrow IPC dataset loading (requires `io-arrow`):

- `cargo run --bin quality_eval --release --features io-arrow -- --task regression --ipc path/to/data.arrow --trees 200 --depth 6`

Parquet dataset loading (requires `io-parquet`):

- `cargo run --bin quality_eval --release --features io-parquet -- --task regression --parquet path/to/data.parquet --trees 200 --depth 6`

Optional baselines:

- XGBoost: add `--features bench-xgboost`
- LightGBM: add `--features bench-lightgbm`

## Results (Criterion, `--release`)

These numbers are from running the commands in **How to run** on the environment above.

### GBDT component training (`cargo bench --bench training_gbdt`)

Quantization (RowMajor -> ColMajor -> Binned):

- `to_binned/max_bins=256/10000x50`: 4.072 ms (122.79 Melem/s)
- `to_binned/max_bins=256/50000x100`: 40.757 ms (122.68 Melem/s)
- `to_binned/max_bins=256/100000x20`: 15.894 ms (125.84 Melem/s)

Training (GBDT, `DepthWise { max_depth: 6 }`):

- regression (`10k x 50`, 50 trees): 125.87 ms (3.972 Melem/s)
- regression (`50k x 100`, 50 trees): 689.03 ms (7.257 Melem/s)
- regression (`100k x 20`, 50 trees): 347.12 ms (5.762 Melem/s)
- binary (`50k x 100`, 50 trees): 871.36 ms (5.738 Melem/s)
- multiclass (`20k x 50`, 10 classes, 30 trees): 1.5150 s (660.08 Kelem/s)

Thread scaling (regression, `50k x 100`, 30 trees):

- 1 thread: 434.77 ms (11.50 Melem/s)
- 2 threads: 275.17 ms (18.17 Melem/s)
- 4 threads: 188.75 ms (26.49 Melem/s)
- 8 threads: 155.41 ms (32.17 Melem/s)

### GBDT end-to-end (`cargo bench --bench e2e_train_predict_gbdt`)

- regression train_then_predict (`50k x 100`, 50 trees, depth=6): 594.32 ms

### GBLinear component training (`cargo bench --features bench-training --bench training_core`)

Regression training throughput:

- train/1000 rows: 2.306 ms (43.36 Melem/s)
- train/10000 rows: 14.378 ms (69.55 Melem/s)
- train/50000 rows: 61.973 ms (80.68 Melem/s)

Updater scaling (10000 rows):

- updater/sequential: 22.852 ms (43.76 Melem/s)
- updater/parallel: 13.111 ms (76.27 Melem/s)

---

## Cross-Library Training Performance Comparison

Training benchmarks comparing booste-rs vs XGBoost and LightGBM.

**Settings (all libraries)**:

- objective: regression (`SquaredLoss` / `reg:squarederror` / `regression`)
- 50 trees, max_depth=6, learning_rate=0.1 or 0.3 (see below)
- 1 thread, no sampling, L2 reg λ=1
- histogram bins: 256

### booste-rs vs XGBoost (`cargo bench --features bench-xgboost --bench training_xgboost`)

| Dataset     | booste-rs warm | booste-rs cold | XGBoost warm | XGBoost cold |
|-------------|----------------|----------------|--------------|--------------|
| small (10k×50) | **624.53 ms** | 620.87 ms      | 649.38 ms    | 705.60 ms    |
| medium (50k×100) | 3.598 s       | 3.654 s        | **2.107 s**  | 2.715 s      |

- "warm" = pre-built dataset structure (BinnedDataset / DMatrix) outside timed region
- "cold" = build data structures inside timed region

**Observation**: On "small" booste-rs is slightly faster; on "medium" XGBoost hist is ~1.7× faster. This is expected—XGBoost hist is highly optimized.

### booste-rs vs LightGBM (`cargo bench --features bench-lightgbm --bench training_lightgbm`)

| Dataset     | booste-rs warm | LightGBM cold |
|-------------|----------------|---------------|
| small (10k×50) | 689.11 ms      | **314.69 ms** |
| medium (50k×100) | 3.894 s        | **1.940 s**   |

**Observation**: LightGBM is ~2× faster on both sizes, even comparing "cold" (dataset build + train) against booste-rs "warm" (pre-binned). LightGBM's native binning + histogram code is highly vectorized.

---

## Cross-Library Model Quality Comparison

Quality harness: `cargo run --bin quality_eval --release --features bench-xgboost,bench-lightgbm`

**Settings (identical for all libraries)**:

- 20,000 synthetic rows, 50 cols
- 50 trees, max_depth=6, learning_rate=0.1
- L2 reg λ=1
- 80/20 train/valid split (fixed seed)

### Regression

| Library    | RMSE      | MAE       |
|------------|-----------|-----------|
| booste-rs  | 2.244     | 1.788     |
| XGBoost    | **1.317** | **1.049** |
| LightGBM   | 1.397     | 1.109     |

### Binary Classification

| Library    | LogLoss   | Accuracy  |
|------------|-----------|-----------|
| booste-rs  | 0.652     | 0.678     |
| XGBoost    | **0.419** | **0.844** |
| LightGBM   | 0.441     | 0.834     |

### Multiclass (5 classes)

| Library    | MLogLoss  | Accuracy  |
|------------|-----------|-----------|
| booste-rs  | 1.452     | 0.408     |
| XGBoost    | 0.937     | 0.712     |
| LightGBM   | **0.854** | **0.731** |

**Observation**: booste-rs underperforms significantly. This is a known issue—the histogram split-finding / gradient accumulation code likely has a bug or the split scoring differs from XGBoost/LightGBM's exact formula. This warrants investigation.

Possible causes (to investigate):

1. Gradient computation scale (XGBoost uses half-scaled hessian for some objectives)
2. Missing split tie-breaking logic
3. Default split scoring differences (exact vs approx)
4. Category/continuous threshold edge-case handling

---

## Notes / caveats

- Training benchmarks are intentionally heavy; Criterion will warn when it can’t collect enough samples in the default target time. This is expected for multi-second training runs.
- Cross-library training comparisons are implemented as benches but require the external libraries to be present/buildable on the machine.
- For real-dataset quality comparisons, prefer Arrow IPC / Parquet and keep split/params in a small sidecar config file (not implemented yet).
