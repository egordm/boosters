# Growth strategy quality comparison (depth-wise vs leaf-wise)

This note compares **booste-rs** depth-wise and leaf-wise tree growth against **XGBoost** and **LightGBM** under a “same-ish params” setup.

The goal is *qualitative parity*: confirm that both growth strategies in booste-rs are on par with established implementations when the constraints are matched as closely as the libraries allow.

## Setup

- Dataset: synthetic dense $X \in \mathbb{R}^{50000 \times 100}$ with train/valid split 80/20.
- Tasks:
  - Regression (linear target + noise)
  - Binary classification (logistic)
  - Multiclass classification ($K=3$)
- Common parameters:
  - `trees=200`
  - `learning_rate=0.1`
  - Histogram bins: `max_bin=256` / `max_bin=256` / `BinnedDatasetBuilder(..., 256)`
  - L2 regularization: `reg_lambda=1.0` / `lambda=1.0` / `lambda_l2=1.0`
  - No subsampling / column sampling (all = 1.0)
  - `seed=42`

Ran via [src/bin/quality_eval.rs](../src/bin/quality_eval.rs).

### Growth strategy mapping

There is no perfect 1:1 mapping across libraries, but the following is the closest “fair” correspondence:

**Depth-wise** (cap depth, allow up to $2^{\text{depth}}$ leaves)

- booste-rs: `GrowthStrategy::DepthWise { max_depth = 6 }`
- XGBoost: `grow_policy=depthwise`, `max_depth=6`, `tree_method=hist`, `max_leaves=0` (unused)
- LightGBM: `max_depth=6`, `num_leaves=2^depth=64` (set high enough to not bind)

**Leaf-wise** (cap leaves, allow depth to vary)

- booste-rs: `GrowthStrategy::LeafWise { max_leaves = 64 }`
- XGBoost: `grow_policy=lossguide`, `max_leaves=64`, `max_depth=0` (unlimited)
- LightGBM: `num_leaves=64`, `max_depth=-1` (unlimited)

## Results (synthetic; seed=42)

All metrics are computed on the same validation split.

### Depth-wise (depth=6, theoretical leaf budget=64)

| Task | booste-rs | XGBoost | LightGBM |
|---|---:|---:|---:|
| Regression (RMSE) | **1.483721** | 1.498711 | 1.493007 |
| Regression (MAE) | **1.186293** | 1.195885 | 1.192656 |
| Binary (LogLoss) | 0.326684 | **0.325241** | 0.326189 |
| Binary (Accuracy) | 0.8780 | **0.8813** | 0.8804 |
| Multiclass (mLogLoss) | **0.359661** | 0.471822 | 0.421798 |
| Multiclass (Accuracy) | **0.8699** | 0.8535 | 0.8640 |

### Leaf-wise (max_leaves=64)

| Task | booste-rs | XGBoost | LightGBM |
|---|---:|---:|---:|
| Regression (RMSE) | **1.420528** | 1.432027 | 1.428714 |
| Regression (MAE) | **1.129851** | 1.140904 | 1.138415 |
| Binary (LogLoss) | 0.311565 | 0.311620 | **0.311223** |
| Binary (Accuracy) | 0.8840 | **0.8858** | 0.8856 |
| Multiclass (mLogLoss) | **0.338635** | 0.449122 | 0.397221 |
| Multiclass (Accuracy) | **0.8768** | 0.8559 | 0.8676 |

## Results (small real-dataset sweep; seed=42)

All runs use the same harness split logic (80/20 train/valid) and the same hyperparams as the synthetic section unless noted.

### Binary: XGBoost demo agaricus (libsvm)

Dataset: `/Users/egordm/projects/rust/xgboost/demo/data/agaricus.txt.train` (mushroom).

Depth-wise (depth=6, leaf budget=64)

| Metric | booste-rs | XGBoost | LightGBM |
|---|---:|---:|---:|
| LogLoss | 0.000716 | 0.000717 | **0.000124** |
| Accuracy | **1.0000** | **1.0000** | **1.0000** |

Leaf-wise (max_leaves=64)

| Metric | booste-rs | XGBoost | LightGBM |
|---|---:|---:|---:|
| LogLoss | 0.000716 | 0.000717 | **0.000121** |
| Accuracy | **1.0000** | **1.0000** | **1.0000** |

### Regression: UCI computer hardware (XGBoost demo)

Dataset: `/Users/egordm/projects/rust/xgboost/demo/data/regression/machine.data`.

Target/Features: predicts PRP using 6 numeric features (MYCT..CHMAX), ignoring vendor/model strings.

Depth-wise (depth=6, leaf budget=64)

| Metric | booste-rs | XGBoost | LightGBM |
|---|---:|---:|---:|
| RMSE | **77.882854** | 83.506004 | 85.899750 |
| MAE | **30.374693** | 32.674996 | 31.734260 |

Leaf-wise (max_leaves=64)

| Metric | booste-rs | XGBoost | LightGBM |
|---|---:|---:|---:|
| RMSE | **80.897583** | 86.490929 | 86.427604 |
| MAE | **30.780661** | 32.342942 | 32.708485 |

### Multiclass: LightGBM example dataset (label-first dense)

Dataset: `/Users/egordm/projects/rust/LightGBM/examples/multiclass_classification/multiclass.train` (K=5).

Depth-wise (depth=6, leaf budget=64)

| Metric | booste-rs | XGBoost | LightGBM |
|---|---:|---:|---:|
| mLogLoss | **1.208078** | 1.293726 | 1.270620 |
| Accuracy | **0.4650** | 0.4250 | 0.4400 |

Leaf-wise (max_leaves=64)

| Metric | booste-rs | XGBoost | LightGBM |
|---|---:|---:|---:|
| mLogLoss | 1.129189 | 1.123613 | **1.106219** |
| Accuracy | **0.5143** | 0.4979 | 0.5136 |

## Qualitative takeaways

- **Depth-wise parity looks good** on regression and binary: booste-rs is essentially on par with XGBoost/LightGBM under matched depth constraints.
- **Leaf-wise parity looks good** on regression and binary: all three improve vs depth-wise, and booste-rs tracks the same trend and magnitude.
- **Multiclass shows a consistent gap in the opposite direction** on this synthetic benchmark (booste-rs better than both XGBoost and LightGBM).
  - This is *not automatically a correctness signal* by itself: synthetic generators can interact with library defaults in non-obvious ways.
  - We verified that XGBoost/LightGBM multiclass outputs are proper probabilities (row sums exactly 1.0) in the harness, so this is not a reshaping/normalization artifact.

## Important caveats

- **“Same params” is approximate** across libraries:
  - Leaf-wise growth is inherently different from depth-wise. Even with the same leaf budget, leaf-wise concentrates splits where gain is highest and can go deeper on a subset of branches.
  - Regularization and minimum-child constraints have similar intent across libraries, but implementation details differ.
- **Synthetic quality can be misleading**. For final parity claims, repeat on a few real datasets (e.g. UCI-like tabular for multiclass) using the same harness.

## Reproducing

Depth-wise:

`cargo run --release --bin quality_eval --features bench-xgboost,bench-lightgbm -- --task regression --synthetic 50000 100 --trees 200 --growth depthwise --depth 6 --seed 42`

Leaf-wise:

`cargo run --release --bin quality_eval --features bench-xgboost,bench-lightgbm -- --task regression --synthetic 50000 100 --trees 200 --growth leafwise --depth 6 --seed 42`

Real datasets:

- Agaricus (binary, libsvm):
  `cargo run --release --bin quality_eval --features bench-xgboost,bench-lightgbm -- --task binary --libsvm /Users/egordm/projects/rust/xgboost/demo/data/agaricus.txt.train --trees 200 --growth depthwise --depth 6 --seed 42`

- UCI machine (regression):
  `cargo run --release --bin quality_eval --features bench-xgboost,bench-lightgbm -- --task regression --uci-machine /Users/egordm/projects/rust/xgboost/demo/data/regression/machine.data --trees 200 --growth depthwise --depth 6 --seed 42`

- LightGBM multiclass example (dense label-first):
  `cargo run --release --bin quality_eval --features bench-xgboost,bench-lightgbm -- --task multiclass --classes 5 --label0 /Users/egordm/projects/rust/LightGBM/examples/multiclass_classification/multiclass.train --trees 200 --growth depthwise --depth 6 --seed 42`
