# booste-rs Merge Plan

This document outlines the plan for merging training infrastructure from `boosters` into `booste-rs`.

## Overview

`boosters` was developed to explore training architecture from scratch, focusing on clean histogram-based GBDT training. `booste-rs` started as an inference library with XGBoost/LightGBM compatibility.

**Goal**: Merge the best of both into `booste-rs` as the unified library.

## Design Principles

1. **Flat, clean APIs** - No complex builder patterns, params bundled in structs
2. **Static dispatch** - No `dyn Trait` where avoidable
3. **Consistent naming** - `gbdt` and `gblinear` throughout
4. **Separation of concerns** - `training/` vs `inference/`
5. **User-friendly Dataset** - Structured input that converts to training formats

---

## Module Structure

```text
src/
├── lib.rs
├── utils.rs
├── testing.rs
├── main.rs
│
├── data/
│   ├── mod.rs
│   ├── matrix.rs                     # DenseMatrix + Layout + DataMatrix + RowView
│   ├── dataset.rs                    # User-facing Dataset type
│   └── binned/
│       ├── mod.rs
│       ├── dataset.rs                # BinnedDataset (from boosters)
│       ├── builder.rs                # BinnedDatasetBuilder
│       ├── bin_mapper.rs             # BinMapper
│       ├── group.rs                  # FeatureGroup
│       └── storage.rs                # BinStorage, FeatureView
│
├── inference/
│   ├── mod.rs
│   │
│   ├── common/                       # Shared inference utilities
│   │   ├── mod.rs
│   │   └── output.rs                 # PredictionOutput
│   │
│   ├── gbdt/                         # GBDT inference
│   │   ├── mod.rs
│   │   ├── tree.rs                   # TreeStorage
│   │   ├── node.rs                   # Node, SplitCondition
│   │   ├── leaf.rs                   # LeafValue, ScalarLeaf, VectorLeaf
│   │   ├── forest.rs                 # Forest
│   │   ├── categories.rs             # CategoriesStorage
│   │   ├── unrolled.rs               # UnrolledTreeLayout
│   │   ├── predictor.rs              # Predictor
│   │   ├── simd.rs                   # Optional SIMD traversal (feature = simd)
│   │   └── traversal.rs              # TreeTraversal strategies
│   │
│   ├── gblinear/                     # GBLinear inference
│   │   ├── mod.rs
│   │   └── model.rs                  # LinearModel
│
├── compat/                           # External model loading
│   ├── mod.rs
│   ├── xgboost/
│   └── lightgbm/
│
└── training/
    ├── mod.rs
    │
    ├── objectives/                   # Objective trait + implementations
    │   ├── mod.rs                    # Objective trait
    │   ├── regression.rs             # SquaredLoss, Pinball, Huber, Poisson
    │   └── classification.rs         # Logistic, Softmax, Hinge, LambdaRank
    │
    ├── metrics/                      # Metric trait + implementations
    │   ├── mod.rs                    # Metric trait
    │   ├── regression.rs             # Rmse, Mae, Mape, etc.
    │   └── classification.rs         # LogLoss, Accuracy, Auc, Ndcg, Map
    │
    ├── eval.rs                       # EvalSet
    ├── gradients.rs                  # GradientBuffer
    ├── callback.rs                   # EarlyStopping, etc.
    ├── logger.rs                     # TrainingLogger, Verbosity
    │
    ├── sampling/
    │   ├── mod.rs
    │   ├── row.rs                    # RowSampler, GOSS
    │   └── column.rs                 # ColSampler
    │
    ├── gbdt/                         # GBDT training
    │   ├── mod.rs
    │   ├── trainer.rs                # GBDTTrainer, GBDTParams
    │   ├── grower.rs                 # TreeGrower
    │   ├── expansion.rs              # GrowthStrategy, GrowthState
    │   ├── partition.rs              # RowPartitioner
    │   ├── categorical.rs            # CatBitset
    │   ├── optimization.rs           # OptimizationProfile
    │   │
    │   ├── histograms/
    │   │   ├── mod.rs
    │   │   ├── pool.rs               # HistogramPool
    │   │   └── ops.rs                # build_histograms, subtract
    │   │
    │   ├── split/
    │   │   ├── mod.rs
    │   │   ├── types.rs              # SplitInfo, SplitType
    │   │   ├── gain.rs               # GainParams
    │   │   └── find.rs               # GreedySplitter
    │   │
    └── gblinear/                     # GBLinear training
        ├── mod.rs
        ├── trainer.rs                # GBLinearTrainer, GBLinearParams
        ├── selector.rs               # Feature selectors
        └── updater.rs                # Coordinate descent updater
```

---

## Key Types

### User-Facing Dataset

```rust
/// User-facing dataset with per-feature storage
pub struct Dataset {
    /// Per-feature columns with type information
    features: Vec<FeatureColumn>,
    /// Number of samples (rows)
    n_samples: usize,
    /// Target values
    targets: Vec<f32>,
    /// Optional sample weights
    weights: Option<Vec<f32>>,
}

/// A single feature column with type and metadata
pub enum FeatureColumn {
    /// Continuous numeric feature
    Numeric {
        values: Vec<f32>,
        name: Option<String>,
    },
    /// Categorical feature with integer categories
    Categorical {
        values: Vec<i32>,
        name: Option<String>,
        n_categories: u32,
    },
}

impl Dataset {
    /// Create from numeric-only features
    pub fn from_numeric(features: ColMatrix<f32>, targets: Vec<f32>) -> Self;
    
    /// Add sample weights
    pub fn with_weights(self, weights: Vec<f32>) -> Self;
    
    /// Convert to BinnedDataset for GBDT training
    pub fn to_binned(&self, max_bins: u32) -> Result<BinnedDataset, BuildError>;
    
    /// Validate and get features for GBLinear (no categoricals allowed)
    pub fn for_gblinear(&self) -> Result<ColMatrix<f32>, DataError>;
    
    // Accessors
    pub fn targets(&self) -> &[f32];
    pub fn weights(&self) -> Option<&[f32]>;
    pub fn n_samples(&self) -> usize;
    pub fn n_features(&self) -> usize;
}
```

### EvalSet

```rust
/// Named evaluation dataset
pub struct EvalSet<'a> {
    pub name: &'a str,
    pub dataset: &'a Dataset,
}

impl<'a> EvalSet<'a> {
    pub fn new(name: &'a str, dataset: &'a Dataset) -> Self {
        Self { name, dataset }
    }
}
```

### Objective Trait

```rust
/// Objective (loss) function for gradient boosting
pub trait Objective: Send + Sync {
    /// Number of outputs per sample (1 for regression/binary, K for multiclass)
    fn n_outputs(&self) -> usize { 1 }
    
    /// Compute gradients and hessians
    fn compute_gradients(
        &self,
        n_rows: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
        gradients: &mut GradientBuffer,
    );
    
    /// Compute initial base score(s)
    fn compute_base_score(
        &self,
        targets: &[f32],
        weights: &[f32],
    ) -> Vec<f32>;
    
    /// Transform raw output to final predictions (sigmoid, softmax, etc.)
    fn transform(&self, output: &mut [f32]) {
        // Default: no transform
    }
    
    fn name(&self) -> &'static str;
}
```

### Metric Trait

```rust
/// Evaluation metric
pub trait Metric: Send + Sync {
    /// Compute metric value
    fn compute(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
    ) -> f64;
    
    /// Whether higher values are better
    fn higher_is_better(&self) -> bool;
    
    fn name(&self) -> &'static str;
}
```

### GradientBuffer

```rust
/// Column-major gradient storage for efficient histogram building
pub struct GradientBuffer {
    grads: Vec<f32>,
    hess: Vec<f32>,
    n_samples: usize,
    n_outputs: usize,
}

impl GradientBuffer {
    pub fn new(n_samples: usize, n_outputs: usize) -> Self;
    
    /// Zero-copy slice for one output (for histogram building)
    pub fn output_grads(&self, output: usize) -> &[f32];
    pub fn output_hess(&self, output: usize) -> &[f32];
    
    /// Sum all samples for an output
    pub fn sum_output(&self, output: usize) -> (f64, f64);
    
    /// Sum specific rows
    pub fn sum_rows(&self, output: usize, rows: &[u32]) -> (f64, f64);
}
```

### GBDT Training

```rust
pub struct GBDTParams {
    pub n_trees: u32,
    pub learning_rate: f32,
    pub growth_strategy: GrowthStrategy,
    pub gain: GainParams,
    pub row_sampling: RowSamplingParams,
    pub col_sampling: ColSamplingParams,
    pub early_stopping: Option<EarlyStoppingParams>,
    pub n_threads: usize,
    pub seed: u64,
}

pub struct GBDTTrainer<O: Objective, M: Metric> {
    objective: O,
    metric: M,
    params: GBDTParams,
}

impl<O: Objective, M: Metric> GBDTTrainer<O, M> {
    pub fn new(objective: O, metric: M, params: GBDTParams) -> Self;
    
    /// Train model
    /// Pass empty slice for eval_sets if no evaluation needed
    pub fn train(
        &self,
        train: &Dataset,
        eval_sets: &[EvalSet],
    ) -> Option<GBDTModel>;
}

pub struct GBDTModel {
    pub forest: inference::gbdt::Forest,
    pub base_scores: Vec<f32>,
    pub n_features: usize,
    pub n_outputs: usize,
}

impl GBDTModel {
    pub fn predict<D: DataMatrix<Element = f32>>(&self, features: &D) -> PredictionOutput;
}
```

### GBLinear Training

```rust
pub struct GBLinearParams {
    pub n_rounds: u32,
    pub learning_rate: f32,
    pub reg_lambda: f32,
    pub reg_alpha: f32,
    pub feature_selector: FeatureSelectorKind,
    pub early_stopping: Option<EarlyStoppingParams>,
    pub seed: u64,
}

pub struct GBLinearTrainer<O: Objective, M: Metric> {
    objective: O,
    metric: M,
    params: GBLinearParams,
}

impl<O: Objective, M: Metric> GBLinearTrainer<O, M> {
    pub fn new(objective: O, metric: M, params: GBLinearParams) -> Self;
    
    /// Train model
    pub fn train(
        &self,
        train: &Dataset,
        eval_sets: &[EvalSet],
    ) -> Option<GBLinearModel>;
}

pub struct GBLinearModel {
    pub linear: inference::gblinear::LinearModel,
    pub base_scores: Vec<f32>,
    pub n_outputs: usize,
}

impl GBLinearModel {
    pub fn predict<D: DataMatrix<Element = f32>>(&self, features: &D) -> PredictionOutput;
}
```

---

## Example Usage

```rust
use booste_rs::{
    data::{Dataset, FeatureColumn},
    training::{
        GBDTTrainer, GBDTParams, GrowthStrategy,
        GBLinearTrainer, GBLinearParams,
        SquaredLoss, Rmse, EvalSet,
    },
};

// Create dataset with mixed feature types
let dataset = Dataset::new(vec![
    FeatureColumn::Numeric { values: col1, name: Some("age".into()) },
    FeatureColumn::Numeric { values: col2, name: Some("income".into()) },
    FeatureColumn::Categorical { values: col3, name: Some("city".into()), n_categories: 10 },
], targets)
.with_weights(weights);

let val_dataset = Dataset::new(val_features, val_targets);

// GBDT training
let trainer = GBDTTrainer::new(
    SquaredLoss,
    Rmse,
    GBDTParams {
        n_trees: 100,
        learning_rate: 0.1,
        ..Default::default()
    },
);

let model = trainer.train(&dataset, &[EvalSet::new("val", &val_dataset)])?;

// Or without eval sets
let model = trainer.train(&dataset, &[])?;

// Prediction
let predictions = model.predict(&test_features);
```

---

## Migration Steps

### Phase 1: Structure

1. Create `inference/` directory structure
2. Move existing tree/forest/predict code to `inference/gbdt/`
3. Move linear model to `inference/gblinear/`
4. Rename `SoAForest` → `Forest`, `SoATreeStorage` → `TreeStorage`

### Phase 2: Data

1. Port `DenseMatrix` from boosters (replace existing)
2. Port `BinnedDataset` from boosters to `data/binned/`
3. Create `Dataset` with `FeatureColumn` enum

### Phase 3: Training Infrastructure

1. Port `objectives/` from boosters, add `n_outputs()` and `transform()`
2. Port `metrics/` from boosters
3. Refactor `GradientBuffer` with column-major layout
4. Port `sampling/` from boosters

### Phase 4: GBDT Training

1. Port `training/gbdt/` from boosters
2. Update to use new `GradientBuffer`
3. Add conversion from training tree to inference `Forest`

### Phase 5: GBLinear Training

1. Refactor existing `GBLinearTrainer` to match GBDT interface
2. Add eval set support
3. Ensure consistent API

### Phase 6: Cleanup

1. Remove deprecated booste-rs training code
2. Update all tests
3. Update examples
4. Archive boosters repo

---

## Code to Remove from booste-rs

| Module                          | Reason                                  |
| ------------------------------- | --------------------------------------- |
| `training::gbtree::quantize/`   | Replaced by `data/binned/` from boosters |
| `training::gbtree::histogram/`  | Replaced by boosters' version           |
| `training::gbtree::partition.rs`| Replaced by boosters' version           |
| `training::gbtree::grower/`     | Replaced by boosters' version           |
| `training::loss/`               | Renamed to `objectives/`                |
| `data/dense.rs`                 | Replaced by boosters' `matrix.rs`       |
| `data/layout.rs`                | Merged into `matrix.rs`                 |

---

## Naming Conventions

| Concept           | Name                                    |
| ----------------- | --------------------------------------- |
| Tree booster      | `gbdt` (not `gbtree`)                   |
| Linear booster    | `gblinear`                              |
| Loss function     | `Objective` (not `Loss`)                |
| Gradient storage  | `GradientBuffer` (in `gradients.rs`)    |
| Tree storage      | `TreeStorage` (not `SoATreeStorage`)    |
| Forest            | `Forest` (not `SoAForest`)              |

---

## Open Questions

1. **Multi-output training**: Should GBDT train K trees per round for K-class, or one tree with vector leaves?
   - Current plan: K trees per round (simpler, matches XGBoost)

2. **Categorical encoding for GBLinear**: One-hot or error?
   - Current plan: Error if categoricals present, user must encode

3. **Feature names propagation**: Should trained models keep feature names?
   - Current plan: Yes, for explainability
