# RFC-0020: API Layering and Usability Redesign

| Status  | Accepted |
|---------|----------|
| Created | 2024-12-20 |
| Authors | @egordm  |

## Summary

This RFC proposes a comprehensive API redesign to establish clear abstraction layers, improve naming consistency, and simplify the library for Rust users while preparing for Python bindings.

### Key Decisions Summary

1. **Three-layer architecture**: Model (high) → Trainer/Predictor (mid) → Repr (low)
2. **Loss-based naming**: `PinballLoss`, `ArctanLoss` (not task-based like `Quantile`)
3. **Objective enum wraps structs**: Enum holds pre-constructed loss structs, no `::new()` in delegation
4. **Streamlined Objective trait**: Only required methods + sensible defaults; removed bloat
5. **Nested param groups**: `TreeParams`, `RegularizationParams`, `SamplingParams` - same at all levels
6. **No wrapper types**: Predictions are just `ColMatrix<f32>`, no forwarding methods
7. **Model stores config**: `model.config()` for introspection and retraining
8. **Builder via bon**: Validation in `build()` returns `Result`

---

## Motivation

The current API has evolved organically and has several pain points:

1. **Inconsistent naming**: Uses "GBDT" and "GBLinear" but also "Forest", "Tree", "LinearModel" interchangeably without clear layer boundaries. Mixed patterns like `TaskKind`, `ObjectiveFunction`, `Objective` trait.

2. **No clear abstraction layers**: Users cannot easily identify which types are for high-level usage vs. low-level customization.

3. **Configuration scattered across calls**: Objective and metric are passed to `train()` rather than being part of the model configuration.

4. **Excessive forwarding methods**: Types like `GBDTModel` forward many calls to internal `Forest`, adding maintenance burden.

5. **PredictionOutput duplication**: `PredictionOutput` is essentially `ColMatrix<f32>` with a different name.

---

## Design Goals

1. **Clear three-layer architecture**:
   - **Model Layer** (High): sklearn-like ease of use, configuration-driven
   - **Trainer/Predictor Layer** (Mid): Component flexibility, swap implementations
   - **Repr Layer** (Low): Raw structures for algorithm developers

2. **Consistent naming conventions** (see Section 2)

3. **Flat, composable types**: Minimize forwarding; expose internal components via accessors

4. **Python binding readiness**: Enums and configurations suitable for PyO3

---

## Detailed Design

### 1. Module Reorganization

```text
boosters/
├── model/              # High-level models (sklearn-like)
│   ├── mod.rs          # Model trait, re-exports
│   ├── gbdt.rs         # GBDTModel, GBDTConfig, TreeParams, etc.
│   └── gblinear.rs     # GBLinearModel, GBLinearConfig
├── training/           # Mid-level trainers, objectives, metrics
│   ├── mod.rs
│   ├── objectives/     # ObjectiveFn trait and implementations
│   │   ├── mod.rs          # ObjectiveFn trait, Objective enum
│   │   ├── regression.rs   # SquaredLoss, PinballLoss, etc.
│   │   └── classification.rs # LogisticLoss, SoftmaxLoss, etc.
│   ├── metrics/        # MetricFn trait and implementations
│   │   ├── mod.rs          # MetricFn trait, Metric enum
│   │   ├── regression.rs   # RmseMetric, MaeMetric, etc.
│   │   └── classification.rs # LogLossMetric, AucMetric, etc.
│   ├── gbdt/
│   │   └── trainer.rs      # GBDTTrainer<O, M>
│   └── gblinear/
│       └── trainer.rs      # GBLinearTrainer<O, M>
├── inference/          # Mid-level predictors
│   ├── mod.rs
│   ├── common.rs       # PredictionKind
│   ├── gbdt/
│   │   └── predictor.rs    # Predictor<T>
│   └── gblinear/
│       └── predict.rs
├── repr/               # Low-level representations
│   ├── mod.rs
│   ├── tree.rs         # Tree, MutableTree
│   ├── forest.rs       # Forest
│   ├── linear.rs       # LinearModel
│   └── ...
├── data/               # Data types
│   ├── matrix.rs       # DenseMatrix, RowMatrix, ColMatrix
│   ├── dataset.rs      # Dataset
│   └── binned.rs       # BinnedDataset
├── explainability/     # SHAP, importance
├── compat/             # XGBoost/LightGBM loading
└── io/                 # Serialization
```

---

### 2. Naming Conventions

Establish consistent naming patterns across the codebase:

| Pattern | Usage | Examples |
|---------|-------|----------|
| `FooFn` (trait) | Behavior contract (function-like) | `ObjectiveFn`, `MetricFn` |
| `Foo` (enum) | Selection enum wrapping implementations | `Objective`, `Metric` |
| `FooKind` (enum) | Simple discriminator (no/minimal data) | `TaskKind`, `PredictionKind` |
| `Foo` (struct) | Concrete implementation | `SquaredLoss`, `PinballLoss`, `Rmse` |
| `FooConfig` (struct) | High-level user configuration | `GBDTConfig`, `GBLinearConfig` |
| `FooParams` (struct) | Mid-level component parameters | `GrowerParams`, `HistogramParams` |

**Rationale**:

- `Fn` suffix for traits that define function-like behavior (like `std::ops::Fn`)
- Base name for enum that selects which implementation to use (`Objective`, `Metric`)
- `Kind` suffix for simple discriminators (no significant embedded data)
- `Config` for user-facing configuration structs, `Params` for internal component parameters

**Migration**:

- `ObjectiveFunction` → `Objective` (enum selecting objective)
- `Objective` (trait) → `ObjectiveFn` (trait defining behavior)
- `MetricFunction` → `Metric`
- `GBDTParams` (high-level) → `GBDTConfig`
- `PredictionOutput` → `ColMatrix<f32>` directly

---

### 3. Objective and Metric Architecture

#### ObjectiveFn Trait (Streamlined)

```rust
// In training/objectives/mod.rs

/// Core trait for gradient-based optimization.
/// 
/// **Design rationale**: The trait is intentionally minimal. Only methods
/// required for training are included. Metadata concerns (validation, defaults)
/// are handled elsewhere.
/// 
/// Named `ObjectiveFn` to distinguish from `Objective` enum (like `std::ops::Fn`).
pub trait ObjectiveFn: Send + Sync {
    // === Required methods ===
    
    /// Compute gradients and hessians in-place.
    /// 
    /// # Layout (column-major)
    /// - `predictions[output * n_rows + row]`
    /// - `grad_hess[output * n_rows + row]`
    fn compute_gradients(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],  // empty = unweighted
        grad_hess: &mut [GradsTuple],
    );
    
    /// Compute initial base score (bias) from targets.
    fn compute_base_score(
        &self,
        n_rows: usize,
        n_outputs: usize,
        targets: &[f32],
        weights: &[f32],
        outputs: &mut [f32],  // length = n_outputs
    );
    
    // === Optional methods with sensible defaults ===
    
    /// Number of outputs per sample (1 for most, K for multiclass/multi-quantile).
    fn n_outputs(&self) -> usize { 1 }
    
    /// Transform raw predictions in-place (margins → probabilities, etc.)
    /// Default: no transformation (regression).
    fn transform_predictions(&self, _raw: &mut [f32], _n_rows: usize, _n_outputs: usize) -> PredictionKind {
        PredictionKind::Value
    }
    
    /// Task kind for metadata/serialization. Default: Regression.
    fn task_kind(&self) -> TaskKind { TaskKind::Regression }
    
    /// Objective name for logging. Default: type name.
    fn name(&self) -> &'static str { std::any::type_name::<Self>() }
}

// **Removed from trait (bloat)**:
// - `target_schema()` - validation belongs at Dataset layer
// - `default_metric()` - users should specify metric explicitly
```

**Design note on `transform_predictions`**:

The transformation from margins to probabilities (sigmoid, softmax) is tied to the **objective**, not the task type. This is because:

1. **Different objectives can have different transforms for the same task**: Logistic loss uses sigmoid, but a custom classification loss might use a different link function.

2. **The objective defines the inverse link**: Gradient boosting trains on the link function's scale (log-odds for logistic). The objective knows its own inverse link.

3. **Consistency with XGBoost/LightGBM**: Both libraries tie transformation to objective, not task.

The model stores the objective, so `predict()` can apply the correct transformation automatically. Users who want raw margins call `predict_raw()`.

#### Objective Enum (Wraps Concrete Loss Structs)

```rust
use std::sync::Arc;

/// Selection enum wrapping pre-constructed loss implementations.
/// 
/// **Design**: The enum holds actual loss structs, not raw data. This avoids
/// repeated `::new()` calls in delegation and keeps loss naming consistent
/// (PinballLoss, not Quantile).
#[derive(Clone)]
pub enum Objective {
    // Regression losses
    SquaredLoss(SquaredLoss),
    AbsoluteLoss(AbsoluteLoss),
    HuberLoss(HuberLoss),
    PseudoHuberLoss(PseudoHuberLoss),
    PoissonLoss(PoissonLoss),
    PinballLoss(PinballLoss),      // For quantile regression
    ArctanLoss(ArctanLoss),         // Robust regression (future)

    // Classification losses
    LogisticLoss(LogisticLoss),
    HingeLoss(HingeLoss),
    SoftmaxLoss(SoftmaxLoss),
    
    // Ranking losses
    LambdaRankLoss(LambdaRankLoss),
    
    // Custom: wraps user-provided implementation
    Custom(Arc<dyn ObjectiveFn>),
}

/// Delegation is trivial - just call inner method, no ::new() needed.
impl ObjectiveFn for Objective {
    fn compute_gradients(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
        grad_hess: &mut [GradsTuple],
    ) {
        match self {
            Self::SquaredLoss(inner) => inner.compute_gradients(n_rows, n_outputs, predictions, targets, weights, grad_hess),
            Self::PinballLoss(inner) => inner.compute_gradients(n_rows, n_outputs, predictions, targets, weights, grad_hess),
            Self::Custom(inner) => inner.compute_gradients(n_rows, n_outputs, predictions, targets, weights, grad_hess),
            // ... all variants delegate to inner
        }
    }
    
    fn compute_base_score(&self, n_rows: usize, n_outputs: usize, targets: &[f32], weights: &[f32], outputs: &mut [f32]) {
        match self {
            Self::SquaredLoss(inner) => inner.compute_base_score(n_rows, n_outputs, targets, weights, outputs),
            Self::PinballLoss(inner) => inner.compute_base_score(n_rows, n_outputs, targets, weights, outputs),
            Self::Custom(inner) => inner.compute_base_score(n_rows, n_outputs, targets, weights, outputs),
            // ...
        }
    }
    
    fn n_outputs(&self) -> usize {
        match self {
            Self::SoftmaxLoss(inner) => inner.n_outputs(),
            Self::PinballLoss(inner) => inner.n_outputs(),
            Self::Custom(inner) => inner.n_outputs(),
            _ => 1,
        }
    }
    
    // ... other trait methods delegate to inner
}

/// Convenience constructors - users don't need to import loss structs.
impl Objective {
    /// Squared error (L2) loss for regression.
    pub fn squared() -> Self {
        Self::SquaredLoss(SquaredLoss)
    }
    
    /// Absolute error (L1) loss for robust regression.
    pub fn absolute() -> Self {
        Self::AbsoluteLoss(AbsoluteLoss)
    }
    
    /// Pinball loss for single quantile (e.g., median with 0.5).
    pub fn pinball(alpha: f32) -> Self {
        Self::PinballLoss(PinballLoss::new(vec![alpha]))
    }
    
    /// Pinball loss for multiple quantiles.
    pub fn pinball_multi(alphas: &[f32]) -> Self {
        Self::PinballLoss(PinballLoss::new(alphas.to_vec()))
    }
    
    /// Huber loss with specified delta.
    pub fn huber(delta: f32) -> Self {
        Self::HuberLoss(HuberLoss::new(delta))
    }
    
    /// Binary logistic loss.
    pub fn logistic() -> Self {
        Self::LogisticLoss(LogisticLoss)
    }
    
    /// Softmax loss for multiclass.
    pub fn softmax(n_classes: usize) -> Self {
        Self::SoftmaxLoss(SoftmaxLoss::new(n_classes))
    }
}

impl Default for Objective {
    fn default() -> Self { Self::SquaredLoss(SquaredLoss) }
}
```

**Key design decisions**:

- Enum wraps pre-constructed loss structs (no `::new()` in match arms)
- Loss-based naming (`PinballLoss`, `ArctanLoss`), not task-based (`Quantile`)
- Users implement `ObjectiveFn` trait for custom objectives
- `Objective::Custom(Arc<dyn ObjectiveFn>)` for runtime polymorphism
- Convenience constructors so users don't need to import loss structs

#### Metric Architecture (Parallel Design)

Metrics follow the exact same pattern as objectives: `MetricFn` trait + `Metric` enum.

```rust
// In training/metrics/mod.rs

/// Metric trait - behavior contract for evaluation metrics.
/// 
/// Named `MetricFn` to distinguish from `Metric` enum (like `std::ops::Fn`).
pub trait MetricFn: Send + Sync {
    fn compute(&self, predictions: &[f32], targets: &[f32], weights: &[f32]) -> f64;
    fn name(&self) -> &'static str { std::any::type_name::<Self>() }
    fn higher_is_better(&self) -> bool;
}

/// Selection enum wrapping pre-constructed metric implementations.
#[derive(Clone)]
pub enum Metric {
    // Regression metrics
    Rmse(RmseMetric),
    Mae(MaeMetric),
    Mape(MapeMetric),
    Huber(HuberMetric),
    PoissonDeviance(PoissonDevianceMetric),
    Quantile(QuantileMetric),
    
    // Classification metrics
    LogLoss(LogLossMetric),
    Accuracy(AccuracyMetric),
    Auc(AucMetric),
    MulticlassLogLoss(MulticlassLogLossMetric),
    MulticlassAccuracy(MulticlassAccuracyMetric),
    
    // Ranking metrics
    Ndcg(NdcgMetric),
    Map(MapMetric),
    
    Custom(Arc<dyn MetricFn>),
}

impl MetricFn for Metric {
    fn compute(&self, predictions: &[f32], targets: &[f32], weights: &[f32]) -> f64 {
        match self {
            Self::Rmse(inner) => inner.compute(predictions, targets, weights),
            Self::Mae(inner) => inner.compute(predictions, targets, weights),
            Self::Custom(inner) => inner.compute(predictions, targets, weights),
            // ... all variants delegate to inner
        }
    }
    
    fn higher_is_better(&self) -> bool {
        match self {
            Self::Auc(_) | Self::Accuracy(_) | Self::MulticlassAccuracy(_) 
            | Self::Ndcg(_) | Self::Map(_) => true,
            _ => false,  // Lower is better for loss metrics
        }
    }
}

/// Convenience constructors.
impl Metric {
    pub fn rmse() -> Self { Self::Rmse(RmseMetric) }
    pub fn mae() -> Self { Self::Mae(MaeMetric) }
    pub fn logloss() -> Self { Self::LogLoss(LogLossMetric) }
    pub fn accuracy(threshold: f32) -> Self { Self::Accuracy(AccuracyMetric::new(threshold)) }
    pub fn auc() -> Self { Self::Auc(AucMetric) }
    pub fn quantile(alphas: &[f32]) -> Self { Self::Quantile(QuantileMetric::new(alphas.to_vec())) }
    pub fn ndcg(k: Option<usize>) -> Self { Self::Ndcg(NdcgMetric::new(k)) }
}

impl Default for Metric {
    fn default() -> Self { Self::Rmse(RmseMetric) }
}
```

#### TaskKind (Single Definition)

```rust
// In training/objectives/mod.rs - single source of truth
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskKind {
    Regression,
    BinaryClassification,
    MulticlassClassification { n_classes: usize },
    Ranking,
}
```

---

### 4. Configuration Layer

#### Nested Parameter Groups

Instead of a flat config with many parameters, use nested structs that group related settings. 
These same structs are used at both high-level (GBDTConfig) and mid-level (GBDTTrainerParams).

**Module location**: Parameter group structs live in their respective model modules:

- `model::gbdt::{TreeParams, RegularizationParams, SamplingParams}` - GBDT-specific params
- `model::gblinear::RegularizationParams` - GBLinear-specific (may differ from GBDT's)

Shared params (if any) would live in a `model::common` module. Currently, `TreeParams` and `SamplingParams` are GBDT-only since GBLinear doesn't use trees or row/column sampling.

`RegularizationParams` exists in both model modules with different fields appropriate to each:
- GBDT: `lambda`, `alpha`, `min_child_weight`, `min_gain`
- GBLinear: `lambda`, `alpha` (no tree-specific params)

```rust
/// Tree structure parameters (GBDT only).
#[derive(Debug, Clone, Default)]
pub struct TreeParams {
    /// Maximum tree depth. Default: 6.
    pub max_depth: u32,
    /// Maximum number of leaves (optional, overrides max_depth).
    pub max_leaves: Option<u32>,
    /// Tree growth strategy (depthwise, lossguide, etc.).
    pub growth_strategy: GrowthStrategy,
}

/// Regularization parameters.
#[derive(Debug, Clone)]
pub struct RegularizationParams {
    /// L2 regularization term. Default: 1.0.
    pub lambda: f32,
    /// L1 regularization term. Default: 0.0.
    pub alpha: f32,
    /// Minimum sum of hessians in a leaf. Default: 1.0.
    pub min_child_weight: f32,
    /// Minimum gain required for a split. Default: 0.0.
    pub min_gain: f32,
}

impl Default for RegularizationParams {
    fn default() -> Self {
        Self { lambda: 1.0, alpha: 0.0, min_child_weight: 1.0, min_gain: 0.0 }
    }
}

/// Sampling parameters.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Row subsampling ratio. Default: 1.0 (no sampling).
    pub subsample: f32,
    /// Column subsampling per tree. Default: 1.0.
    pub colsample_bytree: f32,
    /// Column subsampling per level. Default: 1.0.
    pub colsample_bylevel: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self { subsample: 1.0, colsample_bytree: 1.0, colsample_bylevel: 1.0 }
    }
}
```

#### High-Level GBDTConfig (Composes Parameter Groups)

```rust
use bon::bon;

/// High-level configuration for GBDT models.
/// 
/// Uses nested parameter groups for semantic organization. The same structs
/// are used by both GBDTConfig and GBDTTrainerParams - no conversion overhead.
#[derive(Debug, Clone)]
pub struct GBDTConfig {
    // What to optimize
    pub objective: Objective,
    pub metric: Option<Metric>,
    
    // Boosting parameters (top-level, always relevant)
    pub n_trees: u32,
    pub learning_rate: f32,
    
    // Nested parameter groups
    pub tree: TreeParams,
    pub regularization: RegularizationParams,
    pub sampling: SamplingParams,
    
    // Early stopping
    pub early_stopping_rounds: Option<u32>,
    
    // Resource control
    pub n_threads: Option<NonZeroUsize>,
    pub seed: u64,
    pub verbosity: Verbosity,
}

#[bon]
impl GBDTConfig {
    #[builder(finish_fn = build)]
    pub fn try_new(
        #[builder(default)] objective: Objective,
        #[builder(default)] metric: Option<Metric>,
        #[builder(default = 100)] n_trees: u32,
        #[builder(default = 0.3)] learning_rate: f32,
        #[builder(default)] tree: TreeParams,
        #[builder(default)] regularization: RegularizationParams,
        #[builder(default)] sampling: SamplingParams,
        #[builder(default)] early_stopping_rounds: Option<u32>,
        #[builder(default)] n_threads: Option<NonZeroUsize>,
        #[builder(default = 42)] seed: u64,
        #[builder(default)] verbosity: Verbosity,
    ) -> Result<Self, ConfigError> {
        // Validation
        if learning_rate <= 0.0 || learning_rate > 1.0 {
            return Err(ConfigError::InvalidLearningRate(learning_rate));
        }
        if n_trees == 0 {
            return Err(ConfigError::InvalidNTrees);
        }
        sampling.validate()?;  // Nested struct can have its own validation
        
        Ok(Self {
            objective, metric, n_trees, learning_rate,
            tree, regularization, sampling,
            early_stopping_rounds, n_threads, seed, verbosity,
        })
    }
}
```

**Usage**:

```rust
// All defaults
let config = GBDTConfig::builder().build()?;

// Override top-level params
let config = GBDTConfig::builder()
    .objective(Objective::logistic())
    .n_trees(200)
    .learning_rate(0.1)
    .build()?;

// Override a parameter group
let config = GBDTConfig::builder()
    .regularization(RegularizationParams { lambda: 2.0, ..Default::default() })
    .build()?;

// Mix top-level and nested
let config = GBDTConfig::builder()
    .n_trees(500)
    .tree(TreeParams { max_depth: 8, ..Default::default() })
    .sampling(SamplingParams { subsample: 0.8, ..Default::default() })
    .build()?;
```

**Benefits of nested approach**:

1. **Semantic grouping**: Related params are together (all regularization in one place)
2. **Same structs everywhere**: High-level and mid-level use identical types
3. **Focused customization**: Users can focus on one concern at a time
4. **Clear boundary isn't needed**: The code is clean whether you think of it as high or mid level

---

### 5. Predictions: Just Return ColMatrix

**Problem**: `PredictionOutput` duplicates `ColMatrix<f32>`. The RFC previously proposed a 
`Predictions` wrapper with forwarding methods like `column()` - but that's exactly the 
useless forwarding we want to avoid.

**Solution**: Return `ColMatrix<f32>` directly. The "kind" is implicit in which method you call.

```rust
impl GBDTModel {
    /// Predict with automatic transformation based on objective.
    /// - Classification: returns probabilities
    /// - Regression: returns values
    pub fn predict(&self, features: &impl DataMatrix<Element = f32>) -> ColMatrix<f32>;
    
    /// Predict raw margins without transformation.
    pub fn predict_raw(&self, features: &impl DataMatrix<Element = f32>) -> ColMatrix<f32>;
}
```

**No wrapper needed**. Users know what they're getting based on which method they called.

```rust
// Get probabilities for classification
let probs = model.predict(&features);
let class_0_probs = probs.col_slice(0);  // Direct access, no forwarding

// Get raw margins if needed
let margins = model.predict_raw(&features);
```

**Migration path**:

- Remove `PredictionOutput` entirely
- Remove proposed `Predictions` wrapper
- Prediction methods return `ColMatrix<f32>` directly
- Users access data via `ColMatrix` methods (no forwarding)

---

### 6. Model Layer

```rust
/// High-level GBDT model.
/// 
/// Stores the trained forest, metadata, AND training configuration.
/// This enables introspection and retraining with the same params.
pub struct GBDTModel {
    forest: Forest<ScalarLeaf>,
    meta: ModelMeta,
    config: GBDTConfig,  // Training params are preserved
}

impl GBDTModel {
    /// Train from Dataset.
    /// 
    /// `eval_sets` are optional validation sets for early stopping and monitoring.
    pub fn train(
        dataset: &Dataset,
        config: GBDTConfig,
        eval_sets: &[(&str, &Dataset)],  // Named eval sets
    ) -> Result<Self, TrainError> {
        // Convert Dataset to BinnedDataset (internal)
        let binned = BinnedDataset::from_dataset(dataset, &binning_config)?;
        
        // Use config's nested params directly (same types at all levels)
        let trainer = GBDTTrainer::new(
            config.objective.clone(),
            config.metric.clone(),
            &config,  // Trainer can borrow config directly
        );
        
        // Convert eval sets
        let binned_evals: Vec<_> = eval_sets.iter()
            .map(|(name, ds)| EvalSet::from_dataset(name, ds))
            .collect();
        
        // Train
        let forest = trainer.train(&binned, dataset.targets(), dataset.weights(), &binned_evals)?;
        
        // Task kind from objective
        let task = config.objective.task_kind();
        let meta = ModelMeta { task, /* ... */ };
        
        Ok(Self { forest, meta, config })
    }
    
    /// Construct from trained forest with config.
    pub fn from_parts(forest: Forest<ScalarLeaf>, meta: ModelMeta, config: GBDTConfig) -> Self;
    
    // === Accessors (no forwarding methods!) ===
    
    /// Access underlying forest.
    pub fn forest(&self) -> &Forest<ScalarLeaf> { &self.forest }
    
    /// Access model metadata.
    pub fn meta(&self) -> &ModelMeta { &self.meta }
    
    /// Access training configuration.
    /// Useful for introspection or retraining with same params.
    pub fn config(&self) -> &GBDTConfig { &self.config }
    
    // === Prediction ===
    
    /// Predict with automatic transformation (probabilities for classification).
    pub fn predict(&self, features: &impl DataMatrix<Element = f32>) -> ColMatrix<f32>;
    
    /// Predict raw margins without transformation.
    pub fn predict_raw(&self, features: &impl DataMatrix<Element = f32>) -> ColMatrix<f32>;
    
    // === Serialization ===
    pub fn write_to<W: Write>(&self, writer: W) -> Result<(), SerializeError>;
    pub fn read_from<R: Read>(reader: R) -> Result<Self, DeserializeError>;
}
```

#### GBLinearModel (Parallel Pattern)

`GBLinearModel` follows the same patterns as `GBDTModel`:

```rust
/// High-level gradient-boosted linear model.
pub struct GBLinearModel {
    linear: LinearModel,
    meta: ModelMeta,
    config: GBLinearConfig,
}

impl GBLinearModel {
    /// Train from Dataset.
    pub fn train(
        dataset: &Dataset,
        config: GBLinearConfig,
        eval_sets: &[(&str, &Dataset)],
    ) -> Result<Self, TrainError>;
    
    // Accessors (same pattern as GBDTModel)
    pub fn linear(&self) -> &LinearModel { &self.linear }
    pub fn meta(&self) -> &ModelMeta { &self.meta }
    pub fn config(&self) -> &GBLinearConfig { &self.config }
    
    // Predictions (same pattern as GBDTModel)
    pub fn predict(&self, features: &impl DataMatrix<Element = f32>) -> ColMatrix<f32>;
    pub fn predict_raw(&self, features: &impl DataMatrix<Element = f32>) -> ColMatrix<f32>;
}
```

**Key consistency points**:

1. **Same API shape**: `train()`, `predict()`, `predict_raw()`, accessors
2. **Stores config**: For introspection and retraining
3. **Uses same Objective/Metric enums**: Same objectives work for both model types
4. **Same nested param pattern**: `GBLinearConfig` has its own `RegularizationParams`

---

### 7. Trainer Layer (Mid-Level)

```rust
/// GBDT trainer with pluggable objective and metric.
/// 
/// Uses generics for static dispatch (zero-cost). Objective implements
/// ObjectiveFn, so it can be passed directly.
pub struct GBDTTrainer<O: ObjectiveFn, M: MetricFn> {
    objective: O,
    metric: M,
    config: GBDTConfig,  // Uses the same config type - no conversion needed!
}

impl<O: ObjectiveFn, M: MetricFn> GBDTTrainer<O, M> {
    pub fn new(objective: O, metric: M, config: &GBDTConfig) -> Self;
    
    /// Train on pre-binned data.
    pub fn train(
        &self,
        dataset: &BinnedDataset,
        targets: &[f32],
        weights: &[f32],
        eval_sets: &[EvalSet<'_>],
    ) -> Result<Forest<ScalarLeaf>, TrainError>;
}

/// Training errors.
#[derive(Debug, thiserror::Error)]
pub enum TrainError {
    #[error("targets length {actual} < n_rows {expected}")]
    InvalidTargetsLength { actual: usize, expected: usize },
    
    #[error("weights length {actual} doesn't match n_rows {expected}")]
    InvalidWeightsLength { actual: usize, expected: usize },
}
```

**Key insight**: With nested parameter groups (`TreeParams`, `RegularizationParams`, `SamplingParams`), 
the trainer can use `GBDTConfig` directly. No separate `GBDTTrainerParams` type needed! The boundary 
between "high-level" and "mid-level" is just which methods you call, not different types.

```rust
// Mid-level usage: trainer directly uses config's nested params
let grower_params = GrowerParams {
    tree: config.tree.clone(),
    regularization: config.regularization.clone(),
    // ...
};
```

---

### 8. Reducing Forwarding Methods

Replace forwarding with accessor + direct access:

**Before**:

```rust
impl GBDTModel {
    pub fn n_trees(&self) -> usize { self.forest.n_trees() }
    pub fn n_groups(&self) -> usize { self.forest.n_groups() }
    pub fn base_scores(&self) -> &[f32] { self.forest.base_scores() }
}
```

**After**:

```rust
impl GBDTModel {
    pub fn forest(&self) -> &Forest<ScalarLeaf> { &self.forest }
}

// Usage
let n_trees = model.forest().n_trees();
```

---

### 9. Python Bindings (Future, Undetermined)

> **Note**: Python bindings are not a confirmed feature. This section is exploratory
> and shows how the API *could* map to Python if/when bindings are added. The primary
> focus is the Rust API.

```python
from boosters import GBDTModel, Dataset, GBDTConfig

# Dataset from numpy arrays
dataset = Dataset.from_numpy(X_train, y_train, weights=sample_weights)

# Config with kwargs
config = GBDTConfig(
    n_trees=100,
    learning_rate=0.1,
    objective="binary_logistic",  # String alias supported
)

# Train
model = GBDTModel.train(dataset, eval_sets=[("valid", eval_set)], config=config)

# Predict
predictions = model.predict(X_test)
probabilities = predictions.values.as_numpy()

# Serialize
with open("model.bin", "wb") as f:
    model.write_to(f)
```

---

### 10. Crate-Level Re-exports

```rust
// lib.rs

// High-level API
pub use model::{GBDTModel, GBLinearModel};
pub use data::{Dataset, FeatureColumn};

// Configuration (nested param groups included)
pub use model::gbdt::{GBDTConfig, TreeParams, RegularizationParams, SamplingParams};
pub use model::gblinear::GBLinearConfig;

// Objectives and Metrics (for customization)
pub use training::objectives::{ObjectiveFn, Objective, TaskKind};
pub use training::metrics::{MetricFn, Metric};

// Matrix types (predictions are just ColMatrix)
pub use data::{ColMatrix, RowMatrix, DenseMatrix};
```

---

## Migration Path

1. **Phase 1**: Add `bon` dependency, implement `GBDTConfig` builder
2. **Phase 2**: Rename `ObjectiveFunction` → `Objective`, `Objective (trait)` → `ObjectiveFn`
3. **Phase 3**: Replace `PredictionOutput` with `ColMatrix<f32>`
4. **Phase 4**: Update `GBDTModel::train()` to accept `Dataset` and `GBDTConfig`
5. **Phase 5**: Remove forwarding methods, add accessors
6. **Phase 6**: Add Python bindings (if confirmed)

No deprecation period needed (no external users). Hard cut to new API.

---

## Alternatives Considered

### A. Manual Builder Implementation

**Rejected**: `bon` generates correct code automatically with compile-time checks. Manual implementation is error-prone and verbose.

### B. Keep `ObjectiveFunction` Naming

**Rejected**: Inconsistent with trait naming (`Objective`) and other enum patterns (`TaskKind`).

### C. Keep `PredictionOutput` Separate from `ColMatrix`

**Rejected**: They have identical semantics. Unifying reduces cognitive load and code duplication.

### D. Use "Params" for High-Level Configuration

**Rejected**: "Config" better signals user-facing configuration vs. internal parameters.

### E. Function Pointers for Custom Objectives

**Rejected**: Trait implementation is more idiomatic Rust, enables better error handling, and supports stateful objectives.

---

## Dependencies

New crate dependency:

```toml
[dependencies]
bon = "3.8"
```

---

## Open Questions

None remaining.

---

## References

- [bon crate documentation](https://bon-rs.com/)
- XGBoost Python API (parameter design)
- LightGBM parameter organization
- scikit-learn estimator interface

---

## Changelog

### Design Decisions (Consolidated from 20+ Review Rounds)

**Architecture**:

- Three-layer architecture: Model → Trainer/Predictor → Repr
- Objectives/metrics under `training/` (not root level)
- `TaskKind` single definition in `training/objectives/`

**Naming Conventions**:

- `FooFn` (trait), `Foo` (enum wrapping implementations), `FooKind` (simple discriminator)
- Loss-based naming: `PinballLoss`, `ArctanLoss` (not task-based like `Quantile`)
- `Config` for high-level, `Params` for mid-level

**Objective & Metric Pattern**:

- `ObjectiveFn` / `MetricFn` traits define behavior contracts
- `Objective` / `Metric` enums wrap pre-constructed implementations (no `::new()` in delegation)
- Convenience constructors: `Objective::pinball(0.5)`, `Metric::rmse()`
- `Objective::Custom(Arc<dyn ObjectiveFn>)` for custom objectives
- `transform_predictions` in objective (not task) - objectives know their inverse link

**ObjectiveFn Trait (Streamlined)**:

- Required: `compute_gradients()`, `compute_base_score()`
- Optional with defaults: `n_outputs()`, `transform_predictions()`, `task_kind()`, `name()`
- **Removed**: `target_schema()`, `default_metric()` (bloat)

**Configuration Layer**:

- Nested parameter groups: `TreeParams`, `RegularizationParams`, `SamplingParams`
- Param groups live in their model module (e.g., `model::gbdt::TreeParams`)
- Same structs at high and mid level (no conversion needed)
- Builder pattern via `bon` with `#[builder(finish_fn = build)]`
- Validation in `build()` - invalid configs cannot be constructed

**Predictions**:

- Return `ColMatrix<f32>` directly (no wrapper)
- No forwarding methods - users access data via `ColMatrix` methods
- Implicit kind: `predict()` = transformed, `predict_raw()` = margins

**Model Storage**:

- Model stores config (`GBDTConfig`/`GBLinearConfig`) for introspection and retraining
- `model.config()` accessor for training params

**GBLinearModel Parity**:

- Same API shape as GBDTModel: `train()`, `predict()`, `predict_raw()`, accessors
- Uses same `Objective`/`Metric` enums
- Stores config for introspection

**Training**:

- `train()` accepts `eval_sets: &[(&str, &Dataset)]` for early stopping and monitoring
- High-level uses `Dataset`, mid-level uses `BinnedDataset`

**Encapsulation**:

- `fn forest(&self)`, `fn meta(&self)`, `fn config(&self)` accessors
- No forwarding methods - access internal data directly
