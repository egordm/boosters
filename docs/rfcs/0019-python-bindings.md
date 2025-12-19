# RFC-0019: Python Bindings & Model Abstraction

- **Status**: Draft
- **Created**: 2025-12-19
- **Updated**: 2025-12-19
- **Depends on**: RFC-0001, RFC-0002, RFC-0008, RFC-0014, RFC-0016
- **Scope**: High-level model abstraction for Rust and Python APIs

## Summary

This RFC introduces a high-level model abstraction layer with separate types for GBDT and GBLinear models. Key features:

1. **Separate Rust model types** - `GBDTModel` and `GBLinearModel` for type-safe APIs
2. **Python bindings** - PyO3-based bindings with NumPy/Pandas zero-copy interop
3. **Hierarchical configuration** - Type-safe params (`CommonParams` + model-specific)
4. **Sklearn wrappers** - `GBDTRegressor`, `GBDTClassifier` for familiar API

## Motivation

### Current State

The current API requires users to:

1. Choose between `GBDTTrainer` and `GBLinearTrainer`
2. Construct appropriate data structures (`BinnedDataset` vs `ColMatrix`)
3. Select inference strategies (`Predictor<UnrolledTraversal6>` vs `SimplePredictor`)
4. Manually handle base scores, output transforms, etc.

### Pain Points

| Issue | Impact |
|-------|--------|
| Multiple paths for GBDT/GBLinear | Confusing for new users |
| Manual strategy selection | Suboptimal defaults |
| No high-level model object | Can't serialize/deserialize cleanly |
| No Python bindings | Blocks adoption |

### Goals

1. **Simple by default**: One line to train, one line to predict
2. **Powerful when needed**: Full control via configuration
3. **Zero-copy Python**: No data duplication between Python and Rust  
4. **Type-safe**: Separate model types prevent misuse at compile time

## Design

### Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│                         Python Layer                             │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────────────────┐│
│  │ GBDTBooster  │  │ GBLinBooster  │  │ Dataset (zero-copy)    ││
│  │ (Python)     │  │ (Python)      │  │ NumPy/Pandas wrapper   ││
│  └──────┬───────┘  └───────┬───────┘  └────────────┬───────────┘│
│         │                  │                       │             │
│  ┌──────┴──────────────────┴───────────────────────┘             │
│  │ Sklearn wrappers: GBDTRegressor, GBDTClassifier, etc.        │
│  └───────────────────────────────────────────────────────────────│
└──────────────────────────────┬───────────────────────────────────┘
                               │ PyO3
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Rust Layer                               │
│  ┌─────────────────────────────┐  ┌─────────────────────────┐   │
│  │       GBDTModel             │  │     GBLinearModel       │   │
│  │  ┌───────────┬───────────┐  │  │  ┌─────────┬─────────┐  │   │
│  │  │ forest    │ meta      │  │  │  │ weights │ meta    │  │   │
│  │  │ (Forest)  │ (names)   │  │  │  │ (f32)   │ (names) │  │   │
│  │  └───────────┴───────────┘  │  │  └─────────┴─────────┘  │   │
│  └───────────────┬─────────────┘  └───────────┬─────────────┘   │
│                  │                            │                  │
│                  ▼                            ▼                  │
│  ┌─────────────────┐              ┌──────────────────┐          │
│  │ GBDTTrainer     │              │ GBLinearTrainer  │          │
│  │ (training)      │              │ (training)       │          │
│  └─────────────────┘              └──────────────────┘          │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   Hierarchical Params                      │  │
│  │  CommonParams ──► GBDTParams / GBLinearParams             │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Structures

#### GBDTModel - Tree Ensemble Model

```rust
/// A trained gradient-boosted decision tree model.
/// 
/// The model is automatically optimized for inference after training.
/// Training parameters are stored for serialization and reproducibility.
pub struct GBDTModel {
    /// Tree ensemble
    pub forest: Forest<ScalarLeaf>,
    /// Model metadata
    pub meta: ModelMeta,
    /// Training parameters (for reproducibility/serialization)
    pub params: GBDTParams,
    /// Optimized layout for fast inference (always populated after train/load)
    optimized_layout: UnrolledLayout,
}

impl GBDTModel {
    /// Train a GBDT model.
    /// 
    /// Automatically optimizes the model for inference after training.
    pub fn train(
        data: &ColMatrix,
        labels: &[f32],
        params: GBDTParams,
    ) -> Result<Self>;
    
    /// Predict on features.
    pub fn predict(&self, features: &impl RowMajorMatrix<f32>) -> Vec<f32>;
    
    /// Predict with output margin (no sigmoid/softmax).
    pub fn predict_margin(&self, features: &impl RowMajorMatrix<f32>) -> Vec<f32>;
    
    /// Compute feature importance.
    pub fn feature_importance(&self, importance_type: ImportanceType) -> Vec<f32>;
    
    /// Compute SHAP values for explanations (RFC-0020).
    pub fn shap_values(&self, features: &impl RowMajorMatrix<f32>) -> Result<ShapResult>;
    
    /// Number of trees in the ensemble.
    pub fn n_trees(&self) -> usize;
    
    /// Number of features.
    pub fn n_features(&self) -> usize;
    
    /// Save to file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()>;
    
    /// Load from file (automatically optimizes for inference).
    pub fn load(path: impl AsRef<Path>) -> Result<Self>;
    
    /// Serialize to bytes (for custom storage/transmission).
    pub fn to_bytes(&self) -> Result<Vec<u8>>;
    
    /// Deserialize from bytes (automatically optimizes for inference).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self>;
    
    /// Write to any output stream (streaming serialization, no intermediate buffer).
    pub fn write_to(&self, writer: &mut impl std::io::Write) -> Result<()>;
    
    /// Read from any input stream.
    pub fn read_from(reader: &mut impl std::io::Read) -> Result<Self>;
}
```

#### GBLinearModel - Linear Booster Model

```rust
/// A trained gradient-boosted linear model.
/// 
/// Training parameters are stored for serialization and reproducibility.
pub struct GBLinearModel {
    /// Learned weights: [n_features × n_groups]
    pub weights: Box<[f32]>,
    /// Bias terms: [n_groups]
    pub bias: Box<[f32]>,
    /// Model metadata
    pub meta: ModelMeta,
    /// Training parameters (for reproducibility/serialization)
    pub params: GBLinearParams,
}

impl GBLinearModel {
    /// Train a GBLinear model.
    pub fn train(
        data: &ColMatrix,
        labels: &[f32],
        params: GBLinearParams,
    ) -> Result<Self>;
    
    /// Predict on features.
    pub fn predict(&self, features: &impl RowMajorMatrix<f32>) -> Vec<f32>;
    
    /// Get raw weights as matrix.
    pub fn weights(&self) -> &[f32];
    
    /// Get bias terms.
    pub fn bias(&self) -> &[f32];
    
    /// Number of features.
    pub fn n_features(&self) -> usize;
    
    /// Compute feature importance (absolute weight values).
    pub fn feature_importance(&self) -> Vec<f32>;
    
    /// Compute SHAP values (trivial for linear: weight × feature value).
    pub fn shap_values(&self, features: &impl RowMajorMatrix<f32>) -> Result<ShapResult>;
    
    /// Save to file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()>;
    
    /// Load from file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self>;
    
    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>>;
    
    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self>;
}
```

#### Model Metadata

```rust
/// Model metadata shared between model types.
pub struct ModelMeta {
    /// Feature names (optional)
    pub feature_names: Option<Vec<String>>,
    /// Feature types for categorical handling
    pub feature_types: Option<Vec<FeatureType>>,
    /// Number of features
    pub n_features: usize,
    /// Number of output groups (1 for regression/binary, n_classes for multiclass)
    pub n_groups: usize,
    /// Task type inferred from objective
    pub task: TaskKind,
    /// Best iteration (if early stopping was used)
    pub best_iteration: Option<usize>,
}

/// Feature type for categorical handling.
pub enum FeatureType {
    Numeric,
    Categorical { n_categories: u32 },
}
```

#### Model Hierarchy - Separate Types with Shared Params

Rather than a single unified `Model` with all parameters mixed together (XGBoost's pain point), we use separate model types with clear parameter ownership:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Parameters                          │
├─────────────────────────────────────────────────────────────────┤
│  CommonParams        │  Model-Specific Params                   │
│  ─────────────────   │  ─────────────────────────────────────── │
│  • n_estimators      │  GBDTParams          │  GBLinearParams   │
│  • learning_rate     │  • max_depth         │  • reg_lambda     │
│  • objective         │  • max_leaves        │  • reg_alpha      │
│  • eval_metric       │  • subsample         │  • feature_sel    │
│  • early_stopping    │  • colsample_*       │                   │
│  • seed              │  • min_child_weight  │                   │
│  • n_threads         │  • reg_lambda/alpha  │                   │
│  • verbosity         │  • max_bin           │                   │
│                      │  • linear_tree       │                   │
│                      │  • sampling (GOSS)   │                   │
│                      │  • categorical       │                   │
└──────────────────────┴──────────────────────┴───────────────────┘
```

#### Objectives and Metrics

**Objective naming convention**: We use XGBoost-style prefixes for discoverability:

- `reg:*` - Regression objectives
- `binary:*` - Binary classification objectives  
- `multi:*` - Multiclass classification objectives
- `rank:*` - Learning to rank objectives (future)

**Simple vs Parameterized Objectives**:

- **Simple objectives** can be specified as strings: `objective="reg:squared_error"`
- **Parameterized objectives** require `ObjectiveConfig` with additional parameters:
  ```python
  # Quantile regression needs quantile values
  objective=ObjectiveConfig("reg:quantile", quantiles=[0.1, 0.5, 0.9])
  
  # Multiclass needs number of classes
  objective=ObjectiveConfig("multi:softmax", n_classes=5)
  
  # Tweedie needs variance power
  objective=ObjectiveConfig("reg:tweedie", tweedie_variance_power=1.5)
  ```

**Objectives Reference Table**:

| Objective | Task | Default Metric | Output Shape | Notes |
|-----------|------|----------------|--------------|-------|
| `reg:squared_error` | Regression | RMSE | `(n,)` | Default, simple |
| `reg:absolute_error` | Regression | MAE | `(n,)` | Simple |
| `reg:huber` | Regression | MAE | `(n,)` | Parameterized: `delta` |
| `reg:quantile` | Quantile Regression | Pinball | `(n, q)` | Parameterized: `quantiles` |
| `reg:tweedie` | Regression | Tweedie | `(n,)` | Parameterized: `tweedie_variance_power` |
| `binary:logistic` | Binary Classification | LogLoss | `(n,)` prob | Simple |
| `multi:softmax` | Multiclass | Accuracy | `(n, c)` probs | Parameterized: `n_classes` |
| `multi:one_vs_all` | Multiclass | LogLoss | `(n, c)` probs | Parameterized: `n_classes` |

**Metrics**:

| Metric | Task | Description |
|--------|------|-------------|
| `rmse` | Regression | Root mean squared error |
| `mae` | Regression | Mean absolute error |
| `mape` | Regression | Mean absolute percentage error |
| `logloss` | Classification | Negative log-likelihood |
| `auc` | Binary Classification | Area under ROC curve |
| `accuracy` | Classification | Classification accuracy |
| `ndcg` | Ranking | Normalized discounted cumulative gain |

```rust
/// Objective function for training.
#[derive(Clone, Debug, PartialEq)]
pub enum ObjectiveKind {
    // === Regression ===
    /// Squared error (L2 loss) - default for regression
    SquaredError,
    /// Absolute error (L1 loss)
    AbsoluteError,
    /// Huber loss - robust to outliers
    Huber { delta: f32 },
    /// Quantile regression - predict percentiles
    Quantile { quantiles: Vec<f32> },
    /// Tweedie regression - for insurance/count data
    Tweedie { variance_power: f32 },
    
    // === Binary Classification ===
    /// Logistic regression - outputs probability
    BinaryLogistic,
    /// Hinge loss (SVM-style)
    BinaryHinge,
    
    // === Multiclass Classification ===
    /// Softmax - outputs probability distribution
    MultiSoftmax { n_classes: u32 },
    /// One-vs-All - decomposed binary classifiers
    MultiOneVsAll { n_classes: u32 },
}

impl ObjectiveKind {
    /// Parse from string (XGBoost-compatible).
    pub fn from_str(s: &str) -> Result<Self, ParseError> {
        match s {
            "reg:squared_error" | "squared_error" | "mse" => Ok(Self::SquaredError),
            "reg:absolute_error" | "absolute_error" | "mae" => Ok(Self::AbsoluteError),
            "binary:logistic" | "logistic" => Ok(Self::BinaryLogistic),
            // Parameterized objectives require ObjectiveConfig
            _ => Err(ParseError::UnknownObjective(s.to_string())),
        }
    }
    
    /// Returns true if this is a classification objective.
    pub fn is_classification(&self) -> bool {
        matches!(self, 
            Self::BinaryLogistic | Self::BinaryHinge |
            Self::MultiSoftmax { .. } | Self::MultiOneVsAll { .. })
    }
    
    /// Returns true if this is a regression objective.
    pub fn is_regression(&self) -> bool {
        matches!(self,
            Self::SquaredError | Self::AbsoluteError | Self::Huber { .. } |
            Self::Quantile { .. } | Self::Tweedie { .. })
    }
}

/// Evaluation metric for monitoring and early stopping.
#[derive(Clone, Debug, PartialEq)]
pub enum MetricKind {
    // === Regression ===
    Rmse,
    Mae,
    Mape,
    
    // === Classification ===
    LogLoss,
    Auc,
    Accuracy,
    
    // === Ranking ===
    Ndcg { k: Option<u32> },
}

/// Common training parameters shared across all model types.
#[derive(Clone, Debug)]
pub struct CommonParams {
    /// Number of boosting rounds
    pub n_estimators: u32,
    /// Learning rate (shrinkage)
    pub learning_rate: f32,
    /// Objective function
    pub objective: ObjectiveKind,
    /// Evaluation metric (for early stopping). Auto-selected if None.
    pub eval_metric: Option<MetricKind>,
    /// Early stopping rounds (0 = disabled)
    pub early_stopping_rounds: u32,
    /// Random seed
    pub seed: u64,
    /// Number of threads (0 = auto)
    pub n_threads: usize,
    /// Verbosity level
    pub verbosity: Verbosity,
}

/// GBDT-specific training parameters.
#[derive(Clone, Debug)]
pub struct GBDTParams {
    /// Common training parameters
    pub common: CommonParams,
    
    // === Tree structure ===
    /// Maximum tree depth (for depth-wise growth)
    pub max_depth: u32,
    /// Maximum leaves per tree (for leaf-wise growth)
    pub max_leaves: Option<u32>,
    /// Growth strategy
    pub grow_policy: GrowPolicy,
    
    // === Regularization ===
    pub reg_lambda: f32,
    pub reg_alpha: f32,
    pub min_child_weight: f32,
    pub min_split_gain: f32,
    
    // === Sampling ===
    pub subsample: f32,
    pub colsample_bytree: f32,
    pub colsample_bylevel: f32,
    
    // === Histogram ===
    pub max_bin: u32,
    
    // === Advanced ===
    pub linear_tree: bool,
    pub goss: Option<GOSSParams>,
    pub categorical: CategoricalParams,
}

/// GBLinear-specific training parameters.
#[derive(Clone, Debug)]
pub struct GBLinearParams {
    /// Common training parameters
    pub common: CommonParams,
    
    /// L2 regularization
    pub reg_lambda: f32,
    /// L1 regularization  
    pub reg_alpha: f32,
    /// Feature selection strategy
    pub feature_selector: FeatureSelectorKind,
}

/// GOSS sampling parameters.
#[derive(Clone, Debug)]
pub struct GOSSParams {
    pub top_rate: f32,
    pub other_rate: f32,
}

/// Categorical feature handling parameters.
#[derive(Clone, Debug)]
pub struct CategoricalParams {
    /// Maximum categories for one-hot encoding
    pub max_onehot: u32,
}
```

#### Separate Model Types

Instead of one unified `Model` enum, we expose **separate model types** that are clear about their capabilities:

```rust
/// Gradient-boosted decision tree model.
pub struct GBDTModel {
    forest: Forest<ScalarLeaf>,
    meta: ModelMeta,
    /// Optional cached layout for fast inference
    cached_layout: Option<UnrolledLayout>,
}

/// Gradient-boosted linear model.
pub struct GBLinearModel {
    model: LinearModel,
    meta: ModelMeta,
}

/// Model metadata (shared).
pub struct ModelMeta {
    pub feature_names: Option<Vec<String>>,
    pub feature_types: Option<Vec<FeatureType>>,
    pub n_features: usize,
    pub n_outputs: usize,
    pub best_iteration: Option<usize>,
    /// Feature statistics for SHAP (means, stds)
    pub feature_stats: Option<FeatureStats>,
}
```

#### Trainers

```rust
/// GBDT Trainer (wraps existing GBDTTrainer with params conversion).
impl GBDTModel {
    pub fn train(
        data: &Dataset,
        params: GBDTParams,
    ) -> Result<Self, TrainError> {
        // Delegates to internal GBDTTrainer
    }
}

/// GBLinear Trainer.
impl GBLinearModel {
    pub fn train(
        data: &Dataset,
        params: GBLinearParams,
    ) -> Result<Self, TrainError> {
        // Delegates to internal GBLinearTrainer
    }
}
```

#### Benefits of Separate Types

| Aspect | Unified Model | Separate Types (Chosen) |
|--------|---------------|-------------------------|
| Parameter clarity | ❌ Mixed params, unclear which apply | ✅ Each model has own params |
| Type safety | ⚠️ Runtime checks needed | ✅ Compile-time guarantees |
| API discovery | ❌ All methods visible, some error | ✅ Only relevant methods |
| Python bindings | Simpler (one class) | Slightly more (2-3 classes) |
| XGBoost compat | More similar | Less similar (intentional) |

impl Default for ModelParams {
    fn default() -> Self {
        Self {
            booster: BoosterType::GBTree,
            n_estimators: 100,
            learning_rate: 0.3,
            objective: ObjectiveKind::SquaredError,
            eval_metric: None,
            early_stopping_rounds: 0,
            seed: 42,
            n_threads: 0,
            verbosity: Verbosity::Warning,
            max_depth: Some(6),
            max_leaves: None,
            grow_policy: GrowPolicy::DepthWise,
            reg_lambda: 1.0,
            reg_alpha: 0.0,
            min_child_weight: 1.0,
            min_split_gain: 0.0,
            subsample: 1.0,
            colsample_bytree: 1.0,
            colsample_bylevel: 1.0,
            max_bin: 256,
            linear_tree: false,
            top_rate: None,
            other_rate: None,
            max_cat_threshold: 32,
        }
    }
}
```

### Model API

#### Training

```rust
impl Model {
    /// Train a model from feature matrix and targets.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix (row-major or column-major)
    /// * `targets` - Target values
    /// * `params` - Training parameters
    ///
    /// # Returns
    ///
    /// Trained model, or error if training fails.
    pub fn train<M: DataMatrix<Element = f32>>(
        features: &M,
        targets: &[f32],
        params: ModelParams,
    ) -> Result<Self, TrainError> {
        Self::train_with_options(features, targets, &[], params, TrainOptions::default())
    }

    /// Train with additional options (sample weights, eval sets, callbacks).
    pub fn train_with_options<M: DataMatrix<Element = f32>>(
        features: &M,
        targets: &[f32],
        sample_weights: &[f32],
        params: ModelParams,
        options: TrainOptions,
    ) -> Result<Self, TrainError> {
        // 1. Convert features to appropriate format
        // 2. Dispatch to GBDTTrainer or GBLinearTrainer based on params.booster
        // 3. Wrap result in Model
        /* ... */
    }

    /// Continue training from an existing model (incremental learning).
    pub fn update<M: DataMatrix<Element = f32>>(
        &mut self,
        features: &M,
        targets: &[f32],
        n_additional_rounds: u32,
    ) -> Result<(), TrainError> {
        /* ... */
    }
}

/// Additional training options.
pub struct TrainOptions {
    /// Validation sets for early stopping
    pub eval_sets: Vec<EvalSetConfig>,
    /// Feature names
    pub feature_names: Option<Vec<String>>,
    /// Categorical feature indices
    pub categorical_features: Option<Vec<usize>>,
    /// Per-feature bin limits
    pub feature_bins: Option<Vec<u32>>,
}
```

#### Prediction

```rust
impl Model {
    /// Predict for a single row.
    pub fn predict_row(&self, features: &[f32]) -> Vec<f32> {
        match &self.core {
            ModelCore::GBDT { forest, .. } => forest.predict_row(features),
            ModelCore::GBLinear { model } => model.predict_row(features),
        }
    }

    /// Predict for multiple rows.
    ///
    /// Returns a flat array in row-major order: `[row0_out0, row0_out1, ..., row1_out0, ...]`
    pub fn predict<M: DataMatrix<Element = f32>>(&self, features: &M) -> Vec<f32> {
        match &self.core {
            ModelCore::GBDT { forest, cached_layout } => {
                // Use cached optimized layout if available
                if let Some(layout) = cached_layout {
                    let predictor = UnrolledPredictor6::from_layout(layout);
                    predictor.par_predict(features)
                } else {
                    // Fallback to simple prediction
                    forest.predict_into(features, &mut output);
                    output
                }
            }
            ModelCore::GBLinear { model } => {
                model.predict_batch(features)
            }
        }
    }

    /// Predict with output transform (e.g., sigmoid for probability).
    pub fn predict_proba<M: DataMatrix<Element = f32>>(&self, features: &M) -> Vec<f32> {
        let raw = self.predict(features);
        self.apply_transform(&raw)
    }

    /// Get raw margin predictions (before sigmoid/softmax).
    pub fn predict_margin<M: DataMatrix<Element = f32>>(&self, features: &M) -> Vec<f32> {
        self.predict(features)
    }

    /// Predict leaf indices for tree models.
    ///
    /// Returns `None` for linear models.
    pub fn predict_leaf<M: DataMatrix<Element = f32>>(&self, features: &M) -> Option<Vec<u32>> {
        match &self.core {
            ModelCore::GBDT { forest, .. } => Some(forest.predict_leaf_indices(features)),
            ModelCore::GBLinear { .. } => None,
        }
    }

    /// Optimize model for fast inference.
    ///
    /// Pre-computes optimized layouts for tree traversal.
    /// Call this before batch prediction for best performance.
    pub fn optimize_for_inference(&mut self) {
        if let ModelCore::GBDT { forest, cached_layout } = &mut self.core {
            if cached_layout.is_none() {
                *cached_layout = Some(UnrolledLayout::from_forest(forest));
            }
        }
    }
}
```

#### Serialization

```rust
impl Model {
    /// Save model to file.
    ///
    /// Format is auto-detected from extension:
    /// - `.json` - JSON format (human-readable, XGBoost-compatible)
    /// - `.bin` - Binary format (fast, compact)
    /// - `.txt` - LightGBM text format
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), SaveError> {
        /* ... */
    }

    /// Load model from file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, LoadError> {
        /* ... */
    }

    /// Load from XGBoost JSON format.
    pub fn from_xgboost<R: Read>(reader: R) -> Result<Self, LoadError> {
        let xgb = XgbModel::from_reader(reader)?;
        Self::from_xgb_model(xgb)
    }

    /// Load from LightGBM text format.
    pub fn from_lightgbm<R: Read>(reader: R) -> Result<Self, LoadError> {
        let lgb = LgbModel::from_reader(reader)?;
        Self::from_lgb_model(lgb)
    }

    /// Export to JSON bytes.
    pub fn to_json(&self) -> Result<Vec<u8>, SerializeError> {
        /* ... */
    }

    /// Export to binary bytes.
    pub fn to_binary(&self) -> Result<Vec<u8>, SerializeError> {
        /* ... */
    }
}
```

### Python Bindings

#### Design Philosophy

1. **Zero-copy by default** - DMatrix wraps NumPy/Pandas without copying
2. **NumPy/Pandas only** - No Python lists (inefficient, type-unsafe)
3. **Separate model classes** - `GBDTBooster`, `GBLinearBooster` mirror Rust
4. **Sklearn wrappers** - `GBDTRegressor`, `GBDTClassifier` for familiar API

#### Module Structure

```
boosters/
├── __init__.py          # Main exports
├── _core.pyi            # Type stubs
├── _core.so             # Native extension (PyO3)
└── sklearn.py           # Sklearn-compatible wrappers
```

#### Dataset - Zero-Copy Data Wrapper

The `Dataset` class is a **thin zero-copy wrapper** around NumPy arrays or Pandas DataFrames. It does NOT convert or copy data.

```python
class Dataset:
    """Zero-copy wrapper for training/prediction data.
    
    Wraps NumPy arrays or Pandas DataFrames without copying.
    The underlying data must remain valid for the lifetime of this object.
    
    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Feature matrix. Must be float32 or will be converted (single copy).
    label : np.ndarray or pd.Series, optional
        Target values. Shape validated against objective:
        - Scalar objectives: (n_samples,)
        - Multi-output (e.g., quantile): (n_samples, n_targets)
    weight : np.ndarray or pd.Series, optional  
        Sample weights. Must be shape (n_samples,).
    feature_names : list[str], optional
        Feature names (inferred from DataFrame columns if not provided).
    
    Raises
    ------
    TypeError
        If data is a Python list (not supported - use np.array()).
    ValueError
        If array shapes are inconsistent (e.g., label length != n_rows).
    """
    
    def __init__(
        self,
        data: np.ndarray | pd.DataFrame,
        label: np.ndarray | pd.Series | None = None,
        weight: np.ndarray | pd.Series | None = None,
        feature_names: list[str] | None = None,
    ): ...
    
    @property
    def n_rows(self) -> int: ...
    
    @property  
    def n_cols(self) -> int: ...
    
    @property
    def feature_names(self) -> list[str] | None: ...
    
    def validate_for_training(self, params: CommonParams) -> None:
        """Validate dataset is suitable for training with given params.
        
        Checks:
        - Label is present
        - Label shape matches objective requirements
        - Weight shape matches n_rows (if provided)
        - No NaN in label (unless objective handles it)
        
        Raises
        ------
        ValueError
            If validation fails with descriptive message.
        """
        ...
```

**Key design choices:**

- No `from_pandas()` or similar - constructor handles both
- No Python list support - raises `TypeError` with helpful message
- Accepts `pd.Series` for label/weight (common pandas pattern)
- dtype conversion happens once if needed (float32 required)
- Keeps reference to original array (prevents GC)
- `validate_for_training()` checks shape constraints for multi-output objectives

#### GBDTBooster - Tree Model

```python
class GBDTBooster:
    """Gradient-boosted decision tree model.
    
    For sklearn-style API, use GBDTRegressor or GBDTClassifier instead.
    """
    
    @staticmethod
    def train(
        params: GBDTParams,
        train_data: Dataset,
        eval_data: list[tuple[Dataset, str]] | None = None,
    ) -> "GBDTBooster": ...
    
    def predict(
        self, 
        data: Dataset | np.ndarray,
        output_margin: bool = False,
    ) -> np.ndarray: ...
    
    def predict_leaf(self, data: Dataset | np.ndarray) -> np.ndarray: ...
    
    def feature_importance(self, importance_type: str = "gain") -> dict[str, float]: ...
    
    def shap_values(self, data: Dataset | np.ndarray) -> np.ndarray: ...
    
    @property
    def expected_value(self) -> float | np.ndarray:
        """Base value for SHAP explanations.
        
        For single-output: returns float (mean prediction over training data).
        For multi-output: returns array of shape (n_outputs,).
        """
        ...
    
    def save(self, path: str | Path, format: str = "json") -> None:
        """Save model to file."""
        ...
    
    @staticmethod
    def load(path: str | Path) -> "GBDTBooster":
        """Load model from file."""
        ...
    
    def to_bytes(self, format: str = "json") -> bytes:
        """Serialize model to bytes (for pickling, custom storage)."""
        ...
    
    @staticmethod
    def from_bytes(data: bytes) -> "GBDTBooster":
        """Deserialize model from bytes."""
        ...
    
    def __getstate__(self) -> bytes:
        """Pickle support - returns serialized bytes."""
        return self.to_bytes()
    
    def __setstate__(self, state: bytes) -> None:
        """Pickle support - restores from bytes."""
        ...
    
    @property
    def n_trees(self) -> int: ...
    
    @property
    def n_features(self) -> int: ...
    
    @property
    def best_iteration(self) -> int | None: ...
    
    @property
    def params(self) -> GBDTParams:
        """Training parameters (for reproducibility)."""
        ...
```

#### GBLinearBooster - Linear Model

```python
class GBLinearBooster:
    """Gradient-boosted linear model."""
    
    @staticmethod
    def train(
        params: GBLinearParams,
        train_data: Dataset,
        eval_data: list[tuple[Dataset, str]] | None = None,
    ) -> "GBLinearBooster": ...
    
    def predict(self, data: Dataset | np.ndarray) -> np.ndarray: ...
    
    def feature_importance(self) -> dict[str, float]: ...
    
    def shap_values(self, data: Dataset | np.ndarray) -> np.ndarray: ...
    
    def save(self, path: str | Path) -> None:
        """Save model to file."""
        ...
    
    @staticmethod
    def load(path: str | Path) -> "GBLinearBooster":
        """Load model from file."""
        ...
    
    def to_bytes(self) -> bytes:
        """Serialize model to bytes."""
        ...
    
    @staticmethod
    def from_bytes(data: bytes) -> "GBLinearBooster":
        """Deserialize model from bytes."""
        ...
    
    def __getstate__(self) -> bytes:
        """Pickle support."""
        return self.to_bytes()
    
    def __setstate__(self, state: bytes) -> None:
        """Pickle support."""
        ...
    
    @property
    def weights(self) -> np.ndarray: ...
    
    @property
    def bias(self) -> float | np.ndarray: ...
    
    @property
    def n_features(self) -> int: ...
    
    @property
    def params(self) -> GBLinearParams:
        """Training parameters (for reproducibility)."""
        ...
```

#### Parameter Classes

```python
@dataclass
class CommonParams:
    """Common training parameters."""
    n_estimators: int = 100
    learning_rate: float = 0.3
    objective: str | ObjectiveConfig = "squared_error"
    eval_metric: str | None = None
    early_stopping_rounds: int = 0
    base_score: float | None = None  # Auto-computed if None
    seed: int = 42
    n_threads: int = 0  # 0 = auto
    verbosity: int = 1
    callbacks: list[Callback] | None = None  # Custom callbacks

@dataclass
class ObjectiveConfig:
    """Parameterized objective function.
    
    Required fields by objective:
    - "quantile": requires `quantiles` (list of floats in (0,1))
    - "tweedie": requires `tweedie_variance_power` (float in (1,2))
    - "huber": optional `delta` (default 1.0)
    - "multi:softmax": requires `n_classes` >= 2
    
    Examples:
        ObjectiveConfig("quantile", quantiles=[0.1, 0.5, 0.9])
        ObjectiveConfig("tweedie", tweedie_variance_power=1.5)
        ObjectiveConfig("multi:softmax", n_classes=5)
    """
    name: str
    quantiles: list[float] | None = None
    tweedie_variance_power: float | None = None
    delta: float | None = None
    n_classes: int | None = None

@dataclass  
class GBDTParams:
    """GBDT-specific parameters."""
    common: CommonParams = field(default_factory=CommonParams)
    max_depth: int = 6
    max_leaves: int | None = None
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    min_child_weight: float = 1.0
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    max_bin: int = 256
    linear_tree: bool = False
    # Multi-class: "one_vs_all" or "softmax" (default: "softmax")
    multi_class_strategy: str = "softmax"
    
    @classmethod
    def quick(cls, **kwargs) -> "GBDTParams":
        """Quick constructor with flat kwargs.
        
        Common params (learning_rate, n_estimators, etc.) are
        automatically routed to CommonParams.
        
        Example:
            GBDTParams.quick(learning_rate=0.1, max_depth=8)
        """
        ...
    
    @classmethod
    def for_regression(cls, **kwargs) -> "GBDTParams":
        """Preset for regression tasks.
        
        Sets objective="reg:squared_error", eval_metric="rmse".
        """
        ...
    
    @classmethod
    def for_classification(cls, n_classes: int = 2, **kwargs) -> "GBDTParams":
        """Preset for classification tasks.
        
        Parameters
        ----------
        n_classes : int, default=2
            Number of classes. 
            - n_classes=2: Uses "binary:logistic" (single probability output)
            - n_classes>2: Uses "multi:softmax" (probability per class)
        
        Note: For n_classes=2, the output is a single probability P(y=1).
        For n_classes>2, the output is (n_samples, n_classes) probabilities.
        """
        ...

@dataclass
class GBLinearParams:
    """GBLinear-specific parameters."""
    common: CommonParams = field(default_factory=CommonParams)
    reg_lambda: float = 0.0
    reg_alpha: float = 0.0
    feature_selector: str = "cyclic"
    
    def __init__(self, **kwargs):
        """Allow flat kwargs for convenience."""
        ...


class Callback(Protocol):
    """Training callback interface."""
    
    def on_iteration_end(
        self, 
        iteration: int, 
        train_loss: float,
        eval_results: dict[str, float] | None,
    ) -> bool:
        """Called after each boosting round.
        
        Return False to stop training early.
        """
        ...


class PrintMetricsCallback:
    """Example callback that prints metrics each round."""
    
    def __init__(self, period: int = 10):
        self.period = period
    
    def on_iteration_end(self, iteration, train_loss, eval_results):
        if iteration % self.period == 0:
            print(f"[{iteration}] train_loss={train_loss:.4f}")
            if eval_results:
                for name, value in eval_results.items():
                    print(f"       {name}={value:.4f}")
        return True  # Continue training
```

#### Sklearn-Compatible Wrappers

The sklearn wrappers inherit from `BaseEstimator` which provides:

- **`get_params()` / `set_params()`** - Required for `GridSearchCV`, `RandomizedSearchCV`
- **`clone()`** - Required for `cross_val_score`, pipelines
- **Pickling** - Via `__reduce__` (inherits from BaseEstimator)

All wrappers follow sklearn conventions:
- **Classifiers**: `predict()` returns class labels, `predict_proba()` returns probabilities
- **Regressors**: `predict()` returns continuous values
- **After fit**: `feature_importances_`, `n_features_in_`, `feature_names_in_` available

```python
# boosters/sklearn.py

class GBDTRegressor(RegressorMixin, BaseEstimator):
    """Sklearn-compatible GBDT regressor.
    
    Parameters match sklearn conventions (flat, not nested).
    Inherits get_params(), set_params(), clone() from BaseEstimator.
    
    Objective is automatically set to "reg:squared_error".
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.3,
        max_depth: int = 6,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        n_jobs: int = -1,
        random_state: int | None = None,
        early_stopping_rounds: int = 0,
    ):
        # Store all params as instance attributes for sklearn's get_params()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        # ... all params stored identically
    
    def fit(
        self, 
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        eval_set: list[tuple] | None = None,
    ) -> "GBDTRegressor":
        """Fit the model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        sample_weight : array-like of shape (n_samples,), optional
        eval_set : list of (X, y) tuples, optional
            Validation sets for early stopping.
        """
        # Store feature names if DataFrame
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
        self.n_features_in_ = X.shape[1]
        
        # Convert to Dataset, build GBDTParams, train
        ...
        return self
    
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict regression target for X."""
        ...
    
    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances (gain-based)."""
        ...
    
    @property
    def n_features_in_(self) -> int:
        """Number of features seen during fit."""
        ...
    
    @property
    def feature_names_in_(self) -> np.ndarray | None:
        """Feature names seen during fit (if X was DataFrame)."""
        ...


class GBDTClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible GBDT classifier.
    
    For binary classification, objective is "binary:logistic".
    For multiclass, objective is "multi:softmax" with n_classes auto-detected from y.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.3,
        max_depth: int = 6,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        n_jobs: int = -1,
        random_state: int | None = None,
        early_stopping_rounds: int = 0,
    ): ...
    
    def fit(self, X, y, sample_weight=None, eval_set=None):
        """Fit the classifier.
        
        Automatically detects n_classes from unique values in y.
        """
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        # Sets objective based on n_classes
        ...
    
    def predict(self, X) -> np.ndarray:
        """Predict class labels for X.
        
        Returns array of shape (n_samples,) with class labels.
        """
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
    
    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for X.
        
        Returns array of shape (n_samples, n_classes).
        """
        ...
    
    @property
    def classes_(self) -> np.ndarray:
        """Class labels known to the classifier."""
        ...


class GBLinearRegressor(RegressorMixin, BaseEstimator):
    """Sklearn-compatible linear booster regressor.
    
    Uses gradient-boosted linear model instead of trees.
    Useful for high-dimensional sparse data or when linear model is appropriate.
    """
    ...


class GBLinearClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible linear booster classifier."""
    ...
```

#### PyO3 Module Definition

```rust
/// Python module initialization.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core classes
    m.add_class::<PyDataset>()?;
    m.add_class::<PyGBDTBooster>()?;
    m.add_class::<PyGBLinearBooster>()?;
    
    // Parameter dataclasses
    m.add_class::<PyCommonParams>()?;
    m.add_class::<PyGBDTParams>()?;
    m.add_class::<PyGBLinearParams>()?;
    
    // Version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}
```

## Design Decisions

### DD-1: Separate Model Types (Not Unified Enum)

**Context**: Should we have one `Model` type or separate `GBDTModel` and `GBLinearModel`?

**Options considered**:

1. **Unified enum** - Single `Model` with runtime dispatch
2. **Separate types** - `GBDTModel` and `GBLinearModel` as distinct types

**Decision**: **Separate types** because:

- No runtime dispatch overhead
- Type-safe APIs (e.g., `GBDTBooster.num_trees` exists, but doesn't on Linear)
- Clear parameter separation (tree params vs linear params)
- Better IDE autocompletion and type checking
- Users explicitly choose their model type

**Consequences**: 

- Two Python classes instead of one
- Loading models requires knowing the type (or a `load_any()` helper)
- Slightly more surface area in Python bindings

### DD-2: Zero-Copy NumPy Strategy

**Context**: How to avoid copying data between Python and Rust?

**Options considered**:

1. **Always copy** - Simple but slow for large data
2. **PyO3 views** - Zero-copy but lifetime complexity
3. **Hybrid** - Zero-copy when possible, copy when needed

**Decision**: Hybrid approach:

- Use `PyReadonlyArray` for read-only access (prediction)
- Copy on write when data needs mutation (training binning)
- Document when copies occur

**Consequences**: Best performance for inference, acceptable for training.

### DD-3: XGBoost-Compatible Parameter Names

**Context**: Should parameter names match XGBoost or use our own?

**Options considered**:

1. **XGBoost names** - `n_estimators`, `max_depth`, `learning_rate`
2. **Custom names** - `num_trees`, `max_tree_depth`, `eta`

**Decision**: Use XGBoost/sklearn names because:

- Lower migration friction
- Users already familiar
- IDE autocomplete matches expectations

**Consequences**: Some names are legacy (e.g., `eta` as alias for `learning_rate`).

### DD-4: Feature Names Storage

**Context**: Where to store feature names?

**Options considered**:

1. **In Model** - Names stored with trained model
2. **In DMatrix only** - Names only at data creation
3. **Both** - Copy to model during training

**Decision**: Both - copy to model during training:

- Enables feature importance by name
- Works with loaded models (names in serialization)
- Dataset can validate consistency

### DD-5: Hierarchical vs Flat Parameters (REVISED)

**Context**: XGBoost uses flat parameter dictionaries which become a pain point for IDE support and documentation.

**Options considered**:

1. **Flat dict** - XGBoost-style `{"max_depth": 6, "eta": 0.1}`
2. **Hierarchical dataclasses** - `GBDTParams(common=CommonParams(...), max_depth=6)`

**Decision**: Hierarchical dataclasses because:

- Type-safe parameters with IDE autocompletion
- Model-specific params don't pollute other models
- CommonParams shared cleanly between GBDT and GBLinear
- Sklearn wrappers flatten back for sklearn compatibility

**Consequences**: Slightly more verbose for quick scripts, but much better DX overall.

### DD-6: Automatic Inference Optimization

**Context**: Should users call `optimize_for_inference()` manually?

**Options considered**:

1. **Manual opt-in** - Users call method when ready
2. **Automatic after train/load** - Always optimized

**Decision**: Automatic optimization because:

- Users always want fast inference
- No valid use case for unoptimized model
- Removes footgun of forgetting to optimize
- `load()` also auto-optimizes for consistency

**Consequences**: Slight extra time after train/load (negligible vs training time).

### DD-7: Storing Params in Model

**Context**: Should trained models store their training parameters?

**Decision**: Yes, store `params` in model because:

- Enables reproducibility (re-train with same params)
- Serialization includes full config
- Users can inspect what params were used
- Makes model self-documenting

**Consequences**: Slightly larger serialized models (negligible).

### DD-8: Input Validation Strategy

**Context**: How to validate inputs early and provide good error messages?

**Decision**: Multi-layer validation:

1. **Dataset construction** - Shape consistency (label.len == n_rows)
2. **Dataset.validate_for_training(params)** - Objective-specific checks
3. **Train-time** - Final validation before training starts

Key validations:

- Multi-output objectives (quantile, multi-class) check label dimensions
- Pinball loss validates n_quantiles matches label columns
- Weight must be 1D with correct length
- Feature count consistency across train/eval sets

## Integration

| Component | Integration Point | Notes |
| --------- | ----------------- | ----- |
| RFC-0001 (DataMatrix) | `Dataset::as_matrix()` | Zero-copy view |
| RFC-0002 (Forest) | `GBDTModel.forest` | Direct storage |
| RFC-0008 (Objectives) | `ObjectiveKind` | Validation uses this |
| RFC-0014 (GBLinear) | `GBLinearModel` | Separate type |
| RFC-0016 (Prediction) | Auto-optimize on train/load | Always fast |
| RFC-0020 (Explainability) | `feature_importance()`, `shap_values()` | Future |

## Open Questions

1. **Arrow support**: Should we support Arrow tables directly via `arrow-rs`?
   - Pro: Zero-copy from Pandas with Arrow backing
   - Con: Additional dependency
   - **Tentative**: Yes, behind `io-arrow` feature

2. **Model loading without type knowledge**: User has a saved model but doesn't know if it's GBDT or GBLinear.
   - Option A: Peek at file header, return appropriate type
   - Option B: `load_any()` → `Union[GBDTBooster, GBLinearBooster]`
   - **Tentative**: Option B with type guard helpers

3. **Multi-GPU**: Future consideration for GPU backend
   - Current design doesn't preclude it
   - Models could have device placement info

4. **Serialization format versioning**: How to handle model files across library versions?
   - Format includes magic number + version in header
   - Backward compatibility: newer library reads older formats
   - Forward compatibility: older library fails gracefully with "unsupported version" error
   - **Tentative**: Support backward compat for last 3 minor versions

## Usage Examples

This section shows real-world usage patterns for common tasks.

### Example 1: Quick Regression

```python
import numpy as np
from boosters.sklearn import GBDTRegressor

# Minimal code path
model = GBDTRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Example 2: Binary Classification with Early Stopping

```python
from boosters import GBDTBooster, GBDTParams, Dataset

# Using low-level API
params = GBDTParams.for_classification(n_classes=2)
params.common.early_stopping_rounds = 50
params.common.n_estimators = 1000

train_ds = Dataset(X_train, label=y_train)
valid_ds = Dataset(X_valid, label=y_valid)

model = GBDTBooster.train(
    params=params,
    train_data=train_ds,
    eval_data=[(valid_ds, "validation")],
)

print(f"Best iteration: {model.best_iteration}")

# Predict probabilities
probs = model.predict(X_test)  # Returns probabilities for binary

# Predict class labels (threshold at 0.5)
labels = (probs > 0.5).astype(int)
```

### Example 3: Multiclass Classification

```python
from boosters import GBDTParams, ObjectiveConfig

# Option A: Using preset factory
params = GBDTParams.for_classification(n_classes=5)

# Option B: Explicit configuration
params = GBDTParams.quick(
    objective=ObjectiveConfig("multi:softmax", n_classes=5),
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
)

# Using sklearn wrapper
from boosters.sklearn import GBDTClassifier

clf = GBDTClassifier(objective="multi:softmax", n_classes=5)
clf.fit(X_train, y_train)

# predict() returns class labels (0, 1, 2, 3, 4)
labels = clf.predict(X_test)

# predict_proba() returns probability distribution (n_samples, n_classes)
probs = clf.predict_proba(X_test)
```

### Example 4: Quantile Regression for Prediction Intervals

```python
from boosters import GBDTParams, ObjectiveConfig

# Predict 10th, 50th, and 90th percentiles
params = GBDTParams.quick(
    objective=ObjectiveConfig("reg:quantile", quantiles=[0.1, 0.5, 0.9]),
    n_estimators=100,
)

model = GBDTBooster.train(params, train_data)

# Output shape: (n_samples, 3) for 3 quantiles
preds = model.predict(X_test)
lower_10 = preds[:, 0]
median = preds[:, 1]
upper_90 = preds[:, 2]
```

### Example 5: Sklearn Pipeline with Cross-Validation

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from boosters.sklearn import GBDTClassifier

# Build pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GBDTClassifier(n_estimators=100))
])

# Cross-validation works out of the box
scores = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')
print(f"CV AUC: {scores.mean():.3f} ± {scores.std():.3f}")

# GridSearchCV for hyperparameter tuning
param_grid = {
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.1, 0.3],
    'model__n_estimators': [50, 100, 200],
}

grid = GridSearchCV(pipe, param_grid, cv=3, scoring='roc_auc')
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.3f}")
```

### Example 6: Hyperparameter Tuning with Optuna

```python
import optuna
from sklearn.model_selection import cross_val_score
from boosters.sklearn import GBDTRegressor

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
    }
    
    model = GBDTRegressor(**params)
    scores = cross_val_score(model, X, y, cv=3, scoring='neg_root_mean_squared_error')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best RMSE: {-study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

### Example 7: Rust API Usage

```rust
use boosters::{GBDTModel, GBDTParams, CommonParams, ObjectiveKind, ColMatrix};

// === Regression (default) ===
let params = GBDTParams::default();  // Uses SquaredError objective
let model = GBDTModel::train(&data, &labels, params)?;
let predictions = model.predict(&test_data);

// === Binary Classification ===
let params = GBDTParams {
    common: CommonParams {
        objective: ObjectiveKind::BinaryLogistic,
        n_estimators: 100,
        learning_rate: 0.1,
        ..Default::default()
    },
    max_depth: 6,
    ..Default::default()
};
let model = GBDTModel::train(&data, &labels, params)?;
let probabilities = model.predict(&test_data);

// === Multiclass Classification ===
let params = GBDTParams {
    common: CommonParams {
        objective: ObjectiveKind::MultiSoftmax { n_classes: 5 },
        ..Default::default()
    },
    ..Default::default()
};
let model = GBDTModel::train(&data, &labels, params)?;
// Output shape: n_samples * n_classes (flattened)
let probs = model.predict(&test_data);
```

### Example 8: Feature Importance and SHAP

```python
from boosters import GBDTBooster, GBDTParams, Dataset

model = GBDTBooster.train(params, train_data)

# Feature importance (returns dict: feature_name -> importance)
importance = model.feature_importance(importance_type="gain")
for name, score in sorted(importance.items(), key=lambda x: -x[1])[:10]:
    print(f"{name}: {score:.4f}")

# SHAP values for local explanations
shap_values = model.shap_values(X_test[:100])  # Shape: (100, n_features)

# Works with shap library for plotting
import shap
shap.summary_plot(shap_values, X_test[:100])
```

### Prediction Behavior Summary

| Class | `predict()` Returns | `predict_proba()` Returns |
|-------|---------------------|---------------------------|
| `GBDTBooster` | Raw output (prob for logistic, margin if `output_margin=True`) | N/A |
| `GBDTClassifier` | Class labels (0, 1, 2, ...) | Probabilities `(n, n_classes)` |
| `GBDTRegressor` | Predicted values | N/A |
| `GBLinearBooster` | Raw output | N/A |
| `GBLinearClassifier` | Class labels | Probabilities |
| `GBLinearRegressor` | Predicted values | N/A |

## Future Work

- [ ] ONNX export support
- [ ] GPU inference backend
- [ ] Model compression (pruning, quantization)
- [ ] Custom objective functions from Python
- [ ] Distributed training (future major feature)

## References

- [PyO3 User Guide](https://pyo3.rs/)
- [XGBoost Python Package](https://xgboost.readthedocs.io/en/latest/python/index.html)
- [LightGBM Python API](https://lightgbm.readthedocs.io/en/latest/Python-API.html)
- [numpy-rust interop](https://docs.rs/numpy/latest/numpy/)

## Changelog

- 2025-12-19: Round 5 review updates:
  - Added `ObjectiveKind` and `MetricKind` enum definitions
  - Added comprehensive objectives reference table with task/metric/output mapping
  - Added XGBoost-style objective prefixes (`reg:*`, `binary:*`, `multi:*`)
  - Added Usage Examples section with 8 real-world patterns
  - Clarified predict() vs predict_proba() behavior for all classes
  - Expanded sklearn wrapper documentation (get_params, set_params, clone)
  - Added Rust API usage examples for regression/classification
- 2025-12-19: Round 4 review updates:
  - Added preset factories (`GBDTParams.for_regression()`, `for_classification()`)
  - Added `PrintMetricsCallback` example
  - Added serialization format versioning to Open Questions
- 2025-12-19: Round 3 review updates:
  - Added `multi_class_strategy` to GBDTParams
  - Added `GBDTParams.quick()` classmethod for flat kwargs
  - Added `feature_names_in_` and `n_features_in_` to models
- 2025-12-19: Round 2 review updates:
  - Added `ObjectiveConfig` for parameterized objectives
  - Added `Callback` protocol for training customization
  - Added `callbacks` list to CommonParams
- 2025-12-19: Round 1 review updates:
  - Added `base_score` to CommonParams
  - Added `early_stopping_eval_set` to train()
- 2025-12-19: Pre-review updates:
  - Added `to_bytes()`/`from_bytes()` for in-memory serialization
  - Added `write_to()`/`read_from()` for streaming
  - Added `__reduce_ex__` for pickle support
- 2025-12-19: Second revision based on feedback:
  - DD-6 added: Automatic inference optimization (no manual call needed)
  - DD-7 added: Params stored in model for reproducibility/serialization
  - DD-8 added: Input validation strategy
  - Naming convention: `n_trees`, `n_features`, `n_rows`, `n_cols` (not `num_`)
  - Dataset accepts `pd.Series` for label/weight
  - Added `validate_for_training()` for objective-specific checks
- 2025-12-19: Major revision based on feedback:
  - DD-1 reversed: Separate model types (`GBDTModel`, `GBLinearModel`) instead of unified enum
  - DD-5 added: Hierarchical params (`CommonParams` + model-specific)
  - Renamed `DMatrix` → `Dataset` for clarity
  - Removed Python list support (NumPy/Pandas only)
  - Added sklearn-compatible wrappers (`GBDTRegressor`, `GBDTClassifier`, etc.)
  - Simplified Python code examples (less implementation detail)
- 2025-12-19: Initial draft
