//! Configuration types for Python bindings.
//!
//! Flat config structure matching core Rust configs.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::error::BoostersError;
use crate::metrics::PyMetric;
use crate::objectives::PyObjective;

// =============================================================================
// Growth Strategy
// =============================================================================

/// Tree growth strategy for building decision trees.
///
/// Determines the order in which nodes are expanded during tree construction.
/// Acts like a Python StrEnum - can be compared to strings.
///
/// Attributes:
///     Depthwise: Grow tree level-by-level (like XGBoost).
///         All nodes at depth d are expanded before any node at depth d+1.
///         More balanced trees, better for shallow trees.
///     Leafwise: Grow tree by best-first split (like LightGBM).
///         Always expands the leaf with highest gain.
///         Can produce deeper, more accurate trees but risks overfitting.
///
/// Examples:
///     >>> from boosters import GrowthStrategy
///     >>> strategy = GrowthStrategy.Depthwise
///     >>> strategy == "depthwise"
///     True
///     >>> str(strategy)
///     'depthwise'
#[pyo3_stub_gen::derive::gen_stub_pyclass_enum]
#[pyclass(name = "GrowthStrategy", module = "boosters._boosters_rs", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PyGrowthStrategy {
    /// Grow tree level-by-level (like XGBoost).
    #[default]
    Depthwise = 0,
    /// Grow tree by best-first split (like LightGBM).
    Leafwise = 1,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyGrowthStrategy {
    /// String representation like StrEnum.
    fn __str__(&self) -> &'static str {
        match self {
            PyGrowthStrategy::Depthwise => "depthwise",
            PyGrowthStrategy::Leafwise => "leafwise",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            PyGrowthStrategy::Depthwise => "GrowthStrategy.Depthwise",
            PyGrowthStrategy::Leafwise => "GrowthStrategy.Leafwise",
        }
    }

    /// Hash using the string value for StrEnum-like behavior.
    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.__str__().hash(&mut hasher);
        hasher.finish()
    }

    /// Pickle support: reduce to module path and variant name.
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        let variant_name = match self {
            PyGrowthStrategy::Depthwise => "Depthwise",
            PyGrowthStrategy::Leafwise => "Leafwise",
        };
        // Return a callable that reconstructs the enum
        let boosters = py.import("boosters._boosters_rs")?;
        let growth_strategy = boosters.getattr("GrowthStrategy")?;
        let variant = growth_strategy.getattr(variant_name)?;
        // Return a function that returns the variant directly
        // Use a lambda-like pattern: (lambda: variant,)
        let obj = variant.unbind();
        Ok((obj.getattr(py, "__class__")?, (variant_name.to_string(),)))
    }

    /// Deepcopy support - return self since enum variants are singletons.
    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        *self
    }

    /// Copy support - return self since enum variants are singletons.
    fn __copy__(&self) -> Self {
        *self
    }
}

// =============================================================================
// GBDT Config
// =============================================================================

/// Main configuration for GBDT model.
///
/// This is the primary configuration class for gradient boosted decision trees.
/// All parameters are flat (no nested config objects) matching the core Rust API.
///
/// Args:
///     n_estimators: Number of boosting rounds (trees to train). Default: 100.
///     learning_rate: Step size shrinkage (0.01 - 0.3 typical). Default: 0.3.
///     objective: Loss function for training. Default: Objective.Squared().
///     metric: Evaluation metric. None uses objective's default.
///
///     # Tree Structure
///     growth_strategy: Tree growth strategy. Default: GrowthStrategy.Depthwise.
///     max_depth: Maximum tree depth (only for depthwise). Default: 6.
///     n_leaves: Maximum leaves (only for leafwise). Default: 31.
///     max_onehot_cats: Max categories for one-hot encoding. Default: 4.
///
///     # Regularization
///     l1: L1 regularization on leaf weights. Default: 0.0.
///     l2: L2 regularization on leaf weights. Default: 1.0.
///     min_gain_to_split: Minimum gain required to make a split. Default: 0.0.
///     min_child_weight: Minimum sum of hessians in a leaf. Default: 1.0.
///     min_samples_leaf: Minimum samples in a leaf. Default: 1.
///
///     # Sampling
///     subsample: Row subsampling ratio per tree. Default: 1.0.
///     colsample_bytree: Column subsampling per tree. Default: 1.0.
///     colsample_bylevel: Column subsampling per level. Default: 1.0.
///
///     # Linear Leaves (experimental)
///     linear_leaves: Enable linear models in leaves. Default: False.
///     linear_l2: L2 regularization for linear coefficients. Default: 0.01.
///     linear_l1: L1 regularization for linear coefficients. Default: 0.0.
///
///     # Training Control
///     early_stopping_rounds: Stop if no improvement for this many rounds.
///     seed: Random seed for reproducibility. Default: 42.
///
/// Examples:
///     >>> config = GBDTConfig(
///     ...     n_estimators=500,
///     ...     learning_rate=0.1,
///     ...     objective=Objective.logistic(),
///     ...     max_depth=6,
///     ...     l2=1.0,
///     ... )
#[gen_stub_pyclass]
#[pyclass(name = "GBDTConfig", module = "boosters._boosters_rs")]
#[derive(Debug, Clone)]
pub struct PyGBDTConfig {
    // === Boosting parameters ===
    /// Number of boosting rounds.
    #[pyo3(get)]
    pub n_estimators: u32,
    /// Learning rate (step size shrinkage).
    #[pyo3(get)]
    pub learning_rate: f64,
    /// Objective function.
    pub objective: PyObjective,
    /// Evaluation metric (optional).
    pub metric: Option<PyMetric>,

    // === Tree structure ===
    /// Growth strategy for tree building.
    #[pyo3(get)]
    pub growth_strategy: PyGrowthStrategy,
    /// Maximum depth of tree (for depthwise growth).
    #[pyo3(get)]
    pub max_depth: u32,
    /// Maximum number of leaves (for leafwise growth).
    #[pyo3(get)]
    pub n_leaves: u32,
    /// Maximum categories for one-hot encoding categorical splits.
    #[pyo3(get)]
    pub max_onehot_cats: u32,

    // === Regularization ===
    /// L1 regularization on leaf weights.
    #[pyo3(get)]
    pub l1: f64,
    /// L2 regularization on leaf weights.
    #[pyo3(get)]
    pub l2: f64,
    /// Minimum gain required to make a split.
    #[pyo3(get)]
    pub min_gain_to_split: f64,
    /// Minimum sum of hessians required in a leaf.
    #[pyo3(get)]
    pub min_child_weight: f64,
    /// Minimum number of samples required in a leaf.
    #[pyo3(get)]
    pub min_samples_leaf: u32,

    // === Sampling ===
    /// Row subsampling ratio per tree.
    #[pyo3(get)]
    pub subsample: f64,
    /// Column subsampling ratio per tree.
    #[pyo3(get)]
    pub colsample_bytree: f64,
    /// Column subsampling ratio per level.
    #[pyo3(get)]
    pub colsample_bylevel: f64,

    // === Linear leaves ===
    /// Enable linear models in leaves.
    #[pyo3(get)]
    pub linear_leaves: bool,
    /// L2 regularization for linear coefficients.
    #[pyo3(get)]
    pub linear_l2: f64,
    /// L1 regularization for linear coefficients.
    #[pyo3(get)]
    pub linear_l1: f64,

    // === Training control ===
    /// Early stopping rounds (None = disabled).
    #[pyo3(get)]
    pub early_stopping_rounds: Option<u32>,
    /// Random seed.
    #[pyo3(get)]
    pub seed: u64,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyGBDTConfig {
    #[new]
    #[pyo3(signature = (
        n_estimators = 100,
        learning_rate = 0.3,
        objective = None,
        metric = None,
        growth_strategy = PyGrowthStrategy::Depthwise,
        max_depth = 6,
        n_leaves = 31,
        max_onehot_cats = 4,
        l1 = 0.0,
        l2 = 1.0,
        min_gain_to_split = 0.0,
        min_child_weight = 1.0,
        min_samples_leaf = 1,
        subsample = 1.0,
        colsample_bytree = 1.0,
        colsample_bylevel = 1.0,
        linear_leaves = false,
        linear_l2 = 0.01,
        linear_l1 = 0.0,
        early_stopping_rounds = None,
        seed = 42
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_estimators: u32,
        learning_rate: f64,
        #[gen_stub(override_type(type_repr = "Objective | None"))]
        objective: Option<PyObjective>,
        #[gen_stub(override_type(type_repr = "Metric | None"))]
        metric: Option<PyMetric>,
        growth_strategy: PyGrowthStrategy,
        max_depth: u32,
        n_leaves: u32,
        max_onehot_cats: u32,
        l1: f64,
        l2: f64,
        min_gain_to_split: f64,
        min_child_weight: f64,
        min_samples_leaf: u32,
        subsample: f64,
        colsample_bytree: f64,
        colsample_bylevel: f64,
        linear_leaves: bool,
        linear_l2: f64,
        linear_l1: f64,
        early_stopping_rounds: Option<u32>,
        seed: u64,
    ) -> PyResult<Self> {
        // Validate n_estimators
        if n_estimators == 0 {
            return Err(BoostersError::InvalidParameter {
                name: "n_estimators".to_string(),
                reason: "must be positive".to_string(),
            }
            .into());
        }

        // Validate learning_rate
        if learning_rate <= 0.0 {
            return Err(BoostersError::InvalidParameter {
                name: "learning_rate".to_string(),
                reason: "must be positive".to_string(),
            }
            .into());
        }

        // Validate sampling ratios
        if subsample <= 0.0 || subsample > 1.0 {
            return Err(BoostersError::InvalidParameter {
                name: "subsample".to_string(),
                reason: "must be in (0, 1]".to_string(),
            }
            .into());
        }
        if colsample_bytree <= 0.0 || colsample_bytree > 1.0 {
            return Err(BoostersError::InvalidParameter {
                name: "colsample_bytree".to_string(),
                reason: "must be in (0, 1]".to_string(),
            }
            .into());
        }
        if colsample_bylevel <= 0.0 || colsample_bylevel > 1.0 {
            return Err(BoostersError::InvalidParameter {
                name: "colsample_bylevel".to_string(),
                reason: "must be in (0, 1]".to_string(),
            }
            .into());
        }

        // Validate regularization
        if l1 < 0.0 {
            return Err(BoostersError::InvalidParameter {
                name: "l1".to_string(),
                reason: "must be non-negative".to_string(),
            }
            .into());
        }
        if l2 < 0.0 {
            return Err(BoostersError::InvalidParameter {
                name: "l2".to_string(),
                reason: "must be non-negative".to_string(),
            }
            .into());
        }
        if min_child_weight < 0.0 {
            return Err(BoostersError::InvalidParameter {
                name: "min_child_weight".to_string(),
                reason: "must be non-negative".to_string(),
            }
            .into());
        }
        if min_gain_to_split < 0.0 {
            return Err(BoostersError::InvalidParameter {
                name: "min_gain_to_split".to_string(),
                reason: "must be non-negative".to_string(),
            }
            .into());
        }

        Ok(Self {
            n_estimators,
            learning_rate,
            objective: objective.unwrap_or_default(),
            metric,
            growth_strategy,
            max_depth,
            n_leaves,
            max_onehot_cats,
            l1,
            l2,
            min_gain_to_split,
            min_child_weight,
            min_samples_leaf,
            subsample,
            colsample_bytree,
            colsample_bylevel,
            linear_leaves,
            linear_l2,
            linear_l1,
            early_stopping_rounds,
            seed,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "GBDTConfig(n_estimators={}, learning_rate={}, objective={:?})",
            self.n_estimators, self.learning_rate, self.objective
        )
    }

    /// Get the objective function.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "Objective"))]
    fn objective(&self) -> PyObjective {
        self.objective.clone()
    }

    /// Get the evaluation metric (or None).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "Metric | None"))]
    fn metric(&self) -> Option<PyMetric> {
        self.metric.clone()
    }
}

impl Default for PyGBDTConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.3,
            objective: PyObjective::default(),
            metric: None,
            growth_strategy: PyGrowthStrategy::Depthwise,
            max_depth: 6,
            n_leaves: 31,
            max_onehot_cats: 4,
            l1: 0.0,
            l2: 1.0,
            min_gain_to_split: 0.0,
            min_child_weight: 1.0,
            min_samples_leaf: 1,
            subsample: 1.0,
            colsample_bytree: 1.0,
            colsample_bylevel: 1.0,
            linear_leaves: false,
            linear_l2: 0.01,
            linear_l1: 0.0,
            early_stopping_rounds: None,
            seed: 42,
        }
    }
}

impl PyGBDTConfig {
    /// Convert Python config to core Rust config.
    ///
    /// Called at fit-time to create the Rust training configuration.
    pub fn to_core(&self, _py: Python<'_>) -> PyResult<boosters::GBDTConfig> {
        use boosters::training::gbdt::GrowthStrategy;

        // Convert objective
        let objective: boosters::training::Objective = (&self.objective).into();

        // Convert metric (optional)
        let metric = self.metric.as_ref().map(|m| m.into());

        // Convert growth strategy
        let growth_strategy = match self.growth_strategy {
            PyGrowthStrategy::Depthwise => GrowthStrategy::DepthWise {
                max_depth: self.max_depth,
            },
            PyGrowthStrategy::Leafwise => GrowthStrategy::LeafWise {
                max_leaves: self.n_leaves,
            },
        };

        // Convert linear leaves config (optional)
        let linear_leaves = if self.linear_leaves {
            Some(boosters::training::gbdt::LinearLeafConfig {
                lambda: self.linear_l2 as f32,
                alpha: self.linear_l1 as f32,
                max_iterations: 10,          // Default
                tolerance: 1e-6,             // Default
                min_samples: 50,             // Default
                coefficient_threshold: 1e-6, // Default
                max_features: 10,            // Default
            })
        } else {
            None
        };

        // Build core config using struct constructor.
        // This ensures compile-time errors if new fields are added to GBDTConfig.
        let config = boosters::GBDTConfig {
            objective,
            metric,
            n_trees: self.n_estimators,
            learning_rate: self.learning_rate as f32,
            growth_strategy,
            max_onehot_cats: self.max_onehot_cats,
            lambda: self.l2 as f32,
            alpha: self.l1 as f32,
            min_gain: self.min_gain_to_split as f32,
            min_child_weight: self.min_child_weight as f32,
            min_samples_leaf: self.min_samples_leaf,
            subsample: self.subsample as f32,
            colsample_bytree: self.colsample_bytree as f32,
            colsample_bylevel: self.colsample_bylevel as f32,
            binning: Default::default(),
            linear_leaves,
            early_stopping_rounds: self.early_stopping_rounds,
            cache_size: 8,
            seed: self.seed,
            verbosity: Default::default(),
        };

        // Validate the constructed config
        config.validate().map_err(|e| {
            BoostersError::InvalidParameter {
                name: "config".to_string(),
                reason: e.to_string(),
            }
        })?;

        Ok(config)
    }
}

// =============================================================================
// GBLinear Config
// =============================================================================

/// Main configuration for GBLinear model.
///
/// GBLinear uses gradient boosting to train a linear model via coordinate
/// descent. Simpler than GBDT but can be effective for linear relationships.
///
/// Args:
///     n_estimators: Number of boosting rounds. Default: 100.
///     learning_rate: Step size for weight updates. Default: 0.5.
///     objective: Loss function for training. Default: Objective.Squared().
///     metric: Evaluation metric. None uses objective's default.
///     l1: L1 regularization (alpha). Encourages sparse weights. Default: 0.0.
///     l2: L2 regularization (lambda). Prevents large weights. Default: 1.0.
///     early_stopping_rounds: Stop if no improvement for this many rounds.
///     seed: Random seed for reproducibility. Default: 42.
///
/// Examples:
///     >>> config = GBLinearConfig(
///     ...     n_estimators=200,
///     ...     learning_rate=0.3,
///     ...     objective=Objective.logistic(),
///     ...     l2=0.1,
///     ... )
#[gen_stub_pyclass]
#[pyclass(name = "GBLinearConfig", module = "boosters._boosters_rs")]
#[derive(Debug, Clone)]
pub struct PyGBLinearConfig {
    /// Number of boosting rounds.
    #[pyo3(get)]
    pub n_estimators: u32,
    /// Learning rate (step size).
    #[pyo3(get)]
    pub learning_rate: f64,
    /// Objective function.
    pub objective: PyObjective,
    /// Evaluation metric (optional).
    pub metric: Option<PyMetric>,
    /// L1 regularization (alpha).
    #[pyo3(get)]
    pub l1: f64,
    /// L2 regularization (lambda).
    #[pyo3(get)]
    pub l2: f64,
    /// Early stopping rounds (None = disabled).
    #[pyo3(get)]
    pub early_stopping_rounds: Option<u32>,
    /// Random seed.
    #[pyo3(get)]
    pub seed: u64,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyGBLinearConfig {
    #[new]
    #[pyo3(signature = (
        n_estimators = 100,
        learning_rate = 0.5,
        objective = None,
        metric = None,
        l1 = 0.0,
        l2 = 1.0,
        early_stopping_rounds = None,
        seed = 42
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_estimators: u32,
        learning_rate: f64,
        #[gen_stub(override_type(type_repr = "Objective | None"))]
        objective: Option<PyObjective>,
        #[gen_stub(override_type(type_repr = "Metric | None"))]
        metric: Option<PyMetric>,
        l1: f64,
        l2: f64,
        early_stopping_rounds: Option<u32>,
        seed: u64,
    ) -> PyResult<Self> {
        // Validate n_estimators
        if n_estimators == 0 {
            return Err(BoostersError::InvalidParameter {
                name: "n_estimators".to_string(),
                reason: "must be positive".to_string(),
            }
            .into());
        }

        // Validate learning_rate
        if learning_rate <= 0.0 {
            return Err(BoostersError::InvalidParameter {
                name: "learning_rate".to_string(),
                reason: "must be positive".to_string(),
            }
            .into());
        }

        // Validate regularization
        if l1 < 0.0 {
            return Err(BoostersError::InvalidParameter {
                name: "l1".to_string(),
                reason: "must be non-negative".to_string(),
            }
            .into());
        }

        if l2 < 0.0 {
            return Err(BoostersError::InvalidParameter {
                name: "l2".to_string(),
                reason: "must be non-negative".to_string(),
            }
            .into());
        }

        Ok(Self {
            n_estimators,
            learning_rate,
            objective: objective.unwrap_or_default(),
            metric,
            l1,
            l2,
            early_stopping_rounds,
            seed,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "GBLinearConfig(n_estimators={}, learning_rate={}, l1={}, l2={}, objective={:?})",
            self.n_estimators, self.learning_rate, self.l1, self.l2, self.objective
        )
    }

    /// Get the objective function.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "Objective"))]
    fn objective(&self) -> PyObjective {
        self.objective.clone()
    }

    /// Get the evaluation metric (or None).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "Metric | None"))]
    fn metric(&self) -> Option<PyMetric> {
        self.metric.clone()
    }
}

impl PyGBLinearConfig {
    /// Convert to core GBLinearConfig for training.
    pub fn to_core(&self) -> PyResult<boosters::GBLinearConfig> {
        // Convert objective
        let objective: boosters::training::Objective = (&self.objective).into();

        // Convert metric if present
        let metric = self.metric.as_ref().map(|m| m.into());

        // Build core config using struct constructor instead of builder.
        // This ensures compile-time errors if new fields are added to GBLinearConfig.
        let config = boosters::GBLinearConfig {
            objective,
            metric,
            n_rounds: self.n_estimators,
            learning_rate: self.learning_rate as f32,
            alpha: self.l1 as f32,
            lambda: self.l2 as f32,
            parallel: true, // Default
            feature_selector: Default::default(),
            early_stopping_rounds: self.early_stopping_rounds,
            seed: self.seed,
            verbosity: Default::default(),
        };

        // Validate the constructed config
        config
            .validate()
            .map_err(|e| BoostersError::ValidationError(e.to_string()))?;

        Ok(config)
    }
}

impl Default for PyGBLinearConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.5,
            objective: PyObjective::default(),
            metric: None,
            l1: 0.0,
            l2: 1.0,
            early_stopping_rounds: None,
            seed: 42,
        }
    }
}
