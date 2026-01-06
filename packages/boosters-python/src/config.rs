//! Configuration types for Python bindings.
//!
//! Flat config structure matching core Rust configs.

use boosters::data::BinningConfig;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::metrics::PyMetric;
use crate::objectives::PyObjective;
use crate::types::{PyGBLinearUpdateStrategy, PyVerbosity};
use crate::validation::{validate_non_negative, validate_positive, validate_ratio};

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
/// # Arguments
///
/// * `n_estimators` - Number of boosting rounds (trees to train). Default: 100.
/// * `learning_rate` - Step size shrinkage (0.01 - 0.3 typical). Default: 0.3.
/// * `objective` - Loss function for training. Default: Objective.Squared().
/// * `metric` - Evaluation metric. None uses objective's default.
/// * `growth_strategy` - Tree growth strategy. Default: GrowthStrategy.Depthwise.
/// * `max_depth` - Maximum tree depth (only for depthwise). Default: 6.
/// * `n_leaves` - Maximum leaves (only for leafwise). Default: 31.
/// * `max_onehot_cats` - Max categories for one-hot encoding. Default: 4.
/// * `l1` - L1 regularization on leaf weights. Default: 0.0.
/// * `l2` - L2 regularization on leaf weights. Default: 1.0.
/// * `min_gain_to_split` - Minimum gain required to make a split. Default: 0.0.
/// * `min_child_weight` - Minimum sum of hessians in a leaf. Default: 1.0.
/// * `min_samples_leaf` - Minimum samples in a leaf. Default: 1.
/// * `subsample` - Row subsampling ratio per tree. Default: 1.0.
/// * `colsample_bytree` - Column subsampling per tree. Default: 1.0.
/// * `colsample_bylevel` - Column subsampling per level. Default: 1.0.
/// * `linear_leaves` - Enable linear models in leaves (experimental). Default: False.
/// * `linear_l2` - L2 regularization for linear coefficients. Default: 0.01.
/// * `linear_l1` - L1 regularization for linear coefficients. Default: 0.0.
/// * `early_stopping_rounds` - Stop if no improvement for this many rounds.
/// * `seed` - Random seed for reproducibility. Default: 42.
///
/// # Example (Python)
///
/// ```text
/// config = GBDTConfig(
///     n_estimators=500,
///     learning_rate=0.1,
///     objective=Objective.logistic(),
///     max_depth=6,
///     l2=1.0,
/// )
/// ```
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
    pub learning_rate: f32,
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
    pub l1: f32,
    /// L2 regularization on leaf weights.
    #[pyo3(get)]
    pub l2: f32,
    /// Minimum gain required to make a split.
    #[pyo3(get)]
    pub min_gain_to_split: f32,
    /// Minimum sum of hessians required in a leaf.
    #[pyo3(get)]
    pub min_child_weight: f32,
    /// Minimum number of samples required in a leaf.
    #[pyo3(get)]
    pub min_samples_leaf: u32,

    // === Sampling ===
    /// Row subsampling ratio per tree.
    #[pyo3(get)]
    pub subsample: f32,
    /// Column subsampling ratio per tree.
    #[pyo3(get)]
    pub colsample_bytree: f32,
    /// Column subsampling ratio per level.
    #[pyo3(get)]
    pub colsample_bylevel: f32,

    // === Linear leaves ===
    /// Enable linear models in leaves.
    #[pyo3(get)]
    pub linear_leaves: bool,
    /// L2 regularization for linear coefficients.
    #[pyo3(get)]
    pub linear_l2: f32,
    /// L1 regularization for linear coefficients.
    #[pyo3(get)]
    pub linear_l1: f32,
    /// Maximum coordinate descent iterations for linear leaves.
    #[pyo3(get)]
    pub linear_max_iterations: u32,
    /// Convergence tolerance for linear leaves.
    #[pyo3(get)]
    pub linear_tolerance: f64,
    /// Minimum samples required to fit linear model in leaf.
    #[pyo3(get)]
    pub linear_min_samples: u32,
    /// Threshold for pruning small coefficients.
    #[pyo3(get)]
    pub linear_coefficient_threshold: f32,
    /// Maximum features in linear model per leaf.
    #[pyo3(get)]
    pub linear_max_features: u32,
    /// Use global features instead of path features for linear models.
    /// When true, uses the top-k most frequently split features for all leaves.
    /// This can improve extrapolation by ensuring important features are always included.
    #[pyo3(get)]
    pub linear_use_global_features: bool,
    /// Number of initial trees to skip linear leaf fitting.
    /// Default is 1 (first tree has homogeneous gradients). Set to 0 to enable from first tree.
    #[pyo3(get)]
    pub linear_skip_first_n_trees: u32,

    // === Binning ===
    /// Maximum bins per feature for binning (1-256).
    #[pyo3(get)]
    pub max_bins: u32,

    // === Feature Bundling (EFB) ===
    /// Enable exclusive feature bundling for sparse data. Default: true.
    /// Bundles sparse/one-hot features to reduce memory and speed up training.
    #[pyo3(get)]
    pub enable_bundling: bool,
    /// Sparsity threshold (fraction of zeros to use sparse storage).
    /// Features with density ≤ (1 - threshold) are considered sparse.
    #[pyo3(get)]
    pub sparsity_threshold: f32,
    /// Max cardinality to auto-detect as categorical.
    /// Features with ≤ this many unique integer values may be treated as categorical.
    #[pyo3(get)]
    pub max_categorical_cardinality: u32,
    /// Number of samples for computing bin boundaries (for large datasets).
    #[pyo3(get)]
    pub binning_sample_cnt: usize,

    // === Resource control ===
    /// Histogram cache size (number of slots). Default: 8.
    #[pyo3(get)]
    pub cache_size: usize,

    // === Training control ===
    /// Early stopping rounds (None = disabled).
    #[pyo3(get)]
    pub early_stopping_rounds: Option<u32>,
    /// Random seed.
    #[pyo3(get)]
    pub seed: u64,
    /// Verbosity level for training output.
    #[pyo3(get)]
    pub verbosity: PyVerbosity,
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
        linear_max_iterations = 10,
        linear_tolerance = 1e-6,
        linear_min_samples = 50,
        linear_coefficient_threshold = 1e-6,
        linear_max_features = 10,
        linear_use_global_features = false,
        linear_skip_first_n_trees = 1,
        max_bins = 256,
        enable_bundling = true,
        sparsity_threshold = 0.9,
        max_categorical_cardinality = 0,
        binning_sample_cnt = 200000,
        cache_size = 8,
        early_stopping_rounds = None,
        seed = 42,
        verbosity = PyVerbosity::Silent
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_estimators: u32,
        learning_rate: f32,
        #[gen_stub(override_type(type_repr = "Objective | None"))] objective: Option<PyObjective>,
        #[gen_stub(override_type(type_repr = "Metric | None"))] metric: Option<PyMetric>,
        growth_strategy: PyGrowthStrategy,
        max_depth: u32,
        n_leaves: u32,
        max_onehot_cats: u32,
        l1: f32,
        l2: f32,
        min_gain_to_split: f32,
        min_child_weight: f32,
        min_samples_leaf: u32,
        subsample: f32,
        colsample_bytree: f32,
        colsample_bylevel: f32,
        linear_leaves: bool,
        linear_l2: f32,
        linear_l1: f32,
        linear_max_iterations: u32,
        linear_tolerance: f64,
        linear_min_samples: u32,
        linear_coefficient_threshold: f32,
        linear_max_features: u32,
        linear_use_global_features: bool,
        linear_skip_first_n_trees: u32,
        max_bins: u32,
        enable_bundling: bool,
        sparsity_threshold: f32,
        max_categorical_cardinality: u32,
        binning_sample_cnt: usize,
        cache_size: usize,
        early_stopping_rounds: Option<u32>,
        seed: u64,
        verbosity: PyVerbosity,
    ) -> PyResult<Self> {
        // Validate parameters
        validate_positive("n_estimators", n_estimators)?;
        validate_positive("learning_rate", learning_rate)?;
        validate_ratio("subsample", subsample)?;
        validate_ratio("colsample_bytree", colsample_bytree)?;
        validate_ratio("colsample_bylevel", colsample_bylevel)?;
        validate_non_negative("l1", l1)?;
        validate_non_negative("l2", l2)?;
        validate_non_negative("min_child_weight", min_child_weight)?;
        validate_non_negative("min_gain_to_split", min_gain_to_split)?;
        validate_ratio("sparsity_threshold", sparsity_threshold)?;
        validate_positive("binning_sample_cnt", binning_sample_cnt)?;
        validate_positive("cache_size", cache_size)?;

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
            linear_max_iterations,
            linear_tolerance,
            linear_min_samples,
            linear_coefficient_threshold,
            linear_max_features,
            linear_use_global_features,
            linear_skip_first_n_trees,
            max_bins,
            enable_bundling,
            sparsity_threshold,
            max_categorical_cardinality,
            binning_sample_cnt,
            cache_size,
            early_stopping_rounds,
            seed,
            verbosity,
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
            linear_max_iterations: 10,
            linear_tolerance: 1e-6,
            linear_min_samples: 50,
            linear_coefficient_threshold: 1e-6,
            linear_max_features: 10,
            linear_use_global_features: false,
            linear_skip_first_n_trees: 1,
            max_bins: 256,
            enable_bundling: true,
            sparsity_threshold: 0.9,
            max_categorical_cardinality: 0,
            binning_sample_cnt: 200_000,
            cache_size: 8,
            early_stopping_rounds: None,
            seed: 42,
            verbosity: PyVerbosity::Silent,
        }
    }
}

impl From<&PyGBDTConfig> for boosters::GBDTConfig {
    fn from(py_config: &PyGBDTConfig) -> Self {
        use boosters::training::gbdt::GrowthStrategy;

        // Convert objective
        let objective: boosters::training::Objective = (&py_config.objective).into();

        // Convert metric. If not specified in Python, use the core default.
        let metric: boosters::training::Metric = match py_config.metric.as_ref() {
            Some(m) => m.into(),
            None => boosters::training::default_metric_for_objective(&objective),
        };

        // Convert growth strategy
        let growth_strategy = match py_config.growth_strategy {
            PyGrowthStrategy::Depthwise => GrowthStrategy::DepthWise {
                max_depth: py_config.max_depth,
            },
            PyGrowthStrategy::Leafwise => GrowthStrategy::LeafWise {
                max_leaves: py_config.n_leaves,
            },
        };

        // Convert linear leaves config (optional)
        let linear_leaves = if py_config.linear_leaves {
            use boosters::training::gbdt::LinearFeatureSelection;
            let feature_selection = if py_config.linear_use_global_features {
                LinearFeatureSelection::GlobalFeatures
            } else {
                LinearFeatureSelection::PathFeatures
            };
            Some(boosters::training::gbdt::LinearLeafConfig {
                lambda: py_config.linear_l2,
                alpha: py_config.linear_l1,
                max_iterations: py_config.linear_max_iterations,
                tolerance: py_config.linear_tolerance,
                min_samples: py_config.linear_min_samples as usize,
                coefficient_threshold: py_config.linear_coefficient_threshold,
                max_features: py_config.linear_max_features as usize,
                feature_selection,
                skip_first_n_trees: py_config.linear_skip_first_n_trees,
            })
        } else {
            None
        };

        boosters::GBDTConfig {
            objective,
            metric,
            n_trees: py_config.n_estimators,
            learning_rate: py_config.learning_rate,
            growth_strategy,
            max_onehot_cats: py_config.max_onehot_cats,
            lambda: py_config.l2,
            alpha: py_config.l1,
            min_gain: py_config.min_gain_to_split,
            min_child_weight: py_config.min_child_weight,
            min_samples_leaf: py_config.min_samples_leaf,
            subsample: py_config.subsample,
            colsample_bytree: py_config.colsample_bytree,
            colsample_bylevel: py_config.colsample_bylevel,
            // Binning config with bundling settings
            binning: BinningConfig {
                max_bins: py_config.max_bins,
                sparsity_threshold: py_config.sparsity_threshold,
                enable_bundling: py_config.enable_bundling,
                max_categorical_cardinality: py_config.max_categorical_cardinality,
                sample_cnt: py_config.binning_sample_cnt,
            },
            linear_leaves,
            early_stopping_rounds: py_config.early_stopping_rounds,
            cache_size: py_config.cache_size,
            seed: py_config.seed,
            verbosity: py_config.verbosity.into(),
        }
    }
}

impl From<&boosters::GBDTConfig> for PyGBDTConfig {
    fn from(config: &boosters::GBDTConfig) -> Self {
        use boosters::training::gbdt::GrowthStrategy;

        // Convert objective
        let objective: PyObjective = (&config.objective).into();

        // Convert metric
        let metric = match &config.metric {
            boosters::training::Metric::None => None,
            m => Some(m.into()),
        };

        // Convert growth strategy
        let (growth_strategy, max_depth, n_leaves) = match config.growth_strategy {
            GrowthStrategy::DepthWise { max_depth } => (PyGrowthStrategy::Depthwise, max_depth, 31),
            GrowthStrategy::LeafWise { max_leaves } => (PyGrowthStrategy::Leafwise, 6, max_leaves),
        };

        // Convert linear leaves config
        let (
            linear_leaves,
            linear_l2,
            linear_l1,
            linear_max_iterations,
            linear_tolerance,
            linear_min_samples,
            linear_coefficient_threshold,
            linear_max_features,
            linear_use_global_features,
            linear_skip_first_n_trees,
        ) = if let Some(ref ll) = config.linear_leaves {
            use boosters::training::gbdt::LinearFeatureSelection;
            let use_global = matches!(ll.feature_selection, LinearFeatureSelection::GlobalFeatures);
            (
                true,
                ll.lambda,
                ll.alpha,
                ll.max_iterations,
                ll.tolerance,
                ll.min_samples as u32,
                ll.coefficient_threshold,
                ll.max_features as u32,
                use_global,
                ll.skip_first_n_trees,
            )
        } else {
            (false, 0.01, 0.0, 10, 1e-6, 50, 1e-6, 10, false, 1)
        };

        // Convert verbosity
        let verbosity = match config.verbosity {
            boosters::training::Verbosity::Silent => PyVerbosity::Silent,
            boosters::training::Verbosity::Warning => PyVerbosity::Warning,
            boosters::training::Verbosity::Info => PyVerbosity::Info,
            boosters::training::Verbosity::Debug => PyVerbosity::Debug,
        };

        Self {
            n_estimators: config.n_trees,
            learning_rate: config.learning_rate,
            objective,
            metric,
            growth_strategy,
            max_depth,
            n_leaves,
            max_onehot_cats: config.max_onehot_cats,
            l1: config.alpha,
            l2: config.lambda,
            min_gain_to_split: config.min_gain,
            min_child_weight: config.min_child_weight,
            min_samples_leaf: config.min_samples_leaf,
            subsample: config.subsample,
            colsample_bytree: config.colsample_bytree,
            colsample_bylevel: config.colsample_bylevel,
            linear_leaves,
            linear_l2,
            linear_l1,
            linear_max_iterations,
            linear_tolerance,
            linear_min_samples,
            linear_coefficient_threshold,
            linear_max_features,
            linear_use_global_features,
            linear_skip_first_n_trees,
            max_bins: config.binning.max_bins,
            enable_bundling: config.binning.enable_bundling,
            sparsity_threshold: config.binning.sparsity_threshold,
            max_categorical_cardinality: config.binning.max_categorical_cardinality,
            binning_sample_cnt: config.binning.sample_cnt,
            cache_size: config.cache_size,
            early_stopping_rounds: config.early_stopping_rounds,
            seed: config.seed,
            verbosity,
        }
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
///     l2: L2 regularization (lambda). Prevents large weights. Default: 0.0.
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
    pub learning_rate: f32,
    /// Coordinate descent update strategy.
    #[pyo3(get)]
    pub update_strategy: PyGBLinearUpdateStrategy,
    /// Objective function.
    pub objective: PyObjective,
    /// Evaluation metric (optional).
    pub metric: Option<PyMetric>,
    /// L1 regularization (alpha).
    #[pyo3(get)]
    pub l1: f32,
    /// L2 regularization (lambda).
    #[pyo3(get)]
    pub l2: f32,
    /// Maximum per-coordinate Newton step (stability), in absolute value.
    ///
    /// Set to `0.0` to disable.
    #[pyo3(get)]
    pub max_delta_step: f32,
    /// Early stopping rounds (None = disabled).
    #[pyo3(get)]
    pub early_stopping_rounds: Option<u32>,
    /// Random seed.
    #[pyo3(get)]
    pub seed: u64,
    /// Verbosity level for training output.
    #[pyo3(get)]
    pub verbosity: PyVerbosity,
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
        l2 = 0.0,
        update_strategy = PyGBLinearUpdateStrategy::Shotgun,
        max_delta_step = 0.0,
        early_stopping_rounds = None,
        seed = 42,
        verbosity = PyVerbosity::Silent
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_estimators: u32,
        learning_rate: f32,
        #[gen_stub(override_type(type_repr = "Objective | None"))] objective: Option<PyObjective>,
        #[gen_stub(override_type(type_repr = "Metric | None"))] metric: Option<PyMetric>,
        l1: f32,
        l2: f32,
        update_strategy: PyGBLinearUpdateStrategy,
        max_delta_step: f32,
        early_stopping_rounds: Option<u32>,
        seed: u64,
        verbosity: PyVerbosity,
    ) -> PyResult<Self> {
        // Validate parameters
        validate_positive("n_estimators", n_estimators)?;
        validate_positive("learning_rate", learning_rate)?;
        validate_non_negative("l1", l1)?;
        validate_non_negative("l2", l2)?;
        validate_non_negative("max_delta_step", max_delta_step)?;

        Ok(Self {
            n_estimators,
            learning_rate,
            update_strategy,
            objective: objective.unwrap_or_default(),
            metric,
            l1,
            l2,
            max_delta_step,
            early_stopping_rounds,
            seed,
            verbosity,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "GBLinearConfig(n_estimators={}, learning_rate={}, l1={}, l2={}, update_strategy={:?}, max_delta_step={}, objective={:?})",
            self.n_estimators,
            self.learning_rate,
            self.l1,
            self.l2,
            self.update_strategy,
            self.max_delta_step,
            self.objective
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

impl From<&PyGBLinearConfig> for boosters::GBLinearConfig {
    fn from(py_config: &PyGBLinearConfig) -> Self {
        // Convert objective
        let objective: boosters::training::Objective = (&py_config.objective).into();

        // Convert metric. If not specified in Python, use the core default.
        let metric: boosters::training::Metric = match py_config.metric.as_ref() {
            Some(m) => m.into(),
            None => boosters::training::default_metric_for_objective(&objective),
        };

        boosters::GBLinearConfig {
            objective,
            metric,
            n_rounds: py_config.n_estimators,
            learning_rate: py_config.learning_rate,
            alpha: py_config.l1,
            lambda: py_config.l2,
            update_strategy: py_config.update_strategy.into(),
            feature_selector: Default::default(),
            max_delta_step: py_config.max_delta_step,
            early_stopping_rounds: py_config.early_stopping_rounds,
            seed: py_config.seed,
            verbosity: py_config.verbosity.into(),
        }
    }
}

impl From<&boosters::GBLinearConfig> for PyGBLinearConfig {
    fn from(config: &boosters::GBLinearConfig) -> Self {
        use boosters::training::gblinear::UpdateStrategy;

        // Convert objective
        let objective: PyObjective = (&config.objective).into();

        // Convert metric
        let metric = match &config.metric {
            boosters::training::Metric::None => None,
            m => Some(m.into()),
        };

        // Convert update strategy
        let update_strategy = match config.update_strategy {
            UpdateStrategy::Shotgun => PyGBLinearUpdateStrategy::Shotgun,
            UpdateStrategy::Sequential => PyGBLinearUpdateStrategy::Sequential,
        };

        // Convert verbosity
        let verbosity = match config.verbosity {
            boosters::training::Verbosity::Silent => PyVerbosity::Silent,
            boosters::training::Verbosity::Warning => PyVerbosity::Warning,
            boosters::training::Verbosity::Info => PyVerbosity::Info,
            boosters::training::Verbosity::Debug => PyVerbosity::Debug,
        };

        Self {
            n_estimators: config.n_rounds,
            learning_rate: config.learning_rate,
            update_strategy,
            objective,
            metric,
            l1: config.alpha,
            l2: config.lambda,
            max_delta_step: config.max_delta_step,
            early_stopping_rounds: config.early_stopping_rounds,
            seed: config.seed,
            verbosity,
        }
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
            l2: 0.0,
            update_strategy: PyGBLinearUpdateStrategy::Shotgun,
            max_delta_step: 0.0,
            early_stopping_rounds: None,
            seed: 42,
            verbosity: PyVerbosity::Silent,
        }
    }
}
