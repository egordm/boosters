//! GBDT configuration for Python bindings.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use super::{
    PyCategoricalConfig, PyEFBConfig, PyGrowthStrategy, PyLinearLeavesConfig,
    PyRegularizationConfig, PySamplingConfig, PyTreeConfig,
};
use crate::error::BoostersError;
use crate::metrics::PyMetric;
use crate::objectives::PyObjective;

/// Main configuration for GBDT model.
///
/// This is the primary configuration class for gradient boosted decision trees.
/// It accepts nested configuration objects for tree structure, regularization,
/// sampling, etc.
///
/// Args:
///     n_estimators: Number of boosting rounds (trees to train). Default: 100.
///     learning_rate: Step size shrinkage (0.01 - 0.3 typical). Default: 0.3.
///     objective: Loss function for training. Default: Objective.Squared().
///     metric: Evaluation metric. None uses objective's default.
///     tree: Tree structure parameters.
///     regularization: L1/L2 regularization parameters.
///     sampling: Row and column subsampling parameters.
///     categorical: Categorical feature handling.
///     efb: Exclusive Feature Bundling config.
///     linear_leaves: Linear model in leaves config. None = disabled.
///     early_stopping_rounds: Stop if no improvement for this many rounds.
///     seed: Random seed for reproducibility. Default: 42.
///
/// Examples:
///     >>> config = GBDTConfig(
///     ...     n_estimators=500,
///     ...     learning_rate=0.1,
///     ...     objective=Objective.logistic(),
///     ...     tree=TreeConfig(max_depth=6),
///     ... )
#[gen_stub_pyclass]
#[pyclass(name = "GBDTConfig", module = "boosters._boosters_rs")]
#[derive(Debug, Clone)]
pub struct PyGBDTConfig {
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
    /// Tree structure config.
    #[pyo3(get)]
    pub tree: PyTreeConfig,
    /// Regularization config.
    #[pyo3(get)]
    pub regularization: PyRegularizationConfig,
    /// Sampling config.
    #[pyo3(get)]
    pub sampling: PySamplingConfig,
    /// Categorical feature config.
    #[pyo3(get)]
    pub categorical: PyCategoricalConfig,
    /// Exclusive Feature Bundling config.
    #[pyo3(get)]
    pub efb: PyEFBConfig,
    /// Linear leaves config (None = disabled).
    #[pyo3(get)]
    pub linear_leaves: Option<PyLinearLeavesConfig>,
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
        tree = None,
        regularization = None,
        sampling = None,
        categorical = None,
        efb = None,
        linear_leaves = None,
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
        tree: Option<PyTreeConfig>,
        regularization: Option<PyRegularizationConfig>,
        sampling: Option<PySamplingConfig>,
        categorical: Option<PyCategoricalConfig>,
        efb: Option<PyEFBConfig>,
        linear_leaves: Option<PyLinearLeavesConfig>,
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

        Ok(Self {
            n_estimators,
            learning_rate,
            objective: objective.unwrap_or_default(),
            metric,
            tree: tree.unwrap_or_default(),
            regularization: regularization.unwrap_or_default(),
            sampling: sampling.unwrap_or_default(),
            categorical: categorical.unwrap_or_default(),
            efb: efb.unwrap_or_default(),
            linear_leaves,
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
            tree: PyTreeConfig::default(),
            regularization: PyRegularizationConfig::default(),
            sampling: PySamplingConfig::default(),
            categorical: PyCategoricalConfig::default(),
            efb: PyEFBConfig::default(),
            linear_leaves: None,
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

        // Convert tree config
        // Note: PyTreeConfig.max_depth is i32 (-1 means unlimited), core uses u32
        let growth_strategy = match self.tree.growth_strategy {
            PyGrowthStrategy::Depthwise => {
                let max_depth = if self.tree.max_depth < 0 {
                    6 // Default depth when unlimited
                } else {
                    self.tree.max_depth as u32
                };
                GrowthStrategy::DepthWise { max_depth }
            }
            PyGrowthStrategy::Leafwise => GrowthStrategy::LeafWise {
                max_leaves: self.tree.n_leaves,
            },
        };

        // Convert linear leaves config (optional)
        // Only convert if enabled
        let linear_leaves = self.linear_leaves.as_ref().and_then(|ll| {
            if ll.enable {
                Some(boosters::training::gbdt::LinearLeafConfig {
                    lambda: ll.l2 as f32,
                    alpha: ll.l1 as f32,
                    max_iterations: ll.max_iter,
                    tolerance: ll.tolerance,
                    min_samples: ll.min_samples as usize,
                    coefficient_threshold: 1e-6, // Default
                    max_features: 10,            // Default
                })
            } else {
                None
            }
        });

        // Build the core config using flattened builder pattern
        boosters::GBDTConfig::builder()
            .objective(objective)
            .maybe_metric(metric)
            .n_trees(self.n_estimators)
            .learning_rate(self.learning_rate as f32)
            // Tree params (flattened)
            .growth_strategy(growth_strategy)
            .max_onehot_cats(self.categorical.max_onehot as u32)
            // Regularization params (flattened)
            .lambda(self.regularization.l2 as f32)
            .alpha(self.regularization.l1 as f32)
            .min_gain(self.tree.min_gain_to_split as f32)
            .min_child_weight(self.regularization.min_hessian as f32)
            .min_samples_leaf(self.tree.min_samples_leaf)
            // Sampling params (flattened)
            .subsample(self.sampling.subsample as f32)
            .colsample_bytree(self.sampling.colsample as f32)
            .colsample_bylevel(self.sampling.colsample_bylevel as f32)
            // Other
            .maybe_linear_leaves(linear_leaves)
            .maybe_early_stopping_rounds(self.early_stopping_rounds)
            .seed(self.seed)
            .build()
            .map_err(|e| {
                BoostersError::InvalidParameter {
                    name: "config".to_string(),
                    reason: e.to_string(),
                }
                .into()
            })
    }
}
