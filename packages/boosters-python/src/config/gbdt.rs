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
///     objective: Loss function for training. Default: SquaredLoss().
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
///     ...     tree=TreeConfig(max_depth=6),
///     ... )
#[gen_stub_pyclass]
#[pyclass(name = "GBDTConfig", module = "boosters._boosters_rs")]
#[derive(Debug)]
pub struct PyGBDTConfig {
    /// Number of boosting rounds.
    #[pyo3(get)]
    pub n_estimators: u32,
    /// Learning rate (step size shrinkage).
    #[pyo3(get)]
    pub learning_rate: f64,
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

    // Store objective and metric as Python objects since the enum
    // doesn't implement IntoPy (it only implements FromPyObject for extraction)
    objective_obj: Py<PyAny>,
    metric_obj: Option<Py<PyAny>>,
}

impl Clone for PyGBDTConfig {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            objective_obj: self.objective_obj.clone_ref(py),
            metric_obj: self.metric_obj.as_ref().map(|m| m.clone_ref(py)),
            tree: self.tree.clone(),
            regularization: self.regularization.clone(),
            sampling: self.sampling.clone(),
            categorical: self.categorical.clone(),
            efb: self.efb.clone(),
            linear_leaves: self.linear_leaves.clone(),
            early_stopping_rounds: self.early_stopping_rounds,
            seed: self.seed,
        })
    }
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
        py: Python<'_>,
        n_estimators: u32,
        learning_rate: f64,
        #[gen_stub(override_type(type_repr = "SquaredLoss | AbsoluteLoss | PoissonLoss | LogisticLoss | HingeLoss | HuberLoss | PinballLoss | ArctanLoss | SoftmaxLoss | LambdaRankLoss | None"))]
        objective: Option<&Bound<'_, PyAny>>,
        #[gen_stub(override_type(type_repr = "Rmse | Mae | Mape | LogLoss | Auc | Accuracy | Ndcg | None"))]
        metric: Option<&Bound<'_, PyAny>>,
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

        // Handle objective - create default SquaredLoss if not provided
        let objective_obj = match objective {
            Some(obj) => {
                // Validate it's a valid objective by trying to extract
                let _: PyObjective = obj.extract()?;
                obj.clone().unbind()
            }
            None => {
                // Create default SquaredLoss
                let squared_loss = crate::objectives::PySquaredLoss::default();
                Py::new(py, squared_loss)?.into_any()
            }
        };

        // Handle metric
        let metric_obj = match metric {
            Some(m) => {
                // Validate it's a valid metric by trying to extract
                let _: PyMetric = m.extract()?;
                Some(m.clone().unbind())
            }
            None => None,
        };

        Ok(Self {
            n_estimators,
            learning_rate,
            objective_obj,
            metric_obj,
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

    /// Get the objective as a Python object.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "SquaredLoss | AbsoluteLoss | PoissonLoss | LogisticLoss | HingeLoss | HuberLoss | PinballLoss | ArctanLoss | SoftmaxLoss | LambdaRankLoss"))]
    fn objective(&self, py: Python<'_>) -> Py<PyAny> {
        self.objective_obj.clone_ref(py)
    }

    /// Get the metric as a Python object (or None).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "Rmse | Mae | Mape | LogLoss | Auc | Accuracy | Ndcg | None"))]
    fn metric(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.metric_obj.as_ref().map(|m| m.clone_ref(py))
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let obj_repr = self
            .objective_obj
            .bind(py)
            .repr()
            .map(|r| r.to_string())
            .unwrap_or_else(|_| "?".to_string());
        format!(
            "GBDTConfig(n_estimators={}, learning_rate={}, objective={})",
            self.n_estimators, self.learning_rate, obj_repr
        )
    }
}

impl Default for PyGBDTConfig {
    fn default() -> Self {
        // Note: This requires Python GIL. Use new() in Python context.
        Python::attach(|py| {
            let squared_loss = crate::objectives::PySquaredLoss::default();
            Self {
                n_estimators: 100,
                learning_rate: 0.3,
                objective_obj: Py::new(py, squared_loss).unwrap().into_any(),
                metric_obj: None,
                tree: PyTreeConfig::default(),
                regularization: PyRegularizationConfig::default(),
                sampling: PySamplingConfig::default(),
                categorical: PyCategoricalConfig::default(),
                efb: PyEFBConfig::default(),
                linear_leaves: None,
                early_stopping_rounds: None,
                seed: 42,
            }
        })
    }
}

impl PyGBDTConfig {
    /// Convert Python config to core Rust config.
    ///
    /// Called at fit-time to create the Rust training configuration.
    pub fn to_core(&self, py: Python<'_>) -> PyResult<boosters::GBDTConfig> {
        use boosters::model::gbdt::{RegularizationParams, SamplingParams, TreeParams};
        use boosters::training::gbdt::GrowthStrategy;

        // Convert objective
        let objective = self
            .objective_obj
            .extract::<PyObjective>(py)?
            .to_core();

        // Convert metric (optional)
        let metric = self
            .metric_obj
            .as_ref()
            .map(|m| m.extract::<PyMetric>(py))
            .transpose()?
            .map(|m| m.to_core());

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

        let tree = TreeParams {
            growth_strategy,
            max_onehot_cats: self.categorical.max_onehot as u32,
        };

        // Convert regularization config
        // PyRegularizationConfig.min_hessian -> min_child_weight
        // PyTreeConfig.min_samples_leaf -> min_samples_leaf
        // PyTreeConfig.min_gain_to_split -> min_gain
        let regularization = RegularizationParams {
            lambda: self.regularization.l2 as f32,
            alpha: self.regularization.l1 as f32,
            min_gain: self.tree.min_gain_to_split as f32,
            min_child_weight: self.regularization.min_hessian as f32,
            min_samples_leaf: self.tree.min_samples_leaf,
        };

        // Convert sampling config
        // PySamplingConfig.colsample -> colsample_bytree
        let sampling = SamplingParams {
            subsample: self.sampling.subsample as f32,
            colsample_bytree: self.sampling.colsample as f32,
            colsample_bylevel: self.sampling.colsample_bylevel as f32,
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

        // Build the core config using builder pattern
        // Note: bon builder uses type-state, so we need to chain all calls together
        // without reassignment. For optional fields, we pass the Option value directly.
        boosters::GBDTConfig::builder()
            .objective(objective)
            .maybe_metric(metric)
            .n_trees(self.n_estimators)
            .learning_rate(self.learning_rate as f32)
            .tree(tree)
            .regularization(regularization)
            .sampling(sampling)
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
