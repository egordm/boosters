//! GBLinear configuration for Python bindings.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::error::BoostersError;
use crate::metrics::PyMetric;
use crate::objectives::PyObjective;

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

        // Build core config
        boosters::GBLinearConfig::builder()
            .objective(objective)
            .maybe_metric(metric)
            .n_rounds(self.n_estimators)
            .learning_rate(self.learning_rate as f32)
            .alpha(self.l1 as f32)
            .lambda(self.l2 as f32)
            .maybe_early_stopping_rounds(self.early_stopping_rounds)
            .seed(self.seed)
            .build()
            .map_err(|e| BoostersError::ValidationError(e.to_string()).into())
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
