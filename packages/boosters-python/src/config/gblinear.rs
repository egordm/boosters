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
///     objective: Loss function for training. Default: SquaredLoss().
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
///     ...     l2=0.1,
///     ... )
#[gen_stub_pyclass]
#[pyclass(name = "GBLinearConfig", module = "boosters._boosters_rs")]
#[derive(Debug)]
pub struct PyGBLinearConfig {
    /// Number of boosting rounds.
    #[pyo3(get)]
    pub n_estimators: u32,
    /// Learning rate (step size).
    #[pyo3(get)]
    pub learning_rate: f64,
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

    // Store objective and metric as Python objects
    objective_obj: Py<PyAny>,
    metric_obj: Option<Py<PyAny>>,
}

impl Clone for PyGBLinearConfig {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            objective_obj: self.objective_obj.clone_ref(py),
            metric_obj: self.metric_obj.as_ref().map(|m| m.clone_ref(py)),
            l1: self.l1,
            l2: self.l2,
            early_stopping_rounds: self.early_stopping_rounds,
            seed: self.seed,
        })
    }
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
        py: Python<'_>,
        n_estimators: u32,
        learning_rate: f64,
        #[gen_stub(override_type(type_repr = "SquaredLoss | AbsoluteLoss | PoissonLoss | LogisticLoss | HingeLoss | HuberLoss | PinballLoss | ArctanLoss | SoftmaxLoss | LambdaRankLoss | None"))]
        objective: Option<&Bound<'_, PyAny>>,
        #[gen_stub(override_type(type_repr = "Rmse | Mae | Mape | LogLoss | Auc | Accuracy | Ndcg | None"))]
        metric: Option<&Bound<'_, PyAny>>,
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
            l1,
            l2,
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
            "GBLinearConfig(n_estimators={}, learning_rate={}, l1={}, l2={}, objective={})",
            self.n_estimators, self.learning_rate, self.l1, self.l2, obj_repr
        )
    }
}

impl PyGBLinearConfig {
    /// Convert to core GBLinearConfig for training.
    pub fn to_core(&self, py: Python<'_>) -> PyResult<boosters::GBLinearConfig> {
        // Extract objective
        let py_objective: PyObjective = self.objective_obj.extract(py)?;
        let objective = py_objective.to_core();

        // Extract metric if present
        let metric: Option<boosters::training::Metric> = match &self.metric_obj {
            Some(m) => {
                let py_metric: PyMetric = m.extract(py)?;
                Some(py_metric.to_core())
            }
            None => None,
        };

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
        Python::attach(|py| {
            let squared_loss = crate::objectives::PySquaredLoss::default();
            Self {
                n_estimators: 100,
                learning_rate: 0.5,
                objective_obj: Py::new(py, squared_loss).unwrap().into_any(),
                metric_obj: None,
                l1: 0.0,
                l2: 1.0,
                early_stopping_rounds: None,
                seed: 42,
            }
        })
    }
}
