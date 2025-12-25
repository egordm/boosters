//! GBLinear configuration for Python bindings.

use pyo3::prelude::*;

use crate::error::BoostersError;
use crate::metrics::PyMetric;
use crate::objectives::PyObjective;

/// Main configuration for GBLinear model.
///
/// GBLinear uses gradient boosting to train a linear model via coordinate
/// descent. Simpler than GBDT but can be effective for linear relationships.
///
/// Parameters
/// ----------
/// n_estimators : int, default=100
///     Number of boosting rounds.
/// learning_rate : float, default=0.5
///     Step size for weight updates. Higher values mean faster convergence
///     but risk overshooting.
/// objective : Objective, default=SquaredLoss()
///     Loss function for training.
/// metric : Metric or None, default=None
///     Evaluation metric. If None, uses objective's default metric.
/// l1 : float, default=0.0
///     L1 regularization (alpha). Encourages sparse weights.
/// l2 : float, default=1.0
///     L2 regularization (lambda). Prevents large weights.
/// early_stopping_rounds : int or None, default=None
///     Stop if no improvement for this many rounds. None disables.
/// seed : int, default=42
///     Random seed for reproducibility.
///
/// Examples
/// --------
/// >>> from boosters import GBLinearConfig, SquaredLoss
/// >>> config = GBLinearConfig(
/// ...     n_estimators=200,
/// ...     learning_rate=0.3,
/// ...     objective=SquaredLoss(),
/// ...     l2=0.1,
/// ... )
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
    objective_obj: PyObject,
    metric_obj: Option<PyObject>,
}

impl Clone for PyGBLinearConfig {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
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
        objective: Option<&Bound<'_, PyAny>>,
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
    fn objective(&self, py: Python<'_>) -> PyObject {
        self.objective_obj.clone_ref(py)
    }

    /// Get the metric as a Python object (or None).
    #[getter]
    fn metric(&self, py: Python<'_>) -> Option<PyObject> {
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
        Python::with_gil(|py| {
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
