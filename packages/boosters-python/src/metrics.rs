//! Evaluation metrics for Python bindings.
//!
//! Each metric is a separate `#[pyclass]` type. The `PyMetric` enum
//! wraps them all for type-safe handling in GBDTConfig/GBLinearConfig.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::error::BoostersError;

// =============================================================================
// Parameterless Metrics
// =============================================================================

/// Root Mean Squared Error for regression.
///
/// Examples
/// --------
/// >>> from boosters import Rmse
/// >>> metric = Rmse()
#[gen_stub_pyclass]
#[pyclass(name = "Rmse", module = "boosters._boosters_rs")]
#[derive(Clone, Debug, Default)]
pub struct PyRmse;

#[gen_stub_pymethods]
#[pymethods]
impl PyRmse {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> &'static str {
        "Rmse()"
    }
}

/// Mean Absolute Error for regression.
///
/// Examples
/// --------
/// >>> from boosters import Mae
/// >>> metric = Mae()
#[gen_stub_pyclass]
#[pyclass(name = "Mae", module = "boosters._boosters_rs")]
#[derive(Clone, Debug, Default)]
pub struct PyMae;

#[gen_stub_pymethods]
#[pymethods]
impl PyMae {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> &'static str {
        "Mae()"
    }
}

/// Mean Absolute Percentage Error for regression.
///
/// Examples
/// --------
/// >>> from boosters import Mape
/// >>> metric = Mape()
#[gen_stub_pyclass]
#[pyclass(name = "Mape", module = "boosters._boosters_rs")]
#[derive(Clone, Debug, Default)]
pub struct PyMape;

#[gen_stub_pymethods]
#[pymethods]
impl PyMape {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> &'static str {
        "Mape()"
    }
}

/// Binary Log Loss (cross-entropy) for classification.
///
/// Examples
/// --------
/// >>> from boosters import LogLoss
/// >>> metric = LogLoss()
#[gen_stub_pyclass]
#[pyclass(name = "LogLoss", module = "boosters._boosters_rs")]
#[derive(Clone, Debug, Default)]
pub struct PyLogLoss;

#[gen_stub_pymethods]
#[pymethods]
impl PyLogLoss {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> &'static str {
        "LogLoss()"
    }
}

/// Area Under ROC Curve for binary classification.
///
/// Examples
/// --------
/// >>> from boosters import Auc
/// >>> metric = Auc()
#[gen_stub_pyclass]
#[pyclass(name = "Auc", module = "boosters._boosters_rs")]
#[derive(Clone, Debug, Default)]
pub struct PyAuc;

#[gen_stub_pymethods]
#[pymethods]
impl PyAuc {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> &'static str {
        "Auc()"
    }
}

/// Classification accuracy (binary or multiclass).
///
/// Examples
/// --------
/// >>> from boosters import Accuracy
/// >>> metric = Accuracy()
#[gen_stub_pyclass]
#[pyclass(name = "Accuracy", module = "boosters._boosters_rs")]
#[derive(Clone, Debug, Default)]
pub struct PyAccuracy;

#[gen_stub_pymethods]
#[pymethods]
impl PyAccuracy {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> &'static str {
        "Accuracy()"
    }
}

// =============================================================================
// Parameterized Metrics
// =============================================================================

/// Normalized Discounted Cumulative Gain for ranking.
///
/// Parameters
/// ----------
/// at : int, default=10
///     Truncation point for NDCG calculation (NDCG@k).
///
/// Examples
/// --------
/// >>> from boosters import Ndcg
/// >>> metric = Ndcg(at=5)  # NDCG@5
#[gen_stub_pyclass]
#[pyclass(name = "Ndcg", module = "boosters._boosters_rs", get_all)]
#[derive(Clone, Debug)]
pub struct PyNdcg {
    pub at: u32,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyNdcg {
    #[new]
    #[pyo3(signature = (at = 10))]
    fn new(at: u32) -> PyResult<Self> {
        if at == 0 {
            return Err(BoostersError::InvalidParameter {
                name: "at".to_string(),
                reason: "must be positive".to_string(),
            }
            .into());
        }
        Ok(Self { at })
    }

    fn __repr__(&self) -> String {
        format!("Ndcg(at={})", self.at)
    }
}

impl Default for PyNdcg {
    fn default() -> Self {
        Self { at: 10 }
    }
}

// =============================================================================
// Metric Enum for Type Safety
// =============================================================================

/// Enum wrapper for all metric types.
///
/// This allows accepting any metric type in GBDTConfig.
#[derive(Debug, Clone, FromPyObject)]
pub enum PyMetric {
    Rmse(PyRmse),
    Mae(PyMae),
    Mape(PyMape),
    LogLoss(PyLogLoss),
    Auc(PyAuc),
    Accuracy(PyAccuracy),
    Ndcg(PyNdcg),
}
