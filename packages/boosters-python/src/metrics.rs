//! Evaluation metrics for Python bindings.
//!
//! Uses PyO3 complex enums (0.22+) for a clean Rust-Python type mapping.
//! All variants use struct syntax (even empty ones) as required by PyO3.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;

use crate::error::BoostersError;

/// Evaluation metrics for gradient boosting.
///
/// Each variant represents a different metric for evaluating model performance.
/// Use the static constructor methods for validation.
///
/// Regression:
///     - Metric.Rmse(): Root Mean Squared Error
///     - Metric.Mae(): Mean Absolute Error
///     - Metric.Mape(): Mean Absolute Percentage Error
///
/// Classification:
///     - Metric.LogLoss(): Binary cross-entropy
///     - Metric.Auc(): Area Under ROC Curve
///     - Metric.Accuracy(): Classification accuracy
///
/// Ranking:
///     - Metric.Ndcg(at): Normalized Discounted Cumulative Gain@k
///
/// Examples
/// --------
/// >>> from boosters import Metric
/// >>> metric = Metric.rmse()  # Regression
/// >>> metric = Metric.auc()  # Binary classification
/// >>> metric = Metric.ndcg(at=5)  # Ranking
///
/// Pattern matching:
/// >>> match metric:
/// ...     case Metric.Rmse():
/// ...         print("RMSE")
/// ...     case Metric.Ndcg(at=k):
/// ...         print(f"NDCG@{k}")
#[gen_stub_pyclass_enum]
#[pyclass(name = "Metric", module = "boosters._boosters_rs", eq)]
#[derive(Clone, Debug, PartialEq)]
pub enum PyMetric {
    /// Root Mean Squared Error for regression.
    #[pyo3(constructor = ())]
    Rmse {},

    /// Mean Absolute Error for regression.
    #[pyo3(constructor = ())]
    Mae {},

    /// Mean Absolute Percentage Error for regression.
    #[pyo3(constructor = ())]
    Mape {},

    /// Binary Log Loss (cross-entropy) for classification.
    #[pyo3(constructor = ())]
    LogLoss {},

    /// Area Under ROC Curve for binary classification.
    #[pyo3(constructor = ())]
    Auc {},

    /// Classification accuracy (binary or multiclass).
    #[pyo3(constructor = ())]
    Accuracy {},

    /// Normalized Discounted Cumulative Gain for ranking.
    ///
    /// Parameters:
    ///     at: Truncation point for NDCG calculation (NDCG@k). Default: 10.
    #[pyo3(constructor = (at = 10))]
    Ndcg { at: u32 },
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMetric {
    // Static constructors for convenience and validation

    /// Create RMSE metric.
    #[staticmethod]
    fn rmse() -> Self {
        PyMetric::Rmse {}
    }

    /// Create MAE metric.
    #[staticmethod]
    fn mae() -> Self {
        PyMetric::Mae {}
    }

    /// Create MAPE metric.
    #[staticmethod]
    fn mape() -> Self {
        PyMetric::Mape {}
    }

    /// Create log loss metric.
    #[staticmethod]
    fn logloss() -> Self {
        PyMetric::LogLoss {}
    }

    /// Create AUC metric.
    #[staticmethod]
    fn auc() -> Self {
        PyMetric::Auc {}
    }

    /// Create accuracy metric.
    #[staticmethod]
    fn accuracy() -> Self {
        PyMetric::Accuracy {}
    }

    /// Create NDCG@k metric with validation.
    #[staticmethod]
    #[pyo3(signature = (at = 10))]
    fn ndcg(at: u32) -> PyResult<Self> {
        if at == 0 {
            return Err(BoostersError::InvalidParameter {
                name: "at".to_string(),
                reason: "must be positive".to_string(),
            }
            .into());
        }
        Ok(PyMetric::Ndcg { at })
    }

    fn __repr__(&self) -> String {
        match self {
            PyMetric::Rmse {} => "Metric.Rmse()".to_string(),
            PyMetric::Mae {} => "Metric.Mae()".to_string(),
            PyMetric::Mape {} => "Metric.Mape()".to_string(),
            PyMetric::LogLoss {} => "Metric.LogLoss()".to_string(),
            PyMetric::Auc {} => "Metric.Auc()".to_string(),
            PyMetric::Accuracy {} => "Metric.Accuracy()".to_string(),
            PyMetric::Ndcg { at } => format!("Metric.Ndcg(at={})", at),
        }
    }
}

impl Default for PyMetric {
    fn default() -> Self {
        PyMetric::Rmse {}
    }
}

impl From<&PyMetric> for boosters::training::Metric {
    fn from(py_metric: &PyMetric) -> Self {
        use boosters::training::Metric;

        match py_metric {
            PyMetric::Rmse {} => Metric::rmse(),
            PyMetric::Mae {} => Metric::mae(),
            PyMetric::Mape {} => Metric::mape(),
            PyMetric::LogLoss {} => Metric::logloss(),
            PyMetric::Auc {} => Metric::auc(),
            PyMetric::Accuracy {} => Metric::accuracy(),
            PyMetric::Ndcg { .. } => {
                // NDCG not yet implemented in core - use rmse as placeholder
                // TODO: Add NDCG to core metric enum
                Metric::rmse()
            }
        }
    }
}

impl From<PyMetric> for boosters::training::Metric {
    fn from(py_metric: PyMetric) -> Self {
        (&py_metric).into()
    }
}

impl From<&boosters::training::Metric> for PyMetric {
    fn from(metric: &boosters::training::Metric) -> Self {
        use boosters::training::Metric;

        match metric {
            Metric::None => PyMetric::Rmse {}, // Default fallback
            Metric::Rmse(_) => PyMetric::Rmse {},
            Metric::Mae(_) => PyMetric::Mae {},
            Metric::Mape(_) => PyMetric::Mape {},
            Metric::LogLoss(_) => PyMetric::LogLoss {},
            Metric::Auc(_) => PyMetric::Auc {},
            Metric::Accuracy(_) | Metric::MarginAccuracy(_) => PyMetric::Accuracy {},
            Metric::MulticlassLogLoss(_) => PyMetric::LogLoss {},
            Metric::MulticlassAccuracy(_) => PyMetric::Accuracy {},
            Metric::Quantile(_) => PyMetric::Mae {}, // Best approximation
            Metric::Huber(_) => PyMetric::Mae {},    // Best approximation
            Metric::PoissonDeviance(_) => PyMetric::Rmse {}, // Best approximation
            Metric::Custom(_) => PyMetric::Rmse {},  // Custom can't be round-tripped
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_conversions() {
        use boosters::training::Metric;

        let metric: Metric = (&PyMetric::Rmse {}).into();
        assert!(matches!(metric, Metric::Rmse(_)));

        let metric: Metric = (&PyMetric::Auc {}).into();
        assert!(matches!(metric, Metric::Auc(_)));
    }
}
