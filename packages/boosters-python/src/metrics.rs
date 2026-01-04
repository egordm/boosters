//! Evaluation metrics for Python bindings.
//!
//! Uses PyO3 complex enums (0.22+) for a clean Rust-Python type mapping.
//! All variants use struct syntax (even empty ones) as required by PyO3.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;

use crate::error::BoostersError;
use crate::validation::validate_ratio;

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
/// Note: ranking metrics are not implemented in core yet.
///
/// Examples
/// --------
/// >>> from boosters import Metric
/// >>> metric = Metric.rmse()  # Regression
/// >>> metric = Metric.auc()  # Binary classification
/// >>> metric = Metric.Accuracy(threshold=0.7)
///
/// Pattern matching:
/// >>> match metric:
/// ...     case Metric.Rmse():
/// ...         print("RMSE")
/// ...     case Metric.Accuracy(threshold=t):
/// ...         print(f"Accuracy@{t}")
#[gen_stub_pyclass_enum]
#[pyclass(name = "Metric", module = "boosters._boosters_rs", eq)]
#[derive(Clone, Debug, PartialEq)]
pub enum PyMetric {
    /// No metric - skips evaluation entirely.
    #[pyo3(constructor = ( ))]
    None {},

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
    ///
    /// Parameters:
    ///     threshold: Probability threshold in (0, 1] for positive class. Default: 0.5.
    #[pyo3(constructor = (threshold = 0.5))]
    Accuracy { threshold: f32 },

    /// Accuracy computed on raw margins (no sigmoid/softmax).
    #[pyo3(constructor = ())]
    MarginAccuracy {},

    /// Multiclass log loss (cross-entropy).
    #[pyo3(constructor = ())]
    MulticlassLogLoss {},

    /// Multiclass accuracy.
    #[pyo3(constructor = ())]
    MulticlassAccuracy {},

    /// Quantile loss metric.
    ///
    /// Parameters:
    ///     alpha: List of quantiles to evaluate. Each value must be in (0, 1).
    Quantile { alpha: Vec<f32> },

    /// Poisson deviance for count regression.
    #[pyo3(constructor = ())]
    PoissonDeviance {},
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMetric {
    /// Create no-metric.
    #[staticmethod]
    fn none() -> Self {
        PyMetric::None {}
    }

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

    /// Create accuracy metric with validation.
    ///
    /// Use `Metric.Accuracy()` for default threshold, `Metric.Accuracy(threshold=...)`
    /// for direct construction, or this helper for validation.
    #[staticmethod]
    #[pyo3(signature = (threshold = 0.5))]
    fn accuracy_at(threshold: f32) -> PyResult<Self> {
        validate_ratio("threshold", threshold)?;
        Ok(PyMetric::Accuracy { threshold })
    }

    /// Create margin accuracy metric.
    #[staticmethod]
    fn margin_accuracy() -> Self {
        PyMetric::MarginAccuracy {}
    }

    /// Create multiclass log loss metric.
    #[staticmethod]
    fn multiclass_logloss() -> Self {
        PyMetric::MulticlassLogLoss {}
    }

    /// Create multiclass accuracy metric.
    #[staticmethod]
    fn multiclass_accuracy() -> Self {
        PyMetric::MulticlassAccuracy {}
    }

    /// Create quantile metric with validation.
    #[staticmethod]
    fn quantile(alpha: Vec<f32>) -> PyResult<Self> {
        if alpha.is_empty() {
            return Err(BoostersError::InvalidParameter {
                name: "alpha".to_string(),
                reason: "must have at least one quantile".to_string(),
            }
            .into());
        }
        for (i, &a) in alpha.iter().enumerate() {
            crate::validation::validate_ratio_open(&format!("alpha[{}]", i), a)?;
        }
        Ok(PyMetric::Quantile { alpha })
    }

    /// Create Poisson deviance metric.
    #[staticmethod]
    fn poisson_deviance() -> Self {
        PyMetric::PoissonDeviance {}
    }

    fn __repr__(&self) -> String {
        match self {
            PyMetric::None {} => "Metric.None()".to_string(),
            PyMetric::Rmse {} => "Metric.Rmse()".to_string(),
            PyMetric::Mae {} => "Metric.Mae()".to_string(),
            PyMetric::Mape {} => "Metric.Mape()".to_string(),
            PyMetric::LogLoss {} => "Metric.LogLoss()".to_string(),
            PyMetric::Auc {} => "Metric.Auc()".to_string(),
            PyMetric::Accuracy { threshold } => {
                format!("Metric.Accuracy(threshold={})", threshold)
            }
            PyMetric::MarginAccuracy {} => "Metric.MarginAccuracy()".to_string(),
            PyMetric::MulticlassLogLoss {} => "Metric.MulticlassLogLoss()".to_string(),
            PyMetric::MulticlassAccuracy {} => "Metric.MulticlassAccuracy()".to_string(),
            PyMetric::Quantile { alpha } => format!("Metric.Quantile(alpha={:?})", alpha),
            PyMetric::PoissonDeviance {} => "Metric.PoissonDeviance()".to_string(),
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
            PyMetric::None {} => Metric::None,
            PyMetric::Rmse {} => Metric::Rmse,
            PyMetric::Mae {} => Metric::Mae,
            PyMetric::Mape {} => Metric::Mape,
            PyMetric::LogLoss {} => Metric::LogLoss,
            PyMetric::Auc {} => Metric::Auc,
            PyMetric::Accuracy { threshold } => Metric::Accuracy {
                threshold: *threshold,
            },
            PyMetric::MarginAccuracy {} => Metric::MarginAccuracy,
            PyMetric::MulticlassLogLoss {} => Metric::MulticlassLogLoss,
            PyMetric::MulticlassAccuracy {} => Metric::MulticlassAccuracy,
            PyMetric::Quantile { alpha } => Metric::Quantile {
                alphas: alpha.clone(),
            },
            PyMetric::PoissonDeviance {} => Metric::PoissonDeviance,
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
            Metric::None => PyMetric::None {},
            Metric::Rmse => PyMetric::Rmse {},
            Metric::Mae => PyMetric::Mae {},
            Metric::Mape => PyMetric::Mape {},
            Metric::LogLoss => PyMetric::LogLoss {},
            Metric::Auc => PyMetric::Auc {},
            Metric::Accuracy { threshold } => PyMetric::Accuracy {
                threshold: *threshold,
            },
            Metric::MarginAccuracy => PyMetric::MarginAccuracy {},
            Metric::MulticlassLogLoss => PyMetric::MulticlassLogLoss {},
            Metric::MulticlassAccuracy => PyMetric::MulticlassAccuracy {},
            Metric::Quantile { alphas } => PyMetric::Quantile {
                alpha: alphas.clone(),
            },
            Metric::Huber { .. } => PyMetric::Mae {}, // Not representable in PyMetric yet
            Metric::PoissonDeviance => PyMetric::PoissonDeviance {},
            Metric::Custom(_) => PyMetric::Rmse {}, // Custom can't be round-tripped
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
        assert!(matches!(metric, Metric::Rmse));

        let metric: Metric = (&PyMetric::Auc {}).into();
        assert!(matches!(metric, Metric::Auc));
    }
}
