//! Objective (loss) functions for Python bindings.
//!
//! Uses PyO3 complex enums (0.22+) for a clean Rust-Python type mapping.
//! All variants use struct syntax (even empty ones) as required by PyO3.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;

use crate::error::BoostersError;
use crate::validation::{validate_positive, validate_ratio_open};

/// Objective (loss) functions for gradient boosting.
///
/// Each variant represents a different loss function for training GBDT and
/// GBLinear models. Use the static constructor methods for validation.
///
/// Regression:
///     - Objective.Squared(): Mean squared error (L2)
///     - Objective.Absolute(): Mean absolute error (L1)
///     - Objective.Huber(delta): Pseudo-Huber loss (robust)
///     - Objective.Pinball(alpha): Quantile regression
///     - Objective.Poisson(): Poisson deviance for count data
///
/// Classification:
///     - Objective.Logistic(): Binary cross-entropy
///     - Objective.Hinge(): SVM-style hinge loss
///     - Objective.Softmax(n_classes): Multiclass cross-entropy
///
/// Note: ranking objectives are not implemented in core yet.
///
/// Examples
/// --------
/// >>> from boosters import Objective
/// >>> obj = Objective.squared()  # L2 regression
/// >>> obj = Objective.logistic()  # Binary classification
/// >>> obj = Objective.pinball([0.1, 0.5, 0.9])  # Quantile regression
/// >>> obj = Objective.softmax(10)  # Multiclass classification
///
/// Pattern matching:
/// >>> match obj:
/// ...     case Objective.Squared():
/// ...         print("L2 loss")
/// ...     case Objective.Pinball(alpha=a):
/// ...         print(f"Quantile: {a}")
#[gen_stub_pyclass_enum]
#[pyclass(name = "Objective", module = "boosters._boosters_rs", eq)]
#[derive(Clone, Debug, PartialEq)]
pub enum PyObjective {
    /// Squared error loss (L2) for regression.
    #[pyo3(constructor = ())]
    Squared {},

    /// Absolute error loss (L1) for robust regression.
    #[pyo3(constructor = ())]
    Absolute {},

    /// Poisson loss for count regression.
    #[pyo3(constructor = ())]
    Poisson {},

    /// Logistic loss for binary classification.
    #[pyo3(constructor = ())]
    Logistic {},

    /// Hinge loss for binary classification (SVM-style).
    #[pyo3(constructor = ())]
    Hinge {},

    /// Pseudo-Huber loss for robust regression.
    ///
    /// Parameters:
    ///     delta: Transition point between quadratic and linear loss. Default: 1.0.
    #[pyo3(constructor = (delta = 1.0))]
    Huber { delta: f32 },

    /// Pinball loss for quantile regression.
    ///
    /// Parameters:
    ///     alpha: List of quantiles to predict. Each value must be in (0, 1).
    Pinball { alpha: Vec<f32> },

    /// Softmax loss for multiclass classification.
    ///
    /// Parameters:
    ///     n_classes: Number of classes. Must be >= 2.
    Softmax { n_classes: u32 },
}

#[gen_stub_pymethods]
#[pymethods]
impl PyObjective {
    // Static constructors for convenience and validation

    /// Create squared error loss (L2).
    #[staticmethod]
    fn squared() -> Self {
        PyObjective::Squared {}
    }

    /// Create absolute error loss (L1).
    #[staticmethod]
    fn absolute() -> Self {
        PyObjective::Absolute {}
    }

    /// Create Poisson loss.
    #[staticmethod]
    fn poisson() -> Self {
        PyObjective::Poisson {}
    }

    /// Create logistic loss for binary classification.
    #[staticmethod]
    fn logistic() -> Self {
        PyObjective::Logistic {}
    }

    /// Create hinge loss for binary classification.
    #[staticmethod]
    fn hinge() -> Self {
        PyObjective::Hinge {}
    }

    /// Create Huber loss with validation.
    #[staticmethod]
    #[pyo3(signature = (delta = 1.0))]
    fn huber(delta: f32) -> PyResult<Self> {
        validate_positive("delta", delta)?;
        Ok(PyObjective::Huber { delta })
    }

    /// Create pinball loss with validation.
    #[staticmethod]
    fn pinball(alpha: Vec<f32>) -> PyResult<Self> {
        if alpha.is_empty() {
            return Err(BoostersError::InvalidParameter {
                name: "alpha".to_string(),
                reason: "must have at least one quantile".to_string(),
            }
            .into());
        }
        for (i, &a) in alpha.iter().enumerate() {
            validate_ratio_open(&format!("alpha[{}]", i), a)?;
        }
        Ok(PyObjective::Pinball { alpha })
    }

    /// Create softmax loss with validation.
    #[staticmethod]
    fn softmax(n_classes: u32) -> PyResult<Self> {
        validate_positive("n_classes", n_classes)?;
        if n_classes < 2 {
            return Err(BoostersError::InvalidParameter {
                name: "n_classes".to_string(),
                reason: "must be >= 2".to_string(),
            }
            .into());
        }
        Ok(PyObjective::Softmax { n_classes })
    }

    fn __repr__(&self) -> String {
        match self {
            PyObjective::Squared {} => "Objective.Squared()".to_string(),
            PyObjective::Absolute {} => "Objective.Absolute()".to_string(),
            PyObjective::Poisson {} => "Objective.Poisson()".to_string(),
            PyObjective::Logistic {} => "Objective.Logistic()".to_string(),
            PyObjective::Hinge {} => "Objective.Hinge()".to_string(),
            PyObjective::Huber { delta } => format!("Objective.Huber(delta={})", delta),
            PyObjective::Pinball { alpha } => format!("Objective.Pinball(alpha={:?})", alpha),
            PyObjective::Softmax { n_classes } => {
                format!("Objective.Softmax(n_classes={})", n_classes)
            }
        }
    }
}

impl Default for PyObjective {
    fn default() -> Self {
        PyObjective::Squared {}
    }
}

impl From<&PyObjective> for boosters::training::Objective {
    fn from(py_obj: &PyObjective) -> Self {
        use boosters::training::Objective;

        match py_obj {
            PyObjective::Squared {} => Objective::SquaredLoss,
            PyObjective::Absolute {} => Objective::AbsoluteLoss,
            PyObjective::Poisson {} => Objective::PoissonLoss,
            PyObjective::Logistic {} => Objective::LogisticLoss,
            PyObjective::Hinge {} => Objective::HingeLoss,
            PyObjective::Huber { delta } => Objective::PseudoHuberLoss { delta: *delta },
            PyObjective::Pinball { alpha } => Objective::PinballLoss {
                alphas: alpha.clone(),
            },
            PyObjective::Softmax { n_classes } => Objective::SoftmaxLoss {
                n_classes: *n_classes as usize,
            },
        }
    }
}

impl From<PyObjective> for boosters::training::Objective {
    fn from(py_obj: PyObjective) -> Self {
        (&py_obj).into()
    }
}

impl From<&boosters::training::Objective> for PyObjective {
    fn from(obj: &boosters::training::Objective) -> Self {
        use boosters::training::Objective;

        match obj {
            Objective::SquaredLoss => PyObjective::Squared {},
            Objective::AbsoluteLoss => PyObjective::Absolute {},
            Objective::PoissonLoss => PyObjective::Poisson {},
            Objective::LogisticLoss => PyObjective::Logistic {},
            Objective::HingeLoss => PyObjective::Hinge {},
            Objective::PseudoHuberLoss { delta } => PyObjective::Huber { delta: *delta },
            Objective::PinballLoss { alphas } => PyObjective::Pinball {
                alpha: alphas.clone(),
            },
            Objective::SoftmaxLoss { n_classes } => PyObjective::Softmax {
                n_classes: *n_classes as u32,
            },
            Objective::Custom(_) => {
                // Custom objectives can't be round-tripped
                // Default to squared loss
                PyObjective::Squared {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_objective_conversions() {
        use boosters::training::Objective;

        // Test parameterless objectives
        let obj: Objective = (&PyObjective::Squared {}).into();
        assert!(matches!(obj, Objective::SquaredLoss));

        let obj: Objective = (&PyObjective::Logistic {}).into();
        assert!(matches!(obj, Objective::LogisticLoss));
    }

    #[test]
    fn test_pinball_single_alpha() {
        use boosters::training::Objective;

        let py_objective = PyObjective::Pinball { alpha: vec![0.5] };
        let core: Objective = (&py_objective).into();
        assert!(matches!(core, Objective::PinballLoss { .. }));
    }

    #[test]
    fn test_pinball_multi_alpha() {
        use boosters::training::Objective;

        let py_objective = PyObjective::Pinball {
            alpha: vec![0.1, 0.5, 0.9],
        };
        let core: Objective = (&py_objective).into();
        assert!(matches!(core, Objective::PinballLoss { .. }));
    }

    #[test]
    fn test_softmax_conversion() {
        use boosters::training::Objective;

        let py_objective = PyObjective::Softmax { n_classes: 5 };
        let core: Objective = (&py_objective).into();
        assert!(matches!(core, Objective::SoftmaxLoss { .. }));
    }

    #[test]
    fn test_huber_conversion() {
        use boosters::training::Objective;

        let py_objective = PyObjective::Huber { delta: 2.0 };
        let core: Objective = (&py_objective).into();
        assert!(matches!(core, Objective::PseudoHuberLoss { .. }));
    }
}
