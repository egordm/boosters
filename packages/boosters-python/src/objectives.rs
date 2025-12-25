//! Objective (loss) functions for Python bindings.
//!
//! Each objective is a separate `#[pyclass]` type. The `PyObjective` enum
//! wraps them all for type-safe handling in GBDTConfig/GBLinearConfig.

use pyo3::prelude::*;

use crate::error::BoostersError;

// =============================================================================
// Parameterless Objectives
// =============================================================================

/// Squared error loss (L2) for regression.
///
/// Examples
/// --------
/// >>> from boosters import SquaredLoss
/// >>> obj = SquaredLoss()
#[pyclass(name = "SquaredLoss", module = "boosters._boosters_rs")]
#[derive(Clone, Debug, Default)]
pub struct PySquaredLoss;

#[pymethods]
impl PySquaredLoss {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> &'static str {
        "SquaredLoss()"
    }
}

/// Absolute error loss (L1) for robust regression.
///
/// Examples
/// --------
/// >>> from boosters import AbsoluteLoss
/// >>> obj = AbsoluteLoss()
#[pyclass(name = "AbsoluteLoss", module = "boosters._boosters_rs")]
#[derive(Clone, Debug, Default)]
pub struct PyAbsoluteLoss;

#[pymethods]
impl PyAbsoluteLoss {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> &'static str {
        "AbsoluteLoss()"
    }
}

/// Poisson loss for count regression.
///
/// Examples
/// --------
/// >>> from boosters import PoissonLoss
/// >>> obj = PoissonLoss()
#[pyclass(name = "PoissonLoss", module = "boosters._boosters_rs")]
#[derive(Clone, Debug, Default)]
pub struct PyPoissonLoss;

#[pymethods]
impl PyPoissonLoss {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> &'static str {
        "PoissonLoss()"
    }
}

/// Logistic loss for binary classification.
///
/// Examples
/// --------
/// >>> from boosters import LogisticLoss
/// >>> obj = LogisticLoss()
#[pyclass(name = "LogisticLoss", module = "boosters._boosters_rs")]
#[derive(Clone, Debug, Default)]
pub struct PyLogisticLoss;

#[pymethods]
impl PyLogisticLoss {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> &'static str {
        "LogisticLoss()"
    }
}

/// Hinge loss for binary classification (SVM-style).
///
/// Examples
/// --------
/// >>> from boosters import HingeLoss
/// >>> obj = HingeLoss()
#[pyclass(name = "HingeLoss", module = "boosters._boosters_rs")]
#[derive(Clone, Debug, Default)]
pub struct PyHingeLoss;

#[pymethods]
impl PyHingeLoss {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> &'static str {
        "HingeLoss()"
    }
}

// =============================================================================
// Parameterized Objectives
// =============================================================================

/// Pseudo-Huber loss for robust regression.
///
/// Parameters
/// ----------
/// delta : float, default=1.0
///     Transition point between quadratic and linear loss.
///
/// Examples
/// --------
/// >>> from boosters import HuberLoss
/// >>> obj = HuberLoss(delta=1.5)
#[pyclass(name = "HuberLoss", module = "boosters._boosters_rs", get_all)]
#[derive(Clone, Debug)]
pub struct PyHuberLoss {
    pub delta: f64,
}

#[pymethods]
impl PyHuberLoss {
    #[new]
    #[pyo3(signature = (delta = 1.0))]
    fn new(delta: f64) -> PyResult<Self> {
        if delta <= 0.0 {
            return Err(BoostersError::InvalidParameter {
                name: "delta".to_string(),
                reason: "must be positive".to_string(),
            }
            .into());
        }
        Ok(Self { delta })
    }

    fn __repr__(&self) -> String {
        format!("HuberLoss(delta={})", self.delta)
    }
}

impl Default for PyHuberLoss {
    fn default() -> Self {
        Self { delta: 1.0 }
    }
}

/// Pinball loss for quantile regression.
///
/// Parameters
/// ----------
/// alpha : float or list of float, default=0.5
///     Quantile(s) to predict. Each value must be in (0, 1).
///
/// Examples
/// --------
/// >>> from boosters import PinballLoss
/// >>> obj = PinballLoss(alpha=0.5)  # median
/// >>> obj = PinballLoss(alpha=[0.1, 0.5, 0.9])  # multiple quantiles
#[pyclass(name = "PinballLoss", module = "boosters._boosters_rs", get_all)]
#[derive(Clone, Debug)]
pub struct PyPinballLoss {
    /// Quantile values (always stored as Vec for consistency).
    pub alpha: Vec<f64>,
}

#[pymethods]
impl PyPinballLoss {
    #[new]
    #[pyo3(signature = (alpha = None))]
    fn new(alpha: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let alpha_vec = match alpha {
            None => vec![0.5],
            Some(obj) => {
                // Try to extract as a single float first
                if let Ok(val) = obj.extract::<f64>() {
                    vec![val]
                } else if let Ok(vals) = obj.extract::<Vec<f64>>() {
                    vals
                } else {
                    return Err(BoostersError::TypeError(
                        "alpha must be a float or list of floats".to_string(),
                    )
                    .into());
                }
            }
        };

        // Validate all alpha values
        for (i, &a) in alpha_vec.iter().enumerate() {
            if a <= 0.0 || a >= 1.0 {
                return Err(BoostersError::InvalidParameter {
                    name: format!("alpha[{}]", i),
                    reason: format!("must be in (0, 1), got {}", a),
                }
                .into());
            }
        }

        Ok(Self { alpha: alpha_vec })
    }

    fn __repr__(&self) -> String {
        if self.alpha.len() == 1 {
            format!("PinballLoss(alpha={})", self.alpha[0])
        } else {
            format!("PinballLoss(alpha={:?})", self.alpha)
        }
    }
}

impl Default for PyPinballLoss {
    fn default() -> Self {
        Self { alpha: vec![0.5] }
    }
}

/// Arctan loss for bounded regression.
///
/// Parameters
/// ----------
/// alpha : float, default=0.5
///     Scale parameter. Must be in (0, 1).
///
/// Examples
/// --------
/// >>> from boosters import ArctanLoss
/// >>> obj = ArctanLoss(alpha=0.3)
#[pyclass(name = "ArctanLoss", module = "boosters._boosters_rs", get_all)]
#[derive(Clone, Debug)]
pub struct PyArctanLoss {
    pub alpha: f64,
}

#[pymethods]
impl PyArctanLoss {
    #[new]
    #[pyo3(signature = (alpha = 0.5))]
    fn new(alpha: f64) -> PyResult<Self> {
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(BoostersError::InvalidParameter {
                name: "alpha".to_string(),
                reason: "must be in (0, 1)".to_string(),
            }
            .into());
        }
        Ok(Self { alpha })
    }

    fn __repr__(&self) -> String {
        format!("ArctanLoss(alpha={})", self.alpha)
    }
}

impl Default for PyArctanLoss {
    fn default() -> Self {
        Self { alpha: 0.5 }
    }
}

/// Softmax loss for multiclass classification.
///
/// Parameters
/// ----------
/// n_classes : int
///     Number of classes. Must be >= 2.
///
/// Examples
/// --------
/// >>> from boosters import SoftmaxLoss
/// >>> obj = SoftmaxLoss(n_classes=10)
#[pyclass(name = "SoftmaxLoss", module = "boosters._boosters_rs", get_all)]
#[derive(Clone, Debug)]
pub struct PySoftmaxLoss {
    pub n_classes: u32,
}

#[pymethods]
impl PySoftmaxLoss {
    #[new]
    fn new(n_classes: u32) -> PyResult<Self> {
        if n_classes < 2 {
            return Err(BoostersError::InvalidParameter {
                name: "n_classes".to_string(),
                reason: "must be >= 2".to_string(),
            }
            .into());
        }
        Ok(Self { n_classes })
    }

    fn __repr__(&self) -> String {
        format!("SoftmaxLoss(n_classes={})", self.n_classes)
    }
}

/// LambdaRank loss for learning to rank.
///
/// Parameters
/// ----------
/// ndcg_at : int, default=10
///     Truncation point for NDCG calculation.
///
/// Examples
/// --------
/// >>> from boosters import LambdaRankLoss
/// >>> obj = LambdaRankLoss(ndcg_at=5)
#[pyclass(name = "LambdaRankLoss", module = "boosters._boosters_rs", get_all)]
#[derive(Clone, Debug)]
pub struct PyLambdaRankLoss {
    pub ndcg_at: u32,
}

#[pymethods]
impl PyLambdaRankLoss {
    #[new]
    #[pyo3(signature = (ndcg_at = 10))]
    fn new(ndcg_at: u32) -> PyResult<Self> {
        if ndcg_at == 0 {
            return Err(BoostersError::InvalidParameter {
                name: "ndcg_at".to_string(),
                reason: "must be positive".to_string(),
            }
            .into());
        }
        Ok(Self { ndcg_at })
    }

    fn __repr__(&self) -> String {
        format!("LambdaRankLoss(ndcg_at={})", self.ndcg_at)
    }
}

impl Default for PyLambdaRankLoss {
    fn default() -> Self {
        Self { ndcg_at: 10 }
    }
}

// =============================================================================
// Objective Enum for Type Safety
// =============================================================================

/// Enum wrapper for all objective types.
///
/// This allows accepting any objective type in GBDTConfig.
#[derive(Debug, Clone, FromPyObject)]
pub enum PyObjective {
    SquaredLoss(PySquaredLoss),
    AbsoluteLoss(PyAbsoluteLoss),
    PoissonLoss(PyPoissonLoss),
    LogisticLoss(PyLogisticLoss),
    HingeLoss(PyHingeLoss),
    HuberLoss(PyHuberLoss),
    PinballLoss(PyPinballLoss),
    ArctanLoss(PyArctanLoss),
    SoftmaxLoss(PySoftmaxLoss),
    LambdaRankLoss(PyLambdaRankLoss),
}
