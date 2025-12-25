//! Linear leaves configuration.

use pyo3::prelude::*;

use crate::error::BoostersError;

/// Configuration for linear models in leaf nodes.
///
/// When enabled, each leaf fits a linear regression model on its samples
/// instead of using a constant value.
///
/// Examples
/// --------
/// >>> from boosters import LinearLeavesConfig
/// >>> config = LinearLeavesConfig(enable=True, l2=0.01)
/// >>> config.enable
/// True
#[pyclass(name = "LinearLeavesConfig", module = "boosters._boosters_rs", get_all, set_all)]
#[derive(Clone, Debug)]
pub struct PyLinearLeavesConfig {
    /// Whether to enable linear models in leaves.
    pub enable: bool,
    /// L2 regularization for linear coefficients.
    pub l2: f64,
    /// L1 regularization for linear coefficients.
    pub l1: f64,
    /// Maximum coordinate descent iterations per leaf.
    pub max_iter: u32,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Minimum samples required to fit a linear model.
    pub min_samples: u32,
}

#[pymethods]
impl PyLinearLeavesConfig {
    /// Create a new LinearLeavesConfig.
    ///
    /// Parameters
    /// ----------
    /// enable : bool, default=False
    ///     Whether to enable linear leaves.
    /// l2 : float, default=0.01
    ///     L2 regularization.
    /// l1 : float, default=0.0
    ///     L1 regularization.
    /// max_iter : int, default=10
    ///     Maximum coordinate descent iterations.
    /// tolerance : float, default=1e-6
    ///     Convergence tolerance.
    /// min_samples : int, default=50
    ///     Minimum samples to fit linear model.
    #[new]
    #[pyo3(signature = (
        enable = false,
        l2 = 0.01,
        l1 = 0.0,
        max_iter = 10,
        tolerance = 1e-6,
        min_samples = 50
    ))]
    fn new(
        enable: bool,
        l2: f64,
        l1: f64,
        max_iter: u32,
        tolerance: f64,
        min_samples: u32,
    ) -> PyResult<Self> {
        if l2 < 0.0 {
            return Err(BoostersError::InvalidParameter {
                name: "l2".to_string(),
                reason: "must be non-negative".to_string(),
            }
            .into());
        }
        if l1 < 0.0 {
            return Err(BoostersError::InvalidParameter {
                name: "l1".to_string(),
                reason: "must be non-negative".to_string(),
            }
            .into());
        }
        if tolerance <= 0.0 {
            return Err(BoostersError::InvalidParameter {
                name: "tolerance".to_string(),
                reason: "must be positive".to_string(),
            }
            .into());
        }

        Ok(Self {
            enable,
            l2,
            l1,
            max_iter,
            tolerance,
            min_samples,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "LinearLeavesConfig(enable={}, l2={}, l1={}, max_iter={}, tolerance={}, min_samples={})",
            self.enable, self.l2, self.l1, self.max_iter, self.tolerance, self.min_samples
        )
    }
}

impl Default for PyLinearLeavesConfig {
    fn default() -> Self {
        Self {
            enable: false,
            l2: 0.01,
            l1: 0.0,
            max_iter: 10,
            tolerance: 1e-6,
            min_samples: 50,
        }
    }
}
