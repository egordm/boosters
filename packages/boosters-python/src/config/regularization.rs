//! Regularization configuration for gradient boosting.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::error::BoostersError;

/// Configuration for L1/L2 regularization.
///
/// Examples
/// --------
/// >>> from boosters import RegularizationConfig
/// >>> config = RegularizationConfig(l1=0.1, l2=1.0)
/// >>> config.l2
/// 1.0
#[gen_stub_pyclass]
#[pyclass(name = "RegularizationConfig", module = "boosters._boosters_rs", get_all, set_all)]
#[derive(Clone, Debug)]
pub struct PyRegularizationConfig {
    /// L1 (Lasso) regularization term on leaf weights.
    pub l1: f64,
    /// L2 (Ridge) regularization term on leaf weights.
    pub l2: f64,
    /// Minimum sum of hessians required in a leaf.
    pub min_hessian: f64,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRegularizationConfig {
    /// Create a new RegularizationConfig.
    ///
    /// Parameters
    /// ----------
    /// l1 : float, default=0.0
    ///     L1 regularization term.
    /// l2 : float, default=1.0
    ///     L2 regularization term.
    /// min_hessian : float, default=1.0
    ///     Minimum sum of hessians in a leaf.
    #[new]
    #[pyo3(signature = (l1 = 0.0, l2 = 1.0, min_hessian = 1.0))]
    fn new(l1: f64, l2: f64, min_hessian: f64) -> PyResult<Self> {
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
        if min_hessian < 0.0 {
            return Err(BoostersError::InvalidParameter {
                name: "min_hessian".to_string(),
                reason: "must be non-negative".to_string(),
            }
            .into());
        }

        Ok(Self { l1, l2, min_hessian })
    }

    fn __repr__(&self) -> String {
        format!(
            "RegularizationConfig(l1={}, l2={}, min_hessian={})",
            self.l1, self.l2, self.min_hessian
        )
    }
}

impl Default for PyRegularizationConfig {
    fn default() -> Self {
        Self {
            l1: 0.0,
            l2: 1.0,
            min_hessian: 1.0,
        }
    }
}
