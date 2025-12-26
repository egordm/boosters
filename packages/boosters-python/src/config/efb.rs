//! Exclusive Feature Bundling (EFB) configuration.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::error::BoostersError;

/// Configuration for Exclusive Feature Bundling.
///
/// EFB bundles mutually exclusive features to reduce memory and computation,
/// similar to LightGBM's implementation.
///
/// Attributes:
///     enable: Whether to enable EFB.
///     max_conflict_rate: Maximum conflict rate for bundling features.
///
/// Examples:
///     >>> config = EFBConfig(enable=True, max_conflict_rate=0.0)
///     >>> config.enable
///     True
#[gen_stub_pyclass]
#[pyclass(name = "EFBConfig", module = "boosters._boosters_rs", get_all, set_all)]
#[derive(Clone, Debug)]
pub struct PyEFBConfig {
    /// Whether to enable Exclusive Feature Bundling.
    pub enable: bool,
    /// Maximum conflict rate allowed when bundling features.
    /// 0.0 means only truly exclusive features are bundled.
    pub max_conflict_rate: f64,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyEFBConfig {
    /// Create a new EFBConfig.
    ///
    /// Args:
    ///     enable: Whether to enable EFB. Default: True.
    ///     max_conflict_rate: Maximum conflict rate. Must be in [0, 1). Default: 0.0.
    #[new]
    #[pyo3(signature = (enable = true, max_conflict_rate = 0.0))]
    fn new(enable: bool, max_conflict_rate: f64) -> PyResult<Self> {
        if max_conflict_rate < 0.0 || max_conflict_rate >= 1.0 {
            return Err(BoostersError::InvalidParameter {
                name: "max_conflict_rate".to_string(),
                reason: "must be in [0, 1)".to_string(),
            }
            .into());
        }

        Ok(Self {
            enable,
            max_conflict_rate,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "EFBConfig(enable={}, max_conflict_rate={})",
            self.enable, self.max_conflict_rate
        )
    }
}

impl Default for PyEFBConfig {
    fn default() -> Self {
        Self {
            enable: true,
            max_conflict_rate: 0.0,
        }
    }
}
