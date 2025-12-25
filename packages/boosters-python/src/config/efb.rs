//! Exclusive Feature Bundling (EFB) configuration.

use pyo3::prelude::*;

use crate::error::BoostersError;

/// Configuration for Exclusive Feature Bundling.
///
/// EFB bundles mutually exclusive features to reduce memory and computation,
/// similar to LightGBM's implementation.
///
/// Examples
/// --------
/// >>> from boosters import EFBConfig
/// >>> config = EFBConfig(enable=True, max_conflict_rate=0.0)
/// >>> config.enable
/// True
#[pyclass(name = "EFBConfig", module = "boosters._boosters_rs", get_all, set_all)]
#[derive(Clone, Debug)]
pub struct PyEFBConfig {
    /// Whether to enable Exclusive Feature Bundling.
    pub enable: bool,
    /// Maximum conflict rate allowed when bundling features.
    /// 0.0 means only truly exclusive features are bundled.
    pub max_conflict_rate: f64,
}

#[pymethods]
impl PyEFBConfig {
    /// Create a new EFBConfig.
    ///
    /// Parameters
    /// ----------
    /// enable : bool, default=True
    ///     Whether to enable EFB.
    /// max_conflict_rate : float, default=0.0
    ///     Maximum conflict rate for bundling. Must be in [0, 1).
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
