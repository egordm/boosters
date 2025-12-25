//! Sampling configuration for gradient boosting.

use pyo3::prelude::*;

use crate::error::BoostersError;

/// Configuration for row and column subsampling.
///
/// Examples
/// --------
/// >>> from boosters import SamplingConfig
/// >>> config = SamplingConfig(subsample=0.8, colsample=0.8)
/// >>> config.subsample
/// 0.8
#[pyclass(name = "SamplingConfig", module = "boosters._boosters_rs", get_all, set_all)]
#[derive(Clone, Debug)]
pub struct PySamplingConfig {
    /// Row subsampling ratio (per tree). Value in (0, 1].
    pub subsample: f64,
    /// Column subsampling ratio (per tree). Value in (0, 1].
    pub colsample: f64,
    /// Column subsampling ratio (per level). Value in (0, 1].
    pub colsample_bylevel: f64,
    /// GOSS top percentage (for gradient-based one-side sampling).
    /// If > 0, enables GOSS. Value in [0, 1).
    pub goss_alpha: f64,
    /// GOSS random percentage for small gradients.
    /// Value in [0, 1].
    pub goss_beta: f64,
}

#[pymethods]
impl PySamplingConfig {
    /// Create a new SamplingConfig.
    ///
    /// Parameters
    /// ----------
    /// subsample : float, default=1.0
    ///     Row subsampling ratio. Must be in (0, 1].
    /// colsample : float, default=1.0
    ///     Column subsampling ratio per tree. Must be in (0, 1].
    /// colsample_bylevel : float, default=1.0
    ///     Column subsampling ratio per level. Must be in (0, 1].
    /// goss_alpha : float, default=0.0
    ///     GOSS top percentage. 0 disables GOSS.
    /// goss_beta : float, default=0.0
    ///     GOSS random percentage for small gradients.
    #[new]
    #[pyo3(signature = (
        subsample = 1.0,
        colsample = 1.0,
        colsample_bylevel = 1.0,
        goss_alpha = 0.0,
        goss_beta = 0.0
    ))]
    fn new(
        subsample: f64,
        colsample: f64,
        colsample_bylevel: f64,
        goss_alpha: f64,
        goss_beta: f64,
    ) -> PyResult<Self> {
        // Validate subsample
        if subsample <= 0.0 || subsample > 1.0 {
            return Err(BoostersError::InvalidParameter {
                name: "subsample".to_string(),
                message: "must be in (0, 1]".to_string(),
            }
            .into());
        }
        // Validate colsample
        if colsample <= 0.0 || colsample > 1.0 {
            return Err(BoostersError::InvalidParameter {
                name: "colsample".to_string(),
                message: "must be in (0, 1]".to_string(),
            }
            .into());
        }
        // Validate colsample_bylevel
        if colsample_bylevel <= 0.0 || colsample_bylevel > 1.0 {
            return Err(BoostersError::InvalidParameter {
                name: "colsample_bylevel".to_string(),
                message: "must be in (0, 1]".to_string(),
            }
            .into());
        }
        // Validate goss_alpha
        if goss_alpha < 0.0 || goss_alpha >= 1.0 {
            return Err(BoostersError::InvalidParameter {
                name: "goss_alpha".to_string(),
                message: "must be in [0, 1)".to_string(),
            }
            .into());
        }
        // Validate goss_beta
        if goss_beta < 0.0 || goss_beta > 1.0 {
            return Err(BoostersError::InvalidParameter {
                name: "goss_beta".to_string(),
                message: "must be in [0, 1]".to_string(),
            }
            .into());
        }

        Ok(Self {
            subsample,
            colsample,
            colsample_bylevel,
            goss_alpha,
            goss_beta,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "SamplingConfig(subsample={}, colsample={}, colsample_bylevel={}, goss_alpha={}, goss_beta={})",
            self.subsample, self.colsample, self.colsample_bylevel, self.goss_alpha, self.goss_beta
        )
    }
}

impl Default for PySamplingConfig {
    fn default() -> Self {
        Self {
            subsample: 1.0,
            colsample: 1.0,
            colsample_bylevel: 1.0,
            goss_alpha: 0.0,
            goss_beta: 0.0,
        }
    }
}
