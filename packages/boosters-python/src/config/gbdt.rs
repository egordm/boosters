//! GBDT configuration (placeholder - Story 2.4).

use pyo3::prelude::*;

/// Main configuration for GBDT model.
///
/// This is a placeholder - full implementation in Story 2.4.
#[pyclass(name = "GBDTConfig", module = "boosters._boosters_rs")]
#[derive(Clone, Debug, Default)]
pub struct PyGBDTConfig;

#[pymethods]
impl PyGBDTConfig {
    #[new]
    fn new() -> Self {
        Self
    }
}
