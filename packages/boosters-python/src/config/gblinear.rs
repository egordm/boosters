//! GBLinear configuration (placeholder - Story 2.4).

use pyo3::prelude::*;

/// Main configuration for GBLinear model.
///
/// This is a placeholder - full implementation in Story 2.4.
#[pyclass(name = "GBLinearConfig", module = "boosters._boosters_rs")]
#[derive(Clone, Debug, Default)]
pub struct PyGBLinearConfig;

#[pymethods]
impl PyGBLinearConfig {
    #[new]
    fn new() -> Self {
        Self
    }
}
