//! Python bindings for the boosters library.
//!
//! This crate provides PyO3-based bindings for training and inference
//! with gradient boosted models.

use pyo3::prelude::*;

mod dataset;
mod error;
mod gbdt;
mod linear;
mod params;

/// Get the version of the boosters library.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Python module for boosters gradient boosting library.
#[pymodule]
fn boosters_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(version, m)?)?;

    // Dataset
    m.add_class::<dataset::PyDataset>()?;

    // Parameters
    m.add_class::<params::PyGBDTParams>()?;
    m.add_class::<params::PyGBLinearParams>()?;

    // Models
    m.add_class::<gbdt::PyGBDTBooster>()?;
    m.add_class::<linear::PyGBLinearBooster>()?;

    Ok(())
}
