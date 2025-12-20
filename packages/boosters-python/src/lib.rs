//! Boosters Python bindings.
//!
//! This is a placeholder. Python bindings will be implemented later.

use pyo3::prelude::*;

/// Module initialization.
#[pymodule]
fn _boosters_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    Ok(())
}
