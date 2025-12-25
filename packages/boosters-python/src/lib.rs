//! Boosters Python bindings.
//!
//! This module provides Python bindings for the boosters gradient boosting library
//! via PyO3. It exposes configuration types, dataset handling, and model training/prediction.

mod error;

use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;

/// Python module for boosters.
///
/// This is the native Rust extension module. Users should import from `boosters`
/// package, not directly from `_boosters_rs`.
#[pymodule]
fn _boosters_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}

// Define stub info gatherer for pyo3-stub-gen
define_stub_info_gatherer!(stub_info);
