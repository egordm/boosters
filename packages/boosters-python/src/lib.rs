//! Boosters Python bindings.
//!
//! This module provides Python bindings for the boosters gradient boosting library
//! via PyO3. It exposes configuration types, dataset handling, and model training/prediction.

mod config;
mod data;
mod error;
mod metrics;
mod model;
mod objectives;
mod persist;
mod types;
mod validation;

use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;

pub use config::{PyGBDTConfig, PyGBLinearConfig, PyGrowthStrategy};
use data::{PyDataset, PyDatasetBuilder, PyFeature};
use metrics::PyMetric;
use model::{PyGBDTModel, PyGBLinearModel};
use objectives::PyObjective;
use persist::{PyModel, PyModelInfo};
use types::{PyGBLinearUpdateStrategy, PyImportanceType, PyVerbosity};

/// Python module for boosters.
///
/// This is the native Rust extension module. Users should import from `boosters`
/// package, not directly from `_boosters_rs`.
#[pymodule]
fn _boosters_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Register custom exception types
    error::register_exceptions(py, m)?;

    // Config types
    m.add_class::<PyGrowthStrategy>()?;
    m.add_class::<PyGBDTConfig>()?;
    m.add_class::<PyGBLinearConfig>()?;

    // Data types
    m.add_class::<PyDataset>()?;
    m.add_class::<PyDatasetBuilder>()?;
    m.add_class::<PyFeature>()?;

    // Model types
    m.add_class::<PyGBDTModel>()?;
    m.add_class::<PyGBLinearModel>()?;

    // Persistence types
    m.add_class::<PyModelInfo>()?;
    m.add_class::<PyModel>()?;

    // Objective and Metric enums (complex enums with variants)
    m.add_class::<PyObjective>()?;
    m.add_class::<PyMetric>()?;

    // Explainability types
    m.add_class::<PyImportanceType>()?;

    // GBLinear update strategy
    m.add_class::<PyGBLinearUpdateStrategy>()?;

    // Verbosity
    m.add_class::<PyVerbosity>()?;

    Ok(())
}

// Define stub info gatherer for pyo3-stub-gen
define_stub_info_gatherer!(stub_info);
