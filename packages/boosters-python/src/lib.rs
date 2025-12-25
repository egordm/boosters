//! Boosters Python bindings.
//!
//! This module provides Python bindings for the boosters gradient boosting library
//! via PyO3. It exposes configuration types, dataset handling, and model training/prediction.

mod config;
mod convert;
mod data;
mod error;
mod metrics;
mod model;
mod objectives;
mod threading;

use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;

use config::{
    PyCategoricalConfig, PyEFBConfig, PyGBDTConfig, PyGBLinearConfig, PyLinearLeavesConfig,
    PyRegularizationConfig, PySamplingConfig, PyTreeConfig,
};
use data::{PyDataset, PyEvalSet};
use metrics::{PyAccuracy, PyAuc, PyLogLoss, PyMae, PyMape, PyNdcg, PyRmse};
use model::PyGBDTModel;
use objectives::{
    PyAbsoluteLoss, PyArctanLoss, PyHingeLoss, PyHuberLoss, PyLambdaRankLoss, PyLogisticLoss,
    PyPinballLoss, PyPoissonLoss, PySoftmaxLoss, PySquaredLoss,
};

/// Python module for boosters.
///
/// This is the native Rust extension module. Users should import from `boosters`
/// package, not directly from `_boosters_rs`.
#[pymodule]
fn _boosters_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Config types
    m.add_class::<PyTreeConfig>()?;
    m.add_class::<PyRegularizationConfig>()?;
    m.add_class::<PySamplingConfig>()?;
    m.add_class::<PyCategoricalConfig>()?;
    m.add_class::<PyEFBConfig>()?;
    m.add_class::<PyLinearLeavesConfig>()?;
    m.add_class::<PyGBDTConfig>()?;
    m.add_class::<PyGBLinearConfig>()?;

    // Data types
    m.add_class::<PyDataset>()?;
    m.add_class::<PyEvalSet>()?;

    // Model types
    m.add_class::<PyGBDTModel>()?;

    // Objective types
    m.add_class::<PySquaredLoss>()?;
    m.add_class::<PyAbsoluteLoss>()?;
    m.add_class::<PyPoissonLoss>()?;
    m.add_class::<PyLogisticLoss>()?;
    m.add_class::<PyHingeLoss>()?;
    m.add_class::<PyHuberLoss>()?;
    m.add_class::<PyPinballLoss>()?;
    m.add_class::<PyArctanLoss>()?;
    m.add_class::<PySoftmaxLoss>()?;
    m.add_class::<PyLambdaRankLoss>()?;

    // Metric types
    m.add_class::<PyRmse>()?;
    m.add_class::<PyMae>()?;
    m.add_class::<PyMape>()?;
    m.add_class::<PyLogLoss>()?;
    m.add_class::<PyAuc>()?;
    m.add_class::<PyAccuracy>()?;
    m.add_class::<PyNdcg>()?;

    Ok(())
}

// Define stub info gatherer for pyo3-stub-gen
define_stub_info_gatherer!(stub_info);
