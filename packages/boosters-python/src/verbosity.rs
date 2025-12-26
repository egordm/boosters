//! Verbosity level enum for Python bindings.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pymethods;

use boosters::training::Verbosity as CoreVerbosity;

/// Verbosity level for training output.
///
/// Controls the amount of information logged during training.
///
/// Attributes:
///     Silent: No output.
///     Warning: Errors and warnings only.
///     Info: Progress and important information.
///     Debug: Detailed debugging information.
///
/// Examples:
///     >>> from boosters import Verbosity, GBDTConfig
///     >>> config = GBDTConfig(verbosity=Verbosity.Info)
#[pyo3_stub_gen::derive::gen_stub_pyclass_enum]
#[pyclass(name = "Verbosity", module = "boosters._boosters_rs", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PyVerbosity {
    /// No output.
    #[default]
    Silent = 0,
    /// Errors and warnings only.
    Warning = 1,
    /// Progress and important information.
    Info = 2,
    /// Detailed debugging information.
    Debug = 3,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyVerbosity {
    /// String representation.
    fn __str__(&self) -> &'static str {
        match self {
            PyVerbosity::Silent => "silent",
            PyVerbosity::Warning => "warning",
            PyVerbosity::Info => "info",
            PyVerbosity::Debug => "debug",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            PyVerbosity::Silent => "Verbosity.Silent",
            PyVerbosity::Warning => "Verbosity.Warning",
            PyVerbosity::Info => "Verbosity.Info",
            PyVerbosity::Debug => "Verbosity.Debug",
        }
    }
}

impl From<PyVerbosity> for CoreVerbosity {
    fn from(py_verbosity: PyVerbosity) -> Self {
        match py_verbosity {
            PyVerbosity::Silent => CoreVerbosity::Silent,
            PyVerbosity::Warning => CoreVerbosity::Warning,
            PyVerbosity::Info => CoreVerbosity::Info,
            PyVerbosity::Debug => CoreVerbosity::Debug,
        }
    }
}
