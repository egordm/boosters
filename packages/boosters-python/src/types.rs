//! Common types for Python bindings.
//!
//! Contains enums and simple types used across the bindings.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pymethods;

use boosters::explainability::ImportanceType as CoreImportanceType;
use boosters::training::gblinear::UpdateStrategy as CoreGBLinearUpdateStrategy;
use boosters::training::Verbosity as CoreVerbosity;

use crate::error::BoostersError;

// =============================================================================
// GBLinearUpdateStrategy
// =============================================================================

/// Coordinate descent update strategy for GBLinear.
///
/// Attributes:
///     Shotgun: Parallel (shotgun) coordinate descent (fast, approximate).
///     Sequential: Sequential coordinate descent (slower, deterministic).
#[pyo3_stub_gen::derive::gen_stub_pyclass_enum]
#[pyclass(
    name = "GBLinearUpdateStrategy",
    module = "boosters._boosters_rs",
    eq,
    eq_int
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PyGBLinearUpdateStrategy {
    /// Parallel (shotgun) coordinate descent.
    #[default]
    Shotgun = 0,
    /// Sequential coordinate descent.
    Sequential = 1,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyGBLinearUpdateStrategy {
    fn __str__(&self) -> &'static str {
        match self {
            PyGBLinearUpdateStrategy::Shotgun => "shotgun",
            PyGBLinearUpdateStrategy::Sequential => "sequential",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            PyGBLinearUpdateStrategy::Shotgun => "GBLinearUpdateStrategy.Shotgun",
            PyGBLinearUpdateStrategy::Sequential => "GBLinearUpdateStrategy.Sequential",
        }
    }
}

impl From<PyGBLinearUpdateStrategy> for CoreGBLinearUpdateStrategy {
    fn from(value: PyGBLinearUpdateStrategy) -> Self {
        match value {
            PyGBLinearUpdateStrategy::Shotgun => CoreGBLinearUpdateStrategy::Shotgun,
            PyGBLinearUpdateStrategy::Sequential => CoreGBLinearUpdateStrategy::Sequential,
        }
    }
}

// =============================================================================
// ImportanceType
// =============================================================================

/// Type of feature importance to compute.
///
/// Attributes:
///     Split: Number of times each feature is used in splits.
///     Gain: Total gain from splits using each feature.
///     AverageGain: Average gain per split (Gain / Split count).
///     Cover: Total cover (hessian sum) at nodes splitting on each feature.
///     AverageCover: Average cover per split (Cover / Split count).
///
/// Examples:
///     >>> from boosters import ImportanceType
///     >>> importance = model.feature_importance(ImportanceType.Gain)
#[pyo3_stub_gen::derive::gen_stub_pyclass_enum]
#[pyclass(name = "ImportanceType", module = "boosters._boosters_rs", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PyImportanceType {
    /// Number of times each feature is used in splits.
    #[default]
    Split = 0,
    /// Total gain from splits using each feature.
    Gain = 1,
    /// Average gain per split (Gain / Split count).
    AverageGain = 2,
    /// Total cover (hessian sum) at nodes splitting on each feature.
    Cover = 3,
    /// Average cover per split (Cover / Split count).
    AverageCover = 4,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyImportanceType {
    /// String representation.
    fn __str__(&self) -> &'static str {
        match self {
            PyImportanceType::Split => "split",
            PyImportanceType::Gain => "gain",
            PyImportanceType::AverageGain => "average_gain",
            PyImportanceType::Cover => "cover",
            PyImportanceType::AverageCover => "average_cover",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            PyImportanceType::Split => "ImportanceType.Split",
            PyImportanceType::Gain => "ImportanceType.Gain",
            PyImportanceType::AverageGain => "ImportanceType.AverageGain",
            PyImportanceType::Cover => "ImportanceType.Cover",
            PyImportanceType::AverageCover => "ImportanceType.AverageCover",
        }
    }
}

impl From<PyImportanceType> for CoreImportanceType {
    fn from(py_type: PyImportanceType) -> Self {
        match py_type {
            PyImportanceType::Split => CoreImportanceType::Split,
            PyImportanceType::Gain => CoreImportanceType::Gain,
            PyImportanceType::AverageGain => CoreImportanceType::AverageGain,
            PyImportanceType::Cover => CoreImportanceType::Cover,
            PyImportanceType::AverageCover => CoreImportanceType::AverageCover,
        }
    }
}

impl PyImportanceType {
    /// Parse from string (for backward compatibility with string API).
    pub fn from_str(s: &str) -> PyResult<Self> {
        match s {
            "split" => Ok(PyImportanceType::Split),
            "gain" => Ok(PyImportanceType::Gain),
            "average_gain" => Ok(PyImportanceType::AverageGain),
            "cover" => Ok(PyImportanceType::Cover),
            "average_cover" => Ok(PyImportanceType::AverageCover),
            other => Err(BoostersError::InvalidParameter {
                name: "importance_type".to_string(),
                reason: format!(
                    "expected 'split', 'gain', 'average_gain', 'cover', or 'average_cover', got '{}'",
                    other
                ),
            }
            .into()),
        }
    }
}

// =============================================================================
// Verbosity
// =============================================================================

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
