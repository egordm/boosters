//! Error handling for Python bindings.
//!
//! This module defines error types and conversions from Rust errors to Python exceptions.

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::create_exception;
use thiserror::Error;

// =============================================================================
// Custom Python Exception Types
// =============================================================================

// Create ReadError as a subclass of ValueError.
//
// NOTE: We intentionally use pyo3-stub-gen's `create_exception!` wrapper here,
// not PyO3's, so stub generation reliably emits this exception class.
create_exception!(
    boosters._boosters_rs,
    ReadError,
    PyValueError,
    "Exception raised when reading a serialized model fails.

This exception is raised when model deserialization encounters:
- Invalid file format (wrong magic number)
- Corrupted data (checksum mismatch)
- Unsupported schema version
- Invalid JSON structure

Examples:
    >>> try:
    ...     model = boosters.Model.load_from_bytes(invalid_bytes)
    ... except boosters.ReadError as e:
    ...     print(f'Failed to load model: {e}')
"
);

/// Register custom exception types with the module.
pub fn register_exceptions(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ReadError", m.py().get_type::<ReadError>())?;
    Ok(())
}

// =============================================================================
// Error Types
// =============================================================================

/// Errors that can occur in boosters Python bindings.
#[derive(Debug, Error)]
pub enum BoostersError {
    /// Invalid parameter value.
    #[error("Invalid parameter '{name}': {reason}")]
    InvalidParameter { name: String, reason: String },

    /// Type mismatch error.
    #[error("Type error: {0}")]
    TypeError(String),

    /// Shape mismatch in arrays.
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    /// Model not fitted yet.
    #[error("Model not fitted. Call {method}() after fit().")]
    NotFitted { method: String },

    /// Data conversion error.
    #[error("Data conversion error: {0}")]
    DataConversion(String),

    /// Validation error.
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// Training error.
    #[error("Training error: {0}")]
    TrainingError(String),

    /// Prediction error.
    #[error("Prediction error: {0}")]
    Prediction(String),

    /// Explainability error.
    #[error("Explainability error: {0}")]
    ExplainError(String),

    /// Model serialization/deserialization error.
    #[error("{0}")]
    ModelReadError(String),

    /// Model write error.
    #[error("{0}")]
    WriteError(String),
}

// =============================================================================
// Error Conversion
// =============================================================================

impl From<BoostersError> for PyErr {
    fn from(err: BoostersError) -> PyErr {
        match &err {
            BoostersError::InvalidParameter { .. } => PyValueError::new_err(err.to_string()),
            BoostersError::TypeError(_) => PyTypeError::new_err(err.to_string()),
            BoostersError::ShapeMismatch(_) => PyValueError::new_err(err.to_string()),
            BoostersError::NotFitted { .. } => PyRuntimeError::new_err(err.to_string()),
            BoostersError::DataConversion(_) => PyValueError::new_err(err.to_string()),
            BoostersError::ValidationError(_) => PyValueError::new_err(err.to_string()),
            BoostersError::TrainingError(_) => PyRuntimeError::new_err(err.to_string()),
            BoostersError::Prediction(_) => PyRuntimeError::new_err(err.to_string()),
            BoostersError::ExplainError(_) => PyRuntimeError::new_err(err.to_string()),
            BoostersError::ModelReadError(msg) => ReadError::new_err(msg.clone()),
            BoostersError::WriteError(_) => PyRuntimeError::new_err(err.to_string()),
        }
    }
}

/// Convert boosters core errors to Python exceptions.
impl From<boosters::DatasetError> for BoostersError {
    fn from(err: boosters::DatasetError) -> Self {
        BoostersError::DataConversion(err.to_string())
    }
}

// =============================================================================
// Result Type Alias
// =============================================================================

/// Result type for boosters operations.
#[allow(dead_code)]
pub type Result<T> = std::result::Result<T, BoostersError>;
