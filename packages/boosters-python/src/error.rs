//! Error handling for Python bindings.
//!
//! This module defines error types and conversions from Rust errors to Python exceptions.

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;

// =============================================================================
// Error Types
// =============================================================================

/// Errors that can occur in boosters Python bindings.
#[derive(Debug, Error)]
pub enum BoostersError {
    /// Invalid parameter value.
    #[error("Invalid parameter '{name}': {message}")]
    InvalidParameter { name: String, message: String },

    /// Type mismatch error.
    #[error("Type error: {0}")]
    TypeError(String),

    /// Shape mismatch in arrays.
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    /// Model not fitted yet.
    #[error("Model not fitted. Call fit() first.")]
    NotFitted,

    /// Data conversion error.
    #[error("Data conversion error: {0}")]
    DataConversion(String),

    /// Training error.
    #[error("Training error: {0}")]
    Training(String),

    /// Prediction error.
    #[error("Prediction error: {0}")]
    Prediction(String),
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
            BoostersError::NotFitted => PyRuntimeError::new_err(err.to_string()),
            BoostersError::DataConversion(_) => PyValueError::new_err(err.to_string()),
            BoostersError::Training(_) => PyRuntimeError::new_err(err.to_string()),
            BoostersError::Prediction(_) => PyRuntimeError::new_err(err.to_string()),
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
