//! Error types for Python bindings.

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;

/// Errors that can occur in the Python bindings.
#[derive(Error, Debug)]
pub enum PyBoostersError {
    #[error("Training error: {0}")]
    Training(String),

    #[error("Prediction error: {0}")]
    Prediction(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Deserialization error: {0}")]
    Deserialization(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Invalid data: {0}")]
    InvalidData(String),

    #[error("Explainability error: {0}")]
    Explainability(String),
}

impl From<PyBoostersError> for PyErr {
    fn from(err: PyBoostersError) -> PyErr {
        match err {
            PyBoostersError::Training(msg) => PyRuntimeError::new_err(msg),
            PyBoostersError::Prediction(msg) => PyRuntimeError::new_err(msg),
            PyBoostersError::Serialization(msg) => PyRuntimeError::new_err(msg),
            PyBoostersError::Deserialization(msg) => PyRuntimeError::new_err(msg),
            PyBoostersError::InvalidParameter(msg) => PyValueError::new_err(msg),
            PyBoostersError::InvalidData(msg) => PyTypeError::new_err(msg),
            PyBoostersError::Explainability(msg) => PyRuntimeError::new_err(msg),
        }
    }
}

impl From<boosters::explainability::ExplainError> for PyBoostersError {
    fn from(err: boosters::explainability::ExplainError) -> Self {
        PyBoostersError::Explainability(err.to_string())
    }
}
