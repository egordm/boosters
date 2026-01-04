//! Validation utilities for Python bindings.
//!
//! Reusable validators for parameter validation across configs and models.

use pyo3::PyResult;

use crate::error::BoostersError;

/// Validate that a value is positive (> 0).
pub fn validate_positive<T: PartialOrd + Default + std::fmt::Display>(
    name: &str,
    value: T,
) -> PyResult<()> {
    if value <= T::default() {
        return Err(BoostersError::InvalidParameter {
            name: name.to_string(),
            reason: "must be positive".to_string(),
        }
        .into());
    }
    Ok(())
}

/// Validate that a value is non-negative (>= 0).
pub fn validate_non_negative<T: PartialOrd + Default + std::fmt::Display>(
    name: &str,
    value: T,
) -> PyResult<()> {
    if value < T::default() {
        return Err(BoostersError::InvalidParameter {
            name: name.to_string(),
            reason: "must be non-negative".to_string(),
        }
        .into());
    }
    Ok(())
}

/// Validate that a value is in the exclusive-inclusive range (0, 1].
pub fn validate_ratio(name: &str, value: f32) -> PyResult<()> {
    if value <= 0.0 || value > 1.0 {
        return Err(BoostersError::InvalidParameter {
            name: name.to_string(),
            reason: "must be in (0, 1]".to_string(),
        }
        .into());
    }
    Ok(())
}

/// Validate that a value is in the open unit interval (0, 1).
pub fn validate_ratio_open(name: &str, value: f32) -> PyResult<()> {
    if value <= 0.0 || value >= 1.0 {
        return Err(BoostersError::InvalidParameter {
            name: name.to_string(),
            reason: "must be in (0, 1)".to_string(),
        }
        .into());
    }
    Ok(())
}

/// Validate that the actual feature count matches the expected count.
pub fn validate_feature_count(expected: usize, actual: usize) -> PyResult<()> {
    if actual != expected {
        return Err(BoostersError::ValidationError(format!(
            "Expected {} features, got {}",
            expected, actual
        ))
        .into());
    }
    Ok(())
}
