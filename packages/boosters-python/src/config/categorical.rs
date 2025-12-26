//! Categorical feature configuration.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::error::BoostersError;

/// Configuration for categorical feature handling.
///
/// Boosters supports native categorical splits (like LightGBM) using
/// bitset-based multi-way splits rather than one-hot encoding.
///
/// Attributes:
///     max_categories: Maximum categories for native categorical splits.
///     min_category_count: Minimum samples per category.
///     max_onehot: Maximum categories for one-hot encoding.
///
/// Examples:
///     >>> config = CategoricalConfig(max_categories=256)
///     >>> config.max_categories
///     256
#[gen_stub_pyclass]
#[pyclass(name = "CategoricalConfig", module = "boosters._boosters_rs", get_all, set_all)]
#[derive(Clone, Debug)]
pub struct PyCategoricalConfig {
    /// Maximum number of categories for native categorical splits.
    /// Categories beyond this are treated as continuous.
    pub max_categories: u32,
    /// Minimum count for a category to be considered.
    /// Categories with fewer samples are grouped into "other".
    pub min_category_count: u32,
    /// Maximum categories for one-hot encoding fallback.
    /// If a feature has more categories, use bitset splits.
    pub max_onehot: u32,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCategoricalConfig {
    /// Create a new CategoricalConfig.
    ///
    /// Args:
    ///     max_categories: Maximum categories for native splits. Default: 256.
    ///     min_category_count: Minimum samples per category. Default: 10.
    ///     max_onehot: Maximum categories for one-hot encoding. Default: 4.
    #[new]
    #[pyo3(signature = (max_categories = 256, min_category_count = 10, max_onehot = 4))]
    fn new(max_categories: u32, min_category_count: u32, max_onehot: u32) -> PyResult<Self> {
        if max_categories == 0 {
            return Err(BoostersError::InvalidParameter {
                name: "max_categories".to_string(),
                reason: "must be positive".to_string(),
            }
            .into());
        }

        Ok(Self {
            max_categories,
            min_category_count,
            max_onehot,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "CategoricalConfig(max_categories={}, min_category_count={}, max_onehot={})",
            self.max_categories, self.min_category_count, self.max_onehot
        )
    }
}

impl Default for PyCategoricalConfig {
    fn default() -> Self {
        Self {
            max_categories: 256,
            min_category_count: 10,
            max_onehot: 4,
        }
    }
}
