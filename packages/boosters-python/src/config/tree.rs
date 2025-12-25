//! Tree configuration for gradient boosting.

use pyo3::prelude::*;

use crate::error::BoostersError;

/// Configuration for tree structure.
///
/// Controls tree depth, number of leaves, and split constraints.
///
/// Examples
/// --------
/// >>> from boosters import TreeConfig
/// >>> config = TreeConfig(max_depth=6, n_leaves=31)
/// >>> config.max_depth
/// 6
#[pyclass(name = "TreeConfig", module = "boosters._boosters_rs", get_all, set_all)]
#[derive(Clone, Debug)]
pub struct PyTreeConfig {
    /// Maximum depth of tree. -1 means unlimited (controlled by n_leaves).
    pub max_depth: i32,
    /// Maximum number of leaves. Only used when max_depth is -1.
    pub n_leaves: u32,
    /// Minimum number of samples required in a leaf node.
    pub min_samples_leaf: u32,
    /// Minimum gain required to make a split.
    pub min_gain_to_split: f64,
    /// Growth strategy: "depthwise" or "leafwise".
    pub growth_strategy: String,
}

#[pymethods]
impl PyTreeConfig {
    /// Create a new TreeConfig.
    ///
    /// Parameters
    /// ----------
    /// max_depth : int, default=-1
    ///     Maximum depth of tree. -1 means unlimited.
    /// n_leaves : int, default=31
    ///     Maximum number of leaves (only used when max_depth=-1).
    /// min_samples_leaf : int, default=1
    ///     Minimum samples required in a leaf.
    /// min_gain_to_split : float, default=0.0
    ///     Minimum gain required to make a split.
    /// growth_strategy : str, default="depthwise"
    ///     Tree growth strategy: "depthwise" or "leafwise".
    #[new]
    #[pyo3(signature = (
        max_depth = -1,
        n_leaves = 31,
        min_samples_leaf = 1,
        min_gain_to_split = 0.0,
        growth_strategy = "depthwise".to_string()
    ))]
    fn new(
        max_depth: i32,
        n_leaves: u32,
        min_samples_leaf: u32,
        min_gain_to_split: f64,
        growth_strategy: String,
    ) -> PyResult<Self> {
        // Validate growth_strategy
        if growth_strategy != "depthwise" && growth_strategy != "leafwise" {
            return Err(BoostersError::InvalidParameter {
                name: "growth_strategy".to_string(),
                reason: format!(
                    "expected 'depthwise' or 'leafwise', got '{}'",
                    growth_strategy
                ),
            }
            .into());
        }

        // Validate min_gain_to_split
        if min_gain_to_split < 0.0 {
            return Err(BoostersError::InvalidParameter {
                name: "min_gain_to_split".to_string(),
                reason: "must be non-negative".to_string(),
            }
            .into());
        }

        Ok(Self {
            max_depth,
            n_leaves,
            min_samples_leaf,
            min_gain_to_split,
            growth_strategy,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "TreeConfig(max_depth={}, n_leaves={}, min_samples_leaf={}, min_gain_to_split={}, growth_strategy='{}')",
            self.max_depth, self.n_leaves, self.min_samples_leaf, self.min_gain_to_split, self.growth_strategy
        )
    }
}

impl Default for PyTreeConfig {
    fn default() -> Self {
        Self {
            max_depth: -1,
            n_leaves: 31,
            min_samples_leaf: 1,
            min_gain_to_split: 0.0,
            growth_strategy: "depthwise".to_string(),
        }
    }
}
