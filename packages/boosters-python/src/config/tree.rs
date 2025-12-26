//! Tree configuration for gradient boosting.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// Tree growth strategy for building decision trees.
///
/// Determines the order in which nodes are expanded during tree construction.
/// Acts like a Python StrEnum - can be compared to strings.
///
/// Attributes:
///     Depthwise: Grow tree level-by-level (like XGBoost).
///         All nodes at depth d are expanded before any node at depth d+1.
///         More balanced trees, better for shallow trees.
///     Leafwise: Grow tree by best-first split (like LightGBM).
///         Always expands the leaf with highest gain.
///         Can produce deeper, more accurate trees but risks overfitting.
///
/// Examples:
///     >>> from boosters import GrowthStrategy
///     >>> strategy = GrowthStrategy.Depthwise
///     >>> strategy == "depthwise"
///     True
///     >>> str(strategy)
///     'depthwise'
#[pyo3_stub_gen::derive::gen_stub_pyclass_enum]
#[pyclass(name = "GrowthStrategy", module = "boosters._boosters_rs", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PyGrowthStrategy {
    /// Grow tree level-by-level (like XGBoost).
    #[default]
    Depthwise = 0,
    /// Grow tree by best-first split (like LightGBM).
    Leafwise = 1,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyGrowthStrategy {
    /// String representation like StrEnum.
    fn __str__(&self) -> &'static str {
        match self {
            PyGrowthStrategy::Depthwise => "depthwise",
            PyGrowthStrategy::Leafwise => "leafwise",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            PyGrowthStrategy::Depthwise => "GrowthStrategy.Depthwise",
            PyGrowthStrategy::Leafwise => "GrowthStrategy.Leafwise",
        }
    }

    /// Hash using the string value for StrEnum-like behavior.
    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.__str__().hash(&mut hasher);
        hasher.finish()
    }

    /// Pickle support: reduce to module path and variant name.
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        let variant_name = match self {
            PyGrowthStrategy::Depthwise => "Depthwise",
            PyGrowthStrategy::Leafwise => "Leafwise",
        };
        // Return a callable that reconstructs the enum
        let boosters = py.import("boosters._boosters_rs")?;
        let growth_strategy = boosters.getattr("GrowthStrategy")?;
        let variant = growth_strategy.getattr(variant_name)?;
        // Return a function that returns the variant directly
        // Use a lambda-like pattern: (lambda: variant,)
        let obj = variant.unbind();
        Ok((obj.getattr(py, "__class__")?, (variant_name.to_string(),)))
    }

    /// Deepcopy support - return self since enum variants are singletons.
    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        *self
    }

    /// Copy support - return self since enum variants are singletons.
    fn __copy__(&self) -> Self {
        *self
    }
}

/// Configuration for tree structure.
///
/// Controls tree depth, number of leaves, and split constraints.
///
/// Attributes:
///     max_depth (int): Maximum depth of tree. -1 means unlimited
///         (controlled by n_leaves).
///     n_leaves (int): Maximum number of leaves. Only used when max_depth=-1.
///     min_samples_leaf (int): Minimum number of samples required in a leaf.
///     min_gain_to_split (float): Minimum gain required to make a split.
///     growth_strategy (GrowthStrategy): Growth strategy for tree building.
///
/// Examples:
///     >>> from boosters import TreeConfig, GrowthStrategy
///     >>> config = TreeConfig(max_depth=6)
///     >>> config.max_depth
///     6
///     >>> config = TreeConfig(max_depth=-1, n_leaves=31, growth_strategy=GrowthStrategy.Leafwise)
#[gen_stub_pyclass]
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
    /// Growth strategy for tree building.
    pub growth_strategy: PyGrowthStrategy,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTreeConfig {
    /// Create a new TreeConfig.
    ///
    /// Args:
    ///     max_depth: Maximum depth of tree. -1 means unlimited
    ///         (controlled by n_leaves). Defaults to -1.
    ///     n_leaves: Maximum number of leaves. Only used when max_depth=-1.
    ///         Defaults to 31.
    ///     min_samples_leaf: Minimum samples required in a leaf. Defaults to 1.
    ///     min_gain_to_split: Minimum gain required to make a split.
    ///         Defaults to 0.0.
    ///     growth_strategy: Tree growth strategy. Defaults to Depthwise.
    ///
    /// Returns:
    ///     A new TreeConfig instance.
    ///
    /// Examples:
    ///     >>> from boosters import TreeConfig, GrowthStrategy
    ///     >>> config = TreeConfig(max_depth=6)
    ///     >>> config = TreeConfig(growth_strategy=GrowthStrategy.Leafwise)
    #[new]
    #[pyo3(signature = (
        max_depth = -1,
        n_leaves = 31,
        min_samples_leaf = 1,
        min_gain_to_split = 0.0,
        growth_strategy = PyGrowthStrategy::Depthwise,
    ))]
    fn new(
        max_depth: i32,
        n_leaves: u32,
        min_samples_leaf: u32,
        min_gain_to_split: f64,
        growth_strategy: PyGrowthStrategy,
    ) -> PyResult<Self> {
        // Validate min_gain_to_split
        if min_gain_to_split < 0.0 {
            return Err(crate::error::BoostersError::InvalidParameter {
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
            "TreeConfig(max_depth={}, n_leaves={}, min_samples_leaf={}, min_gain_to_split={}, growth_strategy={:?})",
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
            growth_strategy: PyGrowthStrategy::Depthwise,
        }
    }
}
