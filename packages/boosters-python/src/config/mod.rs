//! Configuration types for Python bindings.
//!
//! All config types are Rust-owned (`#[pyclass]`) with getters/setters.

mod tree;
mod regularization;
mod sampling;
mod categorical;
mod efb;
mod linear_leaves;
mod gbdt;
mod gblinear;

pub use tree::{PyTreeConfig, PyGrowthStrategy};
pub use regularization::PyRegularizationConfig;
pub use sampling::PySamplingConfig;
pub use categorical::PyCategoricalConfig;
pub use efb::PyEFBConfig;
pub use linear_leaves::PyLinearLeavesConfig;
pub use gbdt::PyGBDTConfig;
pub use gblinear::PyGBLinearConfig;
