//! Linear leaf training components.
//!
//! This module provides support for fitting linear models at tree leaves,
//! enabling smoother predictions within each leaf partition.
//!
//! # Components
//!
//! - [`LeafFeatureBuffer`]: Column-major buffer for gathering leaf features
//! - [`WeightedLeastSquaresSolver`]: Coordinate descent solver for weighted least squares
//!
//! # Design
//!
//! Linear leaves fit `intercept + Σ(coef × feature)` at each leaf.
//! See RFC-0015 for design rationale.

mod buffer;
mod solver;

pub use buffer::LeafFeatureBuffer;
pub use solver::WeightedLeastSquaresSolver;
