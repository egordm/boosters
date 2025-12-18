//! Linear leaf training components.
//!
//! This module provides support for fitting linear models at tree leaves,
//! enabling smoother predictions within each leaf partition.
//!
//! # Components
//!
//! - [`LeafFeatureBuffer`]: Column-major buffer for gathering leaf features
//!
//! # Design
//!
//! Linear leaves fit `intercept + Σ(coef × feature)` at each leaf.
//! See RFC-0015 for design rationale.

mod buffer;

pub use buffer::LeafFeatureBuffer;
