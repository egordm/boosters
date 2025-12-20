//! Canonical model representations.
//!
//! This module defines the core data structures used across training, inference,
//! and compatibility loaders. It is intentionally runtime-neutral: optimized
//! execution strategies live in `inference`.

pub mod gbdt;
pub mod gblinear;
