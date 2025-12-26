//! Split finding for histogram-based tree training.
//!
//! This module provides split types, gain computation, and algorithms for finding
//! optimal splits during tree growth.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                           GreedySplitter                                 │
//! │                                                                          │
//! │  Owns:                                                                   │
//! │  - GainParams (regularization, constraints)                             │
//! │  - Feature type flags (numerical vs categorical)                        │
//! │  - Parallelization strategy selection                                   │
//! │                                                                          │
//! │  Key methods:                                                            │
//! │  - find_split() - auto-selects sequential vs parallel                   │
//! │  - find_split_sequential() - scans features one by one                  │
//! │  - find_split_parallel() - parallel feature search with rayon          │
//! │                                                                          │
//! └────────────────────────┬────────────────────────────────────────────────┘
//!                          │
//!            ┌─────────────┼─────────────┐
//!            ▼             ▼             ▼
//!     Numerical       Categorical     Categorical
//!     Split           (One-Hot)       (Sorted)
//!                                                                            
//!   Forward scan      Each cat as     Sort by grad/hess
//!   + backward for    singleton       ratio, scan for
//!   missing values    left partition  optimal partition
//! ```
//!
//! # Split Finding Process
//!
//! 1. **Feature iteration**: For each feature in the histogram
//! 2. **Type dispatch**: Route to numerical or categorical split finder
//! 3. **Candidate evaluation**: Scan bins/categories to find best split point
//! 4. **Gain computation**: Calculate XGBoost gain formula with regularization
//! 5. **Constraint checking**: Verify min_child_weight and min_samples_leaf
//! 6. **Best selection**: Track split with highest gain across all features
//!
//! # Numerical Splits
//!
//! Uses bidirectional scanning to handle missing values optimally:
//! - **Forward scan**: Missing values go right (default_left = false)
//! - **Backward scan**: Missing values go left (default_left = true)
//! - Returns whichever direction yields higher gain
//!
//! # Categorical Splits
//!
//! Two strategies based on cardinality:
//! - **One-hot** (≤ max_onehot_cats): Try each category as singleton left partition
//! - **Sorted partition** (> max_onehot_cats): Sort by grad/hess ratio, scan for optimal cut
//!
//! # Module Organization
//!
//! - [`types`] - Split result types (`SplitInfo`, `SplitType`)
//! - [`gain`] - Gain computation and regularization (`GainParams`)
//! - [`find`] - Split finding algorithms (`GreedySplitter`, `SplitStrategy`)

pub mod find;
pub mod gain;
pub mod types;

// Re-exports for convenience
pub use find::{DEFAULT_MAX_ONEHOT_CATS, GreedySplitter};
pub use gain::GainParams;
pub use types::{SplitInfo, SplitType};
