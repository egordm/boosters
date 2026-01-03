//! Testing utilities for booste-rs.
//!
//! This module provides testing utilities for both unit and integration tests.
//!
//! # Approach
//!
//! For scalar floating-point comparisons, use `approx` crate directly:
//! ```ignore
//! use approx::assert_abs_diff_eq;
//!
//! assert_abs_diff_eq!(1.0f32, 1.0001f32, epsilon = 0.001);
//! ```
//!
//! For ndarray comparisons, use `approx` directly - Array2<f32> implements
//! `AbsDiffEq` which checks both shape and values:
//! ```ignore
//! use approx::assert_abs_diff_eq;
//! use ndarray::arr2;
//!
//! let actual = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
//! let expected = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
//! assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
//! ```
//!
//! For slice comparisons with nice diff output on failure:
//! ```ignore
//! use boosters::testing::assert_slices_approx_eq;
//!
//! let actual = &[1.0f32, 2.0, 3.0];
//! let expected = &[1.0f32, 2.0, 3.0];
//! assert_slices_approx_eq!(actual, expected, 1e-5);
//! ```
//!
//! # Test Data Structures
//!
//! For loading test cases from JSON files, use [`TestInput`] and [`TestExpected`]:
//! ```ignore
//! use boosters::testing::{TestInput, TestExpected};
//!
//! let input: TestInput = serde_json::from_str(json).unwrap();
//! let features = input.to_f32_rows();
//! ```

mod slices;
mod stats;
mod tree;

pub mod synthetic_datasets;

mod cases;

// =============================================================================
// Constants
// =============================================================================

/// Default tolerance for floating point comparisons (f32).
/// This is appropriate for most predictions where values are O(1).
pub const DEFAULT_TOLERANCE: f32 = 1e-5;

/// Same tolerance as f64 for compatibility with test expected values.
pub const DEFAULT_TOLERANCE_F64: f64 = 1e-5;

pub use slices::{format_slice_diff, format_slice_diff_f64};
pub use stats::pearson_correlation;
pub use tree::{scalar_tree_fn, scalar_tree_with_capacity};

pub use cases::{TestExpected, TestInput};

// Re-export the macros at testing module level.
pub use crate::assert_slices_approx_eq;
pub use crate::assert_slices_approx_eq_f64;
pub use crate::cat_bitset;
pub use crate::scalar_tree;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repr::gbdt::ScalarLeaf;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_slices_approx_eq_macro() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0001f32, 2.0001, 3.0001];
        assert_slices_approx_eq!(&a, &b, 0.001);
    }

    #[test]
    fn test_slices_approx_eq_f64_macro() {
        let actual = [1.0f32, 2.0, 3.0];
        let expected = [1.0f64, 2.0, 3.0];
        assert_slices_approx_eq_f64!(&actual, &expected, 1e-5);
    }

    #[test]
    #[should_panic(expected = "slices not approximately equal")]
    fn test_slices_macro_fails() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32, 5.0, 3.0];
        assert_slices_approx_eq!(&a, &b, 1e-5);
    }

    #[test]
    fn test_format_slice_diff() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32, 5.0, 3.0];
        let diff = format_slice_diff(&a, &b, 1e-5);
        assert!(diff.contains("1 values differ"));
        assert!(diff.contains("[   1]"));
    }

    #[test]
    fn test_pearson_correlation_perfect() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0f32, 4.0, 6.0, 8.0, 10.0];
        let corr = pearson_correlation(&a, &b);
        assert_abs_diff_eq!(corr, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pearson_correlation_negative() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [5.0f32, 4.0, 3.0, 2.0, 1.0];
        let corr = pearson_correlation(&a, &b);
        assert_abs_diff_eq!(corr, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pearson_correlation_zero_variance() {
        let a = [1.0f32, 1.0, 1.0];
        let b = [2.0f32, 3.0, 4.0];
        let corr = pearson_correlation(&a, &b);
        assert_eq!(corr, 0.0, "Zero variance should return 0");
    }

    #[test]
    fn test_scalar_tree_macro_numeric() {
        let tree = scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => leaf(1.0),
            2 => leaf(2.0),
        };

        assert_eq!(tree.predict_row(&[0.3]), &ScalarLeaf(1.0));
        assert_eq!(tree.predict_row(&[0.7]), &ScalarLeaf(2.0));
    }

    #[test]
    fn test_scalar_tree_macro_categorical() {
        let tree = scalar_tree! {
            0 => cat(0, [2, 4], L) -> 1, 2,
            1 => leaf(1.0),
            2 => leaf(2.0),
        };

        assert_eq!(tree.predict_row(&[2.0]), &ScalarLeaf(2.0));
        assert_eq!(tree.predict_row(&[4.0]), &ScalarLeaf(2.0));
        assert_eq!(tree.predict_row(&[0.0]), &ScalarLeaf(1.0));
        assert_eq!(tree.predict_row(&[3.0]), &ScalarLeaf(1.0));
    }

    #[test]
    fn test_scalar_tree_macro_deeper() {
        let tree = scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => num(1, 0.25, R) -> 3, 4,
            2 => leaf(3.0),
            3 => leaf(1.0),
            4 => leaf(2.0),
        };

        assert_eq!(tree.predict_row(&[0.1, 0.1]), &ScalarLeaf(1.0));
        assert_eq!(tree.predict_row(&[0.3, 0.3]), &ScalarLeaf(2.0));
        assert_eq!(tree.predict_row(&[0.9, 0.9]), &ScalarLeaf(3.0));
    }
}
