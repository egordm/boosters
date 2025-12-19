//! Common utilities used across the crate.
//!
//! This module provides slice utilities and iterator helpers that are used
//! by various subsystems.

// =============================================================================
// Slice Utilities
// =============================================================================

/// Get two disjoint mutable slices from a single slice.
///
/// Returns `(slice[a_start..a_start+len], slice[b_start..b_start+len])`
/// with the first element mutable and second immutable.
///
/// # Panics
/// Panics if the ranges overlap or are out of bounds.
#[inline]
pub fn disjoint_slices_mut<T>(
    slice: &mut [T],
    a_start: usize,
    b_start: usize,
    len: usize,
) -> (&mut [T], &[T]) {
    debug_assert!(
        a_start + len <= b_start || b_start + len <= a_start,
        "Ranges overlap: [{}..{}] and [{}..{}]",
        a_start,
        a_start + len,
        b_start,
        b_start + len
    );
    debug_assert!(
        a_start + len <= slice.len() && b_start + len <= slice.len(),
        "Range out of bounds"
    );

    if a_start < b_start {
        let (left, right) = slice.split_at_mut(b_start);
        (&mut left[a_start..a_start + len], &right[..len])
    } else {
        let (left, right) = slice.split_at_mut(a_start);
        (&mut right[..len], &left[b_start..b_start + len])
    }
}

// =============================================================================
// Weight Iterator
// =============================================================================

/// Returns an iterator over weights, using 1.0 for empty weights.
///
/// This allows unified handling of weighted and unweighted computations
/// without branching in hot loops. Used by both objectives and metrics.
///
/// # Example
///
/// ```ignore
/// for (i, w) in weight_iter(weights, n_rows).enumerate() {
///     sum += w * values[i];
///     weight_sum += w;
/// }
/// ```
#[inline]
pub(crate) fn weight_iter(weights: &[f32], n_rows: usize) -> impl Iterator<Item = f32> + '_ {
    let use_weights = !weights.is_empty();
    (0..n_rows).map(move |i| if use_weights { weights[i] } else { 1.0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_iter_empty() {
        let weights: &[f32] = &[];
        let result: Vec<f32> = weight_iter(weights, 3).collect();
        assert_eq!(result, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_weight_iter_with_weights() {
        let weights = &[0.5f32, 2.0, 1.5];
        let result: Vec<f32> = weight_iter(weights, 3).collect();
        assert_eq!(result, vec![0.5, 2.0, 1.5]);
    }

    #[test]
    fn test_disjoint_slices_mut() {
        let mut data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let (a, b) = disjoint_slices_mut(&mut data, 0, 5, 3);
        assert_eq!(a, &mut [0, 1, 2]);
        assert_eq!(b, &[5, 6, 7]);

        // Modify a
        a[0] = 100;
        assert_eq!(data[0], 100);
    }

    #[test]
    fn test_disjoint_slices_mut_reversed() {
        let mut data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let (a, b) = disjoint_slices_mut(&mut data, 6, 2, 3);
        assert_eq!(a, &mut [6, 7, 8]);
        assert_eq!(b, &[2, 3, 4]);
    }
}
