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

#[cfg(test)]
mod tests {
    use super::*;

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
