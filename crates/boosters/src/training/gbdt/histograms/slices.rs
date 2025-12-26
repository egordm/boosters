//! Safe iteration over disjoint histogram feature slices.
//!
//! This module provides [`HistogramFeatureIter`], which encapsulates the unsafe
//! pointer arithmetic needed to yield disjoint mutable slices for each feature's
//! histogram region. The invariants (non-overlapping regions, bounds checking)
//! are enforced here so callers can iterate safely.

use super::HistogramBin;
use super::pool::HistogramLayout;

use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

/// Iterator driver for disjoint per-feature histogram slices.
///
/// This encapsulates the unsafe pointer logic needed to yield `&mut [HistogramBin]`
/// slices for each feature in parallel. The key invariant is that each feature's
/// region is non-overlapping, which is validated in debug builds.
pub struct HistogramFeatureIter {
    hist_addr: usize,
    hist_len: usize,
}

impl HistogramFeatureIter {
    /// Create a new iterator driver from a mutable histogram slice.
    ///
    /// In debug builds, validates that all feature regions are disjoint and in-bounds.
    #[inline]
    pub fn new(histogram: &mut [HistogramBin], _feature_metas: &[HistogramLayout]) -> Self {
        let this = Self {
            hist_addr: histogram.as_mut_ptr() as usize,
            hist_len: histogram.len(),
        };

        #[cfg(debug_assertions)]
        this.debug_validate_feature_metas(_feature_metas);

        this
    }

    /// Iterate over features sequentially, calling `f` for each feature's histogram slice.
    #[inline]
    pub fn for_each_mut<F>(&self, feature_metas: &[HistogramLayout], mut f: F)
    where
        F: FnMut(usize, &mut [HistogramBin]),
    {
        for (feature_index, meta) in feature_metas.iter().enumerate() {
            self.with_feature_slice_mut(meta, |slice| f(feature_index, slice));
        }
    }

    /// Iterate over features in parallel, calling `f` for each feature's histogram slice.
    ///
    /// # Safety invariant
    ///
    /// Each feature's slice is disjoint (enforced by `HistogramLayout` offsets), so
    /// parallel mutable access is safe.
    #[inline]
    pub fn par_for_each_mut<F>(&self, feature_metas: &[HistogramLayout], f: F)
    where
        F: Fn(usize, &mut [HistogramBin]) + Sync + Send,
    {
        feature_metas
            .par_iter()
            .enumerate()
            .for_each(|(feature_index, meta)| {
                self.with_feature_slice_mut(meta, |slice| f(feature_index, slice));
            });
    }

    /// Get a mutable slice for a single feature's histogram region.
    #[inline]
    fn with_feature_slice_mut<F, R>(&self, meta: &HistogramLayout, f: F) -> R
    where
        F: FnOnce(&mut [HistogramBin]) -> R,
    {
        let offset = meta.offset as usize;
        let n_bins = meta.n_bins as usize;
        debug_assert!(offset <= self.hist_len);
        debug_assert!(n_bins <= self.hist_len - offset);

        let base_ptr = self.hist_addr as *mut HistogramBin;
        // SAFETY: We validated in debug that regions are disjoint and in-bounds.
        // Each feature writes to its own non-overlapping region.
        let slice = unsafe {
            let feature_ptr = base_ptr.add(offset);
            std::slice::from_raw_parts_mut(feature_ptr, n_bins)
        };
        f(slice)
    }

    #[cfg(debug_assertions)]
    fn debug_validate_feature_metas(&self, feature_metas: &[HistogramLayout]) {
        let mut regions: Vec<(usize, usize)> = Vec::with_capacity(feature_metas.len());
        for meta in feature_metas {
            let start = meta.offset as usize;
            let end = start.saturating_add(meta.n_bins as usize);
            assert!(
                end <= self.hist_len,
                "feature histogram region out of bounds"
            );
            regions.push((start, end));
        }

        regions.sort_unstable_by_key(|(start, _)| *start);
        for w in regions.windows(2) {
            let (_, prev_end) = w[0];
            let (next_start, _) = w[1];
            assert!(prev_end <= next_start, "feature histogram regions overlap");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_features(bin_counts: &[u32]) -> Vec<HistogramLayout> {
        let mut offset = 0;
        bin_counts
            .iter()
            .map(|&n_bins| {
                let meta = HistogramLayout { offset, n_bins };
                offset += n_bins;
                meta
            })
            .collect()
    }

    #[test]
    fn test_sequential_iteration() {
        let features = make_features(&[3, 4, 2]);
        let mut histogram = vec![(0.0, 0.0); 9];

        let iter = HistogramFeatureIter::new(&mut histogram, &features);
        iter.for_each_mut(&features, |f, slice| {
            for bin in slice.iter_mut() {
                bin.0 = f as f64;
            }
        });

        // Feature 0: bins 0-2
        assert_eq!(histogram[0].0, 0.0);
        assert_eq!(histogram[2].0, 0.0);
        // Feature 1: bins 3-6
        assert_eq!(histogram[3].0, 1.0);
        assert_eq!(histogram[6].0, 1.0);
        // Feature 2: bins 7-8
        assert_eq!(histogram[7].0, 2.0);
        assert_eq!(histogram[8].0, 2.0);
    }

    #[test]
    fn test_parallel_iteration() {
        let features = make_features(&[4, 4, 4, 4]);
        let mut histogram = vec![(0.0, 0.0); 16];

        let iter = HistogramFeatureIter::new(&mut histogram, &features);
        iter.par_for_each_mut(&features, |f, slice| {
            for bin in slice.iter_mut() {
                bin.1 = (f + 1) as f64;
            }
        });

        for (i, bin) in histogram.iter().enumerate() {
            let expected_feature = i / 4;
            assert_eq!(bin.1, (expected_feature + 1) as f64);
        }
    }
}
