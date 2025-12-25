//! Binned data accessor for tree traversal.
//!
//! BinnedDataset implements [`DataAccessor`] directly, converting bin indices
//! to midpoint values for tree traversal comparisons.

use super::BinnedDataset;
use crate::data::{DataAccessor, SampleAccessor};

// ============================================================================
// BinnedSample
// ============================================================================

/// A single sample from binned data, implementing `SampleAccessor`.
///
/// Converts bin indices to midpoint values on-demand for tree traversal.
/// This is a lightweight view that holds references to the underlying data.
pub struct BinnedSample<'a> {
    dataset: &'a BinnedDataset,
    row: usize,
}

impl SampleAccessor for BinnedSample<'_> {
    #[inline]
    fn feature(&self, index: usize) -> f32 {
        let bin = self.dataset.get_bin(self.row, index);
        match bin {
            Some(b) => self.dataset.bin_mapper(index).bin_to_midpoint(b) as f32,
            None => f32::NAN,
        }
    }

    #[inline]
    fn n_features(&self) -> usize {
        self.dataset.n_features()
    }
}

// ============================================================================
// DataAccessor for BinnedDataset
// ============================================================================

impl DataAccessor for BinnedDataset {
    type Sample<'a> = BinnedSample<'a> where Self: 'a;

    #[inline]
    fn sample(&self, index: usize) -> Self::Sample<'_> {
        BinnedSample {
            dataset: self,
            row: index,
        }
    }

    #[inline]
    fn n_samples(&self) -> usize {
        self.n_rows()
    }

    #[inline]
    fn n_features(&self) -> usize {
        BinnedDataset::n_features(self)
    }
}
