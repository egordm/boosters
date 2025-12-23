//! Binned data accessor for tree traversal.
//!
//! Provides [`BinnedAccessor`] which converts binned data to midpoint values
//! for tree traversal comparisons.

use super::{BinMapper, BinnedDataset};
use crate::data::FeatureAccessor;

// ============================================================================
// BinnedAccessor
// ============================================================================

/// Feature accessor that converts binned data to midpoint values.
///
/// For each bin, returns the midpoint between lower and upper bounds:
/// - Bin 0: `(min_val + upper_bound[0]) / 2`
/// - Bin n: `(upper_bound[n-1] + upper_bound[n]) / 2`
///
/// For categorical features, returns the category value as f32.
/// For missing bins, returns `f32::NAN`.
///
/// # Panics
///
/// The constructor panics if `bin_mappers.len() != dataset.n_features()`.
pub struct BinnedAccessor<'a> {
    dataset: &'a BinnedDataset,
    bin_mappers: &'a [BinMapper],
}

impl<'a> BinnedAccessor<'a> {
    /// Create a new binned accessor.
    ///
    /// # Panics
    ///
    /// Panics if the number of bin mappers doesn't match the dataset's feature count.
    pub fn new(dataset: &'a BinnedDataset, bin_mappers: &'a [BinMapper]) -> Self {
        assert_eq!(
            dataset.n_features(),
            bin_mappers.len(),
            "BinMapper count ({}) must match feature count ({})",
            bin_mappers.len(),
            dataset.n_features()
        );
        Self {
            dataset,
            bin_mappers,
        }
    }
}

impl FeatureAccessor for BinnedAccessor<'_> {
    #[inline]
    fn get_feature(&self, row: usize, feature: usize) -> f32 {
        let bin = self.dataset.get_bin(row, feature);
        match bin {
            Some(b) => self.bin_mappers[feature].bin_to_midpoint(b) as f32,
            None => f32::NAN,
        }
    }

    #[inline]
    fn n_rows(&self) -> usize {
        self.dataset.n_rows()
    }

    #[inline]
    fn n_features(&self) -> usize {
        self.dataset.n_features()
    }
}
