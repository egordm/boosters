//! Quantized feature matrix.

use std::sync::Arc;

use super::cuts::{BinCuts, BinIndex};

/// Quantized feature matrix storing bin indices.
///
/// Stored in **column-major** order for efficient histogram building:
/// iterating rows for a single feature is contiguous memory access.
///
/// # Type Parameter
///
/// `B` controls bin index width:
/// - `u8`: Up to 256 bins per feature (default, 1 byte per cell)
/// - `u16`: Up to 65536 bins (2 bytes per cell)
/// - `u32`: Unlimited bins (4 bytes per cell)
///
/// # Memory Layout
///
/// ```text
/// For 4 rows × 3 features:
///
/// index: [r0f0, r1f0, r2f0, r3f0,   ← Feature 0 column (contiguous)
///         r0f1, r1f1, r2f1, r3f1,   ← Feature 1 column
///         r0f2, r1f2, r2f2, r3f2]   ← Feature 2 column
/// ```
#[derive(Debug, Clone)]
pub struct QuantizedMatrix<B: BinIndex = u8> {
    /// Bin indices in column-major layout: index[col * num_rows + row]
    index: Box<[B]>,

    /// Number of rows.
    num_rows: u32,

    /// Number of features.
    num_features: u32,

    /// Reference to the bin cuts used for quantization.
    cuts: Arc<BinCuts>,
}

impl<B: BinIndex> QuantizedMatrix<B> {
    /// Create a new quantized matrix.
    ///
    /// # Arguments
    ///
    /// * `index` - Bin indices in column-major layout
    /// * `num_rows` - Number of rows
    /// * `num_features` - Number of features
    /// * `cuts` - Bin cuts used for quantization
    pub fn new(index: Vec<B>, num_rows: u32, num_features: u32, cuts: Arc<BinCuts>) -> Self {
        assert_eq!(
            index.len(),
            (num_rows as usize) * (num_features as usize),
            "Index length must equal num_rows * num_features"
        );

        Self {
            index: index.into_boxed_slice(),
            num_rows,
            num_features,
            cuts,
        }
    }

    /// Number of rows.
    #[inline]
    pub fn num_rows(&self) -> u32 {
        self.num_rows
    }

    /// Number of features.
    #[inline]
    pub fn num_features(&self) -> u32 {
        self.num_features
    }

    /// Get the bin cuts used for quantization.
    #[inline]
    pub fn cuts(&self) -> &BinCuts {
        &self.cuts
    }

    /// Get bin index for a specific cell.
    #[inline]
    pub fn get(&self, row: u32, feature: u32) -> B {
        let idx = (feature as usize) * (self.num_rows as usize) + (row as usize);
        self.index[idx]
    }

    /// Get all bin indices for a feature (contiguous slice).
    ///
    /// This is the primary access pattern for histogram building.
    #[inline]
    pub fn feature_column(&self, feature: u32) -> &[B] {
        let start = (feature as usize) * (self.num_rows as usize);
        let end = start + self.num_rows as usize;
        &self.index[start..end]
    }

    /// Iterate over bin indices for specific rows of a feature.
    ///
    /// Used for histogram building on a subset of rows (e.g., rows in a tree node).
    #[inline]
    pub fn iter_rows_for_feature<'a>(
        &'a self,
        feature: u32,
        rows: &'a [u32],
    ) -> impl Iterator<Item = B> + 'a {
        let col = self.feature_column(feature);
        rows.iter().map(move |&row| col[row as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create BinCuts for testing
    fn make_cuts(features: &[&[f32]]) -> BinCuts {
        let mut cut_values = Vec::new();
        let mut cut_ptrs = vec![0u32];

        for cuts in features {
            cut_values.extend(*cuts);
            cut_ptrs.push(cut_values.len() as u32);
        }

        BinCuts::new(cut_values, cut_ptrs)
    }

    #[test]
    fn test_quantized_matrix_layout() {
        let cuts = Arc::new(make_cuts(&[&[0.5], &[10.0]]));

        // 3 rows, 2 features, column-major
        // Feature 0: [1, 2, 1] (bins for values < 0.5, > 0.5, < 0.5)
        // Feature 1: [1, 2, 1] (bins for values < 10, > 10, < 10)
        let index = vec![
            1u8, 2, 1, // Feature 0 column
            1, 2, 1, // Feature 1 column
        ];

        let qm = QuantizedMatrix::new(index, 3, 2, cuts);

        assert_eq!(qm.num_rows(), 3);
        assert_eq!(qm.num_features(), 2);

        // Test get()
        assert_eq!(qm.get(0, 0), 1);
        assert_eq!(qm.get(1, 0), 2);
        assert_eq!(qm.get(2, 0), 1);
        assert_eq!(qm.get(0, 1), 1);
        assert_eq!(qm.get(1, 1), 2);
        assert_eq!(qm.get(2, 1), 1);

        // Test feature_column()
        assert_eq!(qm.feature_column(0), &[1, 2, 1]);
        assert_eq!(qm.feature_column(1), &[1, 2, 1]);
    }

    #[test]
    fn test_iter_rows_for_feature() {
        let cuts = Arc::new(make_cuts(&[&[0.5]]));
        let index = vec![1u8, 2, 1, 2, 1]; // 5 rows, 1 feature
        let qm = QuantizedMatrix::new(index, 5, 1, cuts);

        // Iterate over subset of rows
        let rows = vec![0, 2, 4];
        let bins: Vec<u8> = qm.iter_rows_for_feature(0, &rows).collect();
        assert_eq!(bins, vec![1, 1, 1]);

        let rows = vec![1, 3];
        let bins: Vec<u8> = qm.iter_rows_for_feature(0, &rows).collect();
        assert_eq!(bins, vec![2, 2]);
    }
}
