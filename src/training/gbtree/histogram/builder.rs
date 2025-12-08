//! Histogram building and subtraction optimization.

use rayon::prelude::*;

use super::node::NodeHistogram;
use crate::training::gbtree::quantize::{BinIndex, QuantizedMatrix};

/// Builds histograms from quantized features and gradients.
///
/// The builder iterates over rows belonging to a node and accumulates
/// gradients into per-feature histograms based on bin assignments.
///
/// # Algorithm
///
/// 1. Reset histogram to zero
/// 2. For each row in the node:
///    - Look up gradient and hessian
///    - For each feature, add (grad, hess) to the corresponding bin
/// 3. Update cached totals
///
/// # Parallelization
///
/// Two parallelization strategies are available:
/// - [`build`](Self::build): Single-threaded, good baseline
/// - [`build_parallel`](Self::build_parallel): Per-feature parallelism
///
/// Per-feature parallelism works well because each feature histogram is
/// independent — no synchronization needed.
#[derive(Debug, Default, Clone, Copy)]
pub struct HistogramBuilder;

impl HistogramBuilder {
    /// Build histogram for a node from its row indices (single-threaded).
    ///
    /// # Arguments
    ///
    /// * `hist` - Histogram to fill (will be reset first)
    /// * `index` - Quantized feature matrix
    /// * `grads` - Gradient slice for all rows (length = n_samples)
    /// * `hess` - Hessian slice for all rows (length = n_samples)
    /// * `rows` - Row indices belonging to this node
    ///
    /// # Panics
    ///
    /// Panics if row indices are out of bounds.
    pub fn build<B: BinIndex>(
        &self,
        hist: &mut NodeHistogram,
        index: &QuantizedMatrix<B>,
        grads: &[f32],
        hess: &[f32],
        rows: &[u32],
    ) {
        debug_assert_eq!(grads.len(), hess.len());
        hist.reset();
        let num_features = hist.num_features();

        for &row in rows {
            let row_idx = row as usize;
            let grad = grads[row_idx];
            let hess_val = hess[row_idx];

            for feat in 0..num_features {
                let bin = index.get(row, feat as u32).to_usize();
                hist.feature_mut(feat).add(bin, grad, hess_val);
            }
        }

        hist.update_totals();
    }

    /// Build histogram with per-feature parallelism.
    ///
    /// Each feature histogram is built independently in parallel using Rayon.
    /// This is the preferred method for datasets with many features.
    ///
    /// # Arguments
    ///
    /// Same as [`build`](Self::build).
    pub fn build_parallel<B: BinIndex>(
        &self,
        hist: &mut NodeHistogram,
        index: &QuantizedMatrix<B>,
        grads: &[f32],
        hess: &[f32],
        rows: &[u32],
    ) {
        debug_assert_eq!(grads.len(), hess.len());
        hist.features_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(feat, feat_hist)| {
                feat_hist.reset();

                for (row, bin) in rows
                    .iter()
                    .zip(index.iter_rows_for_feature(feat as u32, rows))
                {
                    let row_idx = *row as usize;
                    feat_hist.add(bin.to_usize(), grads[row_idx], hess[row_idx]);
                }
            });

        hist.update_totals();
    }

    /// Build histogram using column iteration (cache-friendly).
    ///
    /// Processes one feature at a time, which is cache-friendly for
    /// column-major quantized matrices.
    pub fn build_column_wise<B: BinIndex>(
        &self,
        hist: &mut NodeHistogram,
        index: &QuantizedMatrix<B>,
        grads: &[f32],
        hess: &[f32],
        rows: &[u32],
    ) {
        debug_assert_eq!(grads.len(), hess.len());
        hist.reset();
        let num_features = hist.num_features();

        for feat in 0..num_features {
            let feat_hist = hist.feature_mut(feat);

            for (&row, bin) in rows
                .iter()
                .zip(index.iter_rows_for_feature(feat as u32, rows))
            {
                let row_idx = row as usize;
                feat_hist.add(bin.to_usize(), grads[row_idx], hess[row_idx]);
            }
        }

        hist.update_totals();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ColMatrix;
    use crate::training::gbtree::quantize::{ExactQuantileCuts, Quantizer};

    fn make_test_data() -> (QuantizedMatrix<u8>, Vec<f32>, Vec<f32>) {
        // Create simple data: 10 rows, 2 features
        let data: Vec<f32> = vec![
            // Feature 0: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
            // Feature 1: all same value
            5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
        ];
        let matrix = ColMatrix::from_vec(data, 10, 2);

        // Quantize
        let quantizer = Quantizer::from_data(&matrix, &ExactQuantileCuts::default(), 256);
        let quantized: QuantizedMatrix<u8> = quantizer.quantize(&matrix);

        // Create gradients: grad = row_id, hess = 1.0
        let grads: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let hess: Vec<f32> = vec![1.0; 10];

        (quantized, grads, hess)
    }

    #[test]
    fn test_histogram_builder_basic() {
        let (quantized, grads, hess) = make_test_data();
        let rows: Vec<u32> = (0..10).collect();

        let mut hist = NodeHistogram::new(quantized.cuts());
        HistogramBuilder.build(&mut hist, &quantized, &grads, &hess, &rows);

        // Total grad should be 0+1+2+...+9 = 45
        // Total hess should be 10 (each row contributes 1.0)
        assert_eq!(hist.total_count(), 10);
        assert!((hist.total_grad() - 45.0).abs() < 1e-5);
        assert!((hist.total_hess() - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_histogram_builder_subset() {
        let (quantized, grads, hess) = make_test_data();

        // Build histogram for subset of rows
        let rows: Vec<u32> = vec![0, 2, 4, 6, 8]; // Even rows

        let mut hist = NodeHistogram::new(quantized.cuts());
        HistogramBuilder.build(&mut hist, &quantized, &grads, &hess, &rows);

        // Grad sum: 0+2+4+6+8 = 20, hess: 5
        assert_eq!(hist.total_count(), 5);
        assert!((hist.total_grad() - 20.0).abs() < 1e-5);
        assert!((hist.total_hess() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_histogram_builder_parallel_matches_sequential() {
        let (quantized, grads, hess) = make_test_data();
        let rows: Vec<u32> = (0..10).collect();

        let mut hist_seq = NodeHistogram::new(quantized.cuts());
        HistogramBuilder.build(&mut hist_seq, &quantized, &grads, &hess, &rows);

        let mut hist_par = NodeHistogram::new(quantized.cuts());
        HistogramBuilder.build_parallel(&mut hist_par, &quantized, &grads, &hess, &rows);

        // Compare totals
        assert_eq!(hist_seq.total_count(), hist_par.total_count());
        assert!((hist_seq.total_grad() - hist_par.total_grad()).abs() < 1e-5);
        assert!((hist_seq.total_hess() - hist_par.total_hess()).abs() < 1e-5);

        // Compare per-feature histograms
        for feat in 0..hist_seq.num_features() {
            let seq = hist_seq.feature(feat);
            let par = hist_par.feature(feat);
            for bin in 0..seq.num_bins() as usize {
                let (sg, sh, sc) = seq.bin_stats(bin);
                let (pg, ph, pc) = par.bin_stats(bin);
                assert!(
                    (sg - pg).abs() < 1e-5,
                    "Feature {} bin {} grad mismatch",
                    feat,
                    bin
                );
                assert!(
                    (sh - ph).abs() < 1e-5,
                    "Feature {} bin {} hess mismatch",
                    feat,
                    bin
                );
                assert_eq!(sc, pc, "Feature {} bin {} count mismatch", feat, bin);
            }
        }
    }

    #[test]
    fn test_histogram_subtraction_correctness() {
        let (quantized, grads, hess) = make_test_data();

        // Build parent histogram (all rows)
        let all_rows: Vec<u32> = (0..10).collect();
        let mut parent = NodeHistogram::new(quantized.cuts());
        HistogramBuilder.build(&mut parent, &quantized, &grads, &hess, &all_rows);

        // Build left child histogram (first 6 rows)
        let left_rows: Vec<u32> = vec![0, 1, 2, 3, 4, 5];
        let mut left = NodeHistogram::new(quantized.cuts());
        HistogramBuilder.build(&mut left, &quantized, &grads, &hess, &left_rows);

        // Build right child directly for comparison
        let right_rows: Vec<u32> = vec![6, 7, 8, 9];
        let mut right_direct = NodeHistogram::new(quantized.cuts());
        HistogramBuilder.build(&mut right_direct, &quantized, &grads, &hess, &right_rows);

        // Compute right via subtraction: parent - left = right
        let right_subtracted = &parent - &left;

        // Compare
        assert_eq!(right_direct.total_count(), right_subtracted.total_count());
        assert!((right_direct.total_grad() - right_subtracted.total_grad()).abs() < 1e-5);
        assert!((right_direct.total_hess() - right_subtracted.total_hess()).abs() < 1e-5);

        // Verify actual values: right should have rows 6,7,8,9
        // Grad sum: 6+7+8+9 = 30
        assert!((right_subtracted.total_grad() - 30.0).abs() < 1e-5);
        assert_eq!(right_subtracted.total_count(), 4);
    }

    #[test]
    fn test_column_wise_matches_row_wise() {
        let (quantized, grads, hess) = make_test_data();
        let rows: Vec<u32> = (0..10).collect();

        let mut hist_row = NodeHistogram::new(quantized.cuts());
        HistogramBuilder.build(&mut hist_row, &quantized, &grads, &hess, &rows);

        let mut hist_col = NodeHistogram::new(quantized.cuts());
        HistogramBuilder.build_column_wise(&mut hist_col, &quantized, &grads, &hess, &rows);

        // Compare totals
        assert_eq!(hist_row.total_count(), hist_col.total_count());
        assert!((hist_row.total_grad() - hist_col.total_grad()).abs() < 1e-5);
        assert!((hist_row.total_hess() - hist_col.total_hess()).abs() < 1e-5);
    }
}
