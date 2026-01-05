//! Histogram building and operations.
//!
//! This module provides histogram building for gradient boosting tree training.
//! The main type is [`HistogramBuilder`] which orchestrates histogram construction
//! using feature-parallel or sequential strategies.
//!
//! # Design Philosophy
//!
//! - Simple `(f64, f64)` tuples for bins (no complex trait hierarchies)
//! - LLVM auto-vectorizes the scalar loops effectively
//! - Feature-parallel only (row-parallel was 2.8x slower due to merge overhead)
//! - The subtraction trick (sibling = parent - child) provides 10-44x speedup
//! - **Ordered gradients**: gradients are pre-gathered into partition order before
//!   histogram building, enabling sequential memory access instead of random access
//!
//! # Numeric Precision
//!
//! Histogram bins use `f64` for accumulation despite gradients being stored as `f32`.
//! This is intentional:
//! - Gain computation involves differences of large sums that can lose precision in f32
//! - Memory overhead is acceptable (histograms are small: typically 256 bins × features)
//!
//! # EFB (Exclusive Feature Bundling) Support
//!
//! When EFB is active, bundled columns store encoded bins that combine multiple
//! mutually exclusive features. During histogram building, we decode each encoded
//! bin to its original feature and accumulate into the correct feature's histogram.
//! This preserves the semantic integrity of each feature's gradient statistics.

use super::FeatureView;
use super::pool::HistogramLayout;
use super::slices::HistogramFeatureIter;
use crate::training::GradsTuple;
use crate::utils::Parallelism;

/// A histogram bin storing accumulated (gradient_sum, hessian_sum).
///
/// Uses `f64` for numerical stability in gain computation, even though source
/// gradients are `f32`. The subtraction trick means small differences between
/// large sums are common, which requires extra precision to avoid drift.
pub type HistogramBin = (f64, f64);

// =============================================================================
// HistogramBuilder
// =============================================================================

/// Minimum features to justify parallelizing over features.
const MIN_FEATURES_PARALLEL: usize = 4;

/// Minimum "work" per thread (rows × features) to amortize rayon scheduling.
const MIN_WORK_PER_THREAD: usize = 4096;

/// Builder for histogram construction.
///
/// Encapsulates parallel strategy selection and kernel dispatch for building
/// histograms from gradient data.
#[derive(Clone, Debug)]
pub struct HistogramBuilder {
    /// Parallelism hint (may be corrected per-build based on workload).
    parallelism: Parallelism,
}

impl Default for HistogramBuilder {
    fn default() -> Self {
        Self::new(Parallelism::Sequential)
    }
}

impl HistogramBuilder {
    /// Create a new histogram builder with the given parallelism hint.
    pub fn new(parallelism: Parallelism) -> Self {
        Self { parallelism }
    }

    /// Build histograms using gathered (pre-ordered) gradients.
    ///
    /// This is for nodes with non-contiguous row indices. Gradients have been
    /// pre-gathered into partition order for cache efficiency.
    pub fn build_gathered(
        &self,
        histogram: &mut [HistogramBin],
        ordered_grad_hess: &[GradsTuple],
        indices: &[u32],
        bin_views: &[FeatureView<'_>],
        feature_metas: &[HistogramLayout],
    ) {
        debug_assert_eq!(ordered_grad_hess.len(), indices.len());

        let parallelism = self.suggest_parallelism(indices.len(), feature_metas.len());
        let iter = HistogramFeatureIter::new(histogram, feature_metas);

        if parallelism.is_parallel() {
            iter.par_for_each_mut(feature_metas, |f, hist_slice| {
                build_feature_gathered(hist_slice, ordered_grad_hess, indices, &bin_views[f]);
            });
        } else {
            iter.for_each_mut(feature_metas, |f, hist_slice| {
                build_feature_gathered(hist_slice, ordered_grad_hess, indices, &bin_views[f]);
            });
        }
    }

    /// Build histograms for a contiguous row range.
    ///
    /// This is a fast path when the node's rows are exactly `[start, start+1, ..., start+len-1]`.
    pub fn build_contiguous(
        &self,
        histogram: &mut [HistogramBin],
        ordered_grad_hess: &[GradsTuple],
        start_row: usize,
        bin_views: &[FeatureView<'_>],
        feature_metas: &[HistogramLayout],
    ) {
        let n_rows = ordered_grad_hess.len();
        let parallelism = self.suggest_parallelism(n_rows, feature_metas.len());
        let iter = HistogramFeatureIter::new(histogram, feature_metas);

        if parallelism.is_parallel() {
            iter.par_for_each_mut(feature_metas, |f, hist_slice| {
                build_feature_contiguous(hist_slice, ordered_grad_hess, start_row, &bin_views[f]);
            });
        } else {
            iter.for_each_mut(feature_metas, |f, hist_slice| {
                build_feature_contiguous(hist_slice, ordered_grad_hess, start_row, &bin_views[f]);
            });
        }
    }

    /// Suggest parallelism based on workload (may downgrade to sequential).
    #[inline]
    fn suggest_parallelism(&self, n_rows: usize, n_features: usize) -> Parallelism {
        if !self.parallelism.is_parallel() {
            return Parallelism::Sequential;
        }
        if n_features < MIN_FEATURES_PARALLEL {
            return Parallelism::Sequential;
        }
        let work = n_rows.saturating_mul(n_features);
        if work < MIN_WORK_PER_THREAD {
            return Parallelism::Sequential;
        }
        self.parallelism
    }
}

// =============================================================================
// Single-Feature Kernels (Gathered)
// =============================================================================

/// Build histogram for a single feature using gathered gradients.
#[inline]
fn build_feature_gathered(
    histogram: &mut [HistogramBin],
    ordered_grad_hess: &[GradsTuple],
    indices: &[u32],
    view: &FeatureView<'_>,
) {
    match view {
        FeatureView::U8(bins) => {
            build_u8_gathered(bins, ordered_grad_hess, histogram, indices);
        }
        FeatureView::U16(bins) => {
            build_u16_gathered(bins, ordered_grad_hess, histogram, indices);
        }
        FeatureView::SparseU8 { .. } | FeatureView::SparseU16 { .. } => {
            // Sparse features not supported in gathered path
        }
    }
}

#[inline]
fn build_u8_gathered(
    bins: &[u8],
    ordered_grad_hess: &[GradsTuple],
    histogram: &mut [HistogramBin],
    indices: &[u32],
) {
    for i in 0..indices.len() {
        let row = unsafe { *indices.get_unchecked(i) } as usize;
        let bin = unsafe { *bins.get_unchecked(row) } as usize;
        let slot = unsafe { histogram.get_unchecked_mut(bin) };
        let gh = unsafe { *ordered_grad_hess.get_unchecked(i) };
        slot.0 += gh.grad as f64;
        slot.1 += gh.hess as f64;
    }
}

#[inline]
fn build_u16_gathered(
    bins: &[u16],
    ordered_grad_hess: &[GradsTuple],
    histogram: &mut [HistogramBin],
    indices: &[u32],
) {
    for i in 0..indices.len() {
        let row = unsafe { *indices.get_unchecked(i) } as usize;
        let bin = unsafe { *bins.get_unchecked(row) } as usize;
        let slot = unsafe { histogram.get_unchecked_mut(bin) };
        let gh = unsafe { *ordered_grad_hess.get_unchecked(i) };
        slot.0 += gh.grad as f64;
        slot.1 += gh.hess as f64;
    }
}

// =============================================================================
// Single-Feature Kernels (Contiguous)
// =============================================================================

/// Build histogram for a single feature using a contiguous row range.
#[inline]
fn build_feature_contiguous(
    histogram: &mut [HistogramBin],
    ordered_grad_hess: &[GradsTuple],
    start_row: usize,
    view: &FeatureView<'_>,
) {
    match view {
        FeatureView::U8(bins) => {
            build_u8_contiguous(bins, ordered_grad_hess, histogram, start_row);
        }
        FeatureView::U16(bins) => {
            build_u16_contiguous(bins, ordered_grad_hess, histogram, start_row);
        }
        FeatureView::SparseU8 {
            sample_indices,
            bin_values,
        } => {
            build_sparse_u8_contiguous(
                sample_indices,
                bin_values,
                ordered_grad_hess,
                histogram,
                start_row,
            );
        }
        FeatureView::SparseU16 {
            sample_indices,
            bin_values,
        } => {
            build_sparse_u16_contiguous(
                sample_indices,
                bin_values,
                ordered_grad_hess,
                histogram,
                start_row,
            );
        }
    }
}

#[inline]
fn build_u8_contiguous(
    bins: &[u8],
    ordered_grad_hess: &[GradsTuple],
    histogram: &mut [HistogramBin],
    start_row: usize,
) {
    for i in 0..ordered_grad_hess.len() {
        let row = start_row + i;
        let bin = unsafe { *bins.get_unchecked(row) } as usize;
        let gh = unsafe { *ordered_grad_hess.get_unchecked(i) };
        let slot = unsafe { histogram.get_unchecked_mut(bin) };
        slot.0 += gh.grad as f64;
        slot.1 += gh.hess as f64;
    }
}

#[inline]
fn build_u16_contiguous(
    bins: &[u16],
    ordered_grad_hess: &[GradsTuple],
    histogram: &mut [HistogramBin],
    start_row: usize,
) {
    for i in 0..ordered_grad_hess.len() {
        let row = start_row + i;
        let bin = unsafe { *bins.get_unchecked(row) } as usize;
        let gh = unsafe { *ordered_grad_hess.get_unchecked(i) };
        let slot = unsafe { histogram.get_unchecked_mut(bin) };
        slot.0 += gh.grad as f64;
        slot.1 += gh.hess as f64;
    }
}

#[inline]
fn build_sparse_u8_contiguous(
    sample_indices: &[u32],
    bin_values: &[u8],
    ordered_grad_hess: &[GradsTuple],
    histogram: &mut [HistogramBin],
    start_row: usize,
) {
    let start = start_row as u32;
    let end = start + ordered_grad_hess.len() as u32;
    // Binary search to find first index >= start
    let first = sample_indices.partition_point(|&r| r < start);
    for i in first..sample_indices.len() {
        let row = unsafe { *sample_indices.get_unchecked(i) };
        if row >= end {
            break; // Early exit when past range
        }
        let idx = (row - start) as usize;
        let bin = unsafe { *bin_values.get_unchecked(i) } as usize;
        let gh = unsafe { *ordered_grad_hess.get_unchecked(idx) };
        let slot = unsafe { histogram.get_unchecked_mut(bin) };
        slot.0 += gh.grad as f64;
        slot.1 += gh.hess as f64;
    }
}

#[inline]
fn build_sparse_u16_contiguous(
    row_indices: &[u32],
    bin_values: &[u16],
    ordered_grad_hess: &[GradsTuple],
    histogram: &mut [HistogramBin],
    start_row: usize,
) {
    let start = start_row as u32;
    let end = start + ordered_grad_hess.len() as u32;
    // Binary search to find first index >= start
    let first = row_indices.partition_point(|&r| r < start);
    for i in first..row_indices.len() {
        let row = unsafe { *row_indices.get_unchecked(i) };
        if row >= end {
            break; // Early exit when past range
        }
        let idx = (row - start) as usize;
        let bin = unsafe { *bin_values.get_unchecked(i) } as usize;
        let gh = unsafe { *ordered_grad_hess.get_unchecked(idx) };
        let slot = unsafe { histogram.get_unchecked_mut(bin) };
        slot.0 += gh.grad as f64;
        slot.1 += gh.hess as f64;
    }
}

// =============================================================================
// Histogram Operations
// =============================================================================

/// Subtract histograms: dst -= src
///
/// Used for the subtraction trick: sibling = parent - child
#[inline]
pub fn subtract_histogram(dst: &mut [HistogramBin], src: &[HistogramBin]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        d.0 -= s.0;
        d.1 -= s.1;
    }
}

/// Merge histograms: dst += src
#[inline]
pub fn merge_histogram(dst: &mut [HistogramBin], src: &[HistogramBin]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        d.0 += s.0;
        d.1 += s.1;
    }
}

/// Zero out a histogram.
#[inline]
pub fn clear_histogram(histogram: &mut [HistogramBin]) {
    histogram.fill((0.0, 0.0));
}

/// Sum all bins in a histogram.
#[inline]
pub fn sum_histogram(histogram: &[HistogramBin]) -> (f64, f64) {
    let mut g = 0.0;
    let mut h = 0.0;
    for &(grad, hess) in histogram {
        g += grad;
        h += hess;
    }
    (g, h)
}

// =============================================================================
// Tests
// =============================================================================

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
    fn test_build_histogram_basic() {
        let bins: Vec<u8> = vec![0, 1, 0, 2, 1, 0];
        let grad = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let hess = [0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0];
        let mut histogram = vec![(0.0, 0.0); 3];

        let features = make_features(&[3]);
        let bin_views = vec![FeatureView::U8(&bins)];
        let ordered_grad_hess: Vec<GradsTuple> = grad
            .iter()
            .zip(&hess)
            .map(|(&g, &h)| GradsTuple { grad: g, hess: h })
            .collect();

        let builder = HistogramBuilder::new(Parallelism::Sequential);
        builder.build_contiguous(&mut histogram, &ordered_grad_hess, 0, &bin_views, &features);

        assert!((histogram[0].0 - 10.0).abs() < 1e-10); // 1+3+6
        assert!((histogram[0].1 - 5.0).abs() < 1e-10); // 0.5+1.5+3
        assert!((histogram[1].0 - 7.0).abs() < 1e-10); // 2+5
        assert!((histogram[2].0 - 4.0).abs() < 1e-10); // 4
    }

    #[test]
    fn test_build_histogram_with_indices() {
        let bins: Vec<u8> = vec![0, 1, 2, 0, 1, 2];
        let grad = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let hess = [1.0; 6];
        let mut histogram = vec![(0.0, 0.0); 3];
        let indices: Vec<u32> = vec![0, 2, 4];

        let features = make_features(&[3]);
        let bin_views = vec![FeatureView::U8(&bins)];
        let ordered_grad_hess: Vec<GradsTuple> = indices
            .iter()
            .map(|&r| {
                let r = r as usize;
                GradsTuple {
                    grad: grad[r],
                    hess: hess[r],
                }
            })
            .collect();

        let builder = HistogramBuilder::new(Parallelism::Sequential);
        builder.build_gathered(
            &mut histogram,
            &ordered_grad_hess,
            &indices,
            &bin_views,
            &features,
        );

        assert!((histogram[0].0 - 1.0).abs() < 1e-10);
        assert!((histogram[1].0 - 5.0).abs() < 1e-10);
        assert!((histogram[2].0 - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_subtract_histogram() {
        let mut dst = vec![(10.0, 5.0), (20.0, 10.0)];
        let src = vec![(3.0, 2.0), (8.0, 4.0)];
        subtract_histogram(&mut dst, &src);
        assert!((dst[0].0 - 7.0).abs() < 1e-10);
        assert!((dst[1].0 - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_histogram() {
        let sample_indices: Vec<u32> = vec![0, 2, 4];
        let bin_values: Vec<u8> = vec![1, 0, 2];
        let grad = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let hess = [0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0];
        let mut histogram = vec![(0.0, 0.0); 3];

        let features = make_features(&[3]);
        let bin_views = vec![FeatureView::SparseU8 {
            sample_indices: &sample_indices,
            bin_values: &bin_values,
        }];
        let ordered_grad_hess: Vec<GradsTuple> = grad
            .iter()
            .zip(&hess)
            .map(|(&g, &h)| GradsTuple { grad: g, hess: h })
            .collect();

        let builder = HistogramBuilder::new(Parallelism::Sequential);
        builder.build_contiguous(&mut histogram, &ordered_grad_hess, 0, &bin_views, &features);

        assert!((histogram[0].0 - 3.0).abs() < 1e-10); // row 2
        assert!((histogram[1].0 - 1.0).abs() < 1e-10); // row 0
        assert!((histogram[2].0 - 5.0).abs() < 1e-10); // row 4
    }

    #[test]
    fn test_build_histograms_multi_feature() {
        let features = make_features(&[4, 4, 4]);
        let n_samples = 100;
        let bins_f0: Vec<u8> = (0..n_samples).map(|i| (i % 4) as u8).collect();
        let bins_f1: Vec<u8> = (0..n_samples).map(|i| ((i + 1) % 4) as u8).collect();
        let bins_f2: Vec<u8> = (0..n_samples).map(|i| ((i + 2) % 4) as u8).collect();

        let bin_views = vec![
            FeatureView::U8(&bins_f0),
            FeatureView::U8(&bins_f1),
            FeatureView::U8(&bins_f2),
        ];

        let grad: Vec<f32> = (0..n_samples).map(|i| i as f32).collect();
        let hess: Vec<f32> = vec![1.0; n_samples];
        let mut histogram = vec![(0.0, 0.0); 12];

        let ordered_grad_hess: Vec<GradsTuple> = grad
            .iter()
            .zip(&hess)
            .map(|(&g, &h)| GradsTuple { grad: g, hess: h })
            .collect();

        let builder = HistogramBuilder::new(Parallelism::Sequential);
        builder.build_contiguous(&mut histogram, &ordered_grad_hess, 0, &bin_views, &features);

        let f0_bin0_grad: f64 = (0..n_samples)
            .filter(|i| i % 4 == 0)
            .map(|i| i as f64)
            .sum();
        assert!((histogram[0].0 - f0_bin0_grad).abs() < 1e-10);
    }

    #[test]
    fn test_build_histograms_gathered_matches_naive() {
        let features = make_features(&[4, 4]);
        let n_samples = 128;
        let bins_f0: Vec<u8> = (0..n_samples).map(|i| (i % 4) as u8).collect();
        let bins_f1: Vec<u8> = (0..n_samples).map(|i| ((i + 1) % 4) as u8).collect();

        let bin_views = vec![FeatureView::U8(&bins_f0), FeatureView::U8(&bins_f1)];

        let indices: Vec<u32> = (0..n_samples as u32).step_by(3).collect();

        // Naive reference
        let mut hist_ref = [(0.0, 0.0); 8];
        for &row_u32 in &indices {
            let row = row_u32 as usize;
            let g = (row as f32 * 0.25) as f64;
            let h = (1.0f32 + (row as f32) * 0.01) as f64;

            let b0 = bins_f0[row] as usize;
            hist_ref[b0].0 += g;
            hist_ref[b0].1 += h;

            let b1 = bins_f1[row] as usize;
            hist_ref[4 + b1].0 += g;
            hist_ref[4 + b1].1 += h;
        }

        let ordered_interleaved: Vec<GradsTuple> = indices
            .iter()
            .map(|&r| {
                let r = r as f32;
                GradsTuple {
                    grad: r * 0.25,
                    hess: 1.0 + r * 0.01,
                }
            })
            .collect();

        let mut hist = vec![(0.0, 0.0); 8];
        let builder = HistogramBuilder::new(Parallelism::Sequential);
        builder.build_gathered(
            &mut hist,
            &ordered_interleaved,
            &indices,
            &bin_views,
            &features,
        );

        for i in 0..hist.len() {
            assert!((hist_ref[i].0 - hist[i].0).abs() < 1e-10);
            assert!((hist_ref[i].1 - hist[i].1).abs() < 1e-10);
        }
    }

    #[test]
    fn test_parallel_build() {
        let features = make_features(&[4, 4, 4, 4]);
        let n_samples = 10000;
        let bins: Vec<u8> = (0..n_samples).map(|i| (i % 4) as u8).collect();

        let bin_views: Vec<_> = (0..4).map(|_| FeatureView::U8(&bins)).collect();

        let ordered_grad_hess: Vec<GradsTuple> = (0..n_samples)
            .map(|i| GradsTuple {
                grad: i as f32,
                hess: 1.0,
            })
            .collect();

        // Sequential
        let mut hist_seq = vec![(0.0, 0.0); 16];
        let builder_seq = HistogramBuilder::new(Parallelism::Sequential);
        builder_seq.build_contiguous(&mut hist_seq, &ordered_grad_hess, 0, &bin_views, &features);

        // Parallel
        let mut hist_par = vec![(0.0, 0.0); 16];
        let builder_par = HistogramBuilder::new(Parallelism::Parallel);
        builder_par.build_contiguous(&mut hist_par, &ordered_grad_hess, 0, &bin_views, &features);

        // Should match
        for (s, p) in hist_seq.iter().zip(&hist_par) {
            assert!((s.0 - p.0).abs() < 1e-10);
            assert!((s.1 - p.1).abs() < 1e-10);
        }
    }
}
