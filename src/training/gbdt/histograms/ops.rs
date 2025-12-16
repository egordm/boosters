//! Histogram building and operations.
//!
//! This module provides histogram building functions for gradient boosting tree training.
//! The main entry point is [`build_histograms_ordered_interleaved`] which uses pre-gathered gradients
//! for optimal cache efficiency, following LightGBM's "ordered gradients" technique.
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
//! - Benchmarks showed f32→f64 quantization overhead outweighed memory bandwidth savings
//!

use crate::data::binned::FeatureView;
use crate::training::GradHessF32;
use super::pool::FeatureMeta;

/// A histogram bin storing accumulated (gradient_sum, hessian_sum).
///
/// Uses `f64` for numerical stability in gain computation, even though source
/// gradients are `f32`. The subtraction trick means small differences between
/// large sums are common, which requires extra precision to avoid drift.
pub type HistogramBin = (f64, f64);

// =============================================================================
// Parallel Strategy
// =============================================================================

/// Parallelization strategy for histogram building.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ParallelStrategy {
    /// Sequential processing (no parallelism).
    #[default]
    Sequential,
    /// Parallelize over features (each thread handles different features).
    FeatureParallel,
}

impl ParallelStrategy {
    /// Select the best strategy based on data characteristics.
    ///
    /// Key thresholds tuned based on benchmarks:
    /// - For very small node counts, sequential is faster (avoids thread spawn overhead)
    /// - Feature-parallel kicks in with sufficient features to amortize overhead
    pub fn auto_select(n_rows: usize, n_features: usize, n_threads: usize) -> Self {
        // Minimum rows before parallelism is worthwhile
        const MIN_ROWS_PARALLEL: usize = 1024;
        
        // Minimum features to justify parallelizing over features
        const MIN_FEATURES_PARALLEL: usize = 4;

        if n_rows < MIN_ROWS_PARALLEL || n_threads <= 1 {
            return Self::Sequential;
        }

        if n_features >= MIN_FEATURES_PARALLEL {
            Self::FeatureParallel
        } else {
            Self::Sequential
        }
    }
}

// =============================================================================
// Main Entry Point
// =============================================================================

/// Build histograms using ordered gradients stored as interleaved `(grad, hess)` pairs.
///
/// This uses a single sequential stream for gradient+hessian reads.
pub fn build_histograms_ordered_interleaved(
    histogram: &mut [HistogramBin],
    ordered_grad_hess: &[GradHessF32],
    indices: &[u32],
    bin_views: &[FeatureView<'_>],
    feature_metas: &[FeatureMeta],
) {
    debug_assert_eq!(ordered_grad_hess.len(), indices.len());

    let n_rows = indices.len();
    let n_features = feature_metas.len();
    let n_threads = rayon::current_num_threads();

    let strategy = ParallelStrategy::auto_select(n_rows, n_features, n_threads);

    match strategy {
        ParallelStrategy::Sequential => {
            for (f, meta) in feature_metas.iter().enumerate() {
                let offset = meta.offset as usize;
                let n_bins = meta.n_bins as usize;
                let hist_slice = &mut histogram[offset..offset + n_bins];
                build_feature_ordered_interleaved(
                    hist_slice,
                    ordered_grad_hess,
                    indices,
                    &bin_views[f],
                );
            }
        }
        ParallelStrategy::FeatureParallel => {
            use rayon::prelude::*;
            // SAFETY: Each feature writes to disjoint histogram regions
            feature_metas.par_iter().enumerate().for_each(|(f, meta)| {
                let offset = meta.offset as usize;
                let n_bins = meta.n_bins as usize;
                let hist_slice = unsafe {
                    let ptr = histogram.as_ptr().add(offset) as *mut HistogramBin;
                    std::slice::from_raw_parts_mut(ptr, n_bins)
                };
                build_feature_ordered_interleaved(
                    hist_slice,
                    ordered_grad_hess,
                    indices,
                    &bin_views[f],
                );
            });
        }
    }
}

/// Build histograms using interleaved ordered gradients for a strictly sequential row range.
///
/// This is a specialized, lossless fast path for the case where the node's row indices are
/// exactly `[start, start+1, ..., start+len-1]`.
pub fn build_histograms_ordered_sequential_interleaved(
    histogram: &mut [HistogramBin],
    ordered_grad_hess: &[GradHessF32],
    start_row: usize,
    bin_views: &[FeatureView<'_>],
    feature_metas: &[FeatureMeta],
) {
    let n_rows = ordered_grad_hess.len();
    let n_features = feature_metas.len();
    let n_threads = rayon::current_num_threads();

    let strategy = ParallelStrategy::auto_select(n_rows, n_features, n_threads);

    match strategy {
        ParallelStrategy::Sequential => {
            for (f, meta) in feature_metas.iter().enumerate() {
                let offset = meta.offset as usize;
                let n_bins = meta.n_bins as usize;
                let hist_slice = &mut histogram[offset..offset + n_bins];
                build_feature_ordered_sequential_interleaved(
                    hist_slice,
                    ordered_grad_hess,
                    start_row,
                    &bin_views[f],
                );
            }
        }
        ParallelStrategy::FeatureParallel => {
            use rayon::prelude::*;
            feature_metas.par_iter().enumerate().for_each(|(f, meta)| {
                let offset = meta.offset as usize;
                let n_bins = meta.n_bins as usize;
                let hist_slice = unsafe {
                    let ptr = histogram.as_ptr().add(offset) as *mut HistogramBin;
                    std::slice::from_raw_parts_mut(ptr, n_bins)
                };
                build_feature_ordered_sequential_interleaved(
                    hist_slice,
                    ordered_grad_hess,
                    start_row,
                    &bin_views[f],
                );
            });
        }
    }
}

// =============================================================================
// Single-Feature Building
// =============================================================================

/// Build histogram for a single feature using interleaved ordered gradients.
#[inline]
fn build_feature_ordered_interleaved(
    histogram: &mut [HistogramBin],
    ordered_grad_hess: &[GradHessF32],
    indices: &[u32],
    view: &FeatureView<'_>,
) {
    match view {
        FeatureView::U8 { bins, stride: 1 } => {
            build_histogram_u8_ordered_interleaved(bins, ordered_grad_hess, histogram, indices);
        }
        FeatureView::U16 { bins, stride: 1 } => {
            build_histogram_u16_ordered_interleaved(bins, ordered_grad_hess, histogram, indices);
        }
        FeatureView::U8 { bins, stride } => {
            build_histogram_strided_u8_ordered_interleaved(
                bins,
                *stride,
                ordered_grad_hess,
                histogram,
                indices,
            );
        }
        FeatureView::U16 { bins, stride } => {
            build_histogram_strided_u16_ordered_interleaved(
                bins,
                *stride,
                ordered_grad_hess,
                histogram,
                indices,
            );
        }
        // Sparse features: ordered gradients are not currently supported in the ordered path.
        FeatureView::SparseU8 { row_indices, bin_values } => {
            let _ = (row_indices, bin_values, ordered_grad_hess);
        }
        FeatureView::SparseU16 { row_indices, bin_values } => {
            let _ = (row_indices, bin_values, ordered_grad_hess);
        }
    }
}

#[inline]
fn build_histogram_u8_ordered_interleaved(
    bins: &[u8],
    ordered_grad_hess: &[GradHessF32],
    histogram: &mut [HistogramBin],
    indices: &[u32],
) {
    debug_assert_eq!(ordered_grad_hess.len(), indices.len());
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
fn build_histogram_u16_ordered_interleaved(
    bins: &[u16],
    ordered_grad_hess: &[GradHessF32],
    histogram: &mut [HistogramBin],
    indices: &[u32],
) {
    debug_assert_eq!(ordered_grad_hess.len(), indices.len());
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
fn build_histogram_strided_u8_ordered_interleaved(
    bins: &[u8],
    stride: usize,
    ordered_grad_hess: &[GradHessF32],
    histogram: &mut [HistogramBin],
    indices: &[u32],
) {
    debug_assert_eq!(ordered_grad_hess.len(), indices.len());
    for i in 0..indices.len() {
        let row = unsafe { *indices.get_unchecked(i) } as usize;
        let bin = unsafe { *bins.get_unchecked(row * stride) } as usize;
        let slot = unsafe { histogram.get_unchecked_mut(bin) };
        let gh = unsafe { *ordered_grad_hess.get_unchecked(i) };
        slot.0 += gh.grad as f64;
        slot.1 += gh.hess as f64;
    }
}

#[inline]
fn build_histogram_strided_u16_ordered_interleaved(
    bins: &[u16],
    stride: usize,
    ordered_grad_hess: &[GradHessF32],
    histogram: &mut [HistogramBin],
    indices: &[u32],
) {
    debug_assert_eq!(ordered_grad_hess.len(), indices.len());
    for i in 0..indices.len() {
        let row = unsafe { *indices.get_unchecked(i) } as usize;
        let bin = unsafe { *bins.get_unchecked(row * stride) } as usize;
        let slot = unsafe { histogram.get_unchecked_mut(bin) };
        let gh = unsafe { *ordered_grad_hess.get_unchecked(i) };
        slot.0 += gh.grad as f64;
        slot.1 += gh.hess as f64;
    }
}

/// Build histogram for a single feature using interleaved ordered gradients and a sequential row range.
#[inline]
fn build_feature_ordered_sequential_interleaved(
    histogram: &mut [HistogramBin],
    ordered_grad_hess: &[GradHessF32],
    start_row: usize,
    view: &FeatureView<'_>,
) {
    match view {
        FeatureView::U8 { bins, stride: 1 } => {
            build_histogram_u8_ordered_sequential_interleaved(
                bins,
                ordered_grad_hess,
                histogram,
                start_row,
            );
        }
        FeatureView::U16 { bins, stride: 1 } => {
            build_histogram_u16_ordered_sequential_interleaved(
                bins,
                ordered_grad_hess,
                histogram,
                start_row,
            );
        }
        FeatureView::U8 { bins, stride } => {
            build_histogram_strided_u8_ordered_sequential_interleaved(
                bins,
                *stride,
                ordered_grad_hess,
                histogram,
                start_row,
            );
        }
        FeatureView::U16 { bins, stride } => {
            build_histogram_strided_u16_ordered_sequential_interleaved(
                bins,
                *stride,
                ordered_grad_hess,
                histogram,
                start_row,
            );
        }
        FeatureView::SparseU8 { row_indices, bin_values } => {
            build_histogram_sparse_u8_ordered_sequential_interleaved(
                row_indices,
                bin_values,
                ordered_grad_hess,
                histogram,
                start_row,
            );
        }
        FeatureView::SparseU16 { row_indices, bin_values } => {
            build_histogram_sparse_u16_ordered_sequential_interleaved(
                row_indices,
                bin_values,
                ordered_grad_hess,
                histogram,
                start_row,
            );
        }
    }
}

#[inline]
fn build_histogram_u8_ordered_sequential_interleaved(
    bins: &[u8],
    ordered_grad_hess: &[GradHessF32],
    histogram: &mut [HistogramBin],
    start_row: usize,
) {
    let mut row = start_row;
    for i in 0..ordered_grad_hess.len() {
        debug_assert!(row < bins.len());
        let bin = unsafe { *bins.get_unchecked(row) } as usize;
        debug_assert!(bin < histogram.len());
        let slot = unsafe { histogram.get_unchecked_mut(bin) };
        let gh = unsafe { *ordered_grad_hess.get_unchecked(i) };
        slot.0 += gh.grad as f64;
        slot.1 += gh.hess as f64;
        row += 1;
    }
}

#[inline]
fn build_histogram_u16_ordered_sequential_interleaved(
    bins: &[u16],
    ordered_grad_hess: &[GradHessF32],
    histogram: &mut [HistogramBin],
    start_row: usize,
) {
    let mut row = start_row;
    for i in 0..ordered_grad_hess.len() {
        debug_assert!(row < bins.len());
        let bin = unsafe { *bins.get_unchecked(row) } as usize;
        debug_assert!(bin < histogram.len());
        let slot = unsafe { histogram.get_unchecked_mut(bin) };
        let gh = unsafe { *ordered_grad_hess.get_unchecked(i) };
        slot.0 += gh.grad as f64;
        slot.1 += gh.hess as f64;
        row += 1;
    }
}

#[inline]
fn build_histogram_strided_u8_ordered_sequential_interleaved(
    bins: &[u8],
    stride: usize,
    ordered_grad_hess: &[GradHessF32],
    histogram: &mut [HistogramBin],
    start_row: usize,
) {
    let mut row = start_row;
    for i in 0..ordered_grad_hess.len() {
        let bin_idx = row * stride;
        debug_assert!(bin_idx < bins.len());
        let bin = unsafe { *bins.get_unchecked(bin_idx) } as usize;
        debug_assert!(bin < histogram.len());
        let slot = unsafe { histogram.get_unchecked_mut(bin) };
        let gh = unsafe { *ordered_grad_hess.get_unchecked(i) };
        slot.0 += gh.grad as f64;
        slot.1 += gh.hess as f64;
        row += 1;
    }
}

#[inline]
fn build_histogram_strided_u16_ordered_sequential_interleaved(
    bins: &[u16],
    stride: usize,
    ordered_grad_hess: &[GradHessF32],
    histogram: &mut [HistogramBin],
    start_row: usize,
) {
    let mut row = start_row;
    for i in 0..ordered_grad_hess.len() {
        let bin_idx = row * stride;
        debug_assert!(bin_idx < bins.len());
        let bin = unsafe { *bins.get_unchecked(bin_idx) } as usize;
        debug_assert!(bin < histogram.len());
        let slot = unsafe { histogram.get_unchecked_mut(bin) };
        let gh = unsafe { *ordered_grad_hess.get_unchecked(i) };
        slot.0 += gh.grad as f64;
        slot.1 += gh.hess as f64;
        row += 1;
    }
}

#[inline]
fn build_histogram_sparse_u8_ordered_sequential_interleaved(
    row_indices: &[u32],
    bin_values: &[u8],
    ordered_grad_hess: &[GradHessF32],
    histogram: &mut [HistogramBin],
    start_row: usize,
) {
    let start = start_row as u32;
    let end = start + ordered_grad_hess.len() as u32;
    for (i, &row) in row_indices.iter().enumerate() {
        if row < start || row >= end {
            continue;
        }
        let idx = (row - start) as usize;
        let bin = bin_values[i] as usize;
        let gh = unsafe { *ordered_grad_hess.get_unchecked(idx) };
        histogram[bin].0 += gh.grad as f64;
        histogram[bin].1 += gh.hess as f64;
    }
}

#[inline]
fn build_histogram_sparse_u16_ordered_sequential_interleaved(
    row_indices: &[u32],
    bin_values: &[u16],
    ordered_grad_hess: &[GradHessF32],
    histogram: &mut [HistogramBin],
    start_row: usize,
) {
    let start = start_row as u32;
    let end = start + ordered_grad_hess.len() as u32;
    for (i, &row) in row_indices.iter().enumerate() {
        if row < start || row >= end {
            continue;
        }
        let idx = (row - start) as usize;
        let bin = bin_values[i] as usize;
        let gh = unsafe { *ordered_grad_hess.get_unchecked(idx) };
        histogram[bin].0 += gh.grad as f64;
        histogram[bin].1 += gh.hess as f64;
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

    fn make_features(bin_counts: &[u32]) -> Vec<FeatureMeta> {
        let mut offset = 0;
        bin_counts
            .iter()
            .map(|&n_bins| {
                let meta = FeatureMeta { offset, n_bins };
                offset += n_bins;
                meta
            })
            .collect()
    }

    #[test]
    fn test_parallel_strategy_auto_select() {
        assert_eq!(ParallelStrategy::auto_select(100, 10, 4), ParallelStrategy::Sequential);
        assert_eq!(ParallelStrategy::auto_select(10000, 100, 4), ParallelStrategy::FeatureParallel);
        assert_eq!(ParallelStrategy::auto_select(10000, 100, 1), ParallelStrategy::Sequential);
        assert_eq!(ParallelStrategy::auto_select(10000, 2, 4), ParallelStrategy::Sequential);
    }

    #[test]
    fn test_build_histogram_basic() {
        let bins: Vec<u8> = vec![0, 1, 0, 2, 1, 0];
        let grad = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let hess = vec![0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0];
        let mut histogram = vec![(0.0, 0.0); 3];

        let features = make_features(&[3]);
        let bin_views = vec![FeatureView::U8 { bins: &bins, stride: 1 }];
        let ordered_grad_hess: Vec<GradHessF32> = grad
            .iter()
            .zip(&hess)
            .map(|(&g, &h)| GradHessF32 { grad: g, hess: h })
            .collect();

        build_histograms_ordered_sequential_interleaved(
            &mut histogram,
            &ordered_grad_hess,
            0,
            &bin_views,
            &features,
        );

        assert!((histogram[0].0 - 10.0).abs() < 1e-10); // 1+3+6
        assert!((histogram[0].1 - 5.0).abs() < 1e-10);  // 0.5+1.5+3
        assert!((histogram[1].0 - 7.0).abs() < 1e-10);  // 2+5
        assert!((histogram[2].0 - 4.0).abs() < 1e-10);  // 4
    }

    #[test]
    fn test_build_histogram_with_indices() {
        let bins: Vec<u8> = vec![0, 1, 2, 0, 1, 2];
        let grad = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let hess = vec![1.0; 6];
        let mut histogram = vec![(0.0, 0.0); 3];
        let indices: Vec<u32> = vec![0, 2, 4];

        let features = make_features(&[3]);
        let bin_views = vec![FeatureView::U8 { bins: &bins, stride: 1 }];
        let ordered_grad_hess: Vec<GradHessF32> = indices
            .iter()
            .map(|&r| {
                let r = r as usize;
                GradHessF32 {
                    grad: grad[r],
                    hess: hess[r],
                }
            })
            .collect();

        build_histograms_ordered_interleaved(
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
        let row_indices: Vec<u32> = vec![0, 2, 4];
        let bin_values: Vec<u8> = vec![1, 0, 2];
        let grad = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let hess = vec![0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0];
        let mut histogram = vec![(0.0, 0.0); 3];

        let features = make_features(&[3]);
        let bin_views = vec![FeatureView::SparseU8 {
            row_indices: &row_indices,
            bin_values: &bin_values,
        }];
        let ordered_grad_hess: Vec<GradHessF32> = grad
            .iter()
            .zip(&hess)
            .map(|(&g, &h)| GradHessF32 { grad: g, hess: h })
            .collect();

        build_histograms_ordered_sequential_interleaved(
            &mut histogram,
            &ordered_grad_hess,
            0,
            &bin_views,
            &features,
        );

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
            FeatureView::U8 { bins: &bins_f0, stride: 1 },
            FeatureView::U8 { bins: &bins_f1, stride: 1 },
            FeatureView::U8 { bins: &bins_f2, stride: 1 },
        ];

        let grad: Vec<f32> = (0..n_samples).map(|i| i as f32).collect();
        let hess: Vec<f32> = vec![1.0; n_samples];
        let mut histogram = vec![(0.0, 0.0); 12];

        let ordered_grad_hess: Vec<GradHessF32> = grad
            .iter()
            .zip(&hess)
            .map(|(&g, &h)| GradHessF32 { grad: g, hess: h })
            .collect();

        build_histograms_ordered_sequential_interleaved(
            &mut histogram,
            &ordered_grad_hess,
            0,
            &bin_views,
            &features,
        );

        let f0_bin0_grad: f64 = (0..n_samples)
            .filter(|i| i % 4 == 0)
            .map(|i| i as f64)
            .sum();
        assert!((histogram[0].0 - f0_bin0_grad).abs() < 1e-10);
    }

    #[test]
    fn test_build_histograms_ordered_interleaved_matches_naive() {
        let features = make_features(&[4, 4]);
        let n_samples = 128;
        let bins_f0: Vec<u8> = (0..n_samples).map(|i| (i % 4) as u8).collect();
        let bins_f1: Vec<u8> = (0..n_samples).map(|i| ((i + 1) % 4) as u8).collect();

        let bin_views = vec![
            FeatureView::U8 { bins: &bins_f0, stride: 1 },
            FeatureView::U8 { bins: &bins_f1, stride: 1 },
        ];

        let indices: Vec<u32> = (0..n_samples as u32).step_by(3).collect();
        // Naive reference: accumulate directly from rows.
        let mut hist_ref = vec![(0.0, 0.0); 8];
        for &row_u32 in &indices {
            let row = row_u32 as usize;
            // Match the kernel's `f32 -> f64` accumulation behavior exactly.
            let g = (row as f32 * 0.25) as f64;
            let h = (1.0f32 + (row as f32) * 0.01) as f64;

            let b0 = bins_f0[row] as usize;
            hist_ref[b0].0 += g;
            hist_ref[b0].1 += h;

            let b1 = bins_f1[row] as usize;
            hist_ref[4 + b1].0 += g;
            hist_ref[4 + b1].1 += h;
        }

        let ordered_interleaved: Vec<GradHessF32> = indices
            .iter()
            .map(|&r| {
                let r = r as f32;
                GradHessF32 {
                    grad: r * 0.25,
                    hess: 1.0 + r * 0.01,
                }
            })
            .collect();

        let mut hist = vec![(0.0, 0.0); 8];
        build_histograms_ordered_interleaved(&mut hist, &ordered_interleaved, &indices, &bin_views, &features);

        for i in 0..hist.len() {
            assert!((hist_ref[i].0 - hist[i].0).abs() < 1e-10);
            assert!((hist_ref[i].1 - hist[i].1).abs() < 1e-10);
        }
    }
}
