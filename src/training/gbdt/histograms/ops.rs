//! Histogram building and operations.
//!
//! This module provides histogram building functions for gradient boosting tree training.
//! The main entry point is [`build_histograms`] which automatically selects between
//! sequential and feature-parallel strategies based on data characteristics.
//!
//! # Design Philosophy
//!
//! - Simple `(f64, f64)` tuples for bins (no complex trait hierarchies)
//! - LLVM auto-vectorizes the scalar loops effectively
//! - Feature-parallel only (row-parallel was 2.8x slower due to merge overhead)
//! - The subtraction trick (sibling = parent - child) provides 10-44x speedup
//!
//! # Numeric Precision
//!
//! Histogram bins use `f64` for accumulation despite gradients being stored as `f32`.
//! This is intentional:
//! - Gain computation involves differences of large sums that can lose precision in f32
//! - Memory overhead is acceptable (histograms are small: typically 256 bins × features)
//! - Benchmarks showed f32→f64 quantization overhead outweighed memory bandwidth savings

use crate::data::binned::FeatureView;
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
    pub fn auto_select(n_rows: usize, n_features: usize, n_threads: usize) -> Self {
        const MIN_ROWS_PARALLEL: usize = 1024;
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

/// Build histograms for all features.
///
/// Automatically selects between sequential and feature-parallel strategies.
///
/// # Arguments
/// * `histogram` - Output histogram buffer (total_bins length, should be zeroed)
/// * `grad` - Gradient values for all samples
/// * `hess` - Hessian values for all samples  
/// * `indices` - Row indices to process (empty = all rows)
/// * `bin_views` - Feature bin views from BinnedDataset
/// * `feature_metas` - Feature offset/size metadata
pub fn build_histograms(
    histogram: &mut [HistogramBin],
    grad: &[f32],
    hess: &[f32],
    indices: &[u32],
    bin_views: &[FeatureView<'_>],
    feature_metas: &[FeatureMeta],
) {
    let n_rows = if indices.is_empty() { grad.len() } else { indices.len() };
    let n_features = feature_metas.len();
    let n_threads = rayon::current_num_threads();

    let strategy = ParallelStrategy::auto_select(n_rows, n_features, n_threads);

    match strategy {
        ParallelStrategy::Sequential => {
            for (f, meta) in feature_metas.iter().enumerate() {
                let offset = meta.offset as usize;
                let n_bins = meta.n_bins as usize;
                let hist_slice = &mut histogram[offset..offset + n_bins];
                build_feature(hist_slice, grad, hess, indices, &bin_views[f]);
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
                build_feature(hist_slice, grad, hess, indices, &bin_views[f]);
            });
        }
    }
}

// =============================================================================
// Single-Feature Building
// =============================================================================

/// Build histogram for a single feature (dispatcher).
#[inline]
fn build_feature(
    histogram: &mut [HistogramBin],
    grad: &[f32],
    hess: &[f32],
    indices: &[u32],
    view: &FeatureView<'_>,
) {
    match view {
        FeatureView::U8 { bins, stride: 1 } => {
            build_histogram_u8(bins, grad, hess, histogram, indices);
        }
        FeatureView::U16 { bins, stride: 1 } => {
            build_histogram_u16(bins, grad, hess, histogram, indices);
        }
        FeatureView::U8 { bins, stride } => {
            build_histogram_strided_u8(bins, *stride, grad, hess, histogram, indices);
        }
        FeatureView::U16 { bins, stride } => {
            build_histogram_strided_u16(bins, *stride, grad, hess, histogram, indices);
        }
        FeatureView::SparseU8 { row_indices, bin_values } => {
            let node_indices = if indices.is_empty() { None } else { Some(indices) };
            build_histogram_sparse_u8(row_indices, bin_values, grad, hess, histogram, node_indices);
        }
        FeatureView::SparseU16 { row_indices, bin_values } => {
            let node_indices = if indices.is_empty() { None } else { Some(indices) };
            build_histogram_sparse_u16(row_indices, bin_values, grad, hess, histogram, node_indices);
        }
    }
}

// =============================================================================
// Core Building Functions
// =============================================================================

/// Build histogram for dense u8 bins.
#[inline]
pub fn build_histogram_u8(
    bins: &[u8],
    grad: &[f32],
    hess: &[f32],
    histogram: &mut [HistogramBin],
    indices: &[u32],
) {
    if indices.is_empty() {
        for i in 0..bins.len() {
            let bin = bins[i] as usize;
            histogram[bin].0 += grad[i] as f64;
            histogram[bin].1 += hess[i] as f64;
        }
    } else {
        for &idx in indices {
            let row = idx as usize;
            let bin = bins[row] as usize;
            histogram[bin].0 += grad[row] as f64;
            histogram[bin].1 += hess[row] as f64;
        }
    }
}

/// Build histogram for dense u16 bins.
#[inline]
pub fn build_histogram_u16(
    bins: &[u16],
    grad: &[f32],
    hess: &[f32],
    histogram: &mut [HistogramBin],
    indices: &[u32],
) {
    if indices.is_empty() {
        for i in 0..bins.len() {
            let bin = bins[i] as usize;
            histogram[bin].0 += grad[i] as f64;
            histogram[bin].1 += hess[i] as f64;
        }
    } else {
        for &idx in indices {
            let row = idx as usize;
            let bin = bins[row] as usize;
            histogram[bin].0 += grad[row] as f64;
            histogram[bin].1 += hess[row] as f64;
        }
    }
}

/// Build histogram with strided u8 bins (row-major groups).
#[inline]
pub fn build_histogram_strided_u8(
    bins: &[u8],
    stride: usize,
    grad: &[f32],
    hess: &[f32],
    histogram: &mut [HistogramBin],
    indices: &[u32],
) {
    if indices.is_empty() {
        let n_rows = grad.len();
        for row in 0..n_rows {
            let bin = bins[row * stride] as usize;
            histogram[bin].0 += grad[row] as f64;
            histogram[bin].1 += hess[row] as f64;
        }
    } else {
        for &idx in indices {
            let row = idx as usize;
            let bin = bins[row * stride] as usize;
            histogram[bin].0 += grad[row] as f64;
            histogram[bin].1 += hess[row] as f64;
        }
    }
}

/// Build histogram with strided u16 bins.
#[inline]
pub fn build_histogram_strided_u16(
    bins: &[u16],
    stride: usize,
    grad: &[f32],
    hess: &[f32],
    histogram: &mut [HistogramBin],
    indices: &[u32],
) {
    if indices.is_empty() {
        let n_rows = grad.len();
        for row in 0..n_rows {
            let bin = bins[row * stride] as usize;
            histogram[bin].0 += grad[row] as f64;
            histogram[bin].1 += hess[row] as f64;
        }
    } else {
        for &idx in indices {
            let row = idx as usize;
            let bin = bins[row * stride] as usize;
            histogram[bin].0 += grad[row] as f64;
            histogram[bin].1 += hess[row] as f64;
        }
    }
}

// =============================================================================
// Sparse Feature Building
// =============================================================================

/// Build histogram for sparse u8 features.
#[inline]
pub fn build_histogram_sparse_u8(
    row_indices: &[u32],
    bin_values: &[u8],
    grad: &[f32],
    hess: &[f32],
    histogram: &mut [HistogramBin],
    node_indices: Option<&[u32]>,
) {
    match node_indices {
        None => {
            for (i, &row) in row_indices.iter().enumerate() {
                let r = row as usize;
                let bin = bin_values[i] as usize;
                histogram[bin].0 += grad[r] as f64;
                histogram[bin].1 += hess[r] as f64;
            }
        }
        Some(indices) => {
            // Sorted merge - both arrays are sorted
            let mut sparse_ptr = 0;
            let mut node_ptr = 0;
            while sparse_ptr < row_indices.len() && node_ptr < indices.len() {
                let sparse_row = row_indices[sparse_ptr];
                let node_row = indices[node_ptr];
                match sparse_row.cmp(&node_row) {
                    std::cmp::Ordering::Less => sparse_ptr += 1,
                    std::cmp::Ordering::Greater => node_ptr += 1,
                    std::cmp::Ordering::Equal => {
                        let r = sparse_row as usize;
                        let bin = bin_values[sparse_ptr] as usize;
                        histogram[bin].0 += grad[r] as f64;
                        histogram[bin].1 += hess[r] as f64;
                        sparse_ptr += 1;
                        node_ptr += 1;
                    }
                }
            }
        }
    }
}

/// Build histogram for sparse u16 features.
#[inline]
pub fn build_histogram_sparse_u16(
    row_indices: &[u32],
    bin_values: &[u16],
    grad: &[f32],
    hess: &[f32],
    histogram: &mut [HistogramBin],
    node_indices: Option<&[u32]>,
) {
    match node_indices {
        None => {
            for (i, &row) in row_indices.iter().enumerate() {
                let r = row as usize;
                let bin = bin_values[i] as usize;
                histogram[bin].0 += grad[r] as f64;
                histogram[bin].1 += hess[r] as f64;
            }
        }
        Some(indices) => {
            let mut sparse_ptr = 0;
            let mut node_ptr = 0;
            while sparse_ptr < row_indices.len() && node_ptr < indices.len() {
                let sparse_row = row_indices[sparse_ptr];
                let node_row = indices[node_ptr];
                match sparse_row.cmp(&node_row) {
                    std::cmp::Ordering::Less => sparse_ptr += 1,
                    std::cmp::Ordering::Greater => node_ptr += 1,
                    std::cmp::Ordering::Equal => {
                        let r = sparse_row as usize;
                        let bin = bin_values[sparse_ptr] as usize;
                        histogram[bin].0 += grad[r] as f64;
                        histogram[bin].1 += hess[r] as f64;
                        sparse_ptr += 1;
                        node_ptr += 1;
                    }
                }
            }
        }
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
        let grad = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let hess = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let mut histogram = vec![(0.0, 0.0); 3];

        build_histogram_u8(&bins, &grad, &hess, &mut histogram, &[]);

        assert!((histogram[0].0 - 10.0).abs() < 1e-10); // 1+3+6
        assert!((histogram[0].1 - 5.0).abs() < 1e-10);  // 0.5+1.5+3
        assert!((histogram[1].0 - 7.0).abs() < 1e-10);  // 2+5
        assert!((histogram[2].0 - 4.0).abs() < 1e-10);  // 4
    }

    #[test]
    fn test_build_histogram_with_indices() {
        let bins: Vec<u8> = vec![0, 1, 2, 0, 1, 2];
        let grad = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let hess = vec![1.0; 6];
        let mut histogram = vec![(0.0, 0.0); 3];
        let indices: Vec<u32> = vec![0, 2, 4];

        build_histogram_u8(&bins, &grad, &hess, &mut histogram, &indices);

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

        build_histogram_sparse_u8(&row_indices, &bin_values, &grad, &hess, &mut histogram, None);

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

        build_histograms(&mut histogram, &grad, &hess, &[], &bin_views, &features);

        let f0_bin0_grad: f64 = (0..n_samples)
            .filter(|i| i % 4 == 0)
            .map(|i| i as f64)
            .sum();
        assert!((histogram[0].0 - f0_bin0_grad).abs() < 1e-10);
    }
}
