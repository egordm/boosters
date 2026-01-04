//! Column-major buffer for gathering leaf features.
//!
//! This buffer is allocated once at training start and reused for each leaf.
//! It gathers features into contiguous columns for efficient coordinate descent.

use crate::data::Dataset;

/// Column-major buffer for leaf feature data.
///
/// Allocated once, reused for each leaf during linear fitting.
/// Column-major layout means each feature's values are contiguous,
/// which is optimal for coordinate descent access patterns.
///
/// # Example
///
/// ```ignore
/// use boosters::data::Dataset;
///
/// let mut buffer = LeafFeatureBuffer::new(1000, 10);
/// let dataset = Dataset::from_array(features.view(), None, None);
///
/// // Gather features for rows in this leaf
/// buffer.gather(&leaf_rows, &dataset, &path_features);
///
/// // Access feature columns for coordinate descent
/// for feat_idx in 0..buffer.n_features() {
///     let values = buffer.feature_slice(feat_idx);
///     // ... process contiguous feature column
/// }
/// ```
#[derive(Debug)]
pub struct LeafFeatureBuffer {
    /// Column-major data: `data[feat_idx * max_rows + row_idx]`
    data: Vec<f32>,
    /// Maximum rows this buffer can hold
    max_rows: usize,
    /// Maximum features this buffer can hold
    max_features: usize,
    /// Current number of rows (samples in this leaf)
    n_rows: usize,
    /// Current number of features
    n_features: usize,
}

impl LeafFeatureBuffer {
    /// Create a new buffer with given capacity.
    ///
    /// # Arguments
    ///
    /// * `max_rows` - Maximum samples per leaf (e.g., n_samples / n_leaves * 2)
    /// * `max_features` - Maximum features used in any path (e.g., max_depth)
    ///
    /// # Panics
    ///
    /// Panics in debug builds if either capacity is zero.
    pub fn new(max_rows: usize, max_features: usize) -> Self {
        debug_assert!(max_rows > 0, "max_rows must be positive");
        debug_assert!(max_features > 0, "max_features must be positive");

        Self {
            data: vec![0.0; max_rows * max_features],
            max_rows,
            max_features,
            n_rows: 0,
            n_features: 0,
        }
    }

    /// Gather features from a Dataset into this buffer.
    ///
    /// Gathers raw feature values using `Dataset::gather_feature_values`.
    ///
    /// # Arguments
    ///
    /// * `rows` - Row indices of samples in this leaf
    /// * `data` - Dataset containing raw feature values
    /// * `features` - Feature indices to gather (typically path features)
    ///
    /// # Panics
    ///
    /// Panics if `rows.len() > max_rows` or `features.len() > max_features`.
    pub fn gather(&mut self, rows: &[u32], data: &Dataset, features: &[u32]) {
        debug_assert!(
            rows.len() <= self.max_rows,
            "Too many rows: {} > max {}",
            rows.len(),
            self.max_rows
        );
        debug_assert!(
            features.len() <= self.max_features,
            "Too many features: {} > max {}",
            features.len(),
            self.max_features
        );

        self.n_rows = rows.len();
        self.n_features = features.len();

        // Gather each feature column using Dataset::gather_feature_values
        for (feat_idx, &feat) in features.iter().enumerate() {
            let col_offset = feat_idx * self.max_rows;
            let buffer = &mut self.data[col_offset..col_offset + rows.len()];
            data.gather_feature_values(feat as usize, rows, buffer);

            // Treat NaNs as missing values for linear leaves.
            // Using 0.0 makes them contribute nothing to dot products and keeps
            // the solver numerically stable.
            for v in buffer.iter_mut() {
                // TODO: see if this can be improved.
                if v.is_nan() {
                    *v = 0.0;
                }
            }
        }
    }

    /// Get contiguous slice for a feature column.
    ///
    /// Returns `n_rows` values for the given feature index.
    ///
    /// # Panics
    ///
    /// Panics if `feat_idx >= n_features`.
    #[inline]
    pub fn feature_slice(&self, feat_idx: usize) -> &[f32] {
        debug_assert!(
            feat_idx < self.n_features,
            "feature index {} out of bounds (n_features = {})",
            feat_idx,
            self.n_features
        );
        let start = feat_idx * self.max_rows;
        &self.data[start..start + self.n_rows]
    }

    /// Number of rows currently in the buffer.
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of features currently in the buffer.
    #[inline]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Maximum rows this buffer can hold.
    #[inline]
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    /// Maximum features this buffer can hold.
    #[inline]
    pub fn max_features(&self) -> usize {
        self.max_features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_leaf_buffer_gather() {
        // Create a 3x10 feature-major matrix (3 features, 10 samples)
        // Data layout: [f0_s0..f0_s9, f1_s0..f1_s9, f2_s0..f2_s9]
        let data: Vec<f32> = (0..30).map(|i| i as f32).collect();
        let arr = Array2::from_shape_vec((3, 10), data).unwrap();
        let dataset = Dataset::from_array(arr.view(), None, None);

        let mut buffer = LeafFeatureBuffer::new(10, 5);

        // Gather rows [1, 3, 5] for features [0, 2]
        let rows = vec![1, 3, 5];
        let features = vec![0, 2];
        buffer.gather(&rows, &dataset, &features);

        assert_eq!(buffer.n_rows(), 3);
        assert_eq!(buffer.n_features(), 2);

        // Feature 0, rows 1,3,5 -> values 1.0, 3.0, 5.0
        let feat0 = buffer.feature_slice(0);
        assert_eq!(feat0, &[1.0, 3.0, 5.0]);

        // Feature 2, rows 1,3,5 -> values 21.0, 23.0, 25.0 (offset by 20)
        let feat2 = buffer.feature_slice(1);
        assert_eq!(feat2, &[21.0, 23.0, 25.0]);
    }

    #[test]
    fn test_leaf_buffer_reuse() {
        // 4 features, 5 samples
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let arr = Array2::from_shape_vec((4, 5), data).unwrap();
        let dataset = Dataset::from_array(arr.view(), None, None);

        let mut buffer = LeafFeatureBuffer::new(10, 5);

        // First gather
        buffer.gather(&[0, 1], &dataset, &[0, 1]);
        assert_eq!(buffer.n_rows(), 2);
        assert_eq!(buffer.feature_slice(0), &[0.0, 1.0]);

        // Second gather - should overwrite
        buffer.gather(&[2, 3, 4], &dataset, &[2, 3]);
        assert_eq!(buffer.n_rows(), 3);
        assert_eq!(buffer.n_features(), 2);

        // Feature 2 (index 0 in buffer), rows 2,3,4 -> values 12.0, 13.0, 14.0
        let feat = buffer.feature_slice(0);
        assert_eq!(feat, &[12.0, 13.0, 14.0]);
    }

    #[test]
    #[should_panic(expected = "Too many rows")]
    #[cfg(debug_assertions)]
    fn test_leaf_buffer_overflow_rows() {
        // 3 features, 10 samples
        let data: Vec<f32> = (0..30).map(|i| i as f32).collect();
        let arr = Array2::from_shape_vec((3, 10), data).unwrap();
        let dataset = Dataset::from_array(arr.view(), None, None);

        let mut buffer = LeafFeatureBuffer::new(5, 3); // Only 5 rows capacity

        // Try to gather 10 rows - should panic
        let rows: Vec<u32> = (0..10).collect();
        buffer.gather(&rows, &dataset, &[0]);
    }

    #[test]
    fn test_leaf_buffer_gather_replaces_nans_with_zero() {
        // 2 features, 3 samples (feature-major)
        // f0: [1.0, NaN, 3.0]
        // f1: [NaN, 2.0, NaN]
        let arr = Array2::from_shape_vec((2, 3), vec![1.0, f32::NAN, 3.0, f32::NAN, 2.0, f32::NAN])
            .unwrap();
        let dataset = Dataset::from_array(arr.view(), None, None);

        let mut buffer = LeafFeatureBuffer::new(3, 2);
        buffer.gather(&[0, 1, 2], &dataset, &[0, 1]);

        assert_eq!(buffer.feature_slice(0), &[1.0, 0.0, 3.0]);
        assert_eq!(buffer.feature_slice(1), &[0.0, 2.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "Too many features")]
    #[cfg(debug_assertions)]
    fn test_leaf_buffer_overflow_features() {
        // 3 features, 10 samples
        let data: Vec<f32> = (0..30).map(|i| i as f32).collect();
        let arr = Array2::from_shape_vec((3, 10), data).unwrap();
        let dataset = Dataset::from_array(arr.view(), None, None);

        let mut buffer = LeafFeatureBuffer::new(10, 2); // Only 2 features capacity

        // Try to gather 3 features - should panic
        buffer.gather(&[0, 1], &dataset, &[0, 1, 2]);
    }
}
