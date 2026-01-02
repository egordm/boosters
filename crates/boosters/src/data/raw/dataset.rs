//! Dataset container and builder.
//!
//! This module provides [`Dataset`] and [`DatasetBuilder`] for creating
//! raw feature datasets. For training, create a [`crate::data::BinnedDataset`]
//! using [`BinnedDataset::from_dataset()`](crate::data::BinnedDataset::from_dataset).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

use super::feature::Feature;
use super::schema::{DatasetSchema, FeatureMeta};
use super::views::{TargetsView, WeightsView};
use crate::data::error::DatasetError;

/// The unified dataset container for all boosters models.
///
/// # Storage Layout
///
/// Features are stored as individual columns (per RFC-0021), each containing
/// either dense or sparse values. This enables efficient sparse dataset support.
///
/// Targets are stored as `[n_outputs, n_samples]` for consistency.
///
/// # Construction
///
/// Use [`Dataset::from_array`] for construction from feature-major matrices,
/// or [`Dataset::builder`] for complex scenarios with mixed feature types.
///
/// # Example
///
/// ```
/// use boosters::data::Dataset;
/// use ndarray::array;
///
/// // Feature-major format: 2 features, 3 samples
/// let features = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]; // [n_features, n_samples]
/// let targets = array![[0.0f32, 1.0, 0.0]]; // [n_outputs, n_samples]
/// let ds = Dataset::from_array(features.view(), Some(targets), None);
///
/// assert_eq!(ds.n_samples(), 3);
/// assert_eq!(ds.n_features(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Feature columns: each Feature contains values for all samples.
    /// Supports both dense (Array1<f32>) and sparse (indices + values) storage.
    features: Vec<Feature>,

    /// Number of samples (cached from first feature).
    n_samples: usize,

    /// Feature metadata.
    schema: DatasetSchema,

    /// Target values: `[n_outputs, n_samples]`.
    targets: Option<Array2<f32>>,

    /// Sample weights: length = n_samples.
    weights: Option<Array1<f32>>,
}

impl Dataset {
    /// Create a Dataset from pre-built feature columns and an explicit schema.
    ///
    /// This is the canonical constructor. The schema is immutable and must be
    /// fully determined at construction time.
    ///
    /// Prefer [`DatasetBuilder`] for most use cases.
    ///
    /// # Panics
    ///
    /// Panics if features is empty or if sample counts don't match.
    pub fn new(
        schema: DatasetSchema,
        features: Vec<Feature>,
        targets: Option<Array2<f32>>,
        weights: Option<Array1<f32>>,
    ) -> Self {
        assert!(!features.is_empty(), "features cannot be empty");
        let n_samples = features[0].n_samples();

        // Validate all features have same n_samples
        for feat in &features {
            assert_eq!(
                feat.n_samples(),
                n_samples,
                "all features must have same n_samples"
            );
        }

        if let Some(ref t) = targets {
            assert_eq!(
                t.ncols(),
                n_samples,
                "targets must have same sample count as features"
            );
        }
        if let Some(ref w) = weights {
            assert_eq!(
                w.len(),
                n_samples,
                "weights must have same sample count as features"
            );
        }

        Self {
            features,
            n_samples,
            schema,
            targets,
            weights,
        }
    }

    /// Create a dense dataset from a 2D feature array.
    ///
    /// This is a convenience constructor for creating datasets from dense
    /// feature matrices. Each row of the input array becomes a dense feature.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature-major array `[n_features, n_samples]`
    /// * `targets` - Optional target array `[n_outputs, n_samples]`
    /// * `weights` - Optional sample weights `[n_samples]`
    ///
    /// # Example
    ///
    /// ```
    /// use boosters::data::Dataset;
    /// use ndarray::array;
    ///
    /// // 2 features, 3 samples
    /// let features = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let targets = array![[0.0f32, 1.0, 0.0]];
    /// let ds = Dataset::from_array(features.view(), Some(targets), None);
    ///
    /// assert_eq!(ds.n_samples(), 3);
    /// assert_eq!(ds.n_features(), 2);
    /// ```
    pub fn from_array(
        features: ArrayView2<f32>,
        targets: Option<Array2<f32>>,
        weights: Option<Array1<f32>>,
    ) -> Self {
        let n_features = features.nrows();

        // Convert each row to a dense Feature
        let feature_vec: Vec<Feature> = (0..n_features)
            .map(|i| Feature::dense(features.row(i).to_owned()))
            .collect();

        let schema = DatasetSchema::all_numeric(n_features);

        Self::new(schema, feature_vec, targets, weights)
    }

    /// Create a builder for dataset construction.
    ///
    /// This is the recommended way to create a Dataset.
    ///
    /// # Example
    ///
    /// ```
    /// use boosters::data::Dataset;
    /// use ndarray::array;
    ///
    /// let ds = Dataset::builder()
    ///     .add_feature_unnamed(array![1.0, 2.0, 3.0])
    ///     .add_feature_unnamed(array![4.0, 5.0, 6.0])
    ///     .targets(array![[0.0, 1.0, 0.0]].view())
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(ds.n_samples(), 3);
    /// assert_eq!(ds.n_features(), 2);
    /// ```
    pub fn builder() -> DatasetBuilder {
        DatasetBuilder::new()
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Number of samples.
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Number of features.
    #[inline]
    pub fn n_features(&self) -> usize {
        self.features.len()
    }

    /// Number of output dimensions (returns 0 if no targets).
    #[inline]
    pub fn n_outputs(&self) -> usize {
        self.targets.as_ref().map(|t| t.nrows()).unwrap_or(0)
    }

    /// Get the schema.
    pub fn schema(&self) -> &DatasetSchema {
        &self.schema
    }

    /// Check if any feature is categorical.
    pub fn has_categorical(&self) -> bool {
        self.schema.has_categorical()
    }

    /// Check if dataset has targets.
    pub fn has_targets(&self) -> bool {
        self.targets.is_some()
    }

    /// Check if dataset has weights.
    pub fn has_weights(&self) -> bool {
        self.weights.is_some()
    }

    // =========================================================================
    // Feature access
    // =========================================================================

    /// Get a reference to a specific feature column.
    #[inline]
    pub fn feature(&self, idx: usize) -> &Feature {
        &self.features[idx]
    }

    /// Get references to all feature columns.
    #[inline]
    pub fn feature_columns(&self) -> &[Feature] {
        &self.features
    }

    /// Get a view of the target data.
    ///
    /// Returns `None` if no targets were provided.
    pub fn targets(&self) -> Option<TargetsView<'_>> {
        self.targets.as_ref().map(|t| TargetsView::new(t.view()))
    }

    /// Get sample weights as a WeightsView.
    ///
    /// Returns `WeightsView::None` if no weights were provided,
    /// or `WeightsView::Some(array)` if weights exist.
    pub fn weights(&self) -> WeightsView<'_> {
        match &self.weights {
            Some(w) => WeightsView::from_array(w.view()),
            None => WeightsView::none(),
        }
    }

    // =========================================================================
    // Feature Value Iteration (RFC-0019)
    // =========================================================================

    /// Apply a function to each (sample_idx, raw_value) pair for a feature.
    ///
    /// This is the recommended pattern for iterating over feature values.
    /// The storage type is matched ONCE, then we iterate directly on the
    /// underlying data—no per-iteration branching.
    ///
    /// # Performance
    ///
    /// - Dense: Equivalent to `for (i, &v) in slice.iter().enumerate()`
    /// - Sparse: Iterates only stored (non-default) values
    ///
    /// # Example
    ///
    /// ```
    /// use boosters::data::Dataset;
    /// use ndarray::array;
    ///
    /// let ds = Dataset::from_array(
    ///     array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].view(),
    ///     None,
    ///     None,
    /// );
    ///
    /// let mut sum = 0.0;
    /// ds.for_each_feature_value(0, |_idx, value| {
    ///     sum += value;
    /// });
    /// assert_eq!(sum, 6.0); // 1 + 2 + 3
    /// ```
    #[inline]
    pub fn for_each_feature_value<F>(&self, feature: usize, mut f: F)
    where
        F: FnMut(usize, f32),
    {
        match &self.features[feature] {
            Feature::Dense(values) => {
                // Fast path: contiguous slice iteration
                if let Some(slice) = values.as_slice() {
                    for (idx, &val) in slice.iter().enumerate() {
                        f(idx, val);
                    }
                } else {
                    for (idx, &val) in values.iter().enumerate() {
                        f(idx, val);
                    }
                }
            }
            Feature::Sparse {
                indices, values, ..
            } => {
                // Sparse: iterate only stored (non-default) values
                for (&idx, &val) in indices.iter().zip(values.iter()) {
                    f(idx as usize, val);
                }
            }
        }
    }

    /// Apply a function to each (sample_idx, raw_value) pair for a feature,
    /// including default values for sparse features.
    ///
    /// This is useful when downstream code needs a dense stream of values for
    /// all samples (e.g., binning / quantization), but we still want to avoid
    /// materializing a full dense matrix.
    ///
    /// # Performance
    ///
    /// - Dense: $O(n)$ iteration over the underlying array
    /// - Sparse: $O(n + nnz)$ merge-like iteration yielding defaults
    #[inline]
    pub fn for_each_feature_value_dense<F>(&self, feature: usize, mut f: F)
    where
        F: FnMut(usize, f32),
    {
        match &self.features[feature] {
            Feature::Dense(values) => {
                if let Some(slice) = values.as_slice() {
                    for (idx, &val) in slice.iter().enumerate() {
                        f(idx, val);
                    }
                } else {
                    for (idx, &val) in values.iter().enumerate() {
                        f(idx, val);
                    }
                }
            }
            Feature::Sparse {
                indices,
                values,
                default,
                ..
            } => {
                let mut sparse_pos = 0usize;
                let n_samples = self.n_samples;

                for sample_idx in 0..n_samples {
                    let val = if sparse_pos < indices.len()
                        && indices[sparse_pos] as usize == sample_idx
                    {
                        let v = values[sparse_pos];
                        sparse_pos += 1;
                        v
                    } else {
                        *default
                    };
                    f(sample_idx, val);
                }
            }
        }
    }

    /// Gather raw values for a feature at specified sample indices into a buffer.
    ///
    /// This is the recommended pattern for linear tree fitting where we need
    /// values for a subset of samples (e.g., samples that landed in a leaf).
    ///
    /// # Arguments
    ///
    /// - `feature`: The feature index
    /// - `sample_indices`: Slice of sample indices to gather (should be sorted for sparse efficiency)
    /// - `buffer`: Output buffer, must have length >= sample_indices.len()
    ///
    /// # Performance
    ///
    /// - Dense: Simple indexed gather, O(k) where k = sample_indices.len()
    /// - Sparse: Merge-join since both indices and sparse storage are sorted
    ///
    /// # Example
    ///
    /// ```
    /// use boosters::data::Dataset;
    /// use ndarray::array;
    ///
    /// let ds = Dataset::from_array(
    ///     array![[1.0, 2.0, 3.0, 4.0, 5.0]].view(),
    ///     None,
    ///     None,
    /// );
    ///
    /// let indices = [1, 3]; // gather samples 1 and 3
    /// let mut buffer = vec![0.0; 2];
    /// ds.gather_feature_values(0, &indices, &mut buffer);
    /// assert_eq!(buffer, vec![2.0, 4.0]);
    /// ```
    #[inline]
    pub fn gather_feature_values(
        &self,
        feature: usize,
        sample_indices: &[u32],
        buffer: &mut [f32],
    ) {
        debug_assert!(
            buffer.len() >= sample_indices.len(),
            "buffer must have length >= sample_indices.len()"
        );

        match &self.features[feature] {
            Feature::Dense(values) => {
                // Dense: simple indexed gather
                for (out_idx, &sample_idx) in sample_indices.iter().enumerate() {
                    buffer[out_idx] = values[sample_idx as usize];
                }
            }
            Feature::Sparse {
                indices,
                values,
                default,
                ..
            } => {
                // Sparse: merge-join algorithm (both sorted)
                let mut sparse_pos = 0;
                for (out_idx, &sample_idx) in sample_indices.iter().enumerate() {
                    // Advance sparse_pos to find or pass sample_idx
                    while sparse_pos < indices.len() && indices[sparse_pos] < sample_idx {
                        sparse_pos += 1;
                    }
                    if sparse_pos < indices.len() && indices[sparse_pos] == sample_idx {
                        buffer[out_idx] = values[sparse_pos];
                    } else {
                        buffer[out_idx] = *default;
                    }
                }
            }
        }
    }

    /// Similar to gather but with a callback for each (local_idx, value) pair.
    ///
    /// Useful when you need to process values immediately without allocating.
    ///
    /// # Arguments
    ///
    /// - `feature`: The feature index
    /// - `sample_indices`: Slice of sample indices to gather
    /// - `f`: Callback receiving (index into sample_indices, value)
    #[inline]
    pub fn for_each_gathered_value<F>(&self, feature: usize, sample_indices: &[u32], mut f: F)
    where
        F: FnMut(usize, f32),
    {
        match &self.features[feature] {
            Feature::Dense(values) => {
                for (out_idx, &sample_idx) in sample_indices.iter().enumerate() {
                    f(out_idx, values[sample_idx as usize]);
                }
            }
            Feature::Sparse {
                indices,
                values,
                default,
                ..
            } => {
                // Sparse: merge-join algorithm (both sorted)
                let mut sparse_pos = 0;
                for (out_idx, &sample_idx) in sample_indices.iter().enumerate() {
                    while sparse_pos < indices.len() && indices[sparse_pos] < sample_idx {
                        sparse_pos += 1;
                    }
                    if sparse_pos < indices.len() && indices[sparse_pos] == sample_idx {
                        f(out_idx, values[sparse_pos]);
                    } else {
                        f(out_idx, *default);
                    }
                }
            }
        }
    }

    /// Get a single feature value at a specific sample index.
    ///
    /// # Performance
    ///
    /// This is O(1) for dense features but O(log n) for sparse features
    /// due to binary search. For bulk access, prefer the iteration methods.
    #[inline]
    pub fn get_feature_value(&self, feature: usize, sample: usize) -> f32 {
        self.features[feature].get(sample)
    }

    /// Fill a block buffer with data from the dataset and return a view.
    ///
    /// Dataset stores features as `[n_features, n_samples]` (feature-major).
    /// This transposes to `[block_size, n_features]` (sample-major) in the buffer.
    ///
    /// Returns a `SamplesView` covering only the filled portion (may be less
    /// than block size at the end of the dataset).
    ///
    /// # Arguments
    ///
    /// * `buffer` - Mutable buffer with shape `[block_size, n_features]`
    /// * `start_sample` - First sample index to copy
    ///
    /// # Returns
    ///
    /// A `SamplesView` borrowing from the buffer, covering `[0..samples_filled, ..]`
    pub fn buffer_samples<'b>(
        &self,
        buffer: &'b mut ndarray::Array2<f32>,
        start_sample: usize,
    ) -> super::SamplesView<'b> {
        let n_features = self.n_features();
        let block_size = buffer.nrows();
        debug_assert!(n_features == buffer.ncols(), "block shape mismatch");

        let samples_filled = block_size.min(self.n_samples().saturating_sub(start_sample));

        for feature_idx in 0..n_features {
            // Get mutable column view in the output block
            let mut feature_slice = buffer.column_mut(feature_idx);
            self.buffer_feature_samples(&mut feature_slice, feature_idx, start_sample);
        }

        // Return a view covering only the filled samples
        super::SamplesView::from_array(buffer.slice(s![..samples_filled, ..]))
    }

    fn buffer_feature_samples(
        &self,
        buffer: &mut ndarray::ArrayViewMut1<f32>,
        feature_idx: usize,
        start_sample: usize,
    ) -> usize {
        let block_size = buffer.len();
        let samples_filled = block_size.min(self.n_samples() - start_sample);

        match &self.features[feature_idx] {
            Feature::Dense(values) => {
                // Only fill the portion we have samples for
                let src = values.slice(s![start_sample..start_sample + samples_filled]);
                buffer.slice_mut(s![..samples_filled]).assign(&src);
            }
            Feature::Sparse {
                indices,
                values,
                default,
                ..
            } => {
                // Fill buffer with default value
                buffer.slice_mut(s![..samples_filled]).fill(*default);

                // Binary search to find first index >= start_sample
                let start_index = indices
                    .binary_search(&(start_sample as u32))
                    .unwrap_or_else(|x| x);

                // Fill non-default values
                for i in start_index..indices.len() {
                    let sample_idx = indices[i] as usize;
                    if sample_idx >= start_sample + samples_filled {
                        break;
                    }
                    buffer[sample_idx - start_sample] = values[i];
                }
            }
        }

        samples_filled
    }
}

/// Builder for complex dataset construction.
///
/// Use this when you need:
/// - Mixed dense and sparse columns
/// - Explicit feature types (numeric vs categorical)
/// - Explicit feature names
///
/// # Example
///
/// ```
/// use boosters::data::{DatasetBuilder, FeatureType};
/// use ndarray::array;
///
/// let ds = DatasetBuilder::new()
///     .add_feature("age", array![25.0, 30.0, 35.0])
///     .add_categorical("color", array![0.0, 1.0, 2.0])
///     .targets(array![[0.0, 1.0, 0.0]].view())
///     .build()
///     .unwrap();
///
/// assert_eq!(ds.n_features(), 2);
/// assert!(ds.has_categorical());
/// ```
#[derive(Debug, Default)]
pub struct DatasetBuilder {
    features: Vec<Feature>,
    metas: Vec<FeatureMeta>,
    targets: Option<Array2<f32>>,
    weights: Option<Array1<f32>>,
}

impl DatasetBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a numeric feature column.
    pub fn add_feature(mut self, name: &str, values: Array1<f32>) -> Self {
        self.features.push(Feature::dense(values));
        self.metas.push(FeatureMeta::numeric_named(name));
        self
    }

    /// Add an unnamed numeric feature column.
    pub fn add_feature_unnamed(mut self, values: Array1<f32>) -> Self {
        self.features.push(Feature::dense(values));
        self.metas.push(FeatureMeta::numeric());
        self
    }

    /// Add a categorical feature column.
    ///
    /// Values should be non-negative integers encoded as floats
    /// (e.g., 0.0, 1.0, 2.0).
    pub fn add_categorical(mut self, name: &str, values: Array1<f32>) -> Self {
        self.features.push(Feature::dense(values));
        self.metas.push(FeatureMeta::categorical_named(name));
        self
    }

    /// Add an unnamed categorical feature column.
    pub fn add_categorical_unnamed(mut self, values: Array1<f32>) -> Self {
        self.features.push(Feature::dense(values));
        self.metas.push(FeatureMeta::categorical());
        self
    }

    /// Add a feature column with explicit metadata.
    ///
    /// This is the most flexible way to add features, allowing full control
    /// over both the storage format (dense/sparse) and metadata.
    ///
    /// # Arguments
    ///
    /// * `feature` - The feature storage (dense or sparse)
    /// * `meta` - Feature metadata (type and optional name)
    pub fn add_feature_with_meta(mut self, feature: Feature, meta: FeatureMeta) -> Self {
        self.features.push(feature);
        self.metas.push(meta);
        self
    }

    /// Add a sparse feature column.
    ///
    /// # Arguments
    ///
    /// * `name` - Feature name
    /// * `indices` - Row indices with non-default values (must be sorted, no duplicates)
    /// * `values` - Values at those indices
    /// * `n_samples` - Total number of samples
    /// * `default` - Default value for unspecified indices (typically 0.0)
    pub fn add_sparse(
        mut self,
        name: &str,
        indices: Vec<u32>,
        values: Vec<f32>,
        n_samples: usize,
        default: f32,
    ) -> Self {
        self.features
            .push(Feature::sparse(indices, values, n_samples, default));
        self.metas.push(FeatureMeta::numeric_named(name));
        self
    }

    /// Set target values.
    ///
    /// Shape: `[n_outputs, n_samples]`.
    pub fn targets(mut self, targets: ArrayView2<f32>) -> Self {
        self.targets = Some(targets.to_owned());
        self
    }

    /// Set 1D targets (single output).
    pub fn targets_1d(mut self, targets: ArrayView1<f32>) -> Self {
        // Reshape to [1, n_samples]
        let n = targets.len();
        self.targets = Some(
            targets
                .to_owned()
                .into_shape_with_order((1, n))
                .expect("reshape should succeed"),
        );
        self
    }

    /// Set sample weights.
    pub fn weights(mut self, weights: ArrayView1<f32>) -> Self {
        self.weights = Some(weights.to_owned());
        self
    }

    /// Build the dataset.
    ///
    /// # Errors
    ///
    /// Returns [`DatasetError`] if:
    /// - No features provided
    /// - Features have inconsistent sample counts
    /// - Targets have wrong sample count
    /// - Weights have wrong length
    /// - Sparse indices are unsorted or have duplicates
    pub fn build(self) -> Result<Dataset, DatasetError> {
        if self.features.is_empty() {
            return Err(DatasetError::EmptyFeatures);
        }

        // Determine n_samples from first feature
        let n_samples = self.features[0].n_samples();

        // Validate all features have same n_samples
        for (i, feat) in self.features.iter().enumerate() {
            if feat.n_samples() != n_samples {
                return Err(DatasetError::ShapeMismatch {
                    expected: n_samples,
                    got: feat.n_samples(),
                    field: "features",
                });
            }

            // Validate sparse features
            if let Err((pos, idx)) = feat.validate() {
                // Check if it's unsorted or duplicate
                if let Feature::Sparse { indices, .. } = feat {
                    if pos > 0 && indices[pos] == indices[pos - 1] {
                        return Err(DatasetError::DuplicateSparseIndices {
                            feature_idx: i,
                            index: idx,
                        });
                    } else if idx as usize >= n_samples {
                        return Err(DatasetError::SparseIndexOutOfBounds {
                            feature_idx: i,
                            index: idx,
                            n_samples,
                        });
                    } else {
                        return Err(DatasetError::UnsortedSparseIndices { feature_idx: i });
                    }
                }
            }
        }

        // Validate targets
        if let Some(ref targets) = self.targets
            && targets.ncols() != n_samples
        {
            return Err(DatasetError::ShapeMismatch {
                expected: n_samples,
                got: targets.ncols(),
                field: "targets",
            });
        }

        // Validate weights
        if let Some(ref weights) = self.weights
            && weights.len() != n_samples
        {
            return Err(DatasetError::ShapeMismatch {
                expected: n_samples,
                got: weights.len(),
                field: "weights",
            });
        }

        // Preserve sparse storage - no densification!
        let schema = DatasetSchema::from_features(self.metas);

        Ok(Dataset::new(
            schema,
            self.features,
            self.targets,
            self.weights,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::super::schema::FeatureType;
    use super::*;
    use ndarray::array;

    #[test]
    fn dataset_from_array() {
        // Feature-major [n_features, n_samples] format: 2 features, 3 samples
        let features = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let targets = array![[0.0f32, 1.0, 0.0]]; // [n_outputs, n_samples]
        let ds = Dataset::from_array(features.view(), Some(targets), None);

        assert_eq!(ds.n_samples(), 3);
        assert_eq!(ds.n_features(), 2);
        assert_eq!(ds.n_outputs(), 1);
        assert!(ds.has_targets());
        assert!(!ds.has_weights());
        assert!(!ds.has_categorical());

        // Verify feature access matches input
        assert_eq!(ds.get_feature_value(0, 0), 1.0);
        assert_eq!(ds.get_feature_value(0, 1), 2.0);
        assert_eq!(ds.get_feature_value(0, 2), 3.0);
        assert_eq!(ds.get_feature_value(1, 0), 4.0);
        assert_eq!(ds.get_feature_value(1, 1), 5.0);
        assert_eq!(ds.get_feature_value(1, 2), 6.0);
    }

    #[test]
    fn dataset_from_array_feature_major() {
        // Feature-major [n_features, n_samples] format: 2 features, 3 samples
        let features = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let targets = array![[0.0f32, 1.0, 0.0]]; // [n_outputs, n_samples]
        let ds = Dataset::from_array(features.view(), Some(targets), None);

        assert_eq!(ds.n_samples(), 3);
        assert_eq!(ds.n_features(), 2);

        assert_eq!(ds.get_feature_value(0, 0), 1.0);
        assert_eq!(ds.get_feature_value(0, 2), 3.0);
        assert_eq!(ds.get_feature_value(1, 0), 4.0);
        assert_eq!(ds.get_feature_value(1, 2), 6.0);
    }

    #[test]
    fn dataset_from_array_features_only() {
        let features = array![[1.0f32, 2.0], [3.0, 4.0]]; // [n_features, n_samples]
        let ds = Dataset::from_array(features.view(), None, None);

        assert_eq!(ds.n_samples(), 2);
        assert_eq!(ds.n_features(), 2);
        assert_eq!(ds.n_outputs(), 0);
        assert!(!ds.has_targets());
    }

    #[test]
    fn dataset_with_weights() {
        let features = array![[1.0f32, 2.0]]; // [n_features=1, n_samples=2]
        let targets = array![[0.0f32, 1.0]]; // [n_outputs=1, n_samples=2]
        let weights = array![0.5f32, 1.5];

        let ds = Dataset::from_array(features.view(), Some(targets), Some(weights));

        assert!(ds.has_weights());
        assert_eq!(ds.weights().as_array().unwrap().to_vec(), vec![0.5, 1.5]);
    }

    #[test]
    fn dataset_feature_access() {
        let features = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]; // [n_features=2, n_samples=3]
        let targets = array![[0.0f32, 1.0, 0.0]]; // [n_outputs=1, n_samples=3]
        let ds = Dataset::from_array(features.view(), Some(targets), None);

        // Access via feature() method
        assert_eq!(ds.n_features(), 2);
        assert_eq!(ds.n_samples(), 3);

        // Feature 0 should be [1, 2, 3]
        match ds.feature(0) {
            Feature::Dense(values) => assert_eq!(values.to_vec(), vec![1.0, 2.0, 3.0]),
            _ => panic!("Expected dense feature"),
        }
        // Feature 1 should be [4, 5, 6]
        match ds.feature(1) {
            Feature::Dense(values) => assert_eq!(values.to_vec(), vec![4.0, 5.0, 6.0]),
            _ => panic!("Expected dense feature"),
        }
    }

    #[test]
    fn dataset_targets_view() {
        let features = array![[1.0f32, 2.0]]; // [n_features=1, n_samples=2]
        let targets = array![[0.0f32, 1.0], [1.0, 0.0]]; // [n_outputs=2, n_samples=2]
        let ds = Dataset::from_array(features.view(), Some(targets), None);

        let view = ds.targets().unwrap();
        assert_eq!(view.n_outputs(), 2);
        assert_eq!(view.n_samples(), 2);
        assert_eq!(view.output(0).to_vec(), vec![0.0, 1.0]);
        assert_eq!(view.output(1).to_vec(), vec![1.0, 0.0]);
    }

    #[test]
    fn dataset_targets_1d() {
        let features = array![[1.0f32, 2.0, 3.0]]; // [n_features=1, n_samples=3]
        let targets = array![[0.0f32, 1.0, 0.0]]; // [n_outputs=1, n_samples=3]
        let ds = Dataset::from_array(features.view(), Some(targets), None);

        assert_eq!(
            ds.targets().unwrap().as_single_output().to_vec(),
            vec![0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn builder_basic() {
        let ds = DatasetBuilder::new()
            .add_feature("x", array![1.0, 2.0, 3.0])
            .add_feature("y", array![4.0, 5.0, 6.0])
            .targets(array![[0.0, 1.0, 0.0]].view())
            .build()
            .unwrap();

        assert_eq!(ds.n_features(), 2);
        assert_eq!(ds.n_samples(), 3);
    }

    #[test]
    fn builder_with_categorical() {
        let ds = DatasetBuilder::new()
            .add_feature("age", array![25.0, 30.0])
            .add_categorical("color", array![0.0, 1.0])
            .targets(array![[0.0, 1.0]].view())
            .build()
            .unwrap();

        assert!(ds.has_categorical());
        assert_eq!(ds.schema().feature_type(0), FeatureType::Numeric);
        assert_eq!(ds.schema().feature_type(1), FeatureType::Categorical);
    }

    #[test]
    fn builder_with_sparse() {
        let ds = DatasetBuilder::new()
            .add_sparse("sparse_feature", vec![1, 3], vec![10.0, 30.0], 5, 0.0)
            .targets_1d(array![0.0, 1.0, 0.0, 1.0, 0.0].view())
            .build()
            .unwrap();

        assert_eq!(ds.n_features(), 1);
        assert_eq!(ds.n_samples(), 5);

        // Test via get_feature_value (supports sparse)
        assert_eq!(ds.get_feature_value(0, 0), 0.0); // default
        assert_eq!(ds.get_feature_value(0, 1), 10.0);
        assert_eq!(ds.get_feature_value(0, 3), 30.0);
    }

    #[test]
    fn builder_empty_features_error() {
        let result = DatasetBuilder::new()
            .targets(array![[0.0, 1.0]].view())
            .build();
        assert!(matches!(result, Err(DatasetError::EmptyFeatures)));
    }

    #[test]
    fn builder_shape_mismatch_error() {
        let result = DatasetBuilder::new()
            .add_feature("x", array![1.0, 2.0, 3.0])
            .add_feature("y", array![4.0, 5.0]) // wrong length
            .build();
        assert!(matches!(result, Err(DatasetError::ShapeMismatch { .. })));
    }

    #[test]
    fn builder_targets_mismatch_error() {
        let result = DatasetBuilder::new()
            .add_feature("x", array![1.0, 2.0, 3.0])
            .targets(array![[0.0, 1.0]].view()) // wrong length
            .build();
        assert!(matches!(result, Err(DatasetError::ShapeMismatch { .. })));
    }

    #[test]
    fn builder_unsorted_sparse_error() {
        let result = DatasetBuilder::new()
            .add_sparse("x", vec![3, 1], vec![1.0, 2.0], 5, 0.0) // unsorted
            .build();
        assert!(matches!(
            result,
            Err(DatasetError::UnsortedSparseIndices { .. })
        ));
    }

    #[test]
    fn builder_duplicate_sparse_error() {
        let result = DatasetBuilder::new()
            .add_sparse("x", vec![1, 1], vec![1.0, 2.0], 5, 0.0) // duplicate
            .build();
        assert!(matches!(
            result,
            Err(DatasetError::DuplicateSparseIndices { .. })
        ));
    }

    #[test]
    fn builder_sparse_out_of_bounds_error() {
        let result = DatasetBuilder::new()
            .add_sparse("x", vec![0, 10], vec![1.0, 2.0], 5, 0.0) // 10 >= 5
            .build();
        assert!(matches!(
            result,
            Err(DatasetError::SparseIndexOutOfBounds { .. })
        ));
    }

    // Verify Send + Sync
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn dataset_is_send_sync() {
        assert_send_sync::<Dataset>();
        assert_send_sync::<DatasetBuilder>();
    }

    #[test]
    fn features_are_contiguous() {
        // Create data in feature-major format [n_features, n_samples]
        let n_samples = 100;
        let n_features = 5;
        let mut data = Vec::with_capacity(n_features * n_samples);
        for f in 0..n_features {
            for s in 0..n_samples {
                data.push((f * n_samples + s) as f32);
            }
        }

        let features = Array2::from_shape_vec((n_features, n_samples), data).unwrap();
        let targets =
            Array2::from_shape_vec((1, n_samples), (0..n_samples).map(|i| i as f32).collect())
                .unwrap();

        let ds = Dataset::from_array(features.view(), Some(targets), None);

        // Each feature should be contiguous when stored as Dense
        for f in 0..n_features {
            match ds.feature(f) {
                Feature::Dense(values) => {
                    // Dense features have contiguous underlying storage
                    assert!(!values.is_empty(), "feature {} should have values", f);
                }
                _ => panic!("Expected dense feature"),
            }
        }
    }

    // =========================================================================
    // Feature Value Iteration Tests (RFC-0019)
    // =========================================================================

    #[test]
    fn for_each_feature_value_dense() {
        let ds = Dataset::from_array(
            array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].view(),
            None,
            None,
        );

        // Test feature 0
        let mut values = Vec::new();
        ds.for_each_feature_value_dense(0, |idx, val| {
            values.push((idx, val));
        });
        assert_eq!(values, vec![(0, 1.0), (1, 2.0), (2, 3.0)]);

        // Test feature 1
        values.clear();
        ds.for_each_feature_value_dense(1, |idx, val| {
            values.push((idx, val));
        });
        assert_eq!(values, vec![(0, 4.0), (1, 5.0), (2, 6.0)]);
    }

    #[test]
    fn for_each_feature_value_dense_sparse_includes_defaults() {
        let ds = DatasetBuilder::new()
            .add_sparse("x", vec![1, 3], vec![10.0, 30.0], 5, 0.0)
            .build()
            .unwrap();

        let mut values = Vec::new();
        ds.for_each_feature_value_dense(0, |idx, val| values.push((idx, val)));
        assert_eq!(
            values,
            vec![(0, 0.0), (1, 10.0), (2, 0.0), (3, 30.0), (4, 0.0)]
        );
    }

    #[test]
    fn for_each_feature_value_sum() {
        let ds = Dataset::from_array(array![[1.0f32, 2.0, 3.0, 4.0, 5.0]].view(), None, None);

        let mut sum = 0.0;
        ds.for_each_feature_value(0, |_idx, value| {
            sum += value;
        });
        assert_eq!(sum, 15.0);
    }

    #[test]
    fn gather_feature_values_basic() {
        let ds = Dataset::from_array(array![[1.0f32, 2.0, 3.0, 4.0, 5.0]].view(), None, None);

        let indices = [1u32, 3];
        let mut buffer = vec![0.0; 2];
        ds.gather_feature_values(0, &indices, &mut buffer);
        assert_eq!(buffer, vec![2.0, 4.0]);
    }

    #[test]
    fn gather_feature_values_all() {
        let ds = Dataset::from_array(array![[10.0f32, 20.0, 30.0]].view(), None, None);

        let indices: Vec<u32> = (0..3).collect();
        let mut buffer = vec![0.0; 3];
        ds.gather_feature_values(0, &indices, &mut buffer);
        assert_eq!(buffer, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn gather_feature_values_empty() {
        let ds = Dataset::from_array(array![[1.0f32, 2.0, 3.0]].view(), None, None);

        let indices: &[u32] = &[];
        let mut buffer: Vec<f32> = vec![];
        ds.gather_feature_values(0, indices, &mut buffer);
        assert!(buffer.is_empty());
    }

    #[test]
    fn for_each_gathered_value_basic() {
        let ds = Dataset::from_array(array![[1.0f32, 2.0, 3.0, 4.0, 5.0]].view(), None, None);

        let indices = [0u32, 2, 4];
        let mut values = Vec::new();
        ds.for_each_gathered_value(0, &indices, |local_idx, val| {
            values.push((local_idx, val));
        });
        assert_eq!(values, vec![(0, 1.0), (1, 3.0), (2, 5.0)]);
    }

    #[test]
    fn get_feature_value_basic() {
        let ds = Dataset::from_array(
            array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].view(),
            None,
            None,
        );

        assert_eq!(ds.get_feature_value(0, 0), 1.0);
        assert_eq!(ds.get_feature_value(0, 2), 3.0);
        assert_eq!(ds.get_feature_value(1, 1), 5.0);
    }

    // =========================================================================
    // Buffer Samples Tests
    // =========================================================================

    #[test]
    fn buffer_samples_single_block() {
        // 2 features, 3 samples
        // Feature 0: [1.0, 2.0, 3.0]
        // Feature 1: [4.0, 5.0, 6.0]
        let ds = Dataset::from_array(
            array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].view(),
            None,
            None,
        );

        // Buffer for all 3 samples, shape [3, 2] (samples × features)
        let mut buffer = Array2::<f32>::zeros((3, 2));
        let samples = ds.buffer_samples(&mut buffer, 0);

        assert_eq!(samples.n_samples(), 3);
        // Row 0 = sample 0: [feat0=1.0, feat1=4.0]
        assert_eq!(samples.get(0, 0), 1.0);
        assert_eq!(samples.get(0, 1), 4.0);
        // Row 1 = sample 1: [feat0=2.0, feat1=5.0]
        assert_eq!(samples.get(1, 0), 2.0);
        assert_eq!(samples.get(1, 1), 5.0);
        // Row 2 = sample 2: [feat0=3.0, feat1=6.0]
        assert_eq!(samples.get(2, 0), 3.0);
        assert_eq!(samples.get(2, 1), 6.0);
    }

    #[test]
    fn buffer_samples_partial_block() {
        // 1 feature, 5 samples
        let ds = Dataset::from_array(array![[1.0f32, 2.0, 3.0, 4.0, 5.0]].view(), None, None);

        // Buffer for 2 samples starting at index 3
        let mut buffer = Array2::<f32>::zeros((2, 1));
        let samples = ds.buffer_samples(&mut buffer, 3);

        assert_eq!(samples.n_samples(), 2);
        assert_eq!(samples.get(0, 0), 4.0); // sample 3
        assert_eq!(samples.get(1, 0), 5.0); // sample 4
    }

    #[test]
    fn buffer_samples_last_partial_block() {
        // 1 feature, 5 samples
        let ds = Dataset::from_array(array![[1.0f32, 2.0, 3.0, 4.0, 5.0]].view(), None, None);

        // Buffer for 3 samples starting at index 3 (only 2 available)
        let mut buffer = Array2::<f32>::zeros((3, 1));
        let samples = ds.buffer_samples(&mut buffer, 3);

        assert_eq!(samples.n_samples(), 2); // Only 2 samples available
        assert_eq!(samples.get(0, 0), 4.0); // sample 3
        assert_eq!(samples.get(1, 0), 5.0); // sample 4
    }

    #[test]
    fn buffer_samples_sparse_feature() {
        // Sparse feature: indices [1, 3], values [10.0, 30.0], default 0.0, n_samples=5
        let ds = DatasetBuilder::new()
            .add_sparse("sparse", vec![1, 3], vec![10.0, 30.0], 5, 0.0)
            .build()
            .unwrap();

        // Buffer for all 5 samples
        let mut buffer = Array2::<f32>::zeros((5, 1));
        let samples = ds.buffer_samples(&mut buffer, 0);

        assert_eq!(samples.n_samples(), 5);
        assert_eq!(samples.get(0, 0), 0.0); // default
        assert_eq!(samples.get(1, 0), 10.0); // sparse value
        assert_eq!(samples.get(2, 0), 0.0); // default
        assert_eq!(samples.get(3, 0), 30.0); // sparse value
        assert_eq!(samples.get(4, 0), 0.0); // default
    }

    #[test]
    fn buffer_samples_sparse_partial() {
        // Sparse feature: indices [1, 3, 7], values [10.0, 30.0, 70.0], default 0.0, n_samples=10
        let ds = DatasetBuilder::new()
            .add_sparse("sparse", vec![1, 3, 7], vec![10.0, 30.0, 70.0], 10, 0.0)
            .build()
            .unwrap();

        // Buffer for 4 samples starting at index 2
        let mut buffer = Array2::<f32>::zeros((4, 1));
        let samples = ds.buffer_samples(&mut buffer, 2);

        assert_eq!(samples.n_samples(), 4);
        assert_eq!(samples.get(0, 0), 0.0); // sample 2: default
        assert_eq!(samples.get(1, 0), 30.0); // sample 3: sparse value
        assert_eq!(samples.get(2, 0), 0.0); // sample 4: default
        assert_eq!(samples.get(3, 0), 0.0); // sample 5: default
    }

    #[test]
    fn buffer_samples_mixed_features() {
        // Feature 0: dense [1.0, 2.0, 3.0, 4.0]
        // Feature 1: sparse indices [1, 3], values [100.0, 300.0], default -1.0
        let ds = DatasetBuilder::new()
            .add_feature_unnamed(array![1.0f32, 2.0, 3.0, 4.0])
            .add_sparse("sparse", vec![1, 3], vec![100.0, 300.0], 4, -1.0)
            .build()
            .unwrap();

        // Buffer for all 4 samples
        let mut buffer = Array2::<f32>::zeros((4, 2));
        let samples = ds.buffer_samples(&mut buffer, 0);

        assert_eq!(samples.n_samples(), 4);
        // Sample 0: [1.0, -1.0]
        assert_eq!(samples.get(0, 0), 1.0);
        assert_eq!(samples.get(0, 1), -1.0);
        // Sample 1: [2.0, 100.0]
        assert_eq!(samples.get(1, 0), 2.0);
        assert_eq!(samples.get(1, 1), 100.0);
        // Sample 2: [3.0, -1.0]
        assert_eq!(samples.get(2, 0), 3.0);
        assert_eq!(samples.get(2, 1), -1.0);
        // Sample 3: [4.0, 300.0]
        assert_eq!(samples.get(3, 0), 4.0);
        assert_eq!(samples.get(3, 1), 300.0);
    }
}
