//! Dataset container and builder.
//!
//! This module provides [`Dataset`] and [`DatasetBuilder`] for creating
//! raw feature datasets. For training, create a [`crate::data::BinnedDataset`]
//! using [`BinnedDataset::from_dataset()`](crate::data::BinnedDataset::from_dataset).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::feature::Feature;
use super::schema::{DatasetSchema, FeatureMeta};
use super::views::{FeaturesView, TargetsView, WeightsView};
use crate::data::error::DatasetError;

/// The unified dataset container for all boosters models.
///
/// # Storage Layout
///
/// Features are stored in **feature-major** layout: `[n_features, n_samples]`.
/// Each feature's values across all samples are contiguous in memory.
///
/// Targets are stored as `[n_outputs, n_samples]` for consistency.
///
/// # Construction
///
/// Use [`Dataset::new`] for construction from feature-major matrices,
/// or [`Dataset::builder`] for complex scenarios with mixed feature types.
///
/// # Example
///
/// ```
/// use boosters::data::Dataset;
/// use ndarray::array;
///
/// // Feature-major format: 2 features, 3 samples
/// let features = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; // [n_features, n_samples]
/// let targets = array![[0.0, 1.0, 0.0]]; // [n_outputs, n_samples]
/// let ds = Dataset::new(features.view(), Some(targets.view()), None);
///
/// assert_eq!(ds.n_samples(), 3);
/// assert_eq!(ds.n_features(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Feature data: `[n_features, n_samples]` (feature-major).
    features: Array2<f32>,

    /// Feature metadata.
    schema: DatasetSchema,

    /// Target values: `[n_outputs, n_samples]`.
    targets: Option<Array2<f32>>,

    /// Sample weights: length = n_samples.
    weights: Option<Array1<f32>>,
}

impl Dataset {
    /// Create a dataset from feature-major data.
    ///
    /// This is the primary constructor. All data is expected in feature-major layout.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix `[n_features, n_samples]` (feature-major)
    /// * `targets` - Optional target matrix `[n_outputs, n_samples]`
    /// * `weights` - Optional sample weights (length = n_samples)
    ///
    /// All features are assumed to be numeric. For categorical features or
    /// mixed types, use [`Dataset::builder`].
    ///
    /// # Panics
    ///
    /// Debug-asserts that sample counts match across features, targets, and weights.
    ///
    /// # Example
    ///
    /// ```
    /// use boosters::data::Dataset;
    /// use ndarray::array;
    ///
    /// // Training: features with targets
    /// let features = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let targets = array![[0.0, 1.0, 0.0]];
    /// let ds = Dataset::new(features.view(), Some(targets.view()), None);
    ///
    /// // Prediction: features only
    /// let ds = Dataset::new(features.view(), None, None);
    /// ```
    pub fn new(
        features: ArrayView2<f32>,
        targets: Option<ArrayView2<f32>>,
        weights: Option<ArrayView1<f32>>,
    ) -> Self {
        let n_samples = features.ncols();
        let n_features = features.nrows();

        // Validate sample counts match
        if let Some(ref t) = targets {
            debug_assert_eq!(
                t.ncols(),
                n_samples,
                "targets must have same sample count as features"
            );
        }
        if let Some(ref w) = weights {
            debug_assert_eq!(
                w.len(),
                n_samples,
                "weights must have same sample count as features"
            );
        }

        let schema = DatasetSchema::all_numeric(n_features);

        Self {
            features: features.to_owned(),
            schema,
            targets: targets.map(|t| t.to_owned()),
            weights: weights.map(|w| w.to_owned()),
        }
    }

    /// Create a builder for complex dataset construction.
    pub fn builder() -> DatasetBuilder {
        DatasetBuilder::new()
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Number of samples.
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.features.ncols()
    }

    /// Number of features.
    #[inline]
    pub fn n_features(&self) -> usize {
        self.features.nrows()
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
    // Views
    // =========================================================================

    /// Get a view of the feature data.
    ///
    /// Shape: `[n_features, n_samples]` (feature-major).
    pub fn features(&self) -> FeaturesView<'_> {
        FeaturesView::new(self.features.view(), &self.schema)
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
    /// let ds = Dataset::new(
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
        // Current implementation: features stored as Array2<f32> (dense)
        // Future: will handle Feature::Sparse when Dataset preserves sparse storage
        let row = self.features.row(feature);
        if let Some(slice) = row.as_slice() {
            // Fast path: contiguous slice
            for (idx, &val) in slice.iter().enumerate() {
                f(idx, val);
            }
        } else {
            // Fallback: non-contiguous (shouldn't happen with feature-major layout)
            for (idx, &val) in row.iter().enumerate() {
                f(idx, val);
            }
        }
    }

    /// Iterate over all samples for a feature, filling in defaults for sparse features.
    ///
    /// For dense features, this is identical to `for_each_feature_value()`.
    /// For sparse features (when supported), this yields all n_samples values
    /// including default values for unspecified indices.
    ///
    /// Use this when you need ALL samples, including default values.
    #[inline]
    pub fn for_each_feature_value_dense<F>(&self, feature: usize, f: F)
    where
        F: FnMut(usize, f32),
    {
        // Current implementation: all features are dense
        // Future: will expand sparse features when Dataset preserves sparse storage
        self.for_each_feature_value(feature, f);
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
    /// let ds = Dataset::new(
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
    pub fn gather_feature_values(&self, feature: usize, sample_indices: &[u32], buffer: &mut [f32]) {
        debug_assert!(
            buffer.len() >= sample_indices.len(),
            "buffer must have length >= sample_indices.len()"
        );

        // Current implementation: dense array
        // Future: will handle sparse features with merge-join
        let row = self.features.row(feature);
        for (out_idx, &sample_idx) in sample_indices.iter().enumerate() {
            buffer[out_idx] = row[sample_idx as usize];
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
        // Current implementation: dense array
        // Future: will handle sparse features with merge-join
        let row = self.features.row(feature);
        for (out_idx, &sample_idx) in sample_indices.iter().enumerate() {
            f(out_idx, row[sample_idx as usize]);
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
        self.features[[feature, sample]]
    }

    // =========================================================================
    // Builder-style methods
    // =========================================================================

    /// Attach sample weights.
    ///
    /// # Panics
    ///
    /// Debug-asserts that weights length matches n_samples.
    pub fn with_weights(mut self, weights: Array1<f32>) -> Self {
        debug_assert_eq!(
            weights.len(),
            self.n_samples(),
            "weights length must match n_samples"
        );
        self.weights = Some(weights);
        self
    }

    /// Set the schema.
    ///
    /// # Panics
    ///
    /// Debug-asserts that schema has same number of features.
    pub fn with_schema(mut self, schema: DatasetSchema) -> Self {
        debug_assert_eq!(
            schema.n_features(),
            self.n_features(),
            "schema must have same number of features"
        );
        self.schema = schema;
        self
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
///     .add_feature("age", array![25.0, 30.0, 35.0].view())
///     .add_categorical("color", array![0.0, 1.0, 2.0].view())
///     .targets(array![[0.0, 1.0, 0.0]].view())
///     .build()
///     .unwrap();
///
/// assert_eq!(ds.n_features(), 2);
/// assert!(ds.has_categorical());
/// ```
#[derive(Debug, Default)]
#[deprecated]
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
    pub fn add_feature(mut self, name: &str, values: ArrayView1<f32>) -> Self {
        self.features.push(Feature::dense(values.to_owned()));
        self.metas.push(FeatureMeta::numeric_named(name));
        self
    }

    /// Add an unnamed numeric feature column.
    pub fn add_feature_unnamed(mut self, values: ArrayView1<f32>) -> Self {
        self.features.push(Feature::dense(values.to_owned()));
        self.metas.push(FeatureMeta::numeric());
        self
    }

    /// Add a categorical feature column.
    ///
    /// Values should be non-negative integers encoded as floats
    /// (e.g., 0.0, 1.0, 2.0).
    pub fn add_categorical(mut self, name: &str, values: ArrayView1<f32>) -> Self {
        self.features.push(Feature::dense(values.to_owned()));
        self.metas.push(FeatureMeta::categorical_named(name));
        self
    }

    /// Add an unnamed categorical feature column.
    pub fn add_categorical_unnamed(mut self, values: ArrayView1<f32>) -> Self {
        self.features.push(Feature::dense(values.to_owned()));
        self.metas.push(FeatureMeta::categorical());
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
        let n_features = self.features.len();

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

        // Build feature matrix [n_features, n_samples]
        // Note: Currently densifies sparse features. This will be changed
        // to preserve sparse storage when Dataset is updated to use Box<[Feature]>.
        let mut features_arr = Array2::zeros((n_features, n_samples));
        for (i, feat) in self.features.into_iter().enumerate() {
            let dense = feat.to_dense();
            features_arr.row_mut(i).assign(&dense);
        }

        let schema = DatasetSchema::from_features(self.metas);

        Ok(Dataset {
            features: features_arr,
            schema,
            targets: self.targets,
            weights: self.weights,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::schema::FeatureType;
    use ndarray::array;

    #[test]
    fn dataset_new() {
        // Feature-major [n_features, n_samples] format: 2 features, 3 samples
        let features = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let targets = array![[0.0, 1.0, 0.0]]; // [n_outputs, n_samples]
        let ds = Dataset::new(features.view(), Some(targets.view()), None);

        assert_eq!(ds.n_samples(), 3);
        assert_eq!(ds.n_features(), 2);
        assert_eq!(ds.n_outputs(), 1);
        assert!(ds.has_targets());
        assert!(!ds.has_weights());
        assert!(!ds.has_categorical());

        // Verify layout is correct (no transpose needed for feature-major)
        let view = ds.features();
        // Feature 0 should be [1, 2, 3]
        assert_eq!(view.feature(0).to_vec(), vec![1.0, 2.0, 3.0]);
        // Feature 1 should be [4, 5, 6]
        assert_eq!(view.feature(1).to_vec(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn dataset_new_feature_major() {
        // Feature-major [n_features, n_samples] format: 2 features, 3 samples
        let features = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let targets = array![[0.0, 1.0, 0.0]]; // [n_outputs, n_samples]
        let ds = Dataset::new(features.view(), Some(targets.view()), None);

        assert_eq!(ds.n_samples(), 3);
        assert_eq!(ds.n_features(), 2);

        let view = ds.features();
        assert_eq!(view.feature(0).to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(view.feature(1).to_vec(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn dataset_new_features_only() {
        let features = array![[1.0, 2.0], [3.0, 4.0]]; // [n_features, n_samples]
        let ds = Dataset::new(features.view(), None, None);

        assert_eq!(ds.n_samples(), 2);
        assert_eq!(ds.n_features(), 2);
        assert_eq!(ds.n_outputs(), 0);
        assert!(!ds.has_targets());
    }

    #[test]
    fn dataset_with_weights() {
        let features = array![[1.0, 2.0]]; // [n_features=1, n_samples=2]
        let targets = array![[0.0, 1.0]]; // [n_outputs=1, n_samples=2]
        let weights = array![0.5, 1.5];

        let ds = Dataset::new(features.view(), Some(targets.view()), Some(weights.view()));

        assert!(ds.has_weights());
        assert_eq!(ds.weights().as_array().unwrap().to_vec(), vec![0.5, 1.5]);
    }

    #[test]
    fn dataset_features_view() {
        let features = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; // [n_features=2, n_samples=3]
        let targets = array![[0.0, 1.0, 0.0]]; // [n_outputs=1, n_samples=3]
        let ds = Dataset::new(features.view(), Some(targets.view()), None);

        let view = ds.features();
        assert_eq!(view.n_features(), 2);
        assert_eq!(view.n_samples(), 3);
        assert_eq!(view.feature(0).to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(view.feature(1).to_vec(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn dataset_targets_view() {
        let features = array![[1.0, 2.0]]; // [n_features=1, n_samples=2]
        let targets = array![[0.0, 1.0], [1.0, 0.0]]; // [n_outputs=2, n_samples=2]
        let ds = Dataset::new(features.view(), Some(targets.view()), None);

        let view = ds.targets().unwrap();
        assert_eq!(view.n_outputs(), 2);
        assert_eq!(view.n_samples(), 2);
        assert_eq!(view.output(0).to_vec(), vec![0.0, 1.0]);
        assert_eq!(view.output(1).to_vec(), vec![1.0, 0.0]);
    }

    #[test]
    fn dataset_targets_1d() {
        let features = array![[1.0, 2.0, 3.0]]; // [n_features=1, n_samples=3]
        let targets = array![[0.0, 1.0, 0.0]]; // [n_outputs=1, n_samples=3]
        let ds = Dataset::new(features.view(), Some(targets.view()), None);

        assert_eq!(
            ds.targets().unwrap().as_single_output().to_vec(),
            vec![0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn builder_basic() {
        let ds = DatasetBuilder::new()
            .add_feature("x", array![1.0, 2.0, 3.0].view())
            .add_feature("y", array![4.0, 5.0, 6.0].view())
            .targets(array![[0.0, 1.0, 0.0]].view())
            .build()
            .unwrap();

        assert_eq!(ds.n_features(), 2);
        assert_eq!(ds.n_samples(), 3);
    }

    #[test]
    fn builder_with_categorical() {
        let ds = DatasetBuilder::new()
            .add_feature("age", array![25.0, 30.0].view())
            .add_categorical("color", array![0.0, 1.0].view())
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

        let view = ds.features();
        assert_eq!(view.get(0, 0), 0.0); // default
        assert_eq!(view.get(1, 0), 10.0);
        assert_eq!(view.get(3, 0), 30.0);
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
            .add_feature("x", array![1.0, 2.0, 3.0].view())
            .add_feature("y", array![4.0, 5.0].view()) // wrong length
            .build();
        assert!(matches!(result, Err(DatasetError::ShapeMismatch { .. })));
    }

    #[test]
    fn builder_targets_mismatch_error() {
        let result = DatasetBuilder::new()
            .add_feature("x", array![1.0, 2.0, 3.0].view())
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

        let ds = Dataset::new(features.view(), Some(targets.view()), None);
        let view = ds.features();

        // Each feature should be contiguous (accessible as slice)
        for f in 0..n_features {
            let feature = view.feature(f);
            assert!(
                feature.as_slice().is_some(),
                "feature {} should be contiguous",
                f
            );
        }
    }

    // =========================================================================
    // Feature Value Iteration Tests (RFC-0019)
    // =========================================================================

    #[test]
    fn for_each_feature_value_dense() {
        let ds = Dataset::new(
            array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].view(),
            None,
            None,
        );

        // Test feature 0
        let mut values = Vec::new();
        ds.for_each_feature_value(0, |idx, val| {
            values.push((idx, val));
        });
        assert_eq!(values, vec![(0, 1.0), (1, 2.0), (2, 3.0)]);

        // Test feature 1
        values.clear();
        ds.for_each_feature_value(1, |idx, val| {
            values.push((idx, val));
        });
        assert_eq!(values, vec![(0, 4.0), (1, 5.0), (2, 6.0)]);
    }

    #[test]
    fn for_each_feature_value_sum() {
        let ds = Dataset::new(
            array![[1.0, 2.0, 3.0, 4.0, 5.0]].view(),
            None,
            None,
        );

        let mut sum = 0.0;
        ds.for_each_feature_value(0, |_idx, value| {
            sum += value;
        });
        assert_eq!(sum, 15.0);
    }

    #[test]
    fn for_each_feature_value_dense_same_as_for_each() {
        let ds = Dataset::new(
            array![[1.0, 2.0, 3.0]].view(),
            None,
            None,
        );

        let mut values1 = Vec::new();
        let mut values2 = Vec::new();

        ds.for_each_feature_value(0, |idx, val| values1.push((idx, val)));
        ds.for_each_feature_value_dense(0, |idx, val| values2.push((idx, val)));

        assert_eq!(values1, values2);
    }

    #[test]
    fn gather_feature_values_basic() {
        let ds = Dataset::new(
            array![[1.0, 2.0, 3.0, 4.0, 5.0]].view(),
            None,
            None,
        );

        let indices = [1u32, 3];
        let mut buffer = vec![0.0; 2];
        ds.gather_feature_values(0, &indices, &mut buffer);
        assert_eq!(buffer, vec![2.0, 4.0]);
    }

    #[test]
    fn gather_feature_values_all() {
        let ds = Dataset::new(
            array![[10.0, 20.0, 30.0]].view(),
            None,
            None,
        );

        let indices: Vec<u32> = (0..3).collect();
        let mut buffer = vec![0.0; 3];
        ds.gather_feature_values(0, &indices, &mut buffer);
        assert_eq!(buffer, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn gather_feature_values_empty() {
        let ds = Dataset::new(
            array![[1.0, 2.0, 3.0]].view(),
            None,
            None,
        );

        let indices: &[u32] = &[];
        let mut buffer: Vec<f32> = vec![];
        ds.gather_feature_values(0, indices, &mut buffer);
        assert!(buffer.is_empty());
    }

    #[test]
    fn for_each_gathered_value_basic() {
        let ds = Dataset::new(
            array![[1.0, 2.0, 3.0, 4.0, 5.0]].view(),
            None,
            None,
        );

        let indices = [0u32, 2, 4];
        let mut values = Vec::new();
        ds.for_each_gathered_value(0, &indices, |local_idx, val| {
            values.push((local_idx, val));
        });
        assert_eq!(values, vec![(0, 1.0), (1, 3.0), (2, 5.0)]);
    }

    #[test]
    fn get_feature_value_basic() {
        let ds = Dataset::new(
            array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].view(),
            None,
            None,
        );

        assert_eq!(ds.get_feature_value(0, 0), 1.0);
        assert_eq!(ds.get_feature_value(0, 2), 3.0);
        assert_eq!(ds.get_feature_value(1, 1), 5.0);
    }
}
