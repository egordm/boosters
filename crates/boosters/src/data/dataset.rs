//! Dataset container and builder.
//!
//! This module provides [`Dataset`] and [`DatasetBuilder`].

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::column::Column;
use super::error::DatasetError;
use super::schema::{DatasetSchema, FeatureMeta};
use super::views::{FeaturesView, TargetsView, WeightsView};

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
pub struct DatasetBuilder {
    columns: Vec<Column>,
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
        self.columns.push(Column::dense(values.to_owned()));
        self.metas.push(FeatureMeta::numeric_named(name));
        self
    }

    /// Add an unnamed numeric feature column.
    pub fn add_feature_unnamed(mut self, values: ArrayView1<f32>) -> Self {
        self.columns.push(Column::dense(values.to_owned()));
        self.metas.push(FeatureMeta::numeric());
        self
    }

    /// Add a categorical feature column.
    ///
    /// Values should be non-negative integers encoded as floats
    /// (e.g., 0.0, 1.0, 2.0).
    pub fn add_categorical(mut self, name: &str, values: ArrayView1<f32>) -> Self {
        self.columns.push(Column::dense(values.to_owned()));
        self.metas.push(FeatureMeta::categorical_named(name));
        self
    }

    /// Add an unnamed categorical feature column.
    pub fn add_categorical_unnamed(mut self, values: ArrayView1<f32>) -> Self {
        self.columns.push(Column::dense(values.to_owned()));
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
        self.columns
            .push(Column::sparse(indices, values, n_samples, default));
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
    /// - Columns have inconsistent sample counts
    /// - Targets have wrong sample count
    /// - Weights have wrong length
    /// - Sparse indices are unsorted or have duplicates
    pub fn build(self) -> Result<Dataset, DatasetError> {
        if self.columns.is_empty() {
            return Err(DatasetError::EmptyFeatures);
        }

        // Determine n_samples from first column
        let n_samples = self.columns[0].n_samples();
        let n_features = self.columns.len();

        // Validate all columns have same n_samples
        for (i, col) in self.columns.iter().enumerate() {
            if col.n_samples() != n_samples {
                return Err(DatasetError::ShapeMismatch {
                    expected: n_samples,
                    got: col.n_samples(),
                    field: "features",
                });
            }

            // Validate sparse columns
            if let Column::Sparse(sparse) = col {
                if let Err((pos, idx)) = sparse.validate() {
                    // Check if it's unsorted or duplicate
                    if pos > 0 && sparse.indices[pos] == sparse.indices[pos - 1] {
                        return Err(DatasetError::DuplicateSparseIndices {
                            feature_idx: i,
                            index: idx,
                        });
                    } else {
                        return Err(DatasetError::UnsortedSparseIndices { feature_idx: i });
                    }
                }
                // Check bounds
                for &idx in &sparse.indices {
                    if idx as usize >= n_samples {
                        return Err(DatasetError::SparseIndexOutOfBounds {
                            feature_idx: i,
                            index: idx,
                            n_samples,
                        });
                    }
                }
            }
        }

        // Validate targets
        if let Some(ref targets) = self.targets {
            if targets.ncols() != n_samples {
                return Err(DatasetError::ShapeMismatch {
                    expected: n_samples,
                    got: targets.ncols(),
                    field: "targets",
                });
            }
        }

        // Validate weights
        if let Some(ref weights) = self.weights {
            if weights.len() != n_samples {
                return Err(DatasetError::ShapeMismatch {
                    expected: n_samples,
                    got: weights.len(),
                    field: "weights",
                });
            }
        }

        // Build feature matrix [n_features, n_samples]
        let mut features = Array2::zeros((n_features, n_samples));
        for (i, col) in self.columns.into_iter().enumerate() {
            let dense = col.to_dense();
            features.row_mut(i).assign(&dense);
        }

        let schema = DatasetSchema::from_features(self.metas);

        Ok(Dataset {
            features,
            schema,
            targets: self.targets,
            weights: self.weights,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::FeatureType;
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

        assert_eq!(ds.targets().unwrap().as_single_output().to_vec(), vec![0.0, 1.0, 0.0]);
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
        let targets = Array2::from_shape_vec(
            (1, n_samples),
            (0..n_samples).map(|i| i as f32).collect(),
        )
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
}
