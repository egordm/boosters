//! View types for algorithm access.
//!
//! These provide read-only access to dataset components with appropriate
//! semantics for algorithms.

use ndarray::{ArrayView1, ArrayView2};

use super::schema::{DatasetSchema, FeatureType};

/// Read-only view into feature data.
///
/// Internal storage is feature-major: `[n_features, n_samples]`.
/// This means:
/// - `feature(f)` returns all samples for feature f (contiguous)
/// - `sample(s)` returns all features for sample s (strided)
///
/// The API uses conceptual terms (sample, feature) not array terms (row, col).
/// Schema is optional - when not provided, all features are assumed numeric.
#[derive(Clone, Copy)]
pub struct FeaturesView<'a> {
    /// Shape: [n_features, n_samples] - feature-major
    data: ArrayView2<'a, f32>,
    /// Optional schema. If None, all features are assumed numeric.
    schema: Option<&'a DatasetSchema>,
}

impl<'a> FeaturesView<'a> {
    /// Create a new features view with schema.
    ///
    /// # Arguments
    ///
    /// * `data` - Array with shape `[n_features, n_samples]`
    /// * `schema` - Feature metadata
    pub fn new(data: ArrayView2<'a, f32>, schema: &'a DatasetSchema) -> Self {
        debug_assert_eq!(
            data.nrows(),
            schema.n_features(),
            "data.nrows() must match schema.n_features()"
        );
        Self {
            data,
            schema: Some(schema),
        }
    }

    /// Create a features view without schema (all features assumed numeric).
    ///
    /// # Arguments
    ///
    /// * `data` - Array with shape `[n_features, n_samples]`
    pub fn from_array(data: ArrayView2<'a, f32>) -> Self {
        Self { data, schema: None }
    }

    /// Create from a contiguous slice in feature-major order.
    ///
    /// This is zero-copy.
    ///
    /// Data layout: `[f0_s0, f0_s1, ..., f1_s0, f1_s1, ...]`
    ///
    /// # Arguments
    ///
    /// * `data` - Slice of length `n_samples * n_features`
    /// * `n_samples` - Number of samples
    /// * `n_features` - Number of features
    ///
    /// # Returns
    ///
    /// `None` if the slice length doesn't match `n_samples * n_features`.
    pub fn from_slice(
        data: &'a [f32],
        n_samples: usize,
        n_features: usize,
    ) -> Option<Self> {
        // Shape is [n_features, n_samples] for feature-major
        ArrayView2::from_shape((n_features, n_samples), data)
            .ok()
            .map(|view| Self { data: view, schema: None })
    }

    /// Number of samples (second dimension).
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.data.ncols()
    }

    /// Number of features (first dimension).
    #[inline]
    pub fn n_features(&self) -> usize {
        self.data.nrows()
    }

    /// Get feature value at (sample, feature).
    ///
    /// Internally accesses `[feature, sample]` due to storage layout.
    #[inline]
    pub fn get(&self, sample: usize, feature: usize) -> f32 {
        self.data[[feature, sample]]
    }

    /// Get a contiguous view of all sample values for a feature.
    ///
    /// This is the fast path - returns a contiguous slice.
    #[inline]
    pub fn feature(&self, feature: usize) -> ArrayView1<'_, f32> {
        self.data.row(feature)
    }

    /// Get all features for a sample.
    ///
    /// **Warning**: This returns a strided view, not contiguous.
    /// For performance-critical code, consider block buffering instead.
    /// For tree traversal, use the `DataAccessor::sample` method instead.
    #[inline]
    pub fn sample_view(&self, sample: usize) -> ArrayView1<'_, f32> {
        self.data.column(sample)
    }

    /// Get the type of a feature.
    ///
    /// Returns `Numeric` if no schema was provided.
    #[inline]
    pub fn feature_type(&self, feature: usize) -> FeatureType {
        self.schema
            .map(|s| s.feature_type(feature))
            .unwrap_or(FeatureType::Numeric)
    }

    /// Get the underlying array view.
    ///
    /// Shape is `[n_features, n_samples]`.
    pub fn view(&self) -> ArrayView2<'a, f32> {
        self.data
    }

    /// Get the schema, if available.
    pub fn schema(&self) -> Option<&DatasetSchema> {
        self.schema
    }

    /// Check if any feature is categorical.
    ///
    /// Returns `false` if no schema was provided.
    pub fn has_categorical(&self) -> bool {
        self.schema.map(|s| s.has_categorical()).unwrap_or(false)
    }
}

impl<'a> std::fmt::Debug for FeaturesView<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeaturesView")
            .field("n_features", &self.n_features())
            .field("n_samples", &self.n_samples())
            .finish()
    }
}

/// A strided sample view for feature-major data.
///
/// This wrapper exists because `FeaturesView` stores data in feature-major order,
/// so accessing a sample requires strided access. This implements `SampleAccessor`
/// for use with tree traversal.
#[derive(Clone, Copy)]
pub struct StridedSample<'a> {
    data: ArrayView1<'a, f32>,
}

impl<'a> crate::data::SampleAccessor for StridedSample<'a> {
    #[inline]
    fn feature(&self, index: usize) -> f32 {
        self.data[index]
    }

    #[inline]
    fn n_features(&self) -> usize {
        self.data.len()
    }
}

// Enable tree traversal with FeaturesView (feature-major layout)
impl<'a> crate::data::DataAccessor for FeaturesView<'a> {
    type Sample<'b> = StridedSample<'b> where Self: 'b;

    #[inline]
    fn sample(&self, index: usize) -> Self::Sample<'_> {
        StridedSample {
            data: self.data.column(index),
        }
    }

    #[inline]
    fn n_samples(&self) -> usize {
        self.data.ncols()
    }

    #[inline]
    fn n_features(&self) -> usize {
        self.data.nrows()
    }

    #[inline]
    fn feature_type(&self, feature: usize) -> FeatureType {
        self.schema
            .map(|s| s.feature_type(feature))
            .unwrap_or(FeatureType::Numeric)
    }

    #[inline]
    fn has_categorical(&self) -> bool {
        self.schema.map(|s| s.has_categorical()).unwrap_or(false)
    }
}

/// Read-only view into target values.
///
/// Shape: `[n_outputs, n_samples]` - each output's values are contiguous.
#[derive(Clone, Copy)]
pub struct TargetsView<'a> {
    data: ArrayView2<'a, f32>,
}

impl<'a> TargetsView<'a> {
    /// Create a new targets view.
    ///
    /// # Arguments
    ///
    /// * `data` - Array with shape `[n_outputs, n_samples]`
    pub fn new(data: ArrayView2<'a, f32>) -> Self {
        Self { data }
    }

    /// Number of samples.
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.data.ncols()
    }

    /// Number of output dimensions.
    #[inline]
    pub fn n_outputs(&self) -> usize {
        self.data.nrows()
    }

    /// Get target value for a sample and output.
    #[inline]
    pub fn get(&self, sample: usize, output: usize) -> f32 {
        self.data[[output, sample]]
    }

    /// Get all samples for an output dimension.
    ///
    /// This is contiguous.
    #[inline]
    pub fn output(&self, output: usize) -> ArrayView1<'_, f32> {
        self.data.row(output)
    }

    /// Get as a 1D view for single-output targets.
    ///
    /// # Panics
    ///
    /// Panics if `n_outputs() != 1`.
    pub fn as_single_output(&self) -> ArrayView1<'_, f32> {
        assert_eq!(
            self.n_outputs(),
            1,
            "as_single_output() requires n_outputs == 1, got {}",
            self.n_outputs()
        );
        self.data.row(0)
    }

    /// Get the underlying array view.
    ///
    /// Shape is `[n_outputs, n_samples]`.
    pub fn view(&self) -> ArrayView2<'a, f32> {
        self.data
    }
}

impl<'a> std::fmt::Debug for TargetsView<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TargetsView")
            .field("n_outputs", &self.n_outputs())
            .field("n_samples", &self.n_samples())
            .finish()
    }
}

// =============================================================================
// SamplesView (Sample-Major Layout)
// =============================================================================

/// Read-only view into feature data with sample-major layout.
///
/// Shape: `[n_samples, n_features]` - each sample's features are contiguous in memory.
/// This is the transpose of [`FeaturesView`] and is useful for:
/// - Tree traversal (where we access all features for one sample)
/// - Block-buffered prediction (where we transpose chunks for cache efficiency)
///
/// The API uses conceptual terms (sample, feature) not array terms (row, col).
/// Schema is optional - when not provided, all features are assumed numeric.
#[derive(Clone, Copy)]
pub struct SamplesView<'a> {
    /// Shape: [n_samples, n_features] - sample-major
    data: ArrayView2<'a, f32>,
    /// Optional schema. If None, all features are assumed numeric.
    schema: Option<&'a DatasetSchema>,
}

impl<'a> SamplesView<'a> {
    /// Create a new samples view with schema.
    ///
    /// # Arguments
    ///
    /// * `data` - Array with shape `[n_samples, n_features]`, must be C-order (row-major)
    /// * `schema` - Feature metadata
    ///
    /// # Panics (debug only)
    ///
    /// Debug-asserts that the array is in standard (C) layout.
    pub fn new(data: ArrayView2<'a, f32>, schema: &'a DatasetSchema) -> Self {
        debug_assert!(data.is_standard_layout(), "Array must be in C-order");
        debug_assert_eq!(
            data.ncols(),
            schema.n_features(),
            "data.ncols() must match schema.n_features()"
        );
        Self {
            data,
            schema: Some(schema),
        }
    }

    /// Create a samples view without schema (all features assumed numeric).
    ///
    /// # Arguments
    ///
    /// * `data` - Array with shape `[n_samples, n_features]`, must be C-order
    ///
    /// # Panics (debug only)
    ///
    /// Debug-asserts that the array is in standard (C) layout.
    pub fn from_array(data: ArrayView2<'a, f32>) -> Self {
        debug_assert!(data.is_standard_layout(), "Array must be in C-order");
        Self { data, schema: None }
    }

    /// Create from a contiguous slice in sample-major (row-major) order.
    ///
    /// This is zero-copy.
    ///
    /// # Arguments
    ///
    /// * `data` - Slice of length `n_samples * n_features`
    /// * `n_samples` - Number of samples (rows)
    /// * `n_features` - Number of features (columns)
    ///
    /// # Returns
    ///
    /// `None` if the slice length doesn't match `n_samples * n_features`.
    pub fn from_slice(
        data: &'a [f32],
        n_samples: usize,
        n_features: usize,
    ) -> Option<Self> {
        ArrayView2::from_shape((n_samples, n_features), data)
            .ok()
            .map(|view| Self { data: view, schema: None })
    }

    /// Number of samples (first dimension).
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.data.nrows()
    }

    /// Number of features (second dimension).
    #[inline]
    pub fn n_features(&self) -> usize {
        self.data.ncols()
    }

    /// Get feature value at (sample, feature).
    #[inline]
    pub fn get(&self, sample: usize, feature: usize) -> f32 {
        self.data[[sample, feature]]
    }

    /// Get all features for a sample as a contiguous slice.
    ///
    /// This is the fast path - returns a contiguous slice.
    /// For tree traversal, use the `DataAccessor::sample` method instead.
    #[inline]
    pub fn sample_view(&self, sample: usize) -> ArrayView1<'_, f32> {
        self.data.row(sample)
    }

    /// Get all sample values for a feature.
    ///
    /// **Warning**: This returns a strided view, not contiguous.
    /// For performance-critical code needing feature-contiguous access,
    /// use [`FeaturesView`] instead.
    #[inline]
    pub fn feature(&self, feature: usize) -> ArrayView1<'_, f32> {
        self.data.column(feature)
    }

    /// Get the type of a feature.
    ///
    /// Returns `Numeric` if no schema was provided.
    #[inline]
    pub fn feature_type(&self, feature: usize) -> FeatureType {
        self.schema
            .map(|s| s.feature_type(feature))
            .unwrap_or(FeatureType::Numeric)
    }

    /// Get the underlying array view.
    ///
    /// Shape is `[n_samples, n_features]`.
    pub fn view(&self) -> ArrayView2<'a, f32> {
        self.data
    }

    /// Get the schema, if available.
    pub fn schema(&self) -> Option<&DatasetSchema> {
        self.schema
    }

    /// Check if any feature is categorical.
    ///
    /// Returns `false` if no schema was provided.
    pub fn has_categorical(&self) -> bool {
        self.schema.map(|s| s.has_categorical()).unwrap_or(false)
    }
}

impl<'a> std::fmt::Debug for SamplesView<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SamplesView")
            .field("n_samples", &self.n_samples())
            .field("n_features", &self.n_features())
            .finish()
    }
}

// Enable tree traversal with SamplesView (sample-major layout)
impl<'a> crate::data::DataAccessor for SamplesView<'a> {
    // Samples are contiguous, so we can return a slice directly
    type Sample<'b> = &'b [f32] where Self: 'b;

    #[inline]
    fn sample(&self, index: usize) -> Self::Sample<'_> {
        // For contiguous C-order arrays, get slice directly from underlying storage.
        // SamplesView invariant guarantees C-order layout, so unwrap is safe.
        let slice = self.data.as_slice().unwrap();
        let start = index * self.data.ncols();
        let end = start + self.data.ncols();
        &slice[start..end]
    }

    #[inline]
    fn n_samples(&self) -> usize {
        self.data.nrows()
    }

    #[inline]
    fn n_features(&self) -> usize {
        self.data.ncols()
    }

    #[inline]
    fn feature_type(&self, feature: usize) -> FeatureType {
        self.schema
            .map(|s| s.feature_type(feature))
            .unwrap_or(FeatureType::Numeric)
    }

    #[inline]
    fn has_categorical(&self) -> bool {
        self.schema.map(|s| s.has_categorical()).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{DataAccessor, SampleAccessor};
    use ndarray::array;

    #[test]
    fn features_view_basic() {
        // 2 features, 3 samples: [[1,2,3], [4,5,6]]
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let schema = DatasetSchema::all_numeric(2);
        let view = FeaturesView::new(data.view(), &schema);

        assert_eq!(view.n_features(), 2);
        assert_eq!(view.n_samples(), 3);
        assert_eq!(view.get(0, 0), 1.0); // sample 0, feature 0
        assert_eq!(view.get(0, 1), 4.0); // sample 0, feature 1
        assert_eq!(view.get(2, 0), 3.0); // sample 2, feature 0
    }

    #[test]
    fn features_view_feature_contiguous() {
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let schema = DatasetSchema::all_numeric(2);
        let view = FeaturesView::new(data.view(), &schema);

        // feature() returns contiguous slice
        let f0 = view.feature(0);
        assert!(f0.as_slice().is_some());
        assert_eq!(f0.as_slice().unwrap(), &[1.0, 2.0, 3.0]);

        let f1 = view.feature(1);
        assert_eq!(f1.as_slice().unwrap(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn features_view_sample_strided() {
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let schema = DatasetSchema::all_numeric(2);
        let view = FeaturesView::new(data.view(), &schema);

        // sample() returns StridedSample which implements SampleAccessor
        let s0 = view.sample(0);
        // Verify via SampleAccessor::feature
        assert_eq!(s0.feature(0), 1.0); // feature 0
        assert_eq!(s0.feature(1), 4.0); // feature 1
        assert_eq!(s0.n_features(), 2);
    }

    #[test]
    fn features_view_feature_type() {
        use super::super::schema::FeatureMeta;

        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let schema = DatasetSchema::from_features(vec![
            FeatureMeta::numeric(),
            FeatureMeta::categorical(),
        ]);
        let view = FeaturesView::new(data.view(), &schema);

        assert_eq!(view.feature_type(0), FeatureType::Numeric);
        assert_eq!(view.feature_type(1), FeatureType::Categorical);
        assert!(view.has_categorical());
    }

    #[test]
    fn targets_view_basic() {
        // 2 outputs, 3 samples
        let data = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]];
        let view = TargetsView::new(data.view());

        assert_eq!(view.n_outputs(), 2);
        assert_eq!(view.n_samples(), 3);
        assert_eq!(view.get(0, 0), 0.0);
        assert_eq!(view.get(1, 1), 0.0);
    }

    #[test]
    fn targets_view_single_output() {
        let data = array![[0.0, 1.0, 0.0]];
        let view = TargetsView::new(data.view());

        let single = view.as_single_output();
        assert_eq!(single.to_vec(), vec![0.0, 1.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "n_outputs == 1")]
    fn targets_view_single_output_panics_multi() {
        let data = array![[0.0, 1.0], [1.0, 0.0]];
        let view = TargetsView::new(data.view());
        view.as_single_output(); // should panic
    }

    // Verify Send + Sync
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn views_are_send_sync() {
        assert_send_sync::<FeaturesView<'_>>();
        assert_send_sync::<TargetsView<'_>>();
    }
}
