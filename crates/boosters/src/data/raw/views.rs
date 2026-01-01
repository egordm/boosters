//! View types for algorithm access.
//!
//! These provide read-only access to dataset components with appropriate
//! semantics for algorithms.

use ndarray::{ArrayView1, ArrayView2};

use super::schema::{DatasetSchema, FeatureType};

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
// WeightsView
// =============================================================================

/// Read-only view into sample weights.
///
/// Handles both weighted and unweighted cases uniformly.
/// Unweighted samples are treated as having weight 1.0.
///
/// # Design
///
/// Uses a simple `None`/`Some` pattern similar to `Option`:
/// - `None`: No weights specified - all samples have implicit weight 1.0
/// - `Some(ArrayView1)`: Explicit weights per sample
///
/// This is simpler than the previous design that required users to specify
/// `n_samples` even when they didn't care about weights.
///
/// # Example
///
/// ```ignore
/// use boosters::data::WeightsView;
/// use ndarray::array;
///
/// // No weights (all 1.0)
/// let uniform = WeightsView::none();
/// assert_eq!(uniform.get(0), 1.0);
///
/// // Custom weights
/// let weights = array![0.5, 2.0, 1.5];
/// let weighted = WeightsView::from_array(weights.view());
/// assert_eq!(weighted.get(1), 2.0);
///
/// // From optional (common pattern)
/// let view = WeightsView::from_optional(Some(weights.view()));
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub enum WeightsView<'a> {
    /// No weights - all samples have implicit weight 1.0.
    #[default]
    None,
    /// Explicit weights per sample.
    Some(ArrayView1<'a, f32>),
}

impl<'a> WeightsView<'a> {
    /// Create an empty weights view (all weights are 1.0).
    #[inline]
    pub fn none() -> Self {
        WeightsView::None
    }

    /// Create from weights array.
    #[inline]
    pub fn from_array(weights: ArrayView1<'a, f32>) -> Self {
        WeightsView::Some(weights)
    }

    /// Create from optional array.
    ///
    /// This is the common pattern for APIs that accept optional weights.
    #[inline]
    pub fn from_optional(weights: Option<ArrayView1<'a, f32>>) -> Self {
        match weights {
            Some(w) => WeightsView::Some(w),
            None => WeightsView::None,
        }
    }

    /// Get weight for a sample.
    ///
    /// Returns 1.0 for uniform weights, or the stored weight for weighted.
    #[inline]
    pub fn get(&self, idx: usize) -> f32 {
        match self {
            WeightsView::None => 1.0,
            WeightsView::Some(w) => w[idx],
        }
    }

    /// Returns true if no weights are specified (all 1.0).
    #[inline]
    pub fn is_none(&self) -> bool {
        matches!(self, WeightsView::None)
    }

    /// Returns true if explicit weights are specified.
    #[inline]
    pub fn is_some(&self) -> bool {
        matches!(self, WeightsView::Some(_))
    }

    /// Get the underlying array view if weighted.
    ///
    /// Returns `None` for uniform weights.
    #[inline]
    pub fn as_array(&self) -> Option<ArrayView1<'a, f32>> {
        match self {
            WeightsView::None => None,
            WeightsView::Some(w) => Some(*w),
        }
    }

    /// Iterate over weights for `n_samples` samples.
    ///
    /// For uniform weights, yields 1.0 for each sample.
    /// For explicit weights, yields the stored weights (and panics in debug
    /// mode if the length doesn't match).
    ///
    /// # Arguments
    ///
    /// * `n_samples` - Number of samples to iterate over. Required because
    ///   `WeightsView::None` has no length information.
    #[inline]
    pub fn iter(self, n_samples: usize) -> WeightsIter<'a> {
        match self {
            WeightsView::None => WeightsIter::Uniform {
                remaining: n_samples,
            },
            WeightsView::Some(w) => {
                debug_assert_eq!(w.len(), n_samples, "weights length mismatch");
                WeightsIter::Weighted { view: w, idx: 0 }
            }
        }
    }
}

/// Iterator over weights.
///
/// Uses enum dispatch to avoid branching on each iteration - the branch
/// is done once at construction time, not on every `.next()` call.
#[derive(Clone, Debug)]
pub enum WeightsIter<'a> {
    /// Yields 1.0 for remaining samples.
    Uniform {
        /// Remaining samples to yield.
        remaining: usize,
    },
    /// Yields from weights array.
    Weighted {
        /// The weights array view.
        view: ArrayView1<'a, f32>,
        /// Current index.
        idx: usize,
    },
}

impl<'a> Iterator for WeightsIter<'a> {
    type Item = f32;

    #[inline]
    fn next(&mut self) -> Option<f32> {
        match self {
            WeightsIter::Uniform { remaining } => {
                if *remaining > 0 {
                    *remaining -= 1;
                    Some(1.0)
                } else {
                    None
                }
            }
            WeightsIter::Weighted { view, idx } => {
                if *idx < view.len() {
                    let w = view[*idx];
                    *idx += 1;
                    Some(w)
                } else {
                    None
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = match self {
            WeightsIter::Uniform { remaining } => *remaining,
            WeightsIter::Weighted { view, idx } => view.len() - idx,
        };
        (len, Some(len))
    }
}

impl<'a> ExactSizeIterator for WeightsIter<'a> {}

// =============================================================================
// SamplesView (Sample-Major Layout)
// =============================================================================

/// Read-only view into feature data with sample-major layout.
///
/// Shape: `[n_samples, n_features]` - each sample's features are contiguous in memory.
/// This is useful for:
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
    pub fn from_slice(data: &'a [f32], n_samples: usize, n_features: usize) -> Option<Self> {
        ArrayView2::from_shape((n_samples, n_features), data)
            .ok()
            .map(|view| Self {
                data: view,
                schema: None,
            })
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
    #[inline]
    pub fn sample_view(&self, sample: usize) -> ArrayView1<'_, f32> {
        self.data.row(sample)
    }

    /// Get all sample values for a feature.
    ///
    /// **Warning**: This returns a strided view, not contiguous.
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn samples_view_basic() {
        // 3 samples, 2 features (sample-major)
        let data = array![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
        let schema = DatasetSchema::all_numeric(2);
        let view = SamplesView::new(data.view(), &schema);

        assert_eq!(view.n_samples(), 3);
        assert_eq!(view.n_features(), 2);
        assert_eq!(view.get(0, 0), 1.0);
        assert_eq!(view.get(0, 1), 4.0);
        assert_eq!(view.get(2, 0), 3.0);
        assert_eq!(view.sample_view(0).as_slice().unwrap(), &[1.0, 4.0]);
        assert_eq!(view.feature(0).to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn samples_view_feature_type() {
        use super::super::schema::FeatureMeta;

        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let schema =
            DatasetSchema::from_features(vec![FeatureMeta::numeric(), FeatureMeta::categorical()]);
        let view = SamplesView::new(data.view(), &schema);

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
        assert_send_sync::<SamplesView<'_>>();
        assert_send_sync::<TargetsView<'_>>();
        assert_send_sync::<WeightsView<'_>>();
        assert_send_sync::<WeightsIter<'_>>();
    }

    #[test]
    fn weights_view_none() {
        let view = WeightsView::none();
        assert!(view.is_none());
        assert!(!view.is_some());
        assert!(view.as_array().is_none());
        assert_eq!(view.get(0), 1.0);
        assert_eq!(view.get(1), 1.0);
        assert_eq!(view.get(2), 1.0);

        let collected: Vec<f32> = view.iter(3).collect();
        assert_eq!(collected, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn weights_view_some() {
        let weights = array![0.5f32, 2.0, 1.5];
        let view = WeightsView::from_array(weights.view());

        assert!(!view.is_none());
        assert!(view.is_some());
        assert!(view.as_array().is_some());
        assert_eq!(view.get(0), 0.5);
        assert_eq!(view.get(1), 2.0);
        assert_eq!(view.get(2), 1.5);

        let collected: Vec<f32> = view.iter(3).collect();
        assert_eq!(collected, vec![0.5, 2.0, 1.5]);
    }

    #[test]
    fn weights_view_from_optional() {
        // None -> none
        let uniform = WeightsView::from_optional(None);
        assert!(uniform.is_none());

        // Some -> some
        let weights = array![0.5f32, 2.0, 1.5];
        let weighted = WeightsView::from_optional(Some(weights.view()));
        assert!(weighted.is_some());
        assert_eq!(weighted.get(1), 2.0);
    }

    #[test]
    fn weights_iter_exact_size() {
        let weights = array![1.0f32, 2.0, 3.0];
        let view = WeightsView::from_array(weights.view());
        let iter = view.iter(3);
        assert_eq!(iter.len(), 3);

        let uniform = WeightsView::none();
        let iter = uniform.iter(5);
        assert_eq!(iter.len(), 5);
    }
}
