//! Dataset type for Python bindings.
//!
//! This module provides `Feature`, `Dataset` and `DatasetBuilder` for flexible dataset construction.
//!
//! # Design Notes
//!
//! The Python API is structured in layers:
//!
//! 1. **Feature**: Single feature column (dense or sparse) with metadata (name, categorical).
//!    Python conversions happen here - the Rust side just accepts validated numpy arrays.
//!
//! 2. **DatasetBuilder**: Collects features, labels, weights. Builds a `Dataset`.
//!
//! 3. **Dataset**: Immutable container for training/prediction.
//!
//! This design keeps Rust simple (just numpy arrays) while Python handles type coercion.

use ndarray::{Array1, Array2};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::error::BoostersError;
use boosters::data::{
    transpose_to_c_order, Dataset as CoreDataset, DatasetBuilder as CoreDatasetBuilder,
    DatasetSchema, Feature, FeatureMeta,
};

// =============================================================================
// Feature
// =============================================================================

/// A single feature column with metadata.
///
/// Features can be dense (array of values) or sparse (indices + values + default).
/// The Python wrapper handles all type conversions; this class just holds validated data.
///
/// Attributes:
///     name: Optional feature name.
///     categorical: Whether this feature is categorical.
///     n_samples: Number of samples in this feature.
///     is_sparse: Whether this is a sparse feature.
///
/// Example:
///     >>> from boosters import Feature
///     >>> # Dense feature
///     >>> f = Feature.dense(np.array([1.0, 2.0, 3.0], dtype=np.float32), name="age")
///     >>> # Sparse feature
///     >>> f = Feature.sparse(indices, values, n_samples=1000, name="rare", default=0.0)
#[gen_stub_pyclass]
#[pyclass(name = "Feature", module = "boosters._boosters_rs", subclass)]
#[derive(Clone)]
pub struct PyFeature {
    /// The underlying feature data.
    inner: Feature,
    /// Feature metadata.
    meta: FeatureMeta,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyFeature {
    /// Create a dense feature from a 1D float32 array.
    ///
    /// Args:
    ///     values: 1D float32 array of values (one per sample).
    ///     name: Optional feature name.
    ///     categorical: Whether this feature is categorical. Default: False.
    ///
    /// Returns:
    ///     A new dense Feature.
    #[staticmethod]
    #[pyo3(signature = (values, name=None, categorical=false))]
    pub fn dense(
        values: PyReadonlyArray1<'_, f32>,
        name: Option<String>,
        categorical: bool,
    ) -> Self {
        let values_owned: Array1<f32> = values.as_array().to_owned();
        let inner = Feature::dense(values_owned);
        let meta = if categorical {
            match name {
                Some(n) => FeatureMeta::categorical_named(n),
                None => FeatureMeta::categorical(),
            }
        } else {
            match name {
                Some(n) => FeatureMeta::numeric_named(n),
                None => FeatureMeta::numeric(),
            }
        };
        Self { inner, meta }
    }

    /// Create a sparse feature.
    ///
    /// Args:
    ///     indices: 1D uint32 array of row indices with non-default values.
    ///         Must be sorted in ascending order with no duplicates.
    ///     values: 1D float32 array of values at those indices.
    ///     n_samples: Total number of samples in the dataset.
    ///     name: Optional feature name.
    ///     default: Default value for unspecified indices. Default: 0.0.
    ///     categorical: Whether this feature is categorical. Default: False.
    ///
    /// Returns:
    ///     A new sparse Feature.
    #[staticmethod]
    #[pyo3(signature = (indices, values, n_samples, name=None, default=0.0, categorical=false))]
    pub fn sparse(
        indices: PyReadonlyArray1<'_, u32>,
        values: PyReadonlyArray1<'_, f32>,
        n_samples: usize,
        name: Option<String>,
        default: f32,
        categorical: bool,
    ) -> Self {
        let indices_vec: Vec<u32> = indices.as_array().to_vec();
        let values_vec: Vec<f32> = values.as_array().to_vec();
        let inner = Feature::sparse(indices_vec, values_vec, n_samples, default);
        let meta = if categorical {
            match name {
                Some(n) => FeatureMeta::categorical_named(n),
                None => FeatureMeta::categorical(),
            }
        } else {
            match name {
                Some(n) => FeatureMeta::numeric_named(n),
                None => FeatureMeta::numeric(),
            }
        };
        Self { inner, meta }
    }

    /// Feature name (if provided).
    #[getter]
    pub fn name(&self) -> Option<String> {
        self.meta.name.clone()
    }

    /// Whether this feature is categorical.
    #[getter]
    pub fn categorical(&self) -> bool {
        self.meta.feature_type.is_categorical()
    }

    /// Number of samples in this feature.
    #[getter]
    pub fn n_samples(&self) -> usize {
        self.inner.n_samples()
    }

    /// Whether this is a sparse feature.
    #[getter]
    pub fn is_sparse(&self) -> bool {
        self.inner.is_sparse()
    }

    fn __repr__(&self) -> String {
        let kind = if self.inner.is_sparse() {
            "sparse"
        } else {
            "dense"
        };
        let cat = if self.categorical() {
            ", categorical"
        } else {
            ""
        };
        match &self.meta.name {
            Some(name) => format!(
                "Feature({}, n_samples={}, name={:?}{})",
                kind,
                self.n_samples(),
                name,
                cat
            ),
            None => format!("Feature({}, n_samples={}{})", kind, self.n_samples(), cat),
        }
    }
}

impl PyFeature {
    /// Get the inner Feature.
    #[inline]
    pub fn inner(&self) -> &Feature {
        &self.inner
    }

    /// Get the feature metadata.
    #[inline]
    pub fn meta(&self) -> &FeatureMeta {
        &self.meta
    }

    /// Consume and return inner + meta.
    pub fn into_parts(self) -> (Feature, FeatureMeta) {
        (self.inner, self.meta)
    }
}

// =============================================================================
// Dataset
// =============================================================================

/// Internal dataset holding features, labels, and optional metadata.
///
/// This is a low-level binding that accepts pre-validated numpy arrays.
/// The Python `Dataset` class extends this to provide user-friendly
/// constructors with DataFrame support, type conversion, and validation.
///
/// Note:
///     This class expects C-contiguous float32 arrays. The Python subclass
///     handles all type conversion, validation, and DataFrame support.
///
/// Attributes:
///     n_samples: Number of samples in the dataset.
///     n_features: Number of features in the dataset.
///     has_labels: Whether labels are present.
///     has_weights: Whether weights are present.
///     feature_names: Feature names if provided.
///     categorical_features: Indices of categorical features.
///     shape: Shape as (n_samples, n_features).
#[gen_stub_pyclass]
#[pyclass(name = "Dataset", module = "boosters._boosters_rs", subclass, dict)]
#[derive(Clone)]
pub struct PyDataset {
    /// The core dataset containing features, labels, and weights.
    inner: CoreDataset,

    /// Feature names (optional, for display and export only)
    feature_names: Option<Vec<String>>,

    /// Indices of categorical features (for metadata, used in binning)
    categorical_features: Vec<usize>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyDataset {
    /// Create a new Dataset from pre-validated numpy arrays.
    ///
    /// Args:
    ///     features: C-contiguous float32 array of shape (n_samples, n_features).
    ///     labels: C-contiguous float32 array of shape (n_outputs, n_samples), or None.
    ///     weights: C-contiguous float32 array of shape (n_samples,), or None.
    ///     feature_names: List of feature names, or None.
    ///     categorical_features: List of categorical feature indices, or None.
    ///
    /// Returns:
    ///     Dataset ready for training or prediction.
    #[new]
    #[pyo3(signature = (features, labels=None, weights=None, feature_names=None, categorical_features=None))]
    pub fn new(
        features: PyReadonlyArray2<'_, f32>,
        labels: Option<PyReadonlyArray2<'_, f32>>,
        weights: Option<PyReadonlyArray1<'_, f32>>,
        feature_names: Option<Vec<String>>,
        categorical_features: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        // Get array views - pyo3 ndarray handles the conversion
        let features_view = features.as_array();

        // Transpose features to [n_features, n_samples] for CoreDataset
        let features_transposed = transpose_to_c_order(features_view);

        // Labels are already 2D [n_outputs, n_samples] - just copy
        let labels_2d: Option<Array2<f32>> = labels.map(|l| l.as_array().to_owned());

        // Extract weights if present
        let weights_1d: Option<Array1<f32>> = weights.map(|w| w.as_array().to_owned());

        // Use provided categorical features or empty
        let cats = categorical_features.unwrap_or_default();

        // Create schema with categorical feature information
        let n_features = features_transposed.nrows();
        let schema = DatasetSchema::with_categorical(n_features, &cats);

        // Convert each row to a dense Feature
        let feature_vec: Vec<Feature> = (0..n_features)
            .map(|i| Feature::dense(features_transposed.row(i).to_owned()))
            .collect();

        // Create the core dataset with new API
        let inner = CoreDataset::new(schema, feature_vec, labels_2d, weights_1d);

        Ok(Self {
            inner,
            feature_names,
            categorical_features: cats,
        })
    }

    /// Number of samples in the dataset.
    #[getter]
    pub fn n_samples(&self) -> usize {
        self.inner.n_samples()
    }

    /// Number of features in the dataset.
    #[getter]
    pub fn n_features(&self) -> usize {
        self.inner.n_features()
    }

    /// Whether labels are present.
    #[getter]
    pub fn has_labels(&self) -> bool {
        self.inner.n_outputs() > 0
    }

    /// Whether weights are present.
    #[getter]
    pub fn has_weights(&self) -> bool {
        self.inner.weights().is_some()
    }

    /// Feature names if provided.
    #[getter]
    pub fn feature_names(&self) -> Option<Vec<String>> {
        self.feature_names.clone()
    }

    /// Indices of categorical features.
    #[getter]
    pub fn categorical_features(&self) -> Vec<usize> {
        self.categorical_features.clone()
    }

    /// Shape of the features array as (n_samples, n_features).
    #[getter]
    pub fn shape(&self) -> (usize, usize) {
        (self.inner.n_samples(), self.inner.n_features())
    }

    fn __repr__(&self) -> String {
        format!(
            "Dataset(n_samples={}, n_features={}, has_labels={}, categorical_features={})",
            self.n_samples(),
            self.n_features(),
            self.has_labels(),
            self.categorical_features.len()
        )
    }
}

impl PyDataset {
    /// Get the inner CoreDataset.
    #[inline]
    pub fn inner(&self) -> &CoreDataset {
        &self.inner
    }

    /// Create PyDataset from CoreDataset with metadata.
    pub fn from_core(
        inner: CoreDataset,
        feature_names: Option<Vec<String>>,
        categorical_features: Vec<usize>,
    ) -> Self {
        Self {
            inner,
            feature_names,
            categorical_features,
        }
    }
}

impl AsRef<CoreDataset> for PyDataset {
    fn as_ref(&self) -> &CoreDataset {
        &self.inner
    }
}

// =============================================================================
// DatasetBuilder
// =============================================================================

/// Builder for constructing datasets with flexible feature types.
///
/// DatasetBuilder collects features (via `add_feature`) and builds a Dataset.
/// Features can be dense or sparse, with per-feature metadata.
///
/// Example:
///     >>> from boosters import DatasetBuilder, Feature
///     >>> builder = DatasetBuilder()
///     >>> builder.add_feature(Feature.dense(age_array, name="age"))
///     >>> builder.add_feature(Feature.sparse(idx, vals, n, name="rare"))
///     >>> builder.labels(y)
///     >>> dataset = builder.build()
#[gen_stub_pyclass]
#[pyclass(name = "DatasetBuilder", module = "boosters._boosters_rs", subclass)]
pub struct PyDatasetBuilder {
    /// Collected features with metadata.
    features: Vec<PyFeature>,
    /// Target labels (optional).
    targets: Option<Array2<f32>>,
    /// Sample weights (optional).
    weights: Option<Array1<f32>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyDatasetBuilder {
    /// Create a new empty DatasetBuilder.
    #[new]
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
            targets: None,
            weights: None,
        }
    }

    /// Add a feature to the builder.
    ///
    /// Args:
    ///     feature: A Feature object (created via Feature.dense() or Feature.sparse()).
    ///
    /// Returns:
    ///     Self for method chaining.
    pub fn add_feature<'py>(
        mut slf: PyRefMut<'py, Self>,
        feature: PyFeature,
    ) -> PyRefMut<'py, Self> {
        slf.features.push(feature);
        slf
    }

    /// Set 1D target labels.
    ///
    /// Args:
    ///     labels: 1D float32 array of labels (one per sample).
    ///
    /// Returns:
    ///     Self for method chaining.
    pub fn labels_1d<'py>(
        mut slf: PyRefMut<'py, Self>,
        labels: PyReadonlyArray1<'py, f32>,
    ) -> PyRefMut<'py, Self> {
        let labels_owned: Array1<f32> = labels.as_array().to_owned();
        let n = labels_owned.len();
        slf.targets = Some(
            labels_owned
                .into_shape_with_order((1, n))
                .expect("reshape should succeed"),
        );
        slf
    }

    /// Set 2D multi-output target labels.
    ///
    /// Args:
    ///     labels: 2D float32 array of shape (n_outputs, n_samples).
    ///
    /// Returns:
    ///     Self for method chaining.
    pub fn labels_2d<'py>(
        mut slf: PyRefMut<'py, Self>,
        labels: PyReadonlyArray2<'py, f32>,
    ) -> PyRefMut<'py, Self> {
        slf.targets = Some(labels.as_array().to_owned());
        slf
    }

    /// Set sample weights.
    ///
    /// Args:
    ///     weights: 1D float32 array of weights (one per sample).
    ///
    /// Returns:
    ///     Self for method chaining.
    pub fn weights<'py>(
        mut slf: PyRefMut<'py, Self>,
        weights: PyReadonlyArray1<'py, f32>,
    ) -> PyRefMut<'py, Self> {
        slf.weights = Some(weights.as_array().to_owned());
        slf
    }

    /// Build the dataset.
    ///
    /// Validates all features have the same sample count, indices are valid,
    /// and targets/weights match the sample count.
    ///
    /// Returns:
    ///     Constructed Dataset ready for training or prediction.
    ///
    /// Raises:
    ///     ValueError: If validation fails (shape mismatch, invalid sparse indices, etc.)
    pub fn build(slf: PyRefMut<'_, Self>) -> PyResult<PyDataset> {
        // Build the core dataset
        let mut core_builder = CoreDatasetBuilder::new();

        // Add features with metadata
        for py_feature in slf.features.iter() {
            core_builder = core_builder
                .add_feature_with_meta(py_feature.inner.clone(), py_feature.meta.clone());
        }

        // Add targets if present
        if let Some(ref targets) = slf.targets {
            core_builder = core_builder.targets(targets.view());
        }

        // Add weights if present
        if let Some(ref weights) = slf.weights {
            core_builder = core_builder.weights(weights.view());
        }

        // Build and handle errors
        let inner = core_builder.build().map_err(BoostersError::from)?;

        // Extract categorical indices from features
        let categorical_features: Vec<usize> = slf
            .features
            .iter()
            .enumerate()
            .filter(|(_, f)| f.meta.feature_type.is_categorical())
            .map(|(i, _)| i)
            .collect();

        // Collect feature names
        let feature_names: Option<Vec<String>> =
            if slf.features.iter().any(|f| f.meta.name.is_some()) {
                Some(
                    slf.features
                        .iter()
                        .enumerate()
                        .map(|(i, f)| f.meta.name.clone().unwrap_or_else(|| format!("f{}", i)))
                        .collect(),
                )
            } else {
                None
            };

        Ok(PyDataset::from_core(
            inner,
            feature_names,
            categorical_features,
        ))
    }

    /// Number of features added so far.
    #[getter]
    pub fn n_features(&self) -> usize {
        self.features.len()
    }

    fn __repr__(&self) -> String {
        let has_targets = self.targets.is_some();
        let has_weights = self.weights.is_some();
        format!(
            "DatasetBuilder(n_features={}, has_labels={}, has_weights={})",
            self.features.len(),
            has_targets,
            has_weights
        )
    }
}

impl Default for PyDatasetBuilder {
    fn default() -> Self {
        Self::new()
    }
}
