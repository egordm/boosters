//! Dataset and EvalSet types for Python bindings.
//!
//! This module provides the `Dataset` wrapper for NumPy arrays and pandas DataFrames,
//! along with `EvalSet` for named evaluation datasets.
//!
//! # Design Notes
//!
//! - Dataset is marked as `subclass` so Python can extend it with convenience methods
//! - Categorical features can be auto-detected from pandas or specified explicitly
//! - NaN in features is allowed (treated as missing values)
//! - Inf in features or NaN/Inf in labels raise errors

use ndarray::{Array1, Array2};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use boosters::data::{transpose_to_c_order, Dataset as CoreDataset};
use boosters::training::EvalSet as CoreEvalSet;

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
#[pyclass(name = "Dataset", module = "boosters._boosters_rs", subclass)]
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

        // Create the core dataset
        let inner = CoreDataset::new(
            features_transposed.view(),
            labels_2d.as_ref().map(|l| l.view()),
            weights_1d.as_ref().map(|w| w.view()),
        );

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
}

impl AsRef<CoreDataset> for PyDataset {
    fn as_ref(&self) -> &CoreDataset {
        &self.inner
    }
}

// =============================================================================
// EvalSet
// =============================================================================

/// Named evaluation set for model training.
///
/// An EvalSet wraps a Dataset with a name, which is used to identify
/// the evaluation set in training logs and `eval_results`.
///
/// # Example
///
/// ```python
/// from boosters import Dataset, EvalSet
/// import numpy as np
///
/// X_val = np.random.rand(50, 10).astype(np.float32)
/// y_val = np.random.rand(50).astype(np.float32)
///
/// val_data = Dataset(X_val, y_val)
/// eval_set = EvalSet("validation", val_data)
///
/// # Use in training:
/// # model.fit(train_data, valid=[eval_set])
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "EvalSet", module = "boosters._boosters_rs")]
pub struct PyEvalSet {
    /// Name of the evaluation set (used in logs and eval_results)
    name: String,

    /// The underlying dataset
    dataset: Py<PyDataset>,
}

impl Clone for PyEvalSet {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            name: self.name.clone(),
            dataset: self.dataset.clone_ref(py),
        })
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyEvalSet {
    /// Create a new named evaluation set.
    ///
    /// Args:
    ///     name: Name for this evaluation set (e.g., "validation", "test")
    ///     dataset: Dataset containing features and labels.
    ///
    /// Returns:
    ///     EvalSet ready for use in training
    #[new]
    pub fn new(dataset: Py<PyDataset>, name: String) -> Self {
        Self { name, dataset }
    }

    /// Name of this evaluation set.
    #[getter]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The underlying dataset.
    #[getter]
    pub fn dataset(&self, py: Python<'_>) -> Py<PyDataset> {
        self.dataset.clone_ref(py)
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let dataset = self.dataset.bind(py);
        let n_samples = dataset.borrow().n_samples();
        format!("EvalSet(name='{}', n_samples={})", self.name, n_samples)
    }
}

impl PyEvalSet {
    /// Get the name as a string slice.
    #[inline]
    pub fn name_str(&self) -> &str {
        &self.name
    }

    /// Get the dataset Py reference.
    #[inline]
    pub fn dataset_ref(&self) -> &Py<PyDataset> {
        &self.dataset
    }

    /// Convert to a CoreEvalSet by borrowing the inner dataset.
    ///
    /// The returned CoreEvalSet has the same lifetime as the borrowed PyDataset.
    #[inline]
    pub fn to_core_eval_set<'a>(
        &'a self,
        py: Python<'a>,
    ) -> (PyRef<'a, PyDataset>, impl FnOnce(&'a PyDataset) -> CoreEvalSet<'a>) {
        let dataset_ref = self.dataset.bind(py).borrow();
        let name = &self.name;
        (dataset_ref, move |ds: &'a PyDataset| CoreEvalSet::new(name, ds.inner()))
    }
}
