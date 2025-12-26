//! Dataset type for holding training/prediction data.
//!
//! This module provides a minimal Rust binding that accepts pre-validated
//! numpy arrays. All type conversion and validation logic is in Python.

use ndarray::{Array1, Array2};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use boosters::data::{transpose_to_c_order, Dataset as CoreDataset};

/// Internal dataset holding features, labels, and optional metadata.
///
/// This is a low-level binding that accepts pre-validated numpy arrays.
/// Use the Python `Dataset` wrapper class for user-facing functionality.
///
/// Note:
///     This class expects C-contiguous float32 arrays. The Python wrapper
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
#[pyclass(name = "Dataset", module = "boosters._boosters_rs")]
pub struct PyDataset {
    /// The core dataset containing features, labels, and weights.
    inner: CoreDataset,

    /// Feature names (optional, for display and export only)
    feature_names: Option<Vec<String>>,

    /// Indices of categorical features (for metadata, used in binning)
    categorical_features: Vec<usize>,
}

impl Clone for PyDataset {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            feature_names: self.feature_names.clone(),
            categorical_features: self.categorical_features.clone(),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyDataset {
    /// Create a new Dataset from pre-validated numpy arrays.
    ///
    /// Args:
    ///     features: C-contiguous float32 array of shape (n_samples, n_features).
    ///     labels: C-contiguous float32 array of shape (n_samples,), or None.
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
        #[gen_stub(override_type(type_repr = "numpy.ndarray | None", imports = ("numpy",)))]
        labels: Option<PyReadonlyArray1<'_, f32>>,
        #[gen_stub(override_type(type_repr = "numpy.ndarray | None", imports = ("numpy",)))]
        weights: Option<PyReadonlyArray1<'_, f32>>,
        feature_names: Option<Vec<String>>,
        categorical_features: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        // Get array views - pyo3 ndarray handles the conversion
        let features_view = features.as_array();

        // Transpose features to [n_features, n_samples] for CoreDataset
        let features_transposed = transpose_to_c_order(features_view);

        // Convert 1D labels to 2D [1, n_samples] for CoreDataset
        // CoreDataset expects targets with shape [n_outputs, n_samples]
        let labels_2d: Option<Array2<f32>> = labels.map(|l| {
            let arr = l.as_array();
            let n = arr.len();
            // Reshape from [n_samples] to [1, n_samples] (1 output)
            arr.to_owned()
                .into_shape_with_order((1, n))
                .expect("labels reshape should succeed")
        });

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
    /// Get direct access to the inner CoreDataset.
    pub fn as_core(&self) -> &CoreDataset {
        &self.inner
    }

    /// Get mutable access to the inner CoreDataset.
    pub fn as_core_mut(&mut self) -> &mut CoreDataset {
        &mut self.inner
    }

    /// Consume and return the inner CoreDataset.
    pub fn into_core(self) -> CoreDataset {
        self.inner
    }
}
