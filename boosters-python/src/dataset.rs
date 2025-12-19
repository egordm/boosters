//! Optional Dataset wrapper for NumPy arrays.
//!
//! **DEPRECATED**: Use `model.fit(X, y, ...)` directly instead.
//! This class will be removed in a future version.
//!
//! For most use cases, pass NumPy arrays directly to `fit()`.
//! This class was provided for users who wanted to pre-package data
//! with labels, weights, and feature names, but `fit()` now accepts
//! all of these parameters directly including categorical features.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// Optional dataset wrapper for feature matrix and labels.
///
/// **DEPRECATED**: Use `model.fit(X, y, ...)` directly instead.
/// This class will be removed in a future version.
///
/// The `fit()` method now accepts all parameters directly:
/// - `sample_weight` for weighted training
/// - `feature_names` for interpretability
/// - `categorical_features` for categorical feature support
///
/// # Example
/// ```python
/// from boosters import GBDTBooster
///
/// # Preferred: Direct fit
/// model = GBDTBooster()
/// model.fit(X, y, feature_names=['a', 'b'], categorical_features=[0])
/// ```
#[pyclass(name = "Dataset")]
#[derive(Clone)]
pub struct PyDataset {
    /// Feature matrix stored as row-major f32
    features: Vec<f32>,
    /// Number of rows
    n_rows: usize,
    /// Number of columns (features)
    n_cols: usize,
    /// Optional labels
    labels: Option<Vec<f32>>,
    /// Optional sample weights
    weights: Option<Vec<f32>>,
    /// Optional feature names
    feature_names: Option<Vec<String>>,
}

#[pymethods]
impl PyDataset {
    /// Create a new Dataset from NumPy arrays.
    ///
    /// # Arguments
    /// * `data` - Feature matrix of shape (n_samples, n_features)
    /// * `label` - Optional label array of shape (n_samples,)
    /// * `weight` - Optional weight array of shape (n_samples,)
    /// * `feature_names` - Optional list of feature names
    #[new]
    #[pyo3(signature = (data, label=None, weight=None, feature_names=None))]
    fn new(
        data: PyReadonlyArray2<f32>,
        label: Option<PyReadonlyArray1<f32>>,
        weight: Option<PyReadonlyArray1<f32>>,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let shape = data.shape();
        let n_rows = shape[0];
        let n_cols = shape[1];

        // Copy data to owned storage (NumPy array may not be contiguous)
        let features: Vec<f32> = data.as_slice()?.to_vec();

        // Copy labels if provided
        let labels = if let Some(y) = label {
            let y_slice = y.as_slice()?;
            if y_slice.len() != n_rows {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Label length {} doesn't match data rows {}",
                    y_slice.len(),
                    n_rows
                )));
            }
            Some(y_slice.to_vec())
        } else {
            None
        };

        // Copy weights if provided
        let weights = if let Some(w) = weight {
            let w_slice = w.as_slice()?;
            if w_slice.len() != n_rows {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Weight length {} doesn't match data rows {}",
                    w_slice.len(),
                    n_rows
                )));
            }
            Some(w_slice.to_vec())
        } else {
            None
        };

        // Validate feature names length
        if let Some(ref names) = feature_names {
            if names.len() != n_cols {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Feature names length {} doesn't match data columns {}",
                    names.len(),
                    n_cols
                )));
            }
        }

        Ok(Self {
            features,
            n_rows,
            n_cols,
            labels,
            weights,
            feature_names,
        })
    }

    /// Number of samples (rows).
    #[getter]
    fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of features (columns).
    #[getter]
    fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Feature names, if available.
    #[getter]
    fn feature_names(&self) -> Option<Vec<String>> {
        self.feature_names.clone()
    }

    /// Check if labels are available.
    fn has_label(&self) -> bool {
        self.labels.is_some()
    }

    /// Check if weights are available.
    fn has_weight(&self) -> bool {
        self.weights.is_some()
    }

    /// Get features as a NumPy array.
    fn get_features<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        PyArray2::from_vec2_bound(
            py,
            &self
                .features
                .chunks(self.n_cols)
                .map(|r| r.to_vec())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }

    /// Get labels as a NumPy array.
    fn get_labels<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f32>>> {
        self.labels
            .as_ref()
            .map(|l| PyArray1::from_slice_bound(py, l))
    }

    /// Get weights as a NumPy array.
    fn get_weights<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f32>>> {
        self.weights
            .as_ref()
            .map(|w| PyArray1::from_slice_bound(py, w))
    }
}

impl PyDataset {
    /// Get features as a slice (for internal use).
    pub fn features(&self) -> &[f32] {
        &self.features
    }

    /// Get labels as a slice (for internal use).
    pub fn labels(&self) -> Option<&[f32]> {
        self.labels.as_deref()
    }

    /// Get weights as a slice (for internal use).
    pub fn weights(&self) -> Option<&[f32]> {
        self.weights.as_deref()
    }

    /// Get the number of features.
    pub fn num_features(&self) -> usize {
        self.n_cols
    }

    /// Get the number of samples.
    pub fn num_samples(&self) -> usize {
        self.n_rows
    }

    /// Get the feature names.
    pub fn get_feature_names(&self) -> Option<Vec<String>> {
        self.feature_names.clone()
    }
}
