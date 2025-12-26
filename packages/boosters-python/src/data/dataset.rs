//! Dataset type for holding training/prediction data.

use ndarray::{Array1, Array2};
use numpy::{PyArrayLike1, PyArrayLike2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use boosters::data::{transpose_to_c_order, Dataset as CoreDataset};

use crate::error::BoostersError;

/// Dataset holding features, labels, and optional metadata.
///
/// This class wraps NumPy arrays or pandas DataFrames for use with boosters models.
/// Data is converted to an internal representation on construction for efficient
/// training and prediction.
///
/// # Data Layout
///
/// - C-contiguous (row-major) float32 arrays provide optimal performance
/// - F-contiguous arrays are automatically converted to C-order
/// - float64 arrays are supported but will be converted to float32
///
/// # Categorical Features
///
/// Categorical features can be:
/// 1. Auto-detected from pandas categorical dtype
/// 2. Specified explicitly via `categorical_features` parameter
///
/// # Missing Values
///
/// NaN values in features are treated as missing (like XGBoost).
/// Inf values in features or NaN/Inf in labels raise errors.
///
/// # Example
///
/// ```python
/// import numpy as np
/// from boosters import Dataset
///
/// X = np.random.rand(100, 10).astype(np.float32)
/// y = np.random.rand(100).astype(np.float32)
///
/// dataset = Dataset(X, y)
/// print(f"Samples: {dataset.n_samples}, Features: {dataset.n_features}")
/// ```
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
    /// Create a new Dataset from features and optional labels.
    ///
    /// Args:
    ///     features: 2D NumPy array or pandas DataFrame of shape (n_samples, n_features)
    ///     labels: Optional 1D array of shape (n_samples,)
    ///     weights: Optional 1D array of sample weights (n_samples,)
    ///     groups: Optional 1D array of group labels for ranking (not yet implemented)
    ///     feature_names: Optional list of feature names
    ///     categorical_features: Optional list of categorical feature indices
    ///
    /// Returns:
    ///     Dataset ready for training or prediction
    ///
    /// Raises:
    ///     ValueError: If data is invalid (shape mismatch, Inf values, etc.)
    ///     TypeError: If data types are unsupported
    #[new]
    #[pyo3(signature = (features, labels=None, weights=None, groups=None, feature_names=None, categorical_features=None))]
    pub fn new(
        py: Python<'_>,
        features: PyObject,
        labels: Option<PyObject>,
        weights: Option<PyObject>,
        groups: Option<PyObject>,
        feature_names: Option<Vec<String>>,
        categorical_features: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        // Groups not yet implemented
        if groups.is_some() {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "groups parameter is not yet implemented",
            ));
        }

        // Extract and validate features - convert to [n_features, n_samples]
        let (features_array, detected_cats, extracted_names) =
            Self::extract_features(py, &features)?;
        let n_samples = features_array.ncols();
        let n_features = features_array.nrows();

        // Merge feature names: explicit > extracted from DataFrame
        let final_feature_names = feature_names.or(extracted_names);

        // Merge categorical features: user-specified + auto-detected
        let mut all_cats = categorical_features.unwrap_or_default();
        for cat in detected_cats {
            if !all_cats.contains(&cat) {
                all_cats.push(cat);
            }
        }
        all_cats.sort();

        // Validate categorical indices
        for &idx in &all_cats {
            if idx >= n_features {
                return Err(BoostersError::InvalidParameter {
                    name: "categorical_features".to_string(),
                    reason: format!("index {} out of range for {} features", idx, n_features),
                }
                .into());
            }
        }

        // Extract labels if present - convert to [1, n_samples]
        let labels_array = if let Some(ref labels_obj) = labels {
            Some(Self::extract_labels(py, labels_obj, n_samples)?)
        } else {
            None
        };

        // Extract weights if present
        let weights_array = if let Some(ref weights_obj) = weights {
            Some(Self::extract_weights(py, weights_obj, n_samples)?)
        } else {
            None
        };

        // Create the core dataset
        let inner = CoreDataset::new(
            features_array.view(),
            labels_array.as_ref().map(|l| l.view()),
            weights_array.as_ref().map(|w| w.view()),
        );

        Ok(Self {
            inner,
            feature_names: final_feature_names,
            categorical_features: all_cats,
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

    /// Extract features from various Python input types.
    ///
    /// Returns: (features [n_features, n_samples], detected_categoricals, feature_names)
    fn extract_features(
        py: Python<'_>,
        features: &PyObject,
    ) -> PyResult<(Array2<f32>, Vec<usize>, Option<Vec<String>>)> {
        let features_bound = features.bind(py);

        // Check if it's a pandas DataFrame
        if let Ok(result) = Self::try_extract_dataframe(py, features_bound) {
            return Ok(result);
        }

        // Check if it's a NumPy array
        if let Ok(result) = Self::try_extract_numpy_array(py, features_bound) {
            return Ok(result);
        }

        // Check for scipy sparse matrices and provide helpful error
        if Self::is_sparse_matrix(py, features_bound)? {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "Sparse matrices are not yet supported. Convert to dense array with .toarray() \
                or use a pandas DataFrame. Sparse support is planned for a future release.",
            ));
        }

        // Unknown type
        let type_name = features_bound
            .get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_else(|_| "unknown".to_string());
        Err(BoostersError::TypeError(format!(
            "expected numpy array or pandas DataFrame, got {}",
            type_name
        ))
        .into())
    }

    /// Check if an object is a scipy sparse matrix.
    fn is_sparse_matrix(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(scipy_sparse) = py.import_bound("scipy.sparse") {
            if let Ok(issparse) = scipy_sparse.getattr("issparse") {
                if let Ok(result) = issparse.call1((obj,)) {
                    if let Ok(is_sparse) = result.extract::<bool>() {
                        return Ok(is_sparse);
                    }
                }
            }
        }
        Ok(false)
    }

    /// Try to extract a pandas DataFrame.
    fn try_extract_dataframe(
        py: Python<'_>,
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<(Array2<f32>, Vec<usize>, Option<Vec<String>>)> {
        // Check for DataFrame by checking for 'values' and 'dtypes' attributes
        if !obj.hasattr("values")? || !obj.hasattr("dtypes")? {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Not a pandas DataFrame",
            ));
        }

        // Get values as numpy array
        let values = obj.getattr("values")?;

        // Get shape
        let shape = values.getattr("shape")?;
        let shape_tuple: (usize, usize) = shape.extract()?;
        let (n_samples, n_features) = shape_tuple;

        if n_samples == 0 {
            return Err(BoostersError::InvalidParameter {
                name: "features".to_string(),
                reason: "DataFrame must have at least one row".to_string(),
            }
            .into());
        }

        // Detect categorical columns
        let mut detected_cats = Vec::new();
        if let Ok(dtypes) = obj.getattr("dtypes") {
            for i in 0..n_features {
                if let Ok(dtype) = dtypes.get_item(i) {
                    let dtype_name: String = dtype.getattr("name")?.extract()?;
                    if dtype_name == "category" || dtype_name.starts_with("category") {
                        detected_cats.push(i);
                    }
                }
            }
        }

        // Extract column names
        let feature_names = if let Ok(columns) = obj.getattr("columns") {
            if let Ok(col_list) = columns.call_method0("tolist") {
                col_list.extract::<Vec<String>>().ok()
            } else {
                None
            }
        } else {
            None
        };

        // Convert to numpy float32 and extract
        let numpy = py.import_bound("numpy")?;
        let arr_f32 = numpy
            .getattr("ascontiguousarray")?
            .call1((values, numpy.getattr("float32")?))?;

        // Extract as numpy array and transpose to feature-major [n_features, n_samples]
        let arr_view = arr_f32.extract::<PyArrayLike2<f32>>()?;
        let arr_ro = arr_view.try_readonly()?;
        let features_transposed = transpose_to_c_order(arr_ro.as_array());

        Ok((features_transposed, detected_cats, feature_names))
    }

    /// Try to extract a NumPy array.
    fn try_extract_numpy_array(
        py: Python<'_>,
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<(Array2<f32>, Vec<usize>, Option<Vec<String>>)> {
        // Check for ndim and shape attributes (numpy array)
        if !obj.hasattr("ndim")? || !obj.hasattr("shape")? || !obj.hasattr("dtype")? {
            return Err(pyo3::exceptions::PyTypeError::new_err("Not a numpy array"));
        }

        // Check dimensions
        let ndim: usize = obj.getattr("ndim")?.extract()?;
        if ndim != 2 {
            return Err(BoostersError::InvalidParameter {
                name: "features".to_string(),
                reason: format!("expected 2D array, got {}D", ndim),
            }
            .into());
        }

        // Get shape
        let shape = obj.getattr("shape")?;
        let shape_tuple: (usize, usize) = shape.extract()?;
        let (n_samples, _n_features) = shape_tuple;

        if n_samples == 0 {
            return Err(BoostersError::InvalidParameter {
                name: "features".to_string(),
                reason: "array must have at least one sample".to_string(),
            }
            .into());
        }

        // Convert to float32 and ensure C-contiguous
        let numpy = py.import_bound("numpy")?;
        let arr_f32 = numpy
            .getattr("ascontiguousarray")?
            .call1((obj, numpy.getattr("float32")?))?;

        // Extract as numpy array and transpose to feature-major [n_features, n_samples]
        let arr_view = arr_f32.extract::<PyArrayLike2<f32>>()?;
        let arr_ro = arr_view.try_readonly()?;
        let features_transposed = transpose_to_c_order(arr_ro.as_array());

        // No categorical detection or feature names for raw numpy arrays
        Ok((features_transposed, Vec::new(), None))
    }

    /// Extract and validate labels array.
    ///
    /// Returns labels as [1, n_samples] for CoreDataset format.
    fn extract_labels(
        py: Python<'_>,
        labels: &PyObject,
        expected_samples: usize,
    ) -> PyResult<Array2<f32>> {
        let labels_bound = labels.bind(py);

        // Check for ndim
        if !labels_bound.hasattr("ndim")? {
            let type_name = labels_bound
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "unknown".to_string());
            return Err(BoostersError::TypeError(format!(
                "labels: expected numpy array, got {}",
                type_name
            ))
            .into());
        }

        let ndim: usize = labels_bound.getattr("ndim")?.extract()?;
        if ndim != 1 {
            return Err(BoostersError::InvalidParameter {
                name: "labels".to_string(),
                reason: format!("expected 1D array, got {}D", ndim),
            }
            .into());
        }

        // Check shape
        let shape = labels_bound.getattr("shape")?;
        let shape_tuple: (usize,) = shape.extract()?;
        let n_samples = shape_tuple.0;

        if n_samples != expected_samples {
            return Err(BoostersError::InvalidParameter {
                name: "labels".to_string(),
                reason: format!(
                    "shape mismatch: expected {} samples, got {}",
                    expected_samples, n_samples
                ),
            }
            .into());
        }

        // Check for NaN/Inf in labels
        let numpy = py.import_bound("numpy")?;
        let isfinite = numpy.getattr("isfinite")?;
        let all_fn = numpy.getattr("all")?;
        let all_finite: bool = all_fn.call1((isfinite.call1((labels_bound,))?,))?.extract()?;

        if !all_finite {
            return Err(BoostersError::InvalidParameter {
                name: "labels".to_string(),
                reason: "labels contain NaN or Inf values".to_string(),
            }
            .into());
        }

        // Convert to float32 if needed
        let arr_f32 = numpy
            .getattr("ascontiguousarray")?
            .call1((labels_bound, numpy.getattr("float32")?))?;

        // Extract as 1D array
        let arr_view = arr_f32.extract::<PyArrayLike1<f32>>()?;
        let arr_ro = arr_view.try_readonly()?;
        let labels_1d: Array1<f32> = arr_ro.as_array().to_owned();

        // Reshape to [1, n_samples] for CoreDataset format
        let labels_2d = labels_1d.into_shape((1, expected_samples)).map_err(|_| {
            BoostersError::ValidationError("Failed to reshape labels".to_string())
        })?;

        Ok(labels_2d)
    }

    /// Extract and validate weights array.
    fn extract_weights(
        py: Python<'_>,
        weights: &PyObject,
        expected_samples: usize,
    ) -> PyResult<Array1<f32>> {
        let weights_bound = weights.bind(py);

        // Check for ndim
        if !weights_bound.hasattr("ndim")? {
            let type_name = weights_bound
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "unknown".to_string());
            return Err(BoostersError::TypeError(format!(
                "weights: expected numpy array, got {}",
                type_name
            ))
            .into());
        }

        let ndim: usize = weights_bound.getattr("ndim")?.extract()?;
        if ndim != 1 {
            return Err(BoostersError::InvalidParameter {
                name: "weights".to_string(),
                reason: format!("expected 1D array, got {}D", ndim),
            }
            .into());
        }

        // Check shape
        let shape = weights_bound.getattr("shape")?;
        let shape_tuple: (usize,) = shape.extract()?;
        let n_samples = shape_tuple.0;

        if n_samples != expected_samples {
            return Err(BoostersError::InvalidParameter {
                name: "weights".to_string(),
                reason: format!(
                    "shape mismatch: expected {} samples, got {}",
                    expected_samples, n_samples
                ),
            }
            .into());
        }

        // Convert to float32 if needed
        let numpy = py.import_bound("numpy")?;
        let arr_f32 = numpy
            .getattr("ascontiguousarray")?
            .call1((weights_bound, numpy.getattr("float32")?))?;

        // Extract as 1D array
        let arr_view = arr_f32.extract::<PyArrayLike1<f32>>()?;
        let arr_ro = arr_view.try_readonly()?;
        Ok(arr_ro.as_array().to_owned())
    }
}

#[cfg(test)]
mod tests {
    // Tests are in Python - they require numpy and pandas
}
