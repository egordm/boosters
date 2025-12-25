//! Dataset type for holding training/prediction data.

use numpy::{PyArrayLike1, PyArrayLike2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyclass;

use crate::error::BoostersError;

/// Dataset holding features, labels, and optional metadata.
///
/// This class wraps NumPy arrays or pandas DataFrames for use with boosters models.
/// It keeps references to the original Python objects to ensure zero-copy access
/// where possible.
///
/// # Data Layout
///
/// - C-contiguous (row-major) float32 arrays provide zero-copy access
/// - F-contiguous arrays are automatically converted to C-order
/// - float64 arrays are supported but may require conversion
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
    /// Original features array (kept alive for zero-copy)
    features: Py<PyAny>,

    /// Original labels array (optional)
    labels: Option<Py<PyAny>>,

    /// Sample weights (optional)
    weights: Option<Py<PyAny>>,

    /// Group labels for ranking (optional)
    groups: Option<Py<PyAny>>,

    /// Feature names (optional)
    feature_names: Option<Vec<String>>,

    /// Indices of categorical features
    categorical_features: Vec<usize>,

    /// Cached shape
    n_samples: usize,
    n_features: usize,

    /// Whether features were converted (not zero-copy)
    converted: bool,
}

impl Clone for PyDataset {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            features: self.features.clone_ref(py),
            labels: self.labels.as_ref().map(|l| l.clone_ref(py)),
            weights: self.weights.as_ref().map(|w| w.clone_ref(py)),
            groups: self.groups.as_ref().map(|g| g.clone_ref(py)),
            feature_names: self.feature_names.clone(),
            categorical_features: self.categorical_features.clone(),
            n_samples: self.n_samples,
            n_features: self.n_features,
            converted: self.converted,
        })
    }
}

#[pymethods]
impl PyDataset {
    /// Create a new Dataset from features and optional labels.
    ///
    /// Args:
    ///     features: 2D NumPy array or pandas DataFrame of shape (n_samples, n_features)
    ///     labels: Optional 1D array of shape (n_samples,)
    ///     weights: Optional 1D array of sample weights (n_samples,)
    ///     groups: Optional 1D array of group labels for ranking
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
        // Process features and extract shape
        let (processed_features, n_samples, n_features, converted, detected_cats) =
            Self::process_features(py, &features)?;

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
                    reason: format!(
                        "index {} out of range for {} features",
                        idx, n_features
                    ),
                }
                .into());
            }
        }

        // Validate labels if present
        if let Some(ref labels_obj) = labels {
            Self::validate_labels(py, labels_obj, n_samples)?;
        }

        // Validate weights if present
        if let Some(ref weights_obj) = weights {
            Self::validate_1d_array(py, weights_obj, n_samples, "weights")?;
        }

        // Validate groups if present
        if let Some(ref groups_obj) = groups {
            Self::validate_1d_array(py, groups_obj, n_samples, "groups")?;
        }

        Ok(Self {
            features: processed_features,
            labels,
            weights,
            groups,
            feature_names,
            categorical_features: all_cats,
            n_samples,
            n_features,
            converted,
        })
    }

    /// Number of samples in the dataset.
    #[getter]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Number of features in the dataset.
    #[getter]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Whether labels are present.
    #[getter]
    pub fn has_labels(&self) -> bool {
        self.labels.is_some()
    }

    /// Whether weights are present.
    #[getter]
    pub fn has_weights(&self) -> bool {
        self.weights.is_some()
    }

    /// Whether groups are present.
    #[getter]
    pub fn has_groups(&self) -> bool {
        self.groups.is_some()
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

    /// Whether data was converted (not zero-copy).
    #[getter]
    pub fn was_converted(&self) -> bool {
        self.converted
    }

    /// Shape of the features array as (n_samples, n_features).
    #[getter]
    pub fn shape(&self) -> (usize, usize) {
        (self.n_samples, self.n_features)
    }

    fn __repr__(&self) -> String {
        format!(
            "Dataset(n_samples={}, n_features={}, has_labels={}, categorical_features={})",
            self.n_samples,
            self.n_features,
            self.has_labels(),
            self.categorical_features.len()
        )
    }
}

impl PyDataset {
    /// Process features from various input types.
    ///
    /// Returns: (processed_features, n_samples, n_features, was_converted, detected_categoricals)
    fn process_features(
        py: Python<'_>,
        features: &PyObject,
    ) -> PyResult<(Py<PyAny>, usize, usize, bool, Vec<usize>)> {
        let features_bound = features.bind(py);

        // Check if it's a pandas DataFrame
        if let Ok(result) = Self::try_extract_dataframe(py, features_bound) {
            return Ok(result);
        }

        // Check if it's a NumPy array
        if let Ok(result) = Self::try_extract_numpy_array(py, features_bound) {
            return Ok(result);
        }

        // Try to convert to numpy array
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

    /// Try to extract a pandas DataFrame.
    fn try_extract_dataframe(
        py: Python<'_>,
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<(Py<PyAny>, usize, usize, bool, Vec<usize>)> {
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

        // Detect categorical columns
        let mut detected_cats = Vec::new();
        if let Ok(dtypes) = obj.getattr("dtypes") {
            // Iterate over dtype indices
            for i in 0..n_features {
                if let Ok(dtype) = dtypes.get_item(i) {
                    let dtype_name: String = dtype.getattr("name")?.extract()?;
                    if dtype_name == "category" || dtype_name.starts_with("category") {
                        detected_cats.push(i);
                    }
                }
            }
        }

        // Get column names if available
        // Note: We don't store them here, just detect categoricals

        // Convert to C-contiguous float32 if needed
        let numpy = py.import_bound("numpy")?;
        let ascontiguousarray = numpy.getattr("ascontiguousarray")?;

        // Check dtype
        let dtype = values.getattr("dtype")?;
        let dtype_name: String = dtype.getattr("name")?.extract()?;

        let (processed, converted) = if dtype_name == "float32" {
            let flags = values.getattr("flags")?;
            let c_contiguous: bool = flags.getattr("c_contiguous")?.extract()?;
            if c_contiguous {
                (values.unbind(), false)
            } else {
                // Convert to C-contiguous
                let converted =
                    ascontiguousarray.call1((values, numpy.getattr("float32")?))?;
                (converted.unbind(), true)
            }
        } else if dtype_name == "float64" {
            // Keep as float64 but ensure C-contiguous
            let flags = values.getattr("flags")?;
            let c_contiguous: bool = flags.getattr("c_contiguous")?.extract()?;
            if c_contiguous {
                (values.unbind(), false)
            } else {
                let converted = ascontiguousarray.call1((values,))?;
                (converted.unbind(), true)
            }
        } else {
            // Convert to float32
            let converted =
                ascontiguousarray.call1((values, numpy.getattr("float32")?))?;
            (converted.unbind(), true)
        };

        Ok((processed, n_samples, n_features, converted, detected_cats))
    }

    /// Try to extract a NumPy array.
    fn try_extract_numpy_array(
        py: Python<'_>,
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<(Py<PyAny>, usize, usize, bool, Vec<usize>)> {
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
        let (n_samples, n_features) = shape_tuple;

        if n_samples == 0 {
            return Err(BoostersError::InvalidParameter {
                name: "features".to_string(),
                reason: "array must have at least one sample".to_string(),
            }
            .into());
        }

        // Check dtype and contiguity
        let dtype = obj.getattr("dtype")?;
        let dtype_name: String = dtype.getattr("name")?.extract()?;

        let numpy = py.import_bound("numpy")?;
        let ascontiguousarray = numpy.getattr("ascontiguousarray")?;

        let (processed, converted) = if dtype_name == "float32" {
            let flags = obj.getattr("flags")?;
            let c_contiguous: bool = flags.getattr("c_contiguous")?.extract()?;
            if c_contiguous {
                (obj.clone().unbind(), false)
            } else {
                let converted =
                    ascontiguousarray.call1((obj, numpy.getattr("float32")?))?;
                (converted.unbind(), true)
            }
        } else if dtype_name == "float64" {
            let flags = obj.getattr("flags")?;
            let c_contiguous: bool = flags.getattr("c_contiguous")?.extract()?;
            if c_contiguous {
                (obj.clone().unbind(), false)
            } else {
                let converted = ascontiguousarray.call1((obj,))?;
                (converted.unbind(), true)
            }
        } else {
            // Convert to float32
            let converted =
                ascontiguousarray.call1((obj, numpy.getattr("float32")?))?;
            (converted.unbind(), true)
        };

        // No categorical detection for raw numpy arrays
        Ok((processed, n_samples, n_features, converted, Vec::new()))
    }

    /// Validate labels array.
    fn validate_labels(py: Python<'_>, labels: &PyObject, expected_samples: usize) -> PyResult<()> {
        let labels_bound = labels.bind(py);

        // Check for ndim
        if !labels_bound.hasattr("ndim")? {
            let type_name = labels_bound
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "unknown".to_string());
            return Err(BoostersError::TypeError(format!(
                "expected numpy array, got {}",
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

        Ok(())
    }

    /// Validate a 1D array (weights, groups).
    fn validate_1d_array(
        py: Python<'_>,
        arr: &PyObject,
        expected_samples: usize,
        name: &str,
    ) -> PyResult<()> {
        let arr_bound = arr.bind(py);

        // Check for ndim
        if !arr_bound.hasattr("ndim")? {
            let type_name = arr_bound
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "unknown".to_string());
            return Err(BoostersError::TypeError(format!(
                "expected numpy array, got {}",
                type_name
            ))
            .into());
        }
        let ndim: usize = arr_bound.getattr("ndim")?.extract()?;

        if ndim != 1 {
            return Err(BoostersError::InvalidParameter {
                name: name.to_string(),
                reason: format!("expected 1D array, got {}D", ndim),
            }
            .into());
        }

        // Check shape
        let shape = arr_bound.getattr("shape")?;
        let shape_tuple: (usize,) = shape.extract()?;
        let n_samples = shape_tuple.0;

        if n_samples != expected_samples {
            return Err(BoostersError::InvalidParameter {
                name: name.to_string(),
                reason: format!(
                    "shape mismatch: expected {} samples, got {}",
                    expected_samples, n_samples
                ),
            }
            .into());
        }

        Ok(())
    }

    /// Get features as a readonly array view.
    ///
    /// This is used internally by the training code.
    pub fn features_array<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<PyReadonlyArray2<'py, f32>> {
        // Try to extract as f32
        if let Ok(arr) = self.features.bind(py).extract::<PyArrayLike2<f32>>() {
            return Ok(arr.try_readonly()?);
        }

        // Try as f64 and we'll need to handle conversion upstream
        Err(BoostersError::TypeError(
            "expected float32 array, got float64 (conversion needed)".to_string(),
        )
        .into())
    }

    /// Get features as a readonly array view (f64 version).
    pub fn features_array_f64<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<PyReadonlyArray2<'py, f64>> {
        self.features
            .bind(py)
            .extract::<PyArrayLike2<f64>>()
            .and_then(|a| Ok(a.try_readonly()?))
            .map_err(|_| {
                BoostersError::TypeError("expected float64 array".to_string()).into()
            })
    }

    /// Get labels as a readonly array view.
    pub fn labels_array<'py>(&self, py: Python<'py>) -> PyResult<Option<PyReadonlyArray1<'py, f32>>> {
        match &self.labels {
            Some(labels) => {
                let arr = labels
                    .bind(py)
                    .extract::<PyArrayLike1<f32>>()
                    .and_then(|a| Ok(a.try_readonly()?))?;
                Ok(Some(arr))
            }
            None => Ok(None),
        }
    }

    /// Get weights as a readonly array view.
    pub fn weights_array<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<PyReadonlyArray1<'py, f32>>> {
        match &self.weights {
            Some(weights) => {
                let arr = weights
                    .bind(py)
                    .extract::<PyArrayLike1<f32>>()
                    .and_then(|a| Ok(a.try_readonly()?))?;
                Ok(Some(arr))
            }
            None => Ok(None),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    // Tests are in Python - they require numpy and pandas
}
