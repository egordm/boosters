//! GBLinear Model Python bindings.

use std::io::Cursor;

use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use boosters::persist::{
    BinaryReadOptions, BinaryWriteOptions, JsonWriteOptions, SerializableModel,
};

use crate::config::PyGBLinearConfig;
use crate::data::PyDataset;
use crate::error::BoostersError;
use crate::validation::validate_feature_count;

/// Gradient Boosted Linear model.
///
/// GBLinear uses gradient boosting to train a linear model via coordinate
/// descent. Simpler than GBDT but can be effective for linear relationships.
///
/// Attributes:
///     coef_: Model coefficients after fitting.
///     intercept_: Model intercept after fitting.
///     is_fitted: Whether the model has been trained.
///     n_features_in_: Number of features seen during fit.
///
/// Examples:
///     >>> config = GBLinearConfig(n_estimators=50, learning_rate=0.3)
///     >>> model = GBLinearModel(config=config).fit(train)
///     >>> predictions = model.predict(X_test)
#[gen_stub_pyclass]
#[pyclass(name = "GBLinearModel", module = "boosters._boosters_rs")]
pub struct PyGBLinearModel {
    model: boosters::GBLinearModel,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyGBLinearModel {
    /// Train a new GBLinear model.
    ///
    /// This matches the Rust API style: training is a class-level constructor.
    ///
    /// Args:
    ///     train: Training dataset containing features and labels.
    ///     config: Optional GBLinearConfig. If not provided, uses default config.
    ///     val_set: Optional validation dataset for early stopping and evaluation.
    ///     n_threads: Number of threads for parallel training (0 = auto).
    ///
    /// Returns:
    ///     Trained GBLinearModel.
    #[staticmethod]
    #[pyo3(signature = (train, config=None, val_set=None, n_threads=0))]
    pub fn train(
        py: Python<'_>,
        train: PyRef<'_, PyDataset>,
        config: Option<Py<PyGBLinearConfig>>,
        val_set: Option<PyRef<'_, PyDataset>>,
        n_threads: usize,
    ) -> PyResult<Self> {
        if !train.has_labels() {
            return Err(BoostersError::ValidationError(
                "Training dataset must have labels".to_string(),
            )
            .into());
        }

        let core_config: boosters::GBLinearConfig = match config {
            Some(c) => {
                let borrowed = c.bind(py).borrow();
                (&*borrowed).into()
            }
            None => (&PyGBLinearConfig::default()).into(),
        };

        let core_train = train.inner();
        let core_val_set = val_set.as_ref().map(|ds| ds.inner());

        let trained_model = py.detach(|| {
            boosters::GBLinearModel::train(core_train, core_val_set, core_config, n_threads)
        });

        match trained_model {
            Ok(model) => Ok(Self { model }),
            Err(err) => Err(BoostersError::ValidationError(err.to_string()).into()),
        }
    }

    /// Number of features the model was trained on.
    #[getter]
    pub fn n_features_in_(&self) -> PyResult<usize> {
        Ok(self.model.meta().n_features)
    }

    /// Feature names from training dataset (if provided).
    #[getter]
    pub fn feature_names(&self) -> Option<Vec<String>> {
        self.model.meta().feature_names.clone()
    }

    /// Model coefficients (weights).
    ///
    /// Returns:
    ///     Array with shape (n_features,) for single-output models
    ///     or (n_features, n_outputs) for multi-output models.
    #[getter]
    pub fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let linear = self.model.linear();
        let n_features = linear.n_features();
        let n_groups = linear.n_groups();

        if n_groups == 1 {
            // Single output: return 1D array (n_features,)
            let weights: Vec<f32> = (0..n_features).map(|f| linear.weight(f, 0)).collect();
            let arr = PyArray1::from_vec(py, weights);
            Ok(arr.into_any().unbind())
        } else {
            // Multi-output: return 2D array (n_features, n_groups)
            let coefs = linear.weight_view().to_owned();
            let arr = PyArray2::from_owned_array(py, coefs);
            Ok(arr.into_any().unbind())
        }
    }

    /// Model intercept (bias).
    ///
    /// Returns:
    ///     Array of shape (n_outputs,).
    #[getter]
    pub fn intercept_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let linear = self.model.linear();
        let biases = linear.biases();

        Ok(PyArray1::from_iter(py, biases.iter().copied()))
    }

    /// String representation.
    fn __repr__(&self) -> String {
        format!("GBLinearModel(n_features={})", self.model.meta().n_features)
    }

    /// Pickle support: return (callable, args) for reconstruction.
    ///
    /// Returns:
    ///     Tuple of (from_bytes function, (serialized_bytes,)).
    pub fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Py<PyAny>, (Bound<'py, PyBytes>,))> {
        let bytes = self.to_bytes(py)?;
        let cls = py.get_type::<Self>();
        let from_bytes = cls.getattr("from_bytes")?;
        Ok((from_bytes.into(), (bytes,)))
    }

    /// Make predictions on features.
    ///
    /// Returns transformed predictions (e.g., probabilities for classification).
    /// Output shape is (n_samples, n_outputs) - sklearn convention.
    ///
    /// Args:
    ///     data: Dataset containing features.
    ///     n_threads: Number of threads (unused, for API consistency).
    ///
    /// Returns:
    ///     Predictions array with shape (n_samples, n_outputs).
    #[pyo3(signature = (data, n_threads=0))]
    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        data: PyRef<'_, PyDataset>,
        #[allow(unused_variables)] n_threads: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let dataset = data.inner();

        validate_feature_count(self.model.meta().n_features, dataset.n_features())?;

        // GBLinearModel::predict takes &Dataset
        let output = py.detach(|| self.model.predict(dataset));

        // Transpose from (n_outputs, n_samples) to (n_samples, n_outputs) for sklearn
        Ok(PyArray2::from_owned_array(py, output.t().to_owned()))
    }

    /// Make raw (untransformed) predictions on features.
    ///
    /// Returns raw margin scores without transformation.
    /// Output shape is (n_samples, n_outputs) - sklearn convention.
    ///
    /// Args:
    ///     data: Dataset containing features.
    ///     n_threads: Number of threads (unused, for API consistency).
    ///
    /// Returns:
    ///     Raw scores array with shape (n_samples, n_outputs).
    #[pyo3(signature = (data, n_threads=0))]
    pub fn predict_raw<'py>(
        &self,
        py: Python<'py>,
        data: PyRef<'_, PyDataset>,
        #[allow(unused_variables)] n_threads: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let dataset = data.inner();

        validate_feature_count(self.model.meta().n_features, dataset.n_features())?;

        // GBLinearModel::predict_raw takes &Dataset
        let output = py.detach(|| self.model.predict_raw(dataset));

        // Transpose from (n_outputs, n_samples) to (n_samples, n_outputs) for sklearn
        Ok(PyArray2::from_owned_array(py, output.t().to_owned()))
    }

    // =========================================================================
    // Serialization methods
    // =========================================================================

    /// Serialize model to binary bytes.
    ///
    /// Returns:
    ///     Binary representation of the model (.bstr format).
    ///
    /// Raises:
    ///     RuntimeError: If model is not fitted or serialization fails.
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let mut buf = Vec::new();
        self.model
            .write_into(&mut buf, &BinaryWriteOptions::default())
            .map_err(|e| BoostersError::WriteError(e.to_string()))?;
        Ok(PyBytes::new(py, &buf))
    }

    /// Serialize model to JSON bytes.
    ///
    /// Returns:
    ///     UTF-8 JSON representation of the model (.bstr.json format).
    ///
    /// Raises:
    ///     RuntimeError: If model is not fitted or serialization fails.
    pub fn to_json_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let mut buf = Vec::new();
        self.model
            .write_json_into(&mut buf, &JsonWriteOptions::compact())
            .map_err(|e| BoostersError::WriteError(e.to_string()))?;
        Ok(PyBytes::new(py, &buf))
    }

    /// Load model from binary bytes.
    ///
    /// Args:
    ///     data: Binary bytes in .bstr format.
    ///
    /// Returns:
    ///     Loaded GBLinearModel instance.
    ///
    /// Raises:
    ///     ValueError: If bytes are invalid or corrupted.
    #[staticmethod]
    #[pyo3(signature = (data))]
    pub fn from_bytes(
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "bytes"))] data: &[u8],
    ) -> PyResult<Self> {
        let model =
            boosters::GBLinearModel::read_from(Cursor::new(data), &BinaryReadOptions::default())
                .map_err(|e| BoostersError::ModelReadError(e.to_string()))?;

        let _ = py;
        Ok(Self { model })
    }

    /// Load model from JSON bytes.
    ///
    /// Args:
    ///     data: UTF-8 JSON bytes in .bstr.json format.
    ///
    /// Returns:
    ///     Loaded GBLinearModel instance.
    ///
    /// Raises:
    ///     ValueError: If JSON is invalid.
    #[staticmethod]
    #[pyo3(signature = (data))]
    pub fn from_json_bytes(
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "bytes"))] data: &[u8],
    ) -> PyResult<Self> {
        let model = boosters::GBLinearModel::read_json_from(Cursor::new(data))
            .map_err(|e| BoostersError::ModelReadError(e.to_string()))?;

        let _ = py;
        Ok(Self { model })
    }
}

impl PyGBLinearModel {
    /// Create from a core model (used by polymorphic loading).
    pub(crate) fn from_model(model: boosters::GBLinearModel) -> Self {
        Self { model }
    }
}
