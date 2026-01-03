//! GBDT Model Python bindings.

use std::io::Cursor;

use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use boosters::explainability::ShapValues;
use boosters::persist::{
    BinaryReadOptions, BinaryWriteOptions, JsonWriteOptions, SerializableModel,
};

use crate::config::PyGBDTConfig;
use crate::data::PyDataset;
use crate::error::BoostersError;
use crate::types::PyImportanceType;
use crate::validation::{require_fitted, validate_feature_count};

/// Gradient Boosted Decision Tree model.
///
/// This is the main model class for training and prediction with gradient
/// boosted decision trees.
///
/// Attributes:
///     is_fitted: Whether the model has been fitted.
///     n_trees: Number of trees in the fitted model.
///     n_features: Number of features the model was trained on.
///     config: Model configuration.
///
/// Examples:
///     >>> from boosters import GBDTModel, Dataset
///     >>> train = Dataset(X, y)
///     >>> model = GBDTModel().fit(train)
///     >>> predictions = model.predict(train)
#[gen_stub_pyclass]
#[pyclass(name = "GBDTModel", module = "boosters._boosters_rs")]
pub struct PyGBDTModel {
    config: boosters::GBDTConfig,
    model: Option<boosters::GBDTModel>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyGBDTModel {
    /// Create a new GBDT model.
    ///
    /// Args:
    ///     config: Optional GBDTConfig. If not provided, uses default config.
    ///
    /// Returns:
    ///     New GBDTModel instance (not yet fitted).
    #[new]
    #[pyo3(signature = (config=None))]
    pub fn new(py: Python<'_>, config: Option<Py<PyGBDTConfig>>) -> PyResult<Self> {
        let core_config = match config {
            Some(c) => {
                let borrowed = c.bind(py).borrow();
                (&*borrowed).into()
            }
            None => (&PyGBDTConfig::default()).into(),
        };

        Ok(Self {
            config: core_config,
            model: None,
        })
    }

    /// Whether the model has been fitted.
    #[getter]
    pub fn is_fitted(&self) -> bool {
        self.model.is_some()
    }

    /// Number of trees in the fitted model.
    #[getter]
    pub fn n_trees(&self) -> PyResult<usize> {
        let model = self.fitted_model("n_trees")?;
        Ok(model.forest().n_trees())
    }

    /// Number of features the model was trained on.
    #[getter]
    pub fn n_features(&self) -> PyResult<usize> {
        let model = self.fitted_model("n_features")?;
        Ok(model.meta().n_features)
    }

    /// Get the model configuration.
    #[getter]
    pub fn config(&self, py: Python<'_>) -> PyResult<Py<PyGBDTConfig>> {
        let py_config: PyGBDTConfig = (&self.config).into();
        Py::new(py, py_config)
    }

    /// Get feature importance scores.
    ///
    /// Args:
    ///     importance_type: Type of importance (ImportanceType.Split or ImportanceType.Gain).
    ///
    /// Returns:
    ///     Array of importance scores, one per feature.
    #[pyo3(signature = (importance_type=PyImportanceType::Split))]
    pub fn feature_importance<'py>(
        &self,
        py: Python<'py>,
        importance_type: PyImportanceType,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let model = self.fitted_model("feature_importance")?;

        let importance = model
            .feature_importance(importance_type.into())
            .map_err(|e| BoostersError::ExplainError(e.to_string()))?;

        // Values are already f32
        Ok(PyArray1::from_vec(py, importance.values().to_vec()))
    }

    /// Compute SHAP values for feature contribution analysis.
    ///
    /// Args:
    ///     data: Dataset containing features for SHAP computation.
    ///
    /// Returns:
    ///     Array with shape (n_samples, n_features + 1, n_outputs).
    #[pyo3(signature = (data))]
    pub fn shap_values<'py>(
        &self,
        py: Python<'py>,
        data: PyRef<'_, PyDataset>,
    ) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let model = self.fitted_model("shap_values")?;
        let dataset = data.inner();

        validate_feature_count(model.meta().n_features, dataset.n_features())?;

        // Compute SHAP values with GIL released
        let shap_result: Result<ShapValues, _> = py.detach(|| model.shap_values(dataset));

        match shap_result {
            Ok(shap_values) => {
                let arr = shap_values.as_array().to_owned();
                Ok(PyArray3::from_owned_array(py, arr))
            }
            Err(e) => Err(BoostersError::ExplainError(e.to_string()).into()),
        }
    }

    /// String representation.
    fn __repr__(&self) -> String {
        if let Some(model) = &self.model {
            format!(
                "GBDTModel(n_trees={}, n_features={}, fitted=True)",
                model.forest().n_trees(),
                model.meta().n_features
            )
        } else {
            "GBDTModel(fitted=False)".to_string()
        }
    }

    /// Make predictions on data.
    ///
    /// Returns transformed predictions (e.g., probabilities for classification).
    /// Output shape is (n_samples, n_outputs) - sklearn convention.
    ///
    /// Args:
    ///     data: Dataset containing features for prediction.
    ///     n_threads: Number of threads for parallel prediction (0 = auto).
    ///
    /// Returns:
    ///     Predictions array with shape (n_samples, n_outputs).
    #[pyo3(signature = (data, n_threads=0))]
    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        data: PyRef<'_, PyDataset>,
        n_threads: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let model = self.fitted_model("predict")?;
        let dataset = data.inner();

        validate_feature_count(model.meta().n_features, dataset.n_features())?;

        // Predict with GIL released
        let output = py.detach(|| model.predict(dataset, n_threads));

        // Transpose from (n_outputs, n_samples) to (n_samples, n_outputs) for sklearn
        Ok(PyArray2::from_owned_array(py, output.t().to_owned()))
    }

    /// Make raw (untransformed) predictions on data.
    ///
    /// Returns raw margin scores without transformation.
    /// Output shape is (n_samples, n_outputs) - sklearn convention.
    ///
    /// Args:
    ///     data: Dataset containing features for prediction.
    ///     n_threads: Number of threads for parallel prediction (0 = auto).
    ///
    /// Returns:
    ///     Raw scores array with shape (n_samples, n_outputs).
    #[pyo3(signature = (data, n_threads=0))]
    pub fn predict_raw<'py>(
        &self,
        py: Python<'py>,
        data: PyRef<'_, PyDataset>,
        n_threads: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let model = self.fitted_model("predict_raw")?;
        let dataset = data.inner();

        validate_feature_count(model.meta().n_features, dataset.n_features())?;

        // Predict with GIL released
        let output = py.detach(|| model.predict_raw(dataset, n_threads));

        // Transpose from (n_outputs, n_samples) to (n_samples, n_outputs) for sklearn
        Ok(PyArray2::from_owned_array(py, output.t().to_owned()))
    }

    /// Train the model on a dataset.
    ///
    /// Args:
    ///     train: Training dataset containing features and labels.
    ///     val_set: Optional validation dataset for early stopping and evaluation.
    ///     n_threads: Number of threads for parallel training (0 = auto).
    ///
    /// Returns:
    ///     Self (for method chaining).
    #[pyo3(signature = (train, val_set=None, n_threads=0))]
    pub fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        train: PyRef<'py, PyDataset>,
        val_set: Option<PyRef<'py, PyDataset>>,
        n_threads: usize,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if !train.has_labels() {
            return Err(BoostersError::ValidationError(
                "Training dataset must have labels".to_string(),
            )
            .into());
        }

        let core_config = slf.config.clone();

        let core_train = train.inner();

        // Get validation dataset if provided
        let core_val_set = val_set.as_ref().map(|ds| ds.inner());

        // Train with GIL released
        let trained_model = py.detach(|| {
            boosters::GBDTModel::train(core_train, core_val_set, core_config, n_threads)
        });

        match trained_model {
            Some(model) => {
                slf.config = model.config().clone();
                slf.model = Some(model);
                Ok(slf)
            }
            None => Err(BoostersError::TrainingError(
                "Training failed to produce a model".to_string(),
            )
            .into()),
        }
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
        let model = self.fitted_model("to_bytes")?;
        let mut buf = Vec::new();
        model
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
        let model = self.fitted_model("to_json_bytes")?;
        let mut buf = Vec::new();
        model
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
    ///     Loaded GBDTModel instance.
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
            boosters::GBDTModel::read_from(Cursor::new(data), &BinaryReadOptions::default())
                .map_err(|e| BoostersError::ModelReadError(e.to_string()))?;

        let config = model.config().clone();

        let _ = py;
        Ok(Self {
            config,
            model: Some(model),
        })
    }

    /// Load model from JSON bytes.
    ///
    /// Args:
    ///     data: UTF-8 JSON bytes in .bstr.json format.
    ///
    /// Returns:
    ///     Loaded GBDTModel instance.
    ///
    /// Raises:
    ///     ValueError: If JSON is invalid.
    #[staticmethod]
    #[pyo3(signature = (data))]
    pub fn from_json_bytes(
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "bytes"))] data: &[u8],
    ) -> PyResult<Self> {
        let model = boosters::GBDTModel::read_json_from(Cursor::new(data))
            .map_err(|e| BoostersError::ModelReadError(e.to_string()))?;

        let config = model.config().clone();

        let _ = py;
        Ok(Self {
            config,
            model: Some(model),
        })
    }
}

impl PyGBDTModel {
    fn fitted_model(&self, method: &str) -> PyResult<&boosters::GBDTModel> {
        require_fitted(self.model.as_ref(), method)
    }

    /// Create from a core model (used by polymorphic loading).
    pub(crate) fn from_model(model: boosters::GBDTModel) -> Self {
        Self {
            config: model.config().clone(),
            model: Some(model),
        }
    }
}
