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
use crate::validation::validate_feature_count;

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
    model: boosters::GBDTModel,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyGBDTModel {
    /// Train a new GBDT model.
    ///
    /// This matches the Rust API style: training is a class-level constructor.
    ///
    /// Args:
    ///     train: Training dataset containing features and labels.
    ///     config: Optional GBDTConfig. If not provided, uses default config.
    ///     val_set: Optional validation dataset for early stopping and evaluation.
    ///     n_threads: Number of threads for parallel training (0 = auto).
    ///
    /// Returns:
    ///     Trained GBDTModel.
    #[staticmethod]
    #[pyo3(signature = (train, config=None, val_set=None, n_threads=0))]
    pub fn train(
        py: Python<'_>,
        train: PyRef<'_, PyDataset>,
        config: Option<Py<PyGBDTConfig>>,
        val_set: Option<PyRef<'_, PyDataset>>,
        n_threads: usize,
    ) -> PyResult<Self> {
        if !train.has_labels() {
            return Err(BoostersError::ValidationError(
                "Training dataset must have labels".to_string(),
            )
            .into());
        }

        let core_config: boosters::GBDTConfig = match config {
            Some(c) => {
                let borrowed = c.bind(py).borrow();
                (&*borrowed).into()
            }
            None => (&PyGBDTConfig::default()).into(),
        };

        let core_train = train.inner();
        let core_val_set = val_set.as_ref().map(|ds| ds.inner());

        let trained_model = py.detach(|| {
            boosters::GBDTModel::train(core_train, core_val_set, core_config, n_threads)
        });

        match trained_model {
            Some(model) => Ok(Self { model }),
            None => Err(BoostersError::TrainingError(
                "Training failed to produce a model".to_string(),
            )
            .into()),
        }
    }

    /// Number of trees in the fitted model.
    #[getter]
    pub fn n_trees(&self) -> PyResult<usize> {
        Ok(self.model.forest().n_trees())
    }

    /// Number of features the model was trained on.
    #[getter]
    pub fn n_features(&self) -> PyResult<usize> {
        Ok(self.model.meta().n_features)
    }

    /// Feature names from training dataset (if provided).
    #[getter]
    pub fn feature_names(&self) -> Option<Vec<String>> {
        self.model.meta().feature_names.clone()
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
        let importance = self
            .model
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
        let dataset = data.inner();

        validate_feature_count(self.model.meta().n_features, dataset.n_features())?;

        // Compute SHAP values with GIL released
        let shap_result: Result<ShapValues, _> = py.detach(|| self.model.shap_values(dataset));

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
        format!(
            "GBDTModel(n_trees={}, n_features={})",
            self.model.forest().n_trees(),
            self.model.meta().n_features
        )
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
        let dataset = data.inner();

        validate_feature_count(self.model.meta().n_features, dataset.n_features())?;

        // Predict with GIL released
        let output = py.detach(|| self.model.predict(dataset, n_threads));

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
        let dataset = data.inner();

        validate_feature_count(self.model.meta().n_features, dataset.n_features())?;

        // Predict with GIL released
        let output = py.detach(|| self.model.predict_raw(dataset, n_threads));

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

        let _ = py;
        Ok(Self { model })
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

        let _ = py;
        Ok(Self { model })
    }
}

impl PyGBDTModel {
    /// Create from a core model (used by polymorphic loading).
    pub(crate) fn from_model(model: boosters::GBDTModel) -> Self {
        Self { model }
    }
}
