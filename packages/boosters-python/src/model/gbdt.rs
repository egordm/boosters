//! GBDT Model Python bindings.

use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use boosters::explainability::ShapValues;
use boosters::training::EvalSet as CoreEvalSet;

use crate::config::PyGBDTConfig;
use crate::data::{PyDataset, PyEvalSet};
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
///     best_iteration: Best iteration from early stopping.
///     best_score: Best score from early stopping.
///     eval_results: Evaluation results from training.
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
    /// Configuration for the model.
    config: Py<PyGBDTConfig>,

    /// Inner trained model (None until fit() is called).
    inner: Option<boosters::GBDTModel>,

    /// Evaluation results from training (populated after fit).
    eval_results: Option<Py<PyAny>>,

    /// Best iteration from early stopping (if applicable).
    best_iteration: Option<usize>,

    /// Best score from early stopping (if applicable).
    best_score: Option<f64>,
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
        let config = match config {
            Some(c) => c,
            None => {
                let boosters_mod = py.import("boosters._boosters_rs")?;
                let config_class = boosters_mod.getattr("GBDTConfig")?;
                let config_obj = config_class.call0()?;
                config_obj.extract::<Py<PyGBDTConfig>>()?
            }
        };

        Ok(Self {
            config,
            inner: None,
            eval_results: None,
            best_iteration: None,
            best_score: None,
        })
    }

    /// Whether the model has been fitted.
    #[getter]
    pub fn is_fitted(&self) -> bool {
        self.inner.is_some()
    }

    /// Number of trees in the fitted model.
    #[getter]
    pub fn n_trees(&self) -> PyResult<usize> {
        let model = require_fitted(self.inner.as_ref(), "n_trees")?;
        Ok(model.forest().n_trees())
    }

    /// Number of features the model was trained on.
    #[getter]
    pub fn n_features(&self) -> PyResult<usize> {
        let model = require_fitted(self.inner.as_ref(), "n_features")?;
        Ok(model.meta().n_features)
    }

    /// Best iteration from early stopping.
    #[getter]
    pub fn best_iteration(&self) -> Option<usize> {
        self.best_iteration
            .or_else(|| self.inner.as_ref().and_then(|m| m.meta().best_iteration))
    }

    /// Best score from early stopping.
    #[getter]
    pub fn best_score(&self) -> Option<f64> {
        self.best_score
    }

    /// Evaluation results from training.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, dict[str, list[float]]] | None", imports = ()))]
    pub fn eval_results(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.eval_results.as_ref().map(|r| r.clone_ref(py))
    }

    /// Get the model configuration.
    #[getter]
    pub fn config(&self, py: Python<'_>) -> Py<PyGBDTConfig> {
        self.config.clone_ref(py)
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
        let model = require_fitted(self.inner.as_ref(), "feature_importance")?;

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
        let model = require_fitted(self.inner.as_ref(), "shap_values")?;
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
        if let Some(model) = &self.inner {
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
        let model = require_fitted(self.inner.as_ref(), "predict")?;
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
        let model = require_fitted(self.inner.as_ref(), "predict_raw")?;
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
    ///     valid: Validation set(s) for early stopping and evaluation.
    ///     n_threads: Number of threads for parallel training (0 = auto).
    ///
    /// Returns:
    ///     Self (for method chaining).
    #[pyo3(signature = (train, valid=None, n_threads=0))]
    pub fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        train: PyRef<'py, PyDataset>,
        valid: Option<Vec<PyRef<'py, PyEvalSet>>>,
        n_threads: usize,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if !train.has_labels() {
            return Err(BoostersError::ValidationError(
                "Training dataset must have labels".to_string(),
            )
            .into());
        }

        // Convert Python config to core config
        let config = slf.config.bind(py).borrow();
        let core_config: boosters::GBDTConfig = (&*config).into();
        drop(config);

        let core_train = train.inner();

        // Build eval sets from PyEvalSet references
        let py_eval_sets: Vec<_> = valid.unwrap_or_default();
        let borrowed: Vec<_> = py_eval_sets
            .iter()
            .map(|es| (es.name_str(), es.dataset_ref().bind(py).borrow()))
            .collect();
        let eval_sets: Vec<CoreEvalSet<'_>> = borrowed
            .iter()
            .map(|(name, ds)| CoreEvalSet::new(name, ds.inner()))
            .collect();

        // Train with GIL released
        let trained_model = py
            .detach(|| boosters::GBDTModel::train(core_train, &eval_sets, core_config, n_threads));

        match trained_model {
            Some(model) => {
                slf.best_iteration = model.meta().best_iteration;
                slf.inner = Some(model);
                Ok(slf)
            }
            None => Err(BoostersError::TrainingError(
                "Training failed to produce a model".to_string(),
            )
            .into()),
        }
    }
}
