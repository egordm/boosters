//! GBDT Model Python bindings.

use ndarray::Array2;
use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use boosters::data::transpose_to_c_order;
use boosters::explainability::ShapValues;
use boosters::training::EvalSet as CoreEvalSet;

use crate::config::PyGBDTConfig;
use crate::data::{PyDataset, PyEvalSet};
use crate::error::BoostersError;

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
                // Create default config using Python interface
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
    ///
    /// Raises:
    ///     ValueError: If model has not been fitted.
    #[getter]
    pub fn n_trees(&self) -> PyResult<usize> {
        match &self.inner {
            Some(model) => Ok(model.forest().n_trees()),
            None => Err(BoostersError::NotFitted {
                method: "n_trees".to_string(),
            }
            .into()),
        }
    }

    /// Number of features the model was trained on.
    ///
    /// Raises:
    ///     ValueError: If model has not been fitted.
    #[getter]
    pub fn n_features(&self) -> PyResult<usize> {
        match &self.inner {
            Some(model) => Ok(model.meta().n_features),
            None => Err(BoostersError::NotFitted {
                method: "n_features".to_string(),
            }
            .into()),
        }
    }

    /// Best iteration from early stopping.
    ///
    /// Returns None if early stopping was not used or not triggered.
    #[getter]
    pub fn best_iteration(&self) -> Option<usize> {
        self.best_iteration
            .or_else(|| self.inner.as_ref().and_then(|m| m.meta().best_iteration))
    }

    /// Best score from early stopping.
    ///
    /// Returns None if early stopping was not used or not triggered.
    #[getter]
    pub fn best_score(&self) -> Option<f64> {
        self.best_score
    }

    /// Evaluation results from training.
    ///
    /// Returns a dict mapping eval set names to dicts of metric names to lists
    /// of scores per iteration.
    ///
    /// Examples:
    ///     >>> results = model.eval_results
    ///     >>> # {"train": {"rmse": [0.5, 0.4, ...]}}
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
    ///     importance_type: Type of importance: "split" or "gain".
    ///
    /// Returns:
    ///     Array of importance scores, one per feature.
    ///
    /// Raises:
    ///     ValueError: If model has not been fitted.
    #[pyo3(signature = (importance_type="split"))]
    #[gen_stub(override_return_type(type_repr = "numpy.ndarray", imports = ("numpy",)))]
    pub fn feature_importance(
        &self,
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "typing.Literal['split', 'gain']", imports = ("typing",)))]
        importance_type: &str,
    ) -> PyResult<Py<PyAny>> {
        use boosters::explainability::ImportanceType;
        use numpy::PyArray1;

        let model = self.inner.as_ref().ok_or_else(|| BoostersError::NotFitted {
            method: "feature_importance".to_string(),
        })?;

        let importance = match importance_type {
            "split" => model.feature_importance(ImportanceType::Split),
            "gain" => model.feature_importance(ImportanceType::Gain),
            other => {
                return Err(BoostersError::InvalidParameter {
                    name: "importance_type".to_string(),
                    reason: format!("expected 'split' or 'gain', got '{}'", other),
                }
                .into())
            }
        };

        match importance {
            Ok(fi) => {
                let scores = fi.values();
                let arr = PyArray1::from_slice(py, scores);
                Ok(arr.into_any().unbind())
            }
            Err(e) => Err(BoostersError::ExplainError(e.to_string()).into()),
        }
    }

    /// Compute SHAP values for feature contribution analysis.
    ///
    /// SHAP (SHapley Additive exPlanations) values show how each feature
    /// contributes to individual predictions.
    ///
    /// Args:
    ///     data: Dataset containing features for SHAP computation.
    ///
    /// Returns:
    ///     Array with shape (n_samples, n_features + 1, n_outputs).
    ///     The last feature index contains the base value.
    ///
    /// Raises:
    ///     RuntimeError: If model has not been fitted.
    ///     ValueError: If features have wrong shape.
    ///
    /// Examples:
    ///     >>> shap_values = model.shap_values(test_data)
    #[pyo3(signature = (data))]
    #[gen_stub(override_return_type(type_repr = "numpy.ndarray", imports = ("numpy",)))]
    pub fn shap_values(&self, py: Python<'_>, data: PyRef<'_, PyDataset>) -> PyResult<Py<PyAny>> {
        use numpy::{PyArray2, PyArray3};

        let model = self.inner.as_ref().ok_or_else(|| BoostersError::NotFitted {
            method: "shap_values".to_string(),
        })?;

        let core_dataset = data.as_core();

        // Validate feature count
        let expected_features = model.meta().n_features;
        let actual_features = core_dataset.n_features();
        if actual_features != expected_features {
            return Err(BoostersError::ValidationError(format!(
                "Expected {} features, got {}",
                expected_features, actual_features
            ))
            .into());
        }

        // Compute SHAP values with GIL released
        let shap_result: Result<ShapValues, _> = py.detach(|| model.shap_values(core_dataset));

        match shap_result {
            Ok(shap_values) => {
                let arr = shap_values.as_array();
                let n_outputs = shap_values.n_outputs();

                if n_outputs == 1 {
                    // Single output: squeeze to [n_samples, n_features + 1]
                    let squeezed = arr.slice(ndarray::s![.., .., 0]);
                    let squeezed_owned: Array2<f32> = squeezed.mapv(|v| v as f32);
                    let py_arr = PyArray2::from_owned_array(py, squeezed_owned);
                    Ok(py_arr.into_any().unbind())
                } else {
                    // Multi-output: keep 3D
                    let arr_f32: ndarray::Array3<f32> = arr.mapv(|v| v as f32);
                    let py_arr = PyArray3::from_owned_array(py, arr_f32);
                    Ok(py_arr.into_any().unbind())
                }
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
    ///
    /// Args:
    ///     data: Dataset containing features for prediction.
    ///     n_iterations: Number of trees to use. Defaults to all trees.
    ///
    /// Returns:
    ///     Predictions of shape (n_samples, n_outputs).
    ///
    /// Raises:
    ///     RuntimeError: If model has not been fitted.
    ///     ValueError: If features have wrong shape.
    ///
    /// Examples:
    ///     >>> predictions = model.predict(test_data)
    #[pyo3(signature = (data, n_iterations=None))]
    #[gen_stub(override_return_type(type_repr = "numpy.ndarray", imports = ("numpy",)))]
    pub fn predict(
        &self,
        py: Python<'_>,
        data: PyRef<'_, PyDataset>,
        n_iterations: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        self.predict_internal(py, &data, false, n_iterations)
    }

    /// Make raw (untransformed) predictions on data.
    ///
    /// Returns raw margin scores without transformation.
    /// For classification this means logits instead of probabilities.
    ///
    /// Args:
    ///     data: Dataset containing features for prediction.
    ///     n_iterations: Number of trees to use. Defaults to all trees.
    ///
    /// Returns:
    ///     Raw scores of shape (n_samples, n_outputs).
    ///
    /// Raises:
    ///     RuntimeError: If model has not been fitted.
    ///     ValueError: If features have wrong shape.
    ///
    /// Examples:
    ///     >>> raw_margins = model.predict_raw(test_data)
    #[pyo3(signature = (data, n_iterations=None))]
    #[gen_stub(override_return_type(type_repr = "numpy.ndarray", imports = ("numpy",)))]
    pub fn predict_raw(
        &self,
        py: Python<'_>,
        data: PyRef<'_, PyDataset>,
        n_iterations: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        self.predict_internal(py, &data, true, n_iterations)
    }

    /// Train the model on a dataset.
    ///
    /// Args:
    ///     train: Training dataset containing features and labels.
    ///     valid: Validation set(s) for early stopping and evaluation.
    ///
    /// Returns:
    ///     Self (for method chaining).
    ///
    /// Raises:
    ///     ValueError: If training data is invalid or labels are missing.
    ///
    /// Examples:
    ///     >>> model = GBDTModel().fit(train_dataset)
    #[pyo3(signature = (train, valid=None))]
    pub fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        train: PyRef<'py, PyDataset>,
        valid: Option<Vec<PyRef<'py, PyEvalSet>>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        // Validate that training data has labels
        if !train.has_labels() {
            return Err(BoostersError::ValidationError(
                "Training dataset must have labels".to_string(),
            )
            .into());
        }

        // Convert Python config to core config
        let config = slf.config.bind(py).borrow();
        let core_config = config.to_core(py)?;
        drop(config);

        // Get reference to the core dataset directly
        let core_train = train.as_core();

        // Extract eval sets
        let eval_set_data: Vec<(String, PyRef<'py, PyDataset>)> = valid
            .unwrap_or_default()
            .into_iter()
            .map(|es| {
                let name = es.name().to_string();
                let dataset = es.get_dataset(py);
                (name, dataset)
            })
            .collect();

        // Create EvalSet references for training
        let eval_set_refs: Vec<CoreEvalSet<'_>> = eval_set_data
            .iter()
            .map(|(name, ds)| CoreEvalSet::new(name.as_str(), ds.as_core()))
            .collect();

        // Train with GIL released
        let n_threads = 0; // Auto-detect
        let trained_model = py.detach(|| {
            boosters::GBDTModel::train(core_train, &eval_set_refs, core_config, n_threads)
        });

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

// Internal helper methods
impl PyGBDTModel {
    /// Set the inner model after training.
    pub fn set_inner(&mut self, model: boosters::GBDTModel) {
        self.inner = Some(model);
    }

    /// Get reference to inner model.
    pub fn get_inner(&self) -> Option<&boosters::GBDTModel> {
        self.inner.as_ref()
    }

    /// Internal prediction method shared by predict and predict_raw.
    fn predict_internal(
        &self,
        py: Python<'_>,
        data: &PyRef<'_, PyDataset>,
        raw_score: bool,
        n_iterations: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let model = self.inner.as_ref().ok_or_else(|| BoostersError::NotFitted {
            method: "predict".to_string(),
        })?;

        let core_dataset = data.as_core();

        // Validate feature count
        let expected_features = model.meta().n_features;
        let actual_features = core_dataset.n_features();
        if actual_features != expected_features {
            return Err(BoostersError::ValidationError(format!(
                "Expected {} features, got {}",
                expected_features, actual_features
            ))
            .into());
        }

        // n_iterations accepted for API compatibility but not used
        let _ = n_iterations;

        // Predict with GIL released
        let n_threads = 0;
        let output = py.detach(|| {
            if raw_score {
                model.predict_raw(core_dataset, n_threads)
            } else {
                model.predict(core_dataset, n_threads)
            }
        });

        // Transpose to [n_samples, n_groups] for Python convention
        let output_t = transpose_to_c_order(output.view());
        let arr = PyArray2::from_owned_array(py, output_t);
        Ok(arr.into_any().unbind())
    }
}
