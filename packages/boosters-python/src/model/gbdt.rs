//! GBDT Model Python bindings.

use ndarray::Array2;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use boosters::data::{transpose_to_c_order, Dataset as CoreDataset};
use boosters::explainability::ShapValues;
use boosters::training::EvalSet as CoreEvalSet;

use crate::config::PyGBDTConfig;
use crate::data::{extract_dataset, PyDataset, PyEvalSet};
use crate::error::BoostersError;

/// Gradient Boosted Decision Tree model.
///
/// This is the main model class for training and prediction with gradient
/// boosted decision trees.
///
/// # Example
///
/// ```python
/// from boosters import GBDTModel, GBDTConfig, Dataset
/// import numpy as np
///
/// # Create training data
/// X = np.random.rand(1000, 10).astype(np.float32)
/// y = np.random.rand(1000).astype(np.float32)
/// train = Dataset(X, y)
///
/// # Train with default config
/// model = GBDTModel().fit(train)
///
/// # Or with custom config
/// config = GBDTConfig(n_estimators=50, learning_rate=0.1)
/// model = GBDTModel(config=config).fit(train)
///
/// # Predict
/// predictions = model.predict(X_test)
/// ```
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
    /// Example:
    ///     ```python
    ///     results = model.eval_results
    ///     # {"train": {"rmse": [0.5, 0.4, ...]}, "valid": {"rmse": [0.6, 0.5, ...]}}
    ///     ```
    #[getter]
    pub fn eval_results(&self, py: Python<'_>) -> Option<PyObject> {
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
    ///     importance_type: Type of importance to compute.
    ///         - "split" (default): Number of times a feature is used to split.
    ///         - "gain": Total gain achieved by splits on this feature.
    ///
    /// Returns:
    ///     NumPy array of importance scores, one per feature.
    ///
    /// Raises:
    ///     ValueError: If model has not been fitted.
    #[pyo3(signature = (importance_type="split"))]
    pub fn feature_importance(&self, py: Python<'_>, importance_type: &str) -> PyResult<PyObject> {
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
                    reason: format!(
                        "expected 'split' or 'gain', got '{}'",
                        other
                    ),
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
    /// contributes to individual predictions. The values sum to the difference
    /// between the model's prediction and the base value.
    ///
    /// Args:
    ///     features: Feature array of shape `(n_samples, n_features)` or Dataset.
    ///
    /// Returns:
    ///     NumPy array with shape `(n_samples, n_features + 1, n_outputs)`.
    ///     The last feature index contains the base value (expected value).
    ///     For single-output models, the last dimension is squeezed.
    ///
    /// Raises:
    ///     RuntimeError: If model has not been fitted.
    ///     ValueError: If features have wrong shape.
    ///
    /// Example:
    ///     ```python
    ///     # Get SHAP values for test data
    ///     shap_values = model.shap_values(X_test)
    ///
    ///     # For a single sample, contributions sum to prediction - base_value
    ///     sample_idx = 0
    ///     feature_contribs = shap_values[sample_idx, :-1, 0]  # All features
    ///     base_value = shap_values[sample_idx, -1, 0]  # Base value
    ///     prediction = model.predict(X_test[sample_idx:sample_idx+1])[0]
    ///     # assert np.isclose(base_value + feature_contribs.sum(), prediction)
    ///     ```
    #[pyo3(signature = (features))]
    pub fn shap_values(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        use numpy::{PyArray2, PyArray3};

        let model = self.inner.as_ref().ok_or_else(|| BoostersError::NotFitted {
            method: "shap_values".to_string(),
        })?;

        // Extract features array
        let features_array = Self::extract_features(py, features)?;

        // Validate feature count
        let expected_features = model.meta().n_features;
        let actual_features = features_array.nrows();
        if actual_features != expected_features {
            return Err(BoostersError::ValidationError(format!(
                "Expected {} features, got {}",
                expected_features, actual_features
            ))
            .into());
        }

        // Create temporary dataset for SHAP computation (no labels needed)
        let shap_dataset = CoreDataset::new(features_array.view(), None, None);

        // Compute SHAP values with GIL released
        let shap_result: Result<ShapValues, _> = py.allow_threads(|| {
            model.shap_values(&shap_dataset)
        });

        match shap_result {
            Ok(shap_values) => {
                // ShapValues is [n_samples, n_features + 1, n_outputs]
                let arr = shap_values.as_array();
                let n_outputs = shap_values.n_outputs();

                if n_outputs == 1 {
                    // Single output: squeeze to [n_samples, n_features + 1]
                    // Convert f64 to f32 for consistency with predict()
                    let squeezed = arr.slice(ndarray::s![.., .., 0]);
                    let squeezed_owned: Array2<f32> = squeezed.mapv(|v| v as f32);
                    let py_arr = PyArray2::from_owned_array(py, squeezed_owned);
                    Ok(py_arr.into_any().unbind())
                } else {
                    // Multi-output: keep 3D [n_samples, n_features + 1, n_outputs]
                    // Convert f64 to f32
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

    /// Make predictions on features.
    ///
    /// Returns transformed predictions (e.g., probabilities for classification).
    ///
    /// Args:
    ///     features: Feature array of shape `(n_samples, n_features)` or Dataset.
    ///     n_iterations: Number of trees to use for prediction. If None, uses all trees.
    ///
    /// Returns:
    ///     NumPy array with predictions of shape `(n_samples, n_outputs)`.
    ///     For single-output models, n_outputs is 1.
    ///
    /// Raises:
    ///     RuntimeError: If model has not been fitted.
    ///     ValueError: If features have wrong shape.
    ///
    /// Example:
    ///     ```python
    ///     predictions = model.predict(X_test)  # shape: (n_samples, n_outputs)
    ///     ```
    #[pyo3(signature = (features, n_iterations=None))]
    pub fn predict(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        n_iterations: Option<usize>,
    ) -> PyResult<PyObject> {
        self.predict_internal(py, features, false, n_iterations)
    }

    /// Make raw (untransformed) predictions on features.
    ///
    /// Returns raw margin scores without transformation.
    /// For classification this means logits instead of probabilities.
    ///
    /// Args:
    ///     features: Feature array of shape `(n_samples, n_features)` or Dataset.
    ///     n_iterations: Number of trees to use for prediction. If None, uses all trees.
    ///
    /// Returns:
    ///     NumPy array with raw scores of shape `(n_samples, n_outputs)`.
    ///     For single-output models, n_outputs is 1.
    ///
    /// Raises:
    ///     RuntimeError: If model has not been fitted.
    ///     ValueError: If features have wrong shape.
    ///
    /// Example:
    ///     ```python
    ///     raw_margins = model.predict_raw(X_test)  # shape: (n_samples, n_outputs)
    ///     ```
    #[pyo3(signature = (features, n_iterations=None))]
    pub fn predict_raw(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        n_iterations: Option<usize>,
    ) -> PyResult<PyObject> {
        self.predict_internal(py, features, true, n_iterations)
    }

    /// Train the model on a dataset.
    ///
    /// Args:
    ///     train: Training dataset containing features and labels.
    ///     valid: Optional validation set(s) for early stopping and evaluation.
    ///         Can be a single EvalSet or a list of EvalSets.
    ///
    /// Returns:
    ///     Self (for method chaining).
    ///
    /// Raises:
    ///     ValueError: If training data is invalid or labels are missing.
    ///
    /// Example:
    ///     ```python
    ///     model = GBDTModel().fit(train_dataset)
    ///     model = GBDTModel().fit(train, valid=[EvalSet("val", val_data)])
    ///     ```
    #[pyo3(signature = (train, valid=None))]
    pub fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        train: &Bound<'py, PyAny>,
        valid: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        // Extract PyDataset from train argument
        let train_dataset_py = extract_dataset(py, train)?;
        let train_dataset = train_dataset_py.bind(py).borrow();

        // Validate that training data has labels
        if !train_dataset.has_labels() {
            return Err(BoostersError::ValidationError(
                "Training dataset must have labels".to_string(),
            )
            .into());
        }

        // Convert Python config to core config
        let config = slf.config.bind(py).borrow();
        let core_config = config.to_core(py)?;
        drop(config);

        // Get reference to the core dataset directly - no conversion needed!
        let core_train = train_dataset.as_core();

        // Extract eval sets as Vec<(name, PyRef<PyDataset>)> to keep borrows alive
        let eval_set_data: Vec<(String, PyRef<'py, PyDataset>)> = if let Some(valid_obj) = valid {
            Self::extract_eval_set_refs(py, valid_obj)?
        } else {
            Vec::new()
        };

        // Create EvalSet references for training
        let eval_set_refs: Vec<CoreEvalSet<'_>> = eval_set_data
            .iter()
            .map(|(name, ds)| CoreEvalSet::new(name.as_str(), ds.as_core()))
            .collect();

        // Train with GIL released
        let n_threads = 0; // Auto-detect
        let trained_model = py.allow_threads(|| {
            boosters::GBDTModel::train(core_train, &eval_set_refs, core_config, n_threads)
        });

        match trained_model {
            Some(model) => {
                // Store best iteration/score from model metadata
                slf.best_iteration = model.meta().best_iteration;

                // Store the trained model
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
    ///
    /// Always returns 2D array of shape [n_samples, n_outputs] for consistency.
    fn predict_internal(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        raw_score: bool,
        n_iterations: Option<usize>,
    ) -> PyResult<PyObject> {
        use numpy::PyArray2;

        let model = self.inner.as_ref().ok_or_else(|| BoostersError::NotFitted {
            method: "predict".to_string(),
        })?;

        // Extract features array
        let features_array = Self::extract_features(py, features)?;

        // Validate feature count
        let expected_features = model.meta().n_features;
        let actual_features = features_array.nrows();
        if actual_features != expected_features {
            return Err(BoostersError::ValidationError(format!(
                "Expected {} features, got {}",
                expected_features, actual_features
            ))
            .into());
        }

        // Create temporary dataset for prediction (no labels needed)
        let pred_dataset = CoreDataset::new(features_array.view(), None, None);

        // Note: n_iterations is accepted for API compatibility but not used.
        // Users should train with the desired number of trees.
        let _ = n_iterations;

        // Predict with GIL released
        let n_threads = 0;
        let output = py.allow_threads(|| {
            if raw_score {
                model.predict_raw(&pred_dataset, n_threads)
            } else {
                model.predict(&pred_dataset, n_threads)
            }
        });

        // output shape is [n_groups, n_samples]
        // Transpose to [n_samples, n_groups] for Python convention
        // Always return 2D array for consistent shape
        let output_t = transpose_to_c_order(output.view());
        let arr = PyArray2::from_owned_array(py, output_t);
        Ok(arr.into_any().unbind())
    }

    /// Extract features array from various Python input types.
    ///
    /// Accepts:
    /// - Direct PyDataset (Rust type)
    /// - Python wrapper with _inner attribute
    /// - Raw numpy array [n_samples, n_features]
    ///
    /// Returns features transposed to [n_features, n_samples] for core API.
    fn extract_features(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Array2<f32>> {
        // Try as Dataset first (Rust or Python wrapper)
        if let Ok(dataset_py) = extract_dataset(py, obj) {
            let dataset = dataset_py.bind(py).borrow();
            // CoreDataset already stores features in [n_features, n_samples] format
            return Ok(dataset.as_core().features().view().to_owned());
        }

        // Fall back to numpy array - need to transpose from [n_samples, n_features]
        let features_view: numpy::PyReadonlyArray2<'_, f32> = obj.extract()?;
        Ok(transpose_to_c_order(features_view.as_array()))
    }

    /// Extract evaluation sets from Python input, keeping PyRef borrows alive.
    ///
    /// Returns Vec of (name, PyRef<PyDataset>) so the borrows remain valid.
    fn extract_eval_set_refs<'py>(
        py: Python<'py>,
        valid: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<(String, PyRef<'py, PyDataset>)>> {
        let mut result = Vec::new();

        // Check if it's a list
        if let Ok(list) = valid.downcast::<pyo3::types::PyList>() {
            for item in list.iter() {
                let eval_set: PyRef<'py, PyEvalSet> = item.extract()?;
                let name = eval_set.name().to_string();
                let dataset = eval_set.get_dataset(py);
                result.push((name, dataset));
            }
        } else {
            // Try as single EvalSet
            let eval_set: PyRef<'py, PyEvalSet> = valid.extract()?;
            let name = eval_set.name().to_string();
            let dataset = eval_set.get_dataset(py);
            result.push((name, dataset));
        }

        Ok(result)
    }
}
