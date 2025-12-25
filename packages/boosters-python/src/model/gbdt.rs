//! GBDT Model Python bindings.

use ndarray::{Array1, Array2};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyclass;

use crate::config::PyGBDTConfig;
use crate::data::{PyDataset, PyEvalSet};
use crate::error::BoostersError;
use crate::threading::EvalLogger;

use boosters::data::Dataset as CoreDataset;
use boosters::training::EvalSet as CoreEvalSet;

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
                let boosters_mod = py.import_bound("boosters._boosters_rs")?;
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
                let arr = PyArray1::from_slice_bound(py, scores);
                Ok(arr.into_any().unbind())
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
    /// Args:
    ///     features: Feature array of shape `(n_samples, n_features)` or Dataset.
    ///     raw_score: If True, return raw margin scores without transformation.
    ///         For classification this means logits instead of probabilities.
    ///     n_iterations: Number of trees to use for prediction. If None, uses all trees.
    ///
    /// Returns:
    ///     NumPy array with predictions. Shape depends on the objective:
    ///     - Regression: `(n_samples,)`
    ///     - Binary classification: `(n_samples,)` (probability of positive class)
    ///     - Multiclass: `(n_samples, n_classes)`
    ///     - Multi-quantile: `(n_samples, n_quantiles)`
    ///
    /// Raises:
    ///     RuntimeError: If model has not been fitted.
    ///     ValueError: If features have wrong shape.
    ///
    /// Example:
    ///     ```python
    ///     predictions = model.predict(X_test)
    ///     raw_margins = model.predict(X_test, raw_score=True)
    ///     ```
    #[pyo3(signature = (features, raw_score=false, n_iterations=None))]
    pub fn predict(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        raw_score: bool,
        n_iterations: Option<usize>,
    ) -> PyResult<PyObject> {
        use numpy::{PyArray1, PyArray2};

        let model = self.inner.as_ref().ok_or_else(|| BoostersError::NotFitted {
            method: "predict".to_string(),
        })?;

        // Extract features array - either from Dataset or raw numpy array
        let features_array: Array2<f32> = if let Ok(dataset) = features.extract::<PyRef<'_, PyDataset>>() {
            // Features from Dataset
            let features_view = dataset.features_array(py)?;
            let features_np = features_view.as_array();
            features_np.t().as_standard_layout().into_owned()
        } else {
            // Assume numpy array [n_samples, n_features]
            let features_view: numpy::PyReadonlyArray2<'_, f32> = features.extract()?;
            let features_np = features_view.as_array();
            features_np.t().as_standard_layout().into_owned()
        };

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

        // TODO: Support n_iterations for partial prediction
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
        let output_t = output.t().as_standard_layout().into_owned();

        // Return appropriate shape based on n_groups
        let n_groups = output.nrows();
        if n_groups == 1 {
            // Single output: return 1D array (n_samples,)
            let flat: Vec<f32> = output_t.iter().copied().collect();
            let arr = PyArray1::from_vec_bound(py, flat);
            Ok(arr.into_any().unbind())
        } else {
            // Multi-output: return 2D array (n_samples, n_groups)
            let arr = PyArray2::from_owned_array_bound(py, output_t);
            Ok(arr.into_any().unbind())
        }
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
        train: &PyDataset,
        valid: Option<&Bound<'py, PyAny>>,
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

        // Extract features [n_samples, n_features] and transpose to [n_features, n_samples]
        // IMPORTANT: Must use as_standard_layout() to ensure C-order (row-contiguous) layout
        // Otherwise binning will fail because rows won't be contiguous
        let features_view = train.features_array(py)?;
        let features_np = features_view.as_array();
        let features_transposed: Array2<f32> = features_np.t().as_standard_layout().into_owned();

        // Extract labels [n_samples] and reshape to [1, n_samples]
        let labels_view = train.labels_array(py)?.expect("labels validated above");
        let labels_np = labels_view.as_array();
        let labels_1d: Array1<f32> = labels_np.to_owned();
        let labels_2d: Array2<f32> =
            labels_1d.into_shape((1, features_transposed.ncols())).map_err(|_| {
                BoostersError::ValidationError("Failed to reshape labels".to_string())
            })?;

        // Extract weights (optional)
        let weights_view = train.weights_array(py)?;
        let weights_opt: Option<ndarray::Array1<f32>> =
            weights_view.map(|w| w.as_array().to_owned());

        // Create core dataset
        let core_train = CoreDataset::new(
            features_transposed.view(),
            Some(labels_2d.view()),
            weights_opt.as_ref().map(|w| w.view()),
        );

        // Convert validation sets
        let eval_sets: Vec<(String, CoreDataset)> = if let Some(valid_obj) = valid {
            Self::extract_eval_sets(py, valid_obj)?
        } else {
            Vec::new()
        };

        // Create EvalSet references for training
        let eval_set_refs: Vec<CoreEvalSet<'_>> = eval_sets
            .iter()
            .map(|(name, ds)| CoreEvalSet::new(name.as_str(), ds))
            .collect();

        // Train with GIL released
        let n_threads = 0; // Auto-detect
        let trained_model = py.allow_threads(|| {
            boosters::GBDTModel::train(&core_train, &eval_set_refs, core_config, n_threads)
        });

        match trained_model {
            Some(model) => {
                // Store best iteration/score from model metadata
                slf.best_iteration = model.meta().best_iteration;
                // Note: best_score not stored in meta currently

                // Store the trained model
                slf.inner = Some(model);

                // TODO: Populate eval_results from training logs
                // For now, leave as None (will be implemented when we add
                // callback support to capture per-round metrics)

                Ok(slf)
            }
            None => Err(BoostersError::TrainingError(
                "Training failed to produce a model".to_string(),
            )
            .into()),
        }
    }
}

// Internal methods for fit/predict (to be implemented in Story 4.3/4.4)
impl PyGBDTModel {
    /// Set the inner model after training.
    pub fn set_inner(&mut self, model: boosters::GBDTModel) {
        self.inner = Some(model);
    }

    /// Get reference to inner model.
    pub fn get_inner(&self) -> Option<&boosters::GBDTModel> {
        self.inner.as_ref()
    }

    /// Extract evaluation sets from Python input.
    ///
    /// Accepts either a single EvalSet or a list of EvalSets.
    fn extract_eval_sets(
        py: Python<'_>,
        valid: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<(String, CoreDataset)>> {
        let mut result = Vec::new();

        // Check if it's a list
        if let Ok(list) = valid.downcast::<pyo3::types::PyList>() {
            for item in list.iter() {
                let eval_set: PyRef<'_, PyEvalSet> = item.extract()?;
                let ds = Self::convert_eval_set(py, &eval_set)?;
                result.push(ds);
            }
        } else {
            // Try as single EvalSet
            let eval_set: PyRef<'_, PyEvalSet> = valid.extract()?;
            let ds = Self::convert_eval_set(py, &eval_set)?;
            result.push(ds);
        }

        Ok(result)
    }

    /// Convert a single PyEvalSet to core types.
    fn convert_eval_set(
        py: Python<'_>,
        eval_set: &PyRef<'_, PyEvalSet>,
    ) -> PyResult<(String, CoreDataset)> {
        let name = eval_set.name().to_string();
        let dataset_ref = eval_set.get_dataset(py);
        let dataset = &*dataset_ref;

        // Extract features [n_samples, n_features] and transpose to [n_features, n_samples]
        // IMPORTANT: Must use as_standard_layout() to ensure C-order (row-contiguous) layout
        let features_view = dataset.features_array(py)?;
        let features_np = features_view.as_array();
        let features_transposed: Array2<f32> = features_np.t().as_standard_layout().into_owned();

        // Extract labels [n_samples] and reshape to [1, n_samples]
        let labels_view = dataset.labels_array(py)?;
        let labels_2d = if let Some(lv) = labels_view {
            let labels_np = lv.as_array();
            let labels_1d: Array1<f32> = labels_np.to_owned();
            Some(
                labels_1d
                    .into_shape((1, features_transposed.ncols()))
                    .map_err(|_| {
                        BoostersError::ValidationError("Failed to reshape labels".to_string())
                    })?,
            )
        } else {
            None
        };

        // Extract weights (optional)
        let weights_view = dataset.weights_array(py)?;
        let weights_opt: Option<ndarray::Array1<f32>> =
            weights_view.map(|w| w.as_array().to_owned());

        // Create core dataset
        let core_ds = CoreDataset::new(
            features_transposed.view(),
            labels_2d.as_ref().map(|l| l.view()),
            weights_opt.as_ref().map(|w| w.view()),
        );

        Ok((name, core_ds))
    }
}
