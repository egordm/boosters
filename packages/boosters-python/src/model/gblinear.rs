//! GBLinear Model Python bindings.

use ndarray::{Array1, Array2};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyclass;

use crate::config::PyGBLinearConfig;
use crate::data::{PyDataset, PyEvalSet};
use crate::error::BoostersError;

use boosters::data::Dataset as CoreDataset;
use boosters::training::EvalSet as CoreEvalSet;

/// Gradient Boosted Linear model.
///
/// GBLinear uses gradient boosting to train a linear model via coordinate
/// descent. Simpler than GBDT but can be effective for linear relationships.
///
/// # Example
///
/// ```python
/// from boosters import GBLinearModel, GBLinearConfig, Dataset
/// import numpy as np
///
/// # Create training data
/// X = np.random.rand(1000, 10).astype(np.float32)
/// y = np.random.rand(1000).astype(np.float32)
/// train = Dataset(X, y)
///
/// # Train with default config
/// model = GBLinearModel().fit(train)
///
/// # Or with custom config
/// config = GBLinearConfig(n_estimators=50, learning_rate=0.3, l2=0.1)
/// model = GBLinearModel(config=config).fit(train)
///
/// # Predict
/// predictions = model.predict(X_test)
///
/// # Access weights
/// print(model.coef_)      # shape: (n_features,) or (n_features, n_outputs)
/// print(model.intercept_)  # shape: () or (n_outputs,)
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "GBLinearModel", module = "boosters._boosters_rs")]
pub struct PyGBLinearModel {
    /// Configuration for the model.
    config: Py<PyGBLinearConfig>,

    /// Inner trained model (None until fit() is called).
    inner: Option<boosters::GBLinearModel>,

    /// Evaluation results from training (populated after fit).
    eval_results: Option<Py<PyAny>>,

    /// Best iteration from early stopping (if applicable).
    best_iteration: Option<usize>,

    /// Best score from early stopping (if applicable).
    best_score: Option<f64>,
}

#[pymethods]
impl PyGBLinearModel {
    /// Create a new GBLinear model.
    ///
    /// Args:
    ///     config: Optional GBLinearConfig. If not provided, uses default config.
    ///
    /// Returns:
    ///     New GBLinearModel instance (not yet fitted).
    #[new]
    #[pyo3(signature = (config=None))]
    pub fn new(py: Python<'_>, config: Option<Py<PyGBLinearConfig>>) -> PyResult<Self> {
        let config = match config {
            Some(c) => c,
            None => {
                // Create default config using Python interface
                let boosters_mod = py.import_bound("boosters._boosters_rs")?;
                let config_class = boosters_mod.getattr("GBLinearConfig")?;
                let config_obj = config_class.call0()?;
                config_obj.extract::<Py<PyGBLinearConfig>>()?
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

    /// Number of features the model was trained on.
    ///
    /// Raises:
    ///     RuntimeError: If model has not been fitted.
    #[getter]
    pub fn n_features_in_(&self) -> PyResult<usize> {
        match &self.inner {
            Some(model) => Ok(model.meta().n_features),
            None => Err(BoostersError::NotFitted {
                method: "n_features_in_".to_string(),
            }
            .into()),
        }
    }

    /// Model coefficients (weights).
    ///
    /// Returns:
    ///     NumPy array with shape:
    ///     - `(n_features,)` for single-output models
    ///     - `(n_features, n_outputs)` for multi-output models
    ///
    /// Raises:
    ///     RuntimeError: If model has not been fitted.
    #[getter]
    pub fn coef_(&self, py: Python<'_>) -> PyResult<PyObject> {
        use numpy::{PyArray1, PyArray2};

        let model = self.inner.as_ref().ok_or_else(|| BoostersError::NotFitted {
            method: "coef_".to_string(),
        })?;

        let linear = model.linear();
        let n_features = linear.n_features();
        let n_groups = linear.n_groups();

        if n_groups == 1 {
            // Single output: return 1D array (n_features,)
            // Extract feature weights (excluding bias)
            let weights: Vec<f32> = (0..n_features).map(|f| linear.weight(f, 0)).collect();
            let arr = PyArray1::from_vec_bound(py, weights);
            Ok(arr.into_any().unbind())
        } else {
            // Multi-output: return 2D array (n_features, n_groups)
            let coefs = linear.weight_view().to_owned();
            let arr = PyArray2::from_owned_array_bound(py, coefs);
            Ok(arr.into_any().unbind())
        }
    }

    /// Model intercept (bias).
    ///
    /// Returns:
    ///     NumPy array with shape:
    ///     - `()` (scalar) for single-output models
    ///     - `(n_outputs,)` for multi-output models
    ///
    /// Raises:
    ///     RuntimeError: If model has not been fitted.
    #[getter]
    pub fn intercept_(&self, py: Python<'_>) -> PyResult<PyObject> {
        use numpy::PyArray1;

        let model = self.inner.as_ref().ok_or_else(|| BoostersError::NotFitted {
            method: "intercept_".to_string(),
        })?;

        let linear = model.linear();
        let biases = linear.biases();

        // Return 1D array of biases
        let arr = PyArray1::from_iter_bound(py, biases.iter().copied());
        Ok(arr.into_any().unbind())
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
    #[getter]
    pub fn eval_results(&self, py: Python<'_>) -> Option<PyObject> {
        self.eval_results.as_ref().map(|r| r.clone_ref(py))
    }

    /// Get the model configuration.
    #[getter]
    pub fn config(&self, py: Python<'_>) -> Py<PyGBLinearConfig> {
        self.config.clone_ref(py)
    }

    /// String representation.
    fn __repr__(&self) -> String {
        if let Some(model) = &self.inner {
            format!(
                "GBLinearModel(n_features={}, fitted=True)",
                model.meta().n_features
            )
        } else {
            "GBLinearModel(fitted=False)".to_string()
        }
    }

    /// Make predictions on features.
    ///
    /// Args:
    ///     features: Feature array of shape `(n_samples, n_features)` or Dataset.
    ///     raw_score: If True, return raw margin scores without transformation.
    ///
    /// Returns:
    ///     NumPy array with predictions. Shape depends on the objective:
    ///     - Regression: `(n_samples,)`
    ///     - Binary classification: `(n_samples,)` (probability of positive class)
    ///     - Multiclass: `(n_samples, n_classes)`
    ///
    /// Raises:
    ///     RuntimeError: If model has not been fitted.
    ///     ValueError: If features have wrong shape.
    #[pyo3(signature = (features, raw_score=false))]
    pub fn predict(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        raw_score: bool,
    ) -> PyResult<PyObject> {
        use numpy::{PyArray1, PyArray2};

        let model = self.inner.as_ref().ok_or_else(|| BoostersError::NotFitted {
            method: "predict".to_string(),
        })?;

        // Extract features array - either from Dataset or raw numpy array
        let features_array: Array2<f32> = if let Ok(dataset) = features.extract::<PyRef<'_, PyDataset>>() {
            let features_view = dataset.features_array(py)?;
            let features_np = features_view.as_array();
            features_np.t().as_standard_layout().into_owned()
        } else {
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

        // Create temporary dataset for prediction
        let pred_dataset = CoreDataset::new(features_array.view(), None, None);

        // Predict with GIL released
        let output = py.allow_threads(|| {
            if raw_score {
                model.predict_raw(pred_dataset)
            } else {
                model.predict(pred_dataset)
            }
        });

        // output shape is [n_groups, n_samples]
        // Transpose to [n_samples, n_groups] for Python convention
        let output_t = output.t().as_standard_layout().into_owned();

        // Return appropriate shape based on n_groups
        let n_groups = output.nrows();
        if n_groups == 1 {
            let flat: Vec<f32> = output_t.iter().copied().collect();
            let arr = PyArray1::from_vec_bound(py, flat);
            Ok(arr.into_any().unbind())
        } else {
            let arr = PyArray2::from_owned_array_bound(py, output_t);
            Ok(arr.into_any().unbind())
        }
    }

    /// Train the model on a dataset.
    ///
    /// Args:
    ///     train: Training dataset containing features and labels.
    ///     eval_set: Optional validation set(s) for early stopping and evaluation.
    ///
    /// Returns:
    ///     Self (for method chaining).
    ///
    /// Raises:
    ///     ValueError: If training data is invalid or labels are missing.
    #[pyo3(signature = (train, eval_set=None))]
    pub fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        train: &PyDataset,
        eval_set: Option<&Bound<'py, PyAny>>,
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
        let eval_sets: Vec<(String, CoreDataset)> = if let Some(valid_obj) = eval_set {
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
        let n_threads = 0;
        let trained_model = py.allow_threads(|| {
            boosters::GBLinearModel::train(&core_train, &eval_set_refs, core_config, n_threads)
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

// Internal methods
impl PyGBLinearModel {
    /// Extract evaluation sets from Python input.
    fn extract_eval_sets(
        py: Python<'_>,
        valid: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<(String, CoreDataset)>> {
        let mut result = Vec::new();

        if let Ok(list) = valid.downcast::<pyo3::types::PyList>() {
            for item in list.iter() {
                let eval_set: PyRef<'_, PyEvalSet> = item.extract()?;
                let ds = Self::convert_eval_set(py, &eval_set)?;
                result.push(ds);
            }
        } else {
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

        let features_view = dataset.features_array(py)?;
        let features_np = features_view.as_array();
        let features_transposed: Array2<f32> = features_np.t().as_standard_layout().into_owned();

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

        let weights_view = dataset.weights_array(py)?;
        let weights_opt: Option<ndarray::Array1<f32>> =
            weights_view.map(|w| w.as_array().to_owned());

        let core_ds = CoreDataset::new(
            features_transposed.view(),
            labels_2d.as_ref().map(|l| l.view()),
            weights_opt.as_ref().map(|w| w.view()),
        );

        Ok((name, core_ds))
    }
}
