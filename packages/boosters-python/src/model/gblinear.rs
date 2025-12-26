//! GBLinear Model Python bindings.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use boosters::data::transpose_to_c_order;
use boosters::training::EvalSet as CoreEvalSet;

use crate::config::PyGBLinearConfig;
use crate::data::{PyDataset, PyEvalSet};
use crate::error::BoostersError;

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

#[gen_stub_pymethods]
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
                let boosters_mod = py.import("boosters._boosters_rs")?;
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
    ///     Array with shape (n_features,) for single-output models
    ///     or (n_features, n_outputs) for multi-output models.
    ///
    /// Raises:
    ///     RuntimeError: If model has not been fitted.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "numpy.ndarray", imports = ("numpy",)))]
    pub fn coef_(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
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
    ///     Scalar for single-output models or array of shape (n_outputs,)
    ///     for multi-output models.
    ///
    /// Raises:
    ///     RuntimeError: If model has not been fitted.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "numpy.ndarray", imports = ("numpy",)))]
    pub fn intercept_(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use numpy::PyArray1;

        let model = self.inner.as_ref().ok_or_else(|| BoostersError::NotFitted {
            method: "intercept_".to_string(),
        })?;

        let linear = model.linear();
        let biases = linear.biases();

        // Return 1D array of biases
        let arr = PyArray1::from_iter(py, biases.iter().copied());
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
    #[gen_stub(override_return_type(type_repr = "dict[str, dict[str, list[float]]] | None", imports = ()))]
    pub fn eval_results(&self, py: Python<'_>) -> Option<Py<PyAny>> {
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
    /// Returns transformed predictions (e.g., probabilities for classification).
    ///
    /// Args:
    ///     data: Dataset containing features.
    ///
    /// Returns:
    ///     Predictions of shape (n_samples, n_outputs).
    ///     For single-output models, n_outputs is 1.
    ///
    /// Raises:
    ///     RuntimeError: If model has not been fitted.
    ///     ValueError: If features have wrong shape.
    ///
    /// Examples:
    ///     >>> predictions = model.predict(test_data)
    #[pyo3(signature = (data))]
    #[gen_stub(override_return_type(type_repr = "numpy.ndarray", imports = ("numpy",)))]
    pub fn predict(&self, py: Python<'_>, data: PyRef<'_, PyDataset>) -> PyResult<Py<PyAny>> {
        self.predict_internal(py, &data, false)
    }

    /// Make raw (untransformed) predictions on features.
    ///
    /// Returns raw margin scores without transformation.
    /// For classification this means logits instead of probabilities.
    ///
    /// Args:
    ///     data: Dataset containing features.
    ///
    /// Returns:
    ///     Raw scores of shape (n_samples, n_outputs).
    ///     For single-output models, n_outputs is 1.
    ///
    /// Raises:
    ///     RuntimeError: If model has not been fitted.
    ///     ValueError: If features have wrong shape.
    ///
    /// Examples:
    ///     >>> raw_margins = model.predict_raw(test_data)
    #[pyo3(signature = (data))]
    #[gen_stub(override_return_type(type_repr = "numpy.ndarray", imports = ("numpy",)))]
    pub fn predict_raw(&self, py: Python<'_>, data: PyRef<'_, PyDataset>) -> PyResult<Py<PyAny>> {
        self.predict_internal(py, &data, true)
    }

    /// Train the model on a dataset.
    ///
    /// Args:
    ///     train: Training dataset containing features and labels.
    ///     eval_set: Validation set(s) for early stopping and evaluation.
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
        train: PyRef<'py, PyDataset>,
        eval_set: Option<Vec<PyRef<'py, PyEvalSet>>>,
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
        let core_config = config.to_core()?;
        drop(config);

        // Get reference to the core dataset directly
        let core_train = train.as_core();

        // Extract eval sets
        let eval_set_data: Vec<(String, PyRef<'py, PyDataset>)> = eval_set
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
        let n_threads = 0;
        let trained_model = py.detach(|| {
            boosters::GBLinearModel::train(core_train, &eval_set_refs, core_config, n_threads)
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
impl PyGBLinearModel {
    /// Internal prediction method shared by predict and predict_raw.
    ///
    /// Always returns 2D array of shape [n_samples, n_outputs] for consistency.
    fn predict_internal(
        &self,
        py: Python<'_>,
        data: &PyRef<'_, PyDataset>,
        raw_score: bool,
    ) -> PyResult<Py<PyAny>> {
        use numpy::PyArray2;

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

        // Predict with GIL released
        // Note: GBLinearModel::predict takes Dataset by value, so we clone
        let features_owned = core_dataset.features().view().to_owned();
        let output = py.detach(|| {
            let pred_dataset = boosters::data::Dataset::new(features_owned.view(), None, None);
            if raw_score {
                model.predict_raw(pred_dataset)
            } else {
                model.predict(pred_dataset)
            }
        });

        // output shape is [n_groups, n_samples]
        // Transpose to [n_samples, n_groups] for Python convention
        // Always return 2D array for consistent shape
        let output_t = transpose_to_c_order(output.view());
        let arr = PyArray2::from_owned_array(py, output_t);
        Ok(arr.into_any().unbind())
    }
}
