//! GBLinear Model Python bindings.

use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use boosters::training::EvalSet as CoreEvalSet;

use crate::config::PyGBLinearConfig;
use crate::data::{PyDataset, PyEvalSet};
use crate::error::BoostersError;
use crate::validation::{require_fitted, validate_feature_count};

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
    #[getter]
    pub fn n_features_in_(&self) -> PyResult<usize> {
        let model = require_fitted(self.inner.as_ref(), "n_features_in_")?;
        Ok(model.meta().n_features)
    }

    /// Model coefficients (weights).
    ///
    /// Returns:
    ///     Array with shape (n_features,) for single-output models
    ///     or (n_features, n_outputs) for multi-output models.
    #[getter]
    pub fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let model = require_fitted(self.inner.as_ref(), "coef_")?;

        let linear = model.linear();
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
        let model = require_fitted(self.inner.as_ref(), "intercept_")?;

        let linear = model.linear();
        let biases = linear.biases();

        Ok(PyArray1::from_iter(py, biases.iter().copied()))
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
        let model = require_fitted(self.inner.as_ref(), "predict")?;
        let dataset = data.inner();

        validate_feature_count(model.meta().n_features, dataset.n_features())?;

        // GBLinearModel::predict takes Dataset by value, so we need to create one
        let features_owned = dataset.features().view().to_owned();
        let output = py.detach(|| {
            let pred_dataset = boosters::data::Dataset::new(features_owned.view(), None, None);
            model.predict(pred_dataset)
        });

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
        let model = require_fitted(self.inner.as_ref(), "predict_raw")?;
        let dataset = data.inner();

        validate_feature_count(model.meta().n_features, dataset.n_features())?;

        // GBLinearModel::predict_raw takes Dataset by value, so we need to create one
        let features_owned = dataset.features().view().to_owned();
        let output = py.detach(|| {
            let pred_dataset = boosters::data::Dataset::new(features_owned.view(), None, None);
            model.predict_raw(pred_dataset)
        });

        // Transpose from (n_outputs, n_samples) to (n_samples, n_outputs) for sklearn
        Ok(PyArray2::from_owned_array(py, output.t().to_owned()))
    }

    /// Train the model on a dataset.
    ///
    /// Args:
    ///     train: Training dataset containing features and labels.
    ///     eval_set: Validation set(s) for early stopping and evaluation.
    ///     n_threads: Number of threads for parallel training (0 = auto).
    ///
    /// Returns:
    ///     Self (for method chaining).
    #[pyo3(signature = (train, eval_set=None, n_threads=0))]
    pub fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        train: PyRef<'py, PyDataset>,
        eval_set: Option<Vec<PyRef<'py, PyEvalSet>>>,
        n_threads: usize,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if !train.has_labels() {
            return Err(BoostersError::ValidationError(
                "Training dataset must have labels".to_string(),
            )
            .into());
        }

        // Convert Python config to core config using Into trait
        let config = slf.config.bind(py).borrow();
        let core_config: boosters::GBLinearConfig = (&*config).into();
        drop(config);

        let core_train = train.inner();

        // Build eval sets from PyEvalSet references
        let py_eval_sets: Vec<_> = eval_set.unwrap_or_default();
        let borrowed: Vec<_> = py_eval_sets
            .iter()
            .map(|es| (es.name_str(), es.dataset_ref().bind(py).borrow()))
            .collect();
        let eval_sets: Vec<CoreEvalSet<'_>> = borrowed
            .iter()
            .map(|(name, ds)| CoreEvalSet::new(name, ds.inner()))
            .collect();

        // Train with GIL released
        let trained_model = py.detach(|| {
            boosters::GBLinearModel::train(core_train, &eval_sets, core_config, n_threads)
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
