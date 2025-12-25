//! GBDT Model Python bindings.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyclass;

use crate::config::PyGBDTConfig;
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
                    message: format!(
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
}
