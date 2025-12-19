//! GBLinear booster Python bindings.

use crate::error::PyBoostersError;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::path::PathBuf;

use boosters::data::{ColMatrix, Dataset as BoostersDataset, DenseMatrix};
use boosters::model::{GBLinearModel, ModelMeta, TaskKind};
use boosters::training::{GBLinearParams, GBLinearTrainer, MetricFunction, ObjectiveFunction};

/// Gradient Boosted Linear model with sklearn-style interface.
///
/// This class provides training and prediction for linear booster models.
/// Use the constructor to set hyperparameters, then call `fit()` with data.
///
/// # Example
/// ```python
/// import numpy as np
/// from boosters_python import GBLinearBooster
///
/// X = np.random.randn(1000, 10).astype(np.float32)
/// y = np.random.randn(1000).astype(np.float32)
///
/// model = GBLinearBooster(n_estimators=100, learning_rate=0.5)
/// model.fit(X, y)
/// predictions = model.predict(X)
/// ```
#[pyclass(name = "GBLinearBooster", module = "boosters._boosters_python")]
pub struct PyGBLinearBooster {
    // Model (None until fitted)
    model: Option<GBLinearModel>,
    
    // Hyperparameters
    learning_rate: f32,
    n_estimators: usize,
    objective: String,
    reg_lambda: f32,
    reg_alpha: f32,
    num_class: Option<usize>,
    
    // Metadata
    feature_names_: Option<Vec<String>>,
}

#[pymethods]
impl PyGBLinearBooster {
    /// Create a new GBLinearBooster with the given hyperparameters.
    ///
    /// # Arguments
    /// * `learning_rate` - Step size shrinkage. Default: 0.5
    /// * `n_estimators` - Number of boosting rounds. Default: 100
    /// * `objective` - Objective function. Default: "squared_error"
    /// * `reg_lambda` - L2 regularization. Default: 0.0
    /// * `reg_alpha` - L1 regularization. Default: 0.0
    /// * `num_class` - Number of classes for multiclass. Default: None
    #[new]
    #[pyo3(signature = (
        learning_rate = 0.5,
        n_estimators = 100,
        objective = "squared_error".to_string(),
        reg_lambda = 0.0,
        reg_alpha = 0.0,
        num_class = None,
    ))]
    fn new(
        learning_rate: f32,
        n_estimators: usize,
        objective: String,
        reg_lambda: f32,
        reg_alpha: f32,
        num_class: Option<usize>,
    ) -> Self {
        Self {
            model: None,
            learning_rate,
            n_estimators,
            objective,
            reg_lambda,
            reg_alpha,
            num_class,
            feature_names_: None,
        }
    }

    /// Fit the model to training data.
    ///
    /// # Arguments
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    /// * `y` - Target values of shape (n_samples,)
    /// * `sample_weight` - Sample weights. Default: None
    /// * `feature_names` - Feature names. Default: None
    ///
    /// # Returns
    /// Self for method chaining
    #[pyo3(signature = (x, y, sample_weight = None, feature_names = None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f32>,
        y: PyReadonlyArray1<'py, f32>,
        sample_weight: Option<PyReadonlyArray1<'py, f32>>,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let shape = x.shape();
        let n_rows = shape[0];
        let n_cols = shape[1];
        
        // Handle both C-contiguous and Fortran-contiguous arrays
        let features: Vec<f32> = if x.is_c_contiguous() {
            x.as_slice()?.to_vec()
        } else {
            // Non-contiguous: copy via ndarray view
            x.as_array().iter().copied().collect()
        };
        let targets = y.as_slice()?.to_vec();

        if targets.len() != n_rows {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "x has {} samples but y has {} samples",
                n_rows,
                targets.len()
            )));
        }

        // Get weights if provided
        let weights: Vec<f32> = if let Some(w) = sample_weight {
            let w_slice = w.as_slice()?;
            if w_slice.len() != n_rows {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "x has {} samples but sample_weight has {} samples",
                    n_rows,
                    w_slice.len()
                )));
            }
            w_slice.to_vec()
        } else {
            vec![]
        };

        // Store feature names
        slf.feature_names_ = feature_names.clone();

        // Convert from row-major to column-major (GBLinear requires column-major layout)
        let mut col_major_data = vec![0.0f32; n_rows * n_cols];
        for row in 0..n_rows {
            for col in 0..n_cols {
                col_major_data[col * n_rows + row] = features[row * n_cols + col];
            }
        }
        let col_matrix: ColMatrix<f32, Vec<f32>> = DenseMatrix::new(col_major_data, n_rows, n_cols);

        // Create boosters Dataset from ColMatrix
        let dataset = BoostersDataset::from_numeric(&col_matrix, targets)
            .map_err(|e| PyBoostersError::Training(format!("Failed to create dataset: {}", e)))?;

        // Create training params
        let trainer_params = GBLinearParams {
            n_rounds: slf.n_estimators as u32,
            learning_rate: slf.learning_rate,
            alpha: slf.reg_alpha,
            lambda: slf.reg_lambda,
            ..Default::default()
        };

        // Train with appropriate objective
        let (linear_model, n_groups) = slf.train_with_objective(&dataset, &weights, trainer_params)?;

        // Create metadata
        let task = match slf.objective.as_str() {
            "squared_error" | "reg:squared_error" | "reg:squarederror" => TaskKind::Regression,
            "binary:logistic" | "binary:logitraw" => TaskKind::BinaryClassification,
            _ => TaskKind::Regression,
        };

        let meta = ModelMeta {
            n_features: n_cols,
            n_groups,
            task,
            feature_names: feature_names,
            ..Default::default()
        };

        let model = GBLinearModel::from_linear_model(linear_model, meta, slf.n_estimators as u32);
        slf.model = Some(model);

        Ok(slf)
    }

    /// Whether the model has been fitted.
    #[getter]
    fn is_fitted(&self) -> bool {
        self.model.is_some()
    }

    /// Make predictions on input data.
    ///
    /// # Arguments
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    ///
    /// # Returns
    /// Predictions as a NumPy array
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let model = self.model.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Model not fitted. Call fit() before predict()."
            )
        })?;

        let shape = x.shape();
        let n_samples = shape[0];
        
        // Handle both C-contiguous and Fortran-contiguous arrays
        let features: Vec<f32>;
        let features_slice = if x.is_c_contiguous() {
            x.as_slice()?
        } else {
            features = x.as_array().iter().copied().collect();
            &features
        };

        let predictions = model.predict_batch(features_slice, n_samples);

        Ok(PyArray1::from_vec_bound(py, predictions))
    }

    /// Compute SHAP values for explaining predictions.
    ///
    /// # Arguments
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    /// * `feature_means` - Optional mean value for each feature (background distribution).
    ///                     If not provided, assumes data is centered (zeros).
    ///
    /// # Returns
    /// SHAP values as a NumPy array of shape (n_samples, n_features)
    #[pyo3(signature = (x, feature_means=None))]
    fn shap_values<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
        feature_means: Option<PyReadonlyArray1<'py, f64>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let model = self.model.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Model not fitted. Call fit() before shap_values()."
            )
        })?;

        let shape = x.shape();
        let n_samples = shape[0];
        let n_features = shape[1];
        
        // Handle both C-contiguous and Fortran-contiguous arrays
        let features: Vec<f32>;
        let features_slice = if x.is_c_contiguous() {
            x.as_slice()?
        } else {
            features = x.as_array().iter().copied().collect();
            &features
        };
        
        // Use provided means or default to zeros (centered data assumption)
        let means = match feature_means {
            Some(arr) => arr.as_slice()?.to_vec(),
            None => vec![0.0; n_features],
        };

        let shap = model
            .shap_values(features_slice, n_samples, Some(means))
            .map_err(PyBoostersError::from)?;

        // Convert to 2D array (n_samples, n_features)
        let mut result = Vec::with_capacity(n_samples * n_features);
        for sample in 0..n_samples {
            for feature in 0..n_features {
                result.push(shap.get(sample, feature, 0));
            }
        }

        let arr = PyArray2::from_vec2_bound(
            py,
            &result.chunks(n_features).map(|r| r.to_vec()).collect::<Vec<_>>(),
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(arr)
    }

    /// Get the expected value (base value) for SHAP.
    ///
    /// For linear models, this is the bias term.
    #[getter]
    fn expected_value(&self) -> PyResult<f64> {
        let model = self.model.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Model not fitted. Call fit() first."
            )
        })?;
        Ok(model.bias(0) as f64)
    }

    /// Number of features.
    #[getter]
    fn n_features(&self) -> PyResult<usize> {
        let model = self.model.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Model not fitted. Call fit() first."
            )
        })?;
        Ok(model.n_features())
    }

    /// Number of output groups.
    #[getter]
    fn n_groups(&self) -> PyResult<usize> {
        let model = self.model.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Model not fitted. Call fit() first."
            )
        })?;
        Ok(model.n_groups())
    }

    /// Number of training rounds.
    #[getter]
    fn n_rounds(&self) -> PyResult<usize> {
        let model = self.model.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Model not fitted. Call fit() first."
            )
        })?;
        Ok(model.n_rounds() as usize)
    }

    /// Get model weights as a NumPy array.
    ///
    /// Returns array of shape (n_features,) for single-output,
    /// or (n_features, n_groups) for multi-output.
    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let model = self.model.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Model not fitted. Call fit() first."
            )
        })?;

        let n_features = model.n_features();
        let n_groups = model.n_groups();

        let mut weights = Vec::with_capacity(n_features * n_groups);
        for group in 0..n_groups {
            for feature in 0..n_features {
                weights.push(model.weight(feature, group));
            }
        }

        Ok(PyArray1::from_vec_bound(py, weights))
    }

    /// Get model bias as a NumPy array.
    #[getter]
    fn bias<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let model = self.model.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Model not fitted. Call fit() first."
            )
        })?;

        let bias = model.biases().to_vec();
        Ok(PyArray1::from_vec_bound(py, bias))
    }

    /// Save the model to a file.
    fn save(&self, path: PathBuf) -> PyResult<()> {
        let model = self.model.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Model not fitted. Call fit() first."
            )
        })?;

        model
            .save(&path)
            .map_err(|e| PyBoostersError::Serialization(e.to_string()))?;
        Ok(())
    }

    /// Load a model from a file.
    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        let model = GBLinearModel::load(&path)
            .map_err(|e| PyBoostersError::Deserialization(e.to_string()))?;

        Ok(Self {
            model: Some(model),
            learning_rate: 0.5,
            n_estimators: 100,
            objective: "squared_error".to_string(),
            reg_lambda: 0.0,
            reg_alpha: 0.0,
            num_class: None,
            feature_names_: None,
        })
    }

    /// Serialize the model to bytes.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let model = self.model.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Model not fitted. Call fit() first."
            )
        })?;

        let bytes = model
            .to_bytes()
            .map_err(|e| PyBoostersError::Serialization(e.to_string()))?;

        Ok(PyBytes::new_bound(py, &bytes))
    }

    /// Deserialize a model from bytes.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        let model = GBLinearModel::from_bytes(data)
            .map_err(|e| PyBoostersError::Deserialization(e.to_string()))?;

        Ok(Self {
            model: Some(model),
            learning_rate: 0.5,
            n_estimators: 100,
            objective: "squared_error".to_string(),
            reg_lambda: 0.0,
            reg_alpha: 0.0,
            num_class: None,
            feature_names_: None,
        })
    }

    /// Support for Python pickling.
    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.to_bytes(py)
    }

    /// Support for Python unpickling.
    fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        let loaded = Self::from_bytes(state)?;
        self.model = loaded.model;
        self.learning_rate = loaded.learning_rate;
        self.n_estimators = loaded.n_estimators;
        self.objective = loaded.objective;
        self.reg_lambda = loaded.reg_lambda;
        self.reg_alpha = loaded.reg_alpha;
        self.num_class = loaded.num_class;
        self.feature_names_ = loaded.feature_names_;
        Ok(())
    }

    fn __repr__(&self) -> String {
        if let Some(ref model) = self.model {
            format!(
                "GBLinearBooster(n_features={}, n_groups={}, n_estimators={})",
                model.n_features(),
                model.n_groups(),
                self.n_estimators
            )
        } else {
            format!(
                "GBLinearBooster(n_estimators={}, learning_rate={}, objective='{}', fitted=False)",
                self.n_estimators, self.learning_rate, self.objective
            )
        }
    }
}

impl PyGBLinearBooster {
    /// Train with the appropriate objective and metric based on the objective string.
    fn train_with_objective(
        &self,
        dataset: &boosters::data::Dataset,
        _weights: &[f32],
        params: GBLinearParams,
    ) -> PyResult<(boosters::repr::gblinear::LinearModel, usize)> {
        // Parse objective string to ObjectiveFunction enum
        let (objective, n_groups) = match self.objective.as_str() {
            // Regression objectives
            "squared_error" | "reg:squared_error" | "reg:squarederror" => {
                (ObjectiveFunction::SquaredError, 1)
            }
            // Binary classification
            "binary:logistic" | "binary_logistic" | "logistic" => {
                (ObjectiveFunction::Logistic, 1)
            }
            _ => {
                return Err(PyBoostersError::InvalidParameter(format!(
                    "Unknown objective: '{}'. Supported: squared_error, binary:logistic",
                    self.objective
                ))
                .into());
            }
        };

        // Get matching default metric
        let metric = MetricFunction::from_objective_str(&self.objective);

        let trainer = GBLinearTrainer::new(objective, metric, params);
        let model = trainer
            .train(dataset, &[])
            .ok_or_else(|| PyBoostersError::Training("Training failed".into()))?;
        
        Ok((model, n_groups))
    }
}
