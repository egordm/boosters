//! GBDT booster Python bindings.

use crate::error::PyBoostersError;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashSet;
use std::path::PathBuf;

use boosters::data::{BinnedDatasetBuilder, Dataset, DenseMatrix, FeatureColumn, RowMajor};
use boosters::model::GBDTModel;
use boosters::training::{GBDTParams, GainParams, MetricFunction, ObjectiveFunction};

/// Gradient Boosted Decision Trees model.
///
/// This class provides training and prediction for tree ensemble models,
/// with an sklearn-compatible interface.
///
/// # Example
/// ```python
/// import numpy as np
/// from boosters import GBDTBooster
///
/// X = np.random.randn(1000, 10).astype(np.float32)
/// y = np.random.randn(1000).astype(np.float32)
///
/// # Sklearn-style: constructor takes params, fit() takes data
/// model = GBDTBooster(n_estimators=100, max_depth=6, learning_rate=0.1)
/// model.fit(X, y)
/// predictions = model.predict(X)
///
/// # With feature names
/// model.fit(X, y, feature_names=['f1', 'f2', ...])
/// print(model.feature_importance())  # Returns dict with feature names
/// ```
#[pyclass(name = "GBDTBooster")]
pub struct PyGBDTBooster {
    /// The trained model (None until fit() is called)
    model: Option<GBDTModel>,
    
    // === Common parameters ===
    /// Learning rate (shrinkage). Default: 0.3
    #[pyo3(get, set)]
    pub learning_rate: f32,
    /// Number of boosting rounds. Default: 100
    #[pyo3(get, set)]
    pub n_estimators: usize,
    /// Objective function. Default: "squared_error"
    #[pyo3(get, set)]
    pub objective: String,
    
    // === Tree parameters ===
    /// Maximum depth of each tree. Default: 6
    #[pyo3(get, set)]
    pub max_depth: usize,
    /// Minimum sum of instance weight in a child. Default: 1.0
    #[pyo3(get, set)]
    pub min_child_weight: f32,
    /// L2 regularization on leaf weights. Default: 1.0
    #[pyo3(get, set)]
    pub reg_lambda: f32,
    /// L1 regularization on leaf weights. Default: 0.0
    #[pyo3(get, set)]
    pub reg_alpha: f32,
    /// Minimum loss reduction for split. Default: 0.0
    #[pyo3(get, set)]
    pub gamma: f32,
    /// Maximum number of bins for histogram. Default: 256
    #[pyo3(get, set)]
    pub max_bin: usize,
    /// Subsample ratio of training instances. Default: 1.0
    #[pyo3(get, set)]
    pub subsample: f32,
    /// Subsample ratio of columns for each tree. Default: 1.0
    #[pyo3(get, set)]
    pub colsample_bytree: f32,
    /// Number of classes (for multiclass). Default: None (auto-detect)
    #[pyo3(get, set)]
    pub num_class: Option<usize>,
    /// Random seed. Default: 0
    #[pyo3(get, set)]
    pub random_state: u64,
    
    // === Objective-specific parameters ===
    /// Quantile for quantile regression (0-1). Default: 0.5
    #[pyo3(get, set)]
    pub quantile_alpha: f32,
    /// Delta parameter for Huber loss. Default: 1.0
    #[pyo3(get, set)]
    pub huber_delta: f32,
}

#[pymethods]
impl PyGBDTBooster {
    /// Create a new GBDT booster with the specified parameters.
    ///
    /// # Arguments
    /// * `n_estimators` - Number of boosting rounds (default: 100)
    /// * `learning_rate` - Learning rate/shrinkage (default: 0.3)
    /// * `max_depth` - Maximum tree depth (default: 6)
    /// * `objective` - Loss function: "squared_error", "binary:logistic", "multi:softmax", "quantile", "huber", etc.
    /// * `reg_lambda` - L2 regularization (default: 1.0)
    /// * `reg_alpha` - L1 regularization (default: 0.0)
    /// * `gamma` - Minimum loss reduction for split (default: 0.0)
    /// * `min_child_weight` - Minimum sum of instance weight in child (default: 1.0)
    /// * `subsample` - Row subsampling ratio (default: 1.0)
    /// * `colsample_bytree` - Column subsampling ratio per tree (default: 1.0)
    /// * `max_bin` - Maximum histogram bins (default: 256)
    /// * `num_class` - Number of classes for multiclass (default: auto-detect)
    /// * `random_state` - Random seed (default: 0)
    /// * `quantile_alpha` - Quantile for quantile regression, 0-1 (default: 0.5)
    /// * `huber_delta` - Delta for Huber/pseudo-Huber loss (default: 1.0)
    #[new]
    #[pyo3(signature = (
        n_estimators = 100,
        learning_rate = 0.3,
        max_depth = 6,
        objective = "squared_error".to_string(),
        reg_lambda = 1.0,
        reg_alpha = 0.0,
        gamma = 0.0,
        min_child_weight = 1.0,
        subsample = 1.0,
        colsample_bytree = 1.0,
        max_bin = 256,
        num_class = None,
        random_state = 0,
        quantile_alpha = 0.5,
        huber_delta = 1.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_estimators: usize,
        learning_rate: f32,
        max_depth: usize,
        objective: String,
        reg_lambda: f32,
        reg_alpha: f32,
        gamma: f32,
        min_child_weight: f32,
        subsample: f32,
        colsample_bytree: f32,
        max_bin: usize,
        num_class: Option<usize>,
        random_state: u64,
        quantile_alpha: f32,
        huber_delta: f32,
    ) -> Self {
        Self {
            model: None,
            learning_rate,
            n_estimators,
            objective,
            max_depth,
            min_child_weight,
            reg_lambda,
            reg_alpha,
            gamma,
            max_bin,
            subsample,
            colsample_bytree,
            num_class,
            random_state,
            quantile_alpha,
            huber_delta,
        }
    }

    /// Fit the model to training data.
    ///
    /// # Arguments
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    /// * `y` - Target values of shape (n_samples,)
    /// * `sample_weight` - Optional sample weights
    /// * `feature_names` - Optional feature names for interpretability
    /// * `categorical_features` - Optional list of indices for categorical features.
    ///                            Values at these indices are treated as integer category IDs.
    ///
    /// # Returns
    /// Self (for method chaining in Python)
    ///
    /// # Example
    /// ```python
    /// # With categorical features at indices 0 and 3
    /// model.fit(X, y, categorical_features=[0, 3])
    /// ```
    #[pyo3(signature = (x, y, sample_weight=None, feature_names=None, categorical_features=None))]
    #[allow(clippy::wrong_self_convention)]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f32>,
        y: PyReadonlyArray1<'py, f32>,
        sample_weight: Option<PyReadonlyArray1<'py, f32>>,
        feature_names: Option<Vec<String>>,
        categorical_features: Option<Vec<usize>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let shape = x.shape();
        let n_rows = shape[0];
        let n_cols = shape[1];
        
        // Validate inputs
        let labels = y.as_slice()?;
        if labels.len() != n_rows {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "y length {} doesn't match x rows {}",
                labels.len(),
                n_rows
            )));
        }
        
        // Validate feature names
        if let Some(ref names) = feature_names {
            if names.len() != n_cols {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "feature_names length {} doesn't match X columns {}",
                    names.len(),
                    n_cols
                )));
            }
        }
        
        // Create set of categorical feature indices for fast lookup
        let cat_indices: HashSet<usize> = categorical_features
            .as_ref()
            .map(|v| v.iter().copied().collect())
            .unwrap_or_default();
        
        // Validate categorical indices
        for &idx in &cat_indices {
            if idx >= n_cols {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "categorical_features index {} out of bounds (n_features={})",
                    idx, n_cols
                )));
            }
        }
        
        // Create matrix from input - handles both C-contiguous (NumPy default) 
        // and Fortran-contiguous (pandas DataFrame) arrays
        let features: Vec<f32> = if x.is_c_contiguous() {
            // Fast path: C-contiguous array, can use slice directly
            x.as_slice()?.to_vec()
        } else {
            // Slow path: non-contiguous array (e.g., Fortran-order from pandas)
            // Use as_array() to get ndarray view that handles strides
            x.as_array().iter().copied().collect()
        };
        
        // Build binned dataset using appropriate method
        let binned_dataset = if cat_indices.is_empty() {
            // Fast path: no categorical features, use direct matrix binning
            let matrix: DenseMatrix<f32, RowMajor, _> = DenseMatrix::from_vec(features, n_rows, n_cols);
            BinnedDatasetBuilder::from_row_matrix(&matrix, slf.max_bin as u32)
                .build()
                .map_err(|e| PyBoostersError::Training(format!("Failed to build dataset: {}", e)))?
        } else {
            // Has categorical features: build Dataset with proper FeatureColumn types
            let mut columns = Vec::with_capacity(n_cols);
            
            for col_idx in 0..n_cols {
                let name = feature_names.as_ref().map(|names| names[col_idx].clone());
                
                if cat_indices.contains(&col_idx) {
                    // Categorical: extract column and convert to i32 category IDs
                    let values: Vec<i32> = (0..n_rows)
                        .map(|row| features[row * n_cols + col_idx] as i32)
                        .collect();
                    columns.push(FeatureColumn::Categorical { name, values });
                } else {
                    // Numeric: extract column as f32
                    let values: Vec<f32> = (0..n_rows)
                        .map(|row| features[row * n_cols + col_idx])
                        .collect();
                    columns.push(FeatureColumn::Numeric { name, values });
                }
            }
            
            let dataset = Dataset::new(columns, labels.to_vec())
                .map_err(|e| PyBoostersError::Training(format!("Failed to create dataset: {}", e)))?;
            
            dataset.to_binned(slf.max_bin as u32)
                .map_err(|e| PyBoostersError::Training(format!("Failed to bin dataset: {}", e)))?
        };
        
        // Create training params
        let trainer_params = GBDTParams {
            n_trees: slf.n_estimators as u32,
            learning_rate: slf.learning_rate,
            gain: GainParams {
                reg_lambda: slf.reg_lambda,
                reg_alpha: slf.reg_alpha,
                min_child_weight: slf.min_child_weight,
                min_gain: slf.gamma,
                min_samples_leaf: 1,
            },
            ..Default::default()
        };
        
        // Handle optional sample weights
        let weights: Option<Vec<f32>> = if let Some(w) = sample_weight {
            let w_slice = w.as_slice()?;
            if w_slice.len() != n_rows {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "sample_weight length {} doesn't match x rows {}",
                    w_slice.len(),
                    n_rows
                )));
            }
            Some(w_slice.to_vec())
        } else {
            None
        };
        
        // Train the model
        let model = Self::train_internal(
            &binned_dataset,
            labels,
            weights.as_deref(),
            &slf.objective,
            slf.num_class,
            slf.quantile_alpha,
            slf.huber_delta,
            trainer_params,
        )?;
        
        // Set feature names if provided and not already set via categorical path
        let model = if !cat_indices.is_empty() {
            // Feature names already set via Dataset when using categorical features
            model
        } else if let Some(names) = feature_names {
            model.with_feature_names(names)
        } else {
            model
        };
        
        slf.model = Some(model);
        Ok(slf)
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
            // Non-contiguous: copy via ndarray view
            features = x.as_array().iter().copied().collect();
            &features
        };

        let predictions = model.predict_batch(features_slice, n_samples);
        Ok(PyArray1::from_vec_bound(py, predictions))
    }

    /// Compute feature importance.
    ///
    /// # Arguments
    /// * `importance_type` - Type of importance: "split" (default), "gain", or "cover"
    ///
    /// # Returns
    /// Dictionary mapping feature names to importance scores
    #[pyo3(signature = (importance_type = "split".to_string()))]
    fn feature_importance(
        &self,
        py: Python<'_>,
        importance_type: String,
    ) -> PyResult<PyObject> {
        use boosters::explainability::ImportanceType;
        use pyo3::types::PyDict;

        let model = self.model.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Model not fitted. Call fit() before feature_importance()."
            )
        })?;

        let imp_type = match importance_type.as_str() {
            "split" | "weight" => ImportanceType::Split,
            "gain" => ImportanceType::Gain,
            "cover" => ImportanceType::Cover,
            "average_gain" | "avg_gain" => ImportanceType::AverageGain,
            "average_cover" | "avg_cover" => ImportanceType::AverageCover,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown importance type: {}. Use 'split', 'gain', or 'cover'",
                    importance_type
                )))
            }
        };

        let importance = model
            .feature_importance(imp_type)
            .map_err(|e| PyBoostersError::Explainability(e.to_string()))?;

        let dict = PyDict::new_bound(py);
        
        // Use feature names if available, otherwise use indices
        if let Some(named_map) = importance.to_named_map() {
            for (name, val) in named_map {
                dict.set_item(name, val)?;
            }
        } else {
            for (idx, val) in importance.to_map() {
                dict.set_item(idx, val)?;
            }
        }
        
        Ok(dict.into())
    }

    /// Compute SHAP values for explaining predictions.
    ///
    /// # Arguments
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    ///
    /// # Returns
    /// SHAP values as a NumPy array of shape (n_samples, n_features)
    fn shap_values<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
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

        let shap = model
            .shap_values(features_slice, n_samples)
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
    #[getter]
    fn expected_value(&self) -> PyResult<f64> {
        let model = self.model.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted.")
        })?;
        Ok(model.base_scores().first().copied().unwrap_or(0.0) as f64)
    }

    /// Number of trees in the ensemble.
    #[getter]
    fn n_trees(&self) -> PyResult<usize> {
        let model = self.model.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted.")
        })?;
        Ok(model.n_trees())
    }

    /// Number of features.
    #[getter]
    fn n_features(&self) -> PyResult<usize> {
        let model = self.model.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted.")
        })?;
        Ok(model.n_features())
    }

    /// Feature names, if available.
    #[getter]
    fn feature_names(&self) -> Option<Vec<String>> {
        self.model.as_ref()?.feature_names().map(|v| v.to_vec())
    }
    
    /// Whether the model has been fitted.
    #[getter]
    fn is_fitted(&self) -> bool {
        self.model.is_some()
    }

    /// Save the model to a file.
    fn save(&self, path: PathBuf) -> PyResult<()> {
        let model = self.model.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted.")
        })?;
        model
            .save(&path)
            .map_err(|e| PyBoostersError::Serialization(e.to_string()))?;
        Ok(())
    }

    /// Load a model from a file.
    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        let model = GBDTModel::load(&path)
            .map_err(|e| PyBoostersError::Deserialization(e.to_string()))?;

        // Create booster with default params (TODO: save/load params with model)
        let mut booster = Self::new(
            100, 0.3, 6, "squared_error".to_string(),
            1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 256, None, 0,
            0.5, 1.0  // quantile_alpha, huber_delta
        );
        booster.model = Some(model);
        Ok(booster)
    }

    /// Serialize the model to bytes.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let model = self.model.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted.")
        })?;
        let bytes = model
            .to_bytes()
            .map_err(|e| PyBoostersError::Serialization(e.to_string()))?;
        Ok(PyBytes::new_bound(py, &bytes))
    }

    /// Deserialize a model from bytes.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        let model = GBDTModel::from_bytes(data)
            .map_err(|e| PyBoostersError::Deserialization(e.to_string()))?;
        
        let mut booster = Self::new(
            100, 0.3, 6, "squared_error".to_string(),
            1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 256, None, 0,
            0.5, 1.0  // quantile_alpha, huber_delta
        );
        booster.model = Some(model);
        Ok(booster)
    }

    /// Support for Python pickling.
    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.to_bytes(py)
    }

    /// Support for Python unpickling.
    fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        let loaded = Self::from_bytes(state)?;
        self.model = loaded.model;
        Ok(())
    }

    fn __repr__(&self) -> String {
        if self.model.is_some() {
            format!(
                "GBDTBooster(n_estimators={}, learning_rate={}, objective='{}', is_fitted=True)",
                self.n_estimators, self.learning_rate, self.objective
            )
        } else {
            format!(
                "GBDTBooster(n_estimators={}, learning_rate={}, objective='{}', is_fitted=False)",
                self.n_estimators, self.learning_rate, self.objective
            )
        }
    }
}

impl PyGBDTBooster {
    /// Internal training method with objective dispatch.
    fn train_internal(
        dataset: &boosters::data::BinnedDataset,
        labels: &[f32],
        weights: Option<&[f32]>,
        objective_str: &str,
        num_class: Option<usize>,
        quantile_alpha: f32,
        huber_delta: f32,
        params: GBDTParams,
    ) -> PyResult<GBDTModel> {
        // Parse objective string to ObjectiveFunction enum
        let objective = match objective_str {
            "squared_error" | "reg:squared_error" | "reg:squarederror" => {
                ObjectiveFunction::SquaredError
            }
            "absolute_error" | "reg:absoluteerror" | "mae" => ObjectiveFunction::AbsoluteError,
            "binary:logistic" | "binary_logistic" | "logistic" => ObjectiveFunction::Logistic,
            "binary:hinge" => ObjectiveFunction::Hinge,
            "multi:softmax" | "multiclass" | "softmax" | "multi:softprob" => {
                let n_classes = num_class.ok_or_else(|| {
                    PyBoostersError::InvalidParameter(
                        "num_class required for multiclass objective".into(),
                    )
                })?;
                ObjectiveFunction::Softmax { num_classes: n_classes }
            }
            "reg:quantile" | "quantile" => ObjectiveFunction::Quantile { alpha: quantile_alpha },
            "reg:pseudohuber" | "huber" => ObjectiveFunction::PseudoHuber { delta: huber_delta },
            "poisson" | "count:poisson" => ObjectiveFunction::Poisson,
            _ => {
                return Err(PyBoostersError::InvalidParameter(format!(
                    "Unknown objective: '{}'. Supported: squared_error, absolute_error, binary:logistic, binary:hinge, multi:softmax, quantile, huber, poisson",
                    objective_str
                ))
                .into());
            }
        };

        // Get matching default metric
        let metric = MetricFunction::from_objective_str(objective_str);

        // Train model (use empty slice for no weights)
        let weights_slice = weights.unwrap_or(&[]);
        let model = GBDTModel::train(dataset, labels, weights_slice, objective, metric, params);

        model.ok_or_else(|| PyBoostersError::Training("Training failed".into()).into())
    }
}
