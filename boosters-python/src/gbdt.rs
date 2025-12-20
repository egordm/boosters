//! GBDT booster Python bindings.

use crate::dataset::PyDataset;
use crate::error::PyBoostersError;
use crate::params::PyGBDTParams;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashMap;
use std::path::PathBuf;

use boosters::data::{BinnedDatasetBuilder, DenseMatrix, RowMajor};
use boosters::model::GBDTModel;
use boosters::training::{GBDTParams, GainParams, MetricFunction, ObjectiveFunction};

/// Gradient Boosted Decision Trees model.
///
/// This class provides training and prediction for tree ensemble models.
///
/// # Example
/// ```python
/// import numpy as np
/// from boosters_python import GBDTBooster, GBDTParams, Dataset
///
/// X = np.random.randn(1000, 10).astype(np.float32)
/// y = np.random.randn(1000).astype(np.float32)
///
/// dataset = Dataset(X, y)
/// params = GBDTParams(n_estimators=100, max_depth=6)
///
/// model = GBDTBooster.train(params, dataset)
/// predictions = model.predict(X)
/// ```
#[pyclass(name = "GBDTBooster")]
pub struct PyGBDTBooster {
    model: GBDTModel,
    params: PyGBDTParams,
}

#[pymethods]
impl PyGBDTBooster {
    /// Train a new GBDT model.
    ///
    /// # Arguments
    /// * `params` - Training parameters
    /// * `train_data` - Training dataset
    /// * `eval_data` - Optional evaluation dataset (for early stopping)
    #[staticmethod]
    #[pyo3(signature = (params, train_data, eval_data=None))]
    fn train(
        params: PyGBDTParams,
        train_data: PyDataset,
        eval_data: Option<PyDataset>,
    ) -> PyResult<Self> {
        let _ = eval_data; // TODO: implement early stopping

        let labels = train_data
            .labels()
            .ok_or_else(|| PyBoostersError::InvalidData("Labels required for training".into()))?;

        // Create RowMatrix from the data
        let n_rows = train_data.num_samples();
        let n_cols = train_data.num_features();
        let features = train_data.features().to_vec();
        let matrix: DenseMatrix<f32, RowMajor, _> = DenseMatrix::from_vec(features, n_rows, n_cols);

        // Build binned dataset
        let binned_dataset = BinnedDatasetBuilder::from_row_matrix(&matrix, params.max_bin as u32)
            .build()
            .map_err(|e| PyBoostersError::Training(format!("Failed to build dataset: {}", e)))?;

        // Create training params
        let trainer_params = Self::params_to_trainer_params(&params);

        // Train with appropriate objective based on params
        let model = Self::train_with_objective(
            &binned_dataset,
            labels,
            train_data.weights(),
            &params.objective,
            params.num_class,
            trainer_params,
        )?;

        Ok(Self { model, params })
    }

    /// Make predictions on input data.
    ///
    /// # Arguments
    /// * `data` - Feature matrix of shape (n_samples, n_features)
    ///
    /// # Returns
    /// Predictions as a NumPy array
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let shape = data.shape();
        let n_samples = shape[0];
        let features = data.as_slice()?;

        let predictions = self.model.predict_batch(features, n_samples);

        Ok(PyArray1::from_vec_bound(py, predictions))
    }

    /// Compute feature importance.
    ///
    /// # Arguments
    /// * `importance_type` - Type of importance: "split" (default), "gain", or "cover"
    ///
    /// # Returns
    /// Dictionary mapping feature names/indices to importance scores
    #[pyo3(signature = (importance_type = "split".to_string()))]
    fn feature_importance(
        &self,
        importance_type: String,
    ) -> PyResult<HashMap<usize, f64>> {
        use boosters::explainability::ImportanceType;

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

        let importance = self
            .model
            .feature_importance(imp_type)
            .map_err(|e| PyBoostersError::Explainability(e.to_string()))?;

        Ok(importance.to_map())
    }

    /// Compute SHAP values for explaining predictions.
    ///
    /// # Arguments
    /// * `data` - Feature matrix of shape (n_samples, n_features)
    ///
    /// # Returns
    /// SHAP values as a NumPy array of shape (n_samples, n_features)
    fn shap_values<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = data.shape();
        let n_samples = shape[0];
        let n_features = shape[1];
        let features = data.as_slice()?;

        let shap = self
            .model
            .shap_values(features, n_samples)
            .map_err(PyBoostersError::from)?;

        // Convert to 2D array (n_samples, n_features)
        // Note: shap includes base value at the end, we exclude it here
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
    fn expected_value(&self) -> f64 {
        // Base score from forest
        self.model.base_scores().first().copied().unwrap_or(0.0) as f64
    }

    /// Number of trees in the ensemble.
    #[getter]
    fn n_trees(&self) -> usize {
        self.model.n_trees()
    }

    /// Number of features.
    #[getter]
    fn n_features(&self) -> usize {
        self.model.n_features()
    }

    /// Feature names, if available.
    #[getter]
    fn feature_names(&self) -> Option<Vec<String>> {
        self.model.feature_names().map(|v| v.to_vec())
    }

    /// Save the model to a file.
    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.model
            .save(&path)
            .map_err(|e| PyBoostersError::Serialization(e.to_string()))?;
        Ok(())
    }

    /// Load a model from a file.
    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        let model = GBDTModel::load(&path)
            .map_err(|e| PyBoostersError::Deserialization(e.to_string()))?;

        // Reconstruct params (best effort)
        let params = PyGBDTParams::default();

        Ok(Self { model, params })
    }

    /// Serialize the model to bytes.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self
            .model
            .to_bytes()
            .map_err(|e| PyBoostersError::Serialization(e.to_string()))?;

        Ok(PyBytes::new_bound(py, &bytes))
    }

    /// Deserialize a model from bytes.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        let model = GBDTModel::from_bytes(data)
            .map_err(|e| PyBoostersError::Deserialization(e.to_string()))?;

        let params = PyGBDTParams::default();

        Ok(Self { model, params })
    }

    /// Support for Python pickling.
    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.to_bytes(py)
    }

    /// Support for Python unpickling.
    fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        let loaded = Self::from_bytes(state)?;
        self.model = loaded.model;
        self.params = loaded.params;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "GBDTBooster(n_trees={}, n_features={})",
            self.n_trees(),
            self.n_features()
        )
    }
}

impl PyGBDTBooster {
    fn params_to_trainer_params(params: &PyGBDTParams) -> GBDTParams {
        GBDTParams {
            n_trees: params.n_estimators as u32,
            learning_rate: params.learning_rate,
            gain: GainParams {
                reg_lambda: params.reg_lambda,
                reg_alpha: params.reg_alpha,
                min_child_weight: params.min_child_weight,
                min_gain: params.gamma,
                min_samples_leaf: 1,
            },
            ..Default::default()
        }
    }

    /// Train with the appropriate objective and metric based on the objective string.
    fn train_with_objective(
        dataset: &boosters::data::BinnedDataset,
        labels: &[f32],
        weights: Option<&[f32]>,
        objective_str: &str,
        num_class: Option<usize>,
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
            "reg:quantile" | "quantile" => ObjectiveFunction::Quantile { alpha: 0.5 },
            "reg:pseudohuber" | "huber" => ObjectiveFunction::PseudoHuber { delta: 1.0 },
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

        // Train model
        let model = if let Some(w) = weights {
            GBDTModel::train_with_weights(dataset, labels, w, objective, metric, params)
        } else {
            GBDTModel::train(dataset, labels, objective, metric, params)
        };

        model.ok_or_else(|| PyBoostersError::Training("Training failed".into()).into())
    }
}
