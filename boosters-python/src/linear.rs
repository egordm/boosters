//! GBLinear booster Python bindings.

use crate::dataset::PyDataset;
use crate::error::PyBoostersError;
use crate::params::PyGBLinearParams;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::path::PathBuf;

use boosters::data::{ColMatrix, Dataset as BoostersDataset, DenseMatrix};
use boosters::model::{GBLinearModel, ModelMeta, TaskKind};
use boosters::training::{GBLinearParams, GBLinearTrainer, MetricFunction, ObjectiveFunction};

/// Gradient Boosted Linear model.
///
/// This class provides training and prediction for linear booster models.
///
/// # Example
/// ```python
/// import numpy as np
/// from boosters_python import GBLinearBooster, GBLinearParams, Dataset
///
/// X = np.random.randn(1000, 10).astype(np.float32)
/// y = np.random.randn(1000).astype(np.float32)
///
/// dataset = Dataset(X, y)
/// params = GBLinearParams(n_estimators=100)
///
/// model = GBLinearBooster.train(params, dataset)
/// predictions = model.predict(X)
/// ```
#[pyclass(name = "GBLinearBooster")]
pub struct PyGBLinearBooster {
    model: GBLinearModel,
    params: PyGBLinearParams,
}

#[pymethods]
impl PyGBLinearBooster {
    /// Train a new GBLinear model.
    ///
    /// # Arguments
    /// * `params` - Training parameters
    /// * `train_data` - Training dataset
    #[staticmethod]
    fn train(params: PyGBLinearParams, train_data: PyDataset) -> PyResult<Self> {
        let labels = train_data
            .labels()
            .ok_or_else(|| PyBoostersError::InvalidData("Labels required for training".into()))?;

        // Create ColMatrix from the data (GBLinear requires column-major layout)
        let n_rows = train_data.num_samples();
        let n_cols = train_data.num_features();
        let features = train_data.features().to_vec();
        
        // Convert from row-major to column-major
        let mut col_major_data = vec![0.0f32; n_rows * n_cols];
        for row in 0..n_rows {
            for col in 0..n_cols {
                col_major_data[col * n_rows + row] = features[row * n_cols + col];
            }
        }
        let col_matrix: ColMatrix<f32, Vec<f32>> = DenseMatrix::new(col_major_data, n_rows, n_cols);
        
        // Create boosters Dataset from ColMatrix
        let targets = labels.to_vec();
        let dataset = BoostersDataset::from_numeric(&col_matrix, targets)
            .map_err(|e| PyBoostersError::Training(format!("Failed to create dataset: {}", e)))?;

        // Create training params
        let trainer_params = Self::params_to_trainer_params(&params);

        // Train with appropriate objective based on params
        let (linear_model, n_groups) = Self::train_with_objective(
            &dataset,
            &params.objective,
            trainer_params,
        )?;

        // Create metadata
        let task = match params.objective.as_str() {
            "squared_error" | "reg:squared_error" | "reg:squarederror" => TaskKind::Regression,
            "binary:logistic" | "binary:logitraw" => TaskKind::BinaryClassification,
            _ => TaskKind::Regression,
        };

        let meta = ModelMeta {
            n_features: train_data.num_features(),
            n_groups,
            task,
            feature_names: train_data.get_feature_names(),
            ..Default::default()
        };

        let model = GBLinearModel::from_linear_model(linear_model, meta, params.n_estimators as u32);

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

    /// Compute SHAP values for explaining predictions.
    ///
    /// # Arguments
    /// * `data` - Feature matrix of shape (n_samples, n_features)
    /// * `feature_means` - Mean value for each feature (background distribution)
    ///
    /// # Returns
    /// SHAP values as a NumPy array of shape (n_samples, n_features)
    fn shap_values<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<f32>,
        feature_means: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = data.shape();
        let n_samples = shape[0];
        let n_features = shape[1];
        let features = data.as_slice()?;
        let means = feature_means.as_slice()?.to_vec();

        let shap = self
            .model
            .shap_values(features, n_samples, means)
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
    fn expected_value(&self) -> f64 {
        self.model.bias(0) as f64
    }

    /// Number of features.
    #[getter]
    fn n_features(&self) -> usize {
        self.model.n_features()
    }

    /// Number of output groups.
    #[getter]
    fn n_groups(&self) -> usize {
        self.model.n_groups()
    }

    /// Number of training rounds.
    #[getter]
    fn n_rounds(&self) -> usize {
        self.model.n_rounds() as usize
    }

    /// Get model weights as a NumPy array.
    ///
    /// Returns array of shape (n_features,) for single-output,
    /// or (n_features, n_groups) for multi-output.
    fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let n_features = self.model.n_features();
        let n_groups = self.model.n_groups();

        let mut weights = Vec::with_capacity(n_features * n_groups);
        for group in 0..n_groups {
            for feature in 0..n_features {
                weights.push(self.model.weight(feature, group));
            }
        }

        PyArray1::from_vec_bound(py, weights)
    }

    /// Get model bias as a NumPy array.
    fn bias<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let n_groups = self.model.n_groups();
        let bias: Vec<f32> = (0..n_groups).map(|g| self.model.bias(g)).collect();
        PyArray1::from_vec_bound(py, bias)
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
        let model = GBLinearModel::load(&path)
            .map_err(|e| PyBoostersError::Deserialization(e.to_string()))?;

        let params = PyGBLinearParams::default();

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
        let model = GBLinearModel::from_bytes(data)
            .map_err(|e| PyBoostersError::Deserialization(e.to_string()))?;

        let params = PyGBLinearParams::default();

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
            "GBLinearBooster(n_features={}, n_groups={})",
            self.n_features(),
            self.n_groups()
        )
    }
}

impl PyGBLinearBooster {
    fn params_to_trainer_params(params: &PyGBLinearParams) -> GBLinearParams {
        GBLinearParams {
            n_rounds: params.n_estimators as u32,
            learning_rate: params.learning_rate,
            alpha: params.reg_alpha,
            lambda: params.reg_lambda,
            ..Default::default()
        }
    }

    /// Train with the appropriate objective and metric based on the objective string.
    fn train_with_objective(
        dataset: &boosters::data::Dataset,
        objective_str: &str,
        params: GBLinearParams,
    ) -> PyResult<(boosters::repr::gblinear::LinearModel, usize)> {
        // Parse objective string to ObjectiveFunction enum
        let (objective, n_groups) = match objective_str {
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
                    objective_str
                ))
                .into());
            }
        };

        // Get matching default metric
        let metric = MetricFunction::from_objective_str(objective_str);

        let trainer = GBLinearTrainer::new(objective, metric, params);
        let model = trainer
            .train(dataset, &[])
            .ok_or_else(|| PyBoostersError::Training("Training failed".into()))?;
        
        Ok((model, n_groups))
    }
}
