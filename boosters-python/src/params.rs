//! Parameter types for Python bindings.

use pyo3::prelude::*;

/// Parameters for GBDT training.
///
/// # Example
/// ```python
/// from boosters_python import GBDTParams
///
/// params = GBDTParams(
///     learning_rate=0.1,
///     max_depth=6,
///     n_estimators=100,
/// )
/// ```
#[pyclass(name = "GBDTParams")]
#[derive(Clone, Debug)]
pub struct PyGBDTParams {
    // Common parameters
    /// Learning rate (shrinkage). Default: 0.3
    #[pyo3(get, set)]
    pub learning_rate: f32,

    /// Number of boosting rounds. Default: 100
    #[pyo3(get, set)]
    pub n_estimators: usize,

    /// Objective function. Default: "squared_error"
    #[pyo3(get, set)]
    pub objective: String,

    // Tree-specific parameters
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

    /// Number of parallel threads. Default: 0 (auto)
    #[pyo3(get, set)]
    pub n_threads: usize,

    /// Random seed. Default: 0
    #[pyo3(get, set)]
    pub seed: u64,

    /// Number of classes (for multiclass). Default: None
    #[pyo3(get, set)]
    pub num_class: Option<usize>,

    /// Base score (initial prediction). Default: 0.5
    #[pyo3(get, set)]
    pub base_score: f32,
}

#[pymethods]
impl PyGBDTParams {
    #[new]
    #[pyo3(signature = (
        learning_rate = 0.3,
        n_estimators = 100,
        objective = "squared_error".to_string(),
        max_depth = 6,
        min_child_weight = 1.0,
        reg_lambda = 1.0,
        reg_alpha = 0.0,
        gamma = 0.0,
        max_bin = 256,
        subsample = 1.0,
        colsample_bytree = 1.0,
        n_threads = 0,
        seed = 0,
        num_class = None,
        base_score = 0.5,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        learning_rate: f32,
        n_estimators: usize,
        objective: String,
        max_depth: usize,
        min_child_weight: f32,
        reg_lambda: f32,
        reg_alpha: f32,
        gamma: f32,
        max_bin: usize,
        subsample: f32,
        colsample_bytree: f32,
        n_threads: usize,
        seed: u64,
        num_class: Option<usize>,
        base_score: f32,
    ) -> Self {
        Self {
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
            n_threads,
            seed,
            num_class,
            base_score,
        }
    }

    /// Create parameters for regression.
    #[staticmethod]
    #[pyo3(signature = (**kwargs))]
    fn for_regression(kwargs: Option<&Bound<'_, pyo3::types::PyDict>>) -> PyResult<Self> {
        let mut params = Self::default();
        params.objective = "squared_error".to_string();

        if let Some(kw) = kwargs {
            params.apply_kwargs(kw)?;
        }

        Ok(params)
    }

    /// Create parameters for binary classification.
    #[staticmethod]
    #[pyo3(signature = (**kwargs))]
    fn for_binary_classification(kwargs: Option<&Bound<'_, pyo3::types::PyDict>>) -> PyResult<Self> {
        let mut params = Self::default();
        params.objective = "binary:logistic".to_string();

        if let Some(kw) = kwargs {
            params.apply_kwargs(kw)?;
        }

        Ok(params)
    }

    /// Create parameters for multiclass classification.
    #[staticmethod]
    #[pyo3(signature = (n_classes, **kwargs))]
    fn for_multiclass(n_classes: usize, kwargs: Option<&Bound<'_, pyo3::types::PyDict>>) -> PyResult<Self> {
        let mut params = Self::default();
        params.objective = "multi:softmax".to_string();
        params.num_class = Some(n_classes);

        if let Some(kw) = kwargs {
            params.apply_kwargs(kw)?;
        }

        Ok(params)
    }

    fn __repr__(&self) -> String {
        format!(
            "GBDTParams(learning_rate={}, n_estimators={}, max_depth={}, objective='{}')",
            self.learning_rate, self.n_estimators, self.max_depth, self.objective
        )
    }
}

impl PyGBDTParams {
    fn apply_kwargs(&mut self, kwargs: &Bound<'_, pyo3::types::PyDict>) -> PyResult<()> {
        for (key, value) in kwargs.iter() {
            let key_str: String = key.extract()?;
            match key_str.as_str() {
                "learning_rate" => self.learning_rate = value.extract()?,
                "n_estimators" => self.n_estimators = value.extract()?,
                "objective" => self.objective = value.extract()?,
                "max_depth" => self.max_depth = value.extract()?,
                "min_child_weight" => self.min_child_weight = value.extract()?,
                "reg_lambda" => self.reg_lambda = value.extract()?,
                "reg_alpha" => self.reg_alpha = value.extract()?,
                "gamma" => self.gamma = value.extract()?,
                "max_bin" => self.max_bin = value.extract()?,
                "subsample" => self.subsample = value.extract()?,
                "colsample_bytree" => self.colsample_bytree = value.extract()?,
                "n_threads" => self.n_threads = value.extract()?,
                "seed" => self.seed = value.extract()?,
                "num_class" => self.num_class = value.extract()?,
                "base_score" => self.base_score = value.extract()?,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Unknown parameter: {}",
                        key_str
                    )))
                }
            }
        }
        Ok(())
    }
}

impl Default for PyGBDTParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.3,
            n_estimators: 100,
            objective: "squared_error".to_string(),
            max_depth: 6,
            min_child_weight: 1.0,
            reg_lambda: 1.0,
            reg_alpha: 0.0,
            gamma: 0.0,
            max_bin: 256,
            subsample: 1.0,
            colsample_bytree: 1.0,
            n_threads: 0,
            seed: 0,
            num_class: None,
            base_score: 0.5,
        }
    }
}

/// Parameters for GBLinear training.
///
/// # Example
/// ```python
/// from boosters_python import GBLinearParams
///
/// params = GBLinearParams(
///     learning_rate=0.5,
///     n_estimators=100,
///     reg_lambda=0.01,
/// )
/// ```
#[pyclass(name = "GBLinearParams")]
#[derive(Clone, Debug)]
pub struct PyGBLinearParams {
    /// Learning rate (shrinkage). Default: 0.5
    #[pyo3(get, set)]
    pub learning_rate: f32,

    /// Number of boosting rounds. Default: 100
    #[pyo3(get, set)]
    pub n_estimators: usize,

    /// Objective function. Default: "squared_error"
    #[pyo3(get, set)]
    pub objective: String,

    /// L2 regularization. Default: 0.0
    #[pyo3(get, set)]
    pub reg_lambda: f32,

    /// L1 regularization. Default: 0.0
    #[pyo3(get, set)]
    pub reg_alpha: f32,

    /// Number of classes (for multiclass). Default: None
    #[pyo3(get, set)]
    pub num_class: Option<usize>,
}

#[pymethods]
impl PyGBLinearParams {
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
            learning_rate,
            n_estimators,
            objective,
            reg_lambda,
            reg_alpha,
            num_class,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GBLinearParams(learning_rate={}, n_estimators={}, objective='{}')",
            self.learning_rate, self.n_estimators, self.objective
        )
    }
}

impl Default for PyGBLinearParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.5,
            n_estimators: 100,
            objective: "squared_error".to_string(),
            reg_lambda: 0.0,
            reg_alpha: 0.0,
            num_class: None,
        }
    }
}
