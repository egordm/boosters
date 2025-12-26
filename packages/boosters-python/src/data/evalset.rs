//! EvalSet type for named evaluation datasets.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use super::PyDataset;

/// Named evaluation set for model training.
///
/// An EvalSet wraps a Dataset with a name, which is used to identify
/// the evaluation set in training logs and `eval_results`.
///
/// # Example
///
/// ```python
/// from boosters import Dataset, EvalSet
/// import numpy as np
///
/// X_val = np.random.rand(50, 10).astype(np.float32)
/// y_val = np.random.rand(50).astype(np.float32)
///
/// val_data = Dataset(X_val, y_val)
/// eval_set = EvalSet("validation", val_data)
///
/// # Use in training:
/// # model.fit(train_data, valid=[eval_set])
/// ```
#[gen_stub_pyclass]
#[pyclass(name = "EvalSet", module = "boosters._boosters_rs")]
pub struct PyEvalSet {
    /// Name of the evaluation set (used in logs and eval_results)
    name: String,

    /// The underlying dataset
    dataset: Py<PyDataset>,
}

impl Clone for PyEvalSet {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            name: self.name.clone(),
            dataset: self.dataset.clone_ref(py),
        })
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyEvalSet {
    /// Create a new named evaluation set.
    ///
    /// Args:
    ///     name: Name for this evaluation set (e.g., "validation", "test")
    ///     dataset: Dataset containing features and labels.
    ///
    /// Returns:
    ///     EvalSet ready for use in training
    #[new]
    pub fn new(dataset: Py<PyDataset>, name: String) -> Self {
        Self { name, dataset }
    }

    /// Name of this evaluation set.
    #[getter]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The underlying dataset.
    #[getter]
    pub fn dataset(&self, py: Python<'_>) -> Py<PyDataset> {
        self.dataset.clone_ref(py)
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let dataset = self.dataset.bind(py);
        let n_samples = dataset.borrow().n_samples();
        format!("EvalSet(name='{}', n_samples={})", self.name, n_samples)
    }
}

impl PyEvalSet {
    /// Get the dataset reference.
    pub fn get_dataset<'py>(&self, py: Python<'py>) -> PyRef<'py, PyDataset> {
        self.dataset.bind(py).borrow()
    }
}
