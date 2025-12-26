//! EvalSet type for named evaluation datasets.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use super::PyDataset;
use crate::error::BoostersError;

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
        Python::with_gil(|py| Self {
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
    ///     dataset: Dataset containing features and labels. Accepts either
    ///         the Rust Dataset from `_boosters_rs` or the Python wrapper.
    ///
    /// Returns:
    ///     EvalSet ready for use in training
    #[new]
    pub fn new(py: Python<'_>, name: String, dataset: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = extract_dataset(py, dataset)?;
        Ok(Self { name, dataset: inner })
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

/// Extract PyDataset from various Python input types.
///
/// Accepts:
/// - Direct `PyDataset` (Rust type from `_boosters_rs.Dataset`)
/// - Python wrapper with `_inner` attribute (from `boosters.data.Dataset`)
pub fn extract_dataset(_py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Py<PyDataset>> {
    // Try direct extraction first (Rust PyDataset)
    if let Ok(dataset) = obj.extract::<Py<PyDataset>>() {
        return Ok(dataset);
    }

    // Try extracting from _inner attribute (Python wrapper)
    if let Ok(inner) = obj.getattr("_inner") {
        if let Ok(dataset) = inner.extract::<Py<PyDataset>>() {
            return Ok(dataset);
        }
    }

    // Not a valid dataset type
    let type_name = obj
        .get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    Err(BoostersError::TypeError(format!(
        "expected Dataset, got {}",
        type_name
    ))
    .into())
}
