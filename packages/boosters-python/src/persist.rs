//! Persistence utilities for Python bindings.
//!
//! Provides polymorphic model loading and inspection functions.

use std::io::Cursor;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use boosters::persist::{BinaryReadOptions, FormatByte, Model, ModelInfo as CoreModelInfo};

use crate::error::BoostersError;
use crate::model::{PyGBDTModel, PyGBLinearModel};

// =============================================================================
// ModelInfo
// =============================================================================

/// Information about a serialized model, obtained without full deserialization.
///
/// This is useful for quick inspection of model files to determine their type,
/// format, and schema version before loading the full model.
///
/// Attributes:
///     schema_version: The bstr schema version (e.g., 1).
///     model_type: The type of model ("gbdt" or "gblinear").
///     format: The serialization format ("binary" or "json").
///     is_compressed: Whether the binary payload is compressed (always False for JSON).
///
/// Examples:
///     >>> data = model.to_bytes()
///     >>> info = boosters.Model.inspect_bytes(data)
///     >>> print(f"Model type: {info.model_type}, version: {info.schema_version}")
#[gen_stub_pyclass]
#[pyclass(name = "ModelInfo", module = "boosters._boosters_rs")]
#[derive(Debug, Clone)]
pub struct PyModelInfo {
    /// Schema version (e.g., 1).
    #[pyo3(get)]
    pub schema_version: u32,
    /// Model type string ("gbdt" or "gblinear").
    #[pyo3(get)]
    pub model_type: String,
    /// Format string ("binary" or "json").
    #[pyo3(get)]
    pub format: String,
    /// Whether the payload is compressed (binary only).
    #[pyo3(get)]
    pub is_compressed: bool,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyModelInfo {
    fn __repr__(&self) -> String {
        format!(
            "ModelInfo(schema_version={}, model_type='{}', format='{}', is_compressed={})",
            self.schema_version,
            self.model_type,
            self.format,
            if self.is_compressed { "True" } else { "False" }
        )
    }
}

impl From<CoreModelInfo> for PyModelInfo {
    fn from(info: CoreModelInfo) -> Self {
        Self {
            schema_version: info.schema_version,
            model_type: info.model_type.as_str().to_string(),
            format: match info.format {
                FormatByte::Json => "json".to_string(),
                FormatByte::MessagePack | FormatByte::MessagePackZstd => "binary".to_string(),
            },
            is_compressed: info.is_compressed,
        }
    }
}

// =============================================================================
// Model loader (polymorphic)
// =============================================================================

/// Namespace for polymorphic model loading and inspection.
///
/// Prefer this over a global `loads()` function.
///
/// Examples:
///     >>> data = model.to_bytes()
///     >>> loaded = boosters.Model.load_from_bytes(data)
///     >>> info = boosters.Model.inspect_bytes(data)
#[gen_stub_pyclass]
#[pyclass(name = "Model", module = "boosters._boosters_rs")]
pub struct PyModel;

#[gen_stub_pymethods]
#[pymethods]
impl PyModel {
    /// Load a model from binary or JSON bytes, auto-detecting the format and model type.
    ///
    /// Args:
    ///     data: Binary bytes (.bstr format) or UTF-8 JSON bytes (.bstr.json format).
    ///
    /// Returns:
    ///     The loaded model (either GBDTModel or GBLinearModel).
    ///
    /// Raises:
    ///     ValueError: If the data is invalid, corrupted, or uses an unsupported format.
    #[staticmethod]
    #[pyo3(signature = (data))]
    #[gen_stub(override_return_type(type_repr = "GBDTModel | GBLinearModel"))]
    pub fn load_from_bytes(
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "bytes"))] data: &[u8],
    ) -> PyResult<Py<PyAny>> {
        // Try to detect format from data
        let is_binary = data.len() >= 4 && &data[0..4] == b"BSTR";

        let model = if is_binary {
            Model::read_binary(Cursor::new(data), &BinaryReadOptions::default())
                .map_err(|e| BoostersError::ModelReadError(e.to_string()))?
        } else {
            Model::read_json(Cursor::new(data))
                .map_err(|e| BoostersError::ModelReadError(e.to_string()))?
        };

        // Convert to appropriate Python model type
        match model {
            Model::GBDT(m) => {
                let py_model = PyGBDTModel::from_model(m);
                Ok(Py::new(py, py_model)?.into_any())
            }
            Model::GBLinear(m) => {
                let py_model = PyGBLinearModel::from_model(m);
                Ok(Py::new(py, py_model)?.into_any())
            }
        }
    }

    /// Inspect serialized model bytes without fully deserializing.
    ///
    /// Args:
    ///     data: Binary bytes (.bstr format) or UTF-8 JSON bytes (.bstr.json format).
    ///
    /// Returns:
    ///     ModelInfo object with schema_version, model_type, format, and is_compressed.
    ///
    /// Raises:
    ///     ValueError: If the data is invalid or cannot be inspected.
    #[staticmethod]
    #[pyo3(signature = (data))]
    pub fn inspect_bytes(
        #[gen_stub(override_type(type_repr = "bytes"))] data: &[u8],
    ) -> PyResult<PyModelInfo> {
        // Try to detect format from data
        let is_binary = data.len() >= 4 && &data[0..4] == b"BSTR";

        let info = if is_binary {
            CoreModelInfo::inspect_binary(Cursor::new(data))
                .map_err(|e| BoostersError::ModelReadError(e.to_string()))?
        } else {
            CoreModelInfo::inspect_json(Cursor::new(data))
                .map_err(|e| BoostersError::ModelReadError(e.to_string()))?
        };

        Ok(info.into())
    }
}
