//! JSON serialization format.
//!
//! Provides human-readable JSON serialization for models.
//! JSON format is useful for debugging and interoperability.

use std::io::{Read, Write};

use super::envelope::SCHEMA_VERSION;
use super::error::{ReadError, WriteError};
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Write options for JSON serialization.
#[derive(Debug, Clone, Default)]
pub struct JsonWriteOptions {
    /// Pretty-print with indentation.
    pub pretty: bool,
}

impl JsonWriteOptions {
    /// Create options for compact JSON.
    pub fn compact() -> Self {
        Self { pretty: false }
    }

    /// Create options for pretty-printed JSON.
    pub fn pretty() -> Self {
        Self { pretty: true }
    }
}

pub(crate) fn write_json_schema<W: Write, S: Serialize>(
    schema: &S,
    writer: W,
    options: &JsonWriteOptions,
) -> Result<(), WriteError> {
    if options.pretty {
        serde_json::to_writer_pretty(writer, schema)?;
    } else {
        serde_json::to_writer(writer, schema)?;
    }
    Ok(())
}

pub(crate) fn read_json_schema<R: Read, S: DeserializeOwned>(reader: R) -> Result<S, ReadError> {
    Ok(serde_json::from_reader(reader)?)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct JsonEnvelope<T> {
    pub bstr_version: u32,
    pub model_type: String,
    pub model: T,
}

pub(crate) fn write_json_envelope<W: Write, S: Serialize>(
    schema: &S,
    model_type: &str,
    writer: W,
    options: &JsonWriteOptions,
) -> Result<(), WriteError> {
    let env = JsonEnvelope {
        bstr_version: SCHEMA_VERSION,
        model_type: model_type.to_string(),
        model: schema,
    };

    write_json_schema(&env, writer, options)
}

pub(crate) fn read_json_envelope<R: Read, S: DeserializeOwned>(
    reader: R,
) -> Result<JsonEnvelope<S>, ReadError> {
    read_json_schema(reader)
}
