//! Model serialization for native `.bstr` format.
//!
//! This module provides efficient binary and JSON serialization for booste-rs models.
//! The native `.bstr` format offers:
//!
//! - **Faster loading** than XGBoost/LightGBM JSON (2–4× faster)
//! - **Smaller files** with optional zstd compression
//! - **Forward compatibility** via reserved fields and version checks
//! - **Integrity verification** using CRC32C checksums
//!
//! # Format Overview
//!
//! The binary format has three sections:
//! 1. **Header** (32 bytes): Magic, version, format, model type
//! 2. **Payload**: MessagePack-encoded schema (optionally zstd-compressed)
//! 3. **Trailer** (16 bytes): Payload length and CRC32C checksum
//!
//! # Usage
//!
//! ```ignore
//! use boosters::model::GBDTModel;
//! use boosters::persist::{BinaryReadOptions, BinaryWriteOptions, JsonWriteOptions, SerializableModel};
//!
//! // Streaming write into any `std::io::Write`
//! let model = GBDTModel::new(...);
//! let mut buf = Vec::new();
//! model.write_into(&mut buf, &BinaryWriteOptions::default())?;
//!
//! // Streaming read from any `std::io::Read`
//! let loaded = GBDTModel::read_from(buf.as_slice(), &BinaryReadOptions::default())?;
//!
//! // JSON envelope (human-readable)
//! let mut json = Vec::new();
//! model.write_json_into(&mut json, &JsonWriteOptions::pretty())?;
//! let loaded = GBDTModel::read_json_from(json.as_slice())?;
//! ```
//!
//! # Schema Types
//!
//! Schema types (e.g., [`GBDTModelSchema`], [`TreeSchema`]) are separate from
//! runtime types to enable:
//! - Schema evolution without breaking runtime code
//! - Validation during deserialization
//! - Clear migration paths between versions

mod binary;
mod convert;
mod envelope;
mod error;
mod json;
mod schema;

use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, Write};
use std::path::Path;

use crate::model::{GBDTModel, GBLinearModel};

use serde::Serialize;
use serde::de::DeserializeOwned;

// Re-export error types
pub use error::{ReadError, WriteError};

// Re-export envelope types
pub use envelope::{
    BstrHeader, BstrTrailer, FormatByte, HEADER_SIZE, MAGIC, MAX_SUPPORTED_VERSION, ModelTypeId,
    SCHEMA_VERSION, TRAILER_SIZE,
};

// Re-export schema types
pub use schema::{
    CategoriesSchema, FeatureTypeSchema, ForestSchema, GBDTModelSchema, GBLinearModelSchema,
    LeafValuesSchema, LinearCoefficientsSchema, LinearWeightsSchema, ModelMetaSchema, TreeSchema,
};

// Re-export JSON options
pub use json::JsonWriteOptions;

// Re-export binary options
pub use binary::{
    BinaryReadOptions, BinaryWriteOptions, DEFAULT_COMPRESSION_LEVEL, MAX_COMPRESSION_LEVEL,
};

/// Streaming persistence API for concrete model types.
///
/// This is the primary persistence interface: write into any `Write` and read from any `Read`.
pub trait SerializableModel: Sized {
    /// Schema payload used for serialization.
    type Schema: Serialize + DeserializeOwned;

    /// Model discriminant stored in binary header.
    const TYPE_ID: ModelTypeId;

    /// Model type string stored in JSON envelope.
    const TYPE_STR: &'static str;

    /// Convert runtime model into schema payload.
    fn to_schema(&self) -> Self::Schema;

    /// Convert schema payload into runtime model.
    fn from_schema(schema: Self::Schema) -> Result<Self, ReadError>;

    /// Write a binary `.bstr` stream into `writer`.
    fn write_into<W: Write>(
        &self,
        writer: W,
        options: &BinaryWriteOptions,
    ) -> Result<(), WriteError> {
        binary::write_binary_schema(&self.to_schema(), Self::TYPE_ID, writer, options)
    }

    /// Read a binary `.bstr` stream from `reader`.
    fn read_from<R: Read>(reader: R, options: &BinaryReadOptions) -> Result<Self, ReadError> {
        let schema = binary::read_binary_schema(reader, Self::TYPE_ID, options)?;
        Self::from_schema(schema)
    }

    /// Write a JSON `.bstr.json` stream into `writer`.
    fn write_json_into<W: Write>(
        &self,
        writer: W,
        options: &JsonWriteOptions,
    ) -> Result<(), WriteError> {
        json::write_json_envelope(&self.to_schema(), Self::TYPE_STR, writer, options)
    }

    /// Read a JSON `.bstr.json` stream from `reader`.
    fn read_json_from<R: Read>(reader: R) -> Result<Self, ReadError> {
        let env = json::read_json_envelope::<R, Self::Schema>(reader)?;

        if env.bstr_version > MAX_SUPPORTED_VERSION {
            return Err(ReadError::UnsupportedVersion {
                version: env.bstr_version,
                max_supported: MAX_SUPPORTED_VERSION,
            });
        }

        if env.model_type != Self::TYPE_STR {
            return Err(ReadError::UnknownModelType(env.model_type));
        }

        Self::from_schema(env.model)
    }
}

impl SerializableModel for GBDTModel {
    type Schema = GBDTModelSchema;

    const TYPE_ID: ModelTypeId = ModelTypeId::Gbdt;
    const TYPE_STR: &'static str = GBDTModelSchema::MODEL_TYPE;

    fn to_schema(&self) -> Self::Schema {
        GBDTModelSchema::from(self)
    }

    fn from_schema(schema: Self::Schema) -> Result<Self, ReadError> {
        GBDTModel::try_from(schema)
    }
}

impl SerializableModel for GBLinearModel {
    type Schema = GBLinearModelSchema;

    const TYPE_ID: ModelTypeId = ModelTypeId::GbLinear;
    const TYPE_STR: &'static str = GBLinearModelSchema::MODEL_TYPE;

    fn to_schema(&self) -> Self::Schema {
        GBLinearModelSchema::from(self)
    }

    fn from_schema(schema: Self::Schema) -> Result<Self, ReadError> {
        GBLinearModel::try_from(schema)
    }
}

/// Information about a serialized model, obtained without full deserialization.
///
/// This is useful for quick inspection of model files to determine their type,
/// format, and schema version before loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    /// The schema version of the serialized model.
    pub schema_version: u32,
    /// The type of model (GBDT or GBLinear).
    pub model_type: ModelTypeId,
    /// The serialization format (binary or JSON).
    pub format: FormatByte,
    /// Whether the payload is compressed (only relevant for binary format).
    pub is_compressed: bool,
}

impl ModelInfo {
    /// Inspect a binary model file without fully deserializing it.
    ///
    /// This reads only the 32-byte header to determine model metadata.
    pub fn inspect_binary<R: Read>(mut reader: R) -> Result<Self, ReadError> {
        let header = BstrHeader::read_from(&mut reader)?;
        Ok(Self {
            schema_version: header.schema_version,
            model_type: header.model_type,
            format: header.format,
            is_compressed: header.format.is_compressed(),
        })
    }

    /// Inspect a JSON model file without fully deserializing it.
    ///
    /// This parses only the top-level `version` and `model_type` fields.
    pub fn inspect_json<R: Read>(reader: R) -> Result<Self, ReadError> {
        // Use serde_json to parse just the envelope
        #[derive(serde::Deserialize)]
        struct JsonEnvelope {
            bstr_version: u32,
            model_type: String,
        }

        let envelope: JsonEnvelope = serde_json::from_reader(reader)?;

        let model_type = match envelope.model_type.as_str() {
            "gbdt" => ModelTypeId::Gbdt,
            "gblinear" => ModelTypeId::GbLinear,
            other => return Err(ReadError::UnknownModelType(other.to_string())),
        };

        Ok(Self {
            schema_version: envelope.bstr_version,
            model_type,
            format: FormatByte::Json,
            is_compressed: false,
        })
    }

    /// Inspect a model file, auto-detecting the format.
    ///
    /// This reads the first 4 bytes to check for the binary magic number.
    /// If not found, it assumes JSON format.
    pub fn inspect<R: Read + Seek>(mut reader: R) -> Result<Self, ReadError> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;

        // Rewind to start
        reader.seek(std::io::SeekFrom::Start(0))?;

        if magic == MAGIC {
            Self::inspect_binary(reader)
        } else {
            Self::inspect_json(reader)
        }
    }

    /// Inspect a model file by path.
    pub fn inspect_file<P: AsRef<Path>>(path: P) -> Result<Self, ReadError> {
        let file = File::open(path.as_ref())?;
        let reader = BufReader::new(file);
        Self::inspect(reader)
    }
}

/// Polymorphic model enum for loading models when the type is not known at compile time.
#[derive(Debug)]
pub enum Model {
    /// A gradient boosted decision tree model.
    GBDT(GBDTModel),
    /// A linear model.
    GBLinear(GBLinearModel),
}

impl Model {
    /// Load a model from binary format, auto-detecting the model type.
    pub fn load_binary<P: AsRef<Path>>(path: P) -> Result<Self, ReadError> {
        let file = File::open(path.as_ref())?;
        let reader = BufReader::new(file);
        Self::read_binary(reader, &BinaryReadOptions::default())
    }

    /// Read a model from binary format, auto-detecting the model type.
    pub fn read_binary<R: Read>(
        mut reader: R,
        options: &BinaryReadOptions,
    ) -> Result<Self, ReadError> {
        let mut header_bytes = [0u8; HEADER_SIZE];
        reader.read_exact(&mut header_bytes)?;

        let header = BstrHeader::read_from(Cursor::new(header_bytes))?;
        let chained = Cursor::new(header_bytes).chain(reader);

        match header.model_type {
            ModelTypeId::Gbdt => Ok(Model::GBDT(GBDTModel::read_from(chained, options)?)),
            ModelTypeId::GbLinear => {
                Ok(Model::GBLinear(GBLinearModel::read_from(chained, options)?))
            }
            ModelTypeId::Dart => Err(ReadError::UnknownModelType("dart".to_string())),
        }
    }

    /// Load a model from JSON format, auto-detecting the model type.
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self, ReadError> {
        let contents = std::fs::read(path.as_ref())?;
        Self::read_json(contents.as_slice())
    }

    /// Read a model from JSON format, auto-detecting the model type.
    pub fn read_json<R: Read>(mut reader: R) -> Result<Self, ReadError> {
        // Read full content since we need to inspect and then parse
        let mut contents = Vec::new();
        reader.read_to_end(&mut contents)?;

        let info = ModelInfo::inspect_json(Cursor::new(&contents))?;

        match info.model_type {
            ModelTypeId::Gbdt => Ok(Model::GBDT(GBDTModel::read_json_from(Cursor::new(
                &contents,
            ))?)),
            ModelTypeId::GbLinear => Ok(Model::GBLinear(GBLinearModel::read_json_from(
                Cursor::new(&contents),
            )?)),
            _ => Err(ReadError::UnknownModelType(
                info.model_type.as_str().to_string(),
            )),
        }
    }

    /// Load a model from a file, auto-detecting both format and model type.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ReadError> {
        let file = File::open(path.as_ref())?;
        let mut reader = BufReader::new(file);

        // Check for binary magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;

        // Rewind
        reader.seek(std::io::SeekFrom::Start(0))?;

        if magic == MAGIC {
            Self::read_binary(reader, &BinaryReadOptions::default())
        } else {
            // Read to string for JSON
            let mut contents = Vec::new();
            reader.read_to_end(&mut contents)?;
            Self::read_json(contents.as_slice())
        }
    }

    /// Returns the model type.
    pub fn model_type(&self) -> ModelTypeId {
        match self {
            Model::GBDT(_) => ModelTypeId::Gbdt,
            Model::GBLinear(_) => ModelTypeId::GbLinear,
        }
    }

    /// Returns the inner GBDT model, if this is a GBDT model.
    pub fn into_gbdt(self) -> Option<GBDTModel> {
        match self {
            Model::GBDT(m) => Some(m),
            _ => None,
        }
    }

    /// Returns the inner GBLinear model, if this is a GBLinear model.
    pub fn into_gblinear(self) -> Option<GBLinearModel> {
        match self {
            Model::GBLinear(m) => Some(m),
            _ => None,
        }
    }

    /// Returns a reference to the inner GBDT model, if this is a GBDT model.
    pub fn as_gbdt(&self) -> Option<&GBDTModel> {
        match self {
            Model::GBDT(m) => Some(m),
            _ => None,
        }
    }

    /// Returns a reference to the inner GBLinear model, if this is a GBLinear model.
    pub fn as_gblinear(&self) -> Option<&GBLinearModel> {
        match self {
            Model::GBLinear(m) => Some(m),
            _ => None,
        }
    }

    /// Write model to binary format.
    pub fn write_into<W: Write>(
        &self,
        writer: W,
        options: &BinaryWriteOptions,
    ) -> Result<(), WriteError> {
        match self {
            Model::GBDT(m) => m.write_into(writer, options),
            Model::GBLinear(m) => m.write_into(writer, options),
        }
    }

    /// Write model to JSON format.
    pub fn write_json_into<W: Write>(
        &self,
        writer: W,
        options: &JsonWriteOptions,
    ) -> Result<(), WriteError> {
        match self {
            Model::GBDT(m) => m.write_json_into(writer, options),
            Model::GBLinear(m) => m.write_json_into(writer, options),
        }
    }
}

impl From<GBDTModel> for Model {
    fn from(model: GBDTModel) -> Self {
        Model::GBDT(model)
    }
}

impl From<GBLinearModel> for Model {
    fn from(model: GBLinearModel) -> Self {
        Model::GBLinear(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelMeta;
    use crate::model::OutputTransform;
    use crate::repr::gbdt::Forest;
    use crate::repr::gblinear::LinearModel;
    use crate::scalar_tree;
    use ndarray::array;
    use tempfile::tempdir;

    fn create_test_gbdt() -> GBDTModel {
        let tree = scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => leaf(1.0),
            2 => leaf(2.0),
        };
        let mut forest = Forest::for_regression().with_base_score(vec![0.5]);
        forest.push_tree(tree, 0);
        let meta = ModelMeta::for_regression(2);
        GBDTModel::from_parts(forest, meta, OutputTransform::Identity)
    }

    fn create_test_gblinear() -> GBLinearModel {
        // Weights has shape [n_features + 1, n_groups] where last row is bias
        // 2 features + bias = 3 rows, 3 output groups
        let weights = array![
            [0.1, 0.2, 0.3],    // feature 0 weights
            [0.4, 0.5, 0.6],    // feature 1 weights
            [0.01, 0.02, 0.03]  // biases
        ];
        let linear = LinearModel::new(weights);
        let meta = ModelMeta::for_multiclass(2, 3);
        GBLinearModel::from_parts(linear, meta, OutputTransform::Softmax)
    }

    #[test]
    fn model_info_inspect_binary() {
        let model = create_test_gbdt();
        let mut buf = Vec::new();
        model
            .write_into(&mut buf, &BinaryWriteOptions::default())
            .unwrap();

        let info = ModelInfo::inspect_binary(Cursor::new(&buf)).unwrap();
        assert_eq!(info.schema_version, SCHEMA_VERSION);
        assert_eq!(info.model_type, ModelTypeId::Gbdt);
        assert!(info.is_compressed);
    }

    #[test]
    fn model_info_inspect_json() {
        let model = create_test_gbdt();
        let mut buf = Vec::new();
        model
            .write_json_into(&mut buf, &JsonWriteOptions::compact())
            .unwrap();

        let info = ModelInfo::inspect_json(Cursor::new(&buf)).unwrap();
        assert_eq!(info.schema_version, SCHEMA_VERSION);
        assert_eq!(info.model_type, ModelTypeId::Gbdt);
        assert!(!info.is_compressed);
        assert_eq!(info.format, FormatByte::Json);
    }

    #[test]
    fn model_info_inspect_auto_binary() {
        let model = create_test_gbdt();
        let mut buf = Vec::new();
        model
            .write_into(&mut buf, &BinaryWriteOptions::default())
            .unwrap();

        let info = ModelInfo::inspect(Cursor::new(&buf)).unwrap();
        assert_eq!(info.model_type, ModelTypeId::Gbdt);
        assert!(info.format != FormatByte::Json);
    }

    #[test]
    fn model_info_inspect_auto_json() {
        let model = create_test_gbdt();
        let mut buf = Vec::new();
        model
            .write_json_into(&mut buf, &JsonWriteOptions::compact())
            .unwrap();

        let info = ModelInfo::inspect(Cursor::new(&buf)).unwrap();
        assert_eq!(info.model_type, ModelTypeId::Gbdt);
        assert_eq!(info.format, FormatByte::Json);
    }

    #[test]
    fn polymorphic_load_gbdt_binary() {
        let original = create_test_gbdt();
        let mut buf = Vec::new();
        original
            .write_into(&mut buf, &BinaryWriteOptions::default())
            .unwrap();

        let loaded = Model::read_binary(Cursor::new(&buf), &BinaryReadOptions::default()).unwrap();
        assert_eq!(loaded.model_type(), ModelTypeId::Gbdt);
        assert!(loaded.as_gbdt().is_some());
        assert!(loaded.as_gblinear().is_none());
    }

    #[test]
    fn polymorphic_load_gblinear_binary() {
        let original = create_test_gblinear();
        let mut buf = Vec::new();
        original
            .write_into(&mut buf, &BinaryWriteOptions::default())
            .unwrap();

        let loaded = Model::read_binary(Cursor::new(&buf), &BinaryReadOptions::default()).unwrap();
        assert_eq!(loaded.model_type(), ModelTypeId::GbLinear);
        assert!(loaded.as_gblinear().is_some());
        assert!(loaded.as_gbdt().is_none());
    }

    #[test]
    fn polymorphic_load_gbdt_json() {
        let original = create_test_gbdt();
        let mut buf = Vec::new();
        original
            .write_json_into(&mut buf, &JsonWriteOptions::compact())
            .unwrap();

        let loaded = Model::read_json(Cursor::new(&buf)).unwrap();
        assert_eq!(loaded.model_type(), ModelTypeId::Gbdt);
    }

    #[test]
    fn polymorphic_load_gblinear_json() {
        let original = create_test_gblinear();
        let mut buf = Vec::new();
        original
            .write_json_into(&mut buf, &JsonWriteOptions::compact())
            .unwrap();

        let loaded = Model::read_json(Cursor::new(&buf)).unwrap();
        assert_eq!(loaded.model_type(), ModelTypeId::GbLinear);
    }

    #[test]
    fn polymorphic_load_auto_detect_binary() {
        let original = create_test_gbdt();
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.bstr");

        let file = File::create(&path).unwrap();
        let mut writer = std::io::BufWriter::new(file);
        original
            .write_into(&mut writer, &BinaryWriteOptions::default())
            .unwrap();
        writer.flush().unwrap();

        let loaded = Model::load(&path).unwrap();
        assert_eq!(loaded.model_type(), ModelTypeId::Gbdt);
    }

    #[test]
    fn polymorphic_load_auto_detect_json() {
        let original = create_test_gblinear();
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.bstr.json");

        let file = File::create(&path).unwrap();
        let mut writer = std::io::BufWriter::new(file);
        original
            .write_json_into(&mut writer, &JsonWriteOptions::pretty())
            .unwrap();
        writer.flush().unwrap();

        let loaded = Model::load(&path).unwrap();
        assert_eq!(loaded.model_type(), ModelTypeId::GbLinear);
    }

    #[test]
    fn model_enum_conversion() {
        let gbdt = create_test_gbdt();
        let gblinear = create_test_gblinear();

        let m1: Model = gbdt.into();
        let m2: Model = gblinear.into();

        assert!(m1.into_gbdt().is_some());
        assert!(m2.into_gblinear().is_some());
    }

    #[test]
    fn model_enum_save_binary() {
        let gbdt = create_test_gbdt();
        let model: Model = gbdt.into();

        let mut bytes = Vec::new();
        model
            .write_into(&mut bytes, &BinaryWriteOptions::default())
            .unwrap();
        let loaded =
            Model::read_binary(Cursor::new(&bytes), &BinaryReadOptions::default()).unwrap();
        assert_eq!(loaded.model_type(), ModelTypeId::Gbdt);
    }

    #[test]
    fn model_enum_save_json() {
        let gblinear = create_test_gblinear();
        let model: Model = gblinear.into();

        let mut bytes = Vec::new();
        model
            .write_json_into(&mut bytes, &JsonWriteOptions::compact())
            .unwrap();
        let loaded = Model::read_json(Cursor::new(&bytes)).unwrap();
        assert_eq!(loaded.model_type(), ModelTypeId::GbLinear);
    }

    #[test]
    fn model_info_unknown_model_type() {
        let json = r#"{"bstr_version": 1, "model_type": "unknown_type", "model": {}}"#;
        let result = ModelInfo::inspect_json(Cursor::new(json.as_bytes()));
        assert!(matches!(result, Err(ReadError::UnknownModelType(_))));
    }
}
