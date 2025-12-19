//! I/O module for model serialization and deserialization.
//!
//! This module provides native storage format support for booste-rs models.
//!
//! # Feature Flags
//!
//! - `storage`: Enables native `.bstr` format support
//! - `storage-compression`: Adds zstd compression support

#[cfg(feature = "storage")]
pub mod convert;

#[cfg(feature = "storage")]
pub mod native;

#[cfg(feature = "storage")]
pub mod payload;

#[cfg(feature = "storage")]
pub use native::{
    DeserializeError, FormatFlags, FormatHeader, ModelType, NativeCodec, SerializeError,
    CURRENT_VERSION_MAJOR, CURRENT_VERSION_MINOR, MAGIC,
};

#[cfg(feature = "storage")]
pub use payload::{
    CategoriesPayload, ForestPayload, GbLinearPayload, GbdtPayload, LinearLeavesPayload,
    ModelMetadata, ModelPayload, Payload, PayloadV1, TreePayload,
};
