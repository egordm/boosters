//! Error types for model serialization.

use std::io;

use thiserror::Error;

/// Error reading a `.bstr` model file.
#[derive(Debug, Error)]
pub enum ReadError {
    /// IO error during read.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    /// Invalid magic bytes (not a `.bstr` file).
    #[error("invalid magic bytes: expected BSTR, got {found:?}")]
    InvalidMagic { found: [u8; 4] },

    /// Unsupported schema version.
    #[error("unsupported schema version: {version} (supported: 1..={max_supported})")]
    UnsupportedVersion { version: u32, max_supported: u32 },

    /// Unsupported model type ID.
    #[error("unsupported model type ID: {type_id}")]
    UnsupportedModelType { type_id: u8 },

    /// Unknown model type string (in JSON).
    #[error("unknown model type: {0}")]
    UnknownModelType(String),

    /// Unknown format byte.
    #[error("unknown format byte: 0x{format:02x}")]
    UnknownFormat { format: u8 },

    /// Checksum mismatch (data corruption).
    #[error("checksum mismatch: expected 0x{expected:08x}, got 0x{actual:08x}")]
    ChecksumMismatch { expected: u32, actual: u32 },

    /// Decompression error.
    #[error("decompression error: {0}")]
    Decompression(String),

    /// Deserialization error (MessagePack or JSON).
    #[error("deserialization error: {0}")]
    Deserialize(String),

    /// JSON deserialization error.
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    /// Validation error (schema invariants violated).
    #[error("validation error: {0}")]
    Validation(String),
}

/// Error writing a `.bstr` model file.
#[derive(Debug, Error)]
pub enum WriteError {
    /// IO error during write.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    /// Compression error.
    #[error("compression error: {0}")]
    Compression(String),

    /// Serialization error.
    #[error("serialization error: {0}")]
    Serialize(String),

    /// JSON serialization error.
    #[error("JSON write error: {0}")]
    Json(#[from] serde_json::Error),

    /// Invalid compression level.
    #[error("invalid compression level: {level} (valid: 0-22)")]
    InvalidCompressionLevel { level: u8 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages_are_clear() {
        let err = ReadError::InvalidMagic {
            found: [0x00, 0x01, 0x02, 0x03],
        };
        assert!(err.to_string().contains("BSTR"));

        let err = ReadError::ChecksumMismatch {
            expected: 0xDEADBEEF,
            actual: 0xCAFEBABE,
        };
        assert!(err.to_string().contains("deadbeef"));
        assert!(err.to_string().contains("cafebabe"));

        let err = WriteError::InvalidCompressionLevel { level: 25 };
        assert!(err.to_string().contains("25"));
    }
}
