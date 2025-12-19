//! Native `.bstr` storage format for booste-rs models.
//!
//! This module implements the binary serialization format defined in RFC-0021.
//! The format consists of a 32-byte header followed by a Postcard-encoded payload.
//!
//! # Format Structure
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────┐
//! │                    Header (32 bytes)                        │
//! ├────────────────────────────────────────────────────────────┤
//! │                    Payload (variable)                       │
//! └────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use boosters::io::{NativeCodec, FormatHeader};
//! use boosters::repr::gbdt::Forest;
//!
//! let forest: Forest = /* ... */;
//!
//! // Serialize
//! let codec = NativeCodec::new();
//! let bytes = codec.serialize_forest(&forest)?;
//!
//! // Deserialize
//! let loaded: Forest = codec.deserialize_forest(&bytes)?;
//! ```

use std::io::{Read, Write};

use thiserror::Error;

// ============================================================================
// Constants
// ============================================================================

/// Magic bytes identifying a booste-rs model file.
pub const MAGIC: &[u8; 4] = b"BSTR";

/// Current format version (major).
pub const CURRENT_VERSION_MAJOR: u8 = 1;

/// Current format version (minor).
pub const CURRENT_VERSION_MINOR: u8 = 0;

/// Size of the format header in bytes.
pub const HEADER_SIZE: usize = 32;

/// Minimum payload size for auto-compression (32KB).
#[cfg(feature = "storage-compression")]
pub const COMPRESSION_THRESHOLD: usize = 32 * 1024;

// ============================================================================
// Model Type
// ============================================================================

/// Model type identifier stored in the header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ModelType {
    /// Gradient-boosted decision tree (gbtree).
    Gbdt = 0,
    /// DART (Dropouts meet Multiple Additive Regression Trees).
    Dart = 1,
    /// Gradient-boosted linear model.
    GbLinear = 2,
}

impl ModelType {
    /// Convert from u8, returning None for unknown values.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Gbdt),
            1 => Some(Self::Dart),
            2 => Some(Self::GbLinear),
            _ => None,
        }
    }
}

// ============================================================================
// Format Flags
// ============================================================================

/// Bitfield flags for format features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FormatFlags(u16);

impl FormatFlags {
    /// Payload is compressed with zstd.
    pub const COMPRESSED: u16 = 1 << 0;
    /// Model contains categorical splits.
    pub const HAS_CATEGORICAL: u16 = 1 << 1;
    /// Tree leaves have linear coefficients.
    pub const HAS_LINEAR_LEAVES: u16 = 1 << 2;
    /// Weights are stored as f64 instead of f32.
    pub const DOUBLE_PRECISION: u16 = 1 << 3;

    /// Create empty flags.
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Create flags from raw value.
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Get raw bits.
    pub const fn bits(self) -> u16 {
        self.0
    }

    /// Check if a flag is set.
    pub const fn contains(self, flag: u16) -> bool {
        (self.0 & flag) != 0
    }

    /// Set a flag.
    pub fn set(&mut self, flag: u16) {
        self.0 |= flag;
    }

    /// Clear a flag.
    pub fn clear(&mut self, flag: u16) {
        self.0 &= !flag;
    }
}

// ============================================================================
// Format Header
// ============================================================================

/// 32-byte header for the native storage format.
///
/// # Layout
///
/// ```text
/// Offset  Size  Field
/// ------  ----  -----
/// 0       4     Magic ("BSTR")
/// 4       1     Version major
/// 5       1     Version minor
/// 6       1     Model type
/// 7       1     Reserved (padding)
/// 8       2     Flags (bitfield)
/// 10      2     Reserved
/// 12      4     Payload size (bytes)
/// 16      4     CRC32 checksum of payload
/// 20      4     Number of features
/// 24      4     Number of groups
/// 28      4     Reserved
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatHeader {
    /// Format version (major).
    pub version_major: u8,
    /// Format version (minor).
    pub version_minor: u8,
    /// Model type.
    pub model_type: ModelType,
    /// Feature flags.
    pub flags: FormatFlags,
    /// Size of the payload in bytes.
    pub payload_size: u32,
    /// CRC32 checksum of the payload.
    pub checksum: u32,
    /// Number of input features.
    pub num_features: u32,
    /// Number of output groups.
    pub num_groups: u32,
}

impl FormatHeader {
    /// Create a new header with current version.
    pub fn new(model_type: ModelType, num_features: u32, num_groups: u32) -> Self {
        Self {
            version_major: CURRENT_VERSION_MAJOR,
            version_minor: CURRENT_VERSION_MINOR,
            model_type,
            flags: FormatFlags::empty(),
            payload_size: 0,
            checksum: 0,
            num_features,
            num_groups,
        }
    }

    /// Serialize header to 32 bytes.
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];

        // Magic (offset 0-3)
        buf[0..4].copy_from_slice(MAGIC);

        // Version (offset 4-5)
        buf[4] = self.version_major;
        buf[5] = self.version_minor;

        // Model type (offset 6)
        buf[6] = self.model_type as u8;

        // Reserved (offset 7)
        buf[7] = 0;

        // Flags (offset 8-9, little-endian)
        buf[8..10].copy_from_slice(&self.flags.bits().to_le_bytes());

        // Reserved (offset 10-11)
        buf[10..12].copy_from_slice(&[0, 0]);

        // Payload size (offset 12-15, little-endian)
        buf[12..16].copy_from_slice(&self.payload_size.to_le_bytes());

        // Checksum (offset 16-19, little-endian)
        buf[16..20].copy_from_slice(&self.checksum.to_le_bytes());

        // Num features (offset 20-23, little-endian)
        buf[20..24].copy_from_slice(&self.num_features.to_le_bytes());

        // Num groups (offset 24-27, little-endian)
        buf[24..28].copy_from_slice(&self.num_groups.to_le_bytes());

        // Reserved (offset 28-31)
        buf[28..32].copy_from_slice(&[0, 0, 0, 0]);

        buf
    }

    /// Parse header from 32 bytes.
    pub fn from_bytes(buf: &[u8; HEADER_SIZE]) -> Result<Self, DeserializeError> {
        // Check magic
        if &buf[0..4] != MAGIC {
            return Err(DeserializeError::NotAModel);
        }

        // Version
        let version_major = buf[4];
        let version_minor = buf[5];

        // Check version compatibility
        if version_major > CURRENT_VERSION_MAJOR {
            return Err(DeserializeError::UnsupportedVersion {
                major: version_major,
                minor: version_minor,
            });
        }

        // Model type
        let model_type = ModelType::from_u8(buf[6])
            .ok_or(DeserializeError::CorruptPayload("invalid model type".into()))?;

        // Flags
        let flags = FormatFlags::from_bits(u16::from_le_bytes([buf[8], buf[9]]));

        // Payload size
        let payload_size = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);

        // Checksum
        let checksum = u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]);

        // Num features
        let num_features = u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]);

        // Num groups
        let num_groups = u32::from_le_bytes([buf[24], buf[25], buf[26], buf[27]]);

        Ok(Self {
            version_major,
            version_minor,
            model_type,
            flags,
            payload_size,
            checksum,
            num_features,
            num_groups,
        })
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during serialization.
#[derive(Debug, Error)]
pub enum SerializeError {
    /// I/O error during writing.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Postcard encoding error.
    #[error("encoding error: {0}")]
    Encoding(#[from] postcard::Error),

    /// Compression error.
    #[cfg(feature = "storage-compression")]
    #[error("compression error: {0}")]
    Compression(std::io::Error),
}

/// Errors that can occur during deserialization.
#[derive(Debug, Error)]
pub enum DeserializeError {
    /// File is not a booste-rs model (wrong magic).
    #[error("not a booste-rs model file")]
    NotAModel,

    /// Model requires a newer version of booste-rs.
    #[error("model requires booste-rs {major}.{minor} or later", major = .major, minor = .minor)]
    UnsupportedVersion { major: u8, minor: u8 },

    /// Payload checksum doesn't match.
    #[error("checksum mismatch: expected {expected:#010x}, got {actual:#010x}")]
    ChecksumMismatch { expected: u32, actual: u32 },

    /// File was truncated or incomplete.
    #[error("file truncated: expected {expected} bytes, got {actual}")]
    Truncated { expected: usize, actual: usize },

    /// Payload is corrupt or malformed.
    #[error("corrupt payload: {0}")]
    CorruptPayload(String),

    /// I/O error during reading.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Postcard decoding error.
    #[error("decoding error: {0}")]
    Decoding(#[from] postcard::Error),

    /// Decompression error.
    #[cfg(feature = "storage-compression")]
    #[error("decompression error: {0}")]
    Decompression(std::io::Error),

    /// Payload type mismatch (e.g., expected GBDT but got GBLinear).
    #[error("model type mismatch: expected {expected:?}, got {actual:?}")]
    TypeMismatch { expected: ModelType, actual: ModelType },
}

// ============================================================================
// CRC32 Helper
// ============================================================================

/// Compute CRC32 checksum of data.
pub fn compute_checksum(data: &[u8]) -> u32 {
    crc32fast::hash(data)
}

// ============================================================================
// Native Codec
// ============================================================================

/// Codec for serializing/deserializing models in native format.
#[derive(Debug, Clone)]
pub struct NativeCodec {
    /// Whether to compress payloads.
    #[cfg(feature = "storage-compression")]
    pub compress: bool,

    /// Compression level (1-22, default 3).
    #[cfg(feature = "storage-compression")]
    pub compression_level: i32,
}

impl Default for NativeCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeCodec {
    /// Create a new codec with default settings.
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "storage-compression")]
            compress: true,
            #[cfg(feature = "storage-compression")]
            compression_level: 3,
        }
    }

    /// Create a codec without compression.
    #[cfg(feature = "storage-compression")]
    pub fn without_compression() -> Self {
        Self {
            compress: false,
            compression_level: 0,
        }
    }

    /// Set compression level (1-22).
    #[cfg(feature = "storage-compression")]
    pub fn with_compression_level(mut self, level: i32) -> Self {
        self.compression_level = level.clamp(1, 22);
        self
    }

    /// Write header and payload to a writer.
    pub fn write_to<W: Write>(
        &self,
        writer: &mut W,
        header: &mut FormatHeader,
        payload: &[u8],
    ) -> Result<(), SerializeError> {
        // Optionally compress
        #[cfg(feature = "storage-compression")]
        let (payload_bytes, compressed) = if self.compress && payload.len() >= COMPRESSION_THRESHOLD
        {
            let compressed = zstd::encode_all(payload, self.compression_level)
                .map_err(SerializeError::Compression)?;
            (compressed, true)
        } else {
            (payload.to_vec(), false)
        };

        #[cfg(not(feature = "storage-compression"))]
        let (payload_bytes, compressed) = (payload.to_vec(), false);

        // Update header
        header.payload_size = payload_bytes.len() as u32;
        header.checksum = compute_checksum(&payload_bytes);
        if compressed {
            header.flags.set(FormatFlags::COMPRESSED);
        }

        // Write header
        writer.write_all(&header.to_bytes())?;

        // Write payload
        writer.write_all(&payload_bytes)?;

        Ok(())
    }

    /// Read header and payload from a reader.
    pub fn read_from<R: Read>(&self, reader: &mut R) -> Result<(FormatHeader, Vec<u8>), DeserializeError> {
        // Read header
        let mut header_buf = [0u8; HEADER_SIZE];
        reader.read_exact(&mut header_buf).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                DeserializeError::Truncated {
                    expected: HEADER_SIZE,
                    actual: 0,
                }
            } else {
                DeserializeError::Io(e)
            }
        })?;

        let header = FormatHeader::from_bytes(&header_buf)?;

        // Read payload
        let mut payload = vec![0u8; header.payload_size as usize];
        reader.read_exact(&mut payload).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                DeserializeError::Truncated {
                    expected: header.payload_size as usize,
                    actual: payload.len(),
                }
            } else {
                DeserializeError::Io(e)
            }
        })?;

        // Verify checksum
        let actual_checksum = compute_checksum(&payload);
        if actual_checksum != header.checksum {
            return Err(DeserializeError::ChecksumMismatch {
                expected: header.checksum,
                actual: actual_checksum,
            });
        }

        // Decompress if needed
        #[cfg(feature = "storage-compression")]
        let payload = if header.flags.contains(FormatFlags::COMPRESSED) {
            zstd::decode_all(payload.as_slice()).map_err(DeserializeError::Decompression)?
        } else {
            payload
        };

        #[cfg(not(feature = "storage-compression"))]
        if header.flags.contains(FormatFlags::COMPRESSED) {
            return Err(DeserializeError::CorruptPayload(
                "file is compressed but storage-compression feature is not enabled".into(),
            ));
        }

        Ok((header, payload))
    }

    /// Serialize a payload to bytes with header.
    ///
    /// This is a convenience method that creates a complete serialized model
    /// including the header. For streaming, use `write_to` instead.
    pub fn serialize<T: serde::Serialize>(
        &self,
        model_type: ModelType,
        num_features: u32,
        num_groups: u32,
        payload: &T,
    ) -> Result<Vec<u8>, SerializeError> {
        // Serialize payload with postcard
        let payload_bytes = postcard::to_allocvec(payload)?;

        let mut header = FormatHeader::new(model_type, num_features, num_groups);
        let mut output = Vec::with_capacity(HEADER_SIZE + payload_bytes.len());

        self.write_to(&mut output, &mut header, &payload_bytes)?;
        Ok(output)
    }

    /// Deserialize a payload from bytes.
    ///
    /// This is a convenience method that reads from a byte slice.
    /// For streaming, use `read_from` instead.
    pub fn deserialize<T: for<'de> serde::Deserialize<'de>>(
        &self,
        bytes: &[u8],
    ) -> Result<(FormatHeader, T), DeserializeError> {
        use std::io::Cursor;
        let mut cursor = Cursor::new(bytes);
        let (header, payload_bytes) = self.read_from(&mut cursor)?;
        let payload = postcard::from_bytes(&payload_bytes)?;
        Ok((header, payload))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip() {
        let header = FormatHeader {
            version_major: 1,
            version_minor: 2,
            model_type: ModelType::Gbdt,
            flags: FormatFlags::from_bits(FormatFlags::HAS_CATEGORICAL | FormatFlags::COMPRESSED),
            payload_size: 12345,
            checksum: 0xDEADBEEF,
            num_features: 100,
            num_groups: 3,
        };

        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), HEADER_SIZE);

        let parsed = FormatHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed, header);
    }

    #[test]
    fn header_wrong_magic() {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(b"XXXX");

        let result = FormatHeader::from_bytes(&buf);
        assert!(matches!(result, Err(DeserializeError::NotAModel)));
    }

    #[test]
    fn header_unsupported_version() {
        let mut header = FormatHeader::new(ModelType::Gbdt, 10, 1);
        header.version_major = 99; // Future version
        let bytes = header.to_bytes();

        let result = FormatHeader::from_bytes(&bytes);
        assert!(matches!(
            result,
            Err(DeserializeError::UnsupportedVersion { major: 99, .. })
        ));
    }

    #[test]
    fn checksum_verification() {
        let data = b"hello world";
        let checksum = compute_checksum(data);
        assert_ne!(checksum, 0);

        // Same data should produce same checksum
        assert_eq!(checksum, compute_checksum(data));

        // Different data should produce different checksum
        let different = b"hello worle";
        assert_ne!(checksum, compute_checksum(different));
    }

    #[test]
    fn codec_write_read_roundtrip() {
        let codec = NativeCodec::new();
        let mut header = FormatHeader::new(ModelType::Gbdt, 10, 1);
        let payload = b"test payload data";

        // Write to buffer
        let mut buffer = Vec::new();
        codec.write_to(&mut buffer, &mut header, payload).unwrap();

        // Read back
        let (read_header, read_payload) = codec.read_from(&mut buffer.as_slice()).unwrap();

        assert_eq!(read_header.model_type, ModelType::Gbdt);
        assert_eq!(read_header.num_features, 10);
        assert_eq!(read_header.num_groups, 1);
        assert_eq!(read_payload, payload);
    }

    #[test]
    fn codec_detects_corruption() {
        let codec = NativeCodec::new();
        let mut header = FormatHeader::new(ModelType::GbLinear, 5, 2);
        let payload = b"some model data";

        // Write to buffer
        let mut buffer = Vec::new();
        codec.write_to(&mut buffer, &mut header, payload).unwrap();

        // Corrupt a byte in the payload
        let payload_start = HEADER_SIZE;
        buffer[payload_start + 5] ^= 0xFF;

        // Read should fail with checksum error
        let result = codec.read_from(&mut buffer.as_slice());
        assert!(matches!(result, Err(DeserializeError::ChecksumMismatch { .. })));
    }

    #[test]
    fn model_type_conversion() {
        assert_eq!(ModelType::from_u8(0), Some(ModelType::Gbdt));
        assert_eq!(ModelType::from_u8(1), Some(ModelType::Dart));
        assert_eq!(ModelType::from_u8(2), Some(ModelType::GbLinear));
        assert_eq!(ModelType::from_u8(255), None);
    }

    #[test]
    fn flags_operations() {
        let mut flags = FormatFlags::empty();
        assert!(!flags.contains(FormatFlags::COMPRESSED));

        flags.set(FormatFlags::COMPRESSED);
        assert!(flags.contains(FormatFlags::COMPRESSED));

        flags.set(FormatFlags::HAS_CATEGORICAL);
        assert!(flags.contains(FormatFlags::COMPRESSED));
        assert!(flags.contains(FormatFlags::HAS_CATEGORICAL));

        flags.clear(FormatFlags::COMPRESSED);
        assert!(!flags.contains(FormatFlags::COMPRESSED));
        assert!(flags.contains(FormatFlags::HAS_CATEGORICAL));
    }
}
