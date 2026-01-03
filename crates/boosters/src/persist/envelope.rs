//! Binary envelope structures for `.bstr` format.
//!
//! The binary format uses a fixed header and trailer to support streaming writes:
//! 1. Write 32-byte header
//! 2. Stream payload (MessagePack, optionally zstd-compressed)
//! 3. Write 16-byte trailer with payload length and checksum

use std::io::{Read, Write};

use super::error::{ReadError, WriteError};

/// Magic bytes identifying a `.bstr` file.
pub const MAGIC: [u8; 4] = *b"BSTR";

/// Current schema version.
pub const SCHEMA_VERSION: u32 = 2;

/// Maximum supported schema version for reading.
pub const MAX_SUPPORTED_VERSION: u32 = 2;

/// Header size in bytes.
pub const HEADER_SIZE: usize = 32;

/// Trailer size in bytes.
pub const TRAILER_SIZE: usize = 16;

/// Format byte values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FormatByte {
    /// JSON format (for ModelInfo only, not stored in binary files).
    Json = 0x00,
    /// Uncompressed MessagePack.
    MessagePack = 0x01,
    /// Zstd-compressed MessagePack.
    MessagePackZstd = 0x02,
}

impl FormatByte {
    /// Convert from raw byte.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x00 => Some(Self::Json),
            0x01 => Some(Self::MessagePack),
            0x02 => Some(Self::MessagePackZstd),
            _ => None,
        }
    }

    /// Check if this format uses compression.
    pub fn is_compressed(&self) -> bool {
        matches!(self, Self::MessagePackZstd)
    }
}

/// Model type identifier stored in header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ModelTypeId {
    /// Gradient-boosted decision tree model.
    Gbdt = 1,
    /// Linear gradient boosting model.
    GbLinear = 2,
    /// DART model (reserved for future).
    Dart = 3,
}

impl ModelTypeId {
    /// Convert from raw byte.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            1 => Some(Self::Gbdt),
            2 => Some(Self::GbLinear),
            3 => Some(Self::Dart),
            _ => None,
        }
    }

    /// Convert to JSON string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Gbdt => "gbdt",
            Self::GbLinear => "gblinear",
            Self::Dart => "dart",
        }
    }
}

/// Binary file header (32 bytes).
#[derive(Debug, Clone)]
pub struct BstrHeader {
    /// Schema version number.
    pub schema_version: u32,
    /// Format byte (compression type).
    pub format: FormatByte,
    /// Model type identifier.
    pub model_type: ModelTypeId,
}

impl BstrHeader {
    /// Create a new header with the current schema version.
    pub fn new(format: FormatByte, model_type: ModelTypeId) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            format,
            model_type,
        }
    }

    /// Write header to a writer.
    pub fn write_into<W: Write>(&self, mut writer: W) -> Result<(), WriteError> {
        let mut buf = [0u8; HEADER_SIZE];

        // Magic bytes (offset 0, 4 bytes)
        buf[0..4].copy_from_slice(&MAGIC);

        // Schema version (offset 4, 4 bytes, little-endian)
        buf[4..8].copy_from_slice(&self.schema_version.to_le_bytes());

        // Format byte (offset 8, 1 byte)
        buf[8] = self.format as u8;

        // Model type (offset 9, 1 byte)
        buf[9] = self.model_type as u8;

        // Reserved bytes (offset 10, 22 bytes) - already zero

        writer.write_all(&buf)?;
        Ok(())
    }

    /// Read header from a reader.
    pub fn read_from<R: Read>(mut reader: R) -> Result<Self, ReadError> {
        let mut buf = [0u8; HEADER_SIZE];
        reader.read_exact(&mut buf)?;

        // Check magic bytes
        let magic: [u8; 4] = buf[0..4].try_into().unwrap();
        if magic != MAGIC {
            return Err(ReadError::InvalidMagic { found: magic });
        }

        // Read schema version
        let schema_version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        if schema_version > MAX_SUPPORTED_VERSION {
            return Err(ReadError::UnsupportedVersion {
                version: schema_version,
                max_supported: MAX_SUPPORTED_VERSION,
            });
        }

        // Read format byte
        let format_byte = buf[8];
        let format = FormatByte::from_byte(format_byte).ok_or(ReadError::UnknownFormat {
            format: format_byte,
        })?;

        // Read model type
        let model_type_byte = buf[9];
        let model_type =
            ModelTypeId::from_byte(model_type_byte).ok_or(ReadError::UnsupportedModelType {
                type_id: model_type_byte,
            })?;

        // Reserved bytes are ignored for forward compatibility

        Ok(Self {
            schema_version,
            format,
            model_type,
        })
    }
}

/// Binary file trailer (16 bytes).
#[derive(Debug, Clone)]
pub struct BstrTrailer {
    /// Payload length in bytes.
    pub payload_len: u64,
    /// CRC32C checksum of payload.
    pub checksum: u32,
}

impl BstrTrailer {
    /// Create a new trailer.
    pub fn new(payload_len: u64, checksum: u32) -> Self {
        Self {
            payload_len,
            checksum,
        }
    }

    /// Write trailer to a writer.
    pub fn write_into<W: Write>(&self, mut writer: W) -> Result<(), WriteError> {
        let mut buf = [0u8; TRAILER_SIZE];

        // Payload length (offset 0, 8 bytes, little-endian)
        buf[0..8].copy_from_slice(&self.payload_len.to_le_bytes());

        // Checksum (offset 8, 4 bytes, little-endian)
        buf[8..12].copy_from_slice(&self.checksum.to_le_bytes());

        // Reserved bytes (offset 12, 4 bytes) - already zero

        writer.write_all(&buf)?;
        Ok(())
    }

    /// Read trailer from a byte slice (last 16 bytes of file).
    pub fn from_bytes(bytes: &[u8; TRAILER_SIZE]) -> Self {
        let payload_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let checksum = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        // Reserved bytes ignored

        Self {
            payload_len,
            checksum,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trip() {
        let header = BstrHeader::new(FormatByte::MessagePackZstd, ModelTypeId::Gbdt);

        let mut buf = Vec::new();
        header.write_into(&mut buf).unwrap();
        assert_eq!(buf.len(), HEADER_SIZE);

        let restored = BstrHeader::read_from(&buf[..]).unwrap();
        assert_eq!(restored.schema_version, SCHEMA_VERSION);
        assert_eq!(restored.format, FormatByte::MessagePackZstd);
        assert_eq!(restored.model_type, ModelTypeId::Gbdt);
    }

    #[test]
    fn header_invalid_magic() {
        let buf = [0u8; HEADER_SIZE]; // All zeros, invalid magic
        let err = BstrHeader::read_from(&buf[..]).unwrap_err();
        assert!(matches!(err, ReadError::InvalidMagic { .. }));
    }

    #[test]
    fn header_future_version() {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&MAGIC);
        buf[4..8].copy_from_slice(&99u32.to_le_bytes()); // Future version

        let err = BstrHeader::read_from(&buf[..]).unwrap_err();
        assert!(matches!(
            err,
            ReadError::UnsupportedVersion { version: 99, .. }
        ));
    }

    #[test]
    fn header_unknown_format() {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&MAGIC);
        buf[4..8].copy_from_slice(&1u32.to_le_bytes());
        buf[8] = 0xFF; // Unknown format

        let err = BstrHeader::read_from(&buf[..]).unwrap_err();
        assert!(matches!(err, ReadError::UnknownFormat { format: 0xFF }));
    }

    #[test]
    fn trailer_round_trip() {
        let trailer = BstrTrailer::new(12345, 0xDEADBEEF);

        let mut buf = Vec::new();
        trailer.write_into(&mut buf).unwrap();
        assert_eq!(buf.len(), TRAILER_SIZE);

        let bytes: [u8; TRAILER_SIZE] = buf.try_into().unwrap();
        let restored = BstrTrailer::from_bytes(&bytes);
        assert_eq!(restored.payload_len, 12345);
        assert_eq!(restored.checksum, 0xDEADBEEF);
    }

    #[test]
    fn format_byte_compression() {
        assert!(!FormatByte::MessagePack.is_compressed());
        assert!(FormatByte::MessagePackZstd.is_compressed());
    }

    #[test]
    fn model_type_strings() {
        assert_eq!(ModelTypeId::Gbdt.as_str(), "gbdt");
        assert_eq!(ModelTypeId::GbLinear.as_str(), "gblinear");
        assert_eq!(ModelTypeId::Dart.as_str(), "dart");
    }
}
