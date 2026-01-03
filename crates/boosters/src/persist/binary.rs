//! Binary serialization format (MessagePack with zstd compression).
//!
//! The binary `.bstr` format consists of:
//! 1. **Header** (32 bytes): Magic, version, format, model type
//! 2. **Payload**: MessagePack-encoded schema (optionally zstd-compressed)
//! 3. **Trailer** (16 bytes): Payload length and CRC32C checksum

use std::io::{Read, Write};

use super::envelope::{BstrHeader, BstrTrailer, FormatByte, ModelTypeId, TRAILER_SIZE};
use super::error::{ReadError, WriteError};
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Default zstd compression level (balance of speed and ratio).
pub const DEFAULT_COMPRESSION_LEVEL: u8 = 3;

/// Maximum valid zstd compression level.
pub const MAX_COMPRESSION_LEVEL: u8 = 22;

/// Options for writing binary `.bstr` files.
#[derive(Debug, Clone, Copy)]
pub struct BinaryWriteOptions {
    /// Compression level: 0 = uncompressed MessagePack, 1-22 = zstd levels.
    pub compression_level: u8,
}

impl Default for BinaryWriteOptions {
    fn default() -> Self {
        Self {
            compression_level: DEFAULT_COMPRESSION_LEVEL,
        }
    }
}

impl BinaryWriteOptions {
    /// Uncompressed MessagePack.
    pub fn uncompressed() -> Self {
        Self {
            compression_level: 0,
        }
    }
}

/// Options for reading binary `.bstr` files.
#[derive(Debug, Clone, Copy, Default)]
pub struct BinaryReadOptions {
    /// Skip checksum verification (for benchmarking only).
    pub skip_checksum: bool,
}

pub(crate) fn write_binary_schema<W: Write, S: Serialize>(
    schema: &S,
    model_type: ModelTypeId,
    mut writer: W,
    opts: &BinaryWriteOptions,
) -> Result<(), WriteError> {
    if opts.compression_level > MAX_COMPRESSION_LEVEL {
        return Err(WriteError::InvalidCompressionLevel {
            level: opts.compression_level,
        });
    }

    let format = if opts.compression_level == 0 {
        FormatByte::MessagePack
    } else {
        FormatByte::MessagePackZstd
    };

    // 1) Header
    let header = BstrHeader::new(format, model_type);
    header.write_into(&mut writer)?;

    // 2) Payload (compute checksum + payload length as written)
    let crc_writer = Crc32cWriter::new(writer);

    let crc_writer = if opts.compression_level == 0 {
        let mut payload_writer = crc_writer;
        rmp_serde::encode::write_named(&mut payload_writer, schema)
            .map_err(|e| WriteError::Serialize(e.to_string()))?;
        payload_writer
    } else {
        let mut encoder =
            zstd::stream::write::Encoder::new(crc_writer, i32::from(opts.compression_level))
                .map_err(|e| WriteError::Compression(e.to_string()))?;
        rmp_serde::encode::write_named(&mut encoder, schema)
            .map_err(|e| WriteError::Serialize(e.to_string()))?;
        encoder
            .finish()
            .map_err(|e| WriteError::Compression(e.to_string()))?
    };

    let (mut writer, payload_len, checksum) = crc_writer.into_parts();

    // 3) Trailer
    let trailer = BstrTrailer::new(payload_len, checksum);
    trailer.write_into(&mut writer)?;
    Ok(())
}

pub(crate) fn read_binary_schema<R: Read, S: DeserializeOwned>(
    mut reader: R,
    expected_model_type: ModelTypeId,
    opts: &BinaryReadOptions,
) -> Result<S, ReadError> {
    let header = BstrHeader::read_from(&mut reader)?;

    if header.format == FormatByte::Json {
        return Err(ReadError::Validation(
            "binary header format 0x00 (Json) is not valid for .bstr".into(),
        ));
    }

    if header.model_type != expected_model_type {
        return Err(ReadError::UnsupportedModelType {
            type_id: header.model_type as u8,
        });
    }

    let (payload, trailer) = read_payload_and_trailer(reader)?;

    if payload.len() as u64 != trailer.payload_len {
        return Err(ReadError::Validation(format!(
            "payload length mismatch: trailer says {}, actual {}",
            trailer.payload_len,
            payload.len()
        )));
    }

    if !opts.skip_checksum {
        let actual_checksum = crc32c::crc32c(&payload);
        if actual_checksum != trailer.checksum {
            return Err(ReadError::ChecksumMismatch {
                expected: trailer.checksum,
                actual: actual_checksum,
            });
        }
    }

    let decompressed = if header.format.is_compressed() {
        zstd::decode_all(payload.as_slice()).map_err(|e| ReadError::Decompression(e.to_string()))?
    } else {
        payload
    };

    rmp_serde::from_slice(&decompressed).map_err(|e| ReadError::Deserialize(e.to_string()))
}

fn read_payload_and_trailer<R: Read>(mut reader: R) -> Result<(Vec<u8>, BstrTrailer), ReadError> {
    let mut payload = Vec::new();
    let mut tail = Vec::with_capacity(TRAILER_SIZE);

    let mut buf = [0u8; 8192];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }

        tail.extend_from_slice(&buf[..n]);
        if tail.len() > TRAILER_SIZE {
            let split = tail.len() - TRAILER_SIZE;
            payload.extend_from_slice(&tail[..split]);
            let keep = tail[split..].to_vec();
            tail.clear();
            tail.extend_from_slice(&keep);
        }
    }

    if tail.len() != TRAILER_SIZE {
        return Err(ReadError::Validation("file too small".into()));
    }

    let trailer_bytes: [u8; TRAILER_SIZE] = tail
        .as_slice()
        .try_into()
        .expect("tail length already checked");
    let trailer = BstrTrailer::from_bytes(&trailer_bytes);

    if payload.is_empty() && trailer.payload_len != 0 {
        // Trailer says there's payload, but we never accumulated any bytes.
        // This only happens on truncated/invalid inputs.
        return Err(ReadError::Validation("missing payload".into()));
    }

    Ok((payload, trailer))
}

struct Crc32cWriter<W> {
    inner: W,
    payload_len: u64,
    checksum: u32,
}

impl<W> Crc32cWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            payload_len: 0,
            checksum: 0,
        }
    }

    fn into_parts(self) -> (W, u64, u32) {
        (self.inner, self.payload_len, self.checksum)
    }
}

impl<W: Write> Write for Crc32cWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.payload_len += n as u64;

        // Update checksum with bytes that were actually written.
        self.checksum = crc32c::crc32c_append(self.checksum, &buf[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}
