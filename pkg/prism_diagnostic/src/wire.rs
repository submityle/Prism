//! Minimal self-contained binary codec primitives (`tracy`/`remote` features).
//!
//! M5 needs a serialization layer for the Tracy-compatible emitter and the
//! remote observability protocol, but the diagnostic kernel must stay
//! dependency-light: no `serde`, no `bincode`, no external wire crate. This
//! module hand-rolls a small, explicit little-endian encoder/decoder on top of
//! `alloc::vec::Vec<u8>` so both features share one audited codec.
//!
//! The format is intentionally simple and fully specified:
//! - integers are fixed-width little-endian ([`ByteWriter::put_u32`], ...),
//! - `f64` is encoded via [`f64::to_bits`] (IEEE-754 bit pattern, little-endian),
//! - strings are a `u32` byte length followed by UTF-8 bytes,
//! - booleans are a single `0`/`1` byte.
//!
//! Every decode path is total and returns [`WireError`] rather than panicking,
//! so a malformed or truncated stream can never crash a reader.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

/// A decode error produced while reading a byte stream.
///
/// All variants are recoverable: readers surface them instead of panicking so
/// a corrupt or truncated frame degrades gracefully.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WireError {
    /// The stream ended before the requested bytes were available.
    UnexpectedEof,
    /// A discriminant/tag byte did not match any known variant.
    InvalidTag(u8),
    /// A string field was not valid UTF-8.
    InvalidUtf8,
    /// A field held a value outside its permitted domain.
    InvalidValue,
}

impl core::fmt::Display for WireError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            WireError::UnexpectedEof => f.write_str("unexpected end of input"),
            WireError::InvalidTag(tag) => write!(f, "invalid discriminant tag {tag}"),
            WireError::InvalidUtf8 => f.write_str("invalid UTF-8 in string field"),
            WireError::InvalidValue => f.write_str("field value out of range"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for WireError {}

/// A growable little-endian byte sink.
#[derive(Clone, Debug, Default)]
pub struct ByteWriter {
    buf: Vec<u8>,
}

impl ByteWriter {
    /// Create an empty writer.
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Create a writer preallocating `capacity` bytes.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buf: Vec::with_capacity(capacity),
        }
    }

    /// Number of bytes written so far.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether nothing has been written yet.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Borrow the written bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    /// Consume the writer, returning the written bytes.
    pub fn into_vec(self) -> Vec<u8> {
        self.buf
    }

    /// Append one raw byte.
    pub fn put_u8(&mut self, value: u8) {
        self.buf.push(value);
    }

    /// Append a `u16` in little-endian order.
    pub fn put_u16(&mut self, value: u16) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    /// Append a `u32` in little-endian order.
    pub fn put_u32(&mut self, value: u32) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    /// Append a `u64` in little-endian order.
    pub fn put_u64(&mut self, value: u64) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    /// Append an `i64` in little-endian two's-complement order.
    pub fn put_i64(&mut self, value: i64) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    /// Append an `f64` via its IEEE-754 bit pattern (little-endian).
    pub fn put_f64(&mut self, value: f64) {
        self.buf.extend_from_slice(&value.to_bits().to_le_bytes());
    }

    /// Append a boolean as a single `0`/`1` byte.
    pub fn put_bool(&mut self, value: bool) {
        self.buf.push(u8::from(value));
    }

    /// Append a UTF-8 string as a `u32` byte length followed by its bytes.
    pub fn put_str(&mut self, value: &str) {
        self.put_u32(value.len() as u32);
        self.buf.extend_from_slice(value.as_bytes());
    }

    /// Append raw bytes with no length prefix.
    pub fn put_bytes(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }
}

/// A cursor over a byte slice that decodes little-endian primitives.
#[derive(Clone, Debug)]
pub struct ByteReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    /// Wrap `buf` in a reader positioned at the start.
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Number of bytes not yet consumed.
    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// Whether every byte has been consumed.
    pub fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], WireError> {
        let end = self.pos.checked_add(n).ok_or(WireError::UnexpectedEof)?;
        let slice = self
            .buf
            .get(self.pos..end)
            .ok_or(WireError::UnexpectedEof)?;
        self.pos = end;
        Ok(slice)
    }

    /// Read one raw byte.
    pub fn get_u8(&mut self) -> Result<u8, WireError> {
        Ok(self.take(1)?[0])
    }

    /// Read a little-endian `u16`.
    pub fn get_u16(&mut self) -> Result<u16, WireError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    /// Read a little-endian `u32`.
    pub fn get_u32(&mut self) -> Result<u32, WireError> {
        let bytes = self.take(4)?;
        let mut arr = [0u8; 4];
        arr.copy_from_slice(bytes);
        Ok(u32::from_le_bytes(arr))
    }

    /// Read a little-endian `u64`.
    pub fn get_u64(&mut self) -> Result<u64, WireError> {
        let bytes = self.take(8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(bytes);
        Ok(u64::from_le_bytes(arr))
    }

    /// Read a little-endian `i64`.
    pub fn get_i64(&mut self) -> Result<i64, WireError> {
        let bytes = self.take(8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(bytes);
        Ok(i64::from_le_bytes(arr))
    }

    /// Read an `f64` from its IEEE-754 bit pattern (little-endian).
    pub fn get_f64(&mut self) -> Result<f64, WireError> {
        let bytes = self.take(8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(bytes);
        Ok(f64::from_bits(u64::from_le_bytes(arr)))
    }

    /// Read a boolean encoded as a single `0`/`1` byte.
    pub fn get_bool(&mut self) -> Result<bool, WireError> {
        match self.get_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(WireError::InvalidValue),
        }
    }

    /// Read a length-prefixed UTF-8 string.
    pub fn get_str(&mut self) -> Result<String, WireError> {
        let len = self.get_u32()? as usize;
        let bytes = self.take(len)?;
        core::str::from_utf8(bytes)
            .map(String::from)
            .map_err(|_| WireError::InvalidUtf8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitives_round_trip() {
        let mut w = ByteWriter::new();
        w.put_u8(7);
        w.put_u16(0xBEEF);
        w.put_u32(0xDEAD_BEEF);
        w.put_u64(0x0102_0304_0506_0708);
        w.put_i64(-42);
        w.put_f64(3.5);
        w.put_bool(true);
        w.put_bool(false);
        w.put_str("héllo");
        let bytes = w.into_vec();

        let mut r = ByteReader::new(&bytes);
        assert_eq!(r.get_u8().unwrap(), 7);
        assert_eq!(r.get_u16().unwrap(), 0xBEEF);
        assert_eq!(r.get_u32().unwrap(), 0xDEAD_BEEF);
        assert_eq!(r.get_u64().unwrap(), 0x0102_0304_0506_0708);
        assert_eq!(r.get_i64().unwrap(), -42);
        assert_eq!(r.get_f64().unwrap(), 3.5);
        assert!(r.get_bool().unwrap());
        assert!(!r.get_bool().unwrap());
        assert_eq!(r.get_str().unwrap(), "héllo");
        assert!(r.is_empty());
    }

    #[test]
    fn truncated_input_is_eof_not_panic() {
        let bytes = [1u8, 2, 3];
        let mut r = ByteReader::new(&bytes);
        assert_eq!(r.get_u32(), Err(WireError::UnexpectedEof));
    }

    #[test]
    fn invalid_bool_is_rejected() {
        let bytes = [9u8];
        let mut r = ByteReader::new(&bytes);
        assert_eq!(r.get_bool(), Err(WireError::InvalidValue));
    }

    #[test]
    fn invalid_utf8_is_rejected() {
        let mut w = ByteWriter::new();
        w.put_u32(2);
        w.put_bytes(&[0xFF, 0xFE]);
        let bytes = w.into_vec();
        let mut r = ByteReader::new(&bytes);
        assert_eq!(r.get_str(), Err(WireError::InvalidUtf8));
    }
}
