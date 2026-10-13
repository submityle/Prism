//! LSB-first bit packing (design §15 "delta 压缩 / 位打包").
//!
//! Network deltas spend most of their bytes on small fields — booleans, flags,
//! quantized angles, indices — that waste a whole byte each. [`BitWriter`] /
//! [`BitReader`] pack them at a caller-chosen bit width so a delta pays for the
//! bits a value actually needs. The stream is **LSB-first within each byte** and
//! byte-aligned at the end, which is what a per-field delta writer wants: write a
//! mask bit, then a 10-bit quantized value, then a 5-bit index, with no padding
//! between them.
//!
//! This is a reference packer: `write`/`read` loop per bit, so it is obviously
//! correct and `no_std`-clean. A word-at-a-time packer can replace it later
//! behind the same API if profiling ever wants it.

use alloc::vec::Vec;

/// Packs values into a byte buffer at arbitrary bit widths, LSB-first.
#[derive(Clone, Debug, Default)]
pub struct BitWriter {
    out: Vec<u8>,
    bit_len: usize,
}

impl BitWriter {
    /// A fresh, empty writer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append the low `bits` bits of `value` (`bits <= 64`).
    ///
    /// Bits above `bits` are ignored; `bits == 0` writes nothing. Bits are
    /// appended LSB-first: the least-significant bit of `value` lands at the
    /// current position, the next above it, and so on.
    ///
    /// # Panics
    /// Panics if `bits > 64`.
    pub fn write(&mut self, value: u64, bits: u8) {
        assert!(bits <= 64, "cannot write more than 64 bits");
        for i in 0..bits {
            let bit = (value >> i) & 1;
            let pos = self.bit_len;
            let byte = pos / 8;
            if byte == self.out.len() {
                self.out.push(0);
            }
            self.out[byte] |= (bit as u8) << (pos % 8);
            self.bit_len += 1;
        }
    }

    /// Total bits written so far.
    #[must_use]
    pub const fn bit_len(&self) -> usize {
        self.bit_len
    }

    /// Consume the writer, returning the bytes written (the final byte is
    /// zero-padded in its high bits).
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.out
    }
}

/// Reads values packed by [`BitWriter`], LSB-first.
///
/// The reader is bounded by the **byte** slice it is given, so the final byte's
/// padding bits are readable: a caller decoding a stream must know the schema's
/// exact bit count (or carry an explicit bit-length header) rather than probing
/// with `read` until it fails.
#[derive(Clone, Debug)]
pub struct BitReader<'a> {
    bytes: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    /// A reader over `bytes`.
    #[must_use]
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit_pos: 0 }
    }

    /// Read the next `bits` bits (`bits <= 64`), or `None` if fewer remain.
    ///
    /// # Panics
    /// Panics if `bits > 64`.
    pub fn read(&mut self, bits: u8) -> Option<u64> {
        assert!(bits <= 64, "cannot read more than 64 bits");
        if bits == 0 {
            return Some(0);
        }
        if self.bit_pos + bits as usize > self.bytes.len() * 8 {
            return None;
        }
        let mut value = 0u64;
        for i in 0..bits {
            let pos = self.bit_pos + i as usize;
            let bit = (self.bytes[pos / 8] >> (pos % 8)) & 1;
            value |= (bit as u64) << i;
        }
        self.bit_pos += bits as usize;
        Some(value)
    }

    /// Total bits consumed so far.
    #[must_use]
    pub const fn bits_read(&self) -> usize {
        self.bit_pos
    }

    /// Bits not yet consumed.
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.bytes.len() * 8 - self.bit_pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn mixed_widths_round_trip() {
        let mut writer = BitWriter::new();
        writer.write(1, 1); // bool
        writer.write(0b1011, 4);
        writer.write(0xFFFF_FFFF_FFFF_FFFF, 64); // full width
        writer.write(0x3FF, 10); // quantized angle
        writer.write(0, 0); // no-op
        writer.write(7, 3);
        let bytes = writer.into_bytes();

        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read(1), Some(1));
        assert_eq!(reader.read(4), Some(0b1011));
        assert_eq!(reader.read(64), Some(0xFFFF_FFFF_FFFF_FFFF));
        assert_eq!(reader.read(10), Some(0x3FF));
        assert_eq!(reader.read(0), Some(0));
        assert_eq!(reader.read(3), Some(7));
    }

    #[test]
    fn bits_above_the_width_are_ignored() {
        let mut writer = BitWriter::new();
        writer.write(0xFF, 3); // only the low 3 bits
        assert_eq!(writer.bit_len(), 3, "three bits were written");
        let bytes = writer.into_bytes();
        assert_eq!(bytes, vec![0b111]);
    }

    #[test]
    fn reading_past_the_end_is_none_and_does_not_advance() {
        let mut writer = BitWriter::new();
        writer.write(0b111, 3);
        let bytes = writer.into_bytes(); // one byte: 3 data bits + 5 padding
        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read(9), None, "only 8 bits are stored");
        assert_eq!(reader.bits_read(), 0, "a failed read consumes nothing");
        assert_eq!(reader.remaining(), 8);
        assert_eq!(reader.read(3), Some(0b111));
        assert_eq!(reader.remaining(), 5, "the pad bits remain");
    }
}
