//! `Golomb`-`Rice` integer coding: the geometric-distribution-optimal, bit-level
//! entropy coder that splits each unsigned integer into a *quotient* written in
//! `unary` and a *remainder* written in a fixed/`truncated-binary` field (design
//! §27 attribute compression, residual and delta-stream coding).
//!
//! A value `v` is coded against a parameter `M` as `q = v / M` followed by
//! `r = v % M`. The quotient `q` is emitted in `unary`; the remainder `r` is
//! emitted in a short binary field whose width tracks `M`. `Rice` coding is the
//! special case `M = 2^k`, where the remainder is exactly `k` plain bits and the
//! division/modulo collapse into shifts and masks. This is the *quotient-unary +
//! remainder-binary* member of the integer-coding family and is deliberately
//! distinct from its neighbors:
//!
//! * [`super::elias_gamma_delta`] is a *logarithmic length-prefix* family
//!   (`Elias` gamma/delta/omega): it prefixes each number with an encoding of its
//!   own bit length. `Golomb`-`Rice` has no length prefix; it spends a `unary`
//!   budget proportional to `v / M` and a fixed-ish remainder, which is optimal
//!   for geometrically distributed inputs rather than for arbitrary magnitudes.
//! * [`super::varint_leb128`] is *byte-granular*: one byte per 7 payload bits
//!   with a continuation flag. `Golomb`-`Rice` is *bit-granular* and carries no
//!   continuation bits.
//! * [`super::run_length_encode`] coalesces *repeats* into (value, count) pairs.
//!   `Golomb`-`Rice` never looks at repetition; it codes each value's magnitude
//!   independently.
//!
//! # Conventions (fixed here)
//!
//! * **Bit order is `MSB`-first.** [`BitWriter`] lays the first pushed bit into
//!   the most-significant bit of byte 0.
//! * **`unary(q)` is `q` one-bits followed by a single terminating zero-bit.**
//!   So `q = 0` is `"0"`, `q = 1` is `"10"`, `q = 3` is `"1110"`.
//! * **Quotient/remainder split is `q = v / M`, `r = v % M`.** For `Rice`,
//!   `q = v >> k` and `r = v & (M - 1)` with `M = 2^k`.
//! * **`truncated-binary`** for an alphabet of size `M` with
//!   `b = ceil(log2(M))` and `cutoff = 2^b - M`: the first `cutoff` symbols
//!   `0..cutoff` use `b - 1` bits, and the remaining symbols `cutoff..M` use `b`
//!   bits coded as `r + cutoff`. For `Rice` (`M` a power of two) `cutoff == 0`,
//!   so every remainder uses the full `k` bits.
//!
//! All `ceil(log2(x))` values are computed with the bit trick
//! `32 - (x - 1).leading_zeros()` — this module never calls any floating-point
//! `log`, `pow`, or `ceil` routine.

use alloc::vec::Vec;

/// `ceil(log2(m))` for `m >= 1`, computed purely with `leading_zeros`.
///
/// Returns the number of bits `b` such that `2^(b-1) < m <= 2^b`. For `m == 1`
/// this is `0`; for `m == 2` it is `1`; for any `m` in `5..=8` it is `3`. No
/// floating-point `log2` is used: `(m - 1).leading_zeros()` counts the leading
/// zero bits of `m - 1`, and the complement to 32 is the needed width.
#[must_use]
pub fn ceil_log2(m: u32) -> u32 {
    debug_assert!(m >= 1, "ceil_log2 is defined for m >= 1");
    32 - (m - 1).leading_zeros()
}

/// `MSB`-first bit sink backed by a `Vec<u8>`.
///
/// Bits are appended with [`BitWriter::push_bit`]/[`BitWriter::push_bits`]; the
/// first bit pushed occupies the most-significant bit of the first byte. The
/// final partial byte is zero-padded in its low bits by [`BitWriter::into_bytes`].
#[derive(Clone, Debug, Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    nbits: usize,
}

impl BitWriter {
    /// Creates an empty writer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bytes: Vec::new(),
            nbits: 0,
        }
    }

    /// Number of bits written so far.
    #[must_use]
    pub fn bit_len(&self) -> usize {
        self.nbits
    }

    /// Appends a single bit (`MSB`-first within each byte).
    pub fn push_bit(&mut self, bit: bool) {
        let byte_index = self.nbits >> 3;
        let bit_index = (self.nbits & 7) as u32;
        if byte_index == self.bytes.len() {
            self.bytes.push(0);
        }
        if bit {
            self.bytes[byte_index] |= 1u8 << (7 - bit_index);
        }
        self.nbits += 1;
    }

    /// Appends the low `count` bits of `value`, most-significant bit first.
    ///
    /// `count == 0` writes nothing. `count` must not exceed 32.
    pub fn push_bits(&mut self, value: u32, count: u32) {
        debug_assert!(count <= 32, "cannot push more than 32 bits from a u32");
        for i in (0..count).rev() {
            self.push_bit((value >> i) & 1 == 1);
        }
    }

    /// Consumes the writer and returns the packed bytes (last byte zero-padded).
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// `MSB`-first bit source reading back from a byte slice.
///
/// Mirrors [`BitWriter`]: the first bit read is the most-significant bit of the
/// first byte. Reads past the end of the slice return [`None`].
#[derive(Clone, Debug)]
pub struct BitReader<'a> {
    bytes: &'a [u8],
    nbits: usize,
    pos: usize,
}

impl<'a> BitReader<'a> {
    /// Creates a reader over `bytes` (treating all `bytes.len() * 8` bits as
    /// available).
    #[must_use]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            nbits: bytes.len() * 8,
            pos: 0,
        }
    }

    /// Number of unread bits remaining.
    #[must_use]
    pub fn bits_remaining(&self) -> usize {
        self.nbits - self.pos
    }

    /// Reads the next bit, or [`None`] if the stream is exhausted.
    pub fn read_bit(&mut self) -> Option<bool> {
        if self.pos >= self.nbits {
            return None;
        }
        let byte = self.bytes[self.pos >> 3];
        let bit_index = (self.pos & 7) as u32;
        self.pos += 1;
        Some((byte >> (7 - bit_index)) & 1 == 1)
    }

    /// Reads `count` bits (most-significant bit first) into a `u32`.
    ///
    /// `count == 0` yields `Some(0)` without consuming any bits. Returns
    /// [`None`] if fewer than `count` bits remain.
    pub fn read_bits(&mut self, count: u32) -> Option<u32> {
        let mut value = 0u32;
        for _ in 0..count {
            let bit = self.read_bit()?;
            value = (value << 1) | u32::from(bit);
        }
        Some(value)
    }
}

/// Writes `q` as `unary`: `q` one-bits then one terminating zero-bit.
fn write_unary(q: u32, bits: &mut BitWriter) {
    for _ in 0..q {
        bits.push_bit(true);
    }
    bits.push_bit(false);
}

/// Reads a `unary` codeword, returning the count of leading one-bits.
fn read_unary(reader: &mut BitReader) -> Option<u32> {
    let mut q = 0u32;
    loop {
        if reader.read_bit()? {
            q += 1;
        } else {
            return Some(q);
        }
    }
}

/// Writes `r` (`r` in `0..m`) in `truncated-binary` for alphabet size `m`.
fn write_truncated_binary(r: u32, m: u32, bits: &mut BitWriter) {
    debug_assert!(r < m, "remainder must be < m");
    let b = ceil_log2(m);
    if b == 0 {
        return;
    }
    let cutoff = ((1u64 << b) - u64::from(m)) as u32;
    if r < cutoff {
        bits.push_bits(r, b - 1);
    } else {
        bits.push_bits(r + cutoff, b);
    }
}

/// Reads a `truncated-binary` symbol for alphabet size `m`.
fn read_truncated_binary(m: u32, reader: &mut BitReader) -> Option<u32> {
    let b = ceil_log2(m);
    if b == 0 {
        return Some(0);
    }
    let cutoff = ((1u64 << b) - u64::from(m)) as u32;
    let x = reader.read_bits(b - 1)?;
    if x < cutoff {
        Some(x)
    } else {
        let extra = u32::from(reader.read_bit()?);
        Some(((x << 1) | extra) - cutoff)
    }
}

/// `Rice`-encodes `value` with parameter `k` (so `M = 2^k`) into `bits`.
///
/// Emits `unary(value >> k)` followed by the low `k` bits of `value`. `k` must
/// be in `0..=31`.
pub fn rice_encode(value: u32, k: u32, bits: &mut BitWriter) {
    debug_assert!(k <= 31, "Rice parameter k must be in 0..=31");
    let q = value >> k;
    write_unary(q, bits);
    let r = value & ((1u32 << k) - 1);
    bits.push_bits(r, k);
}

/// Decodes one `Rice`-coded value with parameter `k` from `reader`.
pub fn rice_decode(k: u32, reader: &mut BitReader) -> Option<u32> {
    debug_assert!(k <= 31, "Rice parameter k must be in 0..=31");
    let q = read_unary(reader)?;
    let r = reader.read_bits(k)?;
    Some((q << k) | r)
}

/// `Golomb`-encodes `value` with modulus `m` (any `m >= 1`) into `bits`.
///
/// Emits `unary(value / m)` followed by `value % m` in `truncated-binary`.
pub fn golomb_encode(value: u32, m: u32, bits: &mut BitWriter) {
    debug_assert!(m >= 1, "Golomb modulus m must be >= 1");
    let q = value / m;
    let r = value % m;
    write_unary(q, bits);
    write_truncated_binary(r, m, bits);
}

/// Decodes one `Golomb`-coded value with modulus `m` from `reader`.
pub fn golomb_decode(m: u32, reader: &mut BitReader) -> Option<u32> {
    debug_assert!(m >= 1, "Golomb modulus m must be >= 1");
    let q = read_unary(reader)?;
    let r = read_truncated_binary(m, reader)?;
    Some(q * m + r)
}

/// `Rice`-encodes every value in `values` with parameter `k`, concatenated
/// into one `MSB`-first bit stream, returning the packed bytes.
#[must_use]
pub fn rice_encode_all(values: &[u32], k: u32) -> Vec<u8> {
    let mut writer = BitWriter::new();
    for &value in values {
        rice_encode(value, k, &mut writer);
    }
    writer.into_bytes()
}

/// Decodes exactly `count` `Rice`-coded values with parameter `k` from `bytes`.
#[must_use]
pub fn rice_decode_all(bytes: &[u8], count: usize, k: u32) -> Vec<u32> {
    let mut reader = BitReader::new(bytes);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(rice_decode(k, &mut reader).expect("stream exhausted before count"));
    }
    out
}

/// `Golomb`-encodes every value in `values` with modulus `m`, concatenated
/// into one `MSB`-first bit stream, returning the packed bytes.
#[must_use]
pub fn golomb_encode_all(values: &[u32], m: u32) -> Vec<u8> {
    let mut writer = BitWriter::new();
    for &value in values {
        golomb_encode(value, m, &mut writer);
    }
    writer.into_bytes()
}

/// Decodes exactly `count` `Golomb`-coded values with modulus `m` from `bytes`.
#[must_use]
pub fn golomb_decode_all(bytes: &[u8], count: usize, m: u32) -> Vec<u32> {
    let mut reader = BitReader::new(bytes);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(golomb_decode(m, &mut reader).expect("stream exhausted before count"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Renders the whole content of a writer as an `MSB`-first bit string.
    #[cfg(test)]
    fn encoded_bits(writer: BitWriter) -> String {
        let n = writer.bit_len();
        let bytes = writer.into_bytes();
        let mut s = String::new();
        for i in 0..n {
            let byte = bytes[i >> 3];
            let bit = (byte >> (7 - (i & 7))) & 1;
            s.push(if bit == 1 { '1' } else { '0' });
        }
        s
    }

    /// Encodes a single `Rice` value and returns its bit string.
    #[cfg(test)]
    fn rice_bits(value: u32, k: u32) -> String {
        let mut w = BitWriter::new();
        rice_encode(value, k, &mut w);
        encoded_bits(w)
    }

    /// Encodes a single `Golomb` value and returns its bit string.
    #[cfg(test)]
    fn golomb_bits(value: u32, m: u32) -> String {
        let mut w = BitWriter::new();
        golomb_encode(value, m, &mut w);
        encoded_bits(w)
    }

    /// Small deterministic `LCG` for pseudo-random round-trip fuzzing.
    #[cfg(test)]
    fn lcg(state: &mut u64) -> u32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as u32
    }

    // ---- Rice reference bit strings (k = 2) -------------------------------

    #[test]
    fn rice_k2_value0() {
        assert_eq!(rice_bits(0, 2), "000");
    }

    #[test]
    fn rice_k2_value3() {
        assert_eq!(rice_bits(3, 2), "011");
    }

    #[test]
    fn rice_k2_value4() {
        assert_eq!(rice_bits(4, 2), "1000");
    }

    #[test]
    fn rice_k2_value7() {
        // q = 7 >> 2 = 1 -> "10"; r = 7 & 3 = 3 -> "11"; code = "1011".
        assert_eq!(rice_bits(7, 2), "1011");
    }

    #[test]
    fn rice_k2_value8() {
        assert_eq!(rice_bits(8, 2), "11000");
    }

    #[test]
    fn rice_k2_value11() {
        assert_eq!(rice_bits(11, 2), "11011");
    }

    // ---- Rice k = 0 (pure unary) ------------------------------------------

    #[test]
    fn rice_k0_is_unary() {
        assert_eq!(rice_bits(0, 0), "0");
        assert_eq!(rice_bits(1, 0), "10");
        assert_eq!(rice_bits(3, 0), "1110");
        assert_eq!(rice_bits(5, 0), "111110");
    }

    // ---- Rice k = 1 -------------------------------------------------------

    #[test]
    fn rice_k1_values() {
        assert_eq!(rice_bits(0, 1), "00");
        assert_eq!(rice_bits(1, 1), "01");
        assert_eq!(rice_bits(2, 1), "100");
        assert_eq!(rice_bits(5, 1), "1101");
    }

    // ---- Rice k = 3 -------------------------------------------------------

    #[test]
    fn rice_k3_values() {
        // 10 -> q=1,"10" r=2 "010" = "10010"
        assert_eq!(rice_bits(10, 3), "10010");
        // 20 -> q=2,"110" r=4 "100" = "110100"
        assert_eq!(rice_bits(20, 3), "110100");
    }

    // ---- Rice boundary k = 31 ---------------------------------------------

    #[test]
    fn rice_k31_boundary() {
        // value < 2^31 -> q = 0 ("0") then 31 remainder bits.
        let value = 0x1234_5678u32;
        let mut w = BitWriter::new();
        rice_encode(value, 31, &mut w);
        // 1 unary stop bit + 31 remainder bits = 32 bits total.
        assert_eq!(w.bit_len(), 32);
        let bytes = rice_encode_all(&[value], 31);
        let back = rice_decode_all(&bytes, 1, 31);
        assert_eq!(back, vec![value]);
    }

    #[test]
    fn rice_k31_large_quotient() {
        // value with the top bit set -> q = 1.
        let value = 0x8000_0005u32;
        let bytes = rice_encode_all(&[value], 31);
        assert_eq!(rice_decode_all(&bytes, 1, 31), vec![value]);
    }

    // ---- Golomb reference bit strings (M = 5, non power of two) -----------

    #[test]
    fn golomb_m5_value0() {
        assert_eq!(golomb_bits(0, 5), "000");
    }

    #[test]
    fn golomb_m5_value2() {
        assert_eq!(golomb_bits(2, 5), "010");
    }

    #[test]
    fn golomb_m5_value3() {
        // r = 3 >= cutoff(3): full b=3 bits coded as r+cutoff = 6 -> "110".
        assert_eq!(golomb_bits(3, 5), "0110");
    }

    #[test]
    fn golomb_m5_value4() {
        assert_eq!(golomb_bits(4, 5), "0111");
    }

    #[test]
    fn golomb_m5_value7() {
        // q = 7/5 = 1 -> "10"; r = 2 < cutoff(3) -> b-1=2 bits "10".
        assert_eq!(golomb_bits(7, 5), "1010");
    }

    #[test]
    fn golomb_m5_value12() {
        assert_eq!(golomb_bits(12, 5), "11010");
    }

    #[test]
    fn golomb_m5_value13() {
        assert_eq!(golomb_bits(13, 5), "110110");
    }

    // ---- Golomb M = 3 -----------------------------------------------------

    #[test]
    fn golomb_m3_values() {
        // b=2, cutoff=1: r=0 -> "0" (1 bit); r in {1,2} -> 2 bits r+1.
        assert_eq!(golomb_bits(0, 3), "00");
        assert_eq!(golomb_bits(1, 3), "010");
        assert_eq!(golomb_bits(2, 3), "011");
        assert_eq!(golomb_bits(3, 3), "100");
        assert_eq!(golomb_bits(5, 3), "1011");
    }

    // ---- Golomb M = 1 (pure unary) ----------------------------------------

    #[test]
    fn golomb_m1_is_unary() {
        assert_eq!(golomb_bits(0, 1), "0");
        assert_eq!(golomb_bits(3, 1), "1110");
    }

    // ---- Golomb M = 6 -----------------------------------------------------

    #[test]
    fn golomb_m6_value8() {
        // b=3, cutoff=2: q=1 "10"; r=2 >= cutoff -> 3 bits r+2=4 "100".
        assert_eq!(golomb_bits(8, 6), "10100");
    }

    // ---- Golomb M = 2^k equals Rice ---------------------------------------

    #[test]
    fn golomb_m4_equals_rice_k2() {
        for value in 0..64u32 {
            assert_eq!(golomb_bits(value, 4), rice_bits(value, 2));
        }
    }

    #[test]
    fn golomb_m8_equals_rice_k3() {
        for value in 0..64u32 {
            assert_eq!(golomb_bits(value, 8), rice_bits(value, 3));
        }
    }

    // ---- ceil_log2 bit trick ----------------------------------------------

    #[test]
    fn ceil_log2_reference() {
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(8), 3);
        assert_eq!(ceil_log2(9), 4);
        assert_eq!(ceil_log2(16), 4);
        assert_eq!(ceil_log2(17), 5);
    }

    // ---- Truncated-binary short-codeword count = 2^b - M ------------------

    #[cfg(test)]
    fn short_codeword_count(m: u32) -> u32 {
        let b = ceil_log2(m);
        let mut count = 0u32;
        for r in 0..m {
            let mut w = BitWriter::new();
            write_truncated_binary(r, m, &mut w);
            if b == 0 {
                // M == 1: zero-length remainder.
                assert_eq!(w.bit_len(), 0);
            } else if w.bit_len() as u32 == b - 1 {
                count += 1;
            } else {
                assert_eq!(w.bit_len() as u32, b);
            }
        }
        count
    }

    #[test]
    fn truncated_binary_short_count_matches_formula() {
        for m in 1..=33u32 {
            let b = ceil_log2(m);
            let expected = if b == 0 { 0 } else { (1u32 << b) - m };
            assert_eq!(short_codeword_count(m), expected, "m = {m}");
        }
    }

    #[test]
    fn truncated_binary_specific_cutoffs() {
        assert_eq!(short_codeword_count(5), 3);
        assert_eq!(short_codeword_count(6), 2);
        assert_eq!(short_codeword_count(3), 1);
        assert_eq!(short_codeword_count(7), 1);
        assert_eq!(short_codeword_count(8), 0);
    }

    // ---- BitWriter / BitReader primitives ---------------------------------

    #[test]
    fn bitwriter_push_bit_msb_first() {
        let mut w = BitWriter::new();
        w.push_bit(true);
        w.push_bit(false);
        w.push_bit(true);
        assert_eq!(w.bit_len(), 3);
        let bytes = w.into_bytes();
        // bits 1,0,1 in the top three bits -> 0b1010_0000.
        assert_eq!(bytes, vec![0b1010_0000]);
    }

    #[test]
    fn bitwriter_push_bits() {
        let mut w = BitWriter::new();
        w.push_bits(0b101, 3);
        w.push_bits(0b11, 2);
        assert_eq!(encoded_bits(w), "10111");
    }

    #[test]
    fn bitwriter_push_bits_zero_count_is_noop() {
        let mut w = BitWriter::new();
        w.push_bits(0xFFFF_FFFF, 0);
        assert_eq!(w.bit_len(), 0);
    }

    #[test]
    fn bitreader_read_bit_sequence() {
        let bytes = vec![0b1010_0000u8];
        let mut r = BitReader::new(&bytes);
        assert!(r.read_bit().unwrap());
        assert!(!r.read_bit().unwrap());
        assert!(r.read_bit().unwrap());
        assert!(!r.read_bit().unwrap());
    }

    #[test]
    fn bitreader_read_bits() {
        let bytes = vec![0b1011_1000u8];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_bits(5).unwrap(), 0b10111);
        assert_eq!(r.read_bits(0).unwrap(), 0);
    }

    #[test]
    fn bitreader_exhaustion_returns_none() {
        let bytes = vec![0u8];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_bits(8).unwrap(), 0);
        assert!(r.read_bit().is_none());
    }

    #[test]
    fn bitreader_bits_remaining() {
        let bytes = vec![0u8, 0u8];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.bits_remaining(), 16);
        let _ = r.read_bits(5);
        assert_eq!(r.bits_remaining(), 11);
    }

    // ---- Unary convention -------------------------------------------------

    #[test]
    fn unary_convention_ones_then_zero() {
        let mut w = BitWriter::new();
        write_unary(0, &mut w);
        assert_eq!(encoded_bits(w), "0");
        let mut w = BitWriter::new();
        write_unary(1, &mut w);
        assert_eq!(encoded_bits(w), "10");
        let mut w = BitWriter::new();
        write_unary(4, &mut w);
        assert_eq!(encoded_bits(w), "11110");
    }

    #[test]
    fn unary_round_trip() {
        for q in 0..40u32 {
            let mut w = BitWriter::new();
            write_unary(q, &mut w);
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            assert_eq!(read_unary(&mut r), Some(q));
        }
    }

    // ---- Single-value round trips -----------------------------------------

    #[test]
    fn rice_round_trip_single() {
        for k in 0..=8u32 {
            for value in 0..200u32 {
                let mut w = BitWriter::new();
                rice_encode(value, k, &mut w);
                let bytes = w.into_bytes();
                let mut r = BitReader::new(&bytes);
                assert_eq!(rice_decode(k, &mut r), Some(value), "k={k} v={value}");
            }
        }
    }

    #[test]
    fn golomb_round_trip_single() {
        for m in 1..=20u32 {
            for value in 0..200u32 {
                let mut w = BitWriter::new();
                golomb_encode(value, m, &mut w);
                let bytes = w.into_bytes();
                let mut r = BitReader::new(&bytes);
                assert_eq!(golomb_decode(m, &mut r), Some(value), "m={m} v={value}");
            }
        }
    }

    // ---- Continuous multi-value bit streams -------------------------------

    #[test]
    fn rice_multi_value_stream_packs_contiguously() {
        // 7 -> "1011", 0 -> "000", 4 -> "1000" => "10110001000" (11 bits).
        let mut w = BitWriter::new();
        rice_encode(7, 2, &mut w);
        rice_encode(0, 2, &mut w);
        rice_encode(4, 2, &mut w);
        assert_eq!(encoded_bits(w), "10110001000");
    }

    #[test]
    fn golomb_multi_value_stream_packs_contiguously() {
        // 7 -> "1010", 2 -> "010", 13 -> "110110".
        let mut w = BitWriter::new();
        golomb_encode(7, 5, &mut w);
        golomb_encode(2, 5, &mut w);
        golomb_encode(13, 5, &mut w);
        assert_eq!(encoded_bits(w), "1010010110110");
    }

    // ---- Batch encode/decode ----------------------------------------------

    #[test]
    fn rice_encode_all_decode_all_round_trip() {
        let values = [0u32, 1, 2, 3, 4, 7, 8, 15, 16, 100, 255, 1000];
        for k in 0..=10u32 {
            let bytes = rice_encode_all(&values, k);
            let back = rice_decode_all(&bytes, values.len(), k);
            assert_eq!(back, values.to_vec(), "k={k}");
        }
    }

    #[test]
    fn golomb_encode_all_decode_all_round_trip() {
        let values = [0u32, 1, 2, 3, 5, 7, 11, 13, 50, 123, 999];
        for m in 1..=17u32 {
            let bytes = golomb_encode_all(&values, m);
            let back = golomb_decode_all(&bytes, values.len(), m);
            assert_eq!(back, values.to_vec(), "m={m}");
        }
    }

    #[test]
    fn rice_encode_all_empty() {
        let bytes = rice_encode_all(&[], 3);
        assert!(bytes.is_empty());
        assert_eq!(rice_decode_all(&bytes, 0, 3), Vec::<u32>::new());
    }

    #[test]
    fn golomb_encode_all_empty() {
        let bytes = golomb_encode_all(&[], 5);
        assert!(bytes.is_empty());
        assert_eq!(golomb_decode_all(&bytes, 0, 5), Vec::<u32>::new());
    }

    // ---- Pseudo-random fuzz round trips -----------------------------------

    #[test]
    fn rice_round_trip_pseudo_random() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut values = Vec::new();
        for _ in 0..500 {
            values.push(lcg(&mut state) % 4096);
        }
        for k in [0u32, 1, 2, 4, 7, 11] {
            let bytes = rice_encode_all(&values, k);
            let back = rice_decode_all(&bytes, values.len(), k);
            assert_eq!(back, values, "k={k}");
        }
    }

    #[test]
    fn golomb_round_trip_pseudo_random() {
        let mut state = 0x0fed_cba9_8765_4321u64;
        let mut values = Vec::new();
        for _ in 0..500 {
            values.push(lcg(&mut state) % 2048);
        }
        for m in [1u32, 2, 3, 5, 6, 10, 17, 100] {
            let bytes = golomb_encode_all(&values, m);
            let back = golomb_decode_all(&bytes, values.len(), m);
            assert_eq!(back, values, "m={m}");
        }
    }

    #[test]
    fn mixed_magnitudes_round_trip() {
        let values = [0u32, u32::MAX >> 2, 1, 1 << 20, 2, 1 << 10];
        let bytes = rice_encode_all(&values, 8);
        assert_eq!(rice_decode_all(&bytes, values.len(), 8), values.to_vec());
    }

    #[test]
    fn golomb_large_modulus_round_trip() {
        let values = [0u32, 500, 1000, 65535, 70000];
        let bytes = golomb_encode_all(&values, 1000);
        assert_eq!(
            golomb_decode_all(&bytes, values.len(), 1000),
            values.to_vec()
        );
    }

    #[test]
    fn bit_len_tracks_pushes() {
        let mut w = BitWriter::new();
        assert_eq!(w.bit_len(), 0);
        w.push_bits(0, 13);
        assert_eq!(w.bit_len(), 13);
        w.push_bit(true);
        assert_eq!(w.bit_len(), 14);
    }

    #[test]
    fn truncated_binary_round_trip_direct() {
        for m in 1..=40u32 {
            for r in 0..m {
                let mut w = BitWriter::new();
                write_truncated_binary(r, m, &mut w);
                let bytes = w.into_bytes();
                let mut reader = BitReader::new(&bytes);
                assert_eq!(
                    read_truncated_binary(m, &mut reader),
                    Some(r),
                    "m={m} r={r}"
                );
            }
        }
    }
}
