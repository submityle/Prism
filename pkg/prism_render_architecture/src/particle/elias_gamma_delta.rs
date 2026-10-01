//! `Elias` universal integer codes: gamma, delta, and omega.
//!
//! This module implements the three classic `Elias` universal codes for the
//! positive integers (values `>= 1`): `Elias` gamma, `Elias` delta, and
//! `Elias` omega. These are *bit-level* codes: the output is a sequence of
//! individual bits rather than whole bytes.
//!
//! This is intentionally distinct from `varint_leb128` and
//! `zigzag_delta_encode`, which are *byte-aligned* encodings (each emits whole
//! bytes). `Elias` codes instead emit a variable number of raw bits and are
//! "universal" in the sense that they encode any positive integer without a
//! pre-agreed upper bound.
//!
//! # Primary representation
//! The primary bitstream representation here is a `Vec<bool>`, interpreted
//! `MSB`-first (the first `bool` in the stream is the most-significant bit of
//! the first code word). A tiny [`BitWriter`] helper accumulates bits during
//! encoding; decoding operates directly on a `&[bool]` slice together with a
//! `&mut usize` cursor so that back-to-back code words can be decoded from a
//! single shared stream.
//!
//! # Reference bit patterns
//! The following exact bit strings are produced by this module and are checked
//! in the tests:
//! - gamma: `1`, `010`, `011`, `00100`, `00101`, `0001000` for `1,2,3,4,5,8`.
//! - delta: `1`, `0100`, `01101` for `1,2,5`.
//! - omega: `0`, `100`, `110`, `101000` for `1,2,3,4`.
//!
//! This module is `no_std` friendly and allocates via `alloc`.

use alloc::vec::Vec;

/// A minimal `MSB`-first bit writer backed by a `Vec<bool>`.
///
/// Bits are appended in the order they should appear in the stream, so the
/// first bit pushed is the most-significant bit of the first code word.
pub struct BitWriter {
    bits: Vec<bool>,
}

impl BitWriter {
    /// Creates an empty [`BitWriter`].
    pub fn new() -> BitWriter {
        BitWriter { bits: Vec::new() }
    }

    /// Appends a single bit to the stream.
    pub fn push_bit(&mut self, bit: bool) {
        self.bits.push(bit);
    }

    /// Appends `count` zero bits (a unary-style run of zeros).
    pub fn write_unary_zeros(&mut self, count: u32) {
        let mut i: u32 = 0;
        while i < count {
            self.bits.push(false);
            i = i.wrapping_add(1);
        }
    }

    /// Returns the number of bits written so far.
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    /// Returns `true` when no bits have been written.
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// Borrows the accumulated bits.
    pub fn as_bits(&self) -> &[bool] {
        &self.bits
    }

    /// Consumes the writer and returns the accumulated bits.
    pub fn into_bits(self) -> Vec<bool> {
        self.bits
    }
}

impl Default for BitWriter {
    fn default() -> BitWriter {
        BitWriter::new()
    }
}

/// Returns the number of significant bits of `n` using an integer-only
/// computation (`32 - n.leading_zeros()`). For `n >= 1` this equals
/// `floor(log2(n)) + 1`. Returns `0` for `n == 0`.
fn bit_length(n: u32) -> u32 {
    32u32.wrapping_sub(n.leading_zeros())
}

/// Writes the binary representation of `n` (`MSB`-first, including the leading
/// `1`) using exactly `bit_length(n)` bits.
fn write_binary_msb_first(out: &mut BitWriter, n: u32) {
    let l = bit_length(n);
    let mut i = l;
    while i > 0 {
        i -= 1;
        let bit = ((n >> i) & 1) == 1;
        out.push_bit(bit);
    }
}

// ------------------------------------------------------------------
// Elias gamma
// ------------------------------------------------------------------

/// Encodes `n` (`n >= 1`) with `Elias` gamma into `out`.
///
/// The gamma code of `n` is `N` leading zeros followed by the `N + 1`-bit
/// binary representation of `n`, where `N = floor(log2(n))`.
pub fn elias_gamma_encode(n: u32, out: &mut BitWriter) {
    let l = bit_length(n);
    let num_zeros = l.wrapping_sub(1);
    out.write_unary_zeros(num_zeros);
    write_binary_msb_first(out, n);
}

/// One-shot `Elias` gamma encode returning a fresh `Vec<bool>`.
pub fn elias_gamma_encode_one(n: u32) -> Vec<bool> {
    let mut w = BitWriter::new();
    elias_gamma_encode(n, &mut w);
    w.into_bits()
}

/// Decodes one `Elias` gamma code word starting at `*pos`, advancing `*pos`
/// past the consumed bits. Returns `None` if the stream is truncated.
pub fn elias_gamma_decode(bits: &[bool], pos: &mut usize) -> Option<u32> {
    let mut num_zeros: usize = 0;
    loop {
        let b = *bits.get(*pos)?;
        *pos += 1;
        if b {
            break;
        }
        num_zeros += 1;
    }
    let mut value: u64 = 1;
    let mut i: usize = 0;
    while i < num_zeros {
        let b = *bits.get(*pos)?;
        *pos += 1;
        value = (value << 1) | if b { 1 } else { 0 };
        i += 1;
    }
    Some(value as u32)
}

/// Encodes a slice of positive integers back-to-back with `Elias` gamma.
pub fn elias_gamma_encode_slice(values: &[u32]) -> Vec<bool> {
    let mut w = BitWriter::new();
    for v in values.iter() {
        elias_gamma_encode(*v, &mut w);
    }
    w.into_bits()
}

/// Decodes an entire `Elias` gamma stream into a `Vec<u32>`.
pub fn elias_gamma_decode_all(bits: &[bool]) -> Option<Vec<u32>> {
    let mut out: Vec<u32> = Vec::new();
    let mut pos: usize = 0;
    while pos < bits.len() {
        let v = elias_gamma_decode(bits, &mut pos)?;
        out.push(v);
    }
    Some(out)
}

// ------------------------------------------------------------------
// Elias delta
// ------------------------------------------------------------------

/// Encodes `n` (`n >= 1`) with `Elias` delta into `out`.
///
/// The delta code gamma-encodes `L = bit_length(n)` and then appends the
/// lower `L - 1` bits of `n` (everything after the leading `1`).
pub fn elias_delta_encode(n: u32, out: &mut BitWriter) {
    let l = bit_length(n);
    elias_gamma_encode(l, out);
    let rem = l.wrapping_sub(1);
    let mut i = rem;
    while i > 0 {
        i -= 1;
        let bit = ((n >> i) & 1) == 1;
        out.push_bit(bit);
    }
}

/// One-shot `Elias` delta encode returning a fresh `Vec<bool>`.
pub fn elias_delta_encode_one(n: u32) -> Vec<bool> {
    let mut w = BitWriter::new();
    elias_delta_encode(n, &mut w);
    w.into_bits()
}

/// Decodes one `Elias` delta code word starting at `*pos`, advancing `*pos`
/// past the consumed bits. Returns `None` if the stream is truncated.
pub fn elias_delta_decode(bits: &[bool], pos: &mut usize) -> Option<u32> {
    let l = elias_gamma_decode(bits, pos)?;
    let mut value: u64 = 1;
    let rem = l.wrapping_sub(1);
    let mut i: u32 = 0;
    while i < rem {
        let b = *bits.get(*pos)?;
        *pos += 1;
        value = (value << 1) | if b { 1 } else { 0 };
        i += 1;
    }
    Some(value as u32)
}

/// Encodes a slice of positive integers back-to-back with `Elias` delta.
pub fn elias_delta_encode_slice(values: &[u32]) -> Vec<bool> {
    let mut w = BitWriter::new();
    for v in values.iter() {
        elias_delta_encode(*v, &mut w);
    }
    w.into_bits()
}

/// Decodes an entire `Elias` delta stream into a `Vec<u32>`.
pub fn elias_delta_decode_all(bits: &[bool]) -> Option<Vec<u32>> {
    let mut out: Vec<u32> = Vec::new();
    let mut pos: usize = 0;
    while pos < bits.len() {
        let v = elias_delta_decode(bits, &mut pos)?;
        out.push(v);
    }
    Some(out)
}

// ------------------------------------------------------------------
// Elias omega
// ------------------------------------------------------------------

/// Encodes `n` (`n >= 1`) with `Elias` omega into `out`.
///
/// The omega code recursively prefixes the binary length groups and terminates
/// with a trailing `0` bit. `omega(1)` is the single bit `0`.
pub fn elias_omega_encode(n: u32, out: &mut BitWriter) {
    let mut groups: Vec<Vec<bool>> = Vec::new();
    let mut m = n;
    while m > 1 {
        let l = bit_length(m);
        let mut g: Vec<bool> = Vec::new();
        let mut i = l;
        while i > 0 {
            i -= 1;
            g.push(((m >> i) & 1) == 1);
        }
        groups.push(g);
        m = l.wrapping_sub(1);
    }
    let mut idx = groups.len();
    while idx > 0 {
        idx -= 1;
        for bit in groups[idx].iter() {
            out.push_bit(*bit);
        }
    }
    out.push_bit(false);
}

/// One-shot `Elias` omega encode returning a fresh `Vec<bool>`.
pub fn elias_omega_encode_one(n: u32) -> Vec<bool> {
    let mut w = BitWriter::new();
    elias_omega_encode(n, &mut w);
    w.into_bits()
}

/// Decodes one `Elias` omega code word starting at `*pos`, advancing `*pos`
/// past the consumed bits. Returns `None` if the stream is truncated.
pub fn elias_omega_decode(bits: &[bool], pos: &mut usize) -> Option<u32> {
    let mut n: u64 = 1;
    loop {
        let b = *bits.get(*pos)?;
        *pos += 1;
        if !b {
            return Some(n as u32);
        }
        let mut value: u64 = 1;
        let mut i: u64 = 0;
        while i < n {
            let bb = *bits.get(*pos)?;
            *pos += 1;
            value = (value << 1) | if bb { 1 } else { 0 };
            i += 1;
        }
        n = value;
    }
}

/// Encodes a slice of positive integers back-to-back with `Elias` omega.
pub fn elias_omega_encode_slice(values: &[u32]) -> Vec<bool> {
    let mut w = BitWriter::new();
    for v in values.iter() {
        elias_omega_encode(*v, &mut w);
    }
    w.into_bits()
}

/// Decodes an entire `Elias` omega stream into a `Vec<u32>`.
pub fn elias_omega_decode_all(bits: &[bool]) -> Option<Vec<u32>> {
    let mut out: Vec<u32> = Vec::new();
    let mut pos: usize = 0;
    while pos < bits.len() {
        let v = elias_omega_decode(bits, &mut pos)?;
        out.push(v);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;

    #[cfg(test)]
    fn bits_to_string(bits: &[bool]) -> String {
        let mut s = String::new();
        for b in bits.iter() {
            s.push(if *b { '1' } else { '0' });
        }
        s
    }

    #[cfg(test)]
    fn gamma_str(n: u32) -> String {
        bits_to_string(&elias_gamma_encode_one(n))
    }

    #[cfg(test)]
    fn delta_str(n: u32) -> String {
        bits_to_string(&elias_delta_encode_one(n))
    }

    #[cfg(test)]
    fn omega_str(n: u32) -> String {
        bits_to_string(&elias_omega_encode_one(n))
    }

    // ---- gamma reference patterns ----

    #[test]
    fn gamma_reference_1() {
        assert_eq!(gamma_str(1), "1");
    }

    #[test]
    fn gamma_reference_2() {
        assert_eq!(gamma_str(2), "010");
    }

    #[test]
    fn gamma_reference_3() {
        assert_eq!(gamma_str(3), "011");
    }

    #[test]
    fn gamma_reference_4() {
        assert_eq!(gamma_str(4), "00100");
    }

    #[test]
    fn gamma_reference_5() {
        assert_eq!(gamma_str(5), "00101");
    }

    #[test]
    fn gamma_reference_8() {
        assert_eq!(gamma_str(8), "0001000");
    }

    #[test]
    fn gamma_reference_6_and_7() {
        assert_eq!(gamma_str(6), "00110");
        assert_eq!(gamma_str(7), "00111");
    }

    // ---- delta reference patterns ----

    #[test]
    fn delta_reference_1() {
        assert_eq!(delta_str(1), "1");
    }

    #[test]
    fn delta_reference_2() {
        assert_eq!(delta_str(2), "0100");
    }

    #[test]
    fn delta_reference_5() {
        assert_eq!(delta_str(5), "01101");
    }

    // ---- omega reference patterns ----

    #[test]
    fn omega_reference_1() {
        assert_eq!(omega_str(1), "0");
    }

    #[test]
    fn omega_reference_2() {
        assert_eq!(omega_str(2), "100");
    }

    #[test]
    fn omega_reference_4() {
        assert_eq!(omega_str(4), "101000");
    }

    #[test]
    fn omega_reference_3_and_5() {
        assert_eq!(omega_str(3), "110");
        assert_eq!(omega_str(5), "101010");
    }

    // ---- round-trip 1..=1000 ----

    #[test]
    fn gamma_roundtrip_1_to_1000() {
        let mut n: u32 = 1;
        while n <= 1000 {
            let bits = elias_gamma_encode_one(n);
            let mut pos: usize = 0;
            assert_eq!(elias_gamma_decode(&bits, &mut pos), Some(n));
            assert_eq!(pos, bits.len());
            n += 1;
        }
    }

    #[test]
    fn delta_roundtrip_1_to_1000() {
        let mut n: u32 = 1;
        while n <= 1000 {
            let bits = elias_delta_encode_one(n);
            let mut pos: usize = 0;
            assert_eq!(elias_delta_decode(&bits, &mut pos), Some(n));
            assert_eq!(pos, bits.len());
            n += 1;
        }
    }

    #[test]
    fn omega_roundtrip_1_to_1000() {
        let mut n: u32 = 1;
        while n <= 1000 {
            let bits = elias_omega_encode_one(n);
            let mut pos: usize = 0;
            assert_eq!(elias_omega_decode(&bits, &mut pos), Some(n));
            assert_eq!(pos, bits.len());
            n += 1;
        }
    }

    // ---- back-to-back stream round-trip ----

    #[test]
    fn gamma_stream_roundtrip() {
        let values: [u32; 10] = [1, 2, 3, 4, 5, 8, 16, 100, 999, 1000];
        let bits = elias_gamma_encode_slice(&values);
        let decoded = elias_gamma_decode_all(&bits).unwrap();
        assert_eq!(decoded.as_slice(), &values);
    }

    #[test]
    fn delta_stream_roundtrip() {
        let values: [u32; 10] = [1, 2, 3, 4, 5, 8, 16, 100, 999, 1000];
        let bits = elias_delta_encode_slice(&values);
        let decoded = elias_delta_decode_all(&bits).unwrap();
        assert_eq!(decoded.as_slice(), &values);
    }

    #[test]
    fn omega_stream_roundtrip() {
        let values: [u32; 10] = [1, 2, 3, 4, 5, 8, 16, 100, 999, 1000];
        let bits = elias_omega_encode_slice(&values);
        let decoded = elias_omega_decode_all(&bits).unwrap();
        assert_eq!(decoded.as_slice(), &values);
    }

    // ---- decode position advancement ----

    #[test]
    fn gamma_decode_position_advances() {
        let a = elias_gamma_encode_one(7);
        let b = elias_gamma_encode_one(42);
        let mut bits: Vec<bool> = Vec::new();
        bits.extend_from_slice(&a);
        bits.extend_from_slice(&b);
        let mut pos: usize = 0;
        assert_eq!(elias_gamma_decode(&bits, &mut pos), Some(7));
        assert_eq!(pos, a.len());
        assert_eq!(elias_gamma_decode(&bits, &mut pos), Some(42));
        assert_eq!(pos, a.len() + b.len());
    }

    #[test]
    fn delta_decode_position_advances() {
        let a = elias_delta_encode_one(7);
        let b = elias_delta_encode_one(42);
        let mut bits: Vec<bool> = Vec::new();
        bits.extend_from_slice(&a);
        bits.extend_from_slice(&b);
        let mut pos: usize = 0;
        assert_eq!(elias_delta_decode(&bits, &mut pos), Some(7));
        assert_eq!(pos, a.len());
        assert_eq!(elias_delta_decode(&bits, &mut pos), Some(42));
        assert_eq!(pos, a.len() + b.len());
    }

    #[test]
    fn omega_decode_position_advances() {
        let a = elias_omega_encode_one(7);
        let b = elias_omega_encode_one(42);
        let mut bits: Vec<bool> = Vec::new();
        bits.extend_from_slice(&a);
        bits.extend_from_slice(&b);
        let mut pos: usize = 0;
        assert_eq!(elias_omega_decode(&bits, &mut pos), Some(7));
        assert_eq!(pos, a.len());
        assert_eq!(elias_omega_decode(&bits, &mut pos), Some(42));
        assert_eq!(pos, a.len() + b.len());
    }

    // ---- length relationships ----

    #[test]
    fn gamma_length_formula() {
        let samples: [u32; 7] = [1, 2, 4, 8, 100, 1000, 65535];
        for v in samples.iter() {
            let expected = 2 * (bit_length(*v) - 1) + 1;
            assert_eq!(elias_gamma_encode_one(*v).len() as u32, expected);
        }
    }

    #[test]
    fn delta_shorter_than_gamma_large() {
        let n: u32 = 1_000_000;
        let g = elias_gamma_encode_one(n).len();
        let d = elias_delta_encode_one(n).len();
        assert!(d < g);
    }

    #[test]
    fn omega_shorter_than_gamma_large() {
        let n: u32 = 1_000_000;
        let g = elias_gamma_encode_one(n).len();
        let o = elias_omega_encode_one(n).len();
        assert!(o < g);
    }

    #[test]
    fn delta_omega_gamma_ordering_large() {
        let n: u32 = 1_000_000;
        let g = elias_gamma_encode_one(n).len();
        let d = elias_delta_encode_one(n).len();
        let o = elias_omega_encode_one(n).len();
        assert!(d < o);
        assert!(o < g);
    }

    // ---- powers of two boundaries ----

    #[test]
    fn gamma_power_of_two_boundaries() {
        let mut k: u32 = 0;
        while k <= 20 {
            let n: u32 = 1u32 << k;
            assert!(n.is_power_of_two());
            let bits = elias_gamma_encode_one(n);
            assert_eq!(bits.len() as u32, 2 * k + 1);
            let mut pos: usize = 0;
            assert_eq!(elias_gamma_decode(&bits, &mut pos), Some(n));
            k += 1;
        }
    }

    #[test]
    fn delta_power_of_two_roundtrip() {
        let mut k: u32 = 0;
        while k <= 20 {
            let n: u32 = 1u32 << k;
            let bits = elias_delta_encode_one(n);
            let mut pos: usize = 0;
            assert_eq!(elias_delta_decode(&bits, &mut pos), Some(n));
            assert_eq!(pos, bits.len());
            k += 1;
        }
    }

    #[test]
    fn omega_power_of_two_roundtrip() {
        let mut k: u32 = 0;
        while k <= 20 {
            let n: u32 = 1u32 << k;
            let bits = elias_omega_encode_one(n);
            let mut pos: usize = 0;
            assert_eq!(elias_omega_decode(&bits, &mut pos), Some(n));
            assert_eq!(pos, bits.len());
            k += 1;
        }
    }

    #[test]
    fn boundary_just_below_and_above_power_of_two() {
        let samples: [u32; 6] = [7, 8, 9, 15, 16, 17];
        for v in samples.iter() {
            let g = elias_gamma_encode_one(*v);
            let d = elias_delta_encode_one(*v);
            let o = elias_omega_encode_one(*v);
            let mut pg: usize = 0;
            let mut pd: usize = 0;
            let mut po: usize = 0;
            assert_eq!(elias_gamma_decode(&g, &mut pg), Some(*v));
            assert_eq!(elias_delta_decode(&d, &mut pd), Some(*v));
            assert_eq!(elias_omega_decode(&o, &mut po), Some(*v));
        }
    }

    // ---- truncated streams return None ----

    #[test]
    fn gamma_truncated_returns_none() {
        let full = elias_gamma_encode_one(100);
        let truncated = &full[..full.len() - 1];
        let mut pos: usize = 0;
        assert_eq!(elias_gamma_decode(truncated, &mut pos), None);
    }

    #[test]
    fn delta_truncated_returns_none() {
        let full = elias_delta_encode_one(100);
        let truncated = &full[..full.len() - 1];
        let mut pos: usize = 0;
        assert_eq!(elias_delta_decode(truncated, &mut pos), None);
    }

    #[test]
    fn omega_truncated_returns_none() {
        let full = elias_omega_encode_one(100);
        let truncated = &full[..full.len() - 1];
        let mut pos: usize = 0;
        assert_eq!(elias_omega_decode(truncated, &mut pos), None);
    }

    #[test]
    fn gamma_empty_decode_none() {
        let empty: [bool; 0] = [];
        let mut pos: usize = 0;
        assert_eq!(elias_gamma_decode(&empty, &mut pos), None);
    }

    #[test]
    fn decode_all_empty_returns_empty() {
        let empty: [bool; 0] = [];
        assert_eq!(elias_gamma_decode_all(&empty), Some(Vec::new()));
        assert_eq!(elias_delta_decode_all(&empty), Some(Vec::new()));
        assert_eq!(elias_omega_decode_all(&empty), Some(Vec::new()));
    }

    #[test]
    fn gamma_mixed_stream_decode_all() {
        let values: [u32; 6] = [10, 1, 7, 256, 3, 42];
        let bits = elias_gamma_encode_slice(&values);
        let decoded = elias_gamma_decode_all(&bits).unwrap();
        assert_eq!(decoded.as_slice(), &values);
    }

    #[test]
    fn large_value_roundtrip_all_codes() {
        let n: u32 = 0x7FFF_FFFF;
        let g = elias_gamma_encode_one(n);
        let d = elias_delta_encode_one(n);
        let o = elias_omega_encode_one(n);
        let mut pg: usize = 0;
        let mut pd: usize = 0;
        let mut po: usize = 0;
        assert_eq!(elias_gamma_decode(&g, &mut pg), Some(n));
        assert_eq!(elias_delta_decode(&d, &mut pd), Some(n));
        assert_eq!(elias_omega_decode(&o, &mut po), Some(n));
    }

    #[test]
    fn one_shot_matches_writer() {
        let mut w = BitWriter::new();
        elias_gamma_encode(123, &mut w);
        assert_eq!(w.into_bits(), elias_gamma_encode_one(123));
        let mut w2 = BitWriter::new();
        elias_delta_encode(123, &mut w2);
        assert_eq!(w2.into_bits(), elias_delta_encode_one(123));
    }

    #[test]
    fn bit_length_matches_expected() {
        assert_eq!(bit_length(1), 1);
        assert_eq!(bit_length(2), 2);
        assert_eq!(bit_length(3), 2);
        assert_eq!(bit_length(4), 3);
        assert_eq!(bit_length(255), 8);
        assert_eq!(bit_length(256), 9);
    }

    #[test]
    fn writer_len_and_empty() {
        let mut w = BitWriter::new();
        assert!(w.is_empty());
        w.push_bit(true);
        w.write_unary_zeros(3);
        assert_eq!(w.len(), 4);
        assert!(!w.is_empty());
        assert_eq!(bits_to_string(w.as_bits()), "1000");
    }

    #[test]
    fn max_u32_roundtrip_gamma() {
        let n: u32 = u32::MAX;
        let bits = elias_gamma_encode_one(n);
        let mut pos: usize = 0;
        assert_eq!(elias_gamma_decode(&bits, &mut pos), Some(n));
        assert_eq!(pos, bits.len());
    }
}
