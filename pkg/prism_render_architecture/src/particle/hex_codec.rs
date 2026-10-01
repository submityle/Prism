//! `base16` (hexadecimal) codec — pure-integer, `CPU` golden reference.
//!
//! This module maps an arbitrary byte stream to and from `base16`
//! (hexadecimal) text. Each input byte becomes exactly two hex nibbles, with
//! the most-significant nibble emitted first, so the ratio is a fixed two
//! characters per byte with no padding and no block grouping.
//!
//! The `base16` alphabet is the sixteen symbols `0`-`9` plus `a`-`f`
//! (lower case) or `A`-`F` (upper case). Encoding is case-selectable; decoding
//! folds case and accepts either form.
//!
//! This `base16` codec is DELIBERATELY DISTINCT from the sibling
//! `base32_rfc4648` module (which implements `base32` and the `base32hex`
//! extended alphabet) and from any `base64` module. There is zero overlap:
//! `base16` uses a 16-symbol alphabet and a 1-byte-to-2-character ratio,
//! whereas `base32` uses a 32-symbol alphabet over 5-bit groups and `base64`
//! uses a 64-symbol alphabet over 6-bit groups. None of those modules' tables
//! or helpers are shared here.

use alloc::string::String;
use alloc::vec::Vec;

/// Map a single low nibble (0..=15) to its lower-case `ASCII` hex digit.
///
/// Only the low four bits of `n` are considered; higher bits are ignored.
#[must_use]
pub fn nibble_to_hex_lower(n: u8) -> u8 {
    let v = n & 0x0F;
    if v < 10 {
        b'0'.wrapping_add(v)
    } else {
        b'a'.wrapping_add(v.wrapping_sub(10))
    }
}

/// Map a single low nibble (0..=15) to its upper-case `ASCII` hex digit.
///
/// Only the low four bits of `n` are considered; higher bits are ignored.
#[must_use]
pub fn nibble_to_hex_upper(n: u8) -> u8 {
    let v = n & 0x0F;
    if v < 10 {
        b'0'.wrapping_add(v)
    } else {
        b'A'.wrapping_add(v.wrapping_sub(10))
    }
}

/// Decode a single `ASCII` hex digit character to its nibble value (0..=15).
///
/// Accepts `0`-`9`, `a`-`f`, and `A`-`F`. Returns [`None`] for any other byte.
#[must_use]
pub fn hex_digit_value(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c.wrapping_sub(b'0')),
        b'a'..=b'f' => Some(c.wrapping_sub(b'a').wrapping_add(10)),
        b'A'..=b'F' => Some(c.wrapping_sub(b'A').wrapping_add(10)),
        _ => None,
    }
}

/// Encode `bytes` as lower-case `base16` text.
///
/// The output length is exactly `2 * bytes.len()`.
#[must_use]
pub fn hex_encode_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().wrapping_mul(2));
    for &b in bytes.iter() {
        out.push(nibble_to_hex_lower(b >> 4) as char);
        out.push(nibble_to_hex_lower(b & 0x0F) as char);
    }
    out
}

/// Encode `bytes` as upper-case `base16` text.
///
/// The output length is exactly `2 * bytes.len()`.
#[must_use]
pub fn hex_encode_upper(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().wrapping_mul(2));
    for &b in bytes.iter() {
        out.push(nibble_to_hex_upper(b >> 4) as char);
        out.push(nibble_to_hex_upper(b & 0x0F) as char);
    }
    out
}

/// Decode `base16` text, folding case.
///
/// Accepts `ASCII` hex in either case. Returns [`None`] if the input length is
/// odd or if any byte is not a valid hex digit.
#[must_use]
pub fn hex_decode(s: &[u8]) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    for pair in s.chunks_exact(2) {
        let hi = hex_digit_value(pair[0])?;
        let lo = hex_digit_value(pair[1])?;
        out.push((hi << 4) | lo);
    }
    Some(out)
}

/// Decode `base16` text, accepting only lower-case `ASCII` hex digits.
///
/// Behaves like [`hex_decode`] but rejects upper-case `A`-`F`. Returns [`None`]
/// on odd length or any disallowed byte.
#[must_use]
pub fn hex_decode_lower(s: &[u8]) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    for pair in s.chunks_exact(2) {
        let hi = lower_digit_value(pair[0])?;
        let lo = lower_digit_value(pair[1])?;
        out.push((hi << 4) | lo);
    }
    Some(out)
}

/// Decode a single lower-case-only `ASCII` hex digit to its nibble value.
///
/// Accepts `0`-`9` and `a`-`f`; rejects upper-case `A`-`F`.
#[must_use]
fn lower_digit_value(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c.wrapping_sub(b'0')),
        b'a'..=b'f' => Some(c.wrapping_sub(b'a').wrapping_add(10)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple inline linear congruential generator for pseudo-random bytes.
    ///
    /// Uses the Numerical Recipes constants. Test-only; no external deps.
    #[cfg(test)]
    struct Lcg {
        state: u32,
    }

    #[cfg(test)]
    impl Lcg {
        #[cfg(test)]
        fn new(seed: u32) -> Self {
            Self { state: seed }
        }

        #[cfg(test)]
        fn next_u32(&mut self) -> u32 {
            self.state = self
                .state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            self.state
        }

        #[cfg(test)]
        fn next_byte(&mut self) -> u8 {
            (self.next_u32() >> 24) as u8
        }
    }

    #[cfg(test)]
    fn make_random(seed: u32, len: usize) -> Vec<u8> {
        let mut rng = Lcg::new(seed);
        let mut v = Vec::with_capacity(len);
        for _ in 0..len {
            v.push(rng.next_byte());
        }
        v
    }

    #[test]
    fn encode_empty_lower() {
        assert_eq!(hex_encode_lower(&[]), "");
    }

    #[test]
    fn encode_empty_upper() {
        assert_eq!(hex_encode_upper(&[]), "");
    }

    #[test]
    fn encode_zero_byte_lower() {
        assert_eq!(hex_encode_lower(&[0x00]), "00");
    }

    #[test]
    fn encode_zero_byte_upper() {
        assert_eq!(hex_encode_upper(&[0x00]), "00");
    }

    #[test]
    fn encode_ff_lower() {
        assert_eq!(hex_encode_lower(&[0xFF]), "ff");
    }

    #[test]
    fn encode_ff_upper() {
        assert_eq!(hex_encode_upper(&[0xFF]), "FF");
    }

    #[test]
    fn encode_deadbeef_lower() {
        assert_eq!(hex_encode_lower(&[0xDE, 0xAD, 0xBE, 0xEF]), "deadbeef");
    }

    #[test]
    fn encode_deadbeef_upper() {
        assert_eq!(hex_encode_upper(&[0xDE, 0xAD, 0xBE, 0xEF]), "DEADBEEF");
    }

    #[test]
    fn encode_single_low_nibble() {
        assert_eq!(hex_encode_lower(&[0x0A]), "0a");
        assert_eq!(hex_encode_upper(&[0x0A]), "0A");
    }

    #[test]
    fn encode_single_high_nibble() {
        assert_eq!(hex_encode_lower(&[0xA0]), "a0");
        assert_eq!(hex_encode_upper(&[0xA0]), "A0");
    }

    #[test]
    fn encode_multibyte_sequence() {
        assert_eq!(hex_encode_lower(&[0x01, 0x23, 0x45, 0x67]), "01234567");
        assert_eq!(hex_encode_lower(&[0x89, 0xAB, 0xCD, 0xEF]), "89abcdef");
    }

    #[test]
    fn encode_sample_0x00_to_0xff() {
        let samples: [u8; 9] = [0x00, 0x0F, 0x10, 0x7F, 0x80, 0xAA, 0x55, 0xC3, 0xFF];
        let expected = ["00", "0f", "10", "7f", "80", "aa", "55", "c3", "ff"];
        for (b, e) in samples.iter().zip(expected.iter()) {
            assert_eq!(hex_encode_lower(core::slice::from_ref(b)), *e);
        }
    }

    #[test]
    fn encode_every_byte_two_chars() {
        for n in 0u16..=255u16 {
            let b = n as u8;
            let s = hex_encode_lower(&[b]);
            assert_eq!(s.len(), 2);
        }
    }

    #[test]
    fn decode_empty() {
        assert_eq!(hex_decode(b""), Some(Vec::new()));
    }

    #[test]
    fn decode_zero_byte() {
        assert_eq!(hex_decode(b"00"), Some(alloc::vec![0x00]));
    }

    #[test]
    fn decode_ff_lower() {
        assert_eq!(hex_decode(b"ff"), Some(alloc::vec![0xFF]));
    }

    #[test]
    fn decode_ff_upper() {
        assert_eq!(hex_decode(b"FF"), Some(alloc::vec![0xFF]));
    }

    #[test]
    fn decode_deadbeef_lower() {
        assert_eq!(
            hex_decode(b"deadbeef"),
            Some(alloc::vec![0xDE, 0xAD, 0xBE, 0xEF])
        );
    }

    #[test]
    fn decode_deadbeef_upper() {
        assert_eq!(
            hex_decode(b"DEADBEEF"),
            Some(alloc::vec![0xDE, 0xAD, 0xBE, 0xEF])
        );
    }

    #[test]
    fn decode_case_insensitive_equivalence() {
        assert_eq!(hex_decode(b"DEADBEEF"), hex_decode(b"deadbeef"));
        assert_eq!(hex_decode(b"CaFeBaBe"), hex_decode(b"cafebabe"));
    }

    #[test]
    fn decode_mixed_case() {
        assert_eq!(hex_decode(b"DeAdBeEf"), hex_decode(b"deadbeef"));
    }

    #[test]
    fn decode_odd_length_rejected() {
        assert_eq!(hex_decode(b"0"), None);
        assert_eq!(hex_decode(b"abc"), None);
        assert_eq!(hex_decode(b"deadbee"), None);
    }

    #[test]
    fn decode_non_hex_g_rejected() {
        assert_eq!(hex_decode(b"0g"), None);
        assert_eq!(hex_decode(b"gg"), None);
    }

    #[test]
    fn decode_space_rejected() {
        assert_eq!(hex_decode(b"0 "), None);
        assert_eq!(hex_decode(b"  "), None);
    }

    #[test]
    fn decode_nul_rejected() {
        assert_eq!(hex_decode(&[b'0', 0x00]), None);
    }

    #[test]
    fn decode_punctuation_rejected() {
        assert_eq!(hex_decode(b"0x"), None);
        assert_eq!(hex_decode(b"#0"), None);
    }

    #[test]
    fn decode_lower_only_rejects_upper() {
        assert_eq!(hex_decode_lower(b"FF"), None);
        assert_eq!(hex_decode_lower(b"ff"), Some(alloc::vec![0xFF]));
    }

    #[test]
    fn decode_lower_only_odd_rejected() {
        assert_eq!(hex_decode_lower(b"abc"), None);
    }

    #[test]
    fn decode_lower_only_matches_full() {
        assert_eq!(hex_decode_lower(b"deadbeef"), hex_decode(b"deadbeef"));
    }

    #[test]
    fn nibble_to_hex_lower_digits() {
        assert_eq!(nibble_to_hex_lower(0), b'0');
        assert_eq!(nibble_to_hex_lower(9), b'9');
        assert_eq!(nibble_to_hex_lower(10), b'a');
        assert_eq!(nibble_to_hex_lower(15), b'f');
    }

    #[test]
    fn nibble_to_hex_upper_digits() {
        assert_eq!(nibble_to_hex_upper(0), b'0');
        assert_eq!(nibble_to_hex_upper(9), b'9');
        assert_eq!(nibble_to_hex_upper(10), b'A');
        assert_eq!(nibble_to_hex_upper(15), b'F');
    }

    #[test]
    fn nibble_to_hex_ignores_high_bits() {
        assert_eq!(nibble_to_hex_lower(0xF0), b'0');
        assert_eq!(nibble_to_hex_lower(0xFA), b'a');
    }

    #[test]
    fn hex_digit_value_digits() {
        assert_eq!(hex_digit_value(b'0'), Some(0));
        assert_eq!(hex_digit_value(b'9'), Some(9));
    }

    #[test]
    fn hex_digit_value_lower() {
        assert_eq!(hex_digit_value(b'a'), Some(10));
        assert_eq!(hex_digit_value(b'f'), Some(15));
    }

    #[test]
    fn hex_digit_value_upper() {
        assert_eq!(hex_digit_value(b'A'), Some(10));
        assert_eq!(hex_digit_value(b'F'), Some(15));
    }

    #[test]
    fn hex_digit_value_invalid() {
        assert_eq!(hex_digit_value(b'g'), None);
        assert_eq!(hex_digit_value(b' '), None);
        assert_eq!(hex_digit_value(0x00), None);
        assert_eq!(hex_digit_value(b'G'), None);
    }

    #[test]
    fn roundtrip_lower_single_bytes() {
        for n in 0u16..=255u16 {
            let b = n as u8;
            let enc = hex_encode_lower(&[b]);
            let dec = hex_decode(enc.as_bytes());
            assert_eq!(dec, Some(alloc::vec![b]));
        }
    }

    #[test]
    fn roundtrip_upper_single_bytes() {
        for n in 0u16..=255u16 {
            let b = n as u8;
            let enc = hex_encode_upper(&[b]);
            let dec = hex_decode(enc.as_bytes());
            assert_eq!(dec, Some(alloc::vec![b]));
        }
    }

    #[test]
    fn upper_lower_decode_identity() {
        for n in 0u16..=255u16 {
            let b = n as u8;
            let lo = hex_encode_lower(&[b]);
            let up = hex_encode_upper(&[b]);
            assert_eq!(hex_decode(lo.as_bytes()), hex_decode(up.as_bytes()));
        }
    }

    #[test]
    fn roundtrip_random_lower() {
        for seed in 1u32..=16u32 {
            let data = make_random(seed, 64);
            let enc = hex_encode_lower(&data);
            assert_eq!(hex_decode(enc.as_bytes()), Some(data));
        }
    }

    #[test]
    fn roundtrip_random_upper() {
        for seed in 100u32..=116u32 {
            let data = make_random(seed, 48);
            let enc = hex_encode_upper(&data);
            assert_eq!(hex_decode(enc.as_bytes()), Some(data));
        }
    }

    #[test]
    fn roundtrip_random_varied_lengths() {
        for len in 0usize..=40usize {
            let data = make_random(len as u32 + 7, len);
            let enc = hex_encode_lower(&data);
            assert_eq!(hex_decode(enc.as_bytes()), Some(data));
        }
    }

    #[test]
    fn capacity_output_len_lower() {
        for len in 0usize..=32usize {
            let data = make_random(len as u32 + 3, len);
            let enc = hex_encode_lower(&data);
            assert_eq!(enc.len(), len.wrapping_mul(2));
        }
    }

    #[test]
    fn capacity_output_len_upper() {
        for len in 0usize..=32usize {
            let data = make_random(len as u32 + 5, len);
            let enc = hex_encode_upper(&data);
            assert_eq!(enc.len(), len.wrapping_mul(2));
        }
    }

    #[test]
    fn decode_output_len_half_input() {
        let enc = hex_encode_lower(&make_random(42, 30));
        let dec = hex_decode(enc.as_bytes()).expect("valid hex decodes");
        assert_eq!(dec.len(), enc.len() / 2);
    }

    #[test]
    fn odd_parity_helper_is_odd() {
        let x: u8 = 0x01;
        assert!((x & 1) == 1);
        let y: u8 = 0x02;
        assert!((y & 1) != 1);
    }

    #[test]
    fn encode_decode_identity_full_byte_vector() {
        let data = make_random(2026, 256);
        let enc = hex_encode_lower(&data);
        assert_eq!(hex_decode(enc.as_bytes()), Some(data.clone()));
        let enc_u = hex_encode_upper(&data);
        assert_eq!(hex_decode(enc_u.as_bytes()), Some(data));
    }
}
