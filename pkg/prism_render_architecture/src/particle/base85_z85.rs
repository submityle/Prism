//! `Z85` binary-to-text encoding (`ZeroMQ` `RFC` 32) using the canonical
//! 85-character alphabet
//! `0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#`,
//! implemented with pure integer arithmetic on the `CPU`.
//!
//! `Z85` reinterprets the input as a sequence of 4-byte `big-endian` groups.
//! Each group forms a `u32` value `val`, which is rewritten in `radix` 85 as
//! exactly five printable characters, most significant `base85` digit first:
//! for `j` in `0..5` the emitted digit is `(val / 85^(4 - j)) % 85`. Decoding
//! is the exact inverse — five characters recombine into one `base85`
//! accumulator which must fit in a `u32` and then split back into four
//! `big-endian` bytes.
//!
//! The alphabet is specifically ordered so every character is safe to embed in
//! source code and shell strings; it contains no whitespace, no backslash, and
//! no single/double quote. Encoded output is 25% larger than the input (5
//! characters per 4 bytes).
//!
//! ## Boundaries
//!
//! This module is deliberately independent from the sibling `ascii85_codec` in
//! this crate and shares none of its tables or helpers. The two `base85`
//! family codecs differ in several incompatible ways:
//!
//! * `Z85` uses its own 85-character alphabet (above); `ASCII85` uses the
//!   contiguous range `'!'..='u'`.
//! * `Z85` has no zero-run compression; `ASCII85` collapses an all-zero group
//!   to the single sentinel `'z'` (and sometimes `'y'` for spaces).
//! * `Z85` permits no whitespace and no `<~`/`~>` framing; `ASCII85` ignores
//!   whitespace and may be wrapped in those delimiters.
//! * `Z85` requires the input length to be an exact multiple of 4 bytes (and
//!   decode input an exact multiple of 5 characters) with no short-tail
//!   padding rule; `ASCII85` supports partial trailing groups.
//! * `Z85` groups bytes `big-endian`; both treat each group as a `u32`, but
//!   the alphabets and tail handling make the encodings mutually unreadable.
//!
//! This file implements only `Z85`; it intentionally reproduces none of the
//! `ASCII85` features.
//!
//! All arithmetic here is integer `/`, `%`, and `big-endian` byte shuffling —
//! no floating point and no transcendental functions — so the reference is
//! bit-reproducible on any target.

use alloc::string::String;
use alloc::vec::Vec;

/// The canonical `Z85` alphabet (`ZeroMQ` `RFC` 32): `base85` digit value `i`
/// (`0..=84`) maps to `ALPHABET[i]`.
const ALPHABET: &[u8; 85] =
    b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#";

/// Sentinel stored for every `ASCII` byte that is not a member of the `Z85`
/// alphabet. Valid `base85` digit values are `0..=84`, so `-1` can never
/// collide with a legitimate entry.
const INVALID: i16 = -1;

/// `85^1` — the `radix` used for every grouping step.
const B1: u64 = 85;
/// `85^2`.
const B2: u64 = 85 * 85;
/// `85^3`.
const B3: u64 = 85 * 85 * 85;
/// `85^4`.
const B4: u64 = 85 * 85 * 85 * 85;
/// Largest value a single decoded group may hold (`u32::MAX`). `85^5`
/// (`4_437_053_125`) exceeds `2^32`, so a five-character group can encode a
/// value too large to be a valid `u32`; such groups must be rejected.
const MAX_GROUP: u64 = 0xFFFF_FFFF;

/// Builds the 256-entry reverse-lookup table (`ASCII byte -> base85 digit`).
/// Every byte not present in [`ALPHABET`] maps to [`INVALID`].
const fn build_decode_lut() -> [i16; 256] {
    let mut lut = [INVALID; 256];
    let mut i = 0usize;
    while i < 85 {
        lut[ALPHABET[i] as usize] = i as i16;
        i += 1;
    }
    lut
}

/// Reverse-lookup table mapping each possible input byte to its `base85` digit
/// value, or [`INVALID`] when the byte is not part of the alphabet.
const DECODE_LUT: [i16; 256] = build_decode_lut();

/// Errors produced while encoding to or decoding from `Z85`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Z85Error {
    /// On encode, the input byte length was not a multiple of 4; on decode,
    /// the input character length was not a multiple of 5.
    InvalidLength,
    /// A decode input contained a byte that is not part of the `Z85`
    /// alphabet (including any whitespace).
    InvalidCharacter,
    /// A decoded five-character group represented a `base85` value greater
    /// than `u32::MAX` and therefore cannot be a valid `Z85` group.
    Overflow,
}

/// Encodes `data` as a `Z85` string.
///
/// The input length must be an exact multiple of 4 bytes; otherwise
/// [`Z85Error::InvalidLength`] is returned. Each 4-byte `big-endian` group
/// becomes exactly five `base85` characters, so the output length is
/// `data.len() / 4 * 5`. The empty input maps to the empty string.
pub fn encode(data: &[u8]) -> Result<String, Z85Error> {
    if !data.len().is_multiple_of(4) {
        return Err(Z85Error::InvalidLength);
    }

    let mut out = String::with_capacity(data.len() / 4 * 5);
    for chunk in data.chunks_exact(4) {
        let val = (u32::from(chunk[0]) << 24)
            | (u32::from(chunk[1]) << 16)
            | (u32::from(chunk[2]) << 8)
            | u32::from(chunk[3]);
        let val = u64::from(val);
        out.push(ALPHABET[((val / B4) % B1) as usize] as char);
        out.push(ALPHABET[((val / B3) % B1) as usize] as char);
        out.push(ALPHABET[((val / B2) % B1) as usize] as char);
        out.push(ALPHABET[((val / B1) % B1) as usize] as char);
        out.push(ALPHABET[(val % B1) as usize] as char);
    }
    Ok(out)
}

/// Decodes a `Z85` string back into the original bytes.
///
/// The input length must be an exact multiple of 5 characters; otherwise
/// [`Z85Error::InvalidLength`] is returned. Any character outside the `Z85`
/// alphabet yields [`Z85Error::InvalidCharacter`]. A five-character group whose
/// `base85` value exceeds `u32::MAX` yields [`Z85Error::Overflow`]. The empty
/// string maps to the empty vector.
pub fn decode(text: &str) -> Result<Vec<u8>, Z85Error> {
    let bytes = text.as_bytes();
    if !bytes.len().is_multiple_of(5) {
        return Err(Z85Error::InvalidLength);
    }

    let mut out = Vec::with_capacity(bytes.len() / 5 * 4);
    for chunk in bytes.chunks_exact(5) {
        let mut acc: u64 = 0;
        for &b in chunk {
            let digit = DECODE_LUT[b as usize];
            if digit == INVALID {
                return Err(Z85Error::InvalidCharacter);
            }
            acc = acc * B1 + digit as u64;
        }
        if acc > MAX_GROUP {
            return Err(Z85Error::Overflow);
        }
        out.push(((acc >> 24) & 0xFF) as u8);
        out.push(((acc >> 16) & 0xFF) as u8);
        out.push(((acc >> 8) & 0xFF) as u8);
        out.push((acc & 0xFF) as u8);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// The reference `"HelloWorld"` vector from the `ZeroMQ` `RFC` 32 spec.
    #[cfg(test)]
    const HELLO_BYTES: [u8; 8] = [0x86, 0x4F, 0xD2, 0x6F, 0xB5, 0x59, 0xF7, 0x5B];

    #[test]
    fn alphabet_has_85_unique_chars() {
        let mut seen = [false; 256];
        for &c in ALPHABET.iter() {
            assert!(!seen[c as usize], "duplicate alphabet char");
            seen[c as usize] = true;
        }
        assert_eq!(ALPHABET.len(), 85);
    }

    #[test]
    fn lut_round_trips_alphabet() {
        for (i, &c) in ALPHABET.iter().enumerate() {
            assert_eq!(DECODE_LUT[c as usize], i as i16);
        }
    }

    #[test]
    fn lut_marks_space_invalid() {
        assert_eq!(DECODE_LUT[b' ' as usize], INVALID);
    }

    #[test]
    fn lut_marks_quote_invalid() {
        assert_eq!(DECODE_LUT[b'"' as usize], INVALID);
    }

    #[test]
    fn lut_marks_backslash_invalid() {
        assert_eq!(DECODE_LUT[b'\\' as usize], INVALID);
    }

    #[test]
    fn radix_constants_are_correct() {
        assert_eq!(B1, 85);
        assert_eq!(B2, 7_225);
        assert_eq!(B3, 614_125);
        assert_eq!(B4, 52_200_625);
    }

    #[test]
    fn pow85_of_five_exceeds_u32() {
        assert_eq!(B4 * B1, 4_437_053_125);
        assert!(B4 * B1 > u64::from(u32::MAX));
    }

    #[test]
    fn encode_hello_world() {
        assert_eq!(encode(&HELLO_BYTES).unwrap(), "HelloWorld");
    }

    #[test]
    fn decode_hello_world() {
        assert_eq!(decode("HelloWorld").unwrap(), HELLO_BYTES.to_vec());
    }

    #[test]
    fn hello_world_round_trip() {
        let text = encode(&HELLO_BYTES).unwrap();
        assert_eq!(decode(&text).unwrap(), HELLO_BYTES.to_vec());
    }

    #[test]
    fn encode_empty_is_empty_string() {
        assert_eq!(encode(&[]).unwrap(), "");
    }

    #[test]
    fn decode_empty_is_empty_vec() {
        assert_eq!(decode("").unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn empty_round_trip() {
        assert_eq!(decode(&encode(&[]).unwrap()).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn encode_all_zero_group() {
        assert_eq!(encode(&[0x00, 0x00, 0x00, 0x00]).unwrap(), "00000");
    }

    #[test]
    fn decode_all_zero_group() {
        assert_eq!(decode("00000").unwrap(), vec![0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn zero_group_round_trip() {
        let bytes = [0x00, 0x00, 0x00, 0x00];
        assert_eq!(decode(&encode(&bytes).unwrap()).unwrap(), bytes.to_vec());
    }

    #[test]
    fn encode_all_ff_group_round_trips() {
        let bytes = [0xFF, 0xFF, 0xFF, 0xFF];
        let text = encode(&bytes).unwrap();
        assert_eq!(decode(&text).unwrap(), bytes.to_vec());
    }

    #[test]
    fn all_ff_group_encodes_to_max_string() {
        // 0xFFFF_FFFF is the largest valid group value; its Z85 form is the
        // canonical "%nSc0".
        assert_eq!(encode(&[0xFF, 0xFF, 0xFF, 0xFF]).unwrap(), "%nSc0");
    }

    #[test]
    fn decode_max_valid_string() {
        assert_eq!(decode("%nSc0").unwrap(), vec![0xFF, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn encode_length_not_multiple_of_four() {
        assert_eq!(encode(&[1, 2, 3]), Err(Z85Error::InvalidLength));
    }

    #[test]
    fn encode_length_one_is_invalid() {
        assert_eq!(encode(&[0]), Err(Z85Error::InvalidLength));
    }

    #[test]
    fn encode_length_five_is_invalid() {
        assert_eq!(encode(&[0, 1, 2, 3, 4]), Err(Z85Error::InvalidLength));
    }

    #[test]
    fn decode_length_not_multiple_of_five() {
        assert_eq!(decode("ABCD"), Err(Z85Error::InvalidLength));
    }

    #[test]
    fn decode_length_one_is_invalid() {
        assert_eq!(decode("A"), Err(Z85Error::InvalidLength));
    }

    #[test]
    fn decode_length_six_is_invalid() {
        assert_eq!(decode("000000"), Err(Z85Error::InvalidLength));
    }

    #[test]
    fn decode_space_is_invalid_character() {
        assert_eq!(decode("Hell "), Err(Z85Error::InvalidCharacter));
    }

    #[test]
    fn decode_tab_is_invalid_character() {
        assert_eq!(decode("Hell\t"), Err(Z85Error::InvalidCharacter));
    }

    #[test]
    fn decode_newline_is_invalid_character() {
        assert_eq!(decode("Hell\n"), Err(Z85Error::InvalidCharacter));
    }

    #[test]
    fn decode_quote_is_invalid_character() {
        assert_eq!(decode("Hell\""), Err(Z85Error::InvalidCharacter));
    }

    #[test]
    fn decode_backslash_is_invalid_character() {
        assert_eq!(decode("Hell\\"), Err(Z85Error::InvalidCharacter));
    }

    #[test]
    fn decode_non_ascii_is_invalid_character() {
        // 'Hel' plus the two-byte UTF-8 encoding of 'é' (0xC3 0xA9) is five
        // bytes, neither of which is in the Z85 alphabet.
        assert_eq!(decode("Hel\u{00e9}"), Err(Z85Error::InvalidCharacter));
    }

    #[test]
    fn invalid_character_checked_before_overflow() {
        // Length is a multiple of five and would be an overflow group, but the
        // space must be reported as an invalid character first.
        assert_eq!(decode("%nSc "), Err(Z85Error::InvalidCharacter));
    }

    #[test]
    fn decode_just_above_max_overflows() {
        // "%nSc1" == 0xFFFF_FFFF + 1 == 2^32, one past the largest valid group.
        assert_eq!(decode("%nSc1"), Err(Z85Error::Overflow));
    }

    #[test]
    fn decode_all_hash_overflows() {
        // "#####" == 85^5 - 1 == 4_437_053_124, far above u32::MAX.
        assert_eq!(decode("#####"), Err(Z85Error::Overflow));
    }

    #[test]
    fn overflow_boundary_is_exact() {
        // "%nSc0" is valid, "%nSc1" overflows: the boundary sits exactly at
        // u32::MAX.
        assert!(decode("%nSc0").is_ok());
        assert_eq!(decode("%nSc1"), Err(Z85Error::Overflow));
    }

    #[test]
    fn overflow_in_second_group_detected() {
        // First group is a valid all-zero group; the second overflows.
        assert_eq!(decode("00000#####"), Err(Z85Error::Overflow));
    }

    #[test]
    fn encode_two_groups_concatenates() {
        let bytes = [0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF];
        assert_eq!(encode(&bytes).unwrap(), "00000%nSc0");
    }

    #[test]
    fn decode_two_groups_round_trip() {
        let bytes = [0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF];
        let text = encode(&bytes).unwrap();
        assert_eq!(decode(&text).unwrap(), bytes.to_vec());
    }

    #[test]
    fn encode_output_length_is_five_quarters() {
        let bytes = [0u8; 16];
        assert_eq!(encode(&bytes).unwrap().len(), 20);
    }

    #[test]
    fn single_byte_increments_round_trip() {
        for b in 0u8..=255 {
            let bytes = [b, b, b, b];
            let text = encode(&bytes).unwrap();
            assert_eq!(text.len(), 5);
            assert_eq!(decode(&text).unwrap(), bytes.to_vec());
        }
    }

    #[test]
    fn big_endian_ordering_is_respected() {
        // 0x00_00_00_01 == value 1 -> digits 0,0,0,0,1 -> "00001".
        assert_eq!(encode(&[0x00, 0x00, 0x00, 0x01]).unwrap(), "00001");
        // 0x01_00_00_00 is far larger and must differ.
        assert_ne!(
            encode(&[0x01, 0x00, 0x00, 0x00]).unwrap(),
            encode(&[0x00, 0x00, 0x00, 0x01]).unwrap()
        );
    }

    #[test]
    fn value_one_decodes_to_single_low_byte() {
        assert_eq!(decode("00001").unwrap(), vec![0x00, 0x00, 0x00, 0x01]);
    }

    #[test]
    fn every_alphabet_digit_decodes() {
        // A group of five identical highest-index-safe digits must decode
        // without error as long as it stays within u32 range: value 0 group.
        assert_eq!(decode("00000").unwrap(), vec![0, 0, 0, 0]);
    }

    #[test]
    fn deterministic_pseudo_random_round_trip() {
        // Deterministic xorshift-style stream, length a multiple of 4.
        let mut state: u32 = 0x1234_5678;
        let mut bytes = Vec::new();
        for _ in 0..256 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            bytes.push((state & 0xFF) as u8);
        }
        assert!(bytes.len().is_multiple_of(4));
        let text = encode(&bytes).unwrap();
        assert_eq!(text.len(), bytes.len() / 4 * 5);
        assert_eq!(decode(&text).unwrap(), bytes);
    }

    #[test]
    fn large_all_zero_buffer_round_trip() {
        let bytes = vec![0u8; 400];
        let text = encode(&bytes).unwrap();
        assert_eq!(decode(&text).unwrap(), bytes);
    }

    #[test]
    fn large_all_ff_buffer_round_trip() {
        let bytes = vec![0xFFu8; 400];
        let text = encode(&bytes).unwrap();
        assert_eq!(decode(&text).unwrap(), bytes);
    }

    #[test]
    fn sequential_byte_pattern_round_trip() {
        let mut bytes = Vec::new();
        for i in 0..128u32 {
            bytes.push((i & 0xFF) as u8);
        }
        assert!(bytes.len().is_multiple_of(4));
        let text = encode(&bytes).unwrap();
        assert_eq!(decode(&text).unwrap(), bytes);
    }

    #[test]
    fn encode_rejects_before_processing_short_tail() {
        // Seven bytes: multiple-of-four check must fail even though the first
        // four bytes form a valid group.
        assert_eq!(encode(&[1, 2, 3, 4, 5, 6, 7]), Err(Z85Error::InvalidLength));
    }

    #[test]
    fn decode_rejects_before_processing_short_tail() {
        // Nine characters: multiple-of-five check must fail.
        assert_eq!(decode("000000000"), Err(Z85Error::InvalidLength));
    }

    #[test]
    fn max_u32_value_boundary_bytes() {
        // Explicit u32::MAX byte pattern must survive a round trip and equal
        // the canonical max string.
        let bytes = (u32::MAX).to_be_bytes();
        assert_eq!(encode(&bytes).unwrap(), "%nSc0");
        assert_eq!(decode("%nSc0").unwrap(), bytes.to_vec());
    }
}
