//! `Base36` binary-to-text encoding over the lower-case alphabet
//! `0123456789abcdefghijklmnopqrstuvwxyz`, implemented with pure integer
//! arbitrary-precision arithmetic on the `CPU`.
//!
//! `Base36` reinterprets an arbitrary `big-endian` byte stream as a single
//! nonnegative integer and rewrites that integer in `radix` 36, emitting one
//! `ASCII` character per `base36` digit. Digit values `0..=9` map to the
//! `ASCII` digits `0`-`9` and digit values `10..=35` map to the lower-case
//! letters `a`-`z`. The compact, URL-safe output makes `Base36` a common
//! choice for short identifiers and short links.
//!
//! Encoding emits lower-case only. Decoding is case-insensitive, so the upper
//! -case letter `A` and the lower-case letter `a` both decode to the digit
//! value 10.
//!
//! ## Boundaries
//!
//! This is a deliberately separate implementation from the sibling bit-group
//! codecs in this crate — `base64`, `base32_rfc4648`, and `ascii85_codec`.
//! Those codecs slice the input into fixed-width bit groups (6, 5, or a
//! `base85` quantum of four bytes to five characters) and map each group
//! independently to one output symbol; encoding is a local, position-wise
//! `shift`/`mask` operation and the output length is a fixed, exact function
//! of the input length.
//!
//! `Base36` is fundamentally different: like the sibling `base58_codec` and
//! `base45_codec` radix converters, it performs an arbitrary-precision
//! `radix` conversion. There is no fixed bit grouping and no padding. The
//! whole input is one `Base256` integer that is repeatedly divided by 36 by
//! classic `big-integer` long division (done here on a `big-endian` `Vec<u8>`
//! using nothing but integer `+`, `*`, `/`, and `%`), collecting remainders
//! as `base36` digits. Because division mixes every byte into every digit,
//! output length is not a closed-form function of input length, and a single
//! changed input byte can change the entire output.
//!
//! `Base36` differs from `base58_codec` in both alphabet and `radix`: it uses
//! all ten digits and all twenty-six letters (`radix` 36) rather than the
//! ambiguity-avoiding `Bitcoin` set (`radix` 58), and it is case-insensitive
//! on decode. It differs from `base45_codec` (`radix` 45, pair-oriented) by
//! using a strict single-digit long division with no two-character grouping.
//!
//! ## Leading zeros
//!
//! Leading zero bytes need special handling precisely because they vanish
//! under integer division (a leading `0x00` contributes nothing to the
//! numeric value). `Base36` therefore follows the same convention as the
//! sibling `base58_codec`: each leading zero byte is encoded as one explicit
//! leading `0` character (`0` is the digit for value zero), and on decode each
//! leading `0` character is restored to a leading zero byte. The empty input
//! maps to the empty string.
//!
//! ## Convenience helpers
//!
//! [`u64_to_base36`] and [`base36_to_u64`] offer a fixed-width fast path for
//! callers that only need a single machine word. [`base36_to_u64`] reports
//! [`Base36Error::Overflow`] when the decoded value would exceed [`u64::MAX`];
//! the arbitrary-precision [`decode`] never overflows.
//!
//! All arithmetic is integer only — no floating point and no transcendental
//! functions — so this reference is bit-reproducible on any target.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

/// The lower-case `Base36` alphabet: digit value `i` (`0..=35`) maps to
/// `ALPHABET[i]`.
const ALPHABET: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";

/// The conversion `radix` used throughout this module.
const RADIX: u32 = 36;

/// The number of leading digit entries in [`ALPHABET`] (`0`-`9`); the letters
/// `a`-`z` begin at this index.
const LETTER_START: usize = 10;

/// Sentinel stored for every `ASCII` byte that is not a member of the
/// alphabet. Valid `base36` digit values are `0..=35`, so `-1` can never
/// collide with a legitimate entry.
const INVALID: i8 = -1;

/// Builds the 128-entry reverse-lookup table (`ASCII byte -> base36 digit`).
///
/// Both letter cases are registered so decoding is case-insensitive: the
/// lower-case letters come straight from [`ALPHABET`], and each corresponding
/// upper-case letter is added by subtracting the fixed `ASCII` case gap of 32.
/// Every `ASCII` byte with no `base36` meaning maps to [`INVALID`].
const fn build_decode_lut() -> [i8; 128] {
    let mut lut = [INVALID; 128];
    let mut i = 0usize;
    while i < 36 {
        lut[ALPHABET[i] as usize] = i as i8;
        i += 1;
    }
    // Register the upper-case letters A-Z as aliases of a-z.
    let mut j = LETTER_START;
    while j < 36 {
        let upper = ALPHABET[j] - 32;
        lut[upper as usize] = j as i8;
        j += 1;
    }
    lut
}

/// Reverse lookup from `ASCII` byte to `base36` digit value.
const DECODE_LUT: [i8; 128] = build_decode_lut();

/// Error returned when a `Base36` input cannot be decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Base36Error {
    /// The input contained a character that is not a `base36` digit in either
    /// letter case. The offending character is carried for diagnostics.
    InvalidChar(char),
    /// The decoded value exceeded [`u64::MAX`]. Only [`base36_to_u64`] can
    /// report this; the arbitrary-precision [`decode`] never overflows.
    Overflow,
}

/// Divides the `big-endian` arbitrary-precision integer in `num` by [`RADIX`]
/// in place, returning the remainder (the next `base36` digit, `0..=35`).
///
/// This is one pass of schoolbook long division: the running remainder is
/// shifted left by one `Base256` place (multiplied by 256) and combined with
/// each successive byte, writing the quotient byte back in place.
fn divmod_radix(num: &mut [u8]) -> u8 {
    let mut remainder: u32 = 0;
    for byte in num.iter_mut() {
        let acc = remainder * 256 + (*byte as u32);
        *byte = (acc / RADIX) as u8;
        remainder = acc % RADIX;
    }
    remainder as u8
}

/// Encodes an arbitrary `big-endian` byte slice as a lower-case `Base36`
/// string.
///
/// Leading zero bytes are emitted as leading `0` characters; the remaining
/// value is converted by repeated long division by 36. The empty slice
/// encodes to the empty string.
pub fn encode(data: &[u8]) -> String {
    let zeros = data.iter().take_while(|&&b| b == 0).count();

    // Big-endian working copy consumed by repeated long division. Each pass
    // extracts one base36 digit (the remainder) into `digits`.
    let mut num: Vec<u8> = data.to_vec();
    let mut digits: Vec<u8> = Vec::new();
    while num.iter().any(|&b| b != 0) {
        digits.push(divmod_radix(&mut num));
    }

    let mut out = String::with_capacity(zeros + digits.len());
    for _ in 0..zeros {
        out.push('0');
    }
    for &d in digits.iter().rev() {
        out.push(ALPHABET[d as usize] as char);
    }
    out
}

/// Decodes a case-insensitive `Base36` string back into the original
/// `big-endian` byte slice.
///
/// Leading `0` characters are restored as leading zero bytes; the remaining
/// digits are folded into a `Base256` accumulator by repeated multiply-add.
/// The empty string decodes to the empty vector.
pub fn decode(s: &str) -> Result<Vec<u8>, Base36Error> {
    let zeros = s.chars().take_while(|&c| c == '0').count();

    // Little-endian Base256 accumulator; each base36 digit multiplies the
    // running value by 36 and adds the digit.
    let mut result: Vec<u8> = Vec::new();
    for c in s.chars() {
        let value = decode_digit(c)?;
        let mut carry = value as u32;
        for b in result.iter_mut() {
            carry += (*b as u32) * RADIX;
            *b = (carry % 256) as u8;
            carry /= 256;
        }
        while carry > 0 {
            result.push((carry % 256) as u8);
            carry /= 256;
        }
    }

    let mut out: Vec<u8> = alloc::vec![0u8; zeros];
    out.reserve(result.len());
    out.extend(result.iter().rev());
    Ok(out)
}

/// Encodes a single [`u64`] as lower-case `Base36`.
///
/// Zero encodes to the single character `0`. This is the fixed-width fast
/// path and never allocates a working integer buffer.
pub fn u64_to_base36(mut value: u64) -> String {
    if value == 0 {
        return String::from("0");
    }
    let mut digits: Vec<u8> = Vec::new();
    while value > 0 {
        digits.push((value % RADIX as u64) as u8);
        value /= RADIX as u64;
    }
    let mut out = String::with_capacity(digits.len());
    for &d in digits.iter().rev() {
        out.push(ALPHABET[d as usize] as char);
    }
    out
}

/// Decodes a case-insensitive `Base36` string into a [`u64`].
///
/// Returns [`Base36Error::InvalidChar`] for any non-`base36` character and
/// [`Base36Error::Overflow`] when the value would exceed [`u64::MAX`]. The
/// empty string decodes to 0.
pub fn base36_to_u64(s: &str) -> Result<u64, Base36Error> {
    let mut acc: u64 = 0;
    for c in s.chars() {
        let value = decode_digit(c)? as u64;
        acc = acc.checked_mul(RADIX as u64).ok_or(Base36Error::Overflow)?;
        acc = acc.checked_add(value).ok_or(Base36Error::Overflow)?;
    }
    Ok(acc)
}

/// Maps a single character to its `base36` digit value, accepting either
/// letter case and rejecting any character that is not a `base36` digit.
fn decode_digit(c: char) -> Result<u8, Base36Error> {
    let code = c as u32;
    if code < 128 {
        let value = DECODE_LUT[code as usize];
        if value != INVALID {
            return Ok(value as u8);
        }
    }
    Err(Base36Error::InvalidChar(c))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- u64_to_base36 reference vectors --------------------------------

    #[test]
    fn u64_to_base36_zero_is_zero_char() {
        assert_eq!(u64_to_base36(0), "0");
    }

    #[test]
    fn u64_to_base36_one_is_one() {
        assert_eq!(u64_to_base36(1), "1");
    }

    #[test]
    fn u64_to_base36_ten_is_a() {
        assert_eq!(u64_to_base36(10), "a");
    }

    #[test]
    fn u64_to_base36_thirtyfive_is_z() {
        assert_eq!(u64_to_base36(35), "z");
    }

    #[test]
    fn u64_to_base36_thirtysix_is_ten() {
        assert_eq!(u64_to_base36(36), "10");
    }

    #[test]
    fn u64_to_base36_1295_is_zz() {
        assert_eq!(u64_to_base36(1295), "zz");
    }

    #[test]
    fn u64_to_base36_1296_is_one_hundred() {
        assert_eq!(u64_to_base36(1296), "100");
    }

    #[test]
    fn u64_to_base36_max_round_trips() {
        let encoded = u64_to_base36(u64::MAX);
        assert_eq!(base36_to_u64(&encoded), Ok(u64::MAX));
    }

    // ---- base36_to_u64 reference vectors --------------------------------

    #[test]
    fn base36_to_u64_capital_z_is_35() {
        assert_eq!(base36_to_u64("Z"), Ok(35));
    }

    #[test]
    fn base36_to_u64_lower_z_is_35() {
        assert_eq!(base36_to_u64("z"), Ok(35));
    }

    #[test]
    fn base36_to_u64_ten_is_36() {
        assert_eq!(base36_to_u64("10"), Ok(36));
    }

    #[test]
    fn base36_to_u64_empty_is_zero() {
        assert_eq!(base36_to_u64(""), Ok(0));
    }

    #[test]
    fn base36_to_u64_zero_char_is_zero() {
        assert_eq!(base36_to_u64("0"), Ok(0));
    }

    #[test]
    fn base36_to_u64_mixed_case_matches_lower() {
        assert_eq!(base36_to_u64("1Z"), base36_to_u64("1z"));
    }

    #[test]
    fn base36_to_u64_overflow_is_reported() {
        // Fourteen `z` digits is 36^14 - 1, far beyond u64::MAX.
        assert_eq!(base36_to_u64("zzzzzzzzzzzzzz"), Err(Base36Error::Overflow));
    }

    #[test]
    fn base36_to_u64_rejects_invalid_char() {
        assert_eq!(base36_to_u64("1!2"), Err(Base36Error::InvalidChar('!')));
    }

    #[test]
    fn u64_base36_round_trip_many() {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..512 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            assert_eq!(base36_to_u64(&u64_to_base36(state)), Ok(state));
        }
    }

    // ---- encode -----------------------------------------------------------

    #[test]
    fn encode_empty_is_empty_string() {
        assert_eq!(encode(&[]), "");
    }

    #[test]
    fn encode_single_zero_byte_is_zero_char() {
        assert_eq!(encode(&[0x00]), "0");
    }

    #[test]
    fn encode_two_zero_bytes_is_two_zero_chars() {
        assert_eq!(encode(&[0x00, 0x00]), "00");
    }

    #[test]
    fn encode_five_zero_bytes() {
        assert_eq!(encode(&[0, 0, 0, 0, 0]), "00000");
    }

    #[test]
    fn encode_single_one_byte() {
        assert_eq!(encode(&[0x01]), "1");
    }

    #[test]
    fn encode_single_ff_is_73() {
        // 0xFF = 255 = 7 * 36 + 3.
        assert_eq!(encode(&[0xFF]), "73");
    }

    #[test]
    fn encode_eighteen_is_i() {
        // 0x12 = 18 maps to the single digit `i`.
        assert_eq!(encode(&[0x12]), "i");
    }

    #[test]
    fn encode_zero_then_one_is_zero_one() {
        assert_eq!(encode(&[0x00, 0x01]), "01");
    }

    #[test]
    fn encode_one_then_zero_is_74() {
        // 0x0100 = 256 = 7 * 36 + 4.
        assert_eq!(encode(&[0x01, 0x00]), "74");
    }

    #[test]
    fn encode_two_ff_is_1ekf() {
        // 0xFFFF = 65535 = ((1 * 36 + 14) * 36 + 20) * 36 + 15 -> 1 e k f.
        assert_eq!(encode(&[0xFF, 0xFF]), "1ekf");
    }

    #[test]
    fn encode_leading_zero_bytes_preserved() {
        // Two leading zero bytes become `00`, then 0x12 = 18 -> `i`.
        assert_eq!(encode(&[0x00, 0x00, 0x12]), "00i");
    }

    #[test]
    fn encode_matches_u64_helper_for_small_values() {
        for value in 1u64..=5000 {
            // Minimal big-endian bytes of `value` (the empty slice is the
            // zero case handled separately, so start at one).
            let mut be: Vec<u8> = Vec::new();
            let mut v = value;
            while v > 0 {
                be.insert(0, (v & 0xFF) as u8);
                v >>= 8;
            }
            assert_eq!(encode(&be), u64_to_base36(value));
        }
    }

    #[test]
    fn encode_output_is_lowercase_only() {
        let ramp: Vec<u8> = (0u16..=255).map(|v| v as u8).collect();
        let encoded = encode(&ramp);
        for ch in encoded.chars() {
            let is_digit = ch.is_ascii_digit();
            let is_lower = ch.is_ascii_lowercase();
            assert!(is_digit || is_lower);
        }
    }

    // ---- decode -----------------------------------------------------------

    #[test]
    fn decode_empty_is_empty_vec() {
        assert_eq!(decode(""), Ok(Vec::new()));
    }

    #[test]
    fn decode_zero_char_is_single_zero_byte() {
        assert_eq!(decode("0"), Ok(alloc::vec![0x00]));
    }

    #[test]
    fn decode_two_zero_chars_is_two_zero_bytes() {
        assert_eq!(decode("00"), Ok(alloc::vec![0x00, 0x00]));
    }

    #[test]
    fn decode_73_is_single_ff() {
        assert_eq!(decode("73"), Ok(alloc::vec![0xFF]));
    }

    #[test]
    fn decode_uppercase_equals_lowercase() {
        assert_eq!(decode("ZZ"), decode("zz"));
    }

    #[test]
    fn decode_leading_zero_chars_restored() {
        assert_eq!(decode("00i"), Ok(alloc::vec![0x00, 0x00, 0x12]));
    }

    #[test]
    fn decode_rejects_space() {
        assert_eq!(decode("ab cd"), Err(Base36Error::InvalidChar(' ')));
    }

    #[test]
    fn decode_rejects_punctuation() {
        assert_eq!(decode("1z!"), Err(Base36Error::InvalidChar('!')));
    }

    #[test]
    fn decode_rejects_non_ascii() {
        assert_eq!(
            decode("1\u{00e9}z"),
            Err(Base36Error::InvalidChar('\u{00e9}'))
        );
    }

    // ---- byte-array round trips ------------------------------------------

    #[test]
    fn round_trip_every_single_byte() {
        for value in 0u16..=255 {
            let byte = [value as u8];
            assert_eq!(decode(&encode(&byte)), Ok(byte.to_vec()));
        }
    }

    #[test]
    fn round_trip_all_lengths_of_ramp() {
        let full: Vec<u8> = (0u16..=255).map(|v| v as u8).collect();
        for len in 0..=full.len() {
            let slice = &full[..len];
            assert_eq!(decode(&encode(slice)), Ok(slice.to_vec()));
        }
    }

    #[test]
    fn round_trip_leading_zero_runs() {
        for leading in 0..=8usize {
            let mut data: Vec<u8> = alloc::vec![0u8; leading];
            data.extend_from_slice(&[0x12, 0x34, 0x56, 0x78]);
            let encoded = encode(&data);
            let zero_chars = encoded.chars().take_while(|&c| c == '0').count();
            assert_eq!(zero_chars, leading);
            assert_eq!(decode(&encoded), Ok(data));
        }
    }

    #[test]
    fn round_trip_lcg_random_blocks() {
        let mut state: u64 = 0x0123_4567_89AB_CDEF;
        for _ in 0..256 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let count = (state >> 58) as usize;
            let mut data: Vec<u8> = Vec::with_capacity(count);
            let mut inner = state;
            for _ in 0..count {
                inner = inner
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                data.push((inner >> 56) as u8);
            }
            assert_eq!(decode(&encode(&data)), Ok(data));
        }
    }

    #[test]
    fn round_trip_all_zero_block() {
        let data: Vec<u8> = alloc::vec![0u8; 64];
        let encoded = encode(&data);
        assert_eq!(encoded.len(), 64);
        assert_eq!(decode(&encoded), Ok(data));
    }

    #[test]
    fn round_trip_all_ff_block() {
        let data: Vec<u8> = alloc::vec![0xFFu8; 96];
        assert_eq!(decode(&encode(&data)), Ok(data));
    }

    #[test]
    fn round_trip_ramp_block() {
        let full: Vec<u8> = (0u16..=255).map(|v| v as u8).collect();
        let encoded = encode(&full);
        assert_eq!(decode(&encoded), Ok(full));
    }

    // ---- structural invariants -------------------------------------------

    #[test]
    fn alphabet_has_thirty_six_unique_bytes() {
        let mut seen = [false; 128];
        let mut unique = 0usize;
        for &b in ALPHABET.iter() {
            if !seen[b as usize] {
                seen[b as usize] = true;
                unique += 1;
            }
        }
        assert_eq!(unique, 36);
    }

    #[test]
    fn decode_digit_round_trips_whole_alphabet() {
        for (index, &b) in ALPHABET.iter().enumerate() {
            assert_eq!(decode_digit(b as char), Ok(index as u8));
        }
    }

    #[test]
    fn decode_digit_accepts_uppercase_letters() {
        for (index, &byte) in ALPHABET.iter().enumerate().take(36).skip(LETTER_START) {
            let upper = (byte - 32) as char;
            assert_eq!(decode_digit(upper), Ok(index as u8));
        }
    }

    #[test]
    fn error_is_clone_and_eq() {
        let err = Base36Error::InvalidChar('!');
        assert_eq!(err.clone(), Base36Error::InvalidChar('!'));
        assert_ne!(Base36Error::Overflow, Base36Error::InvalidChar('!'));
    }
}
