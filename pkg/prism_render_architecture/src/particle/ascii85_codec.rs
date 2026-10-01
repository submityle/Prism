//! `Ascii85`/`Base85` binary-to-text codec (the Adobe / `btoa` variant),
//! implemented with pure integer `radix`-85 arithmetic on the `CPU`.
//!
//! `Ascii85` reinterprets a byte stream as a sequence of fixed-size 4-byte
//! groups. Each group is read as one `big-endian` `u32` and rewritten in base
//! 85 as exactly five digits, obtained by repeatedly dividing by 85
//! (`value / 85`, `value % 85`). Every digit `0..=84` is biased by 33 into the
//! printable `ASCII` range `'!'..='u'`. Four input bytes therefore become five
//! output characters, so the text is only 25% larger than the raw bytes —
//! denser than `base64`'s 33% expansion.
//!
//! Two special rules complete the Adobe convention:
//!
//! * **Zero compression (`'z'`)**: a *complete* 4-byte group whose value is 0
//!   is written as the single character `'z'` instead of `"!!!!!"`. This only
//!   applies to full groups; a partial trailing group of zero bytes is never
//!   compressed.
//! * **Short tail**: when the input length is not a multiple of 4, the final
//!   1, 2 or 3 bytes are zero-padded to a full 4-byte group, encoded, and then
//!   only the first `(tail_len + 1)` characters are emitted. Decoding reverses
//!   this: a trailing group of `n` characters (`2..=4`) is padded with the
//!   maximum digit (`'u'`, value 84) to five digits and yields `(n - 1)`
//!   bytes. A lone trailing character is impossible and is rejected.
//!
//! The optional Adobe stream delimiters `<~` (prefix) and `~>` (suffix) are
//! stripped by [`decode`] if present, but [`encode`] emits only the bare body.
//!
//! ## Boundaries
//!
//! `Ascii85`/`Base85` is a fixed-length *4-byte-group `radix`-85 conversion*
//! and is deliberately distinct from the other binary-to-text families that
//! live beside it in this crate:
//!
//! * `base64` ([`super::base64`]) is a `radix`-64 **bit** re-grouping: 3 bytes
//!   (24 bits) map to 4 characters of 6 bits each. There is no arithmetic
//!   division and no `'z'`-style zero compression.
//! * `base58` ([`super::base58_codec`]) is an *arbitrary-precision* big-number
//!   base conversion over the whole message (long division of the full
//!   integer), not an independent per-group transform, and uses a different
//!   `radix` and alphabet.
//! * `base32`/`base16` (e.g. [`super::base32_rfc4648`]) are pure power-of-two
//!   **bit** groupings (5-bit / 4-bit windows) with no `radix` arithmetic.
//!
//! Only `Ascii85` chops the stream into independent fixed 4-byte groups and
//! converts each one through genuine base-85 integer division; that group
//! structure, the `'z'` zero sentinel, and the five-character width are the
//! defining differences from every neighbour above.
//!
//! All arithmetic here is integer `/` and `%` (plus `big-endian` byte
//! shuffling) — no floating point and no transcendental functions — so the
//! reference is bit-reproducible on any target.

use alloc::string::String;
use alloc::vec::Vec;

/// Errors produced while decoding an `Ascii85`/`Base85` string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ascii85Error {
    /// A byte outside the valid body alphabet was encountered. Valid body
    /// characters are `'!'..='u'` (plus the `'z'` sentinel at group
    /// boundaries and the optional `<~`/`~>` delimiters).
    InvalidChar(char),
    /// The zero sentinel `'z'` appeared in the middle of a 5-character group
    /// instead of on a group boundary.
    ZInMiddle,
    /// The input ended with a trailing group of a single character, which can
    /// never represent any number of bytes under the short-tail rule.
    TrailingSingleChar,
    /// A 5-character group (or a padded short tail) decoded to a value larger
    /// than `u32::MAX`, so it cannot be a valid `Ascii85` group.
    Overflow,
}

/// Lowest valid body character (`'!'`, digit value 0).
const LOW: u8 = b'!';
/// Highest valid body character (`'u'`, digit value 84).
const HIGH: u8 = b'u';
/// Maximum base-85 digit value (`'u' - '!'`).
const MAX_DIGIT: u8 = 84;

/// Encodes an arbitrary byte slice into a bare `Ascii85`/`Base85` string
/// (no `<~`/`~>` delimiters).
///
/// Full zero groups are compressed to `'z'`; a short tail of 1..=3 bytes is
/// zero-padded, encoded, and truncated to `tail_len + 1` characters.
pub fn encode(data: &[u8]) -> String {
    let mut out = String::new();
    for chunk in data.chunks(4) {
        if chunk.len() == 4 {
            let value = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if value == 0 {
                // Adobe zero compression: a full all-zero group is `'z'`.
                out.push('z');
            } else {
                encode_group(value, 5, &mut out);
            }
        } else {
            // Short tail: zero-pad to a full group, emit `len + 1` chars.
            let mut bytes = [0u8; 4];
            bytes[..chunk.len()].copy_from_slice(chunk);
            let value = u32::from_be_bytes(bytes);
            encode_group(value, chunk.len() + 1, &mut out);
        }
    }
    out
}

/// Writes the first `count` base-85 digits of `value` (as biased `ASCII`
/// characters) into `out`. `value` is split into five digits most-significant
/// first via repeated division by 85.
fn encode_group(value: u32, count: usize, out: &mut String) {
    let mut n = value;
    let mut digits = [0u8; 5];
    for slot in digits.iter_mut().rev() {
        *slot = (n % 85) as u8;
        n /= 85;
    }
    for &d in digits.iter().take(count) {
        out.push((d + LOW) as char);
    }
}

/// Decodes a bare or delimited `Ascii85`/`Base85` string back into bytes.
///
/// Handles `'z'` expansion, the short-tail rule, and optional `<~`/`~>`
/// delimiters. Returns an [`Ascii85Error`] for any malformed input rather than
/// guessing.
pub fn decode(s: &str) -> Result<Vec<u8>, Ascii85Error> {
    let body = strip_delimiters(s.as_bytes());
    let mut out = Vec::new();
    let mut group = [0u8; 5];
    let mut count = 0usize;

    for &b in body {
        if b == b'z' {
            if count != 0 {
                return Err(Ascii85Error::ZInMiddle);
            }
            out.extend_from_slice(&[0, 0, 0, 0]);
            continue;
        }
        if !(LOW..=HIGH).contains(&b) {
            return Err(Ascii85Error::InvalidChar(b as char));
        }
        group[count] = b - LOW;
        count += 1;
        if count == 5 {
            let value = group_to_u32(&group)?;
            out.extend_from_slice(&value.to_be_bytes());
            count = 0;
        }
    }

    if count == 1 {
        return Err(Ascii85Error::TrailingSingleChar);
    }
    if count > 1 {
        // Pad the short tail with the maximum digit before converting, then
        // keep only `count - 1` leading bytes.
        for slot in group.iter_mut().skip(count) {
            *slot = MAX_DIGIT;
        }
        let value = group_to_u32(&group)?;
        let bytes = value.to_be_bytes();
        out.extend_from_slice(&bytes[..count - 1]);
    }

    Ok(out)
}

/// Converts five base-85 digits (most-significant first) into a `u32`,
/// rejecting any combination whose value exceeds `u32::MAX`.
fn group_to_u32(digits: &[u8; 5]) -> Result<u32, Ascii85Error> {
    let mut acc: u64 = 0;
    for &d in digits.iter() {
        acc = acc * 85 + d as u64;
    }
    if acc > u32::MAX as u64 {
        return Err(Ascii85Error::Overflow);
    }
    Ok(acc as u32)
}

/// Strips optional Adobe `<~` prefix and `~>` suffix delimiters, independently.
fn strip_delimiters(bytes: &[u8]) -> &[u8] {
    let mut b = bytes;
    if b.starts_with(b"<~") {
        b = &b[2..];
    }
    if b.ends_with(b"~>") {
        b = &b[..b.len() - 2];
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic linear-congruential generator for property tests
    /// (keeps the test suite self-contained, no external crates).
    #[cfg(test)]
    fn next_rand(state: &mut u64) -> u8 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as u8
    }

    // --- Authoritative reference vectors -------------------------------------

    #[test]
    fn encode_man_reference_vector() {
        // "Man " = 0x4D616E20; the classic `Ascii85` first group.
        assert_eq!(encode(b"Man "), "9jqo^");
    }

    #[test]
    fn decode_man_reference_vector() {
        assert_eq!(decode("9jqo^").unwrap(), b"Man ".to_vec());
    }

    #[test]
    fn encode_full_zero_group_is_z() {
        assert_eq!(encode(&[0, 0, 0, 0]), "z");
    }

    #[test]
    fn decode_z_expands_to_four_zeros() {
        assert_eq!(decode("z").unwrap(), vec![0, 0, 0, 0]);
    }

    #[test]
    fn encode_empty_is_empty() {
        assert_eq!(encode(&[]), "");
    }

    #[test]
    fn decode_empty_is_empty() {
        assert_eq!(decode("").unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn encode_ffffffff_reference_vector() {
        // 0xFFFFFFFF is the classic maximal group `s8W-!`.
        assert_eq!(encode(&[0xFF, 0xFF, 0xFF, 0xFF]), "s8W-!");
    }

    #[test]
    fn decode_ffffffff_reference_vector() {
        assert_eq!(decode("s8W-!").unwrap(), vec![0xFF, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn encode_sure_reference_vector() {
        // "sure" = [115, 117, 114, 101] -> "F*2M7".
        assert_eq!(encode(b"sure"), "F*2M7");
    }

    #[test]
    fn decode_sure_reference_vector() {
        assert_eq!(decode("F*2M7").unwrap(), b"sure".to_vec());
    }

    #[test]
    fn encode_hello_vector() {
        assert_eq!(encode(b"Hello"), "87cURDZ");
    }

    #[test]
    fn decode_hello_vector() {
        assert_eq!(decode("87cURDZ").unwrap(), b"Hello".to_vec());
    }

    #[test]
    fn roundtrip_hello() {
        let data = b"Hello";
        assert_eq!(decode(&encode(data)).unwrap(), data.to_vec());
    }

    // --- Short-tail (1/2/3 trailing bytes) rules -----------------------------

    #[test]
    fn encode_one_trailing_byte_emits_two_chars() {
        let enc = encode(&[0x4D]);
        assert_eq!(enc.len(), 2);
    }

    #[test]
    fn encode_two_trailing_bytes_emits_three_chars() {
        let enc = encode(&[0x4D, 0x61]);
        assert_eq!(enc.len(), 3);
    }

    #[test]
    fn encode_three_trailing_bytes_emits_four_chars() {
        let enc = encode(&[0x4D, 0x61, 0x6E]);
        assert_eq!(enc.len(), 4);
    }

    #[test]
    fn roundtrip_one_trailing_byte() {
        let data = [0x4D];
        assert_eq!(decode(&encode(&data)).unwrap(), data.to_vec());
    }

    #[test]
    fn roundtrip_two_trailing_bytes() {
        let data = [0x4D, 0x61];
        assert_eq!(decode(&encode(&data)).unwrap(), data.to_vec());
    }

    #[test]
    fn roundtrip_three_trailing_bytes() {
        let data = [0x4D, 0x61, 0x6E];
        assert_eq!(decode(&encode(&data)).unwrap(), data.to_vec());
    }

    // --- Partial zero tails must NOT compress to 'z' -------------------------

    #[test]
    fn encode_single_zero_byte_not_z() {
        assert_eq!(encode(&[0]), "!!");
    }

    #[test]
    fn encode_two_zero_bytes_not_z() {
        assert_eq!(encode(&[0, 0]), "!!!");
    }

    #[test]
    fn encode_three_zero_bytes_not_z() {
        assert_eq!(encode(&[0, 0, 0]), "!!!!");
    }

    #[test]
    fn encode_five_zero_bytes_mixes_z_and_tail() {
        // One full zero group ('z') plus a 1-byte zero tail ("!!").
        assert_eq!(encode(&[0, 0, 0, 0, 0]), "z!!");
    }

    #[test]
    fn roundtrip_five_zero_bytes() {
        let data = [0, 0, 0, 0, 0];
        assert_eq!(decode(&encode(&data)).unwrap(), data.to_vec());
    }

    // --- 'z' sentinel behaviour ---------------------------------------------

    #[test]
    fn multiple_z_expands_to_many_zeros() {
        assert_eq!(decode("zz").unwrap(), vec![0; 8]);
    }

    #[test]
    fn z_at_boundary_then_group() {
        let mut expected = vec![0, 0, 0, 0];
        expected.extend_from_slice(b"Man ");
        assert_eq!(decode("z9jqo^").unwrap(), expected);
    }

    #[test]
    fn z_in_middle_is_rejected() {
        // "9j" starts a group, then 'z' arrives off-boundary.
        assert_eq!(decode("9jz").unwrap_err(), Ascii85Error::ZInMiddle);
    }

    #[test]
    fn z_after_one_char_is_rejected() {
        assert_eq!(decode("!z").unwrap_err(), Ascii85Error::ZInMiddle);
    }

    // --- Error handling ------------------------------------------------------

    #[test]
    fn trailing_single_char_is_rejected() {
        assert_eq!(decode("!").unwrap_err(), Ascii85Error::TrailingSingleChar);
    }

    #[test]
    fn trailing_single_char_after_full_group() {
        assert_eq!(
            decode("9jqo^9").unwrap_err(),
            Ascii85Error::TrailingSingleChar
        );
    }

    #[test]
    fn invalid_char_above_u_is_rejected() {
        // 'v' (118) is just past 'u' (117).
        assert_eq!(decode("v").unwrap_err(), Ascii85Error::InvalidChar('v'));
    }

    #[test]
    fn invalid_char_tilde_is_rejected() {
        assert_eq!(decode("9jqo~").unwrap_err(), Ascii85Error::InvalidChar('~'));
    }

    #[test]
    fn invalid_char_space_is_rejected() {
        // Space (32) is below '!' (33).
        assert_eq!(
            decode("9jq o^").unwrap_err(),
            Ascii85Error::InvalidChar(' ')
        );
    }

    #[test]
    fn overflow_all_u_group_is_rejected() {
        // "uuuuu" = 85^5 - 1 = 4_437_053_124 > u32::MAX.
        assert_eq!(decode("uuuuu").unwrap_err(), Ascii85Error::Overflow);
    }

    #[test]
    fn overflow_just_above_max_is_rejected() {
        // "s8X-!" is one group digit above 0xFFFFFFFF's "s8W-!".
        assert_eq!(decode("s8X-!").unwrap_err(), Ascii85Error::Overflow);
    }

    // --- Delimiter stripping -------------------------------------------------

    #[test]
    fn decode_strips_adobe_delimiters() {
        assert_eq!(decode("<~9jqo^~>").unwrap(), b"Man ".to_vec());
    }

    #[test]
    fn decode_empty_delimited_is_empty() {
        assert_eq!(decode("<~~>").unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn encode_emits_no_delimiters() {
        let enc = encode(b"Man ");
        assert!(!enc.starts_with("<~"));
        assert!(!enc.ends_with("~>"));
    }

    // --- Exhaustive / property round-trips -----------------------------------

    #[test]
    fn roundtrip_all_lengths_0_to_256() {
        for len in 0..=256usize {
            let data: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();
            let enc = encode(&data);
            let dec = decode(&enc).unwrap();
            assert_eq!(dec, data);
        }
    }

    #[test]
    fn roundtrip_every_single_byte() {
        for b in 0u8..=255 {
            let data = [b];
            let dec = decode(&encode(&data)).unwrap();
            assert_eq!(dec, data.to_vec());
        }
    }

    #[test]
    fn roundtrip_random_blocks() {
        let mut state = 0x1234_5678_9abc_def0u64;
        for _ in 0..64 {
            let len = (next_rand(&mut state) as usize) % 300;
            let data: Vec<u8> = (0..len).map(|_| next_rand(&mut state)).collect();
            let enc = encode(&data);
            let dec = decode(&enc).unwrap();
            assert_eq!(dec, data);
        }
    }

    #[test]
    fn roundtrip_large_zero_block_uses_only_z() {
        let data = vec![0u8; 100];
        let enc = encode(&data);
        assert_eq!(enc, "z".repeat(25));
        assert_eq!(decode(&enc).unwrap(), data);
    }

    #[test]
    fn roundtrip_binary_pattern() {
        let data: Vec<u8> = (0..=255u8).rev().collect();
        let dec = decode(&encode(&data)).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn decode_rejects_z_inside_delimited_group() {
        assert_eq!(decode("<~9jz~>").unwrap_err(), Ascii85Error::ZInMiddle);
    }
}
