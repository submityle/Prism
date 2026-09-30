//! Standard `Base64` binary-to-text encoding (`RFC` 4648), including the
//! `URL`-safe alphabet variant, implemented with pure integer bit operations
//! on the `CPU`.
//!
//! `Base64` reinterprets an arbitrary byte stream as a sequence of 6-bit
//! groups and maps each group to one printable `ASCII` character, so binary
//! assets (mesh/texture blobs, `GPU` upload payloads, serialized parameter
//! buffers) can travel through text-only channels — `JSON`, `URL` query
//! strings, source headers — without corruption. Three input bytes (24 bits)
//! become exactly four output characters (4 x 6 bits); the tail of one or two
//! bytes is zero-extended to the next 6-bit boundary and, in the standard
//! form, completed with `=` padding.
//!
//! Two alphabets are provided:
//!
//! * Standard ([`encode_standard`] / [`decode_standard`]): `A-Za-z0-9+/`
//!   with trailing `=` padding, exactly as specified by `RFC` 4648 section 4.
//! * `URL`-safe ([`encode_url_safe`] / [`decode_url_safe`]): `A-Za-z0-9-_`
//!   as specified by `RFC` 4648 section 5. Following the common `URL`-safe
//!   convention this module emits *no* `=` padding, since `=` is itself a
//!   reserved character in query strings; the decoder infers the tail length
//!   from the number of significant characters and also tolerates padding if
//!   present.
//!
//! Decoding is strict: any character outside the selected alphabet, any
//! impossible group length, and (for the standard form) any wrong total length
//! or mismatched padding cause the decoder to return `None` rather than guess.
//!
//! ## Boundaries
//!
//! `Base64` is a lossless 6-bit *re-encoding*, not a codec that removes
//! redundancy or hides content. It is unrelated to the compression and
//! cryptography neighbours in this crate:
//!
//! * It never shrinks data. Output is always about 33% larger than input
//!   (4 characters per 3 bytes), so it is the opposite of the run-length and
//!   quantization primitives such as [`super::run_length_encode`].
//! * It provides no confidentiality or integrity. The mapping is a fixed,
//!   public table; anyone can decode it. Encryption and hashing live
//!   elsewhere.
//!
//! All arithmetic is integer shift/mask/or only — no floating point and no
//! transcendental functions — so this reference is bit-reproducible on any
//! target.

use alloc::string::String;
use alloc::vec::Vec;

/// Sentinel stored in a decode table for every byte that is **not** a member
/// of the corresponding alphabet. Valid sextet values are `0..=63`, so `0xFF`
/// can never collide with a legitimate entry.
const INVALID: u8 = 0xFF;

/// Builds a 64-entry encode table (`sextet -> ASCII byte`) for the shared
/// `A-Za-z0-9` prefix plus the two alphabet-specific characters `c62`/`c63`.
const fn build_encode_table(c62: u8, c63: u8) -> [u8; 64] {
    let mut table = [0u8; 64];
    let mut i = 0usize;
    while i < 26 {
        table[i] = b'A' + i as u8;
        i += 1;
    }
    let mut j = 0usize;
    while j < 26 {
        table[26 + j] = b'a' + j as u8;
        j += 1;
    }
    let mut k = 0usize;
    while k < 10 {
        table[52 + k] = b'0' + k as u8;
        k += 1;
    }
    table[62] = c62;
    table[63] = c63;
    table
}

/// Builds the 256-entry reverse-lookup table (`ASCII byte -> sextet`) for a
/// given encode table. Every byte not present in `enc` maps to [`INVALID`].
const fn build_decode_table(enc: &[u8; 64]) -> [u8; 256] {
    let mut table = [INVALID; 256];
    let mut i = 0usize;
    while i < 64 {
        table[enc[i] as usize] = i as u8;
        i += 1;
    }
    table
}

/// Standard alphabet encode table: `A-Za-z0-9+/`.
const STANDARD_ENCODE: [u8; 64] = build_encode_table(b'+', b'/');
/// `URL`-safe alphabet encode table: `A-Za-z0-9-_`.
const URL_SAFE_ENCODE: [u8; 64] = build_encode_table(b'-', b'_');
/// Reverse lookup for the standard alphabet.
const STANDARD_DECODE: [u8; 256] = build_decode_table(&STANDARD_ENCODE);
/// Reverse lookup for the `URL`-safe alphabet.
const URL_SAFE_DECODE: [u8; 256] = build_decode_table(&URL_SAFE_ENCODE);

/// Core encoder shared by both alphabets.
///
/// `table` selects the alphabet; `pad` controls whether the tail group is
/// completed with `=` characters (standard form) or left bare (`URL`-safe
/// convention).
fn encode_impl(data: &[u8], table: &[u8; 64], pad: bool) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        match *chunk {
            [a, b, c] => {
                out.push(table[(a >> 2) as usize] as char);
                out.push(table[(((a & 0x03) << 4) | (b >> 4)) as usize] as char);
                out.push(table[(((b & 0x0F) << 2) | (c >> 6)) as usize] as char);
                out.push(table[(c & 0x3F) as usize] as char);
            }
            [a, b] => {
                out.push(table[(a >> 2) as usize] as char);
                out.push(table[(((a & 0x03) << 4) | (b >> 4)) as usize] as char);
                out.push(table[((b & 0x0F) << 2) as usize] as char);
                if pad {
                    out.push('=');
                }
            }
            [a] => {
                out.push(table[(a >> 2) as usize] as char);
                out.push(table[((a & 0x03) << 4) as usize] as char);
                if pad {
                    out.push('=');
                    out.push('=');
                }
            }
            _ => {}
        }
    }
    out
}

/// Core decoder shared by both alphabets.
///
/// `table` is the reverse lookup; `strict_padding` requires the total length
/// to be a multiple of four with `=` padding that matches the tail (standard
/// form). When `strict_padding` is `false` (`URL`-safe form) padding is
/// optional and the tail length is inferred from the significant characters.
fn decode_impl(s: &str, table: &[u8; 256], strict_padding: bool) -> Option<Vec<u8>> {
    let bytes = s.as_bytes();
    let len = bytes.len();
    if len == 0 {
        return Some(Vec::new());
    }

    // Count contiguous trailing '=' padding; anything beyond two is illegal.
    let mut pad = 0usize;
    while pad < len && bytes[len - 1 - pad] == b'=' {
        pad += 1;
    }
    if pad > 2 {
        return None;
    }
    let body_len = len - pad;
    let rem = body_len % 4;
    // A trailing group of a single character carries only six bits and can
    // never round-trip from whole bytes.
    if rem == 1 {
        return None;
    }

    if strict_padding {
        if !len.is_multiple_of(4) {
            return None;
        }
        let expected_pad = match rem {
            0 => 0,
            2 => 2,
            3 => 1,
            _ => return None,
        };
        if pad != expected_pad {
            return None;
        }
    }

    let mut out = Vec::with_capacity(body_len / 4 * 3 + 2);
    let mut acc: u32 = 0;
    let mut nbits: u32 = 0;
    for &ch in &bytes[..body_len] {
        let value = table[ch as usize];
        if value == INVALID {
            return None;
        }
        acc = (acc << 6) | value as u32;
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            out.push((acc >> nbits) as u8);
            acc &= (1u32 << nbits) - 1;
        }
    }

    Some(out)
}

/// Encodes `data` with the standard `RFC` 4648 alphabet `A-Za-z0-9+/`,
/// emitting `=` padding so the output length is always a multiple of four.
///
/// The empty input yields the empty string.
#[must_use]
pub fn encode_standard(data: &[u8]) -> String {
    encode_impl(data, &STANDARD_ENCODE, true)
}

/// Encodes `data` with the `URL`-safe `RFC` 4648 alphabet `A-Za-z0-9-_`,
/// emitting **no** `=` padding (the common `URL`-safe convention).
///
/// The empty input yields the empty string.
#[must_use]
pub fn encode_url_safe(data: &[u8]) -> String {
    encode_impl(data, &URL_SAFE_ENCODE, false)
}

/// Decodes a standard `RFC` 4648 (`A-Za-z0-9+/`) string.
///
/// Returns `None` for any illegal character, any length that is not a multiple
/// of four, or padding that does not match the encoded tail.
#[must_use]
pub fn decode_standard(s: &str) -> Option<Vec<u8>> {
    decode_impl(s, &STANDARD_DECODE, true)
}

/// Decodes a `URL`-safe `RFC` 4648 (`A-Za-z0-9-_`) string.
///
/// Padding is optional: unpadded input (as produced by [`encode_url_safe`]) is
/// accepted, and trailing `=` padding is tolerated if present. Returns `None`
/// for any illegal character or impossible tail length.
#[must_use]
pub fn decode_url_safe(s: &str) -> Option<Vec<u8>> {
    decode_impl(s, &URL_SAFE_DECODE, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // ---- RFC 4648 standard test vectors ----------------------------------

    #[test]
    fn rfc_vector_empty() {
        assert_eq!(encode_standard(b""), "");
    }

    #[test]
    fn rfc_vector_f() {
        assert_eq!(encode_standard(b"f"), "Zg==");
    }

    #[test]
    fn rfc_vector_fo() {
        assert_eq!(encode_standard(b"fo"), "Zm8=");
    }

    #[test]
    fn rfc_vector_foo() {
        assert_eq!(encode_standard(b"foo"), "Zm9v");
    }

    #[test]
    fn rfc_vector_foob() {
        assert_eq!(encode_standard(b"foob"), "Zm9vYg==");
    }

    #[test]
    fn rfc_vector_fooba() {
        assert_eq!(encode_standard(b"fooba"), "Zm9vYmE=");
    }

    #[test]
    fn rfc_vector_foobar() {
        assert_eq!(encode_standard(b"foobar"), "Zm9vYmFy");
    }

    // ---- standard decode of the same vectors -----------------------------

    #[test]
    fn decode_vector_empty() {
        assert_eq!(decode_standard(""), Some(Vec::new()));
    }

    #[test]
    fn decode_vector_f() {
        assert_eq!(decode_standard("Zg=="), Some(b"f".to_vec()));
    }

    #[test]
    fn decode_vector_fo() {
        assert_eq!(decode_standard("Zm8="), Some(b"fo".to_vec()));
    }

    #[test]
    fn decode_vector_foo() {
        assert_eq!(decode_standard("Zm9v"), Some(b"foo".to_vec()));
    }

    #[test]
    fn decode_vector_foob() {
        assert_eq!(decode_standard("Zm9vYg=="), Some(b"foob".to_vec()));
    }

    #[test]
    fn decode_vector_fooba() {
        assert_eq!(decode_standard("Zm9vYmE="), Some(b"fooba".to_vec()));
    }

    #[test]
    fn decode_vector_foobar() {
        assert_eq!(decode_standard("Zm9vYmFy"), Some(b"foobar".to_vec()));
    }

    // ---- tail-padding lengths --------------------------------------------

    #[test]
    fn one_byte_tail_has_two_pad() {
        let enc = encode_standard(&[0x00]);
        assert_eq!(enc.len(), 4);
        assert!(enc.ends_with("=="));
    }

    #[test]
    fn two_byte_tail_has_one_pad() {
        let enc = encode_standard(&[0x00, 0x00]);
        assert_eq!(enc.len(), 4);
        assert!(enc.ends_with('='));
        assert!(!enc.ends_with("=="));
    }

    #[test]
    fn three_byte_tail_has_no_pad() {
        let enc = encode_standard(&[0x00, 0x00, 0x00]);
        assert_eq!(enc.len(), 4);
        assert!(!enc.contains('='));
    }

    // ---- URL-safe variant ------------------------------------------------

    #[test]
    fn url_safe_has_no_padding() {
        assert_eq!(encode_url_safe(b"f"), "Zg");
        assert_eq!(encode_url_safe(b"fo"), "Zm8");
        assert_eq!(encode_url_safe(b"foo"), "Zm9v");
    }

    #[test]
    fn standard_uses_plus_and_slash() {
        // 0xFB,0xFF -> sextets 62,63,60 -> "+/8" plus one pad.
        assert_eq!(encode_standard(&[0xFB, 0xFF]), "+/8=");
    }

    #[test]
    fn url_safe_uses_dash_and_underscore() {
        // Same bytes, URL-safe alphabet maps 62->'-', 63->'_', no padding.
        assert_eq!(encode_url_safe(&[0xFB, 0xFF]), "-_8");
    }

    #[test]
    fn url_safe_decode_dash_underscore() {
        assert_eq!(decode_url_safe("-_8"), Some(vec![0xFB, 0xFF]));
    }

    #[test]
    fn url_safe_decode_tolerates_padding() {
        assert_eq!(decode_url_safe("Zg=="), Some(b"f".to_vec()));
    }

    #[test]
    fn standard_decode_plus_slash() {
        assert_eq!(decode_standard("+/8="), Some(vec![0xFB, 0xFF]));
    }

    // ---- strict rejection ------------------------------------------------

    #[test]
    fn standard_rejects_url_safe_characters() {
        assert_eq!(decode_standard("-_8="), None);
    }

    #[test]
    fn url_safe_rejects_standard_characters() {
        assert_eq!(decode_url_safe("+/8="), None);
    }

    #[test]
    fn standard_rejects_illegal_character() {
        assert_eq!(decode_standard("Zm9*"), None);
        assert_eq!(decode_standard("Z@=="), None);
    }

    #[test]
    fn standard_rejects_bad_length() {
        assert_eq!(decode_standard("Zg="), None); // length 3
        assert_eq!(decode_standard("Zm9"), None); // length 3
        assert_eq!(decode_standard("Zm9vY"), None); // length 5
    }

    #[test]
    fn standard_rejects_bad_padding() {
        assert_eq!(decode_standard("Z==="), None);
        assert_eq!(decode_standard("===="), None);
        assert_eq!(decode_standard("=Zg="), None); // embedded '='
    }

    #[test]
    fn url_safe_rejects_lone_tail_character() {
        assert_eq!(decode_url_safe("A"), None);
        assert_eq!(decode_url_safe("Zm9vA"), None);
    }

    // ---- round trips -----------------------------------------------------

    #[test]
    fn round_trip_every_single_byte_standard() {
        for value in 0u16..=255 {
            let byte = [value as u8];
            let enc = encode_standard(&byte);
            assert_eq!(decode_standard(&enc), Some(byte.to_vec()));
        }
    }

    #[test]
    fn round_trip_every_single_byte_url_safe() {
        for value in 0u16..=255 {
            let byte = [value as u8];
            let enc = encode_url_safe(&byte);
            assert_eq!(decode_url_safe(&enc), Some(byte.to_vec()));
        }
    }

    #[test]
    fn round_trip_all_lengths_standard() {
        // Build a 0..=255 ramp and round-trip every prefix length.
        let full: Vec<u8> = (0u16..=255).map(|v| v as u8).collect();
        for len in 0..=full.len() {
            let slice = &full[..len];
            let enc = encode_standard(slice);
            assert!(enc.len().is_multiple_of(4));
            assert_eq!(decode_standard(&enc), Some(slice.to_vec()));
        }
    }

    #[test]
    fn round_trip_all_lengths_url_safe() {
        let full: Vec<u8> = (0u16..=255).map(|v| v as u8).collect();
        for len in 0..=full.len() {
            let slice = &full[..len];
            let enc = encode_url_safe(slice);
            assert!(!enc.contains('='));
            assert_eq!(decode_url_safe(&enc), Some(slice.to_vec()));
        }
    }

    #[test]
    fn round_trip_lcg_random_standard() {
        let mut state: u64 = 0x0123_4567_89AB_CDEF;
        for _ in 0..200 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let count = (state >> 58) as usize; // 0..=63 bytes
            let mut data: Vec<u8> = Vec::with_capacity(count);
            let mut inner = state;
            for _ in 0..count {
                inner = inner
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                data.push((inner >> 56) as u8);
            }
            let enc = encode_standard(&data);
            assert_eq!(decode_standard(&enc), Some(data));
        }
    }

    #[test]
    fn round_trip_lcg_random_url_safe() {
        let mut state: u64 = 0xDEAD_BEEF_CAFE_F00D;
        for _ in 0..200 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let count = (state >> 58) as usize; // 0..=63 bytes
            let mut data: Vec<u8> = Vec::with_capacity(count);
            let mut inner = state ^ 0xA5A5_A5A5_A5A5_A5A5;
            for _ in 0..count {
                inner = inner
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                data.push((inner >> 56) as u8);
            }
            let enc = encode_url_safe(&data);
            assert_eq!(decode_url_safe(&enc), Some(data));
        }
    }

    #[test]
    fn standard_and_url_safe_differ_only_at_62_and_63() {
        let data = [0xFB, 0xFF, 0x00, 0x10, 0x83, 0xFC];
        let std_enc = encode_standard(&data);
        let url_enc = encode_url_safe(&data);
        // Strip standard padding for a character-by-character comparison.
        let std_trimmed = std_enc.trim_end_matches('=');
        assert_eq!(std_trimmed.len(), url_enc.len());
        for (s, u) in std_trimmed.chars().zip(url_enc.chars()) {
            match s {
                '+' => assert_eq!(u, '-'),
                '/' => assert_eq!(u, '_'),
                other => assert_eq!(u, other),
            }
        }
    }
}
