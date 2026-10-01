//! `RFC` 4648 `Base32` binary-to-text encoding, including the `base32hex`
//! extended-alphabet variant, implemented with pure integer bit operations on
//! the `CPU`.
//!
//! `Base32` reinterprets an arbitrary byte stream as a sequence of 5-bit
//! groups and maps each group to one printable `ASCII` character, so binary
//! assets (mesh/texture blobs, `GPU` upload payloads, serialized parameter
//! buffers) survive transport through case-insensitive, text-only channels —
//! `DNS` labels, filesystem names, human-dictated identifiers — without
//! corruption. Five input bytes (40 bits) become exactly eight output
//! characters (8 x 5 bits); a tail of one to four bytes is zero-extended to
//! the next 5-bit boundary and completed with `=` padding so every encoded
//! block is eight characters wide.
//!
//! This is a deliberately separate implementation from the sibling `Base64`
//! module: the group size (5 bits versus 6), the block size (8 characters
//! versus 4), the alphabets, and the padding schedule all differ, and none of
//! that module's tables or helpers are reused here.
//!
//! Two alphabets are provided:
//!
//! * Standard ([`encode`] / [`encode_nopad`] / [`decode`]): `A-Z2-7` with
//!   trailing `=` padding, exactly as specified by `RFC` 4648 section 6.
//! * `base32hex` ([`encode_hex`] / [`decode_hex`]): `0-9A-V` as specified by
//!   `RFC` 4648 section 7. This alphabet keeps the sort order of the encoded
//!   text aligned with the sort order of the underlying bytes.
//!
//! The padding schedule follows `RFC` 4648 section 6 exactly. A final quantum
//! of 1, 2, 3, or 4 bytes produces 2, 4, 5, or 7 significant characters
//! respectively, each padded with `=` up to the full eight-character block; a
//! full 5-byte quantum produces eight characters and no padding.
//!
//! Decoding is strict. The standard alphabet is upper-case only, so this
//! decoder does not fold case: any character outside the selected alphabet,
//! any total length that is not a multiple of eight, and any padding count
//! that cannot arise from a real tail are reported as a [`Base32Error`]
//! rather than guessed.
//!
//! ## Boundaries
//!
//! `Base32` is a lossless 5-bit *re-encoding*, not a codec that removes
//! redundancy or hides content:
//!
//! * It never shrinks data. Output is always exactly 60% larger than input
//!   (eight characters per five bytes, rounded up to a block), so it is the
//!   opposite of the run-length and quantization primitives in this crate.
//! * It provides no confidentiality or integrity. The mapping is a fixed,
//!   public table; anyone can decode it. Encryption and hashing live
//!   elsewhere.
//!
//! All arithmetic is integer shift/mask/or only — no floating point and no
//! transcendental functions — so this reference is bit-reproducible on any
//! target.

extern crate alloc;

use alloc::{string::String, vec::Vec};

/// Sentinel stored in a decode table for every byte that is **not** a member
/// of the corresponding alphabet. Valid quintet values are `0..=31`, so `0xFF`
/// can never collide with a legitimate entry.
const INVALID: u8 = 0xFF;

/// Builds the standard `RFC` 4648 section 6 encode table (`quintet -> ASCII
/// byte`) for the alphabet `A-Z2-7`.
const fn build_standard_encode() -> [u8; 32] {
    let mut table = [0u8; 32];
    let mut i = 0usize;
    while i < 26 {
        table[i] = b'A' + i as u8;
        i += 1;
    }
    let mut j = 0usize;
    while j < 6 {
        table[26 + j] = b'2' + j as u8;
        j += 1;
    }
    table
}

/// Builds the `base32hex` (`RFC` 4648 section 7) encode table for the alphabet
/// `0-9A-V`.
const fn build_hex_encode() -> [u8; 32] {
    let mut table = [0u8; 32];
    let mut i = 0usize;
    while i < 10 {
        table[i] = b'0' + i as u8;
        i += 1;
    }
    let mut j = 0usize;
    while j < 22 {
        table[10 + j] = b'A' + j as u8;
        j += 1;
    }
    table
}

/// Builds the 256-entry reverse-lookup table (`ASCII byte -> quintet`) for a
/// given encode table. Every byte not present in `enc` maps to [`INVALID`].
const fn build_decode_table(enc: &[u8; 32]) -> [u8; 256] {
    let mut table = [INVALID; 256];
    let mut i = 0usize;
    while i < 32 {
        table[enc[i] as usize] = i as u8;
        i += 1;
    }
    table
}

/// Standard alphabet encode table: `A-Z2-7`.
const STANDARD_ENCODE: [u8; 32] = build_standard_encode();
/// `base32hex` alphabet encode table: `0-9A-V`.
const HEX_ENCODE: [u8; 32] = build_hex_encode();
/// Reverse lookup for the standard alphabet.
const STANDARD_DECODE: [u8; 256] = build_decode_table(&STANDARD_ENCODE);
/// Reverse lookup for the `base32hex` alphabet.
const HEX_DECODE: [u8; 256] = build_decode_table(&HEX_ENCODE);

/// Error returned by the strict decoders when the input cannot be a valid
/// `RFC` 4648 `Base32` string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Base32Error {
    /// The input contained a byte that is not a member of the selected
    /// alphabet (and is not trailing `=` padding). Carries the offending
    /// character.
    InvalidChar(char),
    /// The input length is not a positive multiple of eight, so it cannot be a
    /// sequence of complete `Base32` blocks.
    InvalidLength,
    /// The trailing `=` padding count is not one of the values a real tail can
    /// produce (`0`, `1`, `3`, `4`, or `6`).
    InvalidPadding,
}

/// Number of significant (non-`=`) characters produced by a final quantum of
/// `bytes` input bytes, where `bytes` is in `1..=5`.
const fn significant_chars(bytes: usize) -> usize {
    match bytes {
        1 => 2,
        2 => 4,
        3 => 5,
        4 => 7,
        _ => 8,
    }
}

/// Core encoder shared by both alphabets.
///
/// `table` selects the alphabet; `pad` controls whether each tail block is
/// completed with `=` characters (the standard form) or left bare.
fn encode_impl(data: &[u8], table: &[u8; 32], pad: bool) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(5) * 8);
    for chunk in data.chunks(5) {
        let n = chunk.len();
        // Pack up to 40 bits into a u64 accumulator, most-significant byte
        // first, then left-align the significant bits to the top of the field.
        let mut acc: u64 = 0;
        for &byte in chunk {
            acc = (acc << 8) | byte as u64;
        }
        acc <<= 40 - 8 * n;
        let chars = significant_chars(n);
        for k in 0..chars {
            let shift = 40 - 5 * (k + 1);
            let idx = ((acc >> shift) & 0x1F) as usize;
            out.push(table[idx] as char);
        }
        if pad {
            for _ in chars..8 {
                out.push('=');
            }
        }
    }
    out
}

/// Core decoder shared by both alphabets.
///
/// `table` is the reverse lookup. Decoding is strict: the total length must be
/// a positive multiple of eight, the trailing `=` padding count must be one of
/// the values a real tail can produce, and every non-padding byte must be a
/// member of the alphabet.
fn decode_impl(s: &str, table: &[u8; 256]) -> Result<Vec<u8>, Base32Error> {
    let bytes = s.as_bytes();
    let len = bytes.len();
    if len == 0 {
        return Ok(Vec::new());
    }
    if !len.is_multiple_of(8) {
        return Err(Base32Error::InvalidLength);
    }

    // Count contiguous trailing '=' padding. Any '=' that is not trailing is
    // left in the body and rejected below as an invalid character.
    let mut pad = 0usize;
    while pad < len && bytes[len - 1 - pad] == b'=' {
        pad += 1;
    }
    match pad {
        0 | 1 | 3 | 4 | 6 => {}
        _ => return Err(Base32Error::InvalidPadding),
    }

    let body_len = len - pad;
    let mut out = Vec::with_capacity(len / 8 * 5);
    let mut acc: u64 = 0;
    let mut nbits: u32 = 0;
    for &ch in &bytes[..body_len] {
        let value = table[ch as usize];
        if value == INVALID {
            return Err(Base32Error::InvalidChar(char::from(ch)));
        }
        acc = (acc << 5) | value as u64;
        nbits += 5;
        if nbits >= 8 {
            nbits -= 8;
            out.push((acc >> nbits) as u8);
            acc &= (1u64 << nbits) - 1;
        }
    }

    Ok(out)
}

/// Encodes `data` with the standard `RFC` 4648 alphabet `A-Z2-7`, emitting `=`
/// padding so the output length is always a multiple of eight.
///
/// The empty input yields the empty string.
#[must_use]
pub fn encode(data: &[u8]) -> String {
    encode_impl(data, &STANDARD_ENCODE, true)
}

/// Encodes `data` with the standard `RFC` 4648 alphabet `A-Z2-7`, emitting
/// **no** `=` padding.
///
/// The output is the padded [`encode`] result with its trailing `=` characters
/// removed; the empty input yields the empty string.
#[must_use]
pub fn encode_nopad(data: &[u8]) -> String {
    encode_impl(data, &STANDARD_ENCODE, false)
}

/// Decodes a standard `RFC` 4648 (`A-Z2-7`) string.
///
/// Decoding is strict and case-sensitive. Returns a [`Base32Error`] for any
/// illegal character, any length that is not a positive multiple of eight, or
/// a padding count that cannot arise from a real tail.
///
/// # Errors
///
/// Returns [`Base32Error::InvalidChar`], [`Base32Error::InvalidLength`], or
/// [`Base32Error::InvalidPadding`] as described above.
pub fn decode(s: &str) -> Result<Vec<u8>, Base32Error> {
    decode_impl(s, &STANDARD_DECODE)
}

/// Encodes `data` with the `base32hex` `RFC` 4648 alphabet `0-9A-V`, emitting
/// `=` padding so the output length is always a multiple of eight.
///
/// The empty input yields the empty string.
#[must_use]
pub fn encode_hex(data: &[u8]) -> String {
    encode_impl(data, &HEX_ENCODE, true)
}

/// Decodes a `base32hex` `RFC` 4648 (`0-9A-V`) string.
///
/// Decoding is strict and case-sensitive, exactly like [`decode`] but against
/// the `base32hex` alphabet.
///
/// # Errors
///
/// Returns [`Base32Error::InvalidChar`], [`Base32Error::InvalidLength`], or
/// [`Base32Error::InvalidPadding`] as described above.
pub fn decode_hex(s: &str) -> Result<Vec<u8>, Base32Error> {
    decode_impl(s, &HEX_DECODE)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Re-pads an unpadded `Base32` string back to a whole number of
    /// eight-character blocks so the strict decoder can accept it.
    fn pad_to_block(s: &str) -> String {
        let mut padded = String::from(s);
        while !padded.len().is_multiple_of(8) {
            padded.push('=');
        }
        padded
    }

    // ---- RFC 4648 section 10 standard encode vectors ---------------------

    #[test]
    fn rfc_std_encode_empty() {
        assert_eq!(encode(b""), "");
    }

    #[test]
    fn rfc_std_encode_f() {
        assert_eq!(encode(b"f"), "MY======");
    }

    #[test]
    fn rfc_std_encode_fo() {
        assert_eq!(encode(b"fo"), "MZXQ====");
    }

    #[test]
    fn rfc_std_encode_foo() {
        assert_eq!(encode(b"foo"), "MZXW6===");
    }

    #[test]
    fn rfc_std_encode_foob() {
        assert_eq!(encode(b"foob"), "MZXW6YQ=");
    }

    #[test]
    fn rfc_std_encode_fooba() {
        assert_eq!(encode(b"fooba"), "MZXW6YTB");
    }

    #[test]
    fn rfc_std_encode_foobar() {
        assert_eq!(encode(b"foobar"), "MZXW6YTBOI======");
    }

    // ---- standard decode of the same vectors -----------------------------

    #[test]
    fn rfc_std_decode_empty() {
        assert_eq!(decode(""), Ok(Vec::new()));
    }

    #[test]
    fn rfc_std_decode_f() {
        assert_eq!(decode("MY======"), Ok(b"f".to_vec()));
    }

    #[test]
    fn rfc_std_decode_fo() {
        assert_eq!(decode("MZXQ===="), Ok(b"fo".to_vec()));
    }

    #[test]
    fn rfc_std_decode_foo() {
        assert_eq!(decode("MZXW6==="), Ok(b"foo".to_vec()));
    }

    #[test]
    fn rfc_std_decode_foob() {
        assert_eq!(decode("MZXW6YQ="), Ok(b"foob".to_vec()));
    }

    #[test]
    fn rfc_std_decode_fooba() {
        assert_eq!(decode("MZXW6YTB"), Ok(b"fooba".to_vec()));
    }

    #[test]
    fn rfc_std_decode_foobar() {
        assert_eq!(decode("MZXW6YTBOI======"), Ok(b"foobar".to_vec()));
    }

    // ---- RFC 4648 section 10 base32hex encode vectors --------------------

    #[test]
    fn rfc_hex_encode_empty() {
        assert_eq!(encode_hex(b""), "");
    }

    #[test]
    fn rfc_hex_encode_f() {
        assert_eq!(encode_hex(b"f"), "CO======");
    }

    #[test]
    fn rfc_hex_encode_fo() {
        assert_eq!(encode_hex(b"fo"), "CPNG====");
    }

    #[test]
    fn rfc_hex_encode_foo() {
        assert_eq!(encode_hex(b"foo"), "CPNMU===");
    }

    #[test]
    fn rfc_hex_encode_foob() {
        assert_eq!(encode_hex(b"foob"), "CPNMUOG=");
    }

    #[test]
    fn rfc_hex_encode_fooba() {
        assert_eq!(encode_hex(b"fooba"), "CPNMUOJ1");
    }

    #[test]
    fn rfc_hex_encode_foobar() {
        assert_eq!(encode_hex(b"foobar"), "CPNMUOJ1E8======");
    }

    // ---- base32hex decode of the same vectors ----------------------------

    #[test]
    fn rfc_hex_decode_empty() {
        assert_eq!(decode_hex(""), Ok(Vec::new()));
    }

    #[test]
    fn rfc_hex_decode_f() {
        assert_eq!(decode_hex("CO======"), Ok(b"f".to_vec()));
    }

    #[test]
    fn rfc_hex_decode_fo() {
        assert_eq!(decode_hex("CPNG===="), Ok(b"fo".to_vec()));
    }

    #[test]
    fn rfc_hex_decode_foo() {
        assert_eq!(decode_hex("CPNMU==="), Ok(b"foo".to_vec()));
    }

    #[test]
    fn rfc_hex_decode_foob() {
        assert_eq!(decode_hex("CPNMUOG="), Ok(b"foob".to_vec()));
    }

    #[test]
    fn rfc_hex_decode_fooba() {
        assert_eq!(decode_hex("CPNMUOJ1"), Ok(b"fooba".to_vec()));
    }

    #[test]
    fn rfc_hex_decode_foobar() {
        assert_eq!(decode_hex("CPNMUOJ1E8======"), Ok(b"foobar".to_vec()));
    }

    // ---- padding-length schedule -----------------------------------------

    #[test]
    fn tail_padding_lengths_standard() {
        assert_eq!(encode(&[0x00]).matches('=').count(), 6);
        assert_eq!(encode(&[0x00, 0x00]).matches('=').count(), 4);
        assert_eq!(encode(&[0x00, 0x00, 0x00]).matches('=').count(), 3);
        assert_eq!(encode(&[0x00, 0x00, 0x00, 0x00]).matches('=').count(), 1);
        assert_eq!(
            encode(&[0x00, 0x00, 0x00, 0x00, 0x00]).matches('=').count(),
            0
        );
    }

    #[test]
    fn every_block_is_eight_wide_standard() {
        let full: Vec<u8> = (0u16..=255).map(|v| v as u8).collect();
        for len in 0..=full.len() {
            assert!(encode(&full[..len]).len().is_multiple_of(8));
        }
    }

    // ---- nopad behaviour --------------------------------------------------

    #[test]
    fn nopad_matches_trimmed_padded() {
        for len in 0usize..=16 {
            let data: Vec<u8> = (0..len).map(|v| (v * 7 + 1) as u8).collect();
            let trimmed = encode(&data);
            let trimmed = trimmed.trim_end_matches('=');
            assert_eq!(encode_nopad(&data), trimmed);
        }
    }

    #[test]
    fn nopad_never_contains_padding() {
        let full: Vec<u8> = (0u16..=200).map(|v| v as u8).collect();
        for len in 0..=full.len() {
            assert!(!encode_nopad(&full[..len]).contains('='));
        }
    }

    #[test]
    fn nopad_round_trip_via_repad() {
        let full: Vec<u8> = (0u16..=255).map(|v| v as u8).collect();
        for len in 0..=full.len() {
            let slice = &full[..len];
            let encoded = encode_nopad(slice);
            assert_eq!(decode(&pad_to_block(&encoded)), Ok(slice.to_vec()));
        }
    }

    // ---- round trips ------------------------------------------------------

    #[test]
    fn round_trip_every_single_byte_standard() {
        for value in 0u16..=255 {
            let byte = [value as u8];
            assert_eq!(decode(&encode(&byte)), Ok(byte.to_vec()));
        }
    }

    #[test]
    fn round_trip_every_single_byte_hex() {
        for value in 0u16..=255 {
            let byte = [value as u8];
            assert_eq!(decode_hex(&encode_hex(&byte)), Ok(byte.to_vec()));
        }
    }

    #[test]
    fn round_trip_all_lengths_standard() {
        let full: Vec<u8> = (0u16..=255).map(|v| v as u8).collect();
        for len in 0..=full.len() {
            let slice = &full[..len];
            assert_eq!(decode(&encode(slice)), Ok(slice.to_vec()));
        }
    }

    #[test]
    fn round_trip_all_lengths_hex() {
        let full: Vec<u8> = (0u16..=255).map(|v| v as u8).collect();
        for len in 0..=full.len() {
            let slice = &full[..len];
            assert_eq!(decode_hex(&encode_hex(slice)), Ok(slice.to_vec()));
        }
    }

    #[test]
    fn round_trip_lcg_random_standard() {
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
    fn round_trip_lcg_random_hex() {
        let mut state: u64 = 0xDEAD_BEEF_CAFE_F00D;
        for _ in 0..256 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let count = (state >> 58) as usize;
            let mut data: Vec<u8> = Vec::with_capacity(count);
            let mut inner = state ^ 0xA5A5_A5A5_A5A5_A5A5;
            for _ in 0..count {
                inner = inner
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                data.push((inner >> 56) as u8);
            }
            assert_eq!(decode_hex(&encode_hex(&data)), Ok(data));
        }
    }

    // ---- strict rejection -------------------------------------------------

    #[test]
    fn decode_rejects_illegal_character() {
        // '1', '8', '9', '0' are not in the standard A-Z2-7 alphabet.
        assert_eq!(decode("MZXW6YT0"), Err(Base32Error::InvalidChar('0')));
        assert_eq!(decode("MZXW6YT1"), Err(Base32Error::InvalidChar('1')));
        assert_eq!(decode("MZXW6Y!B"), Err(Base32Error::InvalidChar('!')));
    }

    #[test]
    fn decode_rejects_lowercase_as_illegal() {
        // Standard decoding is case-sensitive: lower-case is not folded.
        assert_eq!(decode("my======"), Err(Base32Error::InvalidChar('m')));
    }

    #[test]
    fn decode_rejects_bad_length() {
        assert_eq!(decode("MY====="), Err(Base32Error::InvalidLength)); // 7 chars
        assert_eq!(decode("MZXW6YTBO"), Err(Base32Error::InvalidLength)); // 9 chars
        assert_eq!(decode("MZX"), Err(Base32Error::InvalidLength)); // 3 chars
    }

    #[test]
    fn decode_rejects_bad_padding() {
        // Two trailing '=' can never arise from a real Base32 tail.
        assert_eq!(decode("MZXW6Y=="), Err(Base32Error::InvalidPadding));
        // Five trailing '=' is likewise impossible.
        assert_eq!(decode("MY====="), Err(Base32Error::InvalidLength));
        assert_eq!(decode("M======="), Err(Base32Error::InvalidPadding));
        assert_eq!(decode("========"), Err(Base32Error::InvalidPadding));
    }

    #[test]
    fn decode_rejects_embedded_padding() {
        // A '=' that is not part of the trailing run is an illegal character.
        assert_eq!(decode("M=XW6YQ="), Err(Base32Error::InvalidChar('=')));
    }

    #[test]
    fn hex_decode_rejects_illegal_character() {
        // 'W' and 'Z' are outside the base32hex 0-9A-V alphabet.
        assert_eq!(decode_hex("CPNMUOJW"), Err(Base32Error::InvalidChar('W')));
        assert_eq!(decode_hex("CPNMUOJZ"), Err(Base32Error::InvalidChar('Z')));
    }

    #[test]
    fn hex_decode_rejects_bad_length_and_padding() {
        assert_eq!(decode_hex("CO====="), Err(Base32Error::InvalidLength));
        assert_eq!(decode_hex("CPNMUO=="), Err(Base32Error::InvalidPadding));
    }

    #[test]
    fn standard_and_hex_differ_on_the_same_input() {
        let data = b"foobar";
        assert_eq!(encode(data), "MZXW6YTBOI======");
        assert_eq!(encode_hex(data), "CPNMUOJ1E8======");
    }

    #[test]
    fn error_is_clone_and_eq() {
        let err = Base32Error::InvalidChar('%');
        assert_eq!(err.clone(), Base32Error::InvalidChar('%'));
        assert_ne!(Base32Error::InvalidLength, Base32Error::InvalidPadding);
    }
}
