//! Pure-integer `UTF-8` codec over Unicode scalar values (CPU golden reference).
//!
//! This module implements a true `UTF-8` encoder, decoder, and validator per
//! `RFC 3629`. It operates on raw `u32` code points and `u8` bytes using only
//! integer bit operations, with no floating point and no `unsafe`.
//!
//! This is intentionally distinct from the hashing modules in this crate:
//! `fnv1a_hash` and `murmur3_hash` merely hash raw bytes and are completely
//! unaware of character boundaries or encoding validity. In contrast, this
//! module interprets bytes as `UTF-8` and enforces strict validity rules.
//!
//! Security notes (the common `UTF-8` pitfalls, handled explicitly here):
//! - OVERLONG encodings are rejected. For example `[0xC0, 0x80]` is NOT a valid
//!   encoding of `U+0000`; the shortest form must always be used.
//! - SURROGATE code points `U+D800..=U+DFFF` are rejected on both encode and
//!   decode. For example the byte sequence `[0xED, 0xA0, 0x80]` (which would
//!   decode to `U+D800`) is rejected.
//! - Code points above `U+10FFFF` are rejected.
//!
//! Valid Unicode scalar values span `U+0000..=U+10FFFF` excluding the surrogate
//! range `U+D800..=U+DFFF`.

use alloc::vec::Vec;

/// Returns the number of `UTF-8` bytes required to encode `cp`.
///
/// Returns `None` for surrogate code points (`U+D800..=U+DFFF`) and for code
/// points greater than `U+10FFFF`.
pub fn utf8_len(cp: u32) -> Option<u8> {
    if cp > 0x10FFFF {
        return None;
    }
    if (0xD800..=0xDFFF).contains(&cp) {
        return None;
    }
    if cp <= 0x7F {
        Some(1)
    } else if cp <= 0x7FF {
        Some(2)
    } else if cp <= 0xFFFF {
        Some(3)
    } else {
        Some(4)
    }
}

/// Encodes a single Unicode code point `cp` as 1-4 `UTF-8` bytes, appending the
/// bytes to `out`.
///
/// Returns `false` (and pushes nothing) for invalid code points: surrogates
/// `U+D800..=U+DFFF` and anything above `U+10FFFF`.
pub fn encode_char(cp: u32, out: &mut Vec<u8>) -> bool {
    match utf8_len(cp) {
        Some(1) => {
            out.push(cp as u8);
            true
        }
        Some(2) => {
            out.push(0xC0 | ((cp >> 6) as u8));
            out.push(0x80 | ((cp & 0x3F) as u8));
            true
        }
        Some(3) => {
            out.push(0xE0 | ((cp >> 12) as u8));
            out.push(0x80 | (((cp >> 6) & 0x3F) as u8));
            out.push(0x80 | ((cp & 0x3F) as u8));
            true
        }
        Some(4) => {
            out.push(0xF0 | ((cp >> 18) as u8));
            out.push(0x80 | (((cp >> 12) & 0x3F) as u8));
            out.push(0x80 | (((cp >> 6) & 0x3F) as u8));
            out.push(0x80 | ((cp & 0x3F) as u8));
            true
        }
        _ => false,
    }
}

/// Decodes one Unicode scalar value from `bytes` starting at `*pos`, advancing
/// `*pos` past the consumed bytes on success.
///
/// Returns `None` (leaving `*pos` unchanged) on any malformed sequence:
/// - invalid lead byte (continuation byte in lead position, or `0xF8..=0xFF`),
/// - a continuation byte that does not match the `10xxxxxx` pattern,
/// - OVERLONG encodings (value smaller than the minimum for its length),
/// - SURROGATE code points `U+D800..=U+DFFF`,
/// - values greater than `U+10FFFF`,
/// - truncated sequences (not enough bytes remain).
pub fn decode_next(bytes: &[u8], pos: &mut usize) -> Option<u32> {
    let start = *pos;
    let b0 = *bytes.get(start)?;

    // Determine sequence length, the minimum legal code point for that length
    // (used to reject overlong encodings), and the initial payload bits.
    let (len, min, init): (usize, u32, u32) = if b0 < 0x80 {
        (1, 0x0, b0 as u32)
    } else if (b0 & 0xE0) == 0xC0 {
        (2, 0x80, (b0 & 0x1F) as u32)
    } else if (b0 & 0xF0) == 0xE0 {
        (3, 0x800, (b0 & 0x0F) as u32)
    } else if (b0 & 0xF8) == 0xF0 {
        (4, 0x10000, (b0 & 0x07) as u32)
    } else {
        return None;
    };

    let mut cp = init;
    let mut i = 1usize;
    while i < len {
        let b = *bytes.get(start + i)?;
        if (b & 0xC0) != 0x80 {
            return None;
        }
        cp = (cp << 6) | ((b & 0x3F) as u32);
        i += 1;
    }

    // Reject overlong encodings.
    if cp < min {
        return None;
    }
    // Reject surrogate code points.
    if (0xD800..=0xDFFF).contains(&cp) {
        return None;
    }
    // Reject out-of-range code points.
    if cp > 0x10FFFF {
        return None;
    }

    *pos = start + len;
    Some(cp)
}

/// Strict `RFC 3629` validation: returns `true` only if every byte in `bytes`
/// participates in a well-formed, shortest-form, non-surrogate `UTF-8`
/// sequence. An empty slice is valid.
pub fn validate(bytes: &[u8]) -> bool {
    let mut pos = 0usize;
    while pos < bytes.len() {
        if decode_next(bytes, &mut pos).is_none() {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[cfg(test)]
    fn encode_one(cp: u32) -> Option<Vec<u8>> {
        let mut out = Vec::new();
        if encode_char(cp, &mut out) {
            Some(out)
        } else {
            None
        }
    }

    #[cfg(test)]
    fn decode_one(bytes: &[u8]) -> Option<u32> {
        let mut pos = 0usize;
        let r = decode_next(bytes, &mut pos);
        if r.is_some() {
            assert_eq!(pos, bytes.len());
        }
        r
    }

    // --- Reference vectors -------------------------------------------------

    #[test]
    fn encode_ref_a() {
        assert_eq!(encode_one(0x0041).unwrap().as_slice(), &[0x41]);
    }

    #[test]
    fn encode_ref_cent() {
        assert_eq!(encode_one(0x00A2).unwrap().as_slice(), &[0xC2, 0xA2]);
    }

    #[test]
    fn encode_ref_euro() {
        assert_eq!(encode_one(0x20AC).unwrap().as_slice(), &[0xE2, 0x82, 0xAC]);
    }

    #[test]
    fn encode_ref_gothic() {
        assert_eq!(
            encode_one(0x10348).unwrap().as_slice(),
            &[0xF0, 0x90, 0x8D, 0x88]
        );
    }

    #[test]
    fn decode_ref_a() {
        assert_eq!(decode_one(&[0x41]), Some(0x0041));
    }

    #[test]
    fn decode_ref_cent() {
        assert_eq!(decode_one(&[0xC2, 0xA2]), Some(0x00A2));
    }

    #[test]
    fn decode_ref_euro() {
        assert_eq!(decode_one(&[0xE2, 0x82, 0xAC]), Some(0x20AC));
    }

    #[test]
    fn decode_ref_gothic() {
        assert_eq!(decode_one(&[0xF0, 0x90, 0x8D, 0x88]), Some(0x10348));
    }

    // --- Length boundaries -------------------------------------------------

    #[test]
    fn len_boundary_007f() {
        assert_eq!(utf8_len(0x007F), Some(1));
    }

    #[test]
    fn len_boundary_0080() {
        assert_eq!(utf8_len(0x0080), Some(2));
    }

    #[test]
    fn len_boundary_07ff() {
        assert_eq!(utf8_len(0x07FF), Some(2));
    }

    #[test]
    fn len_boundary_0800() {
        assert_eq!(utf8_len(0x0800), Some(3));
    }

    #[test]
    fn len_boundary_ffff() {
        assert_eq!(utf8_len(0xFFFF), Some(3));
    }

    #[test]
    fn len_boundary_10000() {
        assert_eq!(utf8_len(0x10000), Some(4));
    }

    #[test]
    fn len_boundary_10ffff() {
        assert_eq!(utf8_len(0x10FFFF), Some(4));
    }

    #[test]
    fn boundary_roundtrips() {
        let samples = [0x007F, 0x0080, 0x07FF, 0x0800, 0xFFFF, 0x10000, 0x10FFFF];
        let mut idx = 0usize;
        while idx < samples.len() {
            let cp = samples[idx];
            let enc = encode_one(cp).unwrap();
            assert_eq!(decode_one(enc.as_slice()), Some(cp));
            idx += 1;
        }
    }

    // --- Minimum of each length encodes to the expected shortest form ------

    #[test]
    fn encode_min_each_length() {
        assert_eq!(encode_one(0x0000).unwrap().as_slice(), &[0x00]);
        assert_eq!(encode_one(0x0080).unwrap().as_slice(), &[0xC2, 0x80]);
        assert_eq!(encode_one(0x0800).unwrap().as_slice(), &[0xE0, 0xA0, 0x80]);
        assert_eq!(
            encode_one(0x10000).unwrap().as_slice(),
            &[0xF0, 0x90, 0x80, 0x80]
        );
    }

    #[test]
    fn decode_max_scalar() {
        assert_eq!(decode_one(&[0xF4, 0x8F, 0xBF, 0xBF]), Some(0x10FFFF));
    }

    // --- Overlong rejection ------------------------------------------------

    #[test]
    fn overlong_2byte_reject() {
        assert_eq!(decode_one(&[0xC0, 0x80]), None);
        assert_eq!(decode_one(&[0xC1, 0xBF]), None);
    }

    #[test]
    fn overlong_3byte_reject() {
        assert_eq!(decode_one(&[0xE0, 0x80, 0x80]), None);
        assert_eq!(decode_one(&[0xE0, 0x9F, 0xBF]), None);
    }

    #[test]
    fn overlong_4byte_reject() {
        assert_eq!(decode_one(&[0xF0, 0x80, 0x80, 0x80]), None);
        assert_eq!(decode_one(&[0xF0, 0x8F, 0xBF, 0xBF]), None);
    }

    // --- Surrogate rejection ----------------------------------------------

    #[test]
    fn surrogate_encode_reject_d800() {
        assert!(encode_one(0xD800).is_none());
    }

    #[test]
    fn surrogate_encode_reject_dfff() {
        assert!(encode_one(0xDFFF).is_none());
    }

    #[test]
    fn surrogate_decode_reject_d800() {
        assert_eq!(decode_one(&[0xED, 0xA0, 0x80]), None);
    }

    #[test]
    fn surrogate_decode_reject_dfff() {
        assert_eq!(decode_one(&[0xED, 0xBF, 0xBF]), None);
    }

    #[test]
    fn utf8_len_surrogate_none() {
        assert_eq!(utf8_len(0xD800), None);
        assert_eq!(utf8_len(0xDBFF), None);
        assert_eq!(utf8_len(0xDC00), None);
        assert_eq!(utf8_len(0xDFFF), None);
    }

    // --- Out-of-range rejection -------------------------------------------

    #[test]
    fn utf8_len_out_of_range_none() {
        assert_eq!(utf8_len(0x110000), None);
        assert_eq!(utf8_len(0xFFFF_FFFF), None);
    }

    #[test]
    fn encode_out_of_range_reject() {
        assert!(encode_one(0x110000).is_none());
        assert!(encode_one(0x20_0000).is_none());
    }

    #[test]
    fn decode_out_of_range_reject() {
        // Lead byte 0xF4 with continuation 0x90 would be U+110000.
        assert_eq!(decode_one(&[0xF4, 0x90, 0x80, 0x80]), None);
        // Lead byte 0xF5 is always out of range.
        assert_eq!(decode_one(&[0xF5, 0x80, 0x80, 0x80]), None);
    }

    // --- Truncated sequences ----------------------------------------------

    #[test]
    fn truncated_2byte() {
        let mut pos = 0usize;
        assert_eq!(decode_next(&[0xC2], &mut pos), None);
        assert_eq!(pos, 0);
    }

    #[test]
    fn truncated_3byte() {
        assert_eq!(decode_one(&[0xE2, 0x82]), None);
        assert_eq!(decode_one(&[0xE2]), None);
    }

    #[test]
    fn truncated_4byte() {
        assert_eq!(decode_one(&[0xF0, 0x90, 0x8D]), None);
        assert_eq!(decode_one(&[0xF0, 0x90]), None);
        assert_eq!(decode_one(&[0xF0]), None);
    }

    // --- Invalid lead bytes ------------------------------------------------

    #[test]
    fn invalid_lead_0x80() {
        assert_eq!(decode_one(&[0x80]), None);
    }

    #[test]
    fn invalid_lead_0xbf() {
        assert_eq!(decode_one(&[0xBF]), None);
    }

    #[test]
    fn invalid_lead_0xf8() {
        assert_eq!(decode_one(&[0xF8, 0x80, 0x80, 0x80]), None);
    }

    #[test]
    fn invalid_lead_0xff() {
        assert_eq!(decode_one(&[0xFF]), None);
    }

    // --- Bad continuation bytes -------------------------------------------

    #[test]
    fn bad_continuation_2byte() {
        assert_eq!(decode_one(&[0xC2, 0x00]), None);
        assert_eq!(decode_one(&[0xC2, 0xC2]), None);
    }

    #[test]
    fn bad_continuation_3byte() {
        assert_eq!(decode_one(&[0xE2, 0x82, 0x20]), None);
        assert_eq!(decode_one(&[0xE2, 0x28, 0xAC]), None);
    }

    #[test]
    fn bad_continuation_4byte() {
        assert_eq!(decode_one(&[0xF0, 0x90, 0x8D, 0x28]), None);
    }

    // --- Validation --------------------------------------------------------

    #[test]
    fn validate_empty_is_valid() {
        assert!(validate(&[]));
    }

    #[test]
    fn validate_valid_buffer() {
        // "A¢€𐍈" concatenated.
        let buf = [0x41, 0xC2, 0xA2, 0xE2, 0x82, 0xAC, 0xF0, 0x90, 0x8D, 0x88];
        assert!(validate(&buf));
    }

    #[test]
    fn validate_invalid_buffer_overlong() {
        let buf = [0x41, 0xC0, 0x80, 0x42];
        assert!(!validate(&buf));
    }

    #[test]
    fn validate_invalid_buffer_truncated_tail() {
        let buf = [0x41, 0xE2, 0x82];
        assert!(!validate(&buf));
    }

    #[test]
    fn validate_invalid_buffer_surrogate() {
        let buf = [0x41, 0xED, 0xA0, 0x80];
        assert!(!validate(&buf));
    }

    #[test]
    fn decode_empty_is_none() {
        let mut pos = 0usize;
        assert_eq!(decode_next(&[], &mut pos), None);
        assert_eq!(pos, 0);
    }

    // --- Multi-character stream decoding -----------------------------------

    #[test]
    fn decode_multi_char_stream() {
        let buf = [0x41, 0xC2, 0xA2, 0xE2, 0x82, 0xAC, 0xF0, 0x90, 0x8D, 0x88];
        let mut pos = 0usize;
        assert_eq!(decode_next(&buf, &mut pos), Some(0x0041));
        assert_eq!(decode_next(&buf, &mut pos), Some(0x00A2));
        assert_eq!(decode_next(&buf, &mut pos), Some(0x20AC));
        assert_eq!(decode_next(&buf, &mut pos), Some(0x10348));
        assert_eq!(pos, buf.len());
    }

    // --- Full round-trip sweep over a range of code points -----------------

    #[test]
    fn roundtrip_sweep() {
        let mut cp = 0u32;
        while cp <= 0x10FFFF {
            if !(0xD800..=0xDFFF).contains(&cp) {
                let enc = encode_one(cp).unwrap();
                let n = utf8_len(cp).unwrap() as usize;
                assert_eq!(enc.len(), n);
                let mut pos = 0usize;
                assert_eq!(decode_next(enc.as_slice(), &mut pos), Some(cp));
                assert_eq!(pos, enc.len());
            }
            cp = cp.wrapping_add(0x1111);
        }
    }

    #[test]
    fn roundtrip_dense_low_range() {
        let mut cp = 0u32;
        while cp <= 0x0FFF {
            let enc = encode_one(cp).unwrap();
            assert_eq!(decode_one(enc.as_slice()), Some(cp));
            cp = cp.wrapping_add(1);
        }
    }

    #[test]
    fn encode_appends_without_clearing() {
        let mut out = Vec::new();
        out.push(0xFFu8);
        assert!(encode_char(0x0041, &mut out));
        assert_eq!(out.as_slice(), &[0xFF, 0x41]);
    }
}
