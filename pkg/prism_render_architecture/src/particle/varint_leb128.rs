//! Variable-length integer coding in the `LEB128` byte format: map one integer
//! to (and back from) a little-endian sequence of 7-bit groups where the high
//! bit of each byte is a continuation flag. This is the canonical varint layout
//! used by Protocol Buffers and by `DWARF` debug info, and it is the on-the-wire
//! shape that a `CPU`-side serializer emits before a `GPU` upload reads it back.
//!
//! Two variants live here, both closed and exact (`decode(encode(x)) == x`):
//!
//! * **Unsigned** ([`encode_u64`] / [`decode_u64`]) writes the raw magnitude
//!   seven bits at a time, least-significant group first (`LSB`-first), setting
//!   the continuation bit on every byte except the last.
//! * **Signed** ([`encode_i64`] / [`decode_i64`]) uses the two's-complement
//!   sign-extension rule: emission stops once the remaining bits are all `0`
//!   with a clear group sign bit, or all `1` with a set group sign bit, and the
//!   decoder sign-extends the final group across the high bits.
//!
//! A single `u64`/`i64` needs at most ten bytes (nine full 7-bit groups plus one
//! group carrying the top bit). The decoders return the consumed byte count so a
//! caller can walk a packed buffer, and they reject truncated input (no
//! terminating byte) and overflow (an eleventh byte, or stray bits in the tenth
//! group that cannot fit `64` bits) by returning `None`.
//!
//! All arithmetic is integer only — shifts, masks, and `or` — with no `f32` and
//! no transcendental calls, so this reference is bit-reproducible against a
//! future `GPU` kernel.
//!
//! ## Boundaries
//!
//! This module owns *only* the single-integer variable-length byte codec. It is
//! deliberately distinct from its neighbours:
//!
//! * [`super::zigzag_delta_encode`] is a reversible integer *transform*: it maps
//!   signed values onto small unsigned magnitudes (`ZigZag`) and differences
//!   neighbours (delta). It rewrites the numbers; it does not produce a
//!   continuation-bit byte stream. `ZigZag` is a natural *pre-pass* feeding a
//!   varint, but the two share no types and no functions.
//! * [`super::run_length_encode`] collapses maximal runs of equal values into
//!   `(value, count)` pairs; it compresses *repetition*, not the width of an
//!   individual number.
//! * [`super::compression`] performs higher-level, often lossy numeric packing
//!   (`fp16`, `snorm`/`unorm`, octahedral). Everything here is exact and
//!   integer-preserving.

use alloc::vec::Vec;

/// The continuation flag: set on every `LEB128` byte that has a successor.
const CONTINUE: u8 = 0x80;
/// Mask selecting the seven payload bits of a `LEB128` byte.
const PAYLOAD: u8 = 0x7f;
/// The group sign bit (bit 6) inspected by the signed codec.
const GROUP_SIGN: u8 = 0x40;

/// Append the unsigned `LEB128` encoding of `value` to `out`.
///
/// Each byte carries seven payload bits, least-significant group first; the
/// high bit is set on every byte except the terminating one. `0` encodes as a
/// single `0x00` byte and `u64::MAX` as ten bytes.
pub fn encode_u64(mut value: u64, out: &mut Vec<u8>) {
    loop {
        let byte = (value & u64::from(PAYLOAD)) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | CONTINUE);
    }
}

/// Return the unsigned `LEB128` encoding of `value` as a fresh vector.
#[must_use]
pub fn encode_u64_to_vec(value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    encode_u64(value, &mut out);
    out
}

/// Decode one unsigned `LEB128` integer from the front of `bytes`.
///
/// On success returns `(value, consumed)` where `consumed` is the number of
/// bytes the integer occupied. Returns `None` when the input is truncated (no
/// terminating byte) or when the value overflows `u64` (an eleventh byte, or
/// non-zero padding bits in the tenth group).
#[must_use]
pub fn decode_u64(bytes: &[u8]) -> Option<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    let mut index: usize = 0;
    loop {
        let byte = *bytes.get(index)?;
        index += 1;
        if shift == 63 {
            // Tenth group: only bit 0 (value bit 63) fits a `u64`.
            if byte & 0x7e != 0 {
                return None;
            }
            if byte & CONTINUE != 0 {
                return None;
            }
            result |= u64::from(byte & PAYLOAD) << shift;
            return Some((result, index));
        }
        result |= u64::from(byte & PAYLOAD) << shift;
        if byte & CONTINUE == 0 {
            return Some((result, index));
        }
        shift += 7;
    }
}

/// Append the signed `LEB128` encoding of `value` to `out`.
///
/// Emission uses two's-complement sign extension: it stops once the remaining
/// high bits are all `0` with a clear group sign bit, or all `1` with a set
/// group sign bit. `-1` encodes as a single `0x7f` byte and `i64::MIN` as ten
/// bytes.
pub fn encode_i64(mut value: i64, out: &mut Vec<u8>) {
    loop {
        let byte = (value & i64::from(PAYLOAD)) as u8;
        value >>= 7;
        let sign = byte & GROUP_SIGN;
        if (value == 0 && sign == 0) || (value == -1 && sign != 0) {
            out.push(byte);
            return;
        }
        out.push(byte | CONTINUE);
    }
}

/// Return the signed `LEB128` encoding of `value` as a fresh vector.
#[must_use]
pub fn encode_i64_to_vec(value: i64) -> Vec<u8> {
    let mut out = Vec::new();
    encode_i64(value, &mut out);
    out
}

/// Decode one signed `LEB128` integer from the front of `bytes`.
///
/// On success returns `(value, consumed)`. The final group is sign-extended
/// across the high bits. Returns `None` on truncation (no terminating byte) or
/// overflow (an eleventh byte, or a tenth group whose bits are not a valid
/// sign extension of bit 63).
#[must_use]
pub fn decode_i64(bytes: &[u8]) -> Option<(i64, usize)> {
    let mut result: i64 = 0;
    let mut shift: u32 = 0;
    let mut index: usize = 0;
    loop {
        let byte = *bytes.get(index)?;
        index += 1;
        if shift == 63 {
            // Tenth group: the only value bit is bit 63; the remaining six bits
            // must be a pure sign extension (all `0` or all `1`).
            if byte & CONTINUE != 0 {
                return None;
            }
            let low = byte & PAYLOAD;
            if low != 0x00 && low != 0x7f {
                return None;
            }
            if low & 0x01 != 0 {
                result |= 1i64 << 63;
            }
            return Some((result, index));
        }
        result |= i64::from(byte & PAYLOAD) << shift;
        if byte & CONTINUE == 0 {
            let sign_pos = shift + 7;
            if sign_pos < 64 && byte & GROUP_SIGN != 0 {
                result |= -1i64 << sign_pos;
            }
            return Some((result, index));
        }
        shift += 7;
    }
}

/// Encode a slice of unsigned integers as one concatenated `LEB128` buffer.
#[must_use]
pub fn encode_u64_slice(values: &[u64]) -> Vec<u8> {
    let mut out = Vec::new();
    for &value in values {
        encode_u64(value, &mut out);
    }
    out
}

/// Decode every unsigned `LEB128` integer packed back to back in `bytes`.
///
/// Returns `None` if any integer is truncated or overflows, or if trailing
/// bytes do not form a complete final integer.
#[must_use]
pub fn decode_u64_all(bytes: &[u8]) -> Option<Vec<u64>> {
    let mut out = Vec::new();
    let mut offset: usize = 0;
    while offset < bytes.len() {
        let (value, consumed) = decode_u64(&bytes[offset..])?;
        out.push(value);
        offset += consumed;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small `LCG` producing deterministic pseudo-random 64-bit words.
    fn lcg_next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    }

    #[test]
    fn u64_zero_is_single_zero_byte() {
        assert_eq!(encode_u64_to_vec(0), Vec::from([0x00u8]));
    }

    #[test]
    fn u64_one_is_single_byte() {
        assert_eq!(encode_u64_to_vec(1), Vec::from([0x01u8]));
    }

    #[test]
    fn u64_127_is_single_byte_boundary() {
        assert_eq!(encode_u64_to_vec(127), Vec::from([0x7fu8]));
    }

    #[test]
    fn u64_128_is_two_bytes() {
        assert_eq!(encode_u64_to_vec(128), Vec::from([0x80u8, 0x01u8]));
    }

    #[test]
    fn u64_300_matches_known_encoding() {
        assert_eq!(encode_u64_to_vec(300), Vec::from([0xacu8, 0x02u8]));
    }

    #[test]
    fn u64_max_is_ten_bytes() {
        let encoded = encode_u64_to_vec(u64::MAX);
        assert_eq!(encoded.len(), 10);
        assert_eq!(
            encoded,
            Vec::from([0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01u8])
        );
    }

    #[test]
    fn u64_decode_zero() {
        assert_eq!(decode_u64(&[0x00]), Some((0, 1)));
    }

    #[test]
    fn u64_decode_reports_consumed_bytes() {
        let (value, consumed) = decode_u64(&[0x80, 0x01, 0xff]).unwrap();
        assert_eq!(value, 128);
        assert_eq!(consumed, 2);
    }

    #[test]
    fn u64_truncated_input_is_none() {
        // Continuation bit set but no following byte.
        assert_eq!(decode_u64(&[0x80]), None);
        assert_eq!(decode_u64(&[]), None);
    }

    #[test]
    fn u64_overflow_eleventh_byte_is_none() {
        let too_long = [0x80u8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80];
        assert_eq!(decode_u64(&too_long), None);
    }

    #[test]
    fn u64_overflow_stray_top_bits_is_none() {
        // Tenth group carries bits beyond bit 63.
        let stray = [0x80u8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x02];
        assert_eq!(decode_u64(&stray), None);
    }

    #[test]
    fn u64_roundtrip_selected_values() {
        let cases = [
            0u64,
            1,
            2,
            127,
            128,
            255,
            256,
            16_383,
            16_384,
            1_000_000,
            u64::from(u32::MAX),
            u64::MAX - 1,
            u64::MAX,
        ];
        for &value in &cases {
            let encoded = encode_u64_to_vec(value);
            let (decoded, consumed) = decode_u64(&encoded).unwrap();
            assert_eq!(decoded, value);
            assert_eq!(consumed, encoded.len());
        }
    }

    #[test]
    fn u64_to_vec_matches_in_place() {
        let mut buffer = Vec::new();
        encode_u64(987_654_321, &mut buffer);
        assert_eq!(buffer, encode_u64_to_vec(987_654_321));
    }

    #[test]
    fn i64_zero_is_single_zero_byte() {
        assert_eq!(encode_i64_to_vec(0), Vec::from([0x00u8]));
    }

    #[test]
    fn i64_negative_one_is_single_byte() {
        assert_eq!(encode_i64_to_vec(-1), Vec::from([0x7fu8]));
    }

    #[test]
    fn i64_sixty_three_is_single_byte() {
        assert_eq!(encode_i64_to_vec(63), Vec::from([0x3fu8]));
    }

    #[test]
    fn i64_sixty_four_needs_padding_byte() {
        // 64 has bit 6 set, which would read as negative in one group.
        assert_eq!(encode_i64_to_vec(64), Vec::from([0xc0u8, 0x00u8]));
    }

    #[test]
    fn i64_negative_sixty_five_is_two_bytes() {
        assert_eq!(encode_i64_to_vec(-65), Vec::from([0xbfu8, 0x7fu8]));
    }

    #[test]
    fn i64_min_is_ten_bytes() {
        let encoded = encode_i64_to_vec(i64::MIN);
        assert_eq!(encoded.len(), 10);
        assert_eq!(
            encoded,
            Vec::from([0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7fu8])
        );
    }

    #[test]
    fn i64_max_is_ten_bytes() {
        let encoded = encode_i64_to_vec(i64::MAX);
        assert_eq!(encoded.len(), 10);
        assert_eq!(
            encoded,
            Vec::from([0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00u8])
        );
    }

    #[test]
    fn i64_decode_negative_one() {
        assert_eq!(decode_i64(&[0x7f]), Some((-1, 1)));
    }

    #[test]
    fn i64_min_roundtrips() {
        let encoded = encode_i64_to_vec(i64::MIN);
        assert_eq!(decode_i64(&encoded), Some((i64::MIN, 10)));
    }

    #[test]
    fn i64_max_roundtrips() {
        let encoded = encode_i64_to_vec(i64::MAX);
        assert_eq!(decode_i64(&encoded), Some((i64::MAX, 10)));
    }

    #[test]
    fn i64_truncated_input_is_none() {
        assert_eq!(decode_i64(&[0x80]), None);
        assert_eq!(decode_i64(&[]), None);
    }

    #[test]
    fn i64_overflow_bad_tenth_group_is_none() {
        // Tenth group 0x02 is neither all-zero nor all-one sign extension.
        let bad = [0x80u8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x02];
        assert_eq!(decode_i64(&bad), None);
    }

    #[test]
    fn i64_overflow_eleventh_byte_is_none() {
        let too_long = [0x80u8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80];
        assert_eq!(decode_i64(&too_long), None);
    }

    #[test]
    fn i64_decode_reports_consumed_bytes() {
        let mut buffer = encode_i64_to_vec(-1000);
        buffer.push(0x55);
        let (value, consumed) = decode_i64(&buffer).unwrap();
        assert_eq!(value, -1000);
        assert_eq!(consumed, buffer.len() - 1);
    }

    #[test]
    fn i64_roundtrip_selected_values() {
        let cases = [
            0i64,
            1,
            -1,
            63,
            64,
            -64,
            -65,
            127,
            -128,
            128,
            i64::from(i32::MIN),
            i64::from(i32::MAX),
            i64::MIN + 1,
            i64::MIN,
            i64::MAX,
        ];
        for &value in &cases {
            let encoded = encode_i64_to_vec(value);
            let (decoded, consumed) = decode_i64(&encoded).unwrap();
            assert_eq!(decoded, value);
            assert_eq!(consumed, encoded.len());
        }
    }

    #[test]
    fn i64_to_vec_matches_in_place() {
        let mut buffer = Vec::new();
        encode_i64(-424_242, &mut buffer);
        assert_eq!(buffer, encode_i64_to_vec(-424_242));
    }

    #[test]
    fn batch_u64_roundtrips() {
        let values = [0u64, 1, 127, 128, 300, 70_000, u64::MAX];
        let packed = encode_u64_slice(&values);
        let decoded = decode_u64_all(&packed).unwrap();
        assert_eq!(decoded, Vec::from(values));
    }

    #[test]
    fn batch_empty_roundtrips() {
        let packed = encode_u64_slice(&[]);
        assert!(packed.is_empty());
        assert_eq!(decode_u64_all(&packed), Some(Vec::new()));
    }

    #[test]
    fn batch_truncated_tail_is_none() {
        let mut packed = encode_u64_slice(&[1, 2, 3]);
        packed.push(0x80); // dangling continuation with no terminator
        assert_eq!(decode_u64_all(&packed), None);
    }

    #[test]
    fn sequential_decode_walks_offsets() {
        let mut packed = Vec::new();
        encode_u64(10, &mut packed);
        encode_u64(1_000, &mut packed);
        encode_u64(100_000, &mut packed);
        let mut offset = 0;
        let (a, ca) = decode_u64(&packed[offset..]).unwrap();
        offset += ca;
        let (b, cb) = decode_u64(&packed[offset..]).unwrap();
        offset += cb;
        let (c, cc) = decode_u64(&packed[offset..]).unwrap();
        offset += cc;
        assert_eq!((a, b, c), (10, 1_000, 100_000));
        assert_eq!(offset, packed.len());
    }

    #[test]
    fn lcg_u64_roundtrip_is_identity() {
        let mut state = 0x1234_5678_9abc_def0u64;
        for _ in 0..2_000 {
            let value = lcg_next(&mut state);
            let encoded = encode_u64_to_vec(value);
            let (decoded, consumed) = decode_u64(&encoded).unwrap();
            assert_eq!(decoded, value);
            assert_eq!(consumed, encoded.len());
        }
    }

    #[test]
    fn lcg_i64_roundtrip_is_identity() {
        let mut state = 0x0fed_cba9_8765_4321u64;
        for _ in 0..2_000 {
            let value = lcg_next(&mut state) as i64;
            let encoded = encode_i64_to_vec(value);
            let (decoded, consumed) = decode_i64(&encoded).unwrap();
            assert_eq!(decoded, value);
            assert_eq!(consumed, encoded.len());
        }
    }

    #[test]
    fn lcg_batch_u64_roundtrip_is_identity() {
        let mut state = 0xdead_beef_cafe_babeu64;
        let values: Vec<u64> = (0..500).map(|_| lcg_next(&mut state)).collect();
        let packed = encode_u64_slice(&values);
        assert_eq!(decode_u64_all(&packed), Some(values));
    }
}
