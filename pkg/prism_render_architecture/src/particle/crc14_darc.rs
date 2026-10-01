//! Bit-level `CRC`-14/`DARC` checksum: pure-integer CPU gold standard.
//!
//! This module implements the `CRC`-14/`DARC` cyclic redundancy check as a
//! straightforward, allocation-free, bit-serial reference. It is intended as a
//! trustworthy golden implementation against which faster table- or
//! `SIMD`-driven variants can be validated.
//!
//! Parameters (per the Rocksoft model):
//! - `width` = 14
//! - `poly` = `0x0805`
//! - `init` = `0x0000`
//! - `refin` = `true`
//! - `refout` = `true`
//! - `xorout` = `0x0`
//! - `mask` = `(1 << 14) - 1` = `0x3FFF`
//!
//! The core routine consumes a `&[u8]` and returns a `u16` whose low 14 bits
//! hold the checksum (the result is always `<= 0x3FFF`). Because both `refin`
//! and `refout` are set, each input byte is reflected before processing and the
//! final register is reflected before being returned.
//!
//! Reference check value: `checksum(b"123456789") == 0x082D`.

/// Register width in bits for `CRC`-14/`DARC`.
pub const WIDTH: u32 = 14;

/// Generator polynomial for `CRC`-14/`DARC` (`poly` in the Rocksoft model).
pub const POLY: u16 = 0x0805;

/// Initial register value.
pub const INIT: u16 = 0x0000;

/// Output `XOR` value.
pub const XOROUT: u16 = 0x0000;

/// Low-14-bit mask: `(1 << WIDTH) - 1`.
pub const MASK: u16 = ((1u32 << WIDTH) - 1) as u16;

/// Reflect the low `bits` bits of `value`, i.e. reverse their order.
///
/// For example, reflecting `0b0000_0000_0000_0001` over 14 bits yields
/// `0b10_0000_0000_0000` (`0x2000`). Reflection over the same width is its own
/// inverse for any value that fits in `bits` bits.
fn reflect(value: u16, bits: u32) -> u16 {
    let mut result: u16 = 0;
    let mut i: u32 = 0;
    while i < bits {
        if ((value >> i) & 1) == 1 {
            result |= 1u16 << ((bits - 1) - i);
        }
        i += 1;
    }
    result
}

/// Reflect all 8 bits of a single byte (`refin` handling).
fn reflect8(byte: u8) -> u8 {
    reflect(byte as u16, 8) as u8
}

/// Compute the `CRC`-14/`DARC` checksum starting from an explicit register
/// value `init`.
///
/// Only the low 14 bits of `init` are significant; higher bits are masked away.
/// The returned `u16` is always `<= MASK` (`0x3FFF`).
pub fn checksum_with_init(init: u16, data: &[u8]) -> u16 {
    const { assert!(WIDTH == 14) };
    const { assert!(POLY <= 0x3FFF) };
    const { assert!(MASK == 0x3FFF) };
    const { assert!(INIT == 0x0000) };
    const { assert!(XOROUT == 0x0000) };

    let mut reg: u16 = init & MASK;
    let mut idx: usize = 0;
    while idx < data.len() {
        let byte: u8 = reflect8(data[idx]);
        let mut bitpos: u32 = 8;
        while bitpos > 0 {
            bitpos -= 1;
            let bit: u16 = ((byte >> bitpos) & 1) as u16;
            let top: u16 = (reg >> (WIDTH - 1)) & 1;
            reg = (reg << 1) & MASK;
            if (top ^ bit) == 1 {
                reg ^= POLY;
            }
        }
        idx += 1;
    }
    (reflect(reg, WIDTH) ^ XOROUT) & MASK
}

/// Compute the `CRC`-14/`DARC` checksum of `data` using the standard `init`.
///
/// The returned `u16` is always `<= MASK` (`0x3FFF`).
pub fn checksum(data: &[u8]) -> u16 {
    checksum_with_init(INIT, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only helper: resume a reflected-output `CRC` from a prior result.
    ///
    /// Because `refin` and `refout` are both set, reflecting a previous output
    /// recovers the raw register, so continuing from it yields the checksum of
    /// the concatenated input.
    #[cfg(test)]
    fn resume_from(prev: u16, rest: &[u8]) -> u16 {
        checksum_with_init(reflect(prev, WIDTH), rest)
    }

    // ---- Enshrined hard reference vectors ----

    #[test]
    fn vector_check_123456789() {
        assert_eq!(checksum(b"123456789"), 0x082D);
    }

    #[test]
    fn vector_empty() {
        assert_eq!(checksum(b""), 0x0);
    }

    #[test]
    fn vector_single_a() {
        assert_eq!(checksum(b"A"), 0x2D0);
    }

    #[test]
    fn vector_abc() {
        assert_eq!(checksum(b"abc"), 0x30F3);
    }

    #[test]
    fn vector_single_zero_byte() {
        assert_eq!(checksum(&[0x00u8]), 0x0);
    }

    #[test]
    fn vector_single_ff_byte() {
        assert_eq!(checksum(&[0xFFu8]), 0x3DB6);
    }

    #[test]
    fn vector_quick_brown_fox() {
        assert_eq!(checksum(b"The quick brown fox"), 0x1E73);
    }

    #[test]
    fn vector_two_zero_bytes() {
        assert_eq!(checksum(&[0x00u8, 0x00u8]), 0x0);
    }

    #[test]
    fn vector_four_ff_bytes() {
        assert_eq!(checksum(&[0xFFu8, 0xFFu8, 0xFFu8, 0xFFu8]), 0x1BF6);
    }

    #[test]
    fn vector_table_all() {
        let cases: [(&[u8], u16); 9] = [
            (b"123456789", 0x082D),
            (b"", 0x0),
            (b"A", 0x2D0),
            (b"abc", 0x30F3),
            (&[0x00u8], 0x0),
            (&[0xFFu8], 0x3DB6),
            (b"The quick brown fox", 0x1E73),
            (&[0x00u8, 0x00u8], 0x0),
            (&[0xFFu8, 0xFFu8, 0xFFu8, 0xFFu8], 0x1BF6),
        ];
        let mut i: usize = 0;
        while i < cases.len() {
            let (data, expected) = cases[i];
            assert_eq!(checksum(data), expected);
            i += 1;
        }
    }

    // ---- Result-bound invariant: always <= 0x3FFF ----

    #[test]
    fn bounded_known_vectors() {
        let samples: [&[u8]; 5] = [b"123456789", b"", b"A", b"abc", b"The quick brown fox"];
        let mut i: usize = 0;
        while i < samples.len() {
            assert!(checksum(samples[i]) <= 0x3FFF);
            i += 1;
        }
    }

    #[test]
    fn bounded_all_single_bytes() {
        let mut b: u16 = 0;
        while b <= 0xFF {
            let input: [u8; 1] = [b as u8];
            assert!(checksum(&input) <= MASK);
            b += 1;
        }
    }

    #[test]
    fn bounded_all_single_bytes_with_mask_check() {
        let mut b: u16 = 0;
        while b <= 0xFF {
            let input: [u8; 1] = [b as u8];
            let c = checksum(&input);
            assert_eq!(c & MASK, c);
            b += 1;
        }
    }

    #[test]
    fn bounded_longer_message() {
        let msg: [u8; 16] = [
            0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99,
            0xAA, 0xBB,
        ];
        assert!(checksum(&msg) <= MASK);
    }

    #[test]
    fn bounded_repeated_zero_bytes() {
        let zeros: [u8; 32] = [0u8; 32];
        let mut n: usize = 0;
        while n <= zeros.len() {
            assert!(checksum(&zeros[..n]) <= MASK);
            n += 1;
        }
    }

    #[test]
    fn bounded_increasing_bytes() {
        let mut buf: [u8; 64] = [0u8; 64];
        let mut i: usize = 0;
        while i < buf.len() {
            buf[i] = (i as u8).wrapping_mul(37);
            i += 1;
        }
        assert!(checksum(&buf) <= MASK);
    }

    // ---- Empty-input behaviour ----

    #[test]
    fn empty_input_default_is_zero() {
        assert_eq!(checksum(&[]), 0x0);
    }

    #[test]
    fn empty_input_reflects_init() {
        // With no data, the result is simply reflect(init) ^ xorout.
        let mut init: u16 = 0;
        while init <= MASK {
            assert_eq!(checksum_with_init(init, &[]), reflect(init, WIDTH));
            init += 0x111;
        }
    }

    #[test]
    fn empty_slice_variants_match() {
        let a: [u8; 0] = [];
        assert_eq!(checksum(&a), checksum(b""));
    }

    // ---- checksum vs checksum_with_init ----

    #[test]
    fn with_init_default_matches_checksum() {
        let samples: [&[u8]; 4] = [b"123456789", b"abc", b"A", b"The quick brown fox"];
        let mut i: usize = 0;
        while i < samples.len() {
            assert_eq!(checksum_with_init(INIT, samples[i]), checksum(samples[i]));
            i += 1;
        }
    }

    // ---- Chunked / incremental consistency ----

    #[test]
    fn chunked_123456789_all_splits() {
        let data = b"123456789";
        let full = checksum(data);
        let mut split: usize = 0;
        while split <= data.len() {
            let (head, tail) = data.split_at(split);
            assert_eq!(resume_from(checksum(head), tail), full);
            split += 1;
        }
    }

    #[test]
    fn chunked_fox_all_splits() {
        let data = b"The quick brown fox";
        let full = checksum(data);
        let mut split: usize = 0;
        while split <= data.len() {
            let (head, tail) = data.split_at(split);
            assert_eq!(resume_from(checksum(head), tail), full);
            split += 1;
        }
    }

    #[test]
    fn chunked_abc_all_splits() {
        let data = b"abc";
        let full = checksum(data);
        let mut split: usize = 0;
        while split <= data.len() {
            let (head, tail) = data.split_at(split);
            assert_eq!(resume_from(checksum(head), tail), full);
            split += 1;
        }
    }

    #[test]
    fn chunked_four_ff_all_splits() {
        let data: [u8; 4] = [0xFF, 0xFF, 0xFF, 0xFF];
        let full = checksum(&data);
        let mut split: usize = 0;
        while split <= data.len() {
            let (head, tail) = data.split_at(split);
            assert_eq!(resume_from(checksum(head), tail), full);
            split += 1;
        }
    }

    #[test]
    fn byte_by_byte_consistency() {
        let data = b"123456789";
        let full = checksum(data);
        let mut acc: u16 = INIT;
        let mut i: usize = 0;
        while i < data.len() {
            let one: [u8; 1] = [data[i]];
            acc = reflect(checksum_with_init(acc, &one), WIDTH);
            i += 1;
        }
        assert_eq!(reflect(acc, WIDTH), full);
    }

    #[test]
    fn chunked_only_when_length_is_multiple_of_chunk() {
        let data: [u8; 12] = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C,
        ];
        let chunk: usize = 4;
        // Guard the even-chunking path with an unsigned divisibility check.
        assert!(data.len().is_multiple_of(chunk));
        let full = checksum(&data);
        let mut acc: u16 = INIT;
        let mut i: usize = 0;
        while i < data.len() {
            let end = i + chunk;
            acc = reflect(checksum_with_init(acc, &data[i..end]), WIDTH);
            i = end;
        }
        assert_eq!(reflect(acc, WIDTH), full);
    }

    #[test]
    fn width_is_multiple_of_seven() {
        // Sanity check on the width using unsigned divisibility.
        assert!(WIDTH.is_multiple_of(7));
        assert!(!WIDTH.is_multiple_of(5));
    }

    // ---- Order sensitivity / prefix sensitivity ----

    #[test]
    fn order_matters() {
        assert_ne!(checksum(b"ab"), checksum(b"ba"));
    }

    #[test]
    fn prefix_changes_checksum() {
        assert_ne!(checksum(b"abc"), checksum(b"Xabc"));
    }

    #[test]
    fn appending_changes_checksum() {
        assert_ne!(checksum(b"abc"), checksum(b"abcd"));
    }

    // ---- Bit reflection correctness ----

    #[test]
    fn reflect8_low_bit() {
        assert_eq!(reflect8(0x01), 0x80);
    }

    #[test]
    fn reflect8_high_bit() {
        assert_eq!(reflect8(0x80), 0x01);
    }

    #[test]
    fn reflect8_all_ones() {
        assert_eq!(reflect8(0xFF), 0xFF);
    }

    #[test]
    fn reflect8_zero() {
        assert_eq!(reflect8(0x00), 0x00);
    }

    #[test]
    fn reflect8_nibble() {
        assert_eq!(reflect8(0x0F), 0xF0);
    }

    #[test]
    fn reflect8_specific_pattern() {
        assert_eq!(reflect8(0b0000_0010), 0b0100_0000);
        assert_eq!(reflect8(0b1010_0000), 0b0000_0101);
    }

    #[test]
    fn reflect8_is_involution() {
        let mut b: u16 = 0;
        while b <= 0xFF {
            let v = b as u8;
            assert_eq!(reflect8(reflect8(v)), v);
            b += 1;
        }
    }

    #[test]
    fn reflect14_low_bit() {
        assert_eq!(reflect(0x1, WIDTH), 0x2000);
    }

    #[test]
    fn reflect14_high_bit() {
        assert_eq!(reflect(0x2000, WIDTH), 0x1);
    }

    #[test]
    fn reflect14_full_mask() {
        assert_eq!(reflect(MASK, WIDTH), MASK);
    }

    #[test]
    fn reflect14_zero() {
        assert_eq!(reflect(0x0, WIDTH), 0x0);
    }

    #[test]
    fn reflect14_is_involution() {
        let mut v: u16 = 0;
        while v <= MASK {
            assert_eq!(reflect(reflect(v, WIDTH), WIDTH), v);
            v += 0x51;
        }
    }

    #[test]
    fn reflect_stays_in_width() {
        let mut v: u16 = 0;
        while v <= MASK {
            assert!(reflect(v, WIDTH) <= MASK);
            v += 0x37;
        }
    }

    // ---- Constant sanity ----

    #[test]
    fn const_width() {
        assert_eq!(WIDTH, 14);
    }

    #[test]
    fn const_poly() {
        assert_eq!(POLY, 0x0805);
    }

    #[test]
    fn const_mask() {
        assert_eq!(MASK, 0x3FFF);
        assert_eq!(MASK, ((1u32 << WIDTH) - 1) as u16);
    }

    #[test]
    fn const_init_and_xorout() {
        assert_eq!(INIT, 0x0000);
        assert_eq!(XOROUT, 0x0000);
    }

    #[test]
    fn poly_fits_in_width() {
        assert!(POLY <= MASK);
    }
}
