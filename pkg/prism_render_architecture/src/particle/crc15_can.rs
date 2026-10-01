//! `CRC`-15/`CAN` bit-level checksum: pure-integer, CPU gold reference.
//!
//! Parameters (`CAN` standard): width 15, `poly` `0x4599`, init `0x0000`,
//! `refin` = false, `refout` = false, `xorout` `0x0`. This module is the
//! **non-reflected**, `MSB`-first variant, so there is no byte or register
//! reflection anywhere in this file. The core operates on `&[u8]` and returns
//! a `u16` whose value always fits in the low 15 bits (`<= 0x7FFF`).

/// Register width in bits for `CRC`-15/`CAN`.
pub const WIDTH: u32 = 15;

/// Generator polynomial for `CRC`-15/`CAN` (`0x4599`).
pub const POLY: u16 = 0x4599;

/// Initial register value for `CRC`-15/`CAN` (`0x0000`).
pub const INIT: u16 = 0x0000;

/// Low-15-bit mask applied after every shift and on output.
const MASK: u16 = (1 << WIDTH) - 1;

const _: () = {
    const { assert!(WIDTH == 15) };
    const { assert!(POLY == 0x4599) };
    const { assert!(INIT == 0x0000) };
    const { assert!(MASK == 0x7FFF) };
};

/// Compute the `CRC`-15/`CAN` checksum of `data` using the standard init
/// (`0x0000`). The result is the 15-bit register value (`<= 0x7FFF`).
pub fn checksum(data: &[u8]) -> u16 {
    checksum_with_init(INIT, data)
}

/// Compute the `CRC`-15/`CAN` checksum of `data` starting from an explicit
/// 15-bit register `init`. Because there is no reflection and no `xorout`,
/// the returned `u16` is also a valid resume state: feeding it back as `init`
/// for the next slice continues the running `CRC` over the concatenation.
pub fn checksum_with_init(init: u16, data: &[u8]) -> u16 {
    let mut reg = init & MASK;
    let mut idx = 0;
    while idx < data.len() {
        let byte = data[idx];
        let mut i: u32 = 8;
        while i > 0 {
            i -= 1;
            let bit = ((byte >> i) & 1) as u16;
            let top = (reg >> (WIDTH - 1)) & 1;
            reg = (reg << 1) & MASK;
            if (top ^ bit) == 1 {
                reg ^= POLY;
            }
        }
        idx += 1;
    }
    reg & MASK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(test)]
    fn split_consistency(data: &[u8], at: usize) -> bool {
        let full = checksum(data);
        let (head, tail) = data.split_at(at);
        let resumed = checksum_with_init(checksum(head), tail);
        full == resumed
    }

    #[test]
    fn vector_check_123456789() {
        assert_eq!(checksum(b"123456789"), 0x059E);
    }

    #[test]
    fn vector_empty() {
        assert_eq!(checksum(b""), 0x0);
    }

    #[test]
    fn vector_empty_slice_literal() {
        let data: &[u8] = &[];
        assert_eq!(checksum(data), 0x0);
    }

    #[test]
    fn vector_a() {
        assert_eq!(checksum(b"A"), 0x365C);
    }

    #[test]
    fn vector_abc() {
        assert_eq!(checksum(b"abc"), 0x6AB2);
    }

    #[test]
    fn vector_single_zero() {
        assert_eq!(checksum(&[0x00]), 0x0);
    }

    #[test]
    fn vector_single_ff() {
        assert_eq!(checksum(&[0xFF]), 0x95);
    }

    #[test]
    fn vector_quick_brown_fox() {
        assert_eq!(checksum(b"The quick brown fox"), 0x3EA5);
    }

    #[test]
    fn vector_two_zeros() {
        assert_eq!(checksum(&[0x00, 0x00]), 0x0);
    }

    #[test]
    fn vector_four_ff() {
        assert_eq!(checksum(&[0xFF, 0xFF, 0xFF, 0xFF]), 0x5D58);
    }

    #[test]
    fn many_zeros_stay_zero() {
        assert_eq!(checksum(&[0x00; 16]), 0x0);
    }

    #[test]
    fn init_mask_is_applied() {
        // An out-of-range init must be masked to 15 bits before use.
        let masked = checksum_with_init(0x7FFF, b"");
        let overflow = checksum_with_init(0xFFFF, b"");
        assert_eq!(masked, overflow);
        assert_eq!(overflow, 0x7FFF);
    }

    #[test]
    fn default_init_matches_explicit() {
        let data = b"prism";
        assert_eq!(checksum(data), checksum_with_init(INIT, data));
    }

    #[test]
    fn result_in_range_vectors() {
        assert!(checksum(b"123456789") <= 0x7FFF);
        assert!(checksum(b"The quick brown fox") <= 0x7FFF);
        assert!(checksum(&[0xFF, 0xFF, 0xFF, 0xFF]) <= 0x7FFF);
    }

    #[test]
    fn result_in_range_all_single_bytes() {
        let mut b: u16 = 0;
        while b <= 0xFF {
            let input = [b as u8];
            assert!(checksum(&input) <= MASK);
            b += 1;
        }
    }

    #[test]
    fn result_in_range_sweep_pairs() {
        let samples: [[u8; 2]; 6] = [
            [0x00, 0x01],
            [0x10, 0x20],
            [0x7F, 0x80],
            [0xAA, 0x55],
            [0xFE, 0x01],
            [0xFF, 0xFF],
        ];
        let mut i = 0;
        while i < samples.len() {
            assert!(checksum(&samples[i]) <= 0x7FFF);
            i += 1;
        }
    }

    #[test]
    fn order_sensitive_ab_ba() {
        // Non-reflected MSB-first CRC is order sensitive.
        assert_ne!(checksum(b"AB"), checksum(b"BA"));
    }

    #[test]
    fn order_sensitive_abc_cba() {
        assert_ne!(checksum(b"abc"), checksum(b"cba"));
    }

    #[test]
    fn order_sensitive_01_10() {
        assert_ne!(checksum(&[0x01, 0x02]), checksum(&[0x02, 0x01]));
    }

    #[test]
    fn order_sensitive_distinct_triples() {
        let a = checksum(&[0x12, 0x34, 0x56]);
        let b = checksum(&[0x56, 0x34, 0x12]);
        let c = checksum(&[0x34, 0x12, 0x56]);
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
    }

    #[test]
    fn leading_zero_from_zero_init_is_invariant() {
        // With init 0x0000 the register starts empty, so a leading zero byte
        // leaves it unchanged: prepending 0x00 does not alter the result.
        assert_eq!(checksum(&[0x00, 0xAB]), checksum(&[0xAB]));
    }

    #[test]
    fn trailing_byte_changes_result() {
        // Appending data advances the register and must change the output.
        assert_ne!(checksum(&[0xAB]), checksum(&[0xAB, 0x00]));
        assert_ne!(checksum(&[0xAB]), checksum(&[0xAB, 0xCD]));
    }

    #[test]
    fn chunk_consistency_abc() {
        let data = b"abc";
        assert!(split_consistency(data, 0));
        assert!(split_consistency(data, 1));
        assert!(split_consistency(data, 2));
        assert!(split_consistency(data, 3));
    }

    #[test]
    fn chunk_consistency_check_vector() {
        let data = b"123456789";
        let mut at = 0;
        while at <= data.len() {
            assert!(split_consistency(data, at));
            at += 1;
        }
    }

    #[test]
    fn chunk_consistency_fox() {
        let data = b"The quick brown fox";
        let mut at = 0;
        while at <= data.len() {
            assert!(split_consistency(data, at));
            at += 1;
        }
    }

    #[test]
    fn chunk_at_even_boundaries() {
        let data = b"abcdefghij";
        let mut at = 0;
        while at <= data.len() {
            if at.is_multiple_of(2) {
                assert!(split_consistency(data, at));
            }
            at += 1;
        }
    }

    #[test]
    fn chunk_three_way_split() {
        let data = b"0123456789ABCDEF";
        let s1 = checksum(&data[..5]);
        let s2 = checksum_with_init(s1, &data[5..11]);
        let s3 = checksum_with_init(s2, &data[11..]);
        assert_eq!(s3, checksum(data));
    }

    #[test]
    fn resume_from_full_equals_concat() {
        let left = b"hello";
        let right = b"world";
        let mut joined = [0u8; 10];
        joined[..5].copy_from_slice(left);
        joined[5..].copy_from_slice(right);
        let resumed = checksum_with_init(checksum(left), right);
        assert_eq!(resumed, checksum(&joined));
    }

    #[test]
    fn empty_resume_is_identity() {
        let data = b"state";
        let s = checksum(data);
        assert_eq!(checksum_with_init(s, b""), s);
    }

    #[test]
    fn single_bit_difference_detected() {
        assert_ne!(checksum(&[0x00]), checksum(&[0x01]));
        assert_ne!(checksum(&[0x80]), checksum(&[0x00]));
    }

    #[test]
    fn all_single_bytes_bounded_and_deterministic() {
        let mut b: u16 = 0;
        while b <= 0xFF {
            let input = [b as u8];
            let first = checksum(&input);
            let second = checksum(&input);
            assert_eq!(first, second);
            assert!(first <= MASK);
            b += 1;
        }
    }

    #[test]
    fn deterministic_repeat() {
        let data = b"determinism";
        assert_eq!(checksum(data), checksum(data));
    }

    #[test]
    fn poly_constant_value() {
        assert_eq!(POLY, 0x4599);
    }

    #[test]
    fn width_constant_value() {
        assert_eq!(WIDTH, 15);
    }

    #[test]
    fn mask_matches_width() {
        assert_eq!(MASK, (1u16 << WIDTH) - 1);
        assert_eq!(MASK, 0x7FFF);
    }

    #[test]
    fn init_constant_value() {
        assert_eq!(INIT, 0x0000);
    }

    #[test]
    fn high_bit_byte_sets_poly() {
        // 0x80 flips the top bit on the very first step, forcing a poly xor.
        let expected = {
            let mut reg: u16 = 0;
            let top = (reg >> (WIDTH - 1)) & 1;
            reg = (reg << 1) & MASK;
            if (top ^ 1) == 1 {
                reg ^= POLY;
            }
            // remaining 7 zero bits
            let mut k = 0;
            while k < 7 {
                let t = (reg >> (WIDTH - 1)) & 1;
                reg = (reg << 1) & MASK;
                if (t ^ 0) == 1 {
                    reg ^= POLY;
                }
                k += 1;
            }
            reg
        };
        assert_eq!(checksum(&[0x80]), expected);
    }

    #[test]
    fn zero_byte_from_zero_state_is_zero() {
        // Feeding zero bytes from the zero register keeps it zero.
        assert_eq!(checksum_with_init(0, &[0x00, 0x00, 0x00]), 0);
    }

    #[test]
    fn longer_input_bounded() {
        let data = b"The quick brown fox jumps over the lazy dog";
        assert!(checksum(data) <= 0x7FFF);
    }

    #[test]
    fn repeated_pattern_bounded() {
        let data = [0xA5u8; 32];
        assert!(checksum(&data) <= MASK);
    }

    #[test]
    fn distinct_lengths_distinct_from_empty() {
        assert_ne!(checksum(b"x"), checksum(b""));
    }

    #[test]
    fn concatenation_associative() {
        let a = b"aa";
        let b = b"bb";
        let c = b"cc";
        let via_ab = checksum_with_init(checksum_with_init(checksum(a), b), c);
        let mut joined = [0u8; 6];
        joined[..2].copy_from_slice(a);
        joined[2..4].copy_from_slice(b);
        joined[4..].copy_from_slice(c);
        assert_eq!(via_ab, checksum(&joined));
    }
}
