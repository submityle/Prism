//! `CPU`-verifiable `CRC-31/PHILIPS` contract module for the Prism particle engine.
//!
//! This module implements a bit-wise `CRC` (cyclic redundancy check) using the
//! `CRC-31/PHILIPS` parameter set. The algorithm walks each input byte most
//! significant bit (`MSB`) first, shifting the register left and conditionally
//! applying the generator polynomial whenever the register top bit `XOR` the
//! incoming data bit is set. Least significant bit (`LSB`) first processing is
//! only used when input reflection is enabled, which this parameter set leaves
//! disabled.
//!
//! Parameters: width `31`, polynomial `0x04C1_1DB7`, init `0x7FFF_FFFF`,
//! xor-out `0x7FFF_FFFF`, no input reflection, no output reflection. Because the
//! width is `31`, the mask selects `31` low bits (`0x7FFF_FFFF`) and the top bit
//! is `1 << 30`; shifting `reg << 1` and masking keeps the register `31`-bit.
//! The canonical check value for the `ASCII` sequence `123456789` is
//! `0x0CE9_E46C`.
//!
//! The implementation is deterministic, allocation-free, and relies only on
//! integer arithmetic so it can be verified on the `CPU` without floating
//! point. The crate is `no_std` plus `alloc`, and this module touches neither
//! the heap nor any `std` or `alloc` collection.

/// `CRC` register width in bits.
const WIDTH: u32 = 31;
/// Generator polynomial (`MSB`-first form).
const POLY: u32 = 0x04C1_1DB7;
/// Initial register value.
const INIT: u32 = 0x7FFF_FFFF;
/// Final value `XOR`-ed into the register before returning.
const XOROUT: u32 = 0x7FFF_FFFF;
/// Whether each input byte is bit-reflected (`LSB`-first) before processing.
const REFLECT_IN: bool = false;
/// Whether the final register is bit-reflected before the `XOR`-out step.
const REFLECT_OUT: bool = false;
/// Mask selecting the low `WIDTH` bits.
const MASK: u32 = 0x7FFF_FFFF;
/// Top-bit selector for the register (`1 << (WIDTH - 1)`).
const TOPBIT: u32 = 1u32 << (WIDTH - 1);

/// Reflect the low `bits` of `value` (reverse their order).
const fn reflect(value: u32, bits: u32) -> u32 {
    let mut out = 0u32;
    let mut i = 0u32;
    while i < bits {
        if ((value >> i) & 1) != 0 {
            out |= 1u32 << (bits - 1 - i);
        }
        i += 1;
    }
    out
}

/// Compute the `CRC-31/PHILIPS` of `data`.
pub fn crc31_philips(data: &[u8]) -> u32 {
    let mut reg = INIT & MASK;
    let mut idx = 0usize;
    while idx < data.len() {
        let byte = data[idx];
        let b = if REFLECT_IN {
            reflect(u32::from(byte), 8)
        } else {
            u32::from(byte)
        };
        let mut i = 0u32;
        while i < 8 {
            let bit = (b >> (7 - i)) & 1;
            let msb = u32::from((reg & TOPBIT) != 0);
            reg = (reg << 1) & MASK;
            if (msb ^ bit) != 0 {
                reg ^= POLY;
            }
            reg &= MASK;
            i += 1;
        }
        idx += 1;
    }
    if REFLECT_OUT {
        reg = reflect(reg, WIDTH);
    }
    (reg ^ XOROUT) & MASK
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fill byte for the long repeated-pattern vector.
    const A5_FILL: u8 = 0xa5;
    /// Length of the long repeated-pattern vector.
    const A5_LEN: usize = 1000;

    #[test]
    fn vector_empty() {
        assert!(crc31_philips(b"") == 0x0000_0000);
    }

    #[test]
    fn vector_check() {
        assert!(crc31_philips(b"123456789") == 0x0CE9_E46C);
    }

    #[test]
    fn vector_z00() {
        assert!(crc31_philips(&[0x00]) == 0x22F3_3697);
    }

    #[test]
    fn vector_ff() {
        assert!(crc31_philips(&[0xff]) == 0x0000_00FF);
    }

    #[test]
    fn vector_fox() {
        assert!(crc31_philips(b"The quick brown fox") == 0x6FA9_6CE4);
    }

    #[test]
    fn vector_0123() {
        assert!(crc31_philips(&[0, 1, 2, 3]) == 0x2805_BE2D);
    }

    #[test]
    fn vector_a() {
        assert!(crc31_philips(b"a") == 0x0315_D6D9);
    }

    #[test]
    fn vector_b() {
        assert!(crc31_philips(b"b") == 0x0E56_F000);
    }

    #[test]
    fn vector_ab() {
        assert!(crc31_philips(b"ab") == 0x0106_64B2);
    }

    #[test]
    fn vector_abc() {
        assert!(crc31_philips(b"abc") == 0x0571_64D9);
    }

    #[test]
    fn vector_01() {
        assert!(crc31_philips(&[0x01]) == 0x2632_2B20);
    }

    #[test]
    fn vector_02() {
        assert!(crc31_philips(&[0x02]) == 0x2B71_0DF9);
    }

    #[test]
    fn vector_7f() {
        assert!(crc31_philips(&[0x7f]) == 0x738A_ADA3);
    }

    #[test]
    fn vector_80() {
        assert!(crc31_philips(&[0x80]) == 0x5179_9BCB);
    }

    #[test]
    fn vector_aa_55() {
        assert!(crc31_philips(&[0xaa, 0x55]) == 0x084F_170C);
    }

    #[test]
    fn vector_55_aa() {
        assert!(crc31_philips(&[0x55, 0xaa]) == 0x778A_74A1);
    }

    #[test]
    fn vector_deadbeef() {
        assert!(crc31_philips(&[0xde, 0xad, 0xbe, 0xef]) == 0x183C_0190);
    }

    #[test]
    fn vector_hello() {
        assert!(crc31_philips(b"Hello") == 0x38F2_EEA2);
    }

    #[test]
    fn vector_two_zeros() {
        assert!(crc31_philips(&[0, 0]) == 0x7FC5_9C52);
    }

    #[test]
    fn vector_four_zeros() {
        assert!(crc31_philips(&[0u8; 4]) == 0x6660_AFAA);
    }

    #[test]
    fn vector_four_ff() {
        assert!(crc31_philips(&[0xffu8; 4]) == 0x7B3E_E248);
    }

    #[test]
    fn vector_two_ff() {
        assert!(crc31_philips(&[0xff, 0xff]) == 0x0000_FFFF);
    }

    #[test]
    fn vector_12345678_bytes() {
        assert!(crc31_philips(&[0x12, 0x34, 0x56, 0x78]) == 0x0E70_B9E7);
    }

    #[test]
    fn vector_0_15() {
        let mut buf = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc31_philips(&buf) == 0x7F89_BE22);
    }

    #[test]
    fn vector_all256() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc31_philips(&buf) == 0x5861_6F16);
    }

    #[test]
    fn vector_a5_1000() {
        let buf = [A5_FILL; A5_LEN];
        assert!(crc31_philips(&buf) == 0x02CC_EBC4);
    }

    #[test]
    fn check_constant_is_0x0ce9e46c() {
        let expected: u32 = 0x0CE9_E46C;
        assert!(crc31_philips(b"123456789") == expected);
    }

    #[test]
    fn result_fits_in_31_bits() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!((crc31_philips(&buf) & !MASK) == 0);
    }

    #[test]
    fn single_byte_results_fit_in_31_bits() {
        let mut byte = 0u16;
        while byte < 256 {
            let single = [byte as u8];
            assert!((crc31_philips(&single) & !MASK) == 0);
            byte += 1;
        }
    }

    #[test]
    fn empty_equals_init_xor_xorout() {
        assert!(crc31_philips(b"") == ((INIT ^ XOROUT) & MASK));
    }

    #[test]
    fn determinism_empty() {
        assert!(crc31_philips(b"") == crc31_philips(b""));
    }

    #[test]
    fn determinism_fox() {
        assert!(crc31_philips(b"The quick brown fox") == crc31_philips(b"The quick brown fox"));
    }

    #[test]
    fn determinism_all256() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc31_philips(&buf) == crc31_philips(&buf));
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc31_philips(b"a") != crc31_philips(b"b"));
    }

    #[test]
    fn prefix_ab_differs_from_abc() {
        assert!(crc31_philips(b"ab") != crc31_philips(b"abc"));
    }

    #[test]
    fn length_sensitive_ff() {
        assert!(crc31_philips(&[0xff]) != crc31_philips(&[0xff, 0xff]));
    }

    #[test]
    fn single_byte_01_differs_from_02() {
        assert!(crc31_philips(&[0x01]) != crc31_philips(&[0x02]));
    }

    #[test]
    fn single_byte_7f_differs_from_80() {
        assert!(crc31_philips(&[0x7f]) != crc31_philips(&[0x80]));
    }

    #[test]
    fn single_byte_00_differs_from_ff() {
        assert!(crc31_philips(&[0x00]) != crc31_philips(&[0xff]));
    }

    #[test]
    fn aa55_differs_from_55aa() {
        assert!(crc31_philips(&[0xaa, 0x55]) != crc31_philips(&[0x55, 0xaa]));
    }

    #[test]
    fn two_zeros_differs_from_empty() {
        assert!(crc31_philips(&[0, 0]) != crc31_philips(b""));
    }

    #[test]
    fn single_byte_sample_pairwise_distinct() {
        let samples: [u8; 6] = [0x00, 0x01, 0x02, 0x7f, 0x80, 0xff];
        let mut i = 0usize;
        while i < samples.len() {
            let mut j = i + 1;
            while j < samples.len() {
                assert!(crc31_philips(&[samples[i]]) != crc31_philips(&[samples[j]]));
                j += 1;
            }
            i += 1;
        }
    }

    #[test]
    fn output_deterministic_over_all_bytes() {
        let mut byte = 0u16;
        while byte < 256 {
            let single = [byte as u8];
            assert!(crc31_philips(&single) == crc31_philips(&single));
            byte += 1;
        }
    }

    #[test]
    fn reflect_reverses_bits() {
        assert!(reflect(0b0000_0001, 8) == 0b1000_0000);
    }

    #[test]
    fn reflect_is_involution_over_width() {
        let value: u32 = 0x1234_5678 & MASK;
        assert!(reflect(reflect(value, WIDTH), WIDTH) == value);
    }
}
