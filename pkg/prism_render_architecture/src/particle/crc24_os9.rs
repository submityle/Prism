//! `CPU`-verifiable `CRC-24/OS-9` contract module for the Prism particle engine.
//!
//! This module implements a bit-wise `CRC` (cyclic redundancy check) using the
//! `CRC-24/OS-9` parameter set. The algorithm walks each input byte most
//! significant bit (`MSB`) first, shifting the register left and conditionally
//! applying the generator polynomial whenever the register top bit `XOR` the
//! incoming data bit is set. Least significant bit (`LSB`) first processing is
//! only used when input reflection is enabled, which this parameter set leaves
//! disabled.
//!
//! Parameters: width `24`, polynomial `0x0080_0063`, init `0x00FF_FFFF`,
//! xor-out `0x00FF_FFFF`, no input reflection, no output reflection. The
//! canonical check value for the `ASCII` sequence `123456789` is `0x20_0FA5`.
//!
//! The implementation is deterministic, allocation-free, and relies only on
//! integer arithmetic so it can be verified on the `CPU` without floating
//! point. It is compatible with `no_std` plus `alloc` environments because it
//! touches neither the standard library nor any heap collection.

/// `CRC` register width in bits.
const WIDTH: u32 = 24;
/// Generator polynomial (`MSB`-first form).
const POLY: u32 = 0x0080_0063;
/// Initial register value.
const INIT: u32 = 0x00FF_FFFF;
/// Final value `XOR`-ed into the register before returning.
const XOROUT: u32 = 0x00FF_FFFF;
/// Whether each input byte is bit-reflected (`LSB`-first) before processing.
const REFLECT_IN: bool = false;
/// Whether the final register is bit-reflected before the `XOR`-out step.
const REFLECT_OUT: bool = false;
/// Mask selecting the low `WIDTH` bits.
const MASK: u32 = 0x00FF_FFFF;
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

/// Compute the `CRC-24/OS-9` of `data`.
pub fn crc24_os9(data: &[u8]) -> u32 {
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
        assert!(crc24_os9(b"") == 0x00_0000);
    }

    #[test]
    fn vector_check() {
        assert!(crc24_os9(b"123456789") == 0x20_0FA5);
    }

    #[test]
    fn vector_z00() {
        assert!(crc24_os9(&[0x00]) == 0x00_3EC1);
    }

    #[test]
    fn vector_ff() {
        assert!(crc24_os9(&[0xff]) == 0x00_00FF);
    }

    #[test]
    fn vector_fox() {
        assert!(crc24_os9(b"The quick brown fox") == 0x2C_4CFB);
    }

    #[test]
    fn vector_0123() {
        assert!(crc24_os9(&[0, 1, 2, 3]) == 0x32_B918);
    }

    #[test]
    fn vector_a() {
        assert!(crc24_os9(b"a") == 0x80_2662);
    }

    #[test]
    fn vector_b() {
        assert!(crc24_os9(b"b") == 0x80_26A4);
    }

    #[test]
    fn vector_ab() {
        assert!(crc24_os9(b"ab") == 0x26_6585);
    }

    #[test]
    fn vector_abc() {
        assert!(crc24_os9(b"abc") == 0xE5_AA2A);
    }

    #[test]
    fn vector_two_zeros() {
        assert!(crc24_os9(&[0, 0]) == 0x3E_FFC1);
    }

    #[test]
    fn vector_01() {
        assert!(crc24_os9(&[0x01]) == 0x80_3EA2);
    }

    #[test]
    fn vector_02() {
        assert!(crc24_os9(&[0x02]) == 0x80_3E64);
    }

    #[test]
    fn vector_7f() {
        assert!(crc24_os9(&[0x7f]) == 0x80_21DE);
    }

    #[test]
    fn vector_80() {
        assert!(crc24_os9(&[0x80]) == 0x80_1FE0);
    }

    #[test]
    fn vector_aa_55() {
        assert!(crc24_os9(&[0xaa, 0x55]) == 0x15_3E2B);
    }

    #[test]
    fn vector_55_aa() {
        assert!(crc24_os9(&[0x55, 0xaa]) == 0x2B_3E15);
    }

    #[test]
    fn vector_deadbeef() {
        assert!(crc24_os9(&[0xde, 0xad, 0xbe, 0xef]) == 0xC9_124D);
    }

    #[test]
    fn vector_hello() {
        assert!(crc24_os9(b"Hello") == 0x79_4955);
    }

    #[test]
    fn vector_12345678_bytes() {
        assert!(crc24_os9(&[0x12, 0x34, 0x56, 0x78]) == 0x0C_2C3C);
    }

    #[test]
    fn vector_four_zeros() {
        assert!(crc24_os9(&[0u8; 4]) == 0x70_3DDE);
    }

    #[test]
    fn vector_four_ff() {
        assert!(crc24_os9(&[0xffu8; 4]) == 0xFF_C1C1);
    }

    #[test]
    fn vector_0_15() {
        let mut buf = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc24_os9(&buf) == 0x7A_8AFF);
    }

    #[test]
    fn vector_a5_1000() {
        let buf = [A5_FILL; A5_LEN];
        assert!(crc24_os9(&buf) == 0xA9_00D7);
    }

    #[test]
    fn vector_all256() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc24_os9(&buf) == 0x68_AB1B);
    }

    #[test]
    fn vector_ff_ff() {
        assert!(crc24_os9(&[0xff, 0xff]) == 0x00_FFFF);
    }

    #[test]
    fn check_constant_is_0x200fa5() {
        let expected: u32 = 0x20_0FA5;
        assert!(crc24_os9(b"123456789") == expected);
    }

    #[test]
    fn empty_equals_xorout() {
        assert!(crc24_os9(b"") == ((INIT ^ XOROUT) & MASK));
    }

    #[test]
    fn result_fits_in_width() {
        assert!(crc24_os9(b"123456789") == (crc24_os9(b"123456789") & MASK));
    }

    #[test]
    fn determinism_empty() {
        assert!(crc24_os9(b"") == crc24_os9(b""));
    }

    #[test]
    fn determinism_fox() {
        assert!(crc24_os9(b"The quick brown fox") == crc24_os9(b"The quick brown fox"));
    }

    #[test]
    fn determinism_all256() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc24_os9(&buf) == crc24_os9(&buf));
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc24_os9(b"a") != crc24_os9(b"b"));
    }

    #[test]
    fn prefix_ab_differs_from_abc() {
        assert!(crc24_os9(b"ab") != crc24_os9(b"abc"));
    }

    #[test]
    fn length_sensitive_ff() {
        assert!(crc24_os9(&[0xff]) != crc24_os9(&[0xff, 0xff]));
    }

    #[test]
    fn single_byte_01_differs_from_02() {
        assert!(crc24_os9(&[0x01]) != crc24_os9(&[0x02]));
    }

    #[test]
    fn single_byte_7f_differs_from_80() {
        assert!(crc24_os9(&[0x7f]) != crc24_os9(&[0x80]));
    }

    #[test]
    fn single_byte_00_differs_from_ff() {
        assert!(crc24_os9(&[0x00]) != crc24_os9(&[0xff]));
    }

    #[test]
    fn aa55_differs_from_55aa() {
        assert!(crc24_os9(&[0xaa, 0x55]) != crc24_os9(&[0x55, 0xaa]));
    }

    #[test]
    fn single_byte_sample_pairwise_distinct() {
        let samples: [u8; 6] = [0x00, 0x01, 0x02, 0x7f, 0x80, 0xff];
        let mut i = 0usize;
        while i < samples.len() {
            let mut j = i + 1;
            while j < samples.len() {
                assert!(crc24_os9(&[samples[i]]) != crc24_os9(&[samples[j]]));
                j += 1;
            }
            i += 1;
        }
    }

    #[test]
    fn output_deterministic_over_all_bytes() {
        let mut byte = 0u32;
        while byte < 256 {
            let single = [byte as u8];
            assert!(crc24_os9(&single) == crc24_os9(&single));
            byte += 1;
        }
    }

    #[test]
    fn all_single_byte_results_fit_in_width() {
        let mut byte = 0u32;
        while byte < 256 {
            let single = [byte as u8];
            let value = crc24_os9(&single);
            assert!(value == (value & MASK));
            byte += 1;
        }
    }

    #[test]
    fn reflect_reverses_bits() {
        assert!(reflect(0b0000_0001, 8) == 0b1000_0000);
    }

    #[test]
    fn reflect_full_width_round_trip() {
        let value = 0x00_ABCDu32;
        assert!(reflect(reflect(value, WIDTH), WIDTH) == value);
    }

    #[test]
    fn topbit_is_width_minus_one() {
        assert!(TOPBIT == (1u32 << (WIDTH - 1)));
    }

    #[test]
    fn mask_selects_low_width_bits() {
        assert!(MASK == ((1u32 << WIDTH) - 1));
    }
}
