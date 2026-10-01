//! `CPU`-verifiable `CRC-6/CDMA2000-A` contract module for the Prism particle engine.
//!
//! This module implements a bit-wise `CRC` (cyclic redundancy check) using the
//! `CRC-6/CDMA2000-A` parameter set. The algorithm walks each input byte most
//! significant bit (`MSB`) first, shifting the register left and conditionally
//! applying the generator polynomial whenever the register top bit `XOR` the
//! incoming data bit is set. Least significant bit (`LSB`) first processing is
//! only used when input reflection is enabled, which this parameter set leaves
//! disabled.
//!
//! Parameters: width `6`, polynomial `0x27`, init `0x3f`, xor-out `0x0`,
//! no input reflection, no output reflection. The register is kept within six
//! bits via a mask, held in a `u32`, and returned as a `u8` whose low six bits
//! carry the result. The canonical check value for the `ASCII` sequence
//! `123456789` is `0x0d`.
//!
//! The implementation is deterministic, allocation-free, and relies only on
//! integer arithmetic so it can be verified on the `CPU` without floating
//! point. It is compatible with a `no_std` plus `alloc` environment because it
//! performs no heap work at all.

/// `CRC` register width in bits.
const WIDTH: u32 = 6;
/// Generator polynomial (`MSB`-first form).
const POLY: u32 = 0x27;
/// Initial register value.
const INIT: u32 = 0x3f;
/// Final value `XOR`-ed into the register before returning.
const XOROUT: u32 = 0x0;
/// Whether each input byte is bit-reflected (`LSB`-first) before processing.
const REFLECT_IN: bool = false;
/// Whether the final register is bit-reflected before the `XOR`-out step.
const REFLECT_OUT: bool = false;
/// Mask selecting the low `WIDTH` bits.
const MASK: u32 = 0x3f;
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

/// Compute the `CRC-6/CDMA2000-A` of `data`.
pub fn crc6_cdma2000_a(data: &[u8]) -> u8 {
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
    ((reg ^ XOROUT) & MASK) as u8
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
        assert!(crc6_cdma2000_a(b"") == 0x3f);
    }

    #[test]
    fn vector_check() {
        assert!(crc6_cdma2000_a(b"123456789") == 0x0d);
    }

    #[test]
    fn vector_z00() {
        assert!(crc6_cdma2000_a(&[0x00]) == 0x07);
    }

    #[test]
    fn vector_ff() {
        assert!(crc6_cdma2000_a(&[0xff]) == 0x0e);
    }

    #[test]
    fn vector_fox() {
        assert!(crc6_cdma2000_a(b"The quick brown fox") == 0x36);
    }

    #[test]
    fn vector_0123() {
        assert!(crc6_cdma2000_a(&[0, 1, 2, 3]) == 0x19);
    }

    #[test]
    fn vector_a() {
        assert!(crc6_cdma2000_a(b"a") == 0x1b);
    }

    #[test]
    fn vector_b() {
        assert!(crc6_cdma2000_a(b"b") == 0x15);
    }

    #[test]
    fn vector_ab() {
        assert!(crc6_cdma2000_a(b"ab") == 0x11);
    }

    #[test]
    fn vector_abc() {
        assert!(crc6_cdma2000_a(b"abc") == 0x0f);
    }

    #[test]
    fn vector_hello() {
        assert!(crc6_cdma2000_a(b"Hello") == 0x15);
    }

    #[test]
    fn vector_01() {
        assert!(crc6_cdma2000_a(&[0x01]) == 0x20);
    }

    #[test]
    fn vector_02() {
        assert!(crc6_cdma2000_a(&[0x02]) == 0x2e);
    }

    #[test]
    fn vector_7f() {
        assert!(crc6_cdma2000_a(&[0x7f]) == 0x10);
    }

    #[test]
    fn vector_80() {
        assert!(crc6_cdma2000_a(&[0x80]) == 0x19);
    }

    #[test]
    fn vector_two_zeros() {
        assert!(crc6_cdma2000_a(&[0, 0]) == 0x22);
    }

    #[test]
    fn vector_aa55() {
        assert!(crc6_cdma2000_a(&[0xaa, 0x55]) == 0x06);
    }

    #[test]
    fn vector_55aa() {
        assert!(crc6_cdma2000_a(&[0x55, 0xaa]) == 0x0e);
    }

    #[test]
    fn vector_deadbeef() {
        assert!(crc6_cdma2000_a(&[0xde, 0xad, 0xbe, 0xef]) == 0x2a);
    }

    #[test]
    fn vector_four_zeros() {
        assert!(crc6_cdma2000_a(&[0u8; 4]) == 0x37);
    }

    #[test]
    fn vector_four_ff() {
        assert!(crc6_cdma2000_a(&[0xffu8; 4]) == 0x34);
    }

    #[test]
    fn vector_12345678_bytes() {
        assert!(crc6_cdma2000_a(&[0x12, 0x34, 0x56, 0x78]) == 0x3b);
    }

    #[test]
    fn check_constant_is_0x0d() {
        let expected: u8 = 0x0d;
        assert!(crc6_cdma2000_a(b"123456789") == expected);
    }

    #[test]
    fn empty_equals_init_xorout() {
        assert!(crc6_cdma2000_a(b"") == ((INIT ^ XOROUT) & MASK) as u8);
    }

    #[test]
    fn determinism_empty() {
        assert!(crc6_cdma2000_a(b"") == crc6_cdma2000_a(b""));
    }

    #[test]
    fn determinism_fox() {
        assert!(crc6_cdma2000_a(b"The quick brown fox") == crc6_cdma2000_a(b"The quick brown fox"));
    }

    #[test]
    fn determinism_long_pattern() {
        let buf = [A5_FILL; A5_LEN];
        assert!(crc6_cdma2000_a(&buf) == crc6_cdma2000_a(&buf));
    }

    #[test]
    fn output_within_6_bits_empty() {
        assert!(crc6_cdma2000_a(b"") <= 0x3f);
    }

    #[test]
    fn output_within_6_bits_all_single_bytes() {
        let mut byte = 0u16;
        while byte < 256 {
            let single = [byte as u8];
            assert!(crc6_cdma2000_a(&single) <= 0x3f);
            byte += 1;
        }
    }

    #[test]
    fn output_within_6_bits_long_pattern() {
        let buf = [A5_FILL; A5_LEN];
        assert!(crc6_cdma2000_a(&buf) <= 0x3f);
    }

    #[test]
    fn output_deterministic_over_all_bytes() {
        let mut byte = 0u16;
        while byte < 256 {
            let single = [byte as u8];
            assert!(crc6_cdma2000_a(&single) == crc6_cdma2000_a(&single));
            byte += 1;
        }
    }

    #[test]
    fn a_differs_from_abc() {
        assert!(crc6_cdma2000_a(b"a") != crc6_cdma2000_a(b"abc"));
    }

    #[test]
    fn prefix_ab_differs_from_abc() {
        assert!(crc6_cdma2000_a(b"ab") != crc6_cdma2000_a(b"abc"));
    }

    #[test]
    fn length_sensitive_ff() {
        assert!(crc6_cdma2000_a(&[0xff]) != crc6_cdma2000_a(&[0xff, 0xff]));
    }

    #[test]
    fn single_byte_01_differs_from_02() {
        assert!(crc6_cdma2000_a(&[0x01]) != crc6_cdma2000_a(&[0x02]));
    }

    #[test]
    fn single_byte_7f_differs_from_80() {
        assert!(crc6_cdma2000_a(&[0x7f]) != crc6_cdma2000_a(&[0x80]));
    }

    #[test]
    fn single_byte_00_differs_from_ff() {
        assert!(crc6_cdma2000_a(&[0x00]) != crc6_cdma2000_a(&[0xff]));
    }

    #[test]
    fn aa55_differs_from_55aa() {
        assert!(crc6_cdma2000_a(&[0xaa, 0x55]) != crc6_cdma2000_a(&[0x55, 0xaa]));
    }

    #[test]
    fn single_byte_sample_pairwise_distinct() {
        let samples: [u8; 6] = [0x00, 0x01, 0x02, 0x7f, 0x80, 0xff];
        let mut i = 0usize;
        while i < samples.len() {
            let mut j = i + 1;
            while j < samples.len() {
                assert!(crc6_cdma2000_a(&[samples[i]]) != crc6_cdma2000_a(&[samples[j]]));
                j += 1;
            }
            i += 1;
        }
    }

    #[test]
    fn empty_differs_from_single_zero() {
        assert!(crc6_cdma2000_a(b"") != crc6_cdma2000_a(&[0x00]));
    }

    #[test]
    fn two_zeros_differs_from_single_zero() {
        assert!(crc6_cdma2000_a(&[0, 0]) != crc6_cdma2000_a(&[0x00]));
    }

    #[test]
    fn reflect_reverses_bits() {
        assert!(reflect(0b0000_0001, 8) == 0b1000_0000);
    }

    #[test]
    fn reflect_six_bits_round_trip() {
        assert!(reflect(reflect(0x25, 6), 6) == 0x25);
    }

    #[test]
    fn topbit_matches_width() {
        assert!(TOPBIT == (1u32 << (WIDTH - 1)));
    }

    #[test]
    fn mask_is_six_bits() {
        assert!(MASK == ((1u32 << WIDTH) - 1));
    }
}
