//! `CPU`-verifiable `CRC-8/I-432-1` contract module for the Prism particle engine.
//!
//! This module implements a bit-wise `CRC` (cyclic redundancy check) using the
//! `CRC-8/I-432-1` parameter set, also known as `CRC-8/ITU`. The algorithm walks
//! each input byte most significant bit (`MSB`) first, shifting the register
//! left and conditionally applying the generator polynomial whenever the
//! register top bit `XOR` the incoming data bit is set. Least significant bit
//! (`LSB`) first processing is only used when input reflection is enabled, which
//! this `ITU` parameter set leaves disabled.
//!
//! Parameters: width `8`, polynomial `0x07`, init `0x0`, xor-out `0x55`,
//! no input reflection, no output reflection. The canonical check value for the
//! `ASCII` sequence `123456789` is `0xa1`.
//!
//! The implementation is deterministic, allocation-free, and relies only on
//! integer arithmetic so it can be verified on the `CPU` without floating point.
//! It is compatible with `no_std` and `alloc` environments because it never
//! allocates.

/// `CRC` register width in bits.
const WIDTH: u32 = 8;
/// Generator polynomial (`MSB`-first form).
const POLY: u32 = 0x07;
/// Initial register value.
const INIT: u32 = 0x0;
/// Final value `XOR`-ed into the register before returning.
const XOROUT: u32 = 0x55;
/// Whether each input byte is bit-reflected (`LSB`-first) before processing.
const REFLECT_IN: bool = false;
/// Whether the final register is bit-reflected before the `XOR`-out step.
const REFLECT_OUT: bool = false;
/// Mask selecting the low `WIDTH` bits.
const MASK: u32 = 0xff;
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

/// Compute the `CRC-8/I-432-1` of `data`.
pub fn crc8_i432_1(data: &[u8]) -> u8 {
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
        assert!(crc8_i432_1(b"") == 0x55);
    }

    #[test]
    fn vector_z00() {
        assert!(crc8_i432_1(&[0x00]) == 0x55);
    }

    #[test]
    fn vector_ff() {
        assert!(crc8_i432_1(&[0xff]) == 0xa6);
    }

    #[test]
    fn vector_check() {
        assert!(crc8_i432_1(b"123456789") == 0xa1);
    }

    #[test]
    fn vector_fox() {
        assert!(crc8_i432_1(b"The quick brown fox") == 0x7c);
    }

    #[test]
    fn vector_0123() {
        assert!(crc8_i432_1(&[0, 1, 2, 3]) == 0x1d);
    }

    #[test]
    fn vector_a() {
        assert!(crc8_i432_1(b"a") == 0x75);
    }

    #[test]
    fn vector_b() {
        assert!(crc8_i432_1(b"b") == 0x7c);
    }

    #[test]
    fn vector_ab() {
        assert!(crc8_i432_1(b"ab") == 0x9c);
    }

    #[test]
    fn vector_abc() {
        assert!(crc8_i432_1(b"abc") == 0x0a);
    }

    #[test]
    fn vector_hello() {
        assert!(crc8_i432_1(b"Hello") == 0xa3);
    }

    #[test]
    fn vector_01() {
        assert!(crc8_i432_1(&[0x01]) == 0x52);
    }

    #[test]
    fn vector_02() {
        assert!(crc8_i432_1(&[0x02]) == 0x5b);
    }

    #[test]
    fn vector_7f() {
        assert!(crc8_i432_1(&[0x7f]) == 0x2f);
    }

    #[test]
    fn vector_80() {
        assert!(crc8_i432_1(&[0x80]) == 0xdc);
    }

    #[test]
    fn vector_aa_55() {
        assert!(crc8_i432_1(&[0xaa, 0x55]) == 0x63);
    }

    #[test]
    fn vector_55_aa() {
        assert!(crc8_i432_1(&[0x55, 0xaa]) == 0x47);
    }

    #[test]
    fn vector_deadbeef() {
        assert!(crc8_i432_1(&[0xde, 0xad, 0xbe, 0xef]) == 0x9f);
    }

    #[test]
    fn vector_12345678_bytes() {
        assert!(crc8_i432_1(&[0x12, 0x34, 0x56, 0x78]) == 0x49);
    }

    #[test]
    fn vector_two_zeros() {
        assert!(crc8_i432_1(&[0, 0]) == 0x55);
    }

    #[test]
    fn vector_four_zeros() {
        assert!(crc8_i432_1(&[0u8; 4]) == 0x55);
    }

    #[test]
    fn vector_four_ff() {
        assert!(crc8_i432_1(&[0xffu8; 4]) == 0x8b);
    }

    #[test]
    fn vector_0_15() {
        let mut buf = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc8_i432_1(&buf) == 0x14);
    }

    #[test]
    fn vector_a5_1000() {
        let buf = [A5_FILL; A5_LEN];
        assert!(crc8_i432_1(&buf) == 0xca);
    }

    #[test]
    fn vector_all256() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc8_i432_1(&buf) == 0x41);
    }

    #[test]
    fn check_constant_is_0xa1() {
        let expected: u8 = 0xa1;
        assert!(crc8_i432_1(b"123456789") == expected);
    }

    #[test]
    fn empty_equals_xorout() {
        assert!(crc8_i432_1(b"") == ((INIT ^ XOROUT) & MASK) as u8);
    }

    #[test]
    fn determinism_empty() {
        assert!(crc8_i432_1(b"") == crc8_i432_1(b""));
    }

    #[test]
    fn determinism_fox() {
        assert!(crc8_i432_1(b"The quick brown fox") == crc8_i432_1(b"The quick brown fox"));
    }

    #[test]
    fn determinism_all256() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc8_i432_1(&buf) == crc8_i432_1(&buf));
    }

    #[test]
    fn a_differs_from_abc() {
        assert!(crc8_i432_1(b"a") != crc8_i432_1(b"abc"));
    }

    #[test]
    fn prefix_ab_differs_from_abc() {
        assert!(crc8_i432_1(b"ab") != crc8_i432_1(b"abc"));
    }

    #[test]
    fn length_sensitive_ff() {
        assert!(crc8_i432_1(&[0xff]) != crc8_i432_1(&[0xff, 0xff]));
    }

    #[test]
    fn single_byte_01_differs_from_02() {
        assert!(crc8_i432_1(&[0x01]) != crc8_i432_1(&[0x02]));
    }

    #[test]
    fn single_byte_7f_differs_from_80() {
        assert!(crc8_i432_1(&[0x7f]) != crc8_i432_1(&[0x80]));
    }

    #[test]
    fn single_byte_00_differs_from_ff() {
        assert!(crc8_i432_1(&[0x00]) != crc8_i432_1(&[0xff]));
    }

    #[test]
    fn aa55_differs_from_55aa() {
        assert!(crc8_i432_1(&[0xaa, 0x55]) != crc8_i432_1(&[0x55, 0xaa]));
    }

    #[test]
    fn single_byte_sample_pairwise_distinct() {
        let samples: [u8; 6] = [0x00, 0x01, 0x02, 0x7f, 0x80, 0xff];
        let mut i = 0usize;
        while i < samples.len() {
            let mut j = i + 1;
            while j < samples.len() {
                assert!(crc8_i432_1(&[samples[i]]) != crc8_i432_1(&[samples[j]]));
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
            assert!(crc8_i432_1(&single) == crc8_i432_1(&single));
            byte += 1;
        }
    }

    #[test]
    fn two_zeros_matches_empty_property() {
        assert!(crc8_i432_1(&[0, 0]) == crc8_i432_1(b""));
    }

    #[test]
    fn zero_runs_match_empty() {
        assert!(crc8_i432_1(&[0u8; 4]) == crc8_i432_1(b""));
        assert!(crc8_i432_1(&[0u8; 7]) == crc8_i432_1(b""));
    }

    #[test]
    fn reflect_reverses_bits() {
        assert!(reflect(0b0000_0001, 8) == 0b1000_0000);
    }

    #[test]
    fn reflect_is_involutive_over_byte() {
        let mut v = 0u32;
        while v < 256 {
            assert!(reflect(reflect(v, 8), 8) == v);
            v += 1;
        }
    }

    #[test]
    fn result_fits_in_width() {
        let mut byte = 0u16;
        while byte < 256 {
            let single = [byte as u8];
            let got = u32::from(crc8_i432_1(&single));
            assert!((got & !MASK) == 0);
            byte += 1;
        }
    }

    #[test]
    fn topbit_is_msb() {
        assert!(TOPBIT == 0x80);
    }
}
