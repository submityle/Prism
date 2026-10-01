//! `CPU`-verifiable `CRC-10/GSM` contract module for the Prism particle engine.
//!
//! This module implements a bit-wise `CRC` (cyclic redundancy check) using the
//! `CRC-10/GSM` parameter set. The algorithm walks each input byte most
//! significant bit (`MSB`) first, shifting the register left and conditionally
//! applying the generator polynomial whenever the register top bit `XOR` the
//! incoming data bit is set.
//!
//! Parameters: width `10`, polynomial `0x175`, init `0x0`, xor-out `0x3ff`,
//! no input reflection, no output reflection. The canonical check value for the
//! `ASCII` sequence `123456789` is `0x12a`.
//!
//! The implementation is deterministic, allocation-free, and relies only on
//! integer arithmetic so it can be verified on the `CPU` without floating point.

/// `CRC` register width in bits.
const WIDTH: u32 = 10;
/// Generator polynomial (`MSB`-first form).
const POLY: u32 = 0x175;
/// Initial register value.
const INIT: u32 = 0x0;
/// Final value `XOR`-ed into the register before returning.
const XOROUT: u32 = 0x3ff;
/// Whether each input byte is bit-reflected (`LSB`-first) before processing.
const REFLECT_IN: bool = false;
/// Whether the final register is bit-reflected before the `XOR`-out step.
const REFLECT_OUT: bool = false;
/// Mask selecting the low `WIDTH` bits.
const MASK: u32 = 0x3ff;
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

/// Compute the `CRC-10/GSM` of `data`.
pub fn crc10_gsm(data: &[u8]) -> u16 {
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
    ((reg ^ XOROUT) & MASK) as u16
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
        assert!(crc10_gsm(b"") == 0x3ff);
    }

    #[test]
    fn vector_z00() {
        assert!(crc10_gsm(&[0x00]) == 0x3ff);
    }

    #[test]
    fn vector_ff() {
        assert!(crc10_gsm(&[0xff]) == 0x274);
    }

    #[test]
    fn vector_a() {
        assert!(crc10_gsm(b"a") == 0x20d);
    }

    #[test]
    fn vector_b() {
        assert!(crc10_gsm(b"b") == 0x192);
    }

    #[test]
    fn vector_ab() {
        assert!(crc10_gsm(b"ab") == 0x072);
    }

    #[test]
    fn vector_abc() {
        assert!(crc10_gsm(b"abc") == 0x30b);
    }

    #[test]
    fn vector_two_zeros() {
        assert!(crc10_gsm(&[0, 0]) == 0x3ff);
    }

    #[test]
    fn vector_01() {
        assert!(crc10_gsm(&[0x01]) == 0x28a);
    }

    #[test]
    fn vector_02() {
        assert!(crc10_gsm(&[0x02]) == 0x115);
    }

    #[test]
    fn vector_7f() {
        assert!(crc10_gsm(&[0x7f]) == 0x380);
    }

    #[test]
    fn vector_80() {
        assert!(crc10_gsm(&[0x80]) == 0x20b);
    }

    #[test]
    fn vector_aa_55() {
        assert!(crc10_gsm(&[0xaa, 0x55]) == 0x1e3);
    }

    #[test]
    fn vector_55_aa() {
        assert!(crc10_gsm(&[0x55, 0xaa]) == 0x105);
    }

    #[test]
    fn vector_deadbeef() {
        assert!(crc10_gsm(&[0xde, 0xad, 0xbe, 0xef]) == 0x266);
    }

    #[test]
    fn vector_hello() {
        assert!(crc10_gsm(b"Hello") == 0x168);
    }

    #[test]
    fn vector_fox() {
        assert!(crc10_gsm(b"The quick brown fox") == 0x039);
    }

    #[test]
    fn vector_four_zeros() {
        assert!(crc10_gsm(&[0u8; 4]) == 0x3ff);
    }

    #[test]
    fn vector_four_ff() {
        assert!(crc10_gsm(&[0xffu8; 4]) == 0x07f);
    }

    #[test]
    fn vector_0_15() {
        let mut buf = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc10_gsm(&buf) == 0x337);
    }

    #[test]
    fn vector_12345678_bytes() {
        assert!(crc10_gsm(&[0x12, 0x34, 0x56, 0x78]) == 0x1d2);
    }

    #[test]
    fn vector_check() {
        assert!(crc10_gsm(b"123456789") == 0x12a);
    }

    #[test]
    fn vector_a5_1000() {
        let buf = [A5_FILL; A5_LEN];
        assert!(crc10_gsm(&buf) == 0x16b);
    }

    #[test]
    fn vector_all256() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc10_gsm(&buf) == 0x3b4);
    }

    #[test]
    fn check_constant_is_0x12a() {
        let expected: u16 = 0x12a;
        assert!(crc10_gsm(b"123456789") == expected);
    }

    #[test]
    fn empty_equals_xorout() {
        assert!(crc10_gsm(b"") == (XOROUT & MASK) as u16);
    }

    #[test]
    fn determinism_empty() {
        assert!(crc10_gsm(b"") == crc10_gsm(b""));
    }

    #[test]
    fn determinism_fox() {
        assert!(crc10_gsm(b"The quick brown fox") == crc10_gsm(b"The quick brown fox"));
    }

    #[test]
    fn determinism_all256() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc10_gsm(&buf) == crc10_gsm(&buf));
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc10_gsm(b"a") != crc10_gsm(b"b"));
    }

    #[test]
    fn prefix_ab_differs_from_abc() {
        assert!(crc10_gsm(b"ab") != crc10_gsm(b"abc"));
    }

    #[test]
    fn length_sensitive_ff() {
        assert!(crc10_gsm(&[0xff]) != crc10_gsm(&[0xff, 0xff]));
    }

    #[test]
    fn single_byte_01_differs_from_02() {
        assert!(crc10_gsm(&[0x01]) != crc10_gsm(&[0x02]));
    }

    #[test]
    fn single_byte_7f_differs_from_80() {
        assert!(crc10_gsm(&[0x7f]) != crc10_gsm(&[0x80]));
    }

    #[test]
    fn single_byte_00_differs_from_ff() {
        assert!(crc10_gsm(&[0x00]) != crc10_gsm(&[0xff]));
    }

    #[test]
    fn aa55_differs_from_55aa() {
        assert!(crc10_gsm(&[0xaa, 0x55]) != crc10_gsm(&[0x55, 0xaa]));
    }

    #[test]
    fn single_byte_sample_pairwise_distinct() {
        let samples: [u8; 5] = [0x00, 0x01, 0x02, 0x7f, 0x80];
        let mut i = 0usize;
        while i < samples.len() {
            let mut j = i + 1;
            while j < samples.len() {
                assert!(crc10_gsm(&[samples[i]]) != crc10_gsm(&[samples[j]]));
                j += 1;
            }
            i += 1;
        }
    }

    #[test]
    fn output_within_mask_range() {
        let mut byte = 0u16;
        while byte < 256 {
            let value = crc10_gsm(&[byte as u8]);
            assert!(value <= 0x3ff);
            byte += 1;
        }
    }

    #[test]
    fn two_zeros_matches_empty_property() {
        assert!(crc10_gsm(&[0, 0]) == crc10_gsm(b""));
    }

    #[test]
    fn reflect_reverses_bits() {
        assert!(reflect(0b0000_0001, 8) == 0b1000_0000);
    }
}
