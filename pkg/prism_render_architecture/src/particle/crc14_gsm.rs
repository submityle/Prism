//! `CPU`-verifiable `CRC-14/GSM` contract module for the Prism particle engine.
//!
//! This module implements a bit-wise `CRC` (cyclic redundancy check) using the
//! `CRC-14/GSM` parameter set. The algorithm walks each input byte most
//! significant bit (`MSB`) first, shifting the register left and conditionally
//! applying the generator polynomial whenever the register top bit `XOR` the
//! incoming data bit is set. Least significant bit (`LSB`) first processing is
//! only used when input reflection is enabled, which this parameter set leaves
//! disabled.
//!
//! Parameters: width `14`, polynomial `0x202D`, init `0x0`, xor-out `0x3FFF`,
//! no input reflection, no output reflection. The canonical check value for the
//! `ASCII` sequence `123456789` is `0x30AE`.
//!
//! The implementation is deterministic, allocation-free, and relies only on
//! integer arithmetic so it can be verified on the `CPU` without floating point.
//! It is compatible with `no_std` and does not require `alloc`.

/// `CRC` register width in bits.
const WIDTH: u32 = 14;
/// Generator polynomial (`MSB`-first form).
const POLY: u32 = 0x202D;
/// Initial register value.
const INIT: u32 = 0x0;
/// Final value `XOR`-ed into the register before returning.
const XOROUT: u32 = 0x3FFF;
/// Whether each input byte is bit-reflected (`LSB`-first) before processing.
const REFLECT_IN: bool = false;
/// Whether the final register is bit-reflected before the `XOR`-out step.
const REFLECT_OUT: bool = false;
/// Mask selecting the low `WIDTH` bits.
const MASK: u32 = 0x3FFF;
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

/// Compute the `CRC-14/GSM` of `data`.
pub fn crc14_gsm(data: &[u8]) -> u16 {
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
        assert!(crc14_gsm(b"") == 0x3FFF);
    }

    #[test]
    fn vector_z00() {
        assert!(crc14_gsm(&[0x00]) == 0x3FFF);
    }

    #[test]
    fn vector_ff() {
        assert!(crc14_gsm(&[0xff]) == 0x2DED);
    }

    #[test]
    fn vector_a() {
        assert!(crc14_gsm(b"a") == 0x1492);
    }

    #[test]
    fn vector_b() {
        assert!(crc14_gsm(b"b") == 0x14C8);
    }

    #[test]
    fn vector_ab() {
        assert!(crc14_gsm(b"ab") == 0x054D);
    }

    #[test]
    fn vector_abc() {
        assert!(crc14_gsm(b"abc") == 0x3762);
    }

    #[test]
    fn vector_two_zeros() {
        assert!(crc14_gsm(&[0, 0]) == 0x3FFF);
    }

    #[test]
    fn vector_01() {
        assert!(crc14_gsm(&[0x01]) == 0x1FD2);
    }

    #[test]
    fn vector_02() {
        assert!(crc14_gsm(&[0x02]) == 0x1F88);
    }

    #[test]
    fn vector_7f() {
        assert!(crc14_gsm(&[0x7f]) == 0x16F6);
    }

    #[test]
    fn vector_80() {
        assert!(crc14_gsm(&[0x80]) == 0x04E4);
    }

    #[test]
    fn vector_aa_55() {
        assert!(crc14_gsm(&[0xaa, 0x55]) == 0x05CA);
    }

    #[test]
    fn vector_55_aa() {
        assert!(crc14_gsm(&[0x55, 0xaa]) == 0x09E8);
    }

    #[test]
    fn vector_deadbeef() {
        assert!(crc14_gsm(&[0xde, 0xad, 0xbe, 0xef]) == 0x2504);
    }

    #[test]
    fn vector_hello() {
        assert!(crc14_gsm(b"Hello") == 0x3275);
    }

    #[test]
    fn vector_fox() {
        assert!(crc14_gsm(b"The quick brown fox") == 0x1210);
    }

    #[test]
    fn vector_four_zeros() {
        assert!(crc14_gsm(&[0u8; 4]) == 0x3FFF);
    }

    #[test]
    fn vector_four_ff() {
        assert!(crc14_gsm(&[0xffu8; 4]) == 0x2AA2);
    }

    #[test]
    fn vector_0_15() {
        let mut buf = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc14_gsm(&buf) == 0x1943);
    }

    #[test]
    fn vector_12345678_bytes() {
        assert!(crc14_gsm(&[0x12, 0x34, 0x56, 0x78]) == 0x0958);
    }

    #[test]
    fn vector_check() {
        assert!(crc14_gsm(b"123456789") == 0x30AE);
    }

    #[test]
    fn vector_0_1_2_3() {
        assert!(crc14_gsm(&[0, 1, 2, 3]) == 0x27D8);
    }

    #[test]
    fn vector_a5_1000() {
        let buf = [A5_FILL; A5_LEN];
        assert!(crc14_gsm(&buf) == 0x2147);
    }

    #[test]
    fn vector_all256() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc14_gsm(&buf) == 0x1549);
    }

    #[test]
    fn check_constant_is_0x30ae() {
        let expected: u16 = 0x30AE;
        assert!(crc14_gsm(b"123456789") == expected);
    }

    #[test]
    fn empty_equals_xorout() {
        assert!(crc14_gsm(b"") == ((INIT ^ XOROUT) & MASK) as u16);
    }

    #[test]
    fn determinism_empty() {
        assert!(crc14_gsm(b"") == crc14_gsm(b""));
    }

    #[test]
    fn determinism_fox() {
        assert!(crc14_gsm(b"The quick brown fox") == crc14_gsm(b"The quick brown fox"));
    }

    #[test]
    fn determinism_all256() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc14_gsm(&buf) == crc14_gsm(&buf));
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc14_gsm(b"a") != crc14_gsm(b"b"));
    }

    #[test]
    fn prefix_ab_differs_from_abc() {
        assert!(crc14_gsm(b"ab") != crc14_gsm(b"abc"));
    }

    #[test]
    fn length_sensitive_ff() {
        assert!(crc14_gsm(&[0xff]) != crc14_gsm(&[0xff, 0xff]));
    }

    #[test]
    fn single_byte_01_differs_from_02() {
        assert!(crc14_gsm(&[0x01]) != crc14_gsm(&[0x02]));
    }

    #[test]
    fn single_byte_7f_differs_from_80() {
        assert!(crc14_gsm(&[0x7f]) != crc14_gsm(&[0x80]));
    }

    #[test]
    fn single_byte_00_differs_from_ff() {
        assert!(crc14_gsm(&[0x00]) != crc14_gsm(&[0xff]));
    }

    #[test]
    fn aa55_differs_from_55aa() {
        assert!(crc14_gsm(&[0xaa, 0x55]) != crc14_gsm(&[0x55, 0xaa]));
    }

    #[test]
    fn single_byte_sample_pairwise_distinct() {
        let samples: [u8; 6] = [0x00, 0x01, 0x02, 0x7f, 0x80, 0xff];
        let mut i = 0usize;
        while i < samples.len() {
            let mut j = i + 1;
            while j < samples.len() {
                assert!(crc14_gsm(&[samples[i]]) != crc14_gsm(&[samples[j]]));
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
            assert!(crc14_gsm(&single) == crc14_gsm(&single));
            byte += 1;
        }
    }

    #[test]
    fn two_zeros_matches_empty_property() {
        assert!(crc14_gsm(&[0, 0]) == crc14_gsm(b""));
    }

    #[test]
    fn four_zeros_matches_empty_property() {
        assert!(crc14_gsm(&[0u8; 4]) == crc14_gsm(b""));
    }

    #[test]
    fn prefix_fox_differs_by_length() {
        assert!(crc14_gsm(b"The quick brown") != crc14_gsm(b"The quick brown fox"));
    }

    #[test]
    fn reflect_reverses_bits() {
        assert!(reflect(0b0000_0001, 8) == 0b1000_0000);
    }
}
