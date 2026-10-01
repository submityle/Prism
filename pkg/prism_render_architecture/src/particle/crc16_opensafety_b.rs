//! `CPU`-verifiable `CRC-16/OPENSAFETY-B` contract module for the Prism particle engine.
//!
//! This module implements a bit-wise `CRC` (cyclic redundancy check) using the
//! `CRC-16/OPENSAFETY-B` parameter set. The algorithm walks each input byte most
//! significant bit (`MSB`) first, shifting the register left and conditionally
//! applying the generator polynomial whenever the register top bit `XOR` the
//! incoming data bit is set. Least significant bit (`LSB`) first processing is
//! only used when input reflection is enabled, which this parameter set leaves
//! disabled.
//!
//! Parameters: width `16`, polynomial `0x755B`, init `0x0`, xor-out `0x0`,
//! no input reflection, no output reflection. The canonical check value for the
//! `ASCII` sequence `123456789` is `0x20FE`.
//!
//! The implementation is deterministic, allocation-free, and relies only on
//! integer arithmetic so it can be verified on the `CPU` without floating point.
//! It is compatible with `no_std` and `alloc`-free environments.

/// `CRC` register width in bits.
const WIDTH: u32 = 16;
/// Generator polynomial (`MSB`-first form).
const POLY: u32 = 0x755B;
/// Initial register value.
const INIT: u32 = 0x0;
/// Final value `XOR`-ed into the register before returning.
const XOROUT: u32 = 0x0;
/// Whether each input byte is bit-reflected (`LSB`-first) before processing.
const REFLECT_IN: bool = false;
/// Whether the final register is bit-reflected before the `XOR`-out step.
const REFLECT_OUT: bool = false;
/// Mask selecting the low `WIDTH` bits.
const MASK: u32 = 0xffff;
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

/// Compute the `CRC-16/OPENSAFETY-B` of `data`.
pub fn crc16_opensafety_b(data: &[u8]) -> u16 {
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
        assert!(crc16_opensafety_b(b"") == 0x0000);
    }

    #[test]
    fn vector_z00() {
        assert!(crc16_opensafety_b(&[0x00]) == 0x0000);
    }

    #[test]
    fn vector_ff() {
        assert!(crc16_opensafety_b(&[0xff]) == 0xa41f);
    }

    #[test]
    fn vector_a() {
        assert!(crc16_opensafety_b(b"a") == 0x7d7c);
    }

    #[test]
    fn vector_b() {
        assert!(crc16_opensafety_b(b"b") == 0xe291);
    }

    #[test]
    fn vector_ab() {
        assert!(crc16_opensafety_b(b"ab") == 0x1c85);
    }

    #[test]
    fn vector_abc() {
        assert!(crc16_opensafety_b(b"abc") == 0xeda2);
    }

    #[test]
    fn vector_two_zeros() {
        assert!(crc16_opensafety_b(&[0, 0]) == 0x0000);
    }

    #[test]
    fn vector_01() {
        assert!(crc16_opensafety_b(&[0x01]) == 0x755b);
    }

    #[test]
    fn vector_02() {
        assert!(crc16_opensafety_b(&[0x02]) == 0xeab6);
    }

    #[test]
    fn vector_7f() {
        assert!(crc16_opensafety_b(&[0x7f]) == 0x68a2);
    }

    #[test]
    fn vector_80() {
        assert!(crc16_opensafety_b(&[0x80]) == 0xccbd);
    }

    #[test]
    fn vector_aa_55() {
        assert!(crc16_opensafety_b(&[0xaa, 0x55]) == 0xa661);
    }

    #[test]
    fn vector_55_aa() {
        assert!(crc16_opensafety_b(&[0x55, 0xaa]) == 0xa520);
    }

    #[test]
    fn vector_deadbeef() {
        assert!(crc16_opensafety_b(&[0xde, 0xad, 0xbe, 0xef]) == 0xacd6);
    }

    #[test]
    fn vector_hello() {
        assert!(crc16_opensafety_b(b"Hello") == 0x0f9b);
    }

    #[test]
    fn vector_fox() {
        assert!(crc16_opensafety_b(b"The quick brown fox") == 0x9027);
    }

    #[test]
    fn vector_four_zeros() {
        assert!(crc16_opensafety_b(&[0u8; 4]) == 0x0000);
    }

    #[test]
    fn vector_four_ff() {
        assert!(crc16_opensafety_b(&[0xffu8; 4]) == 0xebd1);
    }

    #[test]
    fn vector_0_15() {
        let mut buf = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc16_opensafety_b(&buf) == 0xf66e);
    }

    #[test]
    fn vector_12345678_bytes() {
        assert!(crc16_opensafety_b(&[0x12, 0x34, 0x56, 0x78]) == 0xb6a1);
    }

    #[test]
    fn vector_check() {
        assert!(crc16_opensafety_b(b"123456789") == 0x20fe);
    }

    #[test]
    fn vector_0_1_2_3() {
        assert!(crc16_opensafety_b(&[0, 1, 2, 3]) == 0x426c);
    }

    #[test]
    fn vector_ff_ff() {
        assert!(crc16_opensafety_b(&[0xff, 0xff]) == 0x0341);
    }

    #[test]
    fn vector_a5_1000() {
        let buf = [A5_FILL; A5_LEN];
        assert!(crc16_opensafety_b(&buf) == 0xafac);
    }

    #[test]
    fn vector_all256() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc16_opensafety_b(&buf) == 0x405d);
    }

    #[test]
    fn check_constant_is_0x20fe() {
        let expected: u16 = 0x20fe;
        assert!(crc16_opensafety_b(b"123456789") == expected);
    }

    #[test]
    fn empty_equals_xorout() {
        assert!(crc16_opensafety_b(b"") == ((INIT ^ XOROUT) & MASK) as u16);
    }

    #[test]
    fn determinism_empty() {
        assert!(crc16_opensafety_b(b"") == crc16_opensafety_b(b""));
    }

    #[test]
    fn determinism_fox() {
        assert!(crc16_opensafety_b(b"The quick brown fox") == crc16_opensafety_b(b"The quick brown fox"));
    }

    #[test]
    fn determinism_all256() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc16_opensafety_b(&buf) == crc16_opensafety_b(&buf));
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc16_opensafety_b(b"a") != crc16_opensafety_b(b"b"));
    }

    #[test]
    fn prefix_ab_differs_from_abc() {
        assert!(crc16_opensafety_b(b"ab") != crc16_opensafety_b(b"abc"));
    }

    #[test]
    fn length_sensitive_ff() {
        assert!(crc16_opensafety_b(&[0xff]) != crc16_opensafety_b(&[0xff, 0xff]));
    }

    #[test]
    fn single_byte_01_differs_from_02() {
        assert!(crc16_opensafety_b(&[0x01]) != crc16_opensafety_b(&[0x02]));
    }

    #[test]
    fn single_byte_7f_differs_from_80() {
        assert!(crc16_opensafety_b(&[0x7f]) != crc16_opensafety_b(&[0x80]));
    }

    #[test]
    fn single_byte_00_differs_from_ff() {
        assert!(crc16_opensafety_b(&[0x00]) != crc16_opensafety_b(&[0xff]));
    }

    #[test]
    fn aa55_differs_from_55aa() {
        assert!(crc16_opensafety_b(&[0xaa, 0x55]) != crc16_opensafety_b(&[0x55, 0xaa]));
    }

    #[test]
    fn single_byte_sample_pairwise_distinct() {
        let samples: [u8; 6] = [0x00, 0x01, 0x02, 0x7f, 0x80, 0xff];
        let mut i = 0usize;
        while i < samples.len() {
            let mut j = i + 1;
            while j < samples.len() {
                assert!(crc16_opensafety_b(&[samples[i]]) != crc16_opensafety_b(&[samples[j]]));
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
            assert!(crc16_opensafety_b(&single) == crc16_opensafety_b(&single));
            byte += 1;
        }
    }

    #[test]
    fn two_zeros_matches_empty_property() {
        assert!(crc16_opensafety_b(&[0, 0]) == crc16_opensafety_b(b""));
    }

    #[test]
    fn four_zeros_matches_empty_property() {
        assert!(crc16_opensafety_b(&[0u8; 4]) == crc16_opensafety_b(b""));
    }

    #[test]
    fn prefix_fox_shorter_differs() {
        assert!(crc16_opensafety_b(b"The quick brown") != crc16_opensafety_b(b"The quick brown fox"));
    }

    #[test]
    fn reflect_reverses_bits() {
        assert!(reflect(0b0000_0001, 8) == 0b1000_0000);
    }

    #[test]
    fn reflect_full_width_roundtrip() {
        assert!(reflect(reflect(0x1234, WIDTH), WIDTH) == 0x1234);
    }
}
