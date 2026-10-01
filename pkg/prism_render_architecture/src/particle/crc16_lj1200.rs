//! `CPU`-verifiable `CRC-16/LJ1200` contract module for the Prism particle engine.
//!
//! This module implements a bit-wise `CRC` (cyclic redundancy check) using the
//! `CRC-16/LJ1200` parameter set. The algorithm walks each input byte most
//! significant bit (`MSB`) first, shifting the register left and conditionally
//! applying the generator polynomial whenever the register top bit `XOR` the
//! incoming data bit is set. Least significant bit (`LSB`) first processing is
//! only used when input reflection is enabled, which this parameter set leaves
//! disabled.
//!
//! Parameters: width `16`, polynomial `0x6f63`, init `0x0`, xor-out `0x0`,
//! no input reflection, no output reflection. The canonical check value for the
//! `ASCII` sequence `123456789` is `0xbdf4`.
//!
//! The implementation is deterministic, allocation-free, and relies only on
//! integer arithmetic so it can be verified on the `CPU` without floating point.

/// `CRC` register width in bits.
const WIDTH: u32 = 16;
/// Generator polynomial (`MSB`-first form).
const POLY: u32 = 0x6f63;
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

/// Compute the `CRC-16/LJ1200` of `data`.
pub fn crc16_lj1200(data: &[u8]) -> u16 {
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
        assert!(crc16_lj1200(b"") == 0x0000);
    }

    #[test]
    fn vector_z00() {
        assert!(crc16_lj1200(&[0x00]) == 0x0000);
    }

    #[test]
    fn vector_ff() {
        assert!(crc16_lj1200(&[0xff]) == 0x22fc);
    }

    #[test]
    fn vector_a() {
        assert!(crc16_lj1200(b"a") == 0xadf3);
    }

    #[test]
    fn vector_b() {
        assert!(crc16_lj1200(b"b") == 0x1c56);
    }

    #[test]
    fn vector_ab() {
        assert!(crc16_lj1200(b"ab") == 0xb0b4);
    }

    #[test]
    fn vector_abc() {
        assert!(crc16_lj1200(b"abc") == 0x15ff);
    }

    #[test]
    fn vector_two_zeros() {
        assert!(crc16_lj1200(&[0, 0]) == 0x0000);
    }

    #[test]
    fn vector_01() {
        assert!(crc16_lj1200(&[0x01]) == 0x6f63);
    }

    #[test]
    fn vector_02() {
        assert!(crc16_lj1200(&[0x02]) == 0xdec6);
    }

    #[test]
    fn vector_7f() {
        assert!(crc16_lj1200(&[0x7f]) == 0x917e);
    }

    #[test]
    fn vector_80() {
        assert!(crc16_lj1200(&[0x80]) == 0xb382);
    }

    #[test]
    fn vector_aa_55() {
        assert!(crc16_lj1200(&[0xaa, 0x55]) == 0xcf4e);
    }

    #[test]
    fn vector_55_aa() {
        assert!(crc16_lj1200(&[0x55, 0xaa]) == 0x5425);
    }

    #[test]
    fn vector_deadbeef() {
        assert!(crc16_lj1200(&[0xde, 0xad, 0xbe, 0xef]) == 0x0347);
    }

    #[test]
    fn vector_hello() {
        assert!(crc16_lj1200(b"Hello") == 0x9147);
    }

    #[test]
    fn vector_fox() {
        assert!(crc16_lj1200(b"The quick brown fox") == 0x30e0);
    }

    #[test]
    fn vector_four_zeros() {
        assert!(crc16_lj1200(&[0u8; 4]) == 0x0000);
    }

    #[test]
    fn vector_four_ff() {
        assert!(crc16_lj1200(&[0xffu8; 4]) == 0x1e6d);
    }

    #[test]
    fn vector_0_15() {
        let mut buf = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc16_lj1200(&buf) == 0x2d2f);
    }

    #[test]
    fn vector_12345678_bytes() {
        assert!(crc16_lj1200(&[0x12, 0x34, 0x56, 0x78]) == 0x97af);
    }

    #[test]
    fn vector_check() {
        assert!(crc16_lj1200(b"123456789") == 0xbdf4);
    }

    #[test]
    fn vector_a5_1000() {
        let buf = [A5_FILL; A5_LEN];
        assert!(crc16_lj1200(&buf) == 0x536f);
    }

    #[test]
    fn vector_all256() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc16_lj1200(&buf) == 0x6ede);
    }

    #[test]
    fn check_constant_is_0xbdf4() {
        let expected: u16 = 0xbdf4;
        assert!(crc16_lj1200(b"123456789") == expected);
    }

    #[test]
    fn empty_equals_xorout() {
        assert!(crc16_lj1200(b"") == ((INIT ^ XOROUT) & MASK) as u16);
    }

    #[test]
    fn determinism_empty() {
        assert!(crc16_lj1200(b"") == crc16_lj1200(b""));
    }

    #[test]
    fn determinism_fox() {
        assert!(crc16_lj1200(b"The quick brown fox") == crc16_lj1200(b"The quick brown fox"));
    }

    #[test]
    fn determinism_all256() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc16_lj1200(&buf) == crc16_lj1200(&buf));
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc16_lj1200(b"a") != crc16_lj1200(b"b"));
    }

    #[test]
    fn prefix_ab_differs_from_abc() {
        assert!(crc16_lj1200(b"ab") != crc16_lj1200(b"abc"));
    }

    #[test]
    fn length_sensitive_ff() {
        assert!(crc16_lj1200(&[0xff]) != crc16_lj1200(&[0xff, 0xff]));
    }

    #[test]
    fn single_byte_01_differs_from_02() {
        assert!(crc16_lj1200(&[0x01]) != crc16_lj1200(&[0x02]));
    }

    #[test]
    fn single_byte_7f_differs_from_80() {
        assert!(crc16_lj1200(&[0x7f]) != crc16_lj1200(&[0x80]));
    }

    #[test]
    fn single_byte_00_differs_from_ff() {
        assert!(crc16_lj1200(&[0x00]) != crc16_lj1200(&[0xff]));
    }

    #[test]
    fn aa55_differs_from_55aa() {
        assert!(crc16_lj1200(&[0xaa, 0x55]) != crc16_lj1200(&[0x55, 0xaa]));
    }

    #[test]
    fn single_byte_sample_pairwise_distinct() {
        let samples: [u8; 6] = [0x00, 0x01, 0x02, 0x7f, 0x80, 0xff];
        let mut i = 0usize;
        while i < samples.len() {
            let mut j = i + 1;
            while j < samples.len() {
                assert!(crc16_lj1200(&[samples[i]]) != crc16_lj1200(&[samples[j]]));
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
            assert!(crc16_lj1200(&single) == crc16_lj1200(&single));
            byte += 1;
        }
    }

    #[test]
    fn two_zeros_matches_empty_property() {
        assert!(crc16_lj1200(&[0, 0]) == crc16_lj1200(b""));
    }

    #[test]
    fn reflect_reverses_bits() {
        assert!(reflect(0b0000_0001, 8) == 0b1000_0000);
    }
}
