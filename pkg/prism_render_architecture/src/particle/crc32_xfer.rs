//! `CPU`-verifiable `CRC-32/XFER` contract module for the Prism particle engine.
//!
//! This module implements a bit-wise `CRC` (cyclic redundancy check) using the
//! `CRC-32/XFER` parameter set. The algorithm walks each input byte most
//! significant bit (`MSB`) first, shifting the register left and conditionally
//! applying the generator polynomial whenever the register top bit `XOR` the
//! incoming data bit is set. Least significant bit (`LSB`) first processing is
//! only used when input reflection is enabled, which this parameter set leaves
//! disabled.
//!
//! Parameters: width `32`, polynomial `0x0000_00AF`, init `0x0`, xor-out `0x0`,
//! no input reflection, no output reflection. The canonical check value for the
//! `ASCII` sequence `123456789` is `0xBD0B_E338`.
//!
//! The implementation is deterministic, allocation-free, and relies only on
//! integer arithmetic so it can be verified on the `CPU` without floating
//! point. It is compatible with `no_std` plus `alloc` environments because it
//! performs no heap work and uses only `u8` and `u32` scalar math.

/// `CRC` register width in bits.
const WIDTH: u32 = 32;
/// Generator polynomial (`MSB`-first form).
const POLY: u32 = 0x0000_00AF;
/// Initial register value.
const INIT: u32 = 0x0;
/// Final value `XOR`-ed into the register before returning.
const XOROUT: u32 = 0x0;
/// Whether each input byte is bit-reflected (`LSB`-first) before processing.
const REFLECT_IN: bool = false;
/// Whether the final register is bit-reflected before the `XOR`-out step.
const REFLECT_OUT: bool = false;
/// Mask selecting the low `WIDTH` bits.
const MASK: u32 = 0xFFFF_FFFF;
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

/// Compute the `CRC-32/XFER` of `data`.
pub fn crc32_xfer(data: &[u8]) -> u32 {
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
    const A5_FILL: u8 = 0xA5;
    /// Length of the long repeated-pattern vector.
    const A5_LEN: usize = 1000;

    #[test]
    fn vector_empty() {
        assert!(crc32_xfer(b"") == 0x0000_0000);
    }

    #[test]
    fn vector_z00() {
        assert!(crc32_xfer(&[0x00]) == 0x0000_0000);
    }

    #[test]
    fn vector_ff() {
        assert!(crc32_xfer(&[0xFF]) == 0x0000_6565);
    }

    #[test]
    fn vector_check() {
        assert!(crc32_xfer(b"123456789") == 0xBD0B_E338);
    }

    #[test]
    fn vector_fox() {
        assert!(crc32_xfer(b"The quick brown fox") == 0x5DDF_5C83);
    }

    #[test]
    fn vector_0_1_2_3() {
        assert!(crc32_xfer(&[0, 1, 2, 3]) == 0x00AE_5FF1);
    }

    #[test]
    fn vector_a() {
        assert!(crc32_xfer(b"a") == 0x0000_3E8F);
    }

    #[test]
    fn vector_b() {
        assert!(crc32_xfer(b"b") == 0x0000_3F7E);
    }

    #[test]
    fn vector_ab() {
        assert!(crc32_xfer(b"ab") == 0x003E_B07E);
    }

    #[test]
    fn vector_abc() {
        assert!(crc32_xfer(b"abc") == 0x3EB0_41D1);
    }

    #[test]
    fn vector_01() {
        assert!(crc32_xfer(&[0x01]) == 0x0000_00AF);
    }

    #[test]
    fn vector_02() {
        assert!(crc32_xfer(&[0x02]) == 0x0000_015E);
    }

    #[test]
    fn vector_7f() {
        assert!(crc32_xfer(&[0x7F]) == 0x0000_32E5);
    }

    #[test]
    fn vector_80() {
        assert!(crc32_xfer(&[0x80]) == 0x0000_5780);
    }

    #[test]
    fn vector_aa_55() {
        assert!(crc32_xfer(&[0xAA, 0x55]) == 0x0046_6523);
    }

    #[test]
    fn vector_55_aa() {
        assert!(crc32_xfer(&[0x55, 0xAA]) == 0x0023_6546);
    }

    #[test]
    fn vector_deadbeef() {
        assert!(crc32_xfer(&[0xDE, 0xAD, 0xBE, 0xEF]) == 0x6F45_5145);
    }

    #[test]
    fn vector_hello() {
        assert!(crc32_xfer(b"Hello") == 0x0ACE_F329);
    }

    #[test]
    fn vector_two_zeros() {
        assert!(crc32_xfer(&[0, 0]) == 0x0000_0000);
    }

    #[test]
    fn vector_four_zeros() {
        assert!(crc32_xfer(&[0u8; 4]) == 0x0000_0000);
    }

    #[test]
    fn vector_four_ff() {
        assert!(crc32_xfer(&[0xFFu8; 4]) == 0x0000_3C56);
    }

    #[test]
    fn vector_ff_ff() {
        assert!(crc32_xfer(&[0xFF, 0xFF]) == 0x0065_0065);
    }

    #[test]
    fn vector_12345678_bytes() {
        assert!(crc32_xfer(&[0x12, 0x34, 0x56, 0x78]) == 0xB38E_E721);
    }

    #[test]
    fn vector_0_15() {
        let mut buf = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc32_xfer(&buf) == 0x56C3_8155);
    }

    #[test]
    fn vector_all256() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc32_xfer(&buf) == 0x3FF2_756B);
    }

    #[test]
    fn vector_a5_1000() {
        let buf = [A5_FILL; A5_LEN];
        assert!(crc32_xfer(&buf) == 0xE0F0_8F4E);
    }

    #[test]
    fn check_constant_is_0xbd0be338() {
        let expected: u32 = 0xBD0B_E338;
        assert!(crc32_xfer(b"123456789") == expected);
    }

    #[test]
    fn empty_equals_xorout() {
        assert!(crc32_xfer(b"") == ((INIT ^ XOROUT) & MASK));
    }

    #[test]
    fn single_01_equals_poly() {
        assert!(crc32_xfer(&[0x01]) == POLY);
    }

    #[test]
    fn determinism_empty() {
        assert!(crc32_xfer(b"") == crc32_xfer(b""));
    }

    #[test]
    fn determinism_fox() {
        assert!(crc32_xfer(b"The quick brown fox") == crc32_xfer(b"The quick brown fox"));
    }

    #[test]
    fn determinism_all256() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc32_xfer(&buf) == crc32_xfer(&buf));
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc32_xfer(b"a") != crc32_xfer(b"b"));
    }

    #[test]
    fn prefix_ab_differs_from_abc() {
        assert!(crc32_xfer(b"ab") != crc32_xfer(b"abc"));
    }

    #[test]
    fn length_sensitive_ff() {
        assert!(crc32_xfer(&[0xFF]) != crc32_xfer(&[0xFF, 0xFF]));
    }

    #[test]
    fn single_byte_01_differs_from_02() {
        assert!(crc32_xfer(&[0x01]) != crc32_xfer(&[0x02]));
    }

    #[test]
    fn single_byte_7f_differs_from_80() {
        assert!(crc32_xfer(&[0x7F]) != crc32_xfer(&[0x80]));
    }

    #[test]
    fn single_byte_00_differs_from_ff() {
        assert!(crc32_xfer(&[0x00]) != crc32_xfer(&[0xFF]));
    }

    #[test]
    fn aa55_differs_from_55aa() {
        assert!(crc32_xfer(&[0xAA, 0x55]) != crc32_xfer(&[0x55, 0xAA]));
    }

    #[test]
    fn single_byte_sample_pairwise_distinct() {
        let samples: [u8; 6] = [0x00, 0x01, 0x02, 0x7F, 0x80, 0xFF];
        let mut i = 0usize;
        while i < samples.len() {
            let mut j = i + 1;
            while j < samples.len() {
                assert!(crc32_xfer(&[samples[i]]) != crc32_xfer(&[samples[j]]));
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
            assert!(crc32_xfer(&single) == crc32_xfer(&single));
            byte += 1;
        }
    }

    #[test]
    fn two_zeros_matches_empty_property() {
        assert!(crc32_xfer(&[0, 0]) == crc32_xfer(b""));
    }

    #[test]
    fn leading_zero_does_not_change_result() {
        assert!(crc32_xfer(&[0x00, 0xFF]) == crc32_xfer(&[0xFF]));
    }

    #[test]
    fn reflect_reverses_bits() {
        assert!(reflect(0b0000_0001, 8) == 0b1000_0000);
    }

    #[test]
    fn reflect_full_width_roundtrip() {
        let value: u32 = 0x1234_5678;
        assert!(reflect(reflect(value, WIDTH), WIDTH) == value);
    }
}
