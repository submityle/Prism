//! `CRC`-8/OPENSAFETY: width=8, poly=`0x2F`, init=`0x00`, refin=false, refout=false, xorout=`0x00` (non-reflected, `MSB`-first `XOR` into a `u8` register), check(`b"123456789"`)=`0x3E`.
//!
//! This module computes the `CRC`-8/OPENSAFETY checksum of a byte slice.
//! Because `refin` and `refout` are both false, each input byte is fed into
//! the register `MSB`-first, and the register is `XOR`-reduced against the
//! polynomial `0x2F` whenever the high bit is set. The register is a plain
//! `u8`; no reflection or final `XOR` is applied.

/// Generator polynomial for `CRC`-8/OPENSAFETY (`x^8 + x^5 + x^3 + x^2 + x + 1`).
const POLY: u8 = 0x2F;

/// Compute the `CRC`-8/OPENSAFETY checksum of `data`.
///
/// Parameters: width=8, poly=`0x2F`, init=`0x00`, refin=false, refout=false,
/// xorout=`0x00`. The check value for `b"123456789"` is `0x3E`.
#[must_use]
pub fn crc8_opensafety(data: &[u8]) -> u8 {
    let mut reg: u8 = 0x00;
    for &byte in data {
        reg ^= byte;
        for _ in 0..8 {
            reg = if (reg & 0x80) != 0 {
                (reg << 1) ^ POLY
            } else {
                reg << 1
            };
        }
    }
    reg
}

#[cfg(test)]
mod tests {
    use super::crc8_opensafety;

    // --- Anchor 1: empty input ---
    #[test]
    fn anchor_empty_is_zero() {
        assert!(crc8_opensafety(b"") == 0x00);
    }

    #[test]
    fn anchor_empty_slice_literal() {
        let empty: [u8; 0] = [];
        assert!(crc8_opensafety(&empty) == 0x00);
    }

    // --- Anchor 2: single 0x00 ---
    #[test]
    fn anchor_single_zero() {
        assert!(crc8_opensafety(&[0x00]) == 0x00);
    }

    #[test]
    fn anchor_single_zero_matches_empty() {
        assert!(crc8_opensafety(&[0x00]) == crc8_opensafety(b""));
    }

    // --- Anchor 3: single 0xFF ---
    #[test]
    fn anchor_single_ff() {
        assert!(crc8_opensafety(&[0xFF]) == 0x42);
    }

    #[test]
    fn anchor_single_ff_nonzero() {
        assert!(crc8_opensafety(&[0xFF]) != 0x00);
    }

    // --- Anchor 4: check value ---
    #[test]
    fn anchor_check_value() {
        assert!(crc8_opensafety(b"123456789") == 0x3E);
    }

    #[test]
    fn anchor_check_value_bytes() {
        let digits = [b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9'];
        assert!(crc8_opensafety(&digits) == 0x3E);
    }

    // --- Single-byte hardcoded vectors ---
    #[test]
    fn single_a() {
        assert!(crc8_opensafety(b"a") == 0xBA);
    }

    #[test]
    fn single_byte_0x01() {
        assert!(crc8_opensafety(&[0x01]) == 0x2F);
    }

    #[test]
    fn single_byte_0x80() {
        assert!(crc8_opensafety(&[0x80]) == 0xE3);
    }

    #[test]
    fn single_byte_0x2f() {
        assert!(crc8_opensafety(&[0x2F]) == 0xE9);
    }

    // --- Multi-byte hardcoded vectors ---
    #[test]
    fn multi_ab() {
        assert!(crc8_opensafety(b"ab") == 0xFC);
    }

    #[test]
    fn multi_abc() {
        assert!(crc8_opensafety(b"abc") == 0xD7);
    }

    #[test]
    fn multi_two_zeros() {
        assert!(crc8_opensafety(&[0x00, 0x00]) == 0x00);
    }

    #[test]
    fn multi_two_ff() {
        assert!(crc8_opensafety(&[0xFF, 0xFF]) == 0xFA);
    }

    #[test]
    fn multi_one_two_three() {
        assert!(crc8_opensafety(&[0x01, 0x02, 0x03]) == 0x82);
    }

    #[test]
    fn multi_hello() {
        assert!(crc8_opensafety(b"Hello") == 0x1D);
    }

    #[test]
    fn multi_fox() {
        assert!(crc8_opensafety(b"The quick brown fox") == 0x10);
    }

    #[test]
    fn multi_12_34_56_78() {
        assert!(crc8_opensafety(&[0x12, 0x34, 0x56, 0x78]) == 0xF1);
    }

    #[test]
    fn multi_deadbeef() {
        assert!(crc8_opensafety(&[0xDE, 0xAD, 0xBE, 0xEF]) == 0xF9);
    }

    #[test]
    fn multi_opensafety_text() {
        assert!(crc8_opensafety(b"OPENSAFETY") == 0x6E);
    }

    #[test]
    fn multi_aa_55() {
        assert!(crc8_opensafety(&[0xAA, 0x55]) == 0xEE);
    }

    #[test]
    fn multi_sixteen_zeros() {
        assert!(crc8_opensafety(&[0x00; 16]) == 0x00);
    }

    #[test]
    fn multi_sixteen_ff() {
        assert!(crc8_opensafety(&[0xFF; 16]) == 0xCA);
    }

    // --- Long input hardcoded + stability ---
    #[test]
    fn long_256_a5() {
        assert!(crc8_opensafety(&[0xA5; 256]) == 0x72);
    }

    #[test]
    fn long_1024_a5() {
        assert!(crc8_opensafety(&[0xA5; 1024]) == 0x93);
    }

    #[test]
    fn long_sequence_0_to_255() {
        let mut buf = [0u8; 256];
        let mut i: usize = 0;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc8_opensafety(&buf) == 0x41);
    }

    #[test]
    fn long_input_deterministic() {
        let data = [0xA5; 1024];
        let first = crc8_opensafety(&data);
        let second = crc8_opensafety(&data);
        assert!(first == second);
    }

    #[test]
    fn long_input_stable_across_many_calls() {
        let data = [0x5Au8; 2048];
        let expected = crc8_opensafety(&data);
        let mut n: u32 = 0;
        while n < 64 {
            assert!(crc8_opensafety(&data) == expected);
            n += 1;
        }
    }

    // --- Determinism / property tests ---
    #[test]
    fn deterministic_empty() {
        assert!(crc8_opensafety(b"") == crc8_opensafety(b""));
    }

    #[test]
    fn deterministic_check_value() {
        assert!(crc8_opensafety(b"123456789") == crc8_opensafety(b"123456789"));
    }

    #[test]
    fn deterministic_single_byte_sweep() {
        let mut b: u16 = 0;
        while b <= 0xFF {
            let byte = b as u8;
            let first = crc8_opensafety(&[byte]);
            let second = crc8_opensafety(&[byte]);
            assert!(first == second);
            b += 1;
        }
    }

    #[test]
    fn result_within_u8_range() {
        let mut b: u16 = 0;
        while b <= 0xFF {
            let value = crc8_opensafety(&[b as u8]);
            assert!((0x00..=0xFF).contains(&value));
            b += 1;
        }
    }

    #[test]
    fn order_sensitive() {
        let forward = crc8_opensafety(&[0x01, 0x02]);
        let backward = crc8_opensafety(&[0x02, 0x01]);
        assert!(forward != backward);
    }

    #[test]
    fn single_byte_sweep_has_nonzero_results() {
        let mut nonzero_seen = false;
        let mut b: u16 = 1;
        while b <= 0xFF {
            if crc8_opensafety(&[b as u8]) != 0x00 {
                nonzero_seen = true;
            }
            b += 1;
        }
        assert!(nonzero_seen);
    }

    #[test]
    fn repeated_zero_prefix_keeps_value() {
        // Leading 0x00 bytes do not change the running register from init.
        assert!(crc8_opensafety(&[0x00, 0x00, 0x61]) == crc8_opensafety(b"a"));
    }

    #[test]
    fn even_length_zero_block() {
        let data = [0x00u8; 8];
        assert!(data.len().is_multiple_of(2));
        assert!(crc8_opensafety(&data) == 0x00);
    }

    #[test]
    fn slice_matches_array_reference() {
        let data = [0x12u8, 0x34, 0x56, 0x78];
        let via_slice = crc8_opensafety(&data[..]);
        assert!(via_slice == 0xF1);
    }

    #[test]
    fn check_value_differs_from_single_a() {
        assert!(crc8_opensafety(b"123456789") != crc8_opensafety(b"a"));
    }
}
