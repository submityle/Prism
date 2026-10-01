//! `CRC`-8/TECH-3250 (alias `CRC`-8/`AES`-`EBU`) checksum.
//!
//! Parameters: width = 8, poly = `0x1D`, init = `0xFF`, refin = `true`,
//! refout = `true`, xorout = `0x00`. Since refin and refout are both `true`,
//! the computation runs in the reflected (`LSB`-first) domain: the reflected
//! polynomial `REFPOLY` = bit-reversed `0x1D` = `0xB8` is used, the reflected
//! initial value is `0xFF`, and the final `XOR` of `0x00` leaves the `u8`
//! result unchanged. The standard check value for the ASCII string
//! "123456789" is `0x97`.

/// Reflected form of the `CRC`-8/TECH-3250 polynomial `0x1D` (`REFPOLY`).
const REFPOLY: u8 = 0xB8;

/// Compute the `CRC`-8/TECH-3250 checksum (`LSB`-first) of `data`.
///
/// Returns the 8-bit (`u8`) checksum; the trailing `XOR` of `0x00` leaves the
/// accumulator unchanged, so the reflected accumulator is returned directly.
pub fn crc8_tech3250(data: &[u8]) -> u8 {
    let mut crc: u8 = 0xFF;
    for &b in data {
        crc ^= b;
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ REFPOLY;
            } else {
                crc >>= 1;
            }
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Anchor vectors (hard references) --------------------------------

    #[test]
    fn anchor_empty() {
        assert!(crc8_tech3250(b"") == 0xff);
    }

    #[test]
    fn anchor_single_zero() {
        assert!(crc8_tech3250(&[0x00]) == 0x23);
    }

    #[test]
    fn anchor_single_ff() {
        assert!(crc8_tech3250(&[0xff]) == 0x00);
    }

    #[test]
    fn anchor_letter_a() {
        assert!(crc8_tech3250(b"a") == 0x35);
    }

    #[test]
    fn anchor_check_value() {
        assert!(crc8_tech3250(b"123456789") == 0x97);
    }

    // ---- Single-byte hard-coded vectors ----------------------------------

    #[test]
    fn single_0x01() {
        assert!(crc8_tech3250(&[0x01]) == 0x47);
    }

    #[test]
    fn single_0x0f() {
        assert!(crc8_tech3250(&[0x0f]) == 0xdd);
    }

    #[test]
    fn single_0x7f() {
        assert!(crc8_tech3250(&[0x7f]) == 0xb8);
    }

    #[test]
    fn single_0x80() {
        assert!(crc8_tech3250(&[0x80]) == 0x9b);
    }

    #[test]
    fn single_space() {
        assert!(crc8_tech3250(b" ") == 0x0d);
    }

    #[test]
    fn single_newline() {
        assert!(crc8_tech3250(b"\n") == 0x58);
    }

    #[test]
    fn single_lower_z() {
        assert!(crc8_tech3250(b"z") == 0x3d);
    }

    #[test]
    fn single_upper_z() {
        assert!(crc8_tech3250(b"Z") == 0x13);
    }

    // ---- Multi-byte hard-coded vectors -----------------------------------

    #[test]
    fn multi_ab() {
        assert!(crc8_tech3250(b"ab") == 0x06);
    }

    #[test]
    fn multi_abc() {
        assert!(crc8_tech3250(b"abc") == 0xf7);
    }

    #[test]
    fn multi_hello_world() {
        assert!(crc8_tech3250(b"Hello, World!") == 0x3e);
    }

    #[test]
    fn multi_quick_fox() {
        assert!(crc8_tech3250(b"The quick brown fox jumps over the lazy dog") == 0xc1);
    }

    #[test]
    fn multi_two_zeros() {
        assert!(crc8_tech3250(&[0x00, 0x00]) == 0x82);
    }

    #[test]
    fn multi_two_ff() {
        assert!(crc8_tech3250(&[0xff, 0xff]) == 0x23);
    }

    #[test]
    fn multi_sequence_1234() {
        assert!(crc8_tech3250(&[0x01, 0x02, 0x03, 0x04]) == 0xea);
    }

    #[test]
    fn multi_deadbeef() {
        assert!(crc8_tech3250(&[0xde, 0xad, 0xbe, 0xef]) == 0x64);
    }

    #[test]
    fn multi_sixteen_0x55() {
        assert!(crc8_tech3250(&[0x55; 16]) == 0x03);
    }

    #[test]
    fn multi_thirtytwo_0xaa() {
        assert!(crc8_tech3250(&[0xaa; 32]) == 0x17);
    }

    #[test]
    fn multi_incrementing_ten() {
        let data = [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        assert!(crc8_tech3250(&data) == 0xd0);
    }

    #[test]
    fn multi_prism() {
        assert!(crc8_tech3250(b"Prism") == 0x78);
    }

    #[test]
    fn multi_codex() {
        assert!(crc8_tech3250(b"codex") == 0x43);
    }

    #[test]
    fn multi_full_range_256() {
        let mut data = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc8_tech3250(&data) == 0xc5);
    }

    // ---- Long / repeated-input hard-coded vectors ------------------------

    #[test]
    fn long_zeros_256() {
        let data = [0x00u8; 256];
        assert!(crc8_tech3250(&data) == 0x23);
    }

    #[test]
    fn long_ff_256() {
        let data = [0xffu8; 256];
        assert!(crc8_tech3250(&data) == 0x00);
    }

    #[test]
    fn long_a5_2048() {
        let data = [0xa5u8; 2048];
        assert!(crc8_tech3250(&data) == 0xee);
    }

    // ---- Determinism -----------------------------------------------------

    #[test]
    fn deterministic_check_value() {
        let first = crc8_tech3250(b"123456789");
        let second = crc8_tech3250(b"123456789");
        assert!(first == second);
    }

    #[test]
    fn deterministic_hello() {
        let first = crc8_tech3250(b"Hello, World!");
        let second = crc8_tech3250(b"Hello, World!");
        assert!(first == second);
    }

    #[test]
    fn deterministic_empty() {
        assert!(crc8_tech3250(b"") == crc8_tech3250(&[]));
    }

    #[test]
    fn deterministic_long_input() {
        let data = [0xa5u8; 2048];
        let first = crc8_tech3250(&data);
        let second = crc8_tech3250(&data);
        assert!(first == second);
    }

    // ---- Stability of long inputs across repeated recomputation ----------

    #[test]
    fn stable_long_zeros_repeated() {
        let data = [0x00u8; 1024];
        let mut iteration = 0usize;
        let baseline = crc8_tech3250(&data);
        while iteration < 16 {
            assert!(crc8_tech3250(&data) == baseline);
            iteration += 1;
        }
    }

    #[test]
    fn stable_long_pattern_repeated() {
        let data = [0x3cu8; 777];
        let baseline = crc8_tech3250(&data);
        let mut iteration = 0usize;
        while iteration < 8 {
            assert!(crc8_tech3250(&data) == baseline);
            iteration += 1;
        }
    }

    // ---- Structural / property checks ------------------------------------

    #[test]
    fn differing_inputs_may_differ() {
        assert!(crc8_tech3250(b"abc") != crc8_tech3250(b"abd"));
    }

    #[test]
    fn order_sensitive() {
        assert!(crc8_tech3250(&[0x01, 0x02]) != crc8_tech3250(&[0x02, 0x01]));
    }

    #[test]
    fn result_in_byte_range() {
        let value = crc8_tech3250(b"range check");
        assert!((0x00u8..=0xffu8).contains(&value));
    }

    #[test]
    fn prefix_subset_differs_from_full() {
        let full = crc8_tech3250(b"123456789");
        let prefix = crc8_tech3250(b"12345");
        assert!(full != prefix);
    }

    #[test]
    fn length_two_matches_slice() {
        let data = [0xde, 0xad, 0xbe, 0xef];
        assert!(crc8_tech3250(&data[0..2]) == crc8_tech3250(&[0xde, 0xad]));
    }
}
