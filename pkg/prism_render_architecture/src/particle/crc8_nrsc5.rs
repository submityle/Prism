//! `CRC`-8/`NRSC`-5 checksum: width 8, poly `0x31`, init `0xFF`,
//! refin false, refout false, xorout `0x00` (non-reflected,
//! `MSB`-first, no final `XOR`). The register is a `u8` and the canonical
//! check value for b"123456789" is `0xF7`.

/// Generator polynomial for `CRC`-8/`NRSC`-5 (represents x^8 + x^5 + x^4 + 1).
const POLY: u8 = 0x31;

/// Computes the `CRC`-8/`NRSC`-5 checksum of data.
///
/// The `u8` register starts at `0xFF`, each byte is folded in and shifted
/// `MSB`-first eight times, and no final `XOR` is applied (xorout `0x00`).
pub fn crc8_nrsc5(data: &[u8]) -> u8 {
    let mut reg: u8 = 0xFF;
    for &byte in data {
        reg ^= byte;
        for _ in 0..8 {
            if (reg & 0x80) != 0 {
                reg = (reg << 1) ^ POLY;
            } else {
                reg <<= 1;
            }
        }
    }
    reg
}

#[cfg(test)]
mod tests {
    use super::crc8_nrsc5;

    #[test]
    fn empty_matches_init() {
        assert!(crc8_nrsc5(b"") == 0xff);
    }

    #[test]
    fn single_zero_byte() {
        assert!(crc8_nrsc5(&[0x00]) == 0xac);
    }

    #[test]
    fn single_ff_byte() {
        assert!(crc8_nrsc5(&[0xff]) == 0x00);
    }

    #[test]
    fn single_lowercase_a() {
        assert!(crc8_nrsc5(b"a") == 0x26);
    }

    #[test]
    fn canonical_check_value() {
        assert!(crc8_nrsc5(b"123456789") == 0xf7);
    }

    #[test]
    fn ascii_abc() {
        assert!(crc8_nrsc5(b"abc") == 0xe2);
    }

    #[test]
    fn ascii_abc_upper() {
        assert!(crc8_nrsc5(b"ABC") == 0xc1);
    }

    #[test]
    fn ascii_hello() {
        assert!(crc8_nrsc5(b"Hello") == 0xed);
    }

    #[test]
    fn ascii_hello_world() {
        assert!(crc8_nrsc5(b"Hello, World!") == 0xc0);
    }

    #[test]
    fn pangram_fox() {
        assert!(crc8_nrsc5(b"The quick brown fox jumps over the lazy dog") == 0x86);
    }

    #[test]
    fn two_zero_bytes() {
        assert!(crc8_nrsc5(&[0x00, 0x00]) == 0x81);
    }

    #[test]
    fn three_zero_bytes() {
        assert!(crc8_nrsc5(&[0x00, 0x00, 0x00]) == 0x4b);
    }

    #[test]
    fn byte_one() {
        assert!(crc8_nrsc5(&[0x01]) == 0x9d);
    }

    #[test]
    fn byte_two() {
        assert!(crc8_nrsc5(&[0x02]) == 0xce);
    }

    #[test]
    fn byte_0x7f() {
        assert!(crc8_nrsc5(&[0x7f]) == 0x7a);
    }

    #[test]
    fn byte_0x80() {
        assert!(crc8_nrsc5(&[0x80]) == 0xd6);
    }

    #[test]
    fn sequence_one_to_four() {
        assert!(crc8_nrsc5(&[0x01, 0x02, 0x03, 0x04]) == 0x29);
    }

    #[test]
    fn deadbeef() {
        assert!(crc8_nrsc5(&[0xde, 0xad, 0xbe, 0xef]) == 0x69);
    }

    #[test]
    fn ff_ff() {
        assert!(crc8_nrsc5(&[0xff, 0xff]) == 0xac);
    }

    #[test]
    fn ff_then_00() {
        assert!(crc8_nrsc5(&[0xff, 0x00]) == 0x00);
    }

    #[test]
    fn zero_then_ff() {
        assert!(crc8_nrsc5(&[0x00, 0xff]) == 0x2d);
    }

    #[test]
    fn digit_zero() {
        assert!(crc8_nrsc5(b"0") == 0x69);
    }

    #[test]
    fn digit_nine() {
        assert!(crc8_nrsc5(b"9") == 0xe1);
    }

    #[test]
    fn four_capital_a() {
        assert!(crc8_nrsc5(b"AAAA") == 0x69);
    }

    #[test]
    fn word_prism() {
        assert!(crc8_nrsc5(b"Prism") == 0xff);
    }

    #[test]
    fn digits_zero_to_nine() {
        assert!(crc8_nrsc5(b"0123456789") == 0x85);
    }

    #[test]
    fn single_space() {
        assert!(crc8_nrsc5(b" ") == 0x2a);
    }

    #[test]
    fn single_newline() {
        assert!(crc8_nrsc5(&[0x0a]) == 0x77);
    }

    #[test]
    fn all_256_byte_values() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc8_nrsc5(&buf) == 0x14);
    }

    #[test]
    fn thousand_zero_bytes() {
        let buf = [0u8; 1000];
        assert!(crc8_nrsc5(&buf) == 0xe7);
    }

    #[test]
    fn thousand_ff_bytes() {
        let buf = [0xffu8; 1000];
        assert!(crc8_nrsc5(&buf) == 0x88);
    }

    #[test]
    fn generated_pattern_512() {
        let mut buf = [0u8; 512];
        let mut i = 0usize;
        while i < 512 {
            buf[i] = (i * 7 + 3) as u8;
            i += 1;
        }
        assert!(crc8_nrsc5(&buf) == 0x74);
    }

    #[test]
    fn deterministic_on_repeat() {
        let data = b"deterministic-input";
        let first = crc8_nrsc5(data);
        let second = crc8_nrsc5(data);
        assert!(first == second);
    }

    #[test]
    fn deterministic_empty_repeat() {
        assert!(crc8_nrsc5(b"") == crc8_nrsc5(b""));
    }

    #[test]
    fn deterministic_check_repeat() {
        assert!(crc8_nrsc5(b"123456789") == crc8_nrsc5(b"123456789"));
    }

    #[test]
    fn byte_order_changes_result() {
        let forward = crc8_nrsc5(&[0xff, 0x00]);
        let reversed = crc8_nrsc5(&[0x00, 0xff]);
        assert!(forward != reversed);
    }

    #[test]
    fn distinct_lengths_differ() {
        let one = crc8_nrsc5(&[0x00]);
        let two = crc8_nrsc5(&[0x00, 0x00]);
        assert!(one != two);
    }

    #[test]
    fn long_input_stability() {
        let mut buf = [0u8; 512];
        let mut i = 0usize;
        while i < 512 {
            buf[i] = (i * 7 + 3) as u8;
            i += 1;
        }
        let a = crc8_nrsc5(&buf);
        let b = crc8_nrsc5(&buf);
        assert!(a == b);
        assert!(a == 0x74);
    }

    #[test]
    fn thousand_zeros_stable() {
        let buf = [0u8; 1000];
        assert!(crc8_nrsc5(&buf) == crc8_nrsc5(&buf));
    }

    #[test]
    fn thousand_ffs_stable() {
        let buf = [0xffu8; 1000];
        assert!(crc8_nrsc5(&buf) == crc8_nrsc5(&buf));
    }

    #[test]
    fn all256_deterministic() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc8_nrsc5(&buf) == crc8_nrsc5(&buf));
    }

    #[test]
    fn single_byte_distinct_from_zero() {
        assert!(crc8_nrsc5(&[0x01]) != crc8_nrsc5(&[0x00]));
    }

    #[test]
    fn empty_differs_from_single_zero() {
        assert!(crc8_nrsc5(b"") != crc8_nrsc5(&[0x00]));
    }
}
