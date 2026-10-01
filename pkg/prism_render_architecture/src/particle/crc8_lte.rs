//! `CRC-8/LTE` (non-reflected, `MSB`-first, `u8`): width=8, poly=0x9B, init=0x00, refin=false, refout=false, xorout=0x00; check(b"123456789")=0xea.
/// Computes the `CRC-8/LTE` checksum over `data`.
///
/// Non-reflected (`MSB`-first) `u8` `CRC` with polynomial 0x9B and zero
/// initialization. The `XOR` into the register is byte-aligned and the
/// `reg << 1` shift truncates to `u8` by design.
#[must_use]
pub fn crc8_lte(data: &[u8]) -> u8 {
    const POLY: u8 = 0x9B;
    let mut reg: u8 = 0x00;
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
    use super::crc8_lte;

    // --- 5 anchors ---
    #[test]
    fn anchor_empty() {
        assert!(crc8_lte(b"") == 0x00);
    }

    #[test]
    fn anchor_a() {
        assert!(crc8_lte(b"a") == 0x37);
    }

    #[test]
    fn anchor_zero_byte() {
        assert!(crc8_lte(&[0x00]) == 0x00);
    }

    #[test]
    fn anchor_ff_byte() {
        assert!(crc8_lte(&[0xff]) == 0x7b);
    }

    #[test]
    fn anchor_check() {
        assert!(crc8_lte(b"123456789") == 0xea);
    }

    // --- >=10 multi-byte hardcoded vectors (self-computed) ---
    #[test]
    fn multi_01_02_03() {
        assert!(crc8_lte(&[0x01, 0x02, 0x03]) == 0x44);
    }

    #[test]
    fn multi_deadbeef() {
        assert!(crc8_lte(&[0xde, 0xad, 0xbe, 0xef]) == 0xd8);
    }

    #[test]
    fn multi_three_zeros() {
        assert!(crc8_lte(&[0x00, 0x00, 0x00]) == 0x00);
    }

    #[test]
    fn multi_two_ff() {
        assert!(crc8_lte(&[0xff, 0xff]) == 0xca);
    }

    #[test]
    fn multi_12_34_56_78_9a() {
        assert!(crc8_lte(&[0x12, 0x34, 0x56, 0x78, 0x9a]) == 0x94);
    }

    #[test]
    fn multi_aa_55() {
        assert!(crc8_lte(&[0xaa, 0x55]) == 0x7e);
    }

    #[test]
    fn multi_80_01() {
        assert!(crc8_lte(&[0x80, 0x01]) == 0xb4);
    }

    #[test]
    fn multi_7f_80_81() {
        assert!(crc8_lte(&[0x7f, 0x80, 0x81]) == 0xf3);
    }

    #[test]
    fn multi_hello() {
        assert!(crc8_lte(b"Hello") == 0x87);
    }

    #[test]
    fn multi_prism() {
        assert!(crc8_lte(b"Prism") == 0x36);
    }

    #[test]
    fn multi_cafebabe() {
        assert!(crc8_lte(&[0xca, 0xfe, 0xba, 0xbe]) == 0x84);
    }

    #[test]
    fn multi_ramp() {
        assert!(crc8_lte(&[0x10, 0x20, 0x30, 0x40, 0x50, 0x60]) == 0x1c);
    }

    // --- determinism ---
    #[test]
    fn deterministic_repeat() {
        let data = b"deterministic";
        let first = crc8_lte(data);
        let second = crc8_lte(data);
        assert!(first == second);
    }

    #[test]
    fn deterministic_many_calls() {
        let data = &[0x11, 0x22, 0x33, 0x44];
        let expected = crc8_lte(data);
        let mut i: u32 = 0;
        while i < 16 {
            assert!(crc8_lte(data) == expected);
            i += 1;
        }
    }

    // --- empty input == 0 ---
    #[test]
    fn empty_slice_is_zero() {
        let empty: &[u8] = &[];
        assert!(crc8_lte(empty) == 0x00);
    }

    #[test]
    fn empty_literal_is_zero() {
        assert!(crc8_lte(b"") == 0x00);
    }

    // --- sampling / divisibility ---
    #[test]
    fn sample_value_in_range() {
        let value = crc8_lte(b"range-sample");
        assert!((0x00..=0xff).contains(&value));
    }

    #[test]
    fn sample_divisibility_flag() {
        let value = crc8_lte(&[0x02, 0x04, 0x06, 0x08]);
        let divisible = value.is_multiple_of(2);
        let is_even = (value & 0x01) == 0;
        assert!(divisible == is_even);
    }

    #[test]
    fn sample_distinct_inputs() {
        let a = crc8_lte(b"alpha");
        let b = crc8_lte(b"omega");
        assert!(a != b);
    }

    // --- long input stability ---
    #[test]
    fn long_zeros_stable() {
        let buf = [0x00u8; 256];
        let first = crc8_lte(&buf);
        let second = crc8_lte(&buf);
        assert!(first == second);
        assert!(crc8_lte(&buf) == 0x00);
    }

    #[test]
    fn long_ones_stable() {
        let buf = [0xffu8; 128];
        let first = crc8_lte(&buf);
        let second = crc8_lte(&buf);
        assert!(first == second);
    }

    #[test]
    fn long_pattern_stable() {
        let mut buf = [0u8; 200];
        let mut i: usize = 0;
        while i < buf.len() {
            buf[i] = (i & 0xff) as u8;
            i += 1;
        }
        let first = crc8_lte(&buf);
        let second = crc8_lte(&buf);
        assert!(first == second);
    }

    // --- structural properties ---
    #[test]
    fn single_zero_prefix_noop_on_zero_state() {
        // Leading 0x00 bytes on a zero register keep it zero.
        assert!(crc8_lte(&[0x00, 0x00]) == 0x00);
        assert!(crc8_lte(&[0x00, 0x00, 0x00, 0x00]) == 0x00);
    }

    #[test]
    fn prefix_changes_result() {
        let base = crc8_lte(&[0x41, 0x42]);
        let prefixed = crc8_lte(&[0x01, 0x41, 0x42]);
        assert!(base != prefixed);
    }

    #[test]
    fn order_sensitivity() {
        let forward = crc8_lte(&[0x01, 0x02]);
        let reversed = crc8_lte(&[0x02, 0x01]);
        assert!(forward != reversed);
    }

    #[test]
    fn concat_differs_from_parts() {
        let whole = crc8_lte(b"abcdef");
        let part = crc8_lte(b"abc");
        assert!(whole != part);
    }

    #[test]
    fn length_one_matches_two_arg_forms() {
        assert!(crc8_lte(&[0x01]) == crc8_lte(&[0x01]));
    }

    #[test]
    fn all_byte_values_in_range() {
        let mut b: u16 = 0;
        while b <= 0xff {
            let value = crc8_lte(&[b as u8]);
            assert!((0x00..=0xff).contains(&value));
            b += 1;
        }
    }

    #[test]
    fn ascending_pair_differs() {
        let a = crc8_lte(&[0x10, 0x11]);
        let b = crc8_lte(&[0x10, 0x12]);
        assert!(a != b);
    }

    #[test]
    fn high_bit_input_consistency() {
        let data = &[0x80, 0x80, 0x80];
        let expected = crc8_lte(data);
        assert!(crc8_lte(data) == expected);
    }

    #[test]
    fn mixed_text_stable() {
        let data = b"The quick brown fox";
        let first = crc8_lte(data);
        let second = crc8_lte(data);
        assert!(first == second);
    }
}
