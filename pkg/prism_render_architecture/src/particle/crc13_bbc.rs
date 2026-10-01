//! `CRC-13/BBC` one-shot checksum over byte slices.
//!
//! This module implements the `CRC-13/BBC` catalog algorithm using a
//! bit-at-a-time, most-significant-bit-first schedule. It is pure integer
//! code with no floating-point or transcendental operations, and the
//! non-test API neither allocates nor formats: it accepts a `&[u8]` and
//! returns the 13-bit result packed into the low bits of a `u16`.
//!
//! Parameters: width `13`, polynomial `0x1cf5`, init `0x0000`,
//! `refin=false`, `refout=false`, `xorout=0x0000`. The catalog check value
//! for the input `b"123456789"` is `0x04fa`.

/// Compute the `CRC-13/BBC` checksum of `data`.
///
/// The returned `u16` carries the 13-bit result in its low bits, so the
/// value is always within `0x0000..=0x1fff`.
pub fn crc13_bbc(data: &[u8]) -> u16 {
    const WIDTH: u32 = 13;
    const POLY: u16 = 0x1cf5;
    const MASK: u16 = (1 << WIDTH) - 1; // 0x1fff
    let mut crc: u16 = 0;
    for &b in data {
        crc ^= (b as u16) << (WIDTH - 8); // align byte into top of the 13-bit register
        for _ in 0..8 {
            if (crc & (1 << (WIDTH - 1))) != 0 {
                crc = ((crc << 1) ^ POLY) & MASK;
            } else {
                crc = (crc << 1) & MASK;
            }
        }
    }
    crc & MASK
}

#[cfg(test)]
mod tests {
    use super::crc13_bbc;

    /// Upper bound of the 13-bit result space, used for range checks.
    const MASK: u16 = 0x1fff;

    // ----- The five external reference anchors -----

    #[test]
    fn anchor_empty_is_zero() {
        assert!(crc13_bbc(b"") == 0x0000);
    }

    #[test]
    fn anchor_single_a() {
        assert!(crc13_bbc(b"a") == 0x1d31);
    }

    #[test]
    fn anchor_zero_byte() {
        assert!(crc13_bbc(&[0x00]) == 0x0000);
    }

    #[test]
    fn anchor_ff_byte() {
        assert!(crc13_bbc(&[0xff]) == 0x04c2);
    }

    #[test]
    fn anchor_catalog_check_value() {
        assert!(crc13_bbc(b"123456789") == 0x04fa);
    }

    // ----- Self-derived single-character ASCII vectors -----

    #[test]
    fn digit_zero_char() {
        assert!(crc13_bbc(b"0") == 0x00e2);
    }

    #[test]
    fn digit_one_char() {
        assert!(crc13_bbc(b"1") == 0x1c17);
    }

    #[test]
    fn digit_nine_char() {
        assert!(crc13_bbc(b"9") == 0x086b);
    }

    #[test]
    fn upper_a_char() {
        assert!(crc13_bbc(b"A") == 0x09de);
    }

    #[test]
    fn lower_z_char() {
        assert!(crc13_bbc(b"z") == 0x04aa);
    }

    // ----- Self-derived multi-byte string vectors -----

    #[test]
    fn pair_ab() {
        assert!(crc13_bbc(b"ab") == 0x0a35);
    }

    #[test]
    fn pair_ba() {
        assert!(crc13_bbc(b"ba") == 0x1dff);
    }

    #[test]
    fn triple_abc() {
        assert!(crc13_bbc(b"abc") == 0x10fd);
    }

    #[test]
    fn triple_cba() {
        assert!(crc13_bbc(b"cba") == 0x1bca);
    }

    #[test]
    fn word_hello_lower() {
        assert!(crc13_bbc(b"hello") == 0x01c0);
    }

    #[test]
    fn word_hello_cap() {
        assert!(crc13_bbc(b"Hello") == 0x1833);
    }

    #[test]
    fn phrase_the_quick() {
        assert!(crc13_bbc(b"The quick") == 0x1dad);
    }

    #[test]
    fn word_prism() {
        assert!(crc13_bbc(b"prism") == 0x1500);
    }

    // ----- Self-derived raw-byte vectors -----

    #[test]
    fn two_zero_bytes() {
        assert!(crc13_bbc(&[0x00, 0x00]) == 0x0000);
    }

    #[test]
    fn three_zero_bytes() {
        assert!(crc13_bbc(&[0x00, 0x00, 0x00]) == 0x0000);
    }

    #[test]
    fn byte_01() {
        assert!(crc13_bbc(&[0x01]) == 0x1cf5);
    }

    #[test]
    fn byte_02() {
        assert!(crc13_bbc(&[0x02]) == 0x051f);
    }

    #[test]
    fn byte_80() {
        assert!(crc13_bbc(&[0x80]) == 0x16a3);
    }

    #[test]
    fn byte_7f() {
        assert!(crc13_bbc(&[0x7f]) == 0x1261);
    }

    #[test]
    fn byte_aa() {
        assert!(crc13_bbc(&[0xaa]) == 0x132f);
    }

    #[test]
    fn byte_55() {
        assert!(crc13_bbc(&[0x55]) == 0x17ed);
    }

    #[test]
    fn pair_01_02() {
        assert!(crc13_bbc(&[0x01, 0x02]) == 0x14ac);
    }

    #[test]
    fn pair_02_01() {
        assert!(crc13_bbc(&[0x02, 0x01]) == 0x0366);
    }

    #[test]
    fn pair_ff_ff() {
        assert!(crc13_bbc(&[0xff, 0xff]) == 0x1d0c);
    }

    #[test]
    fn pair_ff_00() {
        assert!(crc13_bbc(&[0xff, 0x00]) == 0x19ce);
    }

    #[test]
    fn pair_00_ff() {
        assert!(crc13_bbc(&[0x00, 0xff]) == 0x04c2);
    }

    #[test]
    fn pair_aa_55() {
        assert!(crc13_bbc(&[0xaa, 0x55]) == 0x12ca);
    }

    #[test]
    fn quad_deadbeef() {
        assert!(crc13_bbc(&[0xde, 0xad, 0xbe, 0xef]) == 0x0843);
    }

    #[test]
    fn quad_ascending_nibbles() {
        assert!(crc13_bbc(&[0x12, 0x34, 0x56, 0x78]) == 0x1e0e);
    }

    #[test]
    fn sequence_zero_to_seven() {
        assert!(crc13_bbc(&[0, 1, 2, 3, 4, 5, 6, 7]) == 0x18a6);
    }

    #[test]
    fn sequence_one_to_eight() {
        assert!(crc13_bbc(&[1, 2, 3, 4, 5, 6, 7, 8]) == 0x073f);
    }

    // ----- Property / invariant tests -----

    #[test]
    fn single_01_equals_polynomial() {
        // Feeding a lone `0x01` byte through the schedule reproduces the
        // `CRC-13/BBC` polynomial exactly.
        assert!(crc13_bbc(&[0x01]) == 0x1cf5);
    }

    #[test]
    fn leading_zero_byte_is_invariant() {
        // A zero `init` with a zero `xorout` makes leading `0x00` bytes
        // transparent, so prefixing one does not change the result.
        assert!(crc13_bbc(&[0x00, 0xff]) == crc13_bbc(&[0xff]));
    }

    #[test]
    fn computation_is_deterministic() {
        let a = crc13_bbc(b"determinism");
        let b = crc13_bbc(b"determinism");
        assert!(a == b);
    }

    #[test]
    fn order_is_significant_strings() {
        assert!(crc13_bbc(b"ab") != crc13_bbc(b"ba"));
    }

    #[test]
    fn order_is_significant_bytes() {
        assert!(crc13_bbc(&[0x01, 0x02]) != crc13_bbc(&[0x02, 0x01]));
    }

    #[test]
    fn results_stay_within_thirteen_bits() {
        let samples: [&[u8]; 8] = [
            b"",
            b"a",
            b"abc",
            b"123456789",
            &[0xff, 0xff, 0xff],
            &[0xde, 0xad, 0xbe, 0xef],
            &[0x00, 0x01, 0x02, 0x03],
            b"prism-render-architecture",
        ];
        for &s in &samples {
            assert!(crc13_bbc(s) <= MASK);
        }
    }

    #[test]
    fn result_lies_in_valid_range() {
        assert!((0x0000..=MASK).contains(&crc13_bbc(b"range-check")));
    }

    #[test]
    fn anchor_parity_is_odd() {
        // `0x1d31` is odd, so it is not a multiple of two.
        assert!(!crc13_bbc(b"a").is_multiple_of(2));
    }

    #[test]
    fn empty_result_is_multiple_of_mask() {
        // Zero is a multiple of every modulus, including the 13-bit mask.
        assert!(crc13_bbc(b"").is_multiple_of(MASK));
    }
}
