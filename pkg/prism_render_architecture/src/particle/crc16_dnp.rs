//! `CRC-16/DNP` checksum over byte slices.
//!
//! Implements the `CRC-16/DNP` variant used by the `DNP` 3.0 protocol.
//! Parameters: width 16, poly `0x3D65`, init `0x0000`,
//! refin true, refout true, xorout `0xFFFF`.
//!
//! Because refin and refout are both true, the computation runs
//! least-significant-bit first using the reflected polynomial
//! `bitreverse(0x3D65, 16) == 0xA6BC`. The implementation uses only
//! integer bit arithmetic so it stays valid under `no_std` with `alloc`
//! and never touches floating point or transcendental functions.

/// Reflected polynomial for `CRC-16/DNP` (`bitreverse(0x3D65, 16)`).
const REFPOLY: u16 = 0xA6BC;

/// Compute the `CRC-16/DNP` checksum of `data`.
///
/// Processes each byte least-significant-bit first against the reflected
/// polynomial, then applies the final `xorout` of `0xFFFF`.
#[must_use]
pub fn crc16_dnp(data: &[u8]) -> u16 {
    let mut crc: u16 = 0x0000; // init
    for &b in data {
        crc ^= b as u16;
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ REFPOLY;
            } else {
                crc >>= 1;
            }
        }
    }
    crc ^ 0xFFFF // xorout
}

#[cfg(test)]
mod tests {
    use super::crc16_dnp;

    // ---- anchor vectors (ground truth) ----

    #[test]
    fn anchor_empty() {
        assert!(crc16_dnp(b"") == 0xffff);
    }

    #[test]
    fn anchor_a() {
        assert!(crc16_dnp(b"a") == 0xb350);
    }

    #[test]
    fn anchor_zero_byte() {
        assert!(crc16_dnp(&[0x00]) == 0xffff);
    }

    #[test]
    fn anchor_ff_byte() {
        assert!(crc16_dnp(&[0xff]) == 0xedca);
    }

    #[test]
    fn anchor_check_value() {
        assert!(crc16_dnp(b"123456789") == 0xea82);
    }

    // ---- exact computed multi-byte and single-byte values ----

    #[test]
    fn exact_abc() {
        assert!(crc16_dnp(b"abc") == 0xe957);
    }

    #[test]
    fn exact_hello() {
        assert!(crc16_dnp(b"Hello") == 0x40ec);
    }

    #[test]
    fn exact_hello_world() {
        assert!(crc16_dnp(b"Hello, World!") == 0x97d0);
    }

    #[test]
    fn exact_byte_01() {
        assert!(crc16_dnp(&[0x01]) == 0xc9a1);
    }

    #[test]
    fn exact_byte_02() {
        assert!(crc16_dnp(&[0x02]) == 0x9343);
    }

    #[test]
    fn exact_byte_80() {
        assert!(crc16_dnp(&[0x80]) == 0x5943);
    }

    #[test]
    fn exact_two_zeros() {
        assert!(crc16_dnp(&[0x00, 0x00]) == 0xffff);
    }

    #[test]
    fn exact_two_ff() {
        assert!(crc16_dnp(&[0xff, 0xff]) == 0x993a);
    }

    #[test]
    fn exact_counting_quad() {
        assert!(crc16_dnp(&[0x12, 0x34, 0x56, 0x78]) == 0xafdc);
    }

    #[test]
    fn exact_deadbeef() {
        assert!(crc16_dnp(&[0xde, 0xad, 0xbe, 0xef]) == 0x60b8);
    }

    #[test]
    fn exact_quick_brown_fox() {
        assert!(crc16_dnp(b"The quick brown fox") == 0xae0c);
    }

    #[test]
    fn exact_prism() {
        assert!(crc16_dnp(b"prism") == 0xf21e);
    }

    #[test]
    fn exact_ramp_eight() {
        assert!(crc16_dnp(&[0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]) == 0xefe6);
    }

    #[test]
    fn exact_byte_aa() {
        assert!(crc16_dnp(&[0xaa]) == 0xe3d9);
    }

    #[test]
    fn exact_byte_55() {
        assert!(crc16_dnp(&[0x55]) == 0xf1ec);
    }

    #[test]
    fn exact_aa_55() {
        assert!(crc16_dnp(&[0xaa, 0x55]) == 0x6d9b);
    }

    #[test]
    fn exact_55_aa() {
        assert!(crc16_dnp(&[0x55, 0xaa]) == 0x0b5e);
    }

    #[test]
    fn exact_ab() {
        assert!(crc16_dnp(b"AB") == 0xc078);
    }

    #[test]
    fn exact_ba() {
        assert!(crc16_dnp(b"BA") == 0x2a31);
    }

    #[test]
    fn exact_test() {
        assert!(crc16_dnp(b"test") == 0xb475);
    }

    #[test]
    fn exact_cafebabe() {
        assert!(crc16_dnp(&[0xca, 0xfe, 0xba, 0xbe]) == 0x8b4b);
    }

    #[test]
    fn exact_byte_7f() {
        assert!(crc16_dnp(&[0x7f]) == 0x4b76);
    }

    #[test]
    fn exact_byte_81() {
        assert!(crc16_dnp(&[0x81]) == 0x6f1d);
    }

    #[test]
    fn exact_newline() {
        assert!(crc16_dnp(b"\n") == 0x6cca);
    }

    #[test]
    fn exact_two_spaces() {
        assert!(crc16_dnp(b"  ") == 0x2579);
    }

    // ---- boundary / incremental / determinism tests ----

    #[test]
    fn boundary_empty_equals_init_xorout() {
        // init 0x0000 with no bytes xored out by 0xFFFF.
        assert!(crc16_dnp(&[]) == 0xffff);
    }

    #[test]
    fn boundary_single_zero_matches_empty() {
        // A single 0x00 byte folds to the same value as the empty input.
        assert!(crc16_dnp(&[0x00]) == crc16_dnp(b""));
    }

    #[test]
    fn determinism_repeated_calls_match() {
        let value = crc16_dnp(b"123456789");
        assert!(value == crc16_dnp(b"123456789"));
    }

    #[test]
    fn determinism_abc_stable() {
        assert!(crc16_dnp(b"abc") == crc16_dnp(b"abc"));
    }

    #[test]
    fn order_dependence_ab_vs_ba() {
        assert!(crc16_dnp(b"AB") != crc16_dnp(b"BA"));
    }

    #[test]
    fn order_dependence_aa55_vs_55aa() {
        assert!(crc16_dnp(&[0xaa, 0x55]) != crc16_dnp(&[0x55, 0xaa]));
    }

    #[test]
    fn incremental_appending_changes_result() {
        // Extending the input generally changes the checksum here.
        assert!(crc16_dnp(b"abc") != crc16_dnp(b"abcd"));
    }

    #[test]
    fn incremental_prefix_differs_from_whole() {
        assert!(crc16_dnp(b"Hello") != crc16_dnp(b"Hello, World!"));
    }

    #[test]
    fn distinct_single_bytes_differ() {
        assert!(crc16_dnp(&[0x01]) != crc16_dnp(&[0x02]));
    }

    #[test]
    fn complement_bytes_differ() {
        assert!(crc16_dnp(&[0xaa]) != crc16_dnp(&[0x55]));
    }

    #[test]
    fn repeated_zeros_stay_fixed_point() {
        // Leading zero bytes keep folding the all-zero state.
        assert!(crc16_dnp(&[0x00, 0x00]) == crc16_dnp(&[0x00]));
    }

    #[test]
    fn result_fits_u16_range() {
        let value = crc16_dnp(b"The quick brown fox");
        assert!((0x0000..=0xffff).contains(&value));
    }

    #[test]
    fn two_ff_differs_from_one_ff() {
        assert!(crc16_dnp(&[0xff, 0xff]) != crc16_dnp(&[0xff]));
    }

    #[test]
    fn ramp_differs_from_counting_quad() {
        assert!(
            crc16_dnp(&[0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07])
                != crc16_dnp(&[0x12, 0x34, 0x56, 0x78])
        );
    }
}
