//! `CRC`-32/MPEG-2 checksum.
//!
//! Parameters: width = 32, poly = `0x04C11DB7`, init = `0xFFFFFFFF`,
//! `refin` = false, `refout` = false, `xorout` = `0x00000000`.
//! Processing is non-reflected (`MSB`-first) with a final `XOR` of zero,
//! so the register is returned unchanged. The function yields a `u32`.
//! check(`b"123456789"`) = `0x0376E6E7`.

/// Generator polynomial for `CRC`-32/MPEG-2 (`0x04C11DB7`).
const POLY: u32 = 0x04C1_1DB7;

/// Compute the `CRC`-32/MPEG-2 checksum of `data`.
///
/// Non-reflected, `MSB`-first, init `0xFFFFFFFF`, `xorout` `0x00000000`.
pub fn crc32_mpeg2(data: &[u8]) -> u32 {
    let mut reg: u32 = 0xFFFF_FFFF;
    for &b in data {
        reg ^= u32::from(b) << 24;
        for _ in 0..8 {
            if (reg & 0x8000_0000) != 0 {
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
    use super::crc32_mpeg2;

    // --- Anchor 1: empty input returns the untouched init value. ---
    #[test]
    fn anchor_empty() {
        assert!(crc32_mpeg2(b"") == 0xFFFF_FFFF);
    }

    #[test]
    fn anchor_empty_slice() {
        let empty: [u8; 0] = [];
        assert!(crc32_mpeg2(&empty) == 0xFFFF_FFFF);
    }

    // --- Anchor 2: single 0x00 byte. ---
    #[test]
    fn anchor_single_zero() {
        assert!(crc32_mpeg2(&[0x00]) == 0x4E08_BFB4);
    }

    // --- Anchor 3: single 0xFF byte. ---
    #[test]
    fn anchor_single_ff() {
        assert!(crc32_mpeg2(&[0xFF]) == 0xFFFF_FF00);
    }

    // --- Anchor 4: canonical check vector. ---
    #[test]
    fn anchor_check_vector() {
        assert!(crc32_mpeg2(b"123456789") == 0x0376_E6E7);
    }

    // --- Single-byte vectors. ---
    #[test]
    fn single_byte_01() {
        assert!(crc32_mpeg2(&[0x01]) == 0x4AC9_A203);
    }

    #[test]
    fn single_byte_7f() {
        assert!(crc32_mpeg2(&[0x7F]) == 0x96F3_1FEE);
    }

    #[test]
    fn single_byte_80() {
        assert!(crc32_mpeg2(&[0x80]) == 0x2704_5F5A);
    }

    // --- ASCII letter vectors. ---
    #[test]
    fn ascii_a() {
        assert!(crc32_mpeg2(b"a") == 0xE66C_6494);
    }

    #[test]
    fn ascii_ab() {
        assert!(crc32_mpeg2(b"ab") == 0x166C_0232);
    }

    #[test]
    fn ascii_abc() {
        assert!(crc32_mpeg2(b"abc") == 0x9B73_448C);
    }

    #[test]
    fn ascii_abcd() {
        assert!(crc32_mpeg2(b"abcd") == 0xC2B3_CCB4);
    }

    #[test]
    fn ascii_aaa() {
        assert!(crc32_mpeg2(b"aaa") == 0xE01A_2031);
    }

    #[test]
    fn ascii_hello() {
        assert!(crc32_mpeg2(b"hello") == 0xE6CE_9AC2);
    }

    #[test]
    fn ascii_hello_world() {
        assert!(crc32_mpeg2(b"Hello, World!") == 0x1927_0120);
    }

    #[test]
    fn ascii_quick_brown_fox() {
        let msg = b"The quick brown fox jumps over the lazy dog";
        assert!(crc32_mpeg2(msg) == 0xBA62_119E);
    }

    // --- Multi-byte binary vectors. ---
    #[test]
    fn two_zero_bytes() {
        assert!(crc32_mpeg2(&[0x00, 0x00]) == 0x00B7_647D);
    }

    #[test]
    fn two_ff_bytes() {
        assert!(crc32_mpeg2(&[0xFF, 0xFF]) == 0xFFFF_0000);
    }

    #[test]
    fn ff_then_zero() {
        assert!(crc32_mpeg2(&[0xFF, 0x00]) == 0x4E08_40B4);
    }

    #[test]
    fn zero_then_ff() {
        assert!(crc32_mpeg2(&[0x00, 0xFF]) == 0xB140_24C9);
    }

    #[test]
    fn ascending_five() {
        assert!(crc32_mpeg2(&[1, 2, 3, 4, 5]) == 0xE28F_4B83);
    }

    #[test]
    fn deadbeef() {
        assert!(crc32_mpeg2(&[0xDE, 0xAD, 0xBE, 0xEF]) == 0x81DA_1A18);
    }

    #[test]
    fn word_12345678() {
        assert!(crc32_mpeg2(&[0x12, 0x34, 0x56, 0x78]) == 0xDF8A_8A2B);
    }

    #[test]
    fn pattern_sixteen_bytes() {
        let data = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD,
            0xEE, 0xFF,
        ];
        assert!(crc32_mpeg2(&data) == 0x5448_9AA0);
    }

    #[test]
    fn sixteen_zero_bytes() {
        let data = [0x00_u8; 16];
        assert!(crc32_mpeg2(&data) == 0x552D_22C8);
    }

    #[test]
    fn sixteen_ff_bytes() {
        let data = [0xFF_u8; 16];
        assert!(crc32_mpeg2(&data) == 0xA79C_3203);
    }

    #[test]
    fn hundred_a_bytes() {
        let data = [0x41_u8; 100];
        assert!(crc32_mpeg2(&data) == 0xDDB3_927E);
    }

    // --- Generated 256-byte ramp (0x00..=0xFF). ---
    #[test]
    fn ramp_256_bytes() {
        let mut buf = [0_u8; 256];
        let mut i = 0_usize;
        while i < buf.len() {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc32_mpeg2(&buf) == 0x494A_116A);
    }

    // --- Long inputs: stability of a fixed, known value. ---
    #[test]
    fn long_thousand_aa() {
        let data = [0xAA_u8; 1000];
        assert!(crc32_mpeg2(&data) == 0xF72F_3C25);
    }

    #[test]
    fn long_1024_ramp() {
        let mut buf = [0_u8; 1024];
        let mut i = 0_usize;
        while i < buf.len() {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc32_mpeg2(&buf) == 0x1A5C_3E13);
    }

    // --- Determinism: identical input yields identical output. ---
    #[test]
    fn deterministic_empty() {
        assert!(crc32_mpeg2(b"") == crc32_mpeg2(b""));
    }

    #[test]
    fn deterministic_check() {
        let first = crc32_mpeg2(b"123456789");
        let second = crc32_mpeg2(b"123456789");
        assert!(first == second);
    }

    #[test]
    fn deterministic_single_byte() {
        assert!(crc32_mpeg2(&[0x5A]) == crc32_mpeg2(&[0x5A]));
    }

    #[test]
    fn deterministic_long_thousand() {
        let data = [0xAA_u8; 1000];
        let first = crc32_mpeg2(&data);
        let second = crc32_mpeg2(&data);
        assert!(first == second);
    }

    #[test]
    fn deterministic_long_ramp() {
        let mut buf = [0_u8; 1024];
        let mut i = 0_usize;
        while i < buf.len() {
            buf[i] = i as u8;
            i += 1;
        }
        let first = crc32_mpeg2(&buf);
        let second = crc32_mpeg2(&buf);
        assert!(first == second);
    }

    // --- Distinctness: different inputs produce different checksums. ---
    #[test]
    fn distinct_a_vs_b() {
        assert!(crc32_mpeg2(b"a") != crc32_mpeg2(b"b"));
    }

    #[test]
    fn distinct_zero_vs_ff() {
        assert!(crc32_mpeg2(&[0x00]) != crc32_mpeg2(&[0xFF]));
    }

    #[test]
    fn distinct_order_matters() {
        assert!(crc32_mpeg2(&[0xFF, 0x00]) != crc32_mpeg2(&[0x00, 0xFF]));
    }

    #[test]
    fn distinct_empty_vs_zero() {
        assert!(crc32_mpeg2(b"") != crc32_mpeg2(&[0x00]));
    }

    // --- Range sanity for a selection of confirmed anchors. ---
    #[test]
    fn range_contains_check_value() {
        let value = crc32_mpeg2(b"123456789");
        assert!((0x0000_0000..=0xFFFF_FFFF_u32).contains(&value));
        assert!(value == 0x0376_E6E7);
    }

    #[test]
    fn length_prefix_changes_result() {
        let short = crc32_mpeg2(b"abc");
        let long = crc32_mpeg2(b"abcd");
        assert!(short != long);
    }
}
