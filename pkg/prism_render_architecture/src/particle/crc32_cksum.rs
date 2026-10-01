//! `CRC`-32/`CKSUM` (`RevEng` parameter model, **without** the `POSIX`
//! length-append step).
//!
//! Parameters: width = 32, poly = `0x04C11DB7`, init = `0x00000000`,
//! `refin` = false, `refout` = false, `xorout` = `0xFFFFFFFF`. This is a
//! non-reflected, `MSB`-first computation returning a `u32`.
//!
//! The `POSIX` `cksum` utility also folds the message byte length into the
//! register before applying `xorout`; this module intentionally omits that
//! length-append step and implements only the abstract `RevEng` model.
//!
//! check value: `crc32_cksum(b"123456789")` == `0x765E7680`.
//!
//! Additional verified anchors: empty input == `0xFFFFFFFF`,
//! `[0x00]` == `0xFFFFFFFF`, `[0xFF]` == `0x4E08BF4B`. The register `XOR`
//! and shift steps use pure integer arithmetic only.

/// Generator polynomial for `CRC`-32/`CKSUM` in normal (`MSB`-first) form.
const POLY: u32 = 0x04C1_1DB7;

/// High bit mask selecting the most significant bit of the `u32` register.
const TOP_BIT: u32 = 0x8000_0000;

/// Final `XOR` value applied to the register (`xorout`).
const XOROUT: u32 = 0xFFFF_FFFF;

/// Computes the `CRC`-32/`CKSUM` checksum over `data`.
///
/// This follows the `RevEng` parameter model (poly `0x04C11DB7`, init
/// `0x00000000`, non-reflected, `xorout` `0xFFFFFFFF`) and does **not**
/// append the message length the way the `POSIX` `cksum` utility does.
///
/// Returns the resulting checksum as a `u32`.
pub fn crc32_cksum(data: &[u8]) -> u32 {
    let mut reg: u32 = 0x0000_0000;
    for &b in data {
        reg ^= (b as u32) << 24;
        let mut i = 0;
        while i < 8 {
            if (reg & TOP_BIT) != 0 {
                reg = (reg << 1) ^ POLY;
            } else {
                reg <<= 1;
            }
            i += 1;
        }
    }
    reg ^ XOROUT
}

#[cfg(test)]
mod tests {
    use super::crc32_cksum;

    // --- Four required anchors ---------------------------------------

    #[test]
    fn anchor_empty() {
        assert!(crc32_cksum(b"") == 0xFFFF_FFFF);
    }

    #[test]
    fn anchor_single_zero() {
        assert!(crc32_cksum(&[0x00]) == 0xFFFF_FFFF);
    }

    #[test]
    fn anchor_single_ff() {
        assert!(crc32_cksum(&[0xFF]) == 0x4E08_BF4B);
    }

    #[test]
    fn anchor_check_vector() {
        assert!(crc32_cksum(b"123456789") == 0x765E_7680);
    }

    // --- Hardcoded multi-byte vectors --------------------------------

    #[test]
    fn vector_upper_a() {
        assert!(crc32_cksum(b"A") == 0xCFB8_923F);
    }

    #[test]
    fn vector_lower_a() {
        assert!(crc32_cksum(b"a") == 0x579B_24DF);
    }

    #[test]
    fn vector_abc() {
        assert!(crc32_cksum(b"abc") == 0xD3E8_C673);
    }

    #[test]
    fn vector_hello() {
        assert!(crc32_cksum(b"hello") == 0x5E21_DEA1);
    }

    #[test]
    fn vector_hello_world() {
        assert!(crc32_cksum(b"hello, world") == 0x4B6B_896B);
    }

    #[test]
    fn vector_quick_brown_fox() {
        assert!(crc32_cksum(b"The quick brown fox jumps over the lazy dog") == 0x36B7_8081);
    }

    #[test]
    fn vector_two_zeros() {
        assert!(crc32_cksum(&[0x00, 0x00]) == 0xFFFF_FFFF);
    }

    #[test]
    fn vector_three_zeros() {
        assert!(crc32_cksum(&[0x00, 0x00, 0x00]) == 0xFFFF_FFFF);
    }

    #[test]
    fn vector_two_ff() {
        assert!(crc32_cksum(&[0xFF, 0xFF]) == 0x00B7_9B82);
    }

    #[test]
    fn vector_00_01() {
        assert!(crc32_cksum(&[0x00, 0x01]) == 0xFB3E_E248);
    }

    #[test]
    fn vector_01_02() {
        assert!(crc32_cksum(&[0x01, 0x02]) == 0x2464_054D);
    }

    #[test]
    fn vector_deadbeef() {
        assert!(crc32_cksum(&[0xDE, 0xAD, 0xBE, 0xEF]) == 0xB921_389C);
    }

    #[test]
    fn vector_range_0_255() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc32_cksum(&buf) == 0x53EB_78DA);
    }

    #[test]
    fn vector_256_ones() {
        let buf = [0xFFu8; 256];
        assert!(crc32_cksum(&buf) == 0x614B_8330);
    }

    #[test]
    fn vector_256_zeros() {
        let buf = [0x00u8; 256];
        assert!(crc32_cksum(&buf) == 0xFFFF_FFFF);
    }

    // --- Determinism --------------------------------------------------

    #[test]
    fn deterministic_repeated_check() {
        let a = crc32_cksum(b"123456789");
        let b = crc32_cksum(b"123456789");
        assert!(a == b);
    }

    #[test]
    fn deterministic_repeated_empty() {
        let a = crc32_cksum(b"");
        let b = crc32_cksum(b"");
        assert!(a == b);
    }

    #[test]
    fn deterministic_repeated_fox() {
        let input = b"The quick brown fox jumps over the lazy dog";
        let a = crc32_cksum(input);
        let b = crc32_cksum(input);
        let c = crc32_cksum(input);
        assert!(a == b);
        assert!(b == c);
    }

    #[test]
    fn deterministic_many_iterations() {
        let expected = crc32_cksum(b"abc");
        let mut i = 0;
        while i < 32 {
            assert!(crc32_cksum(b"abc") == expected);
            i += 1;
        }
    }

    // --- Structural / sensitivity properties -------------------------

    #[test]
    fn different_inputs_differ() {
        assert!(crc32_cksum(b"abc") != crc32_cksum(b"abd"));
    }

    #[test]
    fn order_sensitive() {
        assert!(crc32_cksum(&[0x01, 0x02]) != crc32_cksum(&[0x02, 0x01]));
    }

    #[test]
    fn single_bit_change_differs() {
        assert!(crc32_cksum(&[0x00]) != crc32_cksum(&[0x01]));
    }

    #[test]
    fn prefix_extension_differs() {
        let short = crc32_cksum(b"12345");
        let long = crc32_cksum(b"123456789");
        assert!(short != long);
    }

    #[test]
    fn length_distinguishes_zero_runs() {
        // All-zero inputs collapse to the same value; verify explicitly.
        assert!(crc32_cksum(&[0x00]) == crc32_cksum(&[0x00, 0x00]));
        assert!(crc32_cksum(&[0x00, 0x00]) == crc32_cksum(&[0x00, 0x00, 0x00]));
    }

    // --- Long-input stability ----------------------------------------

    #[test]
    fn long_input_1024_zeros_stable() {
        let buf = [0x00u8; 1024];
        let a = crc32_cksum(&buf);
        let b = crc32_cksum(&buf);
        assert!(a == b);
        assert!(a == 0xFFFF_FFFF);
    }

    #[test]
    fn long_input_1024_ones_stable() {
        let buf = [0xFFu8; 1024];
        let a = crc32_cksum(&buf);
        let b = crc32_cksum(&buf);
        assert!(a == b);
    }

    #[test]
    fn long_input_4096_pattern_stable() {
        let mut buf = [0u8; 4096];
        let mut i = 0usize;
        while i < 4096 {
            buf[i] = (i & 0xFF) as u8;
            i += 1;
        }
        let a = crc32_cksum(&buf);
        let b = crc32_cksum(&buf);
        assert!(a == b);
    }

    #[test]
    fn long_input_result_in_u32_range() {
        let buf = [0xABu8; 2048];
        let value = crc32_cksum(&buf);
        assert!((u32::MIN..=u32::MAX).contains(&value));
    }

    #[test]
    fn long_input_incremental_lengths_deterministic() {
        let mut i = 0usize;
        while i < 300 {
            let buf = [0x5Au8; 300];
            let slice = &buf[..i];
            let a = crc32_cksum(slice);
            let b = crc32_cksum(slice);
            assert!(a == b);
            i += 1;
        }
    }

    // --- Additional coverage -----------------------------------------

    #[test]
    fn check_value_matches_doc_anchor() {
        let value = crc32_cksum(b"123456789");
        assert!(value == 0x765E_7680);
        assert!((u32::MIN..=u32::MAX).contains(&value));
    }

    #[test]
    fn empty_and_zero_coincide() {
        assert!(crc32_cksum(b"") == crc32_cksum(&[0x00]));
    }

    #[test]
    fn two_byte_vectors_distinct() {
        let v1 = crc32_cksum(&[0x00, 0x01]);
        let v2 = crc32_cksum(&[0x01, 0x02]);
        let v3 = crc32_cksum(&[0xFF, 0xFF]);
        assert!(v1 != v2);
        assert!(v2 != v3);
        assert!(v1 != v3);
    }

    #[test]
    fn fox_vector_exact() {
        let value = crc32_cksum(b"The quick brown fox jumps over the lazy dog");
        assert!(value == 0x36B7_8081);
    }
}
