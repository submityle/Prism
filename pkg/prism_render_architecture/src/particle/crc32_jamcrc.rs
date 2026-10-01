//! `CRC`-32/JAMCRC checksum computed over a byte slice.
//!
//! Parameters: width = 32, `poly` = `0x04C11DB7`, `init` = `0xFFFFFFFF`,
//! `refin` = true, `refout` = true, `xorout` = `0x00000000`.
//! The implementation is `LSB`-first using the reflected polynomial
//! `REFPOLY` = `0xEDB88320`, so the final `XOR` with zero is a no-op and the
//! raw register value is returned directly as a `u32`.
//! check value: `crc32_jamcrc(b"123456789")` equals `0x340BC6D9`.

/// Reflected form of the `CRC`-32 polynomial `0x04C11DB7`.
const REFPOLY: u32 = 0xEDB8_8320;

/// Compute the `CRC`-32/JAMCRC checksum of `data`.
///
/// Processes each byte `LSB`-first, folding in `REFPOLY` on set bits. The
/// `xorout` parameter is `0x00000000`, so no final `XOR` is applied.
pub fn crc32_jamcrc(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= b as u32;
        let mut i = 0;
        while i < 8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ REFPOLY;
            } else {
                crc >>= 1;
            }
            i += 1;
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Four required anchors -------------------------------------------

    #[test]
    fn anchor_empty() {
        assert!(crc32_jamcrc(b"") == 0xFFFF_FFFF);
    }

    #[test]
    fn anchor_single_zero() {
        assert!(crc32_jamcrc(&[0x00]) == 0x2DFD_1072);
    }

    #[test]
    fn anchor_single_ff() {
        assert!(crc32_jamcrc(&[0xFF]) == 0x00FF_FFFF);
    }

    #[test]
    fn anchor_check_vector() {
        assert!(crc32_jamcrc(b"123456789") == 0x340B_C6D9);
    }

    // --- Hardcoded multi-byte / single-byte vectors ----------------------

    #[test]
    fn vector_upper_a() {
        assert!(crc32_jamcrc(b"A") == 0x2C26_6174);
    }

    #[test]
    fn vector_lower_a() {
        assert!(crc32_jamcrc(b"a") == 0x1748_41BC);
    }

    #[test]
    fn vector_abc() {
        assert!(crc32_jamcrc(b"abc") == 0xCADB_BE3D);
    }

    #[test]
    fn vector_hello_world() {
        assert!(crc32_jamcrc(b"Hello, world!") == 0x1419_3919);
    }

    #[test]
    fn vector_quick_brown_fox() {
        let data = b"The quick brown fox jumps over the lazy dog";
        assert!(crc32_jamcrc(data) == 0xBEB0_5CC6);
    }

    #[test]
    fn vector_incrementing_five() {
        assert!(crc32_jamcrc(&[0x00, 0x01, 0x02, 0x03, 0x04]) == 0xAEA5_2C33);
    }

    #[test]
    fn vector_two_ff() {
        assert!(crc32_jamcrc(&[0xFF, 0xFF]) == 0x0000_FFFF);
    }

    #[test]
    fn vector_two_zero() {
        assert!(crc32_jamcrc(&[0x00, 0x00]) == 0xBE26_ED00);
    }

    #[test]
    fn vector_deadbeef() {
        assert!(crc32_jamcrc(&[0xDE, 0xAD, 0xBE, 0xEF]) == 0x8363_5CA5);
    }

    #[test]
    fn vector_alternating_zero_ff() {
        assert!(crc32_jamcrc(&[0x00, 0xFF, 0x00, 0xFF]) == 0x4D21_FB83);
    }

    #[test]
    fn vector_prism_lower() {
        assert!(crc32_jamcrc(b"prism") == 0x712E_4745);
    }

    #[test]
    fn vector_prism_title() {
        assert!(crc32_jamcrc(b"Prism") == 0xB0EF_6841);
    }

    #[test]
    fn vector_one_char() {
        assert!(crc32_jamcrc(b"1") == 0x7C23_1048);
    }

    #[test]
    fn vector_two_chars() {
        assert!(crc32_jamcrc(b"12") == 0xB0AC_BB32);
    }

    #[test]
    fn vector_all_byte_values() {
        let mut data = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            data[i] = (i & 0xFF) as u8;
            i += 1;
        }
        assert!(crc32_jamcrc(&data) == 0xD6FA_738C);
    }

    #[test]
    fn vector_hundred_z() {
        let data = [b'z'; 100];
        assert!(crc32_jamcrc(&data) == 0x5250_4D5A);
    }

    #[test]
    fn vector_thousand_zero() {
        let data = [0u8; 1000];
        assert!(crc32_jamcrc(&data) == 0xF9F4_E87F);
    }

    #[test]
    fn vector_four_full_cycles() {
        let mut data = [0u8; 1024];
        let mut i = 0usize;
        while i < 1024 {
            data[i] = (i & 0xFF) as u8;
            i += 1;
        }
        assert!(crc32_jamcrc(&data) == 0x48F4_B3D9);
    }

    // --- Determinism -----------------------------------------------------

    #[test]
    fn deterministic_empty() {
        assert!(crc32_jamcrc(b"") == crc32_jamcrc(b""));
    }

    #[test]
    fn deterministic_check_vector() {
        let first = crc32_jamcrc(b"123456789");
        let second = crc32_jamcrc(b"123456789");
        assert!(first == second);
    }

    #[test]
    fn deterministic_multibyte() {
        let data = b"The quick brown fox jumps over the lazy dog";
        assert!(crc32_jamcrc(data) == crc32_jamcrc(data));
    }

    #[test]
    fn deterministic_binary_slice() {
        let data = [0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0xFF];
        assert!(crc32_jamcrc(&data) == crc32_jamcrc(&data));
    }

    // --- Long-input stability -------------------------------------------

    #[test]
    fn long_input_zero_stable() {
        let data = [0u8; 4096];
        let first = crc32_jamcrc(&data);
        let second = crc32_jamcrc(&data);
        assert!(first == second);
    }

    #[test]
    fn long_input_pattern_stable() {
        let mut data = [0u8; 8192];
        let mut i = 0usize;
        while i < 8192 {
            data[i] = (i & 0xFF) as u8;
            i += 1;
        }
        let first = crc32_jamcrc(&data);
        let second = crc32_jamcrc(&data);
        assert!(first == second);
    }

    #[test]
    fn long_input_z_fill_stable() {
        let data = [b'z'; 10000];
        let first = crc32_jamcrc(&data);
        let second = crc32_jamcrc(&data);
        assert!(first == second);
    }

    // --- Structural / property checks -----------------------------------

    #[test]
    fn empty_differs_from_single_zero() {
        assert!(crc32_jamcrc(b"") != crc32_jamcrc(&[0x00]));
    }

    #[test]
    fn distinct_single_bytes_differ() {
        assert!(crc32_jamcrc(&[0x00]) != crc32_jamcrc(&[0xFF]));
    }

    #[test]
    fn byte_order_matters() {
        assert!(crc32_jamcrc(&[0x01, 0x02]) != crc32_jamcrc(&[0x02, 0x01]));
    }

    #[test]
    fn appending_byte_changes_result() {
        assert!(crc32_jamcrc(b"abc") != crc32_jamcrc(b"abcd"));
    }

    #[test]
    fn prefix_differs_from_full() {
        assert!(crc32_jamcrc(b"12345") != crc32_jamcrc(b"123456789"));
    }

    #[test]
    fn different_length_zero_runs_differ() {
        let one = [0u8; 1];
        let two = [0u8; 2];
        assert!(crc32_jamcrc(&one) != crc32_jamcrc(&two));
    }

    #[test]
    fn single_bit_flip_changes_result() {
        assert!(crc32_jamcrc(&[0x00]) != crc32_jamcrc(&[0x01]));
    }

    #[test]
    fn result_in_u32_range() {
        let value = crc32_jamcrc(b"range");
        assert!((u32::MIN..=u32::MAX).contains(&value));
    }

    #[test]
    fn case_sensitivity() {
        assert!(crc32_jamcrc(b"prism") != crc32_jamcrc(b"Prism"));
    }

    #[test]
    fn length_is_multiple_of_four_sanity() {
        let data = [0xAAu8; 16];
        assert!(data.len().is_multiple_of(4));
        assert!(crc32_jamcrc(&data) == crc32_jamcrc(&data));
    }
}
