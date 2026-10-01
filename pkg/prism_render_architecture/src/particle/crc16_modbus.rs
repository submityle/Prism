//! CRC-16/MODBUS checksum over byte slices for the particle contract crate.
//!
//! This module implements the catalog `CRC-16/MODBUS` algorithm using the
//! reflected (LSB-first) bit-by-bit form. Because the algorithm specifies
//! `refin = refout = true`, the reflected polynomial `0xA001` (the bit-reverse
//! of `0x8005` over 16 bits) is used directly, the init value `0xFFFF` is fed
//! in reflected form, and the final `xorout` of `0x0000` is a no-op.
//!
//! The public API is `no_std` and `alloc` friendly: it borrows a `&[u8]` and
//! returns a `u16`, performing only pure integer arithmetic (no floating point
//! and no transcendental operations).

/// Reflected form of the `CRC-16/MODBUS` polynomial `0x8005`.
const REFPOLY: u16 = 0xA001;

/// Reflected initial value for `CRC-16/MODBUS`.
const INIT: u16 = 0xffff;

/// Number of bits processed per input byte.
const BITS_PER_BYTE: u32 = 8;

/// Computes the `CRC-16/MODBUS` checksum of `data`.
///
/// Parameters of the algorithm: width = 16, polynomial = `0x8005`,
/// init = `0xFFFF`, `refin` = true, `refout` = true, `xorout` = `0x0000`.
///
/// The reflected (LSB-first) algorithm is used, so the reflected polynomial
/// `0xA001` is applied and no explicit output reflection is required.
pub fn crc16_modbus(data: &[u8]) -> u16 {
    let mut crc: u16 = INIT;
    for &b in data {
        crc ^= b as u16;
        for _ in 0..BITS_PER_BYTE {
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
    use super::crc16_modbus;

    // ---- The 5 external anchor vectors (ground truth) ----

    #[test]
    fn anchor_empty_is_init() {
        assert!(crc16_modbus(b"") == 0xffff);
    }

    #[test]
    fn anchor_lowercase_a() {
        assert!(crc16_modbus(b"a") == 0xa87e);
    }

    #[test]
    fn anchor_single_zero_byte() {
        assert!(crc16_modbus(&[0x00]) == 0x40bf);
    }

    #[test]
    fn anchor_single_ff_byte() {
        assert!(crc16_modbus(&[0xff]) == 0x00ff);
    }

    #[test]
    fn anchor_catalog_check_value() {
        assert!(crc16_modbus(b"123456789") == 0x4b37);
    }

    // ---- Self-derived exact-value vectors (computed with this algorithm) ----

    #[test]
    fn vector_ab() {
        assert!(crc16_modbus(b"ab") == 0xc9a9);
    }

    #[test]
    fn vector_ba() {
        assert!(crc16_modbus(b"ba") == 0x38e9);
    }

    #[test]
    fn vector_abc() {
        assert!(crc16_modbus(b"abc") == 0x5749);
    }

    #[test]
    fn vector_hello() {
        assert!(crc16_modbus(b"hello") == 0x34f6);
    }

    #[test]
    fn vector_world() {
        assert!(crc16_modbus(b"world") == 0xef41);
    }

    #[test]
    fn vector_two_ff_bytes() {
        assert!(crc16_modbus(&[0xff, 0xff]) == 0x0000);
    }

    #[test]
    fn vector_two_zero_bytes() {
        assert!(crc16_modbus(&[0x00, 0x00]) == 0xb001);
    }

    #[test]
    fn vector_one_two_three() {
        assert!(crc16_modbus(&[0x01, 0x02, 0x03]) == 0x6161);
    }

    #[test]
    fn vector_three_two_one() {
        assert!(crc16_modbus(&[0x03, 0x02, 0x01]) == 0x6041);
    }

    #[test]
    fn vector_sequential_five() {
        assert!(crc16_modbus(&[0x01, 0x02, 0x03, 0x04, 0x05]) == 0xbb2a);
    }

    #[test]
    fn vector_double_uppercase_a() {
        assert!(crc16_modbus(&[0x41, 0x41]) == 0xd0f1);
    }

    #[test]
    fn vector_modbus_word() {
        assert!(crc16_modbus(b"Modbus") == 0x5402);
    }

    #[test]
    fn vector_deadbeef() {
        assert!(crc16_modbus(&[0xde, 0xad, 0xbe, 0xef]) == 0xc19b);
    }

    #[test]
    fn vector_bytes_zero_to_fifteen() {
        let data: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        assert!(crc16_modbus(&data) == 0xe7b4);
    }

    #[test]
    fn vector_prism_word() {
        assert!(crc16_modbus(b"Prism") == 0xb5cb);
    }

    #[test]
    fn vector_double_z() {
        assert!(crc16_modbus(&[0x7a, 0x7a]) == 0x33a3);
    }

    #[test]
    fn vector_single_at_sign() {
        assert!(crc16_modbus(&[0x40]) == 0xb0be);
    }

    #[test]
    fn vector_single_one() {
        assert!(crc16_modbus(&[0x01]) == 0x807e);
    }

    #[test]
    fn vector_single_high_bit() {
        assert!(crc16_modbus(&[0x80]) == 0xe0be);
    }

    #[test]
    fn vector_ten_ff_bytes() {
        let data: [u8; 10] = [0xff; 10];
        assert!(crc16_modbus(&data) == 0x8441);
    }

    #[test]
    fn vector_ten_zero_bytes() {
        let data: [u8; 10] = [0x00; 10];
        assert!(crc16_modbus(&data) == 0x0770);
    }

    // ---- Determinism properties ----

    #[test]
    fn determinism_repeated_empty() {
        assert!(crc16_modbus(b"") == crc16_modbus(b""));
    }

    #[test]
    fn determinism_repeated_check_vector() {
        assert!(crc16_modbus(b"123456789") == crc16_modbus(b"123456789"));
    }

    #[test]
    fn determinism_repeated_binary() {
        let data: [u8; 4] = [0xde, 0xad, 0xbe, 0xef];
        assert!(crc16_modbus(&data) == crc16_modbus(&data));
    }

    #[test]
    fn determinism_slice_and_array_match() {
        let data: [u8; 3] = [0x01, 0x02, 0x03];
        let slice: &[u8] = &data;
        assert!(crc16_modbus(&data) == crc16_modbus(slice));
    }

    // ---- Order sensitivity (permutations differ) ----

    #[test]
    fn order_sensitive_ab_vs_ba() {
        assert!(crc16_modbus(b"ab") != crc16_modbus(b"ba"));
    }

    #[test]
    fn order_sensitive_123_vs_321() {
        assert!(crc16_modbus(&[0x01, 0x02, 0x03]) != crc16_modbus(&[0x03, 0x02, 0x01]));
    }

    #[test]
    fn order_sensitive_hello_vs_world() {
        assert!(crc16_modbus(b"hello") != crc16_modbus(b"world"));
    }

    // ---- Length / content distinctions ----

    #[test]
    fn empty_differs_from_single_zero() {
        assert!(crc16_modbus(b"") != crc16_modbus(&[0x00]));
    }

    #[test]
    fn single_vs_double_zero_differ() {
        assert!(crc16_modbus(&[0x00]) != crc16_modbus(&[0x00, 0x00]));
    }

    #[test]
    fn single_zero_vs_single_ff_differ() {
        assert!(crc16_modbus(&[0x00]) != crc16_modbus(&[0xff]));
    }

    #[test]
    fn appending_byte_changes_result() {
        assert!(crc16_modbus(b"ab") != crc16_modbus(b"abc"));
    }

    #[test]
    fn two_ff_bytes_cancel_to_zero() {
        // Property anchor: the reflected init plus two 0xFF bytes yields 0x0000.
        assert!(crc16_modbus(&[0xff, 0xff]) == 0);
    }

    #[test]
    fn result_fits_in_u16_range() {
        let value = crc16_modbus(b"123456789");
        assert!((0x0000..=0xffff).contains(&value));
    }

    #[test]
    fn single_byte_matches_inline_recompute() {
        // Independent inline recomputation of the LSB-first form for one byte.
        let mut crc: u16 = 0xffff;
        crc ^= 0xa5_u16;
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0xa001;
            } else {
                crc >>= 1;
            }
        }
        assert!(crc16_modbus(&[0xa5]) == crc);
    }
}
