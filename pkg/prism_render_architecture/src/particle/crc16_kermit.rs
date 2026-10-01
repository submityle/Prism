//! `CRC-16/KERMIT` (`CRC-16/CCITT`, reflected variant) over byte slices.
//!
//! Parameters: `width = 16`, `poly = 0x1021`, `init = 0x0000`,
//! `refin = true`, `refout = true`, `xorout = 0x0000`.
//!
//! Because `refin = refout = true`, this uses the reflected `LSB`-first
//! form with the reflected polynomial `bitreverse(0x1021, 16) = 0x8408`.
//! The implementation is `no_std` + `alloc`, pure integer, and avoids any
//! floating-point or transcendental operations.

/// Reflected polynomial: `bitreverse(0x1021, 16)`.
const REFPOLY: u16 = 0x8408;

/// Computes the `CRC-16/KERMIT` checksum of `data`.
///
/// Uses the `LSB`-first reflected form. The `init` value `0x0000` is
/// unchanged by reflection, and `xorout` is `0x0000`, so no final fixup is
/// required beyond the reflected processing loop itself.
pub fn crc16_kermit(data: &[u8]) -> u16 {
    let mut crc: u16 = 0x0000;
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
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- 5 ground-truth anchor vectors -----

    #[test]
    fn anchor_empty() {
        assert!(crc16_kermit(b"") == 0x0000);
    }

    #[test]
    fn anchor_single_a() {
        assert!(crc16_kermit(b"a") == 0x728f);
    }

    #[test]
    fn anchor_zero_byte() {
        assert!(crc16_kermit(&[0x00]) == 0x0000);
    }

    #[test]
    fn anchor_ff_byte() {
        assert!(crc16_kermit(&[0xff]) == 0x0f78);
    }

    #[test]
    fn anchor_check_value() {
        assert!(crc16_kermit(b"123456789") == 0x2189);
    }

    // ----- exact-value vectors (computed from this implementation) -----

    #[test]
    fn vec_ab() {
        assert!(crc16_kermit(b"ab") == 0x3c99);
    }

    #[test]
    fn vec_abc() {
        assert!(crc16_kermit(b"abc") == 0x58e9);
    }

    #[test]
    fn vec_abcd() {
        assert!(crc16_kermit(b"abcd") == 0x5fb5);
    }

    #[test]
    fn vec_message_digest() {
        assert!(crc16_kermit(b"message digest") == 0x7bd5);
    }

    #[test]
    fn vec_quick_brown_fox() {
        assert!(crc16_kermit(b"The quick brown fox") == 0x8550);
    }

    #[test]
    fn vec_byte_01() {
        assert!(crc16_kermit(&[0x01]) == 0x1189);
    }

    #[test]
    fn vec_byte_80() {
        assert!(crc16_kermit(&[0x80]) == 0x8408);
    }

    #[test]
    fn vec_byte_55() {
        assert!(crc16_kermit(&[0x55]) == 0x0528);
    }

    #[test]
    fn vec_byte_aa() {
        assert!(crc16_kermit(&[0xaa]) == 0x0a50);
    }

    #[test]
    fn vec_seq_0_7() {
        assert!(crc16_kermit(&[0, 1, 2, 3, 4, 5, 6, 7]) == 0xe171);
    }

    #[test]
    fn vec_seq_0_15() {
        let data: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        assert!(crc16_kermit(&data) == 0xbc40);
    }

    #[test]
    fn vec_char_0() {
        assert!(crc16_kermit(b"0") == 0x3183);
    }

    #[test]
    fn vec_char_00() {
        assert!(crc16_kermit(b"00") == 0x8721);
    }

    #[test]
    fn vec_char_000() {
        assert!(crc16_kermit(b"000") == 0x018f);
    }

    #[test]
    fn vec_hello() {
        assert!(crc16_kermit(b"hello") == 0xfbca);
    }

    #[test]
    fn vec_hello_world() {
        assert!(crc16_kermit(b"Hello, World!") == 0x543e);
    }

    #[test]
    fn vec_two_ff() {
        assert!(crc16_kermit(&[0xff, 0xff]) == 0xf0b8);
    }

    #[test]
    fn vec_two_zero() {
        assert!(crc16_kermit(&[0x00, 0x00]) == 0x0000);
    }

    #[test]
    fn vec_char_a_upper() {
        assert!(crc16_kermit(b"A") == 0x538d);
    }

    #[test]
    fn vec_ab_upper() {
        assert!(crc16_kermit(b"AB") == 0x3ea8);
    }

    #[test]
    fn vec_prism() {
        assert!(crc16_kermit(b"Prism") == 0xed4d);
    }

    #[test]
    fn vec_particle() {
        assert!(crc16_kermit(b"particle") == 0x346d);
    }

    #[test]
    fn vec_crc16() {
        assert!(crc16_kermit(b"crc16") == 0xceaa);
    }

    #[test]
    fn vec_kermit() {
        assert!(crc16_kermit(b"kermit") == 0x339c);
    }

    #[test]
    fn vec_deadbeef() {
        assert!(crc16_kermit(&[0xde, 0xad, 0xbe, 0xef]) == 0x1915);
    }

    #[test]
    fn vec_mixed_eight() {
        let data: [u8; 8] = [0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0];
        assert!(crc16_kermit(&data) == 0xd147);
    }

    // ----- incremental / structural properties -----

    #[test]
    fn incremental_ab_extends_a() {
        // "ab" differs from "a" only by appending 'b'; both are fixed anchors.
        assert!(crc16_kermit(b"a") == 0x728f);
        assert!(crc16_kermit(b"ab") == 0x3c99);
    }

    #[test]
    fn incremental_abc_extends_ab() {
        assert!(crc16_kermit(b"ab") == 0x3c99);
        assert!(crc16_kermit(b"abc") == 0x58e9);
    }

    #[test]
    fn determinism_repeated_calls() {
        let a = crc16_kermit(b"message digest");
        let b = crc16_kermit(b"message digest");
        assert!(a == b);
    }

    #[test]
    fn determinism_slice_vs_array() {
        let arr: [u8; 3] = [b'a', b'b', b'c'];
        assert!(crc16_kermit(&arr) == crc16_kermit(b"abc"));
    }

    #[test]
    fn boundary_many_zeros_is_zero() {
        // All-zero input keeps the register at the reflected init (zero).
        let zeros: [u8; 32] = [0u8; 32];
        assert!(crc16_kermit(&zeros) == 0x0000);
    }

    #[test]
    fn boundary_result_fits_u16() {
        let value = crc16_kermit(b"The quick brown fox");
        assert!((0x0000..=0xffff).contains(&value));
    }

    #[test]
    fn boundary_single_byte_differs() {
        assert!(crc16_kermit(&[0x00]) != crc16_kermit(&[0x01]));
    }

    #[test]
    fn property_order_matters() {
        assert!(crc16_kermit(&[0x01, 0x02]) != crc16_kermit(&[0x02, 0x01]));
    }

    #[test]
    fn property_length_affects_zero_padding() {
        assert!(crc16_kermit(&[0x00]) == crc16_kermit(&[0x00, 0x00]));
    }
}
