//! `CRC-16/USB` golden reference (reflected, `LSB`-first).
//!
//! This module implements the `CRC-16/USB` checksum using the reflected
//! (`LSB`-first) bit-at-a-time algorithm. Catalog parameters:
//! `width` = 16, `poly` = `0x8005`, `init` = `0xFFFF`, `refin` = `true`,
//! `refout` = `true`, `xorout` = `0xFFFF`.
//!
//! Because `refin` and `refout` are both `true`, the implementation shifts
//! right and uses the bit-reversed polynomial `REFPOLY` = `0xA001`, which is
//! the 16-bit reversal of `0x8005`. Each input byte is `XORed` into the low
//! byte of the running register before eight reduction steps.
//!
//! The directory `check` value (the result for the `ASCII` string
//! `"123456789"`) is `0xb4c8`.
//!
//! The computation is pure integer arithmetic: only shifts, masks, and the
//! bitwise `XOR`/`AND` operators are used, with no floating point and no
//! transcendental functions.

/// Compute the `CRC-16/USB` checksum of `data`.
///
/// Returns the final 16-bit `u16` checksum after applying the `xorout`
/// value `0xFFFF`.
#[must_use]
pub fn crc16_usb(data: &[u8]) -> u16 {
    const REFPOLY: u16 = 0xA001;
    let mut crc: u16 = 0xFFFF; // init (symmetric, needs no reflection)
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
    use super::crc16_usb;

    // ---- 5 directory anchors ----

    #[test]
    fn anchor_empty() {
        assert!(crc16_usb(b"") == 0x0000);
    }

    #[test]
    fn anchor_single_a() {
        assert!(crc16_usb(b"a") == 0x5781);
    }

    #[test]
    fn anchor_zero_byte() {
        assert!(crc16_usb(&[0x00]) == 0xbf40);
    }

    #[test]
    fn anchor_ff_byte() {
        assert!(crc16_usb(&[0xff]) == 0xff00);
    }

    #[test]
    fn anchor_check_vector() {
        assert!(crc16_usb(b"123456789") == 0xb4c8);
    }

    // ---- multi-byte self-consistent vectors (truth hard-coded) ----

    #[test]
    fn vec_abc() {
        assert!(crc16_usb(b"abc") == 0xa8b6);
    }

    #[test]
    fn vec_hello() {
        assert!(crc16_usb(b"hello") == 0xcb09);
    }

    #[test]
    fn vec_hello_world() {
        assert!(crc16_usb(b"Hello, World!") == 0xeeb1);
    }

    #[test]
    fn vec_quick_brown_fox() {
        assert!(crc16_usb(b"The quick brown fox") == 0xe5cf);
    }

    #[test]
    fn vec_sequential_four() {
        assert!(crc16_usb(&[0x01, 0x02, 0x03, 0x04]) == 0xd45e);
    }

    #[test]
    fn vec_deadbeef() {
        assert!(crc16_usb(&[0xde, 0xad, 0xbe, 0xef]) == 0x3e64);
    }

    #[test]
    fn vec_four_zeros() {
        assert!(crc16_usb(&[0x00, 0x00, 0x00, 0x00]) == 0xdbff);
    }

    #[test]
    fn vec_four_ff() {
        assert!(crc16_usb(&[0xff, 0xff, 0xff, 0xff]) == 0x4ffe);
    }

    #[test]
    fn vec_five_bytes() {
        assert!(crc16_usb(&[0x12, 0x34, 0x56, 0x78, 0x9a]) == 0xb72f);
    }

    #[test]
    fn vec_prism() {
        assert!(crc16_usb(b"prism") == 0x8db5);
    }

    #[test]
    fn vec_crc16usb_text() {
        assert!(crc16_usb(b"CRC-16/USB") == 0x4749);
    }

    #[test]
    fn vec_alternating_aa55() {
        assert!(crc16_usb(&[0xaa, 0x55, 0xaa, 0x55]) == 0x8c70);
    }

    #[test]
    fn vec_sixteen_zeros() {
        let data = [0x00u8; 16];
        assert!(crc16_usb(&data) == 0x0f41);
    }

    #[test]
    fn vec_sixteen_ff() {
        let data = [0xffu8; 16];
        assert!(crc16_usb(&data) == 0x7f01);
    }

    #[test]
    fn vec_thirtytwo_5a() {
        let data = [0x5au8; 32];
        assert!(crc16_usb(&data) == 0x82ee);
    }

    #[test]
    fn vec_hundred_ones() {
        let data = [0x01u8; 100];
        assert!(crc16_usb(&data) == 0x3fdd);
    }

    #[test]
    fn vec_255_ab() {
        let data = [0xabu8; 255];
        assert!(crc16_usb(&data) == 0x0b1b);
    }

    #[test]
    fn vec_ascii_digits() {
        assert!(crc16_usb(b"0123456789") == 0xbcb2);
    }

    #[test]
    fn vec_single_0x80() {
        assert!(crc16_usb(&[0x80]) == 0x1f41);
    }

    #[test]
    fn vec_single_0x01() {
        assert!(crc16_usb(&[0x01]) == 0x7f81);
    }

    #[test]
    fn vec_single_b() {
        assert!(crc16_usb(b"b") == 0x56c1);
    }

    #[test]
    fn vec_single_aa() {
        assert!(crc16_usb(&[0xaa]) == 0xc0c0);
    }

    #[test]
    fn vec_single_55() {
        assert!(crc16_usb(&[0x55]) == 0x8080);
    }

    #[test]
    fn vec_two_bytes_ab() {
        assert!(crc16_usb(b"ab") == 0x3656);
    }

    #[test]
    fn vec_four_bytes_abcd() {
        assert!(crc16_usb(b"abcd") == 0xe268);
    }

    #[test]
    fn vec_two_zeros() {
        assert!(crc16_usb(&[0x00, 0x00]) == 0x4ffe);
    }

    #[test]
    fn vec_one_two() {
        assert!(crc16_usb(&[0x01, 0x02]) == 0x1e7e);
    }

    #[test]
    fn vec_two_one() {
        assert!(crc16_usb(&[0x02, 0x01]) == 0xef3e);
    }

    #[test]
    fn vec_incrementing_256() {
        let mut data = [0u8; 256];
        let mut i: usize = 0;
        while i < 256 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc16_usb(&data) == 0x2193);
    }

    // ---- property / behavioral tests ----

    #[test]
    fn determinism_repeated_calls() {
        let data = b"123456789";
        let first = crc16_usb(data);
        let second = crc16_usb(data);
        let third = crc16_usb(data);
        assert!(first == second);
        assert!(second == third);
        assert!(first == 0xb4c8);
    }

    #[test]
    fn empty_equals_init_xor_xorout() {
        const INIT: u16 = 0xFFFF;
        const XOROUT: u16 = 0xFFFF;
        assert!(crc16_usb(b"") == (INIT ^ XOROUT));
    }

    #[test]
    fn distinct_inputs_distinct_values() {
        assert!(crc16_usb(b"a") != crc16_usb(b"b"));
        assert!(crc16_usb(b"abc") != crc16_usb(b"abcd"));
        assert!(crc16_usb(&[0x00]) != crc16_usb(&[0xff]));
    }

    #[test]
    fn byte_order_matters() {
        assert!(crc16_usb(&[0x01, 0x02]) != crc16_usb(&[0x02, 0x01]));
    }

    #[test]
    fn length_matters() {
        assert!(crc16_usb(&[0x00]) != crc16_usb(&[0x00, 0x00]));
    }

    #[test]
    fn long_zero_block_stable() {
        let data = [0x00u8; 1000];
        assert!(crc16_usb(&data) == 0xf4ab);
        assert!(crc16_usb(&data) == crc16_usb(&data));
    }

    #[test]
    fn prefix_changes_result() {
        let base = crc16_usb(b"message");
        let prefixed = crc16_usb(b"Xmessage");
        assert!(base != prefixed);
    }
}
