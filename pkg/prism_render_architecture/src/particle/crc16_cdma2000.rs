//! `CRC-16/CDMA2000` checksum over byte slices.
//!
//! Parameters: `width=16`, `poly=0xC867`, `init=0xFFFF`, `refin=false`,
//! `refout=false`, `xorout=0x0000`. The implementation uses the canonical
//! non-reflected `MSB`-first bit algorithm with a `u32` working register
//! masked to 16 bits, so results match the standard reference vectors
//! exactly. The directory `check` value for the input `b"123456789"` is
//! `0x4c06`.
//!
//! This module is `no_std` + `alloc` friendly: it uses only fixed-width
//! integers, no floating point, no transcendental functions, and no `Vec`,
//! `String`, or formatting macros in non-test code.

/// Width mask keeping the working register to 16 bits (`0xFFFF`).
const MASK: u32 = 0xFFFF;

/// Generator polynomial for `CRC-16/CDMA2000` (`0xC867`).
const POLY: u32 = 0xC867;

/// Computes the `CRC-16/CDMA2000` checksum of `data`.
///
/// The register starts at `0xFFFF` (`init`), bytes are processed `MSB`-first
/// (`refin=false`), and the final register is returned unchanged
/// (`refout=false`, `xorout=0x0000`).
pub fn crc16_cdma2000(data: &[u8]) -> u16 {
    let mut reg: u32 = 0xFFFF; // init
    for &byte in data {
        for i in 0..8u32 {
            let bit = ((byte >> (7 - i)) & 1) as u32; // refin=false => MSB-first
            let hi = (reg >> 15) & 1;
            let fb = hi ^ bit;
            reg = (reg << 1) & MASK;
            if fb != 0 {
                reg ^= POLY;
            }
        }
    }
    // refout=false, xorout=0 => return the register directly.
    (reg & MASK) as u16
}

#[cfg(test)]
mod tests {
    use super::crc16_cdma2000;

    // ---- anchor vectors (ground truth) ----

    #[test]
    fn anchor_empty() {
        assert!(crc16_cdma2000(b"") == 0xffff);
    }

    #[test]
    fn anchor_lower_a() {
        assert!(crc16_cdma2000(b"a") == 0x7f83);
    }

    #[test]
    fn anchor_zero_byte() {
        assert!(crc16_cdma2000(&[0x00]) == 0x6b6c);
    }

    #[test]
    fn anchor_ff_byte() {
        assert!(crc16_cdma2000(&[0xff]) == 0xff00);
    }

    #[test]
    fn anchor_check_value() {
        assert!(crc16_cdma2000(b"123456789") == 0x4c06);
    }

    // ---- self-consistent multi-byte vectors ----

    #[test]
    fn vec_ab() {
        assert!(crc16_cdma2000(b"ab") == 0xcd17);
    }

    #[test]
    fn vec_abc() {
        assert!(crc16_cdma2000(b"abc") == 0xf9c7);
    }

    #[test]
    fn vec_abcd() {
        assert!(crc16_cdma2000(b"abcd") == 0xd74d);
    }

    #[test]
    fn vec_message_digest() {
        assert!(crc16_cdma2000(b"message digest") == 0xf0f5);
    }

    #[test]
    fn vec_quick_fox() {
        assert!(crc16_cdma2000(b"The quick brown fox") == 0xc64a);
    }

    #[test]
    fn vec_quick_fox_full() {
        assert!(crc16_cdma2000(b"The quick brown fox jumps over the lazy dog") == 0x27a2);
    }

    #[test]
    fn vec_upper_a() {
        assert!(crc16_cdma2000(b"A") == 0x8c26);
    }

    #[test]
    fn vec_upper_ab() {
        assert!(crc16_cdma2000(b"AB") == 0x144f);
    }

    #[test]
    fn vec_hello() {
        assert!(crc16_cdma2000(b"hello") == 0x2d92);
    }

    #[test]
    fn vec_hello_world() {
        assert!(crc16_cdma2000(b"Hello, World!") == 0x1b05);
    }

    #[test]
    fn vec_digit_zero() {
        assert!(crc16_cdma2000(b"0") == 0x0528);
    }

    #[test]
    fn vec_digit_one() {
        assert!(crc16_cdma2000(b"1") == 0xcd4f);
    }

    #[test]
    fn vec_digit_nine() {
        assert!(crc16_cdma2000(b"9") == 0x678c);
    }

    #[test]
    fn vec_lower_z() {
        assert!(crc16_cdma2000(b"z") == 0xd86f);
    }

    #[test]
    fn vec_upper_z() {
        assert!(crc16_cdma2000(b"Z") == 0x2bca);
    }

    #[test]
    fn vec_space() {
        assert!(crc16_cdma2000(b" ") == 0x98c9);
    }

    #[test]
    fn vec_newline() {
        assert!(crc16_cdma2000(b"\n") == 0x9906);
    }

    #[test]
    fn vec_prism() {
        assert!(crc16_cdma2000(b"prism") == 0xc62f);
    }

    #[test]
    fn vec_particle() {
        assert!(crc16_cdma2000(b"particle") == 0x9b66);
    }

    #[test]
    fn vec_two_zero_bytes() {
        assert!(crc16_cdma2000(&[0x00, 0x00]) == 0x8a85);
    }

    #[test]
    fn vec_two_ff_bytes() {
        assert!(crc16_cdma2000(&[0xff, 0xff]) == 0x0000);
    }

    #[test]
    fn vec_byte_01() {
        assert!(crc16_cdma2000(&[0x01]) == 0xa30b);
    }

    #[test]
    fn vec_byte_02() {
        assert!(crc16_cdma2000(&[0x02]) == 0x33c5);
    }

    #[test]
    fn vec_byte_80() {
        assert!(crc16_cdma2000(&[0x80]) == 0x3536);
    }

    #[test]
    fn vec_byte_7f() {
        assert!(crc16_cdma2000(&[0x7f]) == 0xa15a);
    }

    #[test]
    fn vec_deadbeef() {
        assert!(crc16_cdma2000(&[0xde, 0xad, 0xbe, 0xef]) == 0xaeb8);
    }

    #[test]
    fn vec_counting_bytes() {
        assert!(crc16_cdma2000(&[0x00, 0x01, 0x02, 0x03]) == 0x1f4f);
    }

    #[test]
    fn vec_aa_55() {
        assert!(crc16_cdma2000(&[0xaa, 0x55]) == 0xedef);
    }

    #[test]
    fn vec_ten_digits() {
        assert!(crc16_cdma2000(b"0123456789") == 0xd89d);
    }

    // ---- determinism / incremental / boundary ----

    #[test]
    fn determinism_repeated_call() {
        let input: &[u8] = b"The quick brown fox";
        assert!(crc16_cdma2000(input) == crc16_cdma2000(input));
    }

    #[test]
    fn determinism_all_single_bytes() {
        let mut b: u16 = 0;
        while b <= 0xff {
            let one: [u8; 1] = [b as u8];
            assert!(crc16_cdma2000(&one) == crc16_cdma2000(&one));
            b += 1;
        }
    }

    #[test]
    fn empty_equals_init() {
        // init == 0xFFFF, and an empty slice performs no iterations.
        assert!(crc16_cdma2000(&[]) == 0xffff);
    }

    #[test]
    fn distinct_inputs_distinct_outputs() {
        // Known-different vectors must not collide.
        assert!(crc16_cdma2000(b"a") != crc16_cdma2000(b"ab"));
        assert!(crc16_cdma2000(b"abc") != crc16_cdma2000(b"abcd"));
        assert!(crc16_cdma2000(b"A") != crc16_cdma2000(b"a"));
    }

    #[test]
    fn order_sensitivity() {
        // Byte order matters for the checksum.
        assert!(crc16_cdma2000(&[0x01, 0x02]) != crc16_cdma2000(&[0x02, 0x01]));
    }

    #[test]
    fn result_fits_u16() {
        // The register is masked to 16 bits; the widened value stays in range.
        let value: u32 = crc16_cdma2000(b"particle") as u32;
        assert!(value <= 0xffff);
    }

    #[test]
    fn long_input_stable() {
        // A longer repeated-pattern input stays deterministic across calls.
        let block: [u8; 16] = [0xa5; 16];
        assert!(crc16_cdma2000(&block) == crc16_cdma2000(&block));
    }
}
