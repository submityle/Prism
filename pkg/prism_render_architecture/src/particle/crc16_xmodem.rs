//! `CRC-16/XMODEM` checksum over byte slices.
//!
//! Parameters: `width=16`, `poly=0x1021`, `init=0x0000`, `refin=false`,
//! `refout=false`, `xorout=0x0000`. The implementation uses the canonical
//! `MSB`-first bit algorithm with a `u32` working register masked to 16 bits,
//! so results match the standard reference vectors exactly. The directory
//! `check` value for the input `b"123456789"` is `0x31c3`.
//!
//! This module is `no_std` + `alloc` friendly: it uses only fixed-width
//! integers, no floating point, no transcendental functions, and no `Vec`,
//! `String`, or formatting macros in non-test code.

/// Width mask keeping the working register to 16 bits (`0xFFFF`).
const MASK: u32 = 0xFFFF;

/// Generator polynomial for `CRC-16/XMODEM` (`0x1021`).
const POLY: u32 = 0x1021;

/// Computes the `CRC-16/XMODEM` checksum of `data`.
///
/// The register starts at `0x0000` (`init`), bytes are processed `MSB`-first
/// (`refin=false`), and the final register is returned unchanged
/// (`refout=false`, `xorout=0x0000`).
pub fn crc16_xmodem(data: &[u8]) -> u16 {
    let mut reg: u32 = 0x0000; // init
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
    use super::crc16_xmodem;

    // ---- anchor vectors (ground truth) ----

    #[test]
    fn anchor_empty() {
        assert!(crc16_xmodem(b"") == 0x0000);
    }

    #[test]
    fn anchor_lower_a() {
        assert!(crc16_xmodem(b"a") == 0x7c87);
    }

    #[test]
    fn anchor_zero_byte() {
        assert!(crc16_xmodem(&[0x00]) == 0x0000);
    }

    #[test]
    fn anchor_ff_byte() {
        assert!(crc16_xmodem(&[0xff]) == 0x1ef0);
    }

    #[test]
    fn anchor_check_value() {
        assert!(crc16_xmodem(b"123456789") == 0x31c3);
    }

    // ---- self-consistent multi-byte vectors ----

    #[test]
    fn vec_ab() {
        assert!(crc16_xmodem(b"ab") == 0x74ff);
    }

    #[test]
    fn vec_abc() {
        assert!(crc16_xmodem(b"abc") == 0x9dd6);
    }

    #[test]
    fn vec_abcd() {
        assert!(crc16_xmodem(b"abcd") == 0xa836);
    }

    #[test]
    fn vec_message_digest() {
        assert!(crc16_xmodem(b"message digest") == 0x9ba6);
    }

    #[test]
    fn vec_quick_fox() {
        assert!(crc16_xmodem(b"The quick brown fox") == 0xaddd);
    }

    #[test]
    fn vec_quick_fox_full() {
        assert!(crc16_xmodem(b"The quick brown fox jumps over the lazy dog") == 0xf0c8);
    }

    #[test]
    fn vec_upper_a() {
        assert!(crc16_xmodem(b"A") == 0x58e5);
    }

    #[test]
    fn vec_upper_ab() {
        assert!(crc16_xmodem(b"AB") == 0x567b);
    }

    #[test]
    fn vec_hello() {
        assert!(crc16_xmodem(b"hello") == 0xc362);
    }

    #[test]
    fn vec_hello_world() {
        assert!(crc16_xmodem(b"Hello, World!") == 0x4fd6);
    }

    #[test]
    fn vec_digit_zero() {
        assert!(crc16_xmodem(b"0") == 0x3653);
    }

    #[test]
    fn vec_digit_one() {
        assert!(crc16_xmodem(b"1") == 0x2672);
    }

    #[test]
    fn vec_digit_nine() {
        assert!(crc16_xmodem(b"9") == 0xa77a);
    }

    #[test]
    fn vec_lower_z() {
        assert!(crc16_xmodem(b"z") == 0xdfdd);
    }

    #[test]
    fn vec_upper_z() {
        assert!(crc16_xmodem(b"Z") == 0xfbbf);
    }

    #[test]
    fn vec_space() {
        assert!(crc16_xmodem(b" ") == 0x2462);
    }

    #[test]
    fn vec_newline() {
        assert!(crc16_xmodem(b"\n") == 0xa14a);
    }

    #[test]
    fn vec_prism() {
        assert!(crc16_xmodem(b"prism") == 0x5760);
    }

    #[test]
    fn vec_particle() {
        assert!(crc16_xmodem(b"particle") == 0xfbf1);
    }

    #[test]
    fn vec_two_zero_bytes() {
        assert!(crc16_xmodem(&[0x00, 0x00]) == 0x0000);
    }

    #[test]
    fn vec_two_ff_bytes() {
        assert!(crc16_xmodem(&[0xff, 0xff]) == 0x1d0f);
    }

    #[test]
    fn vec_byte_01() {
        assert!(crc16_xmodem(&[0x01]) == 0x1021);
    }

    #[test]
    fn vec_byte_02() {
        assert!(crc16_xmodem(&[0x02]) == 0x2042);
    }

    #[test]
    fn vec_byte_80() {
        assert!(crc16_xmodem(&[0x80]) == 0x9188);
    }

    #[test]
    fn vec_byte_7f() {
        assert!(crc16_xmodem(&[0x7f]) == 0x8f78);
    }

    #[test]
    fn vec_deadbeef() {
        assert!(crc16_xmodem(&[0xde, 0xad, 0xbe, 0xef]) == 0xc457);
    }

    #[test]
    fn vec_counting_bytes() {
        assert!(crc16_xmodem(&[0x00, 0x01, 0x02, 0x03]) == 0x6131);
    }

    #[test]
    fn vec_aa_55() {
        assert!(crc16_xmodem(&[0xaa, 0x55]) == 0xf8e5);
    }

    #[test]
    fn vec_ten_digits() {
        assert!(crc16_xmodem(b"0123456789") == 0x9c58);
    }

    // ---- determinism / incremental / boundary ----

    #[test]
    fn determinism_repeated_call() {
        let input: &[u8] = b"The quick brown fox";
        assert!(crc16_xmodem(input) == crc16_xmodem(input));
    }

    #[test]
    fn determinism_all_single_bytes() {
        let mut b: u16 = 0;
        while b <= 0xff {
            let one: [u8; 1] = [b as u8];
            assert!(crc16_xmodem(&one) == crc16_xmodem(&one));
            b += 1;
        }
    }

    #[test]
    fn empty_equals_init() {
        // init == 0x0000, and an empty slice performs no iterations.
        assert!(crc16_xmodem(&[]) == 0x0000);
    }

    #[test]
    fn all_zero_runs_stay_zero() {
        // With init 0 and all-zero input the register never leaves 0x0000.
        let eight: [u8; 8] = [0x00; 8];
        let four: [u8; 4] = [0x00; 4];
        assert!(crc16_xmodem(&eight) == 0x0000);
        assert!(crc16_xmodem(&four) == 0x0000);
    }

    #[test]
    fn distinct_inputs_distinct_outputs() {
        // Known-different vectors must not collide.
        assert!(crc16_xmodem(b"a") != crc16_xmodem(b"ab"));
        assert!(crc16_xmodem(b"abc") != crc16_xmodem(b"abcd"));
        assert!(crc16_xmodem(b"A") != crc16_xmodem(b"a"));
    }

    #[test]
    fn order_sensitivity() {
        // Byte order matters for the checksum.
        assert!(crc16_xmodem(&[0x01, 0x02]) != crc16_xmodem(&[0x02, 0x01]));
    }

    #[test]
    fn result_fits_u16() {
        // The register is masked to 16 bits; the widened value stays in range.
        let value: u32 = crc16_xmodem(b"particle") as u32;
        assert!(value <= 0xffff);
    }

    #[test]
    fn single_byte_01_matches_poly() {
        // A lone 0x01 byte shifts the single set bit through to produce POLY.
        assert!(crc16_xmodem(&[0x01]) == 0x1021);
    }
}
