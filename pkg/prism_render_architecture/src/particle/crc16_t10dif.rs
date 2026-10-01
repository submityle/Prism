//! `CRC-16/T10-DIF` checksum over byte slices.
//!
//! Parameters: `width=16`, `poly=0x8BB7`, `init=0x0000`, `refin=false`,
//! `refout=false`, `xorout=0x0000`. The implementation uses the canonical
//! non-reflected (`MSB`-first) bit algorithm with a `u32` working register
//! masked to 16 bits, so results match the standard reference vectors
//! exactly. Here `T10` is the `INCITS` Technical Committee `T10`, `DIF`
//! means Data Integrity Field, and `XOR` denotes bitwise exclusive-or. The
//! directory `check` value for the input `b"123456789"` is `0xd0db`.
//!
//! This module is `no_std` + `alloc` friendly: it uses only fixed-width
//! integers, no floating point, no transcendental functions, and no `Vec`,
//! `String`, or formatting macros in non-test code.

/// Width mask keeping the working register to 16 bits (`0xFFFF`).
const MASK: u32 = 0xFFFF;

/// Generator polynomial for `CRC-16/T10-DIF` (`0x8BB7`).
const POLY: u32 = 0x8BB7;

/// Computes the `CRC-16/T10-DIF` checksum of `data`.
///
/// The register starts at `0x0000` (`init`), bytes are processed `MSB`-first
/// (`refin=false`), and the final register is returned unchanged
/// (`refout=false`, `xorout=0x0000`).
pub fn crc16_t10dif(data: &[u8]) -> u16 {
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
    use super::crc16_t10dif;

    // ---- anchor vectors (ground truth) ----

    #[test]
    fn anchor_empty() {
        assert!(crc16_t10dif(b"") == 0x0000);
    }

    #[test]
    fn anchor_lower_a() {
        assert!(crc16_t10dif(b"a") == 0xfaae);
    }

    #[test]
    fn anchor_zero_byte() {
        assert!(crc16_t10dif(&[0x00]) == 0x0000);
    }

    #[test]
    fn anchor_ff_byte() {
        assert!(crc16_t10dif(&[0xff]) == 0x55b3);
    }

    #[test]
    fn anchor_check_value() {
        assert!(crc16_t10dif(b"123456789") == 0xd0db);
    }

    // ---- self-consistent multi-byte vectors (hardcoded) ----

    #[test]
    fn vec_bytes_010203() {
        assert!(crc16_t10dif(&[0x01, 0x02, 0x03]) == 0x83cc);
    }

    #[test]
    fn vec_deadbeef() {
        assert!(crc16_t10dif(&[0xde, 0xad, 0xbe, 0xef]) == 0x34dc);
    }

    #[test]
    fn vec_hello() {
        assert!(crc16_t10dif(b"hello") == 0xe982);
    }

    #[test]
    fn vec_prism() {
        assert!(crc16_t10dif(b"Prism") == 0x7e98);
    }

    #[test]
    fn vec_four_zero_bytes() {
        assert!(crc16_t10dif(&[0x00, 0x00, 0x00, 0x00]) == 0x0000);
    }

    #[test]
    fn vec_two_ff_bytes() {
        assert!(crc16_t10dif(&[0xff, 0xff]) == 0x534f);
    }

    #[test]
    fn vec_bytes_123456789a() {
        assert!(crc16_t10dif(&[0x12, 0x34, 0x56, 0x78, 0x9a]) == 0x6457);
    }

    #[test]
    fn vec_t10_dif_text() {
        assert!(crc16_t10dif(b"T10-DIF") == 0x2f18);
    }

    #[test]
    fn vec_aa55_pattern() {
        assert!(crc16_t10dif(&[0xaa, 0x55, 0xaa, 0x55]) == 0x5b24);
    }

    #[test]
    fn vec_bytes_8001() {
        assert!(crc16_t10dif(&[0x80, 0x01]) == 0xb484);
    }

    #[test]
    fn vec_abcdefghij() {
        assert!(crc16_t10dif(b"abcdefghij") == 0xb90b);
    }

    #[test]
    fn vec_00ff00ff() {
        assert!(crc16_t10dif(&[0x00, 0xff, 0x00, 0xff]) == 0x876f);
    }

    // ---- additional short/single-byte vectors ----

    #[test]
    fn vec_byte_01_matches_poly() {
        // A lone 0x01 shifts the single set bit through to produce POLY.
        assert!(crc16_t10dif(&[0x01]) == 0x8bb7);
    }

    #[test]
    fn vec_byte_02() {
        assert!(crc16_t10dif(&[0x02]) == 0x9cd9);
    }

    #[test]
    fn vec_byte_80() {
        assert!(crc16_t10dif(&[0x80]) == 0x3ab1);
    }

    #[test]
    fn vec_byte_7f() {
        assert!(crc16_t10dif(&[0x7f]) == 0x6f02);
    }

    #[test]
    fn vec_ab() {
        assert!(crc16_t10dif(b"ab") == 0x2fc1);
    }

    #[test]
    fn vec_abc() {
        assert!(crc16_t10dif(b"abc") == 0x443b);
    }

    #[test]
    fn vec_abcd() {
        assert!(crc16_t10dif(b"abcd") == 0x929a);
    }

    #[test]
    fn vec_upper_a() {
        assert!(crc16_t10dif(b"A") == 0x5334);
    }

    #[test]
    fn vec_two_zero_bytes() {
        assert!(crc16_t10dif(&[0x00, 0x00]) == 0x0000);
    }

    #[test]
    fn vec_ten_digits() {
        assert!(crc16_t10dif(b"0123456789") == 0xde40);
    }

    #[test]
    fn vec_message_digest() {
        assert!(crc16_t10dif(b"message digest") == 0x7a7e);
    }

    #[test]
    fn vec_quick_fox() {
        assert!(crc16_t10dif(b"The quick brown fox") == 0xa1ef);
    }

    #[test]
    fn vec_particle() {
        assert!(crc16_t10dif(b"particle") == 0x3048);
    }

    // ---- determinism / boundary / structural ----

    #[test]
    fn determinism_repeated_call() {
        let input: &[u8] = b"The quick brown fox";
        assert!(crc16_t10dif(input) == crc16_t10dif(input));
    }

    #[test]
    fn determinism_all_single_bytes() {
        let mut b: u16 = 0;
        while b <= 0xff {
            let one: [u8; 1] = [b as u8];
            assert!(crc16_t10dif(&one) == crc16_t10dif(&one));
            b += 1;
        }
    }

    #[test]
    fn empty_equals_init() {
        // init == 0x0000, and an empty slice performs no iterations.
        assert!(crc16_t10dif(&[]) == 0x0000);
    }

    #[test]
    fn all_zero_runs_stay_zero() {
        // With init 0 and all-zero input the register never leaves 0x0000.
        let eight: [u8; 8] = [0x00; 8];
        let four: [u8; 4] = [0x00; 4];
        assert!(crc16_t10dif(&eight) == 0x0000);
        assert!(crc16_t10dif(&four) == 0x0000);
    }

    #[test]
    fn distinct_inputs_distinct_outputs() {
        // Known-different vectors must not collide.
        assert!(crc16_t10dif(b"a") != crc16_t10dif(b"ab"));
        assert!(crc16_t10dif(b"abc") != crc16_t10dif(b"abcd"));
        assert!(crc16_t10dif(b"A") != crc16_t10dif(b"a"));
    }

    #[test]
    fn order_sensitivity() {
        // Byte order matters for the checksum.
        assert!(crc16_t10dif(&[0x01, 0x02]) != crc16_t10dif(&[0x02, 0x01]));
    }

    #[test]
    fn result_fits_u16() {
        // The register is masked to 16 bits; the widened value stays in range.
        let value: u32 = crc16_t10dif(b"particle") as u32;
        assert!(value <= 0xffff);
    }

    #[test]
    fn sampling_divisible_indices_deterministic() {
        // Build a byte run, then checksum only the elements at indices that
        // are multiples of three; the sampled checksum must be reproducible.
        let mut buf: [u8; 64] = [0x00; 64];
        let mut i: usize = 0;
        while i < buf.len() {
            buf[i] = (i as u8).wrapping_mul(31).wrapping_add(7);
            i += 1;
        }
        let mut sampled: [u8; 64] = [0x00; 64];
        let mut count: usize = 0;
        let mut j: usize = 0;
        while j < buf.len() {
            if j.is_multiple_of(3) {
                sampled[count] = buf[j];
                count += 1;
            }
            j += 1;
        }
        let slice: &[u8] = &sampled[..count];
        assert!(crc16_t10dif(slice) == crc16_t10dif(slice));
        // 64 indices, every third starting at 0 => 22 sampled bytes.
        assert!(count == 22);
    }

    #[test]
    fn long_input_256_counting_stable() {
        let mut buf: [u8; 256] = [0x00; 256];
        let mut i: usize = 0;
        while i < buf.len() {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc16_t10dif(&buf) == 0x563b);
        assert!(crc16_t10dif(&buf) == crc16_t10dif(&buf));
    }

    #[test]
    fn long_input_1000_aa_stable() {
        let buf: [u8; 1000] = [0xaa; 1000];
        assert!(crc16_t10dif(&buf) == 0xd878);
    }

    #[test]
    fn long_input_100_a_stable() {
        let buf: [u8; 100] = [0x61; 100];
        assert!(crc16_t10dif(&buf) == 0x4ab6);
    }
}
