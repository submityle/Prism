//! `CRC-12/UMTS` (also known as `CRC-12/3GPP`) implementation.
//!
//! Parameters: width = 12, poly = `0x80F`, init = `0x000`, `refin` = false,
//! `refout` = true, `xorout` = `0x000`.
//!
//! The algorithm is asymmetric: input bytes are processed most-significant-bit
//! first (`refin` = false), while the final 12-bit register is bit-reflected
//! before being returned (`refout` = true). This module uses the canonical
//! `MSB`-first bit algorithm and applies a 12-bit output reflection at the end.
//!
//! All arithmetic is pure integer and masked to 12 bits; no floating point and
//! no heap allocation are used.

/// Reflect the low 12 bits of `v`.
///
/// This mirrors the bit order of a 12-bit value, which implements the
/// `refout` = true behaviour required by `CRC-12/UMTS`.
fn reflect12(mut v: u16) -> u16 {
    let mut r: u16 = 0;
    for _ in 0..12 {
        r = ((r << 1) | (v & 1)) & 0x0FFF;
        v >>= 1;
    }
    r
}

/// Compute the `CRC-12/UMTS` checksum of `data`.
///
/// Returns a 12-bit value in the low bits of a `u16`.
pub fn crc12_umts(data: &[u8]) -> u16 {
    const POLY: u16 = 0x80F;
    const MASK: u16 = 0x0FFF;
    let mut reg: u16 = 0x000;
    for &byte in data {
        for i in 0..8u32 {
            let bit = ((byte >> (7 - i)) & 1) as u16; // refin = false (MSB first)
            let hi = (reg >> 11) & 1;
            let fb = hi ^ bit;
            reg = (reg << 1) & MASK;
            if fb != 0 {
                reg ^= POLY;
            }
        }
    }
    reflect12(reg) & MASK // refout = true, xorout = 0
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- anchor ground-truth vectors ----

    #[test]
    fn anchor_empty() {
        assert!(crc12_umts(b"") == 0x000);
    }

    #[test]
    fn anchor_a() {
        assert!(crc12_umts(b"a") == 0xf3d);
    }

    #[test]
    fn anchor_zero_byte() {
        assert!(crc12_umts(&[0x00]) == 0x000);
    }

    #[test]
    fn anchor_ff_byte() {
        assert!(crc12_umts(&[0xff]) == 0x606);
    }

    #[test]
    fn anchor_check_value() {
        assert!(crc12_umts(b"123456789") == 0xdaf);
    }

    // ---- additional single-byte vectors ----

    #[test]
    fn single_b() {
        assert!(crc12_umts(b"b") == 0x8bd);
    }

    #[test]
    fn single_c() {
        assert!(crc12_umts(b"c") == 0x7bc);
    }

    #[test]
    fn single_0x01() {
        assert!(crc12_umts(&[0x01]) == 0xf01);
    }

    #[test]
    fn single_0x02() {
        assert!(crc12_umts(&[0x02]) == 0x881);
    }

    #[test]
    fn single_0x80() {
        assert!(crc12_umts(&[0x80]) == 0xa0b);
    }

    #[test]
    fn single_0x7f() {
        assert!(crc12_umts(&[0x7f]) == 0xc0d);
    }

    #[test]
    fn single_0xaa() {
        assert!(crc12_umts(&[0xaa]) == 0x202);
    }

    #[test]
    fn single_0x55() {
        assert!(crc12_umts(&[0x55]) == 0x404);
    }

    // ---- multi-byte vectors ----

    #[test]
    fn multi_ab() {
        assert!(crc12_umts(b"ab") == 0x321);
    }

    #[test]
    fn multi_abc() {
        assert!(crc12_umts(b"abc") == 0x6f5);
    }

    #[test]
    fn multi_hello() {
        assert!(crc12_umts(b"hello") == 0xcb6);
    }

    #[test]
    fn multi_world() {
        assert!(crc12_umts(b"world") == 0x847);
    }

    #[test]
    fn multi_two_zero_bytes() {
        assert!(crc12_umts(&[0x00, 0x00]) == 0x000);
    }

    #[test]
    fn multi_two_ff_bytes() {
        assert!(crc12_umts(&[0xff, 0xff]) == 0x63c);
    }

    #[test]
    fn multi_0x12_0x34() {
        assert!(crc12_umts(&[0x12, 0x34]) == 0x61a);
    }

    #[test]
    fn multi_deadbeef() {
        assert!(crc12_umts(&[0xde, 0xad, 0xbe, 0xef]) == 0x231);
    }

    #[test]
    fn multi_sentence() {
        assert!(crc12_umts(b"The quick brown fox") == 0xdf6);
    }

    #[test]
    fn multi_prism() {
        assert!(crc12_umts(b"prism") == 0x744);
    }

    #[test]
    fn multi_crc_label() {
        assert!(crc12_umts(b"CRC-12") == 0xc49);
    }

    #[test]
    fn multi_counting_0_to_7() {
        assert!(crc12_umts(&[0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]) == 0xa6c);
    }

    #[test]
    fn multi_counting_1_to_10() {
        assert!(crc12_umts(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 0x8b2);
    }

    // ---- reflect12 unit tests ----

    #[test]
    fn reflect12_zero() {
        assert!(reflect12(0x000) == 0x000);
    }

    #[test]
    fn reflect12_all_ones() {
        assert!(reflect12(0xfff) == 0xfff);
    }

    #[test]
    fn reflect12_lsb_to_msb() {
        assert!(reflect12(0x001) == 0x800);
    }

    #[test]
    fn reflect12_msb_to_lsb() {
        assert!(reflect12(0x800) == 0x001);
    }

    #[test]
    fn reflect12_known_value() {
        assert!(reflect12(0xf3d) == 0xbcf);
    }

    #[test]
    fn reflect12_is_involution() {
        let mut v: u16 = 0;
        while v < 0x1000 {
            assert!(reflect12(reflect12(v)) == v);
            v += 1;
        }
    }

    #[test]
    fn reflect12_ignores_high_bits_of_input() {
        // Bits above bit 11 must not influence the 12-bit reflection.
        assert!(reflect12(0xf001 & 0x0FFF) == reflect12(0x001));
    }

    // ---- structural / property tests ----

    #[test]
    fn output_is_12_bits() {
        let inputs: [&[u8]; 8] = [
            b"",
            b"a",
            b"123456789",
            &[0xff],
            &[0xde, 0xad, 0xbe, 0xef],
            b"The quick brown fox",
            b"prism",
            &[0x00, 0x01, 0x02, 0x03],
        ];
        for inp in inputs {
            assert!(crc12_umts(inp) < 0x1000);
        }
    }

    #[test]
    fn deterministic_repeat() {
        let data = b"deterministic";
        let first = crc12_umts(data);
        let second = crc12_umts(data);
        assert!(first == second);
    }

    #[test]
    fn order_matters() {
        let forward = crc12_umts(&[0x12, 0x34]);
        let reversed = crc12_umts(&[0x34, 0x12]);
        assert!(forward != reversed);
    }

    #[test]
    fn empty_equals_zero_byte_stream() {
        // init is 0 and processing zero bytes keeps the register at 0.
        assert!(crc12_umts(b"") == crc12_umts(&[0x00, 0x00, 0x00]));
    }

    #[test]
    fn length_sensitivity() {
        let one = crc12_umts(&[0xaa]);
        let two = crc12_umts(&[0xaa, 0xaa]);
        assert!(one != two);
    }

    #[test]
    fn distinct_single_bytes_mostly_distinct() {
        // 0x00 and 0x01 differ, confirming per-byte sensitivity.
        assert!(crc12_umts(&[0x00]) != crc12_umts(&[0x01]));
    }

    #[test]
    fn prefix_changes_result() {
        let base = crc12_umts(b"message");
        let prefixed = crc12_umts(b"Xmessage");
        assert!(base != prefixed);
    }

    #[test]
    fn suffix_changes_result() {
        let base = crc12_umts(b"message");
        let suffixed = crc12_umts(b"messageX");
        assert!(base != suffixed);
    }

    #[test]
    fn single_bit_inputs_differ() {
        let b0 = crc12_umts(&[0x01]);
        let b7 = crc12_umts(&[0x80]);
        assert!(b0 != b7);
    }

    #[test]
    fn long_zero_run_stays_zero() {
        let zeros = [0u8; 32];
        assert!(crc12_umts(&zeros) == 0x000);
    }
}
