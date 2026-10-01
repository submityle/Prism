//! `CRC-32/BZIP2` checksum over byte slices.
//!
//! This module implements the `CRC-32/BZIP2` variant using the canonical
//! `MSB`-first (non-reflected) bitwise algorithm. The parameters are:
//! width = 32, poly = `0x04C11DB7`, init = `0xFFFFFFFF`, refin = false,
//! refout = false, xorout = `0xFFFFFFFF`.
//!
//! The implementation is pure integer arithmetic and is `no_std`-friendly:
//! it only uses `core` primitives, performs no heap allocation, and avoids
//! any floating-point or transcendental operations.

/// Compute the `CRC-32/BZIP2` checksum of `data`.
///
/// Because refin and refout are both false, each input byte is folded into
/// the high 8 bits of the running remainder, and no final bit reversal is
/// applied. The result is XOR-ed with `0xFFFFFFFF` (xorout) before return.
pub fn crc32_bzip2(data: &[u8]) -> u32 {
    const POLY: u32 = 0x04C1_1DB7;
    let mut crc: u32 = 0xFFFF_FFFF; // init

    for &byte in data {
        crc ^= u32::from(byte) << 24; // refin = false: byte enters high 8 bits
        let mut bit = 0;
        while bit < 8 {
            if (crc & 0x8000_0000) != 0 {
                crc = (crc << 1) ^ POLY;
            } else {
                crc <<= 1;
            }
            bit += 1;
        }
    }

    crc ^ 0xFFFF_FFFF // refout = false, xorout
}

#[cfg(test)]
mod tests {
    use super::crc32_bzip2;

    // ---- hard anchor vectors (ground truth) ----

    #[test]
    fn anchor_empty() {
        assert!(crc32_bzip2(b"") == 0x0000_0000);
    }

    #[test]
    fn anchor_a() {
        assert!(crc32_bzip2(b"a") == 0x1993_9b6b);
    }

    #[test]
    fn anchor_zero_byte() {
        assert!(crc32_bzip2(&[0x00]) == 0xb1f7_404b);
    }

    #[test]
    fn anchor_ff_byte() {
        assert!(crc32_bzip2(&[0xff]) == 0x0000_00ff);
    }

    #[test]
    fn anchor_check_value() {
        assert!(crc32_bzip2(b"123456789") == 0xfc89_1918);
    }

    // ---- additional multi-byte exact values ----

    #[test]
    fn value_ab() {
        assert!(crc32_bzip2(b"ab") == 0xe993_fdcd);
    }

    #[test]
    fn value_abc() {
        assert!(crc32_bzip2(b"abc") == 0x648c_bb73);
    }

    #[test]
    fn value_message_digest() {
        assert!(crc32_bzip2(b"message digest") == 0xbfc9_0357);
    }

    #[test]
    fn value_four_zeros() {
        assert!(crc32_bzip2(&[0x00, 0x00, 0x00, 0x00]) == 0x38fb_2284);
    }

    #[test]
    fn value_four_ffs() {
        assert!(crc32_bzip2(&[0xff, 0xff, 0xff, 0xff]) == 0xffff_ffff);
    }

    #[test]
    fn value_one_to_five() {
        assert!(crc32_bzip2(&[1, 2, 3, 4, 5]) == 0x1d70_b47c);
    }

    #[test]
    fn value_deadbeef() {
        assert!(crc32_bzip2(&[0xde, 0xad, 0xbe, 0xef]) == 0x7e25_e5e7);
    }

    #[test]
    fn value_full_byte_range() {
        let mut arr = [0u8; 256];
        let mut i: usize = 0;
        while i < arr.len() {
            arr[i] = i as u8;
            i += 1;
        }
        assert!(crc32_bzip2(&arr) == 0xb6b5_ee95);
    }

    #[test]
    fn value_0x80() {
        assert!(crc32_bzip2(&[0x80]) == 0xd8fb_a0a5);
    }

    #[test]
    fn value_0x01() {
        assert!(crc32_bzip2(&[0x01]) == 0xb536_5dfc);
    }

    #[test]
    fn value_quick_brown_fox() {
        assert!(crc32_bzip2(b"The quick brown fox") == 0x157a_1c4c);
    }

    #[test]
    fn value_0xaa() {
        assert!(crc32_bzip2(&[0xaa]) == 0x6f52_c093);
    }

    #[test]
    fn value_0x55() {
        assert!(crc32_bzip2(&[0x55]) == 0xdea5_8027);
    }

    #[test]
    fn value_aaaa() {
        assert!(crc32_bzip2(b"AAAA") == 0xe16e_6571);
    }

    #[test]
    fn value_zero_one_two() {
        assert!(crc32_bzip2(&[0x00, 0x01, 0x02]) == 0x9300_784d);
    }

    #[test]
    fn value_12() {
        assert!(crc32_bzip2(b"12") == 0xc013_a195);
    }

    #[test]
    fn value_1234() {
        assert!(crc32_bzip2(b"1234") == 0x596a_3b55);
    }

    #[test]
    fn value_12345678() {
        assert!(crc32_bzip2(b"12345678") == 0xb61c_3d04);
    }

    #[test]
    fn value_lowercase_alphabet() {
        assert!(crc32_bzip2(b"abcdefghijklmnopqrstuvwxyz") == 0x77bf_9396);
    }

    #[test]
    fn value_two_zeros() {
        assert!(crc32_bzip2(&[0x00, 0x00]) == 0xff48_9b82);
    }

    #[test]
    fn value_two_ffs() {
        assert!(crc32_bzip2(&[0xff, 0xff]) == 0x0000_ffff);
    }

    #[test]
    fn value_12345678_hex() {
        assert!(crc32_bzip2(&[0x12, 0x34, 0x56, 0x78]) == 0x2075_75d4);
    }

    #[test]
    fn value_eight_a5() {
        assert!(crc32_bzip2(&[0xa5; 8]) == 0x21cc_03cd);
    }

    #[test]
    fn value_sixteen_zeros() {
        assert!(crc32_bzip2(&[0x00; 16]) == 0xaad2_dd37);
    }

    #[test]
    fn value_hello() {
        assert!(crc32_bzip2(b"hello") == 0x1931_653d);
    }

    #[test]
    fn value_hello_world() {
        assert!(crc32_bzip2(b"hello world") == 0x44f7_1378);
    }

    #[test]
    fn value_abcd_bytes() {
        assert!(crc32_bzip2(&[0x61, 0x62, 0x63, 0x64]) == 0x3d4c_334b);
    }

    #[test]
    fn value_capital_z() {
        assert!(crc32_bzip2(b"Z") == 0xe6ea_3d9a);
    }

    #[test]
    fn value_char_zero() {
        assert!(crc32_bzip2(b"0") == 0x65c5_2ddb);
    }

    #[test]
    fn value_0x7f() {
        assert!(crc32_bzip2(&[0x7f]) == 0x690c_e011);
    }

    #[test]
    fn value_0xfe() {
        assert!(crc32_bzip2(&[0xfe]) == 0x04c1_1d48);
    }

    // ---- property / behavioural tests ----

    #[test]
    fn determinism_repeated_calls() {
        let input = b"deterministic input";
        let first = crc32_bzip2(input);
        let second = crc32_bzip2(input);
        assert!(first == second);
    }

    #[test]
    fn equal_slices_match() {
        let a = [0x10u8, 0x20, 0x30, 0x40];
        let b = [0x10u8, 0x20, 0x30, 0x40];
        assert!(crc32_bzip2(&a) == crc32_bzip2(&b));
    }

    #[test]
    fn different_inputs_differ() {
        assert!(crc32_bzip2(b"abc") != crc32_bzip2(b"abd"));
    }

    #[test]
    fn abcd_prefix_matches_subslice() {
        let full = [0x61u8, 0x62, 0x63, 0x64];
        assert!(crc32_bzip2(&full[..4]) == 0x3d4c_334b);
    }

    #[test]
    fn length_zero_is_identity_xor() {
        // init XOR xorout = 0xFFFFFFFF ^ 0xFFFFFFFF = 0.
        assert!(crc32_bzip2(&[]) == 0x0000_0000);
    }

    #[test]
    fn single_byte_ff_equals_byte_value() {
        // Special property of this variant for a single 0xff byte.
        assert!(crc32_bzip2(&[0xff]) == 0x0000_00ff);
    }

    #[test]
    fn result_fits_in_u32_range() {
        let r = crc32_bzip2(b"range check");
        // Trivially true for u32, but guards against type regressions.
        assert!((0..=u32::MAX).contains(&r));
    }

    #[test]
    fn multiple_of_block_length_is_stable() {
        let block = [0xa5u8; 8];
        assert!(block.len().is_multiple_of(4));
        assert!(crc32_bzip2(&block) == 0x21cc_03cd);
    }
}
