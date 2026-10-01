//! `CRC-8/AUTOSAR` golden model: width=8, poly=0x2F, init=0xFF, refin=false, refout=false, xorout=0xFF; check(b"123456789")=0xdf.
//!
//! Non-reflected (`MSB`-first) byte-wise `CRC` over a `u8` register. All math is
//! pure integer; no floating point and no transcendental operations are used.
//! This module is `no_std` + `alloc` and avoids `Vec`/`String`/`format!`.

/// Reduction polynomial for `CRC-8/AUTOSAR` (`0x2F`, `MSB`-first form).
const POLY: u8 = 0x2F;

/// Initial register value before processing input bytes.
const INIT: u8 = 0xFF;

/// Final `XOR` applied to the register to produce the output `CRC`.
const XOROUT: u8 = 0xFF;

/// Compute the `CRC-8/AUTOSAR` checksum of `data`.
///
/// The register is `u8`; each left shift by one truncates to 8 bits, which is
/// exactly the `CRC` reduction behavior (the dropped high bit is accounted for
/// by the `MSB` test before the shift).
pub fn crc8_autosar(data: &[u8]) -> u8 {
    let mut reg: u8 = INIT;
    for &byte in data {
        reg ^= byte;
        for _ in 0..8 {
            if (reg & 0x80) != 0 {
                reg = (reg << 1) ^ POLY;
            } else {
                reg <<= 1;
            }
        }
    }
    reg ^ XOROUT
}

#[cfg(test)]
mod tests {
    use super::crc8_autosar;

    // ---- 5 anchors ----

    #[test]
    fn anchor_empty() {
        assert!(crc8_autosar(b"") == 0x00);
    }

    #[test]
    fn anchor_a() {
        assert!(crc8_autosar(b"a") == 0x07);
    }

    #[test]
    fn anchor_zero_byte() {
        assert!(crc8_autosar(&[0x00]) == 0xbd);
    }

    #[test]
    fn anchor_ff_byte() {
        assert!(crc8_autosar(&[0xff]) == 0xff);
    }

    #[test]
    fn anchor_check_123456789() {
        assert!(crc8_autosar(b"123456789") == 0xdf);
    }

    // ---- multi-byte hardcoded (self-computed golden values) ----

    #[test]
    fn multi_01020304() {
        assert!(crc8_autosar(&[0x01, 0x02, 0x03, 0x04]) == 0x13);
    }

    #[test]
    fn multi_deadbeef() {
        assert!(crc8_autosar(&[0xde, 0xad, 0xbe, 0xef]) == 0xeb);
    }

    #[test]
    fn multi_four_zeros() {
        assert!(crc8_autosar(&[0x00, 0x00, 0x00, 0x00]) == 0x12);
    }

    #[test]
    fn multi_four_ffs() {
        assert!(crc8_autosar(&[0xff, 0xff, 0xff, 0xff]) == 0x6c);
    }

    #[test]
    fn multi_123456789a() {
        assert!(crc8_autosar(&[0x12, 0x34, 0x56, 0x78, 0x9a]) == 0xfe);
    }

    #[test]
    fn multi_aa55aa55() {
        assert!(crc8_autosar(&[0xaa, 0x55, 0xaa, 0x55]) == 0x88);
    }

    #[test]
    fn multi_102030() {
        assert!(crc8_autosar(&[0x10, 0x20, 0x30]) == 0xe1);
    }

    #[test]
    fn multi_7f8081() {
        assert!(crc8_autosar(&[0x7f, 0x80, 0x81]) == 0xf9);
    }

    #[test]
    fn multi_cafebabe() {
        assert!(crc8_autosar(&[0xca, 0xfe, 0xba, 0xbe]) == 0xa6);
    }

    #[test]
    fn multi_single_01() {
        assert!(crc8_autosar(&[0x01]) == 0x92);
    }

    #[test]
    fn multi_single_02() {
        assert!(crc8_autosar(&[0x02]) == 0xe3);
    }

    #[test]
    fn multi_single_80() {
        assert!(crc8_autosar(&[0x80]) == 0x5e);
    }

    #[test]
    fn multi_abc() {
        assert!(crc8_autosar(b"abc") == 0x41);
    }

    #[test]
    fn multi_prism() {
        assert!(crc8_autosar(b"Prism") == 0x77);
    }

    #[test]
    fn multi_f00f_repeated() {
        assert!(crc8_autosar(&[0xf0, 0x0f, 0xf0, 0x0f, 0xf0, 0x0f]) == 0xdb);
    }

    #[test]
    fn multi_12345() {
        assert!(crc8_autosar(b"12345") == 0x92);
    }

    // ---- determinism ----

    #[test]
    fn determinism_repeated_calls() {
        let data = [0x11, 0x22, 0x33, 0x44, 0x55];
        let first = crc8_autosar(&data);
        let second = crc8_autosar(&data);
        let third = crc8_autosar(&data);
        assert!(first == second);
        assert!(second == third);
    }

    #[test]
    fn determinism_check_anchor_repeat() {
        let a = crc8_autosar(b"123456789");
        let b = crc8_autosar(b"123456789");
        assert!(a == b);
        assert!(a == 0xdf);
    }

    #[test]
    fn determinism_loop_sampled() {
        // Sample single-byte inputs and confirm each is stable across calls.
        let mut i: u16 = 0;
        while i < 256 {
            if (i as u8).is_multiple_of(16) {
                let byte = [i as u8];
                let x = crc8_autosar(&byte);
                let y = crc8_autosar(&byte);
                assert!(x == y);
            }
            i += 1;
        }
    }

    // ---- empty input == init ^ xorout == 0 ----

    #[test]
    fn empty_equals_init_xor_xorout() {
        let expected = 0xFF_u8 ^ 0xFF_u8;
        assert!(crc8_autosar(&[]) == expected);
        assert!(expected == 0x00);
    }

    #[test]
    fn empty_slice_variants() {
        let empty: [u8; 0] = [];
        assert!(crc8_autosar(&empty) == 0x00);
        assert!(crc8_autosar(b"") == 0x00);
    }

    // ---- sampling / divisibility ----

    #[test]
    fn sampling_full_byte_range_in_u8() {
        // Every single-byte input yields a value representable in a `u8`
        // (trivially true, but exercises the whole 0..=255 input domain).
        let mut i: u16 = 0;
        let mut count: u16 = 0;
        while i < 256 {
            let byte = [i as u8];
            let _ = crc8_autosar(&byte);
            count += 1;
            i += 1;
        }
        assert!(count == 256);
    }

    #[test]
    fn sampling_multiples_of_four() {
        // Confirm divisibility-based sampling via `is_multiple_of`.
        let data = [0x00_u8, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
        let mut idx: usize = 0;
        let mut seen: usize = 0;
        while idx < data.len() {
            if idx.is_multiple_of(4) {
                let single = [data[idx]];
                let x = crc8_autosar(&single);
                let y = crc8_autosar(&single);
                assert!(x == y);
                seen += 1;
            }
            idx += 1;
        }
        assert!(seen == 2);
    }

    #[test]
    fn sampling_range_contains_index() {
        // Range membership check using `.contains` instead of comparisons.
        let probes = [0x20_u8, 0x40, 0x60, 0x80];
        let mut idx: usize = 0;
        while idx < probes.len() {
            assert!((0..probes.len()).contains(&idx));
            let single = [probes[idx]];
            let v = crc8_autosar(&single);
            let v2 = crc8_autosar(&single);
            assert!(v == v2);
            idx += 1;
        }
    }

    // ---- long input stability ----

    #[test]
    fn long_input_incrementing_256() {
        let mut data = [0u8; 256];
        let mut i: usize = 0;
        while i < 256 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc8_autosar(&data) == 0x06);
    }

    #[test]
    fn long_input_incrementing_256_stable() {
        let mut data = [0u8; 256];
        let mut i: usize = 0;
        while i < 256 {
            data[i] = i as u8;
            i += 1;
        }
        let a = crc8_autosar(&data);
        let b = crc8_autosar(&data);
        assert!(a == b);
    }

    #[test]
    fn long_input_aa_100() {
        let data = [0xAA_u8; 100];
        assert!(crc8_autosar(&data) == 0x03);
    }

    #[test]
    fn long_input_incrementing_64() {
        let mut data = [0u8; 64];
        let mut i: usize = 0;
        while i < 64 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc8_autosar(&data) == 0x0a);
    }

    #[test]
    fn long_input_zeros_8() {
        let data = [0x00_u8; 8];
        assert!(crc8_autosar(&data) == 0x21);
    }

    #[test]
    fn long_input_zeros_large_stable() {
        let data = [0x00_u8; 512];
        let a = crc8_autosar(&data);
        let b = crc8_autosar(&data);
        assert!(a == b);
    }

    // ---- structural / property checks ----

    #[test]
    fn init_xorout_constants() {
        assert!(super::INIT == 0xFF);
        assert!(super::XOROUT == 0xFF);
        assert!(super::POLY == 0x2F);
    }

    #[test]
    fn prefix_extends_consistently() {
        // CRC of a prefix must be stable regardless of surrounding calls.
        let p1 = crc8_autosar(b"12345");
        let p2 = crc8_autosar(b"123456789");
        assert!(p1 == 0x92);
        assert!(p2 == 0xdf);
        assert!(p1 != p2);
    }

    #[test]
    fn differing_single_bytes_differ() {
        let a = crc8_autosar(&[0x00]);
        let b = crc8_autosar(&[0x01]);
        assert!(a != b);
    }

    #[test]
    fn order_sensitivity() {
        let forward = crc8_autosar(&[0x01, 0x02]);
        let reverse = crc8_autosar(&[0x02, 0x01]);
        assert!(forward != reverse);
    }
}
