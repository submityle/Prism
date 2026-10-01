//! `CRC`-16/MCRF4XX checksum (width=16, poly=0x1021, init=0xFFFF, refin=true,
//! refout=true, `XOR`out=0x0000; check=0x6F91 for `b"123456789"`).
//!
//! Because `refin`/`refout` are both `true`, the implementation runs `LSB`-first
//! using the bit-reversed polynomial `REFPOLY` = 0x8408 (the 16-bit reflection
//! of 0x1021). The register is a `u16` whose reflected initial value is
//! reflect(0xFFFF) = 0xFFFF, and the final `XOR`out of 0x0000 is a no-op.

/// Bit-reversed form of poly 0x1021, used for the `LSB`-first `u16` register.
const REFPOLY: u16 = 0x8408;

/// Computes the `CRC`-16/MCRF4XX checksum over `data`.
///
/// Uses the reflected polynomial `REFPOLY` and processes each byte `LSB`-first.
/// The `XOR`out stage is 0x0000, so the register is returned unchanged.
pub fn crc16_mcrf4xx(data: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF; // reflected init
    for &b in data {
        crc ^= b as u16;
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
    crc // xorout = 0
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Anchor 1: empty input ---
    #[test]
    fn test_anchor_empty() {
        assert!(crc16_mcrf4xx(b"") == 0xffff);
    }

    // --- Anchor 2: single 0x00 ---
    #[test]
    fn test_anchor_single_zero() {
        assert!(crc16_mcrf4xx(&[0x00]) == 0x0f87);
    }

    // --- Anchor 3: single 0xff ---
    #[test]
    fn test_anchor_single_ff() {
        assert!(crc16_mcrf4xx(&[0xff]) == 0x00ff);
    }

    // --- Anchor 4a: b"a" ---
    #[test]
    fn test_anchor_a() {
        assert!(crc16_mcrf4xx(b"a") == 0x7d08);
    }

    // --- Anchor 4b: standard check vector ---
    #[test]
    fn test_anchor_check_vector() {
        assert!(crc16_mcrf4xx(b"123456789") == 0x6f91);
    }

    // --- Multi-byte hardcoded vectors ---
    #[test]
    fn test_multi_ab() {
        assert!(crc16_mcrf4xx(b"ab") == 0xcc21);
    }

    #[test]
    fn test_multi_abc() {
        assert!(crc16_mcrf4xx(b"abc") == 0x61da);
    }

    #[test]
    fn test_multi_two_zeros() {
        assert!(crc16_mcrf4xx(&[0x00, 0x00]) == 0xf0b8);
    }

    #[test]
    fn test_multi_single_01() {
        assert!(crc16_mcrf4xx(&[0x01]) == 0x1e0e);
    }

    #[test]
    fn test_multi_single_02() {
        assert!(crc16_mcrf4xx(&[0x02]) == 0x2c95);
    }

    #[test]
    fn test_multi_single_7f() {
        assert!(crc16_mcrf4xx(&[0x7f]) == 0x84f7);
    }

    #[test]
    fn test_multi_single_80() {
        assert!(crc16_mcrf4xx(&[0x80]) == 0x8b8f);
    }

    #[test]
    fn test_multi_aa_55() {
        assert!(crc16_mcrf4xx(&[0xaa, 0x55]) == 0xa71f);
    }

    #[test]
    fn test_multi_deadbeef() {
        assert!(crc16_mcrf4xx(&[0xde, 0xad, 0xbe, 0xef]) == 0x1a34);
    }

    #[test]
    fn test_multi_hello() {
        assert!(crc16_mcrf4xx(b"Hello") == 0xabd3);
    }

    #[test]
    fn test_multi_sentence() {
        assert!(crc16_mcrf4xx(b"The quick brown fox") == 0x039d);
    }

    #[test]
    fn test_multi_four_zeros() {
        assert!(crc16_mcrf4xx(&[0x00, 0x00, 0x00, 0x00]) == 0x0321);
    }

    #[test]
    fn test_multi_four_ff() {
        assert!(crc16_mcrf4xx(&[0xff, 0xff, 0xff, 0xff]) == 0xf0b8);
    }

    #[test]
    fn test_multi_0_to_15() {
        let data = [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        assert!(crc16_mcrf4xx(&data) == 0xec16);
    }

    #[test]
    fn test_multi_12345678() {
        assert!(crc16_mcrf4xx(&[0x12, 0x34, 0x56, 0x78]) == 0x64d1);
    }

    // --- Determinism ---
    #[test]
    fn test_determinism_empty() {
        assert!(crc16_mcrf4xx(b"") == crc16_mcrf4xx(b""));
    }

    #[test]
    fn test_determinism_check() {
        let a = crc16_mcrf4xx(b"123456789");
        let b = crc16_mcrf4xx(b"123456789");
        assert!(a == b);
    }

    #[test]
    fn test_determinism_binary() {
        let data = [0xde, 0xad, 0xbe, 0xef];
        assert!(crc16_mcrf4xx(&data) == crc16_mcrf4xx(&data));
    }

    #[test]
    fn test_determinism_repeated_calls() {
        let first = crc16_mcrf4xx(b"Hello");
        let mut i = 0;
        while i < 10 {
            assert!(crc16_mcrf4xx(b"Hello") == first);
            i += 1;
        }
    }

    // --- Slice views equal array value ---
    #[test]
    fn test_slice_equivalence() {
        let data = [0x12u8, 0x34, 0x56, 0x78];
        assert!(crc16_mcrf4xx(&data[..]) == 0x64d1);
    }

    #[test]
    fn test_subslice_prefix_differs() {
        let data = [0x12u8, 0x34, 0x56, 0x78];
        assert!(crc16_mcrf4xx(&data[..2]) != crc16_mcrf4xx(&data));
    }

    // --- Ordering sensitivity ---
    #[test]
    fn test_order_sensitivity() {
        assert!(crc16_mcrf4xx(&[0xaa, 0x55]) != crc16_mcrf4xx(&[0x55, 0xaa]));
    }

    #[test]
    fn test_different_inputs_differ() {
        assert!(crc16_mcrf4xx(b"a") != crc16_mcrf4xx(b"b"));
    }

    // --- Register stays in u16 range ---
    #[test]
    fn test_output_in_u16_range() {
        let v = crc16_mcrf4xx(b"123456789");
        assert!((0x0000u16..=0xffffu16).contains(&v));
    }

    #[test]
    fn test_empty_in_range() {
        let v = crc16_mcrf4xx(b"");
        assert!((0x0000u16..=0xffffu16).contains(&v));
    }

    // --- Long input stability ---
    #[test]
    fn test_long_a5_1000() {
        let data = [0xA5u8; 1000];
        assert!(crc16_mcrf4xx(&data) == 0xe5f4);
    }

    #[test]
    fn test_long_a5_1000_deterministic() {
        let data = [0xA5u8; 1000];
        assert!(crc16_mcrf4xx(&data) == crc16_mcrf4xx(&data));
    }

    #[test]
    fn test_long_all_bytes_256() {
        let mut data = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc16_mcrf4xx(&data) == 0xcfc3);
    }

    #[test]
    fn test_long_zeros_stability() {
        let data = [0u8; 512];
        let first = crc16_mcrf4xx(&data);
        assert!(crc16_mcrf4xx(&data) == first);
    }

    #[test]
    fn test_long_length_is_multiple() {
        let data = [0xA5u8; 1000];
        assert!(data.len().is_multiple_of(8));
        assert!(crc16_mcrf4xx(&data) == 0xe5f4);
    }

    // --- Incremental independence (fresh init each call) ---
    #[test]
    fn test_no_hidden_state() {
        let _ = crc16_mcrf4xx(b"warmup");
        assert!(crc16_mcrf4xx(b"") == 0xffff);
    }

    #[test]
    fn test_refpoly_value() {
        assert!(REFPOLY == 0x8408);
    }

    #[test]
    fn test_init_matches_empty() {
        // reflected init equals the empty-input result (no bytes processed).
        assert!(crc16_mcrf4xx(b"") == 0xffff);
    }
}
