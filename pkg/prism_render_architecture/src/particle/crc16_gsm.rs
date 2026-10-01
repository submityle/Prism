//! CRC-16/GSM checksum implementation.
//!
//! Parameters: `width=16`, `poly=0x1021`, `init=0x0`,
//! `refin=false`, `refout=false`, `xorout=0xFFFF`,
//! `check=0xce3c`.
//!
//! The implementation is bit-wise and uses only fixed-size arrays and
//! `for`/`while` loops so it is compatible with `no_std` + `alloc`.

const WIDTH: u32 = 16;
const POLY: u32 = 0x1021;
const INIT: u32 = 0x0;
const XOROUT: u32 = 0xFFFF;
const REFLECT_IN: bool = false;
const REFLECT_OUT: bool = false;
const MASK: u32 = 0xffff;
const TOPBIT: u32 = 1u32 << (WIDTH - 1);

const fn reflect(value: u32, bits: u32) -> u32 {
    let mut out = 0u32;
    let mut i = 0u32;
    while i < bits {
        if ((value >> i) & 1) != 0 {
            out |= 1u32 << (bits - 1 - i);
        }
        i += 1;
    }
    out
}

/// Compute the CRC-16/GSM checksum over `data`.
pub fn crc16_gsm(data: &[u8]) -> u16 {
    let mut reg = INIT & MASK;
    let mut idx = 0usize;
    while idx < data.len() {
        let byte = data[idx];
        let b = if REFLECT_IN {
            reflect(byte as u32, 8)
        } else {
            byte as u32
        };
        let mut i = 0u32;
        while i < 8 {
            let bit = (b >> (7 - i)) & 1;
            let msb = u32::from((reg & TOPBIT) != 0);
            reg = (reg << 1) & MASK;
            if (msb ^ bit) != 0 {
                reg ^= POLY;
            }
            i += 1;
        }
        idx += 1;
    }
    if REFLECT_OUT {
        reg = reflect(reg, WIDTH);
    }
    ((reg ^ XOROUT) & MASK) as u16
}

#[cfg(test)]
mod tests {
    use super::crc16_gsm;

    // --- 24 hard-coded reference-vector anchors (truth: /tmp/b42_vectors.txt) ---

    #[test]
    fn vec_empty() {
        assert!(crc16_gsm(b"") == 0xffff);
    }

    #[test]
    fn vec_z00() {
        assert!(crc16_gsm(&[0x00]) == 0xffff);
    }

    #[test]
    fn vec_ff() {
        assert!(crc16_gsm(&[0xff]) == 0xe10f);
    }

    #[test]
    fn vec_a() {
        assert!(crc16_gsm(b"a") == 0x8378);
    }

    #[test]
    fn vec_b() {
        assert!(crc16_gsm(b"b") == 0xb31b);
    }

    #[test]
    fn vec_ab() {
        assert!(crc16_gsm(b"ab") == 0x8b00);
    }

    #[test]
    fn vec_abc() {
        assert!(crc16_gsm(b"abc") == 0x6229);
    }

    #[test]
    fn vec_two_zeros() {
        assert!(crc16_gsm(&[0x00, 0x00]) == 0xffff);
    }

    #[test]
    fn vec_01() {
        assert!(crc16_gsm(&[0x01]) == 0xefde);
    }

    #[test]
    fn vec_02() {
        assert!(crc16_gsm(&[0x02]) == 0xdfbd);
    }

    #[test]
    fn vec_7f() {
        assert!(crc16_gsm(&[0x7f]) == 0x7087);
    }

    #[test]
    fn vec_80() {
        assert!(crc16_gsm(&[0x80]) == 0x6e77);
    }

    #[test]
    fn vec_aa_55() {
        assert!(crc16_gsm(&[0xaa, 0x55]) == 0x071a);
    }

    #[test]
    fn vec_55_aa() {
        assert!(crc16_gsm(&[0x55, 0xaa]) == 0x1a15);
    }

    #[test]
    fn vec_deadbeef() {
        assert!(crc16_gsm(&[0xde, 0xad, 0xbe, 0xef]) == 0x3ba8);
    }

    #[test]
    fn vec_hello() {
        assert!(crc16_gsm(b"Hello") == 0x3429);
    }

    #[test]
    fn vec_fox() {
        assert!(crc16_gsm(b"The quick brown fox") == 0x5222);
    }

    #[test]
    fn vec_four_zeros() {
        assert!(crc16_gsm(&[0x00; 4]) == 0xffff);
    }

    #[test]
    fn vec_four_ff() {
        assert!(crc16_gsm(&[0xff; 4]) == 0x6630);
    }

    #[test]
    fn vec_0_15() {
        let mut data = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc16_gsm(&data) == 0xaec2);
    }

    #[test]
    fn vec_12345678() {
        assert!(crc16_gsm(&[0x12, 0x34, 0x56, 0x78]) == 0x4bd3);
    }

    #[test]
    fn vec_check() {
        assert!(crc16_gsm(b"123456789") == 0xce3c);
    }

    #[test]
    fn vec_a5_1000() {
        let data = [0xA5u8; 1000];
        assert!(crc16_gsm(&data) == 0xd5fb);
    }

    #[test]
    fn vec_all256() {
        let mut data = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc16_gsm(&data) == 0x81aa);
    }

    // --- Property / behaviour tests ---

    #[test]
    fn check_constant_matches() {
        // The documented `check` constant is the CRC of b"123456789".
        assert!(crc16_gsm(b"123456789") == 0xce3c);
    }

    #[test]
    fn determinism_empty() {
        assert!(crc16_gsm(b"") == crc16_gsm(b""));
    }

    #[test]
    fn determinism_check() {
        let first = crc16_gsm(b"123456789");
        let second = crc16_gsm(b"123456789");
        assert!(first == second);
    }

    #[test]
    fn determinism_many_iterations() {
        let expected = crc16_gsm(b"The quick brown fox");
        let mut i = 0u32;
        while i < 64 {
            assert!(crc16_gsm(b"The quick brown fox") == expected);
            i += 1;
        }
    }

    #[test]
    fn a_ne_b() {
        assert!(crc16_gsm(b"a") != crc16_gsm(b"b"));
    }

    #[test]
    fn prefix_differs() {
        // A prefix and its extension produce different checksums.
        assert!(crc16_gsm(b"a") != crc16_gsm(b"ab"));
        assert!(crc16_gsm(b"ab") != crc16_gsm(b"abc"));
    }

    #[test]
    fn order_sensitive() {
        assert!(crc16_gsm(&[0xaa, 0x55]) != crc16_gsm(&[0x55, 0xaa]));
    }

    #[test]
    fn length_sensitive_nonzero() {
        // Differing lengths of distinct data yield distinct checksums here.
        assert!(crc16_gsm(b"a") != crc16_gsm(b"aa"));
        assert!(crc16_gsm(b"aa") != crc16_gsm(b"aaa"));
    }

    #[test]
    fn all_zero_lengths_collapse() {
        // For GSM, all-zero inputs of several lengths map to the same value.
        let v = 0xffffu16;
        assert!(crc16_gsm(b"") == v);
        assert!(crc16_gsm(&[0x00]) == v);
        assert!(crc16_gsm(&[0x00, 0x00]) == v);
        assert!(crc16_gsm(&[0x00; 4]) == v);
    }

    #[test]
    fn single_byte_all_distinct() {
        // All 256 single-byte checksums are pairwise distinct.
        let mut results = [0u16; 256];
        let mut i = 0usize;
        while i < 256 {
            results[i] = crc16_gsm(&[i as u8]);
            i += 1;
        }
        let mut a = 0usize;
        while a < 256 {
            let mut b = a + 1;
            while b < 256 {
                assert!(results[a] != results[b]);
                b += 1;
            }
            a += 1;
        }
    }

    #[test]
    fn result_in_range() {
        // Every checksum is a valid 16-bit value.
        let mut i = 0usize;
        while i < 256 {
            let v = crc16_gsm(&[i as u8]);
            assert!((0..=0xffffu16).contains(&v));
            i += 1;
        }
    }

    #[test]
    fn deadbeef_order_matters() {
        assert!(crc16_gsm(&[0xde, 0xad, 0xbe, 0xef]) != crc16_gsm(&[0xef, 0xbe, 0xad, 0xde]));
    }

    #[test]
    fn ab_differs_from_ba() {
        assert!(crc16_gsm(b"ab") != crc16_gsm(b"ba"));
    }
}
