//! `CRC-24` FlexRay-A checksum over byte slices.
//!
//! Parameters: width 24, poly `0x5D6DCB`, init `0xFEDCBA`, `refin`=false,
//! `refout`=false, `xorout`=`0x0`, check `0x7979BD`. The result is a `u32`
//! masked to 24 bits (`MASK` = `0xff_ffff`).

const WIDTH: u32 = 24;
const POLY: u32 = 0x5D6DCB;
const INIT: u32 = 0xFEDCBA;
const XOROUT: u32 = 0x0;
const REFLECT_IN: bool = false;
const REFLECT_OUT: bool = false;
const MASK: u32 = 0xff_ffff;
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

/// Compute the `CRC-24` FlexRay-A checksum of `data`.
pub fn crc24_flexray_a(data: &[u8]) -> u32 {
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
    (reg ^ XOROUT) & MASK
}

#[cfg(test)]
mod tests {
    use super::crc24_flexray_a;

    #[test]
    fn vector_empty() {
        assert!(crc24_flexray_a(b"") == 0xfedcba);
    }

    #[test]
    fn vector_z00() {
        assert!(crc24_flexray_a(&[0x00]) == 0xfe41fa);
    }

    #[test]
    fn vector_ff() {
        assert!(crc24_flexray_a(&[0xff]) == 0x81d7cb);
    }

    #[test]
    fn vector_a() {
        assert!(crc24_flexray_a(b"a") == 0x8fe324);
    }

    #[test]
    fn vector_b() {
        assert!(crc24_flexray_a(b"b") == 0x685579);
    }

    #[test]
    fn vector_ab() {
        assert!(crc24_flexray_a(b"ab") == 0x85023b);
    }

    #[test]
    fn vector_abc() {
        assert!(crc24_flexray_a(b"abc") == 0xd21ea8);
    }

    #[test]
    fn vector_two_zeros() {
        assert!(crc24_flexray_a(&[0, 0]) == 0x6301fa);
    }

    #[test]
    fn vector_01() {
        assert!(crc24_flexray_a(&[0x01]) == 0xa32c31);
    }

    #[test]
    fn vector_02() {
        assert!(crc24_flexray_a(&[0x02]) == 0x449a6c);
    }

    #[test]
    fn vector_7f() {
        assert!(crc24_flexray_a(&[0x7f]) == 0xef3c07);
    }

    #[test]
    fn vector_80() {
        assert!(crc24_flexray_a(&[0x80]) == 0x90aa36);
    }

    #[test]
    fn vector_aa_55() {
        assert!(crc24_flexray_a(&[0xaa, 0x55]) == 0xb3fb43);
    }

    #[test]
    fn vector_55_aa() {
        assert!(crc24_flexray_a(&[0x55, 0xaa]) == 0x4b218f);
    }

    #[test]
    fn vector_deadbeef() {
        assert!(crc24_flexray_a(&[0xde, 0xad, 0xbe, 0xef]) == 0x4ae9d0);
    }

    #[test]
    fn vector_hello() {
        assert!(crc24_flexray_a(b"Hello") == 0x58cf7e);
    }

    #[test]
    fn vector_fox() {
        assert!(crc24_flexray_a(b"The quick brown fox") == 0x5b70ed);
    }

    #[test]
    fn vector_four_zeros() {
        assert!(crc24_flexray_a(&[0; 4]) == 0x31b872);
    }

    #[test]
    fn vector_four_ff() {
        assert!(crc24_flexray_a(&[0xff; 4]) == 0x3848f3);
    }

    #[test]
    fn vector_0_15() {
        let mut data = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc24_flexray_a(&data) == 0x30074e);
    }

    #[test]
    fn vector_12345678() {
        assert!(crc24_flexray_a(&[0x12, 0x34, 0x56, 0x78]) == 0xccb534);
    }

    #[test]
    fn vector_check() {
        assert!(crc24_flexray_a(b"123456789") == 0x7979bd);
    }

    #[test]
    fn vector_a5_1000() {
        let data = [0xA5u8; 1000];
        assert!(crc24_flexray_a(&data) == 0x298836);
    }

    #[test]
    fn vector_all256() {
        let mut data = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc24_flexray_a(&data) == 0x47588d);
    }

    #[test]
    fn check_constant_matches() {
        let check: u32 = 0x7979bd;
        assert!(crc24_flexray_a(b"123456789") == check);
    }

    #[test]
    fn determinism() {
        let data = [0x11u8, 0x22, 0x33, 0x44];
        assert!(crc24_flexray_a(&data) == crc24_flexray_a(&data));
    }

    #[test]
    fn order_sensitive() {
        assert!(crc24_flexray_a(&[0xaa, 0x55]) != crc24_flexray_a(&[0x55, 0xaa]));
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc24_flexray_a(b"a") != crc24_flexray_a(b"b"));
    }

    #[test]
    fn prefix_differs() {
        assert!(crc24_flexray_a(b"abc") != crc24_flexray_a(b"ab"));
    }

    #[test]
    fn length_sensitive() {
        assert!(crc24_flexray_a(&[0x00]) != crc24_flexray_a(&[0x00, 0x00]));
    }

    #[test]
    fn single_bytes_distinct() {
        let mut seen = [0u32; 256];
        let mut i = 0usize;
        while i < 256 {
            seen[i] = crc24_flexray_a(&[i as u8]);
            i += 1;
        }
        let mut a = 0usize;
        while a < 256 {
            let mut b = a + 1;
            while b < 256 {
                assert!(seen[a] != seen[b]);
                b += 1;
            }
            a += 1;
        }
    }

    #[test]
    fn result_in_range() {
        let data = [0x12u8, 0x34, 0x56, 0x78];
        let v = crc24_flexray_a(&data);
        assert!((0..=0xff_ffffu32).contains(&v));
    }

    #[test]
    fn empty_in_range() {
        let v = crc24_flexray_a(b"");
        assert!((0..=0xff_ffffu32).contains(&v));
    }

    #[test]
    fn check_in_range() {
        let v = crc24_flexray_a(b"123456789");
        assert!((0..=0xff_ffffu32).contains(&v));
    }

    #[test]
    fn two_distinct_inputs() {
        assert!(crc24_flexray_a(b"abc") != crc24_flexray_a(b"abd"));
    }

    #[test]
    fn zero_vs_one() {
        assert!(crc24_flexray_a(&[0x00]) != crc24_flexray_a(&[0x01]));
    }
}
