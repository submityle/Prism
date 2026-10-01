//! `CRC-24/FLEXRAY-B` bit-wise implementation.
//!
//! Parameters: width=24, poly=`0x5D6DCB`, init=`0xABCDEF`, refin=false,
//! refout=false, xorout=`0x0`, check=`0x1F23B8`.

const WIDTH: u32 = 24;
const POLY: u32 = 0x5D6DCB;
const INIT: u32 = 0xABCDEF;
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

/// Compute the `CRC-24/FLEXRAY-B` checksum of `data`.
pub fn crc24_flexray_b(data: &[u8]) -> u32 {
    let mut reg = INIT & MASK;
    let mut idx = 0usize;
    while idx < data.len() {
        let byte = data[idx];
        let b = if REFLECT_IN {
            reflect(u32::from(byte), 8)
        } else {
            u32::from(byte)
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
    use super::crc24_flexray_b;

    #[test]
    fn vector_empty() {
        assert!(crc24_flexray_b(b"") == 0xabcdef);
    }

    #[test]
    fn vector_z00() {
        assert!(crc24_flexray_b(&[0x00]) == 0x0ebdac);
    }

    #[test]
    fn vector_ff() {
        assert!(crc24_flexray_b(&[0xff]) == 0x712b9d);
    }

    #[test]
    fn vector_a() {
        assert!(crc24_flexray_b(b"a") == 0x7f1f72);
    }

    #[test]
    fn vector_b() {
        assert!(crc24_flexray_b(b"b") == 0x98a92f);
    }

    #[test]
    fn vector_ab() {
        assert!(crc24_flexray_b(b"ab") == 0x981b7e);
    }

    #[test]
    fn vector_abc() {
        assert!(crc24_flexray_b(b"abc") == 0x4c32d6);
    }

    #[test]
    fn vector_two_zeros() {
        assert!(crc24_flexray_b(&[0, 0]) == 0x7e18bf);
    }

    #[test]
    fn vector_01() {
        assert!(crc24_flexray_b(&[0x01]) == 0x53d067);
    }

    #[test]
    fn vector_02() {
        assert!(crc24_flexray_b(&[0x02]) == 0xb4663a);
    }

    #[test]
    fn vector_7f() {
        assert!(crc24_flexray_b(&[0x7f]) == 0x1fc051);
    }

    #[test]
    fn vector_80() {
        assert!(crc24_flexray_b(&[0x80]) == 0x605660);
    }

    #[test]
    fn vector_aa_55() {
        assert!(crc24_flexray_b(&[0xaa, 0x55]) == 0xaee206);
    }

    #[test]
    fn vector_55_aa() {
        assert!(crc24_flexray_b(&[0x55, 0xaa]) == 0x5638ca);
    }

    #[test]
    fn vector_deadbeef() {
        assert!(crc24_flexray_b(&[0xde, 0xad, 0xbe, 0xef]) == 0x68a33f);
    }

    #[test]
    fn vector_hello() {
        assert!(crc24_flexray_b(b"Hello") == 0xb3411b);
    }

    #[test]
    fn vector_fox() {
        assert!(crc24_flexray_b(b"The quick brown fox") == 0x9eb0ed);
    }

    #[test]
    fn vector_four_zeros() {
        assert!(crc24_flexray_b(&[0; 4]) == 0x13f29d);
    }

    #[test]
    fn vector_four_ff() {
        assert!(crc24_flexray_b(&[0xff; 4]) == 0x1a021c);
    }

    #[test]
    fn vector_0_15() {
        let mut data = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc24_flexray_b(&data) == 0xa4b54f);
    }

    #[test]
    fn vector_12345678() {
        assert!(crc24_flexray_b(&[0x12, 0x34, 0x56, 0x78]) == 0xeeffdb);
    }

    #[test]
    fn vector_check() {
        assert!(crc24_flexray_b(b"123456789") == 0x1f23b8);
    }

    #[test]
    fn vector_a5_1000() {
        let data = [0xA5u8; 1000];
        assert!(crc24_flexray_b(&data) == 0xf6baed);
    }

    #[test]
    fn vector_all256() {
        let mut data = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc24_flexray_b(&data) == 0xed7a27);
    }

    #[test]
    fn check_constant_matches() {
        const CHECK: u32 = 0x1f23b8;
        assert!(crc24_flexray_b(b"123456789") == CHECK);
    }

    #[test]
    fn determinism() {
        let first = crc24_flexray_b(b"The quick brown fox");
        let second = crc24_flexray_b(b"The quick brown fox");
        assert!(first == second);
    }

    #[test]
    fn order_sensitive() {
        let forward = crc24_flexray_b(&[0xaa, 0x55]);
        let reverse = crc24_flexray_b(&[0x55, 0xaa]);
        assert!(forward != reverse);
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc24_flexray_b(b"a") != crc24_flexray_b(b"b"));
    }

    #[test]
    fn prefix_differs() {
        assert!(crc24_flexray_b(b"ab") != crc24_flexray_b(b"abc"));
    }

    #[test]
    fn length_sensitive() {
        let one = crc24_flexray_b(&[0x00]);
        let two = crc24_flexray_b(&[0x00, 0x00]);
        assert!(one != two);
    }

    #[test]
    fn single_bytes_distinct() {
        let mut seen = [0u32; 256];
        let mut i = 0usize;
        while i < 256 {
            seen[i] = crc24_flexray_b(&[i as u8]);
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
    fn result_within_range() {
        let value = crc24_flexray_b(b"The quick brown fox");
        assert!((0..=0xff_ffffu32).contains(&value));
    }

    #[test]
    fn empty_within_range() {
        let value = crc24_flexray_b(b"");
        assert!((0..=0xff_ffffu32).contains(&value));
    }

    #[test]
    fn all256_within_range() {
        let mut data = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            data[i] = i as u8;
            i += 1;
        }
        let value = crc24_flexray_b(&data);
        assert!((0..=0xff_ffffu32).contains(&value));
    }

    #[test]
    fn large_buffer_within_range() {
        let data = [0xA5u8; 1000];
        let value = crc24_flexray_b(&data);
        assert!((0..=0xff_ffffu32).contains(&value));
    }

    #[test]
    fn two_zeros_matches_four_prefix() {
        assert!(crc24_flexray_b(&[0, 0]) != crc24_flexray_b(&[0; 4]));
    }
}
