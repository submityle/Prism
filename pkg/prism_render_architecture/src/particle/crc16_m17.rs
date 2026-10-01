//! `CRC-16/M17` bit-wise reference implementation.
//!
//! Parameters: width=16, poly=`0x5935`, init=`0xffff`, `refin`=false,
//! `refout`=false, `xorout`=`0x0`, check=`0x772b`.

const WIDTH: u32 = 16;
const POLY: u32 = 0x5935;
const INIT: u32 = 0xffff;
const XOROUT: u32 = 0x0;
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

/// Compute the `CRC-16/M17` checksum of `data`.
pub fn crc16_m17(data: &[u8]) -> u16 {
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
    use super::crc16_m17;

    #[test]
    fn vector_empty() {
        assert!(crc16_m17(b"") == 0xffff);
    }

    #[test]
    fn vector_z00() {
        assert!(crc16_m17(&[0x00]) == 0x4c14);
    }

    #[test]
    fn vector_ff() {
        assert!(crc16_m17(&[0xff]) == 0xff00);
    }

    #[test]
    fn vector_a() {
        assert!(crc16_m17(b"a") == 0x9653);
    }

    #[test]
    fn vector_b() {
        assert!(crc16_m17(b"b") == 0x7d0c);
    }

    #[test]
    fn vector_ab() {
        assert!(crc16_m17(b"ab") == 0x7089);
    }

    #[test]
    fn vector_abc() {
        assert!(crc16_m17(b"abc") == 0x95db);
    }

    #[test]
    fn vector_two_zeros() {
        assert!(crc16_m17(&[0, 0]) == 0x676c);
    }

    #[test]
    fn vector_01() {
        assert!(crc16_m17(&[0x01]) == 0x1521);
    }

    #[test]
    fn vector_02() {
        assert!(crc16_m17(&[0x02]) == 0xfe7e);
    }

    #[test]
    fn vector_7f() {
        assert!(crc16_m17(&[0x7f]) == 0x959e);
    }

    #[test]
    fn vector_80() {
        assert!(crc16_m17(&[0x80]) == 0x268a);
    }

    #[test]
    fn vector_aa_55() {
        assert!(crc16_m17(&[0xaa, 0x55]) == 0x5923);
    }

    #[test]
    fn vector_55_aa() {
        assert!(crc16_m17(&[0x55, 0xaa]) == 0x3e4f);
    }

    #[test]
    fn vector_deadbeef() {
        assert!(crc16_m17(&[0xde, 0xad, 0xbe, 0xef]) == 0x7b8b);
    }

    #[test]
    fn vector_hello() {
        assert!(crc16_m17(b"Hello") == 0x3f21);
    }

    #[test]
    fn vector_fox() {
        assert!(crc16_m17(b"The quick brown fox") == 0xa232);
    }

    #[test]
    fn vector_four_zeros() {
        assert!(crc16_m17(&[0u8; 4]) == 0xaf4e);
    }

    #[test]
    fn vector_four_ff() {
        assert!(crc16_m17(&[0xffu8; 4]) == 0x676c);
    }

    #[test]
    fn vector_0_15() {
        let mut data = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc16_m17(&data) == 0xc632);
    }

    #[test]
    fn vector_12345678() {
        assert!(crc16_m17(&[0x12, 0x34, 0x56, 0x78]) == 0xe66b);
    }

    #[test]
    fn vector_check() {
        assert!(crc16_m17(b"123456789") == 0x772b);
    }

    #[test]
    fn vector_a5_1000() {
        let data = [0xA5u8; 1000];
        assert!(crc16_m17(&data) == 0x948b);
    }

    #[test]
    fn vector_all256() {
        let mut data = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc16_m17(&data) == 0x1c31);
    }

    #[test]
    fn check_constant_matches_spec() {
        const CHECK: u16 = 0x772b;
        assert!(crc16_m17(b"123456789") == CHECK);
    }

    #[test]
    fn determinism() {
        let first = crc16_m17(b"The quick brown fox");
        let second = crc16_m17(b"The quick brown fox");
        assert!(first == second);
    }

    #[test]
    fn order_sensitive() {
        assert!(crc16_m17(&[0xaa, 0x55]) != crc16_m17(&[0x55, 0xaa]));
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc16_m17(b"a") != crc16_m17(b"b"));
    }

    #[test]
    fn prefix_differs() {
        assert!(crc16_m17(b"ab") != crc16_m17(b"abc"));
    }

    #[test]
    fn length_sensitive() {
        assert!(crc16_m17(&[0u8; 2]) != crc16_m17(&[0u8; 4]));
    }

    #[test]
    fn single_bytes_pairwise_distinct() {
        let mut seen = [0u16; 256];
        let mut i = 0usize;
        while i < 256 {
            seen[i] = crc16_m17(&[i as u8]);
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
    fn result_within_u16_range() {
        let value = crc16_m17(b"123456789");
        assert!((0..=0xffffu16).contains(&value));
    }

    #[test]
    fn empty_within_range() {
        let value = crc16_m17(b"");
        assert!((0..=0xffffu16).contains(&value));
    }

    #[test]
    fn two_zeros_matches_four_ff() {
        assert!(crc16_m17(&[0, 0]) == crc16_m17(&[0xffu8; 4]));
    }

    #[test]
    fn single_zero_differs_from_double_zero() {
        assert!(crc16_m17(&[0x00]) != crc16_m17(&[0, 0]));
    }
}
