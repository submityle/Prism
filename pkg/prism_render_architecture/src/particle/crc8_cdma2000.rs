//! `CRC-8/CDMA2000` checksum implemented bit-wise for the particle subsystem.
//!
//! Parameters: `width`=8, `poly`=0x9B, `init`=0xFF, `refin`=false,
//! `refout`=false, `xorout`=0x0, `check`=0xDA. Returns a `u8` register.

const WIDTH: u32 = 8;
const POLY: u32 = 0x9B;
const INIT: u32 = 0xFF;
const XOROUT: u32 = 0x0;
const REFLECT_IN: bool = false;
const REFLECT_OUT: bool = false;
const MASK: u32 = 0xff;
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

pub fn crc8_cdma2000(data: &[u8]) -> u8 {
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
    ((reg ^ XOROUT) & MASK) as u8
}

#[cfg(test)]
mod tests {
    use super::crc8_cdma2000;

    #[test]
    fn vector_empty() {
        assert!(crc8_cdma2000(b"") == 0xff);
    }

    #[test]
    fn vector_z00() {
        assert!(crc8_cdma2000(&[0x00]) == 0x7b);
    }

    #[test]
    fn vector_ff() {
        assert!(crc8_cdma2000(&[0xff]) == 0x00);
    }

    #[test]
    fn vector_a() {
        assert!(crc8_cdma2000(b"a") == 0x4c);
    }

    #[test]
    fn vector_b() {
        assert!(crc8_cdma2000(b"b") == 0x7a);
    }

    #[test]
    fn vector_ab() {
        assert!(crc8_cdma2000(b"ab") == 0x11);
    }

    #[test]
    fn vector_abc() {
        assert!(crc8_cdma2000(b"abc") == 0x33);
    }

    #[test]
    fn vector_two_zeros() {
        assert!(crc8_cdma2000(&[0, 0]) == 0xb1);
    }

    #[test]
    fn vector_01() {
        assert!(crc8_cdma2000(&[0x01]) == 0xe0);
    }

    #[test]
    fn vector_02() {
        assert!(crc8_cdma2000(&[0x02]) == 0xd6);
    }

    #[test]
    fn vector_7f() {
        assert!(crc8_cdma2000(&[0x7f]) == 0x0b);
    }

    #[test]
    fn vector_80() {
        assert!(crc8_cdma2000(&[0x80]) == 0x70);
    }

    #[test]
    fn vector_aa_55() {
        assert!(crc8_cdma2000(&[0xaa, 0x55]) == 0xcf);
    }

    #[test]
    fn vector_55_aa() {
        assert!(crc8_cdma2000(&[0x55, 0xaa]) == 0x05);
    }

    #[test]
    fn vector_deadbeef() {
        assert!(crc8_cdma2000(&[0xde, 0xad, 0xbe, 0xef]) == 0x77);
    }

    #[test]
    fn vector_hello() {
        assert!(crc8_cdma2000(b"Hello") == 0x06);
    }

    #[test]
    fn vector_fox() {
        assert!(crc8_cdma2000(b"The quick brown fox") == 0x23);
    }

    #[test]
    fn vector_four_zeros() {
        assert!(crc8_cdma2000(&[0u8; 4]) == 0xaf);
    }

    #[test]
    fn vector_four_ff() {
        assert!(crc8_cdma2000(&[0xffu8; 4]) == 0x0c);
    }

    #[test]
    fn vector_0_15() {
        let mut buf = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc8_cdma2000(&buf) == 0xd8);
    }

    #[test]
    fn vector_12345678() {
        assert!(crc8_cdma2000(&[0x12, 0x34, 0x56, 0x78]) == 0xa7);
    }

    #[test]
    fn vector_check() {
        assert!(crc8_cdma2000(b"123456789") == 0xda);
    }

    #[test]
    fn vector_a5_1000() {
        let buf = [0xA5u8; 1000];
        assert!(crc8_cdma2000(&buf) == 0x7d);
    }

    #[test]
    fn vector_all256() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc8_cdma2000(&buf) == 0x41);
    }

    #[test]
    fn check_constant_matches() {
        assert!(crc8_cdma2000(b"123456789") == 0xda);
    }

    #[test]
    fn determinism_repeated() {
        let input = b"determinism";
        let first = crc8_cdma2000(input);
        let second = crc8_cdma2000(input);
        assert!(first == second);
    }

    #[test]
    fn order_sensitive() {
        assert!(crc8_cdma2000(&[0xaa, 0x55]) != crc8_cdma2000(&[0x55, 0xaa]));
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc8_cdma2000(b"a") != crc8_cdma2000(b"b"));
    }

    #[test]
    fn prefix_differs() {
        assert!(crc8_cdma2000(b"ab") != crc8_cdma2000(b"abc"));
    }

    #[test]
    fn length_sensitive() {
        assert!(crc8_cdma2000(&[0x00]) != crc8_cdma2000(&[0x00, 0x00]));
    }

    #[test]
    fn single_bytes_distinct() {
        let mut seen = [false; 256];
        let mut i = 0u32;
        while i < 256 {
            let v = crc8_cdma2000(&[i as u8]);
            let slot = v as usize;
            assert!(!seen[slot]);
            seen[slot] = true;
            i += 1;
        }
    }

    #[test]
    fn single_byte_pairs_separable() {
        let mut i = 0u32;
        while i < 256 {
            let mut j = i + 1;
            while j < 256 {
                assert!(crc8_cdma2000(&[i as u8]) != crc8_cdma2000(&[j as u8]));
                j += 1;
            }
            i += 1;
        }
    }

    #[test]
    fn result_in_range() {
        let v = crc8_cdma2000(b"range");
        assert!((0..=0xffu8).contains(&v));
    }

    #[test]
    fn empty_in_range() {
        let v = crc8_cdma2000(b"");
        assert!((0..=0xffu8).contains(&v));
    }
    #[test]
    fn all256_in_range() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        let v = crc8_cdma2000(&buf);
        assert!((0..=0xffu8).contains(&v));
    }
}
