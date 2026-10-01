//! `crc24_ble` reflected `CRC`-24 implemented with a bit-wise algorithm.
//!
//! Parameters: width 24, poly `0x00065B`, init `0x555555`, `refin` true,
//! `refout` true, `xorout` `0x0`, `check` `0xc25a56`. The register is a
//! `u32` masked to `0xff_ffff`.

const WIDTH: u32 = 24;
const POLY: u32 = 0x00065B;
const INIT: u32 = 0x555555;
const XOROUT: u32 = 0x0;
const REFLECT_IN: bool = true;
const REFLECT_OUT: bool = true;
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

pub fn crc24_ble(data: &[u8]) -> u32 {
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
    use super::crc24_ble;

    #[test]
    fn vector_empty() {
        assert!(crc24_ble(b"") == 0xaaaaaa);
    }

    #[test]
    fn vector_z00() {
        assert!(crc24_ble(&[0x00]) == 0xe29d2a);
    }

    #[test]
    fn vector_ff() {
        assert!(crc24_ble(&[0xff]) == 0x71b16a);
    }

    #[test]
    fn vector_a() {
        assert!(crc24_ble(b"a") == 0xb881ea);
    }

    #[test]
    fn vector_b() {
        assert!(crc24_ble(b"b") == 0xba5caa);
    }

    #[test]
    fn vector_ab() {
        assert!(crc24_ble(b"ab") == 0xd77e81);
    }

    #[test]
    fn vector_abc() {
        assert!(crc24_ble(b"abc") == 0x8276fe);
    }

    #[test]
    fn vector_two_zeros() {
        assert!(crc24_ble(&[0, 0]) == 0x38b51d);
    }

    #[test]
    fn vector_01() {
        assert!(crc24_ble(&[0x01]) == 0xe329ea);
    }

    #[test]
    fn vector_02() {
        assert!(crc24_ble(&[0x02]) == 0xe1f4aa);
    }

    #[test]
    fn vector_7f() {
        assert!(crc24_ble(&[0x7f]) == 0xabd16a);
    }

    #[test]
    fn vector_80() {
        assert!(crc24_ble(&[0x80]) == 0x38fd2a);
    }

    #[test]
    fn vector_aa_55() {
        assert!(crc24_ble(&[0xaa, 0x55]) == 0x932cea);
    }

    #[test]
    fn vector_55_aa() {
        assert!(crc24_ble(&[0x55, 0xaa]) == 0x6da386);
    }

    #[test]
    fn vector_deadbeef() {
        assert!(crc24_ble(&[0xde, 0xad, 0xbe, 0xef]) == 0x859f31);
    }

    #[test]
    fn vector_hello() {
        assert!(crc24_ble(b"Hello") == 0x11a59c);
    }

    #[test]
    fn vector_fox() {
        assert!(crc24_ble(b"The quick brown fox") == 0x2c5344);
    }

    #[test]
    fn vector_four_zeros() {
        assert!(crc24_ble(&[0u8; 4]) == 0x479275);
    }

    #[test]
    fn vector_four_ff() {
        assert!(crc24_ble(&[0xffu8; 4]) == 0x6a857a);
    }

    #[test]
    fn vector_0_15() {
        let mut buf = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc24_ble(&buf) == 0xe9cfe6);
    }

    #[test]
    fn vector_12345678() {
        assert!(crc24_ble(&[0x12, 0x34, 0x56, 0x78]) == 0x2eeefd);
    }

    #[test]
    fn vector_check() {
        assert!(crc24_ble(b"123456789") == 0xc25a56);
    }

    #[test]
    fn vector_a5_1000() {
        let buf = [0xA5u8; 1000];
        assert!(crc24_ble(&buf) == 0x36b205);
    }

    #[test]
    fn vector_all256() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc24_ble(&buf) == 0x62b6d7);
    }

    #[test]
    fn check_constant_matches_spec() {
        const CHECK: u32 = 0xc25a56;
        assert!(crc24_ble(b"123456789") == CHECK);
    }

    #[test]
    fn determinism() {
        let input = &[0xde, 0xad, 0xbe, 0xef];
        assert!(crc24_ble(input) == crc24_ble(input));
    }

    #[test]
    fn order_sensitive() {
        assert!(crc24_ble(&[0xaa, 0x55]) != crc24_ble(&[0x55, 0xaa]));
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc24_ble(b"a") != crc24_ble(b"b"));
    }

    #[test]
    fn prefix_differs() {
        assert!(crc24_ble(b"ab") != crc24_ble(b"abc"));
    }

    #[test]
    fn length_sensitive() {
        assert!(crc24_ble(&[0x00]) != crc24_ble(&[0x00, 0x00]));
    }

    #[test]
    fn empty_differs_from_single_zero() {
        assert!(crc24_ble(b"") != crc24_ble(&[0x00]));
    }

    #[test]
    fn single_bytes_pairwise_distinct() {
        let mut seen = [0u32; 256];
        let mut i = 0usize;
        while i < 256 {
            seen[i] = crc24_ble(&[i as u8]);
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
    fn result_within_range_vectors() {
        let inputs: [&[u8]; 5] = [b"", b"a", b"abc", b"123456789", &[0xff, 0x00, 0x55]];
        let mut i = 0usize;
        while i < inputs.len() {
            let v = crc24_ble(inputs[i]);
            assert!((0..=0xff_ffffu32).contains(&v));
            i += 1;
        }
    }

    #[test]
    fn result_within_range_all_single_bytes() {
        let mut i = 0usize;
        while i < 256 {
            let v = crc24_ble(&[i as u8]);
            assert!((0..=0xff_ffffu32).contains(&v));
            i += 1;
        }
    }

    #[test]
    fn two_zeros_differs_from_four_zeros() {
        assert!(crc24_ble(&[0u8; 2]) != crc24_ble(&[0u8; 4]));
    }
}
