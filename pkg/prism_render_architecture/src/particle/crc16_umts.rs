//! `crc16_umts` bit-wise CRC implementation.
//!
//! Parameters: `width`=16, `poly`=0x8005, `init`=0x0, `refin`=false,
//! `refout`=false, `xorout`=0x0, `check`=0xfee8.

const WIDTH: u32 = 16;
const POLY: u32 = 0x8005;
const INIT: u32 = 0x0;
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

pub fn crc16_umts(data: &[u8]) -> u16 {
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
    use super::crc16_umts;

    #[test]
    fn vec_empty() {
        assert!(crc16_umts(b"") == 0x0000);
    }

    #[test]
    fn vec_z00() {
        assert!(crc16_umts(&[0x00]) == 0x0000);
    }

    #[test]
    fn vec_ff() {
        assert!(crc16_umts(&[0xff]) == 0x0202);
    }

    #[test]
    fn vec_a() {
        assert!(crc16_umts(b"a") == 0x8145);
    }

    #[test]
    fn vec_b() {
        assert!(crc16_umts(b"b") == 0x814f);
    }

    #[test]
    fn vec_ab() {
        assert!(crc16_umts(b"ab") == 0xc749);
    }

    #[test]
    fn vec_abc() {
        assert!(crc16_umts(b"abc") == 0xcadb);
    }

    #[test]
    fn vec_two_zeros() {
        assert!(crc16_umts(&[0, 0]) == 0x0000);
    }

    #[test]
    fn vec_01() {
        assert!(crc16_umts(&[0x01]) == 0x8005);
    }

    #[test]
    fn vec_02() {
        assert!(crc16_umts(&[0x02]) == 0x800f);
    }

    #[test]
    fn vec_7f() {
        assert!(crc16_umts(&[0x7f]) == 0x8101);
    }

    #[test]
    fn vec_80() {
        assert!(crc16_umts(&[0x80]) == 0x8303);
    }

    #[test]
    fn vec_aa_55() {
        assert!(crc16_umts(&[0xaa, 0x55]) == 0xfdf4);
    }

    #[test]
    fn vec_55_aa() {
        assert!(crc16_umts(&[0x55, 0xaa]) == 0x7df9);
    }

    #[test]
    fn vec_deadbeef() {
        assert!(crc16_umts(&[0xde, 0xad, 0xbe, 0xef]) == 0x962b);
    }

    #[test]
    fn vec_hello() {
        assert!(crc16_umts(b"Hello") == 0xb7c6);
    }

    #[test]
    fn vec_fox() {
        assert!(crc16_umts(b"The quick brown fox") == 0x9051);
    }

    #[test]
    fn vec_four_zeros() {
        assert!(crc16_umts(&[0u8; 4]) == 0x0000);
    }

    #[test]
    fn vec_four_ff() {
        assert!(crc16_umts(&[0xffu8; 4]) == 0x8029);
    }

    #[test]
    fn vec_0_15() {
        let mut data = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc16_umts(&data) == 0x7f43);
    }

    #[test]
    fn vec_12345678() {
        assert!(crc16_umts(&[0x12, 0x34, 0x56, 0x78]) == 0x1e83);
    }

    #[test]
    fn vec_check() {
        assert!(crc16_umts(b"123456789") == 0xfee8);
    }

    #[test]
    fn vec_a5_1000() {
        let data = [0xA5u8; 1000];
        assert!(crc16_umts(&data) == 0xda13);
    }

    #[test]
    fn vec_all256() {
        let mut data = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc16_umts(&data) == 0x3b7a);
    }

    #[test]
    fn check_constant_matches() {
        const CHECK: u16 = 0xfee8;
        assert!(crc16_umts(b"123456789") == CHECK);
    }

    #[test]
    fn determinism() {
        let input = b"determinism-sample";
        let first = crc16_umts(input);
        let second = crc16_umts(input);
        assert!(first == second);
    }

    #[test]
    fn order_sensitive() {
        let forward = crc16_umts(&[0x01, 0x02, 0x03]);
        let reversed = crc16_umts(&[0x03, 0x02, 0x01]);
        assert!(forward != reversed);
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc16_umts(b"a") != crc16_umts(b"b"));
    }

    #[test]
    fn prefix_differs() {
        let short = crc16_umts(b"abc");
        let long = crc16_umts(b"abcd");
        assert!(short != long);
    }

    #[test]
    fn length_sensitive() {
        let one = crc16_umts(&[0x00]);
        let two = crc16_umts(&[0x00, 0x00]);
        let three = crc16_umts(&[0x00, 0x00, 0x00]);
        assert!(one == two);
        assert!(two == three);
    }

    #[test]
    fn length_sensitive_nonzero() {
        let one = crc16_umts(&[0xffu8; 1]);
        let two = crc16_umts(&[0xffu8; 2]);
        assert!(one != two);
    }

    #[test]
    fn single_bytes_pairwise_distinct() {
        let mut results = [0u16; 256];
        let mut i = 0usize;
        while i < 256 {
            results[i] = crc16_umts(&[i as u8]);
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
    fn result_within_range() {
        let value = crc16_umts(b"range-check-sample");
        assert!((0..=0xffffu16).contains(&value));
    }

    #[test]
    fn result_within_range_all_single_bytes() {
        let mut i = 0usize;
        while i < 256 {
            let value = crc16_umts(&[i as u8]);
            assert!((0..=0xffffu16).contains(&value));
            i += 1;
        }
    }

    #[test]
    fn empty_is_init_derived() {
        assert!(crc16_umts(&[]) == 0x0000);
    }

    #[test]
    fn ab_differs_from_ba() {
        assert!(crc16_umts(b"ab") != crc16_umts(b"ba"));
    }
}
