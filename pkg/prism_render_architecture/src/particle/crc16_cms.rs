//! `CRC-16/CMS` checksum (`width=16`, `poly=0x8005`, `init=0xffff`,
//! `refin=false`, `refout=false`, `xorout=0x0`, `check=0xaee7`).
//!
//! Bit-wise reference implementation operating on raw byte slices. The
//! algorithm feeds each input byte most-significant-bit first through the
//! polynomial division register and returns the final 16-bit remainder.

const WIDTH: u32 = 16;
const POLY: u32 = 0x8005;
const INIT: u32 = 0xFFFF;
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

/// Compute the `CRC-16/CMS` checksum of `data`.
pub fn crc16_cms(data: &[u8]) -> u16 {
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
    use super::crc16_cms;

    // ---- 24 reference-vector anchors (hard truths) ----

    #[test]
    fn vec_empty() {
        assert!(crc16_cms(b"") == 0xffff);
    }

    #[test]
    fn vec_z00() {
        assert!(crc16_cms(&[0x00]) == 0xfd02);
    }

    #[test]
    fn vec_ff() {
        assert!(crc16_cms(&[0xff]) == 0xff00);
    }

    #[test]
    fn vec_a() {
        assert!(crc16_cms(b"a") == 0x7c47);
    }

    #[test]
    fn vec_b() {
        assert!(crc16_cms(b"b") == 0x7c4d);
    }

    #[test]
    fn vec_ab() {
        assert!(crc16_cms(b"ab") == 0x4744);
    }

    #[test]
    fn vec_abc() {
        assert!(crc16_cms(b"abc") == 0x44d8);
    }

    #[test]
    fn vec_two_zeros() {
        assert!(crc16_cms(&[0x00, 0x00]) == 0x800d);
    }

    #[test]
    fn vec_01() {
        assert!(crc16_cms(&[0x01]) == 0x7d07);
    }

    #[test]
    fn vec_02() {
        assert!(crc16_cms(&[0x02]) == 0x7d0d);
    }

    #[test]
    fn vec_7f() {
        assert!(crc16_cms(&[0x7f]) == 0x7c03);
    }

    #[test]
    fn vec_80() {
        assert!(crc16_cms(&[0x80]) == 0x7e01);
    }

    #[test]
    fn vec_aa_55() {
        assert!(crc16_cms(&[0xaa, 0x55]) == 0x7df9);
    }

    #[test]
    fn vec_55_aa() {
        assert!(crc16_cms(&[0x55, 0xaa]) == 0xfdf4);
    }

    #[test]
    fn vec_deadbeef() {
        assert!(crc16_cms(&[0xde, 0xad, 0xbe, 0xef]) == 0x960f);
    }

    #[test]
    fn vec_hello() {
        assert!(crc16_cms(b"Hello") == 0x93c6);
    }

    #[test]
    fn vec_fox() {
        assert!(crc16_cms(b"The quick brown fox") == 0xb475);
    }

    #[test]
    fn vec_four_zeros() {
        assert!(crc16_cms(&[0x00; 4]) == 0x0024);
    }

    #[test]
    fn vec_four_ff() {
        assert!(crc16_cms(&[0xff; 4]) == 0x800d);
    }

    #[test]
    fn vec_0_15() {
        let mut data = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc16_cms(&data) == 0x024c);
    }

    #[test]
    fn vec_12345678() {
        assert!(crc16_cms(&[0x12, 0x34, 0x56, 0x78]) == 0x1ea7);
    }

    #[test]
    fn vec_check() {
        assert!(crc16_cms(b"123456789") == 0xaee7);
    }

    #[test]
    fn vec_a5_1000() {
        let data = [0xA5u8; 1000];
        assert!(crc16_cms(&data) == 0xf0c3);
    }

    #[test]
    fn vec_all256() {
        let mut data = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc16_cms(&data) == 0xc65c);
    }

    // ---- Property / structural tests ----

    #[test]
    fn check_constant_matches() {
        const CHECK: u16 = 0xaee7;
        assert!(crc16_cms(b"123456789") == CHECK);
    }

    #[test]
    fn empty_equals_init() {
        // With `xorout=0x0` and no input, the result is the raw `init` value.
        assert!(crc16_cms(b"") == 0xffff);
    }

    #[test]
    fn determinism_same_input() {
        let first = crc16_cms(b"The quick brown fox");
        let second = crc16_cms(b"The quick brown fox");
        assert!(first == second);
    }

    #[test]
    fn order_sensitive_aa55_vs_55aa() {
        assert!(crc16_cms(&[0xaa, 0x55]) != crc16_cms(&[0x55, 0xaa]));
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc16_cms(b"a") != crc16_cms(b"b"));
    }

    #[test]
    fn prefix_a_vs_ab_differ() {
        assert!(crc16_cms(b"a") != crc16_cms(b"ab"));
    }

    #[test]
    fn length_sensitive_zeros() {
        assert!(crc16_cms(&[0x00]) != crc16_cms(&[0x00, 0x00]));
        assert!(crc16_cms(&[0x00, 0x00]) != crc16_cms(&[0x00; 4]));
    }

    #[test]
    fn abc_vs_ab_differ() {
        assert!(crc16_cms(b"abc") != crc16_cms(b"ab"));
    }

    #[test]
    fn hello_vs_fox_differ() {
        assert!(crc16_cms(b"Hello") != crc16_cms(b"The quick brown fox"));
    }

    #[test]
    fn single_byte_01_vs_02_differ() {
        assert!(crc16_cms(&[0x01]) != crc16_cms(&[0x02]));
    }

    #[test]
    fn single_byte_7f_vs_80_differ() {
        assert!(crc16_cms(&[0x7f]) != crc16_cms(&[0x80]));
    }

    #[test]
    fn single_byte_all_distinct() {
        let mut table = [0u16; 256];
        let mut i = 0usize;
        while i < 256 {
            table[i] = crc16_cms(&[i as u8]);
            i += 1;
        }
        let mut a = 0usize;
        while a < 256 {
            let mut b = a + 1;
            while b < 256 {
                assert!(table[a] != table[b]);
                b += 1;
            }
            a += 1;
        }
    }

    #[test]
    fn results_within_u16_range() {
        let inputs: [&[u8]; 5] = [b"", b"a", b"abc", b"123456789", &[0xff; 4]];
        let mut i = 0usize;
        while i < inputs.len() {
            let value = crc16_cms(inputs[i]);
            assert!((0..=0xffffu16).contains(&value));
            i += 1;
        }
    }
}
