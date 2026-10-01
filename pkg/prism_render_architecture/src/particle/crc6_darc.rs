//! Pure-integer `CRC-6/DARC` gold standard for `no_std` plus `alloc` targets.

const WIDTH: u32 = 6;
const POLY: u32 = 0x19;
const INIT: u32 = 0x0;
const XOROUT: u32 = 0x0;
const REFLECT_IN: bool = true;
const REFLECT_OUT: bool = true;
const MASK: u32 = 0x3f;
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

pub fn crc6_darc(data: &[u8]) -> u8 {
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
    use super::*;

    #[test]
    fn anchor_80() {
        assert!(crc6_darc(&[0x80]) == 0x26);
    }

    #[test]
    fn anchor_12345678() {
        assert!(crc6_darc(&[0x12, 0x34, 0x56, 0x78]) == 0x23);
    }

    #[test]
    fn anchor_empty() {
        assert!(crc6_darc(b"") == 0x00);
    }

    #[test]
    fn anchor_z00() {
        assert!(crc6_darc(&[0x00]) == 0x00);
    }

    #[test]
    fn anchor_ff() {
        assert!(crc6_darc(&[0xff]) == 0x11);
    }

    #[test]
    fn anchor_a() {
        assert!(crc6_darc(b"a") == 0x0e);
    }

    #[test]
    fn anchor_b() {
        assert!(crc6_darc(b"b") == 0x15);
    }

    #[test]
    fn anchor_ab() {
        assert!(crc6_darc(b"ab") == 0x1d);
    }

    #[test]
    fn anchor_abc() {
        assert!(crc6_darc(b"abc") == 0x05);
    }

    #[test]
    fn anchor_two_zeros() {
        assert!(crc6_darc(&[0, 0]) == 0x00);
    }

    #[test]
    fn anchor_01() {
        assert!(crc6_darc(&[0x01]) == 0x32);
    }

    #[test]
    fn anchor_02() {
        assert!(crc6_darc(&[0x02]) == 0x29);
    }

    #[test]
    fn anchor_7f() {
        assert!(crc6_darc(&[0x7f]) == 0x37);
    }

    #[test]
    fn anchor_aa_55() {
        assert!(crc6_darc(&[0xaa, 0x55]) == 0x36);
    }

    #[test]
    fn anchor_55_aa() {
        assert!(crc6_darc(&[0x55, 0xaa]) == 0x24);
    }

    #[test]
    fn anchor_deadbeef() {
        assert!(crc6_darc(&[0xde, 0xad, 0xbe, 0xef]) == 0x05);
    }

    #[test]
    fn anchor_hello() {
        assert!(crc6_darc(b"Hello") == 0x06);
    }

    #[test]
    fn anchor_fox() {
        assert!(crc6_darc(b"The quick brown fox") == 0x15);
    }

    #[test]
    fn anchor_four_zeros() {
        assert!(crc6_darc(&[0u8; 4]) == 0x00);
    }

    #[test]
    fn anchor_four_ff() {
        assert!(crc6_darc(&[0xffu8; 4]) == 0x1d);
    }

    #[test]
    fn anchor_0_15() {
        let data = [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        assert!(crc6_darc(&data) == 0x18);
    }

    #[test]
    fn anchor_check() {
        assert!(crc6_darc(b"123456789") == 0x26);
    }

    #[test]
    fn anchor_a5_1000() {
        let data = [0xA5u8; 1000];
        assert!(crc6_darc(&data) == 0x0c);
    }

    #[test]
    fn anchor_all256() {
        let mut data = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc6_darc(&data) == 0x33);
    }

    #[test]
    fn determinism_empty() {
        assert!(crc6_darc(b"") == crc6_darc(b""));
    }

    #[test]
    fn determinism_fox() {
        let first = crc6_darc(b"The quick brown fox");
        let second = crc6_darc(b"The quick brown fox");
        assert!(first == second);
    }

    #[test]
    fn order_sensitivity() {
        let d1 = crc6_darc(b"ab") != crc6_darc(b"ba");
        let d2 = crc6_darc(&[0x12, 0x34]) != crc6_darc(&[0x34, 0x12]);
        assert!(d1 || d2);
    }

    #[test]
    fn a_neq_b() {
        assert!(crc6_darc(b"a") != crc6_darc(b"b"));
    }

    #[test]
    fn prefix_differs() {
        assert!(crc6_darc(b"abc") != crc6_darc(b"ab"));
    }

    #[test]
    fn prefix_abc_vs_a() {
        assert!(crc6_darc(b"abc") != crc6_darc(b"a"));
    }

    #[test]
    fn length_sensitive() {
        assert!(crc6_darc(&[0xff]) != crc6_darc(&[0xffu8; 4]));
    }

    #[test]
    fn single_byte_in_range_and_varied() {
        let mut seen = [false; 64];
        let mut distinct = 0usize;
        let mut i = 0usize;
        while i < 256 {
            let r = crc6_darc(&[i as u8]);
            assert!((0..=0x3f).contains(&r));
            if !seen[r as usize] {
                seen[r as usize] = true;
                distinct += 1;
            }
            i += 1;
        }
        assert!(distinct > 1);
    }

    #[test]
    fn check_constant_is_0x26() {
        assert!(crc6_darc(b"123456789") == 0x26);
    }

    #[test]
    fn all_anchor_results_in_range() {
        let r0 = crc6_darc(&[0x80]);
        let r1 = crc6_darc(&[0x12, 0x34, 0x56, 0x78]);
        let r2 = crc6_darc(b"Hello");
        let r3 = crc6_darc(&[0xde, 0xad, 0xbe, 0xef]);
        let r4 = crc6_darc(b"The quick brown fox");
        assert!((0..=0x3f).contains(&r0));
        assert!((0..=0x3f).contains(&r1));
        assert!((0..=0x3f).contains(&r2));
        assert!((0..=0x3f).contains(&r3));
        assert!((0..=0x3f).contains(&r4));
    }

    #[test]
    fn empty_is_zero() {
        assert!(crc6_darc(b"") == 0x00);
    }

    #[test]
    fn zeros_all_zero() {
        assert!(crc6_darc(&[0x00]) == 0x00);
        assert!(crc6_darc(&[0, 0]) == 0x00);
        assert!(crc6_darc(&[0u8; 4]) == 0x00);
    }

    #[test]
    fn reflect_known_and_involutive() {
        assert!(reflect(1, 6) == 0x20);
        assert!(reflect(0x20, 6) == 1);
        assert!(reflect(0, 6) == 0);
        assert!(reflect(reflect(0x15, 6), 6) == 0x15);
    }

    #[test]
    fn range_for_various_inputs() {
        let a = crc6_darc(b"a");
        let b = crc6_darc(b"b");
        let c = crc6_darc(b"abc");
        assert!((0..=0x3f).contains(&a));
        assert!((0..=0x3f).contains(&b));
        assert!((0..=0x3f).contains(&c));
    }
}
