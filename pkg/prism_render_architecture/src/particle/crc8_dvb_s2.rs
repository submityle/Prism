//! `CRC`-8/DVB-S2 checksum (width=8, poly=`0xD5`, init=`0x00`, refin=false,
//! refout=false, `XOR`out=`0x00`; check=`0xbc` for `b"123456789"`).
//!
//! Bit-wise `MSB`-first over an 8-bit register.

const WIDTH: u32 = 8;
const POLY: u32 = 0xD5;
const INIT: u32 = 0x00;
const XOROUT: u32 = 0x00;
const REFLECT_IN: bool = false;
const REFLECT_OUT: bool = false;
const MASK: u32 = 0xff;
const TOPBIT: u32 = 1u32 << (WIDTH - 1);

/// Reflects the low `bits` bits of `value`.
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

/// Computes the `CRC`-8/DVB-S2 checksum over `data`.
pub fn crc8_dvb_s2(data: &[u8]) -> u8 {
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
    ((reg ^ XOROUT) & MASK) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_empty() {
        assert!(crc8_dvb_s2(b"") == 0x00);
    }

    #[test]
    fn vec_z00() {
        assert!(crc8_dvb_s2(&[0x00]) == 0x00);
    }

    #[test]
    fn vec_ff() {
        assert!(crc8_dvb_s2(&[0xff]) == 0xf9);
    }

    #[test]
    fn vec_a() {
        assert!(crc8_dvb_s2(b"a") == 0xec);
    }

    #[test]
    fn vec_b() {
        assert!(crc8_dvb_s2(b"b") == 0x46);
    }

    #[test]
    fn vec_ab() {
        assert!(crc8_dvb_s2(b"ab") == 0x47);
    }

    #[test]
    fn vec_abc() {
        assert!(crc8_dvb_s2(b"abc") == 0x5a);
    }

    #[test]
    fn vec_two_zeros() {
        assert!(crc8_dvb_s2(&[0x00, 0x00]) == 0x00);
    }

    #[test]
    fn vec_01() {
        assert!(crc8_dvb_s2(&[0x01]) == 0xd5);
    }

    #[test]
    fn vec_02() {
        assert!(crc8_dvb_s2(&[0x02]) == 0x7f);
    }

    #[test]
    fn vec_7f() {
        assert!(crc8_dvb_s2(&[0x7f]) == 0x16);
    }

    #[test]
    fn vec_80() {
        assert!(crc8_dvb_s2(&[0x80]) == 0xef);
    }

    #[test]
    fn vec_aa_55() {
        assert!(crc8_dvb_s2(&[0xaa, 0x55]) == 0xb4);
    }

    #[test]
    fn vec_55_aa() {
        assert!(crc8_dvb_s2(&[0x55, 0xaa]) == 0x35);
    }

    #[test]
    fn vec_deadbeef() {
        assert!(crc8_dvb_s2(&[0xde, 0xad, 0xbe, 0xef]) == 0xa5);
    }

    #[test]
    fn vec_hello() {
        assert!(crc8_dvb_s2(b"Hello") == 0x8e);
    }

    #[test]
    fn vec_fox() {
        assert!(crc8_dvb_s2(b"The quick brown fox") == 0x8c);
    }

    #[test]
    fn vec_four_zeros() {
        assert!(crc8_dvb_s2(&[0x00, 0x00, 0x00, 0x00]) == 0x00);
    }

    #[test]
    fn vec_four_ff() {
        assert!(crc8_dvb_s2(&[0xff, 0xff, 0xff, 0xff]) == 0x21);
    }

    #[test]
    fn vec_0_15() {
        let mut data = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc8_dvb_s2(&data) == 0x77);
    }

    #[test]
    fn vec_12345678() {
        assert!(crc8_dvb_s2(&[0x12, 0x34, 0x56, 0x78]) == 0x0b);
    }

    #[test]
    fn vec_check() {
        assert!(crc8_dvb_s2(b"123456789") == 0xbc);
    }

    #[test]
    fn vec_a5_1000() {
        let data = [0xA5u8; 1000];
        assert!(crc8_dvb_s2(&data) == 0xc6);
    }

    #[test]
    fn vec_all256() {
        let mut data = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc8_dvb_s2(&data) == 0xca);
    }

    #[test]
    fn determinism() {
        let a = crc8_dvb_s2(b"determinism");
        let b = crc8_dvb_s2(b"determinism");
        assert!(a == b);
    }

    #[test]
    fn order_sensitivity() {
        assert!(crc8_dvb_s2(&[0x61, 0x62]) != crc8_dvb_s2(&[0x62, 0x61]));
    }

    #[test]
    fn a_ne_b() {
        assert!(crc8_dvb_s2(b"a") != crc8_dvb_s2(b"b"));
    }

    #[test]
    fn slice_equivalence() {
        assert!(crc8_dvb_s2(b"abc") == crc8_dvb_s2(&[0x61, 0x62, 0x63]));
    }

    #[test]
    fn prefix_differs() {
        assert!(crc8_dvb_s2(b"ab") != crc8_dvb_s2(b"abc"));
    }

    #[test]
    fn range_contains() {
        let v = crc8_dvb_s2(b"range");
        assert!((0x00..=0xff).contains(&v));
    }

    #[test]
    fn no_hidden_state() {
        let _warmup = crc8_dvb_s2(b"warmup-data-here");
        assert!(crc8_dvb_s2(b"") == 0x00);
    }

    #[test]
    fn const_sanity_poly() {
        assert!(POLY == 0xD5);
    }

    #[test]
    fn const_sanity_width() {
        assert!(WIDTH == 8);
    }

    #[test]
    fn const_sanity_init() {
        assert!(INIT == 0x00);
    }

    #[test]
    fn const_sanity_mask() {
        assert!(MASK == 0xff);
    }

    #[test]
    fn const_sanity_xorout() {
        assert!(XOROUT == 0x00);
    }

    #[test]
    fn two_zeros_equals_empty() {
        assert!(crc8_dvb_s2(&[0x00, 0x00]) == crc8_dvb_s2(b""));
    }

    #[test]
    fn four_zeros_is_zero() {
        assert!(crc8_dvb_s2(&[0x00, 0x00, 0x00, 0x00]) == 0x00);
    }

    #[test]
    fn reflect_identity_disabled() {
        assert!(!REFLECT_IN);
        assert!(!REFLECT_OUT);
    }

    #[test]
    fn check_is_in_range() {
        let v = crc8_dvb_s2(b"123456789");
        assert!((0x00..=0xff).contains(&v));
    }
}
