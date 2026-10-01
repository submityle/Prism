//! `CRC`-8/GSM-A checksum (width=8, poly=`0x1D`, init=`0x00`, refin=false,
//! refout=false, `XOR`out=`0x00`; check=`0x37` for `b"123456789"`).
//!
//! Bit-wise `MSB`-first over an 8-bit register.

const WIDTH: u32 = 8;
const POLY: u32 = 0x1D;
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

/// Computes the `CRC`-8/GSM-A checksum over `data`.
pub fn crc8_gsm_a(data: &[u8]) -> u8 {
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
    fn test_empty() {
        assert!(crc8_gsm_a(b"") == 0x00);
    }

    #[test]
    fn test_z00() {
        assert!(crc8_gsm_a(&[0x00]) == 0x00);
    }

    #[test]
    fn test_ff() {
        assert!(crc8_gsm_a(&[0xff]) == 0xc4);
    }

    #[test]
    fn test_a() {
        assert!(crc8_gsm_a(b"a") == 0x89);
    }

    #[test]
    fn test_b() {
        assert!(crc8_gsm_a(b"b") == 0xae);
    }

    #[test]
    fn test_ab() {
        assert!(crc8_gsm_a(b"ab") == 0x7d);
    }

    #[test]
    fn test_abc() {
        assert!(crc8_gsm_a(b"abc") == 0x6b);
    }

    #[test]
    fn test_two_zeros() {
        assert!(crc8_gsm_a(&[0x00, 0x00]) == 0x00);
    }

    #[test]
    fn test_01() {
        assert!(crc8_gsm_a(&[0x01]) == 0x1d);
    }

    #[test]
    fn test_02() {
        assert!(crc8_gsm_a(&[0x02]) == 0x3a);
    }

    #[test]
    fn test_7f() {
        assert!(crc8_gsm_a(&[0x7f]) == 0xe2);
    }

    #[test]
    fn test_80() {
        assert!(crc8_gsm_a(&[0x80]) == 0x26);
    }

    #[test]
    fn test_aa_55() {
        assert!(crc8_gsm_a(&[0xaa, 0x55]) == 0xc9);
    }

    #[test]
    fn test_55_aa() {
        assert!(crc8_gsm_a(&[0x55, 0xaa]) == 0x4c);
    }

    #[test]
    fn test_deadbeef() {
        assert!(crc8_gsm_a(&[0xde, 0xad, 0xbe, 0xef]) == 0xea);
    }

    #[test]
    fn test_hello() {
        assert!(crc8_gsm_a(b"Hello") == 0x78);
    }

    #[test]
    fn test_fox() {
        assert!(crc8_gsm_a(b"The quick brown fox") == 0x54);
    }

    #[test]
    fn test_four_zeros() {
        assert!(crc8_gsm_a(&[0x00, 0x00, 0x00, 0x00]) == 0x00);
    }

    #[test]
    fn test_four_ff() {
        assert!(crc8_gsm_a(&[0xff, 0xff, 0xff, 0xff]) == 0x2d);
    }

    #[test]
    fn test_0_15() {
        let mut buf = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc8_gsm_a(&buf) == 0x42);
    }

    #[test]
    fn test_12345678() {
        assert!(crc8_gsm_a(&[0x12, 0x34, 0x56, 0x78]) == 0x76);
    }

    #[test]
    fn test_check() {
        assert!(crc8_gsm_a(b"123456789") == 0x37);
    }

    #[test]
    fn test_a5_1000() {
        let buf = [0xA5u8; 1000];
        assert!(crc8_gsm_a(&buf) == 0x7b);
    }

    #[test]
    fn test_all256() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc8_gsm_a(&buf) == 0x3e);
    }

    #[test]
    fn test_determinism() {
        let a = crc8_gsm_a(b"123456789");
        let b = crc8_gsm_a(b"123456789");
        assert!(a == b);
    }

    #[test]
    fn test_order_sensitivity() {
        assert!(crc8_gsm_a(&[0xaa, 0x55]) != crc8_gsm_a(&[0x55, 0xaa]));
    }

    #[test]
    fn test_a_ne_b() {
        assert!(crc8_gsm_a(b"a") != crc8_gsm_a(b"b"));
    }

    #[test]
    fn test_slice_equivalence() {
        let owned = [b'a', b'b', b'c'];
        assert!(crc8_gsm_a(&owned) == crc8_gsm_a(b"abc"));
    }

    #[test]
    fn test_prefix_differs() {
        assert!(crc8_gsm_a(b"ab") != crc8_gsm_a(b"abc"));
    }

    #[test]
    fn test_range_contains() {
        let v = u32::from(crc8_gsm_a(b"123456789"));
        assert!((0x00..=0xff).contains(&v));
    }

    #[test]
    fn test_no_hidden_state() {
        let _warm = crc8_gsm_a(b"warmup data");
        assert!(crc8_gsm_a(b"") == 0x00);
    }

    #[test]
    fn test_const_poly() {
        assert!(POLY == 0x1D);
    }

    #[test]
    fn test_const_init() {
        assert!(INIT == 0x00);
    }

    #[test]
    fn test_const_xorout() {
        assert!(XOROUT == 0x00);
    }

    #[test]
    fn test_const_width() {
        assert!(WIDTH == 8);
    }

    #[test]
    fn test_const_mask() {
        assert!(MASK == 0xff);
    }

    #[test]
    fn test_const_flags() {
        assert!(!REFLECT_IN);
        assert!(!REFLECT_OUT);
    }

    #[test]
    fn test_topbit() {
        assert!(TOPBIT == 0x80);
    }

    #[test]
    fn test_reflect_sanity() {
        assert!(reflect(0x01, 8) == 0x80);
        assert!(reflect(0x80, 8) == 0x01);
    }
}
