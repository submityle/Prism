//! `CRC`-8/GSM-B checksum (width=8, poly=`0x49`, init=`0x00`, refin=false,
//! refout=false, `XOR`out=`0xFF`; check=`0x94` for `b"123456789"`).
//!
//! Bit-wise `MSB`-first over an 8-bit register; the final register is `XOR`ed
//! with `0xFF`.

const WIDTH: u32 = 8;
const POLY: u32 = 0x49;
const INIT: u32 = 0x00;
const XOROUT: u32 = 0xFF;
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

/// Computes the `CRC`-8/GSM-B checksum over `data`.
pub fn crc8_gsm_b(data: &[u8]) -> u8 {
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
    fn vec_empty() {
        assert!(crc8_gsm_b(b"") == 0xff);
    }

    #[test]
    fn vec_z00() {
        assert!(crc8_gsm_b(&[0x00]) == 0xff);
    }

    #[test]
    fn vec_ff() {
        assert!(crc8_gsm_b(&[0xff]) == 0xac);
    }

    #[test]
    fn vec_a() {
        assert!(crc8_gsm_b(b"a") == 0x2a);
    }

    #[test]
    fn vec_b() {
        assert!(crc8_gsm_b(b"b") == 0xf1);
    }

    #[test]
    fn vec_ab() {
        assert!(crc8_gsm_b(b"ab") == 0x59);
    }

    #[test]
    fn vec_abc() {
        assert!(crc8_gsm_b(b"abc") == 0xaa);
    }

    #[test]
    fn vec_two_zeros() {
        assert!(crc8_gsm_b(&[0x00, 0x00]) == 0xff);
    }

    #[test]
    fn vec_01() {
        assert!(crc8_gsm_b(&[0x01]) == 0xb6);
    }

    #[test]
    fn vec_02() {
        assert!(crc8_gsm_b(&[0x02]) == 0x6d);
    }

    #[test]
    fn vec_7f() {
        assert!(crc8_gsm_b(&[0x7f]) == 0xf2);
    }

    #[test]
    fn vec_80() {
        assert!(crc8_gsm_b(&[0x80]) == 0xa1);
    }

    #[test]
    fn vec_aa_55() {
        assert!(crc8_gsm_b(&[0xaa, 0x55]) == 0xc0);
    }

    #[test]
    fn vec_55_aa() {
        assert!(crc8_gsm_b(&[0x55, 0xaa]) == 0x9a);
    }

    #[test]
    fn vec_deadbeef() {
        assert!(crc8_gsm_b(&[0xde, 0xad, 0xbe, 0xef]) == 0x35);
    }

    #[test]
    fn vec_hello() {
        assert!(crc8_gsm_b(b"Hello") == 0x66);
    }

    #[test]
    fn vec_fox() {
        assert!(crc8_gsm_b(b"The quick brown fox") == 0xb0);
    }

    #[test]
    fn vec_four_zeros() {
        assert!(crc8_gsm_b(&[0x00, 0x00, 0x00, 0x00]) == 0xff);
    }

    #[test]
    fn vec_four_ff() {
        assert!(crc8_gsm_b(&[0xff, 0xff, 0xff, 0xff]) == 0x4e);
    }

    #[test]
    fn vec_0_15() {
        let mut buf = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc8_gsm_b(&buf) == 0x9c);
    }

    #[test]
    fn vec_12345678() {
        assert!(crc8_gsm_b(&[0x12, 0x34, 0x56, 0x78]) == 0x6d);
    }

    #[test]
    fn vec_check() {
        assert!(crc8_gsm_b(b"123456789") == 0x94);
    }

    #[test]
    fn vec_a5_1000() {
        let buf = [0xA5u8; 1000];
        assert!(crc8_gsm_b(&buf) == 0x28);
    }

    #[test]
    fn vec_all256() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc8_gsm_b(&buf) == 0x0c);
    }

    #[test]
    fn determinism() {
        assert!(crc8_gsm_b(b"123456789") == crc8_gsm_b(b"123456789"));
    }

    #[test]
    fn order_sensitivity() {
        assert!(crc8_gsm_b(&[0xaa, 0x55]) != crc8_gsm_b(&[0x55, 0xaa]));
    }

    #[test]
    fn a_ne_b() {
        assert!(crc8_gsm_b(b"a") != crc8_gsm_b(b"b"));
    }

    #[test]
    fn slice_equivalence() {
        let data = [0x12u8, 0x34, 0x56];
        assert!(crc8_gsm_b(&data) == crc8_gsm_b(&data[..]));
    }

    #[test]
    fn prefix_differs() {
        assert!(crc8_gsm_b(b"abc") != crc8_gsm_b(b"ab"));
    }

    #[test]
    fn range_contains() {
        let v = crc8_gsm_b(b"anything");
        assert!((0x00..=0xff).contains(&v));
    }

    #[test]
    fn no_hidden_state() {
        let _ = crc8_gsm_b(b"warmup");
        assert!(crc8_gsm_b(b"") == 0xff);
    }

    #[test]
    fn const_poly() {
        assert!(core::hint::black_box(POLY) == 0x49);
    }

    #[test]
    fn const_xorout() {
        assert!(core::hint::black_box(XOROUT) == 0xFF);
    }

    #[test]
    fn const_width() {
        assert!(core::hint::black_box(WIDTH) == 8);
    }

    #[test]
    fn const_init_mask() {
        assert!(core::hint::black_box(INIT) == 0x00);
        assert!(core::hint::black_box(MASK) == 0xff);
    }

    #[test]
    fn reflect_behaviour() {
        assert!(reflect(0x01, 8) == 0x80);
        assert!(reflect(0x80, 8) == 0x01);
    }

    #[test]
    fn longer_differs() {
        assert!(crc8_gsm_b(b"123456789") != crc8_gsm_b(&[0x12, 0x34, 0x56, 0x78]));
    }
}
