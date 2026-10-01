//! `CRC`-8/MIFARE-MAD checksum (width=8, poly=`0x1D`, init=`0xC7`, refin=false,
//! refout=false, `XOR`out=`0x00`; check=`0x99` for `b"123456789"`).
//!
//! Bit-wise `MSB`-first over an 8-bit register.

const WIDTH: u32 = 8;
const POLY: u32 = 0x1D;
const INIT: u32 = 0xC7;
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

/// Computes the `CRC`-8/MIFARE-MAD checksum over `data`.
pub fn crc8_mifare_mad(data: &[u8]) -> u8 {
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
        assert!(crc8_mifare_mad(b"") == 0xc7);
    }

    #[test]
    fn vec_z00() {
        assert!(crc8_mifare_mad(&[0x00]) == 0x66);
    }

    #[test]
    fn vec_ff() {
        assert!(crc8_mifare_mad(&[0xff]) == 0xa2);
    }

    #[test]
    fn vec_a() {
        assert!(crc8_mifare_mad(b"a") == 0xef);
    }

    #[test]
    fn vec_b() {
        assert!(crc8_mifare_mad(b"b") == 0xc8);
    }

    #[test]
    fn vec_ab() {
        assert!(crc8_mifare_mad(b"ab") == 0xa7);
    }

    #[test]
    fn vec_abc() {
        assert!(crc8_mifare_mad(b"abc") == 0x41);
    }

    #[test]
    fn vec_two_zeros() {
        assert!(crc8_mifare_mad(&[0x00, 0x00]) == 0xda);
    }

    #[test]
    fn vec_01() {
        assert!(crc8_mifare_mad(&[0x01]) == 0x7b);
    }

    #[test]
    fn vec_02() {
        assert!(crc8_mifare_mad(&[0x02]) == 0x5c);
    }

    #[test]
    fn vec_7f() {
        assert!(crc8_mifare_mad(&[0x7f]) == 0x84);
    }

    #[test]
    fn vec_80() {
        assert!(crc8_mifare_mad(&[0x80]) == 0x40);
    }

    #[test]
    fn vec_aa_55() {
        assert!(crc8_mifare_mad(&[0xaa, 0x55]) == 0x13);
    }

    #[test]
    fn vec_55_aa() {
        assert!(crc8_mifare_mad(&[0x55, 0xaa]) == 0x96);
    }

    #[test]
    fn vec_deadbeef() {
        assert!(crc8_mifare_mad(&[0xde, 0xad, 0xbe, 0xef]) == 0xbf);
    }

    #[test]
    fn vec_hello() {
        assert!(crc8_mifare_mad(b"Hello") == 0xcf);
    }

    #[test]
    fn vec_fox() {
        assert!(crc8_mifare_mad(b"The quick brown fox") == 0x72);
    }

    #[test]
    fn vec_four_zeros() {
        assert!(crc8_mifare_mad(&[0x00, 0x00, 0x00, 0x00]) == 0x55);
    }

    #[test]
    fn vec_four_ff() {
        assert!(crc8_mifare_mad(&[0xff, 0xff, 0xff, 0xff]) == 0x78);
    }

    #[test]
    fn vec_0_15() {
        let data: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        assert!(crc8_mifare_mad(&data) == 0x8d);
    }

    #[test]
    fn vec_12345678() {
        assert!(crc8_mifare_mad(&[0x12, 0x34, 0x56, 0x78]) == 0x23);
    }

    #[test]
    fn vec_check() {
        assert!(crc8_mifare_mad(b"123456789") == 0x99);
    }

    #[test]
    fn vec_a5_1000() {
        let data = [0xA5u8; 1000];
        assert!(crc8_mifare_mad(&data) == 0x89);
    }

    #[test]
    fn vec_all256() {
        let mut data = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc8_mifare_mad(&data) == 0x58);
    }

    #[test]
    fn determinism() {
        let first = crc8_mifare_mad(b"determinism");
        let second = crc8_mifare_mad(b"determinism");
        assert!(first == second);
    }

    #[test]
    fn order_sensitivity() {
        assert!(crc8_mifare_mad(&[0xaa, 0x55]) != crc8_mifare_mad(&[0x55, 0xaa]));
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc8_mifare_mad(b"a") != crc8_mifare_mad(b"b"));
    }

    #[test]
    fn slice_equivalence() {
        let data: [u8; 5] = [0x48, 0x65, 0x6c, 0x6c, 0x6f];
        let whole = crc8_mifare_mad(&data);
        let via_slice = crc8_mifare_mad(&data[..]);
        assert!(whole == via_slice);
    }

    #[test]
    fn prefix_differs() {
        assert!(crc8_mifare_mad(b"abc") != crc8_mifare_mad(b"ab"));
    }

    #[test]
    fn range_contains() {
        let v = crc8_mifare_mad(b"123456789");
        assert!((0x00u8..=0xffu8).contains(&v));
    }

    #[test]
    fn no_hidden_state() {
        let _ = crc8_mifare_mad(b"warmup payload");
        assert!(crc8_mifare_mad(b"") == 0xc7);
    }

    #[test]
    fn const_poly() {
        assert!(POLY == 0x1D);
    }

    #[test]
    fn const_init() {
        assert!(INIT == 0xC7);
    }

    #[test]
    fn const_width() {
        assert!(WIDTH == 8);
    }

    #[test]
    fn const_xorout() {
        assert!(XOROUT == 0x00);
    }

    #[test]
    fn const_mask() {
        assert!(MASK == 0xff);
    }

    #[test]
    fn const_topbit() {
        assert!(TOPBIT == 0x80);
    }

    #[test]
    fn const_reflect_flags() {
        assert!(u32::from(REFLECT_IN) == 0);
        assert!(u32::from(REFLECT_OUT) == 0);
    }

    #[test]
    fn reflect_fn() {
        assert!(reflect(0x01, 8) == 0x80);
        assert!(reflect(0x80, 8) == 0x01);
        assert!(reflect(0xff, 8) == 0xff);
    }
}
