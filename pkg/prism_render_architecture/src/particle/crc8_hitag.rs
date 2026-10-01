//! `CRC`-8/HITAG checksum (width=8, poly=`0x1d`, init=`0xff`, refin=false,
//! refout=false, `XOR`out=`0x00`; check=`0xb4` for `b"123456789"`).
//!
//! Bit-wise `MSB`-first over an 8-bit register. Pure integer arithmetic only.

/// Register width in bits.
const WIDTH: u32 = 8;
/// Generator polynomial (`0x1d`), `MSB`-first form.
const POLY: u32 = 0x1D;
/// Initial register value (`0xff`).
const INIT: u32 = 0xFF;
/// Final `XOR` applied to the register (`0x00`).
const XOROUT: u32 = 0x00;
/// Whether each input byte is bit-reflected before processing.
const REFLECT_IN: bool = false;
/// Whether the register is bit-reflected after processing.
const REFLECT_OUT: bool = false;
/// Low-byte mask (`0xff`) keeping the register within 8 bits.
const MASK: u32 = 0xff;
/// Most-significant bit of the register.
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

/// Computes the `CRC`-8/HITAG checksum over `data` and returns it as a `u8`.
pub fn crc8_hitag(data: &[u8]) -> u8 {
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
    fn empty_is_init() {
        assert!(crc8_hitag(b"") == 0xff);
    }

    #[test]
    fn z00_vector() {
        assert!(crc8_hitag(&[0x00]) == 0xc4);
    }

    #[test]
    fn ff_vector() {
        assert!(crc8_hitag(&[0xff]) == 0x00);
    }

    #[test]
    fn a_vector() {
        assert!(crc8_hitag(b"a") == 0x4d);
    }

    #[test]
    fn b_vector() {
        assert!(crc8_hitag(b"b") == 0x6a);
    }

    #[test]
    fn ab_vector() {
        assert!(crc8_hitag(b"ab") == 0x3c);
    }

    #[test]
    fn abc_vector() {
        assert!(crc8_hitag(b"abc") == 0x65);
    }

    #[test]
    fn two_zeros_vector() {
        assert!(crc8_hitag(&[0x00, 0x00]) == 0x41);
    }

    #[test]
    fn b01_vector() {
        assert!(crc8_hitag(&[0x01]) == 0xd9);
    }

    #[test]
    fn b02_vector() {
        assert!(crc8_hitag(&[0x02]) == 0xfe);
    }

    #[test]
    fn b7f_vector() {
        assert!(crc8_hitag(&[0x7f]) == 0x26);
    }

    #[test]
    fn b80_vector() {
        assert!(crc8_hitag(&[0x80]) == 0xe2);
    }

    #[test]
    fn aa_55_vector() {
        assert!(crc8_hitag(&[0xaa, 0x55]) == 0x88);
    }

    #[test]
    fn b55_aa_vector() {
        assert!(crc8_hitag(&[0x55, 0xaa]) == 0x0d);
    }

    #[test]
    fn deadbeef_vector() {
        assert!(crc8_hitag(&[0xde, 0xad, 0xbe, 0xef]) == 0x4c);
    }

    #[test]
    fn hello_vector() {
        assert!(crc8_hitag(b"Hello") == 0x97);
    }

    #[test]
    fn fox_vector() {
        assert!(crc8_hitag(b"The quick brown fox") == 0x31);
    }

    #[test]
    fn four_zeros_vector() {
        assert!(crc8_hitag(&[0x00, 0x00, 0x00, 0x00]) == 0xa6);
    }

    #[test]
    fn four_ff_vector() {
        assert!(crc8_hitag(&[0xff, 0xff, 0xff, 0xff]) == 0x8b);
    }

    #[test]
    fn range_0_15_vector() {
        let data: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        assert!(crc8_hitag(&data) == 0x04);
    }

    #[test]
    fn b12345678_vector() {
        assert!(crc8_hitag(&[0x12, 0x34, 0x56, 0x78]) == 0xd0);
    }

    #[test]
    fn check_vector() {
        assert!(crc8_hitag(b"123456789") == 0xb4);
    }

    #[test]
    fn a5_1000_vector() {
        let mut data = [0u8; 1000];
        let mut i = 0usize;
        while i < 1000 {
            data[i] = 0xA5;
            i += 1;
        }
        assert!(crc8_hitag(&data) == 0x5d);
    }

    #[test]
    fn all256_vector() {
        let mut data = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc8_hitag(&data) == 0xfa);
    }

    #[test]
    fn determinism() {
        let a = crc8_hitag(b"The quick brown fox");
        let b = crc8_hitag(b"The quick brown fox");
        assert!(a == b);
    }

    #[test]
    fn determinism_check() {
        let a = crc8_hitag(b"123456789");
        let b = crc8_hitag(b"123456789");
        assert!(a == b);
    }

    #[test]
    fn order_sensitivity() {
        assert!(crc8_hitag(&[0xaa, 0x55]) != crc8_hitag(&[0x55, 0xaa]));
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc8_hitag(b"a") != crc8_hitag(b"b"));
    }

    #[test]
    fn slice_equivalence() {
        let data: [u8; 4] = [0xde, 0xad, 0xbe, 0xef];
        let whole = crc8_hitag(&data);
        let via_slice = crc8_hitag(&data[0..4]);
        assert!(whole == via_slice);
    }

    #[test]
    fn prefix_differs() {
        assert!(crc8_hitag(b"abc") != crc8_hitag(b"ab"));
    }

    #[test]
    fn empty_differs_from_z00() {
        assert!(crc8_hitag(b"") != crc8_hitag(&[0x00]));
    }

    #[test]
    fn range_contains_result() {
        let v = crc8_hitag(b"The quick brown fox");
        assert!((0x00u8..=0xffu8).contains(&v));
    }

    #[test]
    fn range_contains_check() {
        let v = crc8_hitag(b"123456789");
        assert!((0x00u8..=0xffu8).contains(&v));
    }

    #[test]
    fn no_hidden_state() {
        let _warmup = crc8_hitag(b"The quick brown fox");
        assert!(crc8_hitag(b"") == 0xff);
    }

    #[test]
    fn no_hidden_state_after_check() {
        let _warmup = crc8_hitag(b"123456789");
        assert!(crc8_hitag(&[0x00]) == 0xc4);
    }

    #[test]
    fn const_sanity_poly() {
        assert!(POLY == 0x1D);
    }

    #[test]
    fn const_sanity_init() {
        assert!(INIT == 0xFF);
    }

    #[test]
    fn const_sanity_mask() {
        assert!(MASK == 0xff);
    }

    #[test]
    fn const_sanity_width() {
        assert!(WIDTH == 8);
    }

    #[test]
    fn const_sanity_topbit() {
        assert!(TOPBIT == 0x80);
    }

    #[test]
    fn const_sanity_xorout() {
        assert!(XOROUT == 0x00);
    }

    #[test]
    fn reflect_flags_disabled() {
        assert!(!REFLECT_IN);
        assert!(!REFLECT_OUT);
    }

    #[test]
    fn reflect_known_value() {
        assert!(reflect(0x01, 8) == 0x80);
    }

    #[test]
    fn reflect_identity_zero() {
        assert!(reflect(0x00, 8) == 0x00);
    }
}
