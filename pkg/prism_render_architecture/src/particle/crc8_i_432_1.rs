//! `CRC`-8/I-432-1 checksum (width=8, poly=`0x07`, init=`0x00`, refin=false,
//! refout=false, `XOR`out=`0x55`; check=`0xa1` for `b"123456789"`).
//!
//! Bit-wise `MSB`-first over an 8-bit register: each input byte is fed in
//! bit by bit, shifting the register left and conditionally `XOR`ing the
//! polynomial, then the final `XOR`out is applied. This implementation is
//! `no_std`-friendly and uses pure integer operations on `u8`/`u32` only.

/// Register width in bits.
const WIDTH: u32 = 8;
/// Forward (unreflected) generator polynomial.
const POLY: u32 = 0x07;
/// Initial register value.
const INIT: u32 = 0x00;
/// Final `XOR` value applied to the register.
const XOROUT: u32 = 0x55;
/// Whether each input byte is bit-reflected before processing.
const REFLECT_IN: bool = false;
/// Whether the final register is bit-reflected before `XOR`out.
const REFLECT_OUT: bool = false;
/// Mask selecting the low `WIDTH` bits.
const MASK: u32 = 0xff;
/// Most-significant bit (`MSB`) of the `WIDTH`-bit register.
const TOPBIT: u32 = 1u32 << (WIDTH - 1);

/// Reflects the low `bits` bits of `value` (bit-reversal), moving each
/// `LSB`-side bit toward the `MSB` side.
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

/// Computes the `CRC`-8/I-432-1 checksum over `data`.
pub fn crc8_i_432_1(data: &[u8]) -> u8 {
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
    u8::try_from((reg ^ XOROUT) & MASK).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_empty() {
        assert!(crc8_i_432_1(b"") == 0x55);
    }

    #[test]
    fn vec_z00() {
        assert!(crc8_i_432_1(&[0x00]) == 0x55);
    }

    #[test]
    fn vec_ff() {
        assert!(crc8_i_432_1(&[0xff]) == 0xa6);
    }

    #[test]
    fn vec_a() {
        assert!(crc8_i_432_1(b"a") == 0x75);
    }

    #[test]
    fn vec_b() {
        assert!(crc8_i_432_1(b"b") == 0x7c);
    }

    #[test]
    fn vec_ab() {
        assert!(crc8_i_432_1(b"ab") == 0x9c);
    }

    #[test]
    fn vec_abc() {
        assert!(crc8_i_432_1(b"abc") == 0x0a);
    }

    #[test]
    fn vec_two_zeros() {
        assert!(crc8_i_432_1(&[0, 0]) == 0x55);
    }

    #[test]
    fn vec_01() {
        assert!(crc8_i_432_1(&[0x01]) == 0x52);
    }

    #[test]
    fn vec_02() {
        assert!(crc8_i_432_1(&[0x02]) == 0x5b);
    }

    #[test]
    fn vec_7f() {
        assert!(crc8_i_432_1(&[0x7f]) == 0x2f);
    }

    #[test]
    fn vec_80() {
        assert!(crc8_i_432_1(&[0x80]) == 0xdc);
    }

    #[test]
    fn vec_aa_55() {
        assert!(crc8_i_432_1(&[0xaa, 0x55]) == 0x63);
    }

    #[test]
    fn vec_55_aa() {
        assert!(crc8_i_432_1(&[0x55, 0xaa]) == 0x47);
    }

    #[test]
    fn vec_deadbeef() {
        assert!(crc8_i_432_1(&[0xde, 0xad, 0xbe, 0xef]) == 0x9f);
    }

    #[test]
    fn vec_hello() {
        assert!(crc8_i_432_1(b"Hello") == 0xa3);
    }

    #[test]
    fn vec_fox() {
        assert!(crc8_i_432_1(b"The quick brown fox") == 0x7c);
    }

    #[test]
    fn vec_four_zeros() {
        assert!(crc8_i_432_1(&[0, 0, 0, 0]) == 0x55);
    }

    #[test]
    fn vec_four_ff() {
        assert!(crc8_i_432_1(&[0xff, 0xff, 0xff, 0xff]) == 0x8b);
    }

    #[test]
    fn vec_0_15() {
        let mut data = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            data[i] = u8::try_from(i).unwrap();
            i += 1;
        }
        assert!(crc8_i_432_1(&data) == 0x14);
    }

    #[test]
    fn vec_12345678() {
        assert!(crc8_i_432_1(&[0x12, 0x34, 0x56, 0x78]) == 0x49);
    }

    #[test]
    fn vec_check() {
        assert!(crc8_i_432_1(b"123456789") == 0xa1);
    }

    #[test]
    fn vec_a5_1000() {
        let mut data = [0u8; 1000];
        let mut i = 0usize;
        while i < 1000 {
            data[i] = 0xA5;
            i += 1;
        }
        assert!(crc8_i_432_1(&data) == 0xca);
    }

    #[test]
    fn vec_all256() {
        let mut data = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            data[i] = u8::try_from(i).unwrap();
            i += 1;
        }
        assert!(crc8_i_432_1(&data) == 0x41);
    }

    #[test]
    fn determinism() {
        let input = b"determinism";
        assert!(crc8_i_432_1(input) == crc8_i_432_1(input));
    }

    #[test]
    fn determinism_check() {
        assert!(crc8_i_432_1(b"123456789") == crc8_i_432_1(b"123456789"));
    }

    #[test]
    fn order_sensitivity() {
        assert!(crc8_i_432_1(&[0xaa, 0x55]) != crc8_i_432_1(&[0x55, 0xaa]));
    }

    #[test]
    fn different_inputs_differ() {
        assert!(crc8_i_432_1(b"a") != crc8_i_432_1(b"b"));
    }

    #[test]
    fn slice_equivalence() {
        let data = [0xde, 0xad, 0xbe, 0xef];
        assert!(crc8_i_432_1(&data[..]) == crc8_i_432_1(&data));
    }

    #[test]
    fn prefix_differs() {
        let data = [0xde, 0xad, 0xbe, 0xef];
        assert!(crc8_i_432_1(&data[..2]) != crc8_i_432_1(&data));
    }

    #[test]
    fn output_in_range() {
        let v = crc8_i_432_1(b"range");
        assert!((0x00..=0xff).contains(&v));
    }

    #[test]
    fn output_in_range_empty() {
        let v = crc8_i_432_1(b"");
        assert!((0x00..=0xff).contains(&v));
    }

    #[test]
    fn no_hidden_state() {
        let _ = crc8_i_432_1(b"warmup");
        assert!(crc8_i_432_1(b"") == 0x55);
    }

    #[test]
    fn no_hidden_state_repeat() {
        let _ = crc8_i_432_1(b"another warmup payload");
        assert!(crc8_i_432_1(&[0x00]) == 0x55);
    }

    #[test]
    fn const_sanity_poly() {
        assert!(POLY == 0x07);
    }

    #[test]
    fn const_sanity_mask() {
        assert!(MASK == 0xff);
    }

    #[test]
    fn const_sanity_topbit() {
        assert!(TOPBIT == 0x80);
    }

    #[test]
    fn const_sanity_width() {
        assert!(WIDTH == 8);
    }

    #[test]
    fn const_sanity_init_xorout() {
        assert!(INIT == 0x00);
        assert!(XOROUT == 0x55);
    }

    #[test]
    fn const_sanity_reflect_flags() {
        assert!(!REFLECT_IN);
        assert!(!REFLECT_OUT);
    }

    #[test]
    fn reflect_helper_identity_zero() {
        assert!(reflect(0x00, 8) == 0x00);
    }

    #[test]
    fn reflect_helper_known() {
        assert!(reflect(0x01, 8) == 0x80);
    }
}
