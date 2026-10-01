//! `CRC`-16/DECT-R checksum implementation using a bit-wise algorithm.
//!
//! Parameters: width=16, poly=`0x0589`, init=`0x0`, refin=false, refout=false,
//! xorout=`0x0001`, check=`0x007e`.
//!
//! The computation walks each input byte from its most-significant bit (`MSB`)
//! first, shifting the register and conditionally applying the polynomial via
//! an `XOR` operation. This routine is independent of `CPU` endianness.

const WIDTH: u32 = 16;
const POLY: u32 = 0x0589;
const INIT: u32 = 0x0;
const XOROUT: u32 = 0x0001;
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

/// Computes the `CRC`-16/DECT-R checksum of `data`.
pub fn crc16_dect_r(data: &[u8]) -> u16 {
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
    use super::crc16_dect_r;

    #[test]
    fn vector_empty() {
        assert!(crc16_dect_r(b"") == 0x0001);
    }

    #[test]
    fn vector_z00() {
        assert!(crc16_dect_r(&[0x00]) == 0x0001);
    }

    #[test]
    fn vector_ff() {
        assert!(crc16_dect_r(&[0xff]) == 0x751d);
    }

    #[test]
    fn vector_a() {
        assert!(crc16_dect_r(b"a") == 0xd361);
    }

    #[test]
    fn vector_b() {
        assert!(crc16_dect_r(b"b") == 0xddfa);
    }

    #[test]
    fn vector_ab() {
        assert!(crc16_dect_r(b"ab") == 0x43aa);
    }

    #[test]
    fn vector_abc() {
        assert!(crc16_dect_r(b"abc") == 0x1a21);
    }

    #[test]
    fn vector_two_zeros() {
        assert!(crc16_dect_r(&[0, 0]) == 0x0001);
    }

    #[test]
    fn vector_01() {
        assert!(crc16_dect_r(&[0x01]) == 0x0588);
    }

    #[test]
    fn vector_02() {
        assert!(crc16_dect_r(&[0x02]) == 0x0b13);
    }

    #[test]
    fn vector_7f() {
        assert!(crc16_dect_r(&[0x7f]) == 0xba8f);
    }

    #[test]
    fn vector_80() {
        assert!(crc16_dect_r(&[0x80]) == 0xcf93);
    }

    #[test]
    fn vector_aa_55() {
        assert!(crc16_dect_r(&[0xaa, 0x55]) == 0xd26d);
    }

    #[test]
    fn vector_55_aa() {
        assert!(crc16_dect_r(&[0x55, 0xaa]) == 0x26a5);
    }

    #[test]
    fn vector_deadbeef() {
        assert!(crc16_dect_r(&[0xde, 0xad, 0xbe, 0xef]) == 0x303b);
    }

    #[test]
    fn vector_hello() {
        assert!(crc16_dect_r(b"Hello") == 0x3baa);
    }

    #[test]
    fn vector_fox() {
        assert!(crc16_dect_r(b"The quick brown fox") == 0x5db3);
    }

    #[test]
    fn vector_four_zeros() {
        assert!(crc16_dect_r(&[0u8; 4]) == 0x0001);
    }

    #[test]
    fn vector_four_ff() {
        assert!(crc16_dect_r(&[0xffu8; 4]) == 0x983c);
    }

    #[test]
    fn vector_0_15() {
        let mut data = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc16_dect_r(&data) == 0x3ef9);
    }

    #[test]
    fn vector_12345678() {
        assert!(crc16_dect_r(&[0x12, 0x34, 0x56, 0x78]) == 0xf0d1);
    }

    #[test]
    fn vector_check() {
        assert!(crc16_dect_r(b"123456789") == 0x007e);
    }

    #[test]
    fn vector_a5_1000() {
        assert!(crc16_dect_r(&[0xA5u8; 1000]) == 0x55a1);
    }

    #[test]
    fn vector_all256() {
        let mut data = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc16_dect_r(&data) == 0x9a1c);
    }

    #[test]
    fn check_constant_matches() {
        assert!(crc16_dect_r(b"123456789") == 0x007e);
    }

    #[test]
    fn determinism() {
        let data = [0x12u8, 0x34, 0x56, 0x78, 0x9a];
        let first = crc16_dect_r(&data);
        let second = crc16_dect_r(&data);
        assert!(first == second);
    }

    #[test]
    fn order_sensitive() {
        let forward = crc16_dect_r(&[0x01, 0x02, 0x03]);
        let reverse = crc16_dect_r(&[0x03, 0x02, 0x01]);
        assert!(forward != reverse);
    }

    #[test]
    fn a_not_equal_b() {
        assert!(crc16_dect_r(b"a") != crc16_dect_r(b"b"));
    }

    #[test]
    fn prefix_differs() {
        assert!(crc16_dect_r(b"ab") != crc16_dect_r(b"abc"));
    }

    #[test]
    fn length_sensitive() {
        assert!(crc16_dect_r(&[0x01]) != crc16_dect_r(&[0x01, 0x01, 0x01]));
    }

    #[test]
    fn single_bytes_distinguishable() {
        let mut seen = [0u16; 256];
        let mut i = 0usize;
        while i < 256 {
            seen[i] = crc16_dect_r(&[i as u8]);
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
    fn result_in_range() {
        let v = crc16_dect_r(b"range check sample");
        assert!((0..=0xffffu16).contains(&v));
    }

    #[test]
    fn empty_matches_init_xorout() {
        assert!(crc16_dect_r(b"") == 0x0001);
    }

    #[test]
    fn two_distinct_inputs_differ() {
        assert!(crc16_dect_r(b"Hello") != crc16_dect_r(b"World"));
    }

    #[test]
    fn repeated_pattern_stable() {
        let data = [0xA5u8; 1000];
        assert!(crc16_dect_r(&data) == crc16_dect_r(&data));
    }

    #[test]
    fn appended_byte_changes() {
        assert!(crc16_dect_r(b"123456789") != crc16_dect_r(b"1234567890"));
    }
}
