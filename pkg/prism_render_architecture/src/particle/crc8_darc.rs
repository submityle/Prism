//! `CRC-8/DARC` implementation using a bit-wise algorithm.
//!
//! Parameters: width=8, poly=0x39, init=0x0, refin=true, refout=true,
//! xorout=0x0, check=0x15.

const WIDTH: u32 = 8;
const POLY: u32 = 0x39;
const INIT: u32 = 0x0;
const XOROUT: u32 = 0x0;
const REFLECT_IN: bool = true;
const REFLECT_OUT: bool = true;
const MASK: u32 = 0xff;
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

/// Compute the `CRC-8/DARC` checksum of `data`.
pub fn crc8_darc(data: &[u8]) -> u8 {
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
    use super::crc8_darc;

    #[test]
    fn vector_empty() {
        assert!(crc8_darc(b"") == 0x00);
    }

    #[test]
    fn vector_z00() {
        assert!(crc8_darc(&[0x00]) == 0x00);
    }

    #[test]
    fn vector_ff() {
        assert!(crc8_darc(&[0xff]) == 0xc6);
    }

    #[test]
    fn vector_a() {
        assert!(crc8_darc(b"a") == 0x1b);
    }

    #[test]
    fn vector_b() {
        assert!(crc8_darc(b"b") == 0x8d);
    }

    #[test]
    fn vector_ab() {
        assert!(crc8_darc(b"ab") == 0x4f);
    }

    #[test]
    fn vector_abc() {
        assert!(crc8_darc(b"abc") == 0x0d);
    }

    #[test]
    fn vector_two_zeros() {
        assert!(crc8_darc(&[0x00, 0x00]) == 0x00);
    }

    #[test]
    fn vector_01() {
        assert!(crc8_darc(&[0x01]) == 0x72);
    }

    #[test]
    fn vector_02() {
        assert!(crc8_darc(&[0x02]) == 0xe4);
    }

    #[test]
    fn vector_7f() {
        assert!(crc8_darc(&[0x7f]) == 0x5a);
    }

    #[test]
    fn vector_80() {
        assert!(crc8_darc(&[0x80]) == 0x9c);
    }

    #[test]
    fn vector_aa_55() {
        assert!(crc8_darc(&[0xaa, 0x55]) == 0x2f);
    }

    #[test]
    fn vector_55_aa() {
        assert!(crc8_darc(&[0x55, 0xaa]) == 0x2e);
    }

    #[test]
    fn vector_deadbeef() {
        assert!(crc8_darc(&[0xde, 0xad, 0xbe, 0xef]) == 0xc0);
    }

    #[test]
    fn vector_hello() {
        assert!(crc8_darc(b"Hello") == 0x00);
    }

    #[test]
    fn vector_fox() {
        assert!(crc8_darc(b"The quick brown fox") == 0x2b);
    }

    #[test]
    fn vector_four_zeros() {
        assert!(crc8_darc(&[0x00; 4]) == 0x00);
    }

    #[test]
    fn vector_four_ff() {
        assert!(crc8_darc(&[0xff; 4]) == 0x03);
    }

    #[test]
    fn vector_0_15() {
        let mut data = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc8_darc(&data) == 0xbb);
    }

    #[test]
    fn vector_12345678() {
        assert!(crc8_darc(&[0x12, 0x34, 0x56, 0x78]) == 0x6b);
    }

    #[test]
    fn vector_check() {
        assert!(crc8_darc(b"123456789") == 0x15);
    }

    #[test]
    fn vector_a5_1000() {
        let data = [0xA5u8; 1000];
        assert!(crc8_darc(&data) == 0x77);
    }

    #[test]
    fn vector_all256() {
        let mut data = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc8_darc(&data) == 0x3c);
    }

    #[test]
    fn check_constant_matches() {
        const CHECK: u8 = 0x15;
        assert!(crc8_darc(b"123456789") == CHECK);
    }

    #[test]
    fn determinism() {
        let data = [0x12u8, 0x34, 0x56, 0x78];
        assert!(crc8_darc(&data) == crc8_darc(&data));
    }

    #[test]
    fn order_sensitive() {
        assert!(crc8_darc(&[0xaa, 0x55]) != crc8_darc(&[0x55, 0xaa]));
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc8_darc(b"a") != crc8_darc(b"b"));
    }

    #[test]
    fn prefix_differs() {
        assert!(crc8_darc(b"a") != crc8_darc(b"ab"));
    }

    #[test]
    fn length_sensitive() {
        assert!(crc8_darc(&[0x00]) == crc8_darc(&[0x00, 0x00]));
        assert!(crc8_darc(&[0xff]) != crc8_darc(&[0xff, 0xff]));
    }

    #[test]
    fn single_bytes_mostly_distinct() {
        let mut seen = [false; 256];
        let mut collisions = 0u32;
        let mut i = 0usize;
        while i < 256 {
            let v = crc8_darc(&[i as u8]);
            if seen[v as usize] {
                collisions += 1;
            }
            seen[v as usize] = true;
            i += 1;
        }
        assert!(collisions == 0);
    }

    #[test]
    fn result_in_byte_range() {
        let mut i = 0usize;
        while i < 256 {
            let v = crc8_darc(&[i as u8]);
            assert!((0..=0xffu8).contains(&v));
            i += 1;
        }
    }

    #[test]
    fn empty_is_zero() {
        assert!(crc8_darc(&[]) == 0x00);
    }

    #[test]
    fn two_distinct_single_bytes() {
        assert!(crc8_darc(&[0x01]) != crc8_darc(&[0x02]));
        assert!(crc8_darc(&[0x7f]) != crc8_darc(&[0x80]));
    }

    #[test]
    fn deadbeef_differs_from_check() {
        assert!(crc8_darc(&[0xde, 0xad, 0xbe, 0xef]) != crc8_darc(b"123456789"));
    }
}
