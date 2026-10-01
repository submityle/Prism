//! `CRC-30/CDMA` bit-wise implementation.
//!
//! Parameters: width=30, poly=`0x2030B9C7`, init=`0x3FFFFFFF`, refin=false,
//! refout=false, xorout=`0x3FFFFFFF`, check=`0x04c34abf`.

const WIDTH: u32 = 30;
const POLY: u32 = 0x2030B9C7;
const INIT: u32 = 0x3FFFFFFF;
const XOROUT: u32 = 0x3FFFFFFF;
const REFLECT_IN: bool = false;
const REFLECT_OUT: bool = false;
const MASK: u32 = 0x3fff_ffff;
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

/// Compute the `CRC-30/CDMA` checksum of `data`.
pub fn crc30_cdma(data: &[u8]) -> u32 {
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
    (reg ^ XOROUT) & MASK
}

#[cfg(test)]
mod tests {
    use super::crc30_cdma;

    #[test]
    fn vec_empty() {
        assert!(crc30_cdma(b"") == 0x00000000);
    }

    #[test]
    fn vec_z00() {
        assert!(crc30_cdma(&[0x00]) == 0x1faf6629);
    }

    #[test]
    fn vec_ff() {
        assert!(crc30_cdma(&[0xff]) == 0x000000ff);
    }

    #[test]
    fn vec_a() {
        assert!(crc30_cdma(b"a") == 0x33b1ae2e);
    }

    #[test]
    fn vec_b() {
        assert!(crc30_cdma(b"b") == 0x33d0dda0);
    }

    #[test]
    fn vec_ab() {
        assert!(crc30_cdma(b"ab") == 0x3bf61451);
    }

    #[test]
    fn vec_abc() {
        assert!(crc30_cdma(b"abc") == 0x18462cac);
    }

    #[test]
    fn vec_two_zeros() {
        assert!(crc30_cdma(&[0, 0]) == 0x3f2e4585);
    }

    #[test]
    fn vec_01() {
        assert!(crc30_cdma(&[0x01]) == 0x3f9fdfee);
    }

    #[test]
    fn vec_02() {
        assert!(crc30_cdma(&[0x02]) == 0x3ffeac60);
    }

    #[test]
    fn vec_7f() {
        assert!(crc30_cdma(&[0x7f]) == 0x3078d542);
    }

    #[test]
    fn vec_80() {
        assert!(crc30_cdma(&[0x80]) == 0x2fd7b394);
    }

    #[test]
    fn vec_aa_55() {
        assert!(crc30_cdma(&[0xaa, 0x55]) == 0x2aa59842);
    }

    #[test]
    fn vec_55_aa() {
        assert!(crc30_cdma(&[0x55, 0xaa]) == 0x158b2238);
    }

    #[test]
    fn vec_deadbeef() {
        assert!(crc30_cdma(&[0xde, 0xad, 0xbe, 0xef]) == 0x39eb4c4f);
    }

    #[test]
    fn vec_hello() {
        assert!(crc30_cdma(b"Hello") == 0x21045345);
    }

    #[test]
    fn vec_fox() {
        assert!(crc30_cdma(b"The quick brown fox") == 0x0cf499a0);
    }

    #[test]
    fn vec_four_zeros() {
        assert!(crc30_cdma(&[0; 4]) == 0x2c201919);
    }

    #[test]
    fn vec_four_ff() {
        assert!(crc30_cdma(&[0xff; 4]) == 0x3f9e8c71);
    }

    #[test]
    fn vec_0_15() {
        let mut data = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc30_cdma(&data) == 0x17eac421);
    }

    #[test]
    fn vec_12345678() {
        assert!(crc30_cdma(&[0x12, 0x34, 0x56, 0x78]) == 0x10e2f754);
    }

    #[test]
    fn vec_check() {
        assert!(crc30_cdma(b"123456789") == 0x04c34abf);
    }

    #[test]
    fn vec_a5_1000() {
        let data = [0xA5u8; 1000];
        assert!(crc30_cdma(&data) == 0x1f419b48);
    }

    #[test]
    fn vec_all256() {
        let mut data = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc30_cdma(&data) == 0x08189a6e);
    }

    #[test]
    fn check_constant_matches() {
        const CHECK: u32 = 0x04c34abf;
        assert!(crc30_cdma(b"123456789") == CHECK);
    }

    #[test]
    fn determinism() {
        let data = [0x12u8, 0x34, 0x56, 0x78, 0x9a];
        assert!(crc30_cdma(&data) == crc30_cdma(&data));
    }

    #[test]
    fn order_sensitive() {
        assert!(crc30_cdma(&[0xaa, 0x55]) != crc30_cdma(&[0x55, 0xaa]));
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc30_cdma(b"a") != crc30_cdma(b"b"));
    }

    #[test]
    fn prefix_differs() {
        assert!(crc30_cdma(b"ab") != crc30_cdma(b"abc"));
    }

    #[test]
    fn length_sensitive() {
        assert!(crc30_cdma(&[0x00]) != crc30_cdma(&[0x00, 0x00]));
    }

    #[test]
    fn single_bytes_distinct() {
        let mut seen = [0u32; 256];
        let mut i = 0usize;
        while i < 256 {
            seen[i] = crc30_cdma(&[i as u8]);
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
        let inputs: [&[u8]; 5] = [b"", b"a", b"abc", b"123456789", &[0xff; 4]];
        let mut i = 0usize;
        while i < inputs.len() {
            let v = crc30_cdma(inputs[i]);
            assert!((0..=0x3fff_ffffu32).contains(&v));
            i += 1;
        }
    }

    #[test]
    fn empty_is_zero() {
        assert!(crc30_cdma(b"") == 0);
    }

    #[test]
    fn two_zeros_differs_from_four_zeros() {
        assert!(crc30_cdma(&[0; 2]) != crc30_cdma(&[0; 4]));
    }

    #[test]
    fn repeated_byte_length_sensitive() {
        assert!(crc30_cdma(&[0xA5; 3]) != crc30_cdma(&[0xA5; 4]));
    }
}
