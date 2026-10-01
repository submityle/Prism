//! `CRC-21/CAN-FD` bit-wise implementation.
//!
//! Parameters: width = 21, poly = `0x102899`, init = `0x0`, `refin` = false,
//! `refout` = false, `xorout` = `0x0`, check = `0x0ed841`.

const WIDTH: u32 = 21;
const POLY: u32 = 0x102899;
const INIT: u32 = 0x0;
const XOROUT: u32 = 0x0;
const REFLECT_IN: bool = false;
const REFLECT_OUT: bool = false;
const MASK: u32 = 0x1f_ffff;
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

/// Compute the `CRC-21/CAN-FD` checksum of `data`.
pub fn crc21_can_fd(data: &[u8]) -> u32 {
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
    use super::crc21_can_fd;

    #[test]
    fn vec_empty() {
        assert!(crc21_can_fd(b"") == 0x000000);
    }

    #[test]
    fn vec_z00() {
        assert!(crc21_can_fd(&[0x00]) == 0x000000);
    }

    #[test]
    fn vec_ff() {
        assert!(crc21_can_fd(&[0xff]) == 0x104a5a);
    }

    #[test]
    fn vec_a() {
        assert!(crc21_can_fd(b"a") == 0x1a0ed9);
    }

    #[test]
    fn vec_b() {
        assert!(crc21_can_fd(b"b") == 0x1a5feb);
    }

    #[test]
    fn vec_ab() {
        assert!(crc21_can_fd(b"ab") == 0x13dcfc);
    }

    #[test]
    fn vec_abc() {
        assert!(crc21_can_fd(b"abc") == 0x1ccff1);
    }

    #[test]
    fn vec_two_zeros() {
        assert!(crc21_can_fd(&[0, 0]) == 0x000000);
    }

    #[test]
    fn vec_01() {
        assert!(crc21_can_fd(&[0x01]) == 0x102899);
    }

    #[test]
    fn vec_02() {
        assert!(crc21_can_fd(&[0x02]) == 0x1079ab);
    }

    #[test]
    fn vec_7f() {
        assert!(crc21_can_fd(&[0x7f]) == 0x18252d);
    }

    #[test]
    fn vec_80() {
        assert!(crc21_can_fd(&[0x80]) == 0x086f77);
    }

    #[test]
    fn vec_aa_55() {
        assert!(crc21_can_fd(&[0xaa, 0x55]) == 0x13b15e);
    }

    #[test]
    fn vec_55_aa() {
        assert!(crc21_can_fd(&[0x55, 0xaa]) == 0x11b7d8);
    }

    #[test]
    fn vec_deadbeef() {
        assert!(crc21_can_fd(&[0xde, 0xad, 0xbe, 0xef]) == 0x1b60a6);
    }

    #[test]
    fn vec_hello() {
        assert!(crc21_can_fd(b"Hello") == 0x127422);
    }

    #[test]
    fn vec_fox() {
        assert!(crc21_can_fd(b"The quick brown fox") == 0x11c3f8);
    }

    #[test]
    fn vec_four_zeros() {
        assert!(crc21_can_fd(&[0; 4]) == 0x000000);
    }

    #[test]
    fn vec_four_ff() {
        assert!(crc21_can_fd(&[0xff; 4]) == 0x0dbf8e);
    }

    #[test]
    fn vec_0_15() {
        let mut data = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc21_can_fd(&data) == 0x11b0a7);
    }

    #[test]
    fn vec_12345678() {
        assert!(crc21_can_fd(&[0x12, 0x34, 0x56, 0x78]) == 0x02cda1);
    }

    #[test]
    fn vec_check() {
        assert!(crc21_can_fd(b"123456789") == 0x0ed841);
    }

    #[test]
    fn vec_a5_1000() {
        let data = [0xA5u8; 1000];
        assert!(crc21_can_fd(&data) == 0x1cc8ca);
    }

    #[test]
    fn vec_all256() {
        let mut data = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc21_can_fd(&data) == 0x1f07e8);
    }

    #[test]
    fn check_constant_matches() {
        const CHECK: u32 = 0x0ed841;
        assert!(crc21_can_fd(b"123456789") == CHECK);
    }

    #[test]
    fn determinism() {
        let input = b"determinism sample";
        assert!(crc21_can_fd(input) == crc21_can_fd(input));
    }

    #[test]
    fn order_sensitive() {
        assert!(crc21_can_fd(&[0xaa, 0x55]) != crc21_can_fd(&[0x55, 0xaa]));
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc21_can_fd(b"a") != crc21_can_fd(b"b"));
    }

    #[test]
    fn prefix_differs() {
        assert!(crc21_can_fd(b"ab") != crc21_can_fd(b"abc"));
    }

    #[test]
    fn length_sensitive() {
        assert!(crc21_can_fd(&[0x00]) != crc21_can_fd(&[0x00, 0x00, 0x01]));
    }

    #[test]
    fn single_bytes_distinct() {
        let mut seen = [0u32; 256];
        let mut i = 0usize;
        while i < 256 {
            seen[i] = crc21_can_fd(&[i as u8]);
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
    fn result_within_range() {
        let mut i = 0usize;
        while i < 256 {
            let v = crc21_can_fd(&[i as u8]);
            assert!((0..=0x1f_ffffu32).contains(&v));
            i += 1;
        }
    }

    #[test]
    fn empty_within_range() {
        let v = crc21_can_fd(b"");
        assert!((0..=0x1f_ffffu32).contains(&v));
    }

    #[test]
    fn multi_byte_within_range() {
        let v = crc21_can_fd(b"The quick brown fox");
        assert!((0..=0x1f_ffffu32).contains(&v));
    }

    #[test]
    fn zeros_collapse() {
        assert!(crc21_can_fd(&[0x00]) == crc21_can_fd(&[0x00, 0x00]));
    }

    #[test]
    fn repeated_call_stable() {
        let data = [0xA5u8; 1000];
        assert!(crc21_can_fd(&data) == crc21_can_fd(&data));
    }
}
