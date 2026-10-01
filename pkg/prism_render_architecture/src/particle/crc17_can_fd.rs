//! `CRC`-17/CAN-FD implementation using a bit-wise algorithm.
//!
//! Parameters: width = 17, poly = `0x1685B`, init = `0x0`, refin = false,
//! refout = false, xorout = `0x0`, check = `0x04F03`.
//!
//! The algorithm processes each input byte from the most-significant bit
//! (`MSB`) first, feeding bits through the shift register and applying the
//! polynomial via `XOR` whenever the top bit is set. All arithmetic is masked
//! to the 17-bit register width.

const WIDTH: u32 = 17;
const POLY: u32 = 0x1685B;
const INIT: u32 = 0x0;
const XOROUT: u32 = 0x0;
const REFLECT_IN: bool = false;
const REFLECT_OUT: bool = false;
const MASK: u32 = 0x1_ffff;
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

/// Compute the `CRC`-17/CAN-FD checksum over `data`.
///
/// Returns the 17-bit checksum in the low bits of a `u32`.
pub fn crc17_can_fd(data: &[u8]) -> u32 {
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
    use super::*;

    #[test]
    fn vector_empty() {
        assert!(crc17_can_fd(b"") == 0x00000);
    }

    #[test]
    fn vector_z00() {
        assert!(crc17_can_fd(&[0x00]) == 0x00000);
    }

    #[test]
    fn vector_ff() {
        assert!(crc17_can_fd(&[0xff]) == 0x19052);
    }

    #[test]
    fn vector_a() {
        assert!(crc17_can_fd(b"a") == 0x03c43);
    }

    #[test]
    fn vector_b() {
        assert!(crc17_can_fd(b"b") == 0x0ecf5);
    }

    #[test]
    fn vector_ab() {
        assert!(crc17_can_fd(b"ab") == 0x15b9f);
    }

    #[test]
    fn vector_abc() {
        assert!(crc17_can_fd(b"abc") == 0x1cd05);
    }

    #[test]
    fn vector_two_zeros() {
        assert!(crc17_can_fd(&[0, 0]) == 0x00000);
    }

    #[test]
    fn vector_01() {
        assert!(crc17_can_fd(&[0x01]) == 0x1685b);
    }

    #[test]
    fn vector_02() {
        assert!(crc17_can_fd(&[0x02]) == 0x1b8ed);
    }

    #[test]
    fn vector_7f() {
        assert!(crc17_can_fd(&[0x7f]) == 0x1c829);
    }

    #[test]
    fn vector_80() {
        assert!(crc17_can_fd(&[0x80]) == 0x0587b);
    }

    #[test]
    fn vector_aa_55() {
        assert!(crc17_can_fd(&[0xaa, 0x55]) == 0x1b180);
    }

    #[test]
    fn vector_55_aa() {
        assert!(crc17_can_fd(&[0x55, 0xaa]) == 0x180bb);
    }

    #[test]
    fn vector_deadbeef() {
        assert!(crc17_can_fd(&[0xde, 0xad, 0xbe, 0xef]) == 0x08bbb);
    }

    #[test]
    fn vector_hello() {
        assert!(crc17_can_fd(b"Hello") == 0x1f65e);
    }

    #[test]
    fn vector_fox() {
        assert!(crc17_can_fd(b"The quick brown fox") == 0x124ee);
    }

    #[test]
    fn vector_four_zeros() {
        assert!(crc17_can_fd(&[0; 4]) == 0x00000);
    }

    #[test]
    fn vector_four_ff() {
        assert!(crc17_can_fd(&[0xff; 4]) == 0x00c7b);
    }

    #[test]
    fn vector_0_15() {
        let mut data = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc17_can_fd(&data) == 0x1443b);
    }

    #[test]
    fn vector_12345678() {
        assert!(crc17_can_fd(&[0x12, 0x34, 0x56, 0x78]) == 0x08a55);
    }

    #[test]
    fn vector_check() {
        assert!(crc17_can_fd(b"123456789") == 0x04f03);
    }

    #[test]
    fn vector_a5_1000() {
        let data = [0xA5u8; 1000];
        assert!(crc17_can_fd(&data) == 0x0f033);
    }

    #[test]
    fn vector_all256() {
        let mut data = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc17_can_fd(&data) == 0x196f8);
    }

    #[test]
    fn check_constant_matches() {
        let check: u32 = 0x04f03;
        assert!(crc17_can_fd(b"123456789") == check);
    }

    #[test]
    fn determinism() {
        let data = [0x12u8, 0x34, 0x56, 0x78, 0x9a];
        assert!(crc17_can_fd(&data) == crc17_can_fd(&data));
    }

    #[test]
    fn order_sensitive() {
        assert!(crc17_can_fd(&[0xaa, 0x55]) != crc17_can_fd(&[0x55, 0xaa]));
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc17_can_fd(b"a") != crc17_can_fd(b"b"));
    }

    #[test]
    fn prefix_differs() {
        assert!(crc17_can_fd(b"abc") != crc17_can_fd(b"abcd"));
    }

    #[test]
    fn length_sensitive() {
        assert!(crc17_can_fd(&[0xff]) != crc17_can_fd(&[0xff, 0xff]));
    }

    #[test]
    fn single_bytes_pairwise_distinct() {
        let mut seen = [0u32; 256];
        let mut i = 0usize;
        while i < 256 {
            seen[i] = crc17_can_fd(&[i as u8]);
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
        let inputs: [&[u8]; 5] = [b"", b"a", b"abc", b"123456789", b"The quick brown fox"];
        let mut i = 0usize;
        while i < inputs.len() {
            let v = crc17_can_fd(inputs[i]);
            assert!((0..=0x1_ffffu32).contains(&v));
            i += 1;
        }
    }

    #[test]
    fn result_in_range_all_single_bytes() {
        let mut i = 0usize;
        while i < 256 {
            let v = crc17_can_fd(&[i as u8]);
            assert!((0..=0x1_ffffu32).contains(&v));
            i += 1;
        }
    }

    #[test]
    fn empty_is_zero() {
        assert!(crc17_can_fd(&[]) == 0x00000);
    }

    #[test]
    fn reflect_identity_for_zero() {
        assert!(reflect(0, 17) == 0);
    }

    #[test]
    fn mask_bounds_hold() {
        let v = crc17_can_fd(&[0xff; 8]);
        assert!((v & MASK) == v);
    }
}
