//! `CRC-4/G-704` golden reference for the `CPU` side of the particle subsystem.
//!
//! This module provides a pure-integer, `no_std` + `alloc` friendly bit-wise
//! `CRC` computation. It serves as the verifiable gold standard against which
//! other implementations (for example a `GPU` port) can be checked. The
//! algorithm reflects input and output bits and applies the final `XOR` output
//! constant, matching the `CRC-4/G-704` parameters.

const WIDTH: u32 = 4;
const POLY: u32 = 0x3;
const INIT: u32 = 0x0;
const XOROUT: u32 = 0x0;
const REFLECT_IN: bool = true;
const REFLECT_OUT: bool = true;
const MASK: u32 = 0xf;
const TOPBIT: u32 = 1u32 << (WIDTH - 1);

/// Reflect the low `bits` bits of `value`, reversing their order.
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

/// Compute the `CRC-4/G-704` checksum of `data`.
///
/// Each byte is processed most-significant-bit (`MSB`) first after optional
/// input reflection, updating the shift register against the polynomial. The
/// returned value always lies in the inclusive range `0..=0xf`.
pub fn crc4_g704(data: &[u8]) -> u8 {
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
    use super::crc4_g704;

    /// Build the 16-byte sequential input `0..=15` without heap allocation.
    fn seq_0_15() -> [u8; 16] {
        let mut arr = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            arr[i] = i as u8;
            i += 1;
        }
        arr
    }

    /// Build the 256-byte sequential input `0..=255` without heap allocation.
    fn all256() -> [u8; 256] {
        let mut arr = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            arr[i] = i as u8;
            i += 1;
        }
        arr
    }

    // ---- anchor vectors (hard-coded truth) ----
    #[test]
    fn anchor_80() {
        assert!(crc4_g704(&[0x80u8]) == 0xc);
    }
    #[test]
    fn anchor_12345678() {
        assert!(crc4_g704(&[0x12u8, 0x34, 0x56, 0x78]) == 0x3);
    }
    #[test]
    fn anchor_empty() {
        assert!(crc4_g704(b"") == 0x0);
    }
    #[test]
    fn anchor_z00() {
        assert!(crc4_g704(&[0x00u8]) == 0x0);
    }
    #[test]
    fn anchor_ff() {
        assert!(crc4_g704(&[0xffu8]) == 0x2);
    }
    #[test]
    fn anchor_a() {
        assert!(crc4_g704(b"a") == 0x2);
    }
    #[test]
    fn anchor_b() {
        assert!(crc4_g704(b"b") == 0xb);
    }
    #[test]
    fn anchor_ab() {
        assert!(crc4_g704(b"ab") == 0x5);
    }
    #[test]
    fn anchor_abc() {
        assert!(crc4_g704(b"abc") == 0xe);
    }
    #[test]
    fn anchor_two_zeros() {
        assert!(crc4_g704(&[0x00u8, 0x00]) == 0x0);
    }
    #[test]
    fn anchor_01() {
        assert!(crc4_g704(&[0x01u8]) == 0x7);
    }
    #[test]
    fn anchor_02() {
        assert!(crc4_g704(&[0x02u8]) == 0xe);
    }
    #[test]
    fn anchor_7f() {
        assert!(crc4_g704(&[0x7fu8]) == 0xe);
    }
    #[test]
    fn anchor_aa_55() {
        assert!(crc4_g704(&[0xaau8, 0x55]) == 0xa);
    }
    #[test]
    fn anchor_55_aa() {
        assert!(crc4_g704(&[0x55u8, 0xaa]) == 0x6);
    }
    #[test]
    fn anchor_deadbeef() {
        assert!(crc4_g704(&[0xdeu8, 0xad, 0xbe, 0xef]) == 0x0);
    }
    #[test]
    fn anchor_hello() {
        assert!(crc4_g704(b"Hello") == 0x7);
    }
    #[test]
    fn anchor_fox() {
        assert!(crc4_g704(b"The quick brown fox") == 0x6);
    }
    #[test]
    fn anchor_four_zeros() {
        assert!(crc4_g704(&[0x00u8; 4]) == 0x0);
    }
    #[test]
    fn anchor_four_ff() {
        assert!(crc4_g704(&[0xffu8; 4]) == 0xa);
    }
    #[test]
    fn anchor_0_15() {
        let d = seq_0_15();
        assert!(crc4_g704(&d) == 0xf);
    }
    #[test]
    fn anchor_check() {
        assert!(crc4_g704(b"123456789") == 0x7);
    }
    #[test]
    fn anchor_a5_1000() {
        let d = [0xa5u8; 1000];
        assert!(crc4_g704(&d) == 0x8);
    }
    #[test]
    fn anchor_all256() {
        let d = all256();
        assert!(crc4_g704(&d) == 0x5);
    }

    // ---- property / robustness tests ----
    #[test]
    fn determinism_small() {
        assert!(crc4_g704(b"abc") == crc4_g704(b"abc"));
        assert!(crc4_g704(&[0xffu8, 0x55]) == crc4_g704(&[0xffu8, 0x55]));
    }
    #[test]
    fn determinism_large() {
        let d = [0xa5u8; 1000];
        assert!(crc4_g704(&d) == crc4_g704(&d));
    }
    #[test]
    fn order_sensitive() {
        assert!(crc4_g704(&[0xaau8, 0x55]) != crc4_g704(&[0x55u8, 0xaa]));
    }
    #[test]
    fn a_differs_from_b() {
        assert!(crc4_g704(b"a") != crc4_g704(b"b"));
    }
    #[test]
    fn prefix_differs() {
        assert!(crc4_g704(b"a") != crc4_g704(b"ab"));
    }
    #[test]
    fn length_sensitive() {
        assert!(crc4_g704(&[0xffu8]) != crc4_g704(&[0xffu8; 4]));
    }
    #[test]
    fn ab_differs_from_abc() {
        assert!(crc4_g704(b"ab") != crc4_g704(b"abc"));
    }
    #[test]
    fn all_single_bytes_in_range() {
        let mut v = 0u16;
        while v <= 255 {
            let r = crc4_g704(&[v as u8]);
            assert!((0u8..=0xf).contains(&r));
            v += 1;
        }
    }
    #[test]
    fn single_byte_coverage_and_distinctness() {
        let mut counts = [0u32; 16];
        let mut v = 0u16;
        while v <= 255 {
            let r = crc4_g704(&[v as u8]);
            counts[r as usize] += 1;
            v += 1;
        }
        let mut total = 0u32;
        let mut nonzero = 0u32;
        let mut i = 0usize;
        while i < 16 {
            total += counts[i];
            if counts[i] > 0 {
                nonzero += 1;
            }
            i += 1;
        }
        assert!(total == 256);
        assert!(nonzero >= 2);
    }
    #[test]
    fn result_in_range_various() {
        let inputs: [&[u8]; 6] = [
            b"",
            &[0x00u8],
            b"abc",
            &[0xaau8, 0x55],
            b"123456789",
            &[0xa5u8; 10],
        ];
        for inp in inputs {
            let r = crc4_g704(inp);
            assert!((0u8..=0xf).contains(&r));
        }
    }
    #[test]
    fn check_constant_is_7() {
        assert!(crc4_g704(b"123456789") == 0x7);
    }
    #[test]
    fn zeros_family_all_zero() {
        assert!(crc4_g704(b"") == 0x0);
        assert!(crc4_g704(&[0x00u8]) == 0x0);
        assert!(crc4_g704(&[0x00u8, 0x00]) == 0x0);
        assert!(crc4_g704(&[0x00u8; 4]) == 0x0);
    }
    #[test]
    fn ascii_12345678_in_range() {
        let r = crc4_g704(b"12345678");
        assert!((0u8..=0xf).contains(&r));
    }
}
