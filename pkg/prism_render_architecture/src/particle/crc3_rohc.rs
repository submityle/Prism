//! `CRC`-3/`ROHC` reference contract for the Prism particle engine.
//!
//! This module offers a deterministic, `CPU`-verifiable implementation of the
//! reflected `CRC`-3/`ROHC` checksum. The algorithm reflects every input byte
//! before processing (`REFLECT_IN`), reflects the final register
//! (`REFLECT_OUT`) and performs the standard polynomial division using bitwise
//! `XOR` feedback whenever the register `MSB` differs from the next incoming
//! bit. Reflection moves the `LSB` to the `MSB` position for the given bit
//! width. The routine is kept free of heap allocation so it can run in a
//! `no_std` + `alloc` contract crate without pulling in collections.

/// Checksum width in bits.
const WIDTH: u32 = 3;
/// Generator polynomial (non-reflected form, implicit top bit omitted).
const POLY: u32 = 0x3;
/// Initial register value.
const INIT: u32 = 0x7;
/// Final `XOR` applied to the output register.
const XOROUT: u32 = 0x0;
/// Whether each input byte is reflected before being fed in.
const REFLECT_IN: bool = true;
/// Whether the final register is reflected before output.
const REFLECT_OUT: bool = true;
/// Mask that keeps only the `WIDTH` low bits.
const MASK: u32 = 0x7;
/// Mask selecting the register top bit.
const TOPBIT: u32 = 1u32 << (WIDTH - 1);
/// Number of bits contained in a single input byte.
const BITS_PER_BYTE: u32 = 8;

/// Reflects the low `bits` bits of `value`, mirroring bit `i` to bit
/// `bits - 1 - i`. This is used to swap the `LSB`/`MSB` ordering required by
/// the reflected `CRC` definition.
fn reflect(value: u32, bits: u32) -> u32 {
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

/// Computes the `CRC`-3/`ROHC` checksum over `data`.
///
/// Returns the 3-bit checksum packed into the low bits of a [`u8`]; the five
/// high bits are always zero because the result is masked to [`MASK`].
pub fn crc3_rohc(data: &[u8]) -> u8 {
    let mut reg = INIT & MASK;
    let mut idx = 0usize;
    while idx < data.len() {
        let byte = u32::from(data[idx]);
        let b = if REFLECT_IN {
            reflect(byte, BITS_PER_BYTE)
        } else {
            byte
        };
        let mut i = 0u32;
        while i < BITS_PER_BYTE {
            let bit = (b >> (BITS_PER_BYTE - 1 - i)) & 1;
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
    use super::crc3_rohc;
    use super::reflect;

    /// Builds the 256-byte array [0, 1, .., 255] without heap allocation.
    fn all_bytes() -> [u8; 256] {
        let mut out = [0u8; 256];
        let mut i = 0usize;
        while i < out.len() {
            out[i] = i as u8;
            i += 1;
        }
        out
    }

    #[test]
    fn test_empty() {
        assert!(crc3_rohc(b"") == 0x7);
    }

    #[test]
    fn test_check() {
        assert!(crc3_rohc(b"123456789") == 0x6);
    }

    #[test]
    fn test_check_constant() {
        // The published catalogue check constant for this parameter set.
        let check = crc3_rohc(b"123456789");
        assert!(check == 0x6);
    }

    #[test]
    fn test_z00() {
        assert!(crc3_rohc(&[0x00]) == 0x5);
    }

    #[test]
    fn test_ff() {
        assert!(crc3_rohc(&[0xff]) == 0x3);
    }

    #[test]
    fn test_fox() {
        // 19 bytes, no trailing period.
        assert!(crc3_rohc(b"The quick brown fox") == 0x7);
    }

    #[test]
    fn test_arr_0123() {
        assert!(crc3_rohc(&[0, 1, 2, 3]) == 0x7);
    }

    #[test]
    fn test_a() {
        assert!(crc3_rohc(b"a") == 0x7);
    }

    #[test]
    fn test_b() {
        assert!(crc3_rohc(b"b") == 0x0);
    }

    #[test]
    fn test_ab() {
        assert!(crc3_rohc(b"ab") == 0x0);
    }

    #[test]
    fn test_abc() {
        assert!(crc3_rohc(b"abc") == 0x3);
    }

    #[test]
    fn test_two_zeros() {
        assert!(crc3_rohc(&[0x00, 0x00]) == 0x4);
    }

    #[test]
    fn test_four_zeros() {
        assert!(crc3_rohc(&[0u8; 4]) == 0x1);
    }

    #[test]
    fn test_four_ff() {
        assert!(crc3_rohc(&[0xffu8; 4]) == 0x6);
    }

    #[test]
    fn test_01() {
        assert!(crc3_rohc(&[0x01]) == 0x3);
    }

    #[test]
    fn test_02() {
        assert!(crc3_rohc(&[0x02]) == 0x4);
    }

    #[test]
    fn test_7f() {
        assert!(crc3_rohc(&[0x7f]) == 0x5);
    }

    #[test]
    fn test_80() {
        assert!(crc3_rohc(&[0x80]) == 0x3);
    }

    #[test]
    fn test_hello() {
        assert!(crc3_rohc(b"Hello") == 0x5);
    }

    #[test]
    fn test_aa_55() {
        assert!(crc3_rohc(&[0xaa, 0x55]) == 0x4);
    }

    #[test]
    fn test_55_aa() {
        assert!(crc3_rohc(&[0x55, 0xaa]) == 0x1);
    }

    #[test]
    fn test_deadbeef() {
        assert!(crc3_rohc(&[0xde, 0xad, 0xbe, 0xef]) == 0x4);
    }

    #[test]
    fn test_12345678() {
        // Byte array, not an ASCII string.
        assert!(crc3_rohc(&[0x12, 0x34, 0x56, 0x78]) == 0x6);
    }

    #[test]
    fn test_0_15() {
        let data: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        assert!(crc3_rohc(&data) == 0x6);
    }

    #[test]
    fn test_all256() {
        let data = all_bytes();
        assert!(crc3_rohc(&data) == 0x6);
    }

    #[test]
    fn test_a5_1000() {
        let data = [0xA5u8; 1000];
        assert!(crc3_rohc(&data) == 0x4);
    }

    #[test]
    fn test_determinism() {
        let data = b"The quick brown fox";
        let first = crc3_rohc(data);
        let second = crc3_rohc(data);
        assert!(first == second);
    }

    #[test]
    fn test_a_ne_b() {
        assert!(crc3_rohc(b"a") != crc3_rohc(b"b"));
    }

    #[test]
    fn test_prefix_differs() {
        // Sharing a prefix must not force an equal checksum here.
        assert!(crc3_rohc(b"ab") != crc3_rohc(b"abc"));
    }

    #[test]
    fn test_length_sensitive() {
        let one = crc3_rohc(b"a");
        let two = crc3_rohc(b"ab");
        let three = crc3_rohc(b"abc");
        assert!(one != two);
        assert!(two != three);
        assert!(one != three);
    }

    #[test]
    fn test_result_in_range() {
        // Every checksum must fit in the low 3 bits.
        let inputs: [&[u8]; 5] = [b"", b"a", b"abc", b"Hello", b"123456789"];
        let mut i = 0usize;
        while i < inputs.len() {
            let c = crc3_rohc(inputs[i]);
            assert!((0x0u8..=0x7).contains(&c));
            i += 1;
        }
    }

    #[test]
    fn test_range_all_bytes() {
        // The checksum of every single byte stays within the 3-bit range.
        let data = all_bytes();
        let mut i = 0usize;
        while i < data.len() {
            let c = crc3_rohc(&[data[i]]);
            assert!((0x0u8..=0x7).contains(&c));
            i += 1;
        }
    }

    #[test]
    fn test_high_bits_clear() {
        // The five high bits are always zero after masking.
        let data = all_bytes();
        let c = crc3_rohc(&data);
        assert!((c & 0xf8) == 0);
    }

    #[test]
    fn test_order_matters() {
        assert!(crc3_rohc(&[0xaa, 0x55]) != crc3_rohc(&[0x55, 0xaa]));
    }

    #[test]
    fn test_ff_vs_four_ff() {
        assert!(crc3_rohc(&[0xff]) != crc3_rohc(&[0xffu8; 4]));
    }

    #[test]
    fn test_empty_differs_two_zeros() {
        // A non-zero INIT means the empty input is not the all-zero result.
        assert!(crc3_rohc(b"") != crc3_rohc(&[0x00, 0x00]));
    }

    #[test]
    fn test_init_nonzero() {
        // INIT is 0x7, so the empty input differs from a single zero byte.
        assert!(crc3_rohc(b"") != crc3_rohc(&[0x00]));
    }

    #[test]
    fn test_zeros_length_sensitive() {
        let one = crc3_rohc(&[0x00]);
        let two = crc3_rohc(&[0x00, 0x00]);
        let four = crc3_rohc(&[0u8; 4]);
        assert!(one != two);
        assert!(two != four);
    }

    #[test]
    fn test_reflect() {
        assert!(reflect(0x01, 8) == 0x80);
        assert!(reflect(0x80, 8) == 0x01);
        assert!(reflect(0x00, 8) == 0x00);
        assert!(reflect(0xff, 8) == 0xff);
    }

    #[test]
    fn test_reflect_width3() {
        assert!(reflect(0x1, 3) == 0x4);
        assert!(reflect(0x4, 3) == 0x1);
        assert!(reflect(0x2, 3) == 0x2);
        assert!(reflect(0x7, 3) == 0x7);
    }

    #[test]
    fn test_reflect_involution() {
        // Reflecting twice over the same width restores the original value.
        let mut v = 0u32;
        while v < 256 {
            assert!(reflect(reflect(v, 8), 8) == v);
            v += 1;
        }
    }
}
