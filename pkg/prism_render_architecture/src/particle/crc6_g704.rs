//! `CRC`-6/`G704` reference contract for the Prism particle engine.
//!
//! This module offers a deterministic, `CPU`-verifiable implementation of the
//! reflected `CRC`-6/`G704` checksum (also known as `CRC`-6/ITU). The
//! algorithm reflects every input byte before processing (`REFLECT_IN`), reflects
//! the final register (`REFLECT_OUT`) and performs the standard polynomial division
//! using bitwise `XOR` feedback whenever the register `MSB` differs from the next
//! incoming bit. Reflection moves the `LSB` to the `MSB` position for the given
//! bit width. The routine is kept free of heap allocation so it can run in a
//! `no_std` + `alloc` contract crate without pulling in collections.

/// Checksum width in bits.
const WIDTH: u32 = 6;
/// Generator polynomial (non-reflected form, implicit top bit omitted).
const POLY: u32 = 0x3;
/// Initial register value.
const INIT: u32 = 0x0;
/// Final `XOR` applied to the output register.
const XOROUT: u32 = 0x0;
/// Whether each input byte is reflected before being fed in.
const REFLECT_IN: bool = true;
/// Whether the final register is reflected before output.
const REFLECT_OUT: bool = true;
/// Mask that keeps only the `WIDTH` low bits.
const MASK: u32 = 0x3f;
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

/// Computes the `CRC`-6/`G704` checksum over `data`.
///
/// Returns the 6-bit checksum packed into the low bits of a [`u8`]; the two
/// high bits are always zero because the result is masked to [`MASK`].
pub fn crc6_g704(data: &[u8]) -> u8 {
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
    use super::crc6_g704;

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
        assert!(crc6_g704(b"") == 0x00);
    }

    #[test]
    fn test_z00() {
        assert!(crc6_g704(&[0x00]) == 0x00);
    }

    #[test]
    fn test_ff() {
        assert!(crc6_g704(&[0xff]) == 0x2c);
    }

    #[test]
    fn test_a() {
        // Single byte 'a' hashing to zero is a correct boundary result.
        assert!(crc6_g704(b"a") == 0x00);
    }

    #[test]
    fn test_b() {
        assert!(crc6_g704(b"b") == 0x3c);
    }

    #[test]
    fn test_ab() {
        assert!(crc6_g704(b"ab") == 0x3c);
    }

    #[test]
    fn test_abc() {
        assert!(crc6_g704(b"abc") == 0x10);
    }

    #[test]
    fn test_two_zeros() {
        assert!(crc6_g704(&[0x00, 0x00]) == 0x00);
    }

    #[test]
    fn test_01() {
        assert!(crc6_g704(&[0x01]) == 0x14);
    }

    #[test]
    fn test_02() {
        assert!(crc6_g704(&[0x02]) == 0x28);
    }

    #[test]
    fn test_7f() {
        assert!(crc6_g704(&[0x7f]) == 0x1c);
    }

    #[test]
    fn test_80() {
        assert!(crc6_g704(&[0x80]) == 0x30);
    }

    #[test]
    fn test_aa_55() {
        assert!(crc6_g704(&[0xaa, 0x55]) == 0x30);
    }

    #[test]
    fn test_55_aa() {
        assert!(crc6_g704(&[0x55, 0xaa]) == 0x22);
    }

    #[test]
    fn test_deadbeef() {
        assert!(crc6_g704(&[0xde, 0xad, 0xbe, 0xef]) == 0x23);
    }

    #[test]
    fn test_hello() {
        assert!(crc6_g704(b"Hello") == 0x14);
    }

    #[test]
    fn test_fox() {
        assert!(crc6_g704(b"The quick brown fox") == 0x25);
    }

    #[test]
    fn test_four_zeros() {
        assert!(crc6_g704(&[0u8; 4]) == 0x00);
    }

    #[test]
    fn test_four_ff() {
        assert!(crc6_g704(&[0xffu8; 4]) == 0x04);
    }

    #[test]
    fn test_0_15() {
        let data: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        assert!(crc6_g704(&data) == 0x30);
    }

    #[test]
    fn test_12345678() {
        // Byte array, not an ASCII string.
        assert!(crc6_g704(&[0x12, 0x34, 0x56, 0x78]) == 0x3b);
    }

    #[test]
    fn test_check() {
        assert!(crc6_g704(b"123456789") == 0x06);
    }

    #[test]
    fn test_check_constant() {
        // The published check constant for this parameter set.
        let check = crc6_g704(b"123456789");
        assert!(check == 0x06);
    }

    #[test]
    fn test_a5_1000() {
        let data = [0xA5u8; 1000];
        assert!(crc6_g704(&data) == 0x0d);
    }

    #[test]
    fn test_all256() {
        let data = all_bytes();
        assert!(crc6_g704(&data) == 0x2d);
    }

    #[test]
    fn test_determinism() {
        let data = b"The quick brown fox";
        let first = crc6_g704(data);
        let second = crc6_g704(data);
        assert!(first == second);
    }

    #[test]
    fn test_a_ne_b() {
        assert!(crc6_g704(b"a") != crc6_g704(b"b"));
    }

    #[test]
    fn test_prefix_differs() {
        // Sharing a prefix must not force an equal checksum.
        assert!(crc6_g704(b"ab") != crc6_g704(b"abc"));
    }

    #[test]
    fn test_length_sensitive() {
        let one = crc6_g704(b"a");
        let two = crc6_g704(b"ab");
        let three = crc6_g704(b"abc");
        assert!(one != two);
        assert!(two != three);
        assert!(one != three);
    }

    #[test]
    fn test_single_byte_pairwise() {
        // A sampled set of single bytes whose checksums must all differ.
        let samples: [u8; 6] = [0x01, 0x02, 0x7f, 0x80, 0xff, 0x62];
        let mut crcs = [0u8; 6];
        let mut i = 0usize;
        while i < samples.len() {
            crcs[i] = crc6_g704(&[samples[i]]);
            i += 1;
        }
        let mut a = 0usize;
        while a < crcs.len() {
            let mut b = a + 1;
            while b < crcs.len() {
                assert!(crcs[a] != crcs[b]);
                b += 1;
            }
            a += 1;
        }
    }

    #[test]
    fn test_result_in_range() {
        // Every checksum must fit in the low 6 bits.
        let inputs: [&[u8]; 5] = [b"", b"a", b"abc", b"Hello", b"123456789"];
        let mut i = 0usize;
        while i < inputs.len() {
            let c = crc6_g704(inputs[i]);
            assert!((0x00u8..=0x3f).contains(&c));
            i += 1;
        }
    }

    #[test]
    fn test_high_bits_clear() {
        // The two high bits are always zero after masking.
        let data = all_bytes();
        let c = crc6_g704(&data);
        assert!((c & 0xc0) == 0);
    }

    #[test]
    fn test_empty_matches_zero_byte() {
        // INIT is zero and the polynomial leaves a zero byte unchanged.
        assert!(crc6_g704(b"") == crc6_g704(&[0x00]));
    }

    #[test]
    fn test_order_matters() {
        assert!(crc6_g704(&[0xaa, 0x55]) != crc6_g704(&[0x55, 0xaa]));
    }

    #[test]
    fn test_ff_vs_four_ff() {
        assert!(crc6_g704(&[0xff]) != crc6_g704(&[0xffu8; 4]));
    }
}
