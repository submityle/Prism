//! `CRC-15/MPT1327` contract module exposing a deterministic, `CPU`-verifiable
//! checksum over arbitrary byte slices.
//!
//! The implementation uses the classic bit-wise (`MSB`-first) `CRC` register
//! loop. The register is seeded with the `INIT` constant; for each input bit
//! the current `MSB` is `XOR`-ed with the incoming bit, the register is shifted
//! left, and the generator polynomial `POLY` is applied when that combined bit
//! is set. After all bytes are consumed the register is optionally reflected
//! (`LSB`-first) and `XOR`-ed with `XOROUT` to form the final 15-bit result.
//!
//! Parameters for `CRC-15/MPT1327`: width 15, polynomial `0x6815`, init `0x0`,
//! `XOROUT` `0x1`, no input/output reflection. The 15-bit result is returned in
//! the low bits of a `u16`.

const WIDTH: u32 = 15;
const POLY: u32 = 0x6815;
const INIT: u32 = 0x0;
const XOROUT: u32 = 0x1;
const REFLECT_IN: bool = false;
const REFLECT_OUT: bool = false;
const MASK: u32 = 0x7fff;
const TOPBIT: u32 = 1u32 << (WIDTH - 1);
const BITS_PER_BYTE: u32 = 8;

/// Reflect the low `bits` of `value`, reversing their bit order.
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

/// Compute the `CRC-15/MPT1327` checksum of `data`.
///
/// Returns the 15-bit checksum in the low bits of the `u16`.
pub fn crc15_mpt1327(data: &[u8]) -> u16 {
    let mut reg = INIT & MASK;
    let mut idx = 0usize;
    while idx < data.len() {
        let byte = data[idx];
        let b = if REFLECT_IN {
            reflect(u32::from(byte), BITS_PER_BYTE)
        } else {
            u32::from(byte)
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
    ((reg ^ XOROUT) & MASK) as u16
}

#[cfg(test)]
mod tests {
    use super::crc15_mpt1327;

    const EMPTY: &[u8] = b"";
    const BYTE_A: &[u8] = b"a";
    const BYTE_B: &[u8] = b"b";
    const AB: &[u8] = b"ab";
    const ABC: &[u8] = b"abc";
    const HELLO: &[u8] = b"Hello";
    const FOX: &[u8] = b"The quick brown fox";
    const CHECK: &[u8] = b"123456789";

    const ZERO_TO_FIFTEEN_LEN: usize = 16;
    const ALL_BYTES_LEN: usize = 256;
    const A5_REPEAT_LEN: usize = 1000;
    const A5_BYTE: u8 = 0xA5;
    const SAMPLE_COUNT: usize = 5;
    const MAX_RESULT: u32 = 0x7fff;

    #[test]
    fn vector_empty() {
        assert!(crc15_mpt1327(EMPTY) == 0x0001);
    }

    #[test]
    fn vector_single_zero() {
        assert!(crc15_mpt1327(&[0x00]) == 0x0001);
    }

    #[test]
    fn vector_single_ff() {
        assert!(crc15_mpt1327(&[0xff]) == 0x3b07);
    }

    #[test]
    fn vector_byte_a() {
        assert!(crc15_mpt1327(BYTE_A) == 0x0cf8);
    }

    #[test]
    fn vector_byte_b() {
        assert!(crc15_mpt1327(BYTE_B) == 0x5cd2);
    }

    #[test]
    fn vector_ab() {
        assert!(crc15_mpt1327(AB) == 0x54fc);
    }

    #[test]
    fn vector_abc() {
        assert!(crc15_mpt1327(ABC) == 0x6c1a);
    }

    #[test]
    fn vector_two_zeros() {
        assert!(crc15_mpt1327(&[0x00, 0x00]) == 0x0001);
    }

    #[test]
    fn vector_single_01() {
        assert!(crc15_mpt1327(&[0x01]) == 0x6814);
    }

    #[test]
    fn vector_single_02() {
        assert!(crc15_mpt1327(&[0x02]) == 0x383e);
    }

    #[test]
    fn vector_single_7f() {
        assert!(crc15_mpt1327(&[0x7f]) == 0x5d82);
    }

    #[test]
    fn vector_single_80() {
        assert!(crc15_mpt1327(&[0x80]) == 0x6684);
    }

    #[test]
    fn vector_aa_55() {
        assert!(crc15_mpt1327(&[0xaa, 0x55]) == 0x635a);
    }

    #[test]
    fn vector_55_aa() {
        assert!(crc15_mpt1327(&[0x55, 0xaa]) == 0x6323);
    }

    #[test]
    fn vector_deadbeef() {
        assert!(crc15_mpt1327(&[0xde, 0xad, 0xbe, 0xef]) == 0x4a52);
    }

    #[test]
    fn vector_hello() {
        assert!(crc15_mpt1327(HELLO) == 0x7727);
    }

    #[test]
    fn vector_fox() {
        assert!(crc15_mpt1327(FOX) == 0x2ba0);
    }

    #[test]
    fn vector_four_zeros() {
        assert!(crc15_mpt1327(&[0u8; 4]) == 0x0001);
    }

    #[test]
    fn vector_four_ff() {
        assert!(crc15_mpt1327(&[0xffu8; 4]) == 0x2bfc);
    }

    #[test]
    fn vector_zero_to_fifteen() {
        let mut buf = [0u8; ZERO_TO_FIFTEEN_LEN];
        let mut i = 0usize;
        while i < ZERO_TO_FIFTEEN_LEN {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc15_mpt1327(&buf) == 0x40f5);
    }

    #[test]
    fn vector_bytes_12345678() {
        assert!(crc15_mpt1327(&[0x12, 0x34, 0x56, 0x78]) == 0x519e);
    }

    #[test]
    fn vector_check() {
        assert!(crc15_mpt1327(CHECK) == 0x2566);
    }

    #[test]
    fn vector_a5_repeat_1000() {
        let mut buf = [0u8; A5_REPEAT_LEN];
        let mut i = 0usize;
        while i < A5_REPEAT_LEN {
            buf[i] = A5_BYTE;
            i += 1;
        }
        assert!(crc15_mpt1327(&buf) == 0x3f78);
    }

    #[test]
    fn vector_all_bytes() {
        let mut buf = [0u8; ALL_BYTES_LEN];
        let mut i = 0usize;
        while i < ALL_BYTES_LEN {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc15_mpt1327(&buf) == 0x19bf);
    }

    #[test]
    fn check_constant_matches() {
        assert!(crc15_mpt1327(b"123456789") == 0x2566);
    }

    #[test]
    fn determinism_fox() {
        let first = crc15_mpt1327(FOX);
        let second = crc15_mpt1327(FOX);
        assert!(first == second);
    }

    #[test]
    fn determinism_deadbeef() {
        let input = [0xde, 0xad, 0xbe, 0xef];
        assert!(crc15_mpt1327(&input) == crc15_mpt1327(&input));
    }

    #[test]
    fn distinct_a_vs_b() {
        assert!(crc15_mpt1327(BYTE_A) != crc15_mpt1327(BYTE_B));
    }

    #[test]
    fn prefix_differs_abc_vs_ab() {
        assert!(crc15_mpt1327(ABC) != crc15_mpt1327(AB));
    }

    #[test]
    fn length_sensitive_ff() {
        assert!(crc15_mpt1327(&[0xff]) != crc15_mpt1327(&[0xff, 0xff]));
    }

    #[test]
    fn order_sensitive_aa55_vs_55aa() {
        assert!(crc15_mpt1327(&[0xaa, 0x55]) != crc15_mpt1327(&[0x55, 0xaa]));
    }

    #[test]
    fn single_byte_01_vs_02() {
        assert!(crc15_mpt1327(&[0x01]) != crc15_mpt1327(&[0x02]));
    }

    #[test]
    fn single_byte_7f_vs_80() {
        assert!(crc15_mpt1327(&[0x7f]) != crc15_mpt1327(&[0x80]));
    }

    #[test]
    fn single_byte_samples_pairwise_distinct() {
        let inputs: [[u8; 1]; SAMPLE_COUNT] = [[0x01], [0x02], [0x7f], [0x80], [0xff]];
        let mut results = [0u16; SAMPLE_COUNT];
        let mut i = 0usize;
        while i < SAMPLE_COUNT {
            results[i] = crc15_mpt1327(&inputs[i]);
            i += 1;
        }
        let mut a = 0usize;
        while a < SAMPLE_COUNT {
            let mut c = a + 1;
            while c < SAMPLE_COUNT {
                assert!(results[a] != results[c]);
                c += 1;
            }
            a += 1;
        }
    }

    #[test]
    fn result_within_mask_range() {
        let samples: [&[u8]; 4] = [EMPTY, BYTE_A, ABC, FOX];
        let mut i = 0usize;
        while i < samples.len() {
            let r = crc15_mpt1327(samples[i]);
            assert!((0u32..=MAX_RESULT).contains(&u32::from(r)));
            i += 1;
        }
    }

    #[test]
    fn leading_zeros_same() {
        assert!(crc15_mpt1327(&[0x00]) == crc15_mpt1327(&[0x00, 0x00]));
    }

    #[test]
    fn empty_equals_init_xorout() {
        assert!(crc15_mpt1327(EMPTY) == 0x0001);
    }
}
