//! `CRC-12/GSM` contract module: a `CPU`-verifiable, bit-wise cyclic
//! redundancy check (`CRC`) used by the Prism particle engine subsystem.
//!
//! The implementation follows the classic bit-by-bit register algorithm,
//! consuming each input byte most-significant-bit (`MSB`) first. The register
//! is shifted left and conditionally combined with the polynomial using an
//! exclusive-or (`XOR`) operation whenever the shifted-out top bit differs
//! from the incoming data bit. Because the engine runs `no_std`, this module
//! relies solely on fixed-size arrays, slices, and `while` loops, avoiding any
//! heap allocation.
//!
//! Algorithm parameters for `CRC-12/GSM`:
//! width = 12, polynomial = `0xD31`, initial value = `0x000`,
//! final `XOR` = `0xFFF`, with neither input nor output bit reflection.
//! The standard check value, computed over the `ASCII` string `123456789`,
//! is `0xB34`.

/// Register width of the `CRC` in bits.
const WIDTH: u32 = 12;
/// Generator polynomial (`CRC-12/GSM`), omitting the implicit top term.
const POLY: u32 = 0xd31;
/// Initial register value before any data is processed.
const INIT: u32 = 0x0;
/// Value combined with the final register via `XOR` to produce the result.
const XOROUT: u32 = 0xfff;
/// Whether each input byte is bit-reflected (least-significant-bit first).
const REFLECT_IN: bool = false;
/// Whether the final register is bit-reflected before the output `XOR`.
const REFLECT_OUT: bool = false;
/// Mask selecting the low `WIDTH` bits of the register.
const MASK: u32 = 0xfff;
/// Mask selecting the most-significant (`MSB`) bit of the register.
const TOPBIT: u32 = 1u32 << (WIDTH - 1);
/// Number of bits in a single input byte.
const BITS_PER_BYTE: u32 = 8;

/// Reflect the low `bits` of `value`, swapping `MSB` and `LSB` ordering.
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

/// Compute the `CRC-12/GSM` checksum of `data`.
///
/// Returns the 12-bit result in the low bits of a `u16`.
pub fn crc12_gsm(data: &[u8]) -> u16 {
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
    use super::*;

    /// Length of the repeated-`0xA5` stress buffer.
    const A5_LEN: usize = 1000;
    /// Fill byte for the repeated-`0xA5` stress buffer.
    const A5_FILL: u8 = 0xa5;
    /// Length of the all-byte-values buffer (`0x00..=0xFF`).
    const ALL256_LEN: usize = 256;

    #[test]
    fn vec_80() {
        assert!(crc12_gsm(&[0x80]) == 0x82f);
    }

    #[test]
    fn vec_12345678() {
        assert!(crc12_gsm(&[0x12, 0x34, 0x56, 0x78]) == 0xda1);
    }

    #[test]
    fn vec_empty() {
        assert!(crc12_gsm(b"") == 0xfff);
    }

    #[test]
    fn vec_z00() {
        assert!(crc12_gsm(&[0x00]) == 0xfff);
    }

    #[test]
    fn vec_ff() {
        assert!(crc12_gsm(&[0xff]) == 0xe70);
    }

    #[test]
    fn vec_a() {
        assert!(crc12_gsm(b"a") == 0x0d2);
    }

    #[test]
    fn vec_b() {
        assert!(crc12_gsm(b"b") == 0xab0);
    }

    #[test]
    fn vec_ab() {
        assert!(crc12_gsm(b"ab") == 0x5d5);
    }

    #[test]
    fn vec_abc() {
        assert!(crc12_gsm(b"abc") == 0xcf6);
    }

    #[test]
    fn vec_two_zeros() {
        assert!(crc12_gsm(&[0x00, 0x00]) == 0xfff);
    }

    #[test]
    fn vec_01() {
        assert!(crc12_gsm(&[0x01]) == 0x2ce);
    }

    #[test]
    fn vec_02() {
        assert!(crc12_gsm(&[0x02]) == 0x8ac);
    }

    #[test]
    fn vec_7f() {
        assert!(crc12_gsm(&[0x7f]) == 0x9a0);
    }

    #[test]
    fn vec_aa_55() {
        assert!(crc12_gsm(&[0xaa, 0x55]) == 0x580);
    }

    #[test]
    fn vec_55_aa() {
        assert!(crc12_gsm(&[0x55, 0xaa]) == 0xb88);
    }

    #[test]
    fn vec_deadbeef() {
        assert!(crc12_gsm(&[0xde, 0xad, 0xbe, 0xef]) == 0xd85);
    }

    #[test]
    fn vec_hello() {
        assert!(crc12_gsm(b"Hello") == 0xc2d);
    }

    #[test]
    fn vec_fox() {
        assert!(crc12_gsm(b"The quick brown fox") == 0xa0d);
    }

    #[test]
    fn vec_four_zeros() {
        assert!(crc12_gsm(&[0u8; 4]) == 0xfff);
    }

    #[test]
    fn vec_four_ff() {
        assert!(crc12_gsm(&[0xffu8; 4]) == 0x7ee);
    }

    #[test]
    fn vec_0_15() {
        let data: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        assert!(crc12_gsm(&data) == 0xdb4);
    }

    #[test]
    fn vec_check() {
        assert!(crc12_gsm(b"123456789") == 0xb34);
    }

    #[test]
    fn vec_a5_1000() {
        let mut buf = [0u8; A5_LEN];
        let mut i = 0usize;
        while i < A5_LEN {
            buf[i] = A5_FILL;
            i += 1;
        }
        assert!(crc12_gsm(&buf) == 0x0da);
    }

    #[test]
    fn vec_all256() {
        let mut buf = [0u8; ALL256_LEN];
        let mut i = 0usize;
        while i < ALL256_LEN {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc12_gsm(&buf) == 0xe0a);
    }

    #[test]
    fn determinism() {
        let first = crc12_gsm(b"The quick brown fox");
        let second = crc12_gsm(b"The quick brown fox");
        assert!(first == second);
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc12_gsm(b"a") != crc12_gsm(b"b"));
    }

    #[test]
    fn prefix_differs() {
        assert!(crc12_gsm(b"abc") != crc12_gsm(b"ab"));
    }

    #[test]
    fn length_sensitive() {
        assert!(crc12_gsm(b"a") != crc12_gsm(b"aa"));
    }

    #[test]
    fn single_byte_samples_are_pairwise_distinct() {
        const SAMPLES: [u8; 6] = [0x00, 0x01, 0x02, 0x7f, 0x80, 0xff];
        let mut results = [0u16; SAMPLES.len()];
        let mut i = 0usize;
        while i < SAMPLES.len() {
            results[i] = crc12_gsm(&[SAMPLES[i]]);
            i += 1;
        }
        let mut a = 0usize;
        while a < results.len() {
            let mut c = a + 1;
            while c < results.len() {
                assert!(results[a] != results[c]);
                c += 1;
            }
            a += 1;
        }
    }

    #[test]
    fn check_constant_matches_standard() {
        const CHECK: u16 = 0xb34;
        assert!(crc12_gsm(b"123456789") == CHECK);
    }

    #[test]
    fn output_within_mask() {
        let inputs: [&[u8]; 5] = [b"", b"a", b"abc", b"Hello", b"123456789"];
        let mut i = 0usize;
        while i < inputs.len() {
            let result = crc12_gsm(inputs[i]);
            assert!(u32::from(result) <= MASK);
            i += 1;
        }
    }

    #[test]
    fn empty_equals_final_xor() {
        assert!(u32::from(crc12_gsm(b"")) == (XOROUT & MASK));
    }

    #[test]
    fn byte_order_matters() {
        assert!(crc12_gsm(&[0xaa, 0x55]) != crc12_gsm(&[0x55, 0xaa]));
    }

    #[test]
    fn reflect_identity_and_known_values() {
        assert!(reflect(0x001, 8) == 0x080);
        assert!(reflect(0x0ff, 8) == 0x0ff);
        assert!(reflect(0x001, WIDTH) == 0x800);
        assert!(reflect(0xfff, WIDTH) == 0xfff);
    }

    #[test]
    fn all_zero_inputs_are_length_insensitive() {
        let r0 = crc12_gsm(b"");
        let r1 = crc12_gsm(&[0x00]);
        let r2 = crc12_gsm(&[0x00, 0x00]);
        let r4 = crc12_gsm(&[0u8; 4]);
        assert!(r0 == r1);
        assert!(r1 == r2);
        assert!(r2 == r4);
    }
}
