//! Bit-level `CRC-4/ITU` (also known as `CRC-4/G-704`) checksum: a pure-integer,
//! bit-by-bit golden reference used as the `CPU` oracle for verifying particle
//! payload nibbles (design integrity checks).
//!
//! This module computes `CRC-4/ITU` one bit at a time and returns the 4-bit
//! result packed into the low nibble of a `u8`. The algorithm parameters are
//! `width` = 4, `poly` = `0x3`, `init` = `0x0`, `refin` = `true`,
//! `refout` = `true`, and `xorout` = `0x0`.
//!
//! The core loop uses an `MSB`-first shift register together with per-byte input
//! reflection (`refin`) and a single final output reflection (`refout`). This
//! "reflect the input, process `MSB`-first, reflect the output" form is easy to
//! validate against the published `CRC` catalogue: the canonical check value for
//! the ASCII string `b"123456789"` is `0x7`.
//!
//! Every operation is an integer shift, mask, or exclusive-or; there are no
//! floating-point or transcendental operations and no heap allocation. The core
//! API consumes `&[u8]` and returns a `u8` whose high nibble is always zero.

/// Register width in bits for `CRC-4/ITU`.
pub const WIDTH: u32 = 4;

/// Generator polynomial for `CRC-4/ITU`, kept in the low `WIDTH` bits.
pub const POLY: u8 = 0x3;

/// Initial raw register value before any bytes are processed.
pub const INIT: u8 = 0x0;

/// Final value exclusive-or'd into the register after `refout`.
pub const XOROUT: u8 = 0x0;

/// Whether each input byte is bit-reflected before being processed.
pub const REFIN: bool = true;

/// Whether the final register is bit-reflected before `xorout` is applied.
pub const REFOUT: bool = true;

/// Low-nibble mask covering the `WIDTH` significant register bits (`0xF`).
pub const MASK: u8 = (1u8 << WIDTH) - 1;

const _: () = {
    const { assert!(WIDTH == 4) };
    const { assert!(MASK == 0xF) };
    const { assert!((POLY & MASK) == POLY) };
    const { assert!((INIT & MASK) == INIT) };
    const { assert!((XOROUT & MASK) == XOROUT) };
};

/// Reflect all 8 bits of a `u8`, swapping bit `i` with bit `7 - i`.
pub fn reflect_u8(value: u8) -> u8 {
    let mut result: u8 = 0;
    let mut i: u32 = 0;
    while i < 8 {
        if ((value >> i) & 1) == 1 {
            result |= (1u8) << (7u32 - i);
        }
        i += 1;
    }
    result
}

/// Reflect the low `width` bits of a `u8`, swapping bit `i` with bit
/// `(width - 1) - i`. Only the low `width` bits of `value` are considered.
pub fn reflect_bits(value: u8, width: u32) -> u8 {
    let mut result: u8 = 0;
    let mut i: u32 = 0;
    while i < width {
        if ((value >> i) & 1) == 1 {
            result |= (1u8) << ((width - 1) - i);
        }
        i += 1;
    }
    result
}

/// Fold `data` into the running register, starting from `reg`.
///
/// The returned value is the raw shift-register state with neither `refout` nor
/// `xorout` applied, so it can be fed straight back in to continue a checksum
/// across chunk boundaries. Only the low `WIDTH` bits of `reg` are significant.
pub fn update(reg: u8, data: &[u8]) -> u8 {
    let mask: u8 = MASK;
    let mut reg: u8 = reg & mask;
    for &byte in data {
        let b: u8 = if REFIN { reflect_u8(byte) } else { byte };
        let mut i: u32 = 8;
        while i > 0 {
            i -= 1;
            let bit: u8 = (b >> i) & 1;
            let top: u8 = (reg >> (WIDTH - 1)) & 1;
            reg = (reg << 1) & mask;
            if ((top ^ bit) & 1) == 1 {
                reg ^= POLY;
            }
        }
    }
    reg
}

/// Apply the final `refout` reflection and `xorout` to a raw register value.
fn finalize(reg: u8) -> u8 {
    let mut reg: u8 = reg;
    if REFOUT {
        reg = reflect_bits(reg, WIDTH);
    }
    (reg ^ XOROUT) & MASK
}

/// Compute the `CRC-4/ITU` checksum of `data`, returned in the low nibble of a
/// `u8` (the high nibble is always zero).
pub fn checksum(data: &[u8]) -> u8 {
    finalize(update(INIT, data))
}

/// Compute the `CRC-4/ITU` checksum of `data` using a custom raw register
/// `init`. With `init` equal to [`INIT`] this matches [`checksum`]; other
/// values support resuming from a previously saved [`update`] register.
pub fn checksum_with_init(init: u8, data: &[u8]) -> u8 {
    finalize(update(init, data))
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Enshrined hard reference vectors -------------------------------

    #[test]
    fn check_vector_123456789() {
        assert_eq!(checksum(b"123456789"), 0x7);
    }

    #[test]
    fn empty_input_is_zero() {
        assert_eq!(checksum(b""), 0x0);
    }

    #[test]
    fn empty_slice_literal_is_zero() {
        let data: [u8; 0] = [];
        assert_eq!(checksum(&data), 0x0);
    }

    #[test]
    fn single_a() {
        assert_eq!(checksum(b"A"), 0x1);
    }

    #[test]
    fn abc_vector() {
        assert_eq!(checksum(b"abc"), 0xE);
    }

    #[test]
    fn single_zero_byte() {
        assert_eq!(checksum(&[0x00]), 0x0);
    }

    #[test]
    fn single_ff_byte() {
        assert_eq!(checksum(&[0xFF]), 0x2);
    }

    #[test]
    fn quick_brown_fox() {
        assert_eq!(checksum(b"The quick brown fox"), 0x6);
    }

    #[test]
    fn two_zero_bytes() {
        assert_eq!(checksum(&[0x00, 0x00]), 0x0);
    }

    #[test]
    fn four_ff_bytes() {
        assert_eq!(checksum(&[0xFF, 0xFF, 0xFF, 0xFF]), 0xA);
    }

    // --- Byte reflection correctness ------------------------------------

    #[test]
    fn reflect_u8_0x41() {
        assert_eq!(reflect_u8(0x41), 0x82);
    }

    #[test]
    fn reflect_u8_zero() {
        assert_eq!(reflect_u8(0x00), 0x00);
    }

    #[test]
    fn reflect_u8_ff() {
        assert_eq!(reflect_u8(0xFF), 0xFF);
    }

    #[test]
    fn reflect_u8_0x01() {
        assert_eq!(reflect_u8(0x01), 0x80);
    }

    #[test]
    fn reflect_u8_0x80() {
        assert_eq!(reflect_u8(0x80), 0x01);
    }

    #[test]
    fn reflect_u8_0x0f() {
        assert_eq!(reflect_u8(0x0F), 0xF0);
    }

    #[test]
    fn reflect_u8_involution() {
        let samples: [u8; 7] = [0x00, 0x01, 0x41, 0x80, 0xFF, 0xA5, 0x3C];
        let mut i: usize = 0;
        while i < samples.len() {
            let v = samples[i];
            assert_eq!(reflect_u8(reflect_u8(v)), v);
            i += 1;
        }
    }

    // --- Width-limited reflection correctness ---------------------------

    #[test]
    fn reflect_bits_width4_0x8() {
        assert_eq!(reflect_bits(0x8, WIDTH), 0x1);
    }

    #[test]
    fn reflect_bits_width4_0x1() {
        assert_eq!(reflect_bits(0x1, WIDTH), 0x8);
    }

    #[test]
    fn reflect_bits_width4_0x2() {
        assert_eq!(reflect_bits(0x2, WIDTH), 0x4);
    }

    #[test]
    fn reflect_bits_width4_0x4() {
        assert_eq!(reflect_bits(0x4, WIDTH), 0x2);
    }

    #[test]
    fn reflect_bits_width4_0xf() {
        assert_eq!(reflect_bits(0xF, WIDTH), 0xF);
    }

    #[test]
    fn reflect_bits_width4_0x0() {
        assert_eq!(reflect_bits(0x0, WIDTH), 0x0);
    }

    #[test]
    fn reflect_bits_involution() {
        let samples: [u8; 6] = [0x0, 0x1, 0x8, 0xF, 0xA, 0x6];
        let mut i: usize = 0;
        while i < samples.len() {
            let v = samples[i];
            assert_eq!(reflect_bits(reflect_bits(v, WIDTH), WIDTH), v);
            i += 1;
        }
    }

    // --- Chunked (streaming) consistency --------------------------------

    #[test]
    fn chunk_split_123456789() {
        let whole = update(INIT, b"123456789");
        let part = update(INIT, b"12345");
        let combined = update(part, b"6789");
        assert_eq!(whole, combined);
        assert_eq!(finalize(whole), 0x7);
    }

    #[test]
    fn chunk_split_fox() {
        let whole = update(INIT, b"The quick brown fox");
        let part = update(INIT, b"The quick ");
        let combined = update(part, b"brown fox");
        assert_eq!(whole, combined);
    }

    #[test]
    fn chunk_matches_checksum_fox() {
        let part = update(INIT, b"The quick ");
        let combined = update(part, b"brown fox");
        assert_eq!(finalize(combined), checksum(b"The quick brown fox"));
    }

    #[test]
    fn chunk_three_parts() {
        let whole = update(INIT, b"abcdef");
        let p1 = update(INIT, b"ab");
        let p2 = update(p1, b"cd");
        let p3 = update(p2, b"ef");
        assert_eq!(whole, p3);
    }

    #[test]
    fn chunk_byte_at_a_time_matches_whole() {
        let data: [u8; 5] = *b"hello";
        let whole = update(INIT, &data);
        let mut reg = INIT;
        let mut i: usize = 0;
        while i < data.len() {
            reg = update(reg, &[data[i]]);
            i += 1;
        }
        assert_eq!(whole, reg);
    }

    #[test]
    fn chunk_empty_prefix_and_suffix() {
        let whole = update(INIT, b"hello");
        let a = update(INIT, b"");
        let b = update(a, b"hello");
        let c = update(b, b"");
        assert_eq!(whole, c);
    }

    #[test]
    fn update_empty_is_init() {
        assert_eq!(update(INIT, b""), INIT & MASK);
        assert_eq!(update(0x5, b""), 0x5 & MASK);
    }

    #[test]
    fn checksum_with_init_matches_default() {
        assert_eq!(
            checksum_with_init(INIT, b"123456789"),
            checksum(b"123456789")
        );
        assert_eq!(checksum_with_init(INIT, b"abc"), 0xE);
    }

    // --- Mask / nibble boundary -----------------------------------------

    #[test]
    fn result_always_within_mask() {
        let inputs: [&[u8]; 8] = [
            b"",
            b"A",
            b"abc",
            b"123456789",
            b"The quick brown fox",
            &[0x00],
            &[0xFF],
            &[0xFF, 0xFF, 0xFF, 0xFF],
        ];
        let mut i: usize = 0;
        while i < inputs.len() {
            let c = checksum(inputs[i]);
            assert!(c <= 0xF);
            assert_eq!(c & MASK, c);
            i += 1;
        }
    }

    #[test]
    fn all_single_bytes_within_mask() {
        let mut byte: u16 = 0;
        while byte < 256 {
            let c = checksum(&[byte as u8]);
            assert!(c <= MASK);
            byte += 1;
        }
    }

    #[test]
    fn update_result_within_mask() {
        let r = update(INIT, b"some bytes here");
        assert!(r <= MASK);
    }

    #[test]
    fn long_input_within_mask() {
        let data: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let c = checksum(&data);
        assert!(c <= MASK);
    }

    // --- Equivalence between literal forms ------------------------------

    #[test]
    fn checksum_a_equals_bytes_literal() {
        assert_eq!(checksum(b"A"), checksum(&[0x41]));
    }

    #[test]
    fn abc_bytes_equivalence() {
        assert_eq!(checksum(b"abc"), checksum(&[0x61, 0x62, 0x63]));
    }

    #[test]
    fn finalize_of_init_is_zero() {
        assert_eq!(finalize(update(INIT, b"")), 0x0);
    }

    // --- Parameter constants --------------------------------------------

    #[test]
    fn mask_constant() {
        assert_eq!(MASK, 0xF);
    }

    #[test]
    fn width_constant() {
        assert_eq!(WIDTH, 4);
    }

    #[test]
    fn poly_constant() {
        assert_eq!(POLY, 0x3);
    }

    #[test]
    fn init_constant() {
        assert_eq!(INIT, 0x0);
    }

    #[test]
    fn xorout_constant() {
        assert_eq!(XOROUT, 0x0);
    }

    #[test]
    fn refin_refout_flags() {
        assert_eq!((REFIN, REFOUT), (true, true));
    }
}
