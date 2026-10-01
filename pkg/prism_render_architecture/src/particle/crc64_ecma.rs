//! `CRC-64` cyclic-redundancy checks in the two authoritative `64`-bit
//! parameterisations (`ECMA-182` and `XZ`): pure-integer polynomial division
//! over `GF(2)` for verifying large `GPU` resource blobs, archive members, and
//! long network payloads where a `32`-bit register would alias too readily
//! (design § integrity checks).
//!
//! A `CRC-64` treats the input bytes as the coefficients of a polynomial over
//! the binary field `GF(2)` and returns the `64`-bit remainder after dividing
//! by a fixed generator polynomial. The arithmetic is carry-less: "addition" is
//! exclusive-or and there are no carries, so the whole computation reduces to
//! shifts, masks, exclusive-ors, and bit reversals. This module implements two
//! widely used `64`-bit variants that share the generator polynomial
//! `0x42F0E1EBA9EA3693` but differ in their initial register, reflection, and
//! final exclusive-or:
//!
//! - `CRC-64`/`ECMA-182`: `poly` `0x42F0E1EBA9EA3693`, `init` `0x0`, no input or
//!   output reflection, final exclusive-or `0x0`. The canonical *check* value
//!   for the `ASCII` string `123456789` is `0x6C40DF5F0B497347`.
//! - `CRC-64`/`XZ` (also called `CRC-64`/`GO-ECMA`): `poly`
//!   `0x42F0E1EBA9EA3693`, `init` `0xFFFFFFFFFFFFFFFF`, reflected input and
//!   output, final exclusive-or `0xFFFFFFFFFFFFFFFF`. The canonical *check*
//!   value for `123456789` is `0x995DC9BBDF1939FA`. This is the checksum stored
//!   in `.xz` stream and block footers.
//!
//! Two implementations are kept and cross-checked against each other. The
//! low-level [`crc64`] driver is the authority: a generic bit-by-bit routine
//! taking arbitrary parameters (generator `poly` in unreflected,
//! most-significant-bit-first form, initial register `init`, input reflection
//! `refin`, output reflection `refout`, and final exclusive-or `xorout`). When
//! `refin` is set each input byte is bit-reversed with [`u8::reverse_bits`]
//! before being folded in; when `refout` is set the final register is reversed
//! with [`u64::reverse_bits`] before the exclusive-or. The public variants
//! [`crc64_ecma_182`] and [`crc64_xz`] instead run byte-at-a-time over `256`-entry
//! lookup tables (a most-significant-bit-first table for the unreflected
//! `ECMA-182` variant and a least-significant-bit-first table built from the
//! reflected polynomial for `XZ`), and the tests verify, length by length, that
//! the table-driven and bit-by-bit results agree.
//!
//! Scope and boundaries. This is a `64`-bit `CRC`: a `GF(2)` polynomial-modulo
//! error-detection code. It is deliberately distinct from the neighbouring
//! integrity primitives in this crate, which differ in both bit width and the
//! underlying algebra:
//!
//! - `crc8_variants`, `crc16_ccitt`, and `crc32` are the same carry-less `GF(2)`
//!   polynomial division but with narrower registers and different generators;
//!   a `CRC-64` simply carries a `64`-bit remainder with a `64`-bit generator.
//! - `fletcher_checksum` and `adler32` are *not* `CRC`s at all: they are
//!   position-weighted sums computed with ordinary integer modular addition
//!   (running sums reduced modulo a constant), not carry-less polynomial
//!   division. Different algebra, different error profile.
//!
//! None of these are cryptographic. A `CRC-64` is trivially invertible and
//! collisions are easy to craft on purpose, so it must only guard against
//! accidental corruption, never against a malicious adversary. For
//! content-addressing or security use a real hash instead.

/// The `CRC-64` generator polynomial `0x42F0E1EBA9EA3693`, in the unreflected,
/// most-significant-bit-first form shared by both variants.
const POLY: u64 = 0x42F0_E1EB_A9EA_3693;

/// The reflected form of [`POLY`], i.e. `0xC96C5795D7870F42`, used by the
/// least-significant-bit-first table that drives the `XZ` variant.
const POLY_REFLECTED: u64 = 0xC96C_5795_D787_0F42;

/// The all-ones register seed and final mask (`0xFFFFFFFFFFFFFFFF`) used by the
/// `XZ` variant.
const ALL_ONES: u64 = 0xFFFF_FFFF_FFFF_FFFF;

/// The most-significant bit of the `64`-bit register.
const TOP_BIT: u64 = 0x8000_0000_0000_0000;

/// Generic bit-by-bit `CRC-64` driver (the low-level primitive and authority).
///
/// Computes the `64`-bit remainder for arbitrary parameters: generator `poly`
/// in unreflected (most-significant-bit-first) form, initial register `init`,
/// input reflection `refin`, output reflection `refout`, and final
/// exclusive-or `xorout`. When `refin` is set each input byte is bit-reversed
/// before being folded in; when `refout` is set the final register is
/// bit-reversed before the exclusive-or. This matches the standard parametric
/// `CRC` model and is the authority the table-driven variants are checked
/// against.
#[must_use]
pub fn crc64(data: &[u8], poly: u64, init: u64, refin: bool, refout: bool, xorout: u64) -> u64 {
    let mut crc = init;
    for &byte in data {
        let folded = if refin { byte.reverse_bits() } else { byte };
        crc ^= u64::from(folded) << 56;
        for _ in 0..8 {
            if (crc & TOP_BIT) != 0 {
                crc = (crc << 1) ^ poly;
            } else {
                crc <<= 1;
            }
        }
    }
    if refout {
        crc = crc.reverse_bits();
    }
    crc ^ xorout
}

/// Builds the `256`-entry most-significant-bit-first lookup table for an
/// unreflected generator `poly`.
///
/// Entry `i` holds the register contribution of folding the byte `i` into an
/// otherwise-zero register, processing the byte in the high `8` bits.
fn build_table_msb(poly: u64) -> [u64; 256] {
    let mut table = [0u64; 256];
    for (i, slot) in table.iter_mut().enumerate() {
        let mut crc = (i as u64) << 56;
        for _ in 0..8 {
            if (crc & TOP_BIT) != 0 {
                crc = (crc << 1) ^ poly;
            } else {
                crc <<= 1;
            }
        }
        *slot = crc;
    }
    table
}

/// Builds the `256`-entry least-significant-bit-first lookup table for a
/// reflected generator `poly_reflected` (for example `0xC96C5795D7870F42`).
///
/// Entry `i` holds the register contribution of folding the byte `i` into an
/// otherwise-zero register, processing the byte in the low `8` bits.
fn build_table_lsb(poly_reflected: u64) -> [u64; 256] {
    let mut table = [0u64; 256];
    for (i, slot) in table.iter_mut().enumerate() {
        let mut crc = i as u64;
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ poly_reflected;
            } else {
                crc >>= 1;
            }
        }
        *slot = crc;
    }
    table
}

/// Table-driven most-significant-bit-first `CRC-64` (no reflection).
///
/// Used by the unreflected `ECMA-182` variant. The generator `poly` is given in
/// unreflected form.
fn crc64_table_msb(data: &[u8], poly: u64, init: u64, xorout: u64) -> u64 {
    let table = build_table_msb(poly);
    let mut crc = init;
    for &byte in data {
        let index = ((crc >> 56) ^ u64::from(byte)) & 0xFF;
        crc = (crc << 8) ^ table[index as usize];
    }
    crc ^ xorout
}

/// Table-driven least-significant-bit-first `CRC-64` (reflected input/output).
///
/// Used by the reflected `XZ` variant. The generator `poly_reflected` is given
/// in reflected form (for example `0xC96C5795D7870F42`); input and output
/// reflection are intrinsic to this least-significant-bit-first driver, so no
/// explicit bit-reversal is needed.
fn crc64_table_lsb(data: &[u8], poly_reflected: u64, init: u64, xorout: u64) -> u64 {
    let table = build_table_lsb(poly_reflected);
    let mut crc = init;
    for &byte in data {
        let index = (crc ^ u64::from(byte)) & 0xFF;
        crc = (crc >> 8) ^ table[index as usize];
    }
    crc ^ xorout
}

/// `CRC-64`/`ECMA-182`: `poly` `0x42F0E1EBA9EA3693`, `init` `0x0`, no
/// reflection, final exclusive-or `0x0`. The check value for `123456789` is
/// `0x6C40DF5F0B497347` and the empty slice yields `0x0`.
#[must_use]
pub fn crc64_ecma_182(data: &[u8]) -> u64 {
    crc64_table_msb(data, POLY, 0x0, 0x0)
}

/// `CRC-64`/`XZ` (also `CRC-64`/`GO-ECMA`): `poly` `0x42F0E1EBA9EA3693`, `init`
/// `0xFFFFFFFFFFFFFFFF`, reflected input and output, final exclusive-or
/// `0xFFFFFFFFFFFFFFFF`. The check value for `123456789` is
/// `0x995DC9BBDF1939FA` and the empty slice yields `0x0`.
#[must_use]
pub fn crc64_xz(data: &[u8]) -> u64 {
    crc64_table_lsb(data, POLY_REFLECTED, ALL_ONES, ALL_ONES)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// The canonical `CRC` check string.
    const CHECK: &[u8] = b"123456789";

    /// A small deterministic linear-congruential generator for test payloads.
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            // SplitMix64-style mixing: pure integer shifts, multiplies, and xors.
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn bytes(&mut self, len: usize) -> Vec<u8> {
            let mut out = Vec::with_capacity(len);
            while out.len() < len {
                let word = self.next_u64();
                for shift in 0..8 {
                    if out.len() == len {
                        break;
                    }
                    out.push(((word >> (shift * 8)) & 0xFF) as u8);
                }
            }
            out
        }
    }

    // --- Authoritative check values for `123456789` (table-driven) ---

    #[test]
    fn ecma_182_check_vector() {
        assert_eq!(crc64_ecma_182(CHECK), 0x6C40_DF5F_0B49_7347);
    }

    #[test]
    fn xz_check_vector() {
        assert_eq!(crc64_xz(CHECK), 0x995D_C9BB_DF19_39FA);
    }

    // --- Same check values via the generic bit-by-bit driver ---

    #[test]
    fn ecma_182_check_vector_bitwise() {
        assert_eq!(
            crc64(CHECK, POLY, 0x0, false, false, 0x0),
            0x6C40_DF5F_0B49_7347
        );
    }

    #[test]
    fn xz_check_vector_bitwise() {
        assert_eq!(
            crc64(CHECK, POLY, ALL_ONES, true, true, ALL_ONES),
            0x995D_C9BB_DF19_39FA
        );
    }

    // --- Empty input ---

    #[test]
    fn ecma_182_empty_is_zero() {
        assert_eq!(crc64_ecma_182(&[]), 0x0);
    }

    #[test]
    fn xz_empty_is_zero() {
        assert_eq!(crc64_xz(&[]), 0x0);
    }

    #[test]
    fn empty_matches_bitwise() {
        assert_eq!(
            crc64_ecma_182(&[]),
            crc64(&[], POLY, 0x0, false, false, 0x0)
        );
        assert_eq!(
            crc64_xz(&[]),
            crc64(&[], POLY, ALL_ONES, true, true, ALL_ONES)
        );
    }

    // --- Single byte: table-driven agrees with the bit-by-bit driver ---

    #[test]
    fn ecma_182_single_byte_matches_bitwise() {
        let data = [0x5Au8];
        assert_eq!(
            crc64_ecma_182(&data),
            crc64(&data, POLY, 0x0, false, false, 0x0)
        );
    }

    #[test]
    fn xz_single_byte_matches_bitwise() {
        let data = [0x5Au8];
        assert_eq!(
            crc64_xz(&data),
            crc64(&data, POLY, ALL_ONES, true, true, ALL_ONES)
        );
    }

    #[test]
    fn xz_single_zero_byte_vector() {
        assert_eq!(crc64_xz(&[0x00]), 0x1FAD_A173_6467_3F59);
    }

    // --- All-zero input: table-driven agrees with the bit-by-bit driver ---

    #[test]
    fn ecma_182_all_zero_matches_bitwise() {
        let data = [0u8; 32];
        assert_eq!(
            crc64_ecma_182(&data),
            crc64(&data, POLY, 0x0, false, false, 0x0)
        );
    }

    #[test]
    fn xz_all_zero_matches_bitwise() {
        let data = [0u8; 32];
        assert_eq!(
            crc64_xz(&data),
            crc64(&data, POLY, ALL_ONES, true, true, ALL_ONES)
        );
    }

    #[test]
    fn ecma_182_all_zero_is_zero() {
        // With `init` `0x0` and no final exclusive-or, an all-zero message keeps
        // the register at zero regardless of length.
        assert_eq!(crc64_ecma_182(&[0u8; 32]), 0x0);
    }

    #[test]
    fn xz_all_zero_vector() {
        assert_eq!(crc64_xz(&[0u8; 32]), 0xC95A_F861_7CD5_330C);
    }

    // --- All-`0xFF` input: table-driven agrees with the bit-by-bit driver ---

    #[test]
    fn ecma_182_all_ff_matches_bitwise() {
        let data = [0xFFu8; 32];
        assert_eq!(
            crc64_ecma_182(&data),
            crc64(&data, POLY, 0x0, false, false, 0x0)
        );
    }

    #[test]
    fn xz_all_ff_matches_bitwise() {
        let data = [0xFFu8; 32];
        assert_eq!(
            crc64_xz(&data),
            crc64(&data, POLY, ALL_ONES, true, true, ALL_ONES)
        );
    }

    #[test]
    fn ecma_182_all_ff_vector() {
        assert_eq!(crc64_ecma_182(&[0xFFu8; 32]), 0xC395_AE61_FF6C_E004);
    }

    #[test]
    fn xz_all_ff_vector() {
        assert_eq!(crc64_xz(&[0xFFu8; 32]), 0xE95D_CE9E_FAA0_9ACF);
    }

    // --- `abc`: both authoritative vectors and cross-implementation agreement ---

    #[test]
    fn ecma_182_abc_vector() {
        assert_eq!(crc64_ecma_182(b"abc"), 0x6650_1A34_9A0E_0855);
    }

    #[test]
    fn xz_abc_vector() {
        assert_eq!(crc64_xz(b"abc"), 0x2CD8_094A_1A27_7627);
    }

    #[test]
    fn ecma_182_abc_matches_bitwise() {
        assert_eq!(
            crc64_ecma_182(b"abc"),
            crc64(b"abc", POLY, 0x0, false, false, 0x0)
        );
    }

    #[test]
    fn xz_abc_matches_bitwise() {
        assert_eq!(
            crc64_xz(b"abc"),
            crc64(b"abc", POLY, ALL_ONES, true, true, ALL_ONES)
        );
    }

    // --- Long input: table-driven agrees with the bit-by-bit driver ---

    #[test]
    fn ecma_182_long_matches_bitwise() {
        let data = [0xABu8; 1024];
        assert_eq!(
            crc64_ecma_182(&data),
            crc64(&data, POLY, 0x0, false, false, 0x0)
        );
    }

    #[test]
    fn xz_long_matches_bitwise() {
        let data = [0xABu8; 1024];
        assert_eq!(
            crc64_xz(&data),
            crc64(&data, POLY, ALL_ONES, true, true, ALL_ONES)
        );
    }

    // --- Length-by-length cross-validation on pseudorandom payloads ---

    #[test]
    fn ecma_182_table_matches_bitwise_all_lengths() {
        let mut lcg = Lcg::new(0x1234_5678_9ABC_DEF0);
        for len in 0..=257usize {
            let data = lcg.bytes(len);
            assert_eq!(
                crc64_ecma_182(&data),
                crc64(&data, POLY, 0x0, false, false, 0x0)
            );
        }
    }

    #[test]
    fn xz_table_matches_bitwise_all_lengths() {
        let mut lcg = Lcg::new(0x0F0F_0F0F_1234_5678);
        for len in 0..=257usize {
            let data = lcg.bytes(len);
            assert_eq!(
                crc64_xz(&data),
                crc64(&data, POLY, ALL_ONES, true, true, ALL_ONES)
            );
        }
    }

    #[test]
    fn both_variants_match_bitwise_on_random_lengths() {
        let mut lcg = Lcg::new(0xDEAD_BEEF_CAFE_F00D);
        for len in [1usize, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233] {
            let data = lcg.bytes(len);
            assert_eq!(
                crc64_ecma_182(&data),
                crc64(&data, POLY, 0x0, false, false, 0x0)
            );
            assert_eq!(
                crc64_xz(&data),
                crc64(&data, POLY, ALL_ONES, true, true, ALL_ONES)
            );
        }
    }

    // --- Cross-variant distinctness ---

    #[test]
    fn variants_differ_on_check_string() {
        assert_ne!(crc64_ecma_182(CHECK), crc64_xz(CHECK));
    }

    #[test]
    fn variants_differ_on_empty() {
        // Both happen to be zero on the empty slice; a single byte must split
        // them apart.
        assert_ne!(crc64_ecma_182(&[0x00]), crc64_xz(&[0x00]));
    }

    #[test]
    fn variants_differ_on_random_payload() {
        let mut lcg = Lcg::new(0xA1B2_C3D4_E5F6_0718);
        let data = lcg.bytes(97);
        assert_ne!(crc64_ecma_182(&data), crc64_xz(&data));
    }

    // --- Single-bit-flip detection ---

    #[test]
    fn ecma_182_detects_single_bit_flip() {
        let mut lcg = Lcg::new(0x5555_AAAA_3333_CCCC);
        let mut data = lcg.bytes(64);
        let base = crc64_ecma_182(&data);
        data[13] ^= 0x08;
        assert_ne!(crc64_ecma_182(&data), base);
    }

    #[test]
    fn xz_detects_single_bit_flip() {
        let mut lcg = Lcg::new(0x7777_1111_8888_2222);
        let mut data = lcg.bytes(64);
        let base = crc64_xz(&data);
        data[40] ^= 0x01;
        assert_ne!(crc64_xz(&data), base);
    }

    #[test]
    fn ecma_182_detects_every_single_bit_flip_in_small_buffer() {
        let base_data = [0x12u8, 0x34, 0x56, 0x78];
        let base = crc64_ecma_182(&base_data);
        for byte_index in 0..base_data.len() {
            for bit in 0..8u32 {
                let mut flipped = base_data;
                flipped[byte_index] ^= 1u8 << bit;
                assert_ne!(crc64_ecma_182(&flipped), base);
            }
        }
    }

    // --- Reflected polynomial relationship ---

    #[test]
    fn reflected_poly_is_bit_reverse() {
        assert_eq!(POLY_REFLECTED, POLY.reverse_bits());
        assert_eq!(POLY, POLY_REFLECTED.reverse_bits());
    }

    // --- Table structure smoke tests ---

    #[test]
    fn msb_table_entry_zero_is_zero() {
        let table = build_table_msb(POLY);
        assert_eq!(table[0], 0x0);
    }

    #[test]
    fn lsb_table_entry_zero_is_zero() {
        let table = build_table_lsb(POLY_REFLECTED);
        assert_eq!(table[0], 0x0);
    }

    #[test]
    fn msb_and_lsb_tables_differ() {
        let msb = build_table_msb(POLY);
        let lsb = build_table_lsb(POLY_REFLECTED);
        assert_ne!(msb[1], lsb[1]);
    }

    #[test]
    fn msb_table_entries_are_distinct_sample() {
        let table = build_table_msb(POLY);
        assert_ne!(table[1], table[2]);
        assert_ne!(table[2], table[255]);
        assert_ne!(table[128], table[64]);
    }

    #[test]
    fn lsb_table_entries_are_distinct_sample() {
        let table = build_table_lsb(POLY_REFLECTED);
        assert_ne!(table[1], table[2]);
        assert_ne!(table[2], table[255]);
        assert_ne!(table[128], table[64]);
    }

    // --- General sensitivity properties ---

    #[test]
    fn ecma_182_order_sensitivity() {
        assert_ne!(crc64_ecma_182(b"ab"), crc64_ecma_182(b"ba"));
    }

    #[test]
    fn xz_order_sensitivity() {
        assert_ne!(crc64_xz(b"ab"), crc64_xz(b"ba"));
    }

    #[test]
    fn ecma_182_different_data_differs() {
        assert_ne!(crc64_ecma_182(b"abc"), crc64_ecma_182(b"abd"));
        assert_ne!(crc64_ecma_182(b"abc"), crc64_ecma_182(b"acb"));
    }

    #[test]
    fn xz_different_data_differs() {
        assert_ne!(crc64_xz(b"abc"), crc64_xz(b"abd"));
        assert_ne!(crc64_xz(b"abc"), crc64_xz(b"acb"));
    }

    #[test]
    fn ecma_182_appending_zero_changes_result() {
        assert_ne!(crc64_ecma_182(b"data"), crc64_ecma_182(b"data\0"));
    }

    #[test]
    fn xz_appending_zero_changes_result() {
        assert_ne!(crc64_xz(b"data"), crc64_xz(b"data\0"));
    }

    #[test]
    fn length_prefix_changes_result() {
        assert_ne!(crc64_ecma_182(b"12345678"), crc64_ecma_182(b"123456789"));
        assert_ne!(crc64_xz(b"12345678"), crc64_xz(b"123456789"));
    }

    #[test]
    fn repeated_calls_are_pure() {
        let data = b"idempotent payload";
        assert_eq!(crc64_ecma_182(data), crc64_ecma_182(data));
        assert_eq!(crc64_xz(data), crc64_xz(data));
    }

    #[test]
    fn generic_driver_handles_no_reflection_round_trip() {
        // With `refin` and `refout` both false and all-zero init/xorout, the
        // generic driver reproduces the `ECMA-182` surface on arbitrary data.
        let mut lcg = Lcg::new(0x00FF_00FF_00FF_00FF);
        let data = lcg.bytes(200);
        assert_eq!(
            crc64(&data, POLY, 0x0, false, false, 0x0),
            crc64_ecma_182(&data)
        );
    }
}
