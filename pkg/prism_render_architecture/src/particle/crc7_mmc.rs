//! `CRC-7`/`MMC`: the `7`-bit cyclic-redundancy check used by `SD`/`MMC`
//! command and response framing (design § integrity checks).
//!
//! A `CRC-7`/`MMC` treats the input bytes as the coefficients of a polynomial
//! over the binary field `GF(2)` and returns the `7`-bit remainder after
//! dividing by the fixed generator polynomial `x^7 + x^3 + 1`, written `0x09`
//! in unreflected form. The arithmetic is carry-less: "addition" is
//! exclusive-or and there are no carries, so the whole computation reduces to
//! shifts, masks, and exclusive-ors. The register is masked to `7` bits
//! (`0x7F`) throughout.
//!
//! Parameters (the standard parametric `CRC` model): `width` `7`, `poly`
//! `0x09`, `init` `0x00`, no input reflection (`refin` `false`), no output
//! reflection (`refout` `false`), and final exclusive-or `0x00`. The canonical
//! *check* value for the `ASCII` string `123456789` is `0x75`; this is
//! asserted directly in the tests.
//!
//! This module exposes a one-shot [`crc7`] function, an incremental [`Crc7`]
//! register for folding data in chunks, and an internal `256`-entry
//! most-significant-bit-first lookup table that the tests cross-check against
//! the bit-by-bit reference. A host `CPU`, `GPU` upload path, or asset-`hex`
//! tool can share any of these; they all compute the identical `7`-bit value.
//!
//! Scope and boundaries. This is a `7`-bit `CRC`: a `GF(2)` polynomial-modulo
//! error-detection code over a `7`-bit register, specific to `SD`/`MMC`
//! framing. It is deliberately independent of the neighbouring integrity
//! primitives in this crate and does not reference them:
//!
//! - `crc8_variants`, `crc16_ccitt`, `crc24_openpgp`, `crc32`, and
//!   `crc64_ecma` are wider `CRC`s with different generator polynomials and
//!   register widths. They share the same carry-less `GF(2)` algebra but are
//!   separate codes; a `7`-bit remainder is weaker and only suits the very
//!   short command words `SD`/`MMC` framing uses.
//!
//! None of these are cryptographic. A `CRC-7` is trivially invertible and
//! collisions are easy to craft on purpose, so it must only guard against
//! accidental corruption, never against a malicious adversary.

/// Generator polynomial `0x09` (`x^7 + x^3 + 1`), most-significant-bit-first
/// (unreflected) form.
const POLY: u8 = 0x09;

/// Low-`7`-bit mask applied to the register after every shift.
const MASK: u8 = 0x7F;

/// One-shot `CRC-7`/`MMC` over `data`, returning the low `7` bits.
///
/// Processes each byte most-significant-bit-first through the bit-by-bit
/// polynomial-division reference. The register starts at `0x00`, is masked to
/// `7` bits (`0x7F`) after each shift, and there is no final exclusive-or.
pub fn crc7(data: &[u8]) -> u8 {
    let mut crc: u8 = 0;
    for &byte in data {
        for i in (0..8).rev() {
            let bit = (byte >> i) & 1;
            let top = (crc >> 6) & 1;
            crc = (crc << 1) & MASK;
            if (top ^ bit) == 1 {
                crc ^= POLY;
            }
        }
    }
    crc & MASK
}

/// Incremental `CRC-7`/`MMC` register.
///
/// Folds data in arbitrary chunks and yields the same `7`-bit value as the
/// one-shot [`crc7`] over the concatenation of those chunks. Construct with
/// [`Crc7::new`], feed bytes with [`Crc7::update`], and read the result with
/// [`Crc7::finalize`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Crc7 {
    state: u8,
}

impl Crc7 {
    /// Creates a fresh register initialised to `0x00`.
    pub fn new() -> Self {
        Self { state: 0 }
    }

    /// Folds `data` into the register, most-significant-bit-first.
    pub fn update(&mut self, data: &[u8]) {
        let mut crc = self.state;
        for &byte in data {
            for i in (0..8).rev() {
                let bit = (byte >> i) & 1;
                let top = (crc >> 6) & 1;
                crc = (crc << 1) & MASK;
                if (top ^ bit) == 1 {
                    crc ^= POLY;
                }
            }
        }
        self.state = crc;
    }

    /// Returns the current `7`-bit register value.
    pub fn finalize(&self) -> u8 {
        self.state & MASK
    }
}

/// Bit-`7`-aligned form of the generator used by the byte-at-a-time table.
///
/// The generator `x^7 + x^3 + 1` has an implicit `x^7` term; writing it with
/// that term explicit gives `0b1000_1001` = `0x89`. The table driver keeps the
/// `7`-bit register in the low bits of an `8`-bit value and folds against this
/// aligned polynomial so the usual most-significant-bit-first table trick
/// applies to a sub-byte-width `CRC`.
#[cfg(test)]
const POLY_ALIGNED: u8 = 0x89;

/// Builds the `256`-entry byte-at-a-time lookup table.
///
/// Entry `i` holds the register value produced by folding the byte `i` into an
/// otherwise-zero register, processing bits most-significant-bit-first against
/// the bit-`7`-aligned generator `0x89`.
#[cfg(test)]
const fn build_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut v: u8 = if (i & 0x80) != 0 {
            (i as u8) ^ POLY_ALIGNED
        } else {
            i as u8
        };
        let mut j = 1;
        while j < 8 {
            v <<= 1;
            if (v & 0x80) != 0 {
                v ^= POLY_ALIGNED;
            }
            j += 1;
        }
        table[i] = v;
        i += 1;
    }
    table
}

/// Table-driven `CRC-7`/`MMC` used only by the tests to cross-check the
/// bit-by-bit reference.
///
/// Because the `7`-bit register sits in the low bits of a byte, each step
/// indexes the table by `(crc << 1) ^ byte` so the register's top bits line up
/// with the aligned generator. The final value is masked to `7` bits.
#[cfg(test)]
fn crc7_table(data: &[u8]) -> u8 {
    const TABLE: [u8; 256] = build_table();
    let mut crc: u8 = 0;
    for &byte in data {
        crc = TABLE[((crc << 1) ^ byte) as usize];
    }
    crc & MASK
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Hard reference vectors -------------------------------------------

    #[test]
    fn check_value_123456789() {
        assert_eq!(crc7(b"123456789"), 0x75);
    }

    #[test]
    fn empty_input_is_zero() {
        assert_eq!(crc7(b""), 0x00);
    }

    #[test]
    fn check_value_fits_in_seven_bits() {
        assert_eq!(crc7(b"123456789") & 0x80, 0x00);
    }

    // --- Table vs bit-by-bit agreement ------------------------------------

    #[test]
    fn table_matches_bitwise_check_vector() {
        assert_eq!(crc7_table(b"123456789"), 0x75);
    }

    #[test]
    fn table_matches_bitwise_empty() {
        assert_eq!(crc7_table(b""), crc7(b""));
    }

    #[test]
    fn table_matches_bitwise_short_ascii() {
        assert_eq!(crc7_table(b"A"), crc7(b"A"));
    }

    #[test]
    fn table_matches_bitwise_hello() {
        assert_eq!(crc7_table(b"hello"), crc7(b"hello"));
    }

    #[test]
    fn table_matches_bitwise_mixed() {
        let data = [0x00u8, 0x7F, 0x80, 0xFF, 0x09, 0x55, 0xAA, 0x01];
        assert_eq!(crc7_table(&data), crc7(&data));
    }

    #[test]
    fn table_matches_bitwise_all_single_bytes() {
        for b in 0u16..=255 {
            let byte = [b as u8];
            assert_eq!(crc7_table(&byte), crc7(&byte), "byte {b:#04x}");
        }
    }

    #[test]
    fn table_matches_bitwise_all_byte_pairs_sample() {
        for a in (0u16..=255).step_by(17) {
            for b in (0u16..=255).step_by(13) {
                let data = [a as u8, b as u8];
                assert_eq!(crc7_table(&data), crc7(&data), "pair {a:#04x},{b:#04x}");
            }
        }
    }

    #[test]
    fn table_matches_bitwise_counting_sequence() {
        let data: [u8; 64] = core::array::from_fn(|i| i as u8);
        assert_eq!(crc7_table(&data), crc7(&data));
    }

    // --- Incremental vs one-shot agreement --------------------------------

    #[test]
    fn incremental_empty_equals_oneshot() {
        let crc = Crc7::new();
        assert_eq!(crc.finalize(), crc7(b""));
    }

    #[test]
    fn incremental_single_update_equals_oneshot() {
        let mut crc = Crc7::new();
        crc.update(b"123456789");
        assert_eq!(crc.finalize(), crc7(b"123456789"));
    }

    #[test]
    fn incremental_two_chunks_equals_oneshot() {
        let mut crc = Crc7::new();
        crc.update(b"12345");
        crc.update(b"6789");
        assert_eq!(crc.finalize(), crc7(b"123456789"));
    }

    #[test]
    fn incremental_byte_at_a_time_equals_oneshot() {
        let mut crc = Crc7::new();
        for &b in b"123456789" {
            crc.update(&[b]);
        }
        assert_eq!(crc.finalize(), 0x75);
    }

    #[test]
    fn incremental_empty_chunks_do_not_change_state() {
        let mut crc = Crc7::new();
        crc.update(b"");
        crc.update(b"abc");
        crc.update(b"");
        crc.update(b"def");
        assert_eq!(crc.finalize(), crc7(b"abcdef"));
    }

    #[test]
    fn incremental_many_small_chunks() {
        let data: [u8; 48] = core::array::from_fn(|i| (i * 7 + 3) as u8);
        let mut crc = Crc7::new();
        for chunk in data.chunks(5) {
            crc.update(chunk);
        }
        assert_eq!(crc.finalize(), crc7(&data));
    }

    #[test]
    fn incremental_arbitrary_split_points() {
        let data = b"The quick brown fox";
        for split in 0..=data.len() {
            let mut crc = Crc7::new();
            crc.update(&data[..split]);
            crc.update(&data[split..]);
            assert_eq!(crc.finalize(), crc7(data), "split at {split}");
        }
    }

    #[test]
    fn default_matches_new() {
        assert_eq!(Crc7::default(), Crc7::new());
    }

    // --- Single-byte inputs -----------------------------------------------

    #[test]
    fn single_zero_byte() {
        assert_eq!(crc7(&[0x00]), crc7_table(&[0x00]));
    }

    #[test]
    fn single_byte_0x01() {
        assert_eq!(crc7(&[0x01]), crc7_table(&[0x01]));
    }

    #[test]
    fn single_byte_0x80() {
        assert_eq!(crc7(&[0x80]), crc7_table(&[0x80]));
    }

    #[test]
    fn single_byte_0xff() {
        assert_eq!(crc7(&[0xFF]), crc7_table(&[0xFF]));
    }

    // --- Multi-byte inputs ------------------------------------------------

    #[test]
    fn two_bytes_zero() {
        assert_eq!(crc7(&[0x00, 0x00]), crc7_table(&[0x00, 0x00]));
    }

    #[test]
    fn three_ascii_bytes() {
        assert_eq!(crc7(b"abc"), crc7_table(b"abc"));
    }

    #[test]
    fn sd_command_frame_like_input() {
        // Resembles an `SD` command token (command byte plus argument).
        let data = [0x40u8, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(crc7(&data), crc7_table(&data));
    }

    // --- All-zero inputs --------------------------------------------------

    #[test]
    fn all_zero_one_byte_is_zero() {
        assert_eq!(crc7(&[0x00]), 0x00);
    }

    #[test]
    fn all_zero_many_bytes_is_zero() {
        assert_eq!(crc7(&[0x00; 32]), 0x00);
    }

    #[test]
    fn all_zero_incremental_is_zero() {
        let mut crc = Crc7::new();
        crc.update(&[0x00; 10]);
        crc.update(&[0x00; 10]);
        assert_eq!(crc.finalize(), 0x00);
    }

    // --- All-0xFF inputs --------------------------------------------------

    #[test]
    fn all_ones_one_byte() {
        assert_eq!(crc7(&[0xFF]), crc7_table(&[0xFF]));
    }

    #[test]
    fn all_ones_many_bytes() {
        let data = [0xFFu8; 32];
        assert_eq!(crc7(&data), crc7_table(&data));
    }

    #[test]
    fn all_ones_incremental_equals_oneshot() {
        let data = [0xFFu8; 24];
        let mut crc = Crc7::new();
        for chunk in data.chunks(7) {
            crc.update(chunk);
        }
        assert_eq!(crc.finalize(), crc7(&data));
    }

    // --- Long inputs ------------------------------------------------------

    #[test]
    fn long_counting_input_table_agrees() {
        let data: [u8; 256] = core::array::from_fn(|i| i as u8);
        assert_eq!(crc7(&data), crc7_table(&data));
    }

    #[test]
    fn long_input_incremental_agrees() {
        let data: [u8; 512] = core::array::from_fn(|i| (i * 31 + 7) as u8);
        let mut crc = Crc7::new();
        for chunk in data.chunks(37) {
            crc.update(chunk);
        }
        assert_eq!(crc.finalize(), crc7(&data));
    }

    #[test]
    fn long_repeated_pattern() {
        let mut data = [0u8; 300];
        for (i, slot) in data.iter_mut().enumerate() {
            *slot = (i % 3) as u8;
        }
        assert_eq!(crc7(&data), crc7_table(&data));
    }

    #[test]
    fn result_always_within_seven_bits() {
        let data: [u8; 128] = core::array::from_fn(|i| (i * 5 + 1) as u8);
        assert_eq!(crc7(&data) & 0x80, 0x00);
        for len in 0..data.len() {
            assert_eq!(crc7(&data[..len]) & 0x80, 0x00, "len {len}");
        }
    }

    #[test]
    fn appending_zero_byte_changes_result() {
        // Unlike a plain sum, folding a trailing zero still advances the
        // register, so the two values generally differ.
        let base = crc7(b"frame");
        let extended = crc7(b"frame\x00");
        assert_ne!(base, extended);
    }

    #[test]
    fn order_sensitive() {
        assert_ne!(crc7(b"ab"), crc7(b"ba"));
    }

    #[test]
    fn distinct_single_bytes_table_agree() {
        for b in [0x01u8, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80] {
            assert_eq!(crc7(&[b]), crc7_table(&[b]), "byte {b:#04x}");
        }
    }

    #[test]
    fn incremental_matches_table_for_long_input() {
        let data: [u8; 200] = core::array::from_fn(|i| (i ^ 0x5A) as u8);
        let mut crc = Crc7::new();
        crc.update(&data);
        assert_eq!(crc.finalize(), crc7_table(&data));
    }
}
