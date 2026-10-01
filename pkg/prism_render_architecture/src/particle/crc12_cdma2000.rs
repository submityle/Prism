//! `CRC-12`/`CDMA2000` cyclic-redundancy check: pure-integer, bit-by-bit
//! polynomial division over `GF(2)` for the `12`-bit `CDMA2000` check value
//! (design § integrity checks).
//!
//! A `CRC-12` treats the input bytes as the coefficients of a polynomial over
//! the binary field `GF(2)` and returns the `12`-bit remainder after dividing
//! by a fixed generator polynomial. The arithmetic is carry-less: "addition"
//! is exclusive-or and there are no carries, so the whole computation reduces
//! to shifts, masks, and exclusive-ors. The register lives in a `u16` but only
//! the low `12` bits are meaningful, so every intermediate and final value is
//! masked with `0xFFF`.
//!
//! Parameterisation (`Koopman`/`RevEng` catalogue, `CRC-12`/`CDMA2000`):
//!
//! - `width` `12`
//! - `poly` `0xF13` (`MSB`-first / unreflected form)
//! - `init` `0xFFF`
//! - `refin` `false`, `refout` `false`
//! - `xorout` `0x000`
//! - `check` `0xD4D` for the `ASCII` string `123456789`
//!
//! Because the input is not reflected, each input bit is folded in
//! most-significant-bit-first. For every bit the top register bit (bit `11`)
//! is compared against the incoming data bit: the register is shifted left by
//! one (masked back to `12` bits), and when the exclusive-or of the two bits is
//! `1` the generator `poly` is folded in with an exclusive-or. Because
//! `xorout` is `0x000`, finalisation is just the masked register value, so the
//! empty input returns `init` `0xFFF` unchanged.
//!
//! Scope and boundaries. This is a `12`-bit `CRC`: a `GF(2)` polynomial-modulo
//! error-detection code. It is deliberately distinct from the neighbouring
//! integrity primitives in this crate, which differ in bit width, generator,
//! and/or algebra:
//!
//! - `crc10_atm`, `crc16_ccitt`, `crc32` are `CRC`s of other widths (and
//!   generators) built on the same `GF(2)` polynomial-division algebra.
//! - `adler32` and `fletcher_checksum` are *not* `CRC`s: they are
//!   position-weighted sums computed with ordinary integer modular addition,
//!   not carry-less polynomial division. Different algebra, different error
//!   profile.
//!
//! This code is not cryptographic. A `CRC-12` is trivially invertible and
//! collisions are easy to craft on purpose, so it must only guard against
//! accidental corruption, never against a malicious adversary.

/// Generator polynomial `0xF13` for `CRC-12`/`CDMA2000`,
/// most-significant-bit-first (unreflected) form.
const POLY: u16 = 0xF13;

/// Mask selecting the low `12` bits of the register (`0xFFF`).
const MASK: u16 = 0xFFF;

/// The `init` register seed for `CRC-12`/`CDMA2000` (`0xFFF`).
const INIT: u16 = 0xFFF;

/// Bit position of the top (most-significant) bit of the `12`-bit register.
const TOP_SHIFT: u16 = 11;

/// Fold a single byte into `crc`, `MSB`-first, bit by bit.
///
/// For each of the eight bits (most-significant first) the incoming data bit is
/// compared against the top register bit; the register is shifted left and
/// masked back to `12` bits, and the generator `poly` is folded in whenever the
/// exclusive-or of the two bits is `1`.
#[inline]
fn fold_byte(mut crc: u16, byte: u8) -> u16 {
    for i in (0..8).rev() {
        let bit = u16::from((byte >> i) & 1);
        let msb = (crc >> TOP_SHIFT) & 1;
        crc = (crc << 1) & MASK;
        if (msb ^ bit) == 1 {
            crc ^= POLY;
        }
    }
    crc
}

/// Compute the `CRC-12`/`CDMA2000` of `data` in one shot.
///
/// `MSB`-first (unreflected) bit-by-bit reduction starting from `init` `0xFFF`
/// against generator `poly` `0xF13`. The result is always in the range
/// `0..=0xFFF`. The empty slice returns `init` `0xFFF`, and `b"123456789"`
/// yields the catalogue `check` value `0xD4D`.
#[must_use]
pub fn crc12_cdma2000(data: &[u8]) -> u16 {
    let mut crc = INIT;
    for &byte in data {
        crc = fold_byte(crc, byte);
    }
    crc & MASK
}

/// Incremental `CRC-12`/`CDMA2000` accumulator.
///
/// Feed bytes in any number of chunks via [`Crc12Cdma2000::update`]; the
/// finalised value from [`Crc12Cdma2000::finalize`] matches [`crc12_cdma2000`]
/// over the concatenation of all chunks. The register starts at `init` `0xFFF`;
/// because `xorout` is `0x000`, finalisation returns the masked register
/// directly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Crc12Cdma2000 {
    /// Running `12`-bit register (only the low `12` bits are meaningful).
    crc: u16,
}

impl Crc12Cdma2000 {
    /// Create a fresh accumulator initialised to `init` `0xFFF`.
    #[must_use]
    pub fn new() -> Self {
        Self { crc: INIT }
    }

    /// Fold `data` into the running register (`MSB`-first, unreflected).
    pub fn update(&mut self, data: &[u8]) {
        let mut crc = self.crc;
        for &byte in data {
            crc = fold_byte(crc, byte);
        }
        self.crc = crc & MASK;
    }

    /// Return the finalised `12`-bit check value (`xorout` is `0x000`, so this
    /// is just the masked register).
    #[must_use]
    pub fn finalize(&self) -> u16 {
        self.crc & MASK
    }
}

impl Default for Crc12Cdma2000 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Convenience: run the incremental accumulator over a single slice.
    #[cfg(test)]
    fn incremental_once(data: &[u8]) -> u16 {
        let mut c = Crc12Cdma2000::new();
        c.update(data);
        c.finalize()
    }

    #[test]
    fn check_value_123456789() {
        assert_eq!(crc12_cdma2000(b"123456789"), 0xD4D);
    }

    #[test]
    fn empty_input_is_init() {
        assert_eq!(crc12_cdma2000(b""), 0xFFF);
    }

    #[test]
    fn empty_input_equals_init_const() {
        assert_eq!(crc12_cdma2000(b""), INIT);
    }

    #[test]
    fn check_value_in_range() {
        assert!(crc12_cdma2000(b"123456789") <= MASK);
    }

    #[test]
    fn empty_in_range() {
        assert!(crc12_cdma2000(b"") <= MASK);
    }

    #[test]
    fn result_always_within_12_bits() {
        let samples: [&[u8]; 6] = [
            b"",
            b"a",
            b"123456789",
            b"the quick brown fox",
            b"\x00\x01\x02\x03",
            b"\xff\xff\xff\xff",
        ];
        for s in samples {
            assert!(crc12_cdma2000(s) <= MASK);
        }
    }

    #[test]
    fn result_in_inclusive_range() {
        let v = crc12_cdma2000(b"range");
        assert!((0..=MASK).contains(&v));
    }

    #[test]
    fn new_matches_default() {
        assert_eq!(Crc12Cdma2000::new(), Crc12Cdma2000::default());
    }

    #[test]
    fn fresh_accumulator_finalizes_to_init() {
        assert_eq!(Crc12Cdma2000::new().finalize(), INIT);
    }

    #[test]
    fn incremental_once_matches_oneshot() {
        let data = b"123456789";
        assert_eq!(incremental_once(data), crc12_cdma2000(data));
    }

    #[test]
    fn incremental_once_check_value() {
        assert_eq!(incremental_once(b"123456789"), 0xD4D);
    }

    #[test]
    fn incremental_two_chunks_equals_oneshot() {
        let data = b"123456789";
        let mut c = Crc12Cdma2000::new();
        c.update(&data[..4]);
        c.update(&data[4..]);
        assert_eq!(c.finalize(), crc12_cdma2000(data));
    }

    #[test]
    fn incremental_three_chunks_equals_oneshot() {
        let data = b"the quick brown fox";
        let mut c = Crc12Cdma2000::new();
        c.update(&data[..5]);
        c.update(&data[5..11]);
        c.update(&data[11..]);
        assert_eq!(c.finalize(), crc12_cdma2000(data));
    }

    #[test]
    fn incremental_byte_by_byte_equals_oneshot() {
        let data = b"byte-by-byte feed test";
        let mut c = Crc12Cdma2000::new();
        for &b in data {
            c.update(&[b]);
        }
        assert_eq!(c.finalize(), crc12_cdma2000(data));
    }

    #[test]
    fn incremental_empty_chunks_do_not_change() {
        let data = b"has empty chunks";
        let mut c = Crc12Cdma2000::new();
        c.update(b"");
        c.update(&data[..3]);
        c.update(b"");
        c.update(&data[3..]);
        c.update(b"");
        assert_eq!(c.finalize(), crc12_cdma2000(data));
    }

    #[test]
    fn incremental_all_splits_equal_oneshot() {
        let data = b"split everywhere";
        let expected = crc12_cdma2000(data);
        for i in 0..=data.len() {
            let mut c = Crc12Cdma2000::new();
            c.update(&data[..i]);
            c.update(&data[i..]);
            assert_eq!(c.finalize(), expected);
        }
    }

    #[test]
    fn finalize_is_idempotent() {
        let mut c = Crc12Cdma2000::new();
        c.update(b"idempotent");
        let a = c.finalize();
        let b = c.finalize();
        assert_eq!(a, b);
    }

    #[test]
    fn finalize_in_range() {
        let mut c = Crc12Cdma2000::new();
        c.update(b"range check");
        assert!(c.finalize() <= MASK);
    }

    #[test]
    fn clone_is_independent() {
        let mut a = Crc12Cdma2000::new();
        a.update(b"abc");
        let b = a;
        a.update(b"def");
        assert_eq!(b.finalize(), crc12_cdma2000(b"abc"));
        assert_eq!(a.finalize(), crc12_cdma2000(b"abcdef"));
    }

    #[test]
    fn deterministic_repeat() {
        let data = b"determinism";
        assert_eq!(crc12_cdma2000(data), crc12_cdma2000(data));
    }

    #[test]
    fn different_inputs_differ() {
        assert_ne!(crc12_cdma2000(b"abc"), crc12_cdma2000(b"abd"));
    }

    #[test]
    fn order_matters() {
        assert_ne!(crc12_cdma2000(b"ab"), crc12_cdma2000(b"ba"));
    }

    #[test]
    fn length_sensitivity() {
        assert_ne!(crc12_cdma2000(b"a"), crc12_cdma2000(b"aa"));
    }

    #[test]
    fn single_bit_flip_changes_crc() {
        let base = crc12_cdma2000(&[0b0000_0000]);
        let flip = crc12_cdma2000(&[0b0000_0001]);
        assert_ne!(base, flip);
    }

    #[test]
    fn high_bit_flip_changes_crc() {
        let base = crc12_cdma2000(&[0b0000_0000]);
        let flip = crc12_cdma2000(&[0b1000_0000]);
        assert_ne!(base, flip);
    }

    #[test]
    fn repeated_byte_block() {
        let data = [0xAAu8; 32];
        assert_eq!(crc12_cdma2000(&data), incremental_once(&data));
    }

    #[test]
    fn long_run_matches_incremental() {
        let data: Vec<u8> = (0u32..2048)
            .map(|i| (i.wrapping_mul(31) & 0xFF) as u8)
            .collect();
        let mut c = Crc12Cdma2000::new();
        for chunk in data.chunks(13) {
            c.update(chunk);
        }
        assert_eq!(c.finalize(), crc12_cdma2000(&data));
    }

    #[test]
    fn all_byte_values_in_range() {
        let data: Vec<u8> = (0u16..=255).map(|b| b as u8).collect();
        assert!(crc12_cdma2000(&data) <= MASK);
    }

    #[test]
    fn ascii_digits_check_again() {
        let data = b"123456789";
        assert_eq!(crc12_cdma2000(data), 0xD4D);
        assert_eq!(incremental_once(data), 0xD4D);
    }

    #[test]
    fn zero_block_various_lengths_in_range() {
        for len in 0..64 {
            let data = alloc::vec![0u8; len];
            assert!(crc12_cdma2000(&data) <= MASK);
        }
    }

    #[test]
    fn constants_are_consistent() {
        assert_eq!(POLY, 0xF13);
        assert_eq!(MASK, 0xFFF);
        assert_eq!(INIT, 0xFFF);
        assert_eq!(TOP_SHIFT, 11);
    }

    #[test]
    fn poly_fits_in_12_bits() {
        const { assert!(POLY <= MASK) };
    }

    #[test]
    fn init_fits_in_12_bits() {
        const { assert!(INIT <= MASK) };
    }

    #[test]
    fn mask_is_twelve_ones() {
        const { assert!(MASK == (1u16 << 12) - 1) };
    }

    #[test]
    fn interleaved_updates_match_concatenation() {
        let a = b"first-part-";
        let b = b"second-part";
        let mut c = Crc12Cdma2000::new();
        c.update(a);
        c.update(b);
        let mut combined = Vec::new();
        combined.extend_from_slice(a);
        combined.extend_from_slice(b);
        assert_eq!(c.finalize(), crc12_cdma2000(&combined));
    }

    #[test]
    fn empty_then_data_matches_data() {
        let mut c = Crc12Cdma2000::new();
        c.update(b"");
        c.update(b"payload");
        assert_eq!(c.finalize(), crc12_cdma2000(b"payload"));
    }

    #[test]
    fn data_then_empty_matches_data() {
        let mut c = Crc12Cdma2000::new();
        c.update(b"payload");
        c.update(b"");
        assert_eq!(c.finalize(), crc12_cdma2000(b"payload"));
    }

    #[test]
    fn many_small_chunks_sample() {
        let data: Vec<u8> = (0u32..300).map(|i| (i & 0xFF) as u8).collect();
        let expected = crc12_cdma2000(&data);
        let mut c = Crc12Cdma2000::new();
        for chunk in data.chunks(1) {
            c.update(chunk);
        }
        assert_eq!(c.finalize(), expected);
    }

    #[test]
    fn chunk_sizes_all_agree() {
        let data: Vec<u8> = (0u32..512)
            .map(|i| (i.wrapping_mul(97) & 0xFF) as u8)
            .collect();
        let expected = crc12_cdma2000(&data);
        for size in [1usize, 2, 3, 7, 16, 64, 256] {
            let mut c = Crc12Cdma2000::new();
            for chunk in data.chunks(size) {
                c.update(chunk);
            }
            assert_eq!(c.finalize(), expected);
        }
    }

    #[test]
    fn single_zero_byte_value() {
        let v = crc12_cdma2000(&[0x00]);
        assert!(v <= MASK);
        assert_eq!(v, incremental_once(&[0x00]));
    }

    #[test]
    fn all_ones_byte_value() {
        let v = crc12_cdma2000(&[0xFF]);
        assert!(v <= MASK);
        assert_eq!(v, incremental_once(&[0xFF]));
    }

    #[test]
    fn fold_byte_matches_oneshot_single() {
        for byte in 0u16..=255 {
            let b = byte as u8;
            let expected = crc12_cdma2000(&[b]);
            assert_eq!(fold_byte(INIT, b), expected);
        }
    }

    #[test]
    fn two_disjoint_strings_differ() {
        assert_ne!(crc12_cdma2000(b"hello"), crc12_cdma2000(b"world"));
    }

    #[test]
    fn prepended_zero_changes_value() {
        assert_ne!(crc12_cdma2000(b"x"), crc12_cdma2000(b"\x00x"));
    }

    #[test]
    fn large_zero_block_deterministic() {
        let data = alloc::vec![0u8; 1000];
        assert_eq!(crc12_cdma2000(&data), crc12_cdma2000(&data));
    }
}
