//! `CRC-10`/`ATM` cyclic-redundancy check: pure-integer, bit-by-bit polynomial
//! division over `GF(2)` for the `10`-bit `ATM` `AAL` header/`OAM` cell check
//! (design § integrity checks).
//!
//! A `CRC-10` treats the input bytes as the coefficients of a polynomial over
//! the binary field `GF(2)` and returns the `10`-bit remainder after dividing
//! by a fixed generator polynomial. The arithmetic is carry-less: "addition" is
//! exclusive-or and there are no carries, so the entire computation reduces to
//! shifts, masks, and exclusive-ors. The register is held in a `u16` but only
//! the low `10` bits are meaningful, so every intermediate and final value is
//! masked with `0x3FF`.
//!
//! Parameterisation (`Koopman`/`RevEng` catalogue, `CRC-10`/`ATM`):
//!
//! - `width` `10`
//! - `poly` `0x233` (`MSB`-first / unreflected form)
//! - `init` `0x000`
//! - `refin` `false`, `refout` `false`
//! - `xorout` `0x000`
//! - `check` `0x199` for the `ASCII` string `123456789`
//!
//! Because the input is not reflected, each byte is aligned to the top of the
//! `10`-bit register by shifting it left by `2` before the eight per-bit
//! reduction steps. Because `xorout` is `0x000`, finalisation is just the
//! masked register value.
//!
//! Scope and boundaries. This is a `10`-bit `CRC`: a `GF(2)` polynomial-modulo
//! error-detection code. It is deliberately distinct from the neighbouring
//! integrity primitives in this crate, which differ in bit width, generator,
//! and/or algebra:
//!
//! - `crc7_mmc`, `crc16_ccitt`, `crc32` are `CRC`s of other widths (and
//!   generators) built on the same `GF(2)` polynomial-division algebra.
//! - `adler32` and `fletcher_checksum` are *not* `CRC`s: they are
//!   position-weighted sums computed with ordinary integer modular addition,
//!   not carry-less polynomial division. Different algebra, different error
//!   profile.
//!
//! This code is not cryptographic. A `CRC-10` is trivially invertible and
//! collisions are easy to craft on purpose, so it must only guard against
//! accidental corruption, never against a malicious adversary.

/// Generator polynomial `0x233` for `CRC-10`/`ATM`, most-significant-bit-first
/// (unreflected) form.
const POLY: u16 = 0x233;

/// Mask selecting the low `10` bits of the register (`0x3FF`).
const MASK: u16 = 0x3FF;

/// Top bit of the `10`-bit register (`0x200`), tested before each shift.
const TOP_BIT: u16 = 0x200;

/// Compute the `CRC-10`/`ATM` of `data` in one shot.
///
/// `MSB`-first (unreflected) bit-by-bit reduction: each byte is aligned to the
/// top of the `10`-bit register with `(b as u16) << 2`, then folded in eight
/// bits at a time against generator `poly` `0x233`. The result is always in the
/// range `0..=0x3FF`.
pub fn crc10_atm(data: &[u8]) -> u16 {
    let mut crc: u16 = 0x000;
    for &b in data {
        crc ^= (b as u16) << 2;
        for _ in 0..8 {
            if (crc & TOP_BIT) != 0 {
                crc = ((crc << 1) ^ POLY) & MASK;
            } else {
                crc = (crc << 1) & MASK;
            }
        }
    }
    crc & MASK
}

/// Incremental `CRC-10`/`ATM` accumulator.
///
/// Feed bytes in any number of chunks via [`Crc10Atm::update`]; the finalised
/// value from [`Crc10Atm::finalize`] matches [`crc10_atm`] over the
/// concatenation of all chunks. The register starts at `init` `0x000`; because
/// `xorout` is `0x000`, finalisation returns the masked register directly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Crc10Atm {
    /// Running `10`-bit register (only the low `10` bits are meaningful).
    crc: u16,
}

impl Crc10Atm {
    /// Create a fresh accumulator initialised to `init` `0x000`.
    pub fn new() -> Self {
        Self { crc: 0x000 }
    }

    /// Fold `data` into the running register (`MSB`-first, unreflected).
    pub fn update(&mut self, data: &[u8]) {
        let mut crc = self.crc;
        for &b in data {
            crc ^= (b as u16) << 2;
            for _ in 0..8 {
                if (crc & TOP_BIT) != 0 {
                    crc = ((crc << 1) ^ POLY) & MASK;
                } else {
                    crc = (crc << 1) & MASK;
                }
            }
        }
        self.crc = crc & MASK;
    }

    /// Return the finalised `10`-bit check value (`xorout` is `0x000`, so this
    /// is just the masked register).
    pub fn finalize(&self) -> u16 {
        self.crc & MASK
    }
}

impl Default for Crc10Atm {
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
        let mut c = Crc10Atm::new();
        c.update(data);
        c.finalize()
    }

    #[test]
    fn check_value_123456789() {
        assert_eq!(crc10_atm(b"123456789"), 0x199);
    }

    #[test]
    fn empty_input_is_zero() {
        assert_eq!(crc10_atm(b""), 0x000);
    }

    #[test]
    fn empty_input_incremental_is_zero() {
        assert_eq!(incremental_once(b""), 0x000);
    }

    #[test]
    fn new_finalize_before_update_is_zero() {
        let c = Crc10Atm::new();
        assert_eq!(c.finalize(), 0x000);
    }

    #[test]
    fn default_matches_new() {
        assert_eq!(Crc10Atm::default(), Crc10Atm::new());
    }

    #[test]
    fn default_finalize_is_zero() {
        assert_eq!(Crc10Atm::default().finalize(), 0x000);
    }

    #[test]
    fn check_value_incremental() {
        assert_eq!(incremental_once(b"123456789"), 0x199);
    }

    #[test]
    fn result_in_range_check() {
        assert!(crc10_atm(b"123456789") <= MASK);
    }

    #[test]
    fn result_in_range_empty() {
        assert!(crc10_atm(b"") <= MASK);
    }

    #[test]
    fn result_in_range_single_bytes() {
        for b in 0u16..=255 {
            let v = crc10_atm(&[b as u8]);
            assert!(v <= MASK);
        }
    }

    #[test]
    fn result_in_range_two_bytes_sample() {
        for a in (0u16..=255).step_by(7) {
            for b in (0u16..=255).step_by(5) {
                let v = crc10_atm(&[a as u8, b as u8]);
                assert!(v <= MASK);
            }
        }
    }

    #[test]
    fn result_in_range_long_input() {
        let data: Vec<u8> = (0u32..1000).map(|i| (i & 0xFF) as u8).collect();
        assert!(crc10_atm(&data) <= MASK);
    }

    #[test]
    fn deterministic_check() {
        assert_eq!(crc10_atm(b"123456789"), crc10_atm(b"123456789"));
    }

    #[test]
    fn deterministic_repeated_many() {
        let first = crc10_atm(b"deterministic payload");
        for _ in 0..64 {
            assert_eq!(crc10_atm(b"deterministic payload"), first);
        }
    }

    #[test]
    fn deterministic_empty() {
        assert_eq!(crc10_atm(b""), crc10_atm(b""));
    }

    #[test]
    fn single_byte_zero() {
        let v = crc10_atm(&[0x00]);
        assert!(v <= MASK);
    }

    #[test]
    fn single_byte_zero_matches_incremental() {
        assert_eq!(crc10_atm(&[0x00]), incremental_once(&[0x00]));
    }

    #[test]
    fn single_byte_ff() {
        assert_eq!(crc10_atm(&[0xFF]), incremental_once(&[0xFF]));
    }

    #[test]
    fn incremental_two_chunks_equals_oneshot() {
        let data = b"123456789";
        let mut c = Crc10Atm::new();
        c.update(&data[..4]);
        c.update(&data[4..]);
        assert_eq!(c.finalize(), crc10_atm(data));
    }

    #[test]
    fn incremental_three_chunks_equals_oneshot() {
        let data = b"the quick brown fox";
        let mut c = Crc10Atm::new();
        c.update(&data[..5]);
        c.update(&data[5..11]);
        c.update(&data[11..]);
        assert_eq!(c.finalize(), crc10_atm(data));
    }

    #[test]
    fn incremental_byte_by_byte_equals_oneshot() {
        let data = b"byte-by-byte feed test";
        let mut c = Crc10Atm::new();
        for &b in data {
            c.update(&[b]);
        }
        assert_eq!(c.finalize(), crc10_atm(data));
    }

    #[test]
    fn incremental_empty_chunks_do_not_change() {
        let data = b"has empty chunks";
        let mut c = Crc10Atm::new();
        c.update(b"");
        c.update(&data[..3]);
        c.update(b"");
        c.update(&data[3..]);
        c.update(b"");
        assert_eq!(c.finalize(), crc10_atm(data));
    }

    #[test]
    fn incremental_all_splits_equal_oneshot() {
        let data = b"split everywhere";
        let expected = crc10_atm(data);
        for i in 0..=data.len() {
            let mut c = Crc10Atm::new();
            c.update(&data[..i]);
            c.update(&data[i..]);
            assert_eq!(c.finalize(), expected);
        }
    }

    #[test]
    fn finalize_is_idempotent() {
        let mut c = Crc10Atm::new();
        c.update(b"idempotent");
        let a = c.finalize();
        let b = c.finalize();
        assert_eq!(a, b);
    }

    #[test]
    fn finalize_in_range() {
        let mut c = Crc10Atm::new();
        c.update(b"range check");
        assert!(c.finalize() <= MASK);
    }

    #[test]
    fn clone_is_independent() {
        let mut a = Crc10Atm::new();
        a.update(b"abc");
        let b = a;
        a.update(b"def");
        assert_eq!(b.finalize(), crc10_atm(b"abc"));
        assert_eq!(a.finalize(), crc10_atm(b"abcdef"));
    }

    #[test]
    fn different_inputs_differ() {
        assert_ne!(crc10_atm(b"abc"), crc10_atm(b"abd"));
    }

    #[test]
    fn order_matters() {
        assert_ne!(crc10_atm(b"ab"), crc10_atm(b"ba"));
    }

    #[test]
    fn length_sensitivity() {
        assert_ne!(crc10_atm(b"a"), crc10_atm(b"aa"));
    }

    #[test]
    fn single_bit_flip_changes_crc() {
        let base = crc10_atm(&[0b0000_0000]);
        let flip = crc10_atm(&[0b0000_0001]);
        assert_ne!(base, flip);
    }

    #[test]
    fn high_bit_flip_changes_crc() {
        let base = crc10_atm(&[0b0000_0000]);
        let flip = crc10_atm(&[0b1000_0000]);
        assert_ne!(base, flip);
    }

    #[test]
    fn repeated_byte_block() {
        let data = [0xAAu8; 32];
        assert_eq!(crc10_atm(&data), incremental_once(&data));
    }

    #[test]
    fn long_run_matches_incremental() {
        let data: Vec<u8> = (0u32..2048)
            .map(|i| (i.wrapping_mul(31) & 0xFF) as u8)
            .collect();
        let mut c = Crc10Atm::new();
        for chunk in data.chunks(13) {
            c.update(chunk);
        }
        assert_eq!(c.finalize(), crc10_atm(&data));
    }

    #[test]
    fn all_byte_values_in_range() {
        let data: Vec<u8> = (0u16..=255).map(|b| b as u8).collect();
        assert!(crc10_atm(&data) <= MASK);
    }

    #[test]
    fn ascii_digits_check_again() {
        let data = b"123456789";
        assert_eq!(crc10_atm(data), 0x199);
        assert_eq!(incremental_once(data), 0x199);
    }

    #[test]
    fn zero_block_various_lengths_in_range() {
        for len in 0..64 {
            let data = alloc::vec![0u8; len];
            assert!(crc10_atm(&data) <= MASK);
        }
    }

    #[test]
    fn constants_are_consistent() {
        assert_eq!(POLY, 0x233);
        assert_eq!(MASK, 0x3FF);
        assert_eq!(TOP_BIT, 0x200);
    }

    #[test]
    fn poly_fits_in_10_bits_plus_implicit() {
        const { assert!(POLY <= MASK) };
    }

    #[test]
    fn interleaved_updates_match_concatenation() {
        let a = b"first-part-";
        let b = b"second-part";
        let mut c = Crc10Atm::new();
        c.update(a);
        c.update(b);
        let mut combined = Vec::new();
        combined.extend_from_slice(a);
        combined.extend_from_slice(b);
        assert_eq!(c.finalize(), crc10_atm(&combined));
    }

    #[test]
    fn empty_then_data_matches_data() {
        let mut c = Crc10Atm::new();
        c.update(b"");
        c.update(b"payload");
        assert_eq!(c.finalize(), crc10_atm(b"payload"));
    }

    #[test]
    fn many_small_chunks_sample() {
        let data: Vec<u8> = (0u32..300).map(|i| (i & 0xFF) as u8).collect();
        let expected = crc10_atm(&data);
        let mut c = Crc10Atm::new();
        for chunk in data.chunks(1) {
            c.update(chunk);
        }
        assert_eq!(c.finalize(), expected);
    }
}
