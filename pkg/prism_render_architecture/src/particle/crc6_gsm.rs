//! `CRC-6/GSM` checksum implemented with a bit-at-a-time, MSB-first algorithm.
//!
//! This module provides a pure-integer, `no_std`-friendly implementation of the
//! `CRC-6/GSM` algorithm as described by the common `CRC` catalog.
//!
//! Parameters:
//! - width = 6
//! - poly = `0x2f`
//! - init = `0x00`
//! - refin = false
//! - refout = false
//! - xorout = `0x3f`
//!
//! The result is a 6-bit value stored in the low six bits of a [`u8`]. The
//! implementation avoids any floating-point or transcendental operations and
//! relies only on shifts, masks, and exclusive-or so that it behaves identically
//! on any `CPU` or `GPU` integer pipeline.

/// Bit width of the `CRC-6/GSM` register.
const WIDTH: u32 = 6;

/// Generator polynomial for `CRC-6/GSM` (`0x2f`).
const POLY: u8 = 0x2f;

/// Mask selecting the low [`WIDTH`] bits (`0x3f`).
const MASK: u8 = (1 << WIDTH) - 1;

/// Final `xorout` value applied to the register (`0x3f`).
const XOROUT: u8 = 0x3f;

/// Compute the `CRC-6/GSM` checksum of `data`.
///
/// The returned value is always in the inclusive range `0..=0x3f` because only
/// the low six bits of the register are meaningful.
///
/// # Examples
///
/// ```ignore
/// assert!(crc6_gsm(b"123456789") == 0x13);
/// ```
pub fn crc6_gsm(data: &[u8]) -> u8 {
    let mut crc: u8 = 0;
    for &b in data {
        for i in (0..8).rev() {
            let bit = (b >> i) & 1;
            let msb = (crc >> (WIDTH - 1)) & 1;
            crc = (crc << 1) & MASK;
            if (msb ^ bit) == 1 {
                crc ^= POLY;
            }
            crc &= MASK;
        }
    }
    (crc ^ XOROUT) & MASK
}

/// Incremental `CRC-6/GSM` register that can process bytes in chunks.
///
/// The register holds the pre-`xorout` state; call [`Crc6Gsm::finalize`] to
/// obtain the final checksum value.
#[derive(Clone, Copy, Debug)]
pub struct Crc6Gsm {
    /// Current register value, stored in the low [`WIDTH`] bits.
    state: u8,
}

impl Crc6Gsm {
    /// Create a new register seeded with the `init` value (`0x00`).
    #[must_use]
    pub const fn new() -> Self {
        Self { state: 0 }
    }

    /// Feed a slice of bytes into the register.
    pub fn update(&mut self, data: &[u8]) {
        let mut crc = self.state;
        for &b in data {
            for i in (0..8).rev() {
                let bit = (b >> i) & 1;
                let msb = (crc >> (WIDTH - 1)) & 1;
                crc = (crc << 1) & MASK;
                if (msb ^ bit) == 1 {
                    crc ^= POLY;
                }
                crc &= MASK;
            }
        }
        self.state = crc;
    }

    /// Produce the final checksum by applying `xorout` and masking.
    #[must_use]
    pub const fn finalize(&self) -> u8 {
        (self.state ^ XOROUT) & MASK
    }
}

impl Default for Crc6Gsm {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_empty() {
        assert!(crc6_gsm(b"") == 0x3f);
    }

    #[test]
    fn anchor_single_a() {
        assert!(crc6_gsm(b"a") == 0x19);
    }

    #[test]
    fn anchor_zero_byte() {
        assert!(crc6_gsm(&[0x00]) == 0x3f);
    }

    #[test]
    fn anchor_ff_byte() {
        assert!(crc6_gsm(&[0xff]) == 0x17);
    }

    #[test]
    fn anchor_check_value() {
        assert!(crc6_gsm(b"123456789") == 0x13);
    }

    #[test]
    fn single_byte_0x61_matches_a() {
        assert!(crc6_gsm(&[0x61]) == crc6_gsm(b"a"));
    }

    #[test]
    fn result_fits_six_bits_empty() {
        assert!(crc6_gsm(b"") <= MASK);
    }

    #[test]
    fn result_fits_six_bits_zero() {
        assert!(crc6_gsm(&[0x00]) <= 0x3f);
    }

    #[test]
    fn result_fits_six_bits_ff() {
        assert!(crc6_gsm(&[0xff]) <= 0x3f);
    }

    #[test]
    fn result_fits_six_bits_check() {
        assert!(crc6_gsm(b"123456789") <= 0x3f);
    }

    #[test]
    fn all_single_bytes_fit_six_bits() {
        for v in 0u16..=255 {
            let b = v as u8;
            let input = [b];
            assert!(crc6_gsm(&input) <= MASK);
        }
    }

    #[test]
    fn empty_equals_zero_vector_differs() {
        assert!(crc6_gsm(b"") == crc6_gsm(&[]));
    }

    #[test]
    fn empty_differs_from_single_a() {
        assert!(crc6_gsm(b"") != crc6_gsm(b"a"));
    }

    #[test]
    fn zero_differs_from_ff() {
        assert!(crc6_gsm(&[0x00]) != crc6_gsm(&[0xff]));
    }

    #[test]
    fn empty_equals_zero_byte_result() {
        // Both empty and a single zero byte happen to yield 0x3f here.
        assert!(crc6_gsm(b"") == crc6_gsm(&[0x00]));
    }

    #[test]
    fn order_sensitivity_ab_vs_ba() {
        assert!(crc6_gsm(&[0x01, 0x02]) != crc6_gsm(&[0x02, 0x01]));
    }

    #[test]
    fn order_sensitivity_text() {
        assert!(crc6_gsm(b"ab") != crc6_gsm(b"ba"));
    }

    #[test]
    fn two_zero_bytes_fit() {
        assert!(crc6_gsm(&[0x00, 0x00]) <= MASK);
    }

    #[test]
    fn two_zero_bytes_derived_behavior() {
        // Derived by calling the function: additional zero bytes keep the
        // register at zero, so the final value stays 0x3f for all-zero inputs.
        assert!(crc6_gsm(&[0x00, 0x00]) == crc6_gsm(&[0x00]));
        assert!(crc6_gsm(&[0x00, 0x00, 0x00]) == 0x3f);
    }

    #[test]
    fn repeated_ff_fit() {
        assert!(crc6_gsm(&[0xff, 0xff, 0xff]) <= MASK);
    }

    #[test]
    fn deterministic_check_value() {
        assert!(crc6_gsm(b"123456789") == crc6_gsm(b"123456789"));
    }

    #[test]
    fn deterministic_random_like() {
        let data = [0xde, 0xad, 0xbe, 0xef];
        assert!(crc6_gsm(&data) == crc6_gsm(&data));
    }

    #[test]
    fn streaming_matches_oneshot_full() {
        let data = [0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39];
        let mut c = Crc6Gsm::new();
        c.update(&data);
        assert!(c.finalize() == crc6_gsm(&data));
    }

    #[test]
    fn streaming_matches_oneshot_split() {
        let data = [0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39];
        let mut c = Crc6Gsm::new();
        c.update(&data[0..4]);
        c.update(&data[4..9]);
        assert!(c.finalize() == 0x13);
    }

    #[test]
    fn streaming_byte_by_byte() {
        let data = [0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39];
        let mut c = Crc6Gsm::new();
        for &b in &data {
            let one = [b];
            c.update(&one);
        }
        assert!(c.finalize() == 0x13);
    }

    #[test]
    fn streaming_empty_equals_oneshot_empty() {
        let c = Crc6Gsm::new();
        assert!(c.finalize() == crc6_gsm(b""));
    }

    #[test]
    fn default_equals_new() {
        let a = Crc6Gsm::new();
        let b = Crc6Gsm::default();
        assert!(a.finalize() == b.finalize());
    }

    #[test]
    fn streaming_single_a() {
        let mut c = Crc6Gsm::new();
        c.update(b"a");
        assert!(c.finalize() == 0x19);
    }

    #[test]
    fn streaming_zero_byte() {
        let mut c = Crc6Gsm::new();
        c.update(&[0x00]);
        assert!(c.finalize() == 0x3f);
    }

    #[test]
    fn streaming_ff_byte() {
        let mut c = Crc6Gsm::new();
        c.update(&[0xff]);
        assert!(c.finalize() == 0x17);
    }

    #[test]
    fn split_concat_equivalence_text() {
        let whole = b"HelloWorld";
        let mut c = Crc6Gsm::new();
        c.update(&whole[0..5]);
        c.update(&whole[5..10]);
        assert!(c.finalize() == crc6_gsm(whole));
    }

    #[test]
    fn split_at_every_position() {
        let data = [0x10, 0x20, 0x30, 0x40, 0x50];
        let expected = crc6_gsm(&data);
        for split in 0..=data.len() {
            let mut c = Crc6Gsm::new();
            c.update(&data[0..split]);
            c.update(&data[split..]);
            assert!(c.finalize() == expected);
        }
    }

    #[test]
    fn poly_mask_relationship() {
        assert!(MASK == 0x3f);
        assert!(POLY <= MASK + 1);
    }

    #[test]
    fn xorout_is_mask() {
        assert!(XOROUT == MASK);
    }

    #[test]
    fn width_is_six() {
        assert!(WIDTH == 6);
    }

    #[test]
    fn many_messages_fit_six_bits() {
        let samples: [&[u8]; 6] = [
            b"",
            b"a",
            b"abc",
            b"message digest",
            b"123456789",
            &[0x00, 0xff, 0x7f, 0x80],
        ];
        for &s in &samples {
            assert!(crc6_gsm(s) <= MASK);
        }
    }

    #[test]
    fn appending_zero_changes_result() {
        let base = crc6_gsm(b"a");
        let appended = crc6_gsm(&[0x61, 0x00]);
        assert!(base != appended);
    }

    #[test]
    fn length_sensitivity() {
        let one = crc6_gsm(&[0x41]);
        let two = crc6_gsm(&[0x41, 0x41]);
        let three = crc6_gsm(&[0x41, 0x41, 0x41]);
        assert!(one != two);
        assert!(two != three);
    }

    #[test]
    fn prefix_free_distinct() {
        let short = crc6_gsm(b"12345");
        let long = crc6_gsm(b"123456789");
        assert!(short != long);
    }

    #[test]
    fn bit_flip_changes_result_in_most_cases() {
        // Flipping the low bit of a byte frequently changes the checksum.
        let a = crc6_gsm(&[0x00, 0x00, 0x00, 0x00]);
        let b = crc6_gsm(&[0x00, 0x00, 0x00, 0x01]);
        assert!(a != b);
    }

    #[test]
    fn streaming_state_fits_six_bits() {
        let mut c = Crc6Gsm::new();
        c.update(b"intermediate");
        assert!(c.state <= MASK);
    }

    #[test]
    fn exhaustive_two_byte_results_fit() {
        for hi in 0u16..=255 {
            let data = [hi as u8, 0xa5];
            assert!(crc6_gsm(&data) <= MASK);
        }
    }
}
