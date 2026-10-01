//! Fletcher checksum family (`Fletcher-16`, `Fletcher-32`, `Fletcher-64`).
//!
//! This module is a pure-integer, `CPU` golden reference implementation of the
//! Fletcher position-dependent checksum. It is entirely integer based: there is
//! no floating point, no `unsafe`, and no transcendental math anywhere.
//!
//! # How Fletcher differs from `Adler-32`
//!
//! Both algorithms maintain two running sums (`sum1` accumulates the data words,
//! `sum2` accumulates `sum1`), but the reduction modulus is different and that is
//! the whole semantic distinction:
//!
//! * `Fletcher-16` reduces each 8-bit sum modulo `255` (`0xFF`).
//! * `Fletcher-32` reduces each 16-bit sum modulo `65535` (`0xFFFF`).
//! * `Fletcher-64` reduces each 32-bit sum modulo `4294967295` (`0xFFFF_FFFF`,
//!   i.e. `2^32 - 1`).
//!
//! Every Fletcher modulus is a value of the form `2^n - 1`. By contrast,
//! `Adler-32` (implemented in the sibling `adler32` module) reduces modulo the
//! prime `65521`. Because the moduli are different numbers, the two families
//! produce different digests for the same input and there is zero semantic
//! overlap between this module and `adler32`.
//!
//! # Combined word layout
//!
//! The finalized checksum packs the high sum above the low sum:
//!
//! * `Fletcher-16` returns `(sum2 << 8) | sum1`.
//! * `Fletcher-32` returns `(sum2 << 16) | sum1`.
//! * `Fletcher-64` returns `(sum2 << 32) | sum1`.
//!
//! # Reference vectors
//!
//! For the `ASCII` bytes of `"abcde"`, `Fletcher-16` yields `0xC8F0`; for
//! `"abcdef"` it yields `0x2057`; for `"abcdefgh"` it yields `0x0627`. These are
//! verified in the test module below.

/// Modulus used by `Fletcher-16` (`0xFF`).
pub const FLETCHER16_MODULUS: u16 = 255;

/// Modulus used by `Fletcher-32` (`0xFFFF`).
pub const FLETCHER32_MODULUS: u32 = 65535;

/// Modulus used by `Fletcher-64` (`0xFFFF_FFFF`, i.e. `2^32 - 1`).
pub const FLETCHER64_MODULUS: u64 = 4294967295;

/// Incremental state for the `Fletcher-16` checksum.
///
/// The two running sums are reduced modulo `255` (`0xFF`) after every byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fletcher16 {
    sum1: u16,
    sum2: u16,
}

impl Fletcher16 {
    /// Create a fresh, zeroed `Fletcher-16` state.
    #[must_use]
    pub const fn new() -> Self {
        Self { sum1: 0, sum2: 0 }
    }

    /// Fold a slice of bytes into the running state.
    pub fn update(&mut self, data: &[u8]) {
        for &byte in data {
            self.sum1 = (self.sum1 + u16::from(byte)) % FLETCHER16_MODULUS;
            self.sum2 = (self.sum2 + self.sum1) % FLETCHER16_MODULUS;
        }
    }

    /// Produce the combined checksum word `(sum2 << 8) | sum1`.
    #[must_use]
    pub fn finalize(&self) -> u16 {
        (self.sum2 << 8) | self.sum1
    }

    /// Return the low running sum (`sum1`).
    #[must_use]
    pub fn low_sum(&self) -> u16 {
        self.sum1
    }

    /// Return the high running sum (`sum2`).
    #[must_use]
    pub fn high_sum(&self) -> u16 {
        self.sum2
    }
}

impl Default for Fletcher16 {
    fn default() -> Self {
        Self::new()
    }
}

/// Incremental state for the `Fletcher-32` checksum.
///
/// Operates on 16-bit words; the two running sums are reduced modulo `65535`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fletcher32 {
    sum1: u32,
    sum2: u32,
}

impl Fletcher32 {
    /// Create a fresh, zeroed `Fletcher-32` state.
    #[must_use]
    pub const fn new() -> Self {
        Self { sum1: 0, sum2: 0 }
    }

    /// Fold a slice of 16-bit words into the running state.
    pub fn update(&mut self, words: &[u16]) {
        for &word in words {
            self.sum1 = (self.sum1 + u32::from(word)) % FLETCHER32_MODULUS;
            self.sum2 = (self.sum2 + self.sum1) % FLETCHER32_MODULUS;
        }
    }

    /// Fold raw bytes, packing little-endian pairs into 16-bit words.
    ///
    /// An odd trailing byte is zero-padded in the high byte position, matching
    /// the conventional `Fletcher-32` byte-oriented layout.
    pub fn update_bytes(&mut self, data: &[u8]) {
        let mut chunks = data.chunks_exact(2);
        for pair in chunks.by_ref() {
            let word = u16::from_le_bytes([pair[0], pair[1]]);
            self.sum1 = (self.sum1 + u32::from(word)) % FLETCHER32_MODULUS;
            self.sum2 = (self.sum2 + self.sum1) % FLETCHER32_MODULUS;
        }
        let remainder = chunks.remainder();
        if remainder.len() == 1 {
            let word = u16::from_le_bytes([remainder[0], 0]);
            self.sum1 = (self.sum1 + u32::from(word)) % FLETCHER32_MODULUS;
            self.sum2 = (self.sum2 + self.sum1) % FLETCHER32_MODULUS;
        }
    }

    /// Produce the combined checksum word `(sum2 << 16) | sum1`.
    #[must_use]
    pub fn finalize(&self) -> u32 {
        (self.sum2 << 16) | self.sum1
    }

    /// Return the low running sum (`sum1`).
    #[must_use]
    pub fn low_sum(&self) -> u32 {
        self.sum1
    }

    /// Return the high running sum (`sum2`).
    #[must_use]
    pub fn high_sum(&self) -> u32 {
        self.sum2
    }
}

impl Default for Fletcher32 {
    fn default() -> Self {
        Self::new()
    }
}

/// Incremental state for the `Fletcher-64` checksum.
///
/// Operates on 32-bit words; the two running sums are reduced modulo
/// `4294967295` (`2^32 - 1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fletcher64 {
    sum1: u64,
    sum2: u64,
}

impl Fletcher64 {
    /// Create a fresh, zeroed `Fletcher-64` state.
    #[must_use]
    pub const fn new() -> Self {
        Self { sum1: 0, sum2: 0 }
    }

    /// Fold a slice of 32-bit words into the running state.
    pub fn update(&mut self, words: &[u32]) {
        for &word in words {
            self.sum1 = (self.sum1 + u64::from(word)) % FLETCHER64_MODULUS;
            self.sum2 = (self.sum2 + self.sum1) % FLETCHER64_MODULUS;
        }
    }

    /// Fold raw bytes, packing little-endian quartets into 32-bit words.
    ///
    /// Trailing bytes that do not fill a full 32-bit word are zero-padded in the
    /// high byte positions.
    pub fn update_bytes(&mut self, data: &[u8]) {
        let mut chunks = data.chunks_exact(4);
        for quad in chunks.by_ref() {
            let word = u32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]);
            self.sum1 = (self.sum1 + u64::from(word)) % FLETCHER64_MODULUS;
            self.sum2 = (self.sum2 + self.sum1) % FLETCHER64_MODULUS;
        }
        let remainder = chunks.remainder();
        if !remainder.is_empty() {
            let mut bytes = [0u8; 4];
            bytes[..remainder.len()].copy_from_slice(remainder);
            let word = u32::from_le_bytes(bytes);
            self.sum1 = (self.sum1 + u64::from(word)) % FLETCHER64_MODULUS;
            self.sum2 = (self.sum2 + self.sum1) % FLETCHER64_MODULUS;
        }
    }

    /// Produce the combined checksum word `(sum2 << 32) | sum1`.
    #[must_use]
    pub fn finalize(&self) -> u64 {
        (self.sum2 << 32) | self.sum1
    }

    /// Return the low running sum (`sum1`).
    #[must_use]
    pub fn low_sum(&self) -> u64 {
        self.sum1
    }

    /// Return the high running sum (`sum2`).
    #[must_use]
    pub fn high_sum(&self) -> u64 {
        self.sum2
    }
}

impl Default for Fletcher64 {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot `Fletcher-16` over a byte slice.
#[must_use]
pub fn fletcher16(data: &[u8]) -> u16 {
    let mut state = Fletcher16::new();
    state.update(data);
    state.finalize()
}

/// One-shot `Fletcher-32` over a slice of 16-bit words.
#[must_use]
pub fn fletcher32(words: &[u16]) -> u32 {
    let mut state = Fletcher32::new();
    state.update(words);
    state.finalize()
}

/// One-shot `Fletcher-32` over raw bytes, packing little-endian 16-bit words.
///
/// An odd trailing byte is zero-padded in its high byte position.
#[must_use]
pub fn fletcher32_bytes(data: &[u8]) -> u32 {
    let mut state = Fletcher32::new();
    state.update_bytes(data);
    state.finalize()
}

/// One-shot `Fletcher-64` over a slice of 32-bit words.
#[must_use]
pub fn fletcher64(words: &[u32]) -> u64 {
    let mut state = Fletcher64::new();
    state.update(words);
    state.finalize()
}

/// One-shot `Fletcher-64` over raw bytes, packing little-endian 32-bit words.
///
/// Trailing bytes that do not fill a word are zero-padded in high positions.
#[must_use]
pub fn fletcher64_bytes(data: &[u8]) -> u64 {
    let mut state = Fletcher64::new();
    state.update_bytes(data);
    state.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[cfg(test)]
    fn repeating_bytes(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i & 0xff) as u8).collect()
    }

    #[cfg(test)]
    fn repeating_words16(len: usize) -> Vec<u16> {
        (0..len).map(|i| (i & 0xffff) as u16).collect()
    }

    #[cfg(test)]
    fn repeating_words32(len: usize) -> Vec<u32> {
        (0..len)
            .map(|i| (i as u32).wrapping_mul(2654435761))
            .collect()
    }

    #[test]
    fn empty_fletcher16() {
        assert_eq!(fletcher16(&[]), 0);
    }

    #[test]
    fn empty_fletcher32() {
        assert_eq!(fletcher32(&[]), 0);
    }

    #[test]
    fn empty_fletcher64() {
        assert_eq!(fletcher64(&[]), 0);
    }

    #[test]
    fn empty_bytes_helpers() {
        assert_eq!(fletcher32_bytes(&[]), 0);
        assert_eq!(fletcher64_bytes(&[]), 0);
    }

    #[test]
    fn single_byte_fletcher16_zero() {
        assert_eq!(fletcher16(&[0]), 0);
    }

    #[test]
    fn single_byte_fletcher16_one() {
        assert_eq!(fletcher16(&[1]), 0x0101);
    }

    #[test]
    fn single_byte_fletcher16_254() {
        assert_eq!(fletcher16(&[254]), 0xFEFE);
    }

    #[test]
    fn single_byte_fletcher16_255_wraps_to_zero() {
        // 255 reduced modulo 255 is 0 in both sums.
        assert_eq!(fletcher16(&[255]), 0);
    }

    #[test]
    fn single_word_fletcher32_one() {
        assert_eq!(fletcher32(&[1]), 0x0001_0001);
    }

    #[test]
    fn single_word_fletcher32_modulus_wraps() {
        assert_eq!(fletcher32(&[65535]), 0);
    }

    #[test]
    fn single_word_fletcher64_one() {
        assert_eq!(fletcher64(&[1]), 0x0000_0001_0000_0001);
    }

    #[test]
    fn single_word_fletcher64_modulus_wraps() {
        assert_eq!(fletcher64(&[4294967295]), 0);
    }

    #[test]
    fn reference_abcde() {
        assert_eq!(fletcher16(b"abcde"), 0xC8F0);
    }

    #[test]
    fn reference_abcdef() {
        assert_eq!(fletcher16(b"abcdef"), 0x2057);
    }

    #[test]
    fn reference_abcdefgh() {
        assert_eq!(fletcher16(b"abcdefgh"), 0x0627);
    }

    #[test]
    fn combined_layout_matches_sums_16() {
        let mut state = Fletcher16::new();
        state.update(b"abcde");
        let packed = state.finalize();
        assert_eq!(packed, (state.high_sum() << 8) | state.low_sum());
        assert_eq!(state.low_sum(), packed & 0x00FF);
        assert_eq!(state.high_sum(), packed >> 8);
    }

    #[test]
    fn combined_layout_matches_sums_32() {
        let mut state = Fletcher32::new();
        state.update(&[0x1234, 0x5678, 0x9abc]);
        let packed = state.finalize();
        assert_eq!(packed, (state.high_sum() << 16) | state.low_sum());
        assert_eq!(state.low_sum(), packed & 0x0000_FFFF);
        assert_eq!(state.high_sum(), packed >> 16);
    }

    #[test]
    fn combined_layout_matches_sums_64() {
        let mut state = Fletcher64::new();
        state.update(&[0x1111_2222, 0x3333_4444, 0x5555_6666]);
        let packed = state.finalize();
        assert_eq!(packed, (state.high_sum() << 32) | state.low_sum());
        assert_eq!(state.low_sum(), packed & 0x0000_0000_FFFF_FFFF);
        assert_eq!(state.high_sum(), packed >> 32);
    }

    #[test]
    fn incremental_equals_oneshot_16() {
        let data = repeating_bytes(500);
        let one_shot = fletcher16(&data);
        let mut state = Fletcher16::new();
        for chunk in data.chunks(7) {
            state.update(chunk);
        }
        assert_eq!(state.finalize(), one_shot);
    }

    #[test]
    fn incremental_equals_oneshot_32() {
        let data = repeating_words16(400);
        let one_shot = fletcher32(&data);
        let mut state = Fletcher32::new();
        for chunk in data.chunks(13) {
            state.update(chunk);
        }
        assert_eq!(state.finalize(), one_shot);
    }

    #[test]
    fn incremental_equals_oneshot_64() {
        let data = repeating_words32(300);
        let one_shot = fletcher64(&data);
        let mut state = Fletcher64::new();
        for chunk in data.chunks(5) {
            state.update(chunk);
        }
        assert_eq!(state.finalize(), one_shot);
    }

    #[test]
    fn incremental_split_points_16() {
        let data = repeating_bytes(64);
        let expected = fletcher16(&data);
        for split in 0..=data.len() {
            let mut state = Fletcher16::new();
            state.update(&data[..split]);
            state.update(&data[split..]);
            assert_eq!(state.finalize(), expected);
        }
    }

    #[test]
    fn odd_length_padding_32() {
        // A lone byte packs little-endian into the low byte of a word.
        assert_eq!(fletcher32_bytes(&[0x01]), fletcher32(&[0x0001]));
    }

    #[test]
    fn even_length_bytes_32() {
        let word = u16::from_le_bytes([0x01, 0x02]);
        assert_eq!(fletcher32_bytes(&[0x01, 0x02]), fletcher32(&[word]));
    }

    #[test]
    fn padding_matches_manual_32() {
        let data = [0xDE, 0xAD, 0xBE, 0xEF, 0x42];
        let manual = [
            u16::from_le_bytes([0xDE, 0xAD]),
            u16::from_le_bytes([0xBE, 0xEF]),
            u16::from_le_bytes([0x42, 0x00]),
        ];
        assert_eq!(fletcher32_bytes(&data), fletcher32(&manual));
    }

    #[test]
    fn padding_matches_manual_64() {
        let data = [0x01, 0x02, 0x03, 0x04, 0x05];
        let manual = [
            u32::from_le_bytes([0x01, 0x02, 0x03, 0x04]),
            u32::from_le_bytes([0x05, 0x00, 0x00, 0x00]),
        ];
        assert_eq!(fletcher64_bytes(&data), fletcher64(&manual));
    }

    #[test]
    fn byte_helper_three_bytes_64() {
        let data = [0xAA, 0xBB, 0xCC];
        let manual = [u32::from_le_bytes([0xAA, 0xBB, 0xCC, 0x00])];
        assert_eq!(fletcher64_bytes(&data), fletcher64(&manual));
    }

    #[test]
    fn modulus_wraparound_16_double_255() {
        // Two 0xFF bytes both reduce to zero.
        assert_eq!(fletcher16(&[255, 255]), 0);
    }

    #[test]
    fn modulus_wraparound_16_many_ones() {
        // 255 ones drives sum1 back to zero at the final step.
        let data: Vec<u8> = (0..255).map(|_| 1u8).collect();
        let state_sum1 = {
            let mut s = Fletcher16::new();
            s.update(&data);
            s.low_sum()
        };
        assert_eq!(state_sum1, 0);
    }

    #[test]
    fn modulus_wraparound_32_many_max() {
        // Every 0xFFFF word reduces to zero, keeping both sums at zero.
        let data: Vec<u16> = (0..100).map(|_| 65535u16).collect();
        assert_eq!(fletcher32(&data), 0);
    }

    #[test]
    fn modulus_wraparound_64_many_max() {
        let data: Vec<u32> = (0..50).map(|_| 4294967295u32).collect();
        assert_eq!(fletcher64(&data), 0);
    }

    #[test]
    fn determinism_16() {
        let data = repeating_bytes(321);
        assert_eq!(fletcher16(&data), fletcher16(&data));
    }

    #[test]
    fn determinism_32() {
        let data = repeating_words16(321);
        assert_eq!(fletcher32(&data), fletcher32(&data));
    }

    #[test]
    fn determinism_64() {
        let data = repeating_words32(321);
        assert_eq!(fletcher64(&data), fletcher64(&data));
    }

    #[test]
    fn idempotent_finalize_16() {
        let mut state = Fletcher16::new();
        state.update(b"prism");
        let first = state.finalize();
        let second = state.finalize();
        assert_eq!(first, second);
    }

    #[test]
    fn idempotent_finalize_32() {
        let mut state = Fletcher32::new();
        state.update(&[0x0102, 0x0304]);
        assert_eq!(state.finalize(), state.finalize());
    }

    #[test]
    fn idempotent_finalize_64() {
        let mut state = Fletcher64::new();
        state.update(&[0x0102_0304, 0x0506_0708]);
        assert_eq!(state.finalize(), state.finalize());
    }

    #[test]
    fn large_input_16() {
        let data = repeating_bytes(10_000);
        let expected = fletcher16(&data);
        let mut state = Fletcher16::new();
        for chunk in data.chunks(97) {
            state.update(chunk);
        }
        assert_eq!(state.finalize(), expected);
    }

    #[test]
    fn large_input_32() {
        let data = repeating_words16(10_000);
        let expected = fletcher32(&data);
        let mut state = Fletcher32::new();
        for chunk in data.chunks(101) {
            state.update(chunk);
        }
        assert_eq!(state.finalize(), expected);
    }

    #[test]
    fn large_input_64() {
        let data = repeating_words32(10_000);
        let expected = fletcher64(&data);
        let mut state = Fletcher64::new();
        for chunk in data.chunks(103) {
            state.update(chunk);
        }
        assert_eq!(state.finalize(), expected);
    }

    #[test]
    fn large_byte_helper_odd_length_32() {
        let data = repeating_bytes(9_999);
        let expected = fletcher32_bytes(&data);
        let mut state = Fletcher32::new();
        for chunk in data.chunks(2) {
            state.update_bytes(chunk);
        }
        assert_eq!(state.finalize(), expected);
    }

    #[test]
    fn default_equals_new_16() {
        assert_eq!(Fletcher16::default(), Fletcher16::new());
    }

    #[test]
    fn default_equals_new_32() {
        assert_eq!(Fletcher32::default(), Fletcher32::new());
    }

    #[test]
    fn default_equals_new_64() {
        assert_eq!(Fletcher64::default(), Fletcher64::new());
    }

    #[test]
    fn clone_state_matches_16() {
        let mut state = Fletcher16::new();
        state.update(b"half");
        let cloned = state;
        let mut a = state;
        let mut b = cloned;
        a.update(b"rest");
        b.update(b"rest");
        assert_eq!(a.finalize(), b.finalize());
    }

    #[test]
    fn all_zeros_stay_zero_16() {
        let data: Vec<u8> = (0..256).map(|_| 0u8).collect();
        assert_eq!(fletcher16(&data), 0);
    }

    #[test]
    fn all_zeros_stay_zero_32() {
        let data: Vec<u16> = (0..256).map(|_| 0u16).collect();
        assert_eq!(fletcher32(&data), 0);
    }

    #[test]
    fn all_zeros_stay_zero_64() {
        let data: Vec<u32> = (0..256).map(|_| 0u32).collect();
        assert_eq!(fletcher64(&data), 0);
    }

    #[test]
    fn empty_update_is_noop_16() {
        let mut state = Fletcher16::new();
        state.update(b"data");
        let before = state.finalize();
        state.update(&[]);
        assert_eq!(state.finalize(), before);
    }

    #[test]
    fn multiple_updates_accumulate_16() {
        let mut incremental = Fletcher16::new();
        incremental.update(b"abc");
        incremental.update(b"de");
        assert_eq!(incremental.finalize(), fletcher16(b"abcde"));
    }

    #[test]
    fn fresh_state_resets_16() {
        let mut state = Fletcher16::new();
        state.update(b"noise");
        state = Fletcher16::new();
        state.update(b"abcde");
        assert_eq!(state.finalize(), 0xC8F0);
    }

    #[test]
    fn words_vs_bytes_even_length_32() {
        let bytes = [0x11, 0x22, 0x33, 0x44];
        let words = [
            u16::from_le_bytes([0x11, 0x22]),
            u16::from_le_bytes([0x33, 0x44]),
        ];
        assert_eq!(fletcher32_bytes(&bytes), fletcher32(&words));
    }

    #[test]
    fn order_sensitivity_16() {
        // Fletcher is position dependent: reversing input changes the digest.
        let forward = fletcher16(b"abcd");
        let backward = fletcher16(b"dcba");
        assert_ne!(forward, backward);
    }

    #[test]
    fn differs_from_adler_modulus_choice_16() {
        // Sanity check the documented modulus constant is 2^8 - 1, not a prime.
        assert_eq!(FLETCHER16_MODULUS, 0xFF);
        assert_eq!(FLETCHER32_MODULUS, 0xFFFF);
        assert_eq!(FLETCHER64_MODULUS, 0xFFFF_FFFF);
    }
}
