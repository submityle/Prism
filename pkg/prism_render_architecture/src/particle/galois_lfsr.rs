//! Galois- and Fibonacci-form linear feedback shift register (`LFSR`)
//! pseudorandom bit/word sequence generators (pure integer, `no_std`).
//!
//! An `LFSR` is a *linear* feedback shift register: at each clock it shifts the
//! register by one bit and XORs in feedback derived from a fixed tap mask (the
//! characteristic polynomial). The emitted stream is a fully deterministic,
//! linear bit sequence. This is fundamentally different from an avalanche hash
//! such as `PCG`, Wang, or `SplitMix`: those apply nonlinear mixing (multiply,
//! xorshift, rotate) so that a single input-bit change avalanches across the
//! output. An `LFSR` performs no avalanche mixing at all — flipping one state
//! bit perturbs the output only linearly. Treat this module as a classic
//! `PRNG` bit-stream source, not as a hash and not as a cryptographic
//! generator (an `LFSR` output is trivially predictable from a few samples).
//!
//! # Forms
//! Two equivalent realizations of the same polynomial are provided:
//! - Galois form: shift right, and when the bit shifted out is `1`, XOR the
//!   whole register with the tap mask. One XOR per clock.
//! - Fibonacci form: compute a feedback bit as the parity of the tapped bits,
//!   shift right, and insert that feedback bit at the most significant position
//!   (`MSB`). Several taps combined per clock.
//!
//! For a primitive polynomial both forms traverse all `2^n - 1` nonzero states
//! before repeating (maximal length); the two forms produce mirror/reversed
//! sequences but share the same period.
//!
//! # Zero is a fixed point
//! The step function is a linear bijection on the state space with the
//! all-zero register as a fixed point: an all-zero register stays all-zero
//! forever and emits only `0`. Callers must seed with a nonzero value.
//!
//! # Chosen maximal-length polynomials
//! - 16-bit Galois mask `0xB400` is the standard maximal-length (primitive)
//!   tap set for `x^16 + x^14 + x^13 + x^11 + 1`; it yields period `65535`.
//! - 16-bit Fibonacci mask `0x002D` is the reciprocal (bit-reversed) tap set of
//!   `0xB400`; the reciprocal of a primitive polynomial is primitive, so it is
//!   also maximal-length with period `65535`.
//! - 32-bit Galois mask `0x80200003` encodes `x^32 + x^22 + x^2 + x^1 + 1`, a
//!   well-known primitive polynomial (period `2^32 - 1`).
//! - 64-bit Galois mask `0xD800000000000000` encodes
//!   `x^64 + x^63 + x^61 + x^60 + 1`, a well-known primitive polynomial
//!   (period `2^64 - 1`).
//!
//! # Period helper
//! [`period`] clocks a full 16-bit Galois register until it returns to the seed
//! (capped) and is used by tests. For small widths the tests mask a `u16`
//! register down to `n` bits (see the `#[cfg(test)]` masked helpers) to prove a
//! period of `2^n - 1` (for example period `15` at `n = 4` and period `255`
//! would follow the same construction).

use alloc::vec::Vec;

/// Maximal-length (primitive) 16-bit Galois tap mask, `0xB400`.
///
/// Encodes `x^16 + x^14 + x^13 + x^11 + 1`; period `65535` for any nonzero
/// seed.
pub const TAPS_16_GALOIS: u16 = 0xB400;

/// Maximal-length 16-bit Fibonacci tap mask, `0x002D`.
///
/// Reciprocal (bit-reversed) tap set of `0xB400`; also primitive, period
/// `65535`.
pub const TAPS_16_FIBONACCI: u16 = 0x002D;

/// Maximal-length (primitive) 32-bit Galois tap mask, `0x80200003`.
///
/// Encodes `x^32 + x^22 + x^2 + x^1 + 1`; period `2^32 - 1`.
pub const TAPS_32_GALOIS: u32 = 0x8020_0003;

/// Maximal-length 32-bit Fibonacci tap mask, `0xC0000401`.
///
/// Reciprocal (bit-reversed) tap set of `0x80200003`; also primitive.
pub const TAPS_32_FIBONACCI: u32 = 0xC000_0401;

/// Maximal-length (primitive) 64-bit Galois tap mask, `0xD800000000000000`.
///
/// Encodes `x^64 + x^63 + x^61 + x^60 + 1`; period `2^64 - 1`.
pub const TAPS_64_GALOIS: u64 = 0xD800_0000_0000_0000;

macro_rules! define_galois {
    ($name:ident, $t:ty) => {
        /// Galois-form `LFSR` over the given register width.
        ///
        /// Shift right; when the bit shifted out is `1`, XOR the register with
        /// the tap mask. The all-zero register is a fixed point.
        pub struct $name {
            register: $t,
            taps: $t,
        }

        impl $name {
            /// Create a new generator from a `seed` (initial register) and a
            /// tap mask (`taps`). Seed with a nonzero value: an all-zero
            /// register stays all-zero.
            pub const fn new(seed: $t, taps: $t) -> Self {
                Self {
                    register: seed,
                    taps,
                }
            }

            /// Current register state.
            pub const fn state(&self) -> $t {
                self.register
            }

            /// Configured tap mask (polynomial).
            pub const fn taps(&self) -> $t {
                self.taps
            }

            /// Clock the register by one bit; return the output bit (the bit
            /// shifted out of the least significant position).
            pub fn step(&mut self) -> bool {
                let lsb = (self.register & 1) == 1;
                self.register >>= 1;
                if lsb {
                    self.register ^= self.taps;
                }
                lsb
            }

            /// Clock the register `n` times, discarding the output bits.
            pub fn advance(&mut self, n: u64) {
                let mut i: u64 = 0;
                while i < n {
                    let _ = self.step();
                    i += 1;
                }
            }

            /// Clock `k` times (capped at 64) and accumulate the output bits
            /// most significant first (`MSB`-first) into a `u64`.
            pub fn next_bits(&mut self, k: u32) -> u64 {
                let steps = if k > 64 { 64 } else { k };
                let mut acc: u64 = 0;
                let mut i: u32 = 0;
                while i < steps {
                    acc = (acc << 1) | (self.step() as u64);
                    i += 1;
                }
                acc
            }

            /// Clock `k` times and collect each output bit into a `Vec`.
            pub fn collect_bits(&mut self, k: u32) -> Vec<bool> {
                let mut out = Vec::new();
                let mut i: u32 = 0;
                while i < k {
                    out.push(self.step());
                    i += 1;
                }
                out
            }
        }
    };
}

macro_rules! define_fibonacci {
    ($name:ident, $t:ty) => {
        /// Fibonacci-form `LFSR` over the given register width.
        ///
        /// Feedback is the parity of the tapped bits; shift right and insert
        /// the feedback bit at the most significant position (`MSB`). The
        /// all-zero register is a fixed point.
        pub struct $name {
            register: $t,
            taps: $t,
        }

        impl $name {
            /// Create a new generator from a `seed` (initial register) and a
            /// tap mask (`taps`). Seed with a nonzero value: an all-zero
            /// register stays all-zero.
            pub const fn new(seed: $t, taps: $t) -> Self {
                Self {
                    register: seed,
                    taps,
                }
            }

            /// Current register state.
            pub const fn state(&self) -> $t {
                self.register
            }

            /// Configured tap mask (polynomial).
            pub const fn taps(&self) -> $t {
                self.taps
            }

            /// Clock the register by one bit; return the output bit (the bit
            /// shifted out of the least significant position).
            pub fn step(&mut self) -> bool {
                let out = (self.register & 1) == 1;
                let feedback = (self.register & self.taps).count_ones() & 1;
                self.register >>= 1;
                self.register |= (feedback as $t) << (<$t>::BITS - 1);
                out
            }

            /// Clock the register `n` times, discarding the output bits.
            pub fn advance(&mut self, n: u64) {
                let mut i: u64 = 0;
                while i < n {
                    let _ = self.step();
                    i += 1;
                }
            }

            /// Clock `k` times (capped at 64) and accumulate the output bits
            /// most significant first (`MSB`-first) into a `u64`.
            pub fn next_bits(&mut self, k: u32) -> u64 {
                let steps = if k > 64 { 64 } else { k };
                let mut acc: u64 = 0;
                let mut i: u32 = 0;
                while i < steps {
                    acc = (acc << 1) | (self.step() as u64);
                    i += 1;
                }
                acc
            }

            /// Clock `k` times and collect each output bit into a `Vec`.
            pub fn collect_bits(&mut self, k: u32) -> Vec<bool> {
                let mut out = Vec::new();
                let mut i: u32 = 0;
                while i < k {
                    out.push(self.step());
                    i += 1;
                }
                out
            }
        }
    };
}

define_galois!(GaloisLfsr16, u16);
define_galois!(GaloisLfsr32, u32);
define_galois!(GaloisLfsr64, u64);

define_fibonacci!(FibonacciLfsr16, u16);
define_fibonacci!(FibonacciLfsr32, u32);
define_fibonacci!(FibonacciLfsr64, u64);

/// Compute the period of a full 16-bit Galois `LFSR`: clock until the register
/// returns to `seed`, capped at `1 << 17` iterations.
///
/// A nonzero `seed` with a primitive tap mask (for example [`TAPS_16_GALOIS`])
/// yields `65535`. An all-zero `seed` is a fixed point and reports `1`
/// (degenerate).
pub fn period(seed: u16, taps: u16) -> u64 {
    let start = seed;
    let mut lfsr = GaloisLfsr16::new(seed, taps);
    let cap: u64 = 1u64 << 17;
    let mut count: u64 = 0;
    loop {
        let _ = lfsr.step();
        count += 1;
        if lfsr.state() == start || count >= cap {
            break;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(test)]
    fn mask_for(width: u32) -> u16 {
        if width >= 16 {
            u16::MAX
        } else {
            (1u16 << width) - 1
        }
    }

    #[cfg(test)]
    fn galois_step_masked(state: u16, taps: u16, mask: u16) -> u16 {
        let lsb = (state & 1) == 1;
        let mut s = state >> 1;
        if lsb {
            s ^= taps;
        }
        s & mask
    }

    #[cfg(test)]
    fn fibonacci_step_masked(state: u16, taps: u16, width: u32) -> u16 {
        let mask = mask_for(width);
        let feedback = (state & taps).count_ones() & 1;
        let mut s = state >> 1;
        s |= (feedback as u16) << (width - 1);
        s & mask
    }

    #[cfg(test)]
    fn period_galois_masked(seed: u16, taps: u16, width: u32) -> u64 {
        let mask = mask_for(width);
        let start = seed & mask;
        let mut state = start;
        let cap: u64 = 1u64 << 20;
        let mut count: u64 = 0;
        loop {
            state = galois_step_masked(state, taps, mask);
            count += 1;
            if state == start || count >= cap {
                break;
            }
        }
        count
    }

    #[cfg(test)]
    fn period_fibonacci_masked(seed: u16, taps: u16, width: u32) -> u64 {
        let mask = mask_for(width);
        let start = seed & mask;
        let mut state = start;
        let cap: u64 = 1u64 << 20;
        let mut count: u64 = 0;
        loop {
            state = fibonacci_step_masked(state, taps, width);
            count += 1;
            if state == start || count >= cap {
                break;
            }
        }
        count
    }

    #[test]
    fn galois16_new_stores_state_and_taps() {
        let l = GaloisLfsr16::new(0xACE1, TAPS_16_GALOIS);
        assert_eq!(l.state(), 0xACE1);
        assert_eq!(l.taps(), TAPS_16_GALOIS);
    }

    #[test]
    fn galois16_zero_seed_stays_zero() {
        let mut l = GaloisLfsr16::new(0, TAPS_16_GALOIS);
        let mut i = 0;
        while i < 100 {
            let b = l.step();
            assert!(!b);
            assert_eq!(l.state(), 0);
            i += 1;
        }
    }

    #[test]
    fn galois16_step_changes_state_for_nonzero() {
        let mut l = GaloisLfsr16::new(1, TAPS_16_GALOIS);
        let before = l.state();
        let _ = l.step();
        assert_ne!(l.state(), before);
    }

    #[test]
    fn galois16_determinism_two_instances() {
        let mut a = GaloisLfsr16::new(0xBEEF, TAPS_16_GALOIS);
        let mut b = GaloisLfsr16::new(0xBEEF, TAPS_16_GALOIS);
        assert_eq!(a.next_bits(64), b.next_bits(64));
    }

    #[test]
    fn galois16_reproducible_after_reseed() {
        let mut a = GaloisLfsr16::new(0x1234, TAPS_16_GALOIS);
        let first = a.next_bits(40);
        let mut b = GaloisLfsr16::new(0x1234, TAPS_16_GALOIS);
        let second = b.next_bits(40);
        assert_eq!(first, second);
    }

    #[test]
    fn galois16_next_bits_zero_is_zero() {
        let mut l = GaloisLfsr16::new(0x9, TAPS_16_GALOIS);
        assert_eq!(l.next_bits(0), 0);
    }

    #[test]
    fn galois16_next_bits_msb_first() {
        let mut a = GaloisLfsr16::new(0x7, TAPS_16_GALOIS);
        let mut b = GaloisLfsr16::new(0x7, TAPS_16_GALOIS);
        let mut acc: u64 = 0;
        let mut i = 0;
        while i < 10 {
            acc = (acc << 1) | (b.step() as u64);
            i += 1;
        }
        assert_eq!(a.next_bits(10), acc);
    }

    #[test]
    fn galois16_next_bits_caps_at_64() {
        let mut a = GaloisLfsr16::new(0x3, TAPS_16_GALOIS);
        let mut b = GaloisLfsr16::new(0x3, TAPS_16_GALOIS);
        assert_eq!(a.next_bits(100), b.next_bits(64));
    }

    #[test]
    fn galois16_advance_matches_repeated_steps() {
        let mut a = GaloisLfsr16::new(0x55, TAPS_16_GALOIS);
        let mut b = GaloisLfsr16::new(0x55, TAPS_16_GALOIS);
        a.advance(1000);
        let mut i = 0;
        while i < 1000 {
            let _ = b.step();
            i += 1;
        }
        assert_eq!(a.state(), b.state());
    }

    #[test]
    fn galois16_advance_zero_is_noop() {
        let mut a = GaloisLfsr16::new(0x55, TAPS_16_GALOIS);
        let s = a.state();
        a.advance(0);
        assert_eq!(a.state(), s);
    }

    #[test]
    fn galois16_collect_bits_matches_steps() {
        let mut a = GaloisLfsr16::new(0x1F, TAPS_16_GALOIS);
        let mut b = GaloisLfsr16::new(0x1F, TAPS_16_GALOIS);
        let bits = a.collect_bits(20);
        assert_eq!(bits.len(), 20);
        let mut i: usize = 0;
        while i < 20 {
            assert_eq!(bits[i], b.step());
            i += 1;
        }
    }

    #[test]
    fn galois16_full_period_is_65535() {
        assert_eq!(period(1, TAPS_16_GALOIS), 65535);
    }

    #[test]
    fn galois16_full_period_other_seed_is_65535() {
        assert_eq!(period(0xACE1, TAPS_16_GALOIS), 65535);
    }

    #[test]
    fn galois32_zero_seed_stays_zero() {
        let mut l = GaloisLfsr32::new(0, TAPS_32_GALOIS);
        let mut i = 0;
        while i < 100 {
            assert!(!l.step());
            assert_eq!(l.state(), 0);
            i += 1;
        }
    }

    #[test]
    fn galois32_determinism() {
        let mut a = GaloisLfsr32::new(0x1234_5678, TAPS_32_GALOIS);
        let mut b = GaloisLfsr32::new(0x1234_5678, TAPS_32_GALOIS);
        assert_eq!(a.next_bits(64), b.next_bits(64));
    }

    #[test]
    fn galois32_next_bits_state_nonzero() {
        let mut a = GaloisLfsr32::new(0xDEAD_BEEF, TAPS_32_GALOIS);
        let _ = a.next_bits(32);
        assert_ne!(a.state(), 0);
    }

    #[test]
    fn galois32_eventually_outputs_one() {
        let mut l = GaloisLfsr32::new(1, TAPS_32_GALOIS);
        let mut seen = false;
        let mut i = 0;
        while i < 1000 {
            if l.step() {
                seen = true;
                break;
            }
            i += 1;
        }
        assert!(seen);
    }

    #[test]
    fn galois32_large_step_count_deterministic() {
        let mut a = GaloisLfsr32::new(0xABCD, TAPS_32_GALOIS);
        let mut b = GaloisLfsr32::new(0xABCD, TAPS_32_GALOIS);
        a.advance(100_000);
        b.advance(100_000);
        assert_eq!(a.state(), b.state());
    }

    #[test]
    fn galois64_zero_seed_stays_zero() {
        let mut l = GaloisLfsr64::new(0, TAPS_64_GALOIS);
        let mut i = 0;
        while i < 100 {
            assert!(!l.step());
            assert_eq!(l.state(), 0);
            i += 1;
        }
    }

    #[test]
    fn galois64_determinism() {
        let mut a = GaloisLfsr64::new(0x1, TAPS_64_GALOIS);
        let mut b = GaloisLfsr64::new(0x1, TAPS_64_GALOIS);
        assert_eq!(a.next_bits(64), b.next_bits(64));
    }

    #[test]
    fn galois64_next_bits_64() {
        let mut a = GaloisLfsr64::new(0x9E37_79B9_7F4A_7C15, TAPS_64_GALOIS);
        let mut b = GaloisLfsr64::new(0x9E37_79B9_7F4A_7C15, TAPS_64_GALOIS);
        assert_eq!(a.next_bits(64), b.next_bits(64));
    }

    #[test]
    fn galois64_large_step_count_nonzero() {
        let mut a = GaloisLfsr64::new(0xF00D, TAPS_64_GALOIS);
        a.advance(500_000);
        assert_ne!(a.state(), 0);
    }

    #[test]
    fn fib16_zero_seed_stays_zero() {
        let mut l = FibonacciLfsr16::new(0, TAPS_16_FIBONACCI);
        let mut i = 0;
        while i < 100 {
            assert!(!l.step());
            assert_eq!(l.state(), 0);
            i += 1;
        }
    }

    #[test]
    fn fib16_determinism() {
        let mut a = FibonacciLfsr16::new(0xBEEF, TAPS_16_FIBONACCI);
        let mut b = FibonacciLfsr16::new(0xBEEF, TAPS_16_FIBONACCI);
        assert_eq!(a.next_bits(64), b.next_bits(64));
    }

    #[test]
    fn fib16_step_changes_state_for_nonzero() {
        let mut l = FibonacciLfsr16::new(1, TAPS_16_FIBONACCI);
        let before = l.state();
        let _ = l.step();
        assert_ne!(l.state(), before);
    }

    #[test]
    fn fib16_next_bits_msb_first() {
        let mut a = FibonacciLfsr16::new(0x5, TAPS_16_FIBONACCI);
        let mut b = FibonacciLfsr16::new(0x5, TAPS_16_FIBONACCI);
        let mut acc: u64 = 0;
        let mut i = 0;
        while i < 12 {
            acc = (acc << 1) | (b.step() as u64);
            i += 1;
        }
        assert_eq!(a.next_bits(12), acc);
    }

    #[test]
    fn fib16_full_period_is_65535() {
        assert_eq!(period_fibonacci_masked(1, TAPS_16_FIBONACCI, 16), 65535);
    }

    #[test]
    fn fib16_reproducible_after_reseed() {
        let mut a = FibonacciLfsr16::new(0x4321, TAPS_16_FIBONACCI);
        let first = a.next_bits(50);
        let mut b = FibonacciLfsr16::new(0x4321, TAPS_16_FIBONACCI);
        assert_eq!(first, b.next_bits(50));
    }

    #[test]
    fn fib32_zero_seed_stays_zero() {
        let mut l = FibonacciLfsr32::new(0, TAPS_32_FIBONACCI);
        let mut i = 0;
        while i < 100 {
            assert!(!l.step());
            assert_eq!(l.state(), 0);
            i += 1;
        }
    }

    #[test]
    fn fib32_determinism() {
        let mut a = FibonacciLfsr32::new(0x1357_9BDF, TAPS_32_FIBONACCI);
        let mut b = FibonacciLfsr32::new(0x1357_9BDF, TAPS_32_FIBONACCI);
        assert_eq!(a.next_bits(64), b.next_bits(64));
    }

    #[test]
    fn fib32_nonzero_stays_nonzero() {
        let mut a = FibonacciLfsr32::new(0x2468, TAPS_32_FIBONACCI);
        a.advance(50_000);
        assert_ne!(a.state(), 0);
    }

    #[test]
    fn fib64_zero_seed_stays_zero() {
        let mut l = FibonacciLfsr64::new(0, TAPS_32_FIBONACCI as u64);
        let mut i = 0;
        while i < 100 {
            assert!(!l.step());
            assert_eq!(l.state(), 0);
            i += 1;
        }
    }

    #[test]
    fn fib64_determinism() {
        let mut a = FibonacciLfsr64::new(0xDEAD_BEEF_CAFE_F00D, 0x1B);
        let mut b = FibonacciLfsr64::new(0xDEAD_BEEF_CAFE_F00D, 0x1B);
        assert_eq!(a.next_bits(64), b.next_bits(64));
    }

    #[test]
    fn fib64_next_bits_64() {
        let mut a = FibonacciLfsr64::new(0x1, 0x1B);
        let mut b = FibonacciLfsr64::new(0x1, 0x1B);
        assert_eq!(a.next_bits(64), b.next_bits(64));
    }

    #[test]
    fn galois_masked_4bit_period_is_15() {
        assert_eq!(period_galois_masked(1, 0xC, 4), 15);
    }

    #[test]
    fn galois_masked_4bit_all_nonzero_seeds_period_15() {
        let mut seed: u16 = 1;
        while seed < 16 {
            assert_eq!(period_galois_masked(seed, 0xC, 4), 15);
            seed += 1;
        }
    }

    #[test]
    fn galois_masked_3bit_period_is_7() {
        assert_eq!(period_galois_masked(1, 0x6, 3), 7);
    }

    #[test]
    fn galois_masked_2bit_period_is_3() {
        assert_eq!(period_galois_masked(1, 0x3, 2), 3);
    }

    #[test]
    fn fibonacci_masked_4bit_period_is_15() {
        assert_eq!(period_fibonacci_masked(1, 0x3, 4), 15);
    }

    #[test]
    fn fibonacci_masked_3bit_period_is_7() {
        assert_eq!(period_fibonacci_masked(1, 0x3, 3), 7);
    }

    #[test]
    fn fibonacci_masked_2bit_period_is_3() {
        assert_eq!(period_fibonacci_masked(1, 0x3, 2), 3);
    }

    #[test]
    fn galois_fibonacci_period_equivalence_4bit() {
        assert_eq!(
            period_galois_masked(1, 0xC, 4),
            period_fibonacci_masked(1, 0x3, 4)
        );
    }

    #[test]
    fn galois_fibonacci_period_equivalence_3bit() {
        assert_eq!(
            period_galois_masked(1, 0x6, 3),
            period_fibonacci_masked(1, 0x3, 3)
        );
    }

    #[test]
    fn public_period_matches_masked_16bit() {
        assert_eq!(
            period(1, TAPS_16_GALOIS),
            period_galois_masked(1, TAPS_16_GALOIS, 16)
        );
    }

    #[test]
    fn galois_masked_4bit_never_returns_zero_from_nonzero() {
        let mask = mask_for(4);
        let mut state: u16 = 1;
        let mut i = 0;
        while i < 15 {
            state = galois_step_masked(state, 0xC, mask);
            assert_ne!(state, 0);
            i += 1;
        }
    }

    #[test]
    fn fibonacci_masked_4bit_never_returns_zero_from_nonzero() {
        let mut state: u16 = 1;
        let mut i = 0;
        while i < 15 {
            state = fibonacci_step_masked(state, 0x3, 4);
            assert_ne!(state, 0);
            i += 1;
        }
    }
}
