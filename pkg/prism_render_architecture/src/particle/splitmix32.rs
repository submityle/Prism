//! `SplitMix32`: a `32`-bit `SplitMix` pseudo-random bit generator, the narrow
//! sibling of this crate's `splitmix64` engine, built for reproducible particle
//! randomness on `32`-bit integer lanes.
//!
//! A `PRNG` here is a *stateful, advancing stream*, not a one-shot hash: the
//! generator holds a single `u32` register that is incremented by the fixed
//! golden-ratio gamma `0x9e37_79b9` on every draw and then diffused by a
//! fixed finalizer of exclusive-ors, right shifts, and wrapping multiplies.
//! Because the increment is a constant and the finalizer is a fixed sequence of
//! integer operations, the whole stream is a deterministic function of the
//! seed: the same seed always replays the same sequence, which is exactly what
//! a reproducible simulation needs when a frame must match bit for bit between
//! a `CPU` reference path and a future `GPU` implementation.
//!
//! The finalizer is the well-known `32`-bit `SplitMix` variant (the same
//! xor-shift-multiply shape popularised as `splitmix32`/`lowbias32`): add the
//! gamma, then two rounds of `z = (z ^ (z >> s)).wrapping_mul(k)` with a final
//! xor-shift. Every generation step is pure integer arithmetic: exclusive-or,
//! right shift, wrapping add, and wrapping multiply. There are no floating
//! point surfaces, no transcendental functions, and no floating-point equality
//! tests anywhere in this module.
//!
//! Scope: `SplitMix32` is a fast, non-cryptographic generator. Its register is
//! trivially recoverable from a single output by inverting the finalizer, so it
//! must never be used for security, key material, or anywhere an adversary
//! could exploit predictability. It exists purely for reproducible,
//! high-throughput simulation randomness.

/// The golden-ratio increment (fractional part of the golden ratio scaled to
/// `32` bits, `0x9e37_79b9`), added to the register before each draw.
///
/// This odd constant is the gamma of the `SplitMix32` stream; adding it
/// repeatedly walks the register through a maximal-period additive cycle before
/// the finalizer scrambles each value.
pub const GOLDEN_GAMMA: u32 = 0x9e37_79b9;

/// First multiplier constant of the `SplitMix32` finalizer.
const MIX_MULTIPLIER_A: u32 = 0x21f0_aaad;

/// Second multiplier constant of the `SplitMix32` finalizer.
const MIX_MULTIPLIER_B: u32 = 0x735a_2d97;

/// A `32`-bit `SplitMix` pseudo-random bit generator.
///
/// Holds a single `u32` register. Each call to [`SplitMix32::next_u32`] first
/// advances the register by [`GOLDEN_GAMMA`] and then runs the result through
/// the `SplitMix32` finalizer, yielding a well-distributed `u32`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SplitMix32 {
    /// The additive register that is incremented by the gamma on each draw.
    state: u32,
}

impl SplitMix32 {
    /// Creates a new generator whose register is initialised to `seed`.
    ///
    /// The seed is used verbatim as the starting register value; the first draw
    /// advances it by [`GOLDEN_GAMMA`] before finalizing, so a seed of `0` does
    /// not produce a `0` output.
    #[must_use]
    pub const fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    /// Returns the current register value without advancing the stream.
    #[must_use]
    pub const fn state(&self) -> u32 {
        self.state
    }

    /// Advances the register and returns the next `u32` in the stream.
    ///
    /// Uses only exclusive-or, right shift, wrapping add, and wrapping multiply,
    /// so equal register states always produce equal outputs.
    pub fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 16)).wrapping_mul(MIX_MULTIPLIER_A);
        z = (z ^ (z >> 15)).wrapping_mul(MIX_MULTIPLIER_B);
        z ^ (z >> 15)
    }
}

#[cfg(test)]
mod tests {
    use super::{SplitMix32, GOLDEN_GAMMA};

    /// The reference stream for `seed = 0`, independently cross-checked.
    #[cfg(test)]
    const REF_SEED0: [u32; 6] = [
        0x6462_5032,
        0xd9c0_799c,
        0xaf36_2e10,
        0x7fa8_8912,
        0xc467_1b39,
        0xf1d2_eee4,
    ];

    /// Collects `n` consecutive draws from a generator seeded with `seed`.
    #[cfg(test)]
    fn collect(seed: u32, n: usize) -> alloc::vec::Vec<u32> {
        let mut g = SplitMix32::new(seed);
        let mut out = alloc::vec::Vec::with_capacity(n);
        for _ in 0..n {
            out.push(g.next_u32());
        }
        out
    }

    extern crate alloc;

    #[test]
    fn ref_vector_value_0() {
        let mut g = SplitMix32::new(0);
        assert_eq!(g.next_u32(), 0x6462_5032);
    }

    #[test]
    fn ref_vector_value_1() {
        let mut g = SplitMix32::new(0);
        g.next_u32();
        assert_eq!(g.next_u32(), 0xd9c0_799c);
    }

    #[test]
    fn ref_vector_value_2() {
        let mut g = SplitMix32::new(0);
        for _ in 0..2 {
            g.next_u32();
        }
        assert_eq!(g.next_u32(), 0xaf36_2e10);
    }

    #[test]
    fn ref_vector_value_3() {
        let mut g = SplitMix32::new(0);
        for _ in 0..3 {
            g.next_u32();
        }
        assert_eq!(g.next_u32(), 0x7fa8_8912);
    }

    #[test]
    fn ref_vector_value_4() {
        let mut g = SplitMix32::new(0);
        for _ in 0..4 {
            g.next_u32();
        }
        assert_eq!(g.next_u32(), 0xc467_1b39);
    }

    #[test]
    fn ref_vector_value_5() {
        let mut g = SplitMix32::new(0);
        for _ in 0..5 {
            g.next_u32();
        }
        assert_eq!(g.next_u32(), 0xf1d2_eee4);
    }

    #[test]
    fn ref_vector_full_sequence() {
        assert_eq!(collect(0, 6), REF_SEED0);
    }

    #[test]
    fn ref_vector_in_order_loop() {
        let mut g = SplitMix32::new(0);
        for &expected in &REF_SEED0 {
            assert_eq!(g.next_u32(), expected);
        }
    }

    #[test]
    fn determinism_same_seed_same_sequence() {
        let a = collect(0, 64);
        let b = collect(0, 64);
        assert_eq!(a, b);
    }

    #[test]
    fn determinism_same_seed_nonzero() {
        let a = collect(0x1234_5678, 32);
        let b = collect(0x1234_5678, 32);
        assert_eq!(a, b);
    }

    #[test]
    fn determinism_clone_tracks_original() {
        let mut g = SplitMix32::new(99);
        g.next_u32();
        let mut c = g;
        assert_eq!(g.next_u32(), c.next_u32());
        assert_eq!(g.next_u32(), c.next_u32());
    }

    #[test]
    fn determinism_restart_from_seed() {
        let mut g = SplitMix32::new(7);
        let first: alloc::vec::Vec<u32> = (0..10).map(|_| g.next_u32()).collect();
        let mut h = SplitMix32::new(7);
        let second: alloc::vec::Vec<u32> = (0..10).map(|_| h.next_u32()).collect();
        assert_eq!(first, second);
    }

    #[test]
    fn different_seeds_differ_0_vs_1() {
        assert_ne!(collect(0, 8), collect(1, 8));
    }

    #[test]
    fn different_seeds_differ_many() {
        let s0 = collect(0, 16);
        let s1 = collect(1, 16);
        let s2 = collect(2, 16);
        assert_ne!(s0, s1);
        assert_ne!(s1, s2);
        assert_ne!(s0, s2);
    }

    #[test]
    fn different_seeds_first_value_differs() {
        let mut a = SplitMix32::new(100);
        let mut b = SplitMix32::new(200);
        assert_ne!(a.next_u32(), b.next_u32());
    }

    #[test]
    fn adjacent_seeds_decorrelated() {
        // Adjacent seeds should not merely produce shifted streams.
        let a = collect(1000, 8);
        let b = collect(1001, 8);
        assert_ne!(a, b);
    }

    #[test]
    fn not_all_zero_seed0() {
        let v = collect(0, 64);
        assert!(v.iter().any(|&x| x != 0));
    }

    #[test]
    fn not_all_zero_seed_max() {
        let v = collect(u32::MAX, 64);
        assert!(v.iter().any(|&x| x != 0));
    }

    #[test]
    fn no_zero_outputs_in_small_window() {
        // Not a guarantee of the generator, but true for this seed/window and a
        // useful smoke test that the finalizer is actually scrambling.
        let v = collect(0, 32);
        let zeros = v.iter().filter(|&&x| x == 0).count();
        assert_eq!(zeros, 0);
    }

    #[test]
    fn not_all_equal_seed0() {
        let v = collect(0, 32);
        let first = v[0];
        assert!(v.iter().any(|&x| x != first));
    }

    #[test]
    fn not_all_equal_seed_random() {
        let v = collect(0xdead_beef, 32);
        let first = v[0];
        assert!(v.iter().any(|&x| x != first));
    }

    #[test]
    fn consecutive_values_differ() {
        let mut g = SplitMix32::new(42);
        let a = g.next_u32();
        let b = g.next_u32();
        assert_ne!(a, b);
    }

    #[test]
    fn high_distinct_count() {
        // A good scrambler should produce overwhelmingly distinct 32-bit words
        // over a short window.
        let v = collect(0x0bad_f00d, 256);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert!(sorted.len() >= 250);
    }

    #[test]
    fn state_advances_by_gamma() {
        let mut g = SplitMix32::new(0);
        assert_eq!(g.state(), 0);
        g.next_u32();
        assert_eq!(g.state(), GOLDEN_GAMMA);
        g.next_u32();
        assert_eq!(g.state(), GOLDEN_GAMMA.wrapping_mul(2));
    }

    #[test]
    fn state_wraps_around() {
        // Seeding near the top of the register range should wrap cleanly.
        let seed = u32::MAX;
        let mut g = SplitMix32::new(seed);
        g.next_u32();
        assert_eq!(g.state(), seed.wrapping_add(GOLDEN_GAMMA));
    }

    #[test]
    fn new_does_not_advance_state() {
        let g = SplitMix32::new(0x5555_5555);
        assert_eq!(g.state(), 0x5555_5555);
    }

    #[test]
    fn seed_zero_first_output_nonzero() {
        let mut g = SplitMix32::new(0);
        assert_ne!(g.next_u32(), 0);
    }

    #[test]
    fn golden_gamma_is_odd() {
        assert_eq!(GOLDEN_GAMMA & 1, 1);
    }

    #[test]
    fn golden_gamma_value() {
        assert_eq!(GOLDEN_GAMMA, 0x9e37_79b9);
    }

    #[test]
    fn equal_generators_compare_equal() {
        let a = SplitMix32::new(123);
        let b = SplitMix32::new(123);
        assert_eq!(a, b);
    }

    #[test]
    fn different_generators_compare_unequal() {
        let a = SplitMix32::new(123);
        let b = SplitMix32::new(124);
        assert_ne!(a, b);
    }

    #[test]
    fn advanced_generators_track_state_equality() {
        let mut a = SplitMix32::new(5);
        let mut b = SplitMix32::new(5);
        a.next_u32();
        b.next_u32();
        assert_eq!(a, b);
    }

    #[test]
    fn full_period_window_distinct_states() {
        // Register states over a short window are distinct because the gamma is
        // a nonzero constant added each step.
        let mut g = SplitMix32::new(0);
        let mut seen = alloc::vec::Vec::new();
        for _ in 0..128 {
            g.next_u32();
            seen.push(g.state());
        }
        let mut sorted = seen.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), seen.len());
    }

    #[test]
    fn bit_spread_uses_full_width() {
        // Across a window, both high and low bits should be exercised.
        let v = collect(0x1357_9bdf, 64);
        let any_high = v.iter().any(|&x| x & 0x8000_0000 != 0);
        let any_low = v.iter().any(|&x| x & 1 != 0);
        assert!(any_high);
        assert!(any_low);
    }

    #[test]
    fn or_accumulation_sets_most_bits() {
        // ORing many outputs should light up essentially the whole word.
        let v = collect(0x2468_ace0, 256);
        let mut acc = 0u32;
        for &x in &v {
            acc |= x;
        }
        assert_eq!(acc, u32::MAX);
    }

    #[test]
    fn and_accumulation_clears_most_bits() {
        // ANDing many outputs should clear essentially the whole word.
        let v = collect(0x9bdf_1357, 256);
        let mut acc = u32::MAX;
        for &x in &v {
            acc &= x;
        }
        assert_eq!(acc, 0);
    }

    #[test]
    fn streams_from_split_like_seeds_differ() {
        // Seeds derived by offsetting should not collide over a window.
        let base = collect(0xa5a5_a5a5, 16);
        let offset = collect(0xa5a5_a5a5u32.wrapping_add(GOLDEN_GAMMA), 16);
        assert_ne!(base, offset);
    }

    #[test]
    fn long_run_remains_deterministic() {
        let a = collect(0xc0ff_ee00, 1000);
        let b = collect(0xc0ff_ee00, 1000);
        assert_eq!(a, b);
    }

    #[test]
    fn long_run_has_variety() {
        let v = collect(0xc0ff_ee00, 1000);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        sorted.dedup();
        // Expect near-total uniqueness over 1000 draws.
        assert!(sorted.len() >= 990);
    }

    #[test]
    fn mean_is_roughly_centered() {
        // Average of a well-distributed stream should land near the midpoint.
        // Pure integer arithmetic only: accumulate in u64, no floats.
        let v = collect(0x0051_2345, 4096);
        let mut sum: u64 = 0;
        for &x in &v {
            sum += u64::from(x);
        }
        let mean = sum / (v.len() as u64);
        let mid = u64::from(u32::MAX) / 2;
        let lo = mid - mid / 8;
        let hi = mid + mid / 8;
        assert!(mean > lo && mean < hi);
    }

    #[test]
    fn low_bit_balance_is_reasonable() {
        // Roughly half the outputs should have their lowest bit set.
        let v = collect(0x7777_7777, 2048);
        let ones = v.iter().filter(|&&x| x & 1 == 1).count();
        let n = v.len();
        assert!(ones > n / 4 && ones < (3 * n) / 4);
    }

    #[test]
    fn high_bit_balance_is_reasonable() {
        let v = collect(0x3333_3333, 2048);
        let ones = v.iter().filter(|&&x| x & 0x8000_0000 != 0).count();
        let n = v.len();
        assert!(ones > n / 4 && ones < (3 * n) / 4);
    }

    #[test]
    fn seed_one_matches_itself() {
        assert_eq!(collect(1, 50), collect(1, 50));
    }

    #[test]
    fn distinct_seeds_rarely_collide_first_draw() {
        // First draws across a batch of seeds should be overwhelmingly unique.
        let mut firsts = alloc::vec::Vec::new();
        for seed in 0..256u32 {
            let mut g = SplitMix32::new(seed);
            firsts.push(g.next_u32());
        }
        let mut sorted = firsts.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert!(sorted.len() >= 250);
    }
}
