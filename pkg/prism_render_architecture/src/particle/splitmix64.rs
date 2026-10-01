//! `SplitMix64`: Sebastiano Vigna's `splitmix64` pseudo-random bit generator,
//! the same fast integer finalizer that seeds Java 8's `SplittableRandom` and
//! that this crate's `xorshift_rng` engines use to diffuse a raw seed.
//!
//! A `PRNG` here is a *stateful, advancing stream*, not a one-shot hash: the
//! generator holds a single `u64` register that is incremented by the fixed
//! golden-ratio gamma `0x9e3779b97f4a7c15` on every draw and then run through
//! the `splitmix64` finalizer. Because the increment is a constant and the
//! finalizer is a fixed sequence of exclusive-ors, right shifts, and wrapping
//! multiplies, the whole stream is a deterministic function of the seed: the
//! same seed always replays the same sequence, which is exactly what a
//! reproducible particle simulation needs when a frame must match bit for bit
//! between the `CPU` reference path and a future `GPU` implementation.
//!
//! The module exposes three layers. [`splitmix64_mix`] is the stateless
//! finalizer `mix(z: u64) -> u64`, usable as a pure integer scrambler. The
//! [`SplitMix64`] struct wraps a register and advances it with
//! [`SplitMix64::next_u64`]; derived draws cover `u32` words
//! ([`SplitMix64::next_u32`]), unbiased bounded integers via Daniel Lemire's
//! rejection method ([`SplitMix64::next_bounded_u64`]), and unit-interval
//! floating point built from integer mantissas ([`SplitMix64::next_f64_unit`],
//! [`SplitMix64::next_f32_unit`]). Finally [`SplitMix64::split`] derives a fresh
//! independent sub-stream in the spirit of `SplittableRandom`.
//!
//! Every generation step is pure integer arithmetic: exclusive-or, right shift,
//! wrapping add, and wrapping multiply. The only floating-point surfaces are the
//! unit-interval mappings, which construct an integer mantissa first and then
//! perform a single division by a power of two; no transcendental function,
//! rounding, or floating-point equality test is used anywhere.
//!
//! Scope: `splitmix64` is a fast, non-cryptographic generator. Its state is
//! trivially recoverable from a single output by inverting the finalizer, so it
//! must never be used for security, key material, or anywhere an adversary could
//! exploit predictability. It exists purely for reproducible, high-throughput
//! simulation randomness.

/// The golden-ratio increment (fractional part of the golden ratio scaled to
/// `64` bits, `0x9e3779b97f4a7c15`), added to the register before each draw.
///
/// This odd constant is the gamma of the default `splitmix64` stream; adding it
/// repeatedly walks the register through a maximal-period additive cycle before
/// the finalizer scrambles each value.
pub const GOLDEN_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

/// First multiplier constant of the `splitmix64` finalizer.
const MIX_MULTIPLIER_A: u64 = 0xbf58_476d_1ce4_e5b9;

/// Second multiplier constant of the `splitmix64` finalizer.
const MIX_MULTIPLIER_B: u64 = 0x94d0_49bb_1331_11eb;

/// The `splitmix64` finalizer: scrambles one `u64` into a well-distributed
/// `u64` through two xor-shift-multiply rounds and a final xor-shift.
///
/// This is the stateless mixing core shared by every `SplitMix64` draw and by
/// the seed diffuser in `xorshift_rng`. It uses only exclusive-or, right
/// shifts, and wrapping multiplies, so it is referentially transparent: equal
/// inputs always yield equal outputs. Flipping a single input bit flips, on
/// average, close to half of the output bits (strong avalanche).
#[must_use]
pub const fn splitmix64_mix(z: u64) -> u64 {
    let mut z = z;
    z = (z ^ (z >> 30)).wrapping_mul(MIX_MULTIPLIER_A);
    z = (z ^ (z >> 27)).wrapping_mul(MIX_MULTIPLIER_B);
    z ^ (z >> 31)
}

/// A stateful `splitmix64` generator advancing a single `u64` register.
///
/// Each draw first adds [`GOLDEN_GAMMA`] to the register and then returns the
/// [`splitmix64_mix`] of the updated register. Construction stores the seed
/// directly, so the first [`SplitMix64::next_u64`] returns
/// `splitmix64_mix(seed + GOLDEN_GAMMA)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SplitMix64 {
    /// The additive register; incremented by [`GOLDEN_GAMMA`] before each mix.
    state: u64,
}

impl SplitMix64 {
    /// Constructs a generator whose register starts at `seed`.
    ///
    /// Unlike `xorshift`, `splitmix64` has no degenerate fixed point, so any
    /// seed -- including zero -- is a valid starting register and is stored
    /// verbatim.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns the current raw register without advancing the stream.
    #[must_use]
    pub const fn state(&self) -> u64 {
        self.state
    }

    /// Advances the register by [`GOLDEN_GAMMA`] and returns the mixed `u64`.
    ///
    /// This is the fundamental draw; all other draws are derived from it.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        splitmix64_mix(self.state)
    }

    /// Returns the high `32` bits of the next `u64` draw as a `u32`.
    ///
    /// The high bits are taken because the top of a `splitmix64` word carries
    /// the strongest mixing. One full register advance is consumed per call.
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Returns an unbiased integer in `[0, bound)` using Lemire's method.
    ///
    /// When `bound` is zero the half-open range is empty and the function
    /// returns `0`. Otherwise it multiplies a fresh `u64` draw by `bound` in
    /// `128`-bit arithmetic and takes the high half; the rare low-half values
    /// that would skew the distribution are rejected and redrawn, so the result
    /// is exactly uniform with no modulo bias. Only integer multiply, shift,
    /// and remainder are used.
    pub fn next_bounded_u64(&mut self, bound: u64) -> u64 {
        if bound == 0 {
            return 0;
        }
        let mut product = u128::from(self.next_u64()).wrapping_mul(u128::from(bound));
        let mut low = product as u64;
        if low < bound {
            // Rejection threshold: (2^64 mod bound) == (-bound) mod bound.
            let threshold = bound.wrapping_neg() % bound;
            while low < threshold {
                product = u128::from(self.next_u64()).wrapping_mul(u128::from(bound));
                low = product as u64;
            }
        }
        (product >> 64) as u64
    }

    /// Returns an `f64` in the half-open unit interval `[0, 1)`.
    ///
    /// The high `53` bits of a fresh draw form an integer mantissa that is
    /// divided by `2^53`, giving `2^53` equally spaced values. `53` bits is the
    /// full significand width of a binary64 (`IEEE` double), so every
    /// representable value in the grid is reachable. The construction uses one
    /// shift and one division by a power of two; it never calls a transcendental
    /// function.
    pub fn next_f64_unit(&mut self) -> f64 {
        let mantissa = self.next_u64() >> 11;
        let scale = (1_u64 << 53) as f64;
        mantissa as f64 / scale
    }

    /// Returns an `f32` in the half-open unit interval `[0, 1)`.
    ///
    /// The high `24` bits of a fresh draw form an integer mantissa that is
    /// divided by `2^24`, matching the `24`-bit significand of a binary32
    /// (`IEEE` single). The construction uses one shift and one division by a
    /// power of two.
    pub fn next_f32_unit(&mut self) -> f32 {
        let mantissa = (self.next_u64() >> 40) as u32;
        let scale = (1_u32 << 24) as f32;
        mantissa as f32 / scale
    }

    /// Derives a fresh, independent sub-stream in the `SplittableRandom` style.
    ///
    /// A draw is taken from this stream (advancing it) and run through
    /// [`splitmix64_mix`] a second time to decorrelate it; the result seeds the
    /// returned child. The parent and child therefore walk different additive
    /// cycles and produce statistically independent sequences, which is useful
    /// for handing each emitter or worker its own reproducible generator.
    pub fn split(&mut self) -> Self {
        let derived = splitmix64_mix(self.next_u64());
        Self::new(derived)
    }
}

impl Iterator for SplitMix64 {
    type Item = u64;

    /// Yields successive [`SplitMix64::next_u64`] draws; the stream never ends.
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.next_u64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference-vector convention: the register starts at the seed, each draw
    /// adds `GOLDEN_GAMMA` *before* mixing, and these are the published
    /// `splitmix64` outputs for `seed = 0`.
    const SEED0_VECTORS: [u64; 5] = [
        0xe220_a839_7b1d_cdaf,
        0x6e78_9e6a_a1b9_65f4,
        0x06c4_5d18_8009_454f,
        0xf88b_b8a8_724c_81ec,
        0x1b39_896a_51a8_749b,
    ];

    #[test]
    fn reference_vector_first() {
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next_u64(), SEED0_VECTORS[0]);
    }

    #[test]
    fn reference_vector_second() {
        let mut rng = SplitMix64::new(0);
        let _ = rng.next_u64();
        assert_eq!(rng.next_u64(), SEED0_VECTORS[1]);
    }

    #[test]
    fn reference_vector_third() {
        let mut rng = SplitMix64::new(0);
        for _ in 0..2 {
            let _ = rng.next_u64();
        }
        assert_eq!(rng.next_u64(), SEED0_VECTORS[2]);
    }

    #[test]
    fn reference_vector_fourth() {
        let mut rng = SplitMix64::new(0);
        for _ in 0..3 {
            let _ = rng.next_u64();
        }
        assert_eq!(rng.next_u64(), SEED0_VECTORS[3]);
    }

    #[test]
    fn reference_vector_fifth() {
        let mut rng = SplitMix64::new(0);
        for _ in 0..4 {
            let _ = rng.next_u64();
        }
        assert_eq!(rng.next_u64(), SEED0_VECTORS[4]);
    }

    #[test]
    fn reference_vector_full_sequence() {
        let mut rng = SplitMix64::new(0);
        let got: [u64; 5] = [
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ];
        assert_eq!(got, SEED0_VECTORS);
    }

    #[test]
    fn first_draw_equals_mix_of_seed_plus_gamma() {
        let seed = 0x1234_5678_9abc_def0_u64;
        let mut rng = SplitMix64::new(seed);
        let expected = splitmix64_mix(seed.wrapping_add(GOLDEN_GAMMA));
        assert_eq!(rng.next_u64(), expected);
    }

    #[test]
    fn new_stores_seed_verbatim() {
        let seed = 0xdead_beef_cafe_babe_u64;
        let rng = SplitMix64::new(seed);
        assert_eq!(rng.state(), seed);
    }

    #[test]
    fn deterministic_reproducible() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..256 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = SplitMix64::new(1);
        let mut b = SplitMix64::new(2);
        let mut differences = 0_u32;
        for _ in 0..64 {
            if a.next_u64() != b.next_u64() {
                differences += 1;
            }
        }
        // Independent streams should disagree on essentially every draw.
        assert!(differences > 60);
    }

    #[test]
    fn mix_is_deterministic() {
        assert_eq!(splitmix64_mix(12_345), splitmix64_mix(12_345));
    }

    #[test]
    fn mix_fixed_point_at_zero() {
        // The finalizer is built from xor-shifts and wrapping multiplies, all
        // of which map `0` to `0`, so the raw mix has a fixed point at zero.
        // The stateful generator never exposes this: it adds `GOLDEN_GAMMA`
        // before the first mix, so the first emitted draw is nonzero.
        assert_eq!(splitmix64_mix(0), 0);
        assert_ne!(splitmix64_mix(GOLDEN_GAMMA), 0);
    }

    #[test]
    fn mix_distinct_inputs_distinct_outputs() {
        // The finalizer is a bijection, so a block of consecutive inputs must
        // map to a block of distinct outputs.
        let mut seen = [0_u64; 64];
        for (i, slot) in seen.iter_mut().enumerate() {
            *slot = splitmix64_mix(i as u64);
        }
        for i in 0..seen.len() {
            for j in (i + 1)..seen.len() {
                assert_ne!(seen[i], seen[j]);
            }
        }
    }

    #[test]
    fn next_u32_is_high_half_of_u64() {
        let mut wide = SplitMix64::new(777);
        let mut narrow = SplitMix64::new(777);
        for _ in 0..100 {
            let full = wide.next_u64();
            assert_eq!(narrow.next_u32(), (full >> 32) as u32);
        }
    }

    #[test]
    fn next_u32_reproducible() {
        let mut a = SplitMix64::new(9);
        let mut b = SplitMix64::new(9);
        for _ in 0..100 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn bounded_zero_returns_zero() {
        let mut rng = SplitMix64::new(5);
        assert_eq!(rng.next_bounded_u64(0), 0);
    }

    #[test]
    fn bounded_one_always_zero() {
        let mut rng = SplitMix64::new(5);
        for _ in 0..1000 {
            assert_eq!(rng.next_bounded_u64(1), 0);
        }
    }

    #[test]
    fn bounded_within_range() {
        let mut rng = SplitMix64::new(0xabcd);
        for bound in [2_u64, 3, 7, 10, 100, 1000, 65_537] {
            for _ in 0..500 {
                let value = rng.next_bounded_u64(bound);
                assert!(value < bound);
            }
        }
    }

    #[test]
    fn bounded_max_bound_within_range() {
        let mut rng = SplitMix64::new(0x1111);
        let bound = u64::MAX;
        for _ in 0..1000 {
            let value = rng.next_bounded_u64(bound);
            assert!(value < bound);
        }
    }

    #[test]
    fn bounded_reproducible() {
        let mut a = SplitMix64::new(314);
        let mut b = SplitMix64::new(314);
        for _ in 0..500 {
            assert_eq!(a.next_bounded_u64(97), b.next_bounded_u64(97));
        }
    }

    #[test]
    fn bounded_covers_all_values() {
        // Over enough samples every residue in a small range should appear.
        let mut rng = SplitMix64::new(2024);
        let bound = 8_u64;
        let mut seen = [false; 8];
        for _ in 0..2000 {
            let value = rng.next_bounded_u64(bound);
            seen[value as usize] = true;
        }
        assert!(seen.iter().all(|hit| *hit));
    }

    #[test]
    fn bounded_approximately_uniform() {
        // Bucket counts for a small range should stay close to the mean, with
        // no empty bucket and none wildly over-represented.
        let mut rng = SplitMix64::new(98_765);
        let bound = 10_u64;
        let samples = 100_000_u64;
        let mut counts = [0_u64; 10];
        for _ in 0..samples {
            let value = rng.next_bounded_u64(bound);
            counts[value as usize] += 1;
        }
        let expected = samples / bound;
        let tolerance = expected / 10;
        for count in counts {
            let delta = count.abs_diff(expected);
            assert!(delta < tolerance);
        }
    }

    #[test]
    fn f64_unit_in_range() {
        let mut rng = SplitMix64::new(0xf00d);
        for _ in 0..10_000 {
            let value = rng.next_f64_unit();
            assert!(value >= 0.0);
            assert!(value < 1.0);
        }
    }

    #[test]
    fn f64_unit_reproducible() {
        let mut a = SplitMix64::new(55);
        let mut b = SplitMix64::new(55);
        for _ in 0..1000 {
            // Compare bit patterns rather than using float equality.
            assert_eq!(a.next_f64_unit().to_bits(), b.next_f64_unit().to_bits());
        }
    }

    #[test]
    fn f64_unit_mean_near_half() {
        let mut rng = SplitMix64::new(13);
        let samples = 200_000_u32;
        let mut sum = 0.0_f64;
        for _ in 0..samples {
            sum += rng.next_f64_unit();
        }
        let mean = sum / f64::from(samples);
        assert!((mean - 0.5).abs() < 0.01);
    }

    #[test]
    fn f32_unit_in_range() {
        let mut rng = SplitMix64::new(0xbead);
        for _ in 0..10_000 {
            let value = rng.next_f32_unit();
            assert!(value >= 0.0);
            assert!(value < 1.0);
        }
    }

    #[test]
    fn f32_unit_reproducible() {
        let mut a = SplitMix64::new(71);
        let mut b = SplitMix64::new(71);
        for _ in 0..1000 {
            assert_eq!(a.next_f32_unit().to_bits(), b.next_f32_unit().to_bits());
        }
    }

    #[test]
    fn f32_unit_mean_near_half() {
        let mut rng = SplitMix64::new(29);
        let samples = 200_000_u32;
        let mut sum = 0.0_f32;
        for _ in 0..samples {
            sum += rng.next_f32_unit();
        }
        let mean = sum / (samples as f32);
        assert!((mean - 0.5).abs() < 0.01);
    }

    #[test]
    fn split_differs_from_parent() {
        let mut parent = SplitMix64::new(2718);
        let mut child = parent.split();
        let mut agreements = 0_u32;
        for _ in 0..64 {
            if parent.next_u64() == child.next_u64() {
                agreements += 1;
            }
        }
        assert_eq!(agreements, 0);
    }

    #[test]
    fn split_children_differ() {
        let mut parent = SplitMix64::new(161);
        let mut first = parent.split();
        let mut second = parent.split();
        let mut agreements = 0_u32;
        for _ in 0..64 {
            if first.next_u64() == second.next_u64() {
                agreements += 1;
            }
        }
        assert_eq!(agreements, 0);
    }

    #[test]
    fn split_is_reproducible() {
        let mut a = SplitMix64::new(900);
        let mut b = SplitMix64::new(900);
        let mut child_a = a.split();
        let mut child_b = b.split();
        for _ in 0..256 {
            assert_eq!(child_a.next_u64(), child_b.next_u64());
        }
    }

    #[test]
    fn iterator_matches_next_u64() {
        let mut direct = SplitMix64::new(12_321);
        let mut iter = SplitMix64::new(12_321);
        for _ in 0..200 {
            assert_eq!(iter.next(), Some(direct.next_u64()));
        }
    }

    #[test]
    fn iterator_take_is_deterministic() {
        let mut first = [0_u64; 32];
        let mut second = [0_u64; 32];
        for (slot, value) in first.iter_mut().zip(SplitMix64::new(5).take(32)) {
            *slot = value;
        }
        for (slot, value) in second.iter_mut().zip(SplitMix64::new(5).take(32)) {
            *slot = value;
        }
        assert_eq!(first, second);
    }

    #[test]
    fn avalanche_single_bit_flip() {
        // Flipping one input bit should flip roughly half of the 64 output
        // bits (Hamming distance via popcount of the exclusive-or).
        let base_inputs: [u64; 6] = [
            0,
            1,
            0x0123_4567_89ab_cdef,
            0xffff_ffff_ffff_ffff,
            0x5555_5555_5555_5555,
            0xdead_beef_0000_0001,
        ];
        let mut total_bits = 0_u64;
        let mut trials = 0_u64;
        for base in base_inputs {
            let base_out = splitmix64_mix(base);
            for bit in 0..64_u32 {
                let flipped = splitmix64_mix(base ^ (1_u64 << bit));
                total_bits += (base_out ^ flipped).count_ones() as u64;
                trials += 1;
            }
        }
        let mean = total_bits / trials;
        // A strong finalizer averages close to 32 flipped bits out of 64.
        assert!(mean > 28);
        assert!(mean < 36);
    }

    #[test]
    fn stream_not_constant() {
        let mut rng = SplitMix64::new(0);
        let first = rng.next_u64();
        let mut all_same = true;
        for _ in 0..64 {
            if rng.next_u64() != first {
                all_same = false;
            }
        }
        assert!(!all_same);
    }
}
