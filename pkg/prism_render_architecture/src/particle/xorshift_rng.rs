//! Deterministic integer pseudo-random number generators of the `xorshift`
//! family, seeded through the `splitmix64` diffuser, for reproducible particle
//! spawning, jitter, and stochastic effects across `CPU` and `GPU` backends.
//!
//! Unlike the fixed hashes in this crate (`fnv1a_hash`, `crc32`, `murmur3`),
//! which map an input to a single digest, a `PRNG` here is a *stateful,
//! advancing stream*: each generator holds an internal register that mutates on
//! every draw, so repeated calls yield a long, reproducible sequence. Given the
//! same seed a generator always produces the same sequence, which is exactly
//! what a deterministic simulation needs when the same frame must replay
//! identically on different machines or between the `CPU` reference path and a
//! `GPU` implementation.
//!
//! Three integer engines are provided. [`XorShift32`] advances a single `u32`
//! register with the classic `xorshift` triple `(13, 17, 5)`; it is the
//! smallest and fastest but has the shortest period. [`XorShift64`] advances a
//! `u64` register with the triple `(13, 7, 17)` for a far longer period.
//! [`XorShift128`] keeps four `u32` words in Marsaglia's classic
//! `xorshift128` recurrence, giving a period near `2^128 - 1` from a small
//! state.
//!
//! Seeding always runs the user seed through [`splitmix64`] (or, for
//! [`XorShift32`], a non-zero fallback) so that a caller-supplied zero -- or a
//! low-entropy seed such as `1` -- never leaves an engine in or near its
//! degenerate all-zero fixed point, from which `xorshift` can never escape.
//!
//! Every state transition uses only integer exclusive-or, shifts, and wrapping
//! add/multiply -- no floating point is involved in generation. The only `f32`
//! surface is the unit-interval mapping [`u32_to_unit_f32`], which discards the
//! eight low (`LSB`) bits and multiplies by a constant reciprocal of `2^24` to
//! land in the half-open range `[0, 1)`; that mapping performs a single
//! multiply and no division, rounding, or transcendental call.
//!
//! Scope: these are fast, non-cryptographic generators. `xorshift` streams are
//! trivially predictable from a few outputs and must never be used for
//! security, key material, or anywhere an adversary could exploit
//! predictability. They exist purely for reproducible, high-throughput
//! simulation randomness.

/// The reciprocal of `2^24` (`1.0 / 16_777_216.0`), used to scale a 24-bit
/// integer mantissa into the unit interval with a single multiply.
const INV_TWO_POW_24: f32 = 1.0 / 16_777_216.0;

/// A non-zero fallback register for [`XorShift32`] when the caller passes a
/// zero seed; equal to the low 32 bits of the golden-ratio constant.
const XORSHIFT32_FALLBACK: u32 = 0x9E37_79B9;

/// The `splitmix64` seed diffuser: advances `state` by the golden-ratio
/// increment and returns a well-mixed `u64`.
///
/// This is the standard `splitmix64` used to expand a single seed into the
/// higher-quality initial state of the `xorshift` engines. Calling it
/// repeatedly on the same `state` variable produces a deterministic stream of
/// `u64` values; each call mutates `state` in place and returns the mixed
/// output. Only exclusive-or, right shifts, and wrapping add/multiply are used.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A 32-bit `xorshift` generator advancing a single `u32` register.
///
/// Uses the classic shift triple `(13, 17, 5)`. The register is never zero
/// after construction, so the stream never collapses to the all-zero fixed
/// point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct XorShift32 {
    state: u32,
}

/// Constructs a [`XorShift32`] from `seed`.
///
/// If `seed` is zero the register is replaced with a fixed non-zero fallback,
/// because a zero register is the degenerate fixed point of `xorshift` and
/// would emit only zeros. Any non-zero `seed` is used directly.
#[must_use]
pub fn xorshift32_new(seed: u32) -> XorShift32 {
    // A zero seed would trap the `xorshift` recurrence at its all-zero fixed
    // point, so substitute a fixed non-zero constant instead.
    let state = if seed == 0 { XORSHIFT32_FALLBACK } else { seed };
    XorShift32 { state }
}

impl XorShift32 {
    /// Advances the register and returns the next `u32` in the sequence.
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    /// Returns the current internal register value without advancing it.
    #[must_use]
    pub fn state(&self) -> u32 {
        self.state
    }
}

/// A 64-bit `xorshift` generator advancing a single `u64` register.
///
/// Uses the shift triple `(13, 7, 17)` and is seeded through [`splitmix64`]
/// so that even a zero seed yields a non-degenerate register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct XorShift64 {
    state: u64,
}

/// Constructs a [`XorShift64`] by diffusing `seed` through [`splitmix64`].
///
/// Running the seed through `splitmix64` guarantees a well-mixed, effectively
/// never-zero starting register for any input, including zero.
#[must_use]
pub fn xorshift64_new(seed: u64) -> XorShift64 {
    let mut s = seed;
    let mut state = splitmix64(&mut s);
    // Guard against the astronomically unlikely all-zero diffusion output so
    // the recurrence can never start at its fixed point.
    if state == 0 {
        state = 0x9E37_79B9_7F4A_7C15;
    }
    XorShift64 { state }
}

impl XorShift64 {
    /// Advances the register and returns the next `u64` in the sequence.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns the current internal register value without advancing it.
    #[must_use]
    pub fn state(&self) -> u64 {
        self.state
    }
}

/// Marsaglia's classic `xorshift128` generator over four `u32` words.
///
/// Maintains a shift-register of four words; each draw rotates the words and
/// folds the tail word through the `(11, 8, 19)` recurrence, giving a period
/// of `2^128 - 1` from compact state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct XorShift128 {
    s: [u32; 4],
}

/// Constructs a [`XorShift128`] by filling all four words from
/// [`splitmix64`].
///
/// The four words are drawn from successive `splitmix64` outputs of `seed`,
/// which ensures the combined 128-bit state is non-zero (the sole forbidden
/// state) for any seed.
#[must_use]
pub fn xorshift128_new(seed: u64) -> XorShift128 {
    let mut s = seed;
    let mut words = [0u32; 4];
    for word in words.iter_mut() {
        let mixed = splitmix64(&mut s);
        // Take the high 32 bits, which carry the strongest `splitmix64` mixing.
        *word = (mixed >> 32) as u32;
    }
    // Guarantee the 128-bit state is not entirely zero.
    if words == [0u32; 4] {
        words[0] = XORSHIFT32_FALLBACK;
    }
    XorShift128 { s: words }
}

impl XorShift128 {
    /// Advances the four-word register and returns the next `u32`.
    pub fn next_u32(&mut self) -> u32 {
        let mut t = self.s[3];
        let s0 = self.s[0];
        self.s[3] = self.s[2];
        self.s[2] = self.s[1];
        self.s[1] = s0;
        t ^= t << 11;
        t ^= t >> 8;
        let next = t ^ s0 ^ (s0 >> 19);
        self.s[0] = next;
        next
    }

    /// Returns a copy of the four-word internal register without advancing it.
    #[must_use]
    pub fn state(&self) -> [u32; 4] {
        self.s
    }
}

/// Maps a `u32` uniformly into the half-open unit interval `[0, 1)`.
///
/// The eight low (`LSB`) bits are discarded, leaving a 24-bit integer that a
/// `f32` can represent exactly; that value is scaled by the constant
/// reciprocal of `2^24` with a single multiply. The result therefore always
/// satisfies `(0.0..1.0).contains(&x)`: an input of `0` maps to `0.0`, and
/// `u32::MAX` maps to the largest representable value strictly below `1.0`.
/// No division, rounding, or transcendental operation is performed.
#[must_use]
pub fn u32_to_unit_f32(x: u32) -> f32 {
    (x >> 8) as f32 * INV_TWO_POW_24
}

/// Draws the next `f32` uniformly in `[0, 1)` from a [`XorShift32`].
///
/// Equivalent to passing the engine's next `u32` through [`u32_to_unit_f32`],
/// so the result always satisfies `(0.0..1.0).contains(&x)`.
pub fn next_f32(rng: &mut XorShift32) -> f32 {
    u32_to_unit_f32(rng.next_u32())
}

/// Draws the next `u32` uniformly in the half-open range `[lo, hi)`.
///
/// Precondition: `lo < hi`. The result is `lo` plus the engine's next
/// `u32` reduced modulo the span `hi - lo`. Modulo reduction introduces a
/// slight non-uniformity (modulo bias) when the span does not evenly divide
/// `2^32`; the bias is negligible for spans far smaller than `2^32` and is
/// accepted here in exchange for a branch-free single-draw implementation. When
/// `hi - lo == 1` the span is `1` and the function always returns `lo`.
pub fn next_range_u32(rng: &mut XorShift32, lo: u32, hi: u32) -> u32 {
    debug_assert!(lo < hi, "next_range_u32 requires lo < hi");
    let span = hi - lo;
    lo + (rng.next_u32() % span)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_same_seed_same_sequence() {
        let mut a = 0x1234_5678_9ABC_DEF0u64;
        let mut b = 0x1234_5678_9ABC_DEF0u64;
        for _ in 0..64 {
            assert_eq!(splitmix64(&mut a), splitmix64(&mut b));
        }
    }

    #[test]
    fn splitmix64_different_seeds_differ() {
        let mut a = 1u64;
        let mut b = 2u64;
        assert_ne!(splitmix64(&mut a), splitmix64(&mut b));
    }

    #[test]
    fn splitmix64_advances_state() {
        let mut s = 0u64;
        let before = s;
        let _ = splitmix64(&mut s);
        assert_ne!(s, before);
    }

    #[test]
    fn splitmix64_zero_seed_is_nonzero_output() {
        let mut s = 0u64;
        assert_ne!(splitmix64(&mut s), 0);
    }

    #[test]
    fn splitmix64_stream_not_all_equal() {
        let mut s = 42u64;
        let first = splitmix64(&mut s);
        let mut all_equal = true;
        for _ in 0..32 {
            if splitmix64(&mut s) != first {
                all_equal = false;
            }
        }
        assert!(!all_equal);
    }

    #[test]
    fn xorshift32_same_seed_same_sequence() {
        let mut a = xorshift32_new(0xDEAD_BEEF);
        let mut b = xorshift32_new(0xDEAD_BEEF);
        for _ in 0..128 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn xorshift32_first_outputs_self_consistent() {
        let mut a = xorshift32_new(1);
        let mut b = xorshift32_new(1);
        let seq_a = [
            a.next_u32(),
            a.next_u32(),
            a.next_u32(),
            a.next_u32(),
            a.next_u32(),
        ];
        let seq_b = [
            b.next_u32(),
            b.next_u32(),
            b.next_u32(),
            b.next_u32(),
            b.next_u32(),
        ];
        assert_eq!(seq_a, seq_b);
    }

    #[test]
    fn xorshift32_zero_seed_uses_fallback() {
        let mut rng = xorshift32_new(0);
        assert_ne!(rng.state(), 0);
        assert_ne!(rng.next_u32(), 0);
    }

    #[test]
    fn xorshift32_not_degenerate() {
        let mut rng = xorshift32_new(7);
        let first = rng.next_u32();
        let mut all_equal = true;
        for _ in 0..64 {
            if rng.next_u32() != first {
                all_equal = false;
            }
        }
        assert!(!all_equal);
    }

    #[test]
    fn xorshift32_different_seeds_differ() {
        let mut a = xorshift32_new(1);
        let mut b = xorshift32_new(2);
        assert_ne!(a.next_u32(), b.next_u32());
    }

    #[test]
    fn xorshift64_same_seed_same_sequence() {
        let mut a = xorshift64_new(0xCAFE_F00D);
        let mut b = xorshift64_new(0xCAFE_F00D);
        for _ in 0..128 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn xorshift64_first_outputs_self_consistent() {
        let mut a = xorshift64_new(99);
        let mut b = xorshift64_new(99);
        for _ in 0..5 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn xorshift64_zero_seed_is_nondegenerate() {
        let mut rng = xorshift64_new(0);
        assert_ne!(rng.state(), 0);
        let first = rng.next_u64();
        let mut all_equal = true;
        for _ in 0..64 {
            if rng.next_u64() != first {
                all_equal = false;
            }
        }
        assert!(!all_equal);
    }

    #[test]
    fn xorshift64_different_seeds_differ() {
        let mut a = xorshift64_new(10);
        let mut b = xorshift64_new(20);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn xorshift128_same_seed_same_sequence() {
        let mut a = xorshift128_new(0x0BAD_C0DE);
        let mut b = xorshift128_new(0x0BAD_C0DE);
        for _ in 0..128 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn xorshift128_first_outputs_self_consistent() {
        let mut a = xorshift128_new(2024);
        let mut b = xorshift128_new(2024);
        for _ in 0..5 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn xorshift128_state_is_nonzero() {
        let rng = xorshift128_new(0);
        assert_ne!(rng.state(), [0u32; 4]);
    }

    #[test]
    fn xorshift128_not_degenerate() {
        let mut rng = xorshift128_new(3);
        let first = rng.next_u32();
        let mut all_equal = true;
        for _ in 0..64 {
            if rng.next_u32() != first {
                all_equal = false;
            }
        }
        assert!(!all_equal);
    }

    #[test]
    fn xorshift128_different_seeds_differ() {
        let mut a = xorshift128_new(100);
        let mut b = xorshift128_new(200);
        assert_ne!(a.next_u32(), b.next_u32());
    }

    #[test]
    fn unit_f32_zero_maps_to_zero() {
        let v = u32_to_unit_f32(0);
        assert!(v.abs() < 1.0e-9);
        assert!((0.0..1.0).contains(&v));
    }

    #[test]
    fn unit_f32_max_below_one() {
        let v = u32_to_unit_f32(u32::MAX);
        assert!(v < 1.0);
        assert!((0.0..1.0).contains(&v));
    }

    #[test]
    fn unit_f32_always_in_unit_interval() {
        let mut rng = xorshift32_new(0x5151_5151);
        for _ in 0..10_000 {
            let v = u32_to_unit_f32(rng.next_u32());
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn unit_f32_is_monotonic_nondecreasing() {
        let samples = [0u32, 256, 1 << 8, 1 << 16, 1 << 24, 1 << 28, u32::MAX];
        let mut prev = u32_to_unit_f32(samples[0]);
        for &x in samples.iter().skip(1) {
            let cur = u32_to_unit_f32(x);
            assert!(cur >= prev);
            prev = cur;
        }
    }

    #[test]
    fn next_f32_samples_in_unit_interval() {
        let mut rng = xorshift32_new(0xABCD_1234);
        for _ in 0..10_000 {
            let v = next_f32(&mut rng);
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn next_f32_rough_uniformity_four_buckets() {
        let mut rng = xorshift32_new(0x9999_1111);
        let mut buckets = [0u32; 4];
        for _ in 0..10_000 {
            let v = next_f32(&mut rng);
            let idx = ((v * 4.0) as usize).min(3);
            buckets[idx] += 1;
        }
        for count in buckets.iter() {
            assert!(*count > 0);
        }
    }

    #[test]
    fn next_f32_same_seed_same_floats() {
        let mut a = xorshift32_new(555);
        let mut b = xorshift32_new(555);
        for _ in 0..100 {
            let va = next_f32(&mut a);
            let vb = next_f32(&mut b);
            // Identical integer streams give bit-identical scaled floats; use a
            // range check rather than direct float equality.
            assert!((va - vb).abs() < 1.0e-12);
        }
    }

    #[test]
    fn next_range_stays_within_bounds() {
        let mut rng = xorshift32_new(0x2468_ACE0);
        for _ in 0..10_000 {
            let v = next_range_u32(&mut rng, 10, 20);
            assert!((10..20).contains(&v));
        }
    }

    #[test]
    fn next_range_span_one_returns_lo() {
        let mut rng = xorshift32_new(1);
        for _ in 0..1_000 {
            assert_eq!(next_range_u32(&mut rng, 42, 43), 42);
        }
    }

    #[test]
    fn next_range_wide_span_within_bounds() {
        let mut rng = xorshift32_new(0xFEED_FACE);
        let lo = 1_000u32;
        let hi = 1_000_000u32;
        for _ in 0..10_000 {
            let v = next_range_u32(&mut rng, lo, hi);
            assert!((lo..hi).contains(&v));
        }
    }

    #[test]
    fn next_range_covers_low_and_high_ends() {
        let mut rng = xorshift32_new(0x1357_9BDF);
        let mut saw_lo = false;
        let mut saw_near_hi = false;
        for _ in 0..10_000 {
            let v = next_range_u32(&mut rng, 0, 8);
            if v == 0 {
                saw_lo = true;
            }
            if v == 7 {
                saw_near_hi = true;
            }
        }
        assert!(saw_lo);
        assert!(saw_near_hi);
    }

    #[test]
    fn xorshift32_clone_advances_independently() {
        let mut a = xorshift32_new(0x1111_2222);
        let mut b = a;
        let a1 = a.next_u32();
        let b1 = b.next_u32();
        assert_eq!(a1, b1);
        let _ = a.next_u32();
        // b has not advanced past its first draw, so their states now differ.
        assert_ne!(a.state(), b.state());
    }

    #[test]
    fn xorshift64_copy_reproduces_sequence() {
        let mut a = xorshift64_new(0x7777);
        let snapshot = a;
        let seq_a = [a.next_u64(), a.next_u64(), a.next_u64()];
        let mut b = snapshot;
        let seq_b = [b.next_u64(), b.next_u64(), b.next_u64()];
        assert_eq!(seq_a, seq_b);
    }

    #[test]
    fn xorshift128_clone_reproduces_sequence() {
        let mut a = xorshift128_new(0x3333);
        let b_snapshot = a;
        let seq_a = [a.next_u32(), a.next_u32(), a.next_u32(), a.next_u32()];
        let mut b = b_snapshot;
        let seq_b = [b.next_u32(), b.next_u32(), b.next_u32(), b.next_u32()];
        assert_eq!(seq_a, seq_b);
    }

    #[test]
    fn equal_engines_compare_equal() {
        let a = xorshift32_new(12_345);
        let b = xorshift32_new(12_345);
        assert_eq!(a, b);
        let mut cc = a;
        let _ = cc.next_u32();
        assert_ne!(a, cc);
    }

    #[test]
    fn engines_are_hashable() {
        use core::hash::{Hash, Hasher};
        // A trivial manual hasher to exercise the derived Hash impl.
        struct CountHasher(u64);
        impl Hasher for CountHasher {
            fn finish(&self) -> u64 {
                self.0
            }
            fn write(&mut self, bytes: &[u8]) {
                for &b in bytes {
                    self.0 = self.0.wrapping_mul(31).wrapping_add(u64::from(b));
                }
            }
        }
        let a = xorshift32_new(9);
        let b = xorshift32_new(9);
        let mut ha = CountHasher(0);
        let mut hb = CountHasher(0);
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }
}
