//! Widynski's `squares` counter-based pseudo-random number generator
//! (`squares32`), used for reproducible, massively parallel particle
//! randomness (design: stochastic effects).
//!
//! A *counter-based* `RNG` (`CBRNG`) differs fundamentally from a classic
//! stateful `PRNG` such as a linear-feedback or `xorshift` engine: instead of
//! threading a hidden register from one draw to the next, it treats the stream
//! index itself as the input. The i-th output is a pure, stateless function
//! `squares32(i, key)` of a `64`-bit counter `ctr` and a fixed `64`-bit `key`.
//! This is exactly what a deterministic `GPU`/`CPU` particle system wants: any
//! worker, tile, or emitter can jump straight to the random value for a given
//! particle index without replaying the whole sequence, and the same frame
//! replays bit-for-bit identically.
//!
//! The algorithm is Bernard Widynski's `squares` generator. Each draw mixes a
//! `counter`-derived seed through four rounds; every round squares the running
//! value, folds in one of two key-derived addends, and rotates the `64`-bit
//! word by `32` bits via `(x >> 32) | (x << 32)`. The top `32` bits of the
//! final round are returned as the `u32` result. Squaring is what gives the
//! generator its avalanche: a single input bit influences essentially every
//! output bit after the repeated multiply-and-rotate steps.
//!
//! Every operation here is `64`-bit integer arithmetic performed with wrapping
//! (modulo-`2^64`) semantics plus fixed shifts and bitwise-or. There is no
//! floating point, no division, and no transcendental function anywhere in
//! generation, so results are identical on every target.
//!
//! Scope and boundaries: this module is deliberately narrow and completely
//! self-contained. It is the `32`-bit `squares32` function plus a thin stateful
//! wrapper that walks the counter, and it shares no code with the `xoshiro` or
//! `xorshift` engines elsewhere in this crate. Do not conflate them.
//!
//! This generator is fast and non-cryptographic. A `squares` stream is
//! predictable once the `key` is known, so it must never be used for security,
//! key material, or anywhere an adversary could exploit predictability. It
//! exists purely for reproducible, high-throughput simulation randomness.

/// Compute the `squares32` output for counter `ctr` under the `64`-bit `key`.
///
/// This is the canonical four-round Widynski `squares` generator. It is a pure
/// function of its inputs: the same `(ctr, key)` pair always yields the same
/// `u32`, which is the defining property of a counter-based `RNG`. All
/// arithmetic is `64`-bit wrapping multiplication and addition combined with
/// the `32`-bit word rotation `(x >> 32) | (x << 32)`; the returned value is
/// the high `32` bits of the final round, so the `as u32` truncation is exact
/// and lossless for the intended bits.
///
/// # Examples
///
/// ```
/// use prism_render_architecture::particle::squares_rng::squares32;
///
/// let key = 0x9E37_79B9_7F4A_7C15;
/// assert_eq!(squares32(0, key), 0x03D1_0999);
/// assert_eq!(squares32(1, key), 0x4BB2_3CCC);
/// ```
#[inline]
#[must_use]
pub fn squares32(ctr: u64, key: u64) -> u32 {
    let mut x = ctr.wrapping_mul(key);
    let y = x;
    let z = y.wrapping_add(key);

    // Round 1: square, fold in y, rotate the 64-bit word by 32 bits.
    x = x.wrapping_mul(x).wrapping_add(y);
    x = x.rotate_left(32);

    // Round 2: square, fold in z, rotate.
    x = x.wrapping_mul(x).wrapping_add(z);
    x = x.rotate_left(32);

    // Round 3: square, fold in y, rotate.
    x = x.wrapping_mul(x).wrapping_add(y);
    x = x.rotate_left(32);

    // Round 4: square, fold in z, and return the high 32 bits.
    (x.wrapping_mul(x).wrapping_add(z) >> 32) as u32
}

/// A stateful wrapper over [`squares32`] that walks an internal counter.
///
/// `SquaresRng` turns the stateless counter-based `RNG` into a conventional
/// draw-on-demand stream: it holds a fixed `key` and a monotonically
/// increasing `ctr` that starts at zero. Each call to [`SquaresRng::next_u32`]
/// evaluates `squares32(ctr, key)` and then advances `ctr` by one, so the n-th
/// draw of a freshly constructed generator equals `squares32(n, key)` exactly.
/// Because the stream position is just the counter, two generators built from
/// the same key always replay the identical sequence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SquaresRng {
    /// The next counter value to feed into [`squares32`].
    ctr: u64,
    /// The fixed key that selects this stream.
    key: u64,
}

impl SquaresRng {
    /// Create a generator for the stream selected by `key`, with the counter
    /// starting at zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_render_architecture::particle::squares_rng::SquaresRng;
    ///
    /// let mut rng = SquaresRng::new(0x9E37_79B9_7F4A_7C15);
    /// assert_eq!(rng.next_u32(), 0x03D1_0999);
    /// ```
    #[inline]
    #[must_use]
    pub fn new(key: u64) -> Self {
        Self { ctr: 0, key }
    }

    /// The fixed `key` that selects this generator's stream.
    #[inline]
    #[must_use]
    pub fn key(&self) -> u64 {
        self.key
    }

    /// The counter value that the next [`SquaresRng::next_u32`] call consumes.
    #[inline]
    #[must_use]
    pub fn counter(&self) -> u64 {
        self.ctr
    }

    /// Draw the next `32`-bit value and advance the counter by one.
    ///
    /// Returns `squares32(ctr, key)` for the current counter, then increments
    /// the counter with wrapping semantics so the stream never panics even
    /// after `2^64` draws.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let out = squares32(self.ctr, self.key);
        self.ctr = self.ctr.wrapping_add(1);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeSet;

    /// The golden-ratio key used across the hard reference vectors.
    const KEY: u64 = 0x9E37_79B9_7F4A_7C15;

    // --- Hard reference vectors (verified against an independent impl) ---

    #[test]
    fn hard_vector_ctr0() {
        assert_eq!(squares32(0, KEY), 0x03D1_0999);
    }

    #[test]
    fn hard_vector_ctr1() {
        assert_eq!(squares32(1, KEY), 0x4BB2_3CCC);
    }

    #[test]
    fn hard_vector_ctr2() {
        assert_eq!(squares32(2, KEY), 0x3EAE_CE6C);
    }

    #[test]
    fn hard_vector_ctr3() {
        assert_eq!(squares32(3, KEY), 0x434B_336E);
    }

    #[test]
    fn hard_vectors_as_array() {
        let got: [u32; 4] = core::array::from_fn(|i| squares32(i as u64, KEY));
        assert_eq!(got, [0x03D1_0999, 0x4BB2_3CCC, 0x3EAE_CE6C, 0x434B_336E]);
    }

    // --- Determinism of the pure function ---

    #[test]
    fn same_inputs_same_output_ctr0() {
        assert_eq!(squares32(0, KEY), squares32(0, KEY));
    }

    #[test]
    fn same_inputs_same_output_arbitrary() {
        assert_eq!(squares32(12_345, KEY), squares32(12_345, KEY));
    }

    #[test]
    fn same_inputs_same_output_large_ctr() {
        let c = 0xDEAD_BEEF_0000_1234;
        assert_eq!(squares32(c, KEY), squares32(c, KEY));
    }

    #[test]
    fn determinism_over_a_run() {
        for ctr in 0..256u64 {
            assert_eq!(squares32(ctr, KEY), squares32(ctr, KEY));
        }
    }

    // --- Different keys produce different output ---

    #[test]
    fn different_key_differs_ctr0() {
        assert_ne!(squares32(0, KEY), squares32(0, KEY ^ 0x1));
    }

    #[test]
    fn different_key_differs_ctr1() {
        assert_ne!(squares32(1, KEY), squares32(1, 0x1234_5678_9ABC_DEF0));
    }

    #[test]
    fn different_key_differs_many() {
        let other = 0x0123_4567_89AB_CDEF;
        let mut differences = 0u32;
        for ctr in 0..64u64 {
            if squares32(ctr, KEY) != squares32(ctr, other) {
                differences += 1;
            }
        }
        // Essentially all draws should differ between two unrelated keys.
        assert!(differences >= 60);
    }

    // --- Different counters produce different output (within a key) ---

    #[test]
    fn different_counter_differs() {
        assert_ne!(squares32(0, KEY), squares32(1, KEY));
    }

    #[test]
    fn consecutive_counters_mostly_differ() {
        let mut collisions = 0u32;
        for ctr in 0..255u64 {
            if squares32(ctr, KEY) == squares32(ctr + 1, KEY) {
                collisions += 1;
            }
        }
        assert_eq!(collisions, 0);
    }

    // --- Wrapper matches the pure function stream ---

    #[test]
    fn wrapper_matches_squares32_stream() {
        let mut rng = SquaresRng::new(KEY);
        for ctr in 0..512u64 {
            assert_eq!(rng.next_u32(), squares32(ctr, KEY));
        }
    }

    #[test]
    fn wrapper_first_four_match_hard_vectors() {
        let mut rng = SquaresRng::new(KEY);
        assert_eq!(rng.next_u32(), 0x03D1_0999);
        assert_eq!(rng.next_u32(), 0x4BB2_3CCC);
        assert_eq!(rng.next_u32(), 0x3EAE_CE6C);
        assert_eq!(rng.next_u32(), 0x434B_336E);
    }

    #[test]
    fn wrapper_matches_for_other_key() {
        let key = 0xABCD_1234_5678_9F0E;
        let mut rng = SquaresRng::new(key);
        for ctr in 0..128u64 {
            assert_eq!(rng.next_u32(), squares32(ctr, key));
        }
    }

    // --- Counter increments ---

    #[test]
    fn new_starts_counter_at_zero() {
        let rng = SquaresRng::new(KEY);
        assert_eq!(rng.counter(), 0);
    }

    #[test]
    fn counter_increments_by_one() {
        let mut rng = SquaresRng::new(KEY);
        rng.next_u32();
        assert_eq!(rng.counter(), 1);
    }

    #[test]
    fn counter_increments_monotonically() {
        let mut rng = SquaresRng::new(KEY);
        for expected in 0..100u64 {
            assert_eq!(rng.counter(), expected);
            rng.next_u32();
        }
        assert_eq!(rng.counter(), 100);
    }

    #[test]
    fn counter_advances_exactly_once_per_draw() {
        let mut rng = SquaresRng::new(KEY);
        let before = rng.counter();
        rng.next_u32();
        assert_eq!(rng.counter(), before + 1);
    }

    // --- Key is preserved ---

    #[test]
    fn new_preserves_key() {
        let rng = SquaresRng::new(KEY);
        assert_eq!(rng.key(), KEY);
    }

    #[test]
    fn key_unchanged_after_draws() {
        let mut rng = SquaresRng::new(KEY);
        for _ in 0..32 {
            rng.next_u32();
        }
        assert_eq!(rng.key(), KEY);
    }

    // --- Reproducibility of a fresh stream ---

    #[test]
    fn fresh_generators_replay_identically() {
        let mut a = SquaresRng::new(KEY);
        let first: [u32; 32] = core::array::from_fn(|_| a.next_u32());
        let mut b = SquaresRng::new(KEY);
        let second: [u32; 32] = core::array::from_fn(|_| b.next_u32());
        assert_eq!(first, second);
    }

    #[test]
    fn two_keys_produce_different_streams() {
        let mut a = SquaresRng::new(KEY);
        let mut b = SquaresRng::new(KEY ^ 0xFFFF_FFFF_FFFF_FFFF);
        let sa: [u32; 16] = core::array::from_fn(|_| a.next_u32());
        let sb: [u32; 16] = core::array::from_fn(|_| b.next_u32());
        assert_ne!(sa, sb);
    }

    // --- Output is not degenerate ---

    #[test]
    fn output_not_all_zero() {
        let mut rng = SquaresRng::new(KEY);
        let any_nonzero = (0..256).any(|_| rng.next_u32() != 0);
        assert!(any_nonzero);
    }

    #[test]
    fn output_not_all_equal() {
        let mut rng = SquaresRng::new(KEY);
        let first = rng.next_u32();
        let any_different = (0..256).any(|_| rng.next_u32() != first);
        assert!(any_different);
    }

    #[test]
    fn output_has_many_distinct_values() {
        let mut rng = SquaresRng::new(KEY);
        let mut seen = BTreeSet::new();
        for _ in 0..1024 {
            seen.insert(rng.next_u32());
        }
        // A healthy generator should produce almost entirely unique draws.
        assert!(seen.len() >= 1000);
    }

    #[test]
    fn high_and_low_bits_both_toggle() {
        let mut rng = SquaresRng::new(KEY);
        let mut or_acc = 0u32;
        let mut and_acc = u32::MAX;
        for _ in 0..256 {
            let v = rng.next_u32();
            or_acc |= v;
            and_acc &= v;
        }
        // Across many draws, every bit position should see both a 0 and a 1.
        assert_eq!(or_acc, u32::MAX);
        assert_eq!(and_acc, 0);
    }

    // --- Structural / trait sanity ---

    #[test]
    fn clone_copies_position() {
        let mut rng = SquaresRng::new(KEY);
        rng.next_u32();
        rng.next_u32();
        let mut clone = rng;
        assert_eq!(rng.next_u32(), clone.next_u32());
    }

    #[test]
    fn clone_is_independent_after_copy() {
        let mut rng = SquaresRng::new(KEY);
        let clone = rng;
        rng.next_u32();
        // Advancing the original must not move the copy's counter.
        assert_eq!(clone.counter(), 0);
    }

    #[test]
    fn equality_tracks_state() {
        let a = SquaresRng::new(KEY);
        let b = SquaresRng::new(KEY);
        assert_eq!(a, b);
        let mut c = SquaresRng::new(KEY);
        c.next_u32();
        assert_ne!(a, c);
    }

    #[test]
    fn resuming_from_captured_counter_matches() {
        let mut rng = SquaresRng::new(KEY);
        for _ in 0..10 {
            rng.next_u32();
        }
        let resumed = squares32(rng.counter(), KEY);
        assert_eq!(rng.next_u32(), resumed);
    }

    #[test]
    fn interleaved_keys_are_independent() {
        let mut a = SquaresRng::new(0x1111_1111_1111_1111);
        let mut b = SquaresRng::new(0x2222_2222_2222_2222);
        for ctr in 0..64u64 {
            assert_eq!(a.next_u32(), squares32(ctr, 0x1111_1111_1111_1111));
            assert_eq!(b.next_u32(), squares32(ctr, 0x2222_2222_2222_2222));
        }
    }

    #[test]
    fn large_counter_values_are_stable() {
        let base = 0xFFFF_FFFF_FF00_0000;
        for off in 0..16u64 {
            let c = base + off;
            assert_eq!(squares32(c, KEY), squares32(c, KEY));
        }
    }

    #[test]
    fn distinct_counters_distinct_in_sample() {
        let mut seen = BTreeSet::new();
        let mut collisions = 0u32;
        for ctr in 0..2048u64 {
            if !seen.insert(squares32(ctr, KEY)) {
                collisions += 1;
            }
        }
        // Birthday collisions in 2048 draws of a 32-bit space are rare.
        assert!(collisions <= 2);
    }

    #[test]
    fn key_bit_flip_avalanches_output() {
        // Flipping a single key bit should change roughly half the output bits.
        let flipped = KEY ^ (1u64 << 20);
        let a = squares32(777, KEY);
        let b = squares32(777, flipped);
        let diff = (a ^ b).count_ones();
        assert!(diff >= 8);
    }

    #[test]
    fn counter_bit_flip_avalanches_output() {
        let a = squares32(0x0000_0000_0001_0000, KEY);
        let b = squares32(0x0000_0000_0001_0000 ^ 0x40, KEY);
        let diff = (a ^ b).count_ones();
        assert!(diff >= 8);
    }
}
