//! Marsaglia's multiply-with-carry pseudo-random number generator
//! (`MWC32`), a compact stateful `PRNG` used for reproducible particle
//! randomness (design: stochastic effects).
//!
//! A multiply-with-carry generator (`MWC`) threads a small hidden register
//! from one draw to the next. The state is a pair of `32`-bit words: a value
//! `x` and a carry `c`. Each step forms the `64`-bit product
//! `t = A * x + c` for a fixed odd multiplier `A`, keeps the low `32` bits as
//! the new `x`, and promotes the high `32` bits to the new carry `c`. The low
//! word is returned as the `u32` result. This is a classic lagged generator:
//! the full period is governed by the order of the base `2^32` modulo the
//! prime `A * 2^32 - 1`, which for a well-chosen `A` is enormous.
//!
//! The multiplier used here is `A = 0xFFFF_DA61`. It is one of Marsaglia's
//! recommended constants: `A * 2^32 - 1` is prime and `2` is a primitive root,
//! so the stream has a long period and good equidistribution for a generator
//! this small. The generator is non-cryptographic; its state is trivially
//! recoverable from a few outputs, so it must never be used for security, key
//! material, or anywhere predictability could be exploited. It exists purely
//! for reproducible, high-throughput simulation randomness.
//!
//! Every operation in generation is integer arithmetic performed with
//! wrapping (modulo-`2^64`) semantics plus fixed shifts; the `64`-bit product
//! `A * x + c` can never overflow because `A`, `x`, and `c` are each at most
//! `2^32 - 1`. There is no floating point, no division, and no transcendental
//! function anywhere in generation, so results are identical on every target.
//!
//! Seeding uses a single round of the `SplitMix64` finalizer so that even a
//! tiny seed such as `0` or `1` expands into a well-scrambled `(x, c)` pair.
//! The low `32` bits of the mixed word become `x` and the high `32` bits
//! become `c`.
//!
//! Scope and boundaries: this module is deliberately narrow and completely
//! self-contained. It is the `MWC32` recurrence plus a `SplitMix64` seeding
//! helper, and it shares no code with the `squares` or `xorshift` engines
//! elsewhere in this crate. Do not conflate them.

/// The multiply-with-carry multiplier `A`.
///
/// This is Marsaglia's recommended `32`-bit constant for which
/// `A * 2^32 - 1` is prime with `2` as a primitive root, giving the generator
/// its long period. It is widened to `u64` at each step via [`u64::from`] so
/// the product `A * x + c` is computed losslessly.
const MWC32_MULTIPLIER: u32 = 0xFFFF_DA61;

/// The `SplitMix64` additive constant (the `64`-bit golden ratio).
///
/// Seeding advances the raw seed by this odd increment before running the
/// finalizer, matching the canonical `SplitMix64` construction.
const SPLITMIX64_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// `SplitMix64` finalizer: a strong integer bit-mixer with good avalanche.
///
/// This takes an arbitrary `u64` seed, folds in the golden-ratio increment,
/// and scrambles every input bit across the output using only shifts, xors,
/// and odd multiplies. It is a pure function, so it is trivially reproducible
/// on any backend. It is used only to expand a seed into a full `MWC32` state.
///
/// # Examples
///
/// ```
/// use prism_render_architecture::particle::mwc32::splitmix64;
///
/// assert!(splitmix64(1) == 0x910A_2DEC_8902_5CC1);
/// ```
#[inline]
#[must_use]
pub const fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(SPLITMIX64_GAMMA);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A stateful multiply-with-carry generator over the `MWC32` recurrence.
///
/// `Mwc32` holds the two `32`-bit state words `x` and `c`. Each call to
/// [`Mwc32::next_u32`] applies one step of the recurrence and returns the new
/// low word. Two generators built from the same state (or the same seed)
/// always replay the identical sequence, which is exactly what a deterministic
/// particle system needs for bit-for-bit frame replays.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Mwc32 {
    /// The current value word; also the most recently returned output.
    x: u32,
    /// The current carry word (the high half of the last `64`-bit product).
    c: u32,
}

impl Mwc32 {
    /// Build a generator directly from an explicit `(x, c)` state.
    ///
    /// This is the lowest-level constructor; it performs no mixing. Use it to
    /// resume a captured stream position or to drive the generator from a
    /// state produced elsewhere.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_render_architecture::particle::mwc32::Mwc32;
    ///
    /// let rng = Mwc32::from_state(1, 2);
    /// assert!(rng.state() == (1, 2));
    /// ```
    #[inline]
    #[must_use]
    pub const fn from_state(x: u32, c: u32) -> Self {
        Self { x, c }
    }

    /// Build a generator by expanding `seed` through [`splitmix64`].
    ///
    /// The mixed word's low `32` bits become `x` and its high `32` bits become
    /// `c`, so even trivial seeds yield a well-scrambled starting state.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_render_architecture::particle::mwc32::Mwc32;
    ///
    /// let rng = Mwc32::from_seed(1);
    /// assert!(rng.state() == (0x8902_5CC1, 0x910A_2DEC));
    /// ```
    #[inline]
    #[must_use]
    pub const fn from_seed(seed: u64) -> Self {
        let v = splitmix64(seed);
        let x = (v & 0xFFFF_FFFF) as u32;
        let c = (v >> 32) as u32;
        Self { x, c }
    }

    /// Return the current `(x, c)` state words.
    ///
    /// The first element is the value word (equal to the most recently
    /// returned output), and the second is the carry word.
    #[inline]
    #[must_use]
    pub const fn state(&self) -> (u32, u32) {
        (self.x, self.c)
    }

    /// Draw the next `32`-bit value and advance the state by one step.
    ///
    /// Computes the `64`-bit product `A * x + c` (widening with
    /// [`u64::from`] so it is lossless), stores its low word as the new `x`
    /// and its high word as the new carry `c`, then returns the new `x`.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let t = u64::from(MWC32_MULTIPLIER) * u64::from(self.x) + u64::from(self.c);
        self.x = t as u32;
        self.c = (t >> 32) as u32;
        self.x
    }

    /// Draw `N` consecutive values into a fixed-length array.
    ///
    /// This is exactly `N` successive [`Mwc32::next_u32`] calls collected in
    /// order, so `next_array::<N>()` is equivalent to calling `next_u32`
    /// `N` times and gathering the results. `N` is a compile-time constant, so
    /// no heap allocation is involved.
    #[inline]
    #[must_use]
    pub fn next_array<const N: usize>(&mut self) -> [u32; N] {
        core::array::from_fn(|_| self.next_u32())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- SplitMix64 finalizer reference ---

    #[test]
    fn splitmix64_hard_vector_seed1() {
        assert!(splitmix64(1) == 0x910A_2DEC_8902_5CC1);
    }

    #[test]
    fn splitmix64_is_deterministic() {
        assert!(splitmix64(42) == splitmix64(42));
    }

    #[test]
    fn splitmix64_distinct_seeds_differ() {
        assert!(splitmix64(0) != splitmix64(1));
    }

    #[test]
    fn splitmix64_seed0_is_nonzero() {
        assert!(splitmix64(0) != 0);
    }

    // --- from_state hard reference vectors (x = 1, c = 2) ---

    #[test]
    fn from_state_out0() {
        let mut rng = Mwc32::from_state(1, 2);
        assert!(rng.next_u32() == 0xFFFF_DA63);
    }

    #[test]
    fn from_state_out1() {
        let mut rng = Mwc32::from_state(1, 2);
        rng.next_u32();
        assert!(rng.next_u32() == 0x0587_0D83);
    }

    #[test]
    fn from_state_out2() {
        let mut rng = Mwc32::from_state(1, 2);
        for _ in 0..2 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == 0x0C2A_6167);
    }

    #[test]
    fn from_state_out3() {
        let mut rng = Mwc32::from_state(1, 2);
        for _ in 0..3 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == 0x5720_AABB);
    }

    #[test]
    fn from_state_out4() {
        let mut rng = Mwc32::from_state(1, 2);
        for _ in 0..4 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == 0x3633_4E78);
    }

    #[test]
    fn from_state_out5() {
        let mut rng = Mwc32::from_state(1, 2);
        for _ in 0..5 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == 0x42EB_8965);
    }

    #[test]
    fn from_state_out6_as_array() {
        let mut rng = Mwc32::from_state(1, 2);
        let got: [u32; 6] = rng.next_array();
        assert!(
            got == [
                0xFFFF_DA63,
                0x0587_0D83,
                0x0C2A_6167,
                0x5720_AABB,
                0x3633_4E78,
                0x42EB_8965,
            ]
        );
    }

    #[test]
    fn from_state_state_after_six_draws() {
        let mut rng = Mwc32::from_state(1, 2);
        for _ in 0..6 {
            rng.next_u32();
        }
        assert!(rng.state() == (0x42EB_8965, 0x3633_4681));
    }

    // --- from_seed hard reference vectors (seed = 1) ---

    #[test]
    fn from_seed1_initial_x() {
        let rng = Mwc32::from_seed(1);
        assert!(rng.state().0 == 0x8902_5CC1);
    }

    #[test]
    fn from_seed1_initial_c() {
        let rng = Mwc32::from_seed(1);
        assert!(rng.state().1 == 0x910A_2DEC);
    }

    #[test]
    fn from_seed1_out0() {
        let mut rng = Mwc32::from_seed(1);
        assert!(rng.next_u32() == 0x212A_AD0D);
    }

    #[test]
    fn from_seed1_out1() {
        let mut rng = Mwc32::from_seed(1);
        rng.next_u32();
        assert!(rng.next_u32() == 0xC47D_EC8C);
    }

    #[test]
    fn from_seed1_out2() {
        let mut rng = Mwc32::from_seed(1);
        for _ in 0..2 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == 0xE3C4_8139);
    }

    #[test]
    fn from_seed1_out3() {
        let mut rng = Mwc32::from_seed(1);
        for _ in 0..3 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == 0xE6C4_5044);
    }

    #[test]
    fn from_seed1_out4() {
        let mut rng = Mwc32::from_seed(1);
        for _ in 0..4 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == 0x303C_B184);
    }

    #[test]
    fn from_seed1_out5() {
        let mut rng = Mwc32::from_seed(1);
        for _ in 0..5 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == 0x2B69_D95F);
    }

    #[test]
    fn from_seed1_out6_as_array() {
        let mut rng = Mwc32::from_seed(1);
        let got: [u32; 6] = rng.next_array();
        assert!(
            got == [
                0x212A_AD0D,
                0xC47D_EC8C,
                0xE3C4_8139,
                0xE6C4_5044,
                0x303C_B184,
                0x2B69_D95F,
            ]
        );
    }

    // --- from_state round-trips the given state ---

    #[test]
    fn from_state_preserves_x() {
        let rng = Mwc32::from_state(0x1234_5678, 0x9ABC_DEF0);
        assert!(rng.state().0 == 0x1234_5678);
    }

    #[test]
    fn from_state_preserves_c() {
        let rng = Mwc32::from_state(0x1234_5678, 0x9ABC_DEF0);
        assert!(rng.state().1 == 0x9ABC_DEF0);
    }

    #[test]
    fn from_state_round_trip_tuple() {
        let rng = Mwc32::from_state(7, 11);
        assert!(rng.state() == (7, 11));
    }

    // --- Determinism and reproducibility ---

    #[test]
    fn same_state_replays_identically() {
        let mut a = Mwc32::from_state(1, 2);
        let first: [u32; 32] = a.next_array();
        let mut b = Mwc32::from_state(1, 2);
        let second: [u32; 32] = b.next_array();
        assert!(first == second);
    }

    #[test]
    fn same_seed_replays_identically() {
        let mut a = Mwc32::from_seed(0xDEAD_BEEF);
        let first: [u32; 32] = a.next_array();
        let mut b = Mwc32::from_seed(0xDEAD_BEEF);
        let second: [u32; 32] = b.next_array();
        assert!(first == second);
    }

    #[test]
    fn determinism_over_a_run() {
        let mut a = Mwc32::from_seed(123);
        let mut b = Mwc32::from_seed(123);
        for _ in 0..256 {
            assert!(a.next_u32() == b.next_u32());
        }
    }

    // --- next_array matches repeated next_u32 ---

    #[test]
    fn next_array_matches_repeated_next_u32() {
        let mut a = Mwc32::from_seed(99);
        let arr: [u32; 8] = a.next_array();
        let mut b = Mwc32::from_seed(99);
        let manual: [u32; 8] = core::array::from_fn(|_| b.next_u32());
        assert!(arr == manual);
    }

    #[test]
    fn next_array_zero_length_is_empty() {
        let mut rng = Mwc32::from_seed(5);
        let arr: [u32; 0] = rng.next_array();
        assert!(arr.len() == 0);
        // The state must be untouched by a zero-length draw.
        assert!(rng.state() == Mwc32::from_seed(5).state());
    }

    #[test]
    fn next_array_single_matches_next_u32() {
        let mut a = Mwc32::from_state(1, 2);
        let arr: [u32; 1] = a.next_array();
        let mut b = Mwc32::from_state(1, 2);
        assert!(arr[0] == b.next_u32());
    }

    // --- Different seeds / states diverge ---

    #[test]
    fn different_seeds_produce_different_streams() {
        let mut a = Mwc32::from_seed(1);
        let mut b = Mwc32::from_seed(2);
        let sa: [u32; 16] = a.next_array();
        let sb: [u32; 16] = b.next_array();
        assert!(sa != sb);
    }

    #[test]
    fn different_states_produce_different_streams() {
        let mut a = Mwc32::from_state(1, 2);
        let mut b = Mwc32::from_state(3, 4);
        let sa: [u32; 16] = a.next_array();
        let sb: [u32; 16] = b.next_array();
        assert!(sa != sb);
    }

    // --- State advances and tracks the output ---

    #[test]
    fn state_x_tracks_last_output() {
        let mut rng = Mwc32::from_state(1, 2);
        let out = rng.next_u32();
        assert!(rng.state().0 == out);
    }

    #[test]
    fn state_changes_after_draw() {
        let mut rng = Mwc32::from_state(1, 2);
        let before = rng.state();
        rng.next_u32();
        assert!(rng.state() != before);
    }

    // --- Copy semantics ---

    #[test]
    fn copy_duplicates_position() {
        let mut rng = Mwc32::from_seed(7);
        rng.next_u32();
        rng.next_u32();
        let mut copy = rng;
        assert!(rng.next_u32() == copy.next_u32());
    }

    #[test]
    fn copy_is_independent_after_advance() {
        let mut rng = Mwc32::from_seed(7);
        let copy = rng;
        rng.next_u32();
        // Advancing the original must not move the copy.
        assert!(copy.state() == Mwc32::from_seed(7).state());
    }

    #[test]
    fn equality_tracks_state() {
        let a = Mwc32::from_state(1, 2);
        let b = Mwc32::from_state(1, 2);
        assert!(a == b);
        let mut c = Mwc32::from_state(1, 2);
        c.next_u32();
        assert!(a != c);
    }

    // --- Output is not degenerate ---

    #[test]
    fn output_not_all_zero() {
        let mut rng = Mwc32::from_seed(0);
        let mut any_nonzero = false;
        for _ in 0..256 {
            if rng.next_u32() != 0 {
                any_nonzero = true;
            }
        }
        assert!(any_nonzero);
    }

    #[test]
    fn output_not_all_equal() {
        let mut rng = Mwc32::from_seed(0);
        let first = rng.next_u32();
        let mut any_different = false;
        for _ in 0..256 {
            if rng.next_u32() != first {
                any_different = true;
            }
        }
        assert!(any_different);
    }

    #[test]
    fn high_and_low_bits_both_toggle() {
        let mut rng = Mwc32::from_seed(0xABCD_1234);
        let mut or_acc = 0u32;
        let mut and_acc = u32::MAX;
        for _ in 0..512 {
            let v = rng.next_u32();
            or_acc |= v;
            and_acc &= v;
        }
        // Across many draws, every bit position should see both a 0 and a 1.
        assert!(or_acc == u32::MAX);
        assert!(and_acc == 0);
    }

    // --- from_seed and from_state agree on the same starting state ---

    #[test]
    fn from_seed_matches_equivalent_from_state() {
        let seeded = Mwc32::from_seed(1);
        let (x, c) = seeded.state();
        let mut a = Mwc32::from_seed(1);
        let mut b = Mwc32::from_state(x, c);
        let sa: [u32; 8] = a.next_array();
        let sb: [u32; 8] = b.next_array();
        assert!(sa == sb);
    }
}
