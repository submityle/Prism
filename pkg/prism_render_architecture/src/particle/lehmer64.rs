//! Lehmer's `128`-bit multiplicative congruential pseudo-random number
//! generator (`Lehmer64`), used for reproducible, high-throughput particle
//! randomness (design: stochastic effects).
//!
//! A `Lehmer64` engine is a classic stateful `PRNG`: it threads a single
//! `128`-bit multiplicative congruential (`MCG`) register from one draw to the
//! next. Each step multiplies the state by a fixed odd `64`-bit constant inside
//! a `128`-bit accumulator and returns the top `64` bits of the product. This
//! is the generator popularized as `lehmer64`: extremely cheap (one widening
//! multiply per draw), long period, and good statistical quality for
//! simulation work where a deterministic `CPU`/`GPU` particle system wants the
//! same frame to replay bit-for-bit identically.
//!
//! The recurrence is `state <- state * MUL (mod 2^128)`, where `MUL` is the
//! odd multiplier `0xDA94_2042_E4DD_58B5`. The output of a draw is
//! `(state >> 64) as u64`, i.e. the high `64` bits of the freshly updated
//! `128`-bit register. Because the modulus is a power of two and the multiplier
//! is odd, the low bits of a pure `MCG` are weak; returning only the high half
//! sidesteps that and yields a well-mixed `64`-bit word.
//!
//! Every operation here is integer arithmetic performed with wrapping
//! (modulo-`2^128`) semantics plus a fixed shift. There is no floating point,
//! no division, and no transcendental function anywhere in generation, so
//! results are identical on every target.
//!
//! Seeding uses a `splitmix64` finalizer to spread a single `64`-bit seed
//! across the full `128`-bit state: two successive `splitmix64` draws form the
//! high and low halves of the initial register. This avoids the classic
//! failure mode where a small or structured seed leaves the first few draws
//! correlated. The `splitmix64` avalanche (an `XOR`-shift / multiply mixer) has
//! strong bit diffusion, so neighbouring seeds produce unrelated streams.
//!
//! Scope and boundaries: this module is deliberately narrow and completely
//! self-contained. It is the `Lehmer64` recurrence plus a thin seeding helper,
//! and it shares no code with the `squares`, `xoshiro`, or `xorshift` engines
//! elsewhere in this crate. Do not conflate them.
//!
//! This generator is fast and non-cryptographic. A `Lehmer64` stream is
//! trivially predictable once any state is known, so it must never be used for
//! security, key material, or anywhere an adversary could exploit
//! predictability. It exists purely for reproducible, high-throughput
//! simulation randomness.

/// The odd `64`-bit multiplier that drives the `Lehmer64` recurrence.
///
/// It is widened to `u128` so the `state * MUL` product is computed inside the
/// `128`-bit register before the implicit modulo-`2^128` wrap.
const MUL: u128 = 0xDA94_2042_E4DD_58B5;

/// The `splitmix64` additive constant (`floor(2^64 / phi)`, the golden-ratio
/// gamma) that advances the seeding mixer one step at a time.
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// First multiplier of the `splitmix64` finalizer avalanche.
const SPLITMIX_MIX_1: u64 = 0xBF58_476D_1CE4_E5B9;

/// Second multiplier of the `splitmix64` finalizer avalanche.
const SPLITMIX_MIX_2: u64 = 0x94D0_49BB_1331_11EB;

/// Advance a `splitmix64` state once and return the finalized output.
///
/// This is the canonical `splitmix64` step: bump the running `state` by the
/// golden-ratio gamma, then run the finalizer avalanche (two `XOR`-shift /
/// multiply rounds followed by a final `XOR`-shift). It is used only to expand
/// a single `64`-bit seed into a well-mixed `128`-bit `Lehmer64` register; it
/// is not part of the generation hot path.
///
/// The `state` argument is updated in place to the bumped value so that a
/// caller can chain successive draws by threading the same variable.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(SPLITMIX_GAMMA);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(SPLITMIX_MIX_1);
    z = (z ^ (z >> 27)).wrapping_mul(SPLITMIX_MIX_2);
    z ^ (z >> 31)
}

/// A stateful `Lehmer64` multiplicative congruential generator.
///
/// `Lehmer64` holds a single `128`-bit register and advances it by one
/// multiplication per draw. Two generators constructed from the same state (or
/// the same seed) always replay the identical sequence, which is the defining
/// property this particle system relies on for deterministic replay.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Lehmer64 {
    /// The `128`-bit multiplicative congruential register.
    state: u128,
}

impl Lehmer64 {
    /// Create a generator directly from a raw `128`-bit `state`.
    ///
    /// The state is used verbatim as the initial register; the next draw is
    /// `(state * MUL >> 64) as u64`. Any value is accepted, including zero
    /// (which produces an all-zero stream, since zero is the fixed point of a
    /// multiplicative generator). Use [`Lehmer64::from_seed`] for a well-mixed
    /// register derived from a single `64`-bit seed.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_render_architecture::particle::lehmer64::Lehmer64;
    ///
    /// let rng = Lehmer64::from_state(1);
    /// assert!(rng.state() == 1);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_state(state: u128) -> Self {
        Self { state }
    }

    /// Create a generator by expanding a single `64`-bit `seed` through two
    /// `splitmix64` draws.
    ///
    /// The first `splitmix64` output becomes the high `64` bits of the initial
    /// `128`-bit register and the second becomes the low `64` bits, i.e.
    /// `state = (hi << 64) | lo`. This guarantees that even small or structured
    /// seeds (such as `0` or `1`) start from a well-diffused register, so
    /// adjacent seeds yield unrelated streams.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_render_architecture::particle::lehmer64::Lehmer64;
    ///
    /// let rng = Lehmer64::from_seed(1);
    /// assert!(rng.state() == 0x910a_2dec_8902_5cc1_beeb_8da1_658e_ec67);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_seed(seed: u64) -> Self {
        let mut mixer = seed;
        let hi = splitmix64(&mut mixer);
        let lo = splitmix64(&mut mixer);
        Self {
            state: ((hi as u128) << 64) | lo as u128,
        }
    }

    /// The current `128`-bit multiplicative congruential register.
    ///
    /// This is the value that the next [`Lehmer64::next_u64`] call will
    /// multiply by `MUL`. It round-trips exactly through
    /// [`Lehmer64::from_state`].
    #[inline]
    #[must_use]
    pub fn state(&self) -> u128 {
        self.state
    }

    /// Draw the next `64`-bit value and advance the register.
    ///
    /// Multiplies the state by `MUL` with wrapping (modulo-`2^128`) semantics,
    /// stores the result back, and returns the high `64` bits of the updated
    /// register. The `as u64` truncation of `state >> 64` is exact: the shift
    /// has already discarded the low half, so the cast keeps precisely the top
    /// `64` bits.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(MUL);
        (self.state >> 64) as u64
    }

    /// Draw `N` successive values into a fixed-size array.
    ///
    /// This is exactly equivalent to calling [`Lehmer64::next_u64`] `N` times
    /// in order and collecting the results, so the generator is left advanced
    /// by `N` steps. `N` is a compile-time constant, so no heap allocation is
    /// involved; `N == 0` yields an empty array and leaves the state untouched.
    #[inline]
    #[must_use]
    pub fn next_array<const N: usize>(&mut self) -> [u64; N] {
        core::array::from_fn(|_| self.next_u64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Hard reference vectors: from_state(1) ---
    //
    // These were verified against an independent 128-bit implementation. The
    // first draw is deliberately 0x0: 1 * MUL is still less than 2^64, so its
    // high 64 bits are zero. That boundary value must be preserved.

    const FROM_STATE_1_OUT5: [u64; 5] = [
        0x0,
        0xbaa0_9ca7_3f32_65b4,
        0xdb76_c439_96e5_58d0,
        0x5b39_42a4_2b92_b969,
        0x79cb_010e_faeb_6973,
    ];

    const FROM_STATE_1_STATE_AFTER_5: u128 = 0x79cb_010e_faeb_6973_5df7_11cb_f762_5aa5;

    #[test]
    fn from_state_1_out0_is_zero_boundary() {
        let mut rng = Lehmer64::from_state(1);
        assert!(rng.next_u64() == 0x0);
    }

    #[test]
    fn from_state_1_out1() {
        let mut rng = Lehmer64::from_state(1);
        rng.next_u64();
        assert!(rng.next_u64() == 0xbaa0_9ca7_3f32_65b4);
    }

    #[test]
    fn from_state_1_out2() {
        let mut rng = Lehmer64::from_state(1);
        let got: [u64; 3] = rng.next_array();
        assert!(got[2] == 0xdb76_c439_96e5_58d0);
    }

    #[test]
    fn from_state_1_out3() {
        let mut rng = Lehmer64::from_state(1);
        let got: [u64; 4] = rng.next_array();
        assert!(got[3] == 0x5b39_42a4_2b92_b969);
    }

    #[test]
    fn from_state_1_out4() {
        let mut rng = Lehmer64::from_state(1);
        let got: [u64; 5] = rng.next_array();
        assert!(got[4] == 0x79cb_010e_faeb_6973);
    }

    #[test]
    fn from_state_1_out5_as_array() {
        let mut rng = Lehmer64::from_state(1);
        let got: [u64; 5] = rng.next_array();
        assert!(got == FROM_STATE_1_OUT5);
    }

    #[test]
    fn from_state_1_state_after_five_draws() {
        let mut rng = Lehmer64::from_state(1);
        let _: [u64; 5] = rng.next_array();
        assert!(rng.state() == FROM_STATE_1_STATE_AFTER_5);
    }

    #[test]
    fn from_state_1_first_value_boundary_is_exactly_zero() {
        // 1 * MUL < 2^64, so the high 64 bits of the first product are zero.
        // This boundary must stay asserted even though it looks degenerate.
        let mut rng = Lehmer64::from_state(1);
        let first = rng.next_u64();
        assert!(first == 0);
        // The very next draw must already be non-zero.
        assert!(rng.next_u64() != 0);
    }

    // --- Hard reference vectors: from_seed(1) ---

    const FROM_SEED_1_INITIAL_STATE: u128 = 0x910a_2dec_8902_5cc1_beeb_8da1_658e_ec67;

    const FROM_SEED_1_OUT5: [u64; 5] = [
        0x95ff_4196_4c12_e658,
        0xb21d_0bdb_0851_af8d,
        0x3cfe_e867_0eb8_b663,
        0x0cff_06ae_c0e2_cae9,
        0xcd6d_a2e8_bd08_a298,
    ];

    #[test]
    fn from_seed_1_initial_state() {
        let rng = Lehmer64::from_seed(1);
        assert!(rng.state() == FROM_SEED_1_INITIAL_STATE);
    }

    #[test]
    fn from_seed_1_initial_state_halves() {
        // hi = splitmix64 draw 1, lo = splitmix64 draw 2.
        let rng = Lehmer64::from_seed(1);
        let hi = (rng.state() >> 64) as u64;
        let lo = rng.state() as u64;
        assert!(hi == 0x910a_2dec_8902_5cc1);
        assert!(lo == 0xbeeb_8da1_658e_ec67);
    }

    #[test]
    fn from_seed_1_out0() {
        let mut rng = Lehmer64::from_seed(1);
        assert!(rng.next_u64() == 0x95ff_4196_4c12_e658);
    }

    #[test]
    fn from_seed_1_out1() {
        let mut rng = Lehmer64::from_seed(1);
        rng.next_u64();
        assert!(rng.next_u64() == 0xb21d_0bdb_0851_af8d);
    }

    #[test]
    fn from_seed_1_out2() {
        let mut rng = Lehmer64::from_seed(1);
        let got: [u64; 3] = rng.next_array();
        assert!(got[2] == 0x3cfe_e867_0eb8_b663);
    }

    #[test]
    fn from_seed_1_out3() {
        let mut rng = Lehmer64::from_seed(1);
        let got: [u64; 4] = rng.next_array();
        assert!(got[3] == 0x0cff_06ae_c0e2_cae9);
    }

    #[test]
    fn from_seed_1_out4() {
        let mut rng = Lehmer64::from_seed(1);
        let got: [u64; 5] = rng.next_array();
        assert!(got[4] == 0xcd6d_a2e8_bd08_a298);
    }

    #[test]
    fn from_seed_1_out5_as_array() {
        let mut rng = Lehmer64::from_seed(1);
        let got: [u64; 5] = rng.next_array();
        assert!(got == FROM_SEED_1_OUT5);
    }

    // --- from_seed is from_state over the mixed register ---

    #[test]
    fn from_seed_1_matches_from_state_of_initial() {
        let mut a = Lehmer64::from_seed(1);
        let mut b = Lehmer64::from_state(FROM_SEED_1_INITIAL_STATE);
        let sa: [u64; 8] = a.next_array();
        let sb: [u64; 8] = b.next_array();
        assert!(sa == sb);
    }

    // --- Determinism ---

    #[test]
    fn same_state_replays_identically() {
        let mut a = Lehmer64::from_state(0x1234_5678_9abc_def0_0fed_cba9_8765_4321);
        let mut b = Lehmer64::from_state(0x1234_5678_9abc_def0_0fed_cba9_8765_4321);
        for _ in 0..256 {
            assert!(a.next_u64() == b.next_u64());
        }
    }

    #[test]
    fn same_seed_replays_identically() {
        let mut a = Lehmer64::from_seed(0xABCD_1234_5678_9F0E);
        let mut b = Lehmer64::from_seed(0xABCD_1234_5678_9F0E);
        let sa: [u64; 64] = a.next_array();
        let sb: [u64; 64] = b.next_array();
        assert!(sa == sb);
    }

    #[test]
    fn determinism_over_many_seeds() {
        for seed in 0..128u64 {
            let mut a = Lehmer64::from_seed(seed);
            let mut b = Lehmer64::from_seed(seed);
            let va: [u64; 4] = a.next_array();
            let vb: [u64; 4] = b.next_array();
            assert!(va == vb);
        }
    }

    // --- next_array equivalence to repeated next_u64 ---

    #[test]
    fn next_array_matches_repeated_next_u64() {
        let mut a = Lehmer64::from_seed(42);
        let mut b = Lehmer64::from_seed(42);
        let batch: [u64; 16] = a.next_array();
        let mut one_at_a_time = [0u64; 16];
        for slot in &mut one_at_a_time {
            *slot = b.next_u64();
        }
        assert!(batch == one_at_a_time);
    }

    #[test]
    fn next_array_advances_state_like_n_draws() {
        let mut a = Lehmer64::from_seed(7);
        let mut b = Lehmer64::from_seed(7);
        let _: [u64; 10] = a.next_array();
        for _ in 0..10 {
            b.next_u64();
        }
        assert!(a.state() == b.state());
    }

    #[test]
    fn next_array_zero_length_leaves_state_untouched() {
        let mut rng = Lehmer64::from_seed(99);
        let before = rng.state();
        let empty: [u64; 0] = rng.next_array();
        assert!(empty == []);
        assert!(rng.state() == before);
    }

    #[test]
    fn next_array_length_matches_const() {
        let mut rng = Lehmer64::from_seed(1);
        let got: [u64; 12] = rng.next_array();
        assert!(got.len() == 12);
    }

    // --- State progression and round-trips ---

    #[test]
    fn from_state_round_trips_through_state() {
        let raw = 0xDEAD_BEEF_CAFE_F00D_0123_4567_89AB_CDEF;
        let rng = Lehmer64::from_state(raw);
        assert!(rng.state() == raw);
    }

    #[test]
    fn state_changes_after_a_draw() {
        let mut rng = Lehmer64::from_seed(5);
        let before = rng.state();
        rng.next_u64();
        assert!(rng.state() != before);
    }

    #[test]
    fn captured_state_resumes_the_stream() {
        let mut rng = Lehmer64::from_seed(0x1111_2222_3333_4444);
        for _ in 0..20 {
            rng.next_u64();
        }
        let captured = rng.state();
        let mut resumed = Lehmer64::from_state(captured);
        let sa: [u64; 16] = rng.next_array();
        let sb: [u64; 16] = resumed.next_array();
        assert!(sa == sb);
    }

    // --- Distinct seeds / states produce distinct streams ---

    #[test]
    fn different_seeds_differ_initial_state() {
        let a = Lehmer64::from_seed(1);
        let b = Lehmer64::from_seed(2);
        assert!(a.state() != b.state());
    }

    #[test]
    fn different_seeds_produce_different_streams() {
        let mut a = Lehmer64::from_seed(0xAAAA_AAAA_AAAA_AAAA);
        let mut b = Lehmer64::from_seed(0x5555_5555_5555_5555);
        let sa: [u64; 16] = a.next_array();
        let sb: [u64; 16] = b.next_array();
        assert!(sa != sb);
    }

    #[test]
    fn different_states_produce_different_streams() {
        let mut a = Lehmer64::from_state(0x1);
        let mut b = Lehmer64::from_state(0x3);
        let sa: [u64; 16] = a.next_array();
        let sb: [u64; 16] = b.next_array();
        assert!(sa != sb);
    }

    #[test]
    fn adjacent_seeds_are_well_separated() {
        // splitmix64 seeding should decorrelate neighbouring seeds: the first
        // draws of seed n and seed n+1 should essentially never collide.
        let mut collisions = 0u32;
        for seed in 0..256u64 {
            let mut a = Lehmer64::from_seed(seed);
            let mut b = Lehmer64::from_seed(seed + 1);
            if a.next_u64() == b.next_u64() {
                collisions += 1;
            }
        }
        assert!(collisions == 0);
    }

    // --- Non-degenerate output ---

    #[test]
    fn output_not_all_zero() {
        let mut rng = Lehmer64::from_seed(123);
        let any_nonzero = (0..256).any(|_| rng.next_u64() != 0);
        assert!(any_nonzero);
    }

    #[test]
    fn output_not_all_equal() {
        let mut rng = Lehmer64::from_seed(123);
        let first = rng.next_u64();
        let any_different = (0..256).any(|_| rng.next_u64() != first);
        assert!(any_different);
    }

    #[test]
    fn output_has_many_distinct_values() {
        // Draw a fixed batch and count exact duplicates with a nested scan,
        // keeping the test allocation-free. A healthy generator should produce
        // entirely unique draws over this many samples.
        let mut rng = Lehmer64::from_seed(0x0F0F_0F0F_0F0F_0F0F);
        let batch: [u64; 128] = rng.next_array();
        let mut duplicates = 0u32;
        for i in 0..batch.len() {
            for j in (i + 1)..batch.len() {
                if batch[i] == batch[j] {
                    duplicates += 1;
                }
            }
        }
        assert!(duplicates == 0);
    }

    #[test]
    fn consecutive_draws_mostly_differ() {
        let mut rng = Lehmer64::from_seed(0x2468_ACE0_1357_9BDF);
        let mut prev = rng.next_u64();
        let mut collisions = 0u32;
        for _ in 0..1024 {
            let cur = rng.next_u64();
            if cur == prev {
                collisions += 1;
            }
            prev = cur;
        }
        assert!(collisions == 0);
    }

    #[test]
    fn high_and_low_bits_both_toggle() {
        let mut rng = Lehmer64::from_seed(0xCAFE_F00D_1234_5678);
        let mut or_acc = 0u64;
        let mut and_acc = u64::MAX;
        for _ in 0..512 {
            let v = rng.next_u64();
            or_acc |= v;
            and_acc &= v;
        }
        // Across many draws every bit position should see both a 0 and a 1.
        assert!(or_acc == u64::MAX);
        assert!(and_acc == 0);
    }

    #[test]
    fn seed_zero_is_well_mixed() {
        // Even seed 0 must start from a diffused register, not all-zero.
        let rng = Lehmer64::from_seed(0);
        assert!(rng.state() != 0);
        let mut probe = rng;
        assert!(probe.next_u64() != 0);
    }

    #[test]
    fn seed_bit_flip_avalanches_initial_state() {
        // Flipping a single seed bit should change roughly half the state bits.
        let a = Lehmer64::from_seed(0x8000_0000_0000_0000);
        let b = Lehmer64::from_seed(0x8000_0000_0000_0001);
        let diff = (a.state() ^ b.state()).count_ones();
        assert!(diff >= 32);
    }

    // --- Structural / trait sanity ---

    #[test]
    fn clone_copies_position() {
        let mut rng = Lehmer64::from_seed(55);
        rng.next_u64();
        rng.next_u64();
        let mut clone = rng;
        assert!(rng.next_u64() == clone.next_u64());
    }

    #[test]
    fn clone_is_independent_after_copy() {
        let mut rng = Lehmer64::from_seed(55);
        let clone = rng;
        rng.next_u64();
        // Advancing the original must not move the copy's register.
        assert!(clone.state() == Lehmer64::from_seed(55).state());
    }

    #[test]
    fn equality_tracks_state() {
        let a = Lehmer64::from_seed(77);
        let b = Lehmer64::from_seed(77);
        assert!(a == b);
        let mut c = Lehmer64::from_seed(77);
        c.next_u64();
        assert!(a != c);
    }
}
