//! Matsumoto and Panneton's `WELL1024a` well-equidistributed long-period
//! linear pseudo-random number generator (`PRNG`), used for reproducible,
//! massively parallel particle randomness (design: stochastic effects).
//!
//! `WELL1024a` is a member of the `WELL` (Well-Equidistributed Long-period
//! Linear) family. It is a classic *stateful* `PRNG`: unlike a counter-based
//! generator, it threads a hidden register of thirty-two `32`-bit words from
//! one draw to the next. The internal state is therefore `1024` bits wide,
//! which is where the `1024` in the name comes from, and the generator has a
//! period of `2^1024 - 1`. The `a` variant uses one particular, well-studied
//! set of recurrence parameters.
//!
//! Each draw advances a single word of the register. A rolling `index` marks
//! the current word; because the register length is a power of two, the
//! wrap-around addressing is done with a cheap bitwise `AND` mask rather than a
//! division or remainder. The step reads four taps of the state at offsets
//! `M1`, `M2`, and `M3` (plus the previous word), tempers them with a sequence
//! of `XOR` and fixed-shift operations, writes two freshly mixed words back
//! into the register, and moves the `index` one slot backwards (modulo the
//! register length). The newly written current word is the returned value.
//!
//! Every operation here is `32`-bit integer arithmetic: bitwise `XOR`, fixed
//! left and right shifts (whose out-of-range bits are simply discarded), and
//! masked array indexing. There is no multiplication, no division, no floating
//! point, and no transcendental function anywhere in generation, so results are
//! identical on every target. This makes the generator ideal for a
//! deterministic `GPU`/`CPU` particle system that must replay bit-for-bit.
//!
//! Scope and boundaries: this module is deliberately narrow and completely
//! self-contained. It is the `WELL1024a` recurrence plus thin state accessors,
//! and it shares no code with the `squares`, `xoshiro`, or `xorshift` engines
//! elsewhere in this crate. Do not conflate them.
//!
//! This generator is fast and non-cryptographic. A `WELL` stream is fully
//! predictable once any `1024`-bit state snapshot is known, so it must never be
//! used for security, key material, or anywhere an adversary could exploit
//! predictability. It exists purely for reproducible simulation randomness.

/// Number of `32`-bit words in the `WELL1024a` state register.
const STATE_WORDS: usize = 32;

/// Bit mask used to wrap a word offset back into `0..STATE_WORDS`. Because
/// [`STATE_WORDS`] is a power of two, `index & INDEX_MASK` is an exact,
/// branch-free substitute for `index % STATE_WORDS`.
const INDEX_MASK: usize = STATE_WORDS - 1;

/// Offset that selects the word immediately *before* the current one. Since
/// `STATE_WORDS - 1` is congruent to `-1` modulo [`STATE_WORDS`], adding it and
/// masking yields `(index - 1) mod STATE_WORDS` without an underflow check.
const PREV_OFFSET: usize = STATE_WORDS - 1;

/// First recurrence tap offset of the `WELL1024a` `a` parameter set.
const M1: usize = 3;

/// Second recurrence tap offset of the `WELL1024a` `a` parameter set.
const M2: usize = 24;

/// Third recurrence tap offset of the `WELL1024a` `a` parameter set.
const M3: usize = 10;

/// Matsumoto and Panneton's `WELL1024a` generator over a `1024`-bit state.
///
/// `Well1024a` holds the thirty-two-word state register plus the rolling
/// `index` of the current word. Construct it from a full state snapshot with
/// [`Well1024a::from_state`]; the snapshot plus a zero `index` fully determines
/// the stream, so two generators built from the same state always replay the
/// identical sequence. Each call to [`Well1024a::next_u32`] advances the
/// register by exactly one word and returns the freshly mixed `32`-bit result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Well1024a {
    /// The thirty-two-word (`1024`-bit) state register.
    state: [u32; STATE_WORDS],
    /// Index of the current word within [`Well1024a::state`], always kept in
    /// the range `0..STATE_WORDS`.
    index: u32,
}

impl Well1024a {
    /// Build a generator from a full `1024`-bit state snapshot, starting the
    /// rolling word `index` at zero.
    ///
    /// The provided `state` becomes the initial register verbatim. An
    /// all-zero state is a fixed point of the recurrence and therefore yields
    /// an all-zero stream; any state with at least one set bit drives the full
    /// `2^1024 - 1` period.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_render_architecture::particle::well1024a::Well1024a;
    ///
    /// let state = core::array::from_fn(|k| (k as u32) + 1);
    /// let mut rng = Well1024a::from_state(state);
    /// assert!(rng.next_u32() == 0x58C9_82B7);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_state(state: [u32; STATE_WORDS]) -> Self {
        Self { state, index: 0 }
    }

    /// A copy of the current `1024`-bit state register.
    ///
    /// Combined with [`Well1024a::from_state`] this supports snapshot/restore.
    /// Because `from_state` always starts the rolling `index` at zero, a
    /// snapshot faithfully resumes the stream only when it is taken at a word
    /// boundary (`index() == 0`, i.e. after a whole multiple of the register
    /// length). Captured at such a boundary, restoring reproduces the
    /// remaining stream exactly.
    #[inline]
    #[must_use]
    pub fn state(&self) -> [u32; STATE_WORDS] {
        self.state
    }

    /// The index of the current word within the state register.
    ///
    /// The value is always in `0..STATE_WORDS`. It decreases by one (modulo
    /// the register length) on every [`Well1024a::next_u32`] call.
    #[inline]
    #[must_use]
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Draw the next `32`-bit value and advance the register by one word.
    ///
    /// Performs a single step of the `WELL1024a` recurrence: it reads the
    /// previous word and the three taps at offsets `M1`, `M2`, and `M3`,
    /// tempers them with `XOR` and fixed shifts, writes the two mixed words
    /// back, moves the `index` one slot backwards (modulo the register
    /// length), and returns the newly written current word. All arithmetic is
    /// `32`-bit; the left shifts simply discard their overflow bits, so no
    /// explicit wrapping shift is required.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let i = self.index as usize;
        let z0 = self.state[(i + PREV_OFFSET) & INDEX_MASK];
        let vm1 = self.state[(i + M1) & INDEX_MASK];
        let vm2 = self.state[(i + M2) & INDEX_MASK];
        let vm3 = self.state[(i + M3) & INDEX_MASK];

        // Temper the current word with the first tap (matrix T1/T2 action).
        let z1 = self.state[i] ^ (vm1 ^ (vm1 >> 8));
        // Combine the remaining two taps (matrix T3/T4 action).
        let z2 = (vm2 ^ (vm2 << 19)) ^ (vm3 ^ (vm3 << 14));

        // Write the mixed current word, then the new previous word.
        self.state[i] = z1 ^ z2;
        let new_prev = (z0 ^ (z0 << 11)) ^ (z1 ^ (z1 << 7)) ^ (z2 ^ (z2 << 13));
        let prev = (i + PREV_OFFSET) & INDEX_MASK;
        self.state[prev] = new_prev;

        // The current word moves one slot backwards; return the freshly mixed
        // value that now sits there.
        self.index = prev as u32;
        self.state[prev]
    }

    /// Draw `N` consecutive values into a fixed-length array.
    ///
    /// This is exactly equivalent to calling [`Well1024a::next_u32`] `N` times
    /// and collecting the results in order; the generator is advanced by `N`
    /// words. `core::array::from_fn` invokes the closure for indices
    /// `0..N` in ascending order, which preserves the stream ordering.
    #[inline]
    #[must_use]
    pub fn next_array<const N: usize>(&mut self) -> [u32; N] {
        core::array::from_fn(|_| self.next_u32())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference seed state `state[k] = k + 1` used across the hard
    /// reference vectors.
    fn seeded() -> Well1024a {
        Well1024a::from_state(core::array::from_fn(|k| (k as u32) + 1))
    }

    /// The first six outputs of [`seeded`], verified against an independent
    /// implementation of the `WELL1024a` recurrence.
    const OUT6: [u32; 6] = [
        0x58C9_82B7,
        0x6CC8_E0B9,
        0x4001_CD3B,
        0x6599_19EF,
        0xE069_55C5,
        0x8717_7FC9,
    ];

    // --- Hard reference vectors: the first six outputs ---

    #[test]
    fn hard_vector_full_sequence() {
        let mut rng = seeded();
        let got: [u32; 6] = rng.next_array();
        assert!(got == OUT6);
    }

    #[test]
    fn hard_vector_step1() {
        let mut rng = seeded();
        assert!(rng.next_u32() == OUT6[0]);
    }

    #[test]
    fn hard_vector_step2() {
        let mut rng = seeded();
        rng.next_u32();
        assert!(rng.next_u32() == OUT6[1]);
    }

    #[test]
    fn hard_vector_step3() {
        let mut rng = seeded();
        for _ in 0..2 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == OUT6[2]);
    }

    #[test]
    fn hard_vector_step4() {
        let mut rng = seeded();
        for _ in 0..3 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == OUT6[3]);
    }

    #[test]
    fn hard_vector_step5() {
        let mut rng = seeded();
        for _ in 0..4 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == OUT6[4]);
    }

    #[test]
    fn hard_vector_step6() {
        let mut rng = seeded();
        for _ in 0..5 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == OUT6[5]);
    }

    // --- Hard reference vector: index after six draws ---

    #[test]
    fn index_after_six_is_26() {
        let mut rng = seeded();
        for _ in 0..6 {
            rng.next_u32();
        }
        assert!(rng.index() == 26);
    }

    // --- Hard reference vector: full state after six draws ---

    #[test]
    fn state_after_six_all_elements() {
        let mut rng = seeded();
        for _ in 0..6 {
            rng.next_u32();
        }
        let s = rng.state();
        assert!(s[0] == 0x00CA_C017);
        assert!(s[1] == 2);
        assert!(s[2] == 3);
        assert!(s[3] == 4);
        assert!(s[4] == 5);
        assert!(s[5] == 6);
        assert!(s[6] == 7);
        assert!(s[7] == 8);
        assert!(s[8] == 9);
        assert!(s[9] == 10);
        assert!(s[10] == 11);
        assert!(s[11] == 12);
        assert!(s[12] == 13);
        assert!(s[13] == 14);
        assert!(s[14] == 15);
        assert!(s[15] == 16);
        assert!(s[16] == 17);
        assert!(s[17] == 18);
        assert!(s[18] == 19);
        assert!(s[19] == 20);
        assert!(s[20] == 21);
        assert!(s[21] == 22);
        assert!(s[22] == 23);
        assert!(s[23] == 24);
        assert!(s[24] == 25);
        assert!(s[25] == 26);
        assert!(s[26] == 0x8717_7FC9);
        assert!(s[27] == 0x8CD6_07D2);
        assert!(s[28] == 0x3D63_D059);
        assert!(s[29] == 0x4079_C7F2);
        assert!(s[30] == 0x6C72_A0A5);
        assert!(s[31] == 0x580B_02A6);
    }

    #[test]
    fn state_middle_indices_unchanged_after_six() {
        let mut rng = seeded();
        for _ in 0..6 {
            rng.next_u32();
        }
        let s = rng.state();
        // Words 1..=25 are untouched by the first six draws.
        for k in 1..=25usize {
            assert!(s[k] == (k as u32) + 1);
        }
    }

    // --- Determinism ---

    #[test]
    fn determinism_two_instances_same_state() {
        let mut a = seeded();
        let mut b = seeded();
        for _ in 0..512 {
            assert!(a.next_u32() == b.next_u32());
        }
    }

    #[test]
    fn determinism_repeat_single_draw() {
        let mut a = seeded();
        let mut b = seeded();
        assert!(a.next_u32() == b.next_u32());
    }

    #[test]
    fn determinism_long_run() {
        let mut a = seeded();
        let mut b = seeded();
        for _ in 0..1000 {
            assert!(a.next_u32() == b.next_u32());
        }
    }

    #[test]
    fn determinism_from_captured_state() {
        let seed: [u32; STATE_WORDS] =
            core::array::from_fn(|k| (k as u32).wrapping_mul(2_654_435_761) ^ 0x9E37_79B9);
        let mut a = Well1024a::from_state(seed);
        let mut b = Well1024a::from_state(seed);
        for _ in 0..256 {
            assert!(a.next_u32() == b.next_u32());
        }
    }

    // --- next_array equivalence ---

    #[test]
    fn next_array_matches_repeated_next_u32() {
        let mut a = seeded();
        let mut b = seeded();
        let got: [u32; 64] = a.next_array();
        for value in got {
            assert!(value == b.next_u32());
        }
    }

    #[test]
    fn next_array_zero_length() {
        let mut rng = seeded();
        let got: [u32; 0] = rng.next_array();
        assert!(got == []);
        // A zero-length draw must not advance the register.
        assert!(rng.index() == 0);
    }

    #[test]
    fn next_array_length_one() {
        let mut a = seeded();
        let mut b = seeded();
        let got: [u32; 1] = a.next_array();
        assert!(got[0] == b.next_u32());
    }

    #[test]
    fn next_array_first_six_matches_hard_vector() {
        let mut rng = seeded();
        let got: [u32; 6] = rng.next_array();
        assert!(got == OUT6);
    }

    #[test]
    fn next_array_advances_by_n() {
        let mut rng = seeded();
        let _got: [u32; 6] = rng.next_array();
        assert!(rng.index() == 26);
    }

    // --- State snapshot / restore ---

    #[test]
    fn from_state_sets_index_zero() {
        let rng = seeded();
        assert!(rng.index() == 0);
    }

    #[test]
    fn state_roundtrip_before_draws() {
        let seed: [u32; STATE_WORDS] = core::array::from_fn(|k| (k as u32) + 100);
        let rng = Well1024a::from_state(seed);
        assert!(rng.state() == seed);
    }

    #[test]
    fn state_roundtrip_restores_stream() {
        let mut rng = seeded();
        // Capture at a word boundary (a whole multiple of the register length)
        // so the restored generator's zero index addresses the same word.
        for _ in 0..64 {
            rng.next_u32();
        }
        assert!(rng.index() == 0);
        let snapshot = rng.state();
        let mut restored = Well1024a::from_state(snapshot);
        for _ in 0..128 {
            assert!(rng.next_u32() == restored.next_u32());
        }
    }

    #[test]
    fn from_state_preserves_full_state() {
        let seed: [u32; STATE_WORDS] = core::array::from_fn(|k| 0xDEAD_0000 ^ (k as u32));
        let rng = Well1024a::from_state(seed);
        let out = rng.state();
        for k in 0..STATE_WORDS {
            assert!(out[k] == seed[k]);
        }
    }

    // --- Index behaviour ---

    #[test]
    fn index_starts_at_zero() {
        let rng = seeded();
        assert!(rng.index() == 0);
    }

    #[test]
    fn index_after_single_draw_is_31() {
        let mut rng = seeded();
        rng.next_u32();
        assert!(rng.index() == 31);
    }

    #[test]
    fn index_decrements_each_draw() {
        let mut rng = seeded();
        let expected: [u32; 8] = [31, 30, 29, 28, 27, 26, 25, 24];
        for want in expected {
            rng.next_u32();
            assert!(rng.index() == want);
        }
    }

    #[test]
    fn index_stays_in_range() {
        let mut rng = seeded();
        for _ in 0..512 {
            rng.next_u32();
            assert!((rng.index() as usize) < STATE_WORDS);
        }
    }

    #[test]
    fn index_wraps_after_full_cycle() {
        let mut rng = seeded();
        // After 32 draws the index returns to its starting slot.
        for _ in 0..STATE_WORDS {
            rng.next_u32();
        }
        assert!(rng.index() == 0);
    }

    // --- Different seeds produce different streams ---

    #[test]
    fn different_seed_different_first_output() {
        let mut a = seeded();
        let mut b = Well1024a::from_state(core::array::from_fn(|k| (k as u32) + 2));
        assert!(a.next_u32() != b.next_u32());
    }

    #[test]
    fn different_seed_different_sequence() {
        let mut a = seeded();
        let mut b =
            Well1024a::from_state(core::array::from_fn(|k| (k as u32).wrapping_add(0x5555)));
        let sa: [u32; 16] = a.next_array();
        let sb: [u32; 16] = b.next_array();
        assert!(sa != sb);
    }

    #[test]
    fn single_bit_seed_difference_diverges() {
        let mut a = seeded();
        let mut flipped_state = seeded().state();
        flipped_state[0] ^= 1;
        let mut b = Well1024a::from_state(flipped_state);
        let mut differences = 0u32;
        for _ in 0..32 {
            if a.next_u32() != b.next_u32() {
                differences += 1;
            }
        }
        // A one-bit change in the register perturbs essentially every draw.
        assert!(differences >= 30);
    }

    // --- Output quality ---

    #[test]
    fn output_not_all_zero() {
        let mut rng = seeded();
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
        let mut rng = seeded();
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
        let mut rng = seeded();
        let mut or_acc = 0u32;
        let mut and_acc = u32::MAX;
        for _ in 0..2048 {
            let v = rng.next_u32();
            or_acc |= v;
            and_acc &= v;
        }
        // Across many draws, every bit position should see both a 0 and a 1.
        assert!(or_acc == u32::MAX);
        assert!(and_acc == 0);
    }

    #[test]
    fn low_collision_rate_in_sample() {
        let mut rng = seeded();
        let sample: [u32; 64] = rng.next_array();
        let mut collisions = 0u32;
        for a in 0..sample.len() {
            for b in (a + 1)..sample.len() {
                if sample[a] == sample[b] {
                    collisions += 1;
                }
            }
        }
        // Birthday collisions among 64 draws of a 32-bit space are very rare.
        assert!(collisions <= 1);
    }

    // --- Fixed point: the all-zero state ---

    #[test]
    fn zero_state_produces_zero_stream() {
        let mut rng = Well1024a::from_state([0u32; STATE_WORDS]);
        for _ in 0..64 {
            assert!(rng.next_u32() == 0);
        }
    }

    // --- Structural / trait sanity ---

    #[test]
    fn clone_copies_position() {
        let mut rng = seeded();
        rng.next_u32();
        rng.next_u32();
        let mut clone = rng;
        assert!(rng.next_u32() == clone.next_u32());
    }

    #[test]
    fn clone_is_independent_after_copy() {
        let mut rng = seeded();
        let clone = rng;
        rng.next_u32();
        // Advancing the original must not move the copy's index.
        assert!(clone.index() == 0);
    }

    #[test]
    fn equality_tracks_state() {
        let a = seeded();
        let b = seeded();
        assert!(a == b);
        let mut c = seeded();
        c.next_u32();
        assert!(a != c);
    }

    #[test]
    fn fresh_generators_replay_identically() {
        let mut a = seeded();
        let first: [u32; 32] = a.next_array();
        let mut b = seeded();
        let second: [u32; 32] = b.next_array();
        assert!(first == second);
    }
}
