//! `32`-bit Fibonacci linear-feedback shift register (`LFSR`) used as a tiny,
//! fully reproducible bit source for particle randomness (design: cheap
//! deterministic jitter and dithering).
//!
//! A Fibonacci `LFSR` keeps a single `32`-bit `state` register and shifts it
//! one position per step. The output bit is simply the current least
//! significant bit (`LSB`). The new most significant bit (`MSB`) that is
//! shifted in is the exclusive-or (`XOR`) of several *tap* positions selected
//! by a feedback polynomial. This module uses the maximal-length polynomial
//! `x^32 + x^22 + x^2 + x^1 + 1`, which, expressed as right-shift taps on the
//! running register, reduces to combining the register with its shifts by
//! `10`, `30`, and `31` bits before folding down to a single feedback bit.
//!
//! The register value is interpreted purely as `32` raw bits. Every operation
//! is integer shifting, `XOR`, and bitwise-or; there is no multiplication,
//! division, or floating point anywhere, so the stream replays bit-for-bit
//! identically on every target. A `32`-bit maximal `LFSR` cycles through all
//! `2^32 - 1` non-zero states before repeating, so any non-zero seed produces
//! a long, well-mixed bit stream.
//!
//! The all-zero state is a fixed point: it only ever produces zeros. The
//! constructor [`FibonacciLfsr::from_state`] therefore expects a non-zero
//! `state`. This module never panics on a zero seed; it simply yields the
//! degenerate all-zero stream, and keeping the seed non-zero is the caller's
//! responsibility.
//!
//! This is a non-cryptographic generator. An `LFSR` stream is trivially
//! predictable once a handful of outputs are observed, so it must never be
//! used for security, key material, or anywhere predictability could be
//! exploited. It exists purely for reproducible, high-throughput simulation
//! randomness and is intentionally self-contained, sharing no code with the
//! `squares`, `xoshiro`, or `xorshift` engines elsewhere in this crate.

/// Width of the shift register in bits.
const STATE_BITS: u32 = 32;

/// Right-shift tap positions derived from the feedback polynomial
/// `x^32 + x^22 + x^2 + x^1 + 1`. Combined with the raw register they fold
/// down to the single feedback bit shifted into the `MSB` each step.
const TAP_A: u32 = 10;
const TAP_B: u32 = 30;
const TAP_C: u32 = 31;

/// A `32`-bit Fibonacci `LFSR` bit source.
///
/// The generator holds one `32`-bit register. [`FibonacciLfsr::next_bit`]
/// advances it by a single shift and returns the emitted `LSB`;
/// [`FibonacciLfsr::next_u32`] runs `32` such shifts and assembles the bits
/// `MSB`-first into a word. Two generators built from the same non-zero seed
/// replay the identical sequence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FibonacciLfsr {
    /// The current `32`-bit shift register. Should be non-zero; the all-zero
    /// value is a degenerate fixed point that only emits zeros.
    state: u32,
}

impl FibonacciLfsr {
    /// Create a generator seeded with the raw register value `state`.
    ///
    /// The caller is responsible for passing a non-zero `state`: a zero seed
    /// is a fixed point that yields only zeros. This never panics.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_render_architecture::particle::fibonacci_lfsr::FibonacciLfsr;
    ///
    /// let rng = FibonacciLfsr::from_state(1);
    /// assert_eq!(rng.state(), 1);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_state(state: u32) -> Self {
        Self { state }
    }

    /// The current raw value of the shift register.
    ///
    /// This is the exact value that the next [`FibonacciLfsr::next_bit`] call
    /// will consume, so capturing it and feeding it back into
    /// [`FibonacciLfsr::from_state`] resumes the stream precisely.
    #[inline]
    #[must_use]
    pub fn state(&self) -> u32 {
        self.state
    }

    /// Advance the register by one shift and return the emitted output bit.
    ///
    /// The returned value is the register's `LSB` before the shift and is
    /// always `0` or `1`. The feedback bit shifted into the `MSB` is the
    /// `XOR` of the polynomial taps.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_render_architecture::particle::fibonacci_lfsr::FibonacciLfsr;
    ///
    /// let mut rng = FibonacciLfsr::from_state(1);
    /// assert_eq!(rng.next_bit(), 1);
    /// assert_eq!(rng.state(), 0x8000_0000);
    /// ```
    #[inline]
    pub fn next_bit(&mut self) -> u32 {
        let outbit = self.state & 1;
        let fb =
            (self.state ^ (self.state >> TAP_A) ^ (self.state >> TAP_B) ^ (self.state >> TAP_C))
                & 1;
        self.state = (self.state >> 1) | (fb << (STATE_BITS - 1));
        outbit
    }

    /// Draw the next `32`-bit word by running `32` shifts.
    ///
    /// The `32` emitted bits are packed `MSB`-first: the first
    /// [`FibonacciLfsr::next_bit`] result becomes the word's most significant
    /// bit via the running assembly `w = (w << 1) | outbit`.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_render_architecture::particle::fibonacci_lfsr::FibonacciLfsr;
    ///
    /// let mut rng = FibonacciLfsr::from_state(1);
    /// assert_eq!(rng.next_u32(), 0x8000_0000);
    /// ```
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let mut w: u32 = 0;
        for _ in 0..STATE_BITS {
            let outbit = self.next_bit();
            w = (w << 1) | outbit;
        }
        w
    }

    /// Draw `N` consecutive `32`-bit words into a fixed-size array.
    ///
    /// This is exactly `N` back-to-back [`FibonacciLfsr::next_u32`] calls, in
    /// order, advancing the register across the whole array.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_render_architecture::particle::fibonacci_lfsr::FibonacciLfsr;
    ///
    /// let mut rng = FibonacciLfsr::from_state(1);
    /// let out: [u32; 2] = rng.next_array();
    /// assert_eq!(out[0], 0x8000_0000);
    /// ```
    #[inline]
    pub fn next_array<const N: usize>(&mut self) -> [u32; N] {
        core::array::from_fn(|_| self.next_u32())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The canonical seed used across the hard reference vectors.
    const SEED: u32 = 1;

    /// First eight register snapshots after each `next_bit` for `from_state(1)`.
    const SNAPSHOTS: [u32; 8] = [
        0x8000_0000,
        0xC000_0000,
        0x6000_0000,
        0xB000_0000,
        0xD800_0000,
        0x6C00_0000,
        0xB600_0000,
        0xDB00_0000,
    ];

    /// First eight output bits for `from_state(1)`.
    const OUTBITS: [u32; 8] = [1, 0, 0, 0, 0, 0, 0, 0];

    /// Four consecutive `next_u32` draws for `from_state(1)`.
    const OUT4: [u32; 4] = [0x8000_0000, 0xDB6D_B451, 0xE790_9909, 0x5C9D_4B22];

    /// Register value after the four `next_u32` draws above (`128` shifts).
    const STATE_AFTER_OUT4: u32 = 0x9700_8287;

    // --- Hard vector: eight register snapshots (each its own assertion) ---

    #[test]
    fn snapshot_0() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        rng.next_bit();
        assert!(rng.state() == SNAPSHOTS[0]);
    }

    #[test]
    fn snapshot_1() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        rng.next_bit();
        rng.next_bit();
        assert!(rng.state() == SNAPSHOTS[1]);
    }

    #[test]
    fn snapshot_2() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        for _ in 0..3 {
            rng.next_bit();
        }
        assert!(rng.state() == SNAPSHOTS[2]);
    }

    #[test]
    fn snapshot_3() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        for _ in 0..4 {
            rng.next_bit();
        }
        assert!(rng.state() == SNAPSHOTS[3]);
    }

    #[test]
    fn snapshot_4() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        for _ in 0..5 {
            rng.next_bit();
        }
        assert!(rng.state() == SNAPSHOTS[4]);
    }

    #[test]
    fn snapshot_5() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        for _ in 0..6 {
            rng.next_bit();
        }
        assert!(rng.state() == SNAPSHOTS[5]);
    }

    #[test]
    fn snapshot_6() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        for _ in 0..7 {
            rng.next_bit();
        }
        assert!(rng.state() == SNAPSHOTS[6]);
    }

    #[test]
    fn snapshot_7() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        for _ in 0..8 {
            rng.next_bit();
        }
        assert!(rng.state() == SNAPSHOTS[7]);
    }

    #[test]
    fn snapshots_as_sequence() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        let got: [u32; 8] = core::array::from_fn(|_| {
            rng.next_bit();
            rng.state()
        });
        assert!(got == SNAPSHOTS);
    }

    // --- Hard vector: eight output bits (each its own assertion) ---

    #[test]
    fn outbit_0() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        assert!(rng.next_bit() == OUTBITS[0]);
    }

    #[test]
    fn outbit_1() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        rng.next_bit();
        assert!(rng.next_bit() == OUTBITS[1]);
    }

    #[test]
    fn outbit_2() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        for _ in 0..2 {
            rng.next_bit();
        }
        assert!(rng.next_bit() == OUTBITS[2]);
    }

    #[test]
    fn outbit_3() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        for _ in 0..3 {
            rng.next_bit();
        }
        assert!(rng.next_bit() == OUTBITS[3]);
    }

    #[test]
    fn outbit_4() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        for _ in 0..4 {
            rng.next_bit();
        }
        assert!(rng.next_bit() == OUTBITS[4]);
    }

    #[test]
    fn outbit_5() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        for _ in 0..5 {
            rng.next_bit();
        }
        assert!(rng.next_bit() == OUTBITS[5]);
    }

    #[test]
    fn outbit_6() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        for _ in 0..6 {
            rng.next_bit();
        }
        assert!(rng.next_bit() == OUTBITS[6]);
    }

    #[test]
    fn outbit_7() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        for _ in 0..7 {
            rng.next_bit();
        }
        assert!(rng.next_bit() == OUTBITS[7]);
    }

    #[test]
    fn outbits_as_sequence() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        let got: [u32; 8] = core::array::from_fn(|_| rng.next_bit());
        assert!(got == OUTBITS);
    }

    #[test]
    fn every_next_bit_is_zero_or_one() {
        let mut rng = FibonacciLfsr::from_state(0x1234_5678);
        for _ in 0..256 {
            let b = rng.next_bit();
            assert!(b == 0 || b == 1);
        }
    }

    // --- Hard vector: four consecutive next_u32 draws ---

    #[test]
    fn next_u32_draw_0() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        assert!(rng.next_u32() == OUT4[0]);
    }

    #[test]
    fn next_u32_draw_1() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        rng.next_u32();
        assert!(rng.next_u32() == OUT4[1]);
    }

    #[test]
    fn next_u32_draw_2() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        for _ in 0..2 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == OUT4[2]);
    }

    #[test]
    fn next_u32_draw_3() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        for _ in 0..3 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == OUT4[3]);
    }

    #[test]
    fn next_u32_four_as_sequence() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        let got: [u32; 4] = core::array::from_fn(|_| rng.next_u32());
        assert!(got == OUT4);
    }

    #[test]
    fn state_after_four_words() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        for _ in 0..4 {
            rng.next_u32();
        }
        assert!(rng.state() == STATE_AFTER_OUT4);
    }

    // --- Structural / API behaviour ---

    #[test]
    fn from_state_preserves_seed() {
        let rng = FibonacciLfsr::from_state(0xDEAD_BEEF);
        assert!(rng.state() == 0xDEAD_BEEF);
    }

    #[test]
    fn state_is_seed_before_any_draw() {
        let rng = FibonacciLfsr::from_state(SEED);
        assert!(rng.state() == SEED);
    }

    #[test]
    fn next_bit_changes_state() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        let before = rng.state();
        rng.next_bit();
        assert!(rng.state() != before);
    }

    #[test]
    fn state_round_trip_resumes_stream() {
        let mut rng = FibonacciLfsr::from_state(0x0BAD_F00D);
        for _ in 0..17 {
            rng.next_u32();
        }
        let captured = rng.state();
        let expected = rng.next_u32();
        let mut resumed = FibonacciLfsr::from_state(captured);
        assert!(resumed.next_u32() == expected);
    }

    // --- Determinism ---

    #[test]
    fn same_seed_same_sequence() {
        let mut a = FibonacciLfsr::from_state(0x9E37_79B9);
        let mut b = FibonacciLfsr::from_state(0x9E37_79B9);
        let sa: [u32; 32] = core::array::from_fn(|_| a.next_u32());
        let sb: [u32; 32] = core::array::from_fn(|_| b.next_u32());
        assert!(sa == sb);
    }

    #[test]
    fn clone_copies_position() {
        let mut rng = FibonacciLfsr::from_state(0x1357_9BDF);
        rng.next_u32();
        rng.next_u32();
        let mut copy = rng;
        assert!(rng.next_u32() == copy.next_u32());
    }

    #[test]
    fn copy_is_independent() {
        let mut rng = FibonacciLfsr::from_state(0x2468_ACE0 | 1);
        let copy = rng;
        rng.next_u32();
        assert!(copy.state() == (0x2468_ACE0 | 1));
    }

    #[test]
    fn equality_tracks_state() {
        let a = FibonacciLfsr::from_state(SEED);
        let b = FibonacciLfsr::from_state(SEED);
        assert!(a == b);
        let mut c = FibonacciLfsr::from_state(SEED);
        c.next_bit();
        assert!(a != c);
    }

    // --- next_array equivalence ---

    #[test]
    fn next_array_matches_repeated_next_u32() {
        let mut a = FibonacciLfsr::from_state(0xABCD_1234);
        let via_array: [u32; 8] = a.next_array();
        let mut b = FibonacciLfsr::from_state(0xABCD_1234);
        let via_calls: [u32; 8] = core::array::from_fn(|_| b.next_u32());
        assert!(via_array == via_calls);
    }

    #[test]
    fn next_array_matches_hard_vector() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        let got: [u32; 4] = rng.next_array();
        assert!(got == OUT4);
    }

    #[test]
    fn next_array_advances_state_like_calls() {
        let mut a = FibonacciLfsr::from_state(0x5555_AAAA);
        let _: [u32; 5] = a.next_array();
        let mut b = FibonacciLfsr::from_state(0x5555_AAAA);
        for _ in 0..5 {
            b.next_u32();
        }
        assert!(a.state() == b.state());
    }

    // --- next_u32 is exactly 32 next_bit calls, MSB-first ---

    #[test]
    fn next_u32_equals_manual_bit_assembly() {
        let mut a = FibonacciLfsr::from_state(0x0F1E_2D3C);
        let word = a.next_u32();
        let mut b = FibonacciLfsr::from_state(0x0F1E_2D3C);
        let mut manual: u32 = 0;
        for _ in 0..32 {
            manual = (manual << 1) | b.next_bit();
        }
        assert!(word == manual);
    }

    #[test]
    fn next_u32_leaves_states_aligned() {
        let mut a = FibonacciLfsr::from_state(0x0F1E_2D3C);
        a.next_u32();
        let mut b = FibonacciLfsr::from_state(0x0F1E_2D3C);
        for _ in 0..32 {
            b.next_bit();
        }
        assert!(a.state() == b.state());
    }

    // --- Different seeds diverge ---

    #[test]
    fn different_seeds_differ_first_word() {
        let mut a = FibonacciLfsr::from_state(SEED);
        let mut b = FibonacciLfsr::from_state(0x8000_0000);
        assert!(a.next_u32() != b.next_u32());
    }

    #[test]
    fn different_seeds_differ_over_run() {
        let mut a = FibonacciLfsr::from_state(0x0000_0001);
        let mut b = FibonacciLfsr::from_state(0x0000_0003);
        let sa: [u32; 16] = core::array::from_fn(|_| a.next_u32());
        let sb: [u32; 16] = core::array::from_fn(|_| b.next_u32());
        assert!(sa != sb);
    }

    // --- Non-degenerate behaviour of a healthy stream ---

    #[test]
    fn nonzero_seed_stays_nonzero() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        let mut all_nonzero = true;
        for _ in 0..4096 {
            rng.next_bit();
            if rng.state() == 0 {
                all_nonzero = false;
            }
        }
        assert!(all_nonzero);
    }

    #[test]
    fn stream_is_not_constant() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        let first = rng.next_u32();
        let mut varied = false;
        for _ in 0..64 {
            if rng.next_u32() != first {
                varied = true;
            }
        }
        assert!(varied);
    }

    #[test]
    fn every_bit_position_toggles() {
        let mut rng = FibonacciLfsr::from_state(SEED);
        let mut or_acc: u32 = 0;
        let mut and_acc: u32 = u32::MAX;
        for _ in 0..256 {
            let v = rng.next_u32();
            or_acc |= v;
            and_acc &= v;
        }
        assert!(or_acc == u32::MAX);
        assert!(and_acc == 0);
    }

    #[test]
    fn zero_seed_is_degenerate_fixed_point() {
        let mut rng = FibonacciLfsr::from_state(0);
        for _ in 0..64 {
            assert!(rng.next_bit() == 0);
        }
        assert!(rng.state() == 0);
    }
}
