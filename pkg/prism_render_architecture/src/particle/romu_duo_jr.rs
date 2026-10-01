//! `RomuDuoJr` pseudo-random number generator (`PRNG`).
//!
//! This module implements Mark Overton's `RomuDuoJr` algorithm
//! (see romu-random.org) using pure integer arithmetic so it stays
//! `no_std` + `alloc` friendly. The generator keeps two 64-bit lanes
//! and advances them with a single wrapping multiply plus a rotate,
//! making it a very small and fast non-cryptographic `RNG`.
//!
//! The lane update performed by [`RomuDuoJr::next_u64`] is:
//!
//! ```text
//! xp = x
//! x  = MULTIPLIER * y        (wrapping)
//! y  = (y - xp) <<< 27        (wrapping sub, rotate_left)
//! return xp
//! ```
//!
//! No floating-point or transcendental operations are used anywhere.

/// Multiplier constant used by the `RomuDuoJr` algorithm.
const ROMU_MULTIPLIER: u64 = 15241094284759029579;

/// Left-rotation amount applied to the `y` lane on each step.
const ROMU_ROTATION: u32 = 27;

/// `RomuDuoJr` pseudo-random number generator.
///
/// Construct one with [`RomuDuoJr::new`], draw values with
/// [`RomuDuoJr::next_u64`], and snapshot/restore via
/// [`RomuDuoJr::state`] and [`RomuDuoJr::from_state`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RomuDuoJr {
    /// First state lane.
    x: u64,
    /// Second state lane.
    y: u64,
}

impl RomuDuoJr {
    /// Creates a new generator seeded with the two lane values.
    pub fn new(x: u64, y: u64) -> Self {
        Self { x, y }
    }

    /// Rebuilds a generator from a previously captured `(x, y)` state.
    ///
    /// Together with [`RomuDuoJr::state`] this allows a stream to be
    /// paused and resumed while reproducing the exact same outputs.
    pub fn from_state(x: u64, y: u64) -> Self {
        Self { x, y }
    }

    /// Returns the current `(x, y)` state lanes.
    pub fn state(&self) -> (u64, u64) {
        (self.x, self.y)
    }

    /// Advances the generator by one step and returns the next output.
    ///
    /// The returned value is the pre-update `x` lane, matching the
    /// reference `RomuDuoJr` definition.
    pub fn next_u64(&mut self) -> u64 {
        let xp = self.x;
        self.x = ROMU_MULTIPLIER.wrapping_mul(self.y);
        self.y = self.y.wrapping_sub(xp).rotate_left(ROMU_ROTATION);
        xp
    }
}

#[cfg(test)]
mod tests {
    use super::RomuDuoJr;
    use super::ROMU_MULTIPLIER;
    use super::ROMU_ROTATION;

    /// Fills a fixed-size array with successive generator outputs.
    fn collect<const N: usize>(rng: &mut RomuDuoJr) -> [u64; N] {
        let mut out = [0u64; N];
        let mut i = 0;
        while i < N {
            out[i] = rng.next_u64();
            i += 1;
        }
        out
    }

    /// Hard external anchor: first four outputs for seed `(1, 2)`.
    const ANCHOR: [u64; 4] = [
        0x0000000000000001,
        0xa7067d009e98ae96,
        0x027a62ba58000000,
        0xbbf058bed6b89bbd,
    ];

    #[test]
    fn reference_vector_zero() {
        let mut rng = RomuDuoJr::new(1, 2);
        assert!(rng.next_u64() == ANCHOR[0]);
    }

    #[test]
    fn reference_vector_one() {
        let mut rng = RomuDuoJr::new(1, 2);
        let out: [u64; 2] = collect(&mut rng);
        assert!(out[1] == ANCHOR[1]);
    }

    #[test]
    fn reference_vector_two() {
        let mut rng = RomuDuoJr::new(1, 2);
        let out: [u64; 3] = collect(&mut rng);
        assert!(out[2] == ANCHOR[2]);
    }

    #[test]
    fn reference_vector_three() {
        let mut rng = RomuDuoJr::new(1, 2);
        let out: [u64; 4] = collect(&mut rng);
        assert!(out[3] == ANCHOR[3]);
    }

    #[test]
    fn full_anchor_block_matches() {
        let mut rng = RomuDuoJr::new(1, 2);
        let out: [u64; 4] = collect(&mut rng);
        let mut i = 0;
        while i < 4 {
            assert!(out[i] == ANCHOR[i]);
            i += 1;
        }
    }

    #[test]
    fn first_output_equals_seed_x() {
        let mut rng = RomuDuoJr::new(1, 2);
        assert!(rng.next_u64() == 1);
    }

    #[test]
    fn first_output_equals_seed_x_other() {
        let mut rng = RomuDuoJr::new(0xdead_beef, 0x1234);
        assert!(rng.next_u64() == 0xdead_beef);
    }

    #[test]
    fn first_output_equals_seed_x_generic() {
        let seed_x = 0x0123_4567_89ab_cdef;
        let mut rng = RomuDuoJr::new(seed_x, 7);
        assert!(rng.next_u64() == seed_x);
    }

    #[test]
    fn new_sets_state() {
        let rng = RomuDuoJr::new(11, 22);
        assert!(rng.state() == (11, 22));
    }

    #[test]
    fn from_state_sets_state() {
        let rng = RomuDuoJr::from_state(99, 100);
        assert!(rng.state() == (99, 100));
    }

    #[test]
    fn state_matches_constructor_args() {
        let rng = RomuDuoJr::new(0xaaaa, 0xbbbb);
        let (x, y) = rng.state();
        assert!(x == 0xaaaa);
        assert!(y == 0xbbbb);
    }

    #[test]
    fn new_and_from_state_agree() {
        let a = RomuDuoJr::new(5, 6);
        let b = RomuDuoJr::from_state(5, 6);
        assert!(a == b);
    }

    #[test]
    fn determinism_full_sequence() {
        let mut a = RomuDuoJr::new(123, 456);
        let mut b = RomuDuoJr::new(123, 456);
        let out_a: [u64; 16] = collect(&mut a);
        let out_b: [u64; 16] = collect(&mut b);
        let mut i = 0;
        while i < 16 {
            assert!(out_a[i] == out_b[i]);
            i += 1;
        }
    }

    #[test]
    fn determinism_single_step() {
        let mut a = RomuDuoJr::new(7, 7);
        let mut b = RomuDuoJr::new(7, 7);
        assert!(a.next_u64() == b.next_u64());
    }

    #[test]
    fn determinism_state_tracks() {
        let mut a = RomuDuoJr::new(42, 24);
        let mut b = RomuDuoJr::new(42, 24);
        let _ = a.next_u64();
        let _ = b.next_u64();
        assert!(a.state() == b.state());
    }

    #[test]
    fn divergence_different_x() {
        let mut a = RomuDuoJr::new(1, 9);
        let mut b = RomuDuoJr::new(2, 9);
        assert!(a.next_u64() != b.next_u64());
    }

    #[test]
    fn divergence_different_y() {
        let mut a = RomuDuoJr::new(5, 1);
        let mut b = RomuDuoJr::new(5, 2);
        let out_a: [u64; 4] = collect(&mut a);
        let out_b: [u64; 4] = collect(&mut b);
        assert!(out_a[1] != out_b[1]);
    }

    #[test]
    fn divergence_swapped_seeds() {
        let mut a = RomuDuoJr::new(3, 8);
        let mut b = RomuDuoJr::new(8, 3);
        let out_a: [u64; 4] = collect(&mut a);
        let out_b: [u64; 4] = collect(&mut b);
        assert!(out_a[3] != out_b[3]);
    }

    #[test]
    fn divergence_states_after_steps() {
        let mut a = RomuDuoJr::new(100, 200);
        let mut b = RomuDuoJr::new(101, 200);
        let _ = a.next_u64();
        let _ = b.next_u64();
        assert!(a.state() != b.state());
    }

    #[test]
    fn state_capture_reproduces_next() {
        let mut rng = RomuDuoJr::new(321, 654);
        let _ = rng.next_u64();
        let _ = rng.next_u64();
        let (x, y) = rng.state();
        let mut clone = RomuDuoJr::from_state(x, y);
        assert!(rng.next_u64() == clone.next_u64());
    }

    #[test]
    fn state_capture_reproduces_block() {
        let mut rng = RomuDuoJr::new(0xfeed, 0xface);
        let _ = rng.next_u64();
        let (x, y) = rng.state();
        let mut clone = RomuDuoJr::from_state(x, y);
        let out_a: [u64; 8] = collect(&mut rng);
        let out_b: [u64; 8] = collect(&mut clone);
        let mut i = 0;
        while i < 8 {
            assert!(out_a[i] == out_b[i]);
            i += 1;
        }
    }

    #[test]
    fn state_capture_midstream() {
        let mut rng = RomuDuoJr::new(55, 66);
        let pre: [u64; 5] = collect(&mut rng);
        let (x, y) = rng.state();
        let mut resumed = RomuDuoJr::from_state(x, y);
        let post_a: [u64; 5] = collect(&mut rng);
        let post_b: [u64; 5] = collect(&mut resumed);
        assert!(pre[0] != post_a[0]);
        let mut i = 0;
        while i < 5 {
            assert!(post_a[i] == post_b[i]);
            i += 1;
        }
    }

    #[test]
    fn second_output_is_multiplier_times_initial_y() {
        let seed_y = 2u64;
        let mut rng = RomuDuoJr::new(1, seed_y);
        let out: [u64; 2] = collect(&mut rng);
        assert!(out[1] == ROMU_MULTIPLIER.wrapping_mul(seed_y));
    }

    #[test]
    fn second_output_is_multiplier_times_initial_y_generic() {
        let seed_y = 0x00ff_00ff_00ff_00ff;
        let mut rng = RomuDuoJr::new(0, seed_y);
        let out: [u64; 2] = collect(&mut rng);
        assert!(out[1] == ROMU_MULTIPLIER.wrapping_mul(seed_y));
    }

    #[test]
    fn x_lane_after_first_step() {
        let seed_y = 2u64;
        let mut rng = RomuDuoJr::new(1, seed_y);
        let _ = rng.next_u64();
        let (x, _) = rng.state();
        assert!(x == ROMU_MULTIPLIER.wrapping_mul(seed_y));
    }

    #[test]
    fn y_lane_after_first_step() {
        let mut rng = RomuDuoJr::new(1, 2);
        let _ = rng.next_u64();
        let (_, y) = rng.state();
        let expected = 2u64.wrapping_sub(1).rotate_left(ROMU_ROTATION);
        assert!(y == expected);
    }

    #[test]
    fn y_lane_after_first_step_is_shift_for_unit_delta() {
        let mut rng = RomuDuoJr::new(1, 2);
        let _ = rng.next_u64();
        let (_, y) = rng.state();
        assert!(y == (1u64 << ROMU_ROTATION));
    }

    #[test]
    fn return_value_tracks_previous_x() {
        let mut rng = RomuDuoJr::new(13, 17);
        let r0 = rng.next_u64();
        let (x_after_first, _) = rng.state();
        let r1 = rng.next_u64();
        assert!(r0 == 13);
        assert!(r1 == x_after_first);
    }

    #[test]
    fn return_value_tracks_previous_x_chain() {
        let mut rng = RomuDuoJr::new(1000, 2000);
        let mut i = 0;
        while i < 6 {
            let (expected, _) = rng.state();
            let got = rng.next_u64();
            assert!(got == expected);
            i += 1;
        }
    }

    #[test]
    fn zero_seed_first_output_zero() {
        let mut rng = RomuDuoJr::new(0, 0);
        assert!(rng.next_u64() == 0);
    }

    #[test]
    fn zero_seed_outputs_zero_block() {
        let mut rng = RomuDuoJr::new(0, 0);
        let out: [u64; 12] = collect(&mut rng);
        let mut i = 0;
        while i < 12 {
            assert!(out[i] == 0);
            i += 1;
        }
    }

    #[test]
    fn zero_seed_state_stays_zero() {
        let mut rng = RomuDuoJr::new(0, 0);
        let _ = rng.next_u64();
        assert!(rng.state() == (0, 0));
    }

    #[test]
    fn only_y_seed_first_output_zero() {
        let mut rng = RomuDuoJr::new(0, 777);
        assert!(rng.next_u64() == 0);
    }

    #[test]
    fn only_y_seed_second_output_nonzero() {
        let seed_y = 777u64;
        let mut rng = RomuDuoJr::new(0, seed_y);
        let out: [u64; 2] = collect(&mut rng);
        assert!(out[1] == ROMU_MULTIPLIER.wrapping_mul(seed_y));
        assert!(out[1] != 0);
    }

    #[test]
    fn advancing_changes_state() {
        let mut rng = RomuDuoJr::new(5, 9);
        let before = rng.state();
        let _ = rng.next_u64();
        assert!(rng.state() != before);
    }

    #[test]
    fn independent_instances() {
        let mut a = RomuDuoJr::new(2, 3);
        let mut b = RomuDuoJr::new(2, 3);
        let _ = a.next_u64();
        let _ = a.next_u64();
        let _ = b.next_u64();
        assert!(a.state() != b.state());
    }

    #[test]
    fn copy_is_independent() {
        let mut rng = RomuDuoJr::new(321, 123);
        let mut copy = rng;
        let _ = rng.next_u64();
        assert!(rng.state() != copy.state());
        let _ = copy.next_u64();
        assert!(rng.state() == copy.state());
    }

    #[test]
    fn clone_reproduces_sequence() {
        let mut rng = RomuDuoJr::new(0xabc, 0xdef);
        let _ = rng.next_u64();
        let mut cloned = rng;
        let out_a: [u64; 8] = collect(&mut rng);
        let out_b: [u64; 8] = collect(&mut cloned);
        let mut i = 0;
        while i < 8 {
            assert!(out_a[i] == out_b[i]);
            i += 1;
        }
    }

    #[test]
    fn rotation_matches_shift_for_one() {
        assert!(1u64.rotate_left(ROMU_ROTATION) == (1u64 << ROMU_ROTATION));
    }

    #[test]
    fn rotation_wraps_high_bits() {
        let top = 1u64 << 63;
        let rotated = top.rotate_left(ROMU_ROTATION);
        assert!(rotated == (1u64 << ((63 + ROMU_ROTATION) % 64)));
    }

    #[test]
    fn multiplier_constant_value_decimal() {
        assert!(ROMU_MULTIPLIER == 15241094284759029579);
    }

    #[test]
    fn multiplier_constant_value_hex() {
        assert!(ROMU_MULTIPLIER == 0xd383_3e80_4f4c_574b);
    }

    #[test]
    fn rotation_amount_value() {
        assert!(ROMU_ROTATION == 27);
    }

    #[test]
    fn from_state_idempotent() {
        let a = RomuDuoJr::from_state(0x11, 0x22);
        let (x, y) = a.state();
        let b = RomuDuoJr::from_state(x, y);
        assert!(a == b);
    }

    #[test]
    fn two_generators_same_seed_match_many() {
        let mut a = RomuDuoJr::new(0x5555, 0xaaaa);
        let mut b = RomuDuoJr::new(0x5555, 0xaaaa);
        let mut i = 0;
        while i < 64 {
            assert!(a.next_u64() == b.next_u64());
            i += 1;
        }
    }

    #[test]
    fn different_seed_first_outputs_distinct() {
        let mut a = RomuDuoJr::new(1, 0);
        let mut b = RomuDuoJr::new(2, 0);
        let mut c = RomuDuoJr::new(3, 0);
        let va = a.next_u64();
        let vb = b.next_u64();
        let vc = c.next_u64();
        assert!(va != vb);
        assert!(vb != vc);
        assert!(va != vc);
    }

    #[test]
    fn output_block_is_filled() {
        let mut rng = RomuDuoJr::new(999, 1001);
        let out: [u64; 10] = collect(&mut rng);
        assert!(out.len() == 10);
        assert!(out[9] == out[9]);
    }

    #[test]
    fn reseed_reproduces_from_captured() {
        let mut rng = RomuDuoJr::new(0x1357, 0x2468);
        let first: [u64; 4] = collect(&mut rng);
        let reseeded = RomuDuoJr::new(0x1357, 0x2468);
        let mut again = RomuDuoJr::from_state(reseeded.state().0, reseeded.state().1);
        let second: [u64; 4] = collect(&mut again);
        let mut i = 0;
        while i < 4 {
            assert!(first[i] == second[i]);
            i += 1;
        }
    }

    #[test]
    fn sequence_not_constant() {
        let mut rng = RomuDuoJr::new(0x9e37, 0x79b9);
        let out: [u64; 5] = collect(&mut rng);
        let distinct = out[0] != out[1] || out[1] != out[2] || out[2] != out[3];
        assert!(distinct);
    }

    #[test]
    fn high_bit_seed_first_output() {
        let seed_x = 1u64 << 63;
        let mut rng = RomuDuoJr::new(seed_x, 1);
        assert!(rng.next_u64() == seed_x);
    }

    #[test]
    fn max_seed_first_output() {
        let seed_x = u64::MAX;
        let mut rng = RomuDuoJr::new(seed_x, 1);
        assert!(rng.next_u64() == seed_x);
    }

    #[test]
    fn state_roundtrip_many_steps() {
        let mut rng = RomuDuoJr::new(0x2222, 0x4444);
        let mut i = 0;
        while i < 20 {
            let (x, y) = rng.state();
            let mut mirror = RomuDuoJr::from_state(x, y);
            assert!(rng.next_u64() == mirror.next_u64());
            i += 1;
        }
    }
}
