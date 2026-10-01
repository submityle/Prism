//! `RomuQuad` pseudo-random number generator (`PRNG`): Mark Overton's official
//! 64-bit four-register variant, implemented with pure integer `u64`
//! arithmetic.
//!
//! The generator keeps four 64-bit words of state (`w`, `x`, `y`, `z`) and
//! emits a single `u64` per step (the previous `x` word). Every update uses
//! wrapping multiplication/addition/subtraction and bit rotation only. No
//! floating point, transcendental functions, or heap-allocated containers
//! appear anywhere in this module.

/// Fixed odd multiplier constant shared by the Romu family of generators.
const C: u64 = 0xD3833E804F4C574B;

/// `RomuQuad` generator state: four 64-bit words.
pub struct RomuQuad {
    /// First state word.
    pub w: u64,
    /// Second state word (also the value returned by the next step).
    pub x: u64,
    /// Third state word.
    pub y: u64,
    /// Fourth state word.
    pub z: u64,
}

impl RomuQuad {
    /// Builds a `RomuQuad` generator from explicit `(w, x, y, z)` words.
    pub const fn new(w: u64, x: u64, y: u64, z: u64) -> Self {
        Self { w, x, y, z }
    }

    /// Builds a `RomuQuad` generator from a packed `(w, x, y, z)` state tuple.
    pub const fn from_state(state: (u64, u64, u64, u64)) -> Self {
        Self {
            w: state.0,
            x: state.1,
            y: state.2,
            z: state.3,
        }
    }

    /// Returns the current `(w, x, y, z)` state words.
    pub const fn state(&self) -> (u64, u64, u64, u64) {
        (self.w, self.x, self.y, self.z)
    }

    /// Advances the generator by one step and returns the previous `x` word.
    pub fn next_u64(&mut self) -> u64 {
        let wp = self.w;
        let xp = self.x;
        let yp = self.y;
        let zp = self.z;
        self.w = C.wrapping_mul(zp);
        self.x = zp.wrapping_add(wp.rotate_left(52));
        self.y = yp.wrapping_sub(xp);
        self.z = yp.wrapping_add(wp);
        self.z = self.z.rotate_left(19);
        xp
    }
}

#[cfg(test)]
mod tests {
    use super::{RomuQuad, C};

    /// Draws `N` consecutive outputs into a fixed-size array (no heap use).
    fn draw<const N: usize>(rng: &mut RomuQuad) -> [u64; N] {
        let mut out = [0u64; N];
        let mut i = 0;
        while i < N {
            out[i] = rng.next_u64();
            i += 1;
        }
        out
    }

    /// Advances the generator by `n` steps, discarding the drawn outputs.
    fn advance(rng: &mut RomuQuad, n: usize) {
        let mut i = 0;
        while i < n {
            let _ = rng.next_u64();
            i += 1;
        }
    }

    // -- Ground-truth anchor vectors -------------------------------------

    #[test]
    fn anchor_output_zero() {
        let mut rng = RomuQuad::new(1, 2, 3, 4);
        let out: [u64; 4] = draw(&mut rng);
        assert_eq!(out[0], 0x2);
    }

    #[test]
    fn anchor_output_one() {
        let mut rng = RomuQuad::new(1, 2, 3, 4);
        let out: [u64; 4] = draw(&mut rng);
        assert_eq!(out[1], 0x10000000000004);
    }

    #[test]
    fn anchor_output_two() {
        let mut rng = RomuQuad::new(1, 2, 3, 4);
        let out: [u64; 4] = draw(&mut rng);
        assert_eq!(out[2], 0xd2c4e0cfa033d315);
    }

    #[test]
    fn anchor_output_three() {
        let mut rng = RomuQuad::new(1, 2, 3, 4);
        let out: [u64; 4] = draw(&mut rng);
        assert_eq!(out[3], 0xd016ea2982190667);
    }

    #[test]
    fn anchor_full_output_vector() {
        let mut rng = RomuQuad::new(1, 2, 3, 4);
        let out: [u64; 4] = draw(&mut rng);
        let expected: [u64; 4] = [
            0x2,
            0x10000000000004,
            0xd2c4e0cfa033d315,
            0xd016ea2982190667,
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn anchor_state_after_four_w() {
        let mut rng = RomuQuad::new(1, 2, 3, 4);
        advance(&mut rng, 4);
        assert_eq!(rng.state().0, 0x0a2b96b8dac2caa5);
    }

    #[test]
    fn anchor_state_after_four_x() {
        let mut rng = RomuQuad::new(1, 2, 3, 4);
        advance(&mut rng, 4);
        assert_eq!(rng.state().1, 0x3f2df60eacd9df2d);
    }

    #[test]
    fn anchor_state_after_four_y() {
        let mut rng = RomuQuad::new(1, 2, 3, 4);
        advance(&mut rng, 4);
        assert_eq!(rng.state().2, 0x5d143506ddb32681);
    }

    #[test]
    fn anchor_state_after_four_z() {
        let mut rng = RomuQuad::new(1, 2, 3, 4);
        advance(&mut rng, 4);
        assert_eq!(rng.state().3, 0x4ff8ae10e0acbee0);
    }

    #[test]
    fn anchor_state_after_four_tuple() {
        let mut rng = RomuQuad::new(1, 2, 3, 4);
        advance(&mut rng, 4);
        let expected = (
            0x0a2b96b8dac2caa5u64,
            0x3f2df60eacd9df2du64,
            0x5d143506ddb32681u64,
            0x4ff8ae10e0acbee0u64,
        );
        assert!(rng.state() == expected);
    }

    // -- Constant correctness --------------------------------------------

    #[test]
    fn constant_hex_value() {
        assert_eq!(C, 0xD3833E804F4C574B);
    }

    #[test]
    fn constant_decimal_value() {
        assert_eq!(C, 15241094284759029579u64);
    }

    #[test]
    fn constant_is_odd() {
        assert_eq!(C & 1, 1);
    }

    #[test]
    fn constant_hex_matches_decimal() {
        assert!(0xD3833E804F4C574Bu64 == 15241094284759029579u64);
    }

    // -- Determinism ------------------------------------------------------

    #[test]
    fn determinism_same_seed_same_output() {
        let mut a = RomuQuad::new(1, 2, 3, 4);
        let mut b = RomuQuad::new(1, 2, 3, 4);
        let oa: [u64; 16] = draw(&mut a);
        let ob: [u64; 16] = draw(&mut b);
        assert_eq!(oa, ob);
    }

    #[test]
    fn determinism_same_seed_same_state() {
        let mut a = RomuQuad::new(9, 8, 7, 6);
        let mut b = RomuQuad::new(9, 8, 7, 6);
        advance(&mut a, 123);
        advance(&mut b, 123);
        assert!(a.state() == b.state());
    }

    #[test]
    fn determinism_long_run() {
        let mut a = RomuQuad::new(0xdead, 0xbeef, 0xcafe, 0xf00d);
        let mut b = RomuQuad::new(0xdead, 0xbeef, 0xcafe, 0xf00d);
        advance(&mut a, 4096);
        advance(&mut b, 4096);
        assert_eq!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn determinism_zero_seed_repeatable() {
        let mut a = RomuQuad::new(0, 0, 0, 0);
        let mut b = RomuQuad::new(0, 0, 0, 0);
        let oa: [u64; 8] = draw(&mut a);
        let ob: [u64; 8] = draw(&mut b);
        assert_eq!(oa, ob);
    }

    // -- Divergence on differing seeds -----------------------------------

    #[test]
    fn divergence_single_bit_w() {
        let mut a = RomuQuad::new(1, 2, 3, 4);
        let mut b = RomuQuad::new(2, 2, 3, 4);
        let oa: [u64; 8] = draw(&mut a);
        let ob: [u64; 8] = draw(&mut b);
        assert!(oa != ob);
    }

    #[test]
    fn divergence_single_bit_z() {
        let mut a = RomuQuad::new(1, 2, 3, 4);
        let mut b = RomuQuad::new(1, 2, 3, 5);
        let oa: [u64; 8] = draw(&mut a);
        let ob: [u64; 8] = draw(&mut b);
        assert!(oa != ob);
    }

    #[test]
    fn divergence_distinct_seeds() {
        let mut a = RomuQuad::new(0x1111, 0x2222, 0x3333, 0x4444);
        let mut b = RomuQuad::new(0x5555, 0x6666, 0x7777, 0x8888);
        let oa: [u64; 12] = draw(&mut a);
        let ob: [u64; 12] = draw(&mut b);
        assert!(oa != ob);
    }

    #[test]
    fn divergence_states_differ() {
        let mut a = RomuQuad::new(1, 2, 3, 4);
        let mut b = RomuQuad::new(4, 3, 2, 1);
        advance(&mut a, 32);
        advance(&mut b, 32);
        assert!(a.state() != b.state());
    }

    // -- state / from_state round-trips ----------------------------------

    #[test]
    fn roundtrip_new_state() {
        let rng = RomuQuad::new(11, 22, 33, 44);
        assert_eq!(rng.state(), (11, 22, 33, 44));
    }

    #[test]
    fn roundtrip_from_state_state() {
        let rng = RomuQuad::from_state((111, 222, 333, 444));
        assert_eq!(rng.state(), (111, 222, 333, 444));
    }

    #[test]
    fn roundtrip_capture_restore_outputs() {
        let mut rng = RomuQuad::new(7, 11, 13, 17);
        advance(&mut rng, 50);
        let snap = rng.state();
        let expected: [u64; 10] = draw(&mut rng);
        let mut restored = RomuQuad::from_state(snap);
        let actual: [u64; 10] = draw(&mut restored);
        assert_eq!(expected, actual);
    }

    #[test]
    fn roundtrip_capture_restore_state() {
        let mut rng = RomuQuad::new(0xabc, 0xdef, 0x123, 0x456);
        advance(&mut rng, 77);
        let snap = rng.state();
        let mut restored = RomuQuad::from_state(snap);
        advance(&mut rng, 25);
        advance(&mut restored, 25);
        assert!(rng.state() == restored.state());
    }

    #[test]
    fn roundtrip_new_matches_from_state() {
        let a = RomuQuad::new(5, 6, 7, 8);
        let b = RomuQuad::from_state((5, 6, 7, 8));
        assert_eq!(a.state(), b.state());
    }

    #[test]
    fn roundtrip_mid_stream_fields() {
        let mut rng = RomuQuad::new(2, 4, 6, 8);
        advance(&mut rng, 5);
        let (w, x, y, z) = rng.state();
        let rebuilt = RomuQuad::new(w, x, y, z);
        assert_eq!(rebuilt.state(), (w, x, y, z));
    }

    // -- N-th step reproduction ------------------------------------------

    #[test]
    fn nth_step_ten() {
        let mut a = RomuQuad::new(3, 1, 4, 1);
        let mut b = RomuQuad::new(3, 1, 4, 1);
        advance(&mut a, 10);
        advance(&mut b, 10);
        assert_eq!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn nth_step_fifty() {
        let mut a = RomuQuad::new(5, 9, 2, 6);
        let mut b = RomuQuad::new(5, 9, 2, 6);
        advance(&mut a, 50);
        advance(&mut b, 50);
        assert!(a.state() == b.state());
    }

    #[test]
    fn nth_step_hundred() {
        let mut a = RomuQuad::new(2, 7, 1, 8);
        let mut b = RomuQuad::new(2, 7, 1, 8);
        advance(&mut a, 100);
        advance(&mut b, 100);
        assert_eq!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn nth_step_thousand() {
        let mut a = RomuQuad::new(1, 6, 1, 8);
        let mut b = RomuQuad::new(1, 6, 1, 8);
        advance(&mut a, 1000);
        advance(&mut b, 1000);
        assert!(a.state() == b.state());
    }

    #[test]
    fn nth_step_split_matches_continuous() {
        let mut whole = RomuQuad::new(42, 43, 44, 45);
        let mut split = RomuQuad::new(42, 43, 44, 45);
        advance(&mut whole, 200);
        advance(&mut split, 120);
        advance(&mut split, 80);
        assert!(whole.state() == split.state());
    }

    // -- Structural / behavioural properties -----------------------------

    #[test]
    fn first_output_equals_initial_x() {
        let mut rng = RomuQuad::new(100, 200, 300, 400);
        assert_eq!(rng.next_u64(), 200);
    }

    #[test]
    fn second_output_equals_first_updated_x() {
        let mut rng = RomuQuad::new(1, 2, 3, 4);
        let first_x_after = {
            let mut probe = RomuQuad::new(1, 2, 3, 4);
            let _ = probe.next_u64();
            probe.state().1
        };
        let _ = rng.next_u64();
        assert_eq!(rng.next_u64(), first_x_after);
    }

    #[test]
    fn single_step_state_transition() {
        let mut rng = RomuQuad::new(1, 2, 3, 4);
        let _ = rng.next_u64();
        assert_eq!(rng.state().1, 0x10000000000004);
    }

    #[test]
    fn single_step_w_transition() {
        let mut rng = RomuQuad::new(1, 2, 3, 4);
        let _ = rng.next_u64();
        assert_eq!(rng.state().0, C.wrapping_mul(4));
    }

    #[test]
    fn single_step_y_transition() {
        let mut rng = RomuQuad::new(1, 2, 3, 4);
        let _ = rng.next_u64();
        assert_eq!(rng.state().2, 3u64.wrapping_sub(2));
    }

    #[test]
    fn single_step_z_transition() {
        let mut rng = RomuQuad::new(1, 2, 3, 4);
        let _ = rng.next_u64();
        let expected = (3u64.wrapping_add(1)).rotate_left(19);
        assert_eq!(rng.state().3, expected);
    }

    #[test]
    fn outputs_not_all_equal() {
        let mut rng = RomuQuad::new(0x9e3779b97f4a7c15, 0x1, 0x2, 0x3);
        let out: [u64; 8] = draw(&mut rng);
        let mut all_same = true;
        let mut i = 1;
        while i < out.len() {
            if out[i] != out[0] {
                all_same = false;
            }
            i += 1;
        }
        assert!(!all_same);
    }

    #[test]
    fn max_seed_runs_without_panic() {
        let mut rng = RomuQuad::new(u64::MAX, u64::MAX, u64::MAX, u64::MAX);
        let out: [u64; 8] = draw(&mut rng);
        let mut distinct = false;
        let mut i = 1;
        while i < out.len() {
            if out[i] != out[0] {
                distinct = true;
            }
            i += 1;
        }
        assert!(distinct);
    }

    #[test]
    fn high_bit_rotation_in_x() {
        let mut rng = RomuQuad::new(1 << 20, 0, 0, 0);
        let _ = rng.next_u64();
        assert_eq!(rng.state().1, (1u64 << 20).rotate_left(52));
    }

    #[test]
    fn repeated_capture_is_stable() {
        let mut rng = RomuQuad::new(321, 654, 987, 123);
        advance(&mut rng, 15);
        let s1 = rng.state();
        let s2 = rng.state();
        assert!(s1 == s2);
    }

    #[test]
    fn independent_instances_do_not_alias() {
        let mut a = RomuQuad::new(1, 2, 3, 4);
        let b = RomuQuad::new(1, 2, 3, 4);
        advance(&mut a, 5);
        assert!(a.state() != b.state());
    }

    #[test]
    fn long_sequence_has_variety() {
        let mut rng = RomuQuad::new(0xf1, 0xf2, 0xf3, 0xf4);
        let out: [u64; 32] = draw(&mut rng);
        let mut matches = 0;
        let mut i = 0;
        while i < out.len() {
            if out[i] == out[0] {
                matches += 1;
            }
            i += 1;
        }
        assert!(matches < out.len());
    }

    #[test]
    fn state_tuple_order_is_wxyz() {
        let rng = RomuQuad::new(0xaa, 0xbb, 0xcc, 0xdd);
        let (w, x, y, z) = rng.state();
        assert!(w == 0xaa && x == 0xbb && y == 0xcc && z == 0xdd);
    }
}
