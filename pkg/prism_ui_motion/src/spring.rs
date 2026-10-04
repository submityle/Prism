//! Physical spring animation: a damped-harmonic-oscillator integrator.
//!
//! The rest of this crate animates on a fixed *duration* with an easing curve
//! (see [`crate::TransitionTracker`] and [`prism_ui_anim::Tween`]). A spring is
//! the complementary model used by `Framer` Motion, `React` Spring and
//! `SwiftUI`: motion is driven by *physics*, not a clock, so an in-flight
//! animation retargeted mid-interaction keeps its current velocity and settles
//! naturally instead of snapping to a new curve. That continuity is what makes
//! spring motion feel right for drag-release, scroll-overscroll and gesture
//! hand-off.
//!
//! [`Spring`] holds the immutable physical parameters (stiffness, damping,
//! mass) and advances a [`SpringState`] (`value` + `velocity`) toward a target
//! with [`Spring::step`]. Integration is **semi-implicit (symplectic) Euler**,
//! internally sub-stepped so the result is stable and frame-rate independent;
//! because the update is linear in the displacement and velocity it preserves
//! the exact algebraic laws a harmonic oscillator must obey (reflection about
//! the target and amplitude scaling), which is what the test suite pins down.
//!
//! No transcendental functions are used, so the module is `no_std`-friendly and
//! obeys the workspace ban on `f32` transcendentals.
//!
//! # Example
//!
//! ```
//! use prism_ui_motion::{Spring, SpringState};
//!
//! // A snappy, slightly bouncy spring: ~0.4s response, under-damped.
//! let spring = Spring::with_damping_ratio(0.4, 0.7, 1.0);
//! let mut state = SpringState::at_rest(0.0);
//!
//! // Drive 90 frames at 60Hz toward 100.0; the fast residual oscillation of
//! // an under-damped spring needs a touch over a second to fully settle.
//! for _ in 0..90 {
//!     state = spring.step(state, 100.0, 1.0 / 60.0);
//! }
//! assert!(spring.is_settled(state, 100.0));
//! assert!((state.value - 100.0).abs() < 0.5);
//! ```

use core::f32::consts::TAU;

/// Largest sub-step the integrator takes, as a fraction of the spring's
/// oscillation period. Smaller is more accurate; this keeps `dt * omega` well
/// inside the stable region of semi-implicit Euler.
const SUBSTEP_PERIOD_FRACTION: f32 = 0.02;

/// Hard cap on sub-steps per [`Spring::step`] call, so a pathological
/// `dt`/stiffness combination can never cause an unbounded loop.
const MAX_SUBSTEPS: u32 = 1024;

/// The instantaneous state of a spring animation: its current `value` and the
/// `velocity` carried into the next step.
///
/// Keeping velocity explicit is what lets a spring be retargeted mid-flight
/// without discontinuity — the new motion continues from the current speed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpringState {
    /// The current animated value.
    pub value: f32,
    /// The current rate of change of [`SpringState::value`], per second.
    pub velocity: f32,
}

impl SpringState {
    /// A state sitting still at `value` (zero velocity).
    #[must_use]
    pub const fn at_rest(value: f32) -> Self {
        Self {
            value,
            velocity: 0.0,
        }
    }

    /// A state at `value` moving at `velocity`.
    #[must_use]
    pub const fn new(value: f32, velocity: f32) -> Self {
        Self { value, velocity }
    }
}

/// Rest thresholds for [`Spring::is_settled`]: a spring is considered at rest
/// once it is within `distance` of the target *and* slower than `velocity`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpringTolerance {
    /// Maximum distance from the target that still counts as settled.
    pub distance: f32,
    /// Maximum speed that still counts as settled.
    pub velocity: f32,
}

impl SpringTolerance {
    /// A tolerance suitable for pixel-space UI values (sub-pixel distance, slow
    /// residual velocity).
    pub const DEFAULT: Self = Self {
        distance: 0.01,
        velocity: 0.01,
    };
}

impl Default for SpringTolerance {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// A damped harmonic oscillator: `mass` on a spring of `stiffness`, resisted by
/// `damping`.
///
/// Construct one directly with [`Spring::new`] from physical constants, or with
/// [`Spring::with_damping_ratio`] from the designer-friendly `response` /
/// `damping_ratio` parameters used by `SwiftUI` and `React` Spring.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spring {
    /// Spring constant `k` (restoring force per unit displacement). Non-negative.
    stiffness: f32,
    /// Damping coefficient `c` (resistive force per unit velocity). Non-negative.
    damping: f32,
    /// Mass `m` on the spring. Strictly positive.
    mass: f32,
}

impl Spring {
    /// Creates a spring from physical constants.
    ///
    /// # Panics
    ///
    /// Panics unless `mass` is strictly positive and `stiffness`/`damping` are
    /// finite and non-negative.
    #[must_use]
    pub fn new(stiffness: f32, damping: f32, mass: f32) -> Self {
        assert!(mass > 0.0, "mass must be positive");
        assert!(
            stiffness >= 0.0 && stiffness.is_finite(),
            "stiffness must be finite and non-negative"
        );
        assert!(
            damping >= 0.0 && damping.is_finite(),
            "damping must be finite and non-negative"
        );
        Self {
            stiffness,
            damping,
            mass,
        }
    }

    /// Creates a spring from a `response` (the approximate time of one
    /// oscillation, in seconds) and a `damping_ratio`.
    ///
    /// This matches the parameterization used by `SwiftUI`'s `Animation::spring`
    /// and `React` Spring: a `damping_ratio` of `1.0` is critically damped (no
    /// overshoot, fastest non-oscillating approach), below `1.0` is bouncy, and
    /// above `1.0` is sluggish.
    ///
    /// # Panics
    ///
    /// Panics unless `response` and `mass` are strictly positive and
    /// `damping_ratio` is finite and non-negative.
    #[must_use]
    pub fn with_damping_ratio(response: f32, damping_ratio: f32, mass: f32) -> Self {
        assert!(response > 0.0, "response must be positive");
        assert!(mass > 0.0, "mass must be positive");
        assert!(
            damping_ratio >= 0.0 && damping_ratio.is_finite(),
            "damping_ratio must be finite and non-negative"
        );
        // omega_n = 2*pi / response; k = m * omega_n^2; c = 2 * zeta * m * omega_n.
        let omega_n = TAU / response;
        let stiffness = mass * omega_n * omega_n;
        let damping = 2.0 * damping_ratio * mass * omega_n;
        Self {
            stiffness,
            damping,
            mass,
        }
    }

    /// The spring constant `k`.
    #[must_use]
    pub const fn stiffness(&self) -> f32 {
        self.stiffness
    }

    /// The damping coefficient `c`.
    #[must_use]
    pub const fn damping(&self) -> f32 {
        self.damping
    }

    /// The mass `m`.
    #[must_use]
    pub const fn mass(&self) -> f32 {
        self.mass
    }

    /// The undamped natural angular frequency `omega_n = sqrt(k / m)`, in
    /// radians per second.
    #[must_use]
    pub fn angular_frequency(&self) -> f32 {
        (self.stiffness / self.mass).sqrt()
    }

    /// The damping ratio `zeta = c / (2 * sqrt(k * m))`.
    ///
    /// Returns `0.0` for a spring with zero stiffness, where the ratio is
    /// undefined.
    #[must_use]
    pub fn damping_ratio(&self) -> f32 {
        let denom = 2.0 * (self.stiffness * self.mass).sqrt();
        if denom > 0.0 {
            self.damping / denom
        } else {
            0.0
        }
    }

    /// Advances `state` toward `target` over `dt` seconds, returning the new
    /// state.
    ///
    /// A non-positive `dt` is a no-op and returns `state` unchanged. The step
    /// is internally sub-divided so the integration stays stable regardless of
    /// how large `dt` is relative to the spring's period.
    #[must_use]
    pub fn step(&self, state: SpringState, target: f32, dt: f32) -> SpringState {
        // Non-positive or NaN dt is a no-op (a NaN frame delta must not poison
        // the state).
        if dt <= 0.0 || dt.is_nan() {
            return state;
        }

        let substeps = self.substeps_for(dt);
        // Dividing by the (dt-derived) sub-step count keeps the step size a
        // function of `dt` alone, never of `value`/`velocity`, so the update
        // stays linear and the oscillator's reflection/scaling laws hold.
        let h = dt / (substeps as f32);

        let mut value = state.value;
        let mut velocity = state.velocity;
        let inv_mass = 1.0 / self.mass;
        for _ in 0..substeps {
            // Semi-implicit (symplectic) Euler: update velocity from the force
            // at the current position, then advance position with the *new*
            // velocity. This is markedly more stable than explicit Euler.
            let accel = (-self.stiffness * (value - target) - self.damping * velocity) * inv_mass;
            velocity += accel * h;
            value += velocity * h;
        }

        SpringState { value, velocity }
    }

    /// Whether `state` has effectively come to rest at `target` under the given
    /// tolerance.
    #[must_use]
    pub fn is_settled_within(
        &self,
        state: SpringState,
        target: f32,
        tolerance: SpringTolerance,
    ) -> bool {
        (state.value - target).abs() <= tolerance.distance
            && state.velocity.abs() <= tolerance.velocity
    }

    /// Whether `state` has effectively come to rest at `target` under
    /// [`SpringTolerance::DEFAULT`].
    #[must_use]
    pub fn is_settled(&self, state: SpringState, target: f32) -> bool {
        self.is_settled_within(state, target, SpringTolerance::DEFAULT)
    }

    /// Number of sub-steps to split `dt` into for a stable integration.
    fn substeps_for(&self, dt: f32) -> u32 {
        let omega = self.angular_frequency();
        if omega <= 0.0 {
            return 1;
        }
        // Period T = 2*pi / omega; cap each sub-step at a small fraction of it.
        let max_h = SUBSTEP_PERIOD_FRACTION * TAU / omega;
        let count = (dt / max_h).ceil();
        if count < 1.0 {
            return 1;
        }
        if count >= MAX_SUBSTEPS as f32 {
            return MAX_SUBSTEPS;
        }
        count as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// A float in `[lo, hi)`.
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
            lo + unit * (hi - lo)
        }
    }

    fn random_spring(rng: &mut SplitMix64) -> Spring {
        Spring::new(
            rng.range(1.0, 400.0),
            rng.range(0.0, 40.0),
            rng.range(0.2, 4.0),
        )
    }

    #[test]
    fn rest_at_target_is_a_fixed_point() {
        let spring = Spring::with_damping_ratio(0.5, 0.6, 1.3);
        let mut state = SpringState::at_rest(42.0);
        for i in 0..200 {
            // Vary dt to prove it is a fixed point for any step size.
            let dt = 1.0 / (30.0 + (i % 7) as f32);
            state = spring.step(state, 42.0, dt);
            assert_eq!(state.value, 42.0, "value drifted from fixed point");
            assert_eq!(state.velocity, 0.0, "velocity grew from fixed point");
        }
    }

    #[test]
    fn integration_is_deterministic() {
        let spring = Spring::new(120.0, 12.0, 1.0);
        let run = |()| {
            let mut s = SpringState::new(5.0, -3.0);
            let mut trace = Vec::new();
            for _ in 0..100 {
                s = spring.step(s, 1.0, 1.0 / 90.0);
                trace.push(s);
            }
            trace
        };
        assert_eq!(run(()), run(()), "same inputs produced different trajectories");
    }

    #[test]
    fn trajectory_reflects_about_target() {
        // The oscillator is linear in (value - target) and velocity, so a start
        // mirrored about the target produces the exactly mirrored trajectory.
        // Semi-implicit Euler is built from +, * and a dt-only sub-step count,
        // all of which commute with sign flip in IEEE-754, so this is bit-exact.
        //
        // This is exact only in *displacement* space: the integrator stores the
        // absolute `value`, so `target + d` and `target - d` round differently
        // for a non-zero target. Pinning the target at 0 keeps `value` equal to
        // the displacement, which is where the reflection symmetry is bit-exact.
        let mut rng = SplitMix64(0x5151_2323_ABCD_0001);
        for _ in 0..200 {
            let spring = random_spring(&mut rng);
            let target = 0.0;
            let x0 = rng.range(-30.0, 30.0);
            let v0 = rng.range(-30.0, 30.0);
            let dt = rng.range(1.0 / 240.0, 1.0 / 20.0);

            let mut pos = SpringState::new(target + x0, v0);
            let mut neg = SpringState::new(target - x0, -v0);
            for _ in 0..50 {
                pos = spring.step(pos, target, dt);
                neg = spring.step(neg, target, dt);
                assert_eq!(pos.value - target, target - neg.value, "value not mirrored");
                assert_eq!(pos.velocity, -neg.velocity, "velocity not mirrored");
            }
        }
    }

    #[test]
    fn trajectory_scales_with_amplitude() {
        // Scaling the initial displacement and velocity by two scales the whole
        // trajectory by two. Multiplication by a power of two commutes exactly
        // with IEEE-754 rounding, so the relation is bit-exact.
        let mut rng = SplitMix64(0x9090_1212_7777_0002);
        for _ in 0..200 {
            let spring = random_spring(&mut rng);
            let target = 0.0;
            let x0 = rng.range(-20.0, 20.0);
            let v0 = rng.range(-20.0, 20.0);
            let dt = rng.range(1.0 / 240.0, 1.0 / 20.0);

            let mut base = SpringState::new(x0, v0);
            let mut scaled = SpringState::new(2.0 * x0, 2.0 * v0);
            for _ in 0..50 {
                base = spring.step(base, target, dt);
                scaled = spring.step(scaled, target, dt);
                assert_eq!(scaled.value, 2.0 * base.value, "value did not scale");
                assert_eq!(scaled.velocity, 2.0 * base.velocity, "velocity did not scale");
            }
        }
    }

    #[test]
    fn critically_and_over_damped_never_overshoot() {
        // Released from rest, a spring with damping ratio >= 1 approaches the
        // target monotonically and never crosses it.
        let mut rng = SplitMix64(0x1357_9BDF_2468_0003);
        for _ in 0..120 {
            let response = rng.range(0.2, 1.5);
            let ratio = rng.range(1.0, 3.0);
            let mass = rng.range(0.5, 3.0);
            let spring = Spring::with_damping_ratio(response, ratio, mass);

            let amplitude = rng.range(5.0, 40.0);
            let mut state = SpringState::at_rest(amplitude); // target is 0.0
            let mut prev = state.value;
            // Heavily over-damped springs (large ratio, slow response) have a
            // dominant slow pole whose time constant can approach ~1.4s; from a
            // large amplitude they need >11s of wall-clock to reach the 0.01
            // rest tolerance, so give the loop generous headroom (20s @ 120Hz).
            for _ in 0..2400 {
                state = spring.step(state, 0.0, 1.0 / 120.0);
                // Never crosses the target (small epsilon for f32 noise).
                assert!(state.value >= -1e-3, "overshot target: {}", state.value);
                // Monotonically non-increasing toward the target.
                assert!(
                    state.value <= prev + 1e-3,
                    "value increased: {} -> {}",
                    prev,
                    state.value
                );
                prev = state.value;
            }
            assert!(spring.is_settled(state, 0.0), "did not settle");
        }
    }

    #[test]
    fn under_damped_spring_overshoots_then_settles() {
        let spring = Spring::with_damping_ratio(0.3, 0.2, 1.0); // very bouncy
        let mut state = SpringState::at_rest(0.0);
        let mut overshot = false;
        for _ in 0..2000 {
            state = spring.step(state, 10.0, 1.0 / 240.0);
            if state.value > 10.0 {
                overshot = true;
            }
            if spring.is_settled(state, 10.0) {
                break;
            }
        }
        assert!(overshot, "an under-damped spring should overshoot the target");
        assert!(spring.is_settled(state, 10.0), "under-damped spring did not settle");
        assert!((state.value - 10.0).abs() < 0.05);
    }

    #[test]
    fn stable_spring_converges_to_target() {
        let mut rng = SplitMix64(0xACE1_FACE_1234_0004);
        for _ in 0..200 {
            // Guarantee positive stiffness and damping so the spring is stable.
            let spring = Spring::new(
                rng.range(10.0, 400.0),
                rng.range(1.0, 50.0),
                rng.range(0.3, 3.0),
            );
            let target = rng.range(-100.0, 100.0);
            let mut state = SpringState::new(rng.range(-100.0, 100.0), rng.range(-50.0, 50.0));
            let mut settled = false;
            for _ in 0..6000 {
                state = spring.step(state, target, 1.0 / 120.0);
                if spring.is_settled(state, target) {
                    settled = true;
                    break;
                }
            }
            assert!(settled, "stable spring failed to settle");
            assert!((state.value - target).abs() <= SpringTolerance::DEFAULT.distance);
        }
    }

    #[test]
    fn damping_ratio_round_trips() {
        let mass = 1.7;
        for &(response, ratio) in &[(0.3_f32, 0.5_f32), (0.6, 1.0), (1.0, 2.0), (0.45, 0.75)] {
            let spring = Spring::with_damping_ratio(response, ratio, mass);
            assert!(
                (spring.damping_ratio() - ratio).abs() < 1e-4,
                "damping ratio {} != {}",
                spring.damping_ratio(),
                ratio
            );
            // response = 2*pi / omega_n  =>  omega_n = 2*pi / response.
            let expected_omega = TAU / response;
            assert!((spring.angular_frequency() - expected_omega).abs() < 1e-2);
        }
    }

    #[test]
    fn zero_stiffness_has_zero_damping_ratio() {
        let spring = Spring::new(0.0, 5.0, 1.0);
        assert_eq!(spring.damping_ratio(), 0.0);
        assert_eq!(spring.angular_frequency(), 0.0);
    }

    #[test]
    fn non_positive_dt_is_a_noop() {
        let spring = Spring::new(100.0, 10.0, 1.0);
        let state = SpringState::new(3.0, 7.0);
        assert_eq!(spring.step(state, 0.0, 0.0), state);
        assert_eq!(spring.step(state, 0.0, -1.0), state);
    }
}
