//! **M5 — eased transitions.** Easing curves and an eased time-scale ramp.
//!
//! Changing the time scale (bullet-time in/out, pause ramps) with a hard
//! snap looks jarring: velocities, animation rates, and particle emission
//! jump discontinuously. [`ScaleTransition`] ramps smoothly between two
//! scales over a duration using an [`Easing`] curve, so the applied scale is
//! continuous (and, for the smooth curves, so is its first derivative).
//!
//! The curves here are **pure polynomials** so the module is `no_std` with no
//! `libm`/transcendental dependency, and every output is deterministic across
//! runs — a hard requirement for the deterministic replay path.

use crate::Duration;

/// A normalized easing curve mapping progress `t` in `[0, 1]` to an eased
/// value in `[0, 1]`, with `f(0) == 0` and `f(1) == 1`.
///
/// All variants are polynomials (no `sin`/`cos`/`pow`), keeping the crate
/// `no_std` without a math backend and fully deterministic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Easing {
    /// `f(t) = t`. Constant rate; the derivative is discontinuous at the ends.
    Linear,
    /// Hermite `smoothstep`: `3t^2 - 2t^3`. Zero slope at both ends (C1).
    #[default]
    SmoothStep,
    /// Ken Perlin's `smootherstep`: `6t^5 - 15t^4 + 10t^3`. Zero first *and*
    /// second derivative at both ends (C2); the smoothest polynomial ramp.
    SmootherStep,
    /// Accelerating quadratic `t^2` (ease-in).
    InQuad,
    /// Decelerating quadratic `1 - (1 - t)^2` (ease-out).
    OutQuad,
    /// Quadratic ease-in-out (accelerate then decelerate).
    InOutQuad,
    /// Accelerating cubic `t^3` (ease-in).
    InCubic,
    /// Decelerating cubic `1 - (1 - t)^3` (ease-out).
    OutCubic,
    /// Cubic ease-in-out.
    InOutCubic,
}

impl Easing {
    /// Evaluate the curve at `t`. The input is clamped to `[0, 1]`, so callers
    /// may pass raw (possibly slightly out-of-range) progress safely.
    #[inline]
    #[must_use]
    pub fn apply(self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::SmoothStep => t * t * (3.0 - 2.0 * t),
            Self::SmootherStep => t * t * t * (t * (t * 6.0 - 15.0) + 10.0),
            Self::InQuad => t * t,
            Self::OutQuad => {
                let u = 1.0 - t;
                1.0 - u * u
            }
            Self::InOutQuad => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    let u = -2.0 * t + 2.0;
                    1.0 - (u * u) / 2.0
                }
            }
            Self::InCubic => t * t * t,
            Self::OutCubic => {
                let u = 1.0 - t;
                1.0 - u * u * u
            }
            Self::InOutCubic => {
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    let u = -2.0 * t + 2.0;
                    1.0 - (u * u * u) / 2.0
                }
            }
        }
    }

    /// Interpolate from `from` to `to` by the eased progress `apply(t)`.
    #[inline]
    #[must_use]
    pub fn lerp(self, from: f64, to: f64, t: f64) -> f64 {
        let e = self.apply(t);
        from + (to - from) * e
    }
}

/// A smooth, time-driven ramp of a scalar value (typically a time scale)
/// between two endpoints over a fixed duration.
///
/// Feed real (unscaled) frame deltas to [`advance`](Self::advance); the ramp
/// progresses and reports the current eased value. Because the easing output
/// is continuous and the endpoints are held exactly once progress saturates,
/// there is no snap when a transition starts, runs, or completes.
///
/// A transition can be [`retarget`](Self::retarget)ed mid-flight: the new ramp
/// starts from the *current* eased value, so the value stays continuous even
/// when the goal changes (e.g. cancelling a slow-mo ramp halfway).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScaleTransition {
    from: f64,
    to: f64,
    /// Elapsed ramp time.
    elapsed: Duration,
    /// Total ramp duration. Zero means "already complete" (instant).
    duration: Duration,
    easing: Easing,
}

impl ScaleTransition {
    /// Start a transition from `from` to `to` over `duration` using `easing`.
    ///
    /// A zero `duration` yields an already-complete transition that reports
    /// `to` immediately.
    #[inline]
    #[must_use]
    pub fn new(from: f64, to: f64, duration: Duration, easing: Easing) -> Self {
        Self {
            from,
            to,
            elapsed: Duration::ZERO,
            duration,
            easing,
        }
    }

    /// A degenerate transition already resting at `value` (no ramp).
    #[inline]
    #[must_use]
    pub fn settled(value: f64) -> Self {
        Self::new(value, value, Duration::ZERO, Easing::default())
    }

    /// Linear progress in `[0, 1]` (not yet eased). `1.0` once complete.
    #[inline]
    #[must_use]
    pub fn progress(&self) -> f64 {
        if self.duration.is_zero() {
            return 1.0;
        }
        (self.elapsed.as_secs_f64() / self.duration.as_secs_f64()).clamp(0.0, 1.0)
    }

    /// The current eased value. Equals `from` at the start and `to` once
    /// complete.
    #[inline]
    #[must_use]
    pub fn current(&self) -> f64 {
        self.easing.lerp(self.from, self.to, self.progress())
    }

    /// Whether the ramp has reached its target.
    #[inline]
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.duration.is_zero() || self.elapsed >= self.duration
    }

    /// The target value.
    #[inline]
    #[must_use]
    pub fn target(&self) -> f64 {
        self.to
    }

    /// Advance the ramp by `delta` and return the new current eased value.
    ///
    /// Elapsed time is clamped to the total duration, so the value cannot
    /// overshoot `to`.
    #[inline]
    pub fn advance(&mut self, delta: Duration) -> f64 {
        self.elapsed = self.elapsed.saturating_add(delta);
        if self.elapsed > self.duration {
            self.elapsed = self.duration;
        }
        self.current()
    }

    /// Retarget to a new goal over `duration`, starting from the *current*
    /// eased value so the output stays continuous (no snap). The easing curve
    /// is preserved.
    #[inline]
    pub fn retarget(&mut self, to: f64, duration: Duration) {
        let current = self.current();
        self.from = current;
        self.to = to;
        self.elapsed = Duration::ZERO;
        self.duration = duration;
    }

    /// Retarget while also switching the easing curve.
    #[inline]
    pub fn retarget_with(&mut self, to: f64, duration: Duration, easing: Easing) {
        self.easing = easing;
        self.retarget(to, duration);
    }
}

#[cfg(test)]
mod tests {
    use super::{Easing, ScaleTransition};
    use crate::Duration;

    const CURVES: [Easing; 9] = [
        Easing::Linear,
        Easing::SmoothStep,
        Easing::SmootherStep,
        Easing::InQuad,
        Easing::OutQuad,
        Easing::InOutQuad,
        Easing::InCubic,
        Easing::OutCubic,
        Easing::InOutCubic,
    ];

    #[test]
    fn endpoints_are_exact_for_every_curve() {
        for e in CURVES {
            assert!((e.apply(0.0) - 0.0).abs() < 1e-12, "{e:?} f(0)");
            assert!((e.apply(1.0) - 1.0).abs() < 1e-12, "{e:?} f(1)");
            // Midpoint of every symmetric/standard curve is within range.
            let m = e.apply(0.5);
            assert!((0.0..=1.0).contains(&m), "{e:?} f(0.5)={m}");
        }
    }

    #[test]
    fn curves_are_monotonic_nondecreasing() {
        for e in CURVES {
            let mut last = -1.0;
            for i in 0..=100 {
                let v = e.apply(i as f64 / 100.0);
                assert!(
                    v + 1e-12 >= last,
                    "{e:?} not monotonic at {i}: {v} < {last}"
                );
                last = v;
            }
        }
    }

    #[test]
    fn input_is_clamped() {
        assert_eq!(Easing::Linear.apply(-1.0), 0.0);
        assert_eq!(Easing::Linear.apply(2.0), 1.0);
    }

    #[test]
    fn smoothstep_has_flat_ends() {
        // Finite-difference slope near both ends is ~0 for smoothstep.
        let e = Easing::SmoothStep;
        let h = 1e-4;
        let slope_start = (e.apply(h) - e.apply(0.0)) / h;
        let slope_end = (e.apply(1.0) - e.apply(1.0 - h)) / h;
        assert!(slope_start < 1e-2, "start slope {slope_start}");
        assert!(slope_end < 1e-2, "end slope {slope_end}");
    }

    #[test]
    fn transition_ramps_from_to() {
        let mut t = ScaleTransition::new(1.0, 0.1, Duration::from_secs(1), Easing::SmoothStep);
        assert!((t.current() - 1.0).abs() < 1e-12);
        assert!(!t.is_complete());
        // Halfway: smoothstep(0.5) = 0.5 exactly, so value is the midpoint.
        t.advance(Duration::from_millis(500));
        assert!((t.current() - 0.55).abs() < 1e-9, "mid {}", t.current());
        t.advance(Duration::from_millis(500));
        assert!(t.is_complete());
        assert!((t.current() - 0.1).abs() < 1e-12);
    }

    #[test]
    fn advance_cannot_overshoot() {
        let mut t = ScaleTransition::new(0.0, 1.0, Duration::from_millis(100), Easing::Linear);
        t.advance(Duration::from_secs(10));
        assert!(t.is_complete());
        assert_eq!(t.current(), 1.0);
        assert_eq!(t.progress(), 1.0);
    }

    #[test]
    fn zero_duration_is_instantly_complete() {
        let t = ScaleTransition::new(0.0, 1.0, Duration::ZERO, Easing::SmoothStep);
        assert!(t.is_complete());
        assert_eq!(t.current(), 1.0);
    }

    #[test]
    fn retarget_is_continuous() {
        let mut t = ScaleTransition::new(1.0, 0.0, Duration::from_secs(1), Easing::Linear);
        t.advance(Duration::from_millis(400));
        let before = t.current();
        // Change our mind: ramp back to 1.0 over a new window.
        t.retarget(1.0, Duration::from_secs(1));
        let after = t.current();
        assert!((before - after).abs() < 1e-12, "snap: {before} -> {after}");
        assert_eq!(t.target(), 1.0);
    }

    #[test]
    fn settled_holds_value() {
        let mut t = ScaleTransition::settled(0.5);
        assert!(t.is_complete());
        assert_eq!(t.current(), 0.5);
        assert_eq!(t.advance(Duration::from_secs(1)), 0.5);
    }
}
