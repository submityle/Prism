//! Directional sound cone: attenuates an emitter's gain as the listener moves
//! off the emitter's facing axis.
//!
//! A [`Cone`] describes two nested angular regions around the emitter's
//! `forward` axis. Inside the inner cone the source plays at full gain; outside
//! the outer cone it plays at a reduced `outer_gain`; between the two the gain
//! is linearly interpolated. This is the standard directional-source model used
//! by spatial audio APIs (the OpenAL "sound cone" and equivalents), expressed
//! here with full-cone angles.
//!
//! # Determinism
//!
//! The angle between the emitter axis and the emitter-to-listener direction is
//! recovered with [`bevy_math::ops::acos`] (libm-backed) rather than an `f32`
//! intrinsic, and vectors are normalised through [`Vec3::normalize_or_zero`],
//! so cone gains are bit-reproducible across targets.
//!
//! # Provenance
//!
//! This is a from-scratch implementation of the well-known directional sound
//! cone model (OpenAL style: inner angle, outer angle, outer gain). It contains
//! **no Unreal Engine, Unity, Godot, Wwise, or FMOD source or derived code**;
//! only publicly documented acoustics knowledge is used.

use bevy_math::{Vec3, ops};
use prism_audio_core::math::Sample;

use core::f32::consts::TAU;

/// A directional attenuation cone attached to an emitter's facing axis.
///
/// Both angles are **full-cone angles** in radians (the total opening of the
/// cone, measured across the axis) and are constrained to `[0, TAU]`;
/// `outer_gain` is a linear gain multiplier in `[0, 1]`. The half-angles
/// actually used for comparison against a direction are `inner_angle * 0.5` and
/// `outer_angle * 0.5`.
///
/// The invariant `outer_angle >= inner_angle` is maintained by [`Cone::new`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Cone {
    /// Full opening angle of the inner cone in radians, in `[0, TAU]`. Inside
    /// this cone the source plays at full gain.
    pub inner_angle: Sample,
    /// Full opening angle of the outer cone in radians, in `[0, TAU]` and
    /// `>= inner_angle`. Outside this cone the source plays at `outer_gain`.
    pub outer_angle: Sample,
    /// Linear gain applied at and beyond the outer cone, in `[0, 1]`.
    pub outer_gain: Sample,
}

impl Default for Cone {
    /// An omnidirectional cone that never attenuates: both angles are the full
    /// `TAU`, so every direction lies within the inner cone and receives unit
    /// gain regardless of `outer_gain`.
    #[inline]
    fn default() -> Self {
        Self {
            inner_angle: TAU,
            outer_angle: TAU,
            outer_gain: 0.0,
        }
    }
}

impl Cone {
    /// Creates a cone, clamping both angles into `[0, TAU]` and `outer_gain`
    /// into `[0, 1]`, then enforcing `outer_angle >= inner_angle` (the outer
    /// angle is raised to the inner angle if it was smaller).
    #[must_use]
    #[inline]
    pub fn new(inner_angle: Sample, outer_angle: Sample, outer_gain: Sample) -> Self {
        let inner_angle = inner_angle.clamp(0.0, TAU);
        let outer_angle = outer_angle.clamp(0.0, TAU).max(inner_angle);
        let outer_gain = outer_gain.clamp(0.0, 1.0);
        Self { inner_angle, outer_angle, outer_gain }
    }

    /// Computes the cone gain for a precomputed off-axis `angle` in radians.
    ///
    /// `angle` is the angle between the emitter's `forward` axis and the
    /// emitter-to-listener direction, in `[0, PI]`. Let `hi = inner_angle * 0.5`
    /// and `ho = outer_angle * 0.5` be the half-angles. The gain is:
    ///
    /// - `1.0` when `angle <= hi` (inside the inner cone),
    /// - `outer_gain` when `angle >= ho` (outside the outer cone),
    /// - a linear interpolation from `1.0` to `outer_gain` across `[hi, ho]`.
    ///
    /// When `ho <= hi` (a degenerate or clamped-equal configuration) the cone
    /// collapses to a step: `1.0` for `angle <= hi`, otherwise `outer_gain`.
    #[must_use]
    #[inline]
    pub fn gain_from_angle(&self, angle: Sample) -> Sample {
        let hi = self.inner_angle * 0.5;
        let ho = self.outer_angle * 0.5;

        if angle <= hi {
            return 1.0;
        }
        if angle >= ho {
            return self.outer_gain;
        }

        // hi < angle < ho, so ho - hi > 0 and the divisor is safe.
        let t = (angle - hi) / (ho - hi);
        1.0 + t * (self.outer_gain - 1.0)
    }

    /// Computes the cone gain from the emitter's `forward` axis and the
    /// `emitter_to_listener` direction (both world-space vectors of any
    /// non-zero length; only their directions matter).
    ///
    /// Both vectors are normalised with [`Vec3::normalize_or_zero`]; if either
    /// degenerates to zero the off-axis angle is taken as `0`, yielding full
    /// gain. Otherwise the angle is `acos(dot)` with the cosine clamped to
    /// `[-1, 1]` for numerical safety, and [`Cone::gain_from_angle`] is applied.
    #[must_use]
    #[inline]
    pub fn gain(&self, forward: Vec3, emitter_to_listener: Vec3) -> Sample {
        let nf = forward.normalize_or_zero();
        let nd = emitter_to_listener.normalize_or_zero();

        if nf == Vec3::ZERO || nd == Vec3::ZERO {
            return self.gain_from_angle(0.0);
        }

        let cos = nf.dot(nd).clamp(-1.0, 1.0);
        let angle = ops::acos(cos);
        self.gain_from_angle(angle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::{FRAC_PI_2, PI};

    fn approx(a: Sample, b: Sample, eps: Sample) -> bool {
        (a - b).abs() <= eps
    }

    /// A representative directional cone: inner half-angle PI/4 (full PI/2),
    /// outer half-angle PI/2 (full PI), outer gain 0.25.
    fn sample_cone() -> Cone {
        Cone::new(FRAC_PI_2, PI, 0.25)
    }

    #[test]
    fn default_is_omnidirectional_full_gain() {
        let c = Cone::default();
        assert!(approx(c.gain_from_angle(0.0), 1.0, 1e-6));
        assert!(approx(c.gain_from_angle(FRAC_PI_2), 1.0, 1e-6));
        assert!(approx(c.gain_from_angle(PI), 1.0, 1e-6));
    }

    #[test]
    fn new_clamps_and_orders() {
        // Angles out of range are clamped; outer < inner is raised to inner;
        // gain is clamped to [0, 1].
        let c = Cone::new(-1.0, 100.0, 2.0);
        assert!(approx(c.inner_angle, 0.0, 1e-6));
        assert!(approx(c.outer_angle, TAU, 1e-6));
        assert!(approx(c.outer_gain, 1.0, 1e-6));

        let d = Cone::new(PI, FRAC_PI_2, -0.5);
        // outer raised to inner.
        assert!(approx(d.inner_angle, PI, 1e-6));
        assert!(approx(d.outer_angle, PI, 1e-6));
        assert!(approx(d.outer_gain, 0.0, 1e-6));
    }

    #[test]
    fn front_is_full_gain() {
        let c = sample_cone();
        // angle = 0 (straight ahead).
        assert!(approx(c.gain_from_angle(0.0), 1.0, 1e-6));
    }

    #[test]
    fn inside_inner_cone_is_full_gain() {
        let c = sample_cone();
        // inner half-angle is PI/4; anything <= that is full gain.
        assert!(approx(c.gain_from_angle(FRAC_PI_2 * 0.25), 1.0, 1e-6));
        assert!(approx(c.gain_from_angle(FRAC_PI_2 * 0.5), 1.0, 1e-6));
    }

    #[test]
    fn beyond_outer_cone_is_outer_gain() {
        let c = sample_cone();
        // outer half-angle is PI/2; at/after it we get outer_gain.
        assert!(approx(c.gain_from_angle(FRAC_PI_2), 0.25, 1e-6));
        // Fully behind.
        assert!(approx(c.gain_from_angle(PI), 0.25, 1e-6));
    }

    #[test]
    fn midpoint_interpolates() {
        let c = sample_cone();
        // Half-angles hi = PI/4, ho = PI/2; midpoint is 3*PI/8.
        let mid = (FRAC_PI_2 * 0.5 + FRAC_PI_2) * 0.5;
        let expected = (1.0 + c.outer_gain) * 0.5;
        assert!(approx(c.gain_from_angle(mid), expected, 1e-5));
    }

    #[test]
    fn degenerate_equal_angles_step() {
        // inner == outer => step function at the shared half-angle.
        let c = Cone::new(FRAC_PI_2, FRAC_PI_2, 0.3);
        let half = FRAC_PI_2 * 0.5;
        assert!(approx(c.gain_from_angle(half - 0.01), 1.0, 1e-6));
        assert!(approx(c.gain_from_angle(half + 0.01), 0.3, 1e-6));
    }

    #[test]
    fn inner_greater_than_outer_does_not_panic() {
        // new() enforces outer >= inner, so this degenerates to a step and
        // must not produce NaN or panic.
        let c = Cone::new(PI, 0.0, 0.5);
        let g = c.gain_from_angle(FRAC_PI_2);
        assert!(g.is_finite());
        // half-angle is PI/2; at exactly the boundary angle <= hi => 1.0.
        assert!(approx(g, 1.0, 1e-6));
    }

    #[test]
    fn gain_front_source_is_full() {
        let c = sample_cone();
        // forward and emitter->listener aligned => angle 0 => full gain.
        let forward = Vec3::new(0.0, 0.0, -1.0);
        let to_listener = Vec3::new(0.0, 0.0, -3.0);
        assert!(approx(c.gain(forward, to_listener), 1.0, 1e-6));
    }

    #[test]
    fn gain_behind_source_is_outer_gain() {
        let c = sample_cone();
        // Listener directly behind the emitter's facing => angle PI.
        let forward = Vec3::new(0.0, 0.0, -1.0);
        let to_listener = Vec3::new(0.0, 0.0, 4.0);
        assert!(approx(c.gain(forward, to_listener), c.outer_gain, 1e-6));
    }

    #[test]
    fn gain_matches_gain_from_angle() {
        let c = sample_cone();
        // 90 degrees off-axis: forward -Z, listener to the right (+X).
        let forward = Vec3::new(0.0, 0.0, -1.0);
        let to_listener = Vec3::new(2.0, 0.0, 0.0);
        let via_vec = c.gain(forward, to_listener);
        let via_angle = c.gain_from_angle(FRAC_PI_2);
        assert!(approx(via_vec, via_angle, 1e-5));
    }

    #[test]
    fn gain_zero_vectors_are_full_gain_and_finite() {
        let c = sample_cone();
        assert!(approx(c.gain(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)), 1.0, 1e-6));
        assert!(approx(c.gain(Vec3::new(0.0, 0.0, -1.0), Vec3::ZERO), 1.0, 1e-6));
        assert!(c.gain(Vec3::ZERO, Vec3::ZERO).is_finite());
    }

    #[test]
    fn gain_never_nan_across_sweep() {
        let c = sample_cone();
        let mut angle = 0.0f32;
        while angle <= PI {
            let g = c.gain_from_angle(angle);
            assert!(g.is_finite());
            assert!((c.outer_gain..=1.0).contains(&g));
            angle += PI / 64.0;
        }
    }
}
