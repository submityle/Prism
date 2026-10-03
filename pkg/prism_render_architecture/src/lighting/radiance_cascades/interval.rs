//! Radiance interval — the atomic payload a radiance cascade stores per probe
//! and per direction.
//!
//! A cascade probe does not store a single radiance value; it stores, for each
//! angular bin, the radiance gathered along a bounded *radial interval*
//! `[t0, t1)` of that ray together with the interval's **transmittance** (the
//! fraction of light from beyond `t1` that survives the interval). Splitting a
//! ray into contiguous intervals and compositing them is what lets the
//! cascade hierarchy reconstruct a full-range gather from pieces of very
//! different angular/spatial resolution.
//!
//! The compositing operator is ordinary front-to-back "over": if `near` covers
//! `[t0, t1)` and `far` covers `[t1, t2)` along the *same* ray, then
//! `near.over(far)` covers `[t0, t2)`:
//!
//! ```text
//! radiance      = near.radiance + near.transmittance * far.radiance
//! transmittance = near.transmittance * far.transmittance
//! ```
//!
//! This is premultiplied-alpha compositing with `alpha = 1 - transmittance`,
//! so it is associative with the clear interval
//! ([`RadianceInterval::CLEAR`]) as its identity. Those two algebraic facts are
//! exactly what the merge stage relies on, so they are asserted in the golden
//! tests here.
//!
//! # Provenance
//! Radiance intervals and their front-to-back merge are the core data model of
//! Alexander Sannikov's *Radiance Cascades* (2023); the premultiplied "over"
//! operator is the classic Porter–Duff compositing algebra (1984). This is a
//! clean-room classical implementation.
//!
//! # References
//! - A. Sannikov, *Radiance Cascades: A Novel Approach to Calculating Global
//!   Illumination* (2023).
//! - T. Porter, T. Duff, *Compositing Digital Images* (SIGGRAPH 1984).
//!
//! No Unreal Engine source is used anywhere in this module.

use glam::Vec3;

/// Radiance accumulated along a bounded ray interval plus the fraction of
/// background light that survives it.
///
/// `transmittance` is clamped to `[0, 1]`; `radiance` is premultiplied, i.e. it
/// already accounts for absorption *inside* the interval, so it is composited
/// additively rather than being re-attenuated by this interval's own
/// transmittance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadianceInterval {
    /// Premultiplied radiance emitted/scattered toward the probe within the
    /// interval, in linear RGB.
    pub radiance: Vec3,
    /// Fraction of radiance arriving from beyond the interval that reaches the
    /// probe, in `[0, 1]`. `1.0` means the interval is fully transparent.
    pub transmittance: f32,
}

impl RadianceInterval {
    /// A fully transparent, non-emitting interval: the identity of [`over`].
    ///
    /// [`over`]: RadianceInterval::over
    pub const CLEAR: Self = Self {
        radiance: Vec3::ZERO,
        transmittance: 1.0,
    };

    /// A fully opaque, non-emitting interval (a black occluder): nothing from
    /// beyond survives and the interval itself emits nothing.
    pub const OPAQUE: Self = Self {
        radiance: Vec3::ZERO,
        transmittance: 0.0,
    };

    /// Build an interval from premultiplied radiance and transmittance,
    /// clamping transmittance into `[0, 1]`.
    #[must_use]
    pub fn new(radiance: Vec3, transmittance: f32) -> Self {
        Self {
            radiance,
            transmittance: transmittance.clamp(0.0, 1.0),
        }
    }

    /// A fully opaque emitter: it radiates `radiance` and blocks everything
    /// behind it (transmittance `0`).
    #[must_use]
    pub fn emitter(radiance: Vec3) -> Self {
        Self {
            radiance,
            transmittance: 0.0,
        }
    }

    /// Front-to-back composite: `self` is the *near* interval, `far` is the
    /// contiguous interval beyond it along the same ray.
    ///
    /// The result covers the union of both intervals. This is associative and
    /// has [`CLEAR`](Self::CLEAR) as its identity on both sides.
    #[must_use]
    pub fn over(self, far: Self) -> Self {
        Self {
            radiance: self.radiance + self.transmittance * far.radiance,
            transmittance: self.transmittance * far.transmittance,
        }
    }

    /// Linearly interpolate two intervals component-wise (used by the bilinear
    /// probe interpolation in the merge stage). `t` is clamped to `[0, 1]`.
    #[must_use]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            radiance: self.radiance + (other.radiance - self.radiance) * t,
            transmittance: self.transmittance + (other.transmittance - self.transmittance) * t,
        }
    }
}

impl Default for RadianceInterval {
    fn default() -> Self {
        Self::CLEAR
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: Vec3, b: Vec3, eps: f32) -> bool {
        (a - b).abs().max_element() <= eps
    }

    #[test]
    fn clear_is_the_identity_of_over() {
        let x = RadianceInterval::new(Vec3::new(0.3, 0.6, 0.9), 0.4);
        assert_eq!(RadianceInterval::CLEAR.over(x), x);
        assert_eq!(x.over(RadianceInterval::CLEAR), x);
    }

    #[test]
    fn opaque_near_hides_everything_behind_it() {
        let behind = RadianceInterval::new(Vec3::splat(5.0), 1.0);
        let merged = RadianceInterval::emitter(Vec3::new(1.0, 0.0, 0.0)).over(behind);
        assert_eq!(merged.radiance, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(merged.transmittance, 0.0);
    }

    #[test]
    fn over_is_associative() {
        let a = RadianceInterval::new(Vec3::new(0.1, 0.2, 0.3), 0.7);
        let b = RadianceInterval::new(Vec3::new(0.4, 0.1, 0.2), 0.5);
        let c = RadianceInterval::new(Vec3::new(0.2, 0.3, 0.1), 0.6);
        let left = a.over(b).over(c);
        let right = a.over(b.over(c));
        assert!(approx(left.radiance, right.radiance, 1.0e-6));
        assert!((left.transmittance - right.transmittance).abs() <= 1.0e-6);
    }

    #[test]
    fn transmittance_is_clamped() {
        assert_eq!(RadianceInterval::new(Vec3::ZERO, 2.0).transmittance, 1.0);
        assert_eq!(RadianceInterval::new(Vec3::ZERO, -1.0).transmittance, 0.0);
    }

    #[test]
    fn lerp_endpoints_and_midpoint() {
        let a = RadianceInterval::new(Vec3::ZERO, 1.0);
        let b = RadianceInterval::new(Vec3::splat(1.0), 0.0);
        assert_eq!(a.lerp(b, 0.0), a);
        assert_eq!(a.lerp(b, 1.0), b);
        let mid = a.lerp(b, 0.5);
        assert!(approx(mid.radiance, Vec3::splat(0.5), 1.0e-6));
        assert!((mid.transmittance - 0.5).abs() <= 1.0e-6);
    }
}
