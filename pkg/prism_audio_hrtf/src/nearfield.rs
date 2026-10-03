//! Near-field corrections: binaural parallax, per-ear inverse-distance gain,
//! and a spherical-head proximity shadow.
//!
//! Measured HRTF datasets are captured on a sphere at a fixed radius (often
//! `~1 m`) and are therefore *far-field*: they assume the two ears see the
//! source from the same direction and at the same level. Inside roughly one
//! metre this breaks down. Two effects dominate:
//!
//! - **Parallax.** The ears are physically offset from the head centre, so a
//!   near source arrives from a *different direction at each ear*. This module
//!   resolves the source position against explicit left/right ear positions
//!   and reports a per-ear azimuth/elevation, which callers feed back into
//!   [`crate::interpolation`] to pick a slightly different HRIR per ear.
//! - **Near-field inter-aural level difference (ILD).** The `1/r` spreading
//!   law makes the path-length difference between the ears produce a large
//!   level difference up close (a source at the right ear is far louder in the
//!   right ear). A rigid-sphere head additionally shadows the contralateral
//!   ear.
//!
//! # What is exact vs. approximate
//!
//! The parallax geometry and the per-ear inverse-distance gain are **exact**
//! (pure geometry, fully testable). The additional contralateral head-shadow
//! term ([`NearFieldParams::max_shadow_db`]) is a **documented low-frequency
//! proximity approximation** that complements the measured far-field HRTF; it
//! is intentionally a smooth scalar rather than a full rigid-sphere filter,
//! and is labelled as such rather than presented as a physical exactitude.
//!
//! # Real-time contract
//!
//! [`resolve`] is **allocation free, lock free, and panic free**: fixed-size
//! arithmetic with clamps instead of panicking divisions. It may run on the
//! audio thread when a source's position changes.
//!
//! # Determinism
//!
//! All transcendental/length math routes through [`bevy_math::ops`]
//! (libm-backed) and the decibel conversions come from [`prism_audio_core`],
//! so results are bit-reproducible across targets.
//!
//! # Provenance
//!
//! This module is engine-agnostic and contains **no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, or Steam Audio source or derived code**. The parallax
//! geometry, `1/r` law, and rigid-sphere proximity-shadow heuristic are
//! implemented from standard, publicly documented acoustics knowledge
//! (spherical-head near-field model).

use bevy_math::{ops, Vec3};
use prism_audio_core::math::{db_to_linear, linear_to_db, Sample};

/// Default head radius in metres (average adult, the value used by most
/// spherical-head models).
pub const DEFAULT_HEAD_RADIUS: Sample = 0.0875;

/// Directions shorter than this are treated as degenerate and replaced by a
/// fallback to avoid dividing by a near-zero length.
const DIR_EPSILON: Sample = 1.0e-6;

/// Geometry of the listener's head used for near-field resolution.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct HeadGeometry {
    /// Head radius in metres; the ears sit at `+/-radius` along local `X`.
    pub radius: Sample,
}

impl Default for HeadGeometry {
    #[inline]
    fn default() -> Self {
        Self {
            radius: DEFAULT_HEAD_RADIUS,
        }
    }
}

impl HeadGeometry {
    /// Creates a head geometry with the given `radius` (metres), clamped to a
    /// small positive minimum.
    #[must_use]
    #[inline]
    pub fn new(radius: Sample) -> Self {
        Self {
            radius: radius.max(1.0e-4),
        }
    }

    /// Local-frame position of the left ear (`-radius` on `X`).
    #[must_use]
    #[inline]
    pub fn left_ear(&self) -> Vec3 {
        Vec3::new(-self.radius, 0.0, 0.0)
    }

    /// Local-frame position of the right ear (`+radius` on `X`).
    #[must_use]
    #[inline]
    pub fn right_ear(&self) -> Vec3 {
        Vec3::new(self.radius, 0.0, 0.0)
    }
}

/// Tunables for the near-field model.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct NearFieldParams {
    /// Reference radius (metres) at which the inverse-distance gain is unity;
    /// set this to the dataset's measurement radius so the far field matches.
    pub reference_distance: Sample,
    /// Maximum extra contralateral attenuation (dB, non-negative) applied by
    /// the proximity shadow when a source is against the head on the far side.
    pub max_shadow_db: Sample,
    /// Distance (in head radii) beyond which the proximity shadow fades to
    /// zero and only the measured far-field HRTF governs.
    pub far_field_radii: Sample,
    /// Ceiling on the inverse-distance gain (linear) to keep a source at the
    /// ear from producing an unbounded level.
    pub max_distance_gain: Sample,
}

impl Default for NearFieldParams {
    #[inline]
    fn default() -> Self {
        Self {
            reference_distance: 1.0,
            max_shadow_db: 6.0,
            far_field_radii: 8.0,
            max_distance_gain: 8.0,
        }
    }
}

/// Per-ear near-field result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NearFieldEar {
    /// Unit direction from this ear to the source, in the listener-local frame.
    pub direction: Vec3,
    /// Distance from this ear to the source, in metres.
    pub distance: Sample,
    /// Azimuth to the source from this ear (radians, `atan2(x, -z)`).
    pub azimuth: Sample,
    /// Elevation to the source from this ear (radians).
    pub elevation: Sample,
    /// Total near-field gain for this ear in decibels (inverse distance plus
    /// proximity shadow).
    pub gain_db: Sample,
    /// The same gain expressed as a linear multiplier.
    pub gain_linear: Sample,
}

/// The combined near-field result for both ears.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NearFieldResult {
    /// Left-ear result.
    pub left: NearFieldEar,
    /// Right-ear result.
    pub right: NearFieldEar,
}

impl NearFieldResult {
    /// Inter-aural level difference in decibels, `right - left`.
    ///
    /// Positive means the right ear is louder (a source to the right).
    #[must_use]
    #[inline]
    pub fn ild_db(&self) -> Sample {
        self.right.gain_db - self.left.gain_db
    }
}

/// Resolves near-field parallax and gain for a source at `local_dir`
/// (listener-local unit direction) and `distance` metres.
///
/// `local_dir` should be the unit direction from the listener's head centre to
/// the source (e.g. `LocalSource::direction`); a degenerate direction falls
/// back to local forward.
///
/// Real-time safe: no allocation, no panic.
#[must_use]
pub fn resolve(
    local_dir: Vec3,
    distance: Sample,
    head: HeadGeometry,
    params: NearFieldParams,
) -> NearFieldResult {
    let dir = normalize_or(local_dir, Vec3::NEG_Z);
    let source = dir * distance.max(0.0);
    let left = resolve_ear(source, head.left_ear(), Vec3::NEG_X, head, params, dir);
    let right = resolve_ear(source, head.right_ear(), Vec3::X, head, params, dir);
    NearFieldResult { left, right }
}

/// Resolves a single ear against the source position.
fn resolve_ear(
    source: Vec3,
    ear: Vec3,
    ear_normal: Vec3,
    head: HeadGeometry,
    params: NearFieldParams,
    fallback_dir: Vec3,
) -> NearFieldEar {
    let v = source - ear;
    let dist = ops::sqrt(v.dot(v));
    let dir = if dist <= DIR_EPSILON {
        fallback_dir
    } else {
        v / dist
    };

    let azimuth = ops::atan2(dir.x, -dir.z);
    let horizontal = ops::sqrt(dir.x * dir.x + dir.z * dir.z);
    let elevation = ops::atan2(dir.y, horizontal);

    // Inverse-distance gain (unity at the reference radius), capped.
    let safe_dist = dist.max(head.radius);
    let dist_gain = (params.reference_distance / safe_dist).min(params.max_distance_gain);
    let dist_gain_db = linear_to_db(dist_gain);

    // Proximity shadow: contralateral ears (source on the far side) lose level
    // as the source approaches the head. `incidence` is +1 ipsilateral,
    // -1 contralateral.
    let incidence = ear_normal.dot(dir).clamp(-1.0, 1.0);
    let contralateral = (1.0 - incidence) * 0.5; // [0, 1]
    let rho = dist / head.radius;
    let proximity = proximity_factor(rho, params.far_field_radii); // [0, 1]
    let shadow_db = -params.max_shadow_db.max(0.0) * contralateral * proximity;

    let gain_db = dist_gain_db + shadow_db;
    let gain_linear = db_to_linear(gain_db);

    NearFieldEar {
        direction: dir,
        distance: dist,
        azimuth,
        elevation,
        gain_db,
        gain_linear,
    }
}

/// Proximity weight in `[0, 1]`: `1` at the head surface (`rho = 1`), falling
/// linearly to `0` at `far_field_radii` and beyond.
#[inline]
fn proximity_factor(rho: Sample, far_field_radii: Sample) -> Sample {
    let far = far_field_radii.max(1.0 + 1.0e-3);
    ((far - rho) / (far - 1.0)).clamp(0.0, 1.0)
}

/// Normalises `v`, falling back to `fallback` when `v` is too short.
#[inline]
fn normalize_or(v: Vec3, fallback: Vec3) -> Vec3 {
    let len = ops::sqrt(v.dot(v));
    if len <= DIR_EPSILON {
        fallback
    } else {
        v / len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::FRAC_PI_2;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn ear_positions_are_symmetric() {
        let h = HeadGeometry::new(0.09);
        assert_eq!(h.left_ear(), Vec3::new(-0.09, 0.0, 0.0));
        assert_eq!(h.right_ear(), Vec3::new(0.09, 0.0, 0.0));
    }

    #[test]
    fn front_source_is_symmetric() {
        let res = resolve(
            Vec3::NEG_Z,
            0.5,
            HeadGeometry::default(),
            NearFieldParams::default(),
        );
        // Equal distances and mirror-image azimuths, so ILD ~ 0.
        assert!(approx(res.left.distance, res.right.distance, 1e-6));
        assert!(approx(res.ild_db(), 0.0, 1e-5));
        // Parallax: the front source converges, appearing slightly to the
        // inside of each ear (left ear sees +az, right ear sees -az).
        assert!(res.left.azimuth > 0.0);
        assert!(res.right.azimuth < 0.0);
        assert!(approx(res.left.azimuth, -res.right.azimuth, 1e-5));
    }

    #[test]
    fn right_source_favors_right_ear() {
        // Source to the right (+X) at 30 cm.
        let res = resolve(
            Vec3::X,
            0.3,
            HeadGeometry::default(),
            NearFieldParams::default(),
        );
        assert!(res.right.distance < res.left.distance);
        assert!(
            res.ild_db() > 0.0,
            "expected right-favoured ILD, got {}",
            res.ild_db()
        );
        assert!(res.right.gain_linear > res.left.gain_linear);
    }

    #[test]
    fn near_source_has_larger_ild_than_far() {
        let near = resolve(
            Vec3::X,
            0.2,
            HeadGeometry::default(),
            NearFieldParams::default(),
        );
        let far = resolve(
            Vec3::X,
            3.0,
            HeadGeometry::default(),
            NearFieldParams::default(),
        );
        assert!(near.ild_db() > far.ild_db());
    }

    #[test]
    fn far_source_parallax_is_negligible() {
        let res = resolve(
            Vec3::X,
            50.0,
            HeadGeometry::default(),
            NearFieldParams::default(),
        );
        // Both ears nearly agree on azimuth (~+pi/2) for a distant right source.
        assert!(approx(res.left.azimuth, FRAC_PI_2, 1e-2));
        assert!(approx(res.right.azimuth, FRAC_PI_2, 1e-2));
        assert!(approx(res.ild_db(), 0.0, 0.2));
    }

    #[test]
    fn shadow_attenuates_contralateral_ear() {
        // Source hard right against the head: left ear is contralateral and
        // should receive shadow attenuation on top of the distance loss.
        let res = resolve(
            Vec3::X,
            0.12,
            HeadGeometry::default(),
            NearFieldParams::default(),
        );
        // Left (contralateral) gain in dB should be below its pure 1/r value.
        let v = res.left.distance;
        let pure_db = linear_to_db(1.0 / v);
        assert!(res.left.gain_db < pure_db + 1e-3);
    }

    #[test]
    fn proximity_factor_bounds() {
        assert!(approx(proximity_factor(1.0, 8.0), 1.0, 1e-6));
        assert!(approx(proximity_factor(8.0, 8.0), 0.0, 1e-6));
        assert!(approx(proximity_factor(100.0, 8.0), 0.0, 1e-6));
        let mid = proximity_factor(4.5, 8.0);
        assert!(mid > 0.0 && mid < 1.0);
    }

    #[test]
    fn degenerate_direction_falls_back_without_panic() {
        let res = resolve(
            Vec3::ZERO,
            1.0,
            HeadGeometry::default(),
            NearFieldParams::default(),
        );
        // Falls back to forward; still symmetric and finite.
        assert!(res.left.gain_linear.is_finite());
        assert!(res.right.gain_linear.is_finite());
    }

    #[test]
    fn zero_distance_is_finite() {
        let res = resolve(
            Vec3::X,
            0.0,
            HeadGeometry::default(),
            NearFieldParams::default(),
        );
        assert!(res.left.gain_linear.is_finite());
        assert!(res.right.gain_linear.is_finite());
        assert!(res.left.distance > 0.0 && res.right.distance > 0.0);
    }

    #[test]
    fn results_are_deterministic() {
        let a = resolve(
            Vec3::new(0.3, 0.1, -0.9),
            0.4,
            HeadGeometry::default(),
            NearFieldParams::default(),
        );
        let b = resolve(
            Vec3::new(0.3, 0.1, -0.9),
            0.4,
            HeadGeometry::default(),
            NearFieldParams::default(),
        );
        assert_eq!(a, b);
    }
}
