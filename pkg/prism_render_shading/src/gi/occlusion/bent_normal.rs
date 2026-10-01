//! Bent-normal cone reconstruction from hemispherical visibility — CPU golden.
//!
//! A *bent normal* is the average unoccluded direction over a shading point's
//! hemisphere.  Where the geometric normal points straight out of the surface,
//! the bent normal leans toward whatever part of the sky is still visible, so
//! using it to look up ambient / indirect light removes the "flat" look of a
//! scalar ambient-occlusion (AO) term and reintroduces directional contact
//! shadowing.  Jimenez et al.'s GTAO (SIGGRAPH 2016) produces a bent normal as
//! a by-product of its horizon search, and Oat & Sander's *Ambient Aperture
//! Lighting* (2007) fits a *cone* to the visible set so that a single
//! `(direction, half-angle)` pair summarises the open sky.
//!
//! This module is the backend-neutral reference for that reconstruction:
//!
//! * [`BentNormalCone`] bundles the bent-normal direction, the cone
//!   half-angle (*aperture*), and the scalar visibility (AO) in one struct that
//!   mirrors the `f32` layout the GPU twin stores.
//! * [`accumulate_bent_normal`] folds a set of `(direction, visibility)`
//!   samples — e.g. the per-ray visibility of a hemisphere sampler — into a
//!   cone by taking the visibility-weighted mean direction.
//! * [`bent_normal_from_cosine_hemisphere`] builds the same cone from a
//!   geometric normal plus a set of occlusion-flagged sample directions, the
//!   way GTAO-style integrators drive it (cosine-weighted, below-horizon rays
//!   discarded).
//!
//! # Conventions
//! * Directions are right-handed unit `Vec3`s; the stored [`BentNormalCone`]
//!   direction is always normalised (unit length to within `f32` round-off).
//! * The aperture is the cone *half-angle* in radians, in `[0, PI]`.  `0` is a
//!   fully closed (point) cone and `PI` is the whole sphere.  It is recovered
//!   from the mean-resultant length `r` of the visibility-weighted directions
//!   via the uniform-cone centroid identity `r = (1 + cos(aperture)) / 2`,
//!   i.e. `aperture = acos(2*r - 1)`.  A longer resultant (`r -> 1`) means the
//!   visible directions agree, so the cone is tight; a short resultant
//!   (`r -> 0`) means they spread out, so the cone is wide.  This is the same
//!   norm-to-aperture mapping used by *Ambient Aperture Lighting*.
//! * Visibility (AO) is in `[0, 1]`, `1` fully unoccluded.
//! * Degenerate inputs never produce `NaN`: an empty sample set, a zero-length
//!   resultant, or a zero normal fall back to a defined axis (the supplied
//!   geometric normal where one exists, otherwise `+Y`) with either a fully
//!   open or fully closed cone as documented per function.
//! * Every function is a deterministic pure function: no RNG, no I/O, no GPU,
//!   no allocation, and no global state.

use bevy_math::{ops, Vec3};
use core::f32::consts::{FRAC_PI_2, PI};

/// A cone of unoccluded directions fitted to a shading point's hemisphere.
///
/// The cone is described by its central [`direction`](Self::direction) (the
/// bent normal), its half-angle [`aperture`](Self::aperture), and the scalar
/// [`visibility`](Self::visibility) (ambient occlusion) it represents.  The
/// three `f32` fields match the packed layout the GPU twin consumes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BentNormalCone {
    /// Unit-length average unoccluded direction (the bent normal).
    pub direction: Vec3,
    /// Cone half-angle in radians, in `[0, PI]` (`0` closed, `PI` whole sphere).
    pub aperture: f32,
    /// Scalar visibility / ambient occlusion in `[0, 1]` (`1` fully visible).
    pub visibility: f32,
}

impl Default for BentNormalCone {
    fn default() -> Self {
        Self::OCCLUDED
    }
}

impl BentNormalCone {
    /// A fully occluded cone: closed aperture, zero visibility, `+Y` axis.
    ///
    /// Used as the fallback when there is no visibility information to fit a
    /// cone to (for example an empty sample set).
    pub const OCCLUDED: Self = Self {
        direction: Vec3::Y,
        aperture: 0.0,
        visibility: 0.0,
    };

    /// Cosine of the cone half-angle, `cos(aperture)`, clamped to `[-1, 1]`.
    ///
    /// Handy for inside/outside tests against a direction without recomputing
    /// an `acos`: a direction `d` lies inside the cone iff
    /// `direction.dot(d) >= cos_aperture()`.
    #[inline]
    pub fn cos_aperture(&self) -> f32 {
        ops::cos(self.aperture.clamp(0.0, PI))
    }
}

/// Normalises `v`, returning `fallback` for a degenerate (near-zero) input so
/// the caller never divides by zero or propagates a `NaN`.
#[inline]
fn normalize_or(v: Vec3, fallback: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq > f32::MIN_POSITIVE {
        v * len_sq.sqrt().recip()
    } else {
        fallback
    }
}

/// Builds a [`BentNormalCone`] from an accumulated resultant vector.
///
/// `resultant` is the weighted sum `sum_i w_i * dir_i`, `weight` is the sum of
/// weights `sum_i w_i`, `fallback_axis` is the direction to open the cone
/// about when the resultant has no preferred direction, and `visibility` is the
/// scalar AO to store.  The aperture follows the mean-resultant mapping
/// `aperture = acos(2*r - 1)` with `r = |resultant| / weight` clamped to
/// `[0, 1]` (see the module docs).
#[inline]
fn cone_from_resultant(
    resultant: Vec3,
    weight: f32,
    fallback_axis: Vec3,
    visibility: f32,
) -> BentNormalCone {
    let visibility = visibility.clamp(0.0, 1.0);
    let len = resultant.length();
    if weight <= f32::MIN_POSITIVE || len <= f32::MIN_POSITIVE {
        // No net direction (empty, zero-weight, or perfectly cancelling
        // samples): open the cone fully about the fallback axis.
        return BentNormalCone {
            direction: normalize_or(fallback_axis, Vec3::Y),
            aperture: PI,
            visibility,
        };
    }
    let direction = resultant * len.recip();
    // Mean resultant length of the weighted unit directions, in [0, 1].
    let r = (len / weight).clamp(0.0, 1.0);
    // r = (1 + cos(aperture)) / 2  =>  cos(aperture) = 2r - 1.
    let cos_aperture = (2.0 * r - 1.0).clamp(-1.0, 1.0);
    let aperture = ops::acos(cos_aperture);
    BentNormalCone {
        direction,
        aperture,
        visibility,
    }
}

/// Fits a bent-normal cone to a set of `(direction, visibility)` samples.
///
/// Each sample is a (not necessarily normalised) hemisphere direction together
/// with its scalar visibility in `[0, 1]`.  The bent normal is the
/// visibility-weighted mean direction `normalize(sum_i vis_i * dir_i)`; the
/// aperture is recovered from the mean-resultant length of those weighted
/// directions (longer resultant -> tighter cone); and the stored visibility is
/// the arithmetic mean of the per-sample visibilities.
///
/// Degenerate handling (never `NaN`):
/// * An empty slice returns [`BentNormalCone::OCCLUDED`] (closed cone, zero
///   visibility, `+Y` axis) — there is nothing visible to fit.
/// * If the weighted directions cancel to a zero-length resultant (while some
///   visibility remains) the cone opens fully (`aperture = PI`) about `+Y`,
///   keeping the mean visibility.
/// * Individual zero-length sample directions contribute their visibility to
///   the mean but do not bias the bent-normal direction.
pub fn accumulate_bent_normal(samples: &[(Vec3, f32)]) -> BentNormalCone {
    if samples.is_empty() {
        return BentNormalCone::OCCLUDED;
    }
    let mut resultant = Vec3::ZERO;
    let mut weight = 0.0f32;
    for &(dir, visibility) in samples {
        let v = visibility.clamp(0.0, 1.0);
        // A degenerate direction maps to the zero vector so it cannot steer the
        // bent normal, yet it still counts toward the mean visibility below.
        resultant += normalize_or(dir, Vec3::ZERO) * v;
        weight += v;
    }
    let visibility = weight / samples.len() as f32;
    cone_from_resultant(resultant, weight, Vec3::Y, visibility)
}

/// Builds a bent-normal cone from a geometric normal and occlusion-flagged
/// sample directions, GTAO-style.
///
/// Each sample is a (not necessarily normalised) direction tagged with whether
/// it is `occluded` (`true`) or visible (`false`).  Samples are cosine-weighted
/// by `max(normal · dir, 0)`, so directions below the horizon are discarded and
/// grazing directions count for less — matching the clamped-cosine weighting of
/// a Lambertian hemisphere integral.  The bent normal is the cosine-weighted
/// mean of the *visible* directions; the stored visibility is the
/// cosine-weighted fraction of samples that are visible; and the aperture again
/// follows the mean-resultant mapping.
///
/// Degenerate handling (never `NaN`):
/// * An empty slice, or one with no sample in the upper hemisphere, falls back
///   to the (normalised) geometric normal with a hemispherical aperture
///   (`PI/2`) and full visibility — the open-sky default.
/// * If every upper-hemisphere sample is occluded (nothing visible, or the
///   visible directions cancel) the cone closes (`aperture = 0`) about the
///   geometric normal with the computed visibility.
pub fn bent_normal_from_cosine_hemisphere(
    normal: Vec3,
    samples: &[(Vec3, bool)],
) -> BentNormalCone {
    let n = normalize_or(normal, Vec3::Y);
    let mut resultant = Vec3::ZERO;
    let mut visible_weight = 0.0f32;
    let mut total_weight = 0.0f32;
    for &(dir, occluded) in samples {
        let d = normalize_or(dir, Vec3::ZERO);
        let w = n.dot(d).max(0.0); // clamped cosine weight; below-horizon -> 0
        if w <= 0.0 {
            continue;
        }
        total_weight += w;
        if !occluded {
            resultant += d * w;
            visible_weight += w;
        }
    }
    if total_weight <= f32::MIN_POSITIVE {
        // Nothing sampled in the upper hemisphere: assume the sky is open.
        return BentNormalCone {
            direction: n,
            aperture: FRAC_PI_2,
            visibility: 1.0,
        };
    }
    let visibility = (visible_weight / total_weight).clamp(0.0, 1.0);
    if visible_weight <= f32::MIN_POSITIVE || resultant.length() <= f32::MIN_POSITIVE {
        // Everything visible is occluded (or the visible directions cancel):
        // collapse onto the geometric normal with a closed cone.
        return BentNormalCone {
            direction: n,
            aperture: 0.0,
            visibility,
        };
    }
    cone_from_resultant(resultant, visible_weight, n, visibility)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Deterministic Fibonacci-spiral directions over the hemisphere about `n`.
    ///
    /// Returns `count` unit directions whose dot with `n` is in `[0, 1]`.
    fn hemisphere_dirs(n: Vec3, count: usize) -> Vec<Vec3> {
        // Build a tangent frame about n.
        let up = if n.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
        let t = normalize_or(up.cross(n), Vec3::X);
        let b = n.cross(t);
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            // z = cos(theta) uniform in (0, 1] -> upper hemisphere.
            let z = (i as f32 + 0.5) / count as f32;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let phi = core::f32::consts::TAU * (i as f32 * 0.618_034);
            let local = Vec3::new(r * ops::cos(phi), r * ops::sin(phi), z);
            out.push(t * local.x + b * local.y + n * local.z);
        }
        out
    }

    #[test]
    fn full_visibility_recovers_the_normal() {
        let n = Vec3::new(0.2, 0.3, 1.0).normalize();
        let dirs = hemisphere_dirs(n, 256);
        let samples: Vec<(Vec3, f32)> =
            dirs.iter().map(|&d| (d, 1.0)).collect();
        let cone = accumulate_bent_normal(&samples);
        // Bent normal aligns with the geometric normal...
        assert!(cone.direction.dot(n) > 0.99, "dir {:?} n {:?}", cone.direction, n);
        // ...it is unit length...
        assert!((cone.direction.length() - 1.0).abs() < 1e-5);
        // ...AO is ~1 (everything visible)...
        assert!((cone.visibility - 1.0).abs() < 1e-6, "vis {}", cone.visibility);
        // ...and a uniformly-sampled open hemisphere has a ~PI/2 aperture.
        assert!((cone.aperture - FRAC_PI_2).abs() < 0.1, "ap {}", cone.aperture);
    }

    #[test]
    fn half_space_occlusion_biases_toward_the_open_side() {
        let n = Vec3::Z;
        let dirs = hemisphere_dirs(n, 400);
        // Occlude the x < 0 half of the sky (visibility 0), keep x >= 0 open.
        let samples: Vec<(Vec3, f32)> = dirs
            .iter()
            .map(|&d| (d, if d.x >= 0.0 { 1.0 } else { 0.0 }))
            .collect();
        let cone = accumulate_bent_normal(&samples);
        // Bent normal leans toward +x (the open side) while staying in the
        // upper hemisphere.
        assert!(cone.direction.x > 0.05, "dir {:?}", cone.direction);
        assert!(cone.direction.z > 0.0, "dir {:?}", cone.direction);
        assert!((cone.direction.length() - 1.0).abs() < 1e-5);
        // Roughly half the hemisphere is blocked.
        assert!((cone.visibility - 0.5).abs() < 0.1, "vis {}", cone.visibility);
        // Narrower than the fully-open hemisphere (directions less spread out).
        assert!(cone.aperture < FRAC_PI_2, "ap {}", cone.aperture);
    }

    #[test]
    fn empty_sample_set_is_fully_occluded() {
        let cone = accumulate_bent_normal(&[]);
        assert_eq!(cone, BentNormalCone::OCCLUDED);
        assert_eq!(cone.visibility, 0.0);
        assert_eq!(cone.aperture, 0.0);
    }

    #[test]
    fn cancelling_directions_open_the_cone_fully() {
        // Two opposite, equally visible directions cancel -> no net direction.
        let samples = [(Vec3::X, 1.0), (Vec3::NEG_X, 1.0)];
        let cone = accumulate_bent_normal(&samples);
        assert!((cone.aperture - PI).abs() < 1e-5, "ap {}", cone.aperture);
        assert!((cone.visibility - 1.0).abs() < 1e-6);
        assert!((cone.direction.length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn aligned_directions_make_a_tight_cone() {
        // All visibility concentrated on one direction -> resultant length ~1
        // -> aperture ~0.
        let samples = [(Vec3::Z, 1.0), (Vec3::Z, 1.0), (Vec3::Z, 1.0)];
        let cone = accumulate_bent_normal(&samples);
        assert!(cone.direction.dot(Vec3::Z) > 0.999);
        assert!(cone.aperture < 1e-3, "ap {}", cone.aperture);
    }

    #[test]
    fn degenerate_direction_does_not_bias_but_counts_visibility() {
        // A zero-length direction contributes to the mean visibility only.
        let samples = [(Vec3::Z, 1.0), (Vec3::ZERO, 1.0)];
        let cone = accumulate_bent_normal(&samples);
        assert!(cone.direction.dot(Vec3::Z) > 0.999, "dir {:?}", cone.direction);
        // Mean of two unit visibilities.
        assert!((cone.visibility - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_hemisphere_all_visible_matches_normal() {
        let n = Vec3::new(-0.3, 0.7, 0.5).normalize();
        let dirs = hemisphere_dirs(n, 300);
        let samples: Vec<(Vec3, bool)> =
            dirs.iter().map(|&d| (d, false)).collect();
        let cone = bent_normal_from_cosine_hemisphere(n, &samples);
        assert!(cone.direction.dot(n) > 0.99, "dir {:?} n {:?}", cone.direction, n);
        assert!((cone.direction.length() - 1.0).abs() < 1e-5);
        assert!((cone.visibility - 1.0).abs() < 1e-6, "vis {}", cone.visibility);
    }

    #[test]
    fn cosine_hemisphere_half_occluded_biases_open_side() {
        let n = Vec3::Z;
        let dirs = hemisphere_dirs(n, 400);
        // Occlude the y < 0 half.
        let samples: Vec<(Vec3, bool)> =
            dirs.iter().map(|&d| (d, d.y < 0.0)).collect();
        let cone = bent_normal_from_cosine_hemisphere(n, &samples);
        assert!(cone.direction.y > 0.05, "dir {:?}", cone.direction);
        assert!(cone.direction.z > 0.0, "dir {:?}", cone.direction);
        assert!(cone.visibility > 0.3 && cone.visibility < 0.7, "vis {}", cone.visibility);
    }

    #[test]
    fn cosine_hemisphere_fully_occluded_closes_on_normal() {
        let n = Vec3::Y;
        let dirs = hemisphere_dirs(n, 128);
        let samples: Vec<(Vec3, bool)> =
            dirs.iter().map(|&d| (d, true)).collect();
        let cone = bent_normal_from_cosine_hemisphere(n, &samples);
        assert_eq!(cone.direction, n);
        assert_eq!(cone.aperture, 0.0);
        assert!((cone.visibility - 0.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_hemisphere_empty_defaults_to_open_sky() {
        let n = Vec3::new(1.0, 1.0, 1.0).normalize();
        let cone = bent_normal_from_cosine_hemisphere(n, &[]);
        assert!(cone.direction.dot(n) > 0.9999, "dir {:?} n {:?}", cone.direction, n);
        assert!((cone.aperture - FRAC_PI_2).abs() < 1e-6);
        assert!((cone.visibility - 1.0).abs() < 1e-6);
    }

    #[test]
    fn degenerate_normal_falls_back_to_up() {
        let cone = bent_normal_from_cosine_hemisphere(Vec3::ZERO, &[]);
        assert_eq!(cone.direction, Vec3::Y);
    }

    #[test]
    fn results_are_deterministic() {
        let samples = [
            (Vec3::new(0.1, 0.2, 0.9), 0.8f32),
            (Vec3::new(-0.3, 0.5, 0.7), 0.4),
            (Vec3::new(0.6, -0.1, 0.3), 0.9),
        ];
        let a = accumulate_bent_normal(&samples);
        let b = accumulate_bent_normal(&samples);
        assert_eq!(a, b);

        let flagged = [
            (Vec3::new(0.1, 0.2, 0.9), false),
            (Vec3::new(-0.3, 0.5, 0.7), true),
            (Vec3::new(0.6, -0.1, 0.3), false),
        ];
        let c = bent_normal_from_cosine_hemisphere(Vec3::Z, &flagged);
        let d = bent_normal_from_cosine_hemisphere(Vec3::Z, &flagged);
        assert_eq!(c, d);
    }

    #[test]
    fn outputs_are_always_finite_and_bounded() {
        // Sweep a mix of normal and degenerate inputs; nothing may be NaN and
        // every field must stay within its documented range.
        let dirs = [
            Vec3::ZERO,
            Vec3::X,
            Vec3::NEG_Y,
            Vec3::new(0.0, 0.0, 1e-30),
            Vec3::new(3.0, -2.0, 1.0),
        ];
        for &d in &dirs {
            for &v in &[-1.0f32, 0.0, 0.5, 2.0] {
                let cone = accumulate_bent_normal(&[(d, v)]);
                assert!(cone.direction.is_finite());
                assert!((cone.direction.length() - 1.0).abs() < 1e-4 || cone.direction == Vec3::Y);
                assert!(cone.aperture.is_finite() && (0.0..=PI + 1e-4).contains(&cone.aperture));
                assert!((0.0..=1.0).contains(&cone.visibility));
            }
        }
    }
}
