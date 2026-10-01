//! Surfels: world-space oriented discs and their coverage / weight kernels.
//!
//! A *surfel* (surface element) is the atomic cache primitive of a Lumen-style
//! surface cache: a small oriented disc anchored to the geometry at a world
//! position, carrying a surface normal and a world-space radius.  Shading
//! points gather radiance from the surfels that *cover* them, where coverage is
//! a smooth kernel built from three independent terms — how far the point lies
//! inside the disc radius (radial), how far it floats off the disc plane
//! (axial), and how closely the point normal agrees with the surfel normal
//! (orientation).  The same geometric weight drives spatial reuse between
//! neighbouring surfels in [`super::integration`].
//!
//! # Conventions
//! * Positions and normals are right-handed world-space `Vec3`; normals are
//!   normalised defensively on construction (zero / non-finite normals collapse
//!   to `+Z`) and the radius is clamped to a small positive epsilon so the
//!   kernels never divide by zero.
//! * Every weight is finite and in `[0, 1]`; degenerate inputs fall back to a
//!   safe value (zero coverage, identity orientation) rather than emitting
//!   `NaN`.  A point exactly at the disc centre with a matching normal has
//!   coverage `1`; a point beyond the radius has radial weight `0`.
//! * Transcendentals go through [`bevy_math::ops`] (never `f32::exp`), matching
//!   the crate-wide no-`std` numerical contract.  `sqrt` uses the inherent
//!   `f32::sqrt`.
//! * All helpers are deterministic pure functions (no RNG / IO / GPU / unsafe).

use bevy_math::{ops, Vec3};

/// Smallest surfel radius; keeps every radial / axial division finite.
const MIN_RADIUS: f32 = 1.0e-6;

/// A world-space oriented disc: the atomic surface-cache element.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Surfel {
    /// World-space anchor (disc centre).
    pub position: Vec3,
    /// Unit surface normal (disc axis).
    pub normal: Vec3,
    /// Disc radius in world units (strictly positive after construction).
    pub radius: f32,
}

/// Tunables for the surfel coverage kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoverageParams {
    /// Orientation exponent: larger values narrow the normal-agreement lobe.
    pub normal_sharpness: f32,
    /// Off-plane tolerance expressed as a fraction of the surfel radius; the
    /// axial weight falls off as `exp(-|axial| / (radius * axial_tolerance))`.
    pub axial_tolerance: f32,
}

impl Default for CoverageParams {
    fn default() -> Self {
        // Balanced defaults: a moderately wide orientation lobe and an
        // off-plane tolerance of a quarter radius.
        Self {
            normal_sharpness: 8.0,
            axial_tolerance: 0.25,
        }
    }
}

impl Surfel {
    /// Construct a surfel, normalising the normal and clamping the radius.
    ///
    /// A zero-length or non-finite normal collapses to `+Z`; a non-finite or
    /// non-positive radius clamps to [`MIN_RADIUS`] so coverage stays finite.
    #[must_use]
    pub fn new(position: Vec3, normal: Vec3, radius: f32) -> Self {
        Self {
            position: sanitize_vec(position),
            normal: safe_normalize(normal),
            radius: sanitize_radius(radius),
        }
    }

    /// Signed distance from `point` to the disc plane (along the normal).
    ///
    /// Positive in front of the disc (normal side), negative behind it.
    #[must_use]
    pub fn signed_plane_distance(&self, point: Vec3) -> f32 {
        (sanitize_vec(point) - self.position).dot(self.normal)
    }

    /// In-plane (radial) distance from the disc centre to `point`.
    ///
    /// This is the length of the component of `point - position` that lies in
    /// the disc plane; it is always non-negative and finite.
    #[must_use]
    pub fn radial_distance(&self, point: Vec3) -> f32 {
        let delta = sanitize_vec(point) - self.position;
        let axial = delta.dot(self.normal);
        let in_plane = delta - self.normal * axial;
        let len_sq = in_plane.length_squared();
        if len_sq > 0.0 {
            len_sq.sqrt()
        } else {
            0.0
        }
    }

    /// Whether `point` falls within the disc radius (ignoring plane distance).
    #[must_use]
    pub fn covers(&self, point: Vec3) -> bool {
        self.radial_distance(point) <= self.radius
    }

    /// Radial coverage weight in `[0, 1]`: `max(0, 1 - (r / radius)^2)`.
    ///
    /// `1` at the disc centre, smoothly falling to `0` at the rim and staying
    /// `0` beyond it.
    #[must_use]
    pub fn radial_weight(&self, point: Vec3) -> f32 {
        let t = (self.radial_distance(point) / self.radius).clamp(0.0, 1.0);
        (1.0 - t * t).clamp(0.0, 1.0)
    }

    /// Axial coverage weight in `[0, 1]`: `exp(-|axial| / (radius * tol))`.
    ///
    /// Penalises points that float off the disc plane; `tol` is the
    /// [`CoverageParams::axial_tolerance`] expressed as a fraction of the
    /// radius.
    #[must_use]
    pub fn axial_weight(&self, point: Vec3, axial_tolerance: f32) -> f32 {
        let axial = self.signed_plane_distance(point).abs();
        let scale = self.radius * axial_tolerance.max(0.0);
        let denom = scale.max(MIN_RADIUS);
        stable_exp(-axial / denom)
    }

    /// Full coverage weight of this surfel for a shading point in `[0, 1]`.
    ///
    /// Product of the radial, axial, and orientation terms.  `point_normal` is
    /// the shading point's surface normal; a mismatched orientation shrinks the
    /// weight via [`normal_consistency`].  The result is always finite.
    #[must_use]
    pub fn coverage(&self, point: Vec3, point_normal: Vec3, params: &CoverageParams) -> f32 {
        let w_radial = self.radial_weight(point);
        if w_radial <= 0.0 {
            return 0.0;
        }
        let w_axial = self.axial_weight(point, params.axial_tolerance);
        let w_normal = normal_consistency(self.normal, point_normal, params.normal_sharpness);
        (w_radial * w_axial * w_normal).clamp(0.0, 1.0)
    }

    /// Geometric reuse weight against another surfel in `[0, 1]`.
    ///
    /// Combines how far the other surfel's centre lies off *this* disc
    /// (radial + axial, using this surfel's radius) with their normal
    /// agreement.  Drives bilateral spatial filtering between neighbours.
    #[must_use]
    pub fn geometric_weight(&self, other: &Surfel, params: &CoverageParams) -> f32 {
        let w_radial = self.radial_weight(other.position);
        if w_radial <= 0.0 {
            return 0.0;
        }
        let w_axial = self.axial_weight(other.position, params.axial_tolerance);
        let w_normal = normal_consistency(self.normal, other.normal, params.normal_sharpness);
        (w_radial * w_axial * w_normal).clamp(0.0, 1.0)
    }
}

/// Orientation-agreement weight in `[0, 1]`: `max(0, dot(a, b))^sharpness`.
///
/// Both vectors are normalised defensively; a non-positive `sharpness` yields a
/// binary front/back test (`cos^0 == 1` for any non-opposed pair).  Used both
/// for coverage and for neighbour reuse.
#[must_use]
pub fn normal_consistency(a: Vec3, b: Vec3, sharpness: f32) -> f32 {
    let cosine = safe_normalize(a).dot(safe_normalize(b)).clamp(0.0, 1.0);
    ops::powf(cosine, sharpness.max(0.0)).clamp(0.0, 1.0)
}

/// Normalise a vector, falling back to `+Z` for zero-length / non-finite input.
#[must_use]
fn safe_normalize(v: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq.is_finite() && len_sq > 1.0e-24 {
        v / len_sq.sqrt()
    } else {
        Vec3::Z
    }
}

/// Replace any non-finite component of a position with `0`.
#[must_use]
fn sanitize_vec(v: Vec3) -> Vec3 {
    Vec3::new(
        finite_or_zero(v.x),
        finite_or_zero(v.y),
        finite_or_zero(v.z),
    )
}

/// Clamp a radius to a strictly positive, finite value.
#[must_use]
fn sanitize_radius(r: f32) -> f32 {
    if r.is_finite() && r > MIN_RADIUS {
        r
    } else {
        MIN_RADIUS
    }
}

#[must_use]
fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Numerically safe `exp` that never returns `NaN` and saturates for large
/// magnitudes.
#[must_use]
fn stable_exp(x: f32) -> f32 {
    if !x.is_finite() {
        return 0.0;
    }
    ops::exp(x.clamp(-80.0, 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_disc() -> Surfel {
        Surfel::new(Vec3::ZERO, Vec3::Z, 1.0)
    }

    #[test]
    fn constructor_sanitizes_normal_and_radius() {
        let s = Surfel::new(Vec3::ZERO, Vec3::ZERO, -5.0);
        assert!((s.normal.length() - 1.0).abs() < 1e-6);
        assert!(s.radius > 0.0 && s.radius.is_finite());
        let bad = Surfel::new(
            Vec3::new(f32::NAN, 1.0, 2.0),
            Vec3::new(f32::INFINITY, 0.0, 0.0),
            f32::NAN,
        );
        assert!(bad.position.is_finite());
        assert!((bad.normal.length() - 1.0).abs() < 1e-6);
        assert!(bad.radius > 0.0);
    }

    #[test]
    fn coverage_is_one_at_centre() {
        let s = unit_disc();
        let w = s.coverage(Vec3::ZERO, Vec3::Z, &CoverageParams::default());
        assert!((w - 1.0).abs() < 1e-6, "w = {w}");
    }

    #[test]
    fn coverage_zero_beyond_radius() {
        let s = unit_disc();
        // Point two radii out in-plane: radial weight collapses to zero.
        let w = s.coverage(Vec3::new(2.0, 0.0, 0.0), Vec3::Z, &CoverageParams::default());
        assert_eq!(w, 0.0);
        assert!(!s.covers(Vec3::new(2.0, 0.0, 0.0)));
        assert!(s.covers(Vec3::new(0.5, 0.0, 0.0)));
    }

    #[test]
    fn coverage_decreases_off_plane() {
        let s = unit_disc();
        let params = CoverageParams::default();
        let near = s.coverage(Vec3::new(0.0, 0.0, 0.05), Vec3::Z, &params);
        let far = s.coverage(Vec3::new(0.0, 0.0, 0.5), Vec3::Z, &params);
        assert!(near > far, "near {near} should exceed far {far}");
        assert!(far >= 0.0);
    }

    #[test]
    fn coverage_decreases_with_normal_mismatch() {
        let s = unit_disc();
        let params = CoverageParams::default();
        let aligned = s.coverage(Vec3::ZERO, Vec3::Z, &params);
        let tilted = s.coverage(Vec3::ZERO, Vec3::new(1.0, 0.0, 1.0), &params);
        let opposed = s.coverage(Vec3::ZERO, Vec3::NEG_Z, &params);
        assert!(aligned > tilted, "{aligned} vs {tilted}");
        assert!(tilted > opposed, "{tilted} vs {opposed}");
        assert_eq!(opposed, 0.0);
    }

    #[test]
    fn radial_and_plane_distance_decompose_offset() {
        let s = Surfel::new(Vec3::new(1.0, 2.0, 3.0), Vec3::Z, 2.0);
        let p = Vec3::new(1.0 + 0.3, 2.0 - 0.4, 3.0 + 1.5);
        // In-plane offset is (0.3, -0.4) -> length 0.5; axial is +1.5.
        assert!((s.radial_distance(p) - 0.5).abs() < 1e-6);
        assert!((s.signed_plane_distance(p) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn normal_consistency_bounds() {
        assert!((normal_consistency(Vec3::Z, Vec3::Z, 8.0) - 1.0).abs() < 1e-6);
        assert_eq!(normal_consistency(Vec3::Z, Vec3::NEG_Z, 8.0), 0.0);
        // sharpness 0 => any non-opposed pair weighs 1.
        assert!((normal_consistency(Vec3::Z, Vec3::new(1.0, 0.0, 1.0), 0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn geometric_weight_peaks_for_coincident_surfels() {
        let a = unit_disc();
        let b = Surfel::new(Vec3::new(0.1, 0.0, 0.0), Vec3::Z, 1.0);
        let far = Surfel::new(Vec3::new(5.0, 0.0, 0.0), Vec3::Z, 1.0);
        let params = CoverageParams::default();
        assert!(a.geometric_weight(&b, &params) > 0.5);
        assert_eq!(a.geometric_weight(&far, &params), 0.0);
    }

    #[test]
    fn weights_finite_on_degenerate_inputs() {
        let s = Surfel::new(Vec3::splat(f32::NAN), Vec3::ZERO, f32::NAN);
        let w = s.coverage(
            Vec3::splat(f32::INFINITY),
            Vec3::splat(f32::NAN),
            &CoverageParams::default(),
        );
        assert!(w.is_finite());
        assert!((0.0..=1.0).contains(&w));
    }

    #[test]
    fn determinism() {
        let s = unit_disc();
        let params = CoverageParams::default();
        let a = s.coverage(Vec3::new(0.2, 0.1, 0.1), Vec3::new(0.0, 0.1, 1.0), &params);
        let b = s.coverage(Vec3::new(0.2, 0.1, 0.1), Vec3::new(0.0, 0.1, 1.0), &params);
        assert_eq!(a, b);
    }
}
