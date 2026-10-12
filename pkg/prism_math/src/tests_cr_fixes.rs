//! Regression tests for the CR-found core-math edge cases (fixed-point
//! truncation/saturation/remainder and unnormalized plane projection).

use crate::fixed::{Fixed, I16F16};
use crate::geom::aabb::Aabb3;
use crate::geom::frustum::Frustum;
use crate::geom::plane::Plane;
use crate::geom::ray::Ray3;
use crate::geom::sphere::BoundingSphere;
use crate::intersect::{ray_aabb, ray_sphere};
use crate::projection::perspective_infinite_rh;
use crate::quat::Quat;
use crate::vec::Vec3;

#[test]
fn fixed_to_int_truncates_toward_zero() {
    // -0.25 must truncate to 0, not floor to -1.
    assert_eq!(Fixed::from_bits(-(1i64 << 30)).to_int(), 0);
    assert_eq!(Fixed::from_bits(3 * (1i64 << 30)).to_int(), 0);
    assert_eq!(Fixed::from_bits(-3 * (1i64 << 30)).to_int(), 0);
    assert_eq!(Fixed::from_bits(3 * (1i64 << 31)).to_int(), 1);
    assert_eq!(Fixed::from_bits(-3 * (1i64 << 31)).to_int(), -1);
}

#[test]
fn i16f16_to_int_truncates_toward_zero() {
    assert_eq!(I16F16::from_bits(-(1 << 14)).to_int(), 0);
    assert_eq!(I16F16::from_bits(3 << 15).to_int(), 1);
    assert_eq!(I16F16::from_bits(-(3 << 15)).to_int(), -1);
}

#[test]
fn i16f16_from_fixed_saturates() {
    assert_eq!(I16F16::from_fixed(Fixed::MAX), I16F16::MAX);
    assert_eq!(I16F16::from_fixed(Fixed::MIN), I16F16::MIN);
    assert_eq!(
        I16F16::from_fixed(Fixed::ONE).to_bits(),
        I16F16::ONE.to_bits()
    );
}

#[test]
fn remainder_min_by_neg_one_is_zero() {
    assert_eq!((Fixed::MIN % Fixed::from_bits(-1)).to_bits(), 0);
    assert_eq!((I16F16::MIN % I16F16::from_bits(-1)).to_bits(), 0);
}

#[test]
fn plane_project_point_is_orthogonal_for_unnormalized_normal() {
    // Plane `2x = 2` (i.e. `x = 1`) with raw coefficient `2`.
    let pl = Plane::new(Vec3::new(2.0, 0.0, 0.0), -2.0);
    let q = pl.project_point(Vec3::ZERO);
    assert!(
        (q.x - 1.0).abs() < 1.0e-6 && q.y.abs() < 1.0e-6 && q.z.abs() < 1.0e-6,
        "unnormalized projection = {q:?}"
    );
    // A unit-normal plane behaves as before.
    let unit = Plane::from_point_normal(Vec3::new(2.0, 0.0, 0.0), Vec3::X);
    let q = unit.project_point(Vec3::ZERO);
    assert!((q.x - 2.0).abs() < 1.0e-6, "unit projection = {q:?}");
}

#[test]
fn infinite_far_frustum_classifies_visible_points() {
    // An infinite-far projection has an always-satisfied far plane; its zero
    // clip row must not become a NaN plane (CR-found).
    let vp = perspective_infinite_rh(core::f32::consts::FRAC_PI_2, 1.0, 1.0);
    let f = Frustum::from_view_proj(vp);
    // In front of the camera (looking down -Z), beyond the near plane: inside.
    assert!(f.contains_point(Vec3::new(0.0, 0.0, -5.0)));
    // Behind the camera: outside the near plane.
    assert!(!f.contains_point(Vec3::new(0.0, 0.0, 5.0)));
}

#[test]
fn vec3_normalize_handles_extreme_magnitudes() {
    // Squared length overflows to +inf: must still yield a unit vector.
    let n = Vec3::new(1.0e20, 0.0, 0.0).normalize();
    assert!((n.x - 1.0).abs() < 1.0e-6, "big normalize = {n:?}");
    // Squared length underflows to 0: must still yield a unit vector.
    let n = Vec3::new(1.0e-30, 0.0, 0.0).normalize();
    assert!((n.x - 1.0).abs() < 1.0e-6, "small normalize = {n:?}");
    // Normal range is unchanged.
    let n = Vec3::new(3.0, 4.0, 0.0).normalize();
    assert!((n.x - 0.6).abs() < 1.0e-6 && (n.y - 0.8).abs() < 1.0e-6);
}

#[test]
fn quat_normalize_handles_extreme_magnitudes() {
    let q = Quat::from_xyzw(0.0, 0.0, 0.0, 1.0e20).normalize();
    assert!((q.w - 1.0).abs() < 1.0e-6, "big quat normalize = {q:?}");
    let q = Quat::from_xyzw(0.0, 0.0, 0.0, 1.0e-30).normalize();
    assert!((q.w - 1.0).abs() < 1.0e-6, "small quat normalize = {q:?}");
}

#[test]
fn degenerate_rays_are_rejected() {
    // Zero-length direction vs a box: no valid hit (was `t = inf`, NaN point).
    let z = Vec3::ZERO;
    assert!(ray_aabb(Ray3::new(z, z), Aabb3::new(-Vec3::ONE, Vec3::ONE)).is_none());
    // Zero-radius sphere: no valid hit (was a NaN surface normal).
    assert!(ray_sphere(
        Ray3::new(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)),
        BoundingSphere::new(Vec3::ZERO, 0.0),
    )
    .is_none());
}
