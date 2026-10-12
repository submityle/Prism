//! Regression tests for the CR-found core-math edge cases (fixed-point
//! truncation/saturation/remainder and unnormalized plane projection).

use crate::fixed::{Fixed, I16F16};
use crate::geom::plane::Plane;
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
