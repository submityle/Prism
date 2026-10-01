//! Finite/range sanitizers shared by the aero inputs.
//!
//! Every aero coefficient and wind component is pushed through one of these
//! before it reaches the force math, so a `NaN`, infinity, or out-of-range
//! value from authoring data can never propagate into particle velocities.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**.

use glam::Vec3;

use crate::math::scalar::Real;

/// Replaces a non-finite scalar with `0`, leaving finite values unchanged.
#[must_use]
pub(super) fn sanitize_finite(x: Real) -> Real {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Replaces every non-finite component of a vector with `0`.
#[must_use]
pub(super) fn sanitize_vec(v: Vec3) -> Vec3 {
    Vec3::new(
        sanitize_finite(v.x),
        sanitize_finite(v.y),
        sanitize_finite(v.z),
    )
}

/// Clamps a scalar to `0..=1`, mapping any non-finite input to `0`.
#[must_use]
pub(super) fn sanitize_unit(x: Real) -> Real {
    if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Clamps a scalar to be non-negative, mapping any non-finite input to `0`.
#[must_use]
pub(super) fn sanitize_non_negative(x: Real) -> Real {
    if x.is_finite() {
        x.max(0.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_passthrough_and_nan_zeroing() {
        assert_eq!(sanitize_finite(2.5), 2.5);
        assert_eq!(sanitize_finite(Real::NAN), 0.0);
        assert_eq!(sanitize_finite(Real::INFINITY), 0.0);
    }

    #[test]
    fn vec_zeroes_nonfinite_components() {
        let v = sanitize_vec(Vec3::new(Real::NAN, 1.0, Real::NEG_INFINITY));
        assert_eq!(v, Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn unit_clamps_to_zero_one() {
        assert_eq!(sanitize_unit(-1.0), 0.0);
        assert_eq!(sanitize_unit(2.0), 1.0);
        assert_eq!(sanitize_unit(Real::NAN), 0.0);
        assert!((sanitize_unit(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn non_negative_floors_at_zero() {
        assert_eq!(sanitize_non_negative(-3.0), 0.0);
        assert_eq!(sanitize_non_negative(Real::NAN), 0.0);
        assert!((sanitize_non_negative(4.0) - 4.0).abs() < 1e-6);
    }
}
