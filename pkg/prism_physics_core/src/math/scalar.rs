//! Scalar configuration for the physics engine.
//!
//! The engine is built around a single real-number type, [`Real`], so that the
//! precision can be changed in one place. M0 uses [`f32`] to match `glam`'s
//! single-precision `Vec3`/`Quat` types.

/// The real-number scalar type used throughout the physics core.
///
/// All positions, velocities, masses, and time steps are expressed in this
/// type. It is currently [`f32`] to align with the single-precision `glam`
/// linear-algebra types.
pub type Real = f32;

/// Archimedes' constant (π) as a [`Real`].
pub const PI: Real = core::f32::consts::PI;

/// The full-turn constant (τ = 2π) as a [`Real`].
pub const TAU: Real = core::f32::consts::TAU;

/// Machine epsilon for [`Real`], the smallest difference distinguishable near
/// `1.0`.
pub const EPSILON: Real = f32::EPSILON;

/// Returns `true` when `a` and `b` are within `eps` of each other.
///
/// This is an absolute-tolerance comparison (`|a - b| <= eps`), which is the
/// appropriate choice for values of similar magnitude such as coordinates and
/// velocities in a simulation.
#[must_use]
pub fn approx_eq(a: Real, b: Real, eps: Real) -> bool {
    (a - b).abs() <= eps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approx_eq_within_and_outside_tolerance() {
        assert!(approx_eq(1.0, 1.000_01, 1e-3));
        assert!(!approx_eq(1.0, 1.1, 1e-3));
        assert!(approx_eq(0.0, 0.0, 0.0));
    }

    #[test]
    fn constants_have_expected_relationships() {
        assert!(approx_eq(TAU, 2.0 * PI, 1e-6));
        assert!(core::hint::black_box(EPSILON) > 0.0);
    }
}
