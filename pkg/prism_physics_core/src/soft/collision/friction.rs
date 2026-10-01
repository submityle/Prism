//! Shared position-level Coulomb friction primitives for the contact passes.
//!
//! Both body-proxy collision and self-collision rub the tangential slide of a
//! resolved contact with position-level Coulomb friction (Macklin et al. 2014,
//! "Unified Particle Physics for Real-Time Applications"). The coefficient
//! sanitiser and the single-contact friction projection live here so the body
//! and self-collision resolvers share one implementation instead of each
//! carrying its own copy.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! tangential-friction projection is the one published by Macklin et al.
//! (2014).

use glam::Vec3;

use crate::math::scalar::Real;

/// Numerical floor below which a tangential slide is treated as zero, so a
/// friction correction is never normalised from a (near) zero-length vector.
pub(crate) const EPS_FRICTION: Real = 1e-12;

/// Returns `mu` clamped to `0..=1`, mapping any non-finite input to `0` so a
/// mis-authored coefficient can never inject a [`f32::NAN`] into a friction
/// pass.
#[must_use]
pub(crate) fn sanitize_friction(mu: Real) -> Real {
    if mu.is_finite() {
        mu.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Returns `pos` after applying position-level Coulomb friction against a
/// contact whose outward unit `normal` and normal-correction magnitude
/// `normal_push` are known.
///
/// This is the XPBD tangential-friction projection: the particle's tangential
/// slide over the frame (`dx_t`, i.e. `pos - prev` with its normal component
/// removed) is cancelled entirely inside the static-friction cone
/// (`||dx_t|| <= mu * ||dx_n||`) and otherwise shrunk by exactly
/// `mu * ||dx_n||`, leaving its direction unchanged. `mu` is the combined
/// friction coefficient and `normal_push` is `||dx_n||`, the depth the particle
/// was pushed out along `normal`.
///
/// A non-positive `mu`, a non-positive `normal_push`, or a tangential slide at
/// or below [`EPS_FRICTION`] leaves `pos` untouched, so a frictionless material
/// or a purely normal contact is a no-op and no [`f32::NAN`] is produced.
/// `normal` is assumed unit length; callers normalise it before calling.
#[must_use]
pub(crate) fn apply_coulomb_friction(
    pos: Vec3,
    prev: Vec3,
    normal: Vec3,
    normal_push: Real,
    mu: Real,
) -> Vec3 {
    if mu <= 0.0 || normal_push <= 0.0 {
        return pos;
    }
    let delta = pos - prev;
    let normal_amount = delta.dot(normal);
    let tangent = delta - normal * normal_amount;
    let tan_len_sq = tangent.length_squared();
    if tan_len_sq <= EPS_FRICTION {
        return pos;
    }
    let tan_len = tan_len_sq.sqrt();
    // `scale` is `min(mu * ||dx_n|| / ||dx_t||, 1)`: it saturates at 1 inside
    // the static cone (full cancellation) and is `< 1` in the dynamic regime
    // (shrink the slide by `mu * ||dx_n||`).
    let scale = (mu * normal_push / tan_len).min(1.0);
    pos - tangent * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: Real = 1.0e-6;

    #[test]
    fn sanitize_clamps_and_rejects_non_finite() {
        assert!((sanitize_friction(0.5) - 0.5).abs() < TOL);
        assert!((sanitize_friction(-1.0) - 0.0).abs() < TOL);
        assert!((sanitize_friction(2.0) - 1.0).abs() < TOL);
        assert!((sanitize_friction(Real::NAN) - 0.0).abs() < TOL);
        assert!((sanitize_friction(Real::INFINITY) - 0.0).abs() < TOL);
    }

    #[test]
    fn frictionless_or_normal_only_contact_is_a_no_op() {
        let pos = Vec3::new(1.0, 0.0, 0.0);
        let prev = Vec3::new(0.5, 0.3, 0.0);
        let n = Vec3::X;
        assert_eq!(apply_coulomb_friction(pos, prev, n, 1.0, 0.0), pos);
        assert_eq!(apply_coulomb_friction(pos, prev, n, 0.0, 0.5), pos);
    }

    #[test]
    fn static_cone_cancels_whole_tangential_slide() {
        // Normal is +X; slide has a small +Y tangential part well inside the
        // cone, so a unit friction coefficient removes it entirely.
        let prev = Vec3::new(0.0, 0.0, 0.0);
        let pos = Vec3::new(0.3, 0.01, 0.0);
        let out = apply_coulomb_friction(pos, prev, Vec3::X, 1.0, 1.0);
        assert!(out.y.abs() < TOL, "residual tangential y: {}", out.y);
        // Normal (X) component is untouched.
        assert!((out.x - pos.x).abs() < TOL);
    }

    #[test]
    fn dynamic_regime_shrinks_slide_by_mu_times_push() {
        // Large tangential slide outside the cone: shrink by exactly
        // `mu * normal_push`, direction preserved.
        let prev = Vec3::ZERO;
        let pos = Vec3::new(0.0, 1.0, 0.0); // tangential slide of length 1 along +Y
        let mu = 0.25;
        let push = 0.4;
        let out = apply_coulomb_friction(pos, prev, Vec3::X, push, mu);
        // Removed amount is mu*push = 0.1, so y goes 1.0 -> 0.9.
        assert!((out.y - (1.0 - mu * push)).abs() < TOL, "y: {}", out.y);
    }
}
