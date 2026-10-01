//! Deferred-decal G-buffer attribute blending (CPU golden reference).
//!
//! Once [`super::projection`] has established that a decal covers a pixel and
//! produced a coverage weight, the decal pass composites the decal's material
//! attributes over the G-buffer that lies beneath it.  This module is the
//! backend-neutral reference for that composite, covering the attribute classes
//! a deferred decal touches:
//!
//! * **Colour / scalar attributes** (albedo, roughness, metallic, ...) blend by
//!   either *alpha-over* (the decal replaces the base in proportion to its
//!   coverage) or *additive* (the decal adds energy on top of the base).  See
//!   [`DecalBlendMode`].
//! * **Normals** are reoriented and blended.  The decal stores a tangent-space
//!   perturbation whose `+Z` is the surface normal; it is lifted into world
//!   space on a basis built from the base normal, then blended toward the base
//!   by the coverage weight.  This is the "reorient then blend" (UDN-style)
//!   construction used by Unreal and the clustered-decal literature: a flat
//!   decal normal `(0, 0, 1)` leaves the base untouched for any coverage.
//!
//! # Conventions
//! * Pure, deterministic functions: no RNG, IO, GPU, or `unsafe`.
//! * Transcendental functions via [`bevy_math::ops`]; `sqrt` uses the inherent
//!   method.  No `f32::exp()`-style free functions.
//! * Storage layouts mirror the GPU twin (f32 fields, explicit alignment).
//! * `alpha` is the decal coverage weight; it is clamped to `[0, 1]` on entry so
//!   out-of-range coverage from an upstream bug cannot push attributes outside
//!   their domain.
//! * Colour and scalar results are clamped to be finite and non-negative;
//!   normals are always returned unit-length (falling back to the base normal,
//!   then to `+Z`, when every candidate degenerates).
//! * Defensive clamping everywhere so no `NaN`/`inf` ever escapes.

use bevy_math::Vec3;

/// Smallest squared length treated as a usable (non-degenerate) vector.
const MIN_LEN_SQ: f32 = 1.0e-12;

/// How a decal attribute is composited over the underlying G-buffer value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecalBlendMode {
    /// Linear interpolation `lerp(base, decal, alpha)`.
    ///
    /// `alpha = 0` keeps the base untouched; `alpha = 1` fully replaces it.
    AlphaOver,
    /// Additive `base + decal * alpha`.
    ///
    /// The decal adds energy on top of the base; the result is clamped to be
    /// finite and non-negative.
    Additive,
}

/// Clamps a decal coverage weight to the valid `[0, 1]` range.
///
/// Non-finite coverage collapses to `0` (the decal contributes nothing) so a
/// corrupt weight can never corrupt the composite.
#[inline]
fn sanitize_alpha(alpha: f32) -> f32 {
    if alpha.is_finite() {
        alpha.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Makes a scalar finite and non-negative, substituting `0` when non-finite.
#[inline]
fn sanitize_scalar(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

/// Blends a scalar G-buffer attribute (roughness, metallic, AO, ...).
///
/// The inputs are sanitised to be finite and non-negative; the result stays in
/// that domain.  For [`DecalBlendMode::AlphaOver`] the output lies between the
/// two inputs; for [`DecalBlendMode::Additive`] it grows monotonically with
/// `decal` and `alpha`.
#[inline]
pub fn blend_scalar(base: f32, decal: f32, alpha: f32, mode: DecalBlendMode) -> f32 {
    let base = sanitize_scalar(base);
    let decal = sanitize_scalar(decal);
    let a = sanitize_alpha(alpha);
    let out = match mode {
        DecalBlendMode::AlphaOver => base + (decal - base) * a,
        DecalBlendMode::Additive => base + decal * a,
    };
    sanitize_scalar(out)
}

/// Blends an albedo (or any linear RGB) G-buffer attribute component-wise.
///
/// Each channel follows [`blend_scalar`] with the same mode and alpha, so the
/// alpha-over result is bounded by the inputs and the additive result is
/// monotonic and non-negative per channel.
#[inline]
pub fn blend_albedo(base: Vec3, decal: Vec3, alpha: f32, mode: DecalBlendMode) -> Vec3 {
    Vec3::new(
        blend_scalar(base.x, decal.x, alpha, mode),
        blend_scalar(base.y, decal.y, alpha, mode),
        blend_scalar(base.z, decal.z, alpha, mode),
    )
}

/// Normalises `v`, returning `None` for a degenerate (near-zero) input.
#[inline]
fn try_normalize(v: Vec3) -> Option<Vec3> {
    let len_sq = v.length_squared();
    if len_sq.is_finite() && len_sq > MIN_LEN_SQ {
        Some(v * len_sq.sqrt().recip())
    } else {
        None
    }
}

/// Builds a right-handed orthonormal tangent/bitangent pair for a unit normal.
///
/// Uses Duff et al.'s branchless frame so the basis is stable and deterministic
/// across the whole sphere of normals.  Returns `(tangent, bitangent)` such that
/// `(tangent, bitangent, normal)` is orthonormal.
#[inline]
fn orthonormal_basis(n: Vec3) -> (Vec3, Vec3) {
    let sign = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let tangent = Vec3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let bitangent = Vec3::new(b, sign + n.y * n.y * a, -n.y);
    (tangent, bitangent)
}

/// Reorients a tangent-space decal normal onto a base normal and blends it.
///
/// `base_normal` is the geometric G-buffer normal (world space); `decal_normal`
/// is the decal's tangent-space perturbation, with `+Z` meaning "unperturbed"
/// (aligned with the surface).  The decal normal is lifted into world space on a
/// basis built from the base normal, then blended toward the base by `alpha` and
/// renormalised:
///
/// * `alpha = 0` returns the base normal unchanged.
/// * `alpha = 1` returns the fully reoriented decal normal.
/// * A flat decal normal `(0, 0, 1)` returns the base normal for any `alpha`.
///
/// The result is always unit length.  Degenerate inputs fall back to the base
/// normal, then to `+Z`, so the output is never `NaN` or zero.
#[inline]
pub fn blend_normal(base_normal: Vec3, decal_normal: Vec3, alpha: f32) -> Vec3 {
    let a = sanitize_alpha(alpha);
    let base = match try_normalize(base_normal) {
        Some(n) => n,
        None => return Vec3::Z,
    };

    // A degenerate decal normal means "no perturbation": keep the base.
    let ts = match try_normalize(decal_normal) {
        Some(n) => n,
        None => return base,
    };

    // Lift the tangent-space normal onto the base-normal frame.
    let (tangent, bitangent) = orthonormal_basis(base);
    let world_decal = (tangent * ts.x + bitangent * ts.y + base * ts.z).normalize_or_zero();
    let world_decal = if world_decal.length_squared() > 0.0 {
        world_decal
    } else {
        base
    };

    // Blend toward the base by coverage and renormalise.
    let blended = base + (world_decal - base) * a;
    match try_normalize(blended) {
        Some(n) => n,
        None => base,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn approx_vec(a: Vec3, b: Vec3, eps: f32) -> bool {
        approx(a.x, b.x, eps) && approx(a.y, b.y, eps) && approx(a.z, b.z, eps)
    }

    #[test]
    fn alpha_zero_keeps_base() {
        let base = Vec3::new(0.2, 0.4, 0.6);
        let decal = Vec3::new(0.9, 0.1, 0.3);
        let out = blend_albedo(base, decal, 0.0, DecalBlendMode::AlphaOver);
        assert!(approx_vec(out, base, 1.0e-6), "{out:?}");
        assert!(approx(
            blend_scalar(0.3, 0.8, 0.0, DecalBlendMode::AlphaOver),
            0.3,
            1.0e-6
        ));
    }

    #[test]
    fn alpha_one_replaces_base_in_alpha_over() {
        let base = Vec3::new(0.2, 0.4, 0.6);
        let decal = Vec3::new(0.9, 0.1, 0.3);
        let out = blend_albedo(base, decal, 1.0, DecalBlendMode::AlphaOver);
        assert!(approx_vec(out, decal, 1.0e-6), "{out:?}");
    }

    #[test]
    fn alpha_over_is_bounded_by_inputs() {
        let out = blend_scalar(0.2, 0.8, 0.5, DecalBlendMode::AlphaOver);
        assert!(approx(out, 0.5, 1.0e-6), "{out}");
    }

    #[test]
    fn additive_is_monotonic_in_decal_and_alpha() {
        let base = 0.1;
        let lo = blend_scalar(base, 0.2, 0.5, DecalBlendMode::Additive);
        let hi_decal = blend_scalar(base, 0.5, 0.5, DecalBlendMode::Additive);
        let hi_alpha = blend_scalar(base, 0.2, 1.0, DecalBlendMode::Additive);
        assert!(hi_decal > lo, "{hi_decal} !> {lo}");
        assert!(hi_alpha > lo, "{hi_alpha} !> {lo}");
        assert!(approx(
            blend_scalar(0.1, 0.4, 1.0, DecalBlendMode::Additive),
            0.5,
            1.0e-6
        ));
    }

    #[test]
    fn additive_clamps_non_negative() {
        // Sanitisation forces negative inputs to zero before compositing.
        let out = blend_scalar(-1.0, -1.0, 1.0, DecalBlendMode::Additive);
        assert!(out >= 0.0, "{out}");
    }

    #[test]
    fn flat_decal_normal_keeps_base() {
        let base = Vec3::new(0.0, 1.0, 0.0);
        let out = blend_normal(base, Vec3::Z, 1.0);
        assert!(approx_vec(out, base, 1.0e-6), "{out:?}");
    }

    #[test]
    fn blend_normal_alpha_zero_keeps_base() {
        let base = Vec3::new(0.3, 0.2, 0.9).normalize();
        let decal = Vec3::new(0.5, 0.5, 0.70710677);
        let out = blend_normal(base, decal, 0.0);
        assert!(approx_vec(out, base, 1.0e-6), "{out:?}");
    }

    #[test]
    fn blend_normal_is_unit_length() {
        let base = Vec3::new(0.1, 0.2, 0.97).normalize();
        let decal = Vec3::new(0.6, -0.3, 0.74).normalize();
        for i in 0..=10 {
            let a = i as f32 / 10.0;
            let out = blend_normal(base, decal, a);
            assert!(approx(out.length(), 1.0, 1.0e-5), "len = {}", out.length());
        }
    }

    #[test]
    fn blend_normal_alpha_one_matches_reoriented_decal() {
        let base = Vec3::Z;
        // With base = +Z the frame is the identity, so world == tangent space.
        let decal = Vec3::new(0.3, -0.4, 0.86602545).normalize();
        let out = blend_normal(base, decal, 1.0);
        assert!(approx_vec(out, decal, 1.0e-5), "{out:?}");
    }

    #[test]
    fn degenerate_normals_fall_back() {
        // Zero base normal -> +Z fallback.
        assert!(approx_vec(blend_normal(Vec3::ZERO, Vec3::Z, 1.0), Vec3::Z, 1.0e-6));
        // Zero decal normal -> base retained.
        let base = Vec3::new(0.0, 0.0, 1.0);
        assert!(approx_vec(blend_normal(base, Vec3::ZERO, 1.0), base, 1.0e-6));
    }

    #[test]
    fn non_finite_alpha_is_treated_as_zero() {
        let base = Vec3::new(0.2, 0.4, 0.6);
        let decal = Vec3::new(0.9, 0.1, 0.3);
        let out = blend_albedo(base, decal, f32::NAN, DecalBlendMode::AlphaOver);
        assert!(approx_vec(out, base, 1.0e-6), "{out:?}");
    }

    #[test]
    fn orthonormal_basis_is_orthonormal() {
        for n in [
            Vec3::Z,
            Vec3::NEG_Z,
            Vec3::new(0.3, -0.5, 0.81).normalize(),
            Vec3::new(-0.7, 0.2, -0.68).normalize(),
        ] {
            let (t, b) = orthonormal_basis(n);
            assert!(approx(t.length(), 1.0, 1.0e-5));
            assert!(approx(b.length(), 1.0, 1.0e-5));
            assert!(approx(t.dot(b), 0.0, 1.0e-5), "t.b = {}", t.dot(b));
            assert!(approx(t.dot(n), 0.0, 1.0e-5), "t.n = {}", t.dot(n));
            assert!(approx(b.dot(n), 0.0, 1.0e-5), "b.n = {}", b.dot(n));
        }
    }
}
