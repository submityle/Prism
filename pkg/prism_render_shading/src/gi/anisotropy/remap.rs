//! Anisotropy parameter remapping and tangent-frame construction — CPU golden.
//!
//! Translates artist-facing material controls into the GGX widths consumed by
//! [`crate::gi::anisotropy::ndf`] and [`crate::gi::anisotropy::sample`], and
//! builds the orthonormal shading frame the anisotropic lobe is expressed in.
//!
//! It provides:
//!
//! * the Burley / Disney anisotropy remap
//!   `aspect = sqrt(1 - 0.9 · anisotropy)`, `α_t = α / aspect`,
//!   `α_b = α · aspect`, where `α = roughness²`, and
//! * [`orthonormal_tangent_frame`], which re-orthogonalises a surface tangent
//!   against the shading normal with Gram–Schmidt and completes a right-handed
//!   basis `(tangent, bitangent, normal)`.
//!
//! # Conventions
//! * `roughness ∈ [0, 1]` maps to the isotropic GGX width `α = roughness²`
//!   (Burley/Disney), floored at [`MIN_ALPHA`].
//! * `anisotropy ∈ [0, 1]` controls directionality.  `0` is isotropic
//!   (`α_t == α_b`); increasing it *stretches the highlight along the tangent*
//!   by widening `α_t` and narrowing `α_b`.  The input is clamped to `[0, 1)`
//!   via the `aspect` floor so the mapping stays finite.
//! * The returned tangent frame is orthonormal and right-handed:
//!   `bitangent = normal × tangent`, `tangent = bitangent × normal`.  Degenerate
//!   inputs (zero-length or tangent parallel to the normal) fall back to a
//!   deterministic frame built from the normal alone, never producing `NaN`.
//! * Every helper is a deterministic pure function (no RNG, I/O, GPU, globals or
//!   `unsafe`); `f32` arithmetic mirrors the WESL/GPU twin bit-for-bit.
//!
//! # References
//! * Burley 2012, *Physically-Based Shading at Disney* — the `aspect`
//!   anisotropy remap.
//! * Duff et al. 2017, *Building an Orthonormal Basis, Revisited* (JCGT) — the
//!   branchless normal-only frame used as the degenerate fallback.

use bevy_math::{Vec3, ops};

/// Minimum GGX `alpha`, shared with [`crate::gi::anisotropy::ndf::MIN_ALPHA`].
pub use crate::gi::anisotropy::ndf::MIN_ALPHA;

/// Smallest `aspect` ratio.  Guards `α_t = α / aspect` against division by zero
/// as `anisotropy → 1`.
const MIN_ASPECT: f32 = 1.0e-4;

/// Maps a perceptual `roughness ∈ [0, 1]` to the isotropic GGX width
/// `α = roughness²` (Disney/Burley), floored at [`MIN_ALPHA`].
#[inline]
pub fn roughness_to_alpha(roughness: f32) -> f32 {
    let r = roughness.clamp(0.0, 1.0);
    (r * r).max(MIN_ALPHA)
}

/// Disney `aspect` ratio for a given `anisotropy ∈ [0, 1]`:
/// `aspect = sqrt(1 - 0.9 · anisotropy)`, floored at [`MIN_ASPECT`].
///
/// `aspect = 1` at `anisotropy = 0` and shrinks towards `sqrt(0.1) ≈ 0.316` at
/// `anisotropy = 1`.
#[inline]
pub fn anisotropy_aspect(anisotropy: f32) -> f32 {
    let a = anisotropy.clamp(0.0, 1.0);
    (1.0 - 0.9 * a).max(MIN_ASPECT).sqrt()
}

/// Remaps `(roughness, anisotropy)` to the anisotropic GGX widths
/// `(α_t, α_b)` along the tangent / bitangent.
///
/// `α = roughness²`, `aspect = sqrt(1 - 0.9 · anisotropy)`,
/// `α_t = α / aspect`, `α_b = α · aspect`.  Both outputs are floored at
/// [`MIN_ALPHA`]; `anisotropy = 0` yields `α_t == α_b == α`.
#[inline]
pub fn anisotropy_to_alpha(roughness: f32, anisotropy: f32) -> (f32, f32) {
    let alpha = roughness_to_alpha(roughness);
    let aspect = anisotropy_aspect(anisotropy);
    let alpha_t = (alpha / aspect).max(MIN_ALPHA);
    let alpha_b = (alpha * aspect).max(MIN_ALPHA);
    (alpha_t, alpha_b)
}

/// Recovers an approximate `(roughness, anisotropy)` from GGX widths
/// `(α_t, α_b)` — the inverse of [`anisotropy_to_alpha`] when both widths are
/// above [`MIN_ALPHA`].
///
/// Uses `α = sqrt(α_t · α_b)` so `roughness = sqrt(α)` (the geometric mean is
/// invariant to the `aspect` split), and `aspect = sqrt(α_b / α_t)` inverted
/// through `anisotropy = (1 - aspect²) / 0.9`.  The result is clamped to
/// `[0, 1]²`; it is primarily a diagnostic / round-trip helper.
#[inline]
pub fn alpha_to_anisotropy(alpha_t: f32, alpha_b: f32) -> (f32, f32) {
    let at = alpha_t.max(MIN_ALPHA);
    let ab = alpha_b.max(MIN_ALPHA);
    let alpha = (at * ab).max(MIN_ALPHA).sqrt();
    let roughness = alpha.max(0.0).sqrt().clamp(0.0, 1.0);
    let aspect2 = (ab / at).clamp(MIN_ASPECT * MIN_ASPECT, 1.0);
    let anisotropy = ((1.0 - aspect2) / 0.9).clamp(0.0, 1.0);
    (roughness, anisotropy)
}

/// A right-handed orthonormal shading frame.  `tangent` and `bitangent` span the
/// surface; `normal` is the geometric/shading normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TangentFrame {
    /// Unit tangent — the `α_t` ("stretch") axis of the anisotropic lobe.
    pub tangent: Vec3,
    /// Unit bitangent — the `α_b` axis, equal to `normal × tangent`.
    pub bitangent: Vec3,
    /// Unit shading normal (`+Z` of the frame).
    pub normal: Vec3,
}

/// Builds a right-handed orthonormal [`TangentFrame`] from a shading `normal`
/// and a desired surface `tangent`, re-orthogonalising with Gram–Schmidt.
///
/// The tangent is projected to be perpendicular to the normal and renormalised;
/// the bitangent is `normal × tangent`.  If either input is degenerate (zero
/// length, or the tangent is parallel to the normal), a deterministic
/// normal-only basis (Duff et al. 2017) is used so the result is always a valid
/// orthonormal frame with no `NaN`.
#[inline]
pub fn orthonormal_tangent_frame(normal: Vec3, tangent: Vec3) -> TangentFrame {
    let n = normal.normalize_or_zero();
    let n = if n.length_squared() > 0.0 { n } else { Vec3::Z };

    // Gram–Schmidt: remove the normal component from the tangent.
    let t_proj = tangent - n * n.dot(tangent);
    let t = t_proj.normalize_or_zero();

    if t.length_squared() > 0.0 {
        // Perpendicular tangent recovered; complete a right-handed basis.
        let bitangent = n.cross(t).normalize_or_zero();
        if bitangent.length_squared() <= 0.0 {
            // Numerically parallel after projection — fall back.
            return frame_from_normal(n);
        }
        // Re-derive the tangent from bitangent × normal to guarantee exact
        // orthogonality against both axes.
        let tangent = bitangent.cross(n).normalize_or_zero();
        let tangent = if tangent.length_squared() > 0.0 { tangent } else { t };
        TangentFrame { tangent, bitangent, normal: n }
    } else {
        frame_from_normal(n)
    }
}

/// Branchless normal-only orthonormal basis (Duff et al. 2017), used when the
/// caller's tangent is unusable.  `normal` must already be unit length.
#[inline]
fn frame_from_normal(normal: Vec3) -> TangentFrame {
    let n = normal;
    let sign = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let tangent = Vec3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let tangent = tangent.normalize_or_zero();
    let tangent = if tangent.length_squared() > 0.0 { tangent } else { Vec3::X };
    let bitangent = n.cross(tangent).normalize_or_zero();
    let bitangent = if bitangent.length_squared() > 0.0 { bitangent } else { Vec3::Y };
    TangentFrame { tangent, bitangent, normal: n }
}

/// Rotates the `tangent` of a frame about its `normal` by `angle` radians,
/// returning a new right-handed orthonormal frame.
///
/// Models the glTF `KHR_materials_anisotropy` *rotation* control, which spins
/// the anisotropy direction within the surface plane.  The rotation stays in
/// the tangent plane, so the result is re-orthonormalised through
/// [`orthonormal_tangent_frame`] and can never leave the surface or produce
/// `NaN`.
#[inline]
pub fn rotate_tangent_frame(frame: TangentFrame, angle: f32) -> TangentFrame {
    let (sin_a, cos_a) = ops::sin_cos(angle);
    // Rotate within the (tangent, bitangent) plane.
    let rotated = frame.tangent * cos_a + frame.bitangent * sin_a;
    orthonormal_tangent_frame(frame.normal, rotated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isotropic_when_anisotropy_zero() {
        for roughness in [0.1f32, 0.3, 0.6, 1.0] {
            let (at, ab) = anisotropy_to_alpha(roughness, 0.0);
            assert!((at - ab).abs() < 1e-6, "at={at} ab={ab}");
            let alpha = roughness_to_alpha(roughness);
            assert!((at - alpha).abs() < 1e-6, "at={at} alpha={alpha}");
        }
    }

    #[test]
    fn positive_anisotropy_stretches_tangent_axis() {
        let (at, ab) = anisotropy_to_alpha(0.5, 0.8);
        assert!(at > ab, "at={at} ab={ab}");
        // Geometric mean is preserved by the aspect split.
        let alpha = roughness_to_alpha(0.5);
        assert!(((at * ab).sqrt() - alpha).abs() < 1e-5, "mean drifted");
    }

    #[test]
    fn alpha_floor_holds_at_mirror() {
        let (at, ab) = anisotropy_to_alpha(0.0, 1.0);
        assert!(at >= MIN_ALPHA && ab >= MIN_ALPHA, "at={at} ab={ab}");
        assert!(at.is_finite() && ab.is_finite());
    }

    #[test]
    fn aspect_endpoints() {
        assert!((anisotropy_aspect(0.0) - 1.0).abs() < 1e-6);
        assert!((anisotropy_aspect(1.0) - 0.1f32.sqrt()).abs() < 1e-4);
    }

    #[test]
    fn roundtrip_recovers_inputs() {
        for &(r, a) in &[(0.4f32, 0.0f32), (0.6, 0.5), (0.3, 0.8)] {
            let (at, ab) = anisotropy_to_alpha(r, a);
            let (r2, a2) = alpha_to_anisotropy(at, ab);
            assert!((r - r2).abs() < 2e-3, "r={r} r2={r2}");
            assert!((a - a2).abs() < 2e-3, "a={a} a2={a2}");
        }
    }

    #[test]
    fn frame_is_orthonormal_right_handed() {
        let cases = [
            (Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 0.0)),
            (Vec3::new(0.2, 0.3, 0.9), Vec3::new(1.0, 0.0, 0.0)),
            (Vec3::new(-0.5, 0.5, 0.7), Vec3::new(0.0, 1.0, 0.2)),
        ];
        for &(n_in, t_in) in &cases {
            let f = orthonormal_tangent_frame(n_in, t_in);
            assert!((f.tangent.length() - 1.0).abs() < 1e-5);
            assert!((f.bitangent.length() - 1.0).abs() < 1e-5);
            assert!((f.normal.length() - 1.0).abs() < 1e-5);
            assert!(f.tangent.dot(f.normal).abs() < 1e-5, "t·n");
            assert!(f.tangent.dot(f.bitangent).abs() < 1e-5, "t·b");
            assert!(f.bitangent.dot(f.normal).abs() < 1e-5, "b·n");
            // Right-handed: t × b == n.
            assert!(f.tangent.cross(f.bitangent).dot(f.normal) > 0.99, "handedness");
        }
    }

    #[test]
    fn tangent_direction_is_preserved_when_perpendicular() {
        // A tangent already perpendicular to the normal must survive unchanged
        // (up to sign-stable renormalisation).
        let n = Vec3::Z;
        let t_in = Vec3::new(0.6, 0.8, 0.0).normalize();
        let f = orthonormal_tangent_frame(n, t_in);
        assert!(f.tangent.dot(t_in) > 0.999, "tangent drifted: {:?}", f.tangent);
    }


    #[test]
    fn rotating_tangent_by_half_pi_lands_on_bitangent() {
        let base = orthonormal_tangent_frame(Vec3::Z, Vec3::X);
        let rot = rotate_tangent_frame(base, core::f32::consts::FRAC_PI_2);
        // A quarter turn maps the tangent onto the original bitangent.
        assert!(rot.tangent.dot(base.bitangent) > 0.999, "t={:?}", rot.tangent);
        assert!(rot.tangent.dot(base.normal).abs() < 1e-5);
        assert!((rot.tangent.length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn degenerate_tangent_falls_back_to_valid_frame() {
        // Tangent parallel to the normal, and a zero tangent: both must yield a
        // finite orthonormal frame rather than NaN.
        for &(n, t) in &[
            (Vec3::Z, Vec3::Z),
            (Vec3::new(0.0, 0.0, 1.0), Vec3::ZERO),
            (Vec3::new(0.1, 0.2, 0.97).normalize(), Vec3::new(0.1, 0.2, 0.97)),
        ] {
            let f = orthonormal_tangent_frame(n, t);
            assert!(f.tangent.is_finite() && f.bitangent.is_finite() && f.normal.is_finite());
            assert!((f.tangent.length() - 1.0).abs() < 1e-4);
            assert!(f.tangent.dot(f.normal).abs() < 1e-4);
            assert!(f.bitangent.dot(f.normal).abs() < 1e-4);
        }
    }
}
