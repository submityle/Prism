//! Reconnection shift mapping and its geometric Jacobian — CPU golden.
//!
//! Path-space `ReSTIR` reuses a neighbour's path by *shifting* it into the
//! current pixel's domain.  The *reconnection shift* (Lin et al. 2022) keeps the
//! reused tail — the reconnection vertex `x_s` and everything beyond it — fixed,
//! and reconnects it to the destination pixel's primary (shading) vertex.
//! Because the integration measure at the reconnection vertex changes when the
//! primary vertex moves, generalized `RIS` must multiply the reused sample's
//! contribution by the Jacobian determinant of that shift.
//!
//! For a reconnection shift the Jacobian is the ratio of the solid-angle ↔ area
//! geometry terms seen from the reconnection vertex:
//!
//! ```text
//! |J| = (cos_dst / cos_src) * (dist_src^2 / dist_dst^2)
//! ```
//!
//! where `cos_src` / `cos_dst` are the cosines at the reconnection vertex
//! towards the source / destination primary vertex and `dist_src` / `dist_dst`
//! are the corresponding segment lengths.  Mapping a path to a neighbour and
//! back composes to the identity, so `J(a -> b) * J(b -> a) = 1`, and shifting
//! within the same pixel (`a == b`) is `J = 1`.
//!
//! # Conventions
//! * Cosines are clamped away from zero by [`MIN_COS`] (a grazing-angle guard)
//!   and the determinant is clamped to `[`[`J_MIN`]`, `[`J_MAX`]`]`, so the
//!   returned Jacobian is always strictly positive and finite — it can never
//!   inject `NaN` or a divide-by-zero into the resampling weight.
//! * A degenerate configuration (coincident vertices, zero-length segment, or a
//!   degenerate normal) falls back to the identity Jacobian `1.0`, the only
//!   measure-preserving default.
//! * Every function is a deterministic pure function: no RNG, no I/O, no GPU,
//!   no global state, and no `unsafe`.

use bevy_math::Vec3;

use super::vertex::{PathSuffix, PathVertex};

/// Grazing-angle cosine clamp: cosines below this are treated as [`MIN_COS`] so
/// a near-tangent reconnection cannot blow the Jacobian up to infinity.
pub const MIN_COS: f32 = 1.0e-3;

/// Lower clamp on the Jacobian determinant (keeps reused weights strictly
/// positive and well conditioned).
pub const J_MIN: f32 = 1.0e-4;

/// Upper clamp on the Jacobian determinant (bounds the variance a single reused
/// sample can contribute under a near-degenerate shift).
pub const J_MAX: f32 = 1.0e4;

/// Minimum squared segment length treated as non-degenerate.
const MIN_DIST_SQ: f32 = 1.0e-12;

/// Geometric Jacobian determinant of the reconnection shift at `reconnection`.
///
/// Maps a path whose primary (shading) vertex is `src_primary` to one whose
/// primary vertex is `dst_primary`, both reconnecting at `reconnection`.
/// Returns
///
/// ```text
/// |J| = (cos_dst / cos_src) * (dist_src^2 / dist_dst^2)
/// ```
///
/// with the reconnection-vertex cosines clamped to [`MIN_COS`] and the result
/// clamped to `[`[`J_MIN`]`, `[`J_MAX`]`]`.  A degenerate configuration
/// (coincident vertex, zero-length segment, or degenerate normal) returns the
/// identity `1.0`.  The result is always strictly positive and finite.
#[inline]
pub fn reconnection_jacobian(reconnection: PathVertex, src_primary: Vec3, dst_primary: Vec3) -> f32 {
    if reconnection.is_degenerate() {
        return 1.0;
    }
    let dist_src_sq = reconnection.dist_sq_to(src_primary);
    let dist_dst_sq = reconnection.dist_sq_to(dst_primary);
    if dist_src_sq <= MIN_DIST_SQ || dist_dst_sq <= MIN_DIST_SQ {
        return 1.0;
    }

    let cos_src = reconnection.cos_toward(src_primary).max(MIN_COS);
    let cos_dst = reconnection.cos_toward(dst_primary).max(MIN_COS);

    // |J| = (cos_dst / cos_src) * (dist_src^2 / dist_dst^2).
    let jacobian = (cos_dst * dist_src_sq) / (cos_src * dist_dst_sq);
    if jacobian.is_finite() && jacobian > 0.0 {
        jacobian.clamp(J_MIN, J_MAX)
    } else {
        1.0
    }
}

/// Applies the reconnection shift to a base path suffix.
///
/// Re-anchors `base`'s reusable tail to `dst_primary` (via
/// [`PathSuffix::with_primary`]) and returns the shifted suffix together with
/// the Jacobian determinant [`reconnection_jacobian`] of mapping `base`'s own
/// primary vertex to `dst_primary`.  The returned Jacobian is always strictly
/// positive and finite; a degenerate `base` yields the identity `1.0`.
#[inline]
pub fn reconnection_shift(base: &PathSuffix, dst_primary: PathVertex) -> (PathSuffix, f32) {
    let shifted = base.with_primary(dst_primary);
    let jacobian = if base.is_degenerate() {
        1.0
    } else {
        reconnection_jacobian(
            base.reconnection,
            base.primary.position,
            dst_primary.position,
        )
    };
    (shifted, jacobian)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reconnection vertex at the origin facing `+Z`.
    fn anchor() -> PathVertex {
        PathVertex::new(Vec3::ZERO, Vec3::Z)
    }

    #[test]
    fn same_primary_shift_is_identity() {
        let rv = anchor();
        let p = Vec3::new(0.3, -0.2, 2.0);
        let j = reconnection_jacobian(rv, p, p);
        assert!((j - 1.0).abs() < 1e-6, "j={j}");
    }

    #[test]
    fn jacobian_is_symmetric_inverse() {
        // J(a -> b) * J(b -> a) should be 1 for a non-degenerate, unclamped
        // configuration (analytic reciprocity of the reconnection shift).
        let rv = anchor();
        let a = Vec3::new(0.5, 0.0, 2.0);
        let b = Vec3::new(-0.4, 0.3, 1.3);
        let jab = reconnection_jacobian(rv, a, b);
        let jba = reconnection_jacobian(rv, b, a);
        assert!((jab * jba - 1.0).abs() < 1e-4, "jab={jab} jba={jba}");
    }

    #[test]
    fn jacobian_matches_closed_form() {
        let rv = anchor();
        // src straight up at distance 1: cos_src = 1, dist_src^2 = 1.
        let src = Vec3::new(0.0, 0.0, 1.0);
        // dst at 45 degrees, distance sqrt(2): cos_dst = 1/sqrt(2),
        // dist_dst^2 = 2.  Expected |J| = (cos_dst/cos_src)*(1/2).
        let dst = Vec3::new(0.0, 1.0, 1.0);
        let cos_dst = (0.5f32).sqrt();
        let expected = (cos_dst / 1.0) * (1.0 / 2.0);
        let j = reconnection_jacobian(rv, src, dst);
        assert!((j - expected).abs() < 1e-5, "j={j} expected={expected}");
    }

    #[test]
    fn jacobian_falls_off_with_destination_distance() {
        let rv = anchor();
        let src = Vec3::new(0.0, 0.0, 1.0);
        // Moving the destination farther along the same ray shrinks |J|
        // (inverse-square area spreading).
        let near = reconnection_jacobian(rv, src, Vec3::new(0.0, 0.0, 1.0));
        let far = reconnection_jacobian(rv, src, Vec3::new(0.0, 0.0, 2.0));
        assert!((near - 1.0).abs() < 1e-6);
        assert!((far - 0.25).abs() < 1e-5, "far={far}");
    }

    #[test]
    fn grazing_angle_is_clamped_and_finite() {
        let rv = anchor();
        let src = Vec3::new(0.0, 0.0, 1.0);
        // Destination almost in the tangent plane: cos_dst -> 0.  Without the
        // grazing clamp the ratio would collapse to zero; the clamp keeps it
        // strictly positive and finite within the determinant bounds.
        let dst = Vec3::new(1.0, 0.0, 1.0e-5);
        let j = reconnection_jacobian(rv, src, dst);
        assert!(j.is_finite() && j >= J_MIN && j <= J_MAX, "j={j}");
    }

    #[test]
    fn degenerate_configurations_fall_back_to_identity() {
        // Degenerate reconnection normal.
        let bad = PathVertex::new(Vec3::ZERO, Vec3::ZERO);
        assert_eq!(reconnection_jacobian(bad, Vec3::Z, Vec3::X), 1.0);
        // Coincident source primary (zero-length segment).
        let rv = anchor();
        assert_eq!(reconnection_jacobian(rv, Vec3::ZERO, Vec3::Z), 1.0);
    }

    #[test]
    fn reconnection_shift_moves_primary_and_reports_jacobian() {
        let base = PathSuffix::new(
            PathVertex::new(Vec3::new(0.0, 0.0, 1.0), Vec3::NEG_Z),
            anchor(),
            Vec3::splat(2.0),
        );
        let dst = PathVertex::new(Vec3::new(0.0, 1.0, 1.0), Vec3::NEG_Z);
        let (shifted, j) = reconnection_shift(&base, dst);
        // Tail (reconnection + radiance) is preserved, primary swapped.
        assert_eq!(shifted.reconnection, base.reconnection);
        assert_eq!(shifted.radiance, base.radiance);
        assert_eq!(shifted.primary, dst);
        // Jacobian equals the direct closed-form evaluation.
        let direct = reconnection_jacobian(base.reconnection, base.primary.position, dst.position);
        assert!((j - direct).abs() < 1e-6, "j={j} direct={direct}");
    }

    #[test]
    fn reconnection_shift_degenerate_base_is_identity_jacobian() {
        let base = PathSuffix::new(
            PathVertex::new(Vec3::ZERO, Vec3::Z),
            PathVertex::new(Vec3::ZERO, Vec3::NEG_Z), // coincident -> degenerate
            Vec3::ONE,
        );
        let (_, j) = reconnection_shift(&base, PathVertex::new(Vec3::Z, Vec3::NEG_Z));
        assert_eq!(j, 1.0);
    }

    #[test]
    fn results_are_deterministic() {
        let rv = anchor();
        let build = || reconnection_jacobian(rv, Vec3::new(0.2, 0.3, 1.5), Vec3::new(-0.1, 0.4, 1.1));
        assert_eq!(build(), build());
    }
}
