//! Billboard normal reconstruction and camera-facing basis — CPU golden.
//!
//! A particle billboard is a flat quad with no geometric normals, yet directional
//! lighting needs a plausible surface orientation.  "Spherical billboards"
//! (Umenhoffer et al. 2006) treat the quad's texture disc as the projection of a
//! sphere and reconstruct a hemisphere of normals from the quad UVs, so a
//! round sprite (a puff of smoke, a spark) shades like an actual sphere facing
//! the camera.  A cheaper "cylindrical" variant bends the normal only across
//! one axis, suited to axis-aligned streaks and beams.
//!
//! To place and orient the quad this module also builds the orthonormal
//! camera-facing basis (`right`, `up`, `forward`) a billboard is swept along,
//! and rotates a reconstructed view/billboard-space normal into world space so
//! the lighting integrator in [`crate::gi::particle_shade::lighting`] can use a
//! world-space normal and light.
//!
//! * [`spherical_normal`] — hemisphere normal from quad UVs with a circular
//!   coverage mask (`alpha`); UVs outside the unit disc are masked out.
//! * [`cylindrical_normal`] — single-axis bent normal plus a stripe mask.
//! * [`camera_facing_basis`] — Gram-Schmidt orthonormal `(right, up, forward)`
//!   from a camera right/up pair (e.g. the first two rows of a view matrix).
//! * [`billboard_basis_from_view`] — the same basis extracted directly from a
//!   world-to-view [`Mat4`].
//! * [`view_normal_to_world`] / [`billboard_normal_world`] — rotate a
//!   billboard-space normal into world space through the basis.
//!
//! # Conventions
//! * Quad UVs are in `[0, 1]^2` with the disc centre at `(0.5, 0.5)`; the
//!   mapped coordinate is `p = uv * 2 - 1 in [-1, 1]^2`.
//! * The reconstructed normal lives in a right-handed billboard frame whose
//!   `+Z` (`forward`) points **from the particle toward the camera**, matching
//!   the view direction convention used by the lighting pass; `n.z >= 0`.
//! * A returned `alpha` is the circular / stripe coverage in `[0, 1]`: `0`
//!   means the fragment is outside the sprite and should be discarded.
//! * Every function is a deterministic pure function (no RNG / I/O / GPU /
//!   `unsafe`).  Degenerate bases fall back to the world axes and masked / zero
//!   normals default to the forward axis, so results are always finite unit
//!   vectors and never `NaN`.
//!
//! # References
//! * T. Umenhoffer, L. Szirmay-Kalos, G. Szijarto, "Spherical Billboards for
//!   Rendering Volumetric Data", 2006.
//! * NVIDIA GPU Gems 3, Ch. 23 "High-Speed, Off-Screen Particles" (billboarding).
//! * E. Lengyel, *Mathematics for 3D Game Programming and Computer Graphics*,
//!   3rd ed., §4 (orthonormal frames / Gram-Schmidt).

use bevy_math::{Mat3, Mat4, Vec2, Vec3};

/// Fragments whose squared disc radius exceeds this are outside the sprite.
const UNIT_R2: f32 = 1.0;

/// Degenerate-length threshold below which a vector is treated as zero.
const EPS_LEN: f32 = 1.0e-6;

/// Return a finite unit vector, falling back to `fallback` when `v` is zero
/// length or non-finite.
#[inline]
fn safe_normalize(v: Vec3, fallback: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq.is_finite() && len_sq > EPS_LEN * EPS_LEN {
        v / len_sq.sqrt()
    } else {
        fallback
    }
}

/// Saturate a scalar to `[0, 1]`, mapping non-finite input to `0`.
#[inline]
fn saturate(x: f32) -> f32 {
    if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Reconstructed billboard normal with its circular coverage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BillboardNormal {
    /// Unit normal in the billboard frame (`+Z` toward the camera, `n.z >= 0`).
    pub normal: Vec3,
    /// Circular / stripe coverage in `[0, 1]`; `0` discards the fragment.
    pub alpha: f32,
}

/// Spherical-billboard normal reconstruction from quad UVs.
///
/// Maps `uv` to `p = uv * 2 - 1`, interprets `p` as the `xy` of a unit-sphere
/// point, and lifts it to `n.z = sqrt(1 - |p|^2)` — the hemisphere facing the
/// camera.  Fragments with `|p|^2 > 1` lie outside the inscribed disc and are
/// returned with `alpha = 0` and the forward normal `+Z` (so lighting stays
/// finite even if the caller ignores the mask).  Non-finite UVs are treated as
/// the disc centre.
#[inline]
pub fn spherical_normal(uv: Vec2) -> BillboardNormal {
    let px = if uv.x.is_finite() { uv.x * 2.0 - 1.0 } else { 0.0 };
    let py = if uv.y.is_finite() { uv.y * 2.0 - 1.0 } else { 0.0 };
    let r2 = px * px + py * py;
    if r2 > UNIT_R2 {
        return BillboardNormal {
            normal: Vec3::Z,
            alpha: 0.0,
        };
    }
    let nz = (1.0 - r2).max(0.0).sqrt();
    let normal = safe_normalize(Vec3::new(px, py, nz), Vec3::Z);
    BillboardNormal { normal, alpha: 1.0 }
}

/// Soft spherical-billboard coverage for anti-aliased sprite edges.
///
/// Like [`spherical_normal`] but the `alpha` ramps linearly over a `softness`
/// band just inside the disc edge instead of a hard cut, approximating a
/// pre-filtered round sprite.  `softness` is clamped to `[0, 1]`; `0` reproduces
/// the hard mask of [`spherical_normal`].
#[inline]
pub fn spherical_normal_soft(uv: Vec2, softness: f32) -> BillboardNormal {
    let px = if uv.x.is_finite() { uv.x * 2.0 - 1.0 } else { 0.0 };
    let py = if uv.y.is_finite() { uv.y * 2.0 - 1.0 } else { 0.0 };
    let r2 = px * px + py * py;
    let r = r2.max(0.0).sqrt();
    let soft = saturate(softness);
    let inner = (1.0 - soft).max(0.0);
    let alpha = if r <= inner {
        1.0
    } else if r >= 1.0 || soft <= EPS_LEN {
        0.0
    } else {
        saturate((1.0 - r) / (1.0 - inner))
    };
    let nz = (1.0 - r2).max(0.0).sqrt();
    let normal = safe_normalize(Vec3::new(px, py, nz), Vec3::Z);
    BillboardNormal { normal, alpha }
}

/// Cylindrical-billboard normal reconstruction from quad UVs.
///
/// Bends the normal only across the horizontal axis: `n.x = u * 2 - 1`,
/// `n.z = sqrt(1 - n.x^2)`, `n.y = 0`.  The coverage masks a vertical stripe
/// (`|n.x| <= 1`, always inside here) so the sprite reads as a round-sided
/// cylinder / beam.  Non-finite UVs collapse to the centre line (`+Z`).
#[inline]
pub fn cylindrical_normal(uv: Vec2) -> BillboardNormal {
    let nx = if uv.x.is_finite() {
        (uv.x * 2.0 - 1.0).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let nz = (1.0 - nx * nx).max(0.0).sqrt();
    let normal = safe_normalize(Vec3::new(nx, 0.0, nz), Vec3::Z);
    BillboardNormal { normal, alpha: 1.0 }
}

/// Orthonormal camera-facing basis `(right, up, forward)` from a right/up pair.
///
/// `forward` is `right × up` renormalised, then `up` is re-derived as
/// `forward × right` (Gram-Schmidt) so the triple is exactly orthonormal and
/// right-handed even when the inputs are slightly non-orthogonal.  Degenerate
/// inputs fall back to the world axes `(X, Y, Z)`.  The resulting `forward`
/// points toward the camera, matching the billboard-space `+Z` convention.
#[inline]
pub fn camera_facing_basis(camera_right: Vec3, camera_up: Vec3) -> (Vec3, Vec3, Vec3) {
    let right0 = safe_normalize(camera_right, Vec3::X);
    let up0 = safe_normalize(camera_up, Vec3::Y);
    let forward = safe_normalize(right0.cross(up0), Vec3::Z);
    // Re-orthogonalise: right ⟂ forward, then up closes the right-handed frame.
    let right = safe_normalize(up0.cross(forward), right0);
    let up = safe_normalize(forward.cross(right), up0);
    (right, up, forward)
}

/// Camera-facing basis extracted from a world-to-view matrix.
///
/// A column-major world-to-view [`Mat4`] stores the camera's world-space right
/// / up / forward in the first three columns of its rotation block (`x_axis`,
/// `y_axis`, `z_axis`).  This feeds those into [`camera_facing_basis`], so the
/// returned triple is orthonormalised and degenerate matrices fall back to the
/// world axes.
#[inline]
pub fn billboard_basis_from_view(world_to_view: Mat4) -> (Vec3, Vec3, Vec3) {
    let right = world_to_view.x_axis.truncate();
    let up = world_to_view.y_axis.truncate();
    camera_facing_basis(right, up)
}

/// Rotate a billboard/view-space normal into world space through a basis.
///
/// `n = n.x * right + n.y * up + n.z * forward`.  The basis is assumed
/// orthonormal (use [`camera_facing_basis`]); the result is renormalised and
/// falls back to `forward` (then `+Z`) if the combination degenerates.
#[inline]
pub fn view_normal_to_world(normal: Vec3, right: Vec3, up: Vec3, forward: Vec3) -> Vec3 {
    let world = right * normal.x + up * normal.y + forward * normal.z;
    safe_normalize(world, safe_normalize(forward, Vec3::Z))
}

/// Rotate a billboard-space normal into world space via a rotation matrix.
///
/// Equivalent to [`view_normal_to_world`] when `basis`'s columns are
/// `(right, up, forward)`; provided for callers that already hold the
/// billboard→world rotation as a [`Mat3`].
#[inline]
pub fn view_normal_to_world_mat(normal: Vec3, basis: Mat3) -> Vec3 {
    let world = basis * normal;
    safe_normalize(world, Vec3::Z)
}

/// One-shot spherical-billboard world normal from UVs and a right/up pair.
///
/// Reconstructs the spherical normal with [`spherical_normal`], builds the
/// camera-facing basis with [`camera_facing_basis`], and rotates the normal to
/// world space.  Returns the world normal together with the disc coverage so a
/// caller can discard masked fragments.
#[inline]
pub fn billboard_normal_world(
    uv: Vec2,
    camera_right: Vec3,
    camera_up: Vec3,
) -> BillboardNormal {
    let local = spherical_normal(uv);
    let (right, up, forward) = camera_facing_basis(camera_right, camera_up);
    let world = view_normal_to_world(local.normal, right, up, forward);
    BillboardNormal {
        normal: world,
        alpha: local.alpha,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1.0e-5;

    #[test]
    fn spherical_center_faces_camera() {
        let n = spherical_normal(Vec2::new(0.5, 0.5));
        assert!((n.normal - Vec3::Z).length() < TOL);
        assert!((n.alpha - 1.0).abs() < TOL);
    }

    #[test]
    fn spherical_normals_are_unit_and_front_facing() {
        for &(u, v) in &[(0.5, 0.5), (0.7, 0.3), (0.2, 0.9), (0.5, 0.95)] {
            let n = spherical_normal(Vec2::new(u, v));
            if n.alpha > 0.0 {
                assert!((n.normal.length() - 1.0).abs() < TOL, "non-unit at {u},{v}");
                assert!(n.normal.z >= -TOL, "back-facing at {u},{v}");
            }
        }
    }

    #[test]
    fn spherical_outside_disc_is_masked() {
        // Corner (0,0) -> p=(-1,-1), r2=2 > 1 outside.
        let n = spherical_normal(Vec2::new(0.0, 0.0));
        assert!(n.alpha.abs() < TOL);
        assert!(n.normal.is_finite());
    }

    #[test]
    fn spherical_edge_normal_is_tangent() {
        // u=1,v=0.5 -> p=(1,0), r2=1 -> n.z=0, points along +X.
        let n = spherical_normal(Vec2::new(1.0, 0.5));
        assert!(n.normal.z.abs() < 1.0e-3);
        assert!((n.normal.x - 1.0).abs() < 1.0e-3);
    }

    #[test]
    fn spherical_soft_ramps_at_edge() {
        let hard = spherical_normal_soft(Vec2::new(0.5, 0.5), 0.0);
        assert!((hard.alpha - 1.0).abs() < TOL);
        // Mid-radius with softness band gives partial coverage.
        let mid = spherical_normal_soft(Vec2::new(0.95, 0.5), 0.2);
        assert!(mid.alpha > 0.0 && mid.alpha < 1.0, "alpha={}", mid.alpha);
        // Centre always full.
        let c = spherical_normal_soft(Vec2::new(0.5, 0.5), 0.5);
        assert!((c.alpha - 1.0).abs() < TOL);
    }

    #[test]
    fn cylindrical_bends_only_in_x() {
        let n = cylindrical_normal(Vec2::new(0.5, 0.2));
        assert!(n.normal.y.abs() < TOL);
        assert!((n.normal - Vec3::Z).length() < TOL);
        let e = cylindrical_normal(Vec2::new(1.0, 0.7));
        assert!(e.normal.y.abs() < TOL);
        assert!((e.normal.x - 1.0).abs() < 1.0e-3);
    }

    #[test]
    fn basis_is_orthonormal_and_right_handed() {
        let (r, u, f) = camera_facing_basis(Vec3::new(1.0, 0.1, 0.0), Vec3::new(0.0, 1.0, 0.2));
        for v in [r, u, f] {
            assert!((v.length() - 1.0).abs() < TOL);
        }
        assert!(r.dot(u).abs() < TOL);
        assert!(r.dot(f).abs() < TOL);
        assert!(u.dot(f).abs() < TOL);
        // Right-handed: right × up == forward.
        assert!((r.cross(u) - f).length() < TOL);
    }

    #[test]
    fn basis_degenerate_inputs_fall_back_to_world_axes() {
        let (r, u, f) = camera_facing_basis(Vec3::ZERO, Vec3::ZERO);
        assert!((r - Vec3::X).length() < TOL);
        assert!((u - Vec3::Y).length() < TOL);
        assert!((f - Vec3::Z).length() < TOL);
    }

    #[test]
    fn basis_from_view_identity_matches_axes() {
        let (r, u, f) = billboard_basis_from_view(Mat4::IDENTITY);
        assert!((r - Vec3::X).length() < TOL);
        assert!((u - Vec3::Y).length() < TOL);
        assert!((f - Vec3::Z).length() < TOL);
    }

    #[test]
    fn view_to_world_with_identity_basis_is_identity() {
        let n = Vec3::new(0.3, -0.4, 0.866).normalize();
        let w = view_normal_to_world(n, Vec3::X, Vec3::Y, Vec3::Z);
        assert!((w - n).length() < TOL);
    }

    #[test]
    fn view_to_world_rotates_through_basis() {
        // Right/up swapped-and-rotated frame: forward along world +Z still.
        let right = Vec3::new(0.0, 1.0, 0.0);
        let up = Vec3::new(-1.0, 0.0, 0.0);
        let (r, u, f) = camera_facing_basis(right, up);
        let local = Vec3::X; // billboard-space right
        let world = view_normal_to_world(local, r, u, f);
        assert!((world - r).length() < TOL);
    }

    #[test]
    fn view_to_world_mat_matches_component_form() {
        let (r, u, f) = camera_facing_basis(Vec3::new(1.0, 0.2, 0.0), Vec3::new(0.0, 1.0, 0.3));
        let m = Mat3::from_cols(r, u, f);
        let n = Vec3::new(0.2, 0.3, 0.93).normalize();
        let a = view_normal_to_world(n, r, u, f);
        let b = view_normal_to_world_mat(n, m);
        assert!((a - b).length() < TOL);
    }

    #[test]
    fn billboard_world_normal_identity_camera() {
        // Identity camera: billboard world normal equals reconstructed normal.
        let n = billboard_normal_world(Vec2::new(0.7, 0.4), Vec3::X, Vec3::Y);
        let local = spherical_normal(Vec2::new(0.7, 0.4));
        assert!((n.normal - local.normal).length() < TOL);
        assert!((n.alpha - local.alpha).abs() < TOL);
    }

    #[test]
    fn no_nan_on_pathological_inputs() {
        let bad = Vec2::new(f32::NAN, f32::INFINITY);
        assert!(spherical_normal(bad).normal.is_finite());
        assert!(cylindrical_normal(bad).normal.is_finite());
        let n = billboard_normal_world(bad, Vec3::ZERO, Vec3::ZERO);
        assert!(n.normal.is_finite());
        assert!(view_normal_to_world(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO).is_finite());
    }
}
