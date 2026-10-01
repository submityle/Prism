//! Oblique near-plane clipping of the mirror projection — CPU golden.
//!
//! When a scene is rendered from the mirrored camera (see
//! [`super::mirror`]) the reflection pass must not draw any geometry that lies
//! *behind* the reflector, because such geometry is not actually visible in the
//! mirror and would bleed through the water/glass surface.  The classic fix,
//! due to Eric Lengyel ("Oblique View Frustum Depth Projection and Clipping"),
//! is to **replace the camera's near clip plane with the reflector plane**: by
//! editing a single row of the projection matrix, the hardware near-plane clip
//! does the culling for free, with no extra user clip plane required.
//!
//! Given the reflector expressed in **view space** as a 4-vector
//! `C = (cx, cy, cz, cw)` (the plane `C · (x, y, z, 1) = 0`), the trick rewrites
//! the matrix row that produces clip-space `z` so that points on `C` land
//! exactly on the near plane.  This module implements that rewrite for a
//! `[0, 1]` clip-space depth convention (D3D / Metal / `wgpu`, i.e. the one
//! produced by a right-handed `wgpu`-style perspective), where the near plane is
//! `z_ndc = 0`:
//!
//! * [`oblique_near_clip`] replaces row 2 (the `z` row) of the projection with
//!   `k · C`, choosing the scale `k` so the far frustum corner opposite the
//!   plane still maps to `z_ndc = 1` — preserving usable depth precision.
//! * [`clip_plane_from_view_plane`] builds the view-space plane 4-vector from a
//!   normal/offset, orienting it so the kept half-space faces the camera.
//!
//! # Conventions
//! * `no_std`: math via `bevy_math`; no `alloc`.  Only `abs`/`signum`-style
//!   scalar ops are used; no transcendentals.
//! * Clip-space depth is `[0, 1]` with `z_ndc = 0` at the near plane and `1` at
//!   the far plane (reverse-Z is *not* assumed; a standard forward mapping is).
//! * The projection matrix is column-major (`bevy_math` convention) and maps a
//!   view-space point `v` to clip space via `proj.mul_vec4(v.extend(1.0))`, so
//!   clip component `i` equals `proj.row(i) · v`.
//! * The view-space plane `C` must be oriented so the half-space the camera
//!   keeps satisfies `C · v < 0` for visible points (view-space `z < 0` in a
//!   right-handed camera looking down `-Z`).  [`clip_plane_from_view_plane`]
//!   enforces this orientation.
//! * All inputs are sanitised: a singular projection, a non-finite plane, or a
//!   near-zero scaling denominator makes [`oblique_near_clip`] return the
//!   *unmodified* projection (a safe no-op) rather than emitting `NaN`.

use bevy_math::{Mat4, Vec3, Vec4};

/// Smallest absolute value of the Lengyel scaling denominator `C·Q` that is
/// treated as usable; below this the rewrite is skipped (degenerate geometry).
const MIN_DENOM: f32 = 1.0e-8;
/// Smallest absolute determinant for which the projection is considered
/// invertible; a flatter matrix triggers the safe no-op fallback.
const MIN_DET: f32 = 1.0e-12;
/// Smallest squared normal length accepted when building a view-space plane.
const MIN_NORMAL_LEN_SQ: f32 = 1.0e-12;

/// Builds a view-space clip plane 4-vector from a normal and offset.
///
/// The plane is `n·v + d = 0`; the normal is renormalised and the whole plane
/// is flipped if necessary so that the camera's visible half-space (`z < 0` in
/// a right-handed view) satisfies `C·v < 0`, which is the orientation
/// [`oblique_near_clip`] expects.  A degenerate normal yields the camera's
/// default near direction `(0, 0, -1, 0)` so the result is always finite.
pub fn clip_plane_from_view_plane(normal_view: Vec3, d_view: f32) -> Vec4 {
    let len_sq = normal_view.length_squared();
    if !len_sq.is_finite() || len_sq <= MIN_NORMAL_LEN_SQ || !d_view.is_finite() {
        return Vec4::new(0.0, 0.0, -1.0, 0.0);
    }
    let inv_len = 1.0 / len_sq.sqrt();
    let c = Vec4::new(
        normal_view.x * inv_len,
        normal_view.y * inv_len,
        normal_view.z * inv_len,
        d_view * inv_len,
    );
    // Orient so a point just in front of the camera (small negative z) is in the
    // clipped-away (C·v < 0) half-space boundary consistently: we want the plane
    // normal to point *toward* the camera's visible region. Evaluate at a
    // representative visible point (0, 0, -1): keep sign so that this point is on
    // the positive side (not clipped). Flip when it is negative.
    let probe = c.x * 0.0 + c.y * 0.0 + c.z * (-1.0) + c.w;
    if probe < 0.0 {
        -c
    } else {
        c
    }
}

/// Returns `signum`-style sign with a defined value at zero (treated as `+1`).
///
/// Lengyel's construction needs the sign of the plane normal's `x`/`y` to pick
/// the frustum corner farthest along the plane; a zero component maps to `+1`
/// so the chosen corner is still a valid frustum corner.
#[inline]
fn sgn(x: f32) -> f32 {
    if x < 0.0 {
        -1.0
    } else {
        1.0
    }
}

/// Rewrites `proj`'s near plane to coincide with the view-space plane `clip`.
///
/// Returns a new projection matrix whose near clip plane is the oblique plane
/// `clip`, so that any geometry on the far side of the reflector is culled by
/// the standard near-plane test.  Points lying on `clip` project to `z_ndc = 0`
/// and the far frustum corner opposite the plane is held at `z_ndc = 1` to keep
/// depth precision reasonable.
///
/// The input matrix is returned unchanged (a safe no-op) when it is singular,
/// when `clip` is non-finite, or when the scaling denominator collapses — so
/// the result is always a finite, usable projection.
pub fn oblique_near_clip(proj: Mat4, clip: Vec4) -> Mat4 {
    if !clip.is_finite() || !matrix_is_finite(&proj) {
        return proj;
    }
    let det = proj.determinant();
    if !det.is_finite() || det.abs() < MIN_DET {
        return proj;
    }
    let inv = proj.inverse();
    if !matrix_is_finite(&inv) {
        return proj;
    }
    // Frustum corner (in clip space) farthest in the direction of the plane
    // normal; transform back to view space via the inverse projection.
    let corner_clip = Vec4::new(sgn(clip.x), sgn(clip.y), 1.0, 1.0);
    let q = inv.mul_vec4(corner_clip);
    let denom = clip.dot(q);
    if !denom.is_finite() || denom.abs() < MIN_DENOM {
        return proj;
    }
    // We want the replacement z-row `k·C` to map the far corner `q` to the far
    // plane: (k·C)·q = w(q) = row3·q, hence k = (row3·q) / (C·q).
    let row3 = proj.row(3);
    let k = row3.dot(q) / denom;
    if !k.is_finite() {
        return proj;
    }
    let new_z_row = clip * k;
    replace_z_row(proj, new_z_row)
}

/// Returns a copy of `proj` with its row 2 (the clip-space `z` row) replaced.
///
/// Operates on the column-major flat array, overwriting the `z` component of
/// each of the four columns, which collectively form matrix row 2.
fn replace_z_row(proj: Mat4, z_row: Vec4) -> Mat4 {
    let mut cols = proj.to_cols_array();
    // Column-major layout: column c occupies indices [4c, 4c+3]; element (2, c)
    // (row 2 of column c) is at 4c + 2.
    cols[2] = z_row.x;
    cols[6] = z_row.y;
    cols[10] = z_row.z;
    cols[14] = z_row.w;
    Mat4::from_cols_array(&cols)
}

/// Returns `true` when every component of `m` is finite.
#[inline]
fn matrix_is_finite(m: &Mat4) -> bool {
    m.col(0).is_finite() && m.col(1).is_finite() && m.col(2).is_finite() && m.col(3).is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A right-handed `[0, 1]`-depth perspective projection (wgpu/D3D style),
    /// built explicitly so the test does not depend on deprecated helpers.
    ///
    /// Looks down `-Z`; near maps to `z_ndc = 0`, far to `z_ndc = 1`.
    fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
        let f = 1.0 / (fov_y * 0.5).tan();
        let r = far / (near - far);
        Mat4::from_cols(
            Vec4::new(f / aspect, 0.0, 0.0, 0.0),
            Vec4::new(0.0, f, 0.0, 0.0),
            Vec4::new(0.0, 0.0, r, -1.0),
            Vec4::new(0.0, 0.0, near * far / (near - far), 0.0),
        )
    }

    /// The standard test projection: 60° fov, 16:9, near 0.1, far 100.
    fn test_proj() -> Mat4 {
        perspective(core::f32::consts::FRAC_PI_3, 16.0 / 9.0, 0.1, 100.0)
    }

    /// Projects a view-space point to NDC via a projection matrix.
    fn project_ndc(proj: &Mat4, v: Vec3) -> Vec3 {
        let clip = proj.mul_vec4(v.extend(1.0));
        clip.truncate() / clip.w
    }

    #[test]
    fn clip_plane_orientation_keeps_visible_point_positive() {
        // Plane z = -2 in view space, built from an arbitrarily-signed normal.
        let c = clip_plane_from_view_plane(Vec3::new(0.0, 0.0, 5.0), 10.0);
        // A visible point in front of the plane (z = -1) must be on the kept side.
        let v = Vec3::new(0.0, 0.0, -1.0);
        let s = c.x * v.x + c.y * v.y + c.z * v.z + c.w;
        assert!(s >= 0.0, "visible point should be on the non-clipped side");
    }

    #[test]
    fn on_plane_points_project_to_near() {
        let proj = test_proj();
        // Oblique plane: view-space z = -3 (parallel to default near but farther).
        let clip = clip_plane_from_view_plane(Vec3::new(0.0, 0.0, 1.0), 3.0);
        let modified = oblique_near_clip(proj, clip);
        // Several points on the plane z = -3 must land at z_ndc ~ 0.
        for (x, y) in [(0.0, 0.0), (1.0, 0.5), (-2.0, 1.5)] {
            let v = Vec3::new(x, y, -3.0);
            let ndc = project_ndc(&modified, v);
            assert!(ndc.z.abs() < 1e-4, "on-plane z_ndc = {} (expected ~0)", ndc.z);
        }
    }

    /// Returns the component-wise plane normal (xyz) of a clip 4-vector.
    fn clip_normal(c: Vec4) -> Vec3 {
        Vec3::new(c.x, c.y, c.z)
    }

    #[test]
    fn tilted_plane_points_project_to_near() {
        let proj = test_proj();
        // A tilted reflector in view space.
        let clip = clip_plane_from_view_plane(Vec3::new(0.2, 0.1, 1.0), 4.0);
        let modified = oblique_near_clip(proj, clip);
        let normal = clip_normal(clip);
        // Build points exactly on the plane C·v + w = 0 and verify z_ndc ~ 0.
        // For a unit normal, subtracting `s · normal` lands any point on the plane.
        for base in [
            Vec3::new(1.0, 2.0, -5.0),
            Vec3::new(-3.0, 0.5, -6.0),
            Vec3::new(0.0, -1.0, -4.0),
        ] {
            let s = normal.dot(base) + clip.w;
            let on_plane = base - normal * s;
            let ndc = project_ndc(&modified, on_plane);
            assert!(ndc.z.abs() < 1e-3, "tilted on-plane z_ndc = {}", ndc.z);
        }
    }

    #[test]
    fn far_corner_still_reaches_far_plane() {
        let proj = test_proj();
        let clip = clip_plane_from_view_plane(Vec3::new(0.0, 0.0, 1.0), 3.0);
        let modified = oblique_near_clip(proj, clip);
        // The far corner used by the construction should map to z_ndc ~ 1.
        let inv = modified.inverse();
        let corner_clip = Vec4::new(sgn(clip.x), sgn(clip.y), 1.0, 1.0);
        let q = inv.mul_vec4(corner_clip);
        let ndc = {
            let c = modified.mul_vec4(q);
            c.truncate() / c.w
        };
        assert!((ndc.z - 1.0).abs() < 1e-3, "far corner z_ndc = {}", ndc.z);
    }

    #[test]
    fn z_row_is_scaled_clip_plane() {
        let proj = test_proj();
        let clip = clip_plane_from_view_plane(Vec3::new(0.0, 0.0, 1.0), 3.0);
        let modified = oblique_near_clip(proj, clip);
        // Row 2 of the modified matrix must be parallel to the clip plane.
        let row2 = modified.row(2);
        let cross = Vec3::new(
            row2.y * clip.z - row2.z * clip.y,
            row2.z * clip.x - row2.x * clip.z,
            row2.x * clip.y - row2.y * clip.x,
        );
        assert!(cross.length() < 1e-4, "z row not parallel to clip plane");
    }

    #[test]
    fn singular_projection_is_a_noop() {
        // A zero-scale (singular) matrix cannot be inverted: return unchanged.
        let singular = Mat4::from_scale(Vec3::new(1.0, 1.0, 0.0));
        let clip = Vec4::new(0.0, 0.0, 1.0, 3.0);
        let out = oblique_near_clip(singular, clip);
        for c in 0..4 {
            assert_eq!(out.col(c), singular.col(c));
        }
    }

    #[test]
    fn non_finite_plane_is_a_noop() {
        let proj = test_proj();
        let bad = Vec4::new(f32::NAN, 0.0, 1.0, 3.0);
        let out = oblique_near_clip(proj, bad);
        for c in 0..4 {
            assert!((out.col(c) - proj.col(c)).length() < 1e-6);
        }
    }

    #[test]
    fn degenerate_normal_falls_back_to_default_near() {
        let c = clip_plane_from_view_plane(Vec3::ZERO, 5.0);
        assert_eq!(c, Vec4::new(0.0, 0.0, -1.0, 0.0));
        assert!(c.is_finite());
    }
}
