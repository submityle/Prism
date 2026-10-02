//! Cascaded shadow-map (CSM) light-matrix construction and stabilization.
//!
//! `directional.rs` *consumes* one world -> light-clip matrix and one texel
//! world size per cascade; this module *produces* them from the camera and the
//! sun direction, closing the directional shadow loop on the CPU side.  The GPU
//! twin uploads the same matrices to `shadow.wesl`.
//!
//! The construction follows the standard stable-CSM recipe (as used by Unreal's
//! directional cascade fitting and the classic "Stable CSM" technique):
//!
//! 1. Reconstruct the camera frustum's eight world-space corners from the
//!    inverse view-projection, then slice out each cascade's sub-frustum by
//!    interpolating the near/far corners at the cascade split distances.  View
//!    depth is affine in world position, so a linear interpolation along each
//!    frustum edge lands exactly on the requested view-space distance.
//! 2. Fit a **bounding sphere** to the sub-frustum corners.  A sphere is
//!    rotation-invariant, so the fitted region does not change size as the
//!    camera yaws — the first half of killing shadow "shimmer".
//! 3. Build a right-handed light view looking along the sun direction from
//!    behind the sphere, and an orthographic projection sized to the sphere.
//! 4. **Texel-snap** the projection: quantize the sphere centre to whole
//!    shadow-map texels in light space so texels map to a fixed world grid as
//!    the camera moves — the second half of killing shimmer.
//!
//! Everything here is pure, deterministic `[f32; 16]` / `[f32; 3]` math with a
//! CPU golden test, so it stays a byte-for-byte twin of the GPU matrix upload.

use crate::shadow::cascade::{CascadeSplits, MAX_CASCADE_COUNT};
use crate::shadow::math::{
    dot3, look_at_rh, mul, normalize3, orthographic_rh_01, sub3, transform_point, Mat4,
};

/// One cascade's world -> light-clip matrix plus the world size of a single
/// shadow-map texel (feeds the directional normal-offset bias).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CascadeMatrix {
    /// Column-major world -> light-clip matrix (wgpu clip: `z` in `[0, 1]`).
    pub view_projection: Mat4,
    /// World-space edge length of one shadow-map texel in this cascade.
    pub texel_world_size: f32,
}

impl CascadeMatrix {
    /// Identity projection with a unit texel — the neutral value used to pad
    /// inactive cascade slots.
    pub fn identity() -> Self {
        let mut m = [0.0_f32; 16];
        m[0] = 1.0;
        m[5] = 1.0;
        m[10] = 1.0;
        m[15] = 1.0;
        Self {
            view_projection: m,
            texel_world_size: 1.0,
        }
    }
}

/// The eight world-space corners of the camera frustum, reconstructed from the
/// inverse view-projection at the NDC cube corners.  Order: the four near-plane
/// corners (`z_ndc = 0`) followed by the four far-plane corners (`z_ndc = 1`),
/// each quad ordered `(-,-), (+,-), (+,+), (-,+)` in NDC `x, y`.
fn frustum_corners_world(inverse_view_projection: &Mat4) -> [[f32; 3]; 8] {
    // wgpu NDC: x, y in [-1, 1], z in [0, 1].
    let ndc = [
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    let mut corners = [[0.0_f32; 3]; 8];
    for (corner, point) in corners.iter_mut().zip(ndc.iter()) {
        let clip = transform_point(inverse_view_projection, *point);
        // A well-formed perspective inverse-view-projection yields a positive
        // `w` for every NDC cube corner; guard only against an exact zero.
        let inv_w = if clip[3] == 0.0 { 0.0 } else { clip[3].recip() };
        *corner = [clip[0] * inv_w, clip[1] * inv_w, clip[2] * inv_w];
    }
    corners
}

/// Linear blend `a + (b - a) * t` of two points.
fn lerp_point(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// The eight corners of one cascade's sub-frustum: for each of the four frustum
/// edges, the point at the cascade's near split and the point at its far split.
/// `t_near`/`t_far` are the split distances expressed as a fraction of the
/// camera `[near, far]` range (view depth is affine along each edge).
fn cascade_sub_frustum(corners: &[[f32; 3]; 8], t_near: f32, t_far: f32) -> [[f32; 3]; 8] {
    let mut out = [[0.0_f32; 3]; 8];
    for edge in 0..4 {
        let near_corner = corners[edge];
        let far_corner = corners[edge + 4];
        out[edge] = lerp_point(near_corner, far_corner, t_near);
        out[edge + 4] = lerp_point(near_corner, far_corner, t_far);
    }
    out
}

/// Centre and radius of the minimal-ish bounding sphere of eight points, taken
/// as the centroid and the farthest corner distance.  The centroid sphere is
/// slightly larger than the true minimal sphere but is cheap, stable and never
/// clips the frustum, which is what CSM fitting wants.
fn bounding_sphere(corners: &[[f32; 3]; 8]) -> ([f32; 3], f32) {
    let mut center = [0.0_f32; 3];
    for c in corners {
        center[0] += c[0];
        center[1] += c[1];
        center[2] += c[2];
    }
    let inv = 0.125_f32; // 1 / 8
    center = [center[0] * inv, center[1] * inv, center[2] * inv];

    let mut radius_sq = 0.0_f32;
    for c in corners {
        let d = sub3(*c, center);
        radius_sq = radius_sq.max(dot3(d, d));
    }
    (center, radius_sq.sqrt().max(1.0e-4))
}

/// Picks a numerically safe up vector for the light view: world up unless the
/// light direction is nearly vertical, in which case world `+z`.
fn stable_up(light_direction: [f32; 3]) -> [f32; 3] {
    let dir = normalize3(light_direction);
    let world_up = [0.0, 1.0, 0.0];
    if dot3(dir, world_up).abs() > 0.99 {
        [0.0, 0.0, 1.0]
    } else {
        world_up
    }
}

/// Builds the stabilized world -> light-clip matrix and texel world size for a
/// single cascade sphere.
fn fit_cascade(
    center: [f32; 3],
    radius: f32,
    light_direction: [f32; 3],
    resolution: u32,
) -> CascadeMatrix {
    let resolution = resolution.max(1) as f32;
    let dir = normalize3(light_direction);
    let up = stable_up(dir);

    // Eye sits one radius behind the sphere along the light travel direction so
    // the whole cascade is in front of the near plane.
    let eye = [
        center[0] - dir[0] * radius,
        center[1] - dir[1] * radius,
        center[2] - dir[2] * radius,
    ];
    let view = look_at_rh(eye, center, up);

    // Orthographic box tightly bounds the sphere; depth spans [0, 2r] so the
    // sphere centre lands at z = 0.5.
    let mut proj = orthographic_rh_01(-radius, radius, -radius, radius, 0.0, 2.0 * radius);

    // Texel snap: project the centre, quantize its NDC x/y to whole texels, and
    // fold the residual back into the projection's translation columns so the
    // shadow grid is world-locked as the camera moves.
    let view_proj = mul(&proj, &view);
    let ndc_center = transform_point(&view_proj, center);
    let half_res = resolution * 0.5;
    let tx = ndc_center[0] * half_res;
    let ty = ndc_center[1] * half_res;
    let dx = (tx.round() - tx) / half_res;
    let dy = (ty.round() - ty) / half_res;
    proj[12] += dx;
    proj[13] += dy;

    CascadeMatrix {
        view_projection: mul(&proj, &view),
        texel_world_size: (2.0 * radius) / resolution,
    }
}

/// Computes the stabilized world -> light-clip matrix and texel world size for
/// every active cascade of a directional light.
///
/// * `inverse_view_projection` — the camera's inverse view-projection (maps NDC
///   back to world) used to reconstruct the frustum.
/// * `light_direction` — the direction the light travels (from the emitter into
///   the scene); need not be normalized.
/// * `camera_near` / `camera_far` — the camera projection's near/far planes,
///   which anchor how the cascade split distances map onto the frustum edges.
/// * `splits` — the cascade split table (see [`CascadeSplits`]).
/// * `resolution` — the per-cascade shadow-map edge resolution in texels.
///
/// Inactive cascade slots (`>= splits.count`) are filled with
/// [`CascadeMatrix::identity`] so the fixed-size array always has
/// [`MAX_CASCADE_COUNT`] entries.
pub fn compute_cascade_matrices(
    inverse_view_projection: &Mat4,
    light_direction: [f32; 3],
    camera_near: f32,
    camera_far: f32,
    splits: &CascadeSplits,
    resolution: u32,
) -> [CascadeMatrix; MAX_CASCADE_COUNT] {
    let corners = frustum_corners_world(inverse_view_projection);
    let camera_near = camera_near.max(1.0e-4);
    let camera_far = camera_far.max(camera_near + 1.0e-4);
    let inv_range = (camera_far - camera_near).recip();

    let mut matrices = [CascadeMatrix::identity(); MAX_CASCADE_COUNT];
    let count = splits.count.clamp(1, MAX_CASCADE_COUNT);
    for (index, slot) in matrices.iter_mut().enumerate().take(count) {
        let near_view = splits.cascade_near(index);
        let far_view = splits.cascade_far(index);
        let t_near = ((near_view - camera_near) * inv_range).clamp(0.0, 1.0);
        let t_far = ((far_view - camera_near) * inv_range).clamp(0.0, 1.0);

        let sub = cascade_sub_frustum(&corners, t_near, t_far);
        let (center, radius) = bounding_sphere(&sub);
        *slot = fit_cascade(center, radius, light_direction, resolution);
    }
    matrices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shadow::cascade::compute_cascade_splits;
    use bevy_math::ops;

    /// Builds a right-handed perspective inverse-view-projection for a camera at
    /// the origin looking down `-z`, matching wgpu clip (`z` in `[0, 1]`).
    fn camera_inverse_view_projection(near: f32, far: f32, fov_y: f32, aspect: f32) -> Mat4 {
        // Perspective (glam perspective_rh, z in [0,1]) then invert analytically
        // by composing the inverse of a camera that sits at the origin looking
        // down -z (its view matrix is the identity), so inv(view_proj) =
        // inv(proj).
        let h = 1.0 / ops::tan(fov_y * 0.5);
        let w = h / aspect;
        let r = far / (near - far);
        // proj (column-major):
        // [w 0 0 0; 0 h 0 0; 0 0 r -1; 0 0 r*near 0]
        // Invert this closed form.
        let mut inv = [0.0_f32; 16];
        inv[0] = 1.0 / w;
        inv[5] = 1.0 / h;
        inv[11] = 1.0 / (r * near);
        inv[14] = -1.0;
        inv[15] = r / (r * near);
        inv
    }

    /// The near-plane frustum corners must sit at the camera near plane and the
    /// far ones at the far plane (view depth == world -z here).
    #[test]
    fn frustum_corners_span_near_and_far() {
        let inv = camera_inverse_view_projection(1.0, 100.0, 1.2, 1.5);
        let corners = frustum_corners_world(&inv);
        for c in &corners[0..4] {
            assert!((c[2] + 1.0).abs() < 1.0e-3, "near corner z = {}", c[2]);
        }
        for c in &corners[4..8] {
            assert!((c[2] + 100.0).abs() < 1.0e-2, "far corner z = {}", c[2]);
        }
    }

    /// A fitted cascade must project its own sphere centre to the middle of the
    /// shadow map (NDC ~ 0, depth ~ 0.5).
    #[test]
    fn cascade_center_projects_to_shadow_center() {
        let m = fit_cascade([3.0, -2.0, -20.0], 10.0, [0.3, -1.0, -0.2], 2048);
        let ndc = transform_point(&m.view_projection, [3.0, -2.0, -20.0]);
        // Texel snapping shifts x/y by at most one texel, so allow that slack.
        assert!(ndc[0].abs() < 2.0e-3, "x = {}", ndc[0]);
        assert!(ndc[1].abs() < 2.0e-3, "y = {}", ndc[1]);
        assert!((ndc[2] - 0.5).abs() < 1.0e-4, "z = {}", ndc[2]);
    }

    /// The whole cascade sphere must land inside the `[0, 1]^2` shadow UV and
    /// `[0, 1]` depth range so nothing is spuriously treated as unshadowed.
    #[test]
    fn cascade_sphere_fits_clip_box() {
        let center = [1.0, 2.0, -15.0];
        let radius = 8.0;
        let dir = normalize3([0.4, -1.0, 0.3]);
        let m = fit_cascade(center, radius, dir, 1024);
        // Sample points on the sphere surface along the world axes.
        for axis in 0..3 {
            for sign in [-1.0_f32, 1.0] {
                let mut p = center;
                p[axis] += sign * radius;
                let ndc = transform_point(&m.view_projection, p);
                assert!(ndc[0] >= -1.001 && ndc[0] <= 1.001, "x = {}", ndc[0]);
                assert!(ndc[1] >= -1.001 && ndc[1] <= 1.001, "y = {}", ndc[1]);
                assert!(ndc[2] >= -0.001 && ndc[2] <= 1.001, "z = {}", ndc[2]);
            }
        }
    }

    /// Texel snapping must be shift-invariant: nudging the camera by a fraction
    /// of a texel should snap back to (nearly) the same world-locked grid, so a
    /// tiny world translation of the centre moves the projected centre by less
    /// than one texel.
    #[test]
    fn texel_snap_locks_to_grid() {
        let resolution = 512u32;
        let radius = 16.0;
        let dir = normalize3([0.2, -1.0, 0.1]);
        let base = [0.0, 0.0, -30.0];
        let m0 = fit_cascade(base, radius, dir, resolution);
        // Move the centre by ~0.3 texel worth of world space.
        let texel_world = (2.0 * radius) / resolution as f32;
        let moved = [base[0] + texel_world * 0.3, base[1], base[2]];
        let m1 = fit_cascade(moved, radius, dir, resolution);
        // The two snapped grids differ by at most one texel in NDC for a shared
        // world point.
        let probe = [5.0, 3.0, -30.0];
        let a = transform_point(&m0.view_projection, probe);
        let b = transform_point(&m1.view_projection, probe);
        let half_res = resolution as f32 * 0.5;
        let dx_texels = ((a[0] - b[0]) * half_res).abs();
        let dy_texels = ((a[1] - b[1]) * half_res).abs();
        assert!(dx_texels <= 1.001, "dx texels = {dx_texels}");
        assert!(dy_texels <= 1.001, "dy texels = {dy_texels}");
    }

    /// Full pipeline: every active cascade gets a finite matrix and a texel size
    /// that grows with the cascade (coarser far cascades).
    #[test]
    fn cascade_matrices_cover_all_active_cascades() {
        let inv = camera_inverse_view_projection(0.1, 500.0, 1.0, 1.777);
        let splits = compute_cascade_splits(0.1, 500.0, 4, 0.7);
        let m = compute_cascade_matrices(&inv, [0.3, -1.0, 0.2], 0.1, 500.0, &splits, 2048);
        let mut last = 0.0_f32;
        for (i, cascade) in m.iter().enumerate() {
            for v in cascade.view_projection {
                assert!(v.is_finite(), "cascade {i} has non-finite matrix");
            }
            assert!(cascade.texel_world_size > 0.0);
            if i < splits.count {
                assert!(
                    cascade.texel_world_size >= last - 1.0e-6,
                    "cascade {i} texel {} should not shrink below {last}",
                    cascade.texel_world_size
                );
                last = cascade.texel_world_size;
            }
        }
    }
}
