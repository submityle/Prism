//! Hand-written view and projection matrix construction.
//!
//! The renderer needs a single `view_proj` matrix to transform world-space
//! water vertices into clip space. This module builds it from a right-handed
//! look-at view and a reverse-less `zero-to-one` perspective projection that
//! matches the `wgpu` clip-space convention (depth in `[0, 1]`, `y` up).
//!
//! The math is written out explicitly with `f32` arrays rather than pulling in
//! a linear-algebra dependency, so the crate stays lean and the exact matrix
//! layout handed to the shader is unambiguous. All matrices are stored in
//! column-major order, the layout a `WGSL` `mat4x4<f32>` uniform expects.

use prism_render_architecture::water::{cos_approx, sin_approx};

/// A right-handed camera described in world space.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// World-space eye position.
    pub eye: [f32; 3],
    /// World-space point the camera looks at.
    pub target: [f32; 3],
    /// World-space up direction (need not be normalised).
    pub up: [f32; 3],
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Viewport aspect ratio (width divided by height).
    pub aspect: f32,
    /// Near clip-plane distance (strictly positive).
    pub near: f32,
    /// Far clip-plane distance (greater than `near`).
    pub far: f32,
}

impl Camera {
    /// Returns the column-major `view_proj` matrix for this camera.
    #[must_use]
    pub fn view_proj(&self) -> [[f32; 4]; 4] {
        mul(self.projection(), self.view())
    }

    /// Builds the right-handed look-at view matrix (column-major).
    #[must_use]
    pub fn view(&self) -> [[f32; 4]; 4] {
        let f = normalize(sub(self.target, self.eye));
        let s = normalize(cross(f, self.up));
        let u = cross(s, f);
        // Column-major: each inner array is a column.
        [
            [s[0], u[0], -f[0], 0.0],
            [s[1], u[1], -f[1], 0.0],
            [s[2], u[2], -f[2], 0.0],
            [-dot(s, self.eye), -dot(u, self.eye), dot(f, self.eye), 1.0],
        ]
    }

    /// Builds the `zero-to-one` perspective matrix (column-major).
    #[must_use]
    pub fn projection(&self) -> [[f32; 4]; 4] {
        let tan_half = tan_half_fov(self.fov_y);
        let sy = 1.0 / tan_half;
        let sx = sy / self.aspect;
        let range = self.far - self.near;
        let (a, b) = if range.abs() <= f32::EPSILON {
            (0.0, 0.0)
        } else {
            (self.far / range, -(self.far * self.near) / range)
        };
        // Right-handed, depth in `[0, 1]`: the view matrix looks down `-z`, so
        // the perspective divide needs `w = -z_view` (hence the `-1` in the
        // third column) and `z` maps `near -> 0`, `far -> 1`.
        [
            [sx, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, -a, -1.0],
            [0.0, 0.0, b, 0.0],
        ]
    }
}

/// Computes `tan(fov_y / 2)` through the shared deterministic trig.
///
/// The workspace determinism policy forbids the `f32::tan` intrinsic, so this
/// derives the tangent as `sin / cos` from `prism_render_architecture`'s
/// hand-rolled [`sin_approx`]/[`cos_approx`]. Field-of-view angles stay well
/// inside `(0, pi)`, so the half-angle cosine is safely positive; a degenerate
/// near-zero cosine falls back to `1` rather than dividing by zero.
fn tan_half_fov(fov_y: f32) -> f32 {
    let half = fov_y * 0.5;
    let cos = cos_approx(half);
    if cos.abs() <= f32::MIN_POSITIVE {
        return 1.0;
    }
    sin_approx(half) / cos
}

/// Column-major 4x4 multiply, returning `a * b`.
fn mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0_f32; 4]; 4];
    for (col, out_col) in out.iter_mut().enumerate() {
        for (row, cell) in out_col.iter_mut().enumerate() {
            let mut acc = 0.0_f32;
            for k in 0..4 {
                acc += a[k][row] * b[col][k];
            }
            *cell = acc;
        }
    }
    out
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len_sq = dot(v, v);
    if len_sq <= f32::MIN_POSITIVE {
        return [0.0, 0.0, 0.0];
    }
    let inv = 1.0 / len_sq.sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}
