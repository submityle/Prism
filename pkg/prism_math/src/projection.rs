//! Camera **view** and **projection** matrices.
//!
//! These complete the camera-math gap called out by the `prism_math` design
//! doc (§24.1 CPU/GPU consistency, §24.10 honest boundary): the kernel had no
//! `perspective`/`orthographic` constructors, so the GPU shader-mirror for
//! projection could not be delivered. This module supplies the full AAA family
//! so a projection built on the CPU and one built in a shader can be compared
//! bit-close (see `prism_math_gpu`).
//!
//! ## Conventions (inherited from the crate)
//! - **Column-major** matrices that multiply on the left (`m * v`); column `c`
//!   of the returned [`Mat4`] is `from_cols`' `c`-th argument.
//! - **Right-handed** view space is primary: the camera looks down **-Z**, `+X`
//!   is right and `+Y` is up. Left-handed variants (camera looks down `+Z`) are
//!   provided with the `_lh` suffix.
//! - **Clip-space depth** differs by graphics API. The default constructors map
//!   depth to **`[0, 1]`** (Vulkan / Direct3D / Metal / WebGPU). The `_gl`
//!   suffix maps depth to **`[-1, 1]`** (OpenGL).
//! - **Reverse-Z** variants map the near plane to `1` and the far plane to `0`,
//!   which pairs with a `GREATER` depth test and a float depth buffer to spread
//!   precision evenly across the view frustum. Reverse-Z is the modern AAA
//!   default and is only meaningful for the `[0, 1]` clip range.
//!
//! All angles are in **radians**. `aspect` is width / height.

use crate::float::f32 as mf;
use crate::mat::Mat4;
use crate::vec::{Vec3, Vec4};

// ===========================================================================
// Perspective (right-handed, camera looks down -Z)
// ===========================================================================

/// Right-handed perspective projection mapping depth to `[0, 1]`
/// (Vulkan / D3D / Metal / WebGPU).
///
/// `z_near` maps to clip depth `0`, `z_far` to `1`. Requires
/// `z_near > 0`, `z_far > z_near`, finite `aspect > 0`.
#[inline]
#[must_use]
pub fn perspective_rh(fovy_radians: f32, aspect: f32, z_near: f32, z_far: f32) -> Mat4 {
    let f = 1.0 / mf::tan(fovy_radians * 0.5);
    let r = z_far / (z_near - z_far);
    Mat4::from_cols(
        Vec4::new(f / aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, f, 0.0, 0.0),
        Vec4::new(0.0, 0.0, r, -1.0),
        Vec4::new(0.0, 0.0, r * z_near, 0.0),
    )
}

/// Right-handed perspective projection mapping depth to `[-1, 1]` (OpenGL).
#[inline]
#[must_use]
pub fn perspective_rh_gl(fovy_radians: f32, aspect: f32, z_near: f32, z_far: f32) -> Mat4 {
    let f = 1.0 / mf::tan(fovy_radians * 0.5);
    let inv = 1.0 / (z_near - z_far);
    Mat4::from_cols(
        Vec4::new(f / aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, f, 0.0, 0.0),
        Vec4::new(0.0, 0.0, (z_far + z_near) * inv, -1.0),
        Vec4::new(0.0, 0.0, 2.0 * z_far * z_near * inv, 0.0),
    )
}

/// Right-handed **reverse-Z** perspective mapping `z_near -> 1`, `z_far -> 0`
/// (clip range `[0, 1]`). Use with a `GREATER`/`GREATER_EQUAL` depth test and a
/// float depth buffer for maximum precision.
#[inline]
#[must_use]
pub fn perspective_reverse_z_rh(fovy_radians: f32, aspect: f32, z_near: f32, z_far: f32) -> Mat4 {
    let f = 1.0 / mf::tan(fovy_radians * 0.5);
    let inv = 1.0 / (z_far - z_near);
    Mat4::from_cols(
        Vec4::new(f / aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, f, 0.0, 0.0),
        Vec4::new(0.0, 0.0, z_near * inv, -1.0),
        Vec4::new(0.0, 0.0, z_far * z_near * inv, 0.0),
    )
}

/// Right-handed perspective with an **infinite far plane** (clip range
/// `[0, 1)`). `z_near` maps to `0`; depth asymptotically approaches `1`.
#[inline]
#[must_use]
pub fn perspective_infinite_rh(fovy_radians: f32, aspect: f32, z_near: f32) -> Mat4 {
    let f = 1.0 / mf::tan(fovy_radians * 0.5);
    Mat4::from_cols(
        Vec4::new(f / aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, f, 0.0, 0.0),
        Vec4::new(0.0, 0.0, -1.0, -1.0),
        Vec4::new(0.0, 0.0, -z_near, 0.0),
    )
}

/// Right-handed **reverse-Z, infinite far** perspective (clip range `(0, 1]`),
/// mapping `z_near -> 1` and depth asymptotically approaching `0`. This is the
/// modern AAA default projection for open worlds: constant near precision with
/// no far-plane clamp.
#[inline]
#[must_use]
pub fn perspective_infinite_reverse_z_rh(fovy_radians: f32, aspect: f32, z_near: f32) -> Mat4 {
    let f = 1.0 / mf::tan(fovy_radians * 0.5);
    Mat4::from_cols(
        Vec4::new(f / aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, f, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, -1.0),
        Vec4::new(0.0, 0.0, z_near, 0.0),
    )
}

// ===========================================================================
// Perspective (left-handed, camera looks down +Z)
// ===========================================================================

/// Left-handed perspective projection mapping depth to `[0, 1]`.
#[inline]
#[must_use]
pub fn perspective_lh(fovy_radians: f32, aspect: f32, z_near: f32, z_far: f32) -> Mat4 {
    let f = 1.0 / mf::tan(fovy_radians * 0.5);
    let r = z_far / (z_far - z_near);
    Mat4::from_cols(
        Vec4::new(f / aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, f, 0.0, 0.0),
        Vec4::new(0.0, 0.0, r, 1.0),
        Vec4::new(0.0, 0.0, -r * z_near, 0.0),
    )
}

/// Left-handed perspective projection mapping depth to `[-1, 1]` (OpenGL).
#[inline]
#[must_use]
pub fn perspective_lh_gl(fovy_radians: f32, aspect: f32, z_near: f32, z_far: f32) -> Mat4 {
    let f = 1.0 / mf::tan(fovy_radians * 0.5);
    let inv = 1.0 / (z_far - z_near);
    Mat4::from_cols(
        Vec4::new(f / aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, f, 0.0, 0.0),
        Vec4::new(0.0, 0.0, (z_far + z_near) * inv, 1.0),
        Vec4::new(0.0, 0.0, -2.0 * z_far * z_near * inv, 0.0),
    )
}

// ===========================================================================
// Orthographic
// ===========================================================================

/// Right-handed orthographic projection mapping depth to `[0, 1]`.
#[inline]
#[must_use]
pub fn orthographic_rh(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    z_near: f32,
    z_far: f32,
) -> Mat4 {
    let rcp_w = 1.0 / (right - left);
    let rcp_h = 1.0 / (top - bottom);
    let rcp_d = 1.0 / (z_near - z_far);
    Mat4::from_cols(
        Vec4::new(2.0 * rcp_w, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 2.0 * rcp_h, 0.0, 0.0),
        Vec4::new(0.0, 0.0, rcp_d, 0.0),
        Vec4::new(
            -(right + left) * rcp_w,
            -(top + bottom) * rcp_h,
            z_near * rcp_d,
            1.0,
        ),
    )
}

/// Right-handed orthographic projection mapping depth to `[-1, 1]` (OpenGL).
#[inline]
#[must_use]
pub fn orthographic_rh_gl(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    z_near: f32,
    z_far: f32,
) -> Mat4 {
    let rcp_w = 1.0 / (right - left);
    let rcp_h = 1.0 / (top - bottom);
    let rcp_d = 1.0 / (z_far - z_near);
    Mat4::from_cols(
        Vec4::new(2.0 * rcp_w, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 2.0 * rcp_h, 0.0, 0.0),
        Vec4::new(0.0, 0.0, -2.0 * rcp_d, 0.0),
        Vec4::new(
            -(right + left) * rcp_w,
            -(top + bottom) * rcp_h,
            -(z_far + z_near) * rcp_d,
            1.0,
        ),
    )
}

/// Left-handed orthographic projection mapping depth to `[0, 1]`.
#[inline]
#[must_use]
pub fn orthographic_lh(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    z_near: f32,
    z_far: f32,
) -> Mat4 {
    let rcp_w = 1.0 / (right - left);
    let rcp_h = 1.0 / (top - bottom);
    let rcp_d = 1.0 / (z_far - z_near);
    Mat4::from_cols(
        Vec4::new(2.0 * rcp_w, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 2.0 * rcp_h, 0.0, 0.0),
        Vec4::new(0.0, 0.0, rcp_d, 0.0),
        Vec4::new(
            -(right + left) * rcp_w,
            -(top + bottom) * rcp_h,
            -z_near * rcp_d,
            1.0,
        ),
    )
}

// ===========================================================================
// View (look-at) matrices
// ===========================================================================

/// Right-handed look-at view matrix: the camera at `eye` looks toward `target`
/// with the given `up` hint (camera forward is `-Z`).
#[inline]
#[must_use]
pub fn look_at_rh(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    look_to_rh(eye, target - eye, up)
}

/// Right-handed look-to view matrix using an explicit forward `dir`
/// (need not be normalized).
#[inline]
#[must_use]
pub fn look_to_rh(eye: Vec3, dir: Vec3, up: Vec3) -> Mat4 {
    let f = dir.normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(f);
    Mat4::from_cols(
        Vec4::new(s.x, u.x, -f.x, 0.0),
        Vec4::new(s.y, u.y, -f.y, 0.0),
        Vec4::new(s.z, u.z, -f.z, 0.0),
        Vec4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
    )
}

/// Left-handed look-at view matrix (camera forward is `+Z`).
#[inline]
#[must_use]
pub fn look_at_lh(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    look_to_lh(eye, target - eye, up)
}

/// Left-handed look-to view matrix using an explicit forward `dir`.
#[inline]
#[must_use]
pub fn look_to_lh(eye: Vec3, dir: Vec3, up: Vec3) -> Mat4 {
    let f = dir.normalize();
    let s = up.cross(f).normalize();
    let u = f.cross(s);
    Mat4::from_cols(
        Vec4::new(s.x, u.x, f.x, 0.0),
        Vec4::new(s.y, u.y, f.y, 0.0),
        Vec4::new(s.z, u.z, f.z, 0.0),
        Vec4::new(-s.dot(eye), -u.dot(eye), -f.dot(eye), 1.0),
    )
}

#[cfg(test)]
mod tests;
