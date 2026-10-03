//! Scalar reference backend.
//!
//! These implementations are the **behavioural ground truth** for every SIMD
//! backend: they operate on plain `[f32; 4]` lanes using the exact arithmetic
//! (and summation order) of the M0 facade. The SSE2/NEON backends must match
//! these results within the tolerances asserted by the cross-check tests.
//!
//! Lane convention: `Vec4`/`Quat` use all four lanes `[x, y, z, w]`; `Vec3A`
//! packs `[x, y, z, 0.0]` so the padding lane never affects a reduction.
#![allow(
    dead_code,
    reason = "the scalar backend is the cross-check reference and the portable               fallback `imp`; on targets whose facade routes to a SIMD backend               (e.g. aarch64/x86_64) these functions are exercised only by the               cross-check tests, so they look unused in a plain library build."
)]

use crate::float::f32 as mf;

/// Component-wise `a + b`.
#[inline]
pub fn vec4_add(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

/// Component-wise `a - b`.
#[inline]
pub fn vec4_sub(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

/// Component-wise `a * b`.
#[inline]
pub fn vec4_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]]
}

/// Component-wise `a / b`.
#[inline]
pub fn vec4_div(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3]]
}

/// Broadcast scalar multiply `a * s`.
#[inline]
pub fn vec4_scale(a: [f32; 4], s: f32) -> [f32; 4] {
    [a[0] * s, a[1] * s, a[2] * s, a[3] * s]
}

/// 4-lane dot product.
#[inline]
pub fn vec4_dot(a: [f32; 4], b: [f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

/// 4-lane Euclidean length.
#[inline]
pub fn vec4_length(a: [f32; 4]) -> f32 {
    mf::sqrt(vec4_dot(a, a))
}

/// 4-lane normalize (`a / |a|`).
#[inline]
pub fn vec4_normalize(a: [f32; 4]) -> [f32; 4] {
    vec4_scale(a, 1.0 / vec4_length(a))
}

/// 3-lane dot product (lane 3 ignored).
#[inline]
pub fn vec3_dot(a: [f32; 4], b: [f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// 3-lane Euclidean length (lane 3 ignored).
#[inline]
pub fn vec3_length(a: [f32; 4]) -> f32 {
    mf::sqrt(vec3_dot(a, a))
}

/// 3-lane normalize; the padding lane is returned as `0.0`.
#[inline]
pub fn vec3_normalize(a: [f32; 4]) -> [f32; 4] {
    let inv = 1.0 / vec3_length(a);
    [a[0] * inv, a[1] * inv, a[2] * inv, 0.0]
}

/// Column-major `4x4 * 4x4` matrix product (`a * b`).
#[inline]
pub fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    [
        mat4_mul_vec4(a, b[0]),
        mat4_mul_vec4(a, b[1]),
        mat4_mul_vec4(a, b[2]),
        mat4_mul_vec4(a, b[3]),
    ]
}

/// Column-major `4x4 * vec4` (`m.col0*v.x + .. + m.col3*v.w`).
#[inline]
pub fn mat4_mul_vec4(m: &[[f32; 4]; 4], v: [f32; 4]) -> [f32; 4] {
    let c0 = vec4_scale(m[0], v[0]);
    let c1 = vec4_scale(m[1], v[1]);
    let c2 = vec4_scale(m[2], v[2]);
    let c3 = vec4_scale(m[3], v[3]);
    vec4_add(vec4_add(c0, c1), vec4_add(c2, c3))
}

/// Hamilton product `a * b` on `[x, y, z, w]` quaternions.
///
/// Matches [`crate::Quat`]'s `Mul` exactly so the scalar backend is bit-for-bit
/// the quaternion reference.
#[inline]
pub fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let (ax, ay, az, aw) = (a[0], a[1], a[2], a[3]);
    let (bx, by, bz, bw) = (b[0], b[1], b[2], b[3]);
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

/// Rotate `v` (`[x, y, z, 0]`) by unit quaternion `q` (`[x, y, z, w]`).
///
/// Uses the same `v + 2w(u x v) + 2 u x (u x v)` form as the facade.
#[inline]
pub fn quat_mul_vec3(q: [f32; 4], v: [f32; 4]) -> [f32; 4] {
    let u = [q[0], q[1], q[2], 0.0];
    let w = q[3];
    let t = vec4_scale(cross3(u, v), 2.0);
    let r = vec4_add(vec4_add(v, vec4_scale(t, w)), cross3(u, t));
    [r[0], r[1], r[2], 0.0]
}

/// 3-lane cross product (lane 3 cleared).
#[inline]
fn cross3(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
        0.0,
    ]
}
