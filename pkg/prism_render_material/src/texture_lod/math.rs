//! Tiny dependency-free vector helpers shared by the texture-LOD estimators.
//!
//! The material crate deliberately avoids pulling a linear-algebra dependency
//! into its stable ABI surface, so the handful of 2D/3D operations needed here
//! are provided locally over plain `[f32; N]` arrays.
//!
//! # Conventions
//! All operations are the textbook definitions; no normalization is implied
//! unless the function name says so. See the module docs in [`super`].

/// Dot product of two 3D vectors.
#[inline]
#[must_use]
pub(crate) fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Component-wise difference `a - b` of two 3D points.
#[inline]
#[must_use]
pub(crate) fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Cross product of two 3D vectors.
#[inline]
#[must_use]
pub(crate) fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Euclidean length of a 3D vector.
#[inline]
#[must_use]
pub(crate) fn length3(a: [f32; 3]) -> f32 {
    dot3(a, a).sqrt()
}

/// Signed twice-area of the UV triangle `(uv0, uv1, uv2)` (the 2D cross
/// product of its two edge vectors). The sign encodes winding; callers that
/// need an area take the absolute value.
#[inline]
#[must_use]
pub(crate) fn uv_double_area(uv0: [f32; 2], uv1: [f32; 2], uv2: [f32; 2]) -> f32 {
    let e1 = [uv1[0] - uv0[0], uv1[1] - uv0[1]];
    let e2 = [uv2[0] - uv0[0], uv2[1] - uv0[1]];
    e1[0] * e2[1] - e1[1] * e2[0]
}
