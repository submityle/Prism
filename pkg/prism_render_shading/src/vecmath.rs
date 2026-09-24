//! Backend-neutral `[f32; 3]` vector helpers shared by every analytic BSDF
//! lobe (principled, toon, cloth, subsurface).
//!
//! These are deliberately tiny, dependency-free and use only the arithmetic
//! that maps one-to-one onto WESL builtins, so the CPU golden reference stays
//! byte-for-byte in step with the GPU `brdf.wesl` / `cloth.wesl` twins.

/// Dot product of two 3-vectors.
pub(crate) fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Component-wise sum.
pub(crate) fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Component-wise difference.
pub(crate) fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Component-wise (Hadamard) product.
pub(crate) fn mul(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

/// Scales every component by `scalar`.
pub(crate) fn mul_scalar(value: [f32; 3], scalar: f32) -> [f32; 3] {
    [value[0] * scalar, value[1] * scalar, value[2] * scalar]
}

/// Linear interpolation `a + (b - a) * factor`, evaluated component-wise.
pub(crate) fn mix3(a: [f32; 3], b: [f32; 3], factor: f32) -> [f32; 3] {
    add(mul_scalar(a, 1.0 - factor), mul_scalar(b, factor))
}

/// Normalizes `value`, returning `fallback` for degenerate/non-finite inputs.
pub(crate) fn normalize_or(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let length_squared = dot(value, value);
    if length_squared > 1.0e-12 && length_squared.is_finite() {
        mul_scalar(value, length_squared.sqrt().recip())
    } else {
        fallback
    }
}

/// Clamps every component into `[0, 1]`.
pub(crate) fn saturate3(value: [f32; 3]) -> [f32; 3] {
    [
        value[0].clamp(0.0, 1.0),
        value[1].clamp(0.0, 1.0),
        value[2].clamp(0.0, 1.0),
    ]
}
