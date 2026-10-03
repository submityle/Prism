//! Interpolation, easing, and spline helpers.
//!
//! The spline and interpolation functions are generic over any
//! [`Interpolatable`] value (every `f32` vector type and `f32` itself), while
//! the [`easing`] functions remap a scalar parameter `t` in `[0, 1]`.
//!
//! Scalar quaternion `slerp`/`nlerp` live on [`crate::Quat`]; this module reuses
//! them rather than redefining rotation blending.

use core::ops::{Add, Mul, Sub};

pub mod easing;
pub mod spline;

pub use easing::{
    cubic_in, cubic_in_out, cubic_out, expo_in, expo_in_out, expo_out, quad_in, quad_in_out,
    quad_out, sine_in, sine_in_out, sine_out, smootherstep, smoothstep,
};
pub use spline::{
    bezier_cubic, bezier_cubic_tangent, catmull_rom, catmull_rom_tangent, hermite, hermite_tangent,
};

/// A value that supports affine combinations: componentwise add/subtract and
/// scaling by an `f32`. Blanket-implemented for `f32` and every `f32` vector
/// type in this crate.
pub trait Interpolatable:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<f32, Output = Self>
{
}

impl<T> Interpolatable for T where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<f32, Output = T>
{
}

/// Generic linear interpolation: `a` at `t = 0`, `b` at `t = 1`.
///
/// `t` is not clamped, so values outside `[0, 1]` extrapolate.
#[inline]
pub fn lerp<T: Interpolatable>(a: T, b: T, t: f32) -> T {
    a + (b - a) * t
}

/// Inverse of [`lerp`] for scalars: the parameter `t` such that
/// `lerp(a, b, t) == value`. Returns `0.0` when `a == b`.
#[inline]
pub fn inverse_lerp(a: f32, b: f32, value: f32) -> f32 {
    let denom = b - a;
    if denom == 0.0 { 0.0 } else { (value - a) / denom }
}

/// Remap `value` from the input range `[in_min, in_max]` to the output range
/// `[out_min, out_max]`.
#[inline]
pub fn remap(value: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32 {
    lerp(out_min, out_max, inverse_lerp(in_min, in_max, value))
}
