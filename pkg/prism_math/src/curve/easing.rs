//! Scalar easing functions mapping `t` in `[0, 1]` to an eased value.
//!
//! Every function satisfies the boundary conditions `f(0) == 0` and
//! `f(1) == 1`. Inputs are not clamped; callers should pass `t` within
//! `[0, 1]` for the documented behaviour.

use crate::float::f32 as mf;

/// Smooth Hermite interpolation `3t^2 - 2t^3`, clamped to `[0, 1]`.
#[inline]
pub fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Perlin's smoother variant `6t^5 - 15t^4 + 10t^3`, clamped to `[0, 1]`.
#[inline]
pub fn smootherstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Quadratic ease-in (`t^2`).
#[inline]
pub fn quad_in(t: f32) -> f32 {
    t * t
}

/// Quadratic ease-out.
#[inline]
pub fn quad_out(t: f32) -> f32 {
    t * (2.0 - t)
}

/// Quadratic ease-in-out.
#[inline]
pub fn quad_in_out(t: f32) -> f32 {
    if t < 0.5 {
        2.0 * t * t
    } else {
        let u = -2.0 * t + 2.0;
        1.0 - u * u * 0.5
    }
}

/// Cubic ease-in (`t^3`).
#[inline]
pub fn cubic_in(t: f32) -> f32 {
    t * t * t
}

/// Cubic ease-out.
#[inline]
pub fn cubic_out(t: f32) -> f32 {
    let u = 1.0 - t;
    1.0 - u * u * u
}

/// Cubic ease-in-out.
#[inline]
pub fn cubic_in_out(t: f32) -> f32 {
    if t < 0.5 {
        4.0 * t * t * t
    } else {
        let u = -2.0 * t + 2.0;
        1.0 - u * u * u * 0.5
    }
}

/// Sinusoidal ease-in.
#[inline]
pub fn sine_in(t: f32) -> f32 {
    1.0 - mf::cos(t * core::f32::consts::FRAC_PI_2)
}

/// Sinusoidal ease-out.
#[inline]
pub fn sine_out(t: f32) -> f32 {
    mf::sin(t * core::f32::consts::FRAC_PI_2)
}

/// Sinusoidal ease-in-out.
#[inline]
pub fn sine_in_out(t: f32) -> f32 {
    -0.5 * (mf::cos(core::f32::consts::PI * t) - 1.0)
}

/// Exponential ease-in (base-2), with `f(0) == 0` enforced exactly.
#[inline]
pub fn expo_in(t: f32) -> f32 {
    if t <= 0.0 { 0.0 } else { mf::powf(2.0, 10.0 * (t - 1.0)) }
}

/// Exponential ease-out (base-2), with `f(1) == 1` enforced exactly.
#[inline]
pub fn expo_out(t: f32) -> f32 {
    if t >= 1.0 { 1.0 } else { 1.0 - mf::powf(2.0, -10.0 * t) }
}

/// Exponential ease-in-out (base-2), with the endpoints enforced exactly.
#[inline]
pub fn expo_in_out(t: f32) -> f32 {
    if t <= 0.0 {
        0.0
    } else if t >= 1.0 {
        1.0
    } else if t < 0.5 {
        0.5 * mf::powf(2.0, 20.0 * t - 10.0)
    } else {
        1.0 - 0.5 * mf::powf(2.0, -20.0 * t + 10.0)
    }
}
