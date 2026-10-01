//! Easing curves mapping normalized time `[0, 1]` to a normalized progress.
//!
//! All curves accept an input `t` that is clamped into `[0, 1]` before being
//! evaluated. Standard curves satisfy `sample(0.0) == 0.0` and
//! `sample(1.0) == 1.0`. The `Back*` family intentionally overshoots the
//! `[0, 1]` range in its interior while still pinning the endpoints.

use crate::math::{absf, clampf, cosf, exp2f, floorf, sinf, sqrtf};
use core::f32::consts::{FRAC_PI_2, PI};

/// Overshoot constant used by the `Back` easing family (matches the common
/// `back` definition where the overshoot amounts to roughly 10%).
const BACK_C1: f32 = 1.701_58;
/// Secondary overshoot constant for the in/out `Back` variants.
const BACK_C2: f32 = BACK_C1 * 1.525;

/// Where the discontinuous jumps of a [`Easing::Steps`] curve occur.
///
/// These correspond to the CSS `steps()` jump terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepPosition {
    /// Jump at the start of each interval (`jump-start` / `start`).
    JumpStart,
    /// Jump at the end of each interval (`jump-end` / `end`, the default).
    JumpEnd,
    /// No jump at either endpoint (`jump-none`); needs at least two steps.
    JumpNone,
    /// Jump at both endpoints (`jump-both`).
    JumpBoth,
}

/// An easing curve.
///
/// Use [`Easing::sample`] to evaluate the curve at a normalized time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Easing {
    /// Identity curve: `sample(t) == t`.
    Linear,

    /// CSS `ease` preset, `cubic-bezier(0.25, 0.1, 0.25, 1.0)`.
    EaseInOut,
    /// CSS `ease-in` preset, `cubic-bezier(0.42, 0.0, 1.0, 1.0)`.
    EaseIn,
    /// CSS `ease-out` preset, `cubic-bezier(0.0, 0.0, 0.58, 1.0)`.
    EaseOut,

    /// Quadratic ease-in (`t^2`).
    QuadIn,
    /// Quadratic ease-out.
    QuadOut,
    /// Quadratic ease-in-out.
    QuadInOut,

    /// Cubic ease-in (`t^3`).
    CubicIn,
    /// Cubic ease-out.
    CubicOut,
    /// Cubic ease-in-out.
    CubicInOut,

    /// Quartic ease-in (`t^4`).
    QuartIn,
    /// Quartic ease-out.
    QuartOut,
    /// Quartic ease-in-out.
    QuartInOut,

    /// Quintic ease-in (`t^5`).
    QuintIn,
    /// Quintic ease-out.
    QuintOut,
    /// Quintic ease-in-out.
    QuintInOut,

    /// Sinusoidal ease-in.
    SineIn,
    /// Sinusoidal ease-out.
    SineOut,
    /// Sinusoidal ease-in-out.
    SineInOut,

    /// Exponential ease-in.
    ExpoIn,
    /// Exponential ease-out.
    ExpoOut,
    /// Exponential ease-in-out.
    ExpoInOut,

    /// Circular ease-in.
    CircIn,
    /// Circular ease-out.
    CircOut,
    /// Circular ease-in-out.
    CircInOut,

    /// Overshooting ease-in (undershoots below `0.0`).
    BackIn,
    /// Overshooting ease-out (overshoots above `1.0`).
    BackOut,
    /// Overshooting ease-in-out.
    BackInOut,

    /// Stepped curve with `count` steps and a jump [`StepPosition`].
    Steps(u32, StepPosition),

    /// General cubic Bézier easing with control points `(x1, y1, x2, y2)`,
    /// solved like CSS `cubic-bezier()`. The two implicit anchors are fixed at
    /// `(0, 0)` and `(1, 1)`. `x1` and `x2` are clamped into `[0, 1]` so the
    /// curve remains a function of time.
    CubicBezier(f32, f32, f32, f32),
}

impl Default for Easing {
    #[inline]
    fn default() -> Self {
        Easing::Linear
    }
}

impl Easing {
    /// Evaluate the easing curve at normalized time `t`.
    ///
    /// The input is clamped into `[0, 1]`. For standard curves the output is
    /// also within `[0, 1]`; the `Back*` family may overshoot in its interior.
    pub fn sample(&self, t: f32) -> f32 {
        let t = clampf(t, 0.0, 1.0);
        match *self {
            Easing::Linear => t,

            Easing::EaseInOut => cubic_bezier(0.25, 0.1, 0.25, 1.0, t),
            Easing::EaseIn => cubic_bezier(0.42, 0.0, 1.0, 1.0, t),
            Easing::EaseOut => cubic_bezier(0.0, 0.0, 0.58, 1.0, t),

            Easing::QuadIn => t * t,
            Easing::QuadOut => {
                let u = 1.0 - t;
                1.0 - u * u
            }
            Easing::QuadInOut => in_out(t, |x| x * x),

            Easing::CubicIn => t * t * t,
            Easing::CubicOut => {
                let u = 1.0 - t;
                1.0 - u * u * u
            }
            Easing::CubicInOut => in_out(t, |x| x * x * x),

            Easing::QuartIn => t * t * t * t,
            Easing::QuartOut => {
                let u = 1.0 - t;
                1.0 - u * u * u * u
            }
            Easing::QuartInOut => in_out(t, |x| x * x * x * x),

            Easing::QuintIn => t * t * t * t * t,
            Easing::QuintOut => {
                let u = 1.0 - t;
                1.0 - u * u * u * u * u
            }
            Easing::QuintInOut => in_out(t, |x| x * x * x * x * x),

            Easing::SineIn => 1.0 - cosf(t * FRAC_PI_2),
            Easing::SineOut => sinf(t * FRAC_PI_2),
            Easing::SineInOut => -(cosf(PI * t) - 1.0) / 2.0,

            Easing::ExpoIn => {
                if t <= 0.0 {
                    0.0
                } else {
                    exp2f(10.0 * t - 10.0)
                }
            }
            Easing::ExpoOut => {
                if t >= 1.0 {
                    1.0
                } else {
                    1.0 - exp2f(-10.0 * t)
                }
            }
            Easing::ExpoInOut => {
                if t <= 0.0 {
                    0.0
                } else if t >= 1.0 {
                    1.0
                } else if t < 0.5 {
                    exp2f(20.0 * t - 10.0) / 2.0
                } else {
                    (2.0 - exp2f(-20.0 * t + 10.0)) / 2.0
                }
            }

            Easing::CircIn => 1.0 - sqrtf(1.0 - t * t),
            Easing::CircOut => {
                let u = t - 1.0;
                sqrtf(1.0 - u * u)
            }
            Easing::CircInOut => {
                if t < 0.5 {
                    let v = 2.0 * t;
                    (1.0 - sqrtf(1.0 - v * v)) / 2.0
                } else {
                    let v = -2.0 * t + 2.0;
                    (sqrtf(1.0 - v * v) + 1.0) / 2.0
                }
            }

            Easing::BackIn => (BACK_C1 + 1.0) * t * t * t - BACK_C1 * t * t,
            Easing::BackOut => {
                let u = t - 1.0;
                1.0 + (BACK_C1 + 1.0) * u * u * u + BACK_C1 * u * u
            }
            Easing::BackInOut => {
                if t < 0.5 {
                    let v = 2.0 * t;
                    (v * v * ((BACK_C2 + 1.0) * v - BACK_C2)) / 2.0
                } else {
                    let v = 2.0 * t - 2.0;
                    (v * v * ((BACK_C2 + 1.0) * v + BACK_C2) + 2.0) / 2.0
                }
            }

            Easing::Steps(count, position) => steps(t, count, position),

            Easing::CubicBezier(x1, y1, x2, y2) => {
                cubic_bezier(clampf(x1, 0.0, 1.0), y1, clampf(x2, 0.0, 1.0), y2, t)
            }
        }
    }
}

/// Build an in/out curve from an ease-in shape `f` defined on `[0, 1]`.
#[inline]
fn in_out(t: f32, f: impl Fn(f32) -> f32) -> f32 {
    if t < 0.5 {
        f(2.0 * t) / 2.0
    } else {
        1.0 - f(2.0 - 2.0 * t) / 2.0
    }
}

/// Evaluate a stepped easing curve.
fn steps(t: f32, count: u32, position: StepPosition) -> f32 {
    if count == 0 {
        return t;
    }
    let n = count as f32;
    let raw = floorf(t * n);
    match position {
        StepPosition::JumpStart => {
            let step = clampf(raw + 1.0, 0.0, n);
            step / n
        }
        StepPosition::JumpEnd => {
            let step = clampf(raw, 0.0, n);
            step / n
        }
        StepPosition::JumpNone => {
            if count == 1 {
                // Degenerate: a single step with no end jumps collapses to 0.
                return 0.0;
            }
            let step = clampf(raw, 0.0, n - 1.0);
            step / (n - 1.0)
        }
        StepPosition::JumpBoth => {
            let step = clampf(raw, 0.0, n);
            (step + 1.0) / (n + 1.0)
        }
    }
}

/// Maximum Newton iterations when inverting the Bézier `x(s)` mapping.
const BEZIER_NEWTON_ITERS: u32 = 8;
/// Convergence tolerance for the Bézier parameter solve.
const BEZIER_EPSILON: f32 = 1e-6;

/// One coordinate of a cubic Bézier with implicit anchors at 0 and 1.
///
/// `p1` and `p2` are the control-point coordinates; `s` is the Bézier
/// parameter in `[0, 1]`.
#[inline]
fn bezier_coord(p1: f32, p2: f32, s: f32) -> f32 {
    // Expanded Bernstein form with P0 = 0 and P3 = 1.
    let a = 1.0 + 3.0 * p1 - 3.0 * p2;
    let b = 3.0 * p2 - 6.0 * p1;
    let c = 3.0 * p1;
    ((a * s + b) * s + c) * s
}

/// Derivative of [`bezier_coord`] with respect to `s`.
#[inline]
fn bezier_slope(p1: f32, p2: f32, s: f32) -> f32 {
    let a = 1.0 + 3.0 * p1 - 3.0 * p2;
    let b = 3.0 * p2 - 6.0 * p1;
    let c = 3.0 * p1;
    (3.0 * a * s + 2.0 * b) * s + c
}

/// Solve a CSS-style cubic Bézier easing for the output `y` given input `x`.
///
/// First finds the Bézier parameter `s` such that `x(s) == x` using
/// Newton–Raphson with a bisection fallback, then returns `y(s)`.
fn cubic_bezier(x1: f32, y1: f32, x2: f32, y2: f32, x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Newton–Raphson from an identity initial guess.
    let mut s = x;
    let mut i = 0;
    while i < BEZIER_NEWTON_ITERS {
        let error = bezier_coord(x1, x2, s) - x;
        if absf(error) < BEZIER_EPSILON {
            return bezier_coord(y1, y2, s);
        }
        let slope = bezier_slope(x1, x2, s);
        if absf(slope) < 1e-6 {
            break;
        }
        s -= error / slope;
        i += 1;
    }

    // Bisection fallback guarantees convergence on the monotonic x(s) mapping.
    let mut lo = 0.0_f32;
    let mut hi = 1.0_f32;
    let mut mid = clampf(s, 0.0, 1.0);
    let mut j = 0;
    while j < 64 {
        let value = bezier_coord(x1, x2, mid);
        if absf(value - x) < BEZIER_EPSILON {
            break;
        }
        if value < x {
            lo = mid;
        } else {
            hi = mid;
        }
        mid = (lo + hi) / 2.0;
        j += 1;
    }
    bezier_coord(y1, y2, mid)
}
