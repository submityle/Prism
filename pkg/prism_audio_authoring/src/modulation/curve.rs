//! Shaping curves for mapping normalized control values.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the curve mapping requirement of design section 12 (linear,
//! exponential, logarithmic, cubic Bezier, and piecewise response shapes). The
//! same curves shape envelope segments (see `super::envelope`) and modulation
//! matrix routes (see `super::matrix`).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use bevy_math::ops;
use prism_audio_core::Sample;

/// A monotone-friendly response curve mapping an input in `[0, 1]` to an output
/// in `[0, 1]`.
///
/// Inputs are clamped into `[0, 1]` before evaluation so callers never produce
/// out-of-range control values. Determinism is guaranteed by routing every
/// transcendental through [`bevy_math::ops`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum Curve {
    /// Identity mapping: output equals the clamped input.
    Linear,
    /// Power-law mapping `x^exponent`. Exponents above `1` bias toward low
    /// output (slow start); exponents below `1` bias toward high output.
    Power {
        /// Strictly positive power applied to the input.
        exponent: Sample,
    },
    /// Exponential ease `(exp(k*x) - 1) / (exp(k) - 1)` with curvature `k`.
    ///
    /// Positive `k` is convex (accelerating), negative `k` is concave; `k`
    /// approaching zero degrades gracefully to the linear mapping.
    Exponential {
        /// Curvature coefficient; the sign selects convex or concave shaping.
        curvature: Sample,
    },
    /// Logarithmic ease: the reflection of [`Curve::Exponential`], fast at the
    /// start and slow toward the end.
    Logarithmic {
        /// Strictly positive curvature coefficient.
        curvature: Sample,
    },
    /// One-dimensional cubic Bezier ease with endpoints pinned at `0` and `1`.
    ///
    /// The two interior control ordinates `y1` and `y2` bend the response; this
    /// matches the feel of hand-drawn automation curves without exposing the
    /// full two-dimensional parametric form.
    Bezier {
        /// Ordinate of the first interior control point (at `x = 1/3`).
        y1: Sample,
        /// Ordinate of the second interior control point (at `x = 2/3`).
        y2: Sample,
    },
    /// Piecewise-linear mapping across sorted `(input, output)` breakpoints.
    ///
    /// Breakpoints are interpolated linearly and clamped at the endpoints,
    /// giving arbitrary user-authored transfer functions.
    Piecewise {
        /// Breakpoints sorted ascending by input; evaluated by linear search.
        points: Vec<CurvePoint>,
    },
}

/// A single `(input, output)` breakpoint used by [`Curve::Piecewise`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CurvePoint {
    /// Input abscissa of the breakpoint.
    pub input: Sample,
    /// Output ordinate of the breakpoint.
    pub output: Sample,
}

impl CurvePoint {
    /// Builds a breakpoint from an input and output value.
    #[inline]
    #[must_use]
    pub fn new(input: Sample, output: Sample) -> Self {
        Self { input, output }
    }
}

impl Curve {
    /// Evaluates the curve at `x`, clamping the input to `[0, 1]`.
    ///
    /// The returned value is clamped to `[0, 1]` for all variants except
    /// [`Curve::Piecewise`], whose output range is defined by its breakpoints.
    #[must_use]
    pub fn map(&self, x: Sample) -> Sample {
        let t = x.clamp(0.0, 1.0);
        match self {
            Curve::Linear => t,
            Curve::Power { exponent } => {
                let e = exponent.max(1.0e-6);
                ops::powf(t, e)
            }
            Curve::Exponential { curvature } => exponential_ease(t, *curvature),
            Curve::Logarithmic { curvature } => {
                // Logarithmic is the point reflection of the exponential ease.
                1.0 - exponential_ease(1.0 - t, curvature.max(1.0e-6))
            }
            Curve::Bezier { y1, y2 } => bezier_ease(t, *y1, *y2),
            Curve::Piecewise { points } => piecewise(points, t),
        }
    }
}

/// Evaluates `(exp(k*t) - 1) / (exp(k) - 1)` with a linear fallback near `k=0`.
#[inline]
fn exponential_ease(t: Sample, curvature: Sample) -> Sample {
    if curvature.abs() < 1.0e-4 {
        return t;
    }
    let num = ops::exp(curvature * t) - 1.0;
    let den = ops::exp(curvature) - 1.0;
    (num / den).clamp(0.0, 1.0)
}

/// Evaluates a cubic Bezier whose control ordinates are `0, y1, y2, 1`.
///
/// The abscissae are fixed at `0, 1/3, 2/3, 1`, so the parameter equals the
/// clamped input and no Newton inversion is required.
#[inline]
fn bezier_ease(t: Sample, y1: Sample, y2: Sample) -> Sample {
    // Endpoint ordinates are pinned at y0 = 0 and y3 = 1.
    let u = 1.0 - t;
    let b1 = 3.0 * u * u * t;
    let b2 = 3.0 * u * t * t;
    let b3 = t * t * t;
    (b1 * y1 + b2 * y2 + b3).clamp(0.0, 1.0)
}

/// Linearly interpolates across sorted breakpoints with endpoint clamping.
fn piecewise(points: &[CurvePoint], t: Sample) -> Sample {
    if points.is_empty() {
        return t;
    }
    if t <= points[0].input {
        return points[0].output;
    }
    let last = points.len() - 1;
    if t >= points[last].input {
        return points[last].output;
    }
    for w in points.windows(2) {
        let a = w[0];
        let b = w[1];
        if t >= a.input && t <= b.input {
            let span = b.input - a.input;
            if span <= 1.0e-9 {
                return b.output;
            }
            let frac = (t - a.input) / span;
            return a.output + (b.output - a.output) * frac;
        }
    }
    points[last].output
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1.0e-4;

    fn close(a: Sample, b: Sample) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn linear_is_identity() {
        let c = Curve::Linear;
        assert!(close(c.map(0.0), 0.0));
        assert!(close(c.map(0.5), 0.5));
        assert!(close(c.map(1.0), 1.0));
    }

    #[test]
    fn input_is_clamped() {
        let c = Curve::Linear;
        assert!(close(c.map(-2.0), 0.0));
        assert!(close(c.map(3.0), 1.0));
    }

    #[test]
    fn power_pins_endpoints() {
        let c = Curve::Power { exponent: 2.0 };
        assert!(close(c.map(0.0), 0.0));
        assert!(close(c.map(1.0), 1.0));
        // x^2 at 0.5 is 0.25.
        assert!(close(c.map(0.5), 0.25));
    }

    #[test]
    fn exponential_pins_endpoints_and_is_monotone() {
        let c = Curve::Exponential { curvature: 3.0 };
        assert!(close(c.map(0.0), 0.0));
        assert!(close(c.map(1.0), 1.0));
        let mut prev = -1.0;
        for i in 0..=20 {
            let x = i as Sample / 20.0;
            let y = c.map(x);
            assert!(y >= prev - EPS, "non-monotone at {x}: {y} < {prev}");
            prev = y;
        }
    }

    #[test]
    fn exponential_degrades_to_linear() {
        let c = Curve::Exponential { curvature: 0.0 };
        assert!(close(c.map(0.3), 0.3));
    }

    #[test]
    fn logarithmic_is_concave() {
        let c = Curve::Logarithmic { curvature: 3.0 };
        // Concave: value at the midpoint sits above the linear line.
        assert!(c.map(0.5) > 0.5);
        assert!(close(c.map(0.0), 0.0));
        assert!(close(c.map(1.0), 1.0));
    }

    #[test]
    fn bezier_pins_endpoints() {
        let c = Curve::Bezier { y1: 0.0, y2: 1.0 };
        assert!(close(c.map(0.0), 0.0));
        assert!(close(c.map(1.0), 1.0));
    }

    #[test]
    fn piecewise_interpolates() {
        let c = Curve::Piecewise {
            points: alloc::vec![
                CurvePoint::new(0.0, 0.0),
                CurvePoint::new(0.5, 0.8),
                CurvePoint::new(1.0, 1.0),
            ],
        };
        assert!(close(c.map(0.25), 0.4));
        assert!(close(c.map(0.5), 0.8));
        assert!(close(c.map(0.75), 0.9));
    }

    #[test]
    fn piecewise_clamps_outside() {
        let c = Curve::Piecewise {
            points: alloc::vec![CurvePoint::new(0.2, 0.3), CurvePoint::new(0.8, 0.9)],
        };
        assert!(close(c.map(0.0), 0.3));
        assert!(close(c.map(1.0), 0.9));
    }
}
