//! Parameter mapping curves: a monotone-in-x breakpoint function that maps a
//! game-side scalar (an RTPC value, a blend position, a distance) onto a
//! target engine value (a volume in dB, a pitch in semitones, a filter cutoff)
//! through a sequence of segments, each carrying its own interpolation shape.
//!
//! This is the data-driven heart of RTPC and blend containers: a sound
//! designer draws a curve in an authoring tool, and the runtime samples it to
//! translate a continuous game quantity into a continuous engine parameter.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. It is an original
//! data-driven mapping layer built from elementary interpolation mathematics
//! (linear, power, S-curve, logarithmic). It is pure classic math with no
//! AI/ML.
//!
//! # Relationship
//!
//! [`ParameterCurve`] is consumed by [`crate::rtpc`] (game parameter to engine
//! target), by [`crate::container`] blend containers (position to per-layer
//! gain), and anywhere a continuous mapping is authored. It deliberately holds
//! no audio state and performs no DSP; it is a pure function sampler.

use alloc::vec::Vec;
use bevy_math::ops;
use prism_audio_core::math::{lerp, Sample};

/// The interpolation shape applied across a single curve segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Interpolation {
    /// Hold the start value across the whole segment, jumping at the end.
    Constant,
    /// Straight line between the two breakpoints.
    Linear,
    /// Smooth acceleration out of the start point (`t^2`).
    EaseIn,
    /// Smooth deceleration into the end point (`1 - (1 - t)^2`).
    EaseOut,
    /// Symmetric smoothstep S-curve (`3t^2 - 2t^3`).
    SCurve,
    /// Perceptual/logarithmic rise: fast early, slow late. Shaped so a value
    /// swept across the segment sounds even to the ear for gain-like targets.
    Logarithmic,
    /// Exponential rise: slow early, fast late (mirror of [`Self::Logarithmic`]).
    Exponential,
}

impl Interpolation {
    /// Shapes a normalized position `t` in `[0, 1]` according to the curve.
    ///
    /// Returns the eased position, still in `[0, 1]`; the caller linearly
    /// blends the two breakpoint y-values with the shaped position.
    #[must_use]
    pub fn shape(self, t: Sample) -> Sample {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Constant => 0.0,
            Self::Linear => t,
            Self::EaseIn => t * t,
            Self::EaseOut => {
                let u = 1.0 - t;
                1.0 - u * u
            }
            Self::SCurve => t * t * (3.0 - 2.0 * t),
            // ln(1 + k t) / ln(1 + k): monotone, concave-down, 0->0, 1->1.
            Self::Logarithmic => {
                const K: Sample = 9.0;
                ops::ln(1.0 + K * t) / ops::ln(1.0 + K)
            }
            // (exp(k t) - 1) / (exp(k) - 1): monotone, concave-up, 0->0, 1->1.
            Self::Exponential => {
                const K: Sample = 2.197_224_6; // ln(9), mirror of Logarithmic.
                (ops::exp(K * t) - 1.0) / (ops::exp(K) - 1.0)
            }
        }
    }
}

/// A single authored breakpoint: an input coordinate `x` paired with the
/// output value `y` reached there, plus the interpolation used to approach the
/// *next* breakpoint.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Breakpoint {
    /// Input coordinate (game-side quantity).
    pub x: Sample,
    /// Output value at this coordinate (engine-side target).
    pub y: Sample,
    /// Shape used to interpolate from this point toward the next one.
    pub interpolation: Interpolation,
}

impl Breakpoint {
    /// Creates a breakpoint with an explicit interpolation toward the next.
    #[must_use]
    pub fn new(x: Sample, y: Sample, interpolation: Interpolation) -> Self {
        Self { x, y, interpolation }
    }

    /// Creates a linearly-interpolated breakpoint.
    #[must_use]
    pub fn linear(x: Sample, y: Sample) -> Self {
        Self::new(x, y, Interpolation::Linear)
    }
}

/// A piecewise mapping curve made of ordered [`Breakpoint`]s.
///
/// The curve is defined for all real inputs: inputs below the first breakpoint
/// clamp to the first y-value, inputs above the last clamp to the last. Between
/// two breakpoints the segment's [`Interpolation`] shapes the blend. Breakpoints
/// are kept sorted by `x` on insertion; coincident `x` values are allowed and
/// create a step (the later point wins at exactly that `x`).
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ParameterCurve {
    points: Vec<Breakpoint>,
}

impl ParameterCurve {
    /// Creates an empty curve. Sampling an empty curve yields `0.0`.
    #[must_use]
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Builds a curve from an iterator of breakpoints, sorting by `x`.
    #[must_use]
    pub fn from_points<I: IntoIterator<Item = Breakpoint>>(points: I) -> Self {
        let mut curve = Self { points: points.into_iter().collect() };
        curve.sort();
        curve
    }

    /// Convenience constructor for a constant curve returning `value` always.
    #[must_use]
    pub fn constant(value: Sample) -> Self {
        Self::from_points([Breakpoint::linear(0.0, value)])
    }

    /// Convenience constructor for a straight line from `(x0, y0)` to `(x1, y1)`.
    #[must_use]
    pub fn line(x0: Sample, y0: Sample, x1: Sample, y1: Sample) -> Self {
        Self::from_points([Breakpoint::linear(x0, y0), Breakpoint::linear(x1, y1)])
    }

    /// Inserts a breakpoint, keeping the point list sorted by `x`.
    pub fn push(&mut self, point: Breakpoint) {
        self.points.push(point);
        self.sort();
    }

    /// Returns the authored breakpoints in ascending `x` order.
    #[must_use]
    pub fn points(&self) -> &[Breakpoint] {
        &self.points
    }

    /// Returns the number of breakpoints.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns `true` when the curve has no breakpoints.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    fn sort(&mut self) {
        // Stable sort keeps authoring order for coincident x (later wins at x).
        self.points
            .sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(core::cmp::Ordering::Equal));
    }

    /// Samples the curve at input `x`, returning the mapped output value.
    ///
    /// Non-finite inputs are treated as the lowest coordinate (clamp to the
    /// first y-value) so a corrupt game value never produces a non-finite
    /// engine parameter.
    #[must_use]
    pub fn sample(&self, x: Sample) -> Sample {
        if self.points.is_empty() {
            return 0.0;
        }
        if !x.is_finite() {
            return self.points[0].y;
        }
        // Clamp to the authored domain.
        if x <= self.points[0].x {
            return self.points[0].y;
        }
        let last = self.points.len() - 1;
        if x >= self.points[last].x {
            return self.points[last].y;
        }
        // Find the segment [i, i+1] containing x via binary search on x.
        let mut lo = 0usize;
        let mut hi = last;
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if self.points[mid].x <= x {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let a = self.points[lo];
        let b = self.points[hi];
        let span = b.x - a.x;
        if span <= 0.0 {
            // Coincident breakpoints: step to the later value.
            return b.y;
        }
        let t = (x - a.x) / span;
        let shaped = a.interpolation.shape(t);
        lerp(a.y, b.y, shaped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1e-5;

    fn approx(a: Sample, b: Sample) {
        assert!((a - b).abs() <= EPS, "expected {b}, got {a}");
    }

    #[test]
    fn empty_curve_samples_zero() {
        let c = ParameterCurve::new();
        approx(c.sample(0.0), 0.0);
        approx(c.sample(100.0), 0.0);
    }

    #[test]
    fn constant_curve_is_flat() {
        let c = ParameterCurve::constant(-6.0);
        approx(c.sample(-1000.0), -6.0);
        approx(c.sample(1000.0), -6.0);
    }

    #[test]
    fn line_interpolates_linearly() {
        let c = ParameterCurve::line(0.0, 0.0, 1.0, 10.0);
        approx(c.sample(0.0), 0.0);
        approx(c.sample(0.5), 5.0);
        approx(c.sample(1.0), 10.0);
    }

    #[test]
    fn clamps_outside_domain_to_endpoints() {
        let c = ParameterCurve::line(0.0, 2.0, 1.0, 8.0);
        approx(c.sample(-5.0), 2.0);
        approx(c.sample(5.0), 8.0);
    }

    #[test]
    fn constant_segment_holds_start_value() {
        let c = ParameterCurve::from_points([
            Breakpoint::new(0.0, 0.0, Interpolation::Constant),
            Breakpoint::new(1.0, 10.0, Interpolation::Constant),
        ]);
        approx(c.sample(0.0), 0.0);
        approx(c.sample(0.999), 0.0);
        approx(c.sample(1.0), 10.0);
    }

    #[test]
    fn scurve_is_monotone_and_centered() {
        let c = ParameterCurve::from_points([
            Breakpoint::new(0.0, 0.0, Interpolation::SCurve),
            Breakpoint::new(1.0, 1.0, Interpolation::SCurve),
        ]);
        approx(c.sample(0.5), 0.5);
        let a = c.sample(0.25);
        let b = c.sample(0.75);
        assert!(a < 0.5 && b > 0.5 && a < b);
    }

    #[test]
    fn non_finite_input_is_handled() {
        let c = ParameterCurve::line(0.0, 0.0, 1.0, 1.0);
        let v = c.sample(Sample::NAN);
        assert!(v.is_finite());
    }
}
