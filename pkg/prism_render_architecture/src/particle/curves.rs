//! Authored curve and colour-gradient *evaluation and baking* layer (design
//! §5, §8, §30).
//!
//! Production `VFX` stacks drive most per-particle parameters from *over-life*
//! curves and *colour ramps*: Unreal `Niagara`'s curve/colour modules, Unity
//! `VFX Graph`'s "over lifetime" blocks, and Unreal's `Distribution`/`Curve`
//! assets all express "as the particle ages from birth to death, sweep this
//! scalar or colour along an authored shape". This module owns the
//! `CPU`-verifiable *maths* of that contract: it turns authored control points
//! into sampled values and bakes them into the flat look-up tables (`LUT`s) a
//! `GPU` kernel reads as a 1D texture.
//!
//! It is deliberately orthogonal to its two neighbours:
//!
//! * [`super::modules`] owns the *binding declaration* side — a
//!   [`super::modules::ResourceKind::Curve`] slot declared through
//!   [`super::modules::ResourceDecl::curve`] tells the codegen which `GPU`
//!   resource a module reads. It says *that* a curve is bound, not *what shape*
//!   it has.
//! * [`super::authoring`] owns the *authoring reference* side — a
//!   [`super::authoring::BindingSource::Curve`] records that a parameter is
//!   driven by curve number `n`. It says *which* curve, not *how to evaluate*
//!   it.
//!
//! This module is the missing third piece: given the authored control points,
//! it evaluates the curve on the `CPU` and bakes the `LUT` those two layers
//! only ever reference by handle. The three are composed by a higher layer; none
//! of them depends on another's internals.
//!
//! # Determinism
//!
//! Every routine here uses only ordinary `f32` arithmetic plus `floor` (to
//! locate a `LUT` cell). No transcendental function (`sin`/`cos`/`exp`/`ln`/
//! `pow`) is called, so the `CPU` reference stays bit-reproducible against a
//! future `GPU` sampler, matching the determinism contract of the sibling
//! [`super::simulation`] module.

use alloc::vec::Vec;
use core::cmp::Ordering;

/// Absolute tolerance for `f32` equality decisions (segment-width and
/// gradient-width guards). Denominators smaller than this are treated as zero so
/// evaluation never divides by (near) zero and never propagates `NaN`.
pub const EPS: f32 = 1e-9;

/// Clamps `x` into the inclusive range `[lo, hi]`.
///
/// A local helper rather than `f32::clamp` so the intent (and the `lo <= hi`
/// precondition handling) is explicit at every call site; `f32::clamp` would
/// panic if `lo > hi`, which this layer must never do on authored data.
#[must_use]
fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// Linear interpolation between `a` and `b` by `t` (unclamped).
#[must_use]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// How the values *between* two [`Keyframe`]s are reconstructed.
///
/// The four families mirror what authored `VFX` curves expose: a piecewise
/// cubic `Hermite` (the workhorse, tangent-driven), an automatic `Catmull-Rom`
/// spline (tangents inferred from neighbours), a cubic `Bezier` (explicit
/// intermediate control values), and the degenerate step/linear reconstructions.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum InterpolationMode {
    /// Hold the left key's value across the whole segment (a stair-step curve).
    Step,
    /// Straight-line blend between the two bounding keys.
    #[default]
    Linear,
    /// Piecewise cubic `Hermite`: the left key's [`Keyframe::out_tangent`] and
    /// the right key's [`Keyframe::in_tangent`] are the segment-endpoint
    /// *slopes* (value units per time unit).
    Hermite,
    /// Cubic `Catmull-Rom`: endpoint slopes are inferred from the neighbouring
    /// keys, so the stored tangents are ignored and the spline passes smoothly
    /// through every control point.
    CatmullRom,
    /// Cubic `Bezier`: the left key's [`Keyframe::out_tangent`] and the right
    /// key's [`Keyframe::in_tangent`] are the *values* of the two intermediate
    /// control points (`Bezier` control-point form, not slope form).
    Bezier,
}

/// A single authored control point on a scalar [`Curve`].
///
/// `in_tangent`/`out_tangent` are interpreted differently per
/// [`InterpolationMode`] (slopes for [`InterpolationMode::Hermite`], control
/// values for [`InterpolationMode::Bezier`], ignored for the others), which is
/// exactly how authoring tools overload a single "tangent" field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Keyframe {
    /// The key's position along the curve's domain (for an over-life curve this
    /// is the normalized age in `0..=1`).
    pub time: f32,
    /// The value the curve takes exactly at [`Keyframe::time`].
    pub value: f32,
    /// Incoming tangent, used by the segment ending at this key.
    pub in_tangent: f32,
    /// Outgoing tangent, used by the segment starting at this key.
    pub out_tangent: f32,
}

impl Keyframe {
    /// A key with zero tangents (suitable for step/linear/`Catmull-Rom`).
    #[must_use]
    pub const fn new(time: f32, value: f32) -> Self {
        Self {
            time,
            value,
            in_tangent: 0.0,
            out_tangent: 0.0,
        }
    }

    /// A key with explicit incoming and outgoing tangents.
    #[must_use]
    pub const fn with_tangents(time: f32, value: f32, in_tangent: f32, out_tangent: f32) -> Self {
        Self {
            time,
            value,
            in_tangent,
            out_tangent,
        }
    }
}

/// An authored scalar curve: an ordered sequence of [`Keyframe`]s plus the
/// [`InterpolationMode`] used to reconstruct values between them.
///
/// Sampling outside the key domain *holds* the nearest endpoint value (a clamp),
/// matching the "clamp"/"hold" extrapolation of authored `VFX` curves rather
/// than looping or mirroring.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Curve {
    keys: Vec<Keyframe>,
    mode: InterpolationMode,
}

impl Curve {
    /// An empty curve with the given interpolation mode. An empty curve samples
    /// to `0.0` everywhere.
    #[must_use]
    pub const fn new(mode: InterpolationMode) -> Self {
        Self {
            keys: Vec::new(),
            mode,
        }
    }

    /// Builds a curve from a set of keys, sorting them by time ascending.
    ///
    /// A stable sort by time means authored keys may be supplied in any order;
    /// ties keep their relative input order so a deliberately duplicated time
    /// (a hard step) behaves predictably.
    #[must_use]
    pub fn from_keys(mode: InterpolationMode, mut keys: Vec<Keyframe>) -> Self {
        keys.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(Ordering::Equal));
        Self { keys, mode }
    }

    /// The interpolation mode this curve reconstructs with.
    #[must_use]
    pub const fn mode(&self) -> InterpolationMode {
        self.mode
    }

    /// The ordered keys backing this curve.
    #[must_use]
    pub fn keys(&self) -> &[Keyframe] {
        &self.keys
    }

    /// Whether the curve has no keys.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// The number of keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// The `(min_time, max_time)` domain spanned by the keys, or `None` when the
    /// curve is empty.
    #[must_use]
    pub fn domain(&self) -> Option<(f32, f32)> {
        match (self.keys.first(), self.keys.last()) {
            (Some(first), Some(last)) => Some((first.time, last.time)),
            _ => None,
        }
    }

    /// Locates the segment containing `t` and the local parameter `u` in
    /// `0..=1` across that segment. Precondition: at least two keys and
    /// `first.time < t < last.time`.
    fn locate(&self, t: f32) -> (usize, f32) {
        let n = self.keys.len();
        // Count of keys whose time is <= t; since keys are sorted these sit at
        // the front, so the left endpoint of the containing segment is one back.
        let upper = self.keys.partition_point(|k| k.time <= t);
        let i = upper.saturating_sub(1).min(n - 2);
        let t0 = self.keys[i].time;
        let t1 = self.keys[i + 1].time;
        let dt = t1 - t0;
        let u = if dt.abs() > EPS { (t - t0) / dt } else { 0.0 };
        (i, u)
    }

    /// Evaluates a cubic `Hermite` segment `[i, i+1]` at local parameter `u`
    /// using explicit endpoint slopes `m0` (left) and `m1` (right).
    ///
    /// Uses the standard `Hermite` basis `h00 = 2u^3 - 3u^2 + 1`,
    /// `h10 = u^3 - 2u^2 + u`, `h01 = -2u^3 + 3u^2`, `h11 = u^3 - u^2`, with the
    /// tangents scaled by the segment width so the slope is expressed in value
    /// units per time unit.
    fn eval_hermite(&self, i: usize, u: f32, m0: f32, m1: f32) -> f32 {
        let k0 = self.keys[i];
        let k1 = self.keys[i + 1];
        let dt = k1.time - k0.time;
        let u2 = u * u;
        let u3 = u2 * u;
        let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
        let h10 = u3 - 2.0 * u2 + u;
        let h01 = -2.0 * u3 + 3.0 * u2;
        let h11 = u3 - u2;
        h00 * k0.value + h10 * dt * m0 + h01 * k1.value + h11 * dt * m1
    }

    /// Infers the `Catmull-Rom` endpoint slopes for segment `[i, i+1]` from the
    /// neighbouring keys, clamping at the curve ends (the endpoint uses the
    /// bounded one-sided secant).
    fn catmull_tangents(&self, i: usize) -> (f32, f32) {
        let n = self.keys.len();
        let k_prev = self.keys[i.saturating_sub(1)];
        let k0 = self.keys[i];
        let k1 = self.keys[i + 1];
        let k_next = self.keys[(i + 2).min(n - 1)];

        let m0 = {
            let span = k1.time - k_prev.time;
            if span.abs() > EPS {
                (k1.value - k_prev.value) / span
            } else {
                0.0
            }
        };
        let m1 = {
            let span = k_next.time - k0.time;
            if span.abs() > EPS {
                (k_next.value - k0.value) / span
            } else {
                0.0
            }
        };
        (m0, m1)
    }

    /// Evaluates a cubic `Bezier` segment `[i, i+1]` at local parameter `u`,
    /// with the two intermediate control values taken from the bounding keys'
    /// tangent fields (`Bezier` control-point form).
    fn eval_bezier(&self, i: usize, u: f32) -> f32 {
        let k0 = self.keys[i];
        let k1 = self.keys[i + 1];
        let p0 = k0.value;
        let p1 = k0.out_tangent;
        let p2 = k1.in_tangent;
        let p3 = k1.value;
        let inv = 1.0 - u;
        let inv2 = inv * inv;
        let inv3 = inv2 * inv;
        let u2 = u * u;
        let u3 = u2 * u;
        inv3 * p0 + 3.0 * inv2 * u * p1 + 3.0 * inv * u2 * p2 + u3 * p3
    }

    /// Samples the curve at `t`, holding the nearest endpoint value outside the
    /// key domain. An empty curve returns `0.0`; a single-key curve returns that
    /// key's value.
    #[must_use]
    pub fn sample(&self, t: f32) -> f32 {
        let n = self.keys.len();
        if n == 0 {
            return 0.0;
        }
        if n == 1 {
            return self.keys[0].value;
        }
        let first = self.keys[0];
        let last = self.keys[n - 1];
        if t <= first.time {
            return first.value;
        }
        if t >= last.time {
            return last.value;
        }

        let (i, u) = self.locate(t);
        let k0 = self.keys[i];
        let k1 = self.keys[i + 1];
        match self.mode {
            InterpolationMode::Step => k0.value,
            InterpolationMode::Linear => lerp(k0.value, k1.value, u),
            InterpolationMode::Hermite => self.eval_hermite(i, u, k0.out_tangent, k1.in_tangent),
            InterpolationMode::CatmullRom => {
                let (m0, m1) = self.catmull_tangents(i);
                self.eval_hermite(i, u, m0, m1)
            }
            InterpolationMode::Bezier => self.eval_bezier(i, u),
        }
    }

    /// Samples the curve treating `age` as a normalized particle age: `age` is
    /// clamped to `0..=1` before sampling, the canonical over-life convention
    /// (design §8).
    #[must_use]
    pub fn evaluate_over_life(&self, age: f32) -> f32 {
        self.sample(clamp(age, 0.0, 1.0))
    }

    /// Bakes the curve into an equidistantly sampled [`CurveLut`] with
    /// `resolution` texels (clamped to a minimum of two), matching the 1D
    /// texture `LUT` a `GPU` kernel reads.
    #[must_use]
    pub fn bake(&self, resolution: usize) -> CurveLut {
        let count = resolution.max(2);
        let (dmin, dmax) = self.domain().unwrap_or((0.0, 1.0));
        let mut values = Vec::with_capacity(count);
        let span = dmax - dmin;
        for i in 0..count {
            let frac = i as f32 / (count - 1) as f32;
            values.push(self.sample(dmin + span * frac));
        }
        CurveLut {
            values,
            domain_min: dmin,
            domain_max: dmax,
        }
    }
}

/// A baked scalar look-up table: equidistant samples of a [`Curve`] over its
/// domain, sampled at runtime with linear interpolation.
///
/// This is the `CPU` mirror of the 1D texture `LUT` a `GPU` sampler would read:
/// the runtime cost is a floor, one subtract and one blend, independent of the
/// authored curve's complexity.
#[derive(Clone, Debug, PartialEq)]
pub struct CurveLut {
    values: Vec<f32>,
    domain_min: f32,
    domain_max: f32,
}

impl CurveLut {
    /// The baked sample values, one per texel.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// The number of texels.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the table has no texels. A baked `LUT` always has at least two,
    /// so this is provided for completeness alongside [`CurveLut::len`].
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The `(min, max)` domain the table was baked over.
    #[must_use]
    pub fn domain(&self) -> (f32, f32) {
        (self.domain_min, self.domain_max)
    }

    /// Samples the table at `t`, mapping `t` through the baked domain to a texel
    /// position and linearly interpolating between the two straddling texels.
    /// Positions outside the domain hold the nearest texel.
    #[must_use]
    pub fn sample(&self, t: f32) -> f32 {
        let count = self.values.len();
        if count == 0 {
            return 0.0;
        }
        if count == 1 {
            return self.values[0];
        }
        let span = self.domain_max - self.domain_min;
        let p = if span.abs() > EPS {
            clamp((t - self.domain_min) / span, 0.0, 1.0)
        } else {
            0.0
        };
        let scaled = p * (count - 1) as f32;
        let base = scaled.floor();
        let idx = base as usize;
        if idx >= count - 1 {
            return self.values[count - 1];
        }
        let frac = scaled - base;
        lerp(self.values[idx], self.values[idx + 1], frac)
    }

    /// Samples the table treating `age` as a normalized particle age clamped to
    /// `0..=1` (over-life convention).
    #[must_use]
    pub fn evaluate_over_life(&self, age: f32) -> f32 {
        self.sample(clamp(age, 0.0, 1.0))
    }
}

/// A single authored stop on a [`ColorRamp`]: a time and an `RGBA` colour.
///
/// The `rgba` channels are stored in whatever space the author works in; this
/// layer blends them componentwise. See [`ColorRamp::sample`] for the gamma
/// note.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorStop {
    /// The stop's position along the ramp's domain (normalized age for an
    /// over-life ramp).
    pub time: f32,
    /// The `RGBA` colour at this stop, one `f32` per channel.
    pub rgba: [f32; 4],
}

impl ColorStop {
    /// A stop at `time` with the given `RGBA` colour.
    #[must_use]
    pub const fn new(time: f32, rgba: [f32; 4]) -> Self {
        Self { time, rgba }
    }
}

/// An authored colour gradient: an ordered set of [`ColorStop`]s blended
/// linearly per channel.
///
/// This is the direct analogue of a `Niagara`/`VFX Graph` colour ramp. Sampling
/// outside the stop domain holds the nearest endpoint colour.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ColorRamp {
    stops: Vec<ColorStop>,
}

impl ColorRamp {
    /// An empty ramp; it samples to transparent black everywhere.
    #[must_use]
    pub const fn new() -> Self {
        Self { stops: Vec::new() }
    }

    /// Builds a ramp from a set of stops, sorting them by time ascending.
    #[must_use]
    pub fn from_stops(mut stops: Vec<ColorStop>) -> Self {
        stops.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(Ordering::Equal));
        Self { stops }
    }

    /// The ordered stops backing this ramp.
    #[must_use]
    pub fn stops(&self) -> &[ColorStop] {
        &self.stops
    }

    /// Whether the ramp has no stops.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stops.is_empty()
    }

    /// The number of stops.
    #[must_use]
    pub fn len(&self) -> usize {
        self.stops.len()
    }

    /// The `(min_time, max_time)` domain spanned by the stops, or `None` when
    /// the ramp is empty.
    #[must_use]
    pub fn domain(&self) -> Option<(f32, f32)> {
        match (self.stops.first(), self.stops.last()) {
            (Some(first), Some(last)) => Some((first.time, last.time)),
            _ => None,
        }
    }

    /// Samples the ramp at `t`, holding the nearest endpoint colour outside the
    /// stop domain. An empty ramp returns transparent black.
    ///
    /// Interpolation is a straight per-channel blend in the stops' stored space.
    /// If the stops hold gamma-encoded `sRGB` values, this is a direct `RGB`
    /// interpolation (perceptually even but not radiometrically linear); callers
    /// wanting radiometrically linear blends should author linear-space stops.
    #[must_use]
    pub fn sample(&self, t: f32) -> [f32; 4] {
        let n = self.stops.len();
        if n == 0 {
            return [0.0, 0.0, 0.0, 0.0];
        }
        if n == 1 {
            return self.stops[0].rgba;
        }
        let first = self.stops[0];
        let last = self.stops[n - 1];
        if t <= first.time {
            return first.rgba;
        }
        if t >= last.time {
            return last.rgba;
        }

        let upper = self.stops.partition_point(|s| s.time <= t);
        let i = upper.saturating_sub(1).min(n - 2);
        let s0 = self.stops[i];
        let s1 = self.stops[i + 1];
        let span = s1.time - s0.time;
        let u = if span.abs() > EPS {
            (t - s0.time) / span
        } else {
            0.0
        };
        [
            lerp(s0.rgba[0], s1.rgba[0], u),
            lerp(s0.rgba[1], s1.rgba[1], u),
            lerp(s0.rgba[2], s1.rgba[2], u),
            lerp(s0.rgba[3], s1.rgba[3], u),
        ]
    }

    /// Samples the ramp treating `age` as a normalized particle age clamped to
    /// `0..=1` (over-life convention).
    #[must_use]
    pub fn evaluate_over_life(&self, age: f32) -> [f32; 4] {
        self.sample(clamp(age, 0.0, 1.0))
    }

    /// Bakes the ramp into an equidistantly sampled [`ColorLut`] with
    /// `resolution` texels (clamped to a minimum of two), matching the `RGBA`
    /// 1D texture a `GPU` kernel reads.
    #[must_use]
    pub fn bake(&self, resolution: usize) -> ColorLut {
        let count = resolution.max(2);
        let (dmin, dmax) = self.domain().unwrap_or((0.0, 1.0));
        let mut texels = Vec::with_capacity(count);
        let span = dmax - dmin;
        for i in 0..count {
            let frac = i as f32 / (count - 1) as f32;
            texels.push(self.sample(dmin + span * frac));
        }
        ColorLut {
            texels,
            domain_min: dmin,
            domain_max: dmax,
        }
    }
}

/// A baked `RGBA` look-up table: equidistant samples of a [`ColorRamp`] over its
/// domain, sampled at runtime with per-channel linear interpolation.
#[derive(Clone, Debug, PartialEq)]
pub struct ColorLut {
    texels: Vec<[f32; 4]>,
    domain_min: f32,
    domain_max: f32,
}

impl ColorLut {
    /// The baked `RGBA` texels.
    #[must_use]
    pub fn texels(&self) -> &[[f32; 4]] {
        &self.texels
    }

    /// The number of texels.
    #[must_use]
    pub fn len(&self) -> usize {
        self.texels.len()
    }

    /// Whether the table has no texels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.texels.is_empty()
    }

    /// The `(min, max)` domain the table was baked over.
    #[must_use]
    pub fn domain(&self) -> (f32, f32) {
        (self.domain_min, self.domain_max)
    }

    /// Samples the table at `t`, mapping `t` through the baked domain to a texel
    /// position and blending the two straddling texels per channel. Positions
    /// outside the domain hold the nearest texel.
    #[must_use]
    pub fn sample(&self, t: f32) -> [f32; 4] {
        let count = self.texels.len();
        if count == 0 {
            return [0.0, 0.0, 0.0, 0.0];
        }
        if count == 1 {
            return self.texels[0];
        }
        let span = self.domain_max - self.domain_min;
        let p = if span.abs() > EPS {
            clamp((t - self.domain_min) / span, 0.0, 1.0)
        } else {
            0.0
        };
        let scaled = p * (count - 1) as f32;
        let base = scaled.floor();
        let idx = base as usize;
        if idx >= count - 1 {
            return self.texels[count - 1];
        }
        let frac = scaled - base;
        let a = self.texels[idx];
        let b = self.texels[idx + 1];
        [
            lerp(a[0], b[0], frac),
            lerp(a[1], b[1], frac),
            lerp(a[2], b[2], frac),
            lerp(a[3], b[3], frac),
        ]
    }

    /// Samples the table treating `age` as a normalized particle age clamped to
    /// `0..=1` (over-life convention).
    #[must_use]
    pub fn evaluate_over_life(&self, age: f32) -> [f32; 4] {
        self.sample(clamp(age, 0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < TOL
    }

    fn close_rgba(a: [f32; 4], b: [f32; 4]) -> bool {
        close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2]) && close(a[3], b[3])
    }

    fn sample_curve() -> Curve {
        Curve::from_keys(
            InterpolationMode::Linear,
            Vec::from([
                Keyframe::new(0.0, 0.0),
                Keyframe::new(1.0, 2.0),
                Keyframe::new(2.0, 1.0),
            ]),
        )
    }

    #[test]
    fn empty_curve_samples_zero() {
        let c = Curve::new(InterpolationMode::Linear);
        assert!(c.is_empty());
        assert!(close(c.sample(0.0), 0.0));
        assert!(close(c.sample(5.0), 0.0));
        assert_eq!(c.domain(), None);
    }

    #[test]
    fn single_key_curve_holds_value() {
        let c = Curve::from_keys(
            InterpolationMode::Hermite,
            Vec::from([Keyframe::new(3.0, 7.0)]),
        );
        assert!(close(c.sample(-1.0), 7.0));
        assert!(close(c.sample(3.0), 7.0));
        assert!(close(c.sample(100.0), 7.0));
    }

    #[test]
    fn from_keys_sorts_by_time() {
        let c = Curve::from_keys(
            InterpolationMode::Linear,
            Vec::from([
                Keyframe::new(2.0, 1.0),
                Keyframe::new(0.0, 0.0),
                Keyframe::new(1.0, 2.0),
            ]),
        );
        let times: Vec<f32> = c.keys().iter().map(|k| k.time).collect();
        assert!(close(times[0], 0.0) && close(times[1], 1.0) && close(times[2], 2.0));
    }

    #[test]
    fn boundary_clamp_holds_endpoints() {
        let c = sample_curve();
        assert!(close(c.sample(-10.0), 0.0));
        assert!(close(c.sample(10.0), 1.0));
    }

    #[test]
    fn linear_passes_through_control_points_and_midpoints() {
        let c = sample_curve();
        // Exactly on the keys.
        assert!(close(c.sample(0.0), 0.0));
        assert!(close(c.sample(1.0), 2.0));
        assert!(close(c.sample(2.0), 1.0));
        // Midpoints of each linear segment are the averages.
        assert!(close(c.sample(0.5), 1.0));
        assert!(close(c.sample(1.5), 1.5));
    }

    #[test]
    fn linear_segment_is_monotonic() {
        let c = Curve::from_keys(
            InterpolationMode::Linear,
            Vec::from([Keyframe::new(0.0, 0.0), Keyframe::new(1.0, 10.0)]),
        );
        let mut prev = f32::NEG_INFINITY;
        for step in 0..=20 {
            let t = step as f32 / 20.0;
            let v = c.sample(t);
            assert!(v >= prev - TOL, "linear curve must be non-decreasing");
            prev = v;
        }
    }

    #[test]
    fn step_mode_holds_left_value() {
        let c = Curve::from_keys(
            InterpolationMode::Step,
            Vec::from([Keyframe::new(0.0, 5.0), Keyframe::new(1.0, 9.0)]),
        );
        assert!(close(c.sample(0.0), 5.0));
        assert!(close(c.sample(0.999), 5.0));
        assert!(close(c.sample(1.0), 9.0));
    }

    #[test]
    fn hermite_passes_through_control_points() {
        let c = Curve::from_keys(
            InterpolationMode::Hermite,
            Vec::from([
                Keyframe::with_tangents(0.0, 0.0, 0.0, 1.0),
                Keyframe::with_tangents(1.0, 1.0, 1.0, 0.0),
            ]),
        );
        assert!(close(c.sample(0.0), 0.0));
        assert!(close(c.sample(1.0), 1.0));
    }

    #[test]
    fn hermite_with_zero_tangents_matches_smoothstep() {
        // Zero endpoint slopes reduce the Hermite basis to the classic
        // smoothstep 3u^2 - 2u^3, so the midpoint is exactly 0.5.
        let c = Curve::from_keys(
            InterpolationMode::Hermite,
            Vec::from([Keyframe::new(0.0, 0.0), Keyframe::new(1.0, 1.0)]),
        );
        assert!(close(c.sample(0.5), 0.5));
        assert!(close(c.sample(0.25), 3.0 * 0.0625 - 2.0 * 0.015625));
    }

    #[test]
    fn hermite_monotone_tangents_stay_monotonic() {
        // Endpoint slopes equal to the secant slope keep the segment monotone.
        let c = Curve::from_keys(
            InterpolationMode::Hermite,
            Vec::from([
                Keyframe::with_tangents(0.0, 0.0, 1.0, 1.0),
                Keyframe::with_tangents(1.0, 1.0, 1.0, 1.0),
            ]),
        );
        let mut prev = f32::NEG_INFINITY;
        for step in 0..=20 {
            let t = step as f32 / 20.0;
            let v = c.sample(t);
            assert!(v >= prev - TOL, "monotone Hermite must be non-decreasing");
            prev = v;
        }
    }

    #[test]
    fn catmull_rom_passes_through_control_points() {
        let c = Curve::from_keys(
            InterpolationMode::CatmullRom,
            Vec::from([
                Keyframe::new(0.0, 0.0),
                Keyframe::new(1.0, 2.0),
                Keyframe::new(2.0, 0.0),
                Keyframe::new(3.0, 3.0),
            ]),
        );
        assert!(close(c.sample(0.0), 0.0));
        assert!(close(c.sample(1.0), 2.0));
        assert!(close(c.sample(2.0), 0.0));
        assert!(close(c.sample(3.0), 3.0));
    }

    #[test]
    fn catmull_rom_interior_uses_neighbor_slope() {
        // On a straight line the inferred tangents reproduce the line exactly.
        let c = Curve::from_keys(
            InterpolationMode::CatmullRom,
            Vec::from([
                Keyframe::new(0.0, 0.0),
                Keyframe::new(1.0, 1.0),
                Keyframe::new(2.0, 2.0),
                Keyframe::new(3.0, 3.0),
            ]),
        );
        assert!(close(c.sample(1.5), 1.5));
        assert!(close(c.sample(2.5), 2.5));
    }

    #[test]
    fn bezier_passes_through_endpoints() {
        let c = Curve::from_keys(
            InterpolationMode::Bezier,
            Vec::from([
                // out_tangent / in_tangent are the two intermediate control values.
                Keyframe::with_tangents(0.0, 0.0, 0.0, 0.5),
                Keyframe::with_tangents(1.0, 1.0, 0.5, 0.0),
            ]),
        );
        assert!(close(c.sample(0.0), 0.0));
        assert!(close(c.sample(1.0), 1.0));
    }

    #[test]
    fn bezier_flat_control_values_are_constant() {
        // All four control values equal 4 -> constant curve at 4.
        let c = Curve::from_keys(
            InterpolationMode::Bezier,
            Vec::from([
                Keyframe::with_tangents(0.0, 4.0, 0.0, 4.0),
                Keyframe::with_tangents(1.0, 4.0, 4.0, 0.0),
            ]),
        );
        assert!(close(c.sample(0.3), 4.0));
        assert!(close(c.sample(0.7), 4.0));
    }

    #[test]
    fn evaluate_over_life_clamps_age() {
        let c = Curve::from_keys(
            InterpolationMode::Linear,
            Vec::from([Keyframe::new(0.0, 0.0), Keyframe::new(1.0, 1.0)]),
        );
        assert!(close(c.evaluate_over_life(-0.5), 0.0));
        assert!(close(c.evaluate_over_life(0.5), 0.5));
        assert!(close(c.evaluate_over_life(1.5), 1.0));
    }

    #[test]
    fn bake_matches_direct_sampling_within_tolerance() {
        let c = Curve::from_keys(
            InterpolationMode::Hermite,
            Vec::from([
                Keyframe::with_tangents(0.0, 0.0, 0.0, 2.0),
                Keyframe::with_tangents(1.0, 1.0, 0.5, -1.0),
                Keyframe::with_tangents(2.0, 0.5, 0.0, 0.0),
            ]),
        );
        let lut = c.bake(512);
        assert_eq!(lut.len(), 512);
        assert!(!lut.is_empty());
        // Compare the LUT against the analytic curve at arbitrary positions;
        // 512 texels resolve a smooth Hermite well within a loose tolerance.
        for step in 0..=40 {
            let t = 2.0 * (step as f32 / 40.0);
            assert!(
                (lut.sample(t) - c.sample(t)).abs() < 5e-3,
                "baked LUT diverged from the curve at t = {t}"
            );
        }
    }

    #[test]
    fn bake_reproduces_key_values_exactly() {
        let c = sample_curve();
        // Domain is [0, 2]; a 3-texel LUT lands exactly on the three keys.
        let lut = c.bake(3);
        assert!(close(lut.sample(0.0), 0.0));
        assert!(close(lut.sample(1.0), 2.0));
        assert!(close(lut.sample(2.0), 1.0));
        let (dmin, dmax) = lut.domain();
        assert!(close(dmin, 0.0) && close(dmax, 2.0));
    }

    #[test]
    fn curve_lut_clamps_outside_domain() {
        let lut = sample_curve().bake(64);
        assert!(close(lut.sample(-5.0), 0.0));
        assert!(close(lut.sample(50.0), 1.0));
        assert!(close(lut.evaluate_over_life(-1.0), lut.sample(0.0)));
    }

    #[test]
    fn empty_color_ramp_is_transparent_black() {
        let r = ColorRamp::new();
        assert!(r.is_empty());
        assert!(close_rgba(r.sample(0.5), [0.0, 0.0, 0.0, 0.0]));
        assert_eq!(r.domain(), None);
    }

    #[test]
    fn color_ramp_midpoint_is_channel_average() {
        let r = ColorRamp::from_stops(Vec::from([
            ColorStop::new(0.0, [0.0, 0.2, 1.0, 1.0]),
            ColorStop::new(1.0, [1.0, 0.6, 0.0, 0.0]),
        ]));
        let mid = r.sample(0.5);
        assert!(close_rgba(mid, [0.5, 0.4, 0.5, 0.5]));
    }

    #[test]
    fn color_ramp_boundary_clamp_holds_endpoints() {
        let r = ColorRamp::from_stops(Vec::from([
            ColorStop::new(0.2, [1.0, 0.0, 0.0, 1.0]),
            ColorStop::new(0.8, [0.0, 0.0, 1.0, 1.0]),
        ]));
        assert!(close_rgba(r.sample(0.0), [1.0, 0.0, 0.0, 1.0]));
        assert!(close_rgba(r.sample(1.0), [0.0, 0.0, 1.0, 1.0]));
    }

    #[test]
    fn color_ramp_sorts_and_passes_through_stops() {
        let r = ColorRamp::from_stops(Vec::from([
            ColorStop::new(1.0, [0.0, 0.0, 0.0, 1.0]),
            ColorStop::new(0.0, [1.0, 1.0, 1.0, 1.0]),
            ColorStop::new(0.5, [0.5, 0.5, 0.5, 1.0]),
        ]));
        assert!(close(r.stops()[0].time, 0.0));
        assert!(close_rgba(r.sample(0.0), [1.0, 1.0, 1.0, 1.0]));
        assert!(close_rgba(r.sample(0.5), [0.5, 0.5, 0.5, 1.0]));
        assert!(close_rgba(r.sample(1.0), [0.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn color_ramp_single_stop_holds_value() {
        let r = ColorRamp::from_stops(Vec::from([ColorStop::new(0.5, [0.1, 0.2, 0.3, 0.4])]));
        assert!(close_rgba(r.sample(0.0), [0.1, 0.2, 0.3, 0.4]));
        assert!(close_rgba(r.sample(1.0), [0.1, 0.2, 0.3, 0.4]));
    }

    #[test]
    fn color_ramp_evaluate_over_life_clamps_age() {
        let r = ColorRamp::from_stops(Vec::from([
            ColorStop::new(0.0, [0.0, 0.0, 0.0, 1.0]),
            ColorStop::new(1.0, [1.0, 1.0, 1.0, 1.0]),
        ]));
        assert!(close_rgba(r.evaluate_over_life(-1.0), [0.0, 0.0, 0.0, 1.0]));
        assert!(close_rgba(r.evaluate_over_life(0.5), [0.5, 0.5, 0.5, 1.0]));
        assert!(close_rgba(r.evaluate_over_life(2.0), [1.0, 1.0, 1.0, 1.0]));
    }

    #[test]
    fn color_lut_matches_direct_sampling_within_tolerance() {
        let r = ColorRamp::from_stops(Vec::from([
            ColorStop::new(0.0, [0.0, 0.1, 0.2, 1.0]),
            ColorStop::new(0.5, [0.8, 0.4, 0.1, 0.5]),
            ColorStop::new(1.0, [0.2, 0.9, 0.7, 0.0]),
        ]));
        let lut = r.bake(256);
        assert_eq!(lut.len(), 256);
        assert!(!lut.is_empty());
        for step in 0..=40 {
            let t = step as f32 / 40.0;
            let a = lut.sample(t);
            let b = r.sample(t);
            assert!(
                close_rgba(a, b) || (a[0] - b[0]).abs() < 5e-3,
                "baked color LUT diverged at t = {t}"
            );
        }
    }

    #[test]
    fn color_lut_clamps_outside_domain() {
        let r = ColorRamp::from_stops(Vec::from([
            ColorStop::new(0.0, [1.0, 0.0, 0.0, 1.0]),
            ColorStop::new(1.0, [0.0, 1.0, 0.0, 1.0]),
        ]));
        let lut = r.bake(32);
        assert!(close_rgba(lut.sample(-3.0), [1.0, 0.0, 0.0, 1.0]));
        assert!(close_rgba(lut.sample(9.0), [0.0, 1.0, 0.0, 1.0]));
        let (dmin, dmax) = lut.domain();
        assert!(close(dmin, 0.0) && close(dmax, 1.0));
        assert!(close_rgba(
            lut.evaluate_over_life(2.0),
            [0.0, 1.0, 0.0, 1.0]
        ));
    }

    #[test]
    fn bake_clamps_resolution_to_minimum_two() {
        let lut = sample_curve().bake(0);
        assert_eq!(lut.len(), 2);
        let clut = ColorRamp::from_stops(Vec::from([
            ColorStop::new(0.0, [0.0; 4]),
            ColorStop::new(1.0, [1.0; 4]),
        ]))
        .bake(1);
        assert_eq!(clut.len(), 2);
    }

    #[test]
    fn interpolation_mode_default_is_linear() {
        assert_eq!(InterpolationMode::default(), InterpolationMode::Linear);
    }
}
