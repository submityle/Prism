//! Multi-stop `RGBA` `HDR` colour-gradient *evaluation and baking* layer
//! (design §5, §8, §30).
//!
//! Production `VFX` stacks drive a particle's tint from an *over-life* colour
//! ramp: Unreal `Niagara`'s "Colour" module, Unity `VFX Graph`'s "Colour over
//! Life" block, and `PopcornFX`'s gradient samplers all express "as the
//! particle ages from birth to death, sweep its `RGBA` along an authored set of
//! colour stops". This module owns the `CPU`-verifiable *maths* of that
//! contract: it turns authored [`ColorStop`]s into a sampled [`Rgba`] and bakes
//! them into the flat `RGBA` look-up table (`LUT`) a `GPU` kernel reads as a 1D
//! texture (one `vec4` per texel).
//!
//! # Distinction from [`super::curves`]
//!
//! [`super::curves`] evaluates and bakes *scalar* over-life curves: one `f32`
//! per texel. This module is its four-channel sibling — every stop and every
//! baked texel is a full `RGBA` `vec4`, and channel values may exceed `1.0` to
//! carry `HDR` emissive colour. The two never share storage: a curve `LUT` is a
//! scalar table, a gradient `LUT` is a `vec4` table whose `std430` stride is
//! [`super::gpu_layout::VEC4_STRIDE`] bytes.
//!
//! # Determinism
//!
//! Every routine here uses only ordinary `f32` arithmetic plus `floor` (to
//! locate a texel). No transcendental function (`sin`/`cos`/`exp`/`ln`/`pow`)
//! is called, so the `CPU` reference stays bit-reproducible against a future
//! `GPU` sampler, matching the determinism contract of the sibling
//! [`super::simulation`] module.

use crate::particle::gpu_layout::VEC4_STRIDE;
use alloc::vec::Vec;
use core::cmp::Ordering;

/// Absolute tolerance for segment-width guards. A segment whose position span
/// is narrower than this is treated as a hard step so evaluation never divides
/// by (near) zero and never propagates `NaN`.
const SEG_EPS: f32 = 1e-12;

/// Linear interpolation between `a` and `b` by `t` (unclamped).
#[must_use]
fn lerp_scalar(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// A linear `RGBA` colour with `HDR` range: channel values may exceed `1.0` to
/// carry emissive intensity, matching the linear-space colour a `GPU` shader
/// consumes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rgba {
    /// Red channel (linear, may exceed `1.0` for `HDR`).
    pub r: f32,
    /// Green channel (linear, may exceed `1.0` for `HDR`).
    pub g: f32,
    /// Blue channel (linear, may exceed `1.0` for `HDR`).
    pub b: f32,
    /// Alpha / opacity channel (linear, may exceed `1.0` for `HDR` premultiply).
    pub a: f32,
}

impl Rgba {
    /// Fully transparent black `(0, 0, 0, 0)`, used as the empty-gradient guard.
    pub const TRANSPARENT: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };

    /// Builds a colour from its four linear channels.
    #[must_use]
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Component-wise linear interpolation towards `other` by `t` (unclamped),
    /// blending every channel independently.
    #[must_use]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            r: lerp_scalar(self.r, other.r, t),
            g: lerp_scalar(self.g, other.g, t),
            b: lerp_scalar(self.b, other.b, t),
            a: lerp_scalar(self.a, other.a, t),
        }
    }

    /// Scales every channel by `factor`, keeping the colour in linear space.
    #[must_use]
    pub fn scale(self, factor: f32) -> Self {
        Self {
            r: self.r * factor,
            g: self.g * factor,
            b: self.b * factor,
            a: self.a * factor,
        }
    }
}

/// A single authored control point on a [`ColorGradient`].
///
/// `position` is the normalized age in `0..=1` at which the gradient takes
/// exactly [`ColorStop::color`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorStop {
    /// The stop's position along the gradient's normalized life (`0..=1`).
    pub position: f32,
    /// The colour the gradient takes exactly at [`ColorStop::position`].
    pub color: Rgba,
}

impl ColorStop {
    /// Builds a stop at `position` (normalized age) with `color`.
    #[must_use]
    pub const fn new(position: f32, color: Rgba) -> Self {
        Self { position, color }
    }
}

/// An authored multi-stop `RGBA` colour gradient sampled over a particle's
/// normalized life.
///
/// Sampling outside the stop domain *holds* the nearest endpoint colour (a
/// clamp), matching the "clamp"/"hold" extrapolation of authored `VFX` ramps
/// rather than looping or mirroring.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ColorGradient {
    stops: Vec<ColorStop>,
}

impl ColorGradient {
    /// An empty gradient, which samples to [`Rgba::TRANSPARENT`] everywhere.
    #[must_use]
    pub const fn new() -> Self {
        Self { stops: Vec::new() }
    }

    /// Builds a gradient from a set of stops, sorting them by position
    /// ascending.
    ///
    /// A stable sort by position means authored stops may be supplied in any
    /// order; ties keep their relative input order so a deliberately duplicated
    /// position (a hard colour step) behaves predictably.
    #[must_use]
    pub fn from_stops(mut stops: Vec<ColorStop>) -> Self {
        stops.sort_by(|a, b| {
            a.position
                .partial_cmp(&b.position)
                .unwrap_or(Ordering::Equal)
        });
        Self { stops }
    }

    /// The ordered stops backing this gradient.
    #[must_use]
    pub fn stops(&self) -> &[ColorStop] {
        &self.stops
    }

    /// The number of stops.
    #[must_use]
    pub fn stop_count(&self) -> usize {
        self.stops.len()
    }

    /// Whether the gradient has no stops.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stops.is_empty()
    }

    /// Samples the gradient at normalized age `t`.
    ///
    /// An empty gradient returns the [`Rgba::TRANSPARENT`] guard. Otherwise `t`
    /// is clamped to the first / last stop (endpoint hold) and interpolated
    /// component-wise linearly within the containing segment.
    #[must_use]
    pub fn sample(&self, t: f32) -> Rgba {
        let n = self.stops.len();
        if n == 0 {
            return Rgba::TRANSPARENT;
        }
        let first = self.stops[0];
        if t <= first.position {
            return first.color;
        }
        let last = self.stops[n - 1];
        if t >= last.position {
            return last.color;
        }
        // Count of stops whose position is <= t; since stops are sorted these
        // sit at the front, so the left endpoint of the segment is one back.
        let upper = self.stops.partition_point(|s| s.position <= t);
        let i = upper.saturating_sub(1).min(n - 2);
        let s0 = self.stops[i];
        let s1 = self.stops[i + 1];
        let dp = s1.position - s0.position;
        let u = if dp.abs() > SEG_EPS {
            (t - s0.position) / dp
        } else {
            0.0
        };
        s0.color.lerp(s1.color, u)
    }

    /// Bakes the gradient into an equidistantly sampled [`ColorLut`] with
    /// `resolution` texels (clamped to a minimum of two), matching the `RGBA`
    /// 1D texture `LUT` a `GPU` kernel reads (one `vec4` per texel).
    ///
    /// The endpoints of the baked table equal the first and last stop colours,
    /// since the sweep runs across the full normalized `0..=1` life.
    #[must_use]
    pub fn bake(&self, resolution: usize) -> ColorLut {
        let count = resolution.max(2);
        let mut texels = Vec::with_capacity(count);
        let denom = (count - 1) as f32;
        for i in 0..count {
            let frac = i as f32 / denom;
            texels.push(self.sample(frac));
        }
        ColorLut { texels }
    }
}

/// A baked `RGBA` look-up table: equidistant [`ColorGradient`] samples over the
/// normalized `0..=1` life, read at runtime with nearest-texel lookup.
///
/// This is the `CPU` mirror of the 1D `RGBA` texture a `GPU` sampler would
/// read: each texel is a full `vec4` whose `std430` stride is
/// [`VEC4_STRIDE`] bytes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ColorLut {
    texels: Vec<Rgba>,
}

impl ColorLut {
    /// Wraps a pre-sampled sequence of `RGBA` texels.
    #[must_use]
    pub fn new(texels: Vec<Rgba>) -> Self {
        Self { texels }
    }

    /// The baked `RGBA` texels, one `vec4` per texel.
    #[must_use]
    pub fn values(&self) -> &[Rgba] {
        &self.texels
    }

    /// The number of texels.
    #[must_use]
    pub fn len(&self) -> usize {
        self.texels.len()
    }

    /// Whether the table has no texels. A gradient baked through
    /// [`ColorGradient::bake`] always has at least two, so this is provided for
    /// completeness alongside [`ColorLut::len`].
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.texels.is_empty()
    }

    /// The `std430` byte footprint of the table: [`ColorLut::len`] texels each
    /// occupying [`VEC4_STRIDE`] bytes.
    #[must_use]
    pub fn lut_buffer_bytes(&self) -> usize {
        self.texels.len() * VEC4_STRIDE
    }

    /// Samples the table at normalized age `t` by nearest-texel lookup: the
    /// index is `floor(t * (len - 1) + 0.5)` clamped into `0..=len-1`.
    ///
    /// An empty table returns the [`Rgba::TRANSPARENT`] guard.
    #[must_use]
    pub fn sample_nearest(&self, t: f32) -> Rgba {
        let count = self.texels.len();
        if count == 0 {
            return Rgba::TRANSPARENT;
        }
        let last = count - 1;
        let raw = (t * last as f32 + 0.5).floor();
        let idx = if raw < 0.0 {
            0
        } else {
            (raw as usize).min(last)
        };
        self.texels[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Absolute tolerance for `f32` equality decisions in tests.
    const CMP_EPS: f32 = 1e-6;

    fn rgba_close(a: Rgba, b: Rgba) -> bool {
        (a.r - b.r).abs() < CMP_EPS
            && (a.g - b.g).abs() < CMP_EPS
            && (a.b - b.b).abs() < CMP_EPS
            && (a.a - b.a).abs() < CMP_EPS
    }

    #[test]
    fn rgba_lerp_and_scale_are_component_wise() {
        let lo = Rgba::new(0.0, 0.0, 0.0, 0.0);
        let hi = Rgba::new(2.0, 4.0, 6.0, 8.0);
        assert!(rgba_close(lo.lerp(hi, 0.5), Rgba::new(1.0, 2.0, 3.0, 4.0)));
        assert!(rgba_close(lo.lerp(hi, 0.0), lo));
        assert!(rgba_close(lo.lerp(hi, 1.0), hi));
        assert!(rgba_close(hi.scale(0.5), Rgba::new(1.0, 2.0, 3.0, 4.0)));
    }

    #[test]
    fn single_stop_gradient_is_constant() {
        let c = Rgba::new(0.25, 0.5, 0.75, 1.0);
        let g = ColorGradient::from_stops(vec![ColorStop::new(0.5, c)]);
        assert_eq!(g.stop_count(), 1);
        assert!(rgba_close(g.sample(0.0), c));
        assert!(rgba_close(g.sample(0.5), c));
        assert!(rgba_close(g.sample(1.0), c));
    }

    #[test]
    fn two_stop_gradient_interpolates_linearly() {
        let a = Rgba::new(0.0, 0.0, 0.0, 1.0);
        let b = Rgba::new(1.0, 1.0, 1.0, 1.0);
        let g = ColorGradient::from_stops(vec![ColorStop::new(0.0, a), ColorStop::new(1.0, b)]);
        assert!(rgba_close(g.sample(0.25), Rgba::new(0.25, 0.25, 0.25, 1.0)));
        assert!(rgba_close(g.sample(0.5), Rgba::new(0.5, 0.5, 0.5, 1.0)));
        assert!(rgba_close(g.sample(0.75), Rgba::new(0.75, 0.75, 0.75, 1.0)));
    }

    #[test]
    fn multi_stop_gradient_selects_correct_segment() {
        let a = Rgba::new(1.0, 0.0, 0.0, 1.0);
        let b = Rgba::new(0.0, 1.0, 0.0, 1.0);
        let c = Rgba::new(0.0, 0.0, 1.0, 1.0);
        let g = ColorGradient::from_stops(vec![
            ColorStop::new(0.0, a),
            ColorStop::new(0.5, b),
            ColorStop::new(1.0, c),
        ]);
        // Midpoint of the first segment: halfway from red to green.
        assert!(rgba_close(g.sample(0.25), Rgba::new(0.5, 0.5, 0.0, 1.0)));
        // Exactly on the middle stop.
        assert!(rgba_close(g.sample(0.5), b));
        // Midpoint of the second segment: halfway from green to blue.
        assert!(rgba_close(g.sample(0.75), Rgba::new(0.0, 0.5, 0.5, 1.0)));
    }

    #[test]
    fn sample_clamps_to_endpoints() {
        let a = Rgba::new(0.1, 0.2, 0.3, 0.4);
        let b = Rgba::new(0.9, 0.8, 0.7, 0.6);
        let g = ColorGradient::from_stops(vec![ColorStop::new(0.25, a), ColorStop::new(0.75, b)]);
        assert!(rgba_close(g.sample(-1.0), a));
        assert!(rgba_close(g.sample(0.0), a));
        assert!(rgba_close(g.sample(1.0), b));
        assert!(rgba_close(g.sample(2.0), b));
    }

    #[test]
    fn from_stops_sorts_unordered_input_stably() {
        let first = Rgba::new(1.0, 0.0, 0.0, 1.0);
        let dup_a = Rgba::new(0.0, 1.0, 0.0, 1.0);
        let dup_b = Rgba::new(0.0, 0.0, 1.0, 1.0);
        let last = Rgba::new(1.0, 1.0, 1.0, 1.0);
        // Supplied out of order, with a duplicated position (a hard step).
        let g = ColorGradient::from_stops(vec![
            ColorStop::new(1.0, last),
            ColorStop::new(0.5, dup_a),
            ColorStop::new(0.0, first),
            ColorStop::new(0.5, dup_b),
        ]);
        let stops = g.stops();
        assert!((stops[0].position - 0.0).abs() < CMP_EPS);
        assert!((stops[1].position - 0.5).abs() < CMP_EPS);
        assert!((stops[2].position - 0.5).abs() < CMP_EPS);
        assert!((stops[3].position - 1.0).abs() < CMP_EPS);
        // Stable: the first-supplied 0.5 stop (dup_a) precedes the second.
        assert!(rgba_close(stops[1].color, dup_a));
        assert!(rgba_close(stops[2].color, dup_b));
    }

    #[test]
    fn bake_endpoints_equal_first_and_last_stops() {
        let a = Rgba::new(0.2, 0.4, 0.6, 0.8);
        let b = Rgba::new(0.6, 0.4, 0.2, 1.0);
        let g = ColorGradient::from_stops(vec![ColorStop::new(0.0, a), ColorStop::new(1.0, b)]);
        let lut = g.bake(16);
        assert_eq!(lut.len(), 16);
        let texels = lut.values();
        assert!(rgba_close(texels[0], a));
        assert!(rgba_close(texels[texels.len() - 1], b));
    }

    #[test]
    fn bake_clamps_resolution_to_minimum_two() {
        let g = ColorGradient::from_stops(vec![
            ColorStop::new(0.0, Rgba::new(0.0, 0.0, 0.0, 1.0)),
            ColorStop::new(1.0, Rgba::new(1.0, 1.0, 1.0, 1.0)),
        ]);
        assert_eq!(g.bake(0).len(), 2);
        assert_eq!(g.bake(1).len(), 2);
    }

    #[test]
    fn lut_buffer_bytes_counts_vec4_stride() {
        let g = ColorGradient::from_stops(vec![
            ColorStop::new(0.0, Rgba::TRANSPARENT),
            ColorStop::new(1.0, Rgba::new(1.0, 1.0, 1.0, 1.0)),
        ]);
        let lut = g.bake(10);
        assert_eq!(lut.lut_buffer_bytes(), 10 * VEC4_STRIDE);
        assert_eq!(lut.lut_buffer_bytes(), 160);
    }

    #[test]
    fn hdr_values_above_one_are_preserved() {
        let dim = Rgba::new(0.0, 0.0, 0.0, 1.0);
        let bright = Rgba::new(8.0, 4.0, 2.0, 1.0);
        let g =
            ColorGradient::from_stops(vec![ColorStop::new(0.0, dim), ColorStop::new(1.0, bright)]);
        assert!(rgba_close(g.sample(1.0), bright));
        let mid = g.sample(0.5);
        assert!(mid.r > 1.0);
        assert!(rgba_close(mid, Rgba::new(4.0, 2.0, 1.0, 1.0)));
        let lut = g.bake(8);
        assert!(lut.values()[lut.len() - 1].r > 1.0);
    }

    #[test]
    fn empty_gradient_guards_to_transparent_black() {
        let g = ColorGradient::new();
        assert!(g.is_empty());
        assert!(rgba_close(g.sample(0.0), Rgba::TRANSPARENT));
        assert!(rgba_close(g.sample(0.5), Rgba::TRANSPARENT));
    }

    #[test]
    fn empty_lut_sample_nearest_guards_to_transparent_black() {
        let lut = ColorLut::new(Vec::new());
        assert!(lut.is_empty());
        assert!(rgba_close(lut.sample_nearest(0.5), Rgba::TRANSPARENT));
    }

    #[test]
    fn sample_nearest_rounds_and_clamps_index() {
        let texels = vec![
            Rgba::new(0.0, 0.0, 0.0, 1.0),
            Rgba::new(1.0, 0.0, 0.0, 1.0),
            Rgba::new(0.0, 1.0, 0.0, 1.0),
            Rgba::new(0.0, 0.0, 1.0, 1.0),
            Rgba::new(1.0, 1.0, 1.0, 1.0),
        ];
        let lut = ColorLut::new(texels.clone());
        // len - 1 == 4, so t maps onto index round(t * 4).
        assert!(rgba_close(lut.sample_nearest(0.0), texels[0]));
        assert!(rgba_close(lut.sample_nearest(0.24), texels[1]));
        assert!(rgba_close(lut.sample_nearest(0.5), texels[2]));
        assert!(rgba_close(lut.sample_nearest(1.0), texels[4]));
        // Out-of-range positions clamp to the endpoints.
        assert!(rgba_close(lut.sample_nearest(-5.0), texels[0]));
        assert!(rgba_close(lut.sample_nearest(5.0), texels[4]));
    }

    #[test]
    fn position_out_of_range_stops_clamp_on_sample() {
        // Authored positions outside 0..=1 still behave as endpoint holds.
        let a = Rgba::new(0.3, 0.3, 0.3, 1.0);
        let b = Rgba::new(0.7, 0.7, 0.7, 1.0);
        let g = ColorGradient::from_stops(vec![ColorStop::new(-0.5, a), ColorStop::new(1.5, b)]);
        assert!(rgba_close(g.sample(-1.0), a));
        assert!(rgba_close(g.sample(2.0), b));
        assert!(rgba_close(g.sample(0.5), Rgba::new(0.5, 0.5, 0.5, 1.0)));
    }
}
