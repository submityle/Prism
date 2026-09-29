//! Edge-light / `Fresnel` rim term for particle shading (design §16, §17).
//!
//! A rim (a.k.a. edge or `Fresnel`) light brightens a surface where it turns
//! away from the viewer, tracing a luminous outline around silhouettes. It is a
//! staple of both `PBR` reflectance (the `Schlick` `Fresnel` approximation) and
//! `NPR` stylization (a widened, artist-controlled rim band). This module owns
//! the `CPU` reference for that term so a future `GPU` draw kernel matches it
//! bit for bit.
//!
//! The model composes three pieces, all evaluated from the view-space geometry
//! term `NdotV` (the cosine between the surface normal and the view direction):
//!
//! 1. [`fresnel_schlick`] — the physically based `Schlick` approximation
//!    `f0 + (1 - f0) * (1 - cos)^5`, with the fifth power evaluated by an
//!    integer multiply loop ([`power_u32`]) rather than a transcendental
//!    `powf`.
//! 2. [`rim_factor`] — a generalized `Schlick` rim with an artist-chosen integer
//!    exponent, so a larger exponent yields a tighter (narrower) rim.
//! 3. [`rim_intensity`] — a `smoothstep` band that lets the rim brighten toward
//!    grazing angles (small `NdotV`) and fade to nothing head-on.
//!
//! [`RimParams::evaluate`] wires them together for a hand-rolled normal / view
//! pair into a [`RimSample`] (a scalar factor plus a linear-`RGB` contribution).
//!
//! Only `sqrt` and rational / `smoothstep` / integer-power arithmetic are used —
//! no transcendental functions — so the result is deterministic and portable.
//! `GPU` packing follows the shared `std430` `vec4` alignment from
//! [`super::gpu_layout`].

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Minimum squared length below which a direction is treated as degenerate and
/// normalizes to the zero vector instead of dividing by zero.
const MIN_LEN_SQ: f32 = 1e-12;

/// Minimum `smoothstep` interval (and generic denominator guard) below which a
/// soft edge collapses to a hard step, so no division by zero yields a `NaN`.
const MIN_EDGE: f32 = 1e-6;

/// Byte stride of one [`RimParams`] record in a `std430` storage buffer.
///
/// The eight scalars pack into two `vec4` slots: `vec4(rim_color, intensity)`
/// followed by `vec4(f0, power_bits, inner, outer)`.
pub const RIM_PARAMS_STRIDE: usize = 2 * VEC4_STRIDE;

/// Clamps a scalar into the `0..=1` range without branching on equality.
#[must_use]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Dot product of two hand-rolled 3-component vectors.
#[must_use]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Normalizes a 3-component vector, returning the zero vector for a degenerate
/// (near-zero-length) input instead of producing a `NaN`.
#[must_use]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len_sq = dot3(v, v);
    if len_sq < MIN_LEN_SQ {
        return [0.0, 0.0, 0.0];
    }
    let inv_len = 1.0 / len_sq.sqrt();
    [v[0] * inv_len, v[1] * inv_len, v[2] * inv_len]
}

/// Hermite `smoothstep` from `edge0` to `edge1` evaluated at `x`.
///
/// Returns `0.0` at or below `edge0`, `1.0` at or above `edge1`, and the
/// `t * t * (3 - 2 * t)` interpolation in between. A degenerate (near-equal)
/// interval collapses to a hard step at `edge1` rather than dividing by zero.
#[must_use]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if span < MIN_EDGE {
        return if x < edge1 { 0.0 } else { 1.0 };
    }
    let t = clamp01((x - edge0) / span);
    t * t * (3.0 - 2.0 * t)
}

/// Raises `base` to the integer power `exp` by repeated multiplication.
///
/// This replaces `f32::powf` so the rim exponent stays transcendental-free and
/// bit-reproducible across the `CPU` reference and the eventual `GPU` kernel.
/// `power_u32(base, 0)` is `1.0` by the empty-product convention.
#[must_use]
pub fn power_u32(base: f32, exp: u32) -> f32 {
    let mut acc = 1.0;
    let mut remaining = exp;
    while remaining > 0 {
        acc *= base;
        remaining -= 1;
    }
    acc
}

/// The `Schlick` `Fresnel` approximation `f0 + (1 - f0) * (1 - cos)^5`.
///
/// `cos_theta` is the cosine between the surface normal and the view direction
/// and is clamped to `0..=1`. `f0` is the reflectance at normal incidence. At
/// `cos_theta = 1` (head-on) the result is `f0`; at `cos_theta = 0` (grazing)
/// it saturates to `1.0`. The fifth power uses [`power_u32`].
#[must_use]
pub fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    let cos = clamp01(cos_theta);
    let one_minus_cos = 1.0 - cos;
    f0 + (1.0 - f0) * power_u32(one_minus_cos, 5)
}

/// A generalized `Schlick` rim: `f0 + (1 - f0) * (1 - NdotV)^power`.
///
/// Unlike [`fresnel_schlick`]'s fixed fifth power, the exponent is an artist
/// control: a larger `power` makes `(1 - NdotV)^power` decay faster away from
/// the silhouette, so the lit rim gets narrower. `n_dot_v` is clamped to
/// `0..=1`; at `n_dot_v = 1` the rim is `f0`, at `n_dot_v = 0` it saturates to
/// `1.0`.
#[must_use]
pub fn rim_factor(n_dot_v: f32, power: u32, f0: f32) -> f32 {
    let n = clamp01(n_dot_v);
    f0 + (1.0 - f0) * power_u32(1.0 - n, power)
}

/// A `smoothstep` intensity band that brightens the rim toward grazing angles.
///
/// The band is `1.0 - smoothstep(inner, outer, NdotV)`, so it is at full
/// strength where `NdotV <= inner` (grazing, silhouette edge) and fades to
/// `0.0` where `NdotV >= outer` (head-on interior). `inner` is expected to be
/// less than `outer`; a degenerate interval collapses to a hard step via
/// [`smoothstep`]. `n_dot_v` is clamped to `0..=1`.
#[must_use]
pub fn rim_intensity(n_dot_v: f32, inner: f32, outer: f32) -> f32 {
    let n = clamp01(n_dot_v);
    1.0 - smoothstep(inner, outer, n)
}

/// The result of evaluating a rim term for one surface fragment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RimSample {
    /// The unitless rim strength: the generalized `Schlick` factor multiplied by
    /// the `smoothstep` band, in `0..=1`.
    pub factor: f32,
    /// The linear-`RGB` (optionally `HDR`) rim contribution,
    /// `rim_color * factor * intensity`.
    pub rgb: [f32; 3],
}

/// Parameters controlling the `Fresnel` / edge-light rim for a renderer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RimParams {
    /// Reflectance at normal incidence for the `Schlick` rim (`0..=1`).
    pub f0: f32,
    /// Integer rim exponent; larger values tighten the rim (see [`rim_factor`]).
    pub power: u32,
    /// Linear-`RGB` colour of the rim light.
    pub rim_color: [f32; 3],
    /// Scalar multiplier applied to the emitted rim colour (allows `HDR` gain).
    pub intensity: f32,
    /// Inner `NdotV` edge of the `smoothstep` intensity band.
    pub inner: f32,
    /// Outer `NdotV` edge of the `smoothstep` intensity band.
    pub outer: f32,
}

impl RimParams {
    /// Creates rim parameters from all fields.
    #[must_use]
    pub const fn new(
        f0: f32,
        power: u32,
        rim_color: [f32; 3],
        intensity: f32,
        inner: f32,
        outer: f32,
    ) -> Self {
        Self {
            f0,
            power,
            rim_color,
            intensity,
            inner,
            outer,
        }
    }

    /// Evaluates the rim term for a `normal` and a `view_dir`.
    ///
    /// Both inputs are normalized in-place (a degenerate input normalizes to the
    /// zero vector, i.e. grazing), then `NdotV = clamp(dot(n, v), 0, 1)` drives
    /// [`rim_factor`] and [`rim_intensity`]. The returned [`RimSample::factor`]
    /// is their product and [`RimSample::rgb`] is `rim_color * factor *
    /// intensity`.
    #[must_use]
    pub fn evaluate(&self, normal: [f32; 3], view_dir: [f32; 3]) -> RimSample {
        let n = normalize3(normal);
        let v = normalize3(view_dir);
        let n_dot_v = clamp01(dot3(n, v));
        let factor = rim_factor(n_dot_v, self.power, self.f0)
            * rim_intensity(n_dot_v, self.inner, self.outer);
        let scaled = factor * self.intensity;
        RimSample {
            factor,
            rgb: [
                self.rim_color[0] * scaled,
                self.rim_color[1] * scaled,
                self.rim_color[2] * scaled,
            ],
        }
    }

    /// Packs the parameters into their `std430` `vec4`-aligned scalar layout.
    ///
    /// Layout: `[rim_r, rim_g, rim_b, intensity, f0, power_bits, inner, outer]`
    /// — two `vec4` slots, matching [`RIM_PARAMS_STRIDE`]. Float fields are
    /// emitted as their `f32::to_bits` patterns and `power` as its raw `u32`, so
    /// the mixed-type block round-trips exactly without a lossy cast.
    #[must_use]
    pub fn to_std430(&self) -> [u32; 8] {
        [
            self.rim_color[0].to_bits(),
            self.rim_color[1].to_bits(),
            self.rim_color[2].to_bits(),
            self.intensity.to_bits(),
            self.f0.to_bits(),
            self.power,
            self.inner.to_bits(),
            self.outer.to_bits(),
        ]
    }
}

/// Total byte size of a `std430` storage buffer holding `count` packed
/// [`RimParams`] records.
///
/// Uses [`RIM_PARAMS_STRIDE`] and the shared clamp-to-one-element rule from
/// [`storage_bytes`], so an empty set still yields a valid `GPU` binding.
#[must_use]
pub fn rim_params_buffer_bytes(count: usize) -> usize {
    storage_bytes(RIM_PARAMS_STRIDE, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for the float assertions in this module's tests.
    const CMP_EPS: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    #[test]
    fn power_u32_matches_repeated_multiplication() {
        assert!(approx_eq(power_u32(2.0, 0), 1.0));
        assert!(approx_eq(power_u32(2.0, 1), 2.0));
        assert!(approx_eq(power_u32(2.0, 5), 32.0));
        assert!(approx_eq(power_u32(0.5, 3), 0.125));
        // Matches an explicit product for the fifth power used by Schlick.
        let b = 0.3_f32;
        let manual = b * b * b * b * b;
        assert!(approx_eq(power_u32(b, 5), manual));
    }

    #[test]
    fn dot3_and_normalize3_are_hand_rolled() {
        assert!(approx_eq(dot3([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0));
        // A degenerate vector normalizes to zero, not NaN.
        assert_eq!(normalize3([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
        let unit = normalize3([0.0, 5.0, 0.0]);
        assert!(approx_eq(unit[0], 0.0));
        assert!(approx_eq(unit[1], 1.0));
        assert!(approx_eq(unit[2], 0.0));
        // A general vector has unit length after normalization.
        let n = normalize3([3.0, 4.0, 0.0]);
        assert!(approx_eq(dot3(n, n), 1.0));
    }

    #[test]
    fn smoothstep_endpoints_are_zero_and_one() {
        assert!(approx_eq(smoothstep(0.0, 1.0, -0.5), 0.0));
        assert!(approx_eq(smoothstep(0.0, 1.0, 0.0), 0.0));
        assert!(approx_eq(smoothstep(0.0, 1.0, 1.0), 1.0));
        assert!(approx_eq(smoothstep(0.0, 1.0, 1.5), 1.0));
        assert!(approx_eq(smoothstep(0.0, 1.0, 0.5), 0.5));
        // Degenerate interval is a hard step at edge1.
        assert!(approx_eq(smoothstep(0.5, 0.5, 0.4), 0.0));
        assert!(approx_eq(smoothstep(0.5, 0.5, 0.5), 1.0));
    }

    #[test]
    fn fresnel_schlick_hits_its_endpoints() {
        let f0 = 0.04;
        // Head-on: returns f0.
        assert!(approx_eq(fresnel_schlick(1.0, f0), f0));
        // Grazing: saturates to 1.0.
        assert!(approx_eq(fresnel_schlick(0.0, f0), 1.0));
        // Out-of-range cosine clamps to the endpoints.
        assert!(approx_eq(fresnel_schlick(2.0, f0), f0));
        assert!(approx_eq(fresnel_schlick(-1.0, f0), 1.0));
    }

    #[test]
    fn fresnel_schlick_is_monotone_toward_grazing() {
        let f0 = 0.04;
        let head_on = fresnel_schlick(0.9, f0);
        let mid = fresnel_schlick(0.5, f0);
        let grazing = fresnel_schlick(0.1, f0);
        assert!(head_on < mid);
        assert!(mid < grazing);
    }

    #[test]
    fn rim_factor_facing_is_f0_and_grazing_is_one() {
        let f0 = 0.1;
        assert!(approx_eq(rim_factor(1.0, 3, f0), f0));
        assert!(approx_eq(rim_factor(0.0, 3, f0), 1.0));
    }

    #[test]
    fn rim_factor_narrows_as_power_grows() {
        // For a fixed interior angle, a larger exponent yields a smaller (hence
        // narrower) rim contribution: strictly monotone decreasing in power.
        let low = rim_factor(0.5, 2, 0.0);
        let mid = rim_factor(0.5, 4, 0.0);
        let high = rim_factor(0.5, 8, 0.0);
        assert!(high < mid);
        assert!(mid < low);
    }

    #[test]
    fn rim_intensity_is_bright_at_the_edge() {
        // Grazing (small NdotV) is fully lit; head-on is dark.
        assert!(approx_eq(rim_intensity(0.0, 0.2, 0.8), 1.0));
        assert!(approx_eq(rim_intensity(1.0, 0.2, 0.8), 0.0));
        // Monotone decreasing from grazing to head-on across the band.
        let edge = rim_intensity(0.3, 0.2, 0.8);
        let mid = rim_intensity(0.5, 0.2, 0.8);
        let interior = rim_intensity(0.7, 0.2, 0.8);
        assert!(mid < edge);
        assert!(interior < mid);
    }

    #[test]
    fn evaluate_facing_view_has_near_zero_rim() {
        let params = RimParams::new(0.2, 4, [1.0, 0.8, 0.5], 3.0, 0.2, 0.8);
        // Normal and view aligned -> NdotV = 1 -> band is zero -> rim is zero.
        let sample = params.evaluate([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]);
        assert!(approx_eq(sample.factor, 0.0));
        assert!(approx_eq(sample.rgb[0], 0.0));
        assert!(approx_eq(sample.rgb[1], 0.0));
        assert!(approx_eq(sample.rgb[2], 0.0));
    }

    #[test]
    fn evaluate_grazing_view_has_maximal_rim() {
        let params = RimParams::new(0.2, 4, [1.0, 0.8, 0.5], 3.0, 0.2, 0.8);
        // Normal perpendicular to view -> NdotV = 0 -> factor saturates to 1.
        let sample = params.evaluate([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(approx_eq(sample.factor, 1.0));
    }

    #[test]
    fn evaluate_rgb_is_color_times_factor_times_intensity() {
        let params = RimParams::new(0.0, 3, [0.6, 0.3, 0.9], 2.0, 0.2, 0.8);
        let normal = [1.0, 1.0, 0.0];
        let view = [0.0, 0.0, 1.0];
        let sample = params.evaluate(normal, view);
        let scaled = sample.factor * params.intensity;
        assert!(approx_eq(sample.rgb[0], params.rim_color[0] * scaled));
        assert!(approx_eq(sample.rgb[1], params.rim_color[1] * scaled));
        assert!(approx_eq(sample.rgb[2], params.rim_color[2] * scaled));
    }

    #[test]
    fn evaluate_is_deterministic() {
        let params = RimParams::new(0.05, 5, [0.4, 0.7, 1.0], 1.5, 0.1, 0.9);
        let normal = [0.3, 0.6, 0.2];
        let view = [0.1, 0.2, 0.9];
        let a = params.evaluate(normal, view);
        let b = params.evaluate(normal, view);
        assert_eq!(a, b);
    }

    #[test]
    fn std430_stride_and_buffer_bytes() {
        assert_eq!(RIM_PARAMS_STRIDE, 32);
        // Empty set still reserves a single element.
        assert_eq!(rim_params_buffer_bytes(0), RIM_PARAMS_STRIDE);
        assert_eq!(rim_params_buffer_bytes(1), RIM_PARAMS_STRIDE);
        assert_eq!(rim_params_buffer_bytes(4), 4 * RIM_PARAMS_STRIDE);
    }

    #[test]
    fn std430_packing_layout_round_trips() {
        let params = RimParams::new(0.04, 6, [0.2, 0.4, 0.8], 2.5, 0.15, 0.85);
        let packed = params.to_std430();
        assert_eq!(packed.len(), 8);
        // vec4 slot 0: rim_color, intensity.
        assert!(approx_eq(f32::from_bits(packed[0]), 0.2));
        assert!(approx_eq(f32::from_bits(packed[1]), 0.4));
        assert!(approx_eq(f32::from_bits(packed[2]), 0.8));
        assert!(approx_eq(f32::from_bits(packed[3]), 2.5));
        // vec4 slot 1: f0, power (raw u32), inner, outer.
        assert!(approx_eq(f32::from_bits(packed[4]), 0.04));
        assert_eq!(packed[5], 6);
        assert!(approx_eq(f32::from_bits(packed[6]), 0.15));
        assert!(approx_eq(f32::from_bits(packed[7]), 0.85));
    }
}
