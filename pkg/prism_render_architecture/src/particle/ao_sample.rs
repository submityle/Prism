//! Screen-space `hemisphere` ambient-occlusion (`AO`) sampling kernels for the
//! particle shading pass (design §16-§21).
//!
//! Production engines darken creases and contact points with a screen-space
//! ambient-occlusion pass (`SSAO` / `HBAO` style): around each shaded point a
//! small set of offset directions is scattered over the upper `hemisphere`
//! aligned with the surface normal, the scene depth is fetched at each offset,
//! and a point counts as occluded when the sampled geometry sits *nearer* the
//! camera than the offset position. This module owns the `CPU`-verifiable
//! reference for that kernel so a future `GPU` kernel can match it bit for bit.
//!
//! The building blocks are: (1) an integer low-discrepancy sequence maps sample
//! indices onto the upper `hemisphere` with pure `f32::sqrt` and algebra
//! ([`sample_kernel`]) — no trigonometry; (2) one sample's occlusion is a
//! rational distance falloff ([`occlusion_factor`]); (3) a whole kernel folds
//! into an occlusion estimate with a `smoothstep` range check
//! ([`ao_from_samples`]); (4) contrast is shaped by integer exponentiation
//! ([`ao_power`]); and (5) [`AoParams`] gathers the tunables and drives
//! [`AoParams::evaluate`]. `GPU` packing follows the shared `std430` `vec4`
//! alignment from [`super::gpu_layout`].
//!
//! Deliberately out of scope: irradiance / global-illumination probes live in
//! [`super::gi_probe`], and hierarchical-Z occlusion *culling* lives in
//! [`super::occlusion`]. This file neither imports nor re-derives either.
//!
//! Only `f32::sqrt`, `f32::floor`, integer arithmetic, and integer hashing are
//! used — no transcendental functions — so results are deterministic and
//! platform independent.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Generic denominator / soft-edge guard below which a division collapses to a
/// hard step, so no divide-by-zero can produce a `NaN`.
const MIN_EDGE: f32 = 1e-6;

/// `2^32` as an `f32`: the normalizing span turning a `u32` hash word into a
/// unit-interval fraction.
const U32_SPAN: f32 = 4_294_967_296.0;

/// Fixed-point `.32` increment for the first `R2` low-discrepancy axis
/// (`round(2^32 / plastic_constant)`).
const R2_INC_X: u32 = 3_242_174_889;

/// Fixed-point `.32` increment for the second `R2` low-discrepancy axis
/// (`round(2^32 / plastic_constant^2)`).
const R2_INC_Y: u32 = 2_447_445_413;

/// Byte stride of one [`AoParams`] record in a `std430` storage buffer.
///
/// The five scalars pack into two `vec4` slots: `vec4(radius, bias, intensity,
/// power)` followed by `vec4(sample_count, pad, pad, pad)`.
pub const AO_PARAMS_STRIDE: usize = 2 * VEC4_STRIDE;

/// Clamps a scalar into the `0..=1` range without branching on equality.
#[must_use]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
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

/// Integer avalanche hash mixing a `u32` seed into a well-distributed `u32`.
///
/// A self-contained mixer (no dependency on any noise module): xor-shifts and
/// odd-constant multiplies giving a deterministic, platform-independent word.
#[must_use]
fn hash_u32(seed: u32) -> u32 {
    let mut x = seed;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// Normalizes a `u32` hash word into a `0..1` unit fraction.
#[must_use]
fn to_unit(x: u32) -> f32 {
    (x as f32) / U32_SPAN
}

/// Number of `GPU` workgroups needed to cover `sample_count` invocations at
/// `group_size` threads each, via integer `div_ceil` (never zero-sized).
#[must_use]
pub fn dispatch_groups(sample_count: u32, group_size: u32) -> u32 {
    sample_count.div_ceil(group_size.max(1))
}

/// Builds an `n`-direction upper-`hemisphere` sampling kernel seeded by `seed`.
///
/// Each direction lives in tangent space with `z >= 0` (the `hemisphere` axis
/// is `+z`, i.e. the surface normal) and has length `<= 1`. Sample indices are
/// scattered by an integer `R2` low-discrepancy sequence, mapped from the unit
/// square onto the disk with the trig-free *elliptical grid* transform, then
/// lifted onto the `hemisphere` by `z = sqrt(1 - x*x - y*y)`. The per-sample
/// radius accelerates as `0.1 + 0.9 * t*t` (with `t = i / n`) so samples cluster
/// near the shaded point — the standard `SSAO` near-field bias — without any
/// transcendental call. An `n` of `0` yields an empty kernel.
#[must_use]
pub fn sample_kernel(n: u32, seed: u32) -> Vec<[f32; 3]> {
    let base_x = hash_u32(seed);
    let base_y = hash_u32(seed ^ 0x9e37_79b9);
    (0..n)
        .map(|i| {
            let u = to_unit(base_x.wrapping_add(R2_INC_X.wrapping_mul(i)));
            let v = to_unit(base_y.wrapping_add(R2_INC_Y.wrapping_mul(i)));
            // Unit square -> [-1, 1]^2.
            let p = 2.0 * u - 1.0;
            let q = 2.0 * v - 1.0;
            // Elliptical grid map: square -> unit disk using only `sqrt`.
            let x = p * (1.0 - 0.5 * q * q).max(0.0).sqrt();
            let y = q * (1.0 - 0.5 * p * p).max(0.0).sqrt();
            // Lift onto the upper hemisphere; the (x, y, z) triple is unit long.
            let z = (1.0 - x * x - y * y).max(0.0).sqrt();
            let t = (i as f32) / (n as f32);
            let scale = 0.1 + 0.9 * t * t;
            [x * scale, y * scale, z * scale]
        })
        .collect()
}

/// Single-sample occlusion in `0..=1` from a rational distance falloff.
///
/// `sample_depth` is the `view-space` depth of the kernel offset position and
/// `sampled_depth` is the scene depth fetched there; larger depth means farther
/// from the camera. The sample occludes only when the fetched geometry is
/// *nearer* than the offset (`sampled_depth < sample_depth`). Its strength is
/// the rational falloff `range*range / (range*range + delta*delta)`, which is
/// `1` at zero separation and decays past `range` — a bounded stand-in for the
/// usual exponential falloff, with no transcendental call. Equal depths and
/// farther-away geometry contribute `0`.
#[must_use]
pub fn occlusion_factor(sample_depth: f32, sampled_depth: f32, range: f32) -> f32 {
    let delta = sample_depth - sampled_depth;
    if delta <= 0.0 {
        return 0.0;
    }
    let r2 = range * range;
    r2 / (r2 + delta * delta)
}

/// Folds sampled scene depths into an ambient-occlusion term in `0..=1`.
///
/// `depths` holds the scene depth fetched at each kernel offset and
/// `center_depth` is the depth of the shaded point; larger depth means farther
/// from the camera. A sample occludes when it is nearer than the shaded point
/// (`d < center_depth`), weighted by a `smoothstep` range check so occluders
/// far beyond `range` fade out with a soft edge rather than a hard cutoff. The
/// mean occlusion is inverted, so the result follows the convention **`1.0` =
/// fully lit / unoccluded** and **`0.0` = fully occluded**. An empty sample set
/// is unoccluded (`1.0`).
#[must_use]
pub fn ao_from_samples(depths: &[f32], center_depth: f32, range: f32) -> f32 {
    if depths.is_empty() {
        return 1.0;
    }
    let mut occlusion = 0.0;
    for &d in depths {
        let delta = center_depth - d;
        if delta > 0.0 {
            // Soft range check: full weight while `delta < range`, fading out
            // beyond it. `range / delta` is guarded by the `delta > 0` branch.
            occlusion += smoothstep(0.0, 1.0, range / delta);
        }
    }
    let mean = occlusion / (depths.len() as f32);
    clamp01(1.0 - mean)
}

/// Raises `ao` to the integer power `exp` by repeated multiplication.
///
/// Sharpens (or softens) `AO` contrast without `f32::powf`: the input is
/// clamped to `0..=1`, and `exp == 0` yields `1.0` by the usual `x^0`
/// convention. For a fixed base in `0..=1` the result is non-increasing in
/// `exp`; for a fixed `exp` it is non-decreasing in the base.
#[must_use]
pub fn ao_power(ao: f32, exp: u32) -> f32 {
    let base = clamp01(ao);
    let mut acc = 1.0;
    for _ in 0..exp {
        acc *= base;
    }
    acc
}

/// Tunables for the `hemisphere` ambient-occlusion kernel (design §16-§21).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AoParams {
    /// `view-space` radius over which occluders are gathered and range-checked.
    pub radius: f32,
    /// Depth bias subtracted from the shaded point to suppress self-occlusion
    /// acne on flat surfaces.
    pub bias: f32,
    /// Scales how strongly the occlusion term darkens the final result.
    pub intensity: f32,
    /// Integer contrast exponent applied through [`ao_power`].
    pub power: u32,
    /// Number of `hemisphere` samples the kernel scatters per shaded point.
    pub sample_count: u32,
}

impl AoParams {
    /// Creates ambient-occlusion parameters from all fields.
    #[must_use]
    pub const fn new(
        radius: f32,
        bias: f32,
        intensity: f32,
        power: u32,
        sample_count: u32,
    ) -> Self {
        Self {
            radius,
            bias,
            intensity,
            power,
            sample_count,
        }
    }

    /// Evaluates the ambient-occlusion term in `0..=1` for the sampled scene
    /// `depths` around a point at `center_depth`.
    ///
    /// The [`bias`](Self::bias) is subtracted from the shaded depth so an
    /// occluder must be nearer by more than the bias to count (suppressing
    /// self-occlusion), the raw term comes from [`ao_from_samples`] over
    /// [`radius`](Self::radius), contrast is shaped by [`ao_power`] with
    /// [`power`](Self::power), and [`intensity`](Self::intensity) scales the
    /// darkening: `intensity == 0` leaves the point fully lit, `intensity == 1`
    /// applies the term directly. The result follows the same `1.0` = fully lit
    /// convention.
    #[must_use]
    pub fn evaluate(&self, depths: &[f32], center_depth: f32) -> f32 {
        let raw = ao_from_samples(depths, center_depth - self.bias, self.radius);
        let powered = ao_power(raw, self.power);
        clamp01(1.0 - self.intensity * (1.0 - powered))
    }

    /// Packs the parameters into their `std430` `vec4`-aligned word layout.
    ///
    /// Layout: `[radius, bias, intensity, power, sample_count, pad, pad, pad]`
    /// as raw `u32` words (the three `f32` fields via `f32::to_bits`) — two
    /// `vec4` slots, matching [`AO_PARAMS_STRIDE`]. The trailing words are
    /// padding.
    #[must_use]
    pub fn to_std430(&self) -> [u32; 8] {
        [
            self.radius.to_bits(),
            self.bias.to_bits(),
            self.intensity.to_bits(),
            self.power,
            self.sample_count,
            0,
            0,
            0,
        ]
    }
}

/// Total byte size of a `std430` storage buffer holding `count` packed
/// [`AoParams`] records.
///
/// Uses [`AO_PARAMS_STRIDE`] and the shared clamp-to-one-element rule from
/// [`storage_bytes`], so an empty set still yields a valid `GPU` binding.
#[must_use]
pub fn ao_params_buffer_bytes(count: usize) -> usize {
    storage_bytes(AO_PARAMS_STRIDE, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for the float assertions in this module's tests.
    const CMP_EPS: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn length_squared(v: [f32; 3]) -> f32 {
        v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    }

    #[test]
    fn kernel_directions_stay_in_the_upper_hemisphere() {
        let kernel = sample_kernel(128, 0x000a_11ce);
        assert_eq!(kernel.len(), 128);
        for dir in &kernel {
            // Upper hemisphere: the normal-aligned axis is never negative.
            assert!(dir[2] >= 0.0);
            // Radius never exceeds the unit sphere.
            assert!(length_squared(*dir) <= 1.0 + CMP_EPS);
        }
    }

    #[test]
    fn kernel_is_deterministic_for_a_seed() {
        let a = sample_kernel(64, 7);
        let b = sample_kernel(64, 7);
        assert_eq!(a, b);
        // A different seed decorrelates the kernel.
        let c = sample_kernel(64, 8);
        assert!(a != c);
    }

    #[test]
    fn kernel_radius_is_near_field_dense_and_monotone() {
        let kernel = sample_kernel(48, 3);
        // Radius grows with the index (0.1 + 0.9 * t*t), clustering samples near
        // the shaded point.
        for pair in kernel.windows(2) {
            assert!(length_squared(pair[0]) <= length_squared(pair[1]) + CMP_EPS);
        }
        // The first sample sits at the minimum 0.1 radius.
        assert!(approx_eq(length_squared(kernel[0]).sqrt(), 0.1));
    }

    #[test]
    fn empty_kernel_when_zero_samples() {
        assert!(sample_kernel(0, 1).is_empty());
    }

    #[test]
    fn occlusion_factor_needs_a_nearer_occluder() {
        // Equal depth: nothing in front, no occlusion.
        assert!(approx_eq(occlusion_factor(5.0, 5.0, 1.0), 0.0));
        // Farther geometry (behind the point): no occlusion.
        assert!(approx_eq(occlusion_factor(5.0, 6.0, 1.0), 0.0));
        // Nearer geometry occludes.
        let near = occlusion_factor(5.0, 4.8, 1.0);
        assert!(near > 0.0);
        assert!(near <= 1.0);
    }

    #[test]
    fn occlusion_factor_decays_past_range() {
        // At a separation equal to `range` the rational falloff is exactly 0.5.
        assert!(approx_eq(occlusion_factor(2.0, 1.0, 1.0), 0.5));
        // Larger separations occlude strictly less.
        let close = occlusion_factor(2.0, 1.5, 1.0);
        let far = occlusion_factor(2.0, 0.5, 1.0);
        assert!(close > far);
    }

    #[test]
    fn ao_from_samples_conventions() {
        // Empty set is fully lit.
        assert!(approx_eq(ao_from_samples(&[], 5.0, 1.0), 1.0));
        // All samples at the shaded depth: no occluder in front, fully lit.
        assert!(approx_eq(ao_from_samples(&[5.0, 5.0, 5.0], 5.0, 1.0), 1.0));
        // Farther samples (behind): still fully lit.
        assert!(approx_eq(ao_from_samples(&[6.0, 7.0], 5.0, 1.0), 1.0));
    }

    #[test]
    fn ao_from_samples_darkens_with_near_occluders() {
        let lit = ao_from_samples(&[5.0, 5.0, 5.0, 5.0], 5.0, 1.0);
        let occluded = ao_from_samples(&[4.5, 4.6, 4.7, 4.8], 5.0, 1.0);
        assert!(occluded < lit);
        assert!(occluded >= 0.0);
        assert!(occluded <= 1.0);
        // A very-near occluder set stays in range.
        let dark = ao_from_samples(&[4.99, 4.99, 4.99, 4.99], 5.0, 1.0);
        assert!(dark >= 0.0);
        assert!(dark <= 1.0);
    }

    #[test]
    fn ao_power_is_bounded_and_monotone() {
        // x^0 == 1 by convention.
        assert!(approx_eq(ao_power(0.4, 0), 1.0));
        // x^1 == x (clamped).
        assert!(approx_eq(ao_power(0.4, 1), 0.4));
        // Non-increasing in the exponent for a base in 0..=1.
        let mut prev = 2.0;
        for exp in 0..8u32 {
            let v = ao_power(0.6, exp);
            assert!(v <= prev + CMP_EPS);
            assert!(v >= 0.0);
            assert!(v <= 1.0);
            prev = v;
        }
        // Non-decreasing in the base for a fixed exponent.
        assert!(ao_power(0.3, 3) <= ao_power(0.7, 3) + CMP_EPS);
    }

    #[test]
    fn evaluate_respects_intensity_and_bias() {
        let params = AoParams::new(1.0, 0.05, 1.0, 1, 16);
        // Near occluders darken the point.
        let occluded = params.evaluate(&[4.5, 4.6, 4.7, 4.8], 5.0);
        assert!(occluded >= 0.0);
        assert!(occluded <= 1.0);
        assert!(occluded < 1.0);
        // Zero intensity leaves the point fully lit regardless of occlusion.
        let unlit = AoParams::new(1.0, 0.05, 0.0, 2, 16);
        assert!(approx_eq(unlit.evaluate(&[4.5, 4.6, 4.7, 4.8], 5.0), 1.0));
    }

    #[test]
    fn std430_layout_and_bytes() {
        assert_eq!(AO_PARAMS_STRIDE, 32);
        let params = AoParams::new(1.5, 0.05, 0.75, 3, 24);
        let packed = params.to_std430();
        // The three f32 fields survive as raw bits.
        assert_eq!(packed[0], 1.5f32.to_bits());
        assert_eq!(packed[1], 0.05f32.to_bits());
        assert_eq!(packed[2], 0.75f32.to_bits());
        // The two integer fields are stored verbatim.
        assert_eq!(packed[3], 3);
        assert_eq!(packed[4], 24);
        // Padding words are zero.
        assert_eq!(packed[5], 0);
        assert_eq!(packed[6], 0);
        assert_eq!(packed[7], 0);

        assert_eq!(ao_params_buffer_bytes(3), 96);
        // An empty set still reserves one element.
        assert_eq!(ao_params_buffer_bytes(0), AO_PARAMS_STRIDE);
    }

    #[test]
    fn dispatch_groups_round_up() {
        assert_eq!(dispatch_groups(0, 64), 0);
        assert_eq!(dispatch_groups(1, 64), 1);
        assert_eq!(dispatch_groups(64, 64), 1);
        assert_eq!(dispatch_groups(65, 64), 2);
        // A zero group size is guarded to one thread per group.
        assert_eq!(dispatch_groups(10, 0), 10);
    }
}
