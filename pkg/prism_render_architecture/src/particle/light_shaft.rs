//! Screen-space light-shaft (god-ray) radial-blur contract (design §16-§21).
//!
//! A screen-space light shaft is the streak of "volumetric" scattering that
//! appears to radiate outward from a bright light whose on-screen position is
//! known. The classic real-time approximation (Kenny Mitchell, *GPU Gems 3*
//! chapter 13) reconstructs it as a *radial blur*: for every destination pixel
//! the kernel walks a short chain of taps that march from the pixel toward the
//! light's screen `UV`, samples an occlusion/brightness `mask`, and accumulates
//! the taps under a per-step `illuminationDecay` so the scattered energy fades
//! along the ray. This module owns the `CPU`-verifiable maths of that walk and
//! packs its parameters into the `std430` block a future `GPU` post-process
//! kernel binds.
//!
//! # Strict scope
//!
//! This file is *only* the radial, light-centred smear. Each destination pixel
//! reads the `mask` along the single line that connects it to the light's
//! on-screen `UV`; the direction is different for every pixel and always points
//! at the same shared light. That is what distinguishes it from its siblings:
//!
//! - [`super::bloom_upsample`] / [`super::bloom_threshold`] run an *isotropic*
//!   tent-filter pyramid with no preferred direction.
//! - [`super::anamorphic_streak`] smears along a *single fixed* one-dimensional
//!   lens axis that is identical for every pixel, not toward a shared point.
//! - [`super::motion_blur`] taps along a *per-pixel velocity* vector sampled
//!   from a motion buffer, unrelated to any light position.
//! - `volumetrics` integrates a *froxel* volume in view space rather than
//!   blurring a flat screen-space `mask`.
//!
//! # Determinism
//!
//! The determinism-locked contract layer forbids transcendental functions
//! (`sin`/`cos`/`exp`/`ln`/`powf`) and rounding intrinsics beyond `floor`. The
//! `illuminationDecay` term is a pure repeated multiplication by `decay`, the
//! tap positions advance by repeated addition of a fixed step, and the
//! bilinear `mask` fetch uses `floor` plus multiplies only, so a future `GPU`
//! kernel reproduces the `CPU` result closely.

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};
use alloc::vec::Vec;

/// Byte size of the `std430` packing of [`LightShaftParams`].
///
/// The block occupies two `vec4` slots (32 bytes): the `light_uv` `vec2` plus
/// the `density` and `decay` scalars fill the first slot, and `weight`,
/// `exposure`, the `num_samples` `u32`, and a zeroed padding tail fill the
/// second.
pub const LIGHT_SHAFT_STD430_SIZE: usize = 2 * VEC4_STRIDE;

/// Absolute tolerance for the `f32` equality comparisons used by the tests;
/// direct `==` on floating point is intentionally avoided.
#[cfg(test)]
const CMP_EPS: f32 = 1e-6;

/// Clamps a (possibly negative or oversized) texel coordinate to a valid index
/// in `[0, dim - 1]`.
///
/// Callers pass `dim > 0`, so the last valid index `dim - 1` never underflows.
/// The interior branch only runs when the coordinate already lies strictly
/// inside the open interval `(0, dim - 1)`, so truncating toward zero yields a
/// well-defined, in-range index.
fn clamp_coord(v: f32, dim: usize) -> usize {
    if v <= 0.0 {
        return 0;
    }
    let last = dim - 1;
    if v >= last as f32 {
        return last;
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "v lies strictly inside (0, dim-1) here, so truncating toward zero is exact and non-negative"
    )]
    let idx = v as usize;
    idx
}

/// Hand-rolled bilinear fetch of a single-channel `mask` at a texture-space
/// `uv`, using the half-texel-centred convention and clamp-to-edge addressing.
///
/// The `uv` is mapped to texel space as `uv * dim - 0.5` so that the centre of
/// texel `i` sits at `uv = (i + 0.5) / dim`. The four surrounding texels are
/// clamped to the edge and blended with the fractional weights. An empty image
/// (`w == 0` or `h == 0`) samples as `0`.
fn sample_bilinear(mask: &[f32], w: usize, h: usize, uv: [f32; 2]) -> f32 {
    if w == 0 || h == 0 {
        return 0.0;
    }
    let fx = uv[0] * (w as f32) - 0.5;
    let fy = uv[1] * (h as f32) - 0.5;
    let x0f = fx.floor();
    let y0f = fy.floor();
    let tx = fx - x0f;
    let ty = fy - y0f;
    let ix0 = clamp_coord(x0f, w);
    let ix1 = clamp_coord(x0f + 1.0, w);
    let iy0 = clamp_coord(y0f, h);
    let iy1 = clamp_coord(y0f + 1.0, h);
    let c00 = mask[iy0 * w + ix0];
    let c10 = mask[iy0 * w + ix1];
    let c01 = mask[iy1 * w + ix0];
    let c11 = mask[iy1 * w + ix1];
    let top = c00 + (c10 - c00) * tx;
    let bot = c01 + (c11 - c01) * tx;
    top + (bot - top) * ty
}

/// Screen-space light-shaft radial-blur parameters (design §16-§21).
///
/// `light_uv` is the light's on-screen position in `[0, 1]` texture space (it
/// may lie outside the frame for an off-screen light); `num_samples` is the
/// number of taps marched from each pixel toward the light; `density` scales
/// the total marched distance (the fraction of the pixel-to-light vector the
/// tap chain covers); `decay` is the per-step `illuminationDecay` multiplier;
/// `weight` is the gain applied to every accumulated tap; and `exposure` is the
/// global gain applied to the final accumulated shaft.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightShaftParams {
    /// Light's on-screen position in `[0, 1]` texture-space `UV`.
    pub light_uv: [f32; 2],
    /// Number of taps marched from each pixel toward the light.
    pub num_samples: u32,
    /// Fraction of the pixel-to-light vector the tap chain spans.
    pub density: f32,
    /// Per-step multiplier applied to `illuminationDecay`.
    pub decay: f32,
    /// Gain applied to each accumulated tap.
    pub weight: f32,
    /// Global gain applied to the final accumulated shaft.
    pub exposure: f32,
}

impl LightShaftParams {
    /// Builds a parameter set from its raw fields verbatim.
    ///
    /// No clamping or normalization is performed: `light_uv` may sit outside
    /// the frame, and the scalar knobs accept their full ranges so extreme
    /// `density`/`decay` values can be exercised directly.
    #[must_use]
    pub const fn new(
        light_uv: [f32; 2],
        num_samples: u32,
        density: f32,
        decay: f32,
        weight: f32,
        exposure: f32,
    ) -> Self {
        Self {
            light_uv,
            num_samples,
            density,
            decay,
            weight,
            exposure,
        }
    }

    /// Computes the radial-blur shaft value for a single destination pixel.
    ///
    /// The pixel centre `UV` is `((px + 0.5) / w, (py + 0.5) / h)`. When
    /// `num_samples` is zero the pass is the identity and returns the pixel's
    /// own `mask` value. Otherwise the per-step offset is the pixel-to-light
    /// vector scaled by `density / num_samples`; the accumulator seeds with the
    /// pixel's own `mask` value, then each of `num_samples` taps advances one
    /// step toward the light, fetches the bilinear `mask` value, multiplies it
    /// by the running `illuminationDecay` (which starts at `1` and is scaled by
    /// `decay` after every step) and by `weight`, and adds it in. The sum is
    /// finally scaled by `exposure`.
    ///
    /// Out-of-range coordinates, an empty image, or a `mask` shorter than
    /// `w * h` all yield `0` instead of panicking.
    #[must_use]
    pub fn radial_blur_pixel(&self, mask: &[f32], w: usize, h: usize, px: usize, py: usize) -> f32 {
        if w == 0 || h == 0 || px >= w || py >= h || mask.len() < w * h {
            return 0.0;
        }
        let uv = [
            (px as f32 + 0.5) / (w as f32),
            (py as f32 + 0.5) / (h as f32),
        ];
        if self.num_samples == 0 {
            return sample_bilinear(mask, w, h, uv);
        }
        let inv_n = 1.0 / (self.num_samples as f32);
        let step = [
            (self.light_uv[0] - uv[0]) * self.density * inv_n,
            (self.light_uv[1] - uv[1]) * self.density * inv_n,
        ];
        let mut pos = uv;
        let mut color = sample_bilinear(mask, w, h, pos);
        let mut illumination = 1.0f32;
        for _ in 0..self.num_samples {
            pos[0] += step[0];
            pos[1] += step[1];
            let tap = sample_bilinear(mask, w, h, pos) * illumination;
            color += tap * self.weight;
            illumination *= self.decay;
        }
        color * self.exposure
    }

    /// Applies [`Self::radial_blur_pixel`] to every pixel of a `w * h` `mask`,
    /// returning the shaft image in row-major order.
    ///
    /// An empty image or a `mask` shorter than `w * h` yields an empty vector.
    #[must_use]
    pub fn radial_blur(&self, mask: &[f32], w: usize, h: usize) -> Vec<f32> {
        if w == 0 || h == 0 || mask.len() < w * h {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(w * h);
        for py in 0..h {
            for px in 0..w {
                out.push(self.radial_blur_pixel(mask, w, h, px, py));
            }
        }
        out
    }

    /// Packs the parameters into their `std430` uniform-block byte layout.
    ///
    /// The `light_uv` `vec2` and the `density`/`decay` scalars fill the first
    /// `vec4`; `weight`, `exposure`, and the `num_samples` `u32` fill the
    /// second with a zeroed padding tail.
    #[must_use]
    pub fn to_std430(&self) -> [u8; LIGHT_SHAFT_STD430_SIZE] {
        let mut bytes = [0u8; LIGHT_SHAFT_STD430_SIZE];
        bytes[0..4].copy_from_slice(&self.light_uv[0].to_le_bytes());
        bytes[4..8].copy_from_slice(&self.light_uv[1].to_le_bytes());
        bytes[8..12].copy_from_slice(&self.density.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.decay.to_le_bytes());
        bytes[16..20].copy_from_slice(&self.weight.to_le_bytes());
        bytes[20..24].copy_from_slice(&self.exposure.to_le_bytes());
        bytes[24..28].copy_from_slice(&self.num_samples.to_le_bytes());
        bytes
    }

    /// Total `std430` byte size of a storage buffer holding `count` packed
    /// [`LightShaftParams`] blocks, clamped up to a single element so an empty
    /// buffer still yields a valid non-zero `WebGPU` binding.
    #[must_use]
    pub fn gpu_storage_bytes(count: usize) -> usize {
        storage_bytes(LIGHT_SHAFT_STD430_SIZE, count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn sample_params() -> LightShaftParams {
        LightShaftParams::new([0.5, 0.5], 8, 1.0, 0.9, 0.5, 1.0)
    }

    #[test]
    fn new_stores_fields_verbatim() {
        let p = LightShaftParams::new([0.25, 0.75], 12, 1.5, 0.8, 0.4, 2.0);
        assert!(approx(p.light_uv[0], 0.25));
        assert!(approx(p.light_uv[1], 0.75));
        assert_eq!(p.num_samples, 12);
        assert!(approx(p.density, 1.5));
        assert!(approx(p.decay, 0.8));
        assert!(approx(p.weight, 0.4));
        assert!(approx(p.exposure, 2.0));
    }

    #[test]
    fn std430_size_is_two_vec4_slots() {
        assert_eq!(LIGHT_SHAFT_STD430_SIZE, storage_bytes(VEC4_STRIDE, 2));
        assert_eq!(LIGHT_SHAFT_STD430_SIZE, 32);
        assert_eq!(LIGHT_SHAFT_STD430_SIZE % VEC4_STRIDE, 0);
    }

    #[test]
    fn std430_round_trips_the_scalar_fields() {
        let p = LightShaftParams::new([0.3, 0.6], 9, 1.25, 0.85, 0.45, 2.5);
        let bytes = p.to_std430();
        assert_eq!(bytes.len(), LIGHT_SHAFT_STD430_SIZE);
        let read = |i: usize| {
            let mut b = [0u8; 4];
            b.copy_from_slice(&bytes[i..i + 4]);
            f32::from_le_bytes(b)
        };
        assert!(approx(read(0), 0.3));
        assert!(approx(read(4), 0.6));
        assert!(approx(read(8), 1.25));
        assert!(approx(read(12), 0.85));
        assert!(approx(read(16), 0.45));
        assert!(approx(read(20), 2.5));
    }

    #[test]
    fn std430_encodes_num_samples_bits_and_zero_pad() {
        let p = LightShaftParams::new([0.1, 0.2], 7, 1.0, 0.9, 1.0, 1.0);
        let bytes = p.to_std430();
        let mut b = [0u8; 4];
        b.copy_from_slice(&bytes[24..28]);
        assert_eq!(u32::from_le_bytes(b), 7);
        assert_eq!(&bytes[28..32], &[0u8; 4]);
    }

    #[test]
    fn gpu_storage_bytes_scales_and_reserves_one() {
        assert_eq!(
            LightShaftParams::gpu_storage_bytes(0),
            LIGHT_SHAFT_STD430_SIZE
        );
        assert_eq!(
            LightShaftParams::gpu_storage_bytes(4),
            LIGHT_SHAFT_STD430_SIZE * 4
        );
    }

    #[test]
    fn bilinear_at_pixel_center_is_exact() {
        // Row-major 2x2 mask; each pixel centre must fetch its own value.
        let mask = [1.0, 2.0, 3.0, 4.0];
        assert!(approx(sample_bilinear(&mask, 2, 2, [0.25, 0.25]), 1.0));
        assert!(approx(sample_bilinear(&mask, 2, 2, [0.75, 0.25]), 2.0));
        assert!(approx(sample_bilinear(&mask, 2, 2, [0.25, 0.75]), 3.0));
        assert!(approx(sample_bilinear(&mask, 2, 2, [0.75, 0.75]), 4.0));
    }

    #[test]
    fn bilinear_interpolates_midpoint() {
        let mask = [0.0, 1.0];
        // Midpoint between the two texel centres averages them.
        assert!(approx(sample_bilinear(&mask, 2, 1, [0.5, 0.5]), 0.5));
        // A quarter of the way is a quarter blend.
        assert!(approx(sample_bilinear(&mask, 2, 1, [0.375, 0.5]), 0.25));
    }

    #[test]
    fn bilinear_clamps_outside_the_edge() {
        let mask = [0.0, 1.0];
        // uv left of the first texel centre clamps to the first texel value.
        assert!(approx(sample_bilinear(&mask, 2, 1, [0.0, 0.5]), 0.0));
        // uv right of the last texel centre clamps to the last texel value.
        assert!(approx(sample_bilinear(&mask, 2, 1, [1.0, 0.5]), 1.0));
    }

    #[test]
    fn empty_image_samples_zero() {
        let mask: [f32; 0] = [];
        assert!(approx(sample_bilinear(&mask, 0, 0, [0.5, 0.5]), 0.0));
    }

    #[test]
    fn num_samples_zero_is_identity_at_a_pixel() {
        let mask = [1.0, 2.0, 3.0, 4.0];
        let p = LightShaftParams::new([0.5, 0.5], 0, 1.0, 0.9, 0.5, 3.0);
        // Identity: returns the raw mask value, unaffected by weight/exposure.
        assert!(approx(p.radial_blur_pixel(&mask, 2, 2, 0, 0), 1.0));
        assert!(approx(p.radial_blur_pixel(&mask, 2, 2, 1, 1), 4.0));
    }

    #[test]
    fn num_samples_zero_full_image_copies_mask() {
        let mask = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let p = LightShaftParams::new([0.5, 0.5], 0, 1.0, 0.9, 0.5, 2.0);
        let out = p.radial_blur(&mask, 3, 2);
        assert_eq!(out.len(), mask.len());
        for (o, m) in out.iter().zip(mask.iter()) {
            assert!(approx(*o, *m));
        }
    }

    #[test]
    fn all_black_mask_is_zero() {
        let mask = [0.0f32; 16];
        let p = sample_params();
        let out = p.radial_blur(&mask, 4, 4);
        assert_eq!(out.len(), 16);
        for v in out {
            assert!(approx(v, 0.0));
        }
    }

    #[test]
    fn uniform_mask_is_uniform_and_matches_closed_form() {
        // Uniform mask -> every tap reads the same constant, so the closed form
        // color = c * (1 + weight * sum_{i=0}^{n-1} decay^i), scaled by exposure.
        let c = 0.5f32;
        let mask = [c; 16];
        let p = LightShaftParams::new([0.5, 0.5], 4, 1.0, 0.5, 1.0, 1.0);
        let out = p.radial_blur(&mask, 4, 4);
        // sum of decay^i for i in 0..4 with decay = 0.5 -> 1 + .5 + .25 + .125.
        let geom = 1.0 + 0.5 + 0.25 + 0.125;
        let expected = c * (1.0 + 1.0 * geom);
        for v in &out {
            assert!(approx(*v, expected));
        }
        // Uniform in, uniform out.
        for v in &out {
            assert!(approx(*v, out[0]));
        }
    }

    #[test]
    fn light_pixel_is_self_amplified() {
        // A pixel exactly at the light has a zero step, so every tap re-reads
        // its own value: the shaft is a pure amplification above the raw value.
        let mut mask = [0.0f32; 9];
        mask[4] = 1.0; // centre pixel of a 3x3 grid
        let p = LightShaftParams::new([0.5, 0.5], 6, 1.0, 0.9, 0.5, 1.0);
        let raw = mask[4];
        let out = p.radial_blur_pixel(&mask, 3, 3, 1, 1);
        assert!(out > raw);
    }

    #[test]
    fn density_zero_reads_only_the_pixel_itself() {
        // Zero density collapses the step, matching the self-amplified light
        // pixel behaviour for any pixel.
        let mut mask = [0.0f32; 9];
        mask[0] = 1.0;
        let p = LightShaftParams::new([0.5, 0.5], 4, 0.0, 0.5, 1.0, 1.0);
        let geom = 1.0 + 0.5 + 0.25 + 0.125;
        let expected = 1.0 * (1.0 + 1.0 * geom);
        assert!(approx(p.radial_blur_pixel(&mask, 3, 3, 0, 0), expected));
    }

    #[test]
    fn radial_intensity_decays_away_from_the_light() {
        // 1D horizontal setup (h = 1). Light at the right edge, a single bright
        // texel at index 3. Pixels 0,1,2 all march right through the bright
        // texel; the closer a pixel sits to the source along the ray, the lower
        // the sample index at which it crosses it, hence the higher decay
        // weight and the brighter the accumulated shaft.
        let mut mask = [0.0f32; 8];
        mask[3] = 1.0;
        let p = LightShaftParams::new([0.9375, 0.5], 16, 1.0, 0.9, 1.0, 1.0);
        let o0 = p.radial_blur_pixel(&mask, 8, 1, 0, 0);
        let o1 = p.radial_blur_pixel(&mask, 8, 1, 1, 0);
        let o2 = p.radial_blur_pixel(&mask, 8, 1, 2, 0);
        assert!(o2 > o1);
        assert!(o1 > o0);
        assert!(o0 > 0.0);
    }

    #[test]
    fn decay_zero_keeps_only_the_first_tap() {
        // decay = 0: illuminationDecay is 1 for the first tap then 0, so only
        // the first tap past the pixel contributes.
        let c = 0.5f32;
        let mask = [c; 16];
        let p = LightShaftParams::new([0.5, 0.5], 8, 1.0, 0.0, 1.0, 1.0);
        let expected = c + 1.0 * c; // base + first tap only
        assert!(approx(p.radial_blur_pixel(&mask, 4, 4, 0, 0), expected));
    }

    #[test]
    fn decay_one_accumulates_every_tap() {
        // decay = 1: every tap keeps full illumination, the maximal sum.
        let c = 0.5f32;
        let mask = [c; 16];
        let n = 8u32;
        let p = LightShaftParams::new([0.5, 0.5], n, 1.0, 1.0, 1.0, 1.0);
        let expected = c + 1.0 * c * (n as f32);
        assert!(approx(p.radial_blur_pixel(&mask, 4, 4, 0, 0), expected));
    }

    #[test]
    fn decay_one_exceeds_decay_zero() {
        let c = 0.5f32;
        let mask = [c; 16];
        let low = LightShaftParams::new([0.5, 0.5], 8, 1.0, 0.0, 1.0, 1.0);
        let high = LightShaftParams::new([0.5, 0.5], 8, 1.0, 1.0, 1.0, 1.0);
        let lo = low.radial_blur_pixel(&mask, 4, 4, 0, 0);
        let hi = high.radial_blur_pixel(&mask, 4, 4, 0, 0);
        assert!(hi > lo);
    }

    #[test]
    fn exposure_scales_the_output_linearly() {
        let c = 0.5f32;
        let mask = [c; 16];
        let base = LightShaftParams::new([0.5, 0.5], 6, 1.0, 0.9, 0.5, 1.0);
        let scaled = LightShaftParams::new([0.5, 0.5], 6, 1.0, 0.9, 0.5, 3.0);
        let b = base.radial_blur_pixel(&mask, 4, 4, 2, 2);
        let s = scaled.radial_blur_pixel(&mask, 4, 4, 2, 2);
        assert!(approx(s, b * 3.0));
    }

    #[test]
    fn weight_scales_the_accumulated_taps() {
        // With base value c and uniform mask: color = c + weight * c * geom.
        let c = 0.5f32;
        let mask = [c; 16];
        let geom = 1.0 + 0.5 + 0.25 + 0.125;
        let p = LightShaftParams::new([0.5, 0.5], 4, 1.0, 0.5, 2.0, 1.0);
        let expected = c + 2.0 * c * geom;
        assert!(approx(p.radial_blur_pixel(&mask, 4, 4, 0, 0), expected));
    }

    #[test]
    fn light_uv_outside_frame_is_clamped_and_finite() {
        let mask = [0.2f32; 16];
        let p = LightShaftParams::new([-1.5, 2.5], 8, 1.0, 0.9, 0.5, 1.0);
        let v = p.radial_blur_pixel(&mask, 4, 4, 0, 0);
        assert!(v.is_finite());
        assert!(v >= 0.0);
    }

    #[test]
    fn edge_pixels_do_not_panic_and_stay_finite() {
        let mask = [0.3f32; 16];
        let p = LightShaftParams::new([0.9, 0.9], 8, 2.0, 0.9, 0.5, 1.0);
        for &(px, py) in &[(0usize, 0usize), (3, 0), (0, 3), (3, 3)] {
            let v = p.radial_blur_pixel(&mask, 4, 4, px, py);
            assert!(v.is_finite());
        }
    }

    #[test]
    fn full_image_matches_pixelwise_calls() {
        let mask = [
            0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.2,
        ];
        let p = sample_params();
        let img = p.radial_blur(&mask, 4, 4);
        assert_eq!(img.len(), 16);
        for py in 0..4 {
            for px in 0..4 {
                let expected = p.radial_blur_pixel(&mask, 4, 4, px, py);
                assert!(approx(img[py * 4 + px], expected));
            }
        }
    }

    #[test]
    fn oversized_dimensions_return_empty_or_zero() {
        // mask shorter than w * h -> defined zero / empty, never a panic.
        let mask = [0.5f32; 3];
        let p = sample_params();
        assert!(approx(p.radial_blur_pixel(&mask, 4, 4, 0, 0), 0.0));
        assert!(p.radial_blur(&mask, 4, 4).is_empty());
    }

    #[test]
    fn out_of_range_pixel_is_zero() {
        let mask = [0.5f32; 16];
        let p = sample_params();
        assert!(approx(p.radial_blur_pixel(&mask, 4, 4, 4, 0), 0.0));
        assert!(approx(p.radial_blur_pixel(&mask, 4, 4, 0, 4), 0.0));
    }

    #[test]
    fn extreme_decay_above_one_stays_finite() {
        let mask = [0.25f32; 16];
        let p = LightShaftParams::new([0.5, 0.5], 8, 1.0, 2.0, 0.5, 1.0);
        let v = p.radial_blur_pixel(&mask, 4, 4, 1, 1);
        assert!(v.is_finite());
        // With decay > 1 the accumulation grows beyond the raw value.
        assert!(v > mask[5]);
    }
}
