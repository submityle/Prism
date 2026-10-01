//! Mitchell 2007 radial-blur occlusion scattering (CPU golden reference).
//!
//! Implements the post-process volumetric light-scattering march from Kenneth
//! Mitchell, *"Volumetric Light Scattering as a Post-Process"* (GPU Gems 3,
//! chapter 13).  Starting from a screen-space pixel the march steps toward the
//! screen-space light position, sampling an occlusion/emission mask (built by
//! [`super::occlusion`]) at each step and accumulating an exponentially decaying
//! weighted sum:
//!
//! ```text
//! delta        = (uv - light_uv) * (1 / N) * density
//! illum        = mask(uv)                       // centre tap, full weight
//! decay_i      = 1
//! for i in 0..N:
//!     uv      -= delta
//!     illum   += mask(uv) * weight * decay_i
//!     decay_i *= decay
//! result       = illum * exposure
//! ```
//!
//! The mask encodes how much light reaches each screen texel (`1` = light
//! visible / sky, `0` = occluded by geometry), so marching toward the light and
//! summing the still-visible energy reconstructs crepuscular rays ("god rays")
//! cheaply in screen space.  The march returns a scalar scatter intensity which
//! is tinted by the light colour to produce the additive shaft contribution.
//!
//! # Conventions
//! * Screen coordinates are [`Vec2`] in normalized `[0, 1]` texel space, with
//!   the light position `light_uv` in the same space.
//! * Colours are linear-RGB [`Vec3`]; the scalar march is colour-agnostic and is
//!   tinted afterward by [`radial_scatter_color`].
//! * `density`, `weight`, and `exposure` are clamped non-negative and finite;
//!   `decay` is clamped to `[0, 1]`; the sample count is clamped to
//!   `[1, MAX_SAMPLES]`.
//! * The scatter is monotonically non-decreasing in the mask: more occlusion
//!   (smaller mask values) never brightens the shaft.
//! * Non-finite sampler returns are treated as `0`; the accumulator is clamped
//!   non-negative so no `NaN`/`inf` ever escapes.
//! * Transcendental maths (none needed here) would go through
//!   [`bevy_math::ops`]; every function is a deterministic pure function with no
//!   RNG, I/O, GPU, or `unsafe`.

use bevy_math::{Vec2, Vec3};

/// Upper bound on the number of march steps.
///
/// Keeps the inner loop finite and the cost bounded even when a caller passes a
/// pathological sample count.
pub const MAX_SAMPLES: u32 = 256;

/// Tunable parameters of the Mitchell radial-scatter march.
///
/// `density` scales the march step length (how far along the pixel->light ray a
/// full march of `sample_count` steps travels, as a fraction of that distance);
/// `weight` scales every marched tap; `decay` attenuates successive taps
/// geometrically; `exposure` scales the final accumulated intensity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadialScatterParams {
    /// March-step scale; `1.0` marches the full pixel->light distance.
    pub density: f32,
    /// Per-tap weight applied to every marched mask sample.
    pub weight: f32,
    /// Geometric decay in `[0, 1]` applied once per step.
    pub decay: f32,
    /// Final output scale applied to the accumulated intensity.
    pub exposure: f32,
}

impl Default for RadialScatterParams {
    #[inline]
    fn default() -> Self {
        // Values in the spirit of the GPU Gems 3 reference listing.
        Self {
            density: 0.9,
            weight: 0.6,
            decay: 0.95,
            exposure: 0.2,
        }
    }
}

impl RadialScatterParams {
    /// Returns a copy with every field sanitized to its valid, finite range.
    #[inline]
    pub fn sanitized(self) -> Self {
        Self {
            density: clamp_non_negative(self.density),
            weight: clamp_non_negative(self.weight),
            decay: clamp_unit(self.decay),
            exposure: clamp_non_negative(self.exposure),
        }
    }
}

/// Marches the Mitchell radial scatter and returns a scalar intensity.
///
/// `uv` is the originating screen texel and `light_uv` the screen-space light
/// position (both normalized).  `sample_count` is clamped to `[1, MAX_SAMPLES]`.
/// `mask` returns the per-texel light visibility in `[0, 1]`; non-finite returns
/// are treated as `0`.  The returned intensity is non-negative and finite.
///
/// The accumulation sums non-negative, positively weighted samples of `mask`,
/// so the result is monotonically non-decreasing in the mask: increasing
/// occlusion (smaller mask values) can only darken the shaft.
pub fn radial_scatter<F>(
    uv: Vec2,
    light_uv: Vec2,
    sample_count: u32,
    params: RadialScatterParams,
    mask: F,
) -> f32
where
    F: Fn(Vec2) -> f32,
{
    let p = params.sanitized();
    let n = clamp_sample_count(sample_count);

    if !is_finite_vec2(uv) || !is_finite_vec2(light_uv) {
        return 0.0;
    }

    // Per-step vector from the pixel toward the light.  When the pixel coincides
    // with the light the delta is zero and every tap re-samples the same texel,
    // which is the correct degenerate behaviour (a bright core, no streak).
    let delta = (uv - light_uv) * (p.density / n as f32);

    // Centre tap at full weight, matching the GPU Gems 3 listing.
    let mut illum = sample_mask(&mask, uv);
    let mut decay_i = 1.0_f32;
    let mut cursor = uv;

    for _ in 0..n {
        cursor -= delta;
        let s = sample_mask(&mask, cursor);
        illum += s * p.weight * decay_i;
        decay_i *= p.decay;
    }

    clamp_non_negative(illum * p.exposure)
}

/// Tints [`radial_scatter`] by the light colour, yielding the additive shaft
/// contribution in linear-RGB.
///
/// The scalar march is computed once and multiplied by the sanitized
/// `light_color`; every channel is finite and non-negative.
pub fn radial_scatter_color<F>(
    uv: Vec2,
    light_uv: Vec2,
    sample_count: u32,
    params: RadialScatterParams,
    light_color: Vec3,
    mask: F,
) -> Vec3
where
    F: Fn(Vec2) -> f32,
{
    let intensity = radial_scatter(uv, light_uv, sample_count, params, mask);
    sanitize_color(light_color) * intensity
}

/// Edge-clamped bilinear sample of a row-major scalar mask in `[0, 1]` texel
/// space.
///
/// `uv` addresses the buffer with `(0, 0)` at the first texel centre and
/// `(1, 1)` at the last; samples outside the grid clamp to the edge.  An empty
/// or degenerate grid returns `0`.  The result is finite and clamped to the
/// per-texel value range.
pub fn sample_mask_bilinear(mask: &[f32], width: usize, height: usize, uv: Vec2) -> f32 {
    if width == 0 || height == 0 || mask.len() < width * height {
        return 0.0;
    }
    let u = if uv.x.is_finite() { uv.x } else { 0.0 };
    let v = if uv.y.is_finite() { uv.y } else { 0.0 };

    // Map normalized uv to texel-centre coordinates.
    let fx = (u * width as f32 - 0.5).clamp(0.0, (width - 1) as f32);
    let fy = (v * height as f32 - 0.5).clamp(0.0, (height - 1) as f32);

    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;

    let s00 = fetch(mask, width, x0, y0);
    let s10 = fetch(mask, width, x1, y0);
    let s01 = fetch(mask, width, x0, y1);
    let s11 = fetch(mask, width, x1, y1);

    let top = s00 + (s10 - s00) * tx;
    let bot = s01 + (s11 - s01) * tx;
    top + (bot - top) * ty
}

/// Fetches a single mask texel, repairing non-finite storage to `0`.
#[inline]
fn fetch(mask: &[f32], width: usize, x: usize, y: usize) -> f32 {
    let v = mask[y * width + x];
    if v.is_finite() { v } else { 0.0 }
}

/// Samples the user mask and sanitizes the return to a finite, non-negative
/// value so a misbehaving sampler cannot poison the accumulator.
#[inline]
fn sample_mask<F>(mask: &F, uv: Vec2) -> f32
where
    F: Fn(Vec2) -> f32,
{
    let v = mask(uv);
    if v.is_finite() { v.max(0.0) } else { 0.0 }
}

/// Clamps the sample count to `[1, MAX_SAMPLES]`.
#[inline]
fn clamp_sample_count(n: u32) -> u32 {
    n.clamp(1, MAX_SAMPLES)
}

/// Clamps a scalar to be finite and non-negative (non-finite -> `0`).
#[inline]
fn clamp_non_negative(v: f32) -> f32 {
    if v.is_finite() { v.max(0.0) } else { 0.0 }
}

/// Clamps a scalar to the unit interval `[0, 1]` (non-finite -> `0`).
#[inline]
fn clamp_unit(v: f32) -> f32 {
    if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }
}

/// Returns `true` when both components of `v` are finite.
#[inline]
fn is_finite_vec2(v: Vec2) -> bool {
    v.x.is_finite() && v.y.is_finite()
}

/// Sanitizes a colour to be finite and non-negative per channel.
#[inline]
fn sanitize_color(c: Vec3) -> Vec3 {
    Vec3::new(clamp_non_negative(c.x), clamp_non_negative(c.y), clamp_non_negative(c.z))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A constant mask makes the march a closed-form geometric series we can
    /// check exactly: `illum = c * (1 + weight * (1 + decay + ... + decay^(N-1)))`.
    fn expected_constant(c: f32, n: u32, p: RadialScatterParams) -> f32 {
        let mut series = 0.0_f32;
        let mut decay_i = 1.0_f32;
        for _ in 0..n {
            series += decay_i;
            decay_i *= p.decay;
        }
        (c + c * p.weight * series) * p.exposure
    }

    #[test]
    fn constant_mask_matches_closed_form() {
        let p = RadialScatterParams { density: 1.0, weight: 0.5, decay: 0.9, exposure: 0.7 };
        let got = radial_scatter(Vec2::new(0.8, 0.8), Vec2::new(0.5, 0.5), 32, p, |_| 0.4);
        let want = expected_constant(0.4, 32, p);
        assert!((got - want).abs() < 1e-4, "got={got} want={want}");
    }

    #[test]
    fn monotonic_non_decreasing_in_mask() {
        // Brighter (less occluded) masks must never darken the shaft.
        let p = RadialScatterParams::default();
        let light = Vec2::new(0.5, 0.1);
        let uv = Vec2::new(0.7, 0.9);
        let mut prev = -1.0;
        let mut c = 0.0;
        while c <= 1.0 + 1e-6 {
            let got = radial_scatter(uv, light, 48, p, |_| c);
            assert!(got >= prev - 1e-6, "non-monotonic at c={c}: {got} < {prev}");
            prev = got;
            c += 0.05;
        }
    }

    #[test]
    fn more_occlusion_darkens_shaft() {
        // A mask with an occluding band returns strictly less than a fully lit
        // mask of the same geometry.
        let p = RadialScatterParams::default();
        let light = Vec2::new(0.5, 0.05);
        let uv = Vec2::new(0.5, 0.95);
        let lit = radial_scatter(uv, light, 64, p, |_| 1.0);
        let occluded = radial_scatter(uv, light, 64, p, |s| if s.y > 0.4 && s.y < 0.6 { 0.0 } else { 1.0 });
        assert!(occluded < lit, "occluded={occluded} lit={lit}");
    }

    #[test]
    fn higher_decay_accumulates_more() {
        // With a constant mask, a decay closer to 1 retains more of the tail and
        // therefore accumulates a brighter shaft.
        let base = RadialScatterParams { density: 1.0, weight: 0.8, decay: 0.5, exposure: 1.0 };
        let hi = RadialScatterParams { decay: 0.95, ..base };
        let lo = radial_scatter(Vec2::new(0.9, 0.9), Vec2::new(0.1, 0.1), 64, base, |_| 0.5);
        let hi = radial_scatter(Vec2::new(0.9, 0.9), Vec2::new(0.1, 0.1), 64, hi, |_| 0.5);
        assert!(hi > lo, "hi={hi} lo={lo}");
    }

    #[test]
    fn decay_zero_keeps_only_first_marched_tap() {
        // decay = 0 => decay_i is 1 for the first step then 0, so illum =
        // centre + weight * first_tap.
        let p = RadialScatterParams { density: 1.0, weight: 1.0, decay: 0.0, exposure: 1.0 };
        let got = radial_scatter(Vec2::new(0.5, 0.5), Vec2::new(0.5, 0.5), 10, p, |_| 0.3);
        assert!((got - (0.3 + 0.3)).abs() < 1e-5, "got={got}");
    }

    #[test]
    fn zero_distance_light_is_a_bright_core() {
        // When the pixel sits on the light, every tap re-samples the same point.
        let p = RadialScatterParams { density: 1.0, weight: 1.0, decay: 1.0, exposure: 1.0 };
        let got = radial_scatter(Vec2::new(0.5, 0.5), Vec2::new(0.5, 0.5), 8, p, |_| 1.0);
        // centre (1) + 8 taps * weight(1) * decay(1) = 9.
        assert!((got - 9.0).abs() < 1e-5, "got={got}");
    }

    #[test]
    fn non_finite_inputs_fall_back_to_zero() {
        let p = RadialScatterParams::default();
        assert_eq!(radial_scatter(Vec2::new(f32::NAN, 0.0), Vec2::ZERO, 16, p, |_| 1.0), 0.0);
        assert_eq!(radial_scatter(Vec2::ZERO, Vec2::new(f32::INFINITY, 0.0), 16, p, |_| 1.0), 0.0);
        // A NaN-producing sampler is sanitized to zero contribution.
        let got = radial_scatter(Vec2::new(0.6, 0.6), Vec2::new(0.5, 0.5), 16, p, |_| f32::NAN);
        assert_eq!(got, 0.0);
    }

    #[test]
    fn params_sanitize_out_of_range_fields() {
        let p = RadialScatterParams { density: -1.0, weight: f32::NAN, decay: 5.0, exposure: -3.0 }.sanitized();
        assert_eq!(p.density, 0.0);
        assert_eq!(p.weight, 0.0);
        assert_eq!(p.decay, 1.0);
        assert_eq!(p.exposure, 0.0);
    }

    #[test]
    fn sample_count_is_clamped() {
        assert_eq!(clamp_sample_count(0), 1);
        assert_eq!(clamp_sample_count(10_000), MAX_SAMPLES);
    }

    #[test]
    fn bilinear_sampler_edge_clamps_and_interpolates() {
        // 2x2 grid: [[0, 1], [2, 3]] row-major.
        let m = [0.0_f32, 1.0, 2.0, 3.0];
        // Centre of the grid averages all four corners -> 1.5.
        let mid = sample_mask_bilinear(&m, 2, 2, Vec2::new(0.5, 0.5));
        assert!((mid - 1.5).abs() < 1e-5, "mid={mid}");
        // Far outside clamps to the last texel.
        let far = sample_mask_bilinear(&m, 2, 2, Vec2::new(5.0, 5.0));
        assert!((far - 3.0).abs() < 1e-5, "far={far}");
        // Degenerate grids are safe.
        assert_eq!(sample_mask_bilinear(&[], 0, 0, Vec2::ZERO), 0.0);
    }

    #[test]
    fn color_tint_scales_scalar_intensity() {
        let p = RadialScatterParams { density: 1.0, weight: 0.5, decay: 0.9, exposure: 1.0 };
        let scalar = radial_scatter(Vec2::new(0.8, 0.8), Vec2::new(0.5, 0.5), 24, p, |_| 0.6);
        let tint = Vec3::new(1.0, 0.5, 0.25);
        let col = radial_scatter_color(Vec2::new(0.8, 0.8), Vec2::new(0.5, 0.5), 24, p, tint, |_| 0.6);
        assert!((col.x - scalar).abs() < 1e-5);
        assert!((col.y - scalar * 0.5).abs() < 1e-5);
        assert!((col.z - scalar * 0.25).abs() < 1e-5);
    }

    #[test]
    fn negative_light_color_is_sanitized() {
        let p = RadialScatterParams::default();
        let col = radial_scatter_color(Vec2::new(0.8, 0.8), Vec2::new(0.5, 0.5), 16, p, Vec3::new(-1.0, f32::NAN, 2.0), |_| 1.0);
        assert!(col.x >= 0.0 && col.y >= 0.0 && col.z >= 0.0);
        assert!(col.is_finite());
    }
}
