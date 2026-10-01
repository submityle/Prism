//! The water-surface pass's **screen-space global-illumination** `@group(7)`
//! plumbing: the `CPU` half that lets the §5 image-based ambient term pick up a
//! single bounce of *near-field* indirect diffuse, gathered from the already
//! composited opaque `scene_color` along the surface's `GTAO` bent normal.
//!
//! ## Why the water gathers its own one-bounce `SSGI`
//!
//! The image-based ambient term ([`super::surface_shading`]'s `water_ibl`) only
//! carries *distant* environment irradiance (`SH` probe) — it is blind to the
//! light bouncing off the nearby lake bed, pier pilings, or cliff a metre away.
//! `UE5` `Lumen`'s screen traces supply exactly that missing first bounce. This
//! slice mirrors the cheap screen-space half of it: the water fragment reuses
//! the bent normal that the [`super::surface_gtao`] horizon search already
//! produced (the mean unoccluded direction), marches it across the shared
//! reverse-Z Hi-Z pyramid bound at `@group(3)`, and reads the nearby surface's
//! radiance out of the `@group(0)` `scene_color` grab. No ray tracer, no extra
//! prepass, and no texture of its own — the radiance source and the depth
//! pyramid are both already bound for the refraction and `SSR`/`GTAO` slices.
//!
//! ## Why only a uniform (no fallback texture)
//!
//! Like [`super::surface_gtao`], this slice adds **no** texture: the gather
//! reads `scene_color` (`@group(0) @binding(5)`, always resident) and the
//! `@group(3)` Hi-Z pyramid (its own fallback already covers a view with no
//! resident [`ViewSsrTextures`](crate::shading::ViewSsrTextures)). The draw node
//! clears this config's `sample_enable` bit whenever `SSGI` is disabled or no
//! pyramid is resident, so the shader skips the gather and adds nothing. A
//! uniform buffer is always bindable, so no `RenderStartup` fallback resource is
//! required.
//!
//! The sibling [`super::surface_pipeline`] slice owns the raster pipelines; this
//! slice adds the eighth (`@group(7)`) bind group, built once per view by
//! [`super::surface_draw`].

use bevy_material::bind_group_layout_entries::{
    binding_types::uniform_buffer_sized, BindGroupLayoutEntries,
};
use bevy_render::render_resource::ShaderStages;
use bytemuck::{Pod, Zeroable};

/// `GPU`-side mirror of the shader's `WaterSsgiConfig` uniform (the water
/// `@group(7) @binding(0)` block).
///
/// Layout matches the `WGSL` struct byte-for-byte: four `f32` tunables precede
/// two `u32` controls and two trailing `u32` pads, so the block is a 32-byte
/// multiple of the 16-byte `WGSL` uniform alignment. The march basis
/// (projection / world->view matrices, near plane) is **not** duplicated here —
/// the shader reads it from the `@group(3)` [`super::surface_ssr::GpuWaterSsrConfig`]
/// uniform, since the `SSGI` gather shares the `SSR`/`GTAO` reverse-Z
/// reconstruction and projection.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWaterSsgiConfig {
    /// Indirect-diffuse gain applied to the gathered bounce before it is added
    /// to the ambient term. `0` disables the contribution without touching the
    /// `sample_enable` gate; `1` applies the physical Lambert bounce at full
    /// strength (the shader folds in the `1/PI` normalisation itself).
    pub intensity: f32,
    /// View-space gather radius (metres): screen samples farther than this from
    /// the shaded point contribute nothing. Mirrors the opaque
    /// [`PrismShadingSettings::ssgi_max_distance`](crate::shading::PrismShadingSettings).
    pub radius: f32,
    /// Firefly clamp ceiling: a gathered radiance whose Rec.709 luminance
    /// exceeds this is rescaled down to it (hue preserved), so one bright screen
    /// texel cannot smear a sparkle across the surface.
    pub max_luminance: f32,
    /// Minimum `bent_normal · sample_dir` cosine for a screen sample to count:
    /// rejects samples below the surface's bent horizon, keeping the gather on
    /// the unoccluded hemisphere.
    pub min_cosine: f32,
    /// Screen samples marched along the bent-normal direction. Mirrors the
    /// opaque [`PrismShadingSettings::ssgi_sample_count`](crate::shading::PrismShadingSettings),
    /// clamped `>= 1` by the shader.
    pub step_count: u32,
    /// `1` runs the gather; `0` skips it and adds no indirect diffuse. The draw
    /// node clears it when `SSGI` is off or no Hi-Z pyramid is resident this
    /// frame.
    pub sample_enable: u32,
    /// Padding to the 16-byte uniform block alignment; never read.
    pub _pad0: u32,
    /// Padding to the 16-byte uniform block alignment; never read.
    pub _pad1: u32,
}

/// Golden `SSGI` tunable defaults for a view that does not override them: a
/// unit bounce gain, a moderate firefly ceiling, and a small horizon cosine
/// cutoff. The radius and step count come from the shared
/// [`PrismShadingSettings`](crate::shading::PrismShadingSettings) so the water
/// gather tracks the opaque `SSGI` pass's reach and budget.
const DEFAULT_INTENSITY: f32 = 1.0;
const DEFAULT_MAX_LUMINANCE: f32 = 4.0;
const DEFAULT_MIN_COSINE: f32 = 0.05;
/// Lower bound forced onto the gather radius so a zero/garbage setting cannot
/// collapse the march to a point (the shader would then gather nothing).
const MIN_RADIUS: f32 = 1.0e-3;

impl GpuWaterSsgiConfig {
    /// Build the uniform from the golden `SSGI` defaults with the gather radius
    /// and step budget taken from the shared shading settings, packing the
    /// enable bit into [`Self::sample_enable`].
    ///
    /// `radius` is floored at [`MIN_RADIUS`] and `step_count` at `1` so a
    /// disabled or garbage setting still yields a structurally valid gather the
    /// shader can early-out of via the enable bit.
    pub(crate) fn new(sample_enable: bool, radius: f32, step_count: u32) -> Self {
        Self {
            intensity: DEFAULT_INTENSITY,
            radius: radius.max(MIN_RADIUS),
            max_luminance: DEFAULT_MAX_LUMINANCE,
            min_cosine: DEFAULT_MIN_COSINE,
            step_count: step_count.max(1),
            sample_enable: u32::from(sample_enable),
            _pad0: 0,
            _pad1: 0,
        }
    }
}

/// Builds the water-surface `@group(7)` `SSGI` layout entries (one binding): the
/// [`GpuWaterSsgiConfig`] uniform. Declared [`ShaderStages::FRAGMENT`]: the
/// water surface runs the one-bounce gather in its fragment stage, reusing the
/// `@group(0)` `scene_color` grab and the `@group(3)` Hi-Z pyramid, so it needs
/// no texture or sampler of its own.
pub(crate) fn ssgi_layout_entries() -> BindGroupLayoutEntries<1> {
    BindGroupLayoutEntries::sequential(ShaderStages::FRAGMENT, (uniform_buffer_sized(false, None),))
}

/// Rec.709 (sRGB) relative luminance of a linear-RGB colour. A `CPU` twin of
/// the shader's `ssgi_luminance`, matching the golden
/// [`prism_render_shading::gi::denoise::firefly::luminance`] primaries so the
/// firefly clamp agrees between the two by construction.
#[cfg(test)]
fn rec709_luminance(rgb: [f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

/// Karis anti-firefly accumulation weight `w = 1 / (1 + luma)`, a `CPU` twin of
/// the shader's `ssgi_karis_weight` and of the golden
/// [`prism_render_shading::gi::denoise::firefly::karis_weight`]. Monotonically
/// decreasing in luminance (bright outliers contribute proportionally less);
/// negative luminances clamp to `0` so the weight stays in `(0, 1]`.
#[cfg(test)]
fn karis_weight(luma: f32) -> f32 {
    1.0 / (1.0 + luma.max(0.0))
}

/// Luminance firefly clamp: a `CPU` twin of the shader's `ssgi_clamp_firefly`.
/// A colour whose Rec.709 luminance exceeds `max_luminance` is rescaled
/// uniformly across `RGB` down to that ceiling (chromaticity preserved); dim or
/// degenerate colours pass through unchanged. A non-positive ceiling disables
/// the clamp.
#[cfg(test)]
fn clamp_firefly(rgb: [f32; 3], max_luminance: f32) -> [f32; 3] {
    if max_luminance <= 0.0 {
        return rgb;
    }
    let luma = rec709_luminance(rgb);
    if luma > max_luminance && luma > 1.0e-6 {
        let scale = max_luminance / luma;
        return [rgb[0] * scale, rgb[1] * scale, rgb[2] * scale];
    }
    rgb
}

/// Physical one-bounce indirect diffuse: a `CPU` twin of the shader's final
/// `radiance * albedo * (intensity / PI)` combine. Each channel multiplies the
/// gathered incident radiance by the surface albedo and the Lambert `1/PI`
/// normalisation, scaled by the configured gain, and is clamped non-negative so
/// a stray negative input cannot subtract light.
#[cfg(test)]
fn indirect_diffuse(radiance: [f32; 3], albedo: [f32; 3], intensity: f32) -> [f32; 3] {
    let gain = intensity.max(0.0) * core::f32::consts::FRAC_1_PI;
    let chan = |r: f32, a: f32| (r * a.max(0.0) * gain).max(0.0);
    [
        chan(radiance[0], albedo[0]),
        chan(radiance[1], albedo[1]),
        chan(radiance[2], albedo[2]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    const EPS: f32 = 1.0e-6;

    #[test]
    fn config_matches_the_wgsl_uniform_block_size_and_alignment() {
        // Four `f32` (16 B) + four `u32` (16 B) = 32 B, a multiple of the
        // 16-byte `WGSL` uniform block alignment.
        assert_eq!(size_of::<GpuWaterSsgiConfig>(), 32);
        assert_eq!(align_of::<GpuWaterSsgiConfig>(), 4);
    }

    #[test]
    fn new_packs_the_enable_bit_radius_and_step_budget() {
        let enabled = GpuWaterSsgiConfig::new(true, 6.0, 12);
        assert_eq!(enabled.sample_enable, 1);
        assert_eq!(enabled.step_count, 12);
        assert_eq!(enabled._pad0, 0);
        assert_eq!(enabled._pad1, 0);
        assert!((enabled.radius - 6.0).abs() < EPS);
        assert!((enabled.intensity - DEFAULT_INTENSITY).abs() < EPS);
        assert!((enabled.max_luminance - DEFAULT_MAX_LUMINANCE).abs() < EPS);
        assert!((enabled.min_cosine - DEFAULT_MIN_COSINE).abs() < EPS);

        let disabled = GpuWaterSsgiConfig::new(false, 6.0, 12);
        assert_eq!(disabled.sample_enable, 0);
    }

    #[test]
    fn new_floors_a_garbage_radius_and_step_count() {
        let cfg = GpuWaterSsgiConfig::new(true, 0.0, 0);
        assert!(cfg.radius >= MIN_RADIUS);
        assert_eq!(cfg.step_count, 1);
    }

    #[test]
    fn layout_declares_one_fragment_binding() {
        let entries = ssgi_layout_entries();
        assert_eq!(entries.len(), 1);
        for entry in entries.iter() {
            assert!(entry.visibility.contains(ShaderStages::FRAGMENT));
        }
    }

    #[test]
    fn karis_weight_is_one_at_black_and_decreases_with_luminance() {
        assert!((karis_weight(0.0) - 1.0).abs() < EPS);
        assert!(karis_weight(1.0) < karis_weight(0.5));
        assert!(karis_weight(100.0) < karis_weight(1.0));
        // Negative luminance clamps to the black weight rather than exploding.
        assert!((karis_weight(-5.0) - 1.0).abs() < EPS);
        // Always a valid convex weight.
        for &l in &[0.0_f32, 0.3, 2.0, 50.0] {
            let w = karis_weight(l);
            assert!(w > 0.0 && w <= 1.0 + EPS);
        }
    }

    #[test]
    fn clamp_firefly_rescales_only_bright_colours_and_preserves_hue() {
        // A dim colour passes through untouched.
        let dim = [0.1, 0.2, 0.3];
        assert_eq!(clamp_firefly(dim, 4.0), dim);
        // A bright colour is pulled down to the ceiling with its ratios intact.
        let bright = [8.0, 4.0, 2.0];
        let clamped = clamp_firefly(bright, 2.0);
        let lum = rec709_luminance(clamped);
        assert!((lum - 2.0).abs() < 1.0e-4, "lum = {lum}");
        // Hue (channel ratios) is preserved by the uniform rescale.
        assert!((clamped[0] / clamped[2] - bright[0] / bright[2]).abs() < 1.0e-4);
        // A disabled ceiling leaves the colour alone.
        assert_eq!(clamp_firefly(bright, 0.0), bright);
    }

    #[test]
    fn indirect_diffuse_is_lambert_scaled_and_non_negative() {
        let radiance = [2.0, 1.0, 0.5];
        let albedo = [0.5, 0.5, 0.5];
        let out = indirect_diffuse(radiance, albedo, 1.0);
        let inv_pi = core::f32::consts::FRAC_1_PI;
        assert!((out[0] - 2.0 * 0.5 * inv_pi).abs() < EPS);
        assert!((out[1] - 1.0 * 0.5 * inv_pi).abs() < EPS);
        // Zero gain kills the contribution.
        assert_eq!(indirect_diffuse(radiance, albedo, 0.0), [0.0, 0.0, 0.0]);
        // Negative inputs cannot subtract light.
        let neg = indirect_diffuse([-5.0, 1.0, 1.0], [1.0, -1.0, 1.0], 1.0);
        assert!(neg[0].abs() < EPS);
        assert!(neg[1].abs() < EPS);
        assert!(neg[2] > 0.0);
    }

    #[test]
    fn indirect_diffuse_scales_linearly_with_gain() {
        let radiance = [1.0, 1.0, 1.0];
        let albedo = [0.8, 0.8, 0.8];
        let a = indirect_diffuse(radiance, albedo, 1.0);
        let b = indirect_diffuse(radiance, albedo, 2.0);
        assert!((b[0] - 2.0 * a[0]).abs() < EPS);
    }
}
