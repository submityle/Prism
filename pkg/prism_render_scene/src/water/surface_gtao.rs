//! The water-surface pass's **ground-truth ambient occlusion** `@group(6)`
//! plumbing: the `CPU` half that lets the §5 image-based ambient term occlude
//! itself with a horizon-based `GTAO` march (Jimenez et al. 2016) run directly
//! inside the transparent water fragment stage.
//!
//! ## Why the water marches its own `AO` instead of sampling a resolved buffer
//!
//! The opaque `GTAO` prepass ([`crate::shading`], gated by
//! [`PrismShadingSettings::enable_gtao`](crate::shading::PrismShadingSettings))
//! resolves an occlusion term for every *opaque* pixel before the water surface
//! is drawn. For a water pixel that resolved value belongs to the submerged
//! terrain occupying the pixel, not to the water surface itself — sampling it on
//! the water would darken the surface with the lake bed's cavities, which is
//! physically wrong. Instead the water fragment reconstructs its own view-space
//! position and normal and runs the horizon search itself over the shared
//! reverse-Z Hi-Z "nearest depth" pyramid that the [`super::surface_ssr`] slice
//! already binds at `@group(3)`. This reuses the engine's existing depth
//! pyramid (no extra prepass) and mirrors the `CPU` golden
//! [`prism_render_shading::gi::gtao`] horizon/integral/multi-bounce references
//! that the device shader is a twin of.
//!
//! ## Why only a uniform (no fallback texture)
//!
//! Unlike [`super::surface_ssr::WaterSsrFallback`], this slice adds **no**
//! texture of its own: the horizon march reads the same `@group(3)` Hi-Z
//! pyramid, whose fallback already covers a view with no resident
//! [`ViewSsrTextures`](crate::shading::ViewSsrTextures). The draw node clears
//! this config's `sample_enable` bit whenever `GTAO` is disabled or no pyramid
//! is resident, so the shader skips the march and leaves the ambient term fully
//! lit (`AO = 1`). A uniform buffer is always bindable, so no `RenderStartup`
//! fallback resource is required.
//!
//! The sibling [`super::surface_pipeline`] slice owns the raster pipelines; this
//! slice adds the seventh (`@group(6)`) bind group, built once per view by
//! [`super::surface_draw`].

use bevy_material::bind_group_layout_entries::{
    binding_types::uniform_buffer_sized, BindGroupLayoutEntries,
};
use bevy_render::render_resource::ShaderStages;
use bytemuck::{Pod, Zeroable};

/// `GPU`-side mirror of the shader's `WaterGtaoConfig` uniform (the water
/// `@group(6) @binding(0)` block).
///
/// Layout matches the `WGSL` struct byte-for-byte: four `f32` tunables precede
/// three `u32`s and one trailing `u32` pad, so the block is a 32-byte multiple
/// of the 16-byte `WGSL` uniform alignment. The march basis (projection and
/// world->view matrices, near plane) is **not** duplicated here — the shader
/// reads it from the `@group(3)` [`super::surface_ssr::GpuWaterSsrConfig`]
/// uniform, since both the `SSR` and `GTAO` marches share the same reverse-Z
/// reconstruction.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWaterGtaoConfig {
    /// View-space falloff radius (metres): occluders at/beyond it stop
    /// contributing. Mirrors the golden
    /// [`gtao::horizon`](prism_render_shading::gi::gtao::horizon)
    /// `HorizonParams::falloff_radius`.
    pub radius: f32,
    /// Occlusion strength blend in `[0, 1]`: `0` keeps the surface fully lit,
    /// `1` applies the shaped `AO` at full strength (golden `power_intensity`).
    pub intensity: f32,
    /// Contrast power applied to the raw occlusion before the intensity blend
    /// (golden `power_intensity`); `> 1` deepens contact shadows.
    pub power: f32,
    /// Thin-surface recovery factor in `[0, 1]` (golden
    /// `HorizonParams::thickness`): `0` latches a strict running maximum, `1`
    /// lets the horizon fall back toward a receding sample.
    pub thickness: f32,
    /// Number of azimuthal slices the fragment integrates (golden `integrate`
    /// slice count); each covers both tangent sides.
    pub slice_count: u32,
    /// Horizon samples marched per slice side.
    pub step_count: u32,
    /// `1` runs the march; `0` skips it and leaves the ambient term fully lit
    /// (`AO = 1`). The draw node clears it when `GTAO` is off or no Hi-Z pyramid
    /// is resident this frame.
    pub sample_enable: u32,
    /// Padding to the 16-byte uniform block alignment; never read.
    pub _pad0: u32,
}

/// Golden `GTAO` tunable defaults, matching the `CPU`
/// [`prism_render_shading::gi::gtao`] reference's mid-range configuration:
/// a unit falloff radius and a moderate thickness relaxation, with a
/// two-slice / four-step screen march and the common squared contrast power.
const DEFAULT_RADIUS: f32 = 1.0;
const DEFAULT_INTENSITY: f32 = 1.0;
const DEFAULT_POWER: f32 = 2.0;
const DEFAULT_THICKNESS: f32 = 0.5;
const DEFAULT_SLICE_COUNT: u32 = 2;
const DEFAULT_STEP_COUNT: u32 = 4;

impl GpuWaterGtaoConfig {
    /// Build the uniform from the golden `GTAO` defaults with the given enable
    /// bit. The tunables adopt the shared
    /// [`prism_render_shading::gi::gtao`] reference's defaults
    /// (`HorizonParams::default()` falloff/thickness) so the water march agrees
    /// with the opaque `GTAO` golden.
    pub(crate) fn new(sample_enable: bool) -> Self {
        Self {
            radius: DEFAULT_RADIUS,
            intensity: DEFAULT_INTENSITY,
            power: DEFAULT_POWER,
            thickness: DEFAULT_THICKNESS,
            slice_count: DEFAULT_SLICE_COUNT,
            step_count: DEFAULT_STEP_COUNT,
            sample_enable: u32::from(sample_enable),
            _pad0: 0,
        }
    }
}

/// Builds the water-surface `@group(6)` `GTAO` layout entries (one binding): the
/// [`GpuWaterGtaoConfig`] uniform. Declared [`ShaderStages::FRAGMENT`]: the
/// water surface runs the horizon search in its fragment stage, reusing the
/// `@group(3)` Hi-Z pyramid, so it needs no texture or sampler of its own.
pub(crate) fn gtao_layout_entries() -> BindGroupLayoutEntries<1> {
    BindGroupLayoutEntries::sequential(ShaderStages::FRAGMENT, (uniform_buffer_sized(false, None),))
}

/// Quadratic distance falloff weight in `[0, 1]`: `1` at zero separation,
/// decaying to `0` at/after `radius`. A `CPU` twin of the golden
/// [`prism_render_shading::gi::gtao::horizon`] `falloff_weight` (private there),
/// documenting the exact `1 - (dist / radius)^2` attenuation the `@group(6)`
/// shader applies to each horizon candidate so the two agree by construction. A
/// disabled radius (`<= 0`) returns `1` (no attenuation), matching the golden.
#[cfg(test)]
fn falloff_weight(dist: f32, radius: f32) -> f32 {
    if radius <= 0.0 {
        return 1.0;
    }
    let t = dist / radius;
    (1.0 - t * t).clamp(0.0, 1.0)
}

/// Azimuth (radians, in `[0, PI)`) of slice `index` out of `count` evenly
/// spaced slices, each sampling the full `[0, PI)` half-circle (both tangent
/// sides are marched per slice, so the slices need only span a half turn). The
/// half-step offset `(index + 0.5)/count` keeps the first and last slices off
/// the degenerate screen axes. A zero `count` collapses to `0`.
#[cfg(test)]
fn slice_azimuth(index: u32, count: u32) -> f32 {
    if count == 0 {
        return 0.0;
    }
    let step = core::f32::consts::PI / count as f32;
    (index as f32 + 0.5) * step
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::{FRAC_PI_2, PI};
    use core::mem::{align_of, size_of};

    const EPS: f32 = 1.0e-6;

    #[test]
    fn config_matches_the_wgsl_uniform_block_size_and_alignment() {
        // Four `f32` (16 B) + four `u32` (16 B) = 32 B, a multiple of the
        // 16-byte `WGSL` uniform block alignment.
        assert_eq!(size_of::<GpuWaterGtaoConfig>(), 32);
        assert_eq!(align_of::<GpuWaterGtaoConfig>(), 4);
    }

    #[test]
    fn new_packs_the_enable_bit_and_golden_defaults() {
        let enabled = GpuWaterGtaoConfig::new(true);
        assert_eq!(enabled.sample_enable, 1);
        assert_eq!(enabled.slice_count, DEFAULT_SLICE_COUNT);
        assert_eq!(enabled.step_count, DEFAULT_STEP_COUNT);
        assert_eq!(enabled._pad0, 0);
        assert!((enabled.radius - DEFAULT_RADIUS).abs() < EPS);
        assert!((enabled.thickness - DEFAULT_THICKNESS).abs() < EPS);

        let disabled = GpuWaterGtaoConfig::new(false);
        assert_eq!(disabled.sample_enable, 0);
    }

    #[test]
    fn layout_declares_one_fragment_binding() {
        let entries = gtao_layout_entries();
        assert_eq!(entries.len(), 1);
        for entry in entries.iter() {
            assert!(entry.visibility.contains(ShaderStages::FRAGMENT));
        }
    }

    #[test]
    fn falloff_weight_is_one_at_origin_and_zero_past_radius() {
        assert!((falloff_weight(0.0, 2.0) - 1.0).abs() < EPS);
        // Past the radius the weight saturates at zero rather than going
        // negative.
        assert!(falloff_weight(3.0, 2.0).abs() < EPS);
        assert!(falloff_weight(10.0, 2.0).abs() < EPS);
        // A disabled radius disables attenuation.
        assert!((falloff_weight(5.0, 0.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn falloff_weight_decreases_monotonically_inside_the_radius() {
        let radius = 4.0;
        let mut prev = falloff_weight(0.0, radius);
        for step in 1..=8 {
            let dist = step as f32 * 0.5;
            let w = falloff_weight(dist, radius);
            assert!(w <= prev + EPS, "weight rose at dist {dist}");
            prev = w;
        }
    }

    #[test]
    fn slice_azimuth_spans_the_half_circle_without_touching_the_axes() {
        // Two slices land at PI/4 and 3*PI/4: off both screen axes, symmetric
        // about PI/2.
        let a0 = slice_azimuth(0, 2);
        let a1 = slice_azimuth(1, 2);
        assert!((a0 - PI / 4.0).abs() < EPS);
        assert!((a1 - 3.0 * PI / 4.0).abs() < EPS);
        assert!(a0 > 0.0 && a1 < PI);
        // Symmetric pair averages to the vertical axis.
        assert!((0.5 * (a0 + a1) - FRAC_PI_2).abs() < EPS);
        // Degenerate count collapses to zero.
        assert!(slice_azimuth(0, 0).abs() < EPS);
    }

    #[test]
    fn slice_azimuth_is_strictly_increasing_in_index() {
        let count = 5;
        let mut prev = -1.0_f32;
        for index in 0..count {
            let a = slice_azimuth(index, count);
            assert!(a > prev, "azimuth not increasing at index {index}");
            assert!(a > 0.0 && a < PI);
            prev = a;
        }
    }
}
