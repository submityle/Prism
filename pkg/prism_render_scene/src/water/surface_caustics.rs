//! The water-surface pass's **projected-caustics** `@group(9)` plumbing: the
//! `CPU` half that finally *consumes* the light-space caustic-intensity texture
//! the [`super::super::water`] `water_caustics_project` compute kernel already
//! fills every frame.
//!
//! ## Why this slice exists (closing a dead-compute gap)
//!
//! The caustics kernel (twin of
//! [`prism_render_architecture::water::caustics`]) folds the refractive surface
//! `Jacobian` into a focused-sunlight intensity and projects it into a
//! `r32float` `texture_storage_2d` whose texels store the per-cell caustic gain.
//! Until this slice that texture was *written and never read*: the transparent
//! water surface composited screen-space refraction with no focused-light term,
//! so the caustic light patterns the kernel paints onto the floor never reached
//! the underwater view. This slice binds the caustic texture as a sampled `2D`
//! texture in the water fragment stage and *adds* `caustic_gain * strength *
//! water_tint` onto the already underwater-attenuated refraction, matching the
//! `Frostbite` / `Crest` focused-sunlight compositing path.
//!
//! The sibling [`super::surface_froxel`] slice closed the equivalent gap for the
//! underwater froxel volume (`@group(5)`); this slice adds the tenth bind group
//! on top of the `@group(0..=8)` geometry/light/`VSM`/`SSR`/motion/froxel/`GTAO`/
//! `SSGI`/world-`GI` chain.
//!
//! ## Caustic addressing
//!
//! The kernel parameterises the caustic grid over the same water-simulation
//! `UV` tile the surface mesh already carries (`VertexOutput.uv`). The fragment
//! therefore reads the texture at `uv * textureDimensions(caustic_tex)` with an
//! integer `textureLoad` (the `r32float` format is not filterable, so no sampler
//! is bound). A zero-strength body or a disabled feature flag clears the
//! `sample_enable` bit, which the shader reads as "no caustics" and skips, so
//! the surface keeps its raw refraction exactly as before.

use bevy_material::bind_group_layout_entries::{
    binding_types::{texture_2d, uniform_buffer_sized},
    BindGroupLayoutEntries,
};
use bevy_render::render_resource::{ShaderStages, TextureSampleType};
use bytemuck::{Pod, Zeroable};

/// Below this the caustic contribution is treated as off (no focused light), so
/// the shader keeps the raw refraction. A strength this small is already below
/// one `LSB` of any `8`-bit display and cannot brighten a pixel.
const EPS_STRENGTH: f32 = 1.0e-6;

/// `GPU`-side mirror of the shader's `WaterCausticsConfig` uniform (the water
/// `@group(9) @binding(1)` block).
///
/// Four scalars = 16 bytes, exactly one `WGSL` uniform-block alignment unit. The
/// shader multiplies the sampled caustic gain by [`Self::strength`] and gates
/// the whole lookup on [`Self::sample_enable`] so a disabled or zero-strength
/// body costs nothing.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWaterSurfaceCausticsConfig {
    /// Scalar gain the fragment multiplies onto the sampled caustic intensity
    /// before adding it to the underwater refraction. Kept at its authored
    /// value even when the lookup is gated off, so re-enabling needs no reupload.
    pub strength: f32,
    /// `1` when caustics are enabled and the strength is positive, `0` otherwise;
    /// `0` makes the shader keep the raw screen-space refraction.
    pub sample_enable: u32,
    /// Padding to the 16-byte block alignment; never read.
    pub _pad0: u32,
    /// Padding to the 16-byte block alignment; never read.
    pub _pad1: u32,
}

impl GpuWaterSurfaceCausticsConfig {
    /// Packs the caustic config for one body. `enable` is the caller's feature
    /// gate; the lookup is additionally forced off when `strength` is
    /// non-positive (so a zero-strength body never samples a texture it cannot
    /// brighten with).
    #[must_use]
    pub(crate) fn new(enable: bool, strength: f32) -> Self {
        let active = enable && strength > EPS_STRENGTH;
        Self {
            strength,
            sample_enable: u32::from(active),
            _pad0: 0,
            _pad1: 0,
        }
    }
}

/// Builds the water-surface `@group(9)` caustics layout entries (two bindings),
/// in the exact `@binding(n)` order `water_surface_raster.wesl` declares:
///
/// 0. the `r32float` light-space caustic-intensity texture as a sampled `2D`
///    texture (non-filterable float; the fragment reads it with an integer
///    `textureLoad`, so no sampler is bound), and
/// 1. the [`GpuWaterSurfaceCausticsConfig`] uniform.
///
/// Declared [`ShaderStages::FRAGMENT`]: only the lighting fork reads the caustic
/// texture, never the vertex stage.
pub(crate) fn caustics_layout_entries() -> BindGroupLayoutEntries<2> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::FRAGMENT,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            uniform_buffer_sized(false, None),
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn config_matches_the_wgsl_uniform_block_size_and_alignment() {
        // Four 4-byte scalars = 16 bytes, exactly one WGSL uniform-block unit.
        assert_eq!(size_of::<GpuWaterSurfaceCausticsConfig>(), 16);
        assert_eq!(align_of::<GpuWaterSurfaceCausticsConfig>(), 4);
    }

    #[test]
    fn new_packs_the_enable_bit_and_keeps_the_strength() {
        let on = GpuWaterSurfaceCausticsConfig::new(true, 2.5);
        assert_eq!(on.sample_enable, 1);
        assert!((on.strength - 2.5).abs() <= 1.0e-6);
        assert_eq!(on._pad0, 0);
        assert_eq!(on._pad1, 0);

        // A healthy strength the caller gates off still clears the bit but keeps
        // the authored value for a cheap re-enable.
        let gated = GpuWaterSurfaceCausticsConfig::new(false, 2.5);
        assert_eq!(gated.sample_enable, 0);
        assert!((gated.strength - 2.5).abs() <= 1.0e-6);
    }

    #[test]
    fn non_positive_strength_disables_the_lookup() {
        // Zero or negative strength cannot brighten a pixel, so the enable bit
        // clears even when the caller asks for caustics.
        let zero = GpuWaterSurfaceCausticsConfig::new(true, 0.0);
        assert_eq!(zero.sample_enable, 0);

        let negative = GpuWaterSurfaceCausticsConfig::new(true, -1.0);
        assert_eq!(negative.sample_enable, 0);
    }

    #[test]
    fn layout_declares_two_fragment_bindings() {
        let entries = caustics_layout_entries();
        assert_eq!(entries.len(), 2);
        for entry in entries.iter() {
            assert!(entry.visibility.contains(ShaderStages::FRAGMENT));
        }
    }
}
