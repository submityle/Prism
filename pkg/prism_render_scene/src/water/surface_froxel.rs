//! The water-surface pass's **underwater froxel-volume** `@group(5)` plumbing:
//! the `CPU` half that finally *consumes* the single-scatter + transmittance
//! `3D` froxel volume the [`super::super::water`] `water_underwater_volume`
//! compute kernel already fills every frame.
//!
//! ## Why this slice exists (closing a dead-compute gap)
//!
//! The underwater kernel (twin of
//! [`prism_render_architecture::water::underwater`]) integrates `Beer-Lambert`
//! extinction, the `RGB` blue-green colour shift, `Henyey-Greenstein`
//! phase-weighted single scatter, a multiple-scatter boost, and the god-ray
//! in-scatter into a `rgba16float` `texture_storage_3d` whose texels store
//! `vec4(inscatter.rgb, avg_transmittance)` per froxel. Until this slice that
//! volume was *written and never read*: the transparent water surface composited
//! raw screen-space refraction with no participating-medium term. This slice
//! binds the froxel volume as a sampled `3D` texture in the water fragment stage
//! and composites `refracted * transmittance + inscatter`, matching the
//! `Frostbite` froxel-volume transparent-compositing path.
//!
//! The sibling [`super::surface_pipeline`] slice owns the raster pipelines and
//! the earlier `@group(0..=4)` layouts (geometry/light/`VSM`/`SSR`/motion); this
//! slice adds the sixth bind group.
//!
//! ## Froxel addressing
//!
//! The kernel lays the volume out as *screen columns* on the `xy` axes and
//! *eye-to-slice depth* on the `z` axis: froxel `z` slice centre sits at
//! `depth = (z + 0.5) * slice_thickness` metres from the eye. The water
//! fragment therefore samples the volume at `(screen_uv, eye_depth / far)` where
//! `far = slice_thickness * depth_slices` is the furthest slice centre's reach;
//! [`froxel_inv_far`] precomputes `1 / far` on the host so the shader only
//! multiplies. A zero-thickness or zero-slice volume yields `inv_far = 0`, which
//! the shader reads as "no volume" and skips (the `sample_enable` bit also
//! clears), so the surface keeps its raw refraction exactly as before.

use bevy_material::bind_group_layout_entries::{
    binding_types::{sampler, texture_3d, uniform_buffer_sized},
    BindGroupLayoutEntries,
};
use bevy_render::render_resource::{SamplerBindingType, ShaderStages, TextureSampleType};
use bytemuck::{Pod, Zeroable};

/// Below this the froxel volume is treated as degenerate (no participating
/// medium), so the shader keeps the raw refraction. One micrometre of total
/// reach is already far beyond any meaningful water column.
const EPS_FAR: f32 = 1.0e-6;

/// `GPU`-side mirror of the shader's `WaterFroxelParams` uniform (the water
/// `@group(5) @binding(2)` block).
///
/// Four scalars = 16 bytes, exactly one `WGSL` uniform-block alignment unit. The
/// shader multiplies the fragment's eye-space depth by [`Self::inv_far`] to land
/// on the froxel `z` axis, and gates the whole lookup on [`Self::sample_enable`]
/// so a degenerate volume (zero thickness or zero slices) costs nothing.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWaterFroxelParams {
    /// `1 / (slice_thickness * depth_slices)`: scales eye-space depth onto the
    /// froxel `z` axis in `0..=1`. Zero when the volume is degenerate.
    pub inv_far: f32,
    /// `1` when a non-degenerate froxel volume is bound and should be sampled;
    /// `0` makes the shader keep the raw screen-space refraction.
    pub sample_enable: u32,
    /// Padding to the 16-byte block alignment; never read.
    pub _pad0: u32,
    /// Padding to the 16-byte block alignment; never read.
    pub _pad1: u32,
}

/// Reciprocal of the froxel volume's furthest reach, `1 / (slice_thickness *
/// depth_slices)`, or `0` when either factor is non-positive.
///
/// Pure and monotonic: a thicker slice or more slices lengthen the reach and
/// shrink the reciprocal, mapping a fixed eye depth onto a shallower froxel `z`.
#[expect(
    clippy::cast_precision_loss,
    reason = "depth_slices is a froxel grid slice count, far below 2^24, so the \
              widening to f32 is exact."
)]
pub(crate) fn froxel_inv_far(slice_thickness: f32, depth_slices: u32) -> f32 {
    let far = slice_thickness.max(0.0) * depth_slices as f32;
    if far > EPS_FAR {
        1.0 / far
    } else {
        0.0
    }
}

impl GpuWaterFroxelParams {
    /// Packs the froxel mapping for one body. `enable` is the caller's feature
    /// gate; the volume is additionally forced off when it is degenerate (so a
    /// zero-thickness or zero-slice body never samples a meaningless volume).
    #[must_use]
    pub(crate) fn new(slice_thickness: f32, depth_slices: u32, enable: bool) -> Self {
        let inv_far = froxel_inv_far(slice_thickness, depth_slices);
        let active = enable && inv_far > 0.0;
        Self {
            inv_far,
            sample_enable: u32::from(active),
            _pad0: 0,
            _pad1: 0,
        }
    }
}

/// Builds the water-surface `@group(5)` underwater-volume layout entries (three
/// bindings), in the exact `@binding(n)` order `water_surface_raster.wesl`
/// declares:
///
/// 0. the `rgba16float` single-scatter + transmittance froxel volume as a
///    sampled `3D` texture (filterable float; the fragment samples it with
///    `textureSampleLevel`),
/// 1. the filtering sampler the volume lookup uses, and
/// 2. the [`GpuWaterFroxelParams`] uniform.
///
/// Declared [`ShaderStages::FRAGMENT`]: only the lighting fork reads the volume,
/// never the vertex stage.
pub(crate) fn froxel_layout_entries() -> BindGroupLayoutEntries<3> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::FRAGMENT,
        (
            texture_3d(TextureSampleType::Float { filterable: true }),
            sampler(SamplerBindingType::Filtering),
            uniform_buffer_sized(false, None),
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn params_match_the_wgsl_uniform_block_size_and_alignment() {
        // Four 4-byte scalars = 16 bytes, exactly one WGSL uniform-block unit.
        assert_eq!(size_of::<GpuWaterFroxelParams>(), 16);
        assert_eq!(align_of::<GpuWaterFroxelParams>(), 4);
    }

    #[test]
    fn inv_far_is_the_reciprocal_of_the_total_reach() {
        // reach = 0.5 * 64 = 32 metres -> inv_far = 1/32.
        let inv = froxel_inv_far(0.5, 64);
        assert!((inv - 1.0 / 32.0).abs() <= 1.0e-6);
    }

    #[test]
    fn inv_far_is_monotonic_in_both_factors() {
        // A longer reach (thicker slice or more slices) shrinks the reciprocal.
        let base = froxel_inv_far(0.5, 32);
        assert!(froxel_inv_far(1.0, 32) < base);
        assert!(froxel_inv_far(0.5, 64) < base);
    }

    #[test]
    fn degenerate_volumes_disable_the_lookup() {
        // Zero thickness, zero slices, or negative thickness all collapse the
        // reach, so inv_far is zero and the enable bit clears even when asked.
        assert!(froxel_inv_far(0.0, 64).abs() <= 1.0e-6);
        assert!(froxel_inv_far(0.5, 0).abs() <= 1.0e-6);
        assert!(froxel_inv_far(-1.0, 64).abs() <= 1.0e-6);

        let off = GpuWaterFroxelParams::new(0.0, 64, true);
        assert_eq!(off.sample_enable, 0);
        assert!(off.inv_far.abs() <= 1.0e-6);
    }

    #[test]
    fn new_packs_the_enable_bit_and_clears_padding() {
        let on = GpuWaterFroxelParams::new(0.5, 64, true);
        assert_eq!(on.sample_enable, 1);
        assert!((on.inv_far - 1.0 / 32.0).abs() <= 1.0e-6);
        assert_eq!(on._pad0, 0);
        assert_eq!(on._pad1, 0);

        // A healthy volume the caller gates off still clears the bit.
        let gated = GpuWaterFroxelParams::new(0.5, 64, false);
        assert_eq!(gated.sample_enable, 0);
    }

    #[test]
    fn layout_declares_three_fragment_bindings() {
        let entries = froxel_layout_entries();
        assert_eq!(entries.len(), 3);
        for entry in entries.iter() {
            assert!(entry.visibility.contains(ShaderStages::FRAGMENT));
        }
    }
}
