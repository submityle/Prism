//! ABI shared between the halftone compute pass and `shaders/halftone.wesl`.
//!
//! Like the other single-pass post effects, the halftone pass carries one
//! immediate (push-constant) block, [`GpuHalftoneParams`], mirroring the
//! shader's single `var<immediate>` global. Every field mirrors the shader
//! struct byte-for-byte so machines with and without a GPU agree with the CPU
//! golden in [`prism_render_shading::halftone`].
//!
//! The golden [`prism_render_shading::HalftoneParams`] carries two `f32`
//! tunables (`cell_size`, `angle`) plus the enable flag. WGSL gives `vec2<u32>`
//! an 8-byte alignment, so the framebuffer extent leads the block and forces the
//! whole struct to an 8-byte multiple. Layout: the extent at 0, then
//! `cell_size`/`angle`, the `enabled` flag (as `u32`) and an explicit tail pad:
//! `8 + 4 + 4 + 4 + 4 = 24` bytes, a multiple of 8 with no implicit padding,
//! matching the WGSL struct the `vec2<u32>` aligns.

use bevy_math::UVec2;
use bytemuck::{Pod, Zeroable};

use super::settings::PrismHalftoneSettings;

/// Workgroup size (per axis) of the halftone compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `halftone.wesl`; the dispatch rounds
/// its target extent up to a multiple of this on both axes and the shader
/// bounds-checks every invocation.
pub(crate) const HALFTONE_WORKGROUP_SIZE: u32 = 8;

/// Immediate (push-constant) block consumed by the `halftone_main` entry point.
///
/// Mirrors the shader's `GpuHalftoneParams`: the framebuffer extent, the golden
/// `cell_size` and lattice `angle`, and the master `enabled` flag (as `u32`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuHalftoneParams {
    /// Full-resolution framebuffer extent in texels (`vec2<u32>`); leads the
    /// block for its 8-byte alignment.
    pub screen_size: [u32; 2],
    /// Side length of a screen cell, in pixels.
    pub cell_size: f32,
    /// Rotation of the dot lattice, in radians.
    pub angle: f32,
    /// Master enable; `0` passes the scene pixel through (identity).
    pub enabled: u32,
    /// Explicit tail padding so the block is a multiple of 8 bytes, matching the
    /// WGSL struct alignment the leading `vec2<u32>` forces.
    pub _pad: u32,
}

impl GpuHalftoneParams {
    /// Builds the immediate block from the framebuffer extent and the live
    /// [`PrismHalftoneSettings`].
    pub(crate) fn from_settings(size: UVec2, settings: &PrismHalftoneSettings) -> Self {
        Self {
            screen_size: [size.x, size.y],
            cell_size: settings.cell_size,
            angle: settings.angle,
            enabled: u32::from(settings.enabled),
            _pad: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halftone_params_is_the_24_byte_immediate_block() {
        // The `vec2<u32>` extent (8) plus the two `f32` tunables (8), the `u32`
        // enable (4) and the `u32` pad (4) fill 24 bytes, a multiple of 8 with
        // no implicit padding.
        assert_eq!(size_of::<GpuHalftoneParams>(), 24);
        assert_eq!(align_of::<GpuHalftoneParams>(), 4);
    }

    #[test]
    fn workgroup_constant_matches_the_shader() {
        assert_eq!(HALFTONE_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn from_settings_packs_the_extent_and_controls() {
        let settings = PrismHalftoneSettings::default();
        let params = GpuHalftoneParams::from_settings(UVec2::new(1920, 1080), &settings);
        assert_eq!(params.screen_size, [1920, 1080]);
        assert_eq!(params.cell_size, settings.cell_size);
        assert_eq!(params.angle, settings.angle);
        // Default is disabled -> the flag is 0.
        assert_eq!(params.enabled, 0);
    }

    #[test]
    fn from_settings_maps_the_enable_flag() {
        let settings = PrismHalftoneSettings {
            enabled: true,
            ..Default::default()
        };
        let params = GpuHalftoneParams::from_settings(UVec2::new(8, 8), &settings);
        assert_eq!(params.enabled, 1);
    }
}
