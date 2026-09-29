//! ABI shared between the Kuwahara compute pass and `shaders/kuwahara.wesl`.
//!
//! Like the other single-pass post effects, the Kuwahara pass carries one
//! immediate (push-constant) block, [`GpuKuwaharaParams`], mirroring the
//! shader's single `var<immediate>` global. Every field mirrors the shader
//! struct byte-for-byte so machines with and without a GPU agree with the CPU
//! golden in [`prism_render_shading::kuwahara`].
//!
//! The golden [`prism_render_shading::KuwaharaParams`] carries only two scalars
//! (`radius`, `enabled`), so no `vec3`/`vec4` packing is needed: the `vec2<u32>`
//! framebuffer extent leads for its 8-byte alignment, then the `radius` and the
//! `enabled` flag (stored as `u32` for uniform layout). Layout: the extent at
//! offset 0, `radius` at 8 and `enabled` at 12 — `2 * 4 + 2 * 4 = 16` bytes, a
//! multiple of 16 with no implicit padding.

use bevy_math::UVec2;
use bytemuck::{Pod, Zeroable};

use super::settings::PrismKuwaharaSettings;

/// Workgroup size (per axis) of the Kuwahara compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `kuwahara.wesl`; the dispatch rounds
/// its target extent up to a multiple of this on both axes and the shader
/// bounds-checks every invocation.
pub(crate) const KUWAHARA_WORKGROUP_SIZE: u32 = 8;

/// Immediate (push-constant) block consumed by the `kuwahara_main` entry point.
///
/// Mirrors the shader's `GpuKuwaharaParams`: the framebuffer extent, the golden
/// neighbourhood `radius` and the master `enabled` flag (as `u32`). The four
/// overlapping quadrants are each `(radius + 1) x (radius + 1)`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuKuwaharaParams {
    /// Full-resolution framebuffer extent in texels (`vec2<u32>`); leads the
    /// block for its 8-byte alignment.
    pub screen_size: [u32; 2],
    /// Neighbourhood radius `r`; each quadrant is `(r + 1) x (r + 1)`.
    pub radius: u32,
    /// Master enable; `0` passes the centre pixel through (identity).
    pub enabled: u32,
}

impl GpuKuwaharaParams {
    /// Builds the immediate block from the framebuffer extent and the live
    /// [`PrismKuwaharaSettings`].
    pub(crate) fn from_settings(size: UVec2, settings: &PrismKuwaharaSettings) -> Self {
        Self {
            screen_size: [size.x, size.y],
            radius: settings.radius,
            enabled: u32::from(settings.enabled),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kuwahara_params_is_the_16_byte_immediate_block() {
        // The `vec2<u32>` extent (8) plus the two `u32` scalars (8) fill 16
        // bytes, a multiple of 16 with no implicit padding.
        assert_eq!(size_of::<GpuKuwaharaParams>(), 16);
        assert_eq!(align_of::<GpuKuwaharaParams>(), 4);
    }

    #[test]
    fn workgroup_constant_matches_the_shader() {
        assert_eq!(KUWAHARA_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn from_settings_packs_the_extent_and_controls() {
        let settings = PrismKuwaharaSettings::default();
        let params = GpuKuwaharaParams::from_settings(UVec2::new(1920, 1080), &settings);
        assert_eq!(params.screen_size, [1920, 1080]);
        assert_eq!(params.radius, settings.radius);
        // Default is disabled -> the flag is 0.
        assert_eq!(params.enabled, 0);
    }

    #[test]
    fn from_settings_maps_the_enable_flag() {
        let settings = PrismKuwaharaSettings {
            enabled: true,
            ..PrismKuwaharaSettings::default()
        };
        let params = GpuKuwaharaParams::from_settings(UVec2::new(8, 8), &settings);
        assert_eq!(params.enabled, 1);
    }
}
