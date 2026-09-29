//! ABI shared between the `CAS` compute pass and `shaders/cas.wesl`.
//!
//! Like the other single-pass post effects, the sharpen pass carries one
//! immediate (push-constant) block, [`GpuCasParams`], mirroring the shader's
//! single `var<immediate>` global. Every field mirrors the shader struct
//! byte-for-byte so machines with and without a GPU agree with the CPU golden
//! in [`prism_render_shading::cas`].
//!
//! The `vec2<u32>` framebuffer extent leads the block for its 8-byte alignment,
//! then the two `f32` artist scalars (`strength`, `sharpness`): 8 + 4 + 4 = 16
//! bytes, a multiple of 16 with no implicit padding.

use bevy_math::UVec2;
use bytemuck::{Pod, Zeroable};

use super::settings::PrismCasSettings;

/// Workgroup size (per axis) of the `CAS` compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `cas.wesl`; the dispatch rounds its
/// target extent up to a multiple of this on both axes and the shader
/// bounds-checks every invocation.
pub(crate) const CAS_WORKGROUP_SIZE: u32 = 8;

/// Immediate (push-constant) block consumed by the `cas_main` entry point.
///
/// Mirrors the shader's `GpuCasParams`: the framebuffer extent and the two
/// golden `CasParams` controls (the `strength` blend of the sharpened result
/// over the centre and the `sharpness` adaptive-peak selector). `strength = 0`
/// reproduces the input exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuCasParams {
    /// Full-resolution framebuffer extent in texels (`vec2<u32>`). Leads the
    /// block for its 8-byte alignment.
    pub screen_size: [u32; 2],
    /// Blend of the sharpened result over the original centre, `[0, 1]`.
    pub strength: f32,
    /// Adaptive sharpening peak selector, `[0, 1]`.
    pub sharpness: f32,
}

impl GpuCasParams {
    /// Builds the immediate block from the framebuffer extent and the live
    /// [`PrismCasSettings`]. The artist controls are carried through verbatim
    /// so the on-device sharpen is the exact golden twin.
    pub(crate) fn from_settings(size: UVec2, settings: &PrismCasSettings) -> Self {
        Self {
            screen_size: [size.x, size.y],
            strength: settings.strength,
            sharpness: settings.sharpness,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cas_params_is_the_16_byte_immediate_block() {
        // The `vec2<u32>` extent (8) plus the two `f32` scalars (8) fill 16
        // bytes, a multiple of 16 with no implicit padding.
        assert_eq!(size_of::<GpuCasParams>(), 16);
        assert_eq!(align_of::<GpuCasParams>(), 4);
    }

    #[test]
    fn workgroup_constant_matches_the_shader() {
        assert_eq!(CAS_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn from_settings_packs_the_extent_and_artist_controls() {
        let settings = PrismCasSettings {
            enabled: true,
            strength: 0.6,
            sharpness: 0.4,
        };
        let params = GpuCasParams::from_settings(UVec2::new(1920, 1080), &settings);
        assert_eq!(params.screen_size, [1920, 1080]);
        assert_eq!(params.strength, 0.6);
        assert_eq!(params.sharpness, 0.4);
    }
}
