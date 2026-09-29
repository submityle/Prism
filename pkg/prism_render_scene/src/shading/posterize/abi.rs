//! ABI shared between the posterize compute pass and
//! `shaders/posterize.wesl`.
//!
//! Like the other single-pass post effects, the posterize pass carries one
//! immediate (push-constant) block, [`GpuPosterizeParams`], mirroring the
//! shader's single `var<immediate>` global. Every field mirrors the shader
//! struct byte-for-byte so machines with and without a GPU agree with the CPU
//! golden in [`prism_render_shading::posterize`].
//!
//! Layout: the `vec2<u32>` framebuffer extent leads the block for its 8-byte
//! alignment, then the three `f32` controls (`levels`, `luma_levels`,
//! `strength`), the two `u32` flags (`use_luma`, `enabled`) and a `u32` tail
//! pad: 8 + 12 + 8 + 4 = 32 bytes, a multiple of 16 with no implicit padding.

use bevy_math::UVec2;
use bytemuck::{Pod, Zeroable};

use super::settings::PrismPosterizeSettings;

/// Workgroup size (per axis) of the posterize compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `posterize.wesl`; the dispatch
/// rounds its target extent up to a multiple of this on both axes and the
/// shader bounds-checks every invocation.
pub(crate) const POSTERIZE_WORKGROUP_SIZE: u32 = 8;

/// Immediate (push-constant) block consumed by the `posterize_main` entry
/// point.
///
/// Mirrors the shader's `GpuPosterizeParams`: the framebuffer extent plus the
/// golden `PosterizeParams` controls (the per-channel `RGB` band count
/// `levels`, the luma-preserving band count `luma_levels`, the `strength`
/// blend, and the `use_luma` / `enabled` flags carried as `u32`s). `enabled = 0`
/// reproduces the input exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuPosterizeParams {
    /// Full-resolution framebuffer extent in texels (`vec2<u32>`). Leads the
    /// block for its 8-byte alignment.
    pub screen_size: [u32; 2],
    /// Band count for the per-channel `RGB` quantizer.
    pub levels: f32,
    /// Band count for the luma-preserving quantizer.
    pub luma_levels: f32,
    /// Blend of the quantized result over the input, `a + (b - a) * strength`.
    pub strength: f32,
    /// Select the luma-preserving quantizer (`1`) over per-channel `RGB` (`0`).
    pub use_luma: u32,
    /// Master enable (`0` forces a hard identity regardless of the controls).
    pub enabled: u32,
    /// Explicit tail padding so the block is a multiple of 16 bytes, matching
    /// the shader struct's `u32` `_pad`.
    pub _pad: u32,
}

impl GpuPosterizeParams {
    /// Builds the immediate block from the framebuffer extent and the live
    /// [`PrismPosterizeSettings`]. The artist controls are carried through
    /// verbatim so the on-device posterize is the exact golden twin.
    pub(crate) fn from_settings(size: UVec2, settings: &PrismPosterizeSettings) -> Self {
        Self {
            screen_size: [size.x, size.y],
            levels: settings.levels,
            luma_levels: settings.luma_levels,
            strength: settings.strength,
            use_luma: u32::from(settings.use_luma),
            enabled: u32::from(settings.enabled),
            _pad: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posterize_params_is_the_32_byte_immediate_block() {
        // The `vec2<u32>` extent (8) + three `f32` (12) + two `u32` flags (8) +
        // the `u32` pad (4) fill 32 bytes, a multiple of 16 with no implicit
        // padding.
        assert_eq!(size_of::<GpuPosterizeParams>(), 32);
        assert_eq!(align_of::<GpuPosterizeParams>(), 4);
    }

    #[test]
    fn workgroup_constant_matches_the_shader() {
        assert_eq!(POSTERIZE_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn from_settings_packs_the_extent_and_artist_controls() {
        let settings = PrismPosterizeSettings {
            enabled: true,
            levels: 5.0,
            luma_levels: 3.0,
            use_luma: true,
            strength: 0.75,
        };
        let params = GpuPosterizeParams::from_settings(UVec2::new(1920, 1080), &settings);
        assert_eq!(params.screen_size, [1920, 1080]);
        assert_eq!(params.levels, 5.0);
        assert_eq!(params.luma_levels, 3.0);
        assert_eq!(params.strength, 0.75);
        assert_eq!(params.use_luma, 1);
        assert_eq!(params.enabled, 1);
    }
}
