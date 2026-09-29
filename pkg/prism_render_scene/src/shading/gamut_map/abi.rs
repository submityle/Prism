//! ABI shared between the gamut-map compute pass and `shaders/gamut_map.wesl`.
//!
//! Like the other single-pass post effects, the gamut-map pass carries one
//! immediate (push-constant) block, [`GpuGamutMapParams`], mirroring the
//! shader's single `var<immediate>` global. Every field mirrors the shader
//! struct byte-for-byte so machines with and without a GPU agree with the CPU
//! golden in [`prism_render_shading::gamut_map`].
//!
//! WGSL gives `vec4` a 16-byte alignment, so the golden `GamutMapParams`
//! per-channel `vec3`s are padded up to `vec4` (carrying the scalar `power` and
//! the device-only `scale` in their free `w` lane) rather than left as bare
//! `vec3`s (which would force implicit tail padding). Layout: two `vec4`s at
//! offsets 0/16, then the `vec2<u32>` extent at 32 and an explicit `vec2<u32>`
//! pad at 40: 2 * 16 + 8 + 8 = 48 bytes, a multiple of 16 with no implicit
//! padding.

use bevy_math::UVec2;
use bytemuck::{Pod, Zeroable};

use super::settings::PrismGamutMapSettings;

/// Workgroup size (per axis) of the gamut-map compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `gamut_map.wesl`; the dispatch
/// rounds its target extent up to a multiple of this on both axes and the
/// shader bounds-checks every invocation.
pub(crate) const GAMUT_MAP_WORKGROUP_SIZE: u32 = 8;

/// Immediate (push-constant) block consumed by the `gamut_map_main` entry point.
///
/// Mirrors the shader's `GpuGamutMapParams`: the golden `GamutMapParams`
/// per-channel `threshold`/`limit` and the reserved `power` exponent packed into
/// `vec4` lanes, plus the framebuffer extent and a device-only global `scale` in
/// `[0, 1]` that fades the whole map toward identity (`scale = 1` reproduces the
/// golden compression exactly).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuGamutMapParams {
    /// `xyz` = per-channel distance `threshold` (the working gamut), `w` = the
    /// reserved perceptual `power` exponent. `vec4` leads the block for its
    /// 16-byte alignment.
    pub threshold_power: [f32; 4],
    /// `xyz` = per-channel asymptotic `limit` (the gamut-boundary target),
    /// `w` = the device-only global `scale`.
    pub limit_scale: [f32; 4],
    /// Full-resolution framebuffer extent in texels (`vec2<u32>`).
    pub screen_size: [u32; 2],
    /// Explicit tail padding so the block is a multiple of 16 bytes, matching
    /// the shader struct's `vec2<u32>` `_pad`.
    pub _pad: [u32; 2],
}

impl GpuGamutMapParams {
    /// Builds the immediate block from the framebuffer extent and the live
    /// [`PrismGamutMapSettings`].
    ///
    /// The `scale` is clamped to `[0, 1]` on device; the artist controls are
    /// carried through verbatim so the on-device compression is the exact golden
    /// twin (`scale = 1`).
    pub(crate) fn from_settings(size: UVec2, settings: &PrismGamutMapSettings) -> Self {
        Self {
            threshold_power: [
                settings.threshold[0],
                settings.threshold[1],
                settings.threshold[2],
                settings.power,
            ],
            limit_scale: [
                settings.limit[0],
                settings.limit[1],
                settings.limit[2],
                settings.scale,
            ],
            screen_size: [size.x, size.y],
            _pad: [0, 0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamut_map_params_is_the_48_byte_immediate_block() {
        // Two `vec4`s (32) + the `vec2<u32>` extent (8) + the `vec2<u32>` pad (8)
        // fill 48 bytes, a multiple of 16 with no implicit padding.
        assert_eq!(size_of::<GpuGamutMapParams>(), 48);
        assert_eq!(align_of::<GpuGamutMapParams>(), 4);
    }

    #[test]
    fn workgroup_constant_matches_the_shader() {
        assert_eq!(GAMUT_MAP_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn from_settings_packs_the_extent_and_artist_controls() {
        let settings = PrismGamutMapSettings::default();
        let params = GpuGamutMapParams::from_settings(UVec2::new(1920, 1080), &settings);
        assert_eq!(params.screen_size, [1920, 1080]);
        // The per-channel wheels and their packed scalar lanes round-trip.
        assert_eq!(params.threshold_power, [
            settings.threshold[0],
            settings.threshold[1],
            settings.threshold[2],
            settings.power,
        ]);
        assert_eq!(params.limit_scale, [
            settings.limit[0],
            settings.limit[1],
            settings.limit[2],
            settings.scale,
        ]);
        assert_eq!(params._pad, [0, 0]);
    }
}
