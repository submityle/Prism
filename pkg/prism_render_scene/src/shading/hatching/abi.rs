//! ABI shared between the cross-hatching compute pass and
//! `shaders/hatching.wesl`.
//!
//! Like the other single-pass post effects, the hatching pass carries one
//! immediate (push-constant) block, [`GpuHatchingParams`], mirroring the
//! shader's single `var<immediate>` global. Every field mirrors the shader
//! struct byte-for-byte so machines with and without a GPU agree with the CPU
//! golden in [`prism_render_shading::hatching`].
//!
//! The golden [`prism_render_shading::HatchingParams`] carries eight `f32`
//! tunables plus the enable flag. WGSL gives `vec2<u32>` an 8-byte alignment, so
//! the framebuffer extent leads the block and forces the whole struct to an
//! 8-byte multiple. Layout: the extent at 0, then `frequency`/`thickness`, the
//! three stroke `angles`, the three `thresholds`, the `enabled` flag (as `u32`)
//! and an explicit tail pad: `8 + 8 * 4 + 4 + 4 = 48` bytes, a multiple of 8
//! with no implicit padding, matching the WGSL struct the `vec2<u32>` aligns.

use bevy_math::UVec2;
use bytemuck::{Pod, Zeroable};

use super::settings::PrismHatchingSettings;

/// Workgroup size (per axis) of the cross-hatching compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `hatching.wesl`; the dispatch rounds
/// its target extent up to a multiple of this on both axes and the shader
/// bounds-checks every invocation.
pub(crate) const HATCHING_WORKGROUP_SIZE: u32 = 8;

/// Immediate (push-constant) block consumed by the `hatching_main` entry point.
///
/// Mirrors the shader's `GpuHatchingParams`: the framebuffer extent, the golden
/// stripe `frequency` and stroke `thickness`, the three stroke `angles`, the
/// three luminance `thresholds` and the master `enabled` flag (as `u32`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuHatchingParams {
    /// Full-resolution framebuffer extent in texels (`vec2<u32>`); leads the
    /// block for its 8-byte alignment.
    pub screen_size: [u32; 2],
    /// Stripe frequency (periods across the normalized screen).
    pub frequency: f32,
    /// Stroke half-width in fractional-period units.
    pub thickness: f32,
    /// First stroke direction in radians (added first, brightest tier).
    pub angle0: f32,
    /// Second stroke direction in radians.
    pub angle1: f32,
    /// Third stroke direction in radians (added last, darkest tier).
    pub angle2: f32,
    /// Brightest luminance tier at which the first stroke set switches on.
    pub threshold0: f32,
    /// Middle luminance tier.
    pub threshold1: f32,
    /// Darkest luminance tier.
    pub threshold2: f32,
    /// Master enable; `0` passes the scene pixel through (identity).
    pub enabled: u32,
    /// Explicit tail padding so the block is a multiple of 8 bytes, matching the
    /// WGSL struct alignment the leading `vec2<u32>` forces.
    pub _pad: u32,
}

impl GpuHatchingParams {
    /// Builds the immediate block from the framebuffer extent and the live
    /// [`PrismHatchingSettings`].
    pub(crate) fn from_settings(size: UVec2, settings: &PrismHatchingSettings) -> Self {
        Self {
            screen_size: [size.x, size.y],
            frequency: settings.frequency,
            thickness: settings.thickness,
            angle0: settings.angles[0],
            angle1: settings.angles[1],
            angle2: settings.angles[2],
            threshold0: settings.thresholds[0],
            threshold1: settings.thresholds[1],
            threshold2: settings.thresholds[2],
            enabled: u32::from(settings.enabled),
            _pad: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hatching_params_is_the_48_byte_immediate_block() {
        // The `vec2<u32>` extent (8) plus the eight `f32` tunables (32), the
        // `u32` enable (4) and the `u32` pad (4) fill 48 bytes, a multiple of 8
        // with no implicit padding.
        assert_eq!(size_of::<GpuHatchingParams>(), 48);
        assert_eq!(align_of::<GpuHatchingParams>(), 4);
    }

    #[test]
    fn workgroup_constant_matches_the_shader() {
        assert_eq!(HATCHING_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn from_settings_packs_the_extent_and_controls() {
        let settings = PrismHatchingSettings::default();
        let params = GpuHatchingParams::from_settings(UVec2::new(1920, 1080), &settings);
        assert_eq!(params.screen_size, [1920, 1080]);
        assert_eq!(params.frequency, settings.frequency);
        assert_eq!(params.thickness, settings.thickness);
        assert_eq!(
            [params.angle0, params.angle1, params.angle2],
            settings.angles
        );
        assert_eq!(
            [params.threshold0, params.threshold1, params.threshold2],
            settings.thresholds
        );
        // Default is disabled -> the flag is 0.
        assert_eq!(params.enabled, 0);
    }

    #[test]
    fn from_settings_maps_the_enable_flag() {
        let settings = PrismHatchingSettings {
            enabled: true,
            ..PrismHatchingSettings::default()
        };
        let params = GpuHatchingParams::from_settings(UVec2::new(8, 8), &settings);
        assert_eq!(params.enabled, 1);
    }
}
