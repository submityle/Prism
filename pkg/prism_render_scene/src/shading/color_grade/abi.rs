//! ABI shared between the colour-grade compute pass and
//! `shaders/color_grade.wesl`.
//!
//! Like the other single-pass post effects, the grade pass carries one
//! immediate (push-constant) block, [`GpuColorGradeParams`], mirroring the
//! shader's single `var<immediate>` global. Every field mirrors the shader
//! struct byte-for-byte so machines with and without a GPU agree with the CPU
//! golden in [`prism_render_shading::color_grade`].
//!
//! WGSL gives `vec4` a 16-byte alignment, so the golden `ColorGradeParams`
//! scalars are packed into the free `w`/`z` lanes of four `vec4`s rather than
//! trailing them (which would force `vec3` padding). Layout: four `vec4`s at
//! offsets 0/16/32/48 (the three ASC CDL wheels carry `temperature`/`tint`/
//! `contrast` in their `w` lane, the fourth carries `pivot`/`saturation`/`scale`),
//! then the `vec2<u32>` extent at 64 and an explicit `vec2<u32>` pad at 72:
//! 4 * 16 + 8 + 8 = 80 bytes, a multiple of 16 with no implicit padding.

use bevy_math::UVec2;
use bytemuck::{Pod, Zeroable};

use super::settings::PrismColorGradeSettings;

/// Workgroup size (per axis) of the colour-grade compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `color_grade.wesl`; the dispatch
/// rounds its target extent up to a multiple of this on both axes and the
/// shader bounds-checks every invocation.
pub(crate) const COLOR_GRADE_WORKGROUP_SIZE: u32 = 8;

/// Immediate (push-constant) block consumed by the `color_grade_main` entry
/// point.
///
/// Mirrors the shader's `GpuColorGradeParams`: the golden `ColorGradeParams`
/// controls (the von Kries white balance `temperature`/`tint`, the ASC CDL
/// `lift`/`gamma`/`gain` wheels, the `contrast` about `pivot` and the
/// luma-preserving `saturation`) packed into `vec4` lanes, plus the framebuffer
/// extent and a device-only global `scale` in `[0, 1]` that fades the whole
/// grade toward identity (`scale = 1` reproduces the golden grade exactly).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuColorGradeParams {
    /// `xyz` = ASC CDL lift (per-channel offset), `w` = white-balance
    /// temperature. `vec4` leads the block for its 16-byte alignment.
    pub lift_temp: [f32; 4],
    /// `xyz` = ASC CDL gamma (per-channel power), `w` = white-balance tint.
    pub gamma_tint: [f32; 4],
    /// `xyz` = ASC CDL gain (per-channel slope), `w` = contrast.
    pub gain_contrast: [f32; 4],
    /// `x` = contrast pivot, `y` = saturation, `z` = global scale, `w` = pad.
    pub pivot_sat_scale: [f32; 4],
    /// Full-resolution framebuffer extent in texels (`vec2<u32>`).
    pub screen_size: [u32; 2],
    /// Explicit tail padding so the block is a multiple of 16 bytes, matching
    /// the shader struct's `vec2<u32>` `_pad`.
    pub _pad: [u32; 2],
}

impl GpuColorGradeParams {
    /// Builds the immediate block from the framebuffer extent and the live
    /// [`PrismColorGradeSettings`].
    ///
    /// The `scale` is clamped to `[0, 1]` on device; the artist controls are
    /// carried through verbatim so the on-device grade is the exact golden twin
    /// (`scale = 1`).
    pub(crate) fn from_settings(size: UVec2, settings: &PrismColorGradeSettings) -> Self {
        Self {
            lift_temp: [
                settings.lift[0],
                settings.lift[1],
                settings.lift[2],
                settings.temperature,
            ],
            gamma_tint: [
                settings.gamma[0],
                settings.gamma[1],
                settings.gamma[2],
                settings.tint,
            ],
            gain_contrast: [
                settings.gain[0],
                settings.gain[1],
                settings.gain[2],
                settings.contrast,
            ],
            pivot_sat_scale: [settings.pivot, settings.saturation, settings.scale, 0.0],
            screen_size: [size.x, size.y],
            _pad: [0, 0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_grade_params_is_the_80_byte_immediate_block() {
        // Four `vec4`s (64) + the `vec2<u32>` extent (8) + the `vec2<u32>` pad
        // (8) fill 80 bytes, a multiple of 16 with no implicit padding.
        assert_eq!(size_of::<GpuColorGradeParams>(), 80);
        assert_eq!(align_of::<GpuColorGradeParams>(), 4);
    }

    #[test]
    fn workgroup_constant_matches_the_shader() {
        assert_eq!(COLOR_GRADE_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn from_settings_packs_the_extent_and_artist_controls() {
        let settings = PrismColorGradeSettings::default();
        let params = GpuColorGradeParams::from_settings(UVec2::new(1920, 1080), &settings);
        assert_eq!(params.screen_size, [1920, 1080]);
        // The CDL wheels and their packed scalar lanes round-trip.
        assert_eq!(params.lift_temp, [
            settings.lift[0],
            settings.lift[1],
            settings.lift[2],
            settings.temperature,
        ]);
        assert_eq!(params.gamma_tint, [
            settings.gamma[0],
            settings.gamma[1],
            settings.gamma[2],
            settings.tint,
        ]);
        assert_eq!(params.gain_contrast, [
            settings.gain[0],
            settings.gain[1],
            settings.gain[2],
            settings.contrast,
        ]);
        assert_eq!(
            params.pivot_sat_scale,
            [settings.pivot, settings.saturation, settings.scale, 0.0]
        );
    }
}
