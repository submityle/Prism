//! ABI shared between the ordered-dither compute pass and
//! `shaders/ordered_dither.wesl`.
//!
//! Like the other single-pass post effects, the dither pass carries one
//! immediate (push-constant) block, [`GpuOrderedDitherParams`], mirroring the
//! shader's single `var<immediate>` global. Every field mirrors the shader
//! struct byte-for-byte so machines with and without a GPU agree with the CPU
//! golden in [`prism_render_shading::ordered_dither`].
//!
//! The block leads with the `vec2<u32>` framebuffer extent (WGSL gives it an
//! 8-byte alignment) followed by the two scalar controls, so the whole struct is
//! a tightly packed 16 bytes with no implicit padding: `screen_size` at 0,
//! `levels` at 8 and `strength` at 12.

use bevy_math::UVec2;
use bytemuck::{Pod, Zeroable};

use super::settings::PrismOrderedDitherSettings;

/// Workgroup size (per axis) of the ordered-dither compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `ordered_dither.wesl`; the dispatch
/// rounds its target extent up to a multiple of this on both axes and the
/// shader bounds-checks every invocation.
pub(crate) const ORDERED_DITHER_WORKGROUP_SIZE: u32 = 8;

/// Immediate (push-constant) block consumed by the `ordered_dither_main` entry
/// point.
///
/// Mirrors the shader's `GpuOrderedDitherParams`: the framebuffer extent (which
/// also bounds-checks each invocation), the golden `levels` palette size and the
/// golden `strength` blend weight. The dispatch only runs when the host enabled
/// the pass, so the golden `enabled` flag is not carried — it is folded into the
/// [`PrismOrderedDitherSettings`] gate.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuOrderedDitherParams {
    /// Full-resolution framebuffer extent in texels (`vec2<u32>`); leads the
    /// block for its 8-byte alignment and bounds-checks each invocation.
    pub screen_size: [u32; 2],
    /// Number of discrete quantisation steps per channel (golden `levels`); the
    /// shader clamps it up to `2`.
    pub levels: u32,
    /// Blend weight of the dithered result over the input (golden `strength`,
    /// `a + (b - a) * t`); `0` is the identity.
    pub strength: f32,
}

impl GpuOrderedDitherParams {
    /// Builds the immediate block from the framebuffer extent and the live
    /// [`PrismOrderedDitherSettings`].
    ///
    /// The artist controls are carried through verbatim so the on-device dither
    /// is the exact golden twin (per pixel, with `enabled = true` implied by the
    /// pass having dispatched at all).
    pub(crate) fn from_settings(size: UVec2, settings: &PrismOrderedDitherSettings) -> Self {
        Self {
            screen_size: [size.x, size.y],
            levels: settings.levels,
            strength: settings.strength,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_dither_params_is_the_16_byte_immediate_block() {
        // The `vec2<u32>` extent (8) + `levels` (4) + `strength` (4) fill 16
        // bytes with no implicit padding.
        assert_eq!(size_of::<GpuOrderedDitherParams>(), 16);
        assert_eq!(align_of::<GpuOrderedDitherParams>(), 4);
    }

    #[test]
    fn workgroup_constant_matches_the_shader() {
        assert_eq!(ORDERED_DITHER_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn from_settings_packs_the_extent_and_artist_controls() {
        let settings = PrismOrderedDitherSettings::default();
        let params = GpuOrderedDitherParams::from_settings(UVec2::new(1920, 1080), &settings);
        assert_eq!(params.screen_size, [1920, 1080]);
        assert_eq!(params.levels, settings.levels);
        assert_eq!(params.strength, settings.strength);
    }
}
