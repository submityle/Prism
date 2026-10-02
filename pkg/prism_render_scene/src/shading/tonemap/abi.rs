//! ABI shared between the tone-map compute pass and `shaders/tonemap.wesl`.
//!
//! Like the other single-pass post effects, the tone-map pass carries one
//! immediate (push-constant) block, [`GpuTonemapParams`], mirroring the shader's
//! single `var<immediate>` global. Every field mirrors the shader struct
//! byte-for-byte so machines with and without a GPU agree with the CPU golden in
//! [`prism_render_shading::tonemap`].
//!
//! The block is deliberately compact: the operator selector and the extended
//! Reinhard white point are the only tunables the golden `apply_tonemap`
//! consumes (the parameter-free curves and the neutral `AgX` look need nothing
//! more), plus the framebuffer extent for the per-invocation bounds check.
//! Layout: `u32` operator at 0, `f32` white point at 4, then the `vec2<u32>`
//! extent at 8: 16 bytes, a multiple of 16 with no implicit padding.

use bevy_math::UVec2;
use bytemuck::{Pod, Zeroable};

use super::settings::PrismTonemapSettings;

/// Workgroup size (per axis) of the tone-map compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `tonemap.wesl`; the dispatch rounds
/// its target extent up to a multiple of this on both axes and the shader
/// bounds-checks every invocation.
pub(crate) const TONEMAP_WORKGROUP_SIZE: u32 = 8;

/// Immediate (push-constant) block consumed by the `tonemap_main` entry point.
///
/// Mirrors the shader's `GpuTonemapParams`: the operator selector (the CPU
/// [`prism_render_shading::TonemapOperator`] discriminant, `0..=4`), the
/// extended-Reinhard white point and the framebuffer extent.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuTonemapParams {
    /// Operator selector: `0` Reinhard, `1` ReinhardExtended, `2` AcesNarkowicz,
    /// `3` AcesFitted, `4` AgX — matching the CPU enum declaration order and the
    /// shader's `apply_tonemap` dispatch codes.
    pub operator: u32,
    /// White point for the extended-Reinhard operator (radiance mapping to
    /// `1.0`); ignored by the other operators.
    pub reinhard_white_point: f32,
    /// Full-resolution framebuffer extent in texels (`vec2<u32>`).
    pub screen_size: [u32; 2],
}

impl GpuTonemapParams {
    /// Builds the immediate block from the framebuffer extent and the live
    /// [`PrismTonemapSettings`].
    pub(crate) fn from_settings(size: UVec2, settings: &PrismTonemapSettings) -> Self {
        Self {
            operator: settings.operator_code(),
            reinhard_white_point: settings.reinhard_white_point,
            screen_size: [size.x, size.y],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_shading::TonemapOperator;

    #[test]
    fn tonemap_params_is_the_16_byte_immediate_block() {
        // `u32` operator (4) + `f32` white point (4) + `vec2<u32>` extent (8)
        // fill 16 bytes, a multiple of 16 with no implicit padding.
        assert_eq!(size_of::<GpuTonemapParams>(), 16);
        assert_eq!(align_of::<GpuTonemapParams>(), 4);
    }

    #[test]
    fn workgroup_constant_matches_the_shader() {
        assert_eq!(TONEMAP_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn from_settings_packs_the_extent_and_operator_code() {
        let settings = PrismTonemapSettings {
            operator: TonemapOperator::AcesFitted,
            ..Default::default()
        };
        let params = GpuTonemapParams::from_settings(UVec2::new(1920, 1080), &settings);
        assert_eq!(params.screen_size, [1920, 1080]);
        // AcesFitted is the fourth variant (code 3) and the white point rides
        // through verbatim.
        assert_eq!(params.operator, 3);
        assert_eq!(params.reinhard_white_point, settings.reinhard_white_point);
    }
}
