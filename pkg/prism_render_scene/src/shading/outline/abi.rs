//! ABI shared between the outline compute pass and `shaders/outline.wesl`.
//!
//! Like the other single-pass post effects, the outline pass carries one
//! immediate (push-constant) block, [`GpuOutlineParams`], mirroring the
//! shader's single `var<immediate>` global. Every field mirrors the shader
//! struct byte-for-byte so machines with and without a GPU agree with the CPU
//! golden in [`prism_render_shading::outline`].
//!
//! The block leads with a `mat4x4<f32>` inverse projection (the clip -> view
//! matrix that reconstructs a linear view-space distance from the reverse-Z
//! device depth, exactly like `dof.wesl` / `ssgi.wesl`), which forces the
//! struct's size to a multiple of the 16-byte immediate alignment WGSL
//! requires. Layout: `view_from_clip` at 0, the `line_color` `vec4` (rgb =
//! authored ink colour, `w` = coverage->opacity strength) at 64, the four
//! golden edge thresholds at 80/84/88/92, the `vec2<u32>` extent at 96, the
//! `id_edges` flag at 104 and a trailing `u32` pad at 108: 64 + 16 + 16 + 8 +
//! 4 + 4 = 112 bytes, a multiple of 16 with no implicit padding.

use bevy_math::{Mat4, UVec2};
use bytemuck::{Pod, Zeroable};

use super::settings::PrismOutlineSettings;

/// Workgroup size (per axis) of the outline compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `outline.wesl`; the dispatch rounds
/// its target extent up to a multiple of this on both axes and the shader
/// bounds-checks every invocation.
pub(crate) const OUTLINE_WORKGROUP_SIZE: u32 = 8;

/// Immediate (push-constant) block consumed by the `outline_main` entry point.
///
/// Mirrors the shader's `GpuOutlineParams`: the `view_from_clip` inverse
/// projection (reconstructs a linear view-space depth from the reverse-Z device
/// depth), the authored `line_color` and its coverage->opacity strength packed
/// into a `vec4`, the golden [`prism_render_shading::OutlineParams`] edge
/// thresholds (`depth_threshold` / `depth_softness` / `normal_threshold` /
/// `normal_softness`), the framebuffer extent and the `id_edges` flag.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuOutlineParams {
    /// Clip -> view (inverse projection); reconstructs a linear view-space
    /// position from the reverse-Z device depth. Uploaded column-major via
    /// [`Mat4::to_cols_array`]. Leads the block for its 16-byte alignment.
    pub view_from_clip: [f32; 16],
    /// `xyz` = authored ink line colour, `w` = coverage->opacity strength that
    /// scales the `[0, 1]` outline coverage before the composite lerp.
    pub line_color: [f32; 4],
    /// Relative depth-jump threshold where the depth edge begins (golden
    /// `depth_threshold`).
    pub depth_threshold: f32,
    /// Half-width of the depth-edge transition (golden `depth_softness`).
    pub depth_softness: f32,
    /// Normal-turn threshold where the crease edge begins (golden
    /// `normal_threshold`).
    pub normal_threshold: f32,
    /// Half-width of the crease-edge transition (golden `normal_softness`).
    pub normal_softness: f32,
    /// Full-resolution framebuffer extent in texels (`vec2<u32>`).
    pub screen_size: [u32; 2],
    /// `1` when a differing neighbour id draws a hard edge, `0` otherwise
    /// (golden `id_edges`). No outline-id G-buffer exists in the prescribed
    /// binding layout, so the dispatch feeds equal ids and this term stays
    /// inert; it is the hook point for a future id buffer.
    pub id_edges: u32,
    /// Padding so the block is a multiple of 16 bytes, matching the shader
    /// struct's trailing `u32` pad.
    pub _pad0: u32,
}

impl GpuOutlineParams {
    /// Builds the immediate block from the inverse projection, the framebuffer
    /// extent and the live [`PrismOutlineSettings`].
    ///
    /// The matrix is uploaded column-major (via [`Mat4::to_cols_array`]) so the
    /// WGSL `mat4x4<f32>` multiply agrees byte-for-byte, and the golden edge
    /// thresholds are carried through verbatim so the on-device edge reduction
    /// is the exact golden twin.
    pub(crate) fn from_settings(
        view_from_clip: Mat4,
        size: UVec2,
        settings: &PrismOutlineSettings,
    ) -> Self {
        Self {
            view_from_clip: view_from_clip.to_cols_array(),
            line_color: [
                settings.line_color[0],
                settings.line_color[1],
                settings.line_color[2],
                settings.line_strength,
            ],
            depth_threshold: settings.depth_threshold,
            depth_softness: settings.depth_softness,
            normal_threshold: settings.normal_threshold,
            normal_softness: settings.normal_softness,
            screen_size: [size.x, size.y],
            id_edges: u32::from(settings.id_edges),
            _pad0: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outline_params_is_the_112_byte_immediate_block() {
        // mat4x4 (64) + line_color vec4 (16) + four threshold f32 (16) + the
        // vec2<u32> extent (8) + id_edges (4) + pad (4) fill 112 bytes, a
        // multiple of the 16-byte immediate alignment the mat4x4 field forces.
        assert_eq!(size_of::<GpuOutlineParams>(), 112);
        assert_eq!(align_of::<GpuOutlineParams>(), 4);
    }

    #[test]
    fn workgroup_constant_matches_the_shader() {
        assert_eq!(OUTLINE_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn from_settings_uploads_the_matrix_column_major_and_folds_thresholds() {
        let inv = Mat4::from_cols_array(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let settings = PrismOutlineSettings::default();
        let params = GpuOutlineParams::from_settings(inv, UVec2::new(1920, 1080), &settings);
        assert_eq!(params.view_from_clip, inv.to_cols_array());
        assert_eq!(params.screen_size, [1920, 1080]);
        // Golden `OutlineParams` edge defaults round-trip.
        assert_eq!(params.depth_threshold, 0.05);
        assert_eq!(params.depth_softness, 0.05);
        assert_eq!(params.normal_threshold, 0.2);
        assert_eq!(params.normal_softness, 0.2);
        // Golden `id_edges` default is on -> 1.
        assert_eq!(params.id_edges, 1);
        // Line colour + strength pack into the leading `vec4`.
        assert_eq!(params.line_color, [
            settings.line_color[0],
            settings.line_color[1],
            settings.line_color[2],
            settings.line_strength,
        ]);
        assert_eq!(params._pad0, 0);
    }
}
