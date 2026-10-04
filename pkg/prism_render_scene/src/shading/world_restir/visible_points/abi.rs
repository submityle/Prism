//! Device ABI for the world-space `ReSTIR` visible-point producer pass.
//!
//! The producer runs one compute invocation per screen *tile* (one point per
//! `VISIBLE_POINTS_TILE_SIZE x VISIBLE_POINTS_TILE_SIZE` block of pixels),
//! reconstructs the tile-centre shading point from the SSR prepass depth +
//! packed normal, lifts it into world space, and appends it to the per-frame
//! visible-point list the inject pass hashes into its `SHARC` grid. This module
//! freezes that pass's immediate (push-constant) block and the two dispatch
//! constants, byte-for-byte with the WESL twin
//! `shaders/world_restir_visible_points.wesl`.
//!
//! The visible-point *records* reuse the inject pass's frozen
//! [`super::super::abi::GpuWorldRestirInjectPoint`] layout (two `vec4` lanes =
//! 32 bytes), so no new per-point struct is introduced here: the producer and
//! the inject consumer agree on the storage stride through that single ABI.

use bevy_math::{Mat4, UVec2};
use bytemuck::{Pod, Zeroable};

/// Workgroup edge (both x and y) of the `visible_points_main` entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in
/// `world_restir_visible_points.wesl`; the dispatch rounds the tile grid up to
/// a multiple of this on each axis and the shader bounds-checks every
/// invocation against the live `(tiles_x, tiles_y)`.
pub(crate) const VISIBLE_POINTS_WORKGROUP_SIZE: u32 = 8;

/// Tile edge in pixels: one visible point is produced per
/// `VISIBLE_POINTS_TILE_SIZE x VISIBLE_POINTS_TILE_SIZE` framebuffer block.
///
/// Tiling keeps the injected point count bounded (~129600 points @1080p for a
/// tile edge of 4, under the 131072 default reservoir-table capacity) so the
/// open-addressed hash grid never thrashes.
pub(crate) const VISIBLE_POINTS_TILE_SIZE: u32 = 4;

/// Immediate (push-constant) block consumed by the `visible_points_main` entry
/// point.
///
/// Layout (two `mat4x4<f32>` + two `vec4<u32>` lanes, std430, no implicit
/// padding = 160 bytes):
/// 0..16.  `view_from_clip` (inverse reverse-Z projection; reconstructs the
///         view-space shading point from a tile-centre device depth).
/// 16..32. `world_from_view` (lifts the reconstructed point + its normal into
///         world space for the `SHARC` hash).
/// 32..36. `screen_tiles` = (screen_width, screen_height, tiles_x, tiles_y).
/// 36..40. `tile_point` = (tile_size, point_count, 0, 0).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuVisiblePointsParams {
    /// Clip -> view (inverse reverse-Z projection), column-major.
    pub view_from_clip: [f32; 16],
    /// View -> world, column-major.
    pub world_from_view: [f32; 16],
    /// (screen_width, screen_height, tiles_x, tiles_y) in texels / tiles.
    pub screen_tiles: [u32; 4],
    /// (tile_size_px, point_count, 0, 0): tile edge + dispatch extent / bounds.
    pub tile_point: [u32; 4],
}

impl GpuVisiblePointsParams {
    /// Builds the visible-point immediate block from the per-view matrices, the
    /// framebuffer size, the derived tile grid and the per-point counts.
    pub(crate) fn new(
        view_from_clip: Mat4,
        world_from_view: Mat4,
        screen: UVec2,
        tiles: UVec2,
        tile_size: u32,
        point_count: u32,
    ) -> Self {
        Self {
            view_from_clip: view_from_clip.to_cols_array(),
            world_from_view: world_from_view.to_cols_array(),
            screen_tiles: [screen.x, screen.y, tiles.x, tiles.y],
            tile_point: [tile_size, point_count, 0, 0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn params_is_the_160_byte_four_lane_block() {
        // Two mat4x4 (64 B each) + two vec4<u32> (16 B each) = 160 B, every
        // lane naturally 16-aligned so there is no implicit tail padding. A
        // drift here would mismatch `set_immediates` against the shader's
        // `var<immediate>` block.
        assert_eq!(size_of::<GpuVisiblePointsParams>(), 160);
        assert_eq!(align_of::<GpuVisiblePointsParams>(), 4);
    }

    #[test]
    fn builder_packs_matrices_column_major_and_counts_in_place() {
        let view_from_clip = Mat4::from_cols_array(&core::array::from_fn(|i| i as f32));
        let world_from_view = Mat4::from_cols_array(&core::array::from_fn(|i| (i as f32) * 2.0));
        let params = GpuVisiblePointsParams::new(
            view_from_clip,
            world_from_view,
            UVec2::new(1920, 1080),
            UVec2::new(480, 270),
            VISIBLE_POINTS_TILE_SIZE,
            480 * 270,
        );
        assert_eq!(params.view_from_clip, view_from_clip.to_cols_array());
        assert_eq!(params.world_from_view, world_from_view.to_cols_array());
        assert_eq!(params.screen_tiles, [1920, 1080, 480, 270]);
        assert_eq!(
            params.tile_point,
            [VISIBLE_POINTS_TILE_SIZE, 480 * 270, 0, 0]
        );
    }

    #[test]
    fn dispatch_constants_match_the_wesl_twin() {
        // `@workgroup_size(8, 8, 1)` and the 4x4-pixel tiling in
        // `world_restir_visible_points.wesl`.
        assert_eq!(VISIBLE_POINTS_WORKGROUP_SIZE, 8);
        assert_eq!(VISIBLE_POINTS_TILE_SIZE, 4);
    }
}
