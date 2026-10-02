//! ABI shared between the motion-blur compute passes and
//! `shaders/motion_blur.wesl`.
//!
//! The three passes (`TileMax`, `NeighborMax`, reconstruction) share a single
//! immediate (push-constant) block, [`MotionBlurParams`], whose every field
//! mirrors the shader's `MotionBlurParams` struct byte-for-byte so machines
//! with and without a GPU agree with the CPU golden in
//! [`prism_render_shading::motion_blur`]. The block carries the inverse
//! projection the reconstruction pass unprojects reverse-Z device depth with,
//! the framebuffer extent, the derived tile grid, the tile edge, the gather
//! sample count and the golden tunables (`max_velocity_px`, `soft_z_extent`,
//! `exposure_fraction`).
//!
//! The leading `mat4x4<f32>` forces the struct's total size to a multiple of
//! the 16-byte immediate alignment WGSL requires: 64 (matrix) + 24 (six `u32`)
//! + 12 (three `f32`) + 12 (three `u32` pads) = 112 bytes.

use bevy_math::{Mat4, UVec2};
use bytemuck::{Pod, Zeroable};

use super::settings::PrismMotionBlurSettings;

/// Workgroup size (per axis) of every motion-blur compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `motion_blur.wesl`; each dispatch
/// rounds its target extent up to a multiple of this on both axes and the
/// shaders bounds-check every invocation.
pub(crate) const MOTION_BLUR_WORKGROUP_SIZE: u32 = 8;

/// Edge length (in texels) of a `TileMax` / `NeighborMax` tile.
///
/// `TileMax` reduces each `MOTION_BLUR_TILE_SIZE`-square screen tile to one
/// velocity, so the tile textures are sized `ceil(dim / MOTION_BLUR_TILE_SIZE)`
/// per axis. This is a GPU-side budget (not a golden constant): a 16-texel tile
/// bounds the per-tile scan while comfortably exceeding the default half-tile
/// blur budget.
pub(crate) const MOTION_BLUR_TILE_SIZE: u32 = 16;

/// Immediate (push-constant) block consumed by every `motion_blur.wesl` entry
/// point.
///
/// Mirrors the shader's `MotionBlurParams`: the `view_from_clip` inverse
/// projection (reconstructs linear view depth from reverse-Z device depth for
/// the soft-depth term), the framebuffer extent, the tile grid, the tile edge,
/// the gather sample count and the golden tunables, then three trailing `u32`
/// pads to the 16-byte immediate alignment the matrix forces on the struct.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct MotionBlurParams {
    /// Clip -> view (inverse projection); reconstructs linear view-space depth
    /// from reverse-Z device depth so the soft-depth term is scale-correct.
    /// Uploaded column-major via [`Mat4::to_cols_array`].
    pub view_from_clip: [f32; 16],
    /// Full-resolution framebuffer width in texels.
    pub width: u32,
    /// Full-resolution framebuffer height in texels.
    pub height: u32,
    /// Tile-grid width: `ceil(width / tile_size)`.
    pub tiles_x: u32,
    /// Tile-grid height: `ceil(height / tile_size)`.
    pub tiles_y: u32,
    /// Edge length (texels) of a `TileMax` / `NeighborMax` tile.
    pub tile_size: u32,
    /// Reconstruction gather taps per pixel.
    pub sample_count: u32,
    /// Blur budget in pixels; velocities are clamped to this to bound the
    /// gather (golden `MotionBlurParams::max_velocity_px`).
    pub max_velocity_px: f32,
    /// Depth softness (view-space units) fed to `mb_soft_depth_compare` (golden
    /// `MotionBlurParams::soft_z_extent`).
    pub soft_z_extent: f32,
    /// Fraction of the frame the shutter is open, fed to `mb_shutter_velocity`
    /// (golden `MotionBlurParams::exposure_fraction`).
    pub exposure_fraction: f32,
    /// Padding to satisfy the 16-byte immediate alignment.
    pub _pad0: u32,
    /// Padding to satisfy the 16-byte immediate alignment.
    pub _pad1: u32,
    /// Padding to satisfy the 16-byte immediate alignment.
    pub _pad2: u32,
}

impl MotionBlurParams {
    /// Builds the immediate block from the inverse projection, the framebuffer
    /// extent and the live [`PrismMotionBlurSettings`], deriving the tile grid
    /// as `ceil(dim / MOTION_BLUR_TILE_SIZE)`.
    ///
    /// The matrix is uploaded column-major (via [`Mat4::to_cols_array`]) so the
    /// WGSL `mat4x4<f32>` multiply agrees byte-for-byte, and the sample count is
    /// floored at one so a zero request still runs the centre tap rather than
    /// dividing by an empty gather weight.
    pub(crate) fn from_settings(
        view_from_clip: Mat4,
        size: UVec2,
        settings: &PrismMotionBlurSettings,
    ) -> Self {
        Self {
            view_from_clip: view_from_clip.to_cols_array(),
            width: size.x,
            height: size.y,
            tiles_x: size.x.div_ceil(MOTION_BLUR_TILE_SIZE),
            tiles_y: size.y.div_ceil(MOTION_BLUR_TILE_SIZE),
            tile_size: MOTION_BLUR_TILE_SIZE,
            sample_count: settings.sample_count.max(1),
            max_velocity_px: settings.max_velocity_px,
            soft_z_extent: settings.soft_z_extent,
            exposure_fraction: settings.exposure_fraction,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_match_the_shader_immediate_layout() {
        // A mat4x4 (64) + six u32 extents/grid (24) + three f32 tunables (12) +
        // three u32 pads (12) fill 112 bytes, a multiple of the 16-byte
        // immediate alignment the mat4x4 field forces on the struct.
        assert_eq!(size_of::<MotionBlurParams>(), 112);
        assert_eq!(align_of::<MotionBlurParams>(), 4);
    }

    #[test]
    fn tile_and_workgroup_constants_match_the_shader() {
        assert_eq!(MOTION_BLUR_TILE_SIZE, 16);
        assert_eq!(MOTION_BLUR_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn from_settings_folds_the_grid_and_uploads_the_matrix_column_major() {
        let inv = Mat4::from_cols_array(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let settings = PrismMotionBlurSettings::default();
        let params = MotionBlurParams::from_settings(inv, UVec2::new(1920, 1080), &settings);
        assert_eq!(params.view_from_clip, inv.to_cols_array());
        assert_eq!(params.width, 1920);
        assert_eq!(params.height, 1080);
        // ceil(1920 / 16) = 120, ceil(1080 / 16) = 68 (1088 / 16).
        assert_eq!(params.tiles_x, 120);
        assert_eq!(params.tiles_y, 68);
        assert_eq!(params.tile_size, 16);
        // Golden `MotionBlurParams` defaults.
        assert_eq!(params.sample_count, 16);
        assert_eq!(params.max_velocity_px, 64.0);
        assert_eq!(params.soft_z_extent, 1.0);
        assert_eq!(params.exposure_fraction, 0.5);
        assert_eq!(params._pad0, 0);
        assert_eq!(params._pad1, 0);
        assert_eq!(params._pad2, 0);
    }

    #[test]
    fn from_settings_floors_the_sample_count_to_one() {
        let settings = PrismMotionBlurSettings {
            sample_count: 0,
            ..Default::default()
        };
        let params = MotionBlurParams::from_settings(Mat4::IDENTITY, UVec2::new(64, 64), &settings);
        assert_eq!(params.sample_count, 1);
    }
}
