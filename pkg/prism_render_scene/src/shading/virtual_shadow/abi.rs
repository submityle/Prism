//! GPU-side virtual-shadow-map ABI: the `#[repr(C)]` immediate (push-constant)
//! records the page-mark and sample passes bind, kept byte-for-byte in sync
//! with the WESL twins in `shaders/vsm_page_mark.wesl` and
//! `shaders/vsm_sample.wesl`.
//!
//! The virtual shadow map is UE5-style demand paging: receivers drive a
//! per-frame page-request pass, an allocator makes the requested pages resident
//! in a bounded physical page atlas, and the resolve pass samples visibility
//! through a virtual-to-physical page table.  These records carry the clipmap
//! layout ([`prism_render_shading::ClipmapConfig`]) plus the per-pass driving
//! state to the GPU so both shaders address pages exactly as the CPU golden
//! [`prism_render_shading::shadow`] virtual-shadow-map reference does.
//!
//! Every struct here is laid out as 4-byte scalars back to back so it matches
//! the WESL immediate block with no implicit padding; the layout unit tests
//! pin the sizes and offsets against drift.

use bytemuck::{Pod, Zeroable};
use prism_render_shading::{ClipmapConfig, Receiver, ReceiverProjection};

/// Workgroup size of `vsm_page_mark.wesl`'s `vsm_mark_pages` entry: one
/// receiver per invocation along X (`@workgroup_size(64, 1, 1)`).
pub(crate) const VSM_PAGE_MARK_WORKGROUP_SIZE: u32 = 64;

/// Edge of `vsm_sample.wesl`'s `vsm_selftest` workgroup tile
/// (`@workgroup_size(8, 8, 1)`).
pub(crate) const VSM_SAMPLE_WORKGROUP_SIZE: u32 = 8;

/// Sentinel physical-page index for a virtual page that is not resident this
/// frame, matching the WESL `VSM_PAGE_UNMAPPED` (`0xffffffffu`).  It is the GPU
/// twin of [`prism_render_shading::VirtualPageTable::get`] returning `None`.
pub(crate) const VSM_PAGE_UNMAPPED: u32 = u32::MAX;

/// One shadow receiver projected into the light's clipmap plane, the GPU twin
/// of the golden [`prism_render_shading::Receiver`] and of `vsm_page_mark.wesl`'s
/// `VsmReceiver` struct.  Four 4-byte scalars, 16 bytes, no padding.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuVsmReceiver {
    /// Receiver position in the light-space clipmap plane (world units).
    pub light_space_xy: [f32; 2],
    /// Positive view-space distance; selects the clip level (finer when near).
    pub view_distance: f32,
    /// Soft-shadow filter kernel half-width in shadow texels; widens the page
    /// footprint so edge samples still find resident neighbours.
    pub filter_radius_texels: f32,
}

impl GpuVsmReceiver {
    /// Packs a golden [`Receiver`] into its GPU record verbatim.
    pub(crate) fn from_reference(receiver: &Receiver) -> Self {
        Self {
            light_space_xy: [receiver.light_space_xy.x, receiver.light_space_xy.y],
            view_distance: receiver.view_distance,
            filter_radius_texels: receiver.filter_radius_texels,
        }
    }
}

/// Uniform block for `vsm_receiver_gen.wesl`: the camera inverse
/// view-projection used to unproject depth, the light's orthonormal clipmap
/// basis, the camera world position (view-distance origin) and the receiver
/// filter footprint.  Laid out to match the shader's `VsmReceiverGenParams`
/// std140 uniform exactly: the `mat4x4` occupies bytes 0..64 and each following
/// `vec3` lands on a 16-byte boundary (64, 80, 96) with its trailing scalar
/// filling the fourth column, so the plain `#[repr(C)]` scalar packing below is
/// byte-for-byte with std140 without any explicit padding.  112 bytes total.
///
/// Unlike the page-mark / sample params (which ride in the push-constant
/// immediate block), this record is 112 bytes with a `mat4x4`, so it is uploaded
/// through a uniform buffer instead; see `resources.rs`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuVsmReceiverGenParams {
    /// Column-major inverse view-projection (bytes 0..64), byte-identical to the
    /// matrix the CPU golden `reconstruct_world_position` unprojects with.
    pub inverse_view_proj: [f32; 16],
    /// Right axis of the light's clipmap plane (unit, perpendicular to the light).
    pub light_right: [f32; 3],
    /// Soft-shadow filter kernel half-width in shadow texels (fills the fourth
    /// column of `light_right`'s 16-byte std140 slot).
    pub filter_radius_texels: f32,
    /// Up axis of the light's clipmap plane (unit, perpendicular to the light).
    pub light_up: [f32; 3],
    /// Framebuffer width in pixels (fills `light_up`'s fourth std140 column).
    pub viewport_width: u32,
    /// Camera world position; a receiver's view distance is measured from here.
    pub camera_world: [f32; 3],
    /// Framebuffer height in pixels (fills `camera_world`'s fourth std140 column).
    pub viewport_height: u32,
}

impl GpuVsmReceiverGenParams {
    /// Builds the receiver-generation uniform from the camera inverse
    /// view-projection, the light-space [`ReceiverProjection`] basis and the
    /// framebuffer size.  The filter radius is clamped to `>= 0` so a degenerate
    /// setting can never widen the footprint the wrong way, matching the golden
    /// [`ReceiverProjection::project`].
    pub(crate) fn new(
        inverse_view_proj: [f32; 16],
        projection: &ReceiverProjection,
        viewport: [u32; 2],
    ) -> Self {
        Self {
            inverse_view_proj,
            light_right: [
                projection.light_right.x,
                projection.light_right.y,
                projection.light_right.z,
            ],
            filter_radius_texels: projection.filter_radius_texels.max(0.0),
            light_up: [
                projection.light_up.x,
                projection.light_up.y,
                projection.light_up.z,
            ],
            viewport_width: viewport[0],
            camera_world: [
                projection.camera_world.x,
                projection.camera_world.y,
                projection.camera_world.z,
            ],
            viewport_height: viewport[1],
        }
    }
}

/// Immediate block for `vsm_page_mark.wesl`: the clipmap layout plus this
/// frame's driving state (light id, receiver count, camera position for the
/// whole-page window snap).  Twelve 4-byte scalars, 48 bytes, no padding, and
/// byte-for-byte with the shader's `VsmPageMarkParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuVsmPageMarkParams {
    /// Number of clipmap levels (mirrors [`ClipmapConfig::levels`]).
    pub levels: u32,
    /// Edge length of a page in texels (mirrors [`ClipmapConfig::page_size`]).
    pub page_size: u32,
    /// Pages along one edge of a level's resident window.
    pub pages_per_level_edge: u32,
    /// World page coordinate mapped to key coordinate `0`.
    pub page_coord_bias: i32,
    /// World-space edge length of one texel at the finest level.
    pub level0_texel_world_size: f32,
    /// View distance at or below which level `0` is selected.
    pub level0_max_distance: f32,
    /// Light id whose pages this dispatch marks.
    pub light: u32,
    /// Number of valid receivers in the receiver buffer.
    pub receiver_count: u32,
    /// Camera X used to snap the resident window to whole pages.
    pub camera_x: f32,
    /// Camera Y used to snap the resident window to whole pages.
    pub camera_y: f32,
    /// Padding to keep the block a whole number of 16-byte rows.
    pub pad0: u32,
    /// Padding to keep the block a whole number of 16-byte rows.
    pub pad1: u32,
}

impl GpuVsmPageMarkParams {
    /// Builds the page-mark params from the clipmap config plus this frame's
    /// light id, camera position and receiver count.
    pub(crate) fn new(
        clipmap: &ClipmapConfig,
        light: u32,
        camera_xy: [f32; 2],
        receiver_count: u32,
    ) -> Self {
        Self {
            levels: u32::from(clipmap.levels.max(1)),
            page_size: u32::from(clipmap.page_size.max(1)),
            pages_per_level_edge: u32::from(clipmap.pages_per_level_edge.max(1)),
            page_coord_bias: clipmap.page_coord_bias,
            level0_texel_world_size: clipmap.level0_texel_world_size,
            level0_max_distance: clipmap.level0_max_distance,
            light,
            receiver_count,
            camera_x: camera_xy[0],
            camera_y: camera_xy[1],
            pad0: 0,
            pad1: 0,
        }
    }
}

/// Immediate block for `vsm_sample.wesl`: the clipmap layout plus the physical
/// atlas geometry (how many physical pages exist, how they tile the atlas, and
/// the PCF radius).  Twelve 4-byte scalars, 48 bytes, no padding, and
/// byte-for-byte with the shader's `VsmSampleParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuVsmSampleParams {
    /// Number of clipmap levels (mirrors [`ClipmapConfig::levels`]).
    pub levels: u32,
    /// Edge length of a page in texels (mirrors [`ClipmapConfig::page_size`]).
    pub page_size: u32,
    /// Pages along one edge of a level's resident window.
    pub pages_per_level_edge: u32,
    /// World page coordinate mapped to key coordinate `0`.
    pub page_coord_bias: i32,
    /// World-space edge length of one texel at the finest level.
    pub level0_texel_world_size: f32,
    /// View distance at or below which level `0` is selected.
    pub level0_max_distance: f32,
    /// Number of physical pages backing the atlas.
    pub physical_pages: u32,
    /// Physical pages along one edge of the square atlas grid.
    pub physical_pages_per_edge: u32,
    /// Soft-shadow PCF box half-width in physical texels.
    pub pcf_radius: i32,
    /// Padding to keep the block a whole number of 16-byte rows.
    pub pad0: u32,
    /// Padding to keep the block a whole number of 16-byte rows.
    pub pad1: u32,
    /// Padding to keep the block a whole number of 16-byte rows.
    pub pad2: u32,
}

impl GpuVsmSampleParams {
    /// Builds the sample params from the clipmap config plus the physical atlas
    /// geometry and PCF radius.
    pub(crate) fn new(
        clipmap: &ClipmapConfig,
        physical_pages: u32,
        physical_pages_per_edge: u32,
        pcf_radius: i32,
    ) -> Self {
        Self {
            levels: u32::from(clipmap.levels.max(1)),
            page_size: u32::from(clipmap.page_size.max(1)),
            pages_per_level_edge: u32::from(clipmap.pages_per_level_edge.max(1)),
            page_coord_bias: clipmap.page_coord_bias,
            level0_texel_world_size: clipmap.level0_texel_world_size,
            level0_max_distance: clipmap.level0_max_distance,
            physical_pages,
            physical_pages_per_edge: physical_pages_per_edge.max(1),
            pcf_radius: pcf_radius.max(0),
            pad0: 0,
            pad1: 0,
            pad2: 0,
        }
    }
}

/// Number of `u32` slots the per-frame request bitmap / page table needs:
/// `levels * pages_per_level_edge^2`, the flat resident-window slot space both
/// shaders index (`slot = level * edge*edge + local_y * edge + local_x`).
pub(crate) fn window_slot_count(clipmap: &ClipmapConfig) -> u32 {
    let levels = u32::from(clipmap.levels.max(1));
    let edge = u32::from(clipmap.pages_per_level_edge.max(1));
    levels * edge * edge
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::offset_of;
    use bevy_math::{Vec2, Vec3};

    fn clipmap() -> ClipmapConfig {
        ClipmapConfig {
            levels: 4,
            page_size: 128,
            pages_per_level_edge: 8,
            level0_texel_world_size: 0.1,
            level0_max_distance: 10.0,
            page_coord_bias: 32_768,
        }
    }

    #[test]
    fn workgroup_sizes_match_shaders() {
        assert_eq!(VSM_PAGE_MARK_WORKGROUP_SIZE, 64);
        assert_eq!(VSM_SAMPLE_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn page_unmapped_matches_wesl_sentinel() {
        // WESL 0xffffffffu.
        assert_eq!(VSM_PAGE_UNMAPPED, 0xffff_ffff);
    }

    #[test]
    fn receiver_layout_matches_wesl_struct() {
        // Four 4-byte scalars, no padding.
        assert_eq!(size_of::<GpuVsmReceiver>(), 16);
        assert_eq!(align_of::<GpuVsmReceiver>(), 4);

        let gpu = GpuVsmReceiver::from_reference(&Receiver {
            light_space_xy: Vec2::new(3.0, -4.0),
            view_distance: 12.5,
            filter_radius_texels: 2.0,
        });
        assert_eq!(gpu.light_space_xy, [3.0, -4.0]);
        assert_eq!(gpu.view_distance, 12.5);
        assert_eq!(gpu.filter_radius_texels, 2.0);
    }

    #[test]
    fn page_mark_params_layout_matches_the_wesl_immediate_block() {
        // Twelve 4-byte scalars, no padding.
        assert_eq!(size_of::<GpuVsmPageMarkParams>(), 48);
        assert_eq!(align_of::<GpuVsmPageMarkParams>(), 4);

        let c = clipmap();
        let params = GpuVsmPageMarkParams::new(&c, 7, [5.0, -3.0], 42);
        assert_eq!(params.levels, 4);
        assert_eq!(params.page_size, 128);
        assert_eq!(params.pages_per_level_edge, 8);
        assert_eq!(params.page_coord_bias, 32_768);
        assert_eq!(params.level0_texel_world_size, 0.1);
        assert_eq!(params.level0_max_distance, 10.0);
        assert_eq!(params.light, 7);
        assert_eq!(params.receiver_count, 42);
        assert_eq!(params.camera_x, 5.0);
        assert_eq!(params.camera_y, -3.0);
        assert_eq!(params.pad0, 0);
        assert_eq!(params.pad1, 0);
    }

    #[test]
    fn receiver_gen_params_layout_matches_the_wesl_std140_uniform() {
        // mat4 (64B) + three vec3+scalar rows (48B) = 112B, std140-compatible.
        assert_eq!(size_of::<GpuVsmReceiverGenParams>(), 112);
        assert_eq!(align_of::<GpuVsmReceiverGenParams>(), 4);
        // Each vec3 must start on its std140 16-byte boundary and its trailing
        // scalar must fill the fourth column, or the GPU would read the matrix
        // and basis shifted.
        assert_eq!(offset_of!(GpuVsmReceiverGenParams, inverse_view_proj), 0);
        assert_eq!(offset_of!(GpuVsmReceiverGenParams, light_right), 64);
        assert_eq!(offset_of!(GpuVsmReceiverGenParams, filter_radius_texels), 76);
        assert_eq!(offset_of!(GpuVsmReceiverGenParams, light_up), 80);
        assert_eq!(offset_of!(GpuVsmReceiverGenParams, viewport_width), 92);
        assert_eq!(offset_of!(GpuVsmReceiverGenParams, camera_world), 96);
        assert_eq!(offset_of!(GpuVsmReceiverGenParams, viewport_height), 108);
    }

    #[test]
    fn receiver_gen_params_builder_copies_matrix_basis_and_viewport() {
        let inv = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let projection = ReceiverProjection {
            light_right: Vec3::new(1.0, 0.0, 0.0),
            light_up: Vec3::new(0.0, 0.0, 1.0),
            camera_world: Vec3::new(4.0, 3.0, 10.0),
            filter_radius_texels: 2.5,
        };
        let params = GpuVsmReceiverGenParams::new(inv, &projection, [1920, 1080]);
        assert_eq!(params.inverse_view_proj, inv);
        assert_eq!(params.light_right, [1.0, 0.0, 0.0]);
        assert_eq!(params.light_up, [0.0, 0.0, 1.0]);
        assert_eq!(params.camera_world, [4.0, 3.0, 10.0]);
        assert_eq!(params.filter_radius_texels, 2.5);
        assert_eq!(params.viewport_width, 1920);
        assert_eq!(params.viewport_height, 1080);
    }

    #[test]
    fn receiver_gen_params_clamp_negative_filter_radius() {
        let projection = ReceiverProjection {
            light_right: Vec3::X,
            light_up: Vec3::Z,
            camera_world: Vec3::ZERO,
            filter_radius_texels: -4.0,
        };
        let params = GpuVsmReceiverGenParams::new([0.0; 16], &projection, [8, 8]);
        // A negative footprint would shrink pages below their receiver; clamp it.
        assert_eq!(params.filter_radius_texels, 0.0);
    }

    #[test]
    fn sample_params_layout_matches_the_wesl_immediate_block() {
        // Twelve 4-byte scalars, no padding.
        assert_eq!(size_of::<GpuVsmSampleParams>(), 48);
        assert_eq!(align_of::<GpuVsmSampleParams>(), 4);

        let c = clipmap();
        let params = GpuVsmSampleParams::new(&c, 4096, 64, 1);
        assert_eq!(params.levels, 4);
        assert_eq!(params.page_size, 128);
        assert_eq!(params.pages_per_level_edge, 8);
        assert_eq!(params.page_coord_bias, 32_768);
        assert_eq!(params.level0_texel_world_size, 0.1);
        assert_eq!(params.level0_max_distance, 10.0);
        assert_eq!(params.physical_pages, 4096);
        assert_eq!(params.physical_pages_per_edge, 64);
        assert_eq!(params.pcf_radius, 1);
        assert_eq!(params.pad0, 0);
        assert_eq!(params.pad1, 0);
        assert_eq!(params.pad2, 0);
    }

    #[test]
    fn sample_params_clamp_degenerate_geometry() {
        let c = clipmap();
        let params = GpuVsmSampleParams::new(&c, 0, 0, -3);
        // Atlas edge is clamped to at least one page and PCF radius to >= 0 so
        // the shader never divides by zero or loops with a negative bound.
        assert_eq!(params.physical_pages_per_edge, 1);
        assert_eq!(params.pcf_radius, 0);
    }

    #[test]
    fn window_slot_count_is_levels_times_edge_squared() {
        let c = clipmap();
        assert_eq!(window_slot_count(&c), 4 * 8 * 8);
    }
}
