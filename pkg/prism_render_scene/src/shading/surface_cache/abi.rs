//! `ABI` shared between the surface-cache compute passes and their `WESL`
//! twins (`shaders/surface_cache_{alloc,update,spatial_filter,coverage}.wesl`
//! and `shaders/surface_cache_composite.wesl`).
//!
//! The subsystem carries one immediate (push-constant) block per pass, each
//! mirroring that shader's single `var<immediate>` global byte-for-byte so a
//! machine with or without a `GPU` agrees with the `CPU` golden in
//! [`prism_render_shading`]'s `gi::surface_cache`.
//!
//! `WGSL` gives `mat4x4<f32>` a 16-byte alignment and `vec2` an 8-byte
//! alignment, so the matrix-led blocks (`alloc` / `coverage`) begin with
//! `view_from_clip` (offset 0, 64 bytes), then the `vec2`s, then the scalar
//! tail packs to the 16-byte immediate boundary with explicit padding.
//!
//! The persistent surfel storage element [`GpuSurfel`] is an all-scalar
//! std430 struct (48 bytes) shared read-write by `alloc` -> `update` ->
//! `spatial_filter` and read-only by `coverage`.

use bevy_math::{Mat4, UVec2};
use bytemuck::{Pod, Zeroable};

use super::settings::PrismSurfaceCacheSettings;

/// Workgroup size of the 1-D per-surfel passes (`alloc`, `update`,
/// `spatial_filter`); must match their `@workgroup_size(64, 1, 1)`.
pub(crate) const SURFACE_CACHE_WORKGROUP_SIZE_1D: u32 = 64;

/// Workgroup size (per axis) of the 2-D per-pixel passes (`coverage`,
/// composite); must match their `@workgroup_size(8, 8, 1)`.
pub(crate) const SURFACE_CACHE_WORKGROUP_SIZE_2D: u32 = 8;

/// Persistent per-surfel storage element, mirroring the all-scalar `WESL`
/// `Surfel` struct's std430 layout (12 x 4 = 48 bytes, 48-byte array stride).
///
/// Positions and normals are view-space; `sample_count` is the confidence
/// frame count and `valid` is a `0`/`1` occupancy flag. Scalar fields (never
/// `vec3`) keep the std430 stride at 48 bytes with no implicit padding.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuSurfel {
    /// View-space anchor x.
    pub pos_x: f32,
    /// View-space anchor y.
    pub pos_y: f32,
    /// View-space anchor z.
    pub pos_z: f32,
    /// View-space unit normal x.
    pub normal_x: f32,
    /// View-space unit normal y.
    pub normal_y: f32,
    /// View-space unit normal z.
    pub normal_z: f32,
    /// Disc radius in view-space units.
    pub radius: f32,
    /// Accumulated linear-RGB radiance r.
    pub radiance_x: f32,
    /// Accumulated linear-RGB radiance g.
    pub radiance_y: f32,
    /// Accumulated linear-RGB radiance b.
    pub radiance_z: f32,
    /// Confidence frame count (capped at `max_samples`).
    pub sample_count: u32,
    /// Occupancy flag: `1` if this surfel holds valid geometry.
    pub valid: u32,
}

/// Immediate block consumed by the `surface_cache_alloc_main` entry point.
///
/// One invocation per surfel (screen tile): it reconstructs the tile centre's
/// view-space position + normal, samples the pre-exposed scene colour and
/// writes a fresh surfel into the scratch buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuSurfaceCacheAllocParams {
    /// Inverse projection (clip -> view), column-major via
    /// [`Mat4::to_cols_array`]; leads the block for its 16-byte alignment.
    pub view_from_clip: [f32; 16],
    /// Full-resolution framebuffer extent in texels (`vec2<f32>`).
    pub screen_size: [f32; 2],
    /// Surfel-grid dimensions in surfels (`vec2<u32>`).
    pub surfel_grid: [u32; 2],
    /// Tile size in pixels per surfel.
    pub tile: u32,
    /// Positive near-plane distance in front of the camera along `-Z`.
    pub near: f32,
    /// Multiplier on the derived view-space surfel radius.
    pub radius_scale: f32,
    /// Padding to the 16-byte immediate boundary.
    pub _pad0: u32,
}

impl GpuSurfaceCacheAllocParams {
    /// Builds the `alloc` immediate block from the view and live settings.
    pub(crate) fn from_view(
        view_from_clip: Mat4,
        screen_size: UVec2,
        surfel_grid: UVec2,
        near: f32,
        settings: &PrismSurfaceCacheSettings,
    ) -> Self {
        Self {
            view_from_clip: view_from_clip.to_cols_array(),
            screen_size: [screen_size.x as f32, screen_size.y as f32],
            surfel_grid: [surfel_grid.x, surfel_grid.y],
            tile: settings.tile,
            near,
            radius_scale: settings.radius_scale,
            _pad0: 0,
        }
    }
}

/// Immediate block consumed by the `surface_cache_update_main` entry point.
///
/// One invocation per surfel: it blends the freshly sampled surfel into the
/// persistent surfel via the golden confidence-weighted `EMA`, resetting on
/// disocclusion.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuSurfaceCacheUpdateParams {
    /// Surfel count; the invocation bound.
    pub count: u32,
    /// Maximum confidence (golden `max_samples`); caps the `EMA` weight.
    pub max_samples: u32,
    /// Anchor-displacement tolerance as a fraction of the surfel radius.
    pub position_tolerance: f32,
    /// Minimum `dot(prev_normal, curr_normal)` to keep history.
    pub normal_tolerance: f32,
}

impl GpuSurfaceCacheUpdateParams {
    /// Builds the `update` immediate block from the surfel count and settings.
    pub(crate) fn from_view(count: u32, settings: &PrismSurfaceCacheSettings) -> Self {
        Self {
            count,
            max_samples: settings.max_samples,
            position_tolerance: settings.position_tolerance,
            normal_tolerance: settings.normal_tolerance,
        }
    }
}

/// Immediate block consumed by the `surface_cache_spatial_filter_main` entry
/// point.
///
/// One invocation per surfel: it gathers the grid neighbours in a
/// `filter_radius` box and blends their radiance by the golden geometric
/// weight into the filtered scratch buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuSurfaceCacheFilterParams {
    /// Surfel-grid dimensions in surfels (`vec2<u32>`).
    pub surfel_grid: [u32; 2],
    /// Orientation exponent (golden `normal_sharpness`).
    pub normal_sharpness: f32,
    /// Off-plane tolerance as a fraction of the surfel radius (golden
    /// `axial_tolerance`).
    pub axial_tolerance: f32,
    /// Surfel count; the invocation bound.
    pub count: u32,
    /// Half-extent of the neighbour box gathered (in surfels).
    pub filter_radius: u32,
    /// Padding to the 16-byte immediate boundary.
    pub _pad0: u32,
    /// Padding to the 16-byte immediate boundary.
    pub _pad1: u32,
}

impl GpuSurfaceCacheFilterParams {
    /// Builds the `spatial_filter` immediate block from the grid, count and
    /// settings.
    pub(crate) fn from_view(
        surfel_grid: UVec2,
        count: u32,
        settings: &PrismSurfaceCacheSettings,
    ) -> Self {
        Self {
            surfel_grid: [surfel_grid.x, surfel_grid.y],
            normal_sharpness: settings.normal_sharpness,
            axial_tolerance: settings.axial_tolerance,
            count,
            filter_radius: settings.filter_radius,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

/// Immediate block consumed by the `surface_cache_coverage_main` entry point.
///
/// One invocation per pixel: it reconstructs the shading point's view-space
/// position + normal, gathers the surrounding surfels in a `gather_radius` box
/// and writes the coverage-weighted radiance + confidence into the `GI`
/// export.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuSurfaceCacheCoverageParams {
    /// Inverse projection (clip -> view), column-major via
    /// [`Mat4::to_cols_array`]; leads the block for its 16-byte alignment.
    pub view_from_clip: [f32; 16],
    /// Full-resolution framebuffer extent in texels (`vec2<f32>`).
    pub screen_size: [f32; 2],
    /// Surfel-grid dimensions in surfels (`vec2<u32>`).
    pub surfel_grid: [u32; 2],
    /// Tile size in pixels per surfel.
    pub tile: u32,
    /// Positive near-plane distance in front of the camera along `-Z`.
    pub near: f32,
    /// Orientation exponent (golden `normal_sharpness`).
    pub normal_sharpness: f32,
    /// Off-plane tolerance as a fraction of the surfel radius (golden
    /// `axial_tolerance`).
    pub axial_tolerance: f32,
    /// Artistic gain applied to the gathered radiance.
    pub intensity: f32,
    /// Half-extent of the surfel box gathered per pixel (in surfels).
    pub gather_radius: u32,
    /// Padding to the 16-byte immediate boundary.
    pub _pad0: u32,
    /// Padding to the 16-byte immediate boundary.
    pub _pad1: u32,
}

impl GpuSurfaceCacheCoverageParams {
    /// Builds the `coverage` immediate block from the view and live settings.
    pub(crate) fn from_view(
        view_from_clip: Mat4,
        screen_size: UVec2,
        surfel_grid: UVec2,
        near: f32,
        settings: &PrismSurfaceCacheSettings,
    ) -> Self {
        Self {
            view_from_clip: view_from_clip.to_cols_array(),
            screen_size: [screen_size.x as f32, screen_size.y as f32],
            surfel_grid: [surfel_grid.x, surfel_grid.y],
            tile: settings.tile,
            near,
            normal_sharpness: settings.normal_sharpness,
            axial_tolerance: settings.axial_tolerance,
            intensity: settings.intensity,
            gather_radius: settings.gather_radius,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

/// Immediate block consumed by `surface_cache_composite.wesl`'s two entry
/// points.
///
/// Both the base copy and the energy-conserving fold only need the framebuffer
/// extent to bounds-check each invocation; the two trailing `u32`s round the
/// block up to the 16-byte immediate alignment.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuSurfaceCacheCompositeParams {
    /// Framebuffer width in texels.
    pub width: u32,
    /// Framebuffer height in texels.
    pub height: u32,
    /// Padding to satisfy the 16-byte immediate alignment.
    pub _pad0: u32,
    /// Padding to satisfy the 16-byte immediate alignment.
    pub _pad1: u32,
}

impl GpuSurfaceCacheCompositeParams {
    /// Builds the composite params from the framebuffer extent.
    pub(crate) fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surfel_is_the_48_byte_scalar_layout() {
        // 12 scalars x 4 bytes = 48, a multiple of the 16-byte array stride
        // base and matching the all-scalar WESL `Surfel` struct.
        assert_eq!(size_of::<GpuSurfel>(), 48);
        assert_eq!(align_of::<GpuSurfel>(), 4);
    }

    #[test]
    fn alloc_params_is_the_96_byte_immediate_block() {
        assert_eq!(size_of::<GpuSurfaceCacheAllocParams>(), 96);
        assert_eq!(align_of::<GpuSurfaceCacheAllocParams>(), 4);
    }

    #[test]
    fn update_params_is_the_16_byte_immediate_block() {
        assert_eq!(size_of::<GpuSurfaceCacheUpdateParams>(), 16);
        assert_eq!(align_of::<GpuSurfaceCacheUpdateParams>(), 4);
    }

    #[test]
    fn filter_params_is_the_32_byte_immediate_block() {
        assert_eq!(size_of::<GpuSurfaceCacheFilterParams>(), 32);
        assert_eq!(align_of::<GpuSurfaceCacheFilterParams>(), 4);
    }

    #[test]
    fn coverage_params_is_the_112_byte_immediate_block() {
        assert_eq!(size_of::<GpuSurfaceCacheCoverageParams>(), 112);
        assert_eq!(align_of::<GpuSurfaceCacheCoverageParams>(), 4);
    }

    #[test]
    fn composite_params_is_the_16_byte_immediate_block() {
        assert_eq!(size_of::<GpuSurfaceCacheCompositeParams>(), 16);
        assert_eq!(align_of::<GpuSurfaceCacheCompositeParams>(), 4);
    }

    #[test]
    fn workgroup_constants_match_the_shaders() {
        assert_eq!(SURFACE_CACHE_WORKGROUP_SIZE_1D, 64);
        assert_eq!(SURFACE_CACHE_WORKGROUP_SIZE_2D, 8);
    }
}
