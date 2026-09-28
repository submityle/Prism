//! ABI shared between the GTAO geometry prepass and `shaders/gtao_prepass.wesl`.
//!
//! The prepass is dispatched once over the whole framebuffer; the only
//! per-dispatch state it needs is the `view_from_world` transform (to push the
//! world-space vertices the visibility buffer decodes into the RH,
//! camera-at-origin view space GTAO integrates in) plus the framebuffer
//! dimensions used to reject out-of-bounds invocations.

use bytemuck::{Pod, Zeroable};

/// Workgroup size (per axis) of the `gtao_prepass` compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `shaders/gtao_prepass.wesl`; the
/// dispatch rounds the viewport up to a multiple of this on both axes.
pub(crate) const GTAO_PREPASS_WORKGROUP_SIZE: u32 = 8;

/// Immediate (push-constant) block consumed by `gtao_prepass.wesl`.
///
/// `view_from_world` is a column-major `Mat4` (uploaded via
/// [`glam::Mat4::to_cols_array`]) so the WGSL `mat4x4<f32>` multiply agrees
/// byte-for-byte. The 64-byte matrix is already 16-byte aligned, so the four
/// trailing `u32`s (two live, two padding) fill the final 16 bytes exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuGtaoPrepassParams {
    /// Column-major world -> view transform.
    pub view_from_world: [f32; 16],
    /// Framebuffer width in pixels; invocations at or beyond it early-out.
    pub width: u32,
    /// Framebuffer height in pixels.
    pub height: u32,
    /// Padding to round the block out to a 16-byte multiple.
    pub _pad0: u32,
    /// Padding to round the block out to a 16-byte multiple.
    pub _pad1: u32,
}


/// Workgroup size (per axis) of the `gtao` kernel compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `shaders/gtao.wesl`; the dispatch
/// rounds the viewport up to a multiple of this on both axes.
pub(crate) const GTAO_KERNEL_WORKGROUP_SIZE: u32 = 8;

/// Immediate (push-constant) block consumed by `gtao.wesl`'s `compute_gtao`.
///
/// Mirrors the WGSL `GtaoConfig` field-for-field: five `f32` tunables followed
/// by three `u32`s (two live counts + one padding word), 32 bytes total. The
/// two field-of-view tangents are derived from the perspective projection's
/// diagonal exactly as [`prism_render_shading::ao::GtaoCamera::from_projection`]
/// does (`tan_half_fov = 1 / |proj_diag|`), so the CPU golden and this GPU twin
/// reconstruct identical view-space positions.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuGtaoConfig {
    /// `1 / |proj[0][0]|`; scales NDC x into a view-space slope.
    pub tan_half_fov_x: f32,
    /// `1 / |proj[1][1]|`; scales NDC y into a view-space slope.
    pub tan_half_fov_y: f32,
    /// Sampling radius in world units; larger gathers more distant occluders.
    pub world_radius: f32,
    /// Fraction of the radius (`0..=1`) at which distance falloff begins.
    pub falloff: f32,
    /// Occlusion contrast exponent applied to the final visibility.
    pub power: f32,
    /// Slice directions swept through the view vector (clamped `>= 1`).
    pub slice_count: u32,
    /// Marched samples per side per slice (clamped `>= 1`).
    pub steps_per_slice: u32,
    /// Padding word rounding the block to an 8-scalar (32-byte) multiple.
    pub padding: u32,
}

impl GpuGtaoConfig {
    /// Builds the kernel config from the active perspective projection diagonal
    /// and the artist-facing GTAO tunables. `proj_m00`/`proj_m11` are the
    /// `clip_from_view` diagonal entries (`x_axis.x` / `y_axis.y`); the counts
    /// are clamped to at least one, matching the golden and the shader.
    pub fn from_projection(
        proj_m00: f32,
        proj_m11: f32,
        world_radius: f32,
        falloff: f32,
        power: f32,
        slice_count: u32,
        steps_per_slice: u32,
    ) -> Self {
        Self {
            tan_half_fov_x: proj_m00.abs().recip(),
            tan_half_fov_y: proj_m11.abs().recip(),
            world_radius,
            falloff,
            power,
            slice_count: slice_count.max(1),
            steps_per_slice: steps_per_slice.max(1),
            padding: 0,
        }
    }
}

/// Workgroup size (per axis) of the `gtao_denoise` compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `shaders/gtao_denoise.wesl`; the
/// dispatch rounds the viewport up to a multiple of this on both axes.
pub(crate) const GTAO_DENOISE_WORKGROUP_SIZE: u32 = 8;

/// Immediate (push-constant) block consumed by `gtao_denoise.wesl`.
///
/// Mirrors the WGSL `GtaoDenoiseConfig` field-for-field: the kernel half-width
/// plus the three edge-stop sigmas of
/// [`prism_render_shading::ao::GtaoDenoiseConfig`], 16 bytes total (already a
/// 16-byte multiple, so no trailing padding is needed).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuGtaoDenoiseConfig {
    /// Bilateral kernel half-width in pixels (`0` = passthrough identity).
    pub radius: u32,
    /// Gaussian spatial falloff in pixels; larger smooths harder.
    pub spatial_sigma: f32,
    /// Depth edge-stop tolerance as a fraction of the centre pixel's view depth.
    pub depth_sigma: f32,
    /// Normal edge-stop sharpness (`dot(n, n_c)` is raised to this power).
    pub normal_power: f32,
}

impl GpuGtaoDenoiseConfig {
    /// Builds the denoise config from the artist-facing tunables, mirroring the
    /// golden `GtaoDenoiseConfig` defaults' domain.
    pub fn new(radius: u32, spatial_sigma: f32, depth_sigma: f32, normal_power: f32) -> Self {
        Self {
            radius,
            spatial_sigma,
            depth_sigma,
            normal_power,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepass_params_layout_matches_the_wgsl_immediate_block() {
        // 64-byte mat4x4 (16-byte aligned) + four u32 scalars = 80 bytes.
        assert_eq!(size_of::<GpuGtaoPrepassParams>(), 80);
        assert_eq!(align_of::<GpuGtaoPrepassParams>(), 4);
        assert_eq!(GTAO_PREPASS_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn kernel_config_layout_matches_the_wgsl_config_struct() {
        // Five f32 + three u32 = 32 bytes, 4-byte aligned.
        assert_eq!(size_of::<GpuGtaoConfig>(), 32);
        assert_eq!(align_of::<GpuGtaoConfig>(), 4);
        assert_eq!(GTAO_KERNEL_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn denoise_config_layout_matches_the_wgsl_config_struct() {
        // One u32 + three f32 = 16 bytes, 4-byte aligned; no trailing padding.
        assert_eq!(size_of::<GpuGtaoDenoiseConfig>(), 16);
        assert_eq!(align_of::<GpuGtaoDenoiseConfig>(), 4);
        assert_eq!(GTAO_DENOISE_WORKGROUP_SIZE, 8);
        let config = GpuGtaoDenoiseConfig::new(2, 2.0, 0.05, 8.0);
        assert_eq!(config.radius, 2);
        assert_eq!(config.spatial_sigma, 2.0);
        assert_eq!(config.depth_sigma, 0.05);
        assert_eq!(config.normal_power, 8.0);
    }

    #[test]
    fn kernel_config_derives_fov_tangents_and_clamps_counts() {
        // proj diagonal 2.0/1.5 -> tangents 0.5/0.666..; zero counts clamp to 1.
        let config = GpuGtaoConfig::from_projection(2.0, -1.5, 1.25, 0.6, 1.5, 0, 0);
        assert!((config.tan_half_fov_x - 0.5).abs() < 1e-6);
        assert!((config.tan_half_fov_y - (1.0 / 1.5)).abs() < 1e-6);
        assert_eq!(config.world_radius, 1.25);
        assert_eq!(config.slice_count, 1);
        assert_eq!(config.steps_per_slice, 1);
        assert_eq!(config.padding, 0);
    }
}
