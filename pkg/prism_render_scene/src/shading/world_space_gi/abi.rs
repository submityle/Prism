//! ABI shared between the world-space GI compute passes and their WESL twins
//! (`shaders/world_space_gi_probe_update.wesl` /
//! `shaders/world_space_gi_resolve.wesl`).
//!
//! The subsystem carries one immediate (push-constant) block per pass, each
//! mirroring that shader's single `var<immediate>` global byte-for-byte so
//! machines with and without a GPU agree with the CPU golden in
//! [`prism_render_shading`]'s `gi::world_space`.
//!
//! WGSL gives `mat4x4<f32>` a 16-byte alignment and `vec2` an 8-byte
//! alignment, so both blocks lead with the `view_from_clip` matrix (offset 0,
//! 64 bytes), then the `vec2<f32>` `screen_size` at 64 and the `vec2<u32>`
//! `probe_grid` at 72, then the scalar tail packs to the 16-byte boundary:
//! `64 + 8 + 8 + 16 = 96` bytes, a multiple of 16 with no implicit padding.

use bevy_math::{Mat4, UVec2};
use bytemuck::{Pod, Zeroable};

use super::settings::PrismWorldSpaceGiSettings;

/// Workgroup size (per axis) of both world-space GI compute entry points.
///
/// Must match `@workgroup_size(N, N, 1)` in both shaders; the dispatch rounds
/// its target extent up to a multiple of this on both axes and each shader
/// bounds-checks every invocation.
pub(crate) const WORLD_SPACE_GI_WORKGROUP_SIZE: u32 = 8;

/// Immediate block consumed by the `probe_update_main` entry point.
///
/// One invocation per screen probe: it reconstructs the probe centre's
/// view-space position + normal, gathers its tile's radiance into an L1 SH
/// probe (golden `add_directional_radiance`) and writes it to the probe
/// storage buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWorldSpaceGiProbeParams {
    /// Inverse projection (clip -> view), column-major via
    /// [`Mat4::to_cols_array`]; leads the block for its 16-byte alignment.
    pub view_from_clip: [f32; 16],
    /// Full-resolution framebuffer extent in texels (`vec2<f32>`).
    pub screen_size: [f32; 2],
    /// Probe-grid dimensions in probes (`vec2<u32>`).
    pub probe_grid: [u32; 2],
    /// Tile size in pixels per probe (golden `tile`).
    pub tile: u32,
    /// World-space radiance-cache cell size (golden `cell_size`), forwarded so
    /// the shader's `world_to_cell` helper matches the golden quantisation.
    pub cell_size: f32,
    /// Artistic gain baked into the captured probe radiance.
    pub intensity: f32,
    /// Positive near-plane distance in front of the camera along `-Z`.
    pub near: f32,
}

impl GpuWorldSpaceGiProbeParams {
    /// Builds the `probe_update` immediate block from the framebuffer extent,
    /// the inverse projection, the near-plane distance and the live settings.
    pub(crate) fn from_view(
        view_from_clip: Mat4,
        screen_size: UVec2,
        probe_grid: UVec2,
        near: f32,
        settings: &PrismWorldSpaceGiSettings,
    ) -> Self {
        Self {
            view_from_clip: view_from_clip.to_cols_array(),
            screen_size: [screen_size.x as f32, screen_size.y as f32],
            probe_grid: [probe_grid.x, probe_grid.y],
            tile: settings.tile,
            cell_size: settings.cell_size,
            intensity: settings.intensity,
            near,
        }
    }
}

/// Immediate block consumed by the `resolve_main` entry point.
///
/// One invocation per pixel: it reconstructs the shading point's normal +
/// view-space depth, gathers the four surrounding screen probes, resolves the
/// bilinear × geometric weights (golden `resolve_weights`), blends their SH
/// (golden `blend_sh`) and evaluates the clamped-cosine irradiance (golden
/// `evaluate_irradiance`) into the GI export buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWorldSpaceGiResolveParams {
    /// Inverse projection (clip -> view), column-major via
    /// [`Mat4::to_cols_array`]; leads the block for its 16-byte alignment.
    pub view_from_clip: [f32; 16],
    /// Full-resolution framebuffer extent in texels (`vec2<f32>`).
    pub screen_size: [f32; 2],
    /// Probe-grid dimensions in probes (`vec2<u32>`).
    pub probe_grid: [u32; 2],
    /// Tile size in pixels per probe (golden `tile`).
    pub tile: u32,
    /// Minimum `dot(n_probe, n_point)` for a probe to be accepted (golden
    /// `InterpolationConfig::normal_threshold`).
    pub normal_threshold: f32,
    /// Maximum relative depth difference for a probe to be accepted (golden
    /// `InterpolationConfig::depth_rel_threshold`).
    pub depth_rel_threshold: f32,
    /// Artistic gain applied to the resolved GI irradiance.
    pub intensity: f32,
}

impl GpuWorldSpaceGiResolveParams {
    /// Builds the `resolve` immediate block from the framebuffer extent, the
    /// inverse projection, the probe grid and the live settings.
    pub(crate) fn from_view(
        view_from_clip: Mat4,
        screen_size: UVec2,
        probe_grid: UVec2,
        settings: &PrismWorldSpaceGiSettings,
    ) -> Self {
        Self {
            view_from_clip: view_from_clip.to_cols_array(),
            screen_size: [screen_size.x as f32, screen_size.y as f32],
            probe_grid: [probe_grid.x, probe_grid.y],
            tile: settings.tile,
            normal_threshold: settings.normal_threshold,
            depth_rel_threshold: settings.depth_rel_threshold,
            intensity: settings.intensity,
        }
    }
}

/// Immediate (push-constant) block consumed by
/// `world_space_gi_composite.wesl`'s two entry points.
///
/// Both the base copy and the energy-conserving GI fold only need the
/// framebuffer extent to bounds-check each invocation; the two trailing `u32`s
/// round the block up to the 16-byte immediate alignment, mirroring
/// [`super::super::ssgi`]'s `GpuSsgiCompositeParams` byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWorldSpaceGiCompositeParams {
    /// Framebuffer width in texels.
    pub width: u32,
    /// Framebuffer height in texels.
    pub height: u32,
    /// Padding to satisfy the 16-byte immediate alignment.
    pub _pad0: u32,
    /// Padding to satisfy the 16-byte immediate alignment.
    pub _pad1: u32,
}

impl GpuWorldSpaceGiCompositeParams {
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
    fn probe_params_is_the_96_byte_immediate_block() {
        // mat4x4 (64) + vec2<f32> (8) + vec2<u32> (8) + four scalars (16) fill
        // 96 bytes, a multiple of 16 with no implicit padding.
        assert_eq!(size_of::<GpuWorldSpaceGiProbeParams>(), 96);
        assert_eq!(align_of::<GpuWorldSpaceGiProbeParams>(), 4);
    }

    #[test]
    fn resolve_params_is_the_96_byte_immediate_block() {
        assert_eq!(size_of::<GpuWorldSpaceGiResolveParams>(), 96);
        assert_eq!(align_of::<GpuWorldSpaceGiResolveParams>(), 4);
    }

    #[test]
    fn composite_params_is_the_16_byte_immediate_block() {
        // Framebuffer extent (8) + two padding `u32`s (8) = 16 bytes, the
        // minimum immediate alignment, with no implicit padding.
        assert_eq!(size_of::<GpuWorldSpaceGiCompositeParams>(), 16);
        assert_eq!(align_of::<GpuWorldSpaceGiCompositeParams>(), 4);
    }

    #[test]
    fn composite_params_new_packs_the_extent() {
        let params = GpuWorldSpaceGiCompositeParams::new(1920, 1080);
        assert_eq!(params.width, 1920);
        assert_eq!(params.height, 1080);
        assert_eq!(params._pad0, 0);
        assert_eq!(params._pad1, 0);
    }

    #[test]
    fn workgroup_constant_matches_the_shaders() {
        assert_eq!(WORLD_SPACE_GI_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn probe_params_from_view_packs_the_extent_and_grid() {
        let settings = PrismWorldSpaceGiSettings::default();
        let params = GpuWorldSpaceGiProbeParams::from_view(
            Mat4::IDENTITY,
            UVec2::new(1920, 1080),
            UVec2::new(120, 68),
            0.1,
            &settings,
        );
        assert_eq!(params.screen_size, [1920.0, 1080.0]);
        assert_eq!(params.probe_grid, [120, 68]);
        assert_eq!(params.tile, settings.tile);
        assert_eq!(params.cell_size, settings.cell_size);
        assert_eq!(params.intensity, settings.intensity);
        assert_eq!(params.near, 0.1);
        // Identity matrix round-trips column-major.
        assert_eq!(params.view_from_clip, Mat4::IDENTITY.to_cols_array());
    }

    #[test]
    fn resolve_params_from_view_packs_the_thresholds() {
        let settings = PrismWorldSpaceGiSettings::default();
        let params = GpuWorldSpaceGiResolveParams::from_view(
            Mat4::IDENTITY,
            UVec2::new(1280, 720),
            UVec2::new(80, 45),
            &settings,
        );
        assert_eq!(params.screen_size, [1280.0, 720.0]);
        assert_eq!(params.probe_grid, [80, 45]);
        assert_eq!(params.tile, settings.tile);
        assert_eq!(params.normal_threshold, settings.normal_threshold);
        assert_eq!(params.depth_rel_threshold, settings.depth_rel_threshold);
        assert_eq!(params.intensity, settings.intensity);
    }
}
