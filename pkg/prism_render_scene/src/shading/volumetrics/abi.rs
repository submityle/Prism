//! Immediate (push-constant) blocks shared with `shaders/volumetrics.wesl`.
//!
//! Kept byte-for-byte in sync with the shader's `VolumetricsScatterParams` and
//! `VolumetricsIntegrateParams` structs. The froxel fog runs as two chained
//! compute passes over a view-frustum-fitted 3D grid:
//!
//! * **scatter** builds every froxel's participating medium and the light it
//!   in-scatters toward the eye, mirroring the golden
//!   [`prism_render_shading::volumetrics`] `MediumSample` / `in_scatter` /
//!   `froxel_source` math, and
//! * **integrate** marches each froxel column front-to-back, folding each
//!   slice's energy-conserving in-scattering under the accumulated
//!   transmittance exactly like the golden `integrate_slice` /
//!   `integrate_froxel_column`.
//!
//! Both records are laid out as 4-byte scalars back to back (never `vec3`, whose
//! WGSL 16-byte alignment would inject padding), so the `#[repr(C)]` structs
//! match the WESL immediate blocks with no implicit padding. The layout unit
//! tests below pin the sizes against drift.

use bytemuck::{Pod, Zeroable};

/// Edge of the scatter pass workgroup cube, matching the shader's
/// `@workgroup_size(4, 4, 4)`: one froxel per invocation.
pub(crate) const VOLUMETRICS_SCATTER_WORKGROUP_SIZE: u32 = 4;

/// Edge of the integrate pass workgroup tile, matching the shader's
/// `@workgroup_size(8, 8, 1)`: one froxel *column* (all Z slices) per invocation.
pub(crate) const VOLUMETRICS_INTEGRATE_WORKGROUP_SIZE: u32 = 8;

/// Scatter-pass immediate block: the froxel grid dimensions, the view-space
/// depth range the grid is fitted to, the homogeneous medium coefficients, the
/// single directional light driving in-scattering, the Henyey-Greenstein
/// anisotropy, the camera frustum half-tangents (so the per-froxel view ray —
/// and thus the phase cosine — is reconstructed on device) and the exponential
/// slice-distribution power.
///
/// Twenty-four 4-byte scalars laid out back to back, so the `#[repr(C)]` record
/// is 96 bytes with no padding and matches the WESL `VolumetricsScatterParams`
/// struct exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuVolumetricsScatterParams {
    /// Froxel grid width (screen-tiled X resolution).
    pub grid_x: u32,
    /// Froxel grid height (screen-tiled Y resolution).
    pub grid_y: u32,
    /// Froxel grid depth (number of view-space depth slices).
    pub grid_z: u32,
    /// Near plane of the froxel grid in view-space units.
    pub near_plane: f32,
    /// Far plane of the froxel grid in view-space units (fog fit distance).
    pub far_plane: f32,
    /// Medium scattering `sigma_s`, red channel.
    pub scattering_r: f32,
    /// Medium scattering `sigma_s`, green channel.
    pub scattering_g: f32,
    /// Medium scattering `sigma_s`, blue channel.
    pub scattering_b: f32,
    /// Medium absorption `sigma_a`, red channel.
    pub absorption_r: f32,
    /// Medium absorption `sigma_a`, green channel.
    pub absorption_g: f32,
    /// Medium absorption `sigma_a`, blue channel.
    pub absorption_b: f32,
    /// Medium emissive radiance density, red channel.
    pub emissive_r: f32,
    /// Medium emissive radiance density, green channel.
    pub emissive_g: f32,
    /// Medium emissive radiance density, blue channel.
    pub emissive_b: f32,
    /// Direction the light travels, X (world/view space, normalised on device).
    pub light_dir_x: f32,
    /// Direction the light travels, Y.
    pub light_dir_y: f32,
    /// Direction the light travels, Z.
    pub light_dir_z: f32,
    /// Incident light radiance reaching the medium, red channel.
    pub light_radiance_r: f32,
    /// Incident light radiance reaching the medium, green channel.
    pub light_radiance_g: f32,
    /// Incident light radiance reaching the medium, blue channel.
    pub light_radiance_b: f32,
    /// Henyey-Greenstein anisotropy `g` (`0` isotropic, `>0` forward glow).
    pub phase_g: f32,
    /// `tan(fov_x / 2)`: half-width of the frustum at unit view depth,
    /// reconstructing each froxel's view ray X.
    pub tan_half_fov_x: f32,
    /// `tan(fov_y / 2)`: half-height of the frustum at unit view depth.
    pub tan_half_fov_y: f32,
    /// Exponential slice-distribution power (`>= 1`): `1` is linear depth, `>1`
    /// packs more froxels near the camera where fog detail matters.
    pub depth_power: f32,
}

impl GpuVolumetricsScatterParams {
    /// Assembles the scatter params from the froxel grid dimensions, the
    /// view-fitted depth range, the medium coefficients, the driving light and
    /// the reconstructed frustum half-tangents. Every extent is forced to at
    /// least `1` and the depth power to at least `1.0` so the shader never
    /// divides by zero or inverts the slice distribution.
    #[expect(clippy::too_many_arguments, reason = "flat mirror of the WESL immediate block")]
    pub(crate) fn new(
        grid: [u32; 3],
        near_plane: f32,
        far_plane: f32,
        scattering: [f32; 3],
        absorption: [f32; 3],
        emissive: [f32; 3],
        light_direction: [f32; 3],
        light_radiance: [f32; 3],
        phase_g: f32,
        tan_half_fov: [f32; 2],
        depth_power: f32,
    ) -> Self {
        Self {
            grid_x: grid[0].max(1),
            grid_y: grid[1].max(1),
            grid_z: grid[2].max(1),
            near_plane: near_plane.max(1.0e-4),
            far_plane: far_plane.max(near_plane.max(1.0e-4) + 1.0e-4),
            scattering_r: scattering[0].max(0.0),
            scattering_g: scattering[1].max(0.0),
            scattering_b: scattering[2].max(0.0),
            absorption_r: absorption[0].max(0.0),
            absorption_g: absorption[1].max(0.0),
            absorption_b: absorption[2].max(0.0),
            emissive_r: emissive[0].max(0.0),
            emissive_g: emissive[1].max(0.0),
            emissive_b: emissive[2].max(0.0),
            light_dir_x: light_direction[0],
            light_dir_y: light_direction[1],
            light_dir_z: light_direction[2],
            light_radiance_r: light_radiance[0].max(0.0),
            light_radiance_g: light_radiance[1].max(0.0),
            light_radiance_b: light_radiance[2].max(0.0),
            phase_g: phase_g.clamp(-0.99, 0.99),
            tan_half_fov_x: tan_half_fov[0].max(1.0e-4),
            tan_half_fov_y: tan_half_fov[1].max(1.0e-4),
            depth_power: depth_power.max(1.0),
        }
    }
}

/// Integrate-pass immediate block: only the froxel grid dimensions, since the
/// column march reads every per-froxel medium/source/thickness straight from
/// the scatter textures. Three 4-byte scalars, 12 bytes, no padding, matching
/// the WESL `VolumetricsIntegrateParams` struct.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuVolumetricsIntegrateParams {
    /// Froxel grid width.
    pub grid_x: u32,
    /// Froxel grid height.
    pub grid_y: u32,
    /// Froxel grid depth (column march length).
    pub grid_z: u32,
}

impl GpuVolumetricsIntegrateParams {
    /// Builds the integrate params from the froxel grid dimensions, forcing each
    /// extent to at least `1`.
    pub(crate) fn new(grid: [u32; 3]) -> Self {
        Self {
            grid_x: grid[0].max(1),
            grid_y: grid[1].max(1),
            grid_z: grid[2].max(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workgroup_sizes_match_the_shader() {
        assert_eq!(VOLUMETRICS_SCATTER_WORKGROUP_SIZE, 4);
        assert_eq!(VOLUMETRICS_INTEGRATE_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn scatter_params_layout_matches_the_wesl_immediate_block() {
        // Twenty-four 4-byte scalars, no padding.
        assert_eq!(size_of::<GpuVolumetricsScatterParams>(), 96);
        assert_eq!(align_of::<GpuVolumetricsScatterParams>(), 4);

        let params = GpuVolumetricsScatterParams::new(
            [160, 90, 64],
            0.1,
            64.0,
            [0.5, 0.5, 0.5],
            [0.1, 0.1, 0.1],
            [0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [4.0, 4.0, 4.0],
            0.3,
            [1.0, 0.5625],
            2.0,
        );
        assert_eq!(params.grid_x, 160);
        assert_eq!(params.grid_y, 90);
        assert_eq!(params.grid_z, 64);
        assert_eq!(params.near_plane, 0.1);
        assert_eq!(params.far_plane, 64.0);
        assert_eq!(params.scattering_r, 0.5);
        assert_eq!(params.absorption_r, 0.1);
        assert_eq!(params.light_dir_y, -1.0);
        assert_eq!(params.light_radiance_r, 4.0);
        assert_eq!(params.phase_g, 0.3);
        assert_eq!(params.tan_half_fov_x, 1.0);
        assert_eq!(params.depth_power, 2.0);
    }

    #[test]
    fn scatter_params_clamp_degenerate_inputs() {
        let params = GpuVolumetricsScatterParams::new(
            [0, 0, 0],
            -1.0,
            -5.0,
            [-1.0; 3],
            [-1.0; 3],
            [-1.0; 3],
            [0.0, 0.0, 1.0],
            [-2.0; 3],
            5.0,
            [0.0, 0.0],
            0.1,
        );
        // Extents floor to 1 so the dispatch never issues a zero-size grid.
        assert_eq!((params.grid_x, params.grid_y, params.grid_z), (1, 1, 1));
        // Near/far stay ordered and positive.
        assert!(params.near_plane > 0.0);
        assert!(params.far_plane > params.near_plane);
        // Coefficients and radiance never go negative.
        assert_eq!(params.scattering_r, 0.0);
        assert_eq!(params.absorption_r, 0.0);
        assert_eq!(params.emissive_r, 0.0);
        assert_eq!(params.light_radiance_r, 0.0);
        // Anisotropy clamps to the golden's (-0.99, 0.99) domain.
        assert_eq!(params.phase_g, 0.99);
        // Frustum half-tangents stay strictly positive.
        assert!(params.tan_half_fov_x > 0.0 && params.tan_half_fov_y > 0.0);
        // Depth power never inverts the slice distribution.
        assert_eq!(params.depth_power, 1.0);
    }

    #[test]
    fn integrate_params_layout_matches_the_wesl_immediate_block() {
        // Three 4-byte scalars, no padding.
        assert_eq!(size_of::<GpuVolumetricsIntegrateParams>(), 12);
        assert_eq!(align_of::<GpuVolumetricsIntegrateParams>(), 4);

        let params = GpuVolumetricsIntegrateParams::new([160, 90, 64]);
        assert_eq!(params.grid_x, 160);
        assert_eq!(params.grid_y, 90);
        assert_eq!(params.grid_z, 64);

        let clamped = GpuVolumetricsIntegrateParams::new([0, 0, 0]);
        assert_eq!((clamped.grid_x, clamped.grid_y, clamped.grid_z), (1, 1, 1));
    }
}
