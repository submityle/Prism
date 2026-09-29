//! Frame-constant tunables for Prism's froxel volumetric-fog subsystem.
//!
//! [`PrismVolumetricsSettings`] is the single render-world resource the two
//! froxel-fog compute passes read to size the view-frustum-fitted 3D grid and
//! parameterise the participating medium. It maps directly onto the golden
//! [`prism_render_shading::volumetrics`] reduction: the scattering/absorption
//! coefficients feed [`prism_render_shading::volumetrics::MediumSample`], the
//! light and anisotropy feed
//! [`prism_render_shading::volumetrics::in_scatter`] /
//! [`prism_render_shading::volumetrics::henyey_greenstein`], and the grid depth
//! range drives the per-slice thickness the golden
//! [`prism_render_shading::volumetrics::integrate_froxel_column`] folds.
//!
//! There is no architecture-boundary contract for volumetric fog (unlike VSM's
//! [`prism_render_architecture::virtual_shadow::VirtualShadowSettings`]), so the
//! subsystem owns its enable flag here rather than in
//! [`super::super::runtime::PrismShadingSettings`]: the render-world plumbing
//! module boundary forbids touching `runtime.rs`, and gating on a self-owned
//! resource keeps the fog opt-in without coupling to the shared front toggle.
//! A game can overwrite the resource to retune globally without touching any
//! pass code.

use bevy_ecs::prelude::Resource;
use bevy_math::UVec3;

use super::abi::{GpuVolumetricsIntegrateParams, GpuVolumetricsScatterParams};

/// Global froxel volumetric-fog settings consumed by the fog compute passes.
///
/// Every field is a plain tunable mirrored into the golden
/// [`prism_render_shading::volumetrics`] math; the immediate blocks the passes
/// upload are built from these through [`Self::scatter_params`] /
/// [`Self::integrate_params`], which apply the same clamps the ABI constructors
/// enforce (non-negative coefficients, ordered near/far, `depth_power >= 1`).
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PrismVolumetricsSettings {
    /// Master enable. Fog is opt-in: `false` skips every prepare/dispatch step
    /// and drops the cached froxel volumes so nothing lingers resident.
    pub enabled: bool,
    /// Froxel grid dimensions `[x, y, z]`: screen-tiled X/Y and view-space depth
    /// slices. `[160, 90, 64]` fits a 16:9 frustum at a fog-appropriate coarse
    /// screen resolution with 64 depth slices.
    pub grid: [u32; 3],
    /// Near plane of the froxel grid in view-space units (the eye-side depth the
    /// first slice starts at).
    pub near_plane: f32,
    /// Far plane of the froxel grid in view-space units (the fog fit distance
    /// past which the medium is ignored).
    pub far_plane: f32,
    /// Medium scattering coefficient `sigma_s` per channel.
    pub scattering: [f32; 3],
    /// Medium absorption coefficient `sigma_a` per channel.
    pub absorption: [f32; 3],
    /// Medium emissive radiance density per channel (self-lit haze).
    pub emissive: [f32; 3],
    /// Direction the driving directional light travels (world/view space,
    /// normalised on device).
    pub light_direction: [f32; 3],
    /// Incident radiance the driving light delivers to the medium, per channel.
    pub light_radiance: [f32; 3],
    /// Henyey-Greenstein anisotropy `g` (`0` isotropic, `>0` forward glow).
    pub phase_g: f32,
    /// Exponential slice-distribution power (`>= 1`): packs more froxels near
    /// the camera where fog detail matters most.
    pub depth_power: f32,
}

impl Default for PrismVolumetricsSettings {
    fn default() -> Self {
        Self {
            // Opt-in, matching every other shading front's enable default.
            enabled: false,
            // 16:9 froxel grid with 64 depth slices (the golden column length).
            grid: [160, 90, 64],
            near_plane: 0.1,
            far_plane: 64.0,
            // A thin, faintly absorbing neutral haze: visible god-ray glow
            // without washing the scene out.
            scattering: [0.03, 0.03, 0.03],
            absorption: [0.01, 0.01, 0.01],
            emissive: [0.0, 0.0, 0.0],
            // A high sun coming down and slightly across the frame.
            light_direction: [0.35, -1.0, 0.35],
            light_radiance: [4.0, 4.0, 4.0],
            // Mild forward scattering for a natural sun-side haze glow.
            phase_g: 0.3,
            // Bias froxel detail toward the eye.
            depth_power: 2.0,
        }
    }
}

impl PrismVolumetricsSettings {
    /// The froxel grid dimensions as a [`UVec3`], each extent floored to `1` so
    /// the dispatch never issues a zero-size grid.
    pub(crate) fn grid(&self) -> UVec3 {
        UVec3::new(
            self.grid[0].max(1),
            self.grid[1].max(1),
            self.grid[2].max(1),
        )
    }

    /// Builds the scatter-pass immediate block from these tunables and the
    /// camera frustum half-tangents the per-view resource reconstructs from the
    /// projection. All clamps live in
    /// [`GpuVolumetricsScatterParams::new`].
    pub(crate) fn scatter_params(&self, tan_half_fov: [f32; 2]) -> GpuVolumetricsScatterParams {
        GpuVolumetricsScatterParams::new(
            self.grid,
            self.near_plane,
            self.far_plane,
            self.scattering,
            self.absorption,
            self.emissive,
            self.light_direction,
            self.light_radiance,
            self.phase_g,
            tan_half_fov,
            self.depth_power,
        )
    }

    /// Builds the integrate-pass immediate block (just the grid dimensions) from
    /// these tunables. Clamps live in [`GpuVolumetricsIntegrateParams::new`].
    pub(crate) fn integrate_params(&self) -> GpuVolumetricsIntegrateParams {
        GpuVolumetricsIntegrateParams::new(self.grid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_opt_in_with_a_16_9_froxel_grid() {
        let settings = PrismVolumetricsSettings::default();
        assert!(!settings.enabled);
        assert_eq!(settings.grid, [160, 90, 64]);
        assert_eq!(settings.grid(), UVec3::new(160, 90, 64));
        assert!(settings.near_plane > 0.0 && settings.far_plane > settings.near_plane);
    }

    #[test]
    fn scatter_params_mirror_the_settings_and_carry_the_frustum() {
        let settings = PrismVolumetricsSettings::default();
        let params = settings.scatter_params([1.0, 0.5625]);
        assert_eq!((params.grid_x, params.grid_y, params.grid_z), (160, 90, 64));
        assert_eq!(params.near_plane, 0.1);
        assert_eq!(params.far_plane, 64.0);
        assert_eq!(params.scattering_r, 0.03);
        assert_eq!(params.absorption_r, 0.01);
        assert_eq!(params.light_dir_y, -1.0);
        assert_eq!(params.light_radiance_r, 4.0);
        assert_eq!(params.phase_g, 0.3);
        assert_eq!(params.tan_half_fov_x, 1.0);
        assert_eq!(params.tan_half_fov_y, 0.5625);
        assert_eq!(params.depth_power, 2.0);
    }

    #[test]
    fn integrate_params_carry_only_the_grid() {
        let settings = PrismVolumetricsSettings::default();
        let params = settings.integrate_params();
        assert_eq!((params.grid_x, params.grid_y, params.grid_z), (160, 90, 64));
    }

    #[test]
    fn grid_floors_degenerate_extents_to_one() {
        let settings = PrismVolumetricsSettings {
            grid: [0, 0, 0],
            ..PrismVolumetricsSettings::default()
        };
        assert_eq!(settings.grid(), UVec3::new(1, 1, 1));
    }
}
