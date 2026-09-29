//! Render-world resource gating the world-space GI passes and feeding the
//! golden tunables into the [`GpuWorldSpaceGiProbeParams`] /
//! [`GpuWorldSpaceGiResolveParams`] immediate blocks.
//!
//! Folds the golden [`prism_render_shading::InterpolationConfig`] defaults so
//! the render-world resource and the CPU reference stay in lockstep, with two
//! device-side additions the golden geometry helpers do not model: a master
//! `enabled` gate and a `tile` size selecting the screen-probe density.
//!
//! The name is prefixed `Prism` to keep the render-world resource distinct
//! from the golden's own types; a game can overwrite the resource to retune
//! the probe density or the interpolation thresholds globally without touching
//! any pass code, mirroring [`super::super::color_grade`]'s
//! `PrismColorGradeSettings`.

use bevy_ecs::prelude::Resource;
use bevy_math::{Mat4, UVec2};
use prism_render_shading::InterpolationConfig;

use super::abi::{GpuWorldSpaceGiProbeParams, GpuWorldSpaceGiResolveParams};

/// Global world-space GI settings consumed by the probe-update and resolve
/// passes.
///
/// Disabled by default (the subsystem is opt-in; when `false` the passes
/// allocate nothing and dispatch nothing). The interpolation thresholds ship
/// the golden [`InterpolationConfig`] defaults so the on-device blend matches
/// the CPU reference; a host raises `enabled` and dials in the probe density.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PrismWorldSpaceGiSettings {
    /// Master enable; when `false` the passes allocate nothing and dispatch
    /// nothing.
    pub enabled: bool,
    /// Screen-probe tile size in pixels (golden `tile`): one radiance probe is
    /// seeded per `tile` x `tile` framebuffer block.
    pub tile: u32,
    /// World-space radiance-cache cell size (golden `cell_size`), forwarded to
    /// the probe-update shader's `world_to_cell` quantisation.
    pub cell_size: f32,
    /// Minimum `dot(n_probe, n_point)` for a probe to be accepted (golden
    /// [`InterpolationConfig::normal_threshold`]).
    pub normal_threshold: f32,
    /// Maximum relative depth difference for a probe to be accepted (golden
    /// [`InterpolationConfig::depth_rel_threshold`]).
    pub depth_rel_threshold: f32,
    /// Artistic gain applied to the captured / resolved GI irradiance. `1`
    /// reproduces the golden magnitude exactly.
    pub intensity: f32,
}

impl Default for PrismWorldSpaceGiSettings {
    fn default() -> Self {
        // Fold the golden `InterpolationConfig` defaults so the render-world
        // resource and the CPU reference stay in lockstep. `enabled` is off
        // (opt-in) and `intensity` is unity so the default resolve is the exact
        // golden magnitude when a host flips it on.
        let cfg = InterpolationConfig::default();
        Self {
            enabled: false,
            // One probe per 16x16 tile: the Lumen-style default trading probe
            // density against the per-probe gather cost.
            tile: 16,
            // A one-metre radiance-cache voxel; only forwarded to the golden
            // `world_to_cell` helper and never divides by zero (the helper
            // clamps it).
            cell_size: 1.0,
            normal_threshold: cfg.normal_threshold,
            depth_rel_threshold: cfg.depth_rel_threshold,
            intensity: 1.0,
        }
    }
}

impl PrismWorldSpaceGiSettings {
    /// Builds the `probe_update` immediate block from the framebuffer extent,
    /// the inverse projection, the probe grid and the near-plane distance.
    pub(crate) fn probe_params(
        &self,
        screen_size: UVec2,
        probe_grid: UVec2,
        view_from_clip: Mat4,
        near: f32,
    ) -> GpuWorldSpaceGiProbeParams {
        GpuWorldSpaceGiProbeParams::from_view(view_from_clip, screen_size, probe_grid, near, self)
    }

    /// Builds the `resolve` immediate block from the framebuffer extent, the
    /// inverse projection and the probe grid.
    pub(crate) fn resolve_params(
        &self,
        screen_size: UVec2,
        probe_grid: UVec2,
        view_from_clip: Mat4,
    ) -> GpuWorldSpaceGiResolveParams {
        GpuWorldSpaceGiResolveParams::from_view(view_from_clip, screen_size, probe_grid, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_mirror_the_golden_interpolation_config() {
        let settings = PrismWorldSpaceGiSettings::default();
        let cfg = InterpolationConfig::default();
        assert_eq!(settings.normal_threshold, cfg.normal_threshold);
        assert_eq!(settings.depth_rel_threshold, cfg.depth_rel_threshold);
        // Opt-in gate off by default.
        assert!(!settings.enabled);
        // Unity gain; the default resolve is the exact golden magnitude.
        assert_eq!(settings.intensity, 1.0);
        // A sane, non-degenerate probe density and cache cell.
        assert!(settings.tile >= 1);
        assert!(settings.cell_size > 0.0);
    }

    #[test]
    fn probe_params_forwards_the_grid_and_tile() {
        let settings = PrismWorldSpaceGiSettings::default();
        let params =
            settings.probe_params(UVec2::new(1920, 1080), UVec2::new(120, 68), Mat4::IDENTITY, 0.1);
        assert_eq!(params.screen_size, [1920.0, 1080.0]);
        assert_eq!(params.probe_grid, [120, 68]);
        assert_eq!(params.tile, settings.tile);
        assert_eq!(params.near, 0.1);
    }

    #[test]
    fn resolve_params_forwards_the_thresholds() {
        let settings = PrismWorldSpaceGiSettings::default();
        let params = settings.resolve_params(UVec2::new(1280, 720), UVec2::new(80, 45), Mat4::IDENTITY);
        assert_eq!(params.probe_grid, [80, 45]);
        assert_eq!(params.normal_threshold, settings.normal_threshold);
        assert_eq!(params.depth_rel_threshold, settings.depth_rel_threshold);
        assert_eq!(params.intensity, settings.intensity);
    }
}
