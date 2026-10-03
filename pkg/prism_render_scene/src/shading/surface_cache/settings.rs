//! Render-world resource gating the surface-cache passes and feeding the
//! golden tunables into the surface-cache immediate blocks.
//!
//! Folds the golden [`prism_render_shading::gi::surface_cache::CoverageParams`]
//! and [`prism_render_shading::gi::surface_cache::TemporalParams`] defaults so
//! the render-world resource and the `CPU` reference stay in lockstep, with
//! device-side additions the golden geometry helpers do not model: a master
//! `enabled` gate, a `tile` size selecting the surfel density, a `radius_scale`
//! and the per-pass gather radii.
//!
//! The name is prefixed `Prism` to keep the render-world resource distinct from
//! the golden's own types; a game can overwrite the resource to retune the
//! surfel density or the weight thresholds globally without touching any pass
//! code, mirroring [`super::super::world_space_gi`]'s
//! `PrismWorldSpaceGiSettings`.

use bevy_ecs::prelude::Resource;
use bevy_math::{Mat4, UVec2};
use prism_render_shading::gi::surface_cache::{CoverageParams, TemporalParams};

use super::abi::{
    GpuSurfaceCacheAllocParams, GpuSurfaceCacheCoverageParams, GpuSurfaceCacheFilterParams,
    GpuSurfaceCacheUpdateParams,
};

/// Global surface-cache settings consumed by the alloc, update, spatial-filter
/// and coverage passes.
///
/// Disabled by default (the subsystem is opt-in; when `false` the passes
/// allocate nothing and dispatch nothing). The weight thresholds ship the
/// golden [`CoverageParams`] / [`TemporalParams`] defaults so the on-device
/// accumulation matches the `CPU` reference; a host raises `enabled` and dials
/// in the surfel density.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PrismSurfaceCacheSettings {
    /// Master enable; when `false` the passes allocate nothing and dispatch
    /// nothing.
    pub enabled: bool,
    /// Surfel tile size in pixels: one surfel is seeded per `tile` x `tile`
    /// framebuffer block.
    pub tile: u32,
    /// Multiplier on the derived view-space surfel radius.
    pub radius_scale: f32,
    /// Maximum confidence (golden [`TemporalParams::max_samples`]); caps the
    /// `EMA` weight at `1 / max_samples`.
    pub max_samples: u32,
    /// Anchor-displacement tolerance as a fraction of the surfel radius (golden
    /// [`TemporalParams::position_tolerance`]).
    pub position_tolerance: f32,
    /// Minimum `dot(prev_normal, curr_normal)` to keep history (golden
    /// [`TemporalParams::normal_tolerance`]).
    pub normal_tolerance: f32,
    /// Orientation exponent (golden [`CoverageParams::normal_sharpness`]).
    pub normal_sharpness: f32,
    /// Off-plane tolerance as a fraction of the surfel radius (golden
    /// [`CoverageParams::axial_tolerance`]).
    pub axial_tolerance: f32,
    /// Half-extent (in surfels) of the spatial-filter neighbour box.
    pub filter_radius: u32,
    /// Half-extent (in surfels) of the per-pixel coverage gather box.
    pub gather_radius: u32,
    /// Artistic gain applied to the gathered `GI` radiance. `1` reproduces the
    /// golden magnitude exactly.
    pub intensity: f32,
}

impl Default for PrismSurfaceCacheSettings {
    fn default() -> Self {
        // Fold the golden `CoverageParams` / `TemporalParams` defaults so the
        // render-world resource and the CPU reference stay in lockstep.
        // `enabled` is off (opt-in) and `intensity` is unity so the default
        // gather is the exact golden magnitude when a host flips it on.
        let coverage = CoverageParams::default();
        let temporal = TemporalParams::default();
        Self {
            enabled: false,
            // One surfel per 16x16 tile: the Lumen-style default trading surfel
            // density against the per-pixel gather cost.
            tile: 16,
            radius_scale: 1.0,
            max_samples: temporal.max_samples,
            position_tolerance: temporal.position_tolerance,
            normal_tolerance: temporal.normal_tolerance,
            normal_sharpness: coverage.normal_sharpness,
            axial_tolerance: coverage.axial_tolerance,
            filter_radius: 1,
            gather_radius: 1,
            intensity: 1.0,
        }
    }
}

impl PrismSurfaceCacheSettings {
    /// Builds the `alloc` immediate block from the framebuffer extent, the
    /// inverse projection, the surfel grid and the near-plane distance.
    pub(crate) fn alloc_params(
        &self,
        screen_size: UVec2,
        surfel_grid: UVec2,
        view_from_clip: Mat4,
        near: f32,
    ) -> GpuSurfaceCacheAllocParams {
        GpuSurfaceCacheAllocParams::from_view(view_from_clip, screen_size, surfel_grid, near, self)
    }

    /// Builds the `update` immediate block from the surfel count.
    pub(crate) fn update_params(&self, count: u32) -> GpuSurfaceCacheUpdateParams {
        GpuSurfaceCacheUpdateParams::from_view(count, self)
    }

    /// Builds the `spatial_filter` immediate block from the surfel grid and
    /// count.
    pub(crate) fn filter_params(
        &self,
        surfel_grid: UVec2,
        count: u32,
    ) -> GpuSurfaceCacheFilterParams {
        GpuSurfaceCacheFilterParams::from_view(surfel_grid, count, self)
    }

    /// Builds the `coverage` immediate block from the framebuffer extent, the
    /// inverse projection, the surfel grid and the near-plane distance.
    pub(crate) fn coverage_params(
        &self,
        screen_size: UVec2,
        surfel_grid: UVec2,
        view_from_clip: Mat4,
        near: f32,
    ) -> GpuSurfaceCacheCoverageParams {
        GpuSurfaceCacheCoverageParams::from_view(
            view_from_clip,
            screen_size,
            surfel_grid,
            near,
            self,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_mirror_the_golden_coverage_and_temporal_params() {
        let settings = PrismSurfaceCacheSettings::default();
        let coverage = CoverageParams::default();
        let temporal = TemporalParams::default();
        assert_eq!(settings.normal_sharpness, coverage.normal_sharpness);
        assert_eq!(settings.axial_tolerance, coverage.axial_tolerance);
        assert_eq!(settings.max_samples, temporal.max_samples);
        assert_eq!(settings.position_tolerance, temporal.position_tolerance);
        assert_eq!(settings.normal_tolerance, temporal.normal_tolerance);
        // Opt-in gate off by default.
        assert!(!settings.enabled);
        // Unity gain; the default gather is the exact golden magnitude.
        assert_eq!(settings.intensity, 1.0);
        // Sane, non-degenerate density and radii.
        assert!(settings.tile >= 1);
        assert!(settings.radius_scale > 0.0);
    }

    #[test]
    fn alloc_params_forwards_the_grid_and_tile() {
        let settings = PrismSurfaceCacheSettings::default();
        let params = settings.alloc_params(
            UVec2::new(1920, 1080),
            UVec2::new(120, 68),
            Mat4::IDENTITY,
            0.1,
        );
        assert_eq!(params.screen_size, [1920.0, 1080.0]);
        assert_eq!(params.surfel_grid, [120, 68]);
        assert_eq!(params.tile, settings.tile);
        assert_eq!(params.near, 0.1);
        assert_eq!(params.radius_scale, settings.radius_scale);
    }

    #[test]
    fn update_params_forwards_the_temporal_tunables() {
        let settings = PrismSurfaceCacheSettings::default();
        let params = settings.update_params(4096);
        assert_eq!(params.count, 4096);
        assert_eq!(params.max_samples, settings.max_samples);
        assert_eq!(params.position_tolerance, settings.position_tolerance);
        assert_eq!(params.normal_tolerance, settings.normal_tolerance);
    }

    #[test]
    fn coverage_params_forwards_the_coverage_tunables() {
        let settings = PrismSurfaceCacheSettings::default();
        let params = settings.coverage_params(
            UVec2::new(1280, 720),
            UVec2::new(80, 45),
            Mat4::IDENTITY,
            0.1,
        );
        assert_eq!(params.surfel_grid, [80, 45]);
        assert_eq!(params.normal_sharpness, settings.normal_sharpness);
        assert_eq!(params.axial_tolerance, settings.axial_tolerance);
        assert_eq!(params.intensity, settings.intensity);
        assert_eq!(params.gather_radius, settings.gather_radius);
    }
}
