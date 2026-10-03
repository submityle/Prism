//! Render-world resource gating the DDGI irradiance-volume pass and feeding the
//! golden tunables into the [`GpuDdgiVolume`] uniform block.
//!
//! Folds the golden [`prism_render_shading::gi::irradiance_volume`] defaults
//! ([`DEFAULT_DEPTH_SHARPNESS`], the RTXGI probe-distance exponent, and
//! [`DEFAULT_RELOCATION_LIMIT`]) so the render-world resource and the CPU
//! reference stay in lockstep, with the device-side additions the golden
//! field helpers do not model: a master `enabled` gate, the octahedral interior
//! resolutions, the normal / view biases, the temporal hysteresis and an
//! artistic intensity.
//!
//! The name is prefixed `Prism` to keep the render-world resource distinct from
//! the golden's own types; a game overwrites the resource to retune the probe
//! density or the leak-suppression biases globally without touching any pass
//! code, mirroring [`super::super::world_space_gi`]'s
//! `PrismWorldSpaceGiSettings`.

use bevy_ecs::prelude::Resource;
use bevy_math::{IVec3, Vec3};
use prism_render_shading::gi::irradiance_volume::{
    ProbeGrid, DEFAULT_DEPTH_SHARPNESS, DEFAULT_RELOCATION_LIMIT,
};

use super::abi::GpuDdgiVolume;

/// Default interior octahedral resolution of each irradiance probe (RTXGI's
/// `6x6` texel field, excluding the one-texel gutter border).
pub(crate) const DEFAULT_IRRADIANCE_INTERIOR: u32 = 6;

/// Default interior octahedral resolution of each depth / visibility probe
/// (RTXGI's `16x16` moment field).
pub(crate) const DEFAULT_DEPTH_INTERIOR: u32 = 16;

/// Default temporal hysteresis for the probe-update exponential moving average
/// (RTXGI's slow, stable `0.97` blend).
pub(crate) const DEFAULT_HYSTERESIS: f32 = 0.97;

/// Global DDGI settings consumed by the irradiance-volume sample pass.
///
/// Disabled by default (the subsystem is opt-in; when `false` the pass
/// allocates nothing and dispatches nothing). The field tunables ship the
/// golden / RTXGI defaults so the on-device blend matches the CPU reference; a
/// host raises `enabled` and dials in the lattice via [`volume`].
///
/// [`volume`]: PrismDdgiSettings::volume
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PrismDdgiSettings {
    /// Master enable; when `false` the pass allocates nothing and dispatches
    /// nothing.
    pub enabled: bool,
    /// Interior octahedral resolution of each irradiance probe.
    pub irradiance_interior: u32,
    /// Interior octahedral resolution of each depth / visibility probe.
    pub depth_interior: u32,
    /// World-space normal bias pulling the shading point along its normal
    /// before the probe lookup (reduces self-intersection / light leak).
    pub normal_bias: f32,
    /// World-space self-shadow bias subtracted from the probe->point distance
    /// before the Chebyshev visibility test.
    pub view_bias: f32,
    /// Temporal hysteresis for the probe-update exponential moving average.
    pub hysteresis: f32,
    /// Depth cosine exponent (golden [`DEFAULT_DEPTH_SHARPNESS`]).
    pub depth_sharpness: f32,
    /// Bounded probe relocation fraction (golden [`DEFAULT_RELOCATION_LIMIT`]).
    pub relocation_limit: f32,
    /// Artistic gain applied to the resolved GI irradiance. `1` reproduces the
    /// golden magnitude exactly.
    pub intensity: f32,
}

impl Default for PrismDdgiSettings {
    fn default() -> Self {
        // Fold the golden irradiance-volume defaults so the render-world
        // resource and the CPU reference stay in lockstep. `enabled` is off
        // (opt-in) and `intensity` is unity so the default resolve is the exact
        // golden magnitude when a host flips it on.
        Self {
            enabled: false,
            irradiance_interior: DEFAULT_IRRADIANCE_INTERIOR,
            depth_interior: DEFAULT_DEPTH_INTERIOR,
            // Small world-space biases; the host scales these to the probe
            // spacing once the lattice is known.
            normal_bias: 0.1,
            view_bias: 0.1,
            hysteresis: DEFAULT_HYSTERESIS,
            depth_sharpness: DEFAULT_DEPTH_SHARPNESS,
            relocation_limit: DEFAULT_RELOCATION_LIMIT,
            intensity: 1.0,
        }
    }
}

impl PrismDdgiSettings {
    /// Builds the golden [`ProbeGrid`] and its [`GpuDdgiVolume`] uniform twin
    /// for a lattice anchored at `origin` with `spacing` and `counts`.
    ///
    /// The golden `ProbeGrid::new` sanitises `spacing` (positive) and `counts`
    /// (`>= 1`); the returned [`GpuDdgiVolume`] reads the sanitised values back
    /// so the device block and the CPU reference describe the same lattice.
    pub(crate) fn volume(
        &self,
        origin: Vec3,
        spacing: Vec3,
        counts: IVec3,
    ) -> (ProbeGrid, GpuDdgiVolume) {
        let grid = ProbeGrid::new(origin, spacing, counts);
        let gpu = GpuDdgiVolume::new(
            grid.origin.to_array(),
            grid.spacing.to_array(),
            grid.counts.to_array(),
            self.irradiance_interior,
            self.depth_interior,
            self.normal_bias,
            self.view_bias,
            self.hysteresis,
            self.depth_sharpness,
            self.intensity,
        );
        (grid, gpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_fold_the_golden_constants() {
        let settings = PrismDdgiSettings::default();
        assert_eq!(settings.depth_sharpness, DEFAULT_DEPTH_SHARPNESS);
        assert_eq!(settings.relocation_limit, DEFAULT_RELOCATION_LIMIT);
        // Opt-in gate off by default.
        assert!(!settings.enabled);
        // Unity gain; the default resolve is the exact golden magnitude.
        assert_eq!(settings.intensity, 1.0);
        // Sane, non-degenerate octahedral resolutions.
        assert!(settings.irradiance_interior >= 1);
        assert!(settings.depth_interior >= settings.irradiance_interior);
    }

    #[test]
    fn volume_sanitises_and_mirrors_the_grid() {
        let settings = PrismDdgiSettings::default();
        let (grid, gpu) = settings.volume(
            Vec3::new(1.0, 2.0, 3.0),
            // A negative / zero spacing component is sanitised positive.
            Vec3::new(2.0, 0.0, -1.0),
            IVec3::new(8, 4, 0),
        );
        // The GPU block reads the golden-sanitised lattice back.
        assert_eq!(gpu.origin, grid.origin.to_array());
        assert_eq!(gpu.spacing, grid.spacing.to_array());
        assert_eq!(gpu.counts, grid.counts.to_array());
        // counts.z was 0 -> clamped to 1 by the golden grid.
        assert_eq!(gpu.counts[2], 1);
        assert_eq!(gpu.irradiance_interior, settings.irradiance_interior);
        assert_eq!(gpu.depth_interior, settings.depth_interior);
        assert_eq!(gpu.intensity, settings.intensity);
    }

    #[test]
    fn volume_probe_count_matches_the_lattice() {
        let settings = PrismDdgiSettings::default();
        let (_grid, gpu) = settings.volume(Vec3::ZERO, Vec3::splat(1.0), IVec3::new(8, 4, 2));
        assert_eq!(gpu.probe_count(), 8 * 4 * 2);
    }
}
