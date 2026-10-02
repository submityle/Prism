//! Render-world resource gating the world-space `ReSTIR` direct-illumination
//! fill pass and feeding the golden spatial-hash tunables into the
//! [`GpuWorldRestirFillParams`] immediate block.
//!
//! Folds the golden [`prism_render_shading::gi::world_restir::spatial_hash`]
//! `HashGridParams::DEFAULT` so the render-world resource and the CPU reference
//! stay in lockstep, with device-side additions the golden grid helpers do not
//! model: a master `enabled` gate, the resident reservoir-table `capacity`, the
//! `ReSTIR` `M`-cap / artistic gain, and the per-frame `GRIS` spatial-reuse
//! budget.
//!
//! The name is prefixed `Prism` to keep the render-world resource distinct from
//! the golden's own types; a game overwrites the resource to retune the hash
//! grid or the reuse budget globally without touching any pass code, mirroring
//! [`super::super::world_space_gi`]'s `PrismWorldSpaceGiSettings`.

use bevy_ecs::prelude::Resource;
use prism_render_shading::gi::world_restir::spatial_hash::HashGridParams;

use super::abi::WORLD_RESTIR_RESERVOIR_STRIDE;

/// Default resident reservoir-table capacity: `2^17` open-addressed world-cell
/// slots, matching the `SHARC`-style cache size the fill dispatch rounds up to
/// a multiple of the workgroup size.
const DEFAULT_RESTIR_CAPACITY: u32 = 131_072;

/// Default `ReSTIR` `M`-cap: a slot accumulates at most this much temporal
/// confidence before older history is discounted (golden
/// `Reservoir::cap_confidence`), bounding reaction latency to lighting change.
const DEFAULT_RESTIR_M_CAP: f32 = 32.0;

/// Default per-frame `GRIS` spatial-reuse neighbour count each slot merges.
const DEFAULT_RESTIR_SPATIAL_SAMPLES: u32 = 4;

/// Default `GRIS` spatial-reuse search radius in cells around each slot.
const DEFAULT_RESTIR_SPATIAL_RADIUS: u32 = 1;

/// Global world-space `ReSTIR` direct-illumination settings consumed by the
/// fill pass and the water-surface `@group(9)` consumer.
///
/// Disabled by default (the subsystem is opt-in; when `false` the pass
/// allocates nothing and dispatches nothing). The hash-grid tunables ship the
/// golden [`HashGridParams::DEFAULT`] so the on-device cell quantisation
/// matches the CPU reference; a host raises `enabled` and dials in the
/// reservoir capacity and the reuse budget.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PrismWorldRestirSettings {
    /// Master enable; when `false` the fill pass allocates nothing and
    /// dispatches nothing.
    pub enabled: bool,
    /// Resident reservoir-table slot count (clamped to at least `1`): the
    /// dispatch extent and the open-addressing modulus (golden `bucket_index`).
    pub capacity: u32,
    /// Edge length of a level-0 cell in world units (golden
    /// [`HashGridParams::base_cell_size`]).
    pub base_cell_size: f32,
    /// Distance-to-cell-size scale (golden [`HashGridParams::level_scale`]);
    /// `<= 0` disables level scaling (uniform fine grid).
    pub level_scale: f32,
    /// Side count of the `resolution x resolution` normal-bin grid (golden
    /// [`HashGridParams::normal_resolution`]).
    pub normal_resolution: u32,
    /// `ReSTIR` `M`-cap bounding how much temporal history a slot accumulates
    /// (golden `Reservoir::cap_confidence`).
    pub m_cap: f32,
    /// Artistic gain applied to the reservoir radiance before `W`. `1`
    /// reproduces the golden magnitude exactly.
    pub intensity: f32,
    /// Number of spatial neighbours each slot merges per frame (golden `GRIS`
    /// reuse); `0` disables spatial reuse.
    pub spatial_samples: u32,
    /// Spatial-reuse search radius in cells around each slot.
    pub spatial_radius: u32,
}

impl Default for PrismWorldRestirSettings {
    fn default() -> Self {
        // Fold the golden `HashGridParams` defaults so the render-world resource
        // and the CPU reference quantise world cells identically. `enabled` is
        // off (opt-in) and `intensity` is unity so the default fill is the exact
        // golden magnitude when a host flips it on.
        let grid = HashGridParams::DEFAULT;
        Self {
            enabled: false,
            capacity: DEFAULT_RESTIR_CAPACITY,
            base_cell_size: grid.base_cell_size,
            level_scale: grid.level_scale,
            normal_resolution: grid.normal_resolution,
            m_cap: DEFAULT_RESTIR_M_CAP,
            intensity: 1.0,
            spatial_samples: DEFAULT_RESTIR_SPATIAL_SAMPLES,
            spatial_radius: DEFAULT_RESTIR_SPATIAL_RADIUS,
        }
    }
}

impl PrismWorldRestirSettings {
    /// Reconstructs the golden [`HashGridParams`] the fill shader's cell
    /// quantisation must agree with. The per-frame grid-phase jitter is a
    /// dispatch-time input (not persisted in the settings), so this returns the
    /// zero-jitter grid; the caller folds the live jitter into the immediate
    /// block via [`super::abi::GpuWorldRestirFillParams::from_settings`].
    pub(crate) fn grid_params(&self) -> HashGridParams {
        HashGridParams {
            base_cell_size: self.base_cell_size,
            normal_resolution: self.normal_resolution,
            level_scale: self.level_scale,
            jitter: HashGridParams::DEFAULT.jitter,
        }
    }

    /// Byte size of the resident reservoir table: `max(capacity, 1)` slots of
    /// [`WORLD_RESTIR_RESERVOIR_STRIDE`] bytes each. The `max` guards the
    /// degenerate `capacity == 0` case so the buffer is never zero-sized.
    pub(crate) fn reservoir_buffer_size(&self) -> u64 {
        u64::from(self.capacity.max(1)) * WORLD_RESTIR_RESERVOIR_STRIDE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shading::world_restir::abi::GpuWorldRestirFillParams;
    use bevy_math::Vec3;

    #[test]
    fn defaults_mirror_the_golden_hash_grid_params() {
        let settings = PrismWorldRestirSettings::default();
        let grid = HashGridParams::DEFAULT;
        assert_eq!(settings.base_cell_size, grid.base_cell_size);
        assert_eq!(settings.level_scale, grid.level_scale);
        assert_eq!(settings.normal_resolution, grid.normal_resolution);
        // Opt-in gate off by default.
        assert!(!settings.enabled);
        // Unity gain; the default fill is the exact golden magnitude.
        assert_eq!(settings.intensity, 1.0);
        // A sane, non-degenerate cache capacity and reuse budget.
        assert!(settings.capacity >= 1);
        assert!(settings.m_cap > 0.0);
    }

    #[test]
    fn grid_params_round_trips_the_golden_default() {
        // With default tunables the reconstructed grid is byte-for-byte the
        // golden `DEFAULT` (zero jitter), so the device quantisation matches the
        // CPU reference exactly.
        let settings = PrismWorldRestirSettings::default();
        assert_eq!(settings.grid_params(), HashGridParams::DEFAULT);
    }

    #[test]
    fn grid_params_forwards_retuned_tunables() {
        let settings = PrismWorldRestirSettings {
            base_cell_size: 2.5,
            level_scale: 0.0, // uniform grid
            normal_resolution: 4,
            ..Default::default()
        };
        let grid = settings.grid_params();
        assert_eq!(grid.base_cell_size, 2.5);
        assert_eq!(grid.level_scale, 0.0);
        assert_eq!(grid.normal_resolution, 4);
        assert_eq!(grid.jitter, HashGridParams::DEFAULT.jitter);
    }

    #[test]
    fn reservoir_buffer_size_is_capacity_times_stride() {
        let settings = PrismWorldRestirSettings {
            capacity: 1024,
            ..Default::default()
        };
        assert_eq!(
            settings.reservoir_buffer_size(),
            1024 * WORLD_RESTIR_RESERVOIR_STRIDE
        );
    }

    #[test]
    fn reservoir_buffer_size_floors_capacity_at_one() {
        let settings = PrismWorldRestirSettings {
            capacity: 0,
            ..Default::default()
        };
        assert_eq!(
            settings.reservoir_buffer_size(),
            WORLD_RESTIR_RESERVOIR_STRIDE
        );
    }

    #[test]
    fn settings_feed_the_fill_immediate_block() {
        // The settings drive the abi builder end to end: the fill block reads
        // the grid tunables and the reuse budget straight off the resource.
        let settings = PrismWorldRestirSettings::default();
        let params =
            GpuWorldRestirFillParams::from_settings(Vec3::ZERO, Vec3::ZERO, 3, 9, &settings);
        assert_eq!(params.base_cell_size, settings.base_cell_size);
        assert_eq!(params.level_scale, settings.level_scale);
        assert_eq!(params.normal_resolution, settings.normal_resolution);
        assert_eq!(params.m_cap, settings.m_cap);
        assert_eq!(params.spatial_samples, settings.spatial_samples);
        assert_eq!(params.spatial_radius, settings.spatial_radius);
    }
}
