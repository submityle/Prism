//! Configuration bundling the temporal and spatial level-of-detail controllers.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! parameters below describe standard CFL substepping and distance-based LOD.

use crate::math::scalar::Real;

use super::spatial::SpatialController;
use super::temporal::TemporalController;

/// Tunables for adaptive level-of-detail.
///
/// Groups the parameters for the two orthogonal controllers so a caller can
/// keep a single serialisable settings object and build both controllers from
/// it with [`temporal`](Self::temporal) and [`spatial`](Self::spatial).
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct LodConfig {
    /// Target Courant number for adaptive substepping.
    pub cfl_target: Real,
    /// Characteristic length used by the CFL bound.
    pub cell_size: Real,
    /// Lower bound on the adaptive substep count.
    pub min_substeps: u32,
    /// Upper bound on the adaptive substep count.
    pub max_substeps: u32,
    /// Ascending distance thresholds separating spatial tiers.
    pub tier_distances: Vec<Real>,
    /// Per-tier iteration-budget multipliers (`tier_distances.len() + 1` entries;
    /// padded/truncated when the controller is built).
    pub iteration_scales: Vec<Real>,
    /// Floor on the scaled iteration count.
    pub min_iterations: u32,
}

impl LodConfig {
    /// Builds a temporal controller from the substepping parameters.
    #[must_use]
    pub fn temporal(&self) -> TemporalController {
        TemporalController::new(
            self.cfl_target,
            self.cell_size,
            self.min_substeps,
            self.max_substeps,
        )
    }

    /// Builds a spatial controller from the tier parameters.
    #[must_use]
    pub fn spatial(&self) -> SpatialController {
        SpatialController::new(
            self.tier_distances.clone(),
            self.iteration_scales.clone(),
            self.min_iterations,
        )
    }
}

impl Default for LodConfig {
    fn default() -> Self {
        LodConfig {
            cfl_target: 0.5,
            cell_size: 0.1,
            min_substeps: 1,
            max_substeps: 16,
            tier_distances: vec![10.0, 40.0],
            iteration_scales: vec![1.0, 0.5, 0.25],
            min_iterations: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_builds_consistent_controllers() {
        let config = LodConfig::default();
        let temporal = config.temporal();
        assert_eq!(temporal.bounds(), (1, 16));
        let spatial = config.spatial();
        assert_eq!(spatial.tier_count(), 3);
    }

    #[test]
    fn temporal_and_spatial_reflect_config() {
        let config = LodConfig {
            cfl_target: 0.25,
            cell_size: 0.2,
            min_substeps: 2,
            max_substeps: 20,
            tier_distances: vec![1.0, 2.0, 3.0],
            iteration_scales: vec![1.0, 0.75, 0.5, 0.25],
            min_iterations: 3,
        };
        assert_eq!(config.temporal().bounds(), (2, 20));
        assert_eq!(config.spatial().tier_count(), 4);
        assert_eq!(config.spatial().scaled_iterations(1, 100.0), 3);
    }
}
