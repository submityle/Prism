//! Tunables that shape a fracture pattern.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. These are
//! plain authoring parameters with no algorithmic content.

use crate::math::scalar::Real;

/// Parameters controlling how seed sites are scattered and how the resulting
/// convex cells are filtered.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct FractureConfig {
    /// Total number of seed sites (and therefore fragments before filtering).
    pub seed_count: usize,
    /// Fraction of seeds (in `[0, 1]`) concentrated near the impact point for
    /// [`crate::fracture::pattern::scatter_impact`]; the remainder are uniform.
    pub impact_cluster_fraction: Real,
    /// Radius of the impact cluster in world units.
    pub impact_cluster_radius: Real,
    /// Fragments whose volume is below this threshold are discarded as slivers.
    pub min_fragment_volume: Real,
    /// Magnitude of the deterministic symbolic perturbation applied to seeds
    /// to break exact geometric ties.
    pub seed_jitter: Real,
    /// Seed for the deterministic scatter generator.
    pub rng_seed: u64,
}

impl FractureConfig {
    /// Default number of seeds.
    pub const DEFAULT_SEED_COUNT: usize = 16;
    /// Default impact cluster fraction.
    pub const DEFAULT_IMPACT_CLUSTER_FRACTION: Real = 0.6;
    /// Default impact cluster radius.
    pub const DEFAULT_IMPACT_CLUSTER_RADIUS: Real = 0.25;
    /// Default sliver-rejection volume.
    pub const DEFAULT_MIN_FRAGMENT_VOLUME: Real = 1e-6;
    /// Default symbolic-perturbation magnitude.
    pub const DEFAULT_SEED_JITTER: Real = 1e-4;
    /// Default generator seed.
    pub const DEFAULT_RNG_SEED: u64 = 0x5EED;
}

impl Default for FractureConfig {
    fn default() -> Self {
        FractureConfig {
            seed_count: Self::DEFAULT_SEED_COUNT,
            impact_cluster_fraction: Self::DEFAULT_IMPACT_CLUSTER_FRACTION,
            impact_cluster_radius: Self::DEFAULT_IMPACT_CLUSTER_RADIUS,
            min_fragment_volume: Self::DEFAULT_MIN_FRAGMENT_VOLUME,
            seed_jitter: Self::DEFAULT_SEED_JITTER,
            rng_seed: Self::DEFAULT_RNG_SEED,
        }
    }
}
