//! Seed-site scattering strategies that shape the fracture pattern.
//!
//! The Voronoi decomposition is driven entirely by where the seed sites land:
//! a uniform cloud yields evenly sized shards, whereas concentrating seeds near
//! an impact point yields many small fragments at the contact and larger ones
//! far away, matching how real brittle materials shatter.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Uniform
//! and clustered point sampling are elementary, publicly documented techniques.

use glam::Vec3;

use crate::fracture::config::FractureConfig;
use crate::fracture::predicates::symbolic_perturbation;
use crate::fracture::rng::DeterministicRng;
use crate::math::scalar::Real;

/// Scatters `config.seed_count` sites uniformly inside the box `min..=max`.
#[must_use]
pub fn scatter_uniform(min: Vec3, max: Vec3, config: &FractureConfig) -> Vec<Vec3> {
    let mut rng = DeterministicRng::new(config.rng_seed);
    let mut sites = Vec::with_capacity(config.seed_count);
    for _ in 0..config.seed_count {
        sites.push(rng.next_in_box(min, max));
    }
    apply_jitter(&mut sites, config.seed_jitter);
    sites
}

/// Scatters sites with a fraction concentrated near `impact` (within
/// `config.impact_cluster_radius`) and the remainder uniform in `min..=max`.
///
/// Clustered sites are always clamped back inside the box so no seed escapes
/// the shape being fractured.
#[must_use]
pub fn scatter_impact(min: Vec3, max: Vec3, impact: Vec3, config: &FractureConfig) -> Vec<Vec3> {
    let mut rng = DeterministicRng::new(config.rng_seed);
    let clustered = clustered_count(config);
    let mut sites = Vec::with_capacity(config.seed_count);

    for _ in 0..clustered {
        let offset = Vec3::new(
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
        ) * config.impact_cluster_radius;
        sites.push((impact + offset).clamp(min, max));
    }
    for _ in clustered..config.seed_count {
        sites.push(rng.next_in_box(min, max));
    }
    apply_jitter(&mut sites, config.seed_jitter);
    sites
}

/// Number of clustered seeds implied by the impact-cluster fraction.
#[must_use]
pub fn clustered_count(config: &FractureConfig) -> usize {
    let frac = config.impact_cluster_fraction.clamp(0.0, 1.0);
    let n = (config.seed_count as Real) * frac;
    (n as usize).min(config.seed_count)
}

/// Adds a deterministic per-index symbolic perturbation to every site so that
/// structured (e.g. grid-aligned) seed sets are nudged into general position.
pub fn apply_jitter(sites: &mut [Vec3], magnitude: Real) {
    if magnitude <= 0.0 {
        return;
    }
    for (index, site) in sites.iter_mut().enumerate() {
        *site += symbolic_perturbation(index, magnitude);
    }
}
