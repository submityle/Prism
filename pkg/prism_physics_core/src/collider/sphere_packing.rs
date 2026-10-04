//! Deterministic polydisperse sphere packing for granular scene setup.
//!
//! The DEM stack —
//! [`SphereDemIntegrator`](super::sphere_dem_integrator::SphereDemIntegrator),
//! [`SphereDemFrictionIntegrator`](super::sphere_dem_friction_integrator::SphereDemFrictionIntegrator)
//! and the [`SphereBoundaryDriver`](super::sphere_boundary_driver::SphereBoundaryDriver)
//! — all consume a cloud of grains described by parallel position and radius
//! arrays. Authoring those arrays by hand is tedious and error-prone: a real
//! hopper, drum or soil column needs hundreds to thousands of grains of varied
//! size that start out **mutually non-overlapping**, otherwise the first step
//! explodes as the contact law resolves the initial interpenetration.
//!
//! This module generates exactly such an initial condition: a reproducible
//! polydisperse packing inside an axis-aligned box in which no two grains
//! overlap and every grain lies fully inside the box.
//!
//! # Algorithm
//!
//! Packing uses **dart throwing** (rejection sampling): a radius is drawn
//! uniformly from `[radius_min, radius_max]`, a centre is drawn uniformly from
//! the box shrunk by that radius (so the grain is guaranteed to fit), and the
//! candidate is accepted only if its surface clears every already-placed grain
//! by at least `separation`. Sampling continues until `target_count` grains are
//! placed or the `max_attempts` dart budget is spent, so a dense request that
//! cannot be satisfied returns an honest, smaller packing rather than a
//! fabricated one. Overlap testing is the exact `O(n²)` pairwise check, which
//! is appropriate for the hundreds-to-thousands grain counts typical of an
//! initial condition.
//!
//! Randomness comes from the crate's own [`DeterministicRng`] seeded by the
//! caller, so a given [`SpherePackingParams`] always yields the identical
//! packing across machines — a prerequisite for baked caches and networked
//! determinism. Nothing here is derived from Unreal Engine source.

use crate::fracture::rng::DeterministicRng;
use glam::Vec3;

/// Tuning for [`pack_spheres`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpherePackingParams {
    /// Lower corner of the axis-aligned packing box.
    pub min: Vec3,
    /// Upper corner of the axis-aligned packing box; strictly greater than
    /// `min` on every axis.
    pub max: Vec3,
    /// Smallest grain radius to draw; strictly positive.
    pub radius_min: f32,
    /// Largest grain radius to draw; at least `radius_min`. The box must be at
    /// least `2·radius_max` wide on every axis.
    pub radius_max: f32,
    /// Desired number of grains. The result may contain fewer if the dart
    /// budget is exhausted first.
    pub target_count: usize,
    /// Total dart-throw budget. Must be at least `target_count`.
    pub max_attempts: usize,
    /// Extra clearance required between grain surfaces; non-negative.
    pub separation: f32,
    /// Seed for the deterministic generator.
    pub seed: u64,
}

/// A reproducible non-overlapping sphere packing.
///
/// `positions` and `radii` are indexed in lockstep and are ready to feed to the
/// DEM integrators and the boundary driver.
#[derive(Clone, Debug, PartialEq)]
pub struct SpherePacking {
    positions: Vec<Vec3>,
    radii: Vec<f32>,
}

impl SpherePacking {
    /// Grain centres, indexed in lockstep with [`SpherePacking::radii`].
    #[must_use]
    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }

    /// Grain radii, indexed in lockstep with [`SpherePacking::positions`].
    #[must_use]
    pub fn radii(&self) -> &[f32] {
        &self.radii
    }

    /// Number of grains placed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether the packing is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Smallest surface gap `‖cᵢ − cⱼ‖ − rᵢ − rⱼ` over every grain pair. A
    /// successful packing keeps this at or above the requested `separation`.
    /// Returns [`f32::INFINITY`] when fewer than two grains are present.
    #[must_use]
    pub fn min_separation(&self) -> f32 {
        let mut smallest = f32::INFINITY;
        for (i, (&ci, &ri)) in self.positions.iter().zip(self.radii.iter()).enumerate() {
            for (&cj, &rj) in self.positions.iter().zip(self.radii.iter()).skip(i + 1) {
                let gap = ci.distance(cj) - ri - rj;
                smallest = smallest.min(gap);
            }
        }
        smallest
    }
}

/// Generates a non-overlapping polydisperse sphere packing.
///
/// Returns `None` when `params` is invalid: a non-finite or non-strictly-
/// ordered box, a non-positive `radius_min`, `radius_max < radius_min`, a
/// negative `separation`, a zero `target_count`, a `max_attempts` smaller than
/// `target_count`, or a box too small to contain a grain of radius
/// `radius_max` on every axis.
#[must_use]
pub fn pack_spheres(params: &SpherePackingParams) -> Option<SpherePacking> {
    let &SpherePackingParams {
        min,
        max,
        radius_min,
        radius_max,
        target_count,
        max_attempts,
        separation,
        seed,
    } = params;

    if !(min.is_finite() && max.is_finite()) {
        return None;
    }
    if !(radius_min.is_finite() && radius_max.is_finite() && separation.is_finite()) {
        return None;
    }
    if radius_min <= 0.0 || radius_max < radius_min || separation < 0.0 {
        return None;
    }
    if target_count == 0 || max_attempts < target_count {
        return None;
    }
    let extent = max - min;
    if extent.x <= 0.0 || extent.y <= 0.0 || extent.z <= 0.0 {
        return None;
    }
    let diameter = 2.0 * radius_max;
    if extent.x < diameter || extent.y < diameter || extent.z < diameter {
        return None;
    }

    let mut rng = DeterministicRng::new(seed);
    let mut positions: Vec<Vec3> = Vec::with_capacity(target_count);
    let mut radii: Vec<f32> = Vec::with_capacity(target_count);
    let mut attempts = 0_usize;

    while positions.len() < target_count && attempts < max_attempts {
        attempts += 1;
        let radius = rng.next_range(radius_min, radius_max);
        let lo = min + Vec3::splat(radius);
        let hi = max - Vec3::splat(radius);
        // A sampled radius can never exceed radius_max, so the shrunk box is
        // always non-empty; guard defensively regardless.
        if lo.x > hi.x || lo.y > hi.y || lo.z > hi.z {
            continue;
        }
        let center = rng.next_in_box(lo, hi);
        let mut accepted = true;
        for (&other_center, &other_radius) in positions.iter().zip(radii.iter()) {
            let min_dist = radius + other_radius + separation;
            if center.distance_squared(other_center) < min_dist * min_dist {
                accepted = false;
                break;
            }
        }
        if accepted {
            positions.push(center);
            radii.push(radius);
        }
    }

    Some(SpherePacking { positions, radii })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_params() -> SpherePackingParams {
        SpherePackingParams {
            min: Vec3::ZERO,
            max: Vec3::splat(10.0),
            radius_min: 0.4,
            radius_max: 0.6,
            target_count: 50,
            max_attempts: 20_000,
            separation: 0.05,
            seed: 42,
        }
    }

    #[test]
    fn rejects_invalid_params() {
        let mut p = base_params();
        p.max = Vec3::new(-1.0, 10.0, 10.0);
        assert!(pack_spheres(&p).is_none());

        let mut p = base_params();
        p.radius_min = 0.0;
        assert!(pack_spheres(&p).is_none());

        let mut p = base_params();
        p.radius_max = 0.1; // below radius_min
        assert!(pack_spheres(&p).is_none());

        let mut p = base_params();
        p.separation = -1.0;
        assert!(pack_spheres(&p).is_none());

        let mut p = base_params();
        p.target_count = 0;
        assert!(pack_spheres(&p).is_none());

        let mut p = base_params();
        p.max_attempts = p.target_count - 1;
        assert!(pack_spheres(&p).is_none());

        // Box narrower than one grain diameter on an axis.
        let mut p = base_params();
        p.max = Vec3::new(0.5, 10.0, 10.0);
        assert!(pack_spheres(&p).is_none());

        let mut p = base_params();
        p.radius_min = f32::NAN;
        assert!(pack_spheres(&p).is_none());
    }

    #[test]
    fn packs_requested_count_without_overlap() {
        let p = base_params();
        let packing = pack_spheres(&p).expect("packing");
        assert_eq!(packing.len(), p.target_count);
        assert_eq!(packing.positions().len(), packing.radii().len());
        // No pair interpenetrates and the requested clearance is honoured.
        assert!(packing.min_separation() >= p.separation - 1.0e-4);
    }

    #[test]
    fn every_grain_lies_inside_the_box() {
        let p = base_params();
        let packing = pack_spheres(&p).expect("packing");
        for (&c, &r) in packing.positions().iter().zip(packing.radii().iter()) {
            assert!(c.x - r >= p.min.x - 1.0e-4 && c.x + r <= p.max.x + 1.0e-4);
            assert!(c.y - r >= p.min.y - 1.0e-4 && c.y + r <= p.max.y + 1.0e-4);
            assert!(c.z - r >= p.min.z - 1.0e-4 && c.z + r <= p.max.z + 1.0e-4);
            assert!((p.radius_min..=p.radius_max).contains(&r));
        }
    }

    #[test]
    fn is_deterministic_for_a_given_seed() {
        let p = base_params();
        let a = pack_spheres(&p).expect("a");
        let b = pack_spheres(&p).expect("b");
        assert_eq!(a, b);

        let mut q = base_params();
        q.seed = 1337;
        let c = pack_spheres(&q).expect("c");
        // A different seed should move at least one grain.
        assert_ne!(a.positions(), c.positions());
    }

    #[test]
    fn monodisperse_request_yields_equal_radii() {
        let mut p = base_params();
        p.radius_min = 0.5;
        p.radius_max = 0.5;
        let packing = pack_spheres(&p).expect("packing");
        for &r in packing.radii() {
            assert!((r - 0.5).abs() <= 1.0e-6);
        }
    }

    #[test]
    fn dense_request_returns_honest_smaller_packing() {
        // Ask for far more grains than fit, with a tight dart budget: the
        // result is capped by geometry, never fabricated, and still valid.
        let mut p = base_params();
        p.target_count = 100_000;
        p.max_attempts = 100_000;
        let packing = pack_spheres(&p).expect("packing");
        assert!(packing.len() < p.target_count);
        assert!(!packing.is_empty());
        assert!(packing.min_separation() >= p.separation - 1.0e-4);
    }
}
