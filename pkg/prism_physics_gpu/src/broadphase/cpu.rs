//! `CPU` golden twin of the spatial-hash broad phase.
//!
//! This is the reference the `GPU` kernel is measured against. It runs the two
//! stages the kernel runs — populate the hash grid, then scan each particle's
//! `3x3x3` neighbourhood — using the shared [`hash`](super::hash) math and the
//! same fixed capacities from [`BroadphaseConfig`], so its candidate-pair *set*
//! is identical to the device's. It is itself validated against an independent
//! brute-force all-pairs reference in the unit tests, so correctness does not
//! rest on the `GPU` agreeing with it.
//!
//! # Duplicate avoidance
//!
//! A hash bucket can hold particles from several distinct cells (hash
//! collisions), and a neighbour scan visits `27` cell coordinates. To emit each
//! pair exactly once, a candidate `j` is accepted only when its *actual* cell
//! equals the specific neighbour coordinate currently being scanned, so `j` is
//! considered under precisely one of the `27` coordinates.
//!
//! Provenance: Teschner et al. 2003. No Unreal Engine source or derived code.

use super::config::{BroadphaseConfig, BroadphaseError};
use super::hash::{cell_coord, hash_cell};
use super::pair::CandidatePair;
use super::particle::Particle;

/// Runs the spatial-hash broad phase on the `CPU`.
///
/// Returns the candidate pairs in discovery order (index-ascending); callers
/// that compare against another path should sort first, since the `GPU` twin
/// emits pairs in nondeterministic atomic-append order.
///
/// # Errors
///
/// Returns [`BroadphaseError`] when the config is invalid, a bucket overflows
/// `max_per_bucket`, or the pair count exceeds `pair_capacity`. These mirror the
/// fixed `GPU` allocations exactly.
pub fn cpu_broadphase(
    particles: &[Particle],
    config: &BroadphaseConfig,
) -> Result<Vec<CandidatePair>, BroadphaseError> {
    config.validate()?;
    let table = config.table_size as usize;
    let stride = config.max_per_bucket as usize;

    let mut counts = vec![0u32; table];
    let mut entries = vec![0u32; table * stride];

    // Stage 1: populate the hash grid.
    for (i, particle) in particles.iter().enumerate() {
        let cell = cell_coord(particle.position, config.cell_size);
        let bucket = hash_cell(cell, config.table_size) as usize;
        let slot = counts[bucket] as usize;
        if slot >= stride {
            return Err(BroadphaseError::BucketOverflow {
                bucket: bucket as u32,
                capacity: config.max_per_bucket,
            });
        }
        entries[bucket * stride + slot] = i as u32;
        counts[bucket] += 1;
    }

    // Stage 2: scan each particle's 3x3x3 neighbourhood.
    let mut pairs = Vec::new();
    for (i, particle) in particles.iter().enumerate() {
        let base = cell_coord(particle.position, config.cell_size);
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let neighbour = [base[0] + dx, base[1] + dy, base[2] + dz];
                    let bucket = hash_cell(neighbour, config.table_size) as usize;
                    let count = counts[bucket] as usize;
                    for slot in 0..count {
                        let j = entries[bucket * stride + slot] as usize;
                        if j <= i {
                            continue;
                        }
                        let other = &particles[j];
                        // Accept `j` only under its own cell coordinate so a
                        // collision-shared bucket cannot emit the pair twice.
                        if cell_coord(other.position, config.cell_size) != neighbour {
                            continue;
                        }
                        if spheres_overlap(particle, other) {
                            pairs.push(CandidatePair::new(i as u32, j as u32));
                            if pairs.len() > config.pair_capacity as usize {
                                return Err(BroadphaseError::PairCapacityExceeded {
                                    capacity: config.pair_capacity,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(pairs)
}

/// Returns whether two bounding spheres overlap (touching counts as overlap).
#[must_use]
fn spheres_overlap(a: &Particle, b: &Particle) -> bool {
    let delta = a.position - b.position;
    let radius_sum = a.radius + b.radius;
    delta.dot(delta) <= radius_sum * radius_sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    /// Independent brute-force all-pairs reference for the exactness anchor.
    fn brute_force(particles: &[Particle]) -> Vec<CandidatePair> {
        let mut pairs = Vec::new();
        for i in 0..particles.len() {
            for j in (i + 1)..particles.len() {
                let delta = particles[i].position - particles[j].position;
                let radius_sum = particles[i].radius + particles[j].radius;
                if delta.dot(delta) <= radius_sum * radius_sum {
                    pairs.push(CandidatePair::new(i as u32, j as u32));
                }
            }
        }
        pairs
    }

    fn sorted(mut pairs: Vec<CandidatePair>) -> Vec<CandidatePair> {
        pairs.sort_unstable();
        pairs
    }

    fn lattice(dim: i32, spacing: f32, radius: f32) -> Vec<Particle> {
        let mut particles = Vec::new();
        for z in 0..dim {
            for y in 0..dim {
                for x in 0..dim {
                    let position = Vec3::new(x as f32, y as f32, z as f32) * spacing;
                    particles.push(Particle::new(position, radius));
                }
            }
        }
        particles
    }

    #[test]
    fn matches_brute_force_on_dense_lattice() {
        // Spacing 1.0, radius 0.6 -> diameter 1.2 < cell_size 1.5, so every
        // overlap is within the 3x3x3 neighbourhood and the hash is exact.
        let particles = lattice(6, 1.0, 0.6);
        let config = BroadphaseConfig::new(1.5, 4096, 64, 1 << 20);
        let hashed = sorted(cpu_broadphase(&particles, &config).expect("valid input"));
        let brute = sorted(brute_force(&particles));
        assert_eq!(hashed, brute);
        assert!(!hashed.is_empty(), "a dense lattice must produce contacts");
    }

    #[test]
    fn matches_brute_force_with_negative_coordinates() {
        let mut particles = lattice(4, 1.0, 0.55);
        for particle in &mut particles {
            particle.position -= Vec3::splat(3.0);
        }
        let config = BroadphaseConfig::new(1.5, 2048, 64, 1 << 20);
        let hashed = sorted(cpu_broadphase(&particles, &config).expect("valid input"));
        let brute = sorted(brute_force(&particles));
        assert_eq!(hashed, brute);
    }

    #[test]
    fn disjoint_particles_produce_no_pairs() {
        let particles = vec![
            Particle::new(Vec3::new(0.0, 0.0, 0.0), 0.4),
            Particle::new(Vec3::new(10.0, 0.0, 0.0), 0.4),
            Particle::new(Vec3::new(0.0, 20.0, 0.0), 0.4),
        ];
        let config = BroadphaseConfig::new(1.0, 256, 8, 64);
        let pairs = cpu_broadphase(&particles, &config).expect("valid input");
        assert!(pairs.is_empty());
    }

    #[test]
    fn bucket_overflow_is_reported() {
        // Nine coincident particles hash to one bucket with capacity 8.
        let particles = vec![Particle::new(Vec3::ZERO, 0.1); 9];
        let config = BroadphaseConfig::new(1.0, 64, 8, 4096);
        let error = cpu_broadphase(&particles, &config).expect_err("overflow");
        assert!(matches!(
            error,
            BroadphaseError::BucketOverflow { capacity: 8, .. }
        ));
    }

    #[test]
    fn invalid_config_is_rejected() {
        let particles = [Particle::new(Vec3::ZERO, 0.1)];
        let bad = BroadphaseConfig::new(0.0, 64, 8, 64);
        assert!(matches!(
            cpu_broadphase(&particles, &bad),
            Err(BroadphaseError::InvalidConfig(_))
        ));
    }
}
