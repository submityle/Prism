//! Approximate strand self-collision via a uniform spatial hash.
//!
//! Body-proxy collision (see [`super::collision`]) keeps hair off the face and
//! shoulders, but it does nothing to stop a groom from passing through
//! *itself*: without self-collision a thick braid or a wind-blown fringe
//! collapses into a flat sheet as strands freely interpenetrate. Exact
//! all-pairs strand collision is O(n^2) and far too expensive for a groom of
//! hundreds of thousands of segments, so production hair uses a spatial
//! acceleration structure and treats particles as small spheres that softly
//! repel their neighbors (design §6.2 / §8 "自碰撞近似", `TressFX` 4 style).
//!
//! This module owns that approximation purely and deterministically
//! (design §9): particles are bucketed into a uniform grid keyed by integer
//! cell, and each particle is pushed apart only from the neighbors in its own
//! and adjacent cells. That bounds the work to the local density instead of the
//! global count. The grid is an ordered [`BTreeMap`], candidate neighbors are
//! visited in sorted index order, and corrections are applied in place
//! (Gauss-Seidel), so the same input array always produces the same output.
//! Nothing samples a real random source and no input panics: non-finite
//! particles are skipped, coincident particles (no separating direction) are
//! left alone, and two pinned particles never move.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use super::dynamics::{StrandParticle, Vec3};

/// Tuning for the self-collision pass.
///
/// `cell_size` should be at least the collision diameter (`2 * particle_radius`)
/// so that any colliding pair lands in the same or an adjacent cell; a larger
/// cell still works but visits more candidates.
#[derive(Clone, Copy, Debug)]
pub struct SelfCollisionParams {
    /// Half the minimum separation: two particles collide when their centers
    /// are closer than `2 * particle_radius`.
    pub particle_radius: f32,
    /// Fraction of each overlap resolved per call, in `0..=1`. `1` separates
    /// colliding pairs fully in one pass; smaller values relax gradually over
    /// frames for stability.
    pub stiffness: f32,
    /// Edge length of a grid cell in world units. Neighbors are searched in the
    /// 27 cells around each particle.
    pub cell_size: f32,
}

/// Vectors shorter than the square root of this are treated as zero-length.
const EPS_LEN_SQ: f32 = 1.0e-24;

/// Integer grid cell coordinate.
type Cell = (i32, i32, i32);

/// Maps a finite world position to its grid cell for the given cell size.
fn cell_of(position: Vec3, cell_size: f32) -> Cell {
    let inv = 1.0 / cell_size;
    (
        (position.x * inv).floor() as i32,
        (position.y * inv).floor() as i32,
        (position.z * inv).floor() as i32,
    )
}

/// Applies one pass of approximate self-collision to a particle array.
///
/// Particles are treated as spheres of `params.particle_radius`; any two whose
/// centers are closer than the collision diameter are pushed apart along their
/// center line, weighted by inverse mass so a pinned particle stays put and its
/// free partner absorbs the whole correction. Pinned/pinned and coincident
/// pairs are skipped.
///
/// This is a standalone per-frame service, run after the strand solve (like
/// [`super::collision::resolve_strand_collisions`]) rather than inside a
/// substep, because self-collision is a broad approximation, not a hard
/// constraint. The call is a no-op for an empty array or non-positive
/// radius/stiffness/cell size.
pub fn resolve_self_collision(particles: &mut [StrandParticle], params: SelfCollisionParams) {
    if particles.is_empty()
        || params.particle_radius <= 0.0
        || params.stiffness <= 0.0
        || params.cell_size <= 0.0
        || !params.cell_size.is_finite()
    {
        return;
    }

    let min_sep = 2.0 * params.particle_radius;
    let min_sep_sq = min_sep * min_sep;
    let stiffness = params.stiffness.clamp(0.0, 1.0);

    // Bucket every finite particle into its grid cell, in index order so each
    // bucket's contents stay sorted ascending.
    let mut grid: BTreeMap<Cell, Vec<usize>> = BTreeMap::new();
    for (index, particle) in particles.iter().enumerate() {
        let p = particle.position;
        if p.x.is_finite() && p.y.is_finite() && p.z.is_finite() {
            grid.entry(cell_of(p, params.cell_size))
                .or_default()
                .push(index);
        }
    }

    // Resolve each particle against higher-indexed neighbors in the 27 cells
    // around it. Visiting only `j > i` counts every pair once; gathering and
    // sorting candidates keeps the correction order deterministic.
    let count = particles.len();
    let mut candidates: Vec<usize> = Vec::new();
    for i in 0..count {
        let pi = particles[i].position;
        if !(pi.x.is_finite() && pi.y.is_finite() && pi.z.is_finite()) {
            continue;
        }
        let (cx, cy, cz) = cell_of(pi, params.cell_size);

        candidates.clear();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(bucket) = grid.get(&(
                        cx.saturating_add(dx),
                        cy.saturating_add(dy),
                        cz.saturating_add(dz),
                    )) {
                        for &j in bucket {
                            if j > i {
                                candidates.push(j);
                            }
                        }
                    }
                }
            }
        }
        candidates.sort_unstable();

        for &j in &candidates {
            let a = particles[i].position;
            let b = particles[j].position;
            let d = a.sub(b);
            let dist_sq = d.length_squared();
            if dist_sq >= min_sep_sq || dist_sq < EPS_LEN_SQ {
                // Far enough apart, or coincident (no separating direction).
                continue;
            }
            let wi = particles[i].inverse_mass.max(0.0);
            let wj = particles[j].inverse_mass.max(0.0);
            let w = wi + wj;
            if w <= 0.0 {
                continue; // both pinned: nothing to move.
            }
            let dist = dist_sq.sqrt();
            let overlap = min_sep - dist;
            let normal = d.scale(1.0 / dist);
            let correction = normal.scale(overlap * stiffness);
            // Split the push by inverse mass: the lighter (freer) particle moves
            // more; a pinned partner (weight 0) does not move at all.
            particles[i].position = particles[i].position.add(correction.scale(wi / w));
            particles[j].position = particles[j].position.sub(correction.scale(wj / w));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn params() -> SelfCollisionParams {
        SelfCollisionParams {
            particle_radius: 0.5,
            stiffness: 1.0,
            cell_size: 1.0,
        }
    }

    #[test]
    fn overlapping_free_particles_separate() {
        let mut ps = vec![
            StrandParticle::free(Vec3::new(0.0, 0.0, 0.0)),
            StrandParticle::free(Vec3::new(0.4, 0.0, 0.0)),
        ];
        resolve_self_collision(&mut ps, params());
        let sep = ps[0].position.sub(ps[1].position).length();
        // Min separation is 2*radius = 1.0; stiffness 1 resolves it fully.
        assert!(sep >= 1.0 - 1.0e-4, "separation {sep} below min");
        // Symmetric split: each moved the same distance from the midpoint.
        assert!((ps[0].position.x + ps[1].position.x - 0.4).abs() < 1.0e-5);
    }

    #[test]
    fn pinned_partner_absorbs_no_push() {
        let mut ps = vec![
            StrandParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
            StrandParticle::free(Vec3::new(0.4, 0.0, 0.0)),
        ];
        resolve_self_collision(&mut ps, params());
        // Pinned particle stays exactly put; the free one moves the full gap.
        assert!(ps[0].position.length_squared() < 1.0e-12);
        let sep = ps[0].position.sub(ps[1].position).length();
        assert!(sep >= 1.0 - 1.0e-4, "separation {sep} below min");
    }

    #[test]
    fn distant_particles_are_untouched() {
        let mut ps = vec![
            StrandParticle::free(Vec3::new(0.0, 0.0, 0.0)),
            StrandParticle::free(Vec3::new(5.0, 0.0, 0.0)),
        ];
        resolve_self_collision(&mut ps, params());
        assert!(ps[0].position.length_squared() < 1.0e-12);
        assert!((ps[1].position.x - 5.0).abs() < 1.0e-6);
    }

    #[test]
    fn extreme_finite_positions_do_not_overflow() {
        // A finite but extreme coordinate saturates to the i32 cell boundary in
        // `cell_of`; gathering the 27-cell neighborhood must use saturating
        // offsets so `cx + 1` / `cx - 1` never overflows. This particle sits
        // alone in a saturated cell, so it stays put, but the call must not
        // panic (honors the no-op/no-panic contract for finite input).
        let mut ps = vec![
            StrandParticle::free(Vec3::new(1.0e30, -1.0e30, 1.0e30)),
            StrandParticle::free(Vec3::new(0.0, 0.0, 0.0)),
        ];
        resolve_self_collision(&mut ps, params());
        // Extreme particle is alone in its saturated cell: unchanged.
        assert!((ps[0].position.x - 1.0e30).abs() <= 1.0e24);
        assert!(ps[1].position.length_squared() < 1.0e-12);
    }

    #[test]
    fn two_pinned_overlapping_do_not_move() {
        let mut ps = vec![
            StrandParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
            StrandParticle::pinned(Vec3::new(0.3, 0.0, 0.0)),
        ];
        resolve_self_collision(&mut ps, params());
        assert!(ps[0].position.length_squared() < 1.0e-12);
        assert!((ps[1].position.x - 0.3).abs() < 1.0e-6);
    }

    #[test]
    fn coincident_particles_do_not_panic() {
        let mut ps = vec![
            StrandParticle::free(Vec3::ZERO),
            StrandParticle::free(Vec3::ZERO),
        ];
        resolve_self_collision(&mut ps, params());
        // No separating direction: left untouched, must not divide by zero.
        assert!(ps[0].position.length_squared() < 1.0e-12);
        assert!(ps[1].position.length_squared() < 1.0e-12);
    }

    #[test]
    fn empty_and_bad_params_are_no_ops() {
        let mut empty: Vec<StrandParticle> = Vec::new();
        resolve_self_collision(&mut empty, params());
        assert!(empty.is_empty());

        let ps = vec![
            StrandParticle::free(Vec3::ZERO),
            StrandParticle::free(Vec3::new(0.1, 0.0, 0.0)),
        ];
        let before: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
        for bad in [
            SelfCollisionParams {
                particle_radius: 0.0,
                stiffness: 1.0,
                cell_size: 1.0,
            },
            SelfCollisionParams {
                particle_radius: 0.5,
                stiffness: 0.0,
                cell_size: 1.0,
            },
            SelfCollisionParams {
                particle_radius: 0.5,
                stiffness: 1.0,
                cell_size: 0.0,
            },
        ] {
            let mut copy = ps.clone();
            resolve_self_collision(&mut copy, bad);
            for (p, b) in copy.iter().zip(before.iter()) {
                assert!((p.position.sub(*b)).length_squared() < 1.0e-12);
            }
        }
    }

    #[test]
    fn non_finite_particle_is_skipped() {
        let mut ps = vec![
            StrandParticle::free(Vec3::new(f32::NAN, 0.0, 0.0)),
            StrandParticle::free(Vec3::new(0.0, 0.0, 0.0)),
            StrandParticle::free(Vec3::new(0.4, 0.0, 0.0)),
        ];
        // The NaN particle is ignored; the finite overlapping pair still splits.
        resolve_self_collision(&mut ps, params());
        let sep = ps[1].position.sub(ps[2].position).length();
        assert!(
            sep >= 1.0 - 1.0e-4,
            "finite pair separation {sep} below min"
        );
    }

    #[test]
    fn stiffness_scales_partial_resolution() {
        let mut ps = vec![
            StrandParticle::free(Vec3::new(0.0, 0.0, 0.0)),
            StrandParticle::free(Vec3::new(0.4, 0.0, 0.0)),
        ];
        let soft = SelfCollisionParams {
            particle_radius: 0.5,
            stiffness: 0.5,
            cell_size: 1.0,
        };
        resolve_self_collision(&mut ps, soft);
        let sep = ps[0].position.sub(ps[1].position).length();
        // Overlap was 0.6; half-resolved leaves separation ~0.4 + 0.3 = 0.7.
        assert!(
            sep > 0.4 && sep < 1.0,
            "partial separation {sep} out of range"
        );
    }
}
