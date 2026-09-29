//! Jacobi (parallel-safe) self-collision resolver — the GPU-faithful golden.
//!
//! [`super::self_collision::resolve_self_collision`] resolves strand
//! self-collision in **Gauss-Seidel** order: it walks particles in index order
//! and each correction is applied in place, so a later particle already sees
//! the moved position of an earlier one. That is the sequential CPU reference,
//! but it does not map to a `GPU` compute kernel: on the `GPU` every particle
//! is updated in parallel from the *same* read-only snapshot, which is a
//! **Jacobi** iteration, not Gauss-Seidel. The two converge to the same
//! separated state but are not bit-for-bit identical on any single pass, so a
//! faithful `hair_self_collision.wesl` twin needs its own golden rather than
//! borrowing the Gauss-Seidel one (design §9: no fake parity).
//!
//! This module owns that Jacobi golden purely and deterministically. Each
//! iteration reads positions through a prebuilt
//! [`super::self_collision_grid::UniformGrid`], accumulates every particle's
//! total correction against its neighbors from that read-only snapshot, and
//! only then applies all corrections at once. Because one invocation owns one
//! particle and reads only neighbor positions, [`accumulate_jacobi_corrections`]
//! is exactly the body of a per-particle `GPU` kernel; the `GPU` twin can mirror
//! it value-for-value. Corrections are summed in ascending neighbor-index order
//! (the grid returns sorted neighbors) so the float reduction is deterministic.
//!
//! Nothing samples a real random source and no input panics: non-finite
//! particles neither push nor are pushed, coincident particles (no separating
//! direction) are skipped, pinned particles receive zero correction because
//! each term is weighted by inverse mass, and degenerate parameters make the
//! whole call a no-op — matching the guard in
//! [`super::self_collision::resolve_self_collision`].

use alloc::vec::Vec;

use super::dynamics::{StrandParticle, Vec3};
use super::self_collision::SelfCollisionParams;
use super::self_collision_grid::{grid_cell_of, UniformGrid};

/// Vectors whose squared length is below this are treated as zero-length.
///
/// Kept in lockstep with [`super::self_collision`] so both resolvers agree on
/// which pairs are coincident.
const EPS_LEN_SQ: f32 = 1.0e-24;

/// Returns whether a position is fully finite on every axis.
#[must_use]
fn is_finite(p: Vec3) -> bool {
    p.x.is_finite() && p.y.is_finite() && p.z.is_finite()
}

/// Accumulates each particle's total Jacobi self-collision correction.
///
/// For every finite particle `i`, this gathers its neighbors from the 27 cells
/// around it in `grid`, and for each finite neighbor `j != i` whose center is
/// closer than the collision diameter `2 * particle_radius`, adds `i`'s
/// inverse-mass-weighted share of the separating push to `out[i]`. All reads
/// are from the *current* particle positions (never from `out`), so the result
/// is independent of evaluation order and maps directly to a per-invocation
/// `GPU` kernel.
///
/// `out` is resized to `particles.len()` and fully overwritten (cleared to
/// [`Vec3::ZERO`] first). The call writes only corrections; it does not move
/// any particle — see [`apply_corrections`]. A degenerate `cell_size` or
/// non-positive `particle_radius`/`stiffness` yields all-zero corrections.
pub fn accumulate_jacobi_corrections(
    particles: &[StrandParticle],
    grid: &UniformGrid,
    params: SelfCollisionParams,
    out: &mut Vec<Vec3>,
) {
    out.clear();
    out.resize(particles.len(), Vec3::ZERO);

    if params.particle_radius <= 0.0
        || params.stiffness <= 0.0
        || params.cell_size <= 0.0
        || !params.cell_size.is_finite()
    {
        return;
    }

    let min_sep = 2.0 * params.particle_radius;
    let min_sep_sq = min_sep * min_sep;
    let stiffness = params.stiffness.clamp(0.0, 1.0);
    let cell_size = grid.cell_size();

    let mut neighbors: Vec<u32> = Vec::new();
    for (i, particle) in particles.iter().enumerate() {
        let pi = particle.position;
        if !is_finite(pi) {
            continue;
        }
        let wi = particle.inverse_mass.max(0.0);
        if wi <= 0.0 {
            continue; // pinned: absorbs no correction.
        }

        grid.neighbors(grid_cell_of(pi, cell_size), &mut neighbors);
        let mut delta = Vec3::ZERO;
        for &j in &neighbors {
            let j = j as usize;
            if j == i {
                continue; // never collide with self.
            }
            let pj = particles[j].position;
            if !is_finite(pj) {
                continue;
            }
            let d = pi.sub(pj);
            let dist_sq = d.length_squared();
            if dist_sq >= min_sep_sq || dist_sq < EPS_LEN_SQ {
                continue; // far enough, or coincident (no separating axis).
            }
            let wj = particles[j].inverse_mass.max(0.0);
            let w = wi + wj;
            if w <= 0.0 {
                continue;
            }
            let dist = dist_sq.sqrt();
            let overlap = min_sep - dist;
            let normal = d.scale(1.0 / dist);
            // This particle's inverse-mass share of the full separating push.
            delta = delta.add(normal.scale(overlap * stiffness * (wi / w)));
        }
        out[i] = delta;
    }
}

/// Adds each accumulated correction to its particle in place.
///
/// Corrections shorter than `out.len()` leave the tail untouched; non-finite
/// particles are skipped so a stray correction cannot revive a NaN position.
pub fn apply_corrections(particles: &mut [StrandParticle], corrections: &[Vec3]) {
    let n = particles.len().min(corrections.len());
    for i in 0..n {
        let p = particles[i].position;
        if is_finite(p) {
            particles[i].position = p.add(corrections[i]);
        }
    }
}

/// Runs `iterations` Jacobi self-collision passes over the particle array.
///
/// Each iteration rebuilds the uniform grid from the current positions (a
/// [`super::self_collision_grid::UniformGrid`] build plus one
/// [`accumulate_jacobi_corrections`] pass, mirroring a `GPU` build+resolve
/// dispatch pair) and then applies every correction simultaneously. Unlike the
/// Gauss-Seidel [`super::self_collision::resolve_self_collision`], within a pass
/// no particle sees another's updated position, which is the behavior a
/// parallel `GPU` kernel produces.
///
/// The call is a no-op for an empty array, zero iterations, or non-positive
/// radius/stiffness/cell size.
pub fn resolve_self_collision_jacobi(
    particles: &mut [StrandParticle],
    params: SelfCollisionParams,
    iterations: u32,
) {
    if particles.is_empty()
        || iterations == 0
        || params.particle_radius <= 0.0
        || params.stiffness <= 0.0
        || params.cell_size <= 0.0
        || !params.cell_size.is_finite()
    {
        return;
    }

    let mut corrections: Vec<Vec3> = Vec::new();
    for _ in 0..iterations {
        let grid = UniformGrid::build(particles, params.cell_size);
        accumulate_jacobi_corrections(particles, &grid, params, &mut corrections);
        apply_corrections(particles, &corrections);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn params() -> SelfCollisionParams {
        SelfCollisionParams {
            particle_radius: 0.5,
            stiffness: 1.0,
            cell_size: 1.0,
        }
    }

    #[test]
    fn no_op_on_degenerate_inputs() {
        let mut none: [StrandParticle; 0] = [];
        resolve_self_collision_jacobi(&mut none, params(), 4);

        let base = [
            StrandParticle::free(Vec3::new(0.0, 0.0, 0.0)),
            StrandParticle::free(Vec3::new(0.1, 0.0, 0.0)),
        ];
        // Zero iterations, or any non-positive parameter, leaves positions put.
        for (p, iters) in [
            (params(), 0u32),
            (
                SelfCollisionParams {
                    particle_radius: 0.0,
                    ..params()
                },
                4,
            ),
            (
                SelfCollisionParams {
                    stiffness: 0.0,
                    ..params()
                },
                4,
            ),
            (
                SelfCollisionParams {
                    cell_size: 0.0,
                    ..params()
                },
                4,
            ),
            (
                SelfCollisionParams {
                    cell_size: f32::NAN,
                    ..params()
                },
                4,
            ),
        ] {
            let mut ps = base;
            resolve_self_collision_jacobi(&mut ps, p, iters);
            assert!((ps[0].position.x - 0.0).abs() < 1.0e-9);
            assert!((ps[1].position.x - 0.1).abs() < 1.0e-9);
        }
    }

    #[test]
    fn overlapping_pair_pushed_apart_symmetrically() {
        // Two free equal-mass particles overlapping along +x by min_sep - 0.1.
        let mut ps = [
            StrandParticle::free(Vec3::new(0.0, 0.0, 0.0)),
            StrandParticle::free(Vec3::new(0.1, 0.0, 0.0)),
        ];
        let grid = UniformGrid::build(&ps, params().cell_size);
        let mut corr: Vec<Vec3> = Vec::new();
        accumulate_jacobi_corrections(&ps, &grid, params(), &mut corr);

        // Equal mass ⇒ each moves half the overlap, in opposite directions.
        let overlap = 1.0 - 0.1; // min_sep(=1.0) - dist(=0.1)
        assert!((corr[0].x + overlap * 0.5).abs() < 1.0e-6); // particle 0 pushed -x
        assert!((corr[1].x - overlap * 0.5).abs() < 1.0e-6); // particle 1 pushed +x
                                                             // No motion off the collision axis.
        assert!(corr[0].y.abs() < 1.0e-9 && corr[0].z.abs() < 1.0e-9);

        apply_corrections(&mut ps, &corr);
        let dist = ps[1].position.sub(ps[0].position).length();
        // One full-stiffness Jacobi pass separates the pair to exactly min_sep.
        assert!((dist - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn pinned_partner_absorbs_no_correction() {
        // Particle 0 pinned (inverse_mass 0), particle 1 free and overlapping.
        let mut ps = [
            StrandParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
            StrandParticle::free(Vec3::new(0.1, 0.0, 0.0)),
        ];
        let grid = UniformGrid::build(&ps, params().cell_size);
        let mut corr: Vec<Vec3> = Vec::new();
        accumulate_jacobi_corrections(&ps, &grid, params(), &mut corr);

        // Pinned particle never moves; free partner takes the whole overlap.
        assert!(corr[0].x.abs() < 1.0e-9);
        let overlap = 1.0 - 0.1;
        assert!((corr[1].x - overlap).abs() < 1.0e-6);

        apply_corrections(&mut ps, &corr);
        assert!((ps[0].position.x - 0.0).abs() < 1.0e-9);
    }

    #[test]
    fn coincident_and_non_finite_pairs_are_skipped() {
        // 0 and 1 coincident (no separating axis); 2 is NaN.
        let mut ps = [
            StrandParticle::free(Vec3::new(0.0, 0.0, 0.0)),
            StrandParticle::free(Vec3::new(0.0, 0.0, 0.0)),
            StrandParticle::free(Vec3::new(f32::NAN, 0.0, 0.0)),
        ];
        let grid = UniformGrid::build(&ps, params().cell_size);
        let mut corr: Vec<Vec3> = Vec::new();
        accumulate_jacobi_corrections(&ps, &grid, params(), &mut corr);
        for c in &corr {
            assert!(c.x.abs() < 1.0e-12 && c.y.abs() < 1.0e-12 && c.z.abs() < 1.0e-12);
        }
        // Full resolve still does not panic or produce NaN on finite ones.
        resolve_self_collision_jacobi(&mut ps, params(), 3);
        assert!(is_finite(ps[0].position) && is_finite(ps[1].position));
    }

    #[test]
    fn jacobi_is_order_independent_within_a_pass() {
        // A symmetric cluster: the accumulated corrections must not depend on
        // the order particles are visited, which is the whole point of Jacobi.
        let build = || {
            [
                StrandParticle::free(Vec3::new(0.0, 0.0, 0.0)),
                StrandParticle::free(Vec3::new(0.15, 0.0, 0.0)),
                StrandParticle::free(Vec3::new(0.0, 0.15, 0.0)),
                StrandParticle::free(Vec3::new(0.15, 0.15, 0.0)),
            ]
        };
        let forward = build();
        let grid_f = UniformGrid::build(&forward, params().cell_size);
        let mut corr_f: Vec<Vec3> = Vec::new();
        accumulate_jacobi_corrections(&forward, &grid_f, params(), &mut corr_f);

        // Reversed storage order, then map corrections back to original index.
        let mut reversed = build();
        reversed.reverse();
        let grid_r = UniformGrid::build(&reversed, params().cell_size);
        let mut corr_r: Vec<Vec3> = Vec::new();
        accumulate_jacobi_corrections(&reversed, &grid_r, params(), &mut corr_r);

        let n = corr_f.len();
        for i in 0..n {
            let mirrored = corr_r[n - 1 - i];
            assert!((corr_f[i].x - mirrored.x).abs() < 1.0e-6);
            assert!((corr_f[i].y - mirrored.y).abs() < 1.0e-6);
            assert!((corr_f[i].z - mirrored.z).abs() < 1.0e-6);
        }
    }

    #[test]
    fn iterations_monotonically_separate_a_pair() {
        let start = [
            StrandParticle::free(Vec3::new(0.0, 0.0, 0.0)),
            StrandParticle::free(Vec3::new(0.2, 0.0, 0.0)),
        ];
        let soft = SelfCollisionParams {
            stiffness: 0.5,
            ..params()
        };
        let mut prev_gap = 0.2f32;
        let mut ps = start;
        for _ in 0..5 {
            resolve_self_collision_jacobi(&mut ps, soft, 1);
            let gap = ps[1].position.sub(ps[0].position).length();
            assert!(gap >= prev_gap - 1.0e-9); // never regresses
            assert!(gap <= 1.0 + 1.0e-6); // never overshoots min_sep
            prev_gap = gap;
        }
    }
}
