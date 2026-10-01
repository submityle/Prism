//! Jacobi (parallel-safe) virtual-particle self-collision — the GPU-faithful
//! golden.
//!
//! [`resolve_self_collision_virtual`](super::resolve_self_collision_virtual)
//! resolves the `NvCloth`-style virtual-particle tier in **Gauss-Seidel**
//! order: it walks the sample pairs in a fixed order and each half-correction
//! is scattered onto the real vertices *in place*, so a later pair already sees
//! the moved positions of an earlier one. That is the sequential CPU reference,
//! but it does not map to a `GPU` compute kernel: on the `GPU` every sample is
//! updated in parallel from the *same* read-only snapshot, which is a
//! **Jacobi** iteration, not Gauss-Seidel. The two converge to the same
//! separated state but are never bit-for-bit identical on a single pass, so a
//! faithful cloth-self-collision `WGSL`/`WESL` twin needs its own golden rather
//! than borrowing the Gauss-Seidel one (no fake parity).
//!
//! This module owns that Jacobi golden purely and deterministically:
//!
//! * **Phase 1 (per-sample own-slot).** For every sample `a` (real vertex or
//!   virtual particle), [`accumulate_virtual_jacobi_corrections`] gathers its
//!   27-cell neighborhood from a prebuilt uniform hash and sums *`a`'s own half*
//!   of the separating push against every penetrating neighbor `b != a`,
//!   reading positions only from the frozen input snapshot. One invocation owns
//!   one `sample_dp[a]` slot and never reads another slot, so this phase is
//!   exactly the body of a per-sample `GPU` kernel with no atomics.
//! * **Phase 2 (per-vertex own-slot).** Each real vertex `v` then sums the
//!   barycentric share `weights[k] * inverse_mass_v / eff_a` of `sample_dp[a]`
//!   over the samples `a` that carry `v` as an active vertex, again writing only
//!   its own `out[v]` slot. This is the scatter of
//!   [`super::virtual_particles`] hoisted into a second own-slot kernel.
//!
//! Both phases reduce in strictly ascending index order (ascending
//! [`BTreeMap`] cells, ascending bucket indices, ascending sample indices), so
//! the float reductions are deterministic and the `GPU` twin can mirror the
//! result value-for-value. The call writes only corrections; it never moves a
//! particle — see [`apply_corrections`]. Guards match
//! [`resolve_self_collision_virtual`](super::resolve_self_collision_virtual): a
//! non-positive `cell_size` or `thickness`, an `inverse_masses` length that
//! differs from `positions`, or fewer than two samples, yields all-zero
//! corrections; pairs sharing an active vertex are skipped; pinned samples
//! (`eff <= 0`) receive nothing; coincident samples separate along `+X`.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! virtual-particle technique is the published `NvCloth` method; the Jacobi
//! split into own-slot accumulate/apply phases is a standard parallel
//! position-based-dynamics reformulation.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use glam::Vec3;

use crate::math::scalar::Real;

use super::virtual_particles::{PairScope, Sample, VirtualParticle, shares_active_vertex};
use super::{cell_of, EPS_LEN_SQ};

/// Accumulates each sample's Jacobi self-collision correction, scattered onto
/// the real vertices.
///
/// `out` is resized to `positions.len()` and fully overwritten (cleared to
/// [`Vec3::ZERO`] first); entry `out[v]` is the total correction real vertex
/// `v` should receive this pass. All reads are from the *input* `positions`
/// snapshot (never from `out`), so the result is independent of evaluation
/// order and maps directly to a per-invocation `GPU` kernel. See
/// [`apply_corrections`] to fold the result back into positions.
///
/// `positions` is the particle position column and `inverse_masses` the
/// index-aligned inverse-mass column (`0` marks a pinned particle). `scope`
/// selects [`PairScope::All`] (a self-contained tier: real-vs-real pairs
/// included) or [`PairScope::VirtualOnly`] (augment mode: real-vs-real pairs are
/// left to the friction point-to-point tier). A non-positive
/// `cell_size`/`thickness`, an `inverse_masses` slice whose length differs from
/// `positions`, or fewer than two samples yields all-zero corrections.
pub(crate) fn accumulate_virtual_jacobi_corrections(
    positions: &[Vec3],
    inverse_masses: &[Real],
    virtuals: &[VirtualParticle],
    cell_size: Real,
    thickness: Real,
    scope: PairScope,
    out: &mut Vec<Vec3>,
) {
    out.clear();
    out.resize(positions.len(), Vec3::ZERO);

    if cell_size <= 0.0 || thickness <= 0.0 || inverse_masses.len() != positions.len() {
        return;
    }
    let real_count = positions.len();

    // Fixed sample order: every real particle first, then in-range virtual
    // particles in generation order — identical to the Gauss-Seidel core so the
    // two goldens fold the same sample set.
    let mut samples: Vec<Sample> = Vec::with_capacity(real_count.saturating_add(virtuals.len()));
    for i in 0..real_count {
        samples.push(Sample::real(i as u32));
    }
    for &vp in virtuals {
        let in_range = vp.verts.iter().all(|&v| (v as usize) < real_count);
        if in_range {
            samples.push(Sample::virtual_particle(vp));
        }
    }
    if samples.len() < 2 {
        return;
    }

    // Bucket samples by their frozen position. Indices are pushed in ascending
    // order, so both the cell traversal and per-bucket traversal are stable.
    let mut grid: BTreeMap<(i32, i32, i32), Vec<u32>> = BTreeMap::new();
    for (index, sample) in samples.iter().enumerate() {
        let cell = cell_of(sample.position(positions), cell_size);
        grid.entry(cell).or_default().push(index as u32);
    }

    // Phase 1: each sample owns `sample_dp[a]` and sums only its own half of
    // every penetrating pair, reading the frozen snapshot. This is the body of
    // a per-sample `GPU` kernel: it writes one slot and reads no other slot.
    let thickness_sq = thickness * thickness;
    let mut sample_dp: Vec<Vec3> = Vec::new();
    sample_dp.resize(samples.len(), Vec3::ZERO);
    for (&cell, bucket) in &grid {
        for &a in bucket {
            let ai = a as usize;
            let mut acc = Vec3::ZERO;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let neighbor = (cell.0 + dx, cell.1 + dy, cell.2 + dz);
                        let Some(nbucket) = grid.get(&neighbor) else {
                            continue;
                        };
                        for &b in nbucket {
                            if b == a {
                                continue;
                            }
                            let bi = b as usize;
                            if scope == PairScope::VirtualOnly
                                && ai < real_count
                                && bi < real_count
                            {
                                // Both samples are real vertices; the friction
                                // point-to-point tier already owns this pair.
                                continue;
                            }
                            acc += half_correction(
                                positions,
                                inverse_masses,
                                &samples,
                                ai,
                                bi,
                                thickness,
                                thickness_sq,
                            );
                        }
                    }
                }
            }
            sample_dp[ai] = acc;
        }
    }

    // Phase 2: each real vertex owns `out[v]` and sums the barycentric share of
    // every incident sample's displacement, again from the frozen snapshot.
    // Iterating samples in ascending order gives every vertex a fixed reduction
    // order, so the float sum is deterministic.
    for (ai, sample) in samples.iter().enumerate() {
        let dp = sample_dp[ai];
        if dp.x == 0.0 && dp.y == 0.0 && dp.z == 0.0 {
            continue;
        }
        let eff = sample.inverse_mass_eff(inverse_masses);
        if eff <= 0.0 {
            continue;
        }
        for k in 0..3 {
            let w = sample.weights[k];
            if w == 0.0 {
                continue;
            }
            let j = sample.verts[k] as usize;
            let im = inverse_masses[j].max(0.0);
            if im <= 0.0 {
                continue;
            }
            let coeff = w * im / eff;
            out[j] += dp * coeff;
        }
    }
}

/// Sample `ai`'s own half of the separating push against neighbor `bi`.
///
/// Returns [`Vec3::ZERO`] when the pair shares an active vertex, is farther
/// apart than `thickness`, or is jointly immovable. `dir` points from `ai`
/// toward `bi`, so `ai` is pushed the opposite way, weighted by its inverse
/// mass share; coincident samples separate along `+X`. Positions come from the
/// frozen `positions` snapshot, so the value is order-independent and matches
/// what a `GPU` invocation for `ai` would compute.
fn half_correction(
    positions: &[Vec3],
    inverse_masses: &[Real],
    samples: &[Sample],
    ai: usize,
    bi: usize,
    thickness: Real,
    thickness_sq: Real,
) -> Vec3 {
    let sample_a = samples[ai];
    let sample_b = samples[bi];
    if shares_active_vertex(&sample_a, &sample_b) {
        return Vec3::ZERO;
    }

    let pa = sample_a.position(positions);
    let pb = sample_b.position(positions);
    let delta = pb - pa;
    let dist_sq = delta.length_squared();
    if dist_sq >= thickness_sq {
        return Vec3::ZERO;
    }

    let wa = sample_a.inverse_mass_eff(inverse_masses);
    let wb = sample_b.inverse_mass_eff(inverse_masses);
    let w_sum = wa + wb;
    if w_sum <= 0.0 {
        return Vec3::ZERO;
    }

    let (dir, penetration) = if dist_sq <= EPS_LEN_SQ {
        (Vec3::new(1.0, 0.0, 0.0), thickness)
    } else {
        let dist = dist_sq.sqrt();
        (delta / dist, thickness - dist)
    };

    // `dir` points from A toward B; A is pushed the opposite way by its share.
    dir * (-penetration * (wa / w_sum))
}

/// Adds the accumulated per-vertex corrections into the particle positions.
///
/// This is the trivial "apply" half of a Jacobi step: after
/// [`accumulate_virtual_jacobi_corrections`] has produced `corrections`, every
/// particle advances by its own entry. `corrections` shorter than `positions`
/// leaves the tail untouched; extra entries are ignored.
pub(crate) fn apply_corrections(positions: &mut [Vec3], corrections: &[Vec3]) {
    let n = positions.len().min(corrections.len());
    for i in 0..n {
        positions[i] += corrections[i];
    }
}

/// Runs one Jacobi virtual-particle self-collision pass over the full sample set
/// (real-vs-real included), the parallel-safe twin of
/// [`resolve_self_collision_virtual`](super::resolve_self_collision_virtual).
///
/// One call is a single Jacobi iteration (accumulate from the frozen snapshot,
/// then apply); repeated calls converge to the same separated state the
/// Gauss-Seidel core reaches, without ever depending on evaluation order.
///
/// `positions` is the particle position column and `inverse_masses` the
/// index-aligned inverse-mass column (`0` marks a pinned particle). A
/// non-positive `cell_size`/`thickness`, a mismatched `inverse_masses` length,
/// or fewer than two samples, is a no-op.
pub fn resolve_self_collision_virtual_jacobi(
    positions: &mut [Vec3],
    inverse_masses: &[Real],
    virtuals: &[VirtualParticle],
    cell_size: Real,
    thickness: Real,
) {
    let mut corrections = Vec::new();
    accumulate_virtual_jacobi_corrections(
        positions,
        inverse_masses,
        virtuals,
        cell_size,
        thickness,
        PairScope::All,
        &mut corrections,
    );
    apply_corrections(positions, &corrections);
}

/// Runs one Jacobi virtual-particle augment pass (virtual-touching pairs only),
/// the parallel-safe twin of
/// [`resolve_self_collision_virtual_augment`](super::resolve_self_collision_virtual_augment).
///
/// Real-vs-real pairs are skipped so this layers on top of a friction
/// point-to-point tier without stripping its tangential friction.
///
/// `positions` is the particle position column and `inverse_masses` the
/// index-aligned inverse-mass column. A non-positive `cell_size`/`thickness`, a
/// mismatched `inverse_masses` length, or fewer than two samples, is a no-op.
pub fn resolve_self_collision_virtual_augment_jacobi(
    positions: &mut [Vec3],
    inverse_masses: &[Real],
    virtuals: &[VirtualParticle],
    cell_size: Real,
    thickness: Real,
) {
    let mut corrections = Vec::new();
    accumulate_virtual_jacobi_corrections(
        positions,
        inverse_masses,
        virtuals,
        cell_size,
        thickness,
        PairScope::VirtualOnly,
        &mut corrections,
    );
    apply_corrections(positions, &corrections);
}

#[cfg(test)]
mod tests {
    use super::super::virtual_particles::{
        VirtualParticlePattern, generate_virtual_particles, resolve_self_collision_virtual,
    };
    use super::*;
    use alloc::vec;

    /// The two stacked triangles from the Gauss-Seidel suite: their centroid
    /// virtual particles collide.
    fn two_layers() -> (Vec<Vec3>, Vec<Real>) {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(0.0, 0.0, 0.05),
            Vec3::new(4.0, 0.0, 0.05),
            Vec3::new(0.0, 4.0, 0.05),
        ];
        let inverse_masses = vec![1.0; 6];
        (positions, inverse_masses)
    }

    #[test]
    fn accumulation_is_bit_identical_across_runs() {
        let (positions, inverse_masses) = two_layers();
        let virtuals = generate_virtual_particles(
            &[[0, 1, 2], [3, 4, 5]],
            &VirtualParticlePattern::nvcloth_default(),
        );

        let mut a = Vec::new();
        let mut b = Vec::new();
        accumulate_virtual_jacobi_corrections(
            &positions,
            &inverse_masses,
            &virtuals,
            1.0,
            0.2,
            PairScope::All,
            &mut a,
        );
        accumulate_virtual_jacobi_corrections(
            &positions,
            &inverse_masses,
            &virtuals,
            1.0,
            0.2,
            PairScope::All,
            &mut b,
        );
        assert_eq!(a, b, "Jacobi accumulation must be deterministic");
        assert_eq!(a.len(), positions.len());
    }

    #[test]
    fn degenerate_parameters_are_a_no_op() {
        let (positions, inverse_masses) = two_layers();
        let virtuals = generate_virtual_particles(
            &[[0, 1, 2], [3, 4, 5]],
            &VirtualParticlePattern::nvcloth_default(),
        );
        let mut out = Vec::new();
        for (cell, thick) in [(0.0, 0.2), (1.0, 0.0), (-1.0, -1.0)] {
            accumulate_virtual_jacobi_corrections(
                &positions,
                &inverse_masses,
                &virtuals,
                cell,
                thick,
                PairScope::All,
                &mut out,
            );
            assert_eq!(out.len(), positions.len());
            assert!(
                out.iter().all(|c| *c == Vec3::ZERO),
                "cell={cell} thick={thick}"
            );
        }
    }

    #[test]
    fn mismatched_inverse_mass_length_is_a_no_op() {
        let (positions, _) = two_layers();
        let short = vec![1.0; 5];
        let mut out = Vec::new();
        accumulate_virtual_jacobi_corrections(
            &positions,
            &short,
            &[],
            1.0,
            0.2,
            PairScope::All,
            &mut out,
        );
        assert_eq!(out.len(), positions.len());
        assert!(out.iter().all(|c| *c == Vec3::ZERO));
    }

    #[test]
    fn empty_virtuals_yields_zero_when_reals_are_apart() {
        // Two well-separated free vertices: with no virtual particles and no
        // real pair within thickness, every correction slot stays zero.
        let positions = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(5.0, 0.0, 0.0)];
        let inverse_masses = vec![1.0, 1.0];
        let mut out = Vec::new();
        accumulate_virtual_jacobi_corrections(
            &positions,
            &inverse_masses,
            &[],
            1.0,
            0.2,
            PairScope::All,
            &mut out,
        );
        assert!(out.iter().all(|c| *c == Vec3::ZERO));
    }

    #[test]
    fn pinned_triangle_pushes_only_the_single_intruder() {
        // A pinned triangle with one free intruder above its centroid. Only the
        // centroid virtual particle is within thickness, so a single Jacobi
        // pass matches the Gauss-Seidel single-pair result exactly.
        let base = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(4.0 / 3.0, 4.0 / 3.0, 0.05),
        ];
        let inverse_masses = vec![0.0, 0.0, 0.0, 1.0];
        let virtuals =
            generate_virtual_particles(&[[0, 1, 2]], &VirtualParticlePattern::nvcloth_default());

        let mut jac = base.clone();
        resolve_self_collision_virtual_jacobi(&mut jac, &inverse_masses, &virtuals, 1.0, 0.2);
        // Pinned corners never move.
        for i in 0..3 {
            assert_eq!(jac[i], base[i]);
        }
        // The intruder ends a full thickness out.
        assert!((jac[3].z - 0.2).abs() < 1e-5, "z = {}", jac[3].z);
    }

    #[test]
    fn augment_skips_real_pairs_that_the_full_pass_resolves() {
        // Two near-coincident free real vertices with no triangle: PairScope::All
        // separates them, PairScope::VirtualOnly leaves them to the point tier.
        let positions = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.05, 0.0, 0.0)];
        let inverse_masses = vec![1.0, 1.0];

        let mut full = Vec::new();
        accumulate_virtual_jacobi_corrections(
            &positions,
            &inverse_masses,
            &[],
            1.0,
            0.2,
            PairScope::All,
            &mut full,
        );
        assert!(full[0] != Vec3::ZERO && full[1] != Vec3::ZERO);

        let mut aug = Vec::new();
        accumulate_virtual_jacobi_corrections(
            &positions,
            &inverse_masses,
            &[],
            1.0,
            0.2,
            PairScope::VirtualOnly,
            &mut aug,
        );
        assert!(aug.iter().all(|c| *c == Vec3::ZERO));
    }

    #[test]
    fn iterated_jacobi_converges_to_the_same_separated_state() {
        // Both goldens must separate the two stacked layers. Gauss-Seidel does
        // it in one call; Jacobi needs a few iterations but reaches the same
        // qualitative state (the +z/-z gap opens past the fabric thickness).
        let virtuals = generate_virtual_particles(
            &[[0, 1, 2], [3, 4, 5]],
            &VirtualParticlePattern::nvcloth_default(),
        );

        let (mut gs, gs_im) = two_layers();
        resolve_self_collision_virtual(&mut gs, &gs_im, &virtuals, 1.0, 0.2);

        let (mut jac, jac_im) = two_layers();
        for _ in 0..64 {
            resolve_self_collision_virtual_jacobi(&mut jac, &jac_im, &virtuals, 1.0, 0.2);
        }

        // Lower face sinks in -z, upper face rises in +z, for both solvers.
        for i in 0..3 {
            assert!(gs[i].z < -1e-6);
            assert!(jac[i].z < -1e-6);
        }
        for i in 3..6 {
            assert!(gs[i].z > 1e-6);
            assert!(jac[i].z > 1e-6);
        }
        // The converged Jacobi centroid gap reaches at least the fabric
        // thickness, matching the Gauss-Seidel separation target.
        let jac_gap = jac[3].z - jac[0].z;
        assert!(jac_gap >= 0.2 - 1e-3, "converged gap {jac_gap} < thickness");
    }
}
