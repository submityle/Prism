//! Jacobi (parallel-safe) point self-collision — the GPU-faithful golden.
//!
//! [`resolve_self_collision`](super::resolve_self_collision) resolves the
//! point-to-point (vertex-vertex) self-collision tier in **Gauss-Seidel**
//! order: it walks the pairs in a fixed traversal and scatters each
//! half-correction onto the two particles *in place*, so a later pair already
//! sees the moved positions of an earlier one. That is the sequential CPU
//! reference, but it does not map onto a `GPU` compute kernel: on the `GPU`
//! every particle is updated in parallel from the *same* read-only snapshot,
//! which is a **Jacobi** iteration, not Gauss-Seidel. The two converge to the
//! same separated state but are never bit-for-bit identical on a single pass,
//! so a faithful point-self-collision `WGSL`/`WESL` twin needs its own golden
//! rather than borrowing the Gauss-Seidel one (no fake parity).
//!
//! This module owns that Jacobi golden purely and deterministically. For every
//! particle `a`, [`accumulate_self_collision_jacobi_corrections`] gathers its
//! 27-cell neighborhood from a prebuilt uniform hash and sums *`a`'s own half*
//! of the separating push against every penetrating neighbor `b != a`, reading
//! positions only from the frozen input snapshot. One invocation owns one
//! `out[a]` slot and never reads another slot, so this is exactly the body of a
//! per-particle `GPU` kernel with no atomics. [`apply_corrections`] then folds
//! the accumulated corrections back into the positions — the trivial "apply"
//! half of a Jacobi step.
//!
//! The friction variant layers position-level Coulomb friction on top of the
//! normal push exactly as the Gauss-Seidel
//! [`resolve_self_collision_with_friction`](super::resolve_self_collision_with_friction)
//! does, but folds each pair's total per-particle displacement (normal push
//! *minus* the inverse-mass-weighted tangential correction) into the owning
//! slot so the whole pass stays order-independent.
//!
//! Both phases reduce in strictly ascending index order (ascending
//! [`BTreeMap`] cells, ascending bucket indices), so the float reductions are
//! deterministic and the `GPU` twin can mirror the result value-for-value. A
//! single isolated pair resolves bit-for-bit identically to the Gauss-Seidel
//! core in one pass; stacked clusters converge to the same separated state over
//! a few iterations. Guards match the Gauss-Seidel core: a non-positive
//! `cell_size`/`thickness`, an `inverse_masses` length that differs from
//! `positions`, or fewer than two particles yields all-zero corrections; pinned
//! particles (`inverse_mass <= 0`) never move; coincident particles separate
//! along `+X`.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! uniform spatial hash and inverse-mass-weighted separation are standard
//! position-based-dynamics techniques; the tangential-friction projection is
//! the one published by Macklin et al. (2014), "Unified Particle Physics for
//! Real-Time Applications"; the Jacobi split into own-slot accumulate/apply
//! phases is a standard parallel position-based-dynamics reformulation.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use glam::Vec3;

use crate::math::scalar::Real;

use super::friction::{sanitize_friction, EPS_FRICTION};
use super::{cell_of, EPS_LEN_SQ};

/// Accumulates each particle's Jacobi point self-collision correction.
///
/// `out` is resized to `positions.len()` and fully overwritten (cleared to
/// [`Vec3::ZERO`] first); entry `out[a]` is the total correction particle `a`
/// should receive this pass. All reads are from the *input* `positions`
/// snapshot (never from `out`), so the result is independent of evaluation
/// order and maps directly to a per-invocation `GPU` kernel. See
/// [`apply_corrections`] to fold the result back into positions.
///
/// `positions` is the particle position column and `inverse_masses` the
/// index-aligned inverse-mass column (`0` marks a pinned particle). A
/// non-positive `cell_size`/`thickness`, an `inverse_masses` slice whose length
/// differs from `positions`, or fewer than two particles yields all-zero
/// corrections.
pub(crate) fn accumulate_self_collision_jacobi_corrections(
    positions: &[Vec3],
    inverse_masses: &[Real],
    cell_size: Real,
    thickness: Real,
    out: &mut Vec<Vec3>,
) {
    out.clear();
    out.resize(positions.len(), Vec3::ZERO);

    if cell_size <= 0.0
        || thickness <= 0.0
        || positions.len() < 2
        || inverse_masses.len() != positions.len()
    {
        return;
    }

    let grid = build_grid(positions, cell_size);
    let thickness_sq = thickness * thickness;
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
                            acc += half_correction(
                                positions,
                                inverse_masses,
                                ai,
                                b as usize,
                                thickness,
                                thickness_sq,
                            );
                        }
                    }
                }
            }
            out[ai] = acc;
        }
    }
}

/// Accumulates each particle's Jacobi point self-collision correction with
/// position-level Coulomb friction layered on the tangential slide.
///
/// Identical to [`accumulate_self_collision_jacobi_corrections`] in traversal,
/// determinism, and guards, but each pair's contribution is the owning
/// particle's full per-pair displacement — the normal push minus its
/// inverse-mass-weighted share of the friction-limited relative tangential
/// slide, measured against `prev_positions` (the frame-start positions). A
/// `prev_positions` slice shorter than `positions` degrades to no tangential
/// slide (hence no friction) for the missing indices. `mu` is assumed already
/// sanitised to `0..=1` by the caller.
pub(crate) fn accumulate_self_collision_with_friction_jacobi_corrections(
    positions: &[Vec3],
    prev_positions: &[Vec3],
    inverse_masses: &[Real],
    cell_size: Real,
    thickness: Real,
    mu: Real,
    out: &mut Vec<Vec3>,
) {
    out.clear();
    out.resize(positions.len(), Vec3::ZERO);

    if cell_size <= 0.0
        || thickness <= 0.0
        || positions.len() < 2
        || inverse_masses.len() != positions.len()
    {
        return;
    }

    let grid = build_grid(positions, cell_size);
    let thickness_sq = thickness * thickness;
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
                            acc += half_correction_with_friction(
                                positions,
                                prev_positions,
                                inverse_masses,
                                ai,
                                b as usize,
                                thickness,
                                thickness_sq,
                                mu,
                            );
                        }
                    }
                }
            }
            out[ai] = acc;
        }
    }
}

/// Adds the accumulated per-particle corrections into the particle positions.
///
/// This is the trivial "apply" half of a Jacobi step: after an accumulate pass
/// has produced `corrections`, every particle advances by its own entry.
/// `corrections` shorter than `positions` leaves the tail untouched; extra
/// entries are ignored.
pub(crate) fn apply_corrections(positions: &mut [Vec3], corrections: &[Vec3]) {
    for (pos, corr) in positions.iter_mut().zip(corrections.iter()) {
        *pos += *corr;
    }
}

/// Runs one Jacobi point self-collision pass in place — the parallel-safe twin
/// of [`resolve_self_collision`](super::resolve_self_collision).
///
/// One call is a single Jacobi iteration (accumulate from the frozen snapshot,
/// then apply). A single isolated pair resolves bit-for-bit identically to the
/// Gauss-Seidel core in one pass; dense clusters converge to the same separated
/// state over a few iterations without ever depending on evaluation order.
///
/// `positions` is the particle position column and `inverse_masses` the
/// index-aligned inverse-mass column (`0` marks a pinned particle). A
/// non-positive `cell_size`/`thickness`, a mismatched `inverse_masses` length,
/// or fewer than two particles, is a no-op.
pub fn resolve_self_collision_jacobi(
    positions: &mut [Vec3],
    inverse_masses: &[Real],
    cell_size: Real,
    thickness: Real,
) {
    let mut corrections = Vec::new();
    accumulate_self_collision_jacobi_corrections(
        positions,
        inverse_masses,
        cell_size,
        thickness,
        &mut corrections,
    );
    apply_corrections(positions, &corrections);
}

/// Runs one Jacobi point self-collision pass with position-level Coulomb
/// friction — the parallel-safe twin of
/// [`resolve_self_collision_with_friction`](super::resolve_self_collision_with_friction).
///
/// The normal push and traversal match [`resolve_self_collision_jacobi`];
/// friction is layered on using each partner's frame-start position from
/// `prev_positions`. `friction` is clamped to `0..=1` (non-finite maps to `0`);
/// a value of `0` delegates straight to [`resolve_self_collision_jacobi`]. A
/// non-positive `cell_size`/`thickness`, fewer than two particles, or a
/// mismatched `inverse_masses` length is a no-op, and a `prev_positions` slice
/// shorter than `positions` degrades to no tangential slide for the missing
/// indices.
pub fn resolve_self_collision_with_friction_jacobi(
    positions: &mut [Vec3],
    prev_positions: &[Vec3],
    inverse_masses: &[Real],
    cell_size: Real,
    thickness: Real,
    friction: Real,
) {
    if cell_size <= 0.0
        || thickness <= 0.0
        || positions.len() < 2
        || inverse_masses.len() != positions.len()
    {
        return;
    }
    let mu = sanitize_friction(friction);
    if mu <= 0.0 {
        resolve_self_collision_jacobi(positions, inverse_masses, cell_size, thickness);
        return;
    }

    let mut corrections = Vec::new();
    accumulate_self_collision_with_friction_jacobi_corrections(
        positions,
        prev_positions,
        inverse_masses,
        cell_size,
        thickness,
        mu,
        &mut corrections,
    );
    apply_corrections(positions, &corrections);
}

/// Buckets every particle index into a [`BTreeMap`] keyed by integer cell.
///
/// Indices are pushed in ascending order, so each bucket is sorted, which makes
/// both the cell traversal and the per-bucket traversal deterministic.
fn build_grid(positions: &[Vec3], cell_size: Real) -> BTreeMap<(i32, i32, i32), Vec<u32>> {
    let mut grid: BTreeMap<(i32, i32, i32), Vec<u32>> = BTreeMap::new();
    for (index, &pos) in positions.iter().enumerate() {
        let cell = cell_of(pos, cell_size);
        grid.entry(cell).or_default().push(index as u32);
    }
    grid
}

/// Particle `ai`'s own half of the separating push against neighbor `bi`.
///
/// Returns [`Vec3::ZERO`] when the pair is farther apart than `thickness` or is
/// jointly immovable (both pinned). `dir` points from `ai` toward `bi`, so `ai`
/// is pushed the opposite way, weighted by its inverse-mass share; coincident
/// particles separate along `+X`. Positions come from the frozen `positions`
/// snapshot, so the value is order-independent and matches what a `GPU`
/// invocation for `ai` would compute.
fn half_correction(
    positions: &[Vec3],
    inverse_masses: &[Real],
    ai: usize,
    bi: usize,
    thickness: Real,
    thickness_sq: Real,
) -> Vec3 {
    let pa = positions[ai];
    let pb = positions[bi];
    let delta = pb - pa;
    let dist_sq = delta.length_squared();
    if dist_sq >= thickness_sq {
        return Vec3::ZERO;
    }

    let wa = inverse_masses[ai].max(0.0);
    let wb = inverse_masses[bi].max(0.0);
    let w_sum = wa + wb;
    if w_sum <= 0.0 {
        return Vec3::ZERO;
    }

    let (dir, penetration) = separation(delta, dist_sq, thickness, ai, bi);
    // `dir` points from A toward B; A is pushed the opposite way by its share.
    dir * (-penetration * (wa / w_sum))
}

/// Particle `ai`'s own per-pair displacement against neighbor `bi` with
/// position-level Coulomb friction.
///
/// The normal push matches [`half_correction`]; friction then removes the
/// friction-limited part of the relative tangential slide
/// `(sep_a - prev_a) - (sep_b - prev_b)` and credits `ai` its inverse-mass
/// share. Returns the *displacement* `ai` should accumulate (post-position
/// minus frozen position), so summing it over all neighbors and applying it
/// reproduces the Gauss-Seidel result for an isolated pair. A coincident pair
/// or a below-threshold slide falls back to the plain normal push without
/// producing a [`f32::NAN`].
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the Gauss-Seidel resolve_pair_with_friction parameter set"
)]
fn half_correction_with_friction(
    positions: &[Vec3],
    prev_positions: &[Vec3],
    inverse_masses: &[Real],
    ai: usize,
    bi: usize,
    thickness: Real,
    thickness_sq: Real,
    mu: Real,
) -> Vec3 {
    let pa = positions[ai];
    let pb = positions[bi];
    let delta = pb - pa;
    let dist_sq = delta.length_squared();
    if dist_sq >= thickness_sq {
        return Vec3::ZERO;
    }

    let wa = inverse_masses[ai].max(0.0);
    let wb = inverse_masses[bi].max(0.0);
    let w_sum = wa + wb;
    if w_sum <= 0.0 {
        return Vec3::ZERO;
    }

    let (dir, penetration) = separation(delta, dist_sq, thickness, ai, bi);

    // Normal separation, inverse-mass weighted (identical to `half_correction`).
    let move_a = -penetration * (wa / w_sum);
    let sep_a = pa + dir * move_a;
    let move_b = penetration * (wb / w_sum);
    let sep_b = pb + dir * move_b;

    // Relative tangential slide since frame start. `dir` is the contact normal
    // and `penetration` is the pair's normal-correction magnitude.
    let prev_a = prev_positions.get(ai).copied().unwrap_or(pa);
    let prev_b = prev_positions.get(bi).copied().unwrap_or(pb);
    let rel = (sep_a - prev_a) - (sep_b - prev_b);
    let normal_amount = rel.dot(dir);
    let tangent = rel - dir * normal_amount;
    let tan_len_sq = tangent.length_squared();
    if tan_len_sq <= EPS_FRICTION {
        // No tangential slide to arrest: just the normal displacement.
        return sep_a - pa;
    }
    let tan_len = tan_len_sq.sqrt();
    let scale = (mu * penetration / tan_len).min(1.0);
    let corr = tangent * scale;
    // `ai`'s displacement is its separated position minus its inverse-mass
    // share of the tangential correction, relative to the frozen position.
    (sep_a - corr * (wa / w_sum)) - pa
}

/// Returns the unit separation direction (from `a` toward `b`) and the
/// penetration depth for a pair whose offset `delta = pb - pa` has squared
/// length `dist_sq` and whose contact `thickness` is known.
///
/// A coincident pair (`dist_sq <= EPS_LEN_SQ`) has no defined direction. Unlike
/// the Gauss-Seidel core — which resolves each unordered pair once and can hard
/// code a `+X` axis — a Jacobi own-slot pass evaluates both roles of the pair
/// independently, so the fallback axis must stay *antisymmetric* under the
/// role swap or both particles would push the same way. The axis is therefore
/// oriented by index (`+X` when `ai < bi`, `-X` otherwise), which reproduces
/// the Gauss-Seidel convention (the lower index moves `-X`, the higher `+X`)
/// deterministically and without a [`f32::NAN`].
fn separation(delta: Vec3, dist_sq: Real, thickness: Real, ai: usize, bi: usize) -> (Vec3, Real) {
    if dist_sq <= EPS_LEN_SQ {
        let axis = if ai < bi { 1.0 } else { -1.0 };
        (Vec3::new(axis, 0.0, 0.0), thickness)
    } else {
        let dist = dist_sq.sqrt();
        (delta / dist, thickness - dist)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soft::collision::{resolve_self_collision, resolve_self_collision_with_friction};

    const TOL: Real = 1.0e-6;

    /// Asserts two vectors are equal within [`TOL`].
    fn approx_eq(a: Vec3, b: Vec3) {
        assert!((a.x - b.x).abs() < TOL, "x: {} vs {}", a.x, b.x);
        assert!((a.y - b.y).abs() < TOL, "y: {} vs {}", a.y, b.y);
        assert!((a.z - b.z).abs() < TOL, "z: {} vs {}", a.z, b.z);
    }

    #[test]
    fn single_pass_matches_gauss_seidel_for_isolated_pair() {
        // One penetrating pair has no cross-dependency, so a single Jacobi pass
        // reproduces the Gauss-Seidel result bit-for-bit.
        let mut jac = [Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)];
        let mut gs = jac;
        let inverse_masses = [1.0, 1.0];
        resolve_self_collision_jacobi(&mut jac, &inverse_masses, 2.0, 1.0);
        resolve_self_collision(&mut gs, &inverse_masses, 2.0, 1.0);
        approx_eq(jac[0], gs[0]);
        approx_eq(jac[1], gs[1]);
        // And it reaches exactly `thickness`.
        let gap = (jac[1] - jac[0]).length();
        assert!((gap - 1.0).abs() < TOL, "gap {gap}");
    }

    #[test]
    fn single_pass_friction_matches_gauss_seidel_for_isolated_pair() {
        let prev = [Vec3::new(0.0, -0.01, 0.0), Vec3::new(0.5, 0.01, 0.0)];
        let inverse_masses = [1.0, 1.0];
        let mut jac = [Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)];
        let mut gs = jac;
        resolve_self_collision_with_friction_jacobi(&mut jac, &prev, &inverse_masses, 2.0, 1.0, 1.0);
        resolve_self_collision_with_friction(&mut gs, &prev, &inverse_masses, 2.0, 1.0, 1.0);
        approx_eq(jac[0], gs[0]);
        approx_eq(jac[1], gs[1]);
    }

    #[test]
    fn coincident_free_pair_separates_along_x_by_half_thickness_each() {
        let mut positions = [Vec3::ZERO, Vec3::ZERO];
        let inverse_masses = [1.0, 1.0];
        resolve_self_collision_jacobi(&mut positions, &inverse_masses, 1.0, 2.0);
        approx_eq(positions[0], Vec3::new(-1.0, 0.0, 0.0));
        approx_eq(positions[1], Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn pinned_partner_takes_whole_correction() {
        // A pinned particle at the origin and a free one inside thickness: the
        // free one takes the full push, the pinned one never moves.
        let mut positions = [Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)];
        let inverse_masses = [0.0, 1.0];
        resolve_self_collision_jacobi(&mut positions, &inverse_masses, 2.0, 1.0);
        approx_eq(positions[0], Vec3::ZERO);
        approx_eq(positions[1], Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn both_pinned_pair_does_not_move() {
        let mut positions = [Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)];
        let inverse_masses = [0.0, 0.0];
        resolve_self_collision_jacobi(&mut positions, &inverse_masses, 2.0, 1.0);
        approx_eq(positions[0], Vec3::ZERO);
        approx_eq(positions[1], Vec3::new(0.5, 0.0, 0.0));
    }

    #[test]
    fn pair_outside_thickness_does_not_move() {
        let mut positions = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let inverse_masses = [1.0, 1.0];
        resolve_self_collision_jacobi(&mut positions, &inverse_masses, 2.0, 1.0);
        approx_eq(positions[0], Vec3::ZERO);
        approx_eq(positions[1], Vec3::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn friction_mu_zero_matches_plain_jacobi() {
        let prev = [Vec3::new(-0.3, 0.0, 0.0), Vec3::new(0.8, 0.2, 0.0)];
        let inverse_masses = [1.0, 1.0];
        let mut with = [Vec3::ZERO, Vec3::new(0.5, 0.2, 0.0)];
        let mut plain = with;
        resolve_self_collision_with_friction_jacobi(
            &mut with,
            &prev,
            &inverse_masses,
            2.0,
            1.0,
            0.0,
        );
        resolve_self_collision_jacobi(&mut plain, &inverse_masses, 2.0, 1.0);
        approx_eq(with[0], plain[0]);
        approx_eq(with[1], plain[1]);
    }

    #[test]
    fn non_finite_friction_is_sanitised_to_zero() {
        let prev = [Vec3::new(-0.3, 0.0, 0.0), Vec3::new(0.8, 0.2, 0.0)];
        let inverse_masses = [1.0, 1.0];
        let mut with = [Vec3::ZERO, Vec3::new(0.5, 0.2, 0.0)];
        let mut plain = with;
        resolve_self_collision_with_friction_jacobi(
            &mut with,
            &prev,
            &inverse_masses,
            2.0,
            1.0,
            Real::NAN,
        );
        resolve_self_collision_jacobi(&mut plain, &inverse_masses, 2.0, 1.0);
        approx_eq(with[0], plain[0]);
        approx_eq(with[1], plain[1]);
    }

    #[test]
    fn non_positive_cell_size_is_a_no_op() {
        let mut positions = [Vec3::ZERO, Vec3::new(0.1, 0.0, 0.0)];
        let inverse_masses = [1.0, 1.0];
        resolve_self_collision_jacobi(&mut positions, &inverse_masses, 0.0, 1.0);
        approx_eq(positions[0], Vec3::ZERO);
        approx_eq(positions[1], Vec3::new(0.1, 0.0, 0.0));
    }

    #[test]
    fn non_positive_thickness_is_a_no_op() {
        let mut positions = [Vec3::ZERO, Vec3::new(0.1, 0.0, 0.0)];
        let inverse_masses = [1.0, 1.0];
        resolve_self_collision_jacobi(&mut positions, &inverse_masses, 1.0, 0.0);
        approx_eq(positions[0], Vec3::ZERO);
        approx_eq(positions[1], Vec3::new(0.1, 0.0, 0.0));
    }

    #[test]
    fn fewer_than_two_particles_is_a_no_op() {
        let mut positions = [Vec3::ZERO];
        let inverse_masses = [1.0];
        resolve_self_collision_jacobi(&mut positions, &inverse_masses, 1.0, 1.0);
        approx_eq(positions[0], Vec3::ZERO);
    }

    #[test]
    fn mismatched_inverse_mass_length_is_a_no_op() {
        let mut positions = [Vec3::ZERO, Vec3::new(0.1, 0.0, 0.0)];
        let inverse_masses = [1.0];
        resolve_self_collision_jacobi(&mut positions, &inverse_masses, 1.0, 1.0);
        approx_eq(positions[0], Vec3::ZERO);
        approx_eq(positions[1], Vec3::new(0.1, 0.0, 0.0));
    }

    #[test]
    fn iterated_jacobi_converges_towards_gauss_seidel_cluster() {
        // A dense cluster where multiple pairs couple each particle: one Jacobi
        // pass undershoots, but iterating converges to a separated state whose
        // minimum pair gap matches what Gauss-Seidel reaches.
        let make = || {
            [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.3, 0.0, 0.0),
                Vec3::new(0.0, 0.3, 0.0),
                Vec3::new(0.3, 0.3, 0.0),
            ]
        };
        let inverse_masses = [1.0, 1.0, 1.0, 1.0];
        let cell = 1.0;
        let thickness = 0.5;

        let mut gs = make();
        resolve_self_collision(&mut gs, &inverse_masses, cell, thickness);

        let mut jac = make();
        for _ in 0..128 {
            resolve_self_collision_jacobi(&mut jac, &inverse_masses, cell, thickness);
        }

        let min_gap = |p: &[Vec3; 4]| {
            let mut m = Real::INFINITY;
            for i in 0..4 {
                for j in (i + 1)..4 {
                    m = m.min((p[j] - p[i]).length());
                }
            }
            m
        };
        let jac_gap = min_gap(&jac);
        // The converged Jacobi cluster separates every pair to at least the
        // fabric thickness, matching the Gauss-Seidel separation target.
        assert!(jac_gap >= thickness - 1e-3, "converged min gap {jac_gap}");
    }

    #[test]
    fn result_is_deterministic() {
        let run = || {
            let mut positions = [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.3, 0.1, 0.0),
                Vec3::new(0.1, 0.2, 0.1),
                Vec3::new(0.2, 0.0, 0.3),
            ];
            let inverse_masses = [1.0, 1.0, 1.0, 1.0];
            resolve_self_collision_jacobi(&mut positions, &inverse_masses, 1.0, 0.5);
            positions
        };
        assert_eq!(run(), run());
    }
}
