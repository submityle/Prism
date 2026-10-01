//! Cloth / soft-body self-collision via a deterministic uniform spatial hash.
//!
//! Particles are bucketed into integer cells keyed in a [`BTreeMap`] so both
//! the cell traversal and (because indices are inserted in ascending order) the
//! per-bucket traversal are deterministic. For each particle only its 27-cell
//! neighborhood is examined, and each unordered pair is tested exactly once (by
//! requiring the neighbor index to exceed the current index), keeping the pass
//! near `O(n)` for well-distributed particles.
//!
//! A pair closer than `thickness` is separated along the line joining them,
//! split by inverse mass: two equal free particles each move half the
//! penetration, while a free particle paired with a pinned one takes the whole
//! correction. Coincident particles are separated along a fixed `+X` axis so
//! the result stays deterministic and free of [`f32::NAN`]. Corrections are
//! applied in place as they are found (Gauss-Seidel style), which is
//! deterministic given the fixed traversal order.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! uniform spatial hash and inverse-mass-weighted separation are standard
//! position-based-dynamics techniques; the tangential-friction projection is
//! the one published by Macklin et al. (2014).

use alloc::collections::BTreeMap;

use glam::Vec3;

use crate::math::scalar::Real;

use super::{cell_of, EPS_LEN_SQ};

/// Numerical floor below which a tangential slide is treated as zero, so a
/// friction correction is never normalised from a (near) zero-length vector.
const EPS_FRICTION: Real = 1e-12;

/// Resolves soft-body self-collision in place with a deterministic uniform
/// spatial hash.
///
/// `positions` is the particle position column and `inverse_masses` the
/// index-aligned inverse-mass column (`0` marks a pinned particle). Particles
/// are bucketed by integer cell of side `cell_size`; for each particle only its
/// 27-cell neighborhood is tested, and each unordered pair at most once. A pair
/// closer than `thickness` is pushed symmetrically apart, split by inverse
/// mass, so a pinned partner never moves and its free partner takes the whole
/// correction.
///
/// A non-positive `cell_size` or `thickness`, fewer than two particles, or an
/// `inverse_masses` slice whose length differs from `positions` is a no-op.
/// Coincident particles separate along `+X`; both-pinned pairs do not move.
pub fn resolve_self_collision(
    positions: &mut [Vec3],
    inverse_masses: &[Real],
    cell_size: Real,
    thickness: Real,
) {
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
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let neighbor = (cell.0 + dx, cell.1 + dy, cell.2 + dz);
                        let Some(nbucket) = grid.get(&neighbor) else {
                            continue;
                        };
                        for &b in nbucket {
                            if b <= a {
                                continue;
                            }
                            resolve_pair(
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
        }
    }
}

/// Resolves self-collision like [`resolve_self_collision`], but rubs the
/// tangential slide of every separated pair with position-level Coulomb
/// friction so stacked cloth layers grip instead of shearing freely.
///
/// The spatial hash, traversal order, and inverse-mass-weighted normal push are
/// identical to [`resolve_self_collision`]; friction is layered on inside
/// [`resolve_pair_with_friction`] using each partner's frame-start position from
/// `prev_positions`. `friction` is clamped to `0..=1`; a value of `0` delegates
/// straight to [`resolve_self_collision`]. A non-positive `cell_size` or
/// `thickness`, fewer than two particles, or a mismatched `inverse_masses`
/// length is a no-op, and a `prev_positions` slice shorter than `positions`
/// degrades to no tangential slide (hence no friction) for the missing indices.
pub fn resolve_self_collision_with_friction(
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
        resolve_self_collision(positions, inverse_masses, cell_size, thickness);
        return;
    }

    let grid = build_grid(positions, cell_size);
    let thickness_sq = thickness * thickness;
    for (&cell, bucket) in &grid {
        for &a in bucket {
            let ai = a as usize;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let neighbor = (cell.0 + dx, cell.1 + dy, cell.2 + dz);
                        let Some(nbucket) = grid.get(&neighbor) else {
                            continue;
                        };
                        for &b in nbucket {
                            if b <= a {
                                continue;
                            }
                            resolve_pair_with_friction(
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
        }
    }
}

/// Buckets every particle index into a [`BTreeMap`] keyed by integer cell.
///
/// Indices are pushed in ascending order, so each bucket is sorted, which makes
/// the whole resolution pass deterministic.
fn build_grid(positions: &[Vec3], cell_size: Real) -> BTreeMap<(i32, i32, i32), Vec<u32>> {
    let mut grid: BTreeMap<(i32, i32, i32), Vec<u32>> = BTreeMap::new();
    for (index, &pos) in positions.iter().enumerate() {
        let cell = cell_of(pos, cell_size);
        grid.entry(cell).or_default().push(index as u32);
    }
    grid
}

/// Returns `mu` clamped to `0..=1`, mapping any non-finite input to `0` so a
/// mis-authored coefficient can never inject a [`f32::NAN`] into a friction
/// pass.
#[must_use]
fn sanitize_friction(mu: Real) -> Real {
    if mu.is_finite() {
        mu.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Separates the particle pair `(ai, bi)` if they are closer than `thickness`.
///
/// The penetration is split by inverse mass so pinned partners stay put. When
/// the two positions coincide (no defined separating direction) a fixed `+X`
/// axis is used for determinism. Reads and writes go through distinct indices
/// (`ai != bi`), so there is no aliasing.
fn resolve_pair(
    positions: &mut [Vec3],
    inverse_masses: &[Real],
    ai: usize,
    bi: usize,
    thickness: Real,
    thickness_sq: Real,
) {
    let pa = positions[ai];
    let pb = positions[bi];
    let delta = pb - pa;
    let dist_sq = delta.length_squared();
    if dist_sq >= thickness_sq {
        return;
    }

    let wa = inverse_masses[ai].max(0.0);
    let wb = inverse_masses[bi].max(0.0);
    let w_sum = wa + wb;
    if w_sum <= 0.0 {
        // Both pinned: nothing can move.
        return;
    }

    let (dir, penetration) = separation(delta, dist_sq, thickness);
    let move_a = -penetration * (wa / w_sum);
    let move_b = penetration * (wb / w_sum);
    positions[ai] = pa + dir * move_a;
    positions[bi] = pb + dir * move_b;
}

/// Separates the pair `(ai, bi)` like [`resolve_pair`], then removes the
/// friction-limited part of their relative tangential slide.
///
/// The normal push (`dir`, `penetration`) is computed exactly as in
/// [`resolve_pair`]. Friction then acts on the relative frame slide
/// `(sep_a - prev_a) - (sep_b - prev_b)` projected onto the contact tangent
/// plane: the removed amount is `min(mu * penetration / ||dx_t||, 1) * dx_t`,
/// split between the partners by inverse mass so a pinned partner never moves
/// and the heavier partner moves less. Reads and writes go through distinct
/// indices, so there is no aliasing; a coincident pair or a below-threshold
/// slide falls back to the plain normal separation without producing a
/// [`f32::NAN`].
fn resolve_pair_with_friction(
    positions: &mut [Vec3],
    prev_positions: &[Vec3],
    inverse_masses: &[Real],
    ai: usize,
    bi: usize,
    thickness: Real,
    thickness_sq: Real,
    mu: Real,
) {
    let pa = positions[ai];
    let pb = positions[bi];
    let delta = pb - pa;
    let dist_sq = delta.length_squared();
    if dist_sq >= thickness_sq {
        return;
    }

    let wa = inverse_masses[ai].max(0.0);
    let wb = inverse_masses[bi].max(0.0);
    let w_sum = wa + wb;
    if w_sum <= 0.0 {
        return;
    }

    let (dir, penetration) = separation(delta, dist_sq, thickness);

    // Normal separation, inverse-mass weighted (identical to `resolve_pair`).
    let move_a = -penetration * (wa / w_sum);
    let move_b = penetration * (wb / w_sum);
    let sep_a = pa + dir * move_a;
    let sep_b = pb + dir * move_b;

    // Relative tangential slide since frame start. `dir` is the contact normal
    // and `penetration` is the pair's normal-correction magnitude `||dx_n||`.
    let prev_a = prev_positions.get(ai).copied().unwrap_or(pa);
    let prev_b = prev_positions.get(bi).copied().unwrap_or(pb);
    let rel = (sep_a - prev_a) - (sep_b - prev_b);
    let normal_amount = rel.dot(dir);
    let tangent = rel - dir * normal_amount;
    let tan_len_sq = tangent.length_squared();
    if tan_len_sq <= EPS_FRICTION {
        positions[ai] = sep_a;
        positions[bi] = sep_b;
        return;
    }
    let tan_len = tan_len_sq.sqrt();
    let scale = (mu * penetration / tan_len).min(1.0);
    let corr = tangent * scale;
    // Split the relative tangential correction by inverse mass so the change in
    // the `(a - b)` relative slide equals `-corr`.
    positions[ai] = sep_a - corr * (wa / w_sum);
    positions[bi] = sep_b + corr * (wb / w_sum);
}

/// Returns the unit separation direction (from `a` toward `b`) and the
/// penetration depth for a pair whose offset `delta = pb - pa` has squared
/// length `dist_sq` and whose contact `thickness` is known.
///
/// A coincident pair (`dist_sq <= EPS_LEN_SQ`) has no defined direction, so a
/// fixed `+X` axis and the full `thickness` are returned for a deterministic,
/// non-`NaN` result.
fn separation(delta: Vec3, dist_sq: Real, thickness: Real) -> (Vec3, Real) {
    if dist_sq <= EPS_LEN_SQ {
        (Vec3::new(1.0, 0.0, 0.0), thickness)
    } else {
        let dist = dist_sq.sqrt();
        (delta / dist, thickness - dist)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: Real = 1.0e-6;

    /// Asserts two vectors are equal within [`TOL`].
    fn approx_eq(a: Vec3, b: Vec3) {
        assert!((a.x - b.x).abs() < TOL, "x: {} vs {}", a.x, b.x);
        assert!((a.y - b.y).abs() < TOL, "y: {} vs {}", a.y, b.y);
        assert!((a.z - b.z).abs() < TOL, "z: {} vs {}", a.z, b.z);
    }

    #[test]
    fn coincident_free_pair_separates_along_x_by_half_thickness_each() {
        let mut positions = [Vec3::ZERO, Vec3::ZERO];
        let inverse_masses = [1.0, 1.0];
        resolve_self_collision(&mut positions, &inverse_masses, 1.0, 2.0);
        approx_eq(positions[0], Vec3::new(-1.0, 0.0, 0.0));
        approx_eq(positions[1], Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn pair_inside_thickness_separates_to_exactly_thickness() {
        let mut positions = [Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)];
        let inverse_masses = [1.0, 1.0];
        resolve_self_collision(&mut positions, &inverse_masses, 2.0, 1.0);
        let gap = (positions[1] - positions[0]).length();
        assert!((gap - 1.0).abs() < TOL, "gap {gap}");
        // Symmetric split: midpoint is preserved.
        approx_eq(positions[0], Vec3::new(-0.25, 0.0, 0.0));
        approx_eq(positions[1], Vec3::new(0.75, 0.0, 0.0));
    }

    #[test]
    fn pair_outside_thickness_does_not_move() {
        let mut positions = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let inverse_masses = [1.0, 1.0];
        resolve_self_collision(&mut positions, &inverse_masses, 1.0, 1.0);
        approx_eq(positions[0], Vec3::ZERO);
        approx_eq(positions[1], Vec3::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn pinned_partner_stays_and_free_partner_takes_whole_correction() {
        let mut positions = [Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)];
        // Particle 0 is pinned (inverse mass 0).
        let inverse_masses = [0.0, 1.0];
        resolve_self_collision(&mut positions, &inverse_masses, 2.0, 1.0);
        approx_eq(positions[0], Vec3::ZERO);
        approx_eq(positions[1], Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn non_positive_cell_size_is_a_no_op() {
        let mut positions = [Vec3::ZERO, Vec3::new(0.1, 0.0, 0.0)];
        let inverse_masses = [1.0, 1.0];
        resolve_self_collision(&mut positions, &inverse_masses, 0.0, 1.0);
        approx_eq(positions[0], Vec3::ZERO);
        approx_eq(positions[1], Vec3::new(0.1, 0.0, 0.0));
    }

    #[test]
    fn non_positive_thickness_is_a_no_op() {
        let mut positions = [Vec3::ZERO, Vec3::new(0.1, 0.0, 0.0)];
        let inverse_masses = [1.0, 1.0];
        resolve_self_collision(&mut positions, &inverse_masses, 1.0, 0.0);
        approx_eq(positions[0], Vec3::ZERO);
        approx_eq(positions[1], Vec3::new(0.1, 0.0, 0.0));
    }

    #[test]
    fn fewer_than_two_particles_is_a_no_op() {
        let mut positions = [Vec3::ZERO];
        let inverse_masses = [1.0];
        resolve_self_collision(&mut positions, &inverse_masses, 1.0, 1.0);
        approx_eq(positions[0], Vec3::ZERO);
    }

    #[test]
    fn mismatched_inverse_mass_length_is_a_no_op() {
        let mut positions = [Vec3::ZERO, Vec3::new(0.1, 0.0, 0.0)];
        let inverse_masses = [1.0];
        resolve_self_collision(&mut positions, &inverse_masses, 1.0, 1.0);
        approx_eq(positions[0], Vec3::ZERO);
        approx_eq(positions[1], Vec3::new(0.1, 0.0, 0.0));
    }

    #[test]
    fn both_pinned_pair_does_not_move() {
        let mut positions = [Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)];
        let inverse_masses = [0.0, 0.0];
        resolve_self_collision(&mut positions, &inverse_masses, 2.0, 1.0);
        approx_eq(positions[0], Vec3::ZERO);
        approx_eq(positions[1], Vec3::new(0.5, 0.0, 0.0));
    }

    #[test]
    fn friction_mu_zero_matches_plain_self_collision() {
        let mut with = [Vec3::ZERO, Vec3::new(0.5, 0.2, 0.0)];
        let mut plain = with;
        let prev = [Vec3::new(-0.3, 0.0, 0.0), Vec3::new(0.8, 0.2, 0.0)];
        let inverse_masses = [1.0, 1.0];
        resolve_self_collision_with_friction(&mut with, &prev, &inverse_masses, 2.0, 1.0, 0.0);
        resolve_self_collision(&mut plain, &inverse_masses, 2.0, 1.0);
        approx_eq(with[0], plain[0]);
        approx_eq(with[1], plain[1]);
    }

    #[test]
    fn static_friction_cone_cancels_whole_tangential_slide() {
        // Two particles overlapping along X; their relative slide is purely
        // tangential (along Y) and small enough to sit inside the cone, so a
        // high friction coefficient should cancel it entirely.
        let mut positions = [Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)];
        // Give each a tangential (Y) slide since frame start.
        let prev = [Vec3::new(0.0, -0.01, 0.0), Vec3::new(0.5, 0.01, 0.0)];
        let inverse_masses = [1.0, 1.0];
        resolve_self_collision_with_friction(&mut positions, &prev, &inverse_masses, 2.0, 1.0, 1.0);
        // Normal separation is along X to exactly thickness; friction only
        // shifts the particles tangentially (in Y), so the X separation stays
        // at the thickness while the Euclidean distance may grow slightly.
        let x_gap = (positions[1].x - positions[0].x).abs();
        assert!((x_gap - 1.0).abs() < TOL, "x gap {x_gap}");
        // The relative tangential (Y) slide `(a - prev_a) - (b - prev_b)` must
        // be fully removed, so both end at the same Y they started the frame.
        let rel_y = (positions[0].y - prev[0].y) - (positions[1].y - prev[1].y);
        assert!(rel_y.abs() < TOL, "residual tangential slide {rel_y}");
    }

    #[test]
    fn dynamic_friction_shrinks_but_keeps_tangential_slide() {
        // A large tangential slide outside the cone is shrunk by exactly
        // `mu * penetration`, not cancelled, so some slide survives.
        let mut positions = [Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)];
        let prev = [Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.5, 1.0, 0.0)];
        let inverse_masses = [1.0, 1.0];
        let mu = 0.2;
        resolve_self_collision_with_friction(&mut positions, &prev, &inverse_masses, 2.0, 1.0, mu);
        let rel_y = (positions[0].y - prev[0].y) - (positions[1].y - prev[1].y);
        // Residual slide is non-zero (not fully cancelled) ...
        assert!(rel_y.abs() > TOL, "slide was fully cancelled: {rel_y}");
        // ... and smaller in magnitude than the original relative slide (2.0).
        let original_rel_y = (0.0 - prev[0].y) - (0.0 - prev[1].y);
        assert!(
            rel_y.abs() < original_rel_y.abs(),
            "slide not shrunk: {rel_y} vs {original_rel_y}"
        );
    }

    #[test]
    fn non_finite_friction_is_sanitised_to_zero() {
        let mut with = [Vec3::ZERO, Vec3::new(0.5, 0.2, 0.0)];
        let mut plain = with;
        let prev = [Vec3::new(-0.3, 0.0, 0.0), Vec3::new(0.8, 0.2, 0.0)];
        let inverse_masses = [1.0, 1.0];
        resolve_self_collision_with_friction(
            &mut with,
            &prev,
            &inverse_masses,
            2.0,
            1.0,
            Real::NAN,
        );
        resolve_self_collision(&mut plain, &inverse_masses, 2.0, 1.0);
        approx_eq(with[0], plain[0]);
        approx_eq(with[1], plain[1]);
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
            resolve_self_collision(&mut positions, &inverse_masses, 1.0, 0.5);
            positions
        };
        assert_eq!(run(), run());
    }
}
