//! Linear-stiffness position projections for curve / strand soft bodies.
//!
//! Unlike the compliant XPBD constraints elsewhere in this module (which
//! accumulate a Lagrange multiplier and normalise compliance by `dt^2`), these
//! projections move a *free* particle a fixed **fraction** of the way toward a
//! geometric target in a single sweep. The fraction (`stiffness`, clamped by
//! the caller to `0..=1`) is intentionally iteration- and step-count
//! dependent: it is the `TressFX` / groom-style formulation used by hair guide
//! solvers for Laplacian smoothing, shape-goal pull-back, and one-sided
//! long-range tethers. These primitives live in the unified kernel so the
//! render-side hair module does not keep a second copy of the arithmetic — it
//! projects strand positions through this single, authoritative source.
//!
//! A particle is *free* when its inverse mass is strictly positive; a pinned
//! particle (`inverse_mass <= 0`) is never moved. For a free particle the
//! correction is applied at full magnitude (these are geometric blends, not
//! mass-weighted impulses), matching the guide-solver convention where a
//! `stiffness` of `1` snaps a particle exactly onto its target.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! Laplacian smoothing, shape pull-back, and long-range tether formulations are
//! standard position-based-dynamics techniques (see Müller et al. and the
//! publicly documented `TressFX` guide solver).

use glam::Vec3;

use crate::math::scalar::Real;

/// Returns whether particle `i` is free to move: its inverse mass is present
/// and strictly positive. A missing entry is treated as pinned (never moved),
/// so a short `inverse_masses` slice simply freezes the trailing particles
/// instead of panicking.
#[inline]
#[must_use]
fn is_free(inverse_masses: &[Real], i: usize) -> bool {
    inverse_masses.get(i).copied().unwrap_or(0.0) > 0.0
}

/// Projects a discrete-Laplacian smoothing step over every interior particle
/// once.
///
/// Each free interior particle is drawn a `stiffness` fraction of the way to
/// the midpoint of its two neighbours, a discrete Laplacian that penalises
/// sharp kinks and keeps a strand from folding onto itself. The two endpoints
/// have only one neighbour and are left untouched (the caller resolves them
/// with length and shape constraints). The sweep is sequential (Gauss-Seidel):
/// particle `i` sees the already-updated position of `i - 1` and the
/// not-yet-updated position of `i + 1`.
///
/// The projection is a no-op when `stiffness <= 0` or there are fewer than
/// three particles.
pub fn project_laplacian_smooth(positions: &mut [Vec3], inverse_masses: &[Real], stiffness: Real) {
    let count = positions.len();
    if count < 3 || stiffness <= 0.0 {
        return;
    }
    let mut i = 1;
    while i + 1 < count {
        let prev_pos = positions[i - 1];
        let next_pos = positions[i + 1];
        if is_free(inverse_masses, i) {
            let midpoint = (prev_pos + next_pos) * 0.5;
            let correction = (midpoint - positions[i]) * stiffness;
            positions[i] += correction;
        }
        i += 1;
    }
}

/// Projects a shape-goal pull-back over every particle once.
///
/// Each free particle that has a `targets` entry is pulled a `stiffness`
/// fraction of the way toward that target; a `stiffness` of `1` snaps it
/// exactly onto the target. Particles without a target entry (short `targets`
/// slice) are left untouched.
///
/// The projection is a no-op when `stiffness <= 0`.
pub fn project_pull_to_target(
    positions: &mut [Vec3],
    inverse_masses: &[Real],
    targets: &[Vec3],
    stiffness: Real,
) {
    if stiffness <= 0.0 {
        return;
    }
    let count = positions.len();
    let mut i = 0;
    while i < count {
        if let Some(&target) = targets.get(i)
            && is_free(inverse_masses, i)
        {
            let correction = (target - positions[i]) * stiffness;
            positions[i] += correction;
        }
        i += 1;
    }
}

/// Projects a one-sided long-range-attachment (LRA / tether) step once.
///
/// Every particle is tethered to the chain *root* (index `0`): its distance
/// from the root may not exceed the cumulative rest length of the segments
/// between them. `rest_lengths[i]` is the length of the segment leaving
/// particle `i`, so the tether radius of particle `k` is the sum of
/// `rest_lengths[0..k]`. When a free particle has been flung past that radius,
/// it is pulled a `stiffness` fraction of the way back onto the tether sphere
/// along its current radial direction; a `stiffness` of `1` snaps it exactly
/// onto the radius. The constraint is one-sided — a particle closer than its
/// tether length is never pushed outward — so it removes over-stretch without
/// injecting energy.
///
/// A missing `rest_lengths` entry stops the cumulative walk (later particles
/// then have no tether), and degenerate (zero-radius or coincident) cases are
/// skipped using `eps_len`. The projection is a no-op when `stiffness <= 0` or
/// there are fewer than two particles.
pub fn project_linear_tether(
    positions: &mut [Vec3],
    inverse_masses: &[Real],
    rest_lengths: &[Real],
    stiffness: Real,
    eps_len: Real,
) {
    let count = positions.len();
    if count < 2 || stiffness <= 0.0 {
        return;
    }
    let root = positions[0];
    let mut max_distance = 0.0;
    let mut i = 1;
    while i < count {
        // The tether radius grows by the rest length of the segment leaving the
        // previous particle; a missing entry ends the reachable chain.
        let Some(&segment) = rest_lengths.get(i - 1) else {
            break;
        };
        max_distance += segment.max(0.0);
        if is_free(inverse_masses, i) && max_distance > eps_len {
            let delta = positions[i] - root;
            let distance = delta.length();
            if distance > max_distance && distance > eps_len {
                // Target point on the tether sphere along the current radial
                // direction, then move a `stiffness` fraction toward it.
                let target = root + delta * (max_distance / distance);
                let correction = (target - positions[i]) * stiffness;
                positions[i] += correction;
            }
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS_LEN: Real = 1.0e-12;

    #[test]
    fn laplacian_is_inert_below_three_particles() {
        let mut positions = [Vec3::ZERO, Vec3::new(1.0, 1.0, 0.0)];
        project_laplacian_smooth(&mut positions, &[1.0, 1.0], 1.0);
        assert_eq!(positions[1], Vec3::new(1.0, 1.0, 0.0));
    }

    #[test]
    fn laplacian_snaps_interior_onto_midpoint_at_full_stiffness() {
        // A single kinked interior particle is pulled exactly onto the midpoint
        // of its neighbours when stiffness == 1.
        let mut positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        project_laplacian_smooth(&mut positions, &[1.0, 1.0, 1.0], 1.0);
        assert_eq!(positions[1], Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn laplacian_skips_pinned_interior() {
        let mut positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        // Interior inverse mass 0 marks it pinned; it must not move.
        project_laplacian_smooth(&mut positions, &[1.0, 0.0, 1.0], 1.0);
        assert_eq!(positions[1], Vec3::new(1.0, 1.0, 0.0));
    }

    #[test]
    fn pull_snaps_free_particle_onto_target() {
        let mut positions = [Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0)];
        let targets = [Vec3::ZERO, Vec3::new(1.0, 2.0, 3.0)];
        project_pull_to_target(&mut positions, &[1.0, 1.0], &targets, 1.0);
        assert_eq!(positions[1], Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn pull_moves_half_way_at_half_stiffness() {
        let mut positions = [Vec3::new(0.0, 0.0, 0.0)];
        let targets = [Vec3::new(4.0, 0.0, 0.0)];
        project_pull_to_target(&mut positions, &[1.0], &targets, 0.5);
        assert_eq!(positions[0], Vec3::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn pull_leaves_targetless_particles_untouched() {
        let mut positions = [Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0)];
        // Short targets slice: only particle 0 has a goal.
        project_pull_to_target(&mut positions, &[1.0, 1.0], &[Vec3::ZERO], 1.0);
        assert_eq!(positions[1], Vec3::new(5.0, 0.0, 0.0));
    }

    #[test]
    fn tether_pulls_overstretched_particle_onto_radius() {
        // Root at origin, one segment of rest length 1; the tip at distance 3
        // is snapped back onto radius 1 at full stiffness.
        let mut positions = [Vec3::ZERO, Vec3::new(3.0, 0.0, 0.0)];
        project_linear_tether(&mut positions, &[0.0, 1.0], &[1.0], 1.0, EPS_LEN);
        assert_eq!(positions[1], Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn tether_leaves_slack_particle_untouched() {
        // The tip is already inside its tether radius; it must not be pushed out.
        let mut positions = [Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)];
        project_linear_tether(&mut positions, &[0.0, 1.0], &[1.0], 1.0, EPS_LEN);
        assert_eq!(positions[1], Vec3::new(0.5, 0.0, 0.0));
    }

    #[test]
    fn tether_radius_accumulates_rest_lengths() {
        // Two unit segments give the second particle a tether radius of 2.
        let mut positions = [
            Vec3::ZERO,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
        ];
        project_linear_tether(&mut positions, &[0.0, 1.0, 1.0], &[1.0, 1.0], 1.0, EPS_LEN);
        assert_eq!(positions[2], Vec3::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn tether_missing_rest_length_ends_chain() {
        let mut positions = [
            Vec3::ZERO,
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(9.0, 0.0, 0.0),
        ];
        // Only one rest length: the walk stops before particle 2, leaving it free.
        project_linear_tether(&mut positions, &[0.0, 1.0, 1.0], &[1.0], 1.0, EPS_LEN);
        assert_eq!(positions[1], Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(positions[2], Vec3::new(9.0, 0.0, 0.0));
    }

    #[test]
    fn tether_skips_pinned_particle() {
        let mut positions = [Vec3::ZERO, Vec3::new(3.0, 0.0, 0.0)];
        // Tip pinned (inverse mass 0): the tether must not move it.
        project_linear_tether(&mut positions, &[0.0, 0.0], &[1.0], 1.0, EPS_LEN);
        assert_eq!(positions[1], Vec3::new(3.0, 0.0, 0.0));
    }
}
