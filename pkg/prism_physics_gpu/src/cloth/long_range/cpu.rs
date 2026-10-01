//! The `CPU` golden twin for the cloth long-range-attachment kernel.
//!
//! The authoritative per-leash projection lives in [`prism_physics_core`] as
//! `project_long_range`; rather than copy that arithmetic and risk it drifting,
//! [`cpu_cloth_long_range`] *delegates* every projection to it and only owns the
//! colour-batched sweep schedule, which is exactly what the
//! [`GpuClothLongRange`](super::gpu::GpuClothLongRange) kernel runs. The parity
//! suite then compares the two applied-position fields within a tight tolerance.
//!
//! The twin is in turn anchored, in this module's tests, against an independent
//! brute-force reference that re-derives the one-sided compliant projection
//! inline over the identical colour schedule, closing the loop from first
//! principles (no fake parity).
//!
//! # Provenance
//!
//! The one-sided long-range-attachment leash is a published position-based
//! dynamics technique (Kim et al., "Long Range Attachments"). No Unreal Engine
//! source or derived code.

use alloc::vec::Vec;

use glam::Vec3;
use prism_physics_core::project_long_range;

use super::coloring::colour_long_range;
use super::ClothLongRangeConstraint;

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// Runs `iterations` colour-batched long-range sweeps on the `CPU`, returning
/// the applied positions.
///
/// This is the golden twin of
/// [`GpuClothLongRange::solve`](super::gpu::GpuClothLongRange::solve): the
/// leashes are graph-coloured once, then each iteration visits every colour
/// class in ascending order and projects its leashes. The accumulated `XPBD`
/// Lagrange multipliers are cleared once at the start and carried across all
/// iterations (standard `XPBD` warm-free accumulation), never reset between
/// sweeps.
///
/// An empty constraint set, zero iterations, or an `inverse_masses` slice
/// shorter than the particles the leashes reference leaves `positions`
/// unchanged (out-of-range leashes are individually skipped by
/// [`project_long_range`]).
#[must_use]
pub fn cpu_cloth_long_range(
    positions: &[Vec3],
    inverse_masses: &[Real],
    constraints: &[ClothLongRangeConstraint],
    dt: Real,
    iterations: u32,
) -> Vec<Vec3> {
    let mut out = positions.to_vec();
    if constraints.is_empty() || iterations == 0 {
        return out;
    }

    let coloring = colour_long_range(constraints);
    // One accumulated multiplier per leash, cleared once for the whole solve so
    // each iteration warm-continues the previous sweep.
    let mut lambdas: Vec<Real> = Vec::new();
    lambdas.resize(constraints.len(), 0.0);

    for _ in 0..iterations {
        for &(start, count) in &coloring.ranges {
            for slot in start..start + count {
                let ci = coloring.order[slot as usize] as usize;
                let c = &constraints[ci];
                lambdas[ci] = project_long_range(
                    &mut out,
                    inverse_masses,
                    c.particle,
                    c.anchor(),
                    c.max_distance,
                    c.compliance,
                    lambdas[ci],
                    dt,
                );
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An independent brute-force reference that re-derives the one-sided
    /// compliant leash projection inline over the identical colour schedule,
    /// with no call into [`project_long_range`]. If this agrees with
    /// [`cpu_cloth_long_range`] the delegation is proven arithmetically
    /// faithful.
    fn brute_force(
        positions: &[Vec3],
        inverse_masses: &[Real],
        constraints: &[ClothLongRangeConstraint],
        dt: Real,
        iterations: u32,
    ) -> Vec<Vec3> {
        let mut out = positions.to_vec();
        if constraints.is_empty() || iterations == 0 {
            return out;
        }
        let coloring = colour_long_range(constraints);
        let mut lambdas = Vec::new();
        lambdas.resize(constraints.len(), 0.0_f32);
        for _ in 0..iterations {
            for &(start, count) in &coloring.ranges {
                for slot in start..start + count {
                    let ci = coloring.order[slot as usize] as usize;
                    let c = &constraints[ci];
                    let i = c.particle as usize;
                    let (Some(&w), Some(&position)) = (inverse_masses.get(i), out.get(i)) else {
                        continue;
                    };
                    if w <= 0.0 {
                        continue;
                    }
                    let delta = position - c.anchor();
                    let length = delta.length();
                    if length < f32::EPSILON {
                        continue;
                    }
                    let err = length - c.max_distance;
                    if err <= 0.0 {
                        continue;
                    }
                    let normal = delta / length;
                    let alpha_tilde = c.compliance / (dt * dt);
                    let delta_lambda = (-err - alpha_tilde * lambdas[ci]) / (w + alpha_tilde);
                    lambdas[ci] += delta_lambda;
                    out[i] += normal * (delta_lambda * w);
                }
            }
        }
        out
    }

    fn max_abs_diff(a: &[Vec3], b: &[Vec3]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x - *y).abs().max_element())
            .fold(0.0_f32, f32::max)
    }

    /// A spray of overstretched particles leashed to assorted anchors, plus one
    /// slack particle and two leashes sharing a particle (to exercise the
    /// multi-colour schedule).
    fn scene() -> (Vec<Vec3>, Vec<Real>, Vec<ClothLongRangeConstraint>) {
        let positions = alloc::vec![
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(0.5, 0.0, 0.0), // slack: inside its leash
            Vec3::new(2.0, 2.0, 1.0),
        ];
        let inverse_masses = alloc::vec![1.0_f32; positions.len()];
        let constraints = alloc::vec![
            ClothLongRangeConstraint::new(0, Vec3::ZERO, 1.0, 0.0),
            ClothLongRangeConstraint::new(1, Vec3::ZERO, 1.0, 1.0e-4),
            ClothLongRangeConstraint::new(2, Vec3::ZERO, 1.0, 0.0),
            ClothLongRangeConstraint::new(3, Vec3::new(1.0, 1.0, 0.0), 0.5, 0.0),
            // A second leash on particle 3, from a different anchor.
            ClothLongRangeConstraint::new(3, Vec3::new(0.0, 0.0, 0.0), 2.0, 0.0),
        ];
        (positions, inverse_masses, constraints)
    }

    #[test]
    fn empty_or_zero_iterations_is_noop() {
        let (positions, inverse_masses, constraints) = scene();
        let none = cpu_cloth_long_range(&positions, &inverse_masses, &[], 1.0 / 60.0, 8);
        assert_eq!(none, positions);
        let zero = cpu_cloth_long_range(&positions, &inverse_masses, &constraints, 1.0 / 60.0, 0);
        assert_eq!(zero, positions);
    }

    #[test]
    fn matches_brute_force_reference() {
        let (positions, inverse_masses, constraints) = scene();
        for iterations in [1_u32, 3, 8] {
            let golden = cpu_cloth_long_range(
                &positions,
                &inverse_masses,
                &constraints,
                1.0 / 60.0,
                iterations,
            );
            let brute = brute_force(
                &positions,
                &inverse_masses,
                &constraints,
                1.0 / 60.0,
                iterations,
            );
            assert!(
                max_abs_diff(&golden, &brute) <= 1.0e-6,
                "golden drifted from the brute-force anchor at {iterations} iterations"
            );
        }
    }

    #[test]
    fn rigid_leash_snaps_onto_sphere() {
        let positions = alloc::vec![Vec3::new(3.0, 0.0, 0.0)];
        let inverse_masses = alloc::vec![1.0_f32];
        let constraints = alloc::vec![ClothLongRangeConstraint::new(0, Vec3::ZERO, 1.0, 0.0)];
        let out = cpu_cloth_long_range(&positions, &inverse_masses, &constraints, 1.0 / 60.0, 1);
        assert!((out[0].length() - 1.0).abs() <= 1.0e-5, "pos {:?}", out[0]);
    }

    #[test]
    fn slack_particle_is_inert() {
        let positions = alloc::vec![Vec3::new(0.5, 0.0, 0.0)];
        let inverse_masses = alloc::vec![1.0_f32];
        let constraints = alloc::vec![ClothLongRangeConstraint::new(0, Vec3::ZERO, 1.0, 0.0)];
        let out = cpu_cloth_long_range(&positions, &inverse_masses, &constraints, 1.0 / 60.0, 8);
        assert_eq!(out, positions);
    }

    #[test]
    fn pinned_particle_does_not_move() {
        let positions = alloc::vec![Vec3::new(3.0, 0.0, 0.0)];
        let inverse_masses = alloc::vec![0.0_f32];
        let constraints = alloc::vec![ClothLongRangeConstraint::new(0, Vec3::ZERO, 1.0, 0.0)];
        let out = cpu_cloth_long_range(&positions, &inverse_masses, &constraints, 1.0 / 60.0, 8);
        assert_eq!(out, positions);
    }
}
