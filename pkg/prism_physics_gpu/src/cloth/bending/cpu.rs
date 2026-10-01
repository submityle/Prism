//! The `CPU` golden twin for the cloth bending kernel.
//!
//! The authoritative per-joint bending projection lives in
//! [`prism_physics_core`] as `project_bending`; rather than copy that
//! arithmetic and risk it drifting, [`cpu_cloth_bending`] *delegates* every
//! projection to it and only owns the colour-batched sweep schedule, which is
//! exactly what the [`GpuClothBending`](super::gpu::GpuClothBending) kernel
//! runs. The parity suite then compares the two applied-position fields within a
//! tight tolerance.
//!
//! The twin is in turn anchored, in this module's tests, against an independent
//! brute-force reference that re-derives the compliant projection formula
//! inline over the identical colour schedule, closing the loop from first
//! principles (no fake parity).
//!
//! # Provenance
//!
//! The compliant `XPBD` bending projection is the published Müller et al.
//! position-based-dynamics technique. No Unreal Engine source or derived code.

use alloc::vec::Vec;

use glam::Vec3;
use prism_physics_core::project_bending;

use super::coloring::colour_bending;
use super::ClothBendingConstraint;

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// Runs `iterations` colour-batched bending sweeps on the `CPU`, returning the
/// applied positions.
///
/// This is the golden twin of
/// [`GpuClothBending::solve`](super::gpu::GpuClothBending::solve): the
/// constraints are graph-coloured once, then each iteration visits every colour
/// class in ascending order and projects its constraints. The accumulated
/// `XPBD` Lagrange multipliers are cleared once at the start and carried across
/// all iterations (standard `XPBD` warm-free accumulation), never reset between
/// sweeps.
///
/// An empty constraint set, zero iterations, or an `inverse_masses` slice
/// shorter than the particles the constraints reference leaves `positions`
/// unchanged (out-of-range joints are individually skipped by
/// [`project_bending`]).
#[must_use]
pub fn cpu_cloth_bending(
    positions: &[Vec3],
    inverse_masses: &[Real],
    constraints: &[ClothBendingConstraint],
    dt: Real,
    iterations: u32,
) -> Vec<Vec3> {
    let mut out = positions.to_vec();
    if constraints.is_empty() || iterations == 0 {
        return out;
    }

    let coloring = colour_bending(constraints);
    // One accumulated multiplier per constraint, cleared once for the whole
    // solve so each iteration warm-continues the previous sweep.
    let mut lambdas: Vec<Real> = Vec::new();
    lambdas.resize(constraints.len(), 0.0);

    for _ in 0..iterations {
        for &(start, count) in &coloring.ranges {
            for slot in start..start + count {
                let ci = coloring.order[slot as usize] as usize;
                let c = &constraints[ci];
                lambdas[ci] = project_bending(
                    &mut out,
                    inverse_masses,
                    c.a,
                    c.center,
                    c.b,
                    c.rest_offset,
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

    /// An independent brute-force reference that re-derives the compliant
    /// bending projection inline over the identical colour schedule, with no
    /// call into [`project_bending`]. If this agrees with [`cpu_cloth_bending`]
    /// the delegation is proven arithmetically faithful.
    fn brute_force(
        positions: &[Vec3],
        inverse_masses: &[Real],
        constraints: &[ClothBendingConstraint],
        dt: Real,
        iterations: u32,
    ) -> Vec<Vec3> {
        let mut out = positions.to_vec();
        if constraints.is_empty() || iterations == 0 {
            return out;
        }
        let coloring = colour_bending(constraints);
        let mut lambdas = Vec::new();
        lambdas.resize(constraints.len(), 0.0_f32);
        for _ in 0..iterations {
            for &(start, count) in &coloring.ranges {
                for slot in start..start + count {
                    let ci = coloring.order[slot as usize] as usize;
                    let c = &constraints[ci];
                    let (ia, ic, ib) = (c.a as usize, c.center as usize, c.b as usize);
                    let (Some(&wa), Some(&wc), Some(&wb)) = (
                        inverse_masses.get(ia),
                        inverse_masses.get(ic),
                        inverse_masses.get(ib),
                    ) else {
                        continue;
                    };
                    let denom = wc + 0.25 * (wa + wb);
                    if denom <= 0.0 {
                        continue;
                    }
                    let midpoint = (out[ia] + out[ib]) * 0.5;
                    let delta = out[ic] - midpoint;
                    let length = delta.length();
                    if length < f32::EPSILON {
                        continue;
                    }
                    let normal = delta / length;
                    let err = length - c.rest_offset;
                    let alpha_tilde = c.compliance / (dt * dt);
                    let delta_lambda = (-err - alpha_tilde * lambdas[ci]) / (denom + alpha_tilde);
                    lambdas[ci] += delta_lambda;
                    out[ic] += normal * (delta_lambda * wc);
                    out[ia] -= normal * (delta_lambda * wa * 0.5);
                    out[ib] -= normal * (delta_lambda * wb * 0.5);
                }
            }
        }
        out
    }

    fn cst(a: u32, center: u32, b: u32, rest: f32, compliance: f32) -> ClothBendingConstraint {
        ClothBendingConstraint {
            a,
            center,
            b,
            rest_offset: rest,
            compliance,
        }
    }

    /// A small folded strip: a row of particles with a kink at the hinges so the
    /// bending constraints have a real correction to apply.
    fn folded_strip() -> (Vec<Vec3>, Vec<Real>, Vec<ClothBendingConstraint>) {
        let positions = alloc::vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.6, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.7, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(5.0, 0.5, 0.0),
            Vec3::new(6.0, 0.0, 0.0),
        ];
        let inverse_masses = alloc::vec![1.0_f32; positions.len()];
        let constraints = alloc::vec![
            cst(0, 1, 2, 0.0, 0.0),
            cst(1, 2, 3, 0.0, 0.0),
            cst(2, 3, 4, 0.0, 1.0e-4),
            cst(3, 4, 5, 0.0, 0.0),
            cst(4, 5, 6, 0.0, 0.0),
        ];
        (positions, inverse_masses, constraints)
    }

    fn max_abs_diff(a: &[Vec3], b: &[Vec3]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x - *y).abs().max_element())
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn empty_or_zero_iterations_is_noop() {
        let (positions, inverse_masses, constraints) = folded_strip();
        let none = cpu_cloth_bending(&positions, &inverse_masses, &[], 1.0 / 60.0, 8);
        assert_eq!(none, positions);
        let zero = cpu_cloth_bending(&positions, &inverse_masses, &constraints, 1.0 / 60.0, 0);
        assert_eq!(zero, positions);
    }

    #[test]
    fn matches_brute_force_reference() {
        let (positions, inverse_masses, constraints) = folded_strip();
        for iterations in [1_u32, 3, 8] {
            let golden =
                cpu_cloth_bending(&positions, &inverse_masses, &constraints, 1.0 / 60.0, iterations);
            let brute =
                brute_force(&positions, &inverse_masses, &constraints, 1.0 / 60.0, iterations);
            assert!(
                max_abs_diff(&golden, &brute) <= 1.0e-6,
                "golden drifted from the brute-force anchor at {iterations} iterations"
            );
        }
    }

    #[test]
    fn pinned_endpoints_do_not_move() {
        let (positions, mut inverse_masses, constraints) = folded_strip();
        // Pin the two ends.
        inverse_masses[0] = 0.0;
        let last = inverse_masses.len() - 1;
        inverse_masses[last] = 0.0;
        let out = cpu_cloth_bending(&positions, &inverse_masses, &constraints, 1.0 / 60.0, 8);
        assert_eq!(out[0], positions[0]);
        assert_eq!(out[last], positions[last]);
    }

    #[test]
    fn flattens_a_rigid_hinge() {
        // A single rigid (zero-compliance, zero-rest) hinge should pull the
        // centre onto the midpoint of its neighbours.
        let positions = alloc::vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let inverse_masses = alloc::vec![1.0_f32; 3];
        let constraints = alloc::vec![cst(0, 1, 2, 0.0, 0.0)];
        let out = cpu_cloth_bending(&positions, &inverse_masses, &constraints, 1.0 / 60.0, 40);
        let midpoint = (out[0] + out[2]) * 0.5;
        assert!(
            (out[1] - midpoint).length() <= 1.0e-4,
            "rigid hinge failed to flatten onto the midpoint"
        );
    }
}
