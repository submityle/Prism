//! The `CPU` golden twin for the cloth strain-limit kernel.
//!
//! The authoritative per-edge clamp lives in [`prism_physics_core`] as
//! `project_strain_limit`; rather than copy that arithmetic and risk it
//! drifting, [`cpu_cloth_strain_limit`] *delegates* every projection to it and
//! only owns the colour-batched sweep schedule, which is exactly what the
//! [`GpuClothStrainLimit`](super::gpu::GpuClothStrainLimit) kernel runs. The
//! parity suite then compares the two applied-position fields within a tight
//! tolerance.
//!
//! The twin is in turn anchored, in this module's tests, against an independent
//! brute-force reference that re-derives the biphasic clamp formula inline over
//! the identical colour schedule, closing the loop from first principles (no
//! fake parity).
//!
//! # Provenance
//!
//! Biphasic strain limiting is a standard, publicly documented cloth technique
//! (Provot 1995; Thomaszewski et al. 2009). No Unreal Engine source or derived
//! code.

use alloc::vec::Vec;

use glam::Vec3;
use prism_physics_core::project_strain_limit;

use super::coloring::colour_strain_limit;
use super::ClothStrainLimitConstraint;

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// Runs `iterations` colour-batched strain-limit sweeps on the `CPU`, returning
/// the applied positions.
///
/// This is the golden twin of
/// [`GpuClothStrainLimit::solve`](super::gpu::GpuClothStrainLimit::solve): the
/// edges are graph-coloured once, then each iteration visits every colour class
/// in ascending order and clamps its edges. The clamp is stateless (a pure
/// geometric projection), so there is no accumulator to carry across sweeps.
///
/// An empty constraint set, zero iterations, or an `inverse_masses` slice
/// shorter than the particles the edges reference leaves `positions` unchanged
/// (out-of-range edges are individually skipped by [`project_strain_limit`]).
#[must_use]
pub fn cpu_cloth_strain_limit(
    positions: &[Vec3],
    inverse_masses: &[Real],
    constraints: &[ClothStrainLimitConstraint],
    iterations: u32,
) -> Vec<Vec3> {
    let mut out = positions.to_vec();
    if constraints.is_empty() || iterations == 0 {
        return out;
    }

    let coloring = colour_strain_limit(constraints);
    for _ in 0..iterations {
        for &(start, count) in &coloring.ranges {
            for slot in start..start + count {
                let ci = coloring.order[slot as usize] as usize;
                let c = &constraints[ci];
                project_strain_limit(
                    &mut out,
                    inverse_masses,
                    c.a,
                    c.b,
                    c.rest_length,
                    c.max_scale,
                    c.min_scale,
                );
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An independent brute-force reference that re-derives the biphasic clamp
    /// inline over the identical colour schedule, with no call into
    /// [`project_strain_limit`]. If this agrees with [`cpu_cloth_strain_limit`]
    /// the delegation is proven arithmetically faithful.
    fn brute_force(
        positions: &[Vec3],
        inverse_masses: &[Real],
        constraints: &[ClothStrainLimitConstraint],
        iterations: u32,
    ) -> Vec<Vec3> {
        let mut out = positions.to_vec();
        if constraints.is_empty() || iterations == 0 {
            return out;
        }
        let coloring = colour_strain_limit(constraints);
        for _ in 0..iterations {
            for &(start, count) in &coloring.ranges {
                for slot in start..start + count {
                    let ci = coloring.order[slot as usize] as usize;
                    let c = &constraints[ci];
                    let (ia, ib) = (c.a as usize, c.b as usize);
                    if ia == ib {
                        continue;
                    }
                    let (Some(&wa), Some(&wb)) = (inverse_masses.get(ia), inverse_masses.get(ib))
                    else {
                        continue;
                    };
                    let w_sum = wa + wb;
                    if w_sum <= 0.0 {
                        continue;
                    }
                    let delta = out[ia] - out[ib];
                    let length = delta.length();
                    if length < f32::EPSILON {
                        continue;
                    }
                    let max_len = c.rest_length * c.max_scale;
                    let min_len = c.rest_length * c.min_scale;
                    let error = if length > max_len {
                        length - max_len
                    } else if c.min_scale > 0.0 && length < min_len {
                        length - min_len
                    } else {
                        continue;
                    };
                    let direction = delta / length;
                    let correction = direction * error;
                    out[ia] -= correction * (wa / w_sum);
                    out[ib] += correction * (wb / w_sum);
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

    /// A row of particles pulled apart so several edges are overstretched and a
    /// couple are compressed, exercising both arms of the biphasic clamp.
    fn stretched_row() -> (Vec<Vec3>, Vec<Real>, Vec<ClothStrainLimitConstraint>) {
        let positions = alloc::vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.6, 0.0, 0.0),
            Vec3::new(3.4, 0.0, 0.0),
            Vec3::new(3.6, 0.0, 0.0),
            Vec3::new(5.3, 0.0, 0.0),
            Vec3::new(7.0, 0.0, 0.0),
        ];
        let inverse_masses = alloc::vec![1.0_f32; positions.len()];
        // Rest length 1, 10% stretch cap, 50% compression floor.
        let constraints: Vec<ClothStrainLimitConstraint> = (0..5)
            .map(|i| ClothStrainLimitConstraint::new(i, i + 1, 1.0, 1.1, 0.5))
            .collect();
        (positions, inverse_masses, constraints)
    }

    #[test]
    fn empty_or_zero_iterations_is_noop() {
        let (positions, inverse_masses, constraints) = stretched_row();
        let none = cpu_cloth_strain_limit(&positions, &inverse_masses, &[], 8);
        assert_eq!(none, positions);
        let zero = cpu_cloth_strain_limit(&positions, &inverse_masses, &constraints, 0);
        assert_eq!(zero, positions);
    }

    #[test]
    fn matches_brute_force_reference() {
        let (positions, inverse_masses, constraints) = stretched_row();
        for iterations in [1_u32, 3, 8] {
            let golden =
                cpu_cloth_strain_limit(&positions, &inverse_masses, &constraints, iterations);
            let brute = brute_force(&positions, &inverse_masses, &constraints, iterations);
            assert!(
                max_abs_diff(&golden, &brute) <= 1.0e-6,
                "golden drifted from the brute-force anchor at {iterations} iterations"
            );
        }
    }

    #[test]
    fn pinned_endpoints_do_not_move() {
        let (positions, mut inverse_masses, constraints) = stretched_row();
        inverse_masses[0] = 0.0;
        let last = inverse_masses.len() - 1;
        inverse_masses[last] = 0.0;
        let out = cpu_cloth_strain_limit(&positions, &inverse_masses, &constraints, 8);
        assert_eq!(out[0], positions[0]);
        assert_eq!(out[last], positions[last]);
    }

    #[test]
    fn caps_every_edge_within_the_band_after_enough_sweeps() {
        let (positions, inverse_masses, constraints) = stretched_row();
        let out = cpu_cloth_strain_limit(&positions, &inverse_masses, &constraints, 64);
        for c in &constraints {
            let length = (out[c.a as usize] - out[c.b as usize]).length();
            let max_len = c.rest_length * c.max_scale;
            let min_len = c.rest_length * c.min_scale;
            assert!(
                length <= max_len + 1.0e-4 && length >= min_len - 1.0e-4,
                "edge {}-{} length {length} escaped band [{min_len}, {max_len}]",
                c.a,
                c.b
            );
        }
    }
}
