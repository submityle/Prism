//! Jacobi (parallel-safe) virtual-particle self-collision — the GPU-faithful
//! golden, delegated to the physics engine single source of truth.
//!
//! [`super::virtual_particles::resolve_self_collision_virtual`] resolves the
//! `NvCloth`-style virtual-particle tier in **Gauss-Seidel** order: it walks the
//! sample pairs in a fixed order and each half-correction is scattered onto the
//! real vertices *in place*, so a later pair already sees the moved positions of
//! an earlier one. That is the sequential CPU reference, but it does not map to a
//! `GPU` compute kernel: on the `GPU` every sample is updated in parallel from
//! the *same* read-only snapshot, which is a **Jacobi** iteration, not
//! Gauss-Seidel. The two converge to the same separated state but are never
//! bit-for-bit identical on a single pass, so a faithful cloth-self-collision
//! `WGSL`/`WESL` twin needs its own golden rather than borrowing the
//! Gauss-Seidel one (design §9: no fake parity).
//!
//! The Jacobi algorithm itself — the two own-slot phases (per-sample half-push
//! accumulation from the frozen snapshot, then per-vertex barycentric scatter),
//! the deterministic ascending reductions, and every degenerate fallback — now
//! lives once in
//! [`prism_physics_core::soft::collision::resolve_self_collision_virtual_jacobi`]
//! and its augment twin. The render-facing [`ClothParticle`]-based entry points
//! below convert to the physics SoA layout, delegate, and write the solved
//! positions back, so the CPU golden the `WESL` twin mirrors is byte-for-byte
//! the physics solver's output.

use super::virtual_particles::VirtualParticle;
use super::{physics_bridge, ClothParticle};
use prism_physics_core::soft::collision as physics_collision;

/// Runs one Jacobi virtual-particle self-collision pass over the full sample set
/// (real-vs-real included), the parallel-safe twin of
/// [`super::virtual_particles::resolve_self_collision_virtual`].
///
/// One call is a single Jacobi iteration (accumulate from the frozen snapshot,
/// then apply); repeated calls converge to the same separated state the
/// Gauss-Seidel core reaches, without ever depending on evaluation order.
///
/// The particles are converted to the physics SoA layout (pinned particles map
/// to inverse mass `0`), resolved by
/// [`prism_physics_core::soft::collision::resolve_self_collision_virtual_jacobi`],
/// and the solved positions are written back. A non-positive `cell_size` or
/// `thickness` is a cheap no-op that skips the conversion entirely.
pub fn resolve_self_collision_virtual_jacobi(
    particles: &mut [ClothParticle],
    virtuals: &[VirtualParticle],
    cell_size: f32,
    thickness: f32,
) {
    if cell_size <= 0.0 || thickness <= 0.0 {
        return;
    }
    let (mut positions, inverse_masses) = physics_bridge::to_soa(particles);
    physics_collision::resolve_self_collision_virtual_jacobi(
        &mut positions,
        &inverse_masses,
        virtuals,
        cell_size,
        thickness,
    );
    physics_bridge::write_positions_back(particles, &positions);
}

/// Runs one Jacobi virtual-particle augment pass (virtual-touching pairs only),
/// the parallel-safe twin of
/// [`super::virtual_particles::resolve_self_collision_virtual_augment`].
///
/// Real-vs-real pairs are skipped so this layers on top of a friction
/// point-to-point tier without stripping its tangential friction (design §6.2).
///
/// Delegates to
/// [`prism_physics_core::soft::collision::resolve_self_collision_virtual_augment_jacobi`]
/// after the same SoA conversion as [`resolve_self_collision_virtual_jacobi`].
pub fn resolve_self_collision_virtual_augment_jacobi(
    particles: &mut [ClothParticle],
    virtuals: &[VirtualParticle],
    cell_size: f32,
    thickness: f32,
) {
    if cell_size <= 0.0 || thickness <= 0.0 {
        return;
    }
    let (mut positions, inverse_masses) = physics_bridge::to_soa(particles);
    physics_collision::resolve_self_collision_virtual_augment_jacobi(
        &mut positions,
        &inverse_masses,
        virtuals,
        cell_size,
        thickness,
    );
    physics_bridge::write_positions_back(particles, &positions);
}

#[cfg(test)]
mod tests {
    use super::super::virtual_particles::{
        generate_virtual_particles, resolve_self_collision_virtual, VirtualParticlePattern,
    };
    use super::super::Vec3;
    use super::*;

    fn particle(x: f32, y: f32, z: f32, inverse_mass: f32) -> ClothParticle {
        ClothParticle::new(Vec3::new(x, y, z), inverse_mass)
    }

    /// The two stacked triangles from the Gauss-Seidel suite: their centroid
    /// virtual particles collide.
    fn two_layers() -> [ClothParticle; 6] {
        [
            particle(0.0, 0.0, 0.0, 1.0),
            particle(4.0, 0.0, 0.0, 1.0),
            particle(0.0, 4.0, 0.0, 1.0),
            particle(0.0, 0.0, 0.05, 1.0),
            particle(4.0, 0.0, 0.05, 1.0),
            particle(0.0, 4.0, 0.05, 1.0),
        ]
    }

    #[test]
    fn resolution_is_bit_identical_across_runs() {
        let base = two_layers();
        let virtuals = generate_virtual_particles(
            &[[0, 1, 2], [3, 4, 5]],
            &VirtualParticlePattern::nvcloth_default(),
        );

        let mut a = base;
        resolve_self_collision_virtual_jacobi(&mut a, &virtuals, 1.0, 0.2);
        let mut b = base;
        resolve_self_collision_virtual_jacobi(&mut b, &virtuals, 1.0, 0.2);
        for i in 0..base.len() {
            assert_eq!(a[i].position, b[i].position, "Jacobi pass must be deterministic");
        }
    }

    #[test]
    fn degenerate_parameters_are_a_no_op() {
        let base = two_layers();
        let virtuals = generate_virtual_particles(
            &[[0, 1, 2], [3, 4, 5]],
            &VirtualParticlePattern::nvcloth_default(),
        );
        for (cell, thick) in [(0.0, 0.2), (1.0, 0.0), (-1.0, -1.0)] {
            let mut particles = base;
            resolve_self_collision_virtual_jacobi(&mut particles, &virtuals, cell, thick);
            for i in 0..base.len() {
                assert_eq!(
                    particles[i].position, base[i].position,
                    "cell={cell} thick={thick}"
                );
            }
        }
    }

    #[test]
    fn empty_virtuals_is_a_no_op_when_reals_are_apart() {
        // Two well-separated free vertices: with no virtual particles and no
        // real pair within thickness, nothing moves.
        let base = [particle(0.0, 0.0, 0.0, 1.0), particle(5.0, 0.0, 0.0, 1.0)];
        let mut particles = base;
        resolve_self_collision_virtual_jacobi(&mut particles, &[], 1.0, 0.2);
        for i in 0..base.len() {
            assert_eq!(particles[i].position, base[i].position);
        }
    }

    #[test]
    fn pinned_triangle_pushes_only_the_single_intruder() {
        // A pinned triangle with one free intruder above its centroid. Only the
        // centroid virtual particle is within thickness, so a single Jacobi
        // pass matches the Gauss-Seidel single-pair result exactly.
        let base = [
            particle(0.0, 0.0, 0.0, 0.0),
            particle(4.0, 0.0, 0.0, 0.0),
            particle(0.0, 4.0, 0.0, 0.0),
            particle(4.0 / 3.0, 4.0 / 3.0, 0.05, 1.0),
        ];
        let virtuals =
            generate_virtual_particles(&[[0, 1, 2]], &VirtualParticlePattern::nvcloth_default());

        let mut jac = base;
        resolve_self_collision_virtual_jacobi(&mut jac, &virtuals, 1.0, 0.2);
        // Pinned corners never move.
        for i in 0..3 {
            assert_eq!(jac[i].position, base[i].position);
        }
        // The intruder ends a full thickness out.
        assert!(
            (jac[3].position.z - 0.2).abs() < 1e-5,
            "z = {}",
            jac[3].position.z
        );
    }

    #[test]
    fn augment_skips_real_pairs_that_the_full_pass_resolves() {
        // Two near-coincident free real vertices with no triangle: the full pass
        // separates them, the augment pass leaves them to the point tier.
        let base = [particle(0.0, 0.0, 0.0, 1.0), particle(0.05, 0.0, 0.0, 1.0)];

        let mut full = base;
        resolve_self_collision_virtual_jacobi(&mut full, &[], 1.0, 0.2);
        assert!(
            full[0].position != base[0].position && full[1].position != base[1].position,
            "the full pass must separate the real-vs-real pair"
        );

        let mut aug = base;
        resolve_self_collision_virtual_augment_jacobi(&mut aug, &[], 1.0, 0.2);
        for i in 0..base.len() {
            assert_eq!(
                aug[i].position, base[i].position,
                "the augment pass must leave real-vs-real pairs untouched"
            );
        }
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

        let mut gs = two_layers();
        resolve_self_collision_virtual(&mut gs, &virtuals, 1.0, 0.2);

        let mut jac = two_layers();
        for _ in 0..64 {
            resolve_self_collision_virtual_jacobi(&mut jac, &virtuals, 1.0, 0.2);
        }

        // Lower face sinks in -z, upper face rises in +z, for both solvers.
        for i in 0..3 {
            assert!(gs[i].position.z < -1e-6);
            assert!(jac[i].position.z < -1e-6);
        }
        for i in 3..6 {
            assert!(gs[i].position.z > 1e-6);
            assert!(jac[i].position.z > 1e-6);
        }
        // The converged Jacobi centroid gap reaches at least the fabric
        // thickness, matching the Gauss-Seidel separation target.
        let jac_gap = jac[3].position.z - jac[0].position.z;
        assert!(jac_gap >= 0.2 - 1e-3, "converged gap {jac_gap} < thickness");
    }
}
