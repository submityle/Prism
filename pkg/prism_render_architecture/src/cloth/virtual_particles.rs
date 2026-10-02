//! `NvCloth`-style virtual particles for cloth self-collision — render-side
//! thin wrapper that delegates to the physics engine single source of truth.
//!
//! The discrete self-collision tier in [`collision`](super::collision) is a
//! point-to-point spatial hash: it separates cloth *vertices* that come within
//! a fabric `thickness` of one another. That catches most layer stacking, but
//! it has a well-known blind spot (design §6.2): a lone vertex can slip
//! straight *through the interior of a large triangle* without ever coming
//! within `thickness` of any of that triangle's three corner vertices, so no
//! point-to-point pair ever fires and the layers interpenetrate.
//!
//! NVIDIA `NvCloth` closes that gap without paying for a full continuous
//! triangle-triangle test by seeding each triangle with a handful of **virtual
//! particles**: fixed barycentric sample points on the face (its centroid and
//! edge midpoints) that are fed into the *same* uniform spatial hash as the
//! real vertices. A vertex diving through a face now finds a virtual particle
//! sitting on that face and is pushed back out, while the correction applied to
//! the virtual particle is scattered back onto the triangle's three real
//! vertices by the barycentric weights so momentum and mass are conserved.
//!
//! The algorithm (sample ordering, deterministic spatial hash, Gauss-Seidel
//! scatter, incident-pair suppression, pinned handling and degenerate
//! fallbacks) now lives once in
//! [`prism_physics_core::soft::collision`]. The [`VirtualParticle`] and
//! [`VirtualParticlePattern`] data types are re-exported from physics so there
//! is a single definition, and the render-facing [`ClothParticle`]-based
//! entry points below simply convert to the physics SoA layout, delegate, and
//! write the solved positions back.
//!
//! Passing an empty virtual-particle slice makes
//! [`resolve_self_collision_virtual`] behave exactly like the point-to-point
//! [`resolve_self_collision`](super::collision::resolve_self_collision): the
//! real-vertex samples still collide with one another, so this is a strict
//! superset of the base tier rather than a replacement.

use alloc::vec::Vec;

use super::{physics_bridge, ClothParticle};
use prism_physics_core::soft::collision as physics_collision;

/// The `NvCloth` barycentric sample pattern — re-exported from the physics
/// engine so the render tier and the solver share a single definition.
pub use physics_collision::VirtualParticlePattern;

/// A barycentric sample bound to one triangle's three real vertices —
/// re-exported from the physics engine so there is a single definition.
pub use physics_collision::VirtualParticle;

/// Seeds every triangle with the pattern's virtual particles.
///
/// Emits one [`VirtualParticle`] per `(triangle, pattern row)` pair, so the
/// result has length `triangles.len() * pattern.len()` for meshes of
/// non-degenerate triangles. Degenerate triangles (any two corner indices
/// equal) are skipped: their zero-area face cannot host a meaningful
/// barycentric sample and would break the mass-conserving scatter, so they
/// contribute no virtual particles and the result may be shorter.
///
/// This is a straight delegation to
/// [`prism_physics_core::soft::collision::generate_virtual_particles`].
#[must_use]
pub fn generate_virtual_particles(
    triangles: &[[u32; 3]],
    pattern: &VirtualParticlePattern,
) -> Vec<VirtualParticle> {
    physics_collision::generate_virtual_particles(triangles, pattern)
}

/// Runs the full virtual-particle self-collision tier.
///
/// This resolves every sample pair — real-vertex versus real-vertex included —
/// so with an empty `virtuals` slice it reduces exactly to the point-to-point
/// [`resolve_self_collision`](super::collision::resolve_self_collision), and
/// with virtual particles it additionally catches vertices tunnelling through
/// triangle interiors. Use this when the virtual tier is the *only*
/// self-collision pass; use [`resolve_self_collision_virtual_augment`] to layer
/// it on top of the friction point-to-point pass.
///
/// The particles are converted to the physics SoA layout (pinned particles map
/// to inverse mass `0`), resolved by
/// [`prism_physics_core::soft::collision::resolve_self_collision_virtual`], and
/// the solved positions are written back. A non-positive `cell_size` or
/// `thickness` is a cheap no-op that skips the conversion entirely.
pub fn resolve_self_collision_virtual(
    particles: &mut [ClothParticle],
    virtuals: &[VirtualParticle],
    cell_size: f32,
    thickness: f32,
) {
    if cell_size <= 0.0 || thickness <= 0.0 {
        return;
    }
    let (mut positions, inverse_masses) = physics_bridge::to_soa(particles);
    physics_collision::resolve_self_collision_virtual(
        &mut positions,
        &inverse_masses,
        virtuals,
        cell_size,
        thickness,
    );
    physics_bridge::write_positions_back(particles, &positions);
}

/// Augments an existing point-to-point self-collision pass with virtual
/// particles, resolving only pairs where at least one sample is virtual.
///
/// The friction point-to-point tier
/// ([`resolve_self_collision_with_friction`](super::collision::resolve_self_collision_with_friction))
/// already separates and rubs real-vertex pairs, so re-resolving them here would
/// undo their tangential friction. This pass therefore skips real-vertex versus
/// real-vertex pairs and adds only the vertex-versus-face and face-versus-face
/// coverage the point tier cannot see (design §6.2). An empty `virtuals` slice
/// is a no-op.
///
/// Delegates to
/// [`prism_physics_core::soft::collision::resolve_self_collision_virtual_augment`]
/// after the same SoA conversion as [`resolve_self_collision_virtual`].
pub fn resolve_self_collision_virtual_augment(
    particles: &mut [ClothParticle],
    virtuals: &[VirtualParticle],
    cell_size: f32,
    thickness: f32,
) {
    if cell_size <= 0.0 || thickness <= 0.0 {
        return;
    }
    let (mut positions, inverse_masses) = physics_bridge::to_soa(particles);
    physics_collision::resolve_self_collision_virtual_augment(
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
    use super::*;
    use super::super::Vec3;

    fn particle(x: f32, y: f32, z: f32, inverse_mass: f32) -> ClothParticle {
        ClothParticle::new(Vec3::new(x, y, z), inverse_mass)
    }

    #[test]
    fn default_pattern_rows_are_normalized_and_non_negative() {
        let pattern = VirtualParticlePattern::nvcloth_default();
        assert_eq!(pattern.len(), 4);
        for row in pattern.rows() {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "row {row:?} sums to {sum}");
            assert!(row.iter().all(|&w| w >= 0.0), "row {row:?} has a negative");
        }
    }

    #[test]
    fn sanitize_clamps_negatives_and_renormalizes() {
        let pattern = VirtualParticlePattern::from_weights(&[[2.0, -1.0, 0.0]]);
        let row = pattern.rows()[0];
        // The negative is dropped, the rest renormalized: [2, 0, 0] -> [1, 0, 0].
        assert_eq!(row, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn sanitize_falls_back_to_centroid_for_degenerate_rows() {
        let pattern = VirtualParticlePattern::from_weights(&[
            [0.0, 0.0, 0.0],
            [f32::NAN, -1.0, f32::INFINITY],
        ]);
        for row in pattern.rows() {
            assert_eq!(*row, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        }
    }

    #[test]
    fn generate_emits_one_particle_per_triangle_row() {
        let pattern = VirtualParticlePattern::nvcloth_default();
        let triangles = [[0u32, 1, 2], [1, 2, 3]];
        let virtuals = generate_virtual_particles(&triangles, &pattern);
        assert_eq!(virtuals.len(), triangles.len() * pattern.len());
        assert_eq!(virtuals[0].verts, [0, 1, 2]);
        assert_eq!(virtuals[pattern.len()].verts, [1, 2, 3]);
    }

    #[test]
    fn generate_skips_degenerate_triangles() {
        let pattern = VirtualParticlePattern::nvcloth_default();
        let triangles = [[0u32, 0, 1], [2, 3, 4]];
        let virtuals = generate_virtual_particles(&triangles, &pattern);
        // Only the second, non-degenerate triangle contributes.
        assert_eq!(virtuals.len(), pattern.len());
        assert!(virtuals.iter().all(|v| v.verts == [2, 3, 4]));
    }

    /// The headline case: a vertex diving through a large triangle's interior
    /// is missed by the point-to-point tier but caught by a virtual particle.
    #[test]
    fn virtual_particle_catches_a_vertex_through_a_triangle() {
        // Big triangle (indices 0,1,2) in the z=0 plane, intruder (index 3)
        // just above its centroid, well inside `thickness` of the face but far
        // from every corner vertex.
        let base = [
            particle(0.0, 0.0, 0.0, 1.0),
            particle(4.0, 0.0, 0.0, 1.0),
            particle(0.0, 4.0, 0.0, 1.0),
            particle(4.0 / 3.0, 4.0 / 3.0, 0.05, 1.0),
        ];
        let thickness = 0.2;
        let cell_size = 1.0;

        // Point-to-point tier: the intruder is far from all corners, so it
        // never moves.
        let mut plain = base;
        super::super::collision::resolve_self_collision(&mut plain, cell_size, thickness);
        assert_eq!(plain[3].position, base[3].position);

        // Virtual tier: the centroid virtual particle sits under the intruder
        // and pushes it back out along +z.
        let mut virt = base;
        let pattern = VirtualParticlePattern::nvcloth_default();
        let virtuals = generate_virtual_particles(&[[0, 1, 2]], &pattern);
        resolve_self_collision_virtual(&mut virt, &virtuals, cell_size, thickness);
        assert!(
            virt[3].position.z > base[3].position.z + 1e-4,
            "intruder z should be pushed out, got {}",
            virt[3].position.z
        );
    }

    #[test]
    fn pinned_triangle_only_moves_the_intruder() {
        let base = [
            particle(0.0, 0.0, 0.0, 0.0),
            particle(4.0, 0.0, 0.0, 0.0),
            particle(0.0, 4.0, 0.0, 0.0),
            particle(4.0 / 3.0, 4.0 / 3.0, 0.05, 1.0),
        ];
        let thickness = 0.2;
        let mut particles = base;
        let virtuals =
            generate_virtual_particles(&[[0, 1, 2]], &VirtualParticlePattern::nvcloth_default());
        resolve_self_collision_virtual(&mut particles, &virtuals, 1.0, thickness);

        // The pinned corners are untouched.
        for i in 0..3 {
            assert_eq!(particles[i].position, base[i].position);
        }
        // The intruder absorbs the whole correction, ending a full thickness out.
        assert!((particles[3].position.z - thickness).abs() < 1e-5);
    }

    #[test]
    fn correction_scatters_onto_all_three_triangle_vertices() {
        let base = [
            particle(0.0, 0.0, 0.0, 1.0),
            particle(4.0, 0.0, 0.0, 1.0),
            particle(0.0, 4.0, 0.0, 1.0),
            particle(4.0 / 3.0, 4.0 / 3.0, 0.05, 1.0),
        ];
        let mut particles = base;
        let virtuals =
            generate_virtual_particles(&[[0, 1, 2]], &VirtualParticlePattern::nvcloth_default());
        resolve_self_collision_virtual(&mut particles, &virtuals, 1.0, 0.2);

        // Every free corner shares the reaction, so all three move in -z.
        for i in 0..3 {
            assert!(
                particles[i].position.z < base[i].position.z - 1e-6,
                "corner {i} should recoil in -z, got {}",
                particles[i].position.z
            );
        }
        assert!(particles[3].position.z > base[3].position.z);
    }

    #[test]
    fn incident_vertex_of_the_triangle_is_not_separated() {
        // Vertex 0 is a corner of the triangle, so its real sample shares an
        // active vertex with every one of the triangle's virtual particles and
        // must never be pushed by them.
        let base = [
            particle(0.0, 0.0, 0.0, 1.0),
            particle(4.0, 0.0, 0.0, 1.0),
            particle(0.0, 4.0, 0.0, 1.0),
        ];
        let mut particles = base;
        let virtuals =
            generate_virtual_particles(&[[0, 1, 2]], &VirtualParticlePattern::nvcloth_default());
        resolve_self_collision_virtual(&mut particles, &virtuals, 4.0, 0.5);
        for i in 0..3 {
            assert_eq!(particles[i].position, base[i].position);
        }
    }

    #[test]
    fn two_virtual_layers_separate_and_scatter_to_disjoint_vertices() {
        // Two parallel triangles (disjoint vertex sets 0..3 and 3..6) stacked
        // 0.05 apart in z: their centroid virtual particles collide and push
        // the two faces apart.
        let base = [
            particle(0.0, 0.0, 0.0, 1.0),
            particle(4.0, 0.0, 0.0, 1.0),
            particle(0.0, 4.0, 0.0, 1.0),
            particle(0.0, 0.0, 0.05, 1.0),
            particle(4.0, 0.0, 0.05, 1.0),
            particle(0.0, 4.0, 0.05, 1.0),
        ];
        let mut particles = base;
        let virtuals = generate_virtual_particles(
            &[[0, 1, 2], [3, 4, 5]],
            &VirtualParticlePattern::nvcloth_default(),
        );
        resolve_self_collision_virtual(&mut particles, &virtuals, 1.0, 0.2);

        // Lower face recoils to -z, upper face to +z: the gap widens.
        for i in 0..3 {
            assert!(particles[i].position.z < base[i].position.z - 1e-6);
        }
        for i in 3..6 {
            assert!(particles[i].position.z > base[i].position.z + 1e-6);
        }
    }

    #[test]
    fn empty_virtuals_matches_point_to_point_self_collision() {
        // With no virtual particles the virtual tier must reduce exactly to the
        // point-to-point solver on a pair of near-coincident free vertices.
        let base = [particle(0.0, 0.0, 0.0, 1.0), particle(0.05, 0.0, 0.0, 1.0)];
        let mut plain = base;
        super::super::collision::resolve_self_collision(&mut plain, 1.0, 0.2);

        let mut virt = base;
        resolve_self_collision_virtual(&mut virt, &[], 1.0, 0.2);

        for i in 0..2 {
            assert_eq!(virt[i].position, plain[i].position);
        }
    }

    #[test]
    fn augment_skips_real_real_but_catches_face_penetration() {
        // Triangle 0,1,2 with an intruder (3) above its centroid, plus a bare
        // pair of near-coincident free vertices (4,5) that belong to no
        // triangle. The augment sweep must push the intruder out (real-vs-face)
        // yet leave the bare real-vs-real pair to the point-to-point tier.
        let base = [
            particle(0.0, 0.0, 0.0, 1.0),
            particle(4.0, 0.0, 0.0, 1.0),
            particle(0.0, 4.0, 0.0, 1.0),
            particle(4.0 / 3.0, 4.0 / 3.0, 0.05, 1.0),
            particle(10.0, 10.0, 10.0, 1.0),
            particle(10.03, 10.0, 10.0, 1.0),
        ];
        let mut particles = base;
        let virtuals =
            generate_virtual_particles(&[[0, 1, 2]], &VirtualParticlePattern::nvcloth_default());
        resolve_self_collision_virtual_augment(&mut particles, &virtuals, 1.0, 0.2);

        // Face penetration resolved: the intruder is pushed out along +z.
        assert!(particles[3].position.z > base[3].position.z + 1e-4);
        // The bare real-vertex pair is untouched by the augment sweep.
        assert_eq!(particles[4].position, base[4].position);
        assert_eq!(particles[5].position, base[5].position);
    }

    #[test]
    fn augment_with_empty_virtuals_is_a_full_noop() {
        // No virtual particles means no pair involves a virtual sample, so the
        // augment sweep must not touch even a colliding real-vertex pair.
        let base = [particle(0.0, 0.0, 0.0, 1.0), particle(0.05, 0.0, 0.0, 1.0)];
        let mut particles = base;
        resolve_self_collision_virtual_augment(&mut particles, &[], 1.0, 0.2);
        assert_eq!(particles, base);
    }

    #[test]
    fn resolution_is_deterministic() {
        let base = [
            particle(0.0, 0.0, 0.0, 1.0),
            particle(4.0, 0.0, 0.0, 1.0),
            particle(0.0, 4.0, 0.0, 1.0),
            particle(4.0 / 3.0, 4.0 / 3.0, 0.05, 1.0),
        ];
        let virtuals =
            generate_virtual_particles(&[[0, 1, 2]], &VirtualParticlePattern::nvcloth_default());

        let mut first = base;
        resolve_self_collision_virtual(&mut first, &virtuals, 1.0, 0.2);
        let mut second = base;
        resolve_self_collision_virtual(&mut second, &virtuals, 1.0, 0.2);

        for i in 0..4 {
            assert_eq!(first[i].position, second[i].position);
        }
    }

    #[test]
    fn non_positive_parameters_and_short_input_are_noops() {
        let base = [
            particle(0.0, 0.0, 0.0, 1.0),
            particle(4.0, 0.0, 0.0, 1.0),
            particle(0.0, 4.0, 0.0, 1.0),
            particle(4.0 / 3.0, 4.0 / 3.0, 0.05, 1.0),
        ];
        let virtuals =
            generate_virtual_particles(&[[0, 1, 2]], &VirtualParticlePattern::nvcloth_default());

        let mut zero_cell = base;
        resolve_self_collision_virtual(&mut zero_cell, &virtuals, 0.0, 0.2);
        assert_eq!(zero_cell, base);

        let mut zero_thickness = base;
        resolve_self_collision_virtual(&mut zero_thickness, &virtuals, 1.0, 0.0);
        assert_eq!(zero_thickness, base);

        let mut single = [particle(0.0, 0.0, 0.0, 1.0)];
        resolve_self_collision_virtual(&mut single, &[], 1.0, 0.2);
        assert_eq!(single[0].position, Vec3::ZERO);
    }
}
