//! Race-free, per-vertex *gather* form of the cloth aerodynamic pass.
//!
//! The sibling [`super::wind`] module owns the reference aerodynamics: for each
//! mesh triangle it computes a face force from the relative wind and *scatters*
//! a third of it onto the triangle's three vertices (see
//! [`super::wind::apply_aero_forces`]). On the `CPU` that scatter is a plain
//! serial loop, but a `GPU` compute pass that launches one thread per triangle
//! would have three threads racing to add into the same vertex's velocity — a
//! classic scatter data race. Production `GPU` cloth (`UE5` `Chaos` Cloth and
//! NVIDIA `NvCloth`) sidesteps it by inverting the loop: one thread per
//! **vertex** *gathers* the forces of the triangles incident to that vertex, so
//! every velocity is written by exactly one thread and no atomics are needed.
//!
//! This module provides both halves of that `GPU`-faithful pass:
//!
//! * [`VertexTriangleAdjacency`] — a compact `CSR` (compressed-sparse-row)
//!   vertex→triangle map built once per topology, listing for each vertex the
//!   slice indices of the triangles that touch it, in ascending order.
//! * [`accumulate_aero_gather`] — the per-vertex gather itself, reading a
//!   single velocity snapshot so every face sees the same input state (an
//!   explicit / Jacobi update), which is exactly what a race-free per-vertex
//!   `WESL` kernel over [`super::gpu::kernels::DispatchDomain::Particle`] can
//!   mirror byte-for-byte.
//!
//! ### Relationship to the sequential scatter
//!
//! [`super::wind::apply_aero_forces`] reads and writes velocities in triangle
//! order, so a vertex shared by several triangles feeds each later face its
//! already-updated velocity (a Gauss-Seidel coupling). The gather here instead
//! reads a frozen pre-pass snapshot for every face (a Jacobi update). The two
//! agree exactly when no vertex is shared (e.g. a single triangle) and differ
//! by the intra-step velocity feedback otherwise. The Jacobi form is the one a
//! parallel `GPU` pass can reproduce deterministically, so it — not the
//! sequential scatter — is the golden reference the shader mirrors. Everything
//! here is deterministic and finite: the gather order is the ascending `CSR`
//! order, pinned vertices are skipped, and out-of-range triangles are dropped
//! at build time rather than panicking.

use super::wind::{AeroParams, WindField};
use super::{physics_bridge, ClothParticle};

pub use prism_physics_core::soft::aero::VertexTriangleAdjacency;

/// Applies the aerodynamic force to `particles` by a race-free per-vertex
/// gather over `adjacency`, reading a single pre-pass velocity snapshot.
///
/// For each free vertex the gather sums, over the triangles incident to it (in
/// ascending `CSR` order), a third of each face's [`super::wind::triangle_wind_force`] — the
/// same force the scatter spreads — evaluated against the frozen snapshot, then
/// applies `sum * inverse_mass * dt` as a single velocity increment. Because
/// every face reads the snapshot and every velocity is written once, the pass
/// is an explicit (Jacobi) update with no data race and is bit-reproducible: a
/// `WESL` per-vertex kernel performing the same ascending gather matches it
/// exactly. A non-positive or non-finite `dt`, an empty particle set, and an
/// empty triangle set are all no-ops; pinned vertices are skipped; and triangle
/// indices outside `particles` were already dropped when the adjacency was
/// built, so nothing here can panic.
pub fn accumulate_aero_gather(
    particles: &mut [ClothParticle],
    triangles: &[[u32; 3]],
    adjacency: &VertexTriangleAdjacency,
    wind: &WindField,
    aero: AeroParams,
    dt: f32,
) {
    // Delegate to the single-source physics-engine Jacobi gather. The render
    // particles are projected into the `(positions, velocities, inverse_masses)`
    // columns the kernel consumes (pinned particles map to a zero inverse mass),
    // the wind and coefficients are forwarded verbatim, and the solved
    // velocities are written back. `prism_physics_core` freezes the same
    // pre-pass velocity snapshot, walks the same ascending `CSR` adjacency, and
    // reuses the shared per-face force, so the pass is bit-for-bit identical to
    // the former inline loop (including every empty/degenerate no-op) and stays
    // in lock-step with the WESL per-vertex twin.
    let (positions, mut velocities, inverse_masses) = physics_bridge::to_soa_full(particles);
    prism_physics_core::soft::aero::accumulate_aero_gather(
        &positions,
        &mut velocities,
        &inverse_masses,
        triangles,
        adjacency,
        &prism_physics_core::soft::aero::WindField::new(
            physics_bridge::to_glam(wind.velocity),
            wind.turbulence,
        ),
        prism_physics_core::soft::aero::AeroParams::new(aero.drag, aero.lift)
            .with_air_density(aero.air_density),
        dt,
    );
    physics_bridge::write_velocities_back(particles, &velocities);
}

#[cfg(test)]
mod tests {
    use super::super::Vec3;
    use super::*;

    /// Small tolerance for the `f32` invariant checks; comparisons are always
    /// magnitude-based, never bit equality.
    const EPS: f32 = 1.0e-5;

    /// Two triangles sharing the edge `1-2`, forming a unit quad in the XY
    /// plane so both faces have the `+Z` normal and area `0.5`.
    fn quad() -> ([ClothParticle; 4], [[u32; 3]; 2]) {
        let particles = [
            ClothParticle::new(Vec3::new(0.0, 0.0, 0.0), 1.0),
            ClothParticle::new(Vec3::new(1.0, 0.0, 0.0), 1.0),
            ClothParticle::new(Vec3::new(0.0, 1.0, 0.0), 1.0),
            ClothParticle::new(Vec3::new(1.0, 1.0, 0.0), 1.0),
        ];
        let triangles = [[0u32, 1, 2], [2u32, 1, 3]];
        (particles, triangles)
    }

    #[test]
    fn adjacency_is_csr_and_counts_incidences() {
        let (_, triangles) = quad();
        let adj = VertexTriangleAdjacency::build(4, &triangles);
        assert_eq!(adj.vertex_count(), 4);
        // Two triangles, three incidences each.
        assert_eq!(adj.incidence_count(), 6);
        // Vertices 1 and 2 are shared by both faces; 0 and 3 by one each.
        assert_eq!(adj.incident(0), &[0]);
        assert_eq!(adj.incident(1), &[0, 1]);
        assert_eq!(adj.incident(2), &[0, 1]);
        assert_eq!(adj.incident(3), &[1]);
    }

    #[test]
    fn adjacency_runs_are_ascending() {
        let (_, triangles) = quad();
        let adj = VertexTriangleAdjacency::build(4, &triangles);
        for v in 0..adj.vertex_count() {
            let run = adj.incident(v);
            for pair in run.windows(2) {
                assert!(pair[0] < pair[1]);
            }
        }
    }

    #[test]
    fn adjacency_drops_out_of_range_triangles() {
        // Second triangle indexes vertex 9, which does not exist.
        let triangles = [[0u32, 1, 2], [0u32, 1, 9]];
        let adj = VertexTriangleAdjacency::build(3, &triangles);
        // Only the first triangle survives: three incidences, one per vertex.
        assert_eq!(adj.incidence_count(), 3);
        assert_eq!(adj.incident(0), &[0]);
        assert_eq!(adj.incident(1), &[0]);
        assert_eq!(adj.incident(2), &[0]);
    }

    #[test]
    fn adjacency_empty_for_out_of_range_vertex() {
        let (_, triangles) = quad();
        let adj = VertexTriangleAdjacency::build(4, &triangles);
        assert!(adj.incident(4).is_empty());
        assert!(adj.incident(999).is_empty());
    }

    #[test]
    fn single_triangle_gather_matches_expected_third() {
        let mut particles = [
            ClothParticle::new(Vec3::new(0.0, 0.0, 0.0), 1.0),
            ClothParticle::new(Vec3::new(1.0, 0.0, 0.0), 1.0),
            ClothParticle::new(Vec3::new(0.0, 1.0, 0.0), 1.0),
        ];
        let triangles = [[0u32, 1, 2]];
        let adj = VertexTriangleAdjacency::build(3, &triangles);
        let wind = WindField::new(Vec3::new(0.0, 0.0, 2.0), 0.0);
        accumulate_aero_gather(
            &mut particles,
            &triangles,
            &adj,
            &wind,
            AeroParams::new(1.0, 0.0),
            1.0,
        );
        // Force (0,0,1) split three ways: each vertex gains z = 1/3.
        for particle in &particles {
            assert!(particle.velocity.x.abs() < EPS);
            assert!(particle.velocity.y.abs() < EPS);
            assert!((particle.velocity.z - 1.0 / 3.0).abs() < EPS);
        }
    }

    #[test]
    fn shared_vertices_gather_both_faces() {
        let (particles, triangles) = quad();
        let mut particles = particles;
        let adj = VertexTriangleAdjacency::build(4, &triangles);
        let wind = WindField::new(Vec3::new(0.0, 0.0, 2.0), 0.0);
        accumulate_aero_gather(
            &mut particles,
            &triangles,
            &adj,
            &wind,
            AeroParams::new(1.0, 0.0),
            1.0,
        );
        // Each face contributes z = 1/3 per incident vertex. Corners 0 and 3
        // touch one face (1/3); shared 1 and 2 touch both (2/3).
        assert!((particles[0].velocity.z - 1.0 / 3.0).abs() < EPS);
        assert!((particles[3].velocity.z - 1.0 / 3.0).abs() < EPS);
        assert!((particles[1].velocity.z - 2.0 / 3.0).abs() < EPS);
        assert!((particles[2].velocity.z - 2.0 / 3.0).abs() < EPS);
    }

    #[test]
    fn gather_matches_scatter_for_disjoint_triangles() {
        // No shared vertices: Jacobi gather and sequential scatter coincide.
        let mut gather_particles = [
            ClothParticle::new(Vec3::new(0.0, 0.0, 0.0), 1.0),
            ClothParticle::new(Vec3::new(1.0, 0.0, 0.0), 1.0),
            ClothParticle::new(Vec3::new(0.0, 1.0, 0.0), 1.0),
            ClothParticle::new(Vec3::new(5.0, 0.0, 0.0), 1.0),
            ClothParticle::new(Vec3::new(6.0, 0.0, 0.0), 1.0),
            ClothParticle::new(Vec3::new(5.0, 1.0, 0.0), 1.0),
        ];
        let mut scatter_particles = gather_particles;
        let triangles = [[0u32, 1, 2], [3u32, 4, 5]];
        let wind = WindField::new(Vec3::new(0.3, 0.1, 2.0), 0.4);
        let aero = AeroParams::new(1.2, 0.7);

        let adj = VertexTriangleAdjacency::build(6, &triangles);
        accumulate_aero_gather(&mut gather_particles, &triangles, &adj, &wind, aero, 0.5);
        super::super::wind::apply_aero_forces(&mut scatter_particles, &triangles, &wind, aero, 0.5);

        for (g, s) in gather_particles.iter().zip(scatter_particles.iter()) {
            assert!((g.velocity.x - s.velocity.x).abs() < EPS);
            assert!((g.velocity.y - s.velocity.y).abs() < EPS);
            assert!((g.velocity.z - s.velocity.z).abs() < EPS);
        }
    }

    #[test]
    fn gather_is_deterministic() {
        let (particles, triangles) = quad();
        let adj = VertexTriangleAdjacency::build(4, &triangles);
        let wind = WindField::new(Vec3::new(0.7, -0.2, 1.5), 0.6);
        let aero = AeroParams::new(0.9, 0.4);

        let mut a = particles;
        let mut b = particles;
        accumulate_aero_gather(&mut a, &triangles, &adj, &wind, aero, 0.25);
        accumulate_aero_gather(&mut b, &triangles, &adj, &wind, aero, 0.25);
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.velocity, pb.velocity);
        }
    }

    #[test]
    fn gather_skips_pinned_vertices() {
        let (particles, triangles) = quad();
        let mut particles = particles;
        particles[1] = ClothParticle::pinned(Vec3::new(1.0, 0.0, 0.0));
        let adj = VertexTriangleAdjacency::build(4, &triangles);
        let wind = WindField::new(Vec3::new(0.0, 0.0, 2.0), 0.0);
        accumulate_aero_gather(
            &mut particles,
            &triangles,
            &adj,
            &wind,
            AeroParams::new(1.0, 0.0),
            1.0,
        );
        // The pinned vertex never moves.
        assert_eq!(particles[1].velocity, Vec3::ZERO);
        // Free shared vertex 2 still gathers both faces.
        assert!((particles[2].velocity.z - 2.0 / 3.0).abs() < EPS);
    }

    #[test]
    fn gather_is_noop_for_bad_dt_and_empty_inputs() {
        let (particles, triangles) = quad();
        let adj = VertexTriangleAdjacency::build(4, &triangles);
        let wind = WindField::new(Vec3::new(0.0, 0.0, 2.0), 0.0);
        let aero = AeroParams::new(1.0, 0.0);

        let mut zero_dt = particles;
        accumulate_aero_gather(&mut zero_dt, &triangles, &adj, &wind, aero, 0.0);
        for p in &zero_dt {
            assert_eq!(p.velocity, Vec3::ZERO);
        }

        let mut nan_dt = particles;
        accumulate_aero_gather(&mut nan_dt, &triangles, &adj, &wind, aero, f32::NAN);
        for p in &nan_dt {
            assert_eq!(p.velocity, Vec3::ZERO);
        }

        let mut no_tris = particles;
        accumulate_aero_gather(&mut no_tris, &[], &adj, &wind, aero, 1.0);
        for p in &no_tris {
            assert_eq!(p.velocity, Vec3::ZERO);
        }
    }

    #[test]
    fn gather_velocities_stay_finite() {
        let (particles, triangles) = quad();
        let mut particles = particles;
        let adj = VertexTriangleAdjacency::build(4, &triangles);
        let wind = WindField::new(Vec3::new(2.0, -3.0, 4.0), 1.0);
        accumulate_aero_gather(
            &mut particles,
            &triangles,
            &adj,
            &wind,
            AeroParams::new(1.5, 1.5),
            0.5,
        );
        for p in &particles {
            assert!(p.velocity.x.is_finite());
            assert!(p.velocity.y.is_finite());
            assert!(p.velocity.z.is_finite());
        }
    }
}
