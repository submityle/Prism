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

use alloc::vec::Vec;

use super::wind::{triangle_wind_force, turbulence_offset, AeroParams, WindField};
use super::{ClothParticle, Vec3};

/// A `CSR` vertex→triangle adjacency: for each vertex, the ascending slice
/// indices of the triangles incident to it.
///
/// `offsets` has `vertex_count + 1` entries; the triangles incident to vertex
/// `v` are `entries[offsets[v] .. offsets[v + 1]]`, and each stored value is an
/// index into the triangle slice the adjacency was built from. Storing the
/// triangle *index* (rather than the face's three vertex ids) keeps the gather
/// able to recompute the exact same per-face force the scatter would, including
/// the index-derived turbulence jitter.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VertexTriangleAdjacency {
    /// Per-vertex start offsets into `entries`; length is `vertex_count + 1`
    /// and the values are non-decreasing.
    offsets: Vec<u32>,
    /// Flattened, per-vertex-contiguous triangle indices, ascending within each
    /// vertex's run.
    entries: Vec<u32>,
}

impl VertexTriangleAdjacency {
    /// Builds the adjacency for `vertex_count` vertices from `triangles`.
    ///
    /// Only triangles whose three indices are all in `0..vertex_count` are
    /// recorded, so the gather skips exactly the faces
    /// [`super::wind::apply_aero_forces`] skips. The build is a deterministic
    /// counting sort: a first pass tallies the incident-triangle count per
    /// vertex, a prefix sum turns those into `offsets`, and a second pass fills
    /// `entries` in triangle order, so each vertex's run is ascending. The pass
    /// is `O(vertex_count + triangles.len())` with no hidden quadratic scan.
    #[must_use]
    pub fn build(vertex_count: usize, triangles: &[[u32; 3]]) -> Self {
        let mut counts = alloc::vec![0u32; vertex_count + 1];

        // First pass: count incident triangles per vertex (in-range faces only).
        for tri in triangles {
            if !Self::in_range(*tri, vertex_count) {
                continue;
            }
            for &vi in tri {
                counts[vi as usize] = counts[vi as usize].saturating_add(1);
            }
        }

        // Prefix sum -> offsets. `offsets[v]` is where vertex `v`'s run starts.
        let mut offsets = alloc::vec![0u32; vertex_count + 1];
        let mut running = 0u32;
        for v in 0..vertex_count {
            offsets[v] = running;
            running = running.saturating_add(counts[v]);
        }
        offsets[vertex_count] = running;

        // Second pass: place each triangle index into its vertices' runs. A
        // per-vertex write cursor advances through the run; iterating triangles
        // in order keeps every run ascending.
        let mut cursor = offsets.clone();
        let mut entries = alloc::vec![0u32; running as usize];
        for (t, tri) in triangles.iter().enumerate() {
            if !Self::in_range(*tri, vertex_count) {
                continue;
            }
            for &vi in tri {
                let slot = cursor[vi as usize] as usize;
                entries[slot] = t as u32;
                cursor[vi as usize] += 1;
            }
        }

        Self { offsets, entries }
    }

    /// Returns `true` when every index of `tri` is a valid vertex id.
    fn in_range(tri: [u32; 3], vertex_count: usize) -> bool {
        (tri[0] as usize) < vertex_count
            && (tri[1] as usize) < vertex_count
            && (tri[2] as usize) < vertex_count
    }

    /// Number of vertices the adjacency was built for.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Total number of (vertex, triangle) incidences stored; equals three times
    /// the number of in-range triangles.
    #[must_use]
    pub fn incidence_count(&self) -> usize {
        self.entries.len()
    }

    /// The ascending triangle indices incident to `vertex`, or an empty slice
    /// when `vertex` is out of range or touches no in-range triangle.
    #[must_use]
    pub fn incident(&self, vertex: usize) -> &[u32] {
        if vertex + 1 >= self.offsets.len() {
            return &[];
        }
        let start = self.offsets[vertex] as usize;
        let end = self.offsets[vertex + 1] as usize;
        &self.entries[start..end]
    }
}

/// Applies the aerodynamic force to `particles` by a race-free per-vertex
/// gather over `adjacency`, reading a single pre-pass velocity snapshot.
///
/// For each free vertex the gather sums, over the triangles incident to it (in
/// ascending `CSR` order), a third of each face's [`triangle_wind_force`] — the
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
    if dt <= 0.0 || !dt.is_finite() || particles.is_empty() || triangles.is_empty() {
        return;
    }
    let field = wind.sanitized();
    let aero = aero.sanitized();
    let count = particles.len();

    // Frozen pre-pass velocities so every face sees the same input state.
    let snapshot: Vec<Vec3> = particles.iter().map(|p| p.velocity).collect();

    let vertices = adjacency.vertex_count().min(count);
    for v in 0..vertices {
        let particle = &particles[v];
        if particle.is_pinned() {
            continue;
        }
        let inverse_mass = particle.inverse_mass;

        let mut accum = Vec3::ZERO;
        for &t in adjacency.incident(v) {
            let Some(tri) = triangles.get(t as usize) else {
                continue;
            };
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            if i0 >= count || i1 >= count || i2 >= count {
                continue;
            }

            let wind_vec = field
                .velocity
                .add(turbulence_offset(*tri, field.turbulence));
            let force = triangle_wind_force(
                particles[i0].position,
                particles[i1].position,
                particles[i2].position,
                snapshot[i0],
                snapshot[i1],
                snapshot[i2],
                wind_vec,
                aero,
            );
            accum = accum.add(force.scale(1.0 / 3.0));
        }

        let delta = accum.scale(inverse_mass * dt);
        particles[v].velocity = particles[v].velocity.add(delta);
    }
}

#[cfg(test)]
mod tests {
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
