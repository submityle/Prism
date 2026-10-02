//! Race-free (Jacobi) per-vertex aero gather — the GPU-faithful twin of the
//! sequential scatter in [`super::apply::apply_aero_forces`].
//!
//! [`apply_aero_forces`](super::apply::apply_aero_forces) walks the mesh in
//! *triangle* order and spreads each face force onto its three vertices,
//! reading each vertex's already-updated velocity (a Gauss-Seidel coupling). A
//! parallel GPU kernel cannot do that without a data race, so the device path
//! instead has every vertex *gather*: it sums, over the triangles incident to
//! it, a third of each incident face's force evaluated against a single frozen
//! pre-pass velocity snapshot, then writes its velocity exactly once. That is
//! an explicit (Jacobi) update — deterministic, race-free, and bit-reproducible
//! — and it agrees with the scatter exactly when no vertex is shared.
//!
//! This module is the single source for that Jacobi gather so the render
//! engine, its WESL per-vertex kernel, and any solver path all share one
//! implementation:
//!
//! * [`VertexTriangleAdjacency`] — the `CSR` vertex→triangle incidence the
//!   gather walks, built once per topology.
//! * [`accumulate_aero_gather`] — the per-vertex velocity pre-pass itself,
//!   reusing the shared [`triangle_aero_force`] and [`turbulence_offset`]
//!   kernels so the per-face force is identical to the scatter's.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The `CSR`
//! adjacency is a textbook counting-sort build and the gather is the standard
//! Jacobi reformulation of the per-triangle aero scatter.

use alloc::vec::Vec;

use glam::Vec3;

use crate::math::scalar::Real;

use super::field::{AeroParams, WindField};
use super::triangle::{triangle_aero_force, turbulence_offset};

/// A `CSR` vertex→triangle adjacency: for each vertex, the ascending slice of
/// the triangle indices incident to it.
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
    /// [`apply_aero_forces`](super::apply::apply_aero_forces) skips. The build
    /// is a deterministic counting sort: a first pass tallies the
    /// incident-triangle count per vertex, a prefix sum turns those into
    /// `offsets`, and a second pass fills `entries` in triangle order, so each
    /// vertex's run is ascending. The pass is
    /// `O(vertex_count + triangles.len())` with no hidden quadratic scan.
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

    /// The raw `CSR` start offsets, one per vertex plus a trailing total.
    ///
    /// Length is `vertex_count + 1` and the values are non-decreasing, so the
    /// triangles incident to vertex `v` occupy `entries()[offsets()[v] ..
    /// offsets()[v + 1]]`. Exposed so a `GPU` per-vertex gather kernel can
    /// upload the same flattened adjacency the `CPU` gather walks, keeping the
    /// device pass byte-for-byte faithful to the golden reference.
    #[must_use]
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// The flattened, per-vertex-contiguous triangle indices (ascending within
    /// each vertex's run) the `offsets` slice windows into.
    #[must_use]
    pub fn entries(&self) -> &[u32] {
        &self.entries
    }
}

/// Applies the aerodynamic force by a race-free per-vertex gather over
/// `adjacency`, reading a single pre-pass velocity snapshot.
///
/// `positions`, `velocities`, and `inverse_masses` are index-aligned particle
/// columns; `velocities` is mutated in place. For each free vertex the gather
/// sums, over the triangles incident to it (in ascending `CSR` order), a third
/// of each face's [`triangle_aero_force`] — the same force the scatter spreads
/// — evaluated against the frozen snapshot, then applies
/// `sum * inverse_mass * dt` as a single velocity increment. Because every face
/// reads the snapshot and every velocity is written once, the pass is an
/// explicit (Jacobi) update with no data race and is bit-reproducible: a `WESL`
/// per-vertex kernel performing the same ascending gather matches it exactly.
///
/// A non-positive or non-finite `dt`, an empty particle set, an empty triangle
/// set, or mismatched column lengths are all no-ops; pinned vertices (inverse
/// mass `0`) are skipped; and triangle indices outside the particle set were
/// already dropped when the adjacency was built, so nothing here can panic.
pub fn accumulate_aero_gather(
    positions: &[Vec3],
    velocities: &mut [Vec3],
    inverse_masses: &[Real],
    triangles: &[[u32; 3]],
    adjacency: &VertexTriangleAdjacency,
    wind: &WindField,
    aero: AeroParams,
    dt: Real,
) {
    let count = positions.len();
    if dt <= 0.0
        || !dt.is_finite()
        || count == 0
        || triangles.is_empty()
        || velocities.len() != count
        || inverse_masses.len() != count
    {
        return;
    }
    let field = wind.sanitized();
    let aero = aero.sanitized();

    // Frozen pre-pass velocities so every face sees the same input state.
    let snapshot: Vec<Vec3> = velocities.to_vec();

    let vertices = adjacency.vertex_count().min(count);
    for v in 0..vertices {
        let inverse_mass = inverse_masses[v];
        if inverse_mass <= 0.0 {
            continue;
        }

        let mut accum = Vec3::ZERO;
        for &t in adjacency.incident(v) {
            let Some(tri) = triangles.get(t as usize) else {
                continue;
            };
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            if i0 >= count || i1 >= count || i2 >= count {
                continue;
            }

            let wind_vec = field.velocity + turbulence_offset(*tri, field.turbulence);
            let force = triangle_aero_force(
                positions[i0],
                positions[i1],
                positions[i2],
                snapshot[i0],
                snapshot[i1],
                snapshot[i2],
                wind_vec,
                aero,
            );
            accum += force * (1.0 / 3.0);
        }

        velocities[v] += accum * (inverse_mass * dt);
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
    fn quad() -> ([Vec3; 4], [Real; 4], [[u32; 3]; 2]) {
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        let inv = [1.0; 4];
        let triangles = [[0u32, 1, 2], [2u32, 1, 3]];
        (positions, inv, triangles)
    }

    #[test]
    fn adjacency_is_csr_and_counts_incidences() {
        let (_, _, triangles) = quad();
        let adj = VertexTriangleAdjacency::build(4, &triangles);
        assert_eq!(adj.vertex_count(), 4);
        assert_eq!(adj.incidence_count(), 6);
        assert_eq!(adj.incident(0), &[0]);
        assert_eq!(adj.incident(1), &[0, 1]);
        assert_eq!(adj.incident(2), &[0, 1]);
        assert_eq!(adj.incident(3), &[1]);
    }

    #[test]
    fn adjacency_runs_are_ascending() {
        let (_, _, triangles) = quad();
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
        let triangles = [[0u32, 1, 2], [0u32, 1, 9]];
        let adj = VertexTriangleAdjacency::build(3, &triangles);
        assert_eq!(adj.incidence_count(), 3);
        assert_eq!(adj.incident(0), &[0]);
        assert_eq!(adj.incident(1), &[0]);
        assert_eq!(adj.incident(2), &[0]);
    }

    #[test]
    fn adjacency_empty_for_out_of_range_vertex() {
        let (_, _, triangles) = quad();
        let adj = VertexTriangleAdjacency::build(4, &triangles);
        assert!(adj.incident(4).is_empty());
        assert!(adj.incident(999).is_empty());
    }

    #[test]
    fn single_triangle_gather_pushes_free_vertices_downwind() {
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let mut velocities = [Vec3::ZERO; 3];
        let inv = [1.0, 1.0, 1.0];
        let triangles = [[0u32, 1, 2]];
        let adj = VertexTriangleAdjacency::build(3, &triangles);
        accumulate_aero_gather(
            &positions,
            &mut velocities,
            &inv,
            &triangles,
            &adj,
            &WindField::new(Vec3::new(0.0, 0.0, 2.0), 0.0),
            AeroParams::new(1.0, 0.0),
            1.0,
        );
        for v in &velocities {
            assert!(v.z > 0.0, "velocity {v:?}");
            assert!(v.x.abs() < EPS && v.y.abs() < EPS);
        }
        assert!((velocities[0] - velocities[1]).length() < EPS);
        assert!((velocities[1] - velocities[2]).length() < EPS);
    }

    #[test]
    fn gather_matches_scatter_when_no_vertex_is_shared() {
        // Two disjoint triangles (no shared vertex): the Jacobi gather and the
        // Gauss-Seidel scatter must agree exactly.
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(5.0, 1.0, 0.0),
        ];
        let inv = [1.0; 6];
        let triangles = [[0u32, 1, 2], [3u32, 4, 5]];
        let wind = WindField::new(Vec3::new(0.3, 0.0, 2.0), 0.0);
        let aero = AeroParams::new(1.0, 0.5);
        let dt = 1.0 / 60.0;

        let mut scatter = [Vec3::ZERO; 6];
        super::super::apply::apply_aero_forces(
            &positions, &mut scatter, &inv, &triangles, &wind, aero, dt,
        );

        let adj = VertexTriangleAdjacency::build(6, &triangles);
        let mut gather = [Vec3::ZERO; 6];
        accumulate_aero_gather(&positions, &mut gather, &inv, &triangles, &adj, &wind, aero, dt);

        for (a, b) in scatter.iter().zip(gather.iter()) {
            assert!((*a - *b).length() < EPS, "scatter {a:?} gather {b:?}");
        }
    }

    #[test]
    fn pinned_vertex_is_not_pushed() {
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let mut velocities = [Vec3::ZERO; 3];
        let inv = [0.0, 1.0, 1.0];
        let triangles = [[0u32, 1, 2]];
        let adj = VertexTriangleAdjacency::build(3, &triangles);
        accumulate_aero_gather(
            &positions,
            &mut velocities,
            &inv,
            &triangles,
            &adj,
            &WindField::new(Vec3::new(0.0, 0.0, 2.0), 0.0),
            AeroParams::new(1.0, 0.0),
            1.0,
        );
        assert!(velocities[0].length() < EPS, "pinned moved: {:?}", velocities[0]);
        assert!(velocities[1].z > 0.0);
    }

    #[test]
    fn non_positive_dt_is_a_noop() {
        let (positions, inv, triangles) = quad();
        let adj = VertexTriangleAdjacency::build(4, &triangles);
        let mut velocities = [Vec3::ZERO; 4];
        accumulate_aero_gather(
            &positions,
            &mut velocities,
            &inv,
            &triangles,
            &adj,
            &WindField::new(Vec3::new(0.0, 0.0, 2.0), 0.0),
            AeroParams::new(1.0, 0.0),
            0.0,
        );
        for v in &velocities {
            assert!(v.length() < EPS);
        }
    }
}
