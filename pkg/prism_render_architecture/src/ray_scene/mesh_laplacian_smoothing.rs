//! Laplacian and Taubin (λ|μ) mesh fairing for the `CPU` golden path.
//!
//! Scanned, voxelized, or coarsely authored meshes often carry high-frequency
//! noise that normal mapping cannot hide — faceting on what should be a smooth
//! shell. The umbrella (uniform Laplacian) operator removes it by nudging every
//! vertex toward the centroid of its one-ring neighbours:
//! `Δp_i = (1/|N_i|) · Σ_{j∈N_i} (p_j − p_i)`. A single step
//! `p_i ← p_i + λ · Δp_i` with `0 < λ < 1` is a low-pass filter on the surface.
//!
//! ## Shrinkage and the Taubin fix
//!
//! Pure Laplacian smoothing is a diffusion: iterate it enough and the mesh
//! collapses toward its centroid. Taubin's λ|μ scheme (SIGGRAPH '95) cancels
//! that drift by following every shrinking pass (`λ > 0`) with an inflating
//! pass using a slightly larger negative factor (`μ < −λ < 0`). The pair acts
//! as a band-stop filter: low frequencies pass through essentially unchanged
//! while high-frequency noise is attenuated, so the volume is preserved across
//! iterations.
//!
//! ## Boundaries
//!
//! Open borders need care, or fairing would erode the silhouette. Boundary
//! vertices (endpoints of an edge used by a single face) are either pinned in
//! place or smoothed with a *restricted* one-ring containing only their
//! boundary neighbours — a one-dimensional curve filter that keeps the border
//! on its own curve instead of pulling it into the interior.
//!
//! ## Scope
//!
//! Only positions move; connectivity, `UV`s, and the stored normals are left
//! untouched because topology is unchanged. Since smoothing invalidates
//! shading normals, recompute them afterward (see
//! [`super::mesh_smooth_normals::with_smooth_normals`]). Every operation is a
//! weighted average — purely linear, no transcendental calls — so the module
//! stays within the golden-path float policy.

use std::collections::HashMap;
use std::collections::HashSet;

use super::triangle_mesh::{TriangleMesh, TriangleMeshError};

/// How boundary (open-edge) vertices are treated during fairing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryRule {
    /// Boundary vertices never move, exactly preserving the border polyline.
    Pin,
    /// Boundary vertices are smoothed using only their boundary neighbours, a
    /// one-dimensional curve filter that keeps the border on its own curve.
    CurveSmooth,
}

/// A configured Laplacian or Taubin fairing operator.
///
/// Build one with [`LaplacianSmoothing::laplacian`] (single-factor low-pass,
/// which shrinks) or [`LaplacianSmoothing::taubin`] (λ|μ band-stop, which
/// preserves volume), then apply it with [`LaplacianSmoothing::smooth`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LaplacianSmoothing {
    /// Number of full iterations. A Taubin iteration is one λ pass plus one μ
    /// pass; a Laplacian iteration is a single λ pass.
    iterations: u32,
    /// Positive shrinking factor applied to the umbrella delta each λ pass.
    lambda: f32,
    /// Negative inflating factor for the μ pass; `None` selects plain
    /// Laplacian smoothing with no de-shrinking pass.
    mu: Option<f32>,
    /// How open-boundary vertices are handled.
    boundary: BoundaryRule,
}

impl LaplacianSmoothing {
    /// Builds a plain Laplacian low-pass operator running `iterations` single
    /// λ passes. `lambda` is clamped to `[0, 1]`. This filter shrinks the mesh;
    /// prefer [`Self::taubin`] when volume matters. Boundaries are pinned by
    /// default; override with [`Self::with_boundary`].
    #[must_use]
    pub fn laplacian(iterations: u32, lambda: f32) -> Self {
        Self {
            iterations,
            lambda: lambda.clamp(0.0, 1.0),
            mu: None,
            boundary: BoundaryRule::Pin,
        }
    }

    /// Builds a Taubin λ|μ band-stop operator. Each of `iterations` iterations
    /// runs a `lambda` shrinking pass then a `mu` inflating pass. `lambda` is
    /// clamped to `[0, 1]` and `mu` to `[-1, 0]`; for volume preservation keep
    /// `-mu` slightly larger than `lambda` (e.g. `0.33` / `-0.34`). Boundaries
    /// are pinned by default; override with [`Self::with_boundary`].
    #[must_use]
    pub fn taubin(iterations: u32, lambda: f32, mu: f32) -> Self {
        Self {
            iterations,
            lambda: lambda.clamp(0.0, 1.0),
            mu: Some(mu.clamp(-1.0, 0.0)),
            boundary: BoundaryRule::Pin,
        }
    }

    /// Returns the operator with its boundary handling replaced by `rule`.
    #[must_use]
    pub fn with_boundary(mut self, rule: BoundaryRule) -> Self {
        self.boundary = rule;
        self
    }

    /// Fairs `mesh`, returning a new mesh with smoothed positions and the
    /// original connectivity, `UV`s, and normals.
    ///
    /// # Errors
    ///
    /// Propagates [`TriangleMeshError`] from rebuilding the result; by
    /// construction the pools stay valid, so this does not fail in practice.
    pub fn smooth(&self, mesh: &TriangleMesh) -> Result<TriangleMesh, TriangleMeshError> {
        let vertex_count = mesh.vertex_count();
        if vertex_count == 0 || self.iterations == 0 {
            return Ok(mesh.clone());
        }

        let topology = Topology::build(mesh);
        let mut positions: Vec<[f32; 3]> = mesh.positions().to_vec();

        for _ in 0..self.iterations {
            self.apply_pass(&mut positions, &topology, self.lambda);
            if let Some(mu) = self.mu {
                self.apply_pass(&mut positions, &topology, mu);
            }
        }

        TriangleMesh::new(
            positions,
            mesh.normals().to_vec(),
            mesh.uvs().to_vec(),
            mesh.indices().to_vec(),
        )
    }

    /// Runs one umbrella pass with the signed `factor`, writing displaced
    /// positions back into `positions` from a snapshot so neighbours are read
    /// consistently.
    fn apply_pass(&self, positions: &mut [[f32; 3]], topology: &Topology, factor: f32) {
        let snapshot = positions.to_vec();
        for v in 0..positions.len() {
            let is_boundary = topology.is_boundary[v];
            if is_boundary && self.boundary == BoundaryRule::Pin {
                continue;
            }
            let neighbours = if is_boundary {
                &topology.boundary_neighbours[v]
            } else {
                &topology.neighbours[v]
            };
            if neighbours.is_empty() {
                continue;
            }
            let p = snapshot[v];
            let mut sum = [0.0f32; 3];
            for &n in neighbours {
                let q = snapshot[n as usize];
                sum[0] += q[0] - p[0];
                sum[1] += q[1] - p[1];
                sum[2] += q[2] - p[2];
            }
            let inv = factor / neighbours.len() as f32;
            positions[v] = [
                p[0] + sum[0] * inv,
                p[1] + sum[1] * inv,
                p[2] + sum[2] * inv,
            ];
        }
    }
}

/// One-ring adjacency plus boundary classification for a mesh.
struct Topology {
    /// For each vertex, its full set of one-ring neighbour vertex ids.
    neighbours: Vec<Vec<u32>>,
    /// For each vertex, the subset of neighbours reached across boundary edges
    /// (empty for interior vertices).
    boundary_neighbours: Vec<Vec<u32>>,
    /// `true` when the vertex lies on an open boundary edge.
    is_boundary: Vec<bool>,
}

impl Topology {
    /// Builds adjacency and boundary flags from a mesh's index buffer.
    fn build(mesh: &TriangleMesh) -> Self {
        let vertex_count = mesh.vertex_count();
        let mut neighbour_sets: Vec<HashSet<u32>> = vec![HashSet::new(); vertex_count];
        // Count how many faces use each undirected edge to find open borders.
        let mut edge_count: HashMap<(u32, u32), u32> = HashMap::new();

        for tri in mesh.indices() {
            let [a, b, c] = *tri;
            if a == b || b == c || a == c {
                continue;
            }
            for &(u, v) in &[(a, b), (b, c), (c, a)] {
                neighbour_sets[u as usize].insert(v);
                neighbour_sets[v as usize].insert(u);
                let key = if u < v { (u, v) } else { (v, u) };
                *edge_count.entry(key).or_insert(0) += 1;
            }
        }

        let mut is_boundary = vec![false; vertex_count];
        let mut boundary_pairs: Vec<HashSet<u32>> = vec![HashSet::new(); vertex_count];
        for (&(u, v), &count) in &edge_count {
            if count == 1 {
                is_boundary[u as usize] = true;
                is_boundary[v as usize] = true;
                boundary_pairs[u as usize].insert(v);
                boundary_pairs[v as usize].insert(u);
            }
        }

        let neighbours = neighbour_sets
            .into_iter()
            .map(|set| {
                let mut v: Vec<u32> = set.into_iter().collect();
                v.sort_unstable();
                v
            })
            .collect();
        let boundary_neighbours = boundary_pairs
            .into_iter()
            .map(|set| {
                let mut v: Vec<u32> = set.into_iter().collect();
                v.sort_unstable();
                v
            })
            .collect();

        Self {
            neighbours,
            boundary_neighbours,
            is_boundary,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a flat `grid × grid` quad lattice on `z = 0`, two triangles per
    /// cell, then displaces each interior vertex by `bump` along `+Z` to create
    /// high-frequency noise for the filter to remove.
    fn noisy_grid(grid: usize, bump: f32) -> TriangleMesh {
        let stride = grid + 1;
        let mut positions = Vec::new();
        for j in 0..=grid {
            for i in 0..=grid {
                let interior = i > 0 && i < grid && j > 0 && j < grid;
                // Alternate the bump sign in a checkerboard for pure high freq.
                let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                let z = if interior { bump * sign } else { 0.0 };
                positions.push([i as f32, j as f32, z]);
            }
        }
        let mut indices = Vec::new();
        for j in 0..grid as u32 {
            for i in 0..grid as u32 {
                let a = j * stride as u32 + i;
                let b = a + 1;
                let c = a + stride as u32;
                let d = c + 1;
                indices.push([a, b, c]);
                indices.push([b, d, c]);
            }
        }
        TriangleMesh::new(positions, Vec::new(), Vec::new(), indices).unwrap()
    }

    /// Peak absolute `z` deviation from the flat plane over all vertices.
    fn peak_z(mesh: &TriangleMesh) -> f32 {
        mesh.positions()
            .iter()
            .map(|p| p[2].abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn zero_iterations_is_identity() {
        let mesh = noisy_grid(4, 0.3);
        let out = LaplacianSmoothing::laplacian(0, 0.5).smooth(&mesh).unwrap();
        assert_eq!(out.positions(), mesh.positions());
    }

    #[test]
    fn laplacian_reduces_high_frequency_noise() {
        let mesh = noisy_grid(6, 0.4);
        let before = peak_z(&mesh);
        let out = LaplacianSmoothing::laplacian(10, 0.5).smooth(&mesh).unwrap();
        let after = peak_z(&out);
        assert!(after < before * 0.25, "before {before}, after {after}");
    }

    #[test]
    fn pinned_boundary_does_not_move() {
        let grid = 5;
        let mesh = noisy_grid(grid, 0.3);
        let out = LaplacianSmoothing::laplacian(8, 0.5)
            .with_boundary(BoundaryRule::Pin)
            .smooth(&mesh)
            .unwrap();
        let stride = grid + 1;
        for j in 0..=grid {
            for i in 0..=grid {
                let on_border = i == 0 || i == grid || j == 0 || j == grid;
                if on_border {
                    let idx = j * stride + i;
                    let a = mesh.positions()[idx];
                    let b = out.positions()[idx];
                    assert!((a[0] - b[0]).abs() < 1.0e-6);
                    assert!((a[1] - b[1]).abs() < 1.0e-6);
                    assert!((a[2] - b[2]).abs() < 1.0e-6);
                }
            }
        }
    }

    #[test]
    fn taubin_preserves_volume_better_than_laplacian() {
        // A hemispherical-ish bump: shrinking should pull its apex down; Taubin
        // should keep it far closer to the original height than pure Laplacian.
        let grid = 10;
        let stride = grid + 1;
        let mut positions = Vec::new();
        let center = grid as f32 / 2.0;
        for j in 0..=grid {
            for i in 0..=grid {
                let dx = i as f32 - center;
                let dy = j as f32 - center;
                let r2 = dx * dx + dy * dy;
                let z = (center * center - r2).max(0.0) * 0.05;
                positions.push([i as f32, j as f32, z]);
            }
        }
        let mut indices = Vec::new();
        for j in 0..grid as u32 {
            for i in 0..grid as u32 {
                let a = j * stride as u32 + i;
                let b = a + 1;
                let c = a + stride as u32;
                let d = c + 1;
                indices.push([a, b, c]);
                indices.push([b, d, c]);
            }
        }
        let mesh = TriangleMesh::new(positions, Vec::new(), Vec::new(), indices).unwrap();

        let apex = (grid / 2) * stride + (grid / 2);
        let original_apex = mesh.positions()[apex][2];

        let lap = LaplacianSmoothing::laplacian(40, 0.5).smooth(&mesh).unwrap();
        let tau = LaplacianSmoothing::taubin(40, 0.5, -0.53)
            .smooth(&mesh)
            .unwrap();

        let lap_drop = original_apex - lap.positions()[apex][2];
        let tau_drop = original_apex - tau.positions()[apex][2];
        assert!(lap_drop > 0.0, "laplacian should shrink the apex");
        assert!(
            tau_drop < lap_drop * 0.6,
            "taubin drop {tau_drop} vs laplacian {lap_drop}"
        );
    }

    #[test]
    fn attributes_and_connectivity_survive() {
        let grid = 4;
        let base = noisy_grid(grid, 0.2);
        let normals = vec![[0.0, 0.0, 1.0]; base.vertex_count()];
        let uvs: Vec<[f32; 2]> = base
            .positions()
            .iter()
            .map(|p| [p[0], p[1]])
            .collect();
        let mesh = TriangleMesh::new(
            base.positions().to_vec(),
            normals.clone(),
            uvs.clone(),
            base.indices().to_vec(),
        )
        .unwrap();
        let out = LaplacianSmoothing::taubin(5, 0.33, -0.34)
            .smooth(&mesh)
            .unwrap();
        assert_eq!(out.indices(), mesh.indices());
        assert_eq!(out.uvs(), uvs.as_slice());
        assert_eq!(out.normals(), normals.as_slice());
        assert_eq!(out.vertex_count(), mesh.vertex_count());
    }

    #[test]
    fn curve_smoothed_boundary_stays_on_border_line() {
        // On a flat (z = 0) grid the border is already straight, so curve
        // smoothing must leave boundary vertices on z = 0 and within the
        // original xy extent.
        let grid = 6;
        let mesh = noisy_grid(grid, 0.0);
        let out = LaplacianSmoothing::laplacian(6, 0.5)
            .with_boundary(BoundaryRule::CurveSmooth)
            .smooth(&mesh)
            .unwrap();
        for p in out.positions() {
            assert!(p[2].abs() < 1.0e-5);
            assert!(p[0] >= -1.0e-5 && p[0] <= grid as f32 + 1.0e-5);
            assert!(p[1] >= -1.0e-5 && p[1] <= grid as f32 + 1.0e-5);
        }
    }

    #[test]
    fn flat_mesh_stays_planar() {
        // A perfectly flat grid stays in the z = 0 plane under fairing: all
        // neighbours share z = 0, so the umbrella delta has no z component.
        // (Tangential xy drift is expected on an irregular triangulation and
        // is not asserted here.)
        let grid = 5;
        let mesh = noisy_grid(grid, 0.0);
        let out = LaplacianSmoothing::taubin(20, 0.33, -0.34)
            .with_boundary(BoundaryRule::CurveSmooth)
            .smooth(&mesh)
            .unwrap();
        for p in out.positions() {
            assert!(p[2].abs() < 1.0e-5, "z drifted to {}", p[2]);
        }
    }

    #[test]
    fn empty_mesh_round_trips() {
        let mesh = TriangleMesh::new(Vec::new(), Vec::new(), Vec::new(), Vec::new()).unwrap();
        let out = LaplacianSmoothing::laplacian(5, 0.5).smooth(&mesh).unwrap();
        assert_eq!(out.vertex_count(), 0);
        assert_eq!(out.triangle_count(), 0);
    }
}
