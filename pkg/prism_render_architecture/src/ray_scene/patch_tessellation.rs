//! Hardware-tessellator-style, crack-free triangulation of a parametric patch.
//!
//! A GPU tessellator subdivides a quad domain `(u, v) ∈ [0, 1]²` using **four
//! independent outer tessellation factors** — one per domain edge — plus an
//! **inner factor** controlling the interior grid density. The outer factors
//! are the mechanism that keeps *adjacent* patches crack-free: because each
//! edge is split into exactly `outer` segments at the shared parameters
//! `k / outer`, two patches that agree on the shared edge's outer factor emit
//! bit-identical boundary vertices and leave no T-junctions between them.
//!
//! [`PatchTessellation`] reproduces that model on the `CPU` for any
//! [`displaced_surface::ParametricSurface`](super::displaced_surface::ParametricSurface):
//!
//! - an **interior block** — the uniform `inner × inner` grid restricted to its
//!   strictly interior nodes — is triangulated directly;
//! - each of the four boundary edges is sampled at its own outer factor and
//!   **zipped** to the matching side of the interior block with the classic
//!   two-pointer polyline stitch, so a dense edge meets a sparse interior (or
//!   vice-versa) without cracks;
//! - the four interior-block corner nodes are *shared* between the two edge
//!   strips that meet there, so the whole patch is watertight (every interior
//!   edge is referenced by exactly two triangles) and every boundary edge is
//!   referenced by exactly one.
//!
//! All arithmetic is division, comparison, and multiply — no transcendental
//! calls — so the subdivision matches the integer-spacing `GPU` tessellator and
//! stays within the project's `clippy` float policy. The output is a
//! [`TriangleMesh`] (positions, re-sampled analytic normals, and the `(u, v)`
//! domain coordinates as texture coordinates) plus an optional
//! [`TriangleMeshBvh`] for ray queries.

use super::triangle_mesh::{TriangleMesh, TriangleMeshBvh, TriangleMeshError};
use crate::ray_scene::displaced_surface::ParametricSurface;

/// Largest tessellation factor accepted on any edge or the interior.
///
/// Mirrors the common `GPU` cap of 64 segments per edge; factors are clamped
/// into `1..=MAX_FACTOR` (the interior is additionally floored at `2`) so a
/// caller can never request an unbounded vertex pool.
pub const MAX_FACTOR: u32 = 64;

/// A crack-free quad-patch tessellation recipe: one inner factor and four
/// per-edge outer factors.
///
/// Construct with [`PatchTessellation::uniform`] or [`PatchTessellation::new`],
/// then call [`PatchTessellation::tessellate`] (or
/// [`PatchTessellation::tessellate_bvh`]) against any parametric surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatchTessellation {
    /// Interior grid density in both parameter directions, clamped to
    /// `2..=MAX_FACTOR`. The strictly interior nodes form the directly
    /// triangulated block; the outermost ring of grid cells is replaced by the
    /// four edge strips.
    inner: u32,
    /// Outer segment counts, each clamped to `1..=MAX_FACTOR`, indexed by
    /// [`Edge`]: `[bottom (v=0), right (u=1), top (v=1), left (u=0)]`.
    outer: [u32; 4],
}

/// The four domain edges of the unit `(u, v)` square, used to index
/// [`PatchTessellation::outer`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge {
    /// The `v = 0` edge, parameterised by increasing `u`.
    Bottom,
    /// The `u = 1` edge, parameterised by increasing `v`.
    Right,
    /// The `v = 1` edge, parameterised by increasing `u`.
    Top,
    /// The `u = 0` edge, parameterised by increasing `v`.
    Left,
}

impl Edge {
    /// Returns the index this edge occupies in [`PatchTessellation::outer`].
    const fn index(self) -> usize {
        match self {
            Edge::Bottom => 0,
            Edge::Right => 1,
            Edge::Top => 2,
            Edge::Left => 3,
        }
    }
}

/// Clamps a requested tessellation factor into `1..=MAX_FACTOR`.
fn clamp_factor(factor: u32) -> u32 {
    factor.clamp(1, MAX_FACTOR)
}

impl PatchTessellation {
    /// Builds a recipe with the same `factor` on every edge and the interior.
    ///
    /// The interior is floored at `2` (a one-cell interior block) and all
    /// factors are capped at [`MAX_FACTOR`].
    pub fn uniform(factor: u32) -> Self {
        let outer = clamp_factor(factor);
        Self {
            inner: outer.max(2),
            outer: [outer; 4],
        }
    }

    /// Builds a recipe from an explicit inner factor and four outer factors.
    ///
    /// `outer` is ordered `[bottom, right, top, left]` (see [`Edge`]). The inner
    /// factor is clamped to `2..=MAX_FACTOR` and each outer factor to
    /// `1..=MAX_FACTOR`.
    pub fn new(inner: u32, outer: [u32; 4]) -> Self {
        Self {
            inner: clamp_factor(inner).max(2),
            outer: outer.map(clamp_factor),
        }
    }

    /// Returns the clamped interior factor actually used.
    pub fn inner(&self) -> u32 {
        self.inner
    }

    /// Returns the clamped outer factor used for `edge`.
    pub fn outer(&self, edge: Edge) -> u32 {
        self.outer[edge.index()]
    }

    /// Overrides a single edge's outer factor (clamped to `1..=MAX_FACTOR`),
    /// returning the updated recipe for chaining.
    pub fn with_outer(mut self, edge: Edge, factor: u32) -> Self {
        self.outer[edge.index()] = clamp_factor(factor);
        self
    }

    /// Tessellates `surface` into a watertight [`TriangleMesh`].
    ///
    /// Positions and shading normals are sampled analytically from `surface`;
    /// the `(u, v)` domain coordinate of each vertex is stored as its texture
    /// coordinate. Triangle winding is made consistently counter-clockwise in
    /// `(u, v)` parameter space.
    ///
    /// # Errors
    ///
    /// Propagates [`TriangleMeshError`] from [`TriangleMesh::new`]; by
    /// construction the generated indices and attribute pools are always valid,
    /// so this does not fail in practice.
    pub fn tessellate<S: ParametricSurface>(
        &self,
        surface: &S,
    ) -> Result<TriangleMesh, TriangleMeshError> {
        let mut builder = MeshBuilder::new(surface);
        self.append_interior(&mut builder);
        self.append_edge_strips(&mut builder);
        builder.into_mesh()
    }

    /// Tessellates `surface` and builds a [`TriangleMeshBvh`] over the result.
    ///
    /// # Errors
    ///
    /// Propagates any [`TriangleMeshError`] from [`PatchTessellation::tessellate`].
    pub fn tessellate_bvh<S: ParametricSurface>(
        &self,
        surface: &S,
    ) -> Result<TriangleMeshBvh, TriangleMeshError> {
        Ok(TriangleMeshBvh::build(self.tessellate(surface)?))
    }

    /// Triangulates the strictly interior nodes of the `inner × inner` grid.
    ///
    /// Interior nodes are `grid(i, j)` for `i, j ∈ 1..=inner-1`; the cells among
    /// them (`i, j ∈ 1..=inner-2`) are split into two counter-clockwise
    /// triangles. When `inner == 2` the interior block collapses to a single
    /// node and this adds no triangles — the four edge strips then fan to that
    /// shared centre node.
    fn append_interior<S: ParametricSurface>(&self, builder: &mut MeshBuilder<'_, S>) {
        let n = self.inner;
        if n < 3 {
            return;
        }
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                let a = builder.grid_vertex(i, j, n);
                let b = builder.grid_vertex(i + 1, j, n);
                let c = builder.grid_vertex(i + 1, j + 1, n);
                let d = builder.grid_vertex(i, j + 1, n);
                builder.push_quad(a, b, c, d);
            }
        }
    }

    /// Builds the four boundary transition strips, zipping each edge's outer
    /// sampling to the matching side of the interior block.
    fn append_edge_strips<S: ParametricSurface>(&self, builder: &mut MeshBuilder<'_, S>) {
        let n = self.inner;
        // Interior-block side indices (grid coordinates) per edge, ordered to
        // match the increasing-parameter direction of that edge. The shared
        // corner nodes appear as the first/last entry of two adjacent edges.
        //
        // Bottom (v=0, increasing u): inner side is the row j = 1.
        self.zip_edge(
            builder,
            Edge::Bottom,
            |t| [t, 0.0],
            (1..=n - 1).map(|i| (i, 1)).collect(),
        );
        // Right (u=1, increasing v): inner side is the column i = n-1.
        self.zip_edge(
            builder,
            Edge::Right,
            |t| [1.0, t],
            (1..=n - 1).map(|j| (n - 1, j)).collect(),
        );
        // Top (v=1, increasing u): inner side is the row j = n-1.
        self.zip_edge(
            builder,
            Edge::Top,
            |t| [t, 1.0],
            (1..=n - 1).map(|i| (i, n - 1)).collect(),
        );
        // Left (u=0, increasing v): inner side is the column i = 1.
        self.zip_edge(
            builder,
            Edge::Left,
            |t| [0.0, t],
            (1..=n - 1).map(|j| (1, j)).collect(),
        );
    }

    /// Zips one boundary edge to its interior-block side.
    ///
    /// `outer_uv` maps a scalar edge parameter `t ∈ [0, 1]` to the boundary
    /// `(u, v)`. `inner_cells` lists the interior-block grid nodes along this
    /// edge, in increasing-parameter order; its endpoints are the shared corner
    /// nodes. The two pointer walk advances whichever polyline is behind in
    /// normalized arc fraction, so a dense edge and a sparse interior stitch
    /// without T-junctions.
    fn zip_edge<S: ParametricSurface>(
        &self,
        builder: &mut MeshBuilder<'_, S>,
        edge: Edge,
        outer_uv: impl Fn(f32) -> [f32; 2],
        inner_cells: Vec<(u32, u32)>,
    ) {
        let n = self.inner;
        let segments = self.outer(edge);
        // Outer boundary vertices sampled at k / segments.
        let outer: Vec<usize> = (0..=segments)
            .map(|k| {
                let t = k as f32 / segments as f32;
                let uv = outer_uv(t);
                builder.vertex(uv[0], uv[1])
            })
            .collect();
        // Inner-block side vertices (reusing the shared grid nodes).
        let inner: Vec<usize> = inner_cells
            .iter()
            .map(|&(i, j)| builder.grid_vertex(i, j, n))
            .collect();

        if inner.len() == 1 {
            // Degenerate interior block (inner == 2): fan every outer segment to
            // the single shared centre node.
            for k in 0..outer.len() - 1 {
                builder.push_tri(outer[k], outer[k + 1], inner[0]);
            }
            return;
        }

        let outer_last = outer.len() - 1;
        let inner_last = inner.len() - 1;
        let mut oi = 0usize;
        let mut ii = 0usize;
        // Advance by comparing normalized fractions along the edge.
        while oi < outer_last || ii < inner_last {
            let of = oi as f32 / outer_last as f32;
            let if_ = ii as f32 / inner_last as f32;
            let advance_outer = ii == inner_last || (oi < outer_last && of <= if_);
            if advance_outer {
                builder.push_tri(outer[oi], outer[oi + 1], inner[ii]);
                oi += 1;
            } else {
                builder.push_tri(inner[ii], inner[ii + 1], outer[oi]);
                ii += 1;
            }
        }
    }
}

/// Accumulates positions, analytic normals, `(u, v)` texture coordinates, and
/// triangle indices while a [`PatchTessellation`] is built, deduplicating the
/// shared interior-grid nodes by grid coordinate.
struct MeshBuilder<'s, S: ParametricSurface> {
    /// The surface being sampled for positions and normals.
    surface: &'s S,
    /// Vertex positions in object space.
    positions: Vec<[f32; 3]>,
    /// Per-vertex analytic surface normals.
    normals: Vec<[f32; 3]>,
    /// Per-vertex `(u, v)` domain coordinates stored as texture coordinates.
    uvs: Vec<[f32; 2]>,
    /// Triangle index triples (winding corrected to CCW in `(u, v)` space).
    indices: Vec<[u32; 3]>,
    /// Reuse table for interior-grid nodes, keyed by packed `(i, j)` so a node
    /// shared between the interior block and an edge strip is emitted once.
    grid_cache: std::collections::HashMap<(u32, u32), usize>,
}

impl<'s, S: ParametricSurface> MeshBuilder<'s, S> {
    /// Creates an empty builder bound to `surface`.
    fn new(surface: &'s S) -> Self {
        Self {
            surface,
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            grid_cache: std::collections::HashMap::new(),
        }
    }

    /// Emits a fresh vertex at domain coordinate `(u, v)` and returns its index.
    fn vertex(&mut self, u: f32, v: f32) -> usize {
        let index = self.positions.len();
        self.positions.push(self.surface.point(u, v));
        self.normals.push(self.surface.normal(u, v));
        self.uvs.push([u, v]);
        index
    }

    /// Returns the deduplicated vertex index for grid node `(i, j)` of an
    /// `n × n` grid, creating it on first use.
    fn grid_vertex(&mut self, i: u32, j: u32, n: u32) -> usize {
        if let Some(&index) = self.grid_cache.get(&(i, j)) {
            return index;
        }
        let u = i as f32 / n as f32;
        let v = j as f32 / n as f32;
        let index = self.vertex(u, v);
        self.grid_cache.insert((i, j), index);
        index
    }

    /// Pushes a triangle, flipping its winding when it is clockwise in `(u, v)`
    /// parameter space so the whole mesh is consistently counter-clockwise.
    fn push_tri(&mut self, a: usize, b: usize, c: usize) {
        let pa = self.uvs[a];
        let pb = self.uvs[b];
        let pc = self.uvs[c];
        let area2 = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0]);
        let tri = if area2 < 0.0 {
            [a as u32, c as u32, b as u32]
        } else {
            [a as u32, b as u32, c as u32]
        };
        self.indices.push(tri);
    }

    /// Splits quad `a-b-c-d` (counter-clockwise in `(u, v)`) into two triangles.
    fn push_quad(&mut self, a: usize, b: usize, c: usize, d: usize) {
        self.push_tri(a, b, c);
        self.push_tri(a, c, d);
    }

    /// Finalizes the accumulated pools into a [`TriangleMesh`].
    fn into_mesh(self) -> Result<TriangleMesh, TriangleMeshError> {
        TriangleMesh::new(self.positions, self.normals, self.uvs, self.indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::traversal::Ray;

    /// A flat unit `(u, v)` plane in the `z = 0` surface, used so expected
    /// positions and watertightness are trivial to reason about.
    #[derive(Debug)]
    struct Plane;

    impl ParametricSurface for Plane {
        /// Maps `(u, v)` directly to `(u, v, 0)`.
        fn point(&self, u: f32, v: f32) -> [f32; 3] {
            [u, v, 0.0]
        }
        /// The plane normal is constant `+z`.
        fn normal(&self, _u: f32, _v: f32) -> [f32; 3] {
            [0.0, 0.0, 1.0]
        }
    }

    /// A paraboloid bowl `z = u² + v²` giving a curved, non-degenerate surface.
    #[derive(Debug)]
    struct Bowl;

    impl ParametricSurface for Bowl {
        /// Lifts `(u, v)` onto the bowl `z = u² + v²`.
        fn point(&self, u: f32, v: f32) -> [f32; 3] {
            [u, v, u * u + v * v]
        }
        /// Returns a unit normal to the bowl at `(u, v)`.
        fn normal(&self, u: f32, v: f32) -> [f32; 3] {
            // Gradient of (u, v, u²+v²): tangents (1,0,2u),(0,1,2v) -> n = (-2u,-2v,1).
            let n = [-2.0 * u, -2.0 * v, 1.0];
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            [n[0] / len, n[1] / len, n[2] / len]
        }
    }

    /// Counts how many triangles reference each undirected edge.
    fn edge_counts(mesh: &TriangleMesh) -> std::collections::HashMap<(u32, u32), u32> {
        let mut counts = std::collections::HashMap::new();
        for tri in mesh.indices() {
            for k in 0..3 {
                let a = tri[k];
                let b = tri[(k + 1) % 3];
                let key = if a < b { (a, b) } else { (b, a) };
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        counts
    }

    #[test]
    fn uniform_tessellation_is_watertight() {
        let mesh = PatchTessellation::uniform(5).tessellate(&Plane).unwrap();
        for (_, count) in edge_counts(&mesh) {
            // Interior edges are shared by exactly two triangles; boundary edges
            // by exactly one. Nothing may be referenced three or more times.
            assert!(count == 1 || count == 2, "edge used {count} times");
        }
    }

    #[test]
    fn degenerate_interior_block_fans_to_centre() {
        // inner clamps to 2 -> single centre node, all strips fan to it.
        let mesh = PatchTessellation::new(1, [3, 3, 3, 3]).tessellate(&Plane).unwrap();
        for (_, count) in edge_counts(&mesh) {
            assert!(count == 1 || count == 2, "edge used {count} times");
        }
        // The shared centre node (0.5, 0.5) must exist exactly once.
        let centres = mesh
            .uvs()
            .iter()
            .filter(|uv| (uv[0] - 0.5).abs() < 1e-6 && (uv[1] - 0.5).abs() < 1e-6)
            .count();
        assert_eq!(centres, 1);
    }

    #[test]
    fn boundary_sampled_at_outer_factor() {
        let tess = PatchTessellation::new(4, [6, 2, 3, 5]);
        let mesh = tess.tessellate(&Plane).unwrap();
        // The bottom edge (v == 0) should contain exactly outer+1 vertices at
        // parameters k / outer.
        let mut bottom_us: Vec<f32> = mesh
            .uvs()
            .iter()
            .filter(|uv| uv[1].abs() < 1e-6)
            .map(|uv| uv[0])
            .collect();
        bottom_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        bottom_us.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        assert_eq!(bottom_us.len(), 7, "bottom edge should have outer+1 vertices");
        for (k, u) in bottom_us.iter().enumerate() {
            assert!((u - k as f32 / 6.0).abs() < 1e-5);
        }
    }

    #[test]
    fn adjacent_patches_share_edge_vertices() {
        // Two patches that meet along u=1 of A / u=0 of B must agree on the
        // shared edge when both use the same outer factor there.
        let tess_a = PatchTessellation::new(3, [4, 7, 4, 2]);
        let tess_b = PatchTessellation::new(5, [6, 3, 6, 7]);
        let a = tess_a.tessellate(&Plane).unwrap();
        let b = tess_b.tessellate(&Plane).unwrap();
        // A's right edge (u == 1) vertices.
        let mut a_edge: Vec<f32> = a
            .uvs()
            .iter()
            .filter(|uv| (uv[0] - 1.0).abs() < 1e-6)
            .map(|uv| uv[1])
            .collect();
        // B's left edge (u == 0) vertices.
        let mut b_edge: Vec<f32> = b
            .uvs()
            .iter()
            .filter(|uv| uv[0].abs() < 1e-6)
            .map(|uv| uv[1])
            .collect();
        a_edge.sort_by(|x, y| x.partial_cmp(y).unwrap());
        a_edge.dedup_by(|x, y| (*x - *y).abs() < 1e-6);
        b_edge.sort_by(|x, y| x.partial_cmp(y).unwrap());
        b_edge.dedup_by(|x, y| (*x - *y).abs() < 1e-6);
        // Both edges used outer factor 7, so both carry v = k/7, k = 0..=7.
        assert_eq!(a_edge.len(), 8);
        assert_eq!(b_edge.len(), 8);
        for (x, y) in a_edge.iter().zip(b_edge.iter()) {
            assert!((x - y).abs() < 1e-5, "edge mismatch {x} vs {y}");
        }
    }

    #[test]
    fn factors_are_clamped() {
        let tess = PatchTessellation::new(0, [0, 999, 1, 999]);
        assert_eq!(tess.inner(), 2);
        assert_eq!(tess.outer(Edge::Bottom), 1);
        assert_eq!(tess.outer(Edge::Right), MAX_FACTOR);
        assert_eq!(tess.outer(Edge::Top), 1);
        assert_eq!(tess.outer(Edge::Left), MAX_FACTOR);
    }

    #[test]
    fn with_outer_overrides_one_edge() {
        let tess = PatchTessellation::uniform(4).with_outer(Edge::Top, 9);
        assert_eq!(tess.outer(Edge::Top), 9);
        assert_eq!(tess.outer(Edge::Bottom), 4);
    }

    #[test]
    fn all_vertices_lie_on_the_surface() {
        let mesh = PatchTessellation::uniform(6).tessellate(&Bowl).unwrap();
        for (pos, uv) in mesh.positions().iter().zip(mesh.uvs().iter()) {
            let expected = [uv[0], uv[1], uv[0] * uv[0] + uv[1] * uv[1]];
            for c in 0..3 {
                assert!((pos[c] - expected[c]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn curved_patch_is_watertight() {
        let mesh = PatchTessellation::new(4, [6, 3, 5, 2])
            .tessellate(&Bowl)
            .unwrap();
        for (_, count) in edge_counts(&mesh) {
            assert!(count == 1 || count == 2, "edge used {count} times");
        }
    }

    #[test]
    fn tessellation_is_deterministic() {
        let tess = PatchTessellation::new(4, [5, 3, 6, 2]);
        let a = tess.tessellate(&Bowl).unwrap();
        let b = tess.tessellate(&Bowl).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn bvh_hits_the_patch() {
        let bvh = PatchTessellation::uniform(5).tessellate_bvh(&Plane).unwrap();
        // Shoot down the +z axis at an off-seam point onto the z=0 plane.
        let ray = Ray::infinite([0.53, 0.47, 1.0], [0.0, 0.0, -1.0]);
        let hit = bvh.closest_hit(&ray).expect("ray should hit the plane");
        assert!((hit.position[2]).abs() < 1e-4);
        assert!((hit.position[0] - 0.53).abs() < 1e-4);
        assert!((hit.position[1] - 0.47).abs() < 1e-4);
    }

    #[test]
    fn triangle_count_scales_with_factors() {
        let coarse = PatchTessellation::uniform(3).tessellate(&Plane).unwrap();
        let fine = PatchTessellation::uniform(10).tessellate(&Plane).unwrap();
        assert!(fine.triangle_count() > coarse.triangle_count());
    }

    #[test]
    fn boundary_normals_follow_surface() {
        let mesh = PatchTessellation::uniform(4).tessellate(&Bowl).unwrap();
        // The (0,0) corner normal of the bowl is the unit +z vector.
        let idx = mesh
            .uvs()
            .iter()
            .position(|uv| uv[0].abs() < 1e-6 && uv[1].abs() < 1e-6)
            .unwrap();
        let n = mesh.normals()[idx];
        assert!((n[0]).abs() < 1e-6 && (n[1]).abs() < 1e-6 && (n[2] - 1.0).abs() < 1e-6);
    }
}
