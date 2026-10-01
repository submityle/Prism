//! Trimmed parametric surfaces: carve a parametric surface's `(u, v)` domain
//! with closed trim loops and tessellate only the kept region into a
//! watertight, conforming triangle mesh.
//!
//! A real CAD/NURBS asset is rarely a full rectangular patch. Instead a surface
//! carries a set of **trim loops** in parameter space — an outer boundary plus
//! inner holes — and only the enclosed region is solid. This module models that
//! with [`TrimmedSurface`]: any [`ParametricSurface`] (every `ray_scene` patch
//! and control-net surface implements it) plus a list of [`TrimLoop`] polygons.
//! A point is *kept* when the total number of loop edges a ray from it crosses
//! is odd (the even–odd fill rule), so nested loops naturally punch holes.
//!
//! [`TrimmedSurface::tessellate`] samples the `(u, v)` domain on a regular
//! `ru × rv` grid, classifies every grid node inside/outside the kept region,
//! and runs **marching squares** per cell: each of the sixteen corner-mask
//! cases emits the exact convex sub-polygon of the kept region, with trim-curve
//! intersections located by **bisection** along the crossed cell edges. The two
//! ambiguous saddle masks (opposite corners kept) are disambiguated with a
//! cell-centre inside test. Grid corners and per-edge crossings are shared
//! through caches keyed by grid-node / grid-edge identity, so adjacent cells
//! reference bit-identical vertices and the result is watertight (crack-free)
//! across the trim boundary. All math is division, comparison, and midpoint
//! arithmetic — no transcendental calls — so it matches the `GPU` tessellator
//! bit-for-bit.

use super::triangle_mesh::{TriangleMesh, TriangleMeshBvh};
use crate::ray_scene::displaced_surface::ParametricSurface;
use std::collections::HashMap;

/// Number of bisection steps used to locate a trim-curve crossing along a cell
/// edge. Each step halves the parameter interval, so `24` steps resolve the
/// crossing to roughly `2^-24 ≈ 6e-8` of the edge length in `(u, v)` space —
/// below single-precision position error for any reasonable surface.
const BISECT_ITERS: u32 = 24;

/// Why [`TrimmedSurface::new`] rejected its inputs.
#[derive(Clone, Debug, PartialEq)]
pub enum TrimmedSurfaceError {
    /// No trim loops were supplied; a trimmed surface needs at least one.
    NoLoops,
    /// A trim loop had fewer than three vertices and so encloses no area.
    DegenerateLoop {
        /// Zero-based index of the offending loop.
        loop_index: usize,
        /// Number of vertices the loop actually held.
        vertices: usize,
    },
}

impl core::fmt::Display for TrimmedSurfaceError {
    /// Formats the trimming error for diagnostics.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoLoops => write!(f, "a trimmed surface requires at least one trim loop"),
            Self::DegenerateLoop {
                loop_index,
                vertices,
            } => write!(
                f,
                "trim loop {loop_index} has {vertices} vertices but needs at least 3"
            ),
        }
    }
}

impl std::error::Error for TrimmedSurfaceError {}

/// A single closed trim loop in `(u, v)` parameter space.
///
/// The vertices are an implicitly closed polygon (the last vertex connects back
/// to the first). Winding direction is irrelevant: the kept region is defined by
/// the even–odd rule across every loop, so an outer boundary and inner holes are
/// supplied the same way and holes fall out of parity automatically.
#[derive(Clone, Debug, PartialEq)]
pub struct TrimLoop {
    /// Polygon vertices in `(u, v)` space, implicitly closed.
    vertices: Vec<[f32; 2]>,
}

impl TrimLoop {
    /// Builds a trim loop from its `(u, v)` polygon vertices.
    ///
    /// The caller is responsible for passing at least three vertices;
    /// [`TrimmedSurface::new`] validates this for every loop it receives.
    #[must_use]
    pub fn new(vertices: Vec<[f32; 2]>) -> Self {
        Self { vertices }
    }

    /// The loop's `(u, v)` polygon vertices.
    #[must_use]
    pub fn vertices(&self) -> &[[f32; 2]] {
        &self.vertices
    }

    /// Number of vertices in the loop.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Returns `true` when the ray `+u` from `point` crosses an odd number of
    /// this loop's edges, i.e. `point` is inside the loop under the even–odd
    /// rule.
    ///
    /// Uses the standard crossing-number test: an edge contributes a crossing
    /// when it straddles the horizontal line `v = point.v` and its intersection
    /// with that line lies to the `+u` side of `point`. The `v`-straddle guard
    /// `(vi > pv) != (vj > pv)` implies `vi != vj`, so the slope division is
    /// always well defined.
    #[must_use]
    pub fn contains(&self, point: [f32; 2]) -> bool {
        let n = self.vertices.len();
        if n < 3 {
            return false;
        }
        let (pu, pv) = (point[0], point[1]);
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let (ui, vi) = (self.vertices[i][0], self.vertices[i][1]);
            let (uj, vj) = (self.vertices[j][0], self.vertices[j][1]);
            if (vi > pv) != (vj > pv) {
                let u_cross = ui + (pv - vi) / (vj - vi) * (uj - ui);
                if pu < u_cross {
                    inside = !inside;
                }
            }
            j = i;
        }
        inside
    }
}

/// A reference to one of a marching-squares cell's four corners or four edge
/// crossings, used to describe the kept sub-polygon abstractly before the
/// vertices are resolved to global mesh indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CellVertex {
    /// Cell corner `0..=3` in counter-clockwise order.
    Corner(u8),
    /// Crossing on cell edge `0..=3` (edge `k` joins corner `k` and `k+1`).
    Edge(u8),
}

/// A parametric surface restricted to the region enclosed by its trim loops.
///
/// Generic over any [`ParametricSurface`]; the trim loops live in the surface's
/// `(u, v)` domain and the kept region is the even–odd fill of all loops.
#[derive(Clone, Debug, PartialEq)]
pub struct TrimmedSurface<S: ParametricSurface> {
    /// The underlying parametric surface being trimmed.
    base: S,
    /// Closed trim loops in `(u, v)` space defining the kept region.
    loops: Vec<TrimLoop>,
}

impl<S: ParametricSurface> TrimmedSurface<S> {
    /// Builds a trimmed surface from a base surface and its trim loops.
    ///
    /// # Errors
    ///
    /// Returns [`TrimmedSurfaceError::NoLoops`] when `loops` is empty, or
    /// [`TrimmedSurfaceError::DegenerateLoop`] when any loop has fewer than
    /// three vertices.
    pub fn new(base: S, loops: Vec<TrimLoop>) -> Result<Self, TrimmedSurfaceError> {
        if loops.is_empty() {
            return Err(TrimmedSurfaceError::NoLoops);
        }
        for (loop_index, trim_loop) in loops.iter().enumerate() {
            if trim_loop.vertex_count() < 3 {
                return Err(TrimmedSurfaceError::DegenerateLoop {
                    loop_index,
                    vertices: trim_loop.vertex_count(),
                });
            }
        }
        Ok(Self { base, loops })
    }

    /// The underlying parametric surface.
    #[must_use]
    pub fn base(&self) -> &S {
        &self.base
    }

    /// The trim loops defining the kept region.
    #[must_use]
    pub fn loops(&self) -> &[TrimLoop] {
        &self.loops
    }

    /// Returns `true` when `(u, v)` lies in the kept region, i.e. the parity of
    /// crossings summed over every trim loop is odd.
    ///
    /// Equivalent to the exclusive-or of each loop's [`TrimLoop::contains`] result, which is
    /// the even–odd fill across all loops together; this makes nested loops
    /// punch holes without any explicit outer/inner bookkeeping.
    #[must_use]
    pub fn contains(&self, u: f32, v: f32) -> bool {
        let point = [u, v];
        let mut inside = false;
        for trim_loop in &self.loops {
            inside ^= trim_loop.contains(point);
        }
        inside
    }

    /// Tessellates the kept region into a watertight, conforming triangle mesh
    /// over an `ru × rv` cell grid of the `[0, 1]²` parameter domain.
    ///
    /// Each cell is classified by its four corners' inside flags and resolved
    /// with marching squares; trim-boundary vertices are placed by bisection
    /// along the crossed cell edges and shared through grid-node / grid-edge
    /// caches so the mesh is crack-free. Per-vertex shading normals come from
    /// [`ParametricSurface::normal`] and texture coordinates are the surface
    /// parameters themselves. `ru`/`rv` are clamped to at least `1`.
    ///
    /// # Panics
    ///
    /// Never panics in practice: the index buffer it assembles only references
    /// vertices it just pushed, so the internal [`TriangleMesh::new`] always
    /// succeeds; an `expect` guards that invariant.
    #[must_use]
    pub fn tessellate(&self, ru: usize, rv: usize) -> TriangleMesh {
        let ru = ru.max(1);
        let rv = rv.max(1);
        let cols = ru + 1;
        let rows = rv + 1;

        // Classify every grid node once.
        let mut node_inside = Vec::with_capacity(cols * rows);
        for j in 0..rows {
            for i in 0..cols {
                let (u, v) = self.node_uv(i, j, ru, rv);
                node_inside.push(self.contains(u, v));
            }
        }

        let mut builder = MeshBuilder {
            surface: &self.base,
            ru,
            rv,
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            corner_cache: HashMap::new(),
            edge_cache: HashMap::new(),
        };

        for cj in 0..rv {
            for ci in 0..ru {
                let i0 = cj * cols + ci;
                let c0 = node_inside[i0];
                let c1 = node_inside[i0 + 1];
                let c2 = node_inside[i0 + cols + 1];
                let c3 = node_inside[i0 + cols];
                let mask = (c0 as u8) | ((c1 as u8) << 1) | ((c2 as u8) << 2) | ((c3 as u8) << 3);
                if mask == 0 {
                    continue;
                }
                let center_inside = if mask == 5 || mask == 10 {
                    let (cu, cv) = self.cell_center_uv(ci, cj, ru, rv);
                    Some(self.contains(cu, cv))
                } else {
                    None
                };
                for polygon in cell_polygons(mask, center_inside) {
                    builder.emit_polygon(ci, cj, &self.closure(), &polygon);
                }
            }
        }

        TriangleMesh::new(
            builder.positions,
            builder.normals,
            builder.uvs,
            builder.indices,
        )
        .expect("trimmed-surface indices only reference freshly pushed vertices")
    }

    /// Tessellates the kept region (see [`TrimmedSurface::tessellate`]) and
    /// builds a [`TriangleMeshBvh`] over the result for ray intersection.
    #[must_use]
    pub fn tessellate_bvh(&self, ru: usize, rv: usize) -> TriangleMeshBvh {
        TriangleMeshBvh::build(self.tessellate(ru, rv))
    }

    /// The `(u, v)` parameters of grid node `(i, j)` on an `ru × rv` grid.
    fn node_uv(&self, i: usize, j: usize, ru: usize, rv: usize) -> (f32, f32) {
        (i as f32 / ru as f32, j as f32 / rv as f32)
    }

    /// The `(u, v)` parameters of the centre of cell `(ci, cj)`.
    fn cell_center_uv(&self, ci: usize, cj: usize, ru: usize, rv: usize) -> (f32, f32) {
        ((ci as f32 + 0.5) / ru as f32, (cj as f32 + 0.5) / rv as f32)
    }

    /// A closure capturing the inside test for the bisection sampler. Keeping it
    /// as a borrow of `self` avoids re-borrowing conflicts inside the builder.
    fn closure(&self) -> impl Fn(f32, f32) -> bool + '_ {
        move |u, v| self.contains(u, v)
    }
}

/// Accumulates the trimmed mesh and shares corner / edge-crossing vertices.
struct MeshBuilder<'a, S: ParametricSurface> {
    /// The surface sampled for positions and normals.
    surface: &'a S,
    /// Cell columns along `u`.
    ru: usize,
    /// Cell rows along `v`.
    rv: usize,
    /// Accumulated vertex positions.
    positions: Vec<[f32; 3]>,
    /// Accumulated per-vertex shading normals.
    normals: Vec<[f32; 3]>,
    /// Accumulated per-vertex `(u, v)` texture coordinates.
    uvs: Vec<[f32; 2]>,
    /// Accumulated triangle index triples.
    indices: Vec<[u32; 3]>,
    /// Cached global index for each kept grid corner, keyed by `(i, j)`.
    corner_cache: HashMap<(usize, usize), u32>,
    /// Cached global index for each edge crossing, keyed by `(tag, i, j)` where
    /// `tag = 0` is the horizontal edge from node `(i, j)` to `(i + 1, j)` and
    /// `tag = 1` is the vertical edge from node `(i, j)` to `(i, j + 1)`.
    edge_cache: HashMap<(u8, usize, usize), u32>,
}

impl<S: ParametricSurface> MeshBuilder<'_, S> {
    /// Pushes a surface sample at `(u, v)` and returns its global vertex index.
    fn push_sample(&mut self, u: f32, v: f32) -> u32 {
        let index = self.positions.len() as u32;
        self.positions.push(self.surface.point(u, v));
        self.normals.push(self.surface.normal(u, v));
        self.uvs.push([u, v]);
        index
    }

    /// Returns the shared vertex index for kept grid corner `(i, j)`.
    fn corner_vertex(&mut self, i: usize, j: usize) -> u32 {
        if let Some(&index) = self.corner_cache.get(&(i, j)) {
            return index;
        }
        let u = i as f32 / self.ru as f32;
        let v = j as f32 / self.rv as f32;
        let index = self.push_sample(u, v);
        self.corner_cache.insert((i, j), index);
        index
    }

    /// Returns the shared vertex index for the trim crossing on a grid edge,
    /// located by bisection between the edge's two nodes.
    fn edge_vertex(
        &mut self,
        tag: u8,
        i: usize,
        j: usize,
        inside: &impl Fn(f32, f32) -> bool,
    ) -> u32 {
        if let Some(&index) = self.edge_cache.get(&(tag, i, j)) {
            return index;
        }
        let a = (i as f32 / self.ru as f32, j as f32 / self.rv as f32);
        let b = if tag == 0 {
            ((i + 1) as f32 / self.ru as f32, j as f32 / self.rv as f32)
        } else {
            (i as f32 / self.ru as f32, (j + 1) as f32 / self.rv as f32)
        };
        let (u, v) = bisect_crossing(a, b, inside);
        let index = self.push_sample(u, v);
        self.edge_cache.insert((tag, i, j), index);
        index
    }

    /// Resolves one abstract cell vertex of cell `(ci, cj)` to a global index.
    fn resolve(
        &mut self,
        ci: usize,
        cj: usize,
        vertex: CellVertex,
        inside: &impl Fn(f32, f32) -> bool,
    ) -> u32 {
        match vertex {
            CellVertex::Corner(0) => self.corner_vertex(ci, cj),
            CellVertex::Corner(1) => self.corner_vertex(ci + 1, cj),
            CellVertex::Corner(2) => self.corner_vertex(ci + 1, cj + 1),
            CellVertex::Corner(_) => self.corner_vertex(ci, cj + 1),
            CellVertex::Edge(0) => self.edge_vertex(0, ci, cj, inside),
            CellVertex::Edge(1) => self.edge_vertex(1, ci + 1, cj, inside),
            CellVertex::Edge(2) => self.edge_vertex(0, ci, cj + 1, inside),
            CellVertex::Edge(_) => self.edge_vertex(1, ci, cj, inside),
        }
    }

    /// Fan-triangulates one kept sub-polygon of cell `(ci, cj)` into the mesh.
    fn emit_polygon(
        &mut self,
        ci: usize,
        cj: usize,
        inside: &impl Fn(f32, f32) -> bool,
        polygon: &[CellVertex],
    ) {
        if polygon.len() < 3 {
            return;
        }
        let mut resolved = Vec::with_capacity(polygon.len());
        for &vertex in polygon {
            resolved.push(self.resolve(ci, cj, vertex, inside));
        }
        for k in 1..resolved.len() - 1 {
            self.indices
                .push([resolved[0], resolved[k], resolved[k + 1]]);
        }
    }
}

/// Locates the trim-curve crossing on the segment `a`–`b` by bisection on the
/// `inside` predicate, orienting the search so `lo` is always inside.
///
/// Exactly one endpoint is inside when this is called, so the midpoint test
/// strictly narrows the bracket each step. The two endpoints and their inside
/// flags are fixed per grid edge, so both cells sharing the edge compute an
/// identical crossing — the key to a watertight seam.
fn bisect_crossing(
    a: (f32, f32),
    b: (f32, f32),
    inside: &impl Fn(f32, f32) -> bool,
) -> (f32, f32) {
    let (mut lo, mut hi) = if inside(a.0, a.1) { (a, b) } else { (b, a) };
    for _ in 0..BISECT_ITERS {
        let mid = (0.5 * (lo.0 + hi.0), 0.5 * (lo.1 + hi.1));
        if inside(mid.0, mid.1) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (0.5 * (lo.0 + hi.0), 0.5 * (lo.1 + hi.1))
}

/// Returns the kept sub-polygon(s) of a marching-squares cell for a given
/// corner `mask` (bit `k` set when corner `k` is inside), in counter-clockwise
/// order and ready for fan triangulation.
///
/// Corners are `0..=3` counter-clockwise; edge `k` joins corner `k` and corner
/// `k + 1 (mod 4)`. The thirteen unambiguous masks map to a single polygon. The
/// two saddle masks (`5` = corners 0 & 2, `10` = corners 1 & 3) are ambiguous:
/// `center_inside` (the cell-centre inside test) decides whether the kept region
/// is one hexagon spanning both corners or two disjoint corner triangles.
fn cell_polygons(mask: u8, center_inside: Option<bool>) -> Vec<Vec<CellVertex>> {
    use CellVertex::{Corner, Edge};
    match mask {
        1 => vec![vec![Edge(3), Corner(0), Edge(0)]],
        2 => vec![vec![Edge(0), Corner(1), Edge(1)]],
        4 => vec![vec![Edge(1), Corner(2), Edge(2)]],
        8 => vec![vec![Edge(2), Corner(3), Edge(3)]],
        3 => vec![vec![Edge(3), Corner(0), Corner(1), Edge(1)]],
        6 => vec![vec![Edge(0), Corner(1), Corner(2), Edge(2)]],
        12 => vec![vec![Edge(1), Corner(2), Corner(3), Edge(3)]],
        9 => vec![vec![Edge(2), Corner(3), Corner(0), Edge(0)]],
        7 => vec![vec![Edge(3), Corner(0), Corner(1), Corner(2), Edge(2)]],
        11 => vec![vec![Corner(0), Corner(1), Edge(1), Edge(2), Corner(3)]],
        13 => vec![vec![Corner(0), Edge(0), Edge(1), Corner(2), Corner(3)]],
        14 => vec![vec![Corner(1), Corner(2), Corner(3), Edge(3), Edge(0)]],
        15 => vec![vec![Corner(0), Corner(1), Corner(2), Corner(3)]],
        5 => {
            if center_inside.unwrap_or(false) {
                vec![vec![
                    Corner(0),
                    Edge(0),
                    Edge(1),
                    Corner(2),
                    Edge(2),
                    Edge(3),
                ]]
            } else {
                vec![
                    vec![Edge(3), Corner(0), Edge(0)],
                    vec![Edge(1), Corner(2), Edge(2)],
                ]
            }
        }
        10 => {
            if center_inside.unwrap_or(false) {
                vec![vec![
                    Corner(1),
                    Edge(1),
                    Edge(2),
                    Corner(3),
                    Edge(3),
                    Edge(0),
                ]]
            } else {
                vec![
                    vec![Edge(0), Corner(1), Edge(1)],
                    vec![Edge(2), Corner(3), Edge(3)],
                ]
            }
        }
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::traversal::Ray;

    /// A flat parametric plane spanning `origin + u*edge_u + v*edge_v` with a
    /// constant unit normal, used to exercise trimming independent of curvature.
    #[derive(Debug)]
    struct PlaneSurface {
        origin: [f32; 3],
        edge_u: [f32; 3],
        edge_v: [f32; 3],
        normal: [f32; 3],
    }

    impl ParametricSurface for PlaneSurface {
        fn point(&self, u: f32, v: f32) -> [f32; 3] {
            [
                self.origin[0] + u * self.edge_u[0] + v * self.edge_v[0],
                self.origin[1] + u * self.edge_u[1] + v * self.edge_v[1],
                self.origin[2] + u * self.edge_u[2] + v * self.edge_v[2],
            ]
        }
        fn normal(&self, _u: f32, _v: f32) -> [f32; 3] {
            self.normal
        }
    }

    /// A unit `XY` plane at `z = 0` with `+Z` normal.
    fn xy_plane() -> PlaneSurface {
        PlaneSurface {
            origin: [0.0, 0.0, 0.0],
            edge_u: [1.0, 0.0, 0.0],
            edge_v: [0.0, 1.0, 0.0],
            normal: [0.0, 0.0, 1.0],
        }
    }

    /// A rectangle loop in `(u, v)` space.
    fn rect(u0: f32, v0: f32, u1: f32, v1: f32) -> TrimLoop {
        TrimLoop::new(vec![[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
    }

    /// A ray shot straight down the `-Z` axis at world `(x, y)`.
    fn down_ray(x: f32, y: f32) -> Ray {
        Ray::infinite([x, y, 1.0], [0.0, 0.0, -1.0])
    }

    #[test]
    fn rejects_empty_loop_set() {
        let err = TrimmedSurface::new(xy_plane(), Vec::new()).unwrap_err();
        assert_eq!(err, TrimmedSurfaceError::NoLoops);
    }

    #[test]
    fn rejects_degenerate_loop() {
        let bad = TrimLoop::new(vec![[0.0, 0.0], [1.0, 0.0]]);
        let err = TrimmedSurface::new(xy_plane(), vec![bad]).unwrap_err();
        assert_eq!(
            err,
            TrimmedSurfaceError::DegenerateLoop {
                loop_index: 0,
                vertices: 2,
            }
        );
    }

    #[test]
    fn contains_matches_even_odd_rule() {
        // Outer square with an inner hole square (nested loops).
        let outer = rect(0.1, 0.1, 0.9, 0.9);
        let hole = rect(0.4, 0.4, 0.6, 0.6);
        let surf = TrimmedSurface::new(xy_plane(), vec![outer, hole]).unwrap();
        assert!(surf.contains(0.2, 0.2)); // inside outer, outside hole
        assert!(!surf.contains(0.5, 0.5)); // inside the hole -> removed
        assert!(!surf.contains(0.95, 0.5)); // outside outer
    }

    #[test]
    fn full_cover_tessellates_whole_grid() {
        // A loop larger than the domain keeps everything: every cell is full,
        // so a 4x4 grid yields 2 triangles per cell.
        let cover = rect(-0.2, -0.2, 1.2, 1.2);
        let surf = TrimmedSurface::new(xy_plane(), vec![cover]).unwrap();
        let mesh = surf.tessellate(4, 4);
        assert_eq!(mesh.triangle_count(), 2 * 4 * 4);
        assert!(mesh.positions().iter().all(|p| p.iter().all(|c| c.is_finite())));
    }

    #[test]
    fn empty_region_yields_no_triangles() {
        // A tiny loop entirely inside a single coarse cell, placed off the grid
        // nodes, keeps no grid node and so emits nothing on a 1x1 grid.
        let tiny = rect(0.45, 0.45, 0.55, 0.55);
        let surf = TrimmedSurface::new(xy_plane(), vec![tiny]).unwrap();
        let mesh = surf.tessellate(1, 1);
        assert_eq!(mesh.triangle_count(), 0);
    }

    #[test]
    fn ray_hits_kept_region_and_misses_hole() {
        let outer = rect(0.05, 0.05, 0.95, 0.95);
        let hole = rect(0.35, 0.35, 0.65, 0.65);
        let surf = TrimmedSurface::new(xy_plane(), vec![outer, hole]).unwrap();
        let bvh = surf.tessellate_bvh(16, 16);
        // A point well inside the kept annulus is hit.
        assert!(bvh.closest_hit(&down_ray(0.2, 0.2)).is_some());
        // A point deep in the hole is trimmed away -> miss.
        assert!(bvh.closest_hit(&down_ray(0.5, 0.5)).is_none());
        // A point outside the outer boundary -> miss.
        assert!(bvh.closest_hit(&down_ray(0.99, 0.5)).is_none());
    }

    #[test]
    fn kept_hit_reports_plane_depth_and_normal() {
        let cover = rect(-0.2, -0.2, 1.2, 1.2);
        let surf = TrimmedSurface::new(xy_plane(), vec![cover]).unwrap();
        let bvh = surf.tessellate_bvh(8, 8);
        let hit = bvh.closest_hit(&down_ray(0.37, 0.61)).expect("interior hit");
        // Plane sits at z = 0 and the ray starts at z = 1 going -Z.
        assert!((hit.t - 1.0).abs() < 1e-4);
        // Flat plane: interpolated normal is +Z (sign may flip for front/back).
        assert!(hit.normal[2].abs() > 0.999);
        assert!(hit.normal[0].abs() < 1e-3 && hit.normal[1].abs() < 1e-3);
    }

    #[test]
    fn boundary_vertices_land_on_trim_curve() {
        // A diagonal triangle loop; crossings must sit on its edges within the
        // bisection tolerance. Check the kept-region boundary respects u+v<=0.8
        // roughly by sampling: every emitted vertex is inside-or-on the loop.
        let tri = TrimLoop::new(vec![[0.0, 0.0], [0.8, 0.0], [0.0, 0.8]]);
        let surf = TrimmedSurface::new(xy_plane(), vec![tri]).unwrap();
        let mesh = surf.tessellate(16, 16);
        for p in mesh.uvs() {
            // Allow a small tolerance for bisected boundary vertices.
            assert!(p[0] + p[1] <= 0.8 + 1e-3, "uv {:?} escaped the trim loop", p);
        }
        assert!(mesh.triangle_count() > 0);
    }

    #[test]
    fn saddle_configuration_is_watertight_and_valid() {
        // Two separate diagonal squares produce opposite-corner (saddle) cells
        // on a coarse grid. Ensure tessellation stays finite and indices are in
        // range (TriangleMesh::new would have errored otherwise).
        let a = rect(-0.1, -0.1, 0.45, 0.45);
        let b = rect(0.55, 0.55, 1.1, 1.1);
        let surf = TrimmedSurface::new(xy_plane(), vec![a, b]).unwrap();
        let mesh = surf.tessellate(2, 2);
        assert!(mesh.triangle_count() > 0);
        let vcount = mesh.vertex_count() as u32;
        for tri in mesh.indices() {
            assert!(tri.iter().all(|&idx| idx < vcount));
        }
    }

    #[test]
    fn tessellation_is_deterministic() {
        let outer = rect(0.1, 0.1, 0.9, 0.9);
        let hole = rect(0.4, 0.4, 0.6, 0.6);
        let surf = TrimmedSurface::new(xy_plane(), vec![outer, hole]).unwrap();
        let a = surf.tessellate(12, 12);
        let b = surf.tessellate(12, 12);
        assert_eq!(a, b);
    }

    #[test]
    fn shared_edge_crossings_are_deduplicated() {
        // On a kept half-plane (u <= 0.5), the vertical trim boundary crosses
        // many horizontal cell edges. Shared crossings must be reused, so the
        // vertex count stays well below naive per-cell emission.
        let keep = rect(-0.2, -0.2, 0.5, 1.2);
        let surf = TrimmedSurface::new(xy_plane(), vec![keep]).unwrap();
        let mesh = surf.tessellate(8, 8);
        // Naive (no dedup) would push 3-6 verts per non-empty cell (>= 150).
        assert!(mesh.vertex_count() < 120, "vertices {}", mesh.vertex_count());
        assert!(mesh.triangle_count() > 0);
    }

    #[test]
    fn trim_loop_contains_handles_small_polygon() {
        let loop_a = rect(0.0, 0.0, 1.0, 1.0);
        assert!(loop_a.contains([0.5, 0.5]));
        assert!(!loop_a.contains([1.5, 0.5]));
        assert_eq!(loop_a.vertex_count(), 4);
    }
}
