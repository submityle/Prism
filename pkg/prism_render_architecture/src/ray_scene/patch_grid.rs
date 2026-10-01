//! Heterogeneous-tolerance, crack-free patch grid for the `CPU` golden path.
//!
//! [`crate::ray_scene::adaptive_tessellation::AdaptiveTessellation`] refines a
//! surface with a *single* global tolerance, so every shared sub-patch edge is
//! measured identically from both sides and automatically agrees. Production
//! LOD authoring, however, wants **spatially varying** detail — a dense
//! tolerance over a character's face, a loose one over the torso — on the same
//! surface. Varying the tolerance per cell naively re-introduces T-junctions:
//! the two cells flanking a shared edge would demand *different* segment counts.
//!
//! [`PatchGrid`] fixes this by resolving every interior edge to the **maximum**
//! of the two adjacent cells' demands. Because both demands are computed from
//! the *same* global edge endpoints (only the tolerance differs), taking the
//! max makes both cells emit the identical, bit-for-bit boundary samples, and
//! the position-keyed weld fuses them into one watertight seam. Interior cell
//! density is the max of its four resolved outer factors and two mid-cross
//! probes at the cell's own tolerance, so a fine cell is never starved by a
//! coarse neighbour.
//!
//! The per-cell factor search reuses the same equal-chord deviation metric as
//! the adaptive driver, and the final triangulation of each cell is delegated
//! to [`PatchTessellation`]. All math is linear plus the segment-distance
//! projection, honouring the golden-path ban on `f32` transcendental functions.

use std::collections::HashMap;

use super::displaced_surface::ParametricSurface;
use super::patch_tessellation::{PatchTessellation, MAX_FACTOR};
use super::triangle_mesh::{TriangleMesh, TriangleMeshBvh, TriangleMeshError};

/// Dense per-edge probe count for the chordal-deviation test.
const PROBE_SAMPLES: u32 = 32;

/// A rectangular grid of sub-patches over one parametric surface, each cell
/// carrying its own refinement tolerance, stitched crack-free.
///
/// Cells are addressed `(col, row)` with `col ∈ 0..cols`, `row ∈ 0..rows`, and
/// tolerances are stored row-major (`row * cols + col`). A smaller tolerance
/// refines a cell more aggressively.
#[derive(Clone, Debug, PartialEq)]
pub struct PatchGrid {
    /// Sub-patch columns across the `u` domain (clamped to `>= 1`).
    cols: u32,
    /// Sub-patch rows across the `v` domain (clamped to `>= 1`).
    rows: u32,
    /// Per-edge segment cap, clamped into `1..=`[`MAX_FACTOR`].
    max_factor: u32,
    /// Row-major per-cell chordal tolerances, each floored at a small epsilon.
    tolerances: Vec<f32>,
}

impl PatchGrid {
    /// Builds a `cols × rows` grid with a uniform `tolerance` everywhere.
    ///
    /// `cols`/`rows` are floored at `1`, `tolerance` at a small positive
    /// epsilon, and `max_factor` is clamped into `1..=`[`MAX_FACTOR`].
    pub fn new(cols: u32, rows: u32, tolerance: f32, max_factor: u32) -> Self {
        let cols = cols.max(1);
        let rows = rows.max(1);
        let tol = tolerance.max(1e-6);
        Self {
            cols,
            rows,
            max_factor: max_factor.clamp(1, MAX_FACTOR),
            tolerances: vec![tol; (cols * rows) as usize],
        }
    }

    /// Returns the clamped column count.
    pub fn cols(&self) -> u32 {
        self.cols
    }

    /// Returns the clamped row count.
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Returns the clamped per-edge segment cap.
    pub fn max_factor(&self) -> u32 {
        self.max_factor
    }

    /// Returns the tolerance stored for cell `(col, row)`, or `None` when the
    /// address is out of range.
    pub fn tolerance(&self, col: u32, row: u32) -> Option<f32> {
        self.cell_index(col, row).map(|i| self.tolerances[i])
    }

    /// Overrides the tolerance of cell `(col, row)` (floored at a small
    /// epsilon). Out-of-range addresses are ignored. Returns `self` for
    /// chaining.
    pub fn with_tolerance(mut self, col: u32, row: u32, tolerance: f32) -> Self {
        if let Some(i) = self.cell_index(col, row) {
            self.tolerances[i] = tolerance.max(1e-6);
        }
        self
    }

    /// Adaptively tessellates `surface` into one watertight [`TriangleMesh`],
    /// honouring each cell's tolerance while keeping shared edges crack-free.
    ///
    /// # Errors
    ///
    /// Propagates [`TriangleMeshError`] from [`TriangleMesh::new`]; by
    /// construction the generated pools are valid, so this does not fail in
    /// practice.
    pub fn tessellate<S: ParametricSurface>(
        &self,
        surface: &S,
    ) -> Result<TriangleMesh, TriangleMeshError> {
        let mut merge = Merge::new();
        for row in 0..self.rows {
            for col in 0..self.cols {
                let outer = self.resolved_outer(surface, col, row);
                let inner = self.inner_factor(surface, col, row, outer);
                let sub = SubPatch {
                    surface,
                    col,
                    row,
                    cols: self.cols,
                    rows: self.rows,
                };
                let mesh = PatchTessellation::new(inner, outer).tessellate(&sub)?;
                merge.append(&mesh, col, row, self.cols, self.rows);
            }
        }
        merge.into_mesh()
    }

    /// Adaptively tessellates `surface` and builds a [`TriangleMeshBvh`].
    ///
    /// # Errors
    ///
    /// Propagates any [`TriangleMeshError`] from [`PatchGrid::tessellate`].
    pub fn tessellate_bvh<S: ParametricSurface>(
        &self,
        surface: &S,
    ) -> Result<TriangleMeshBvh, TriangleMeshError> {
        Ok(TriangleMeshBvh::build(self.tessellate(surface)?))
    }

    /// Maps a cell address to its row-major tolerance index, if in range.
    fn cell_index(&self, col: u32, row: u32) -> Option<usize> {
        (col < self.cols && row < self.rows).then(|| (row * self.cols + col) as usize)
    }

    /// Returns the stored tolerance for an in-range cell (callers guarantee
    /// range during tessellation).
    fn cell_tolerance(&self, col: u32, row: u32) -> f32 {
        self.tolerances[(row * self.cols + col) as usize]
    }

    /// Resolves the four outer factors of cell `(col, row)` in
    /// [`PatchTessellation`] order `[bottom (v low), right (u high), top (v
    /// high), left (u low)]`.
    ///
    /// Interior edges take the maximum demand of both adjacent cells so the two
    /// sides agree; boundary edges use the cell's own demand.
    fn resolved_outer<S: ParametricSurface>(
        &self,
        surface: &S,
        col: u32,
        row: u32,
    ) -> [u32; 4] {
        let u0 = col as f32 / self.cols as f32;
        let u1 = (col + 1) as f32 / self.cols as f32;
        let v0 = row as f32 / self.rows as f32;
        let v1 = (row + 1) as f32 / self.rows as f32;
        let own = self.cell_tolerance(col, row);

        // Bottom edge (v = v0): shared with the cell below (row - 1).
        let bottom = {
            let mut tol = own;
            if row > 0 {
                tol = tol.min(self.cell_tolerance(col, row - 1));
            }
            self.edge_factor(surface, [u0, v0], [u1, v0], tol)
        };
        // Right edge (u = u1): shared with the cell to the right (col + 1).
        let right = {
            let mut tol = own;
            if col + 1 < self.cols {
                tol = tol.min(self.cell_tolerance(col + 1, row));
            }
            self.edge_factor(surface, [u1, v0], [u1, v1], tol)
        };
        // Top edge (v = v1): shared with the cell above (row + 1).
        let top = {
            let mut tol = own;
            if row + 1 < self.rows {
                tol = tol.min(self.cell_tolerance(col, row + 1));
            }
            self.edge_factor(surface, [u0, v1], [u1, v1], tol)
        };
        // Left edge (u = u0): shared with the cell to the left (col - 1).
        let left = {
            let mut tol = own;
            if col > 0 {
                tol = tol.min(self.cell_tolerance(col - 1, row));
            }
            self.edge_factor(surface, [u0, v0], [u0, v1], tol)
        };
        [bottom, right, top, left]
    }

    /// Picks the interior factor as the maximum of the four resolved outer
    /// factors and two mid-cross probes at the cell's own tolerance.
    fn inner_factor<S: ParametricSurface>(
        &self,
        surface: &S,
        col: u32,
        row: u32,
        outer: [u32; 4],
    ) -> u32 {
        let u0 = col as f32 / self.cols as f32;
        let u1 = (col + 1) as f32 / self.cols as f32;
        let v0 = row as f32 / self.rows as f32;
        let v1 = (row + 1) as f32 / self.rows as f32;
        let um = 0.5 * (u0 + u1);
        let vm = 0.5 * (v0 + v1);
        let own = self.cell_tolerance(col, row);
        let mid_u = self.edge_factor(surface, [u0, vm], [u1, vm], own);
        let mid_v = self.edge_factor(surface, [um, v0], [um, v1], own);
        outer
            .iter()
            .copied()
            .chain([mid_u, mid_v])
            .max()
            .unwrap_or(1)
    }

    /// Returns the smallest segment count in `1..=max_factor` whose equal-chord
    /// approximation of the surface along the global edge `a -> b` keeps every
    /// dense probe within `tolerance` of its chord.
    fn edge_factor<S: ParametricSurface>(
        &self,
        surface: &S,
        a: [f32; 2],
        b: [f32; 2],
        tolerance: f32,
    ) -> u32 {
        for n in 1..self.max_factor {
            if chord_fits(surface, a, b, n, tolerance) {
                return n;
            }
        }
        self.max_factor
    }
}

/// Tests whether an `n`-segment chord of the global edge `a -> b` keeps the
/// surface within `tolerance` at every dense probe.
fn chord_fits<S: ParametricSurface>(
    surface: &S,
    a: [f32; 2],
    b: [f32; 2],
    n: u32,
    tolerance: f32,
) -> bool {
    let tol2 = tolerance * tolerance;
    for d in 0..=PROBE_SAMPLES {
        let t = d as f32 / PROBE_SAMPLES as f32;
        let probe = lerp2(a, b, t);
        let p = surface.point(probe[0], probe[1]);
        let seg = ((t * n as f32).floor() as u32).min(n - 1);
        let c0_uv = lerp2(a, b, seg as f32 / n as f32);
        let c1_uv = lerp2(a, b, (seg + 1) as f32 / n as f32);
        let c0 = surface.point(c0_uv[0], c0_uv[1]);
        let c1 = surface.point(c1_uv[0], c1_uv[1]);
        if point_segment_distance_sq(p, c0, c1) > tol2 {
            return false;
        }
    }
    true
}

/// Linearly interpolates between two `(u, v)` parameters.
fn lerp2(a: [f32; 2], b: [f32; 2], t: f32) -> [f32; 2] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

/// Returns the squared distance from point `p` to segment `c0 c1`.
fn point_segment_distance_sq(p: [f32; 3], c0: [f32; 3], c1: [f32; 3]) -> f32 {
    let d = sub(c1, c0);
    let len2 = dot(d, d);
    let t = if len2 > 0.0 {
        (dot(sub(p, c0), d) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let proj = [c0[0] + d[0] * t, c0[1] + d[1] * t, c0[2] + d[2] * t];
    let diff = sub(p, proj);
    dot(diff, diff)
}

/// Component-wise `a - b`.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Dot product of two vectors.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// A single cell viewed as a parametric surface over its own local
/// `(lu, lv) ∈ [0, 1]²`, mapping to the parent surface's global domain.
///
/// The mapping `global = (base_index + local) / count` makes two neighbouring
/// cells evaluate a shared edge at bit-identical global parameters, which is
/// what makes the merged mesh watertight.
struct SubPatch<'s, S: ParametricSurface> {
    /// The parent surface being sampled.
    surface: &'s S,
    /// This cell's column index in `0..cols`.
    col: u32,
    /// This cell's row index in `0..rows`.
    row: u32,
    /// Total columns, the `u` denominator.
    cols: u32,
    /// Total rows, the `v` denominator.
    rows: u32,
}

impl<S: ParametricSurface> SubPatch<'_, S> {
    /// Maps a local `(lu, lv)` to the parent's global `(u, v)`.
    fn to_global(&self, lu: f32, lv: f32) -> [f32; 2] {
        [
            (self.col as f32 + lu) / self.cols as f32,
            (self.row as f32 + lv) / self.rows as f32,
        ]
    }
}

impl<S: ParametricSurface> ParametricSurface for SubPatch<'_, S> {
    /// Forwards the global position of the local parameter.
    fn point(&self, lu: f32, lv: f32) -> [f32; 3] {
        let g = self.to_global(lu, lv);
        self.surface.point(g[0], g[1])
    }
    /// Forwards the global normal of the local parameter.
    fn normal(&self, lu: f32, lv: f32) -> [f32; 3] {
        let g = self.to_global(lu, lv);
        self.surface.normal(g[0], g[1])
    }
}

/// Position-keyed welding accumulator that fuses per-cell tessellations into
/// one watertight global mesh.
struct Merge {
    /// Global vertex positions.
    positions: Vec<[f32; 3]>,
    /// Global per-vertex normals.
    normals: Vec<[f32; 3]>,
    /// Global per-vertex `(u, v)` coordinates remapped to the parent domain.
    uvs: Vec<[f32; 2]>,
    /// Global triangle indices.
    indices: Vec<[u32; 3]>,
    /// Weld table keyed by the three position component bit patterns.
    cache: HashMap<[u32; 3], u32>,
}

impl Merge {
    /// Creates an empty merge accumulator.
    fn new() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            cache: HashMap::new(),
        }
    }

    /// Appends a cell tessellation (whose uvs are *local*), welding vertices by
    /// position bits and remapping uvs into the parent domain for `(col, row)`.
    fn append(&mut self, mesh: &TriangleMesh, col: u32, row: u32, cols: u32, rows: u32) {
        let mut remap = vec![0u32; mesh.vertex_count()];
        for (local, pos) in mesh.positions().iter().enumerate() {
            let key = [pos[0].to_bits(), pos[1].to_bits(), pos[2].to_bits()];
            let global = if let Some(&existing) = self.cache.get(&key) {
                existing
            } else {
                let idx = self.positions.len() as u32;
                self.positions.push(*pos);
                self.normals.push(mesh.normals()[local]);
                let luv = mesh.uvs()[local];
                self.uvs.push([
                    (col as f32 + luv[0]) / cols as f32,
                    (row as f32 + luv[1]) / rows as f32,
                ]);
                self.cache.insert(key, idx);
                idx
            };
            remap[local] = global;
        }
        for tri in mesh.indices() {
            self.indices.push([
                remap[tri[0] as usize],
                remap[tri[1] as usize],
                remap[tri[2] as usize],
            ]);
        }
    }

    /// Finalizes the welded pools into a [`TriangleMesh`].
    fn into_mesh(self) -> Result<TriangleMesh, TriangleMeshError> {
        TriangleMesh::new(self.positions, self.normals, self.uvs, self.indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::traversal::Ray;

    /// A flat unit plane `z = 0` with constant `+z` normal.
    #[derive(Debug)]
    struct Plane;

    impl ParametricSurface for Plane {
        /// Maps `(u, v)` to `(u, v, 0)`.
        fn point(&self, u: f32, v: f32) -> [f32; 3] {
            [u, v, 0.0]
        }
        /// Constant `+z` normal.
        fn normal(&self, _u: f32, _v: f32) -> [f32; 3] {
            [0.0, 0.0, 1.0]
        }
    }

    /// A steep bowl `z = 6 (u² + v²)`; curvature forces refinement.
    #[derive(Debug)]
    struct Bowl;

    impl ParametricSurface for Bowl {
        /// Lifts `(u, v)` onto the steep bowl.
        fn point(&self, u: f32, v: f32) -> [f32; 3] {
            [u, v, 6.0 * (u * u + v * v)]
        }
        /// Returns a unit normal to the bowl.
        fn normal(&self, u: f32, v: f32) -> [f32; 3] {
            let n = [-12.0 * u, -12.0 * v, 1.0];
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            [n[0] / len, n[1] / len, n[2] / len]
        }
    }

    /// Counts how many triangles reference each undirected index edge.
    fn edge_counts(mesh: &TriangleMesh) -> HashMap<(u32, u32), u32> {
        let mut counts = HashMap::new();
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

    /// Returns `true` when `(u, v)` lies on the global domain boundary.
    fn on_domain_boundary(uv: [f32; 2]) -> bool {
        uv[0].abs() < 1e-5
            || (uv[0] - 1.0).abs() < 1e-5
            || uv[1].abs() < 1e-5
            || (uv[1] - 1.0).abs() < 1e-5
    }

    #[test]
    fn config_is_clamped() {
        let grid = PatchGrid::new(0, 0, -1.0, 999);
        assert_eq!(grid.cols(), 1);
        assert_eq!(grid.rows(), 1);
        assert_eq!(grid.max_factor(), MAX_FACTOR);
        assert!(grid.tolerance(0, 0).expect("in range") > 0.0);
        assert_eq!(grid.tolerance(5, 5), None);
    }

    #[test]
    fn with_tolerance_overrides_one_cell() {
        let grid = PatchGrid::new(2, 2, 0.1, 16).with_tolerance(1, 0, 0.001);
        assert!((grid.tolerance(1, 0).expect("in range") - 0.001).abs() < 1e-9);
        assert!((grid.tolerance(0, 0).expect("in range") - 0.1).abs() < 1e-9);
    }

    #[test]
    fn flat_plane_uses_minimum_factors() {
        let grid = PatchGrid::new(3, 3, 0.01, 16);
        let f = grid.edge_factor(&Plane, [0.0, 0.0], [1.0, 0.0], 0.01);
        assert_eq!(f, 1);
    }

    #[test]
    fn curved_surface_refines() {
        let grid = PatchGrid::new(1, 1, 0.02, 32);
        let f = grid.edge_factor(&Bowl, [0.0, 0.0], [1.0, 0.0], 0.02);
        assert!(f > 1, "steep bowl edge should need more than one segment");
    }

    #[test]
    fn finer_cell_adds_triangles() {
        let uniform = PatchGrid::new(2, 2, 0.1, 48);
        let refined = PatchGrid::new(2, 2, 0.1, 48).with_tolerance(0, 0, 0.005);
        let a = uniform.tessellate(&Bowl).expect("uniform");
        let b = refined.tessellate(&Bowl).expect("refined");
        assert!(
            b.triangle_count() > a.triangle_count(),
            "a tighter cell tolerance must add triangles: {} vs {}",
            b.triangle_count(),
            a.triangle_count()
        );
    }

    #[test]
    fn heterogeneous_grid_is_index_watertight() {
        // Deliberately mismatch neighbouring tolerances: without the shared-edge
        // max resolution this would leave T-junctions.
        let mesh = PatchGrid::new(3, 2, 0.08, 32)
            .with_tolerance(0, 0, 0.004)
            .with_tolerance(2, 1, 0.01)
            .tessellate(&Bowl)
            .expect("tessellation");
        let positions = mesh.positions();
        for (&(a, b), &count) in &edge_counts(&mesh) {
            if count == 1 {
                let pa = positions[a as usize];
                let pb = positions[b as usize];
                assert!(
                    on_domain_boundary([pa[0], pa[1]]) && on_domain_boundary([pb[0], pb[1]]),
                    "interior edge referenced only once: {a}-{b}"
                );
            } else {
                assert_eq!(count, 2, "manifold edge {a}-{b} referenced {count} times");
            }
        }
    }

    #[test]
    fn shared_edge_matches_both_sides() {
        // The right edge of cell (0,0) and the left edge of cell (1,0) share the
        // line u = 1/2; both must resolve to the same factor (the finer demand).
        let grid = PatchGrid::new(2, 1, 0.05, 48).with_tolerance(0, 0, 0.002);
        let left_cell = grid.resolved_outer(&Bowl, 0, 0);
        let right_cell = grid.resolved_outer(&Bowl, 1, 0);
        // index 1 = right edge of (0,0); index 3 = left edge of (1,0).
        assert_eq!(left_cell[1], right_cell[3]);
    }

    #[test]
    fn tessellation_is_deterministic() {
        let grid = PatchGrid::new(2, 3, 0.03, 24).with_tolerance(1, 1, 0.006);
        let a = grid.tessellate(&Bowl).expect("run a");
        let b = grid.tessellate(&Bowl).expect("run b");
        assert_eq!(a, b);
    }

    #[test]
    fn positions_lie_on_the_surface() {
        let mesh = PatchGrid::new(2, 2, 0.02, 24)
            .tessellate(&Bowl)
            .expect("tessellation");
        for p in mesh.positions() {
            let expected_z = 6.0 * (p[0] * p[0] + p[1] * p[1]);
            assert!(
                (p[2] - expected_z).abs() < 1e-4,
                "vertex {p:?} is off the bowl"
            );
        }
    }

    #[test]
    fn uvs_cover_the_global_domain() {
        let mesh = PatchGrid::new(2, 2, 0.03, 24)
            .tessellate(&Bowl)
            .expect("tessellation");
        let uvs = mesh.uvs();
        assert!(!uvs.is_empty());
        let mut min_u = f32::INFINITY;
        let mut max_u = f32::NEG_INFINITY;
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        for uv in uvs {
            min_u = min_u.min(uv[0]);
            max_u = max_u.max(uv[0]);
            min_v = min_v.min(uv[1]);
            max_v = max_v.max(uv[1]);
        }
        assert!(min_u.abs() < 1e-5 && (max_u - 1.0).abs() < 1e-5);
        assert!(min_v.abs() < 1e-5 && (max_v - 1.0).abs() < 1e-5);
    }

    #[test]
    fn bvh_hits_the_patch() {
        let bvh = PatchGrid::new(2, 2, 0.05, 24)
            .tessellate_bvh(&Plane)
            .expect("bvh");
        let ray = Ray::new([0.53, 0.47, 10.0], [0.0, 0.0, -1.0], 0.0, 100.0);
        assert!(bvh.closest_hit(&ray).is_some(), "ray should strike the plane");
    }
}
