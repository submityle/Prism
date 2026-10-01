//! Curvature-adaptive, crack-free tessellation of a parametric patch.
//!
//! [`crate::ray_scene::patch_tessellation::PatchTessellation`] subdivides a
//! single quad at a *given* set of factors. This module decides those factors
//! *automatically* from the surface's shape: it splits the `(u, v)` domain into
//! a grid of sub-patches and, for every sub-patch edge, measures the **chordal
//! deviation** of the true surface from the straight chord and chooses the
//! smallest segment count that keeps that deviation under a tolerance. Flat
//! regions collapse to a single segment; tightly curved regions are refined up
//! to a cap. This is the classic level-of-detail driver behind continuous
//! hardware tessellation and adaptive terrain/displacement rendering.
//!
//! **Crack-free by construction.** A shared edge between two neighbouring
//! sub-patches is described by the *same* global parameter endpoints, so each
//! side computes the *same* segment count from the *same* samples, and the
//! sub-patch mapping emits bit-identical boundary positions `(base + local) /
//! count`. The merge then deduplicates vertices by position bits, so adjacent
//! sub-patches share one welded edge — no T-junctions, no gaps. Interior
//! density per sub-patch is driven by the maximum of its four edge factors and
//! two mid-cross probes, so steep interiors are not under-tessellated.
//!
//! All math is division, comparison, multiply, `floor`, and `sqrt` (for the
//! point-to-segment distance) — no transcendental calls — matching the project
//! float policy and the integer-spacing `GPU` tessellator.

use super::patch_tessellation::PatchTessellation;
use super::triangle_mesh::{TriangleMesh, TriangleMeshBvh, TriangleMeshError};
use crate::ray_scene::displaced_surface::ParametricSurface;
use std::collections::HashMap;

/// Dense probe count used along an edge (or mid-cross line) when measuring
/// chordal deviation. Higher values catch sharper wiggles at more cost.
const PROBE_SAMPLES: u32 = 32;

/// A curvature-adaptive tessellation recipe for a parametric patch.
///
/// Build with [`AdaptiveTessellation::new`], then call
/// [`AdaptiveTessellation::tessellate`] against any
/// [`ParametricSurface`]. The domain is split into `patches_u × patches_v`
/// sub-patches; each sub-patch edge is tessellated just finely enough to keep
/// the surface within `tolerance` of the chord, capped at `max_factor`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveTessellation {
    /// Number of sub-patch columns across the `u` domain (clamped to `>= 1`).
    patches_u: u32,
    /// Number of sub-patch rows across the `v` domain (clamped to `>= 1`).
    patches_v: u32,
    /// Maximum allowed chordal deviation, in object-space units. Smaller
    /// tolerances refine more aggressively.
    tolerance: f32,
    /// Per-edge segment cap (clamped into `1..=`[`patch_tessellation::MAX_FACTOR`](super::patch_tessellation::MAX_FACTOR)).
    max_factor: u32,
}

impl AdaptiveTessellation {
    /// Builds a recipe.
    ///
    /// `patches_u`/`patches_v` are floored at `1`, `tolerance` at a small
    /// positive epsilon, and `max_factor` is clamped into
    /// `1..=`[`MAX_FACTOR`](super::patch_tessellation::MAX_FACTOR).
    pub fn new(patches_u: u32, patches_v: u32, tolerance: f32, max_factor: u32) -> Self {
        Self {
            patches_u: patches_u.max(1),
            patches_v: patches_v.max(1),
            tolerance: tolerance.max(1e-6),
            max_factor: max_factor.clamp(1, super::patch_tessellation::MAX_FACTOR),
        }
    }

    /// Returns the clamped sub-patch column count.
    pub fn patches_u(&self) -> u32 {
        self.patches_u
    }

    /// Returns the clamped sub-patch row count.
    pub fn patches_v(&self) -> u32 {
        self.patches_v
    }

    /// Returns the clamped chordal tolerance.
    pub fn tolerance(&self) -> f32 {
        self.tolerance
    }

    /// Returns the clamped per-edge segment cap.
    pub fn max_factor(&self) -> u32 {
        self.max_factor
    }

    /// Adaptively tessellates `surface` into one watertight [`TriangleMesh`].
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
        for row in 0..self.patches_v {
            for col in 0..self.patches_u {
                let outer = self.patch_outer_factors(surface, col, row);
                let inner = self.patch_inner_factor(surface, col, row, outer);
                let sub = SubPatch {
                    surface,
                    col,
                    row,
                    patches_u: self.patches_u,
                    patches_v: self.patches_v,
                };
                let mesh = PatchTessellation::new(inner, outer).tessellate(&sub)?;
                merge.append(&mesh, col, row, self.patches_u, self.patches_v);
            }
        }
        merge.into_mesh()
    }

    /// Adaptively tessellates `surface` and builds a [`TriangleMeshBvh`].
    ///
    /// # Errors
    ///
    /// Propagates any [`TriangleMeshError`] from
    /// [`AdaptiveTessellation::tessellate`].
    pub fn tessellate_bvh<S: ParametricSurface>(
        &self,
        surface: &S,
    ) -> Result<TriangleMeshBvh, TriangleMeshError> {
        Ok(TriangleMeshBvh::build(self.tessellate(surface)?))
    }

    /// Computes the four outer factors for sub-patch `(col, row)` in
    /// `PatchTessellation` order `[bottom (v low), right (u high), top (v high),
    /// left (u low)]`, in **global** parameter space so both sides of every
    /// shared edge agree.
    fn patch_outer_factors<S: ParametricSurface>(
        &self,
        surface: &S,
        col: u32,
        row: u32,
    ) -> [u32; 4] {
        let u0 = col as f32 / self.patches_u as f32;
        let u1 = (col + 1) as f32 / self.patches_u as f32;
        let v0 = row as f32 / self.patches_v as f32;
        let v1 = (row + 1) as f32 / self.patches_v as f32;
        let bottom = self.edge_factor(surface, [u0, v0], [u1, v0]);
        let right = self.edge_factor(surface, [u1, v0], [u1, v1]);
        let top = self.edge_factor(surface, [u0, v1], [u1, v1]);
        let left = self.edge_factor(surface, [u0, v0], [u0, v1]);
        [bottom, right, top, left]
    }

    /// Picks the interior factor for sub-patch `(col, row)` as the maximum of
    /// its four edge factors and two mid-cross probes, so a steep interior is
    /// not under-tessellated relative to its edges.
    fn patch_inner_factor<S: ParametricSurface>(
        &self,
        surface: &S,
        col: u32,
        row: u32,
        outer: [u32; 4],
    ) -> u32 {
        let u0 = col as f32 / self.patches_u as f32;
        let u1 = (col + 1) as f32 / self.patches_u as f32;
        let v0 = row as f32 / self.patches_v as f32;
        let v1 = (row + 1) as f32 / self.patches_v as f32;
        let um = 0.5 * (u0 + u1);
        let vm = 0.5 * (v0 + v1);
        let mid_u = self.edge_factor(surface, [u0, vm], [u1, vm]);
        let mid_v = self.edge_factor(surface, [um, v0], [um, v1]);
        outer
            .iter()
            .copied()
            .chain([mid_u, mid_v])
            .max()
            .unwrap_or(1)
    }

    /// Returns the smallest segment count in `1..=max_factor` whose equal-chord
    /// approximation of the surface along the global edge `a -> b` keeps every
    /// dense probe within [`AdaptiveTessellation::tolerance`] of its chord.
    fn edge_factor<S: ParametricSurface>(&self, surface: &S, a: [f32; 2], b: [f32; 2]) -> u32 {
        for n in 1..self.max_factor {
            if self.chord_fits(surface, a, b, n) {
                return n;
            }
        }
        self.max_factor
    }

    /// Tests whether an `n`-segment chord of the global edge `a -> b` keeps the
    /// surface within tolerance at every dense probe.
    fn chord_fits<S: ParametricSurface>(
        &self,
        surface: &S,
        a: [f32; 2],
        b: [f32; 2],
        n: u32,
    ) -> bool {
        let tol2 = self.tolerance * self.tolerance;
        for d in 0..=PROBE_SAMPLES {
            let t = d as f32 / PROBE_SAMPLES as f32;
            let p = surface.point(lerp2(a, b, t)[0], lerp2(a, b, t)[1]);
            // Which chord segment contains parameter t?
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
}

/// Linearly interpolates between two `(u, v)` parameters.
fn lerp2(a: [f32; 2], b: [f32; 2], t: f32) -> [f32; 2] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

/// Returns the squared distance from point `p` to segment `c0 c1`.
fn point_segment_distance_sq(p: [f32; 3], c0: [f32; 3], c1: [f32; 3]) -> f32 {
    let d = sub(c1, c0);
    let len2 = dot(d, d);
    let w = sub(p, c0);
    let t = if len2 > 0.0 {
        (dot(w, d) / len2).clamp(0.0, 1.0)
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

/// A single sub-patch viewed as a parametric surface over its own local
/// `(lu, lv) ∈ [0, 1]²`, mapping to the parent surface's global domain.
///
/// The mapping is `global = (base_index + local) / count` so that two
/// neighbouring sub-patches evaluate a shared edge at bit-identical global
/// parameters (and therefore bit-identical positions), which is what makes the
/// merged mesh watertight.
struct SubPatch<'s, S: ParametricSurface> {
    /// The parent surface being sampled.
    surface: &'s S,
    /// This sub-patch's column index in `0..patches_u`.
    col: u32,
    /// This sub-patch's row index in `0..patches_v`.
    row: u32,
    /// Total sub-patch columns, the `u` denominator.
    patches_u: u32,
    /// Total sub-patch rows, the `v` denominator.
    patches_v: u32,
}

impl<S: ParametricSurface> SubPatch<'_, S> {
    /// Maps a local `(lu, lv)` to the parent's global `(u, v)`.
    fn to_global(&self, lu: f32, lv: f32) -> [f32; 2] {
        [
            (self.col as f32 + lu) / self.patches_u as f32,
            (self.row as f32 + lv) / self.patches_v as f32,
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

/// Accumulates sub-patch meshes into one global mesh, welding vertices that
/// share bit-identical positions so sub-patch seams become single edges.
struct Merge {
    /// Global vertex positions.
    positions: Vec<[f32; 3]>,
    /// Global per-vertex normals.
    normals: Vec<[f32; 3]>,
    /// Global per-vertex `(u, v)` coordinates (remapped to the parent domain).
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

    /// Appends `mesh` (a sub-patch tessellation whose uvs are *local*),
    /// remapping its vertices into the welded global pools and its uvs into the
    /// parent domain for sub-patch `(col, row)`.
    fn append(&mut self, mesh: &TriangleMesh, col: u32, row: u32, patches_u: u32, patches_v: u32) {
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
                    (col as f32 + luv[0]) / patches_u as f32,
                    (row as f32 + luv[1]) / patches_v as f32,
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

    /// A flat unit plane `z = 0`; every chord deviation is zero.
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

    /// A tall bowl `z = 4 (u² + v²)`; curvature forces refinement.
    #[derive(Debug)]
    struct Bowl;

    impl ParametricSurface for Bowl {
        /// Lifts `(u, v)` onto the steep bowl.
        fn point(&self, u: f32, v: f32) -> [f32; 3] {
            [u, v, 4.0 * (u * u + v * v)]
        }
        /// Returns a unit normal to the bowl.
        fn normal(&self, u: f32, v: f32) -> [f32; 3] {
            let n = [-8.0 * u, -8.0 * v, 1.0];
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
    fn flat_surface_uses_minimum_factors() {
        let tess = AdaptiveTessellation::new(3, 3, 0.01, 16);
        // Every edge is perfectly flat, so each edge factor is 1.
        let f = tess.edge_factor(&Plane, [0.0, 0.0], [1.0, 0.0]);
        assert_eq!(f, 1);
    }

    #[test]
    fn curved_surface_refines() {
        let tess = AdaptiveTessellation::new(1, 1, 0.02, 32);
        let f = tess.edge_factor(&Bowl, [0.0, 0.0], [1.0, 0.0]);
        assert!(f > 1, "steep bowl edge should need more than one segment");
    }

    #[test]
    fn merged_mesh_is_index_watertight() {
        // Welding makes shared sub-patch edges single edges: interior edges are
        // referenced exactly twice, and every once-referenced edge lies on the
        // true domain boundary.
        let mesh = AdaptiveTessellation::new(3, 2, 0.03, 24)
            .tessellate(&Bowl)
            .unwrap();
        let counts = edge_counts(&mesh);
        for (&(a, b), &count) in &counts {
            assert!(count == 1 || count == 2, "edge used {count} times");
            if count == 1 {
                assert!(
                    on_domain_boundary(mesh.uvs()[a as usize])
                        && on_domain_boundary(mesh.uvs()[b as usize]),
                    "open edge not on the domain boundary"
                );
            }
        }
    }

    #[test]
    fn flat_grid_is_watertight() {
        let mesh = AdaptiveTessellation::new(4, 4, 0.01, 16)
            .tessellate(&Plane)
            .unwrap();
        let counts = edge_counts(&mesh);
        for (&(a, b), &count) in &counts {
            assert!(count == 1 || count == 2);
            if count == 1 {
                assert!(
                    on_domain_boundary(mesh.uvs()[a as usize])
                        && on_domain_boundary(mesh.uvs()[b as usize])
                );
            }
        }
    }

    #[test]
    fn tighter_tolerance_adds_triangles() {
        let coarse = AdaptiveTessellation::new(2, 2, 0.2, 32)
            .tessellate(&Bowl)
            .unwrap();
        let fine = AdaptiveTessellation::new(2, 2, 0.01, 32)
            .tessellate(&Bowl)
            .unwrap();
        assert!(fine.triangle_count() > coarse.triangle_count());
    }

    #[test]
    fn uvs_cover_the_global_domain() {
        let mesh = AdaptiveTessellation::new(2, 2, 0.05, 16)
            .tessellate(&Bowl)
            .unwrap();
        let has_origin = mesh.uvs().iter().any(|uv| uv[0].abs() < 1e-5 && uv[1].abs() < 1e-5);
        let has_far = mesh
            .uvs()
            .iter()
            .any(|uv| (uv[0] - 1.0).abs() < 1e-5 && (uv[1] - 1.0).abs() < 1e-5);
        assert!(has_origin && has_far, "uvs should span the full [0,1]^2 domain");
    }

    #[test]
    fn positions_lie_on_the_surface() {
        let mesh = AdaptiveTessellation::new(3, 3, 0.02, 24)
            .tessellate(&Bowl)
            .unwrap();
        for (pos, uv) in mesh.positions().iter().zip(mesh.uvs().iter()) {
            let expected = [uv[0], uv[1], 4.0 * (uv[0] * uv[0] + uv[1] * uv[1])];
            for c in 0..3 {
                assert!((pos[c] - expected[c]).abs() < 1e-4, "{pos:?} vs {expected:?}");
            }
        }
    }

    #[test]
    fn tessellation_is_deterministic() {
        let tess = AdaptiveTessellation::new(3, 2, 0.03, 24);
        let a = tess.tessellate(&Bowl).unwrap();
        let b = tess.tessellate(&Bowl).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn bvh_hits_the_patch() {
        let bvh = AdaptiveTessellation::new(3, 3, 0.02, 24)
            .tessellate_bvh(&Plane)
            .unwrap();
        let ray = Ray::infinite([0.53, 0.47, 1.0], [0.0, 0.0, -1.0]);
        let hit = bvh.closest_hit(&ray).expect("ray should hit the plane");
        assert!(hit.position[2].abs() < 1e-4);
        assert!((hit.position[0] - 0.53).abs() < 1e-4);
        assert!((hit.position[1] - 0.47).abs() < 1e-4);
    }

    #[test]
    fn config_is_clamped() {
        let tess = AdaptiveTessellation::new(0, 0, -1.0, 999);
        assert_eq!(tess.patches_u(), 1);
        assert_eq!(tess.patches_v(), 1);
        assert!(tess.tolerance() > 0.0);
        assert_eq!(tess.max_factor(), super::super::patch_tessellation::MAX_FACTOR);
    }

    #[test]
    fn point_segment_distance_matches_endpoints() {
        let d = point_segment_distance_sq([0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!((d - 1.0).abs() < 1e-6);
        let d0 = point_segment_distance_sq([-2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!((d0 - 1.0).abs() < 1e-6, "beyond the start clamps to the endpoint");
    }
}
