//! View-dependent (silhouette-adaptive) tessellation of a parametric patch.
//!
//! [`super::adaptive_tessellation::AdaptiveTessellation`] drives subdivision
//! from the surface's *intrinsic* curvature. This module instead drives it from
//! the *viewer*: a surface is visually most demanding along its **silhouette**,
//! the locus where the surface normal is perpendicular to the eye direction
//! (`n · v ≈ 0`). There a coarse polygon's straight edge is seen in profile and
//! reads as a faceted outline, whereas front- and back-facing interiors tolerate
//! far coarser meshes. Refining only near the silhouette concentrates triangles
//! exactly where the eye can see the geometric error — the standard
//! view-adaptive level-of-detail strategy for character and terrain patches.
//!
//! For every sub-patch edge the module probes the signed facing term `s = n · v`
//! (with `v` the unit direction from the sample point toward the eye) at a dense
//! set of points. If the sign flips along the edge the silhouette *crosses* it,
//! so the edge is refined to the cap; otherwise the edge factor ramps linearly
//! from `base_factor` up to `max_factor` as the closest `|s|` approaches zero
//! within a configurable band.
//!
//! **Crack-free by construction.** Exactly as in the curvature-adaptive path,
//! every edge factor is a pure function of the *global* edge endpoints and the
//! shared eye position, so two sub-patches touching a shared edge compute the
//! identical factor and emit bit-identical boundary vertices; the merge welds
//! them by position bits. Interior density is the maximum of the four edge
//! factors and two mid-cross probes so a silhouette grazing the interior is not
//! under-tessellated.
//!
//! All math is multiply/divide/compare, `floor`, and a single `sqrt` per
//! view-direction normalization — no transcendental calls — matching the
//! project float policy and the integer-spacing `GPU` tessellator.

use super::patch_tessellation::PatchTessellation;
use super::triangle_mesh::{TriangleMesh, TriangleMeshBvh, TriangleMeshError};
use crate::ray_scene::displaced_surface::ParametricSurface;
use std::collections::HashMap;

/// Dense probe count used along an edge (or mid-cross line) when measuring the
/// facing term. Higher values catch a silhouette that only grazes the edge.
const PROBE_SAMPLES: u32 = 32;

/// A view-dependent tessellation recipe for a parametric patch.
///
/// Build with [`SilhouetteTessellation::new`], then call
/// [`SilhouetteTessellation::tessellate`] against any [`ParametricSurface`].
/// The domain is split into `patches_u × patches_v` sub-patches; each sub-patch
/// edge is tessellated between `base_factor` (clearly front/back facing) and
/// `max_factor` (on the silhouette), ramping within the `band` of `|n · v|`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SilhouetteTessellation {
    /// Number of sub-patch columns across the `u` domain (clamped to `>= 1`).
    patches_u: u32,
    /// Number of sub-patch rows across the `v` domain (clamped to `>= 1`).
    patches_v: u32,
    /// Eye (camera) position; the facing term uses the unit direction from each
    /// sample toward this point, so perspective foreshortening is respected.
    eye: [f32; 3],
    /// Segment count for edges clearly off the silhouette (`|n · v| >= band`).
    base_factor: u32,
    /// Segment cap for edges on or crossing the silhouette.
    max_factor: u32,
    /// Facing-term band in `(0, 1]`: refinement ramps from `base_factor` toward
    /// `max_factor` as the closest `|n · v|` falls from `band` to `0`.
    band: f32,
}

impl SilhouetteTessellation {
    /// Builds a recipe.
    ///
    /// `patches_u`/`patches_v` are floored at `1`; `base_factor` is clamped into
    /// `1..=max_factor`; `max_factor` is clamped into
    /// `1..=`[`MAX_FACTOR`](super::patch_tessellation::MAX_FACTOR); and `band`
    /// is clamped into `(epsilon, 1]`.
    pub fn new(
        patches_u: u32,
        patches_v: u32,
        eye: [f32; 3],
        base_factor: u32,
        max_factor: u32,
        band: f32,
    ) -> Self {
        let max_factor = max_factor.clamp(1, super::patch_tessellation::MAX_FACTOR);
        Self {
            patches_u: patches_u.max(1),
            patches_v: patches_v.max(1),
            eye,
            base_factor: base_factor.clamp(1, max_factor),
            max_factor,
            band: band.clamp(1e-4, 1.0),
        }
    }

    /// Returns the clamped sub-patch column count.
    #[must_use]
    pub fn patches_u(&self) -> u32 {
        self.patches_u
    }

    /// Returns the clamped sub-patch row count.
    #[must_use]
    pub fn patches_v(&self) -> u32 {
        self.patches_v
    }

    /// Returns the eye position driving the facing term.
    #[must_use]
    pub fn eye(&self) -> [f32; 3] {
        self.eye
    }

    /// Returns the clamped off-silhouette base factor.
    #[must_use]
    pub fn base_factor(&self) -> u32 {
        self.base_factor
    }

    /// Returns the clamped silhouette segment cap.
    #[must_use]
    pub fn max_factor(&self) -> u32 {
        self.max_factor
    }

    /// Returns the clamped facing-term band.
    #[must_use]
    pub fn band(&self) -> f32 {
        self.band
    }

    /// View-adaptively tessellates `surface` into one watertight
    /// [`TriangleMesh`].
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

    /// View-adaptively tessellates `surface` and builds a [`TriangleMeshBvh`].
    ///
    /// # Errors
    ///
    /// Propagates any [`TriangleMeshError`] from
    /// [`SilhouetteTessellation::tessellate`].
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
    /// its four edge factors and two mid-cross probes, so a silhouette that only
    /// grazes the interior still refines it.
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

    /// Returns the segment count for the global edge `a -> b` from how close it
    /// comes to the silhouette.
    ///
    /// A sign change in the facing term `n · v` along the edge means the
    /// silhouette crosses it, forcing the cap. Otherwise the factor ramps
    /// linearly from `base_factor` to `max_factor` as the smallest `|n · v|`
    /// falls from `band` to `0`.
    fn edge_factor<S: ParametricSurface>(&self, surface: &S, a: [f32; 2], b: [f32; 2]) -> u32 {
        let mut min_abs = f32::INFINITY;
        let mut saw_pos = false;
        let mut saw_neg = false;
        for d in 0..=PROBE_SAMPLES {
            let t = d as f32 / PROBE_SAMPLES as f32;
            let uv = lerp2(a, b, t);
            let s = self.facing_term(surface, uv);
            let abs_s = s.abs();
            if abs_s < min_abs {
                min_abs = abs_s;
            }
            // A hard zero counts as both sides so a tangent edge is treated as a
            // crossing (maximum refinement).
            if s >= 0.0 {
                saw_pos = true;
            }
            if s <= 0.0 {
                saw_neg = true;
            }
        }

        if saw_pos && saw_neg {
            return self.max_factor;
        }
        if min_abs >= self.band {
            return self.base_factor;
        }
        // Linear ramp: fraction 0 at |s| == band, 1 at |s| == 0.
        let frac = (self.band - min_abs) / self.band;
        let span = (self.max_factor - self.base_factor) as f32;
        let factor = self.base_factor as f32 + span * frac;
        // Round to nearest and clamp into the valid band.
        let rounded = (factor + 0.5).floor() as u32;
        rounded.clamp(self.base_factor, self.max_factor)
    }

    /// Signed facing term `n · v` at global parameter `uv`, with `v` the unit
    /// direction from the sample point toward the eye. Degenerate normals or a
    /// zero-length view vector collapse the term to `1` (treated as clearly
    /// facing, never a false silhouette).
    fn facing_term<S: ParametricSurface>(&self, surface: &S, uv: [f32; 2]) -> f32 {
        let p = surface.point(uv[0], uv[1]);
        let n = surface.normal(uv[0], uv[1]);
        let to_eye = sub(self.eye, p);
        let len2 = dot(to_eye, to_eye);
        if len2 <= 1.0e-24 || dot(n, n) <= 1.0e-24 {
            return 1.0;
        }
        let inv = 1.0 / len2.sqrt();
        let v = [to_eye[0] * inv, to_eye[1] * inv, to_eye[2] * inv];
        dot(n, v)
    }
}

/// Linearly interpolates between two `(u, v)` parameters.
fn lerp2(a: [f32; 2], b: [f32; 2], t: f32) -> [f32; 2] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
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
/// The mapping is `global = (base_index + local) / count` so neighbouring
/// sub-patches evaluate a shared edge at bit-identical global parameters (and
/// therefore bit-identical positions), which is what makes the merged mesh
/// watertight.
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

    /// A flat unit quad in the `z = 0` plane spanning `[0, 1]²`, normal `+Z`.
    struct Plane;

    impl ParametricSurface for Plane {
        fn point(&self, u: f32, v: f32) -> [f32; 3] {
            [u, v, 0.0]
        }
        fn normal(&self, _u: f32, _v: f32) -> [f32; 3] {
            [0.0, 0.0, 1.0]
        }
    }

    /// A quadratic bowl `z = (u-0.5)² + (v-0.5)²` over `[0, 1]²`, whose normal
    /// sweeps through a wide range of directions so a side view sees a
    /// silhouette somewhere across the patch.
    struct Bowl;

    impl ParametricSurface for Bowl {
        fn point(&self, u: f32, v: f32) -> [f32; 3] {
            let x = u - 0.5;
            let y = v - 0.5;
            [u, v, x * x + y * y]
        }
        fn normal(&self, u: f32, v: f32) -> [f32; 3] {
            // dz/du = 2(u-0.5), dz/dv = 2(v-0.5); normal ∝ (-dz/du, -dz/dv, 1).
            let nx = -2.0 * (u - 0.5);
            let ny = -2.0 * (v - 0.5);
            let nz = 1.0;
            let inv = 1.0 / (nx * nx + ny * ny + nz * nz).sqrt();
            [nx * inv, ny * inv, nz * inv]
        }
    }

    #[test]
    fn clamps_degenerate_parameters() {
        let t = SilhouetteTessellation::new(0, 0, [0.0, 0.0, 5.0], 0, 100, 0.0);
        assert_eq!(t.patches_u(), 1);
        assert_eq!(t.patches_v(), 1);
        assert_eq!(t.max_factor(), super::super::patch_tessellation::MAX_FACTOR);
        assert_eq!(t.base_factor(), 1);
        assert!(t.band() > 0.0 && t.band() <= 1.0);
    }

    #[test]
    fn plane_facing_eye_uses_base_factor() {
        // Eye straight above the plane: n·v == 1 everywhere, far outside the
        // band, so every edge takes the base factor (coarse).
        let t = SilhouetteTessellation::new(2, 2, [0.5, 0.5, 10.0], 1, 16, 0.3);
        let base = t.patch_outer_factors(&Plane, 0, 0);
        assert_eq!(base, [1, 1, 1, 1], "front-facing plane must stay coarse");
    }

    #[test]
    fn plane_viewed_edge_on_is_all_silhouette() {
        // Eye in the plane (far +X, same z): n·v == 0 everywhere → silhouette,
        // so every edge is capped at max_factor.
        let t = SilhouetteTessellation::new(2, 2, [100.0, 0.5, 0.0], 1, 12, 0.3);
        let f = t.patch_outer_factors(&Plane, 0, 0);
        assert_eq!(f, [12, 12, 12, 12], "edge-on plane must refine to the cap");
    }

    #[test]
    fn silhouette_edges_beat_front_facing_edges() {
        // A low, wide eye sees the bowl's rim in profile: edges that cross the
        // silhouette must get more segments than the base factor.
        let t = SilhouetteTessellation::new(4, 4, [0.5, 8.0, 0.5], 1, 32, 0.4);
        let mut saw_refined = false;
        for row in 0..4 {
            for col in 0..4 {
                let f = t.patch_outer_factors(&Bowl, col, row);
                if f.iter().any(|&x| x > 1) {
                    saw_refined = true;
                }
            }
        }
        assert!(saw_refined, "some bowl edge should sit near the silhouette");
    }

    #[test]
    fn tessellation_is_watertight_and_hittable() {
        // The welded mesh must be a valid, non-empty mesh that a ray can hit.
        let t = SilhouetteTessellation::new(3, 3, [0.5, 8.0, 0.5], 2, 16, 0.4);
        let bvh = t.tessellate_bvh(&Bowl).expect("tessellate");
        assert!(!bvh.is_empty());
        // Shoot straight down at an off-seam parameter; the bowl surface there
        // is z = (0.53-0.5)² + (0.47-0.5)² ≈ 0.0018, so the ray hits near z≈0.
        let ray = Ray::infinite([0.53, 0.47, 10.0], [0.0, 0.0, -1.0]);
        let hit = bvh.closest_hit(&ray).expect("ray should hit the bowl");
        assert!(hit.position[2] < 0.1 && hit.position[2] > -0.1);
    }

    #[test]
    fn shared_edge_factors_match_across_neighbours() {
        // The right edge of sub-patch (0,0) is the left edge of (1,0): both must
        // compute the identical factor so the seam is crack-free.
        let t = SilhouetteTessellation::new(3, 1, [0.5, 8.0, 0.5], 1, 24, 0.4);
        let left_patch = t.patch_outer_factors(&Bowl, 0, 0); // [bottom,right,top,left]
        let right_patch = t.patch_outer_factors(&Bowl, 1, 0);
        assert_eq!(left_patch[1], right_patch[3], "shared seam factors must agree");
    }

    #[test]
    fn interior_factor_is_at_least_the_max_edge() {
        let t = SilhouetteTessellation::new(2, 2, [0.5, 8.0, 0.5], 1, 20, 0.4);
        let outer = t.patch_outer_factors(&Bowl, 0, 0);
        let inner = t.patch_inner_factor(&Bowl, 0, 0, outer);
        assert!(inner >= *outer.iter().max().unwrap());
    }

    #[test]
    fn narrower_band_refines_fewer_edges() {
        // Shrinking the band means only edges very close to the silhouette
        // refine, so the total segment budget cannot grow.
        let wide = SilhouetteTessellation::new(4, 4, [0.5, 8.0, 0.5], 1, 32, 0.6);
        let narrow = SilhouetteTessellation::new(4, 4, [0.5, 8.0, 0.5], 1, 32, 0.1);
        let sum = |t: &SilhouetteTessellation| -> u32 {
            let mut s = 0;
            for row in 0..4 {
                for col in 0..4 {
                    s += t.patch_outer_factors(&Bowl, col, row).iter().sum::<u32>();
                }
            }
            s
        };
        assert!(sum(&narrow) <= sum(&wide), "narrower band must not refine more");
    }
}
