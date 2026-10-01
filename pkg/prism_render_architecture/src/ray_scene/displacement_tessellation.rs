//! Displacement-aware adaptive tessellation for the `CPU` golden path.
//!
//! [`crate::ray_scene::adaptive_tessellation::AdaptiveTessellation`] refines a
//! parametric patch wherever the *base* surface curves. Real AAA displacement
//! pipelines must also refine wherever the **height map** adds high-frequency
//! relief — mortar lines, bark ridges, terrain micro-detail — even on a
//! perfectly flat base. This module closes that gap without reimplementing the
//! adaptive driver: it wraps a base [`ParametricSurface`] plus a sampled
//! [`HeightMap`] in a thin [`DisplacedField`] adapter that itself *is* a
//! [`ParametricSurface`], evaluating
//!
//! ```text
//! P'(u, v) = P(u, v) + scale · h(u, v) · N(u, v)
//! ```
//!
//! and re-deriving the post-displacement shading normal by central differences
//! of that displaced field. Feeding the adapter straight into
//! [`AdaptiveTessellation`] makes the existing chordal-deviation driver measure
//! the *displaced* surface, so curvature **and** displacement both drive
//! refinement through one well-tested code path, and the crack-free welding of
//! the adaptive merge is inherited unchanged.
//!
//! The public [`DisplacementTessellation`] bundles the adaptive recipe with a
//! convenience `tessellate(base, height, scale)` entry point and a BVH variant.
//!
//! All evaluation is linear plus a single `sqrt` for normal normalization, so
//! the module honours the golden-path ban on `f32` transcendental functions.

use super::adaptive_tessellation::AdaptiveTessellation;
use super::displaced_surface::{HeightMap, ParametricSurface};
use super::triangle_mesh::{TriangleMesh, TriangleMeshBvh, TriangleMeshError};

/// Parameter-space step used for the central-difference displaced normal.
///
/// Small enough to localize the stencil to a single sub-patch cell at typical
/// resolutions, large enough to stay well clear of `f32` cancellation.
const NORMAL_EPS: f32 = 1.0e-3;

/// A base [`ParametricSurface`] displaced along its own normal by a scaled
/// [`HeightMap`], exposed itself as a [`ParametricSurface`] so any sampler —
/// including the adaptive tessellator — can treat the bumpy result uniformly.
///
/// Unlike [`super::displaced_surface::DisplacedSurface`], which bakes a fixed
/// uniform grid, this adapter answers point/normal queries at *arbitrary*
/// `(u, v)`, which is exactly what curvature-adaptive refinement needs.
#[derive(Clone, Copy, Debug)]
pub struct DisplacedField<'a, S: ParametricSurface> {
    /// The smooth base surface being displaced.
    base: &'a S,
    /// The scalar height field sampled in `(u, v) ∈ [0, 1]²`.
    height: &'a HeightMap,
    /// Linear multiplier applied to every sampled height.
    scale: f32,
}

impl<'a, S: ParametricSurface> DisplacedField<'a, S> {
    /// Borrows `base` and `height` and fixes the displacement `scale`.
    pub fn new(base: &'a S, height: &'a HeightMap, scale: f32) -> Self {
        Self {
            base,
            height,
            scale,
        }
    }

    /// Returns the borrowed base surface.
    pub fn base(&self) -> &S {
        self.base
    }

    /// Returns the borrowed height map.
    pub fn height_map(&self) -> &HeightMap {
        self.height
    }

    /// Returns the displacement scale.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Evaluates the displaced position `P + scale·h·N` at `(u, v)`.
    fn displaced_point(&self, u: f32, v: f32) -> [f32; 3] {
        let p = self.base.point(u, v);
        let n = self.base.normal(u, v);
        let d = self.scale * self.height.sample(u, v);
        [p[0] + d * n[0], p[1] + d * n[1], p[2] + d * n[2]]
    }
}

impl<S: ParametricSurface> ParametricSurface for DisplacedField<'_, S> {
    /// Returns the displaced position at `(u, v)`.
    fn point(&self, u: f32, v: f32) -> [f32; 3] {
        self.displaced_point(u, v)
    }

    /// Returns the post-displacement unit normal at `(u, v)`.
    ///
    /// The height map destroys the base surface's analytic normal, so the
    /// normal is rebuilt as `normalize(∂P'/∂u × ∂P'/∂v)` using a central
    /// difference whose stencil is clamped to `[0, 1]²` at the border. If the
    /// cross product degenerates (e.g. a flat zero-height patch at a pole), the
    /// base normal is returned; the result is finally oriented into the same
    /// hemisphere as the base normal so lighting stays consistent.
    fn normal(&self, u: f32, v: f32) -> [f32; 3] {
        let u0 = (u - NORMAL_EPS).max(0.0);
        let u1 = (u + NORMAL_EPS).min(1.0);
        let v0 = (v - NORMAL_EPS).max(0.0);
        let v1 = (v + NORMAL_EPS).min(1.0);

        let pu0 = self.displaced_point(u0, v);
        let pu1 = self.displaced_point(u1, v);
        let pv0 = self.displaced_point(u, v0);
        let pv1 = self.displaced_point(u, v1);

        let du = [pu1[0] - pu0[0], pu1[1] - pu0[1], pu1[2] - pu0[2]];
        let dv = [pv1[0] - pv0[0], pv1[1] - pv0[1], pv1[2] - pv0[2]];

        let mut n = [
            du[1] * dv[2] - du[2] * dv[1],
            du[2] * dv[0] - du[0] * dv[2],
            du[0] * dv[1] - du[1] * dv[0],
        ];

        let base_n = self.base.normal(u, v);
        let len2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
        if len2 <= 1.0e-24 {
            return base_n;
        }
        let inv = 1.0 / len2.sqrt();
        n = [n[0] * inv, n[1] * inv, n[2] * inv];

        // Orient into the base normal's hemisphere so shading never flips.
        if n[0] * base_n[0] + n[1] * base_n[1] + n[2] * base_n[2] < 0.0 {
            n = [-n[0], -n[1], -n[2]];
        }
        n
    }
}

/// A displacement-aware adaptive tessellation recipe.
///
/// Thin wrapper around [`AdaptiveTessellation`]: it holds the same sub-patch
/// grid, chordal tolerance, and per-edge segment cap, but tessellates the
/// *displaced* field so the height map's relief drives refinement alongside
/// base curvature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisplacementTessellation {
    /// The underlying adaptive recipe applied to the displaced field.
    adaptive: AdaptiveTessellation,
}

impl DisplacementTessellation {
    /// Builds a recipe from the same parameters as [`AdaptiveTessellation::new`]
    /// (counts floored at `1`, tolerance at a small positive epsilon, cap
    /// clamped into `1..=`[`MAX_FACTOR`](super::patch_tessellation::MAX_FACTOR)).
    pub fn new(patches_u: u32, patches_v: u32, tolerance: f32, max_factor: u32) -> Self {
        Self {
            adaptive: AdaptiveTessellation::new(patches_u, patches_v, tolerance, max_factor),
        }
    }

    /// Builds a recipe directly from an existing [`AdaptiveTessellation`].
    pub fn from_adaptive(adaptive: AdaptiveTessellation) -> Self {
        Self { adaptive }
    }

    /// Returns the wrapped adaptive recipe.
    pub fn adaptive(&self) -> AdaptiveTessellation {
        self.adaptive
    }

    /// Adaptively tessellates `base` displaced by `height · scale` into one
    /// watertight [`TriangleMesh`].
    ///
    /// # Errors
    ///
    /// Propagates [`TriangleMeshError`] from the underlying adaptive
    /// tessellation; by construction the generated pools are valid, so this
    /// does not fail in practice.
    pub fn tessellate<S: ParametricSurface>(
        &self,
        base: &S,
        height: &HeightMap,
        scale: f32,
    ) -> Result<TriangleMesh, TriangleMeshError> {
        let field = DisplacedField::new(base, height, scale);
        self.adaptive.tessellate(&field)
    }

    /// Adaptively tessellates the displaced surface and builds a
    /// [`TriangleMeshBvh`].
    ///
    /// # Errors
    ///
    /// Propagates any [`TriangleMeshError`] from
    /// [`DisplacementTessellation::tessellate`].
    pub fn tessellate_bvh<S: ParametricSurface>(
        &self,
        base: &S,
        height: &HeightMap,
        scale: f32,
    ) -> Result<TriangleMeshBvh, TriangleMeshError> {
        Ok(TriangleMeshBvh::build(self.tessellate(base, height, scale)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::traversal::Ray;
    use std::collections::HashMap;

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

    /// A flat height map of all zeros at the requested resolution.
    fn flat_map(w: usize, h: usize) -> HeightMap {
        HeightMap::new(vec![0.0; w * h], w, h).expect("valid flat map")
    }

    /// A checkerboard-ish ridged height map: alternating high/low texels so the
    /// displaced surface gains sharp relief a flat base never would.
    fn ridged_map(w: usize, h: usize) -> HeightMap {
        let mut data = vec![0.0; w * h];
        for r in 0..h {
            for c in 0..w {
                data[r * w + c] = if (r + c) % 2 == 0 { 1.0 } else { -1.0 };
            }
        }
        HeightMap::new(data, w, h).expect("valid ridged map")
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
    fn displaced_point_matches_formula() {
        let height = ridged_map(5, 5);
        let field = DisplacedField::new(&Plane, &height, 0.25);
        let p = field.point(0.3, 0.7);
        let expected_z = 0.25 * height.sample(0.3, 0.7);
        assert!((p[0] - 0.3).abs() < 1e-6);
        assert!((p[1] - 0.7).abs() < 1e-6);
        assert!((p[2] - expected_z).abs() < 1e-6);
    }

    #[test]
    fn zero_scale_leaves_base_untouched() {
        let height = ridged_map(8, 8);
        let field = DisplacedField::new(&Plane, &height, 0.0);
        let p = field.point(0.42, 0.58);
        assert!((p[0] - 0.42).abs() < 1e-6);
        assert!((p[1] - 0.58).abs() < 1e-6);
        assert!(p[2].abs() < 1e-6);
    }

    #[test]
    fn displaced_normal_is_unit_and_outward() {
        let height = ridged_map(6, 6);
        let field = DisplacedField::new(&Plane, &height, 0.3);
        let n = field.normal(0.53, 0.47);
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-5, "normal must be unit, got {len}");
        // Same hemisphere as the flat base's +z normal.
        assert!(n[2] > 0.0, "displaced normal should point up, got {n:?}");
    }

    #[test]
    fn flat_map_leaves_normal_as_base() {
        let height = flat_map(4, 4);
        let field = DisplacedField::new(&Plane, &height, 1.0);
        let n = field.normal(0.5, 0.5);
        assert!((n[0]).abs() < 1e-6);
        assert!((n[1]).abs() < 1e-6);
        assert!((n[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn flat_base_flat_map_uses_minimum_factors() {
        let height = flat_map(4, 4);
        let tess = DisplacementTessellation::new(3, 3, 0.01, 16);
        let mesh = tess
            .tessellate(&Plane, &height, 1.0)
            .expect("flat tessellation");
        // A flat plane under a zero height map carries no relief, so the
        // displaced tessellation must collapse to the plain adaptive
        // tessellation of the base plane at the same recipe.
        let base = AdaptiveTessellation::new(3, 3, 0.01, 16)
            .tessellate(&Plane)
            .expect("base adaptive tessellation");
        assert_eq!(mesh.triangle_count(), base.triangle_count());
    }

    #[test]
    fn displacement_refines_a_flat_base() {
        let flat = flat_map(16, 16);
        let ridged = ridged_map(16, 16);
        let tess = DisplacementTessellation::new(2, 2, 0.02, 32);
        let smooth = tess
            .tessellate(&Plane, &flat, 1.0)
            .expect("flat tessellation");
        let bumpy = tess
            .tessellate(&Plane, &ridged, 1.0)
            .expect("ridged tessellation");
        assert!(
            bumpy.triangle_count() > smooth.triangle_count(),
            "relief should force more triangles: {} vs {}",
            bumpy.triangle_count(),
            smooth.triangle_count()
        );
    }

    #[test]
    fn larger_scale_adds_triangles() {
        let ridged = ridged_map(16, 16);
        let tess = DisplacementTessellation::new(2, 2, 0.05, 48);
        let gentle = tess
            .tessellate(&Plane, &ridged, 0.1)
            .expect("gentle tessellation");
        let steep = tess
            .tessellate(&Plane, &ridged, 1.0)
            .expect("steep tessellation");
        assert!(
            steep.triangle_count() >= gentle.triangle_count(),
            "a larger displacement scale must not reduce refinement: {} vs {}",
            steep.triangle_count(),
            gentle.triangle_count()
        );
        assert!(steep.triangle_count() > gentle.triangle_count());
    }

    #[test]
    fn zero_scale_matches_base_adaptive() {
        let ridged = ridged_map(16, 16);
        let tess = DisplacementTessellation::new(3, 3, 0.02, 32);
        let displaced = tess
            .tessellate(&Plane, &ridged, 0.0)
            .expect("zero-scale tessellation");
        let base = tess
            .adaptive()
            .tessellate(&Plane)
            .expect("base adaptive tessellation");
        // Zero displacement collapses the displaced field back to the base, so
        // the tessellation must be identical to adaptively tessellating it.
        assert_eq!(displaced.triangle_count(), base.triangle_count());
        assert_eq!(displaced.vertex_count(), base.vertex_count());
    }

    #[test]
    fn tessellation_is_deterministic() {
        let ridged = ridged_map(12, 12);
        let tess = DisplacementTessellation::new(2, 3, 0.03, 24);
        let a = tess.tessellate(&Plane, &ridged, 0.5).expect("run a");
        let b = tess.tessellate(&Plane, &ridged, 0.5).expect("run b");
        assert_eq!(a, b);
    }

    #[test]
    fn merged_mesh_is_index_watertight() {
        let ridged = ridged_map(16, 16);
        let mesh = DisplacementTessellation::new(3, 2, 0.03, 24)
            .tessellate(&Plane, &ridged, 0.4)
            .expect("tessellation");
        let positions = mesh.positions();
        for (&(a, b), &count) in &edge_counts(&mesh) {
            if count == 1 {
                // A once-referenced edge must lie on the true domain boundary;
                // the plane maps (x, y) → (u, v) so x,y are the parameters.
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
    fn uvs_cover_the_global_domain() {
        let ridged = ridged_map(16, 16);
        let mesh = DisplacementTessellation::new(2, 2, 0.03, 24)
            .tessellate(&Plane, &ridged, 0.4)
            .expect("tessellation");
        let uvs = mesh.uvs();
        assert!(!uvs.is_empty(), "displaced mesh should carry uvs");
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
    fn bvh_hits_the_displaced_patch() {
        let ridged = ridged_map(8, 8);
        let bvh = DisplacementTessellation::new(2, 2, 0.05, 24)
            .tessellate_bvh(&Plane, &ridged, 0.5)
            .expect("bvh");
        // Shoot straight down at an off-center, off-seam parameter so the ray
        // does not graze a welded vertex or sub-patch seam.
        let ray = Ray::new([0.53, 0.47, 10.0], [0.0, 0.0, -1.0], 0.0, 100.0);
        let hit = bvh.closest_hit(&ray);
        assert!(hit.is_some(), "ray should strike the displaced plane");
    }
}
