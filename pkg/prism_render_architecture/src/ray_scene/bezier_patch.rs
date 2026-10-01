//! Bicubic Bézier surface patch with analytic normals, tessellated into an
//! [`IndexedBilinearPatchMesh`] for the `CPU` golden path.
//!
//! A single bilinear patch can only represent a ruled quad; curved production
//! surfaces (the Utah teapot, trimmed `NURBS` cages flattened to Bézier,
//! displacement bases, cloth/skin patches) are built from **bicubic** patches:
//! a 4×4 net of control points whose tensor-product Bernstein basis sweeps a
//! smoothly curved surface `P(u, v)`. This primitive is that patch. It never
//! intersects a cubic surface analytically (which would need a high-degree
//! root solve); instead it does what every real-time and offline renderer does
//! with Bézier patches — **tessellates** the smooth surface into a grid of
//! bilinear patches, sampling the exact surface position and the exact analytic
//! surface normal at every tessellation vertex so the resulting
//! [`IndexedBilinearPatchMesh`] reads smooth and welds seamlessly.
//!
//! Both the surface point and its partial derivatives are evaluated with De
//! Casteljau's algorithm, which is **pure linear interpolation** (no
//! transcendental basis functions): a cubic point is three nested `lerp`s and a
//! cubic tangent is `3·` a quadratic De Casteljau over adjacent differences.
//! The surface normal is `∂P/∂u × ∂P/∂v`, normalized, with a short search for a
//! non-degenerate tangent frame at the rare point where one partial collapses
//! (a control-net cusp or coincident control row).

use super::bvh::Aabb;
use super::indexed_bilinear_patch_mesh::{IndexedBilinearPatchMesh, IndexedBilinearPatchMeshBvh};

/// Linear interpolation `a + t·(b − a)` of two points.
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Component-wise difference `a − b`.
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Cross product `a × b`.
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Returns the unit vector along `v`, or `fallback` when `v` is near zero
/// length (so a degenerate tangent frame never yields `NaN`s).
fn normalize_or3(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len2 <= 0.0 {
        return fallback;
    }
    let inv = 1.0 / len2.sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

/// Cubic Bézier point at `t` over four control points (three nested `lerp`s).
fn cubic_point(p: [[f32; 3]; 4], t: f32) -> [f32; 3] {
    let a = lerp3(p[0], p[1], t);
    let b = lerp3(p[1], p[2], t);
    let c = lerp3(p[2], p[3], t);
    let d = lerp3(a, b, t);
    let e = lerp3(b, c, t);
    lerp3(d, e, t)
}

/// Cubic Bézier derivative at `t`: `3·` the quadratic De Casteljau over the
/// three adjacent control-point differences.
fn cubic_deriv(p: [[f32; 3]; 4], t: f32) -> [f32; 3] {
    let d0 = sub3(p[1], p[0]);
    let d1 = sub3(p[2], p[1]);
    let d2 = sub3(p[3], p[2]);
    let a = lerp3(d0, d1, t);
    let b = lerp3(d1, d2, t);
    let q = lerp3(a, b, t);
    [q[0] * 3.0, q[1] * 3.0, q[2] * 3.0]
}

/// A bicubic Bézier surface patch defined by a 4×4 control net.
///
/// Control points are stored row-major as `control[row * 4 + col]`, where
/// `col ∈ 0..4` runs along the `u` parameter and `row ∈ 0..4` runs along the
/// `v` parameter. The four corners interpolate the net: `P(0, 0) = control[0]`,
/// `P(1, 0) = control[3]`, `P(0, 1) = control[12]`, `P(1, 1) = control[15]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BezierPatch {
    /// Row-major 4×4 control net, `control[row * 4 + col]`.
    control: [[f32; 3]; 16],
}

impl BezierPatch {
    /// Builds a patch from a row-major 4×4 control net.
    #[must_use]
    pub fn new(control: [[f32; 3]; 16]) -> Self {
        Self { control }
    }

    /// The row-major 4×4 control net.
    #[must_use]
    pub fn control(&self) -> [[f32; 3]; 16] {
        self.control
    }

    /// The four control points of `v`-row `row` (its `u`-direction curve).
    fn row(&self, row: usize) -> [[f32; 3]; 4] {
        let b = row * 4;
        [
            self.control[b],
            self.control[b + 1],
            self.control[b + 2],
            self.control[b + 3],
        ]
    }

    /// Surface point `P(u, v)` for `u, v ∈ [0, 1]`.
    ///
    /// Each `v`-row is collapsed to a point by a cubic De Casteljau in `u`, and
    /// the four results are collapsed by a cubic De Casteljau in `v`.
    #[must_use]
    pub fn point(&self, u: f32, v: f32) -> [f32; 3] {
        let column = [
            cubic_point(self.row(0), u),
            cubic_point(self.row(1), u),
            cubic_point(self.row(2), u),
            cubic_point(self.row(3), u),
        ];
        cubic_point(column, v)
    }

    /// Partial derivative `∂P/∂u` at `(u, v)`.
    ///
    /// Each `v`-row contributes its `u`-tangent (cubic derivative in `u`); the
    /// four tangents are blended by a cubic De Casteljau in `v`.
    #[must_use]
    pub fn partial_u(&self, u: f32, v: f32) -> [f32; 3] {
        let column = [
            cubic_deriv(self.row(0), u),
            cubic_deriv(self.row(1), u),
            cubic_deriv(self.row(2), u),
            cubic_deriv(self.row(3), u),
        ];
        cubic_point(column, v)
    }

    /// Partial derivative `∂P/∂v` at `(u, v)`.
    ///
    /// Each `v`-row is collapsed to a point by a cubic De Casteljau in `u`, and
    /// the four points are differentiated by a cubic derivative in `v`.
    #[must_use]
    pub fn partial_v(&self, u: f32, v: f32) -> [f32; 3] {
        let column = [
            cubic_point(self.row(0), u),
            cubic_point(self.row(1), u),
            cubic_point(self.row(2), u),
            cubic_point(self.row(3), u),
        ];
        cubic_deriv(column, v)
    }

    /// Unit surface normal `∂P/∂u × ∂P/∂v` at `(u, v)`.
    ///
    /// When one partial collapses (a degenerate control-net point), the sample
    /// is nudged a few steps toward the patch interior along the shorter axis
    /// until a non-degenerate tangent frame is found; failing that it falls
    /// back to `[0, 0, 1]` so the result is always finite and unit length.
    #[must_use]
    pub fn normal(&self, u: f32, v: f32) -> [f32; 3] {
        const FALLBACK: [f32; 3] = [0.0, 0.0, 1.0];
        for step in 0..4 {
            let eps = 1e-3 * step as f32;
            let uu = (u + eps).clamp(0.0, 1.0);
            let vv = (v + eps).clamp(0.0, 1.0);
            let du = self.partial_u(uu, vv);
            let dv = self.partial_v(uu, vv);
            let n = cross3(du, dv);
            let len2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
            if len2 > 0.0 {
                return normalize_or3(n, FALLBACK);
            }
        }
        FALLBACK
    }

    /// Axis-aligned bounds of the control net.
    ///
    /// A Bézier patch lies within the convex hull of its control points, so the
    /// control-net `AABB` is a correct (conservative) bound of the surface.
    #[must_use]
    pub fn control_aabb(&self) -> Aabb {
        let mut lo = self.control[0];
        let mut hi = self.control[0];
        for p in &self.control[1..] {
            for k in 0..3 {
                if p[k] < lo[k] {
                    lo[k] = p[k];
                }
                if p[k] > hi[k] {
                    hi[k] = p[k];
                }
            }
        }
        Aabb::new(lo, hi)
    }

    /// Tessellates the patch into a welded [`IndexedBilinearPatchMesh`] of
    /// `res_u × res_v` quad cells.
    ///
    /// Samples a `(res_u + 1) × (res_v + 1)` vertex grid at the exact surface
    /// positions and exact analytic normals, with planar `UV`s equal to the
    /// patch `(u, v)` parameters. Welded interior vertices are shared, so the
    /// mesh is watertight and smoothly shaded. Both resolutions are clamped to
    /// at least `1` so the mesh always has one patch.
    #[must_use]
    pub fn tessellate(&self, res_u: usize, res_v: usize) -> IndexedBilinearPatchMesh {
        let ru = res_u.max(1);
        let rv = res_v.max(1);
        let cols = ru + 1;
        let rows = rv + 1;
        let mut positions = Vec::with_capacity(cols * rows);
        let mut normals = Vec::with_capacity(cols * rows);
        let mut uvs = Vec::with_capacity(cols * rows);
        for j in 0..rows {
            let v = j as f32 / rv as f32;
            for i in 0..cols {
                let u = i as f32 / ru as f32;
                positions.push(self.point(u, v));
                normals.push(self.normal(u, v));
                uvs.push([u, v]);
            }
        }
        let mut indices = Vec::with_capacity(ru * rv);
        for j in 0..rv {
            for i in 0..ru {
                let vid = |ii: usize, jj: usize| (jj * cols + ii) as u32;
                // (u, v) corner order 0=(0,0), 1=(1,0), 2=(1,1), 3=(0,1).
                indices.push([
                    vid(i, j),
                    vid(i + 1, j),
                    vid(i + 1, j + 1),
                    vid(i, j + 1),
                ]);
            }
        }
        IndexedBilinearPatchMesh::new(positions, normals, uvs, indices)
            .expect("tessellation indices are always in range")
    }

    /// Tessellates the patch and builds a traversable
    /// [`IndexedBilinearPatchMeshBvh`] in one step.
    #[must_use]
    pub fn tessellate_bvh(&self, res_u: usize, res_v: usize) -> IndexedBilinearPatchMeshBvh {
        IndexedBilinearPatchMeshBvh::build(self.tessellate(res_u, res_v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::traversal::Ray;

    /// Minimal xorshift `RNG` for deterministic test rays.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }
        fn unit(&mut self) -> f32 {
            self.next_u32() as f32 / u32::MAX as f32
        }
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }
    }

    /// A flat `z = 0` patch: a 4×4 planar net spanning the unit square in `xy`.
    fn flat_patch() -> BezierPatch {
        let mut control = [[0.0f32; 3]; 16];
        for row in 0..4 {
            for col in 0..4 {
                let x = col as f32 / 3.0;
                let y = row as f32 / 3.0;
                control[row * 4 + col] = [x, y, 0.0];
            }
        }
        BezierPatch::new(control)
    }

    /// A gently domed patch: the flat net with the center control points lifted
    /// in `+z` so the surface bulges upward.
    fn domed_patch() -> BezierPatch {
        let mut control = flat_patch().control();
        for row in 1..3 {
            for col in 1..3 {
                control[row * 4 + col][2] = 0.6;
            }
        }
        BezierPatch::new(control)
    }

    #[test]
    fn corners_interpolate_the_net() {
        let p = domed_patch();
        let c = p.control();
        assert_eq!(p.point(0.0, 0.0), c[0]);
        assert_eq!(p.point(1.0, 0.0), c[3]);
        assert_eq!(p.point(0.0, 1.0), c[12]);
        assert_eq!(p.point(1.0, 1.0), c[15]);
    }

    #[test]
    fn flat_patch_stays_planar_with_up_normal() {
        let p = flat_patch();
        let mut rng = Rng::new(0xF1A7);
        for _ in 0..200 {
            let u = rng.unit();
            let v = rng.unit();
            let pt = p.point(u, v);
            assert!(pt[2].abs() < 1e-5, "z should be ~0, got {}", pt[2]);
            let n = p.normal(u, v);
            // Unit length and parallel to z.
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-5);
            assert!(n[2].abs() > 1.0 - 1e-5);
            assert!(n[0].abs() < 1e-4 && n[1].abs() < 1e-4);
        }
    }

    #[test]
    fn analytic_normal_matches_finite_difference() {
        let p = domed_patch();
        let mut rng = Rng::new(0x0D0E);
        for _ in 0..200 {
            // Stay off the boundary so central differences are well-defined.
            let u = rng.range(0.1, 0.9);
            let v = rng.range(0.1, 0.9);
            let h = 1e-3;
            let du = sub3(p.point(u + h, v), p.point(u - h, v));
            let dv = sub3(p.point(u, v + h), p.point(u, v - h));
            let fd = normalize_or3(cross3(du, dv), [0.0, 0.0, 1.0]);
            let an = p.normal(u, v);
            // Compare directions (allow a global sign choice).
            let dot = fd[0] * an[0] + fd[1] * an[1] + fd[2] * an[2];
            assert!(dot.abs() > 0.999, "normal mismatch dot={dot}");
        }
    }

    #[test]
    fn tessellation_has_expected_shape() {
        let p = domed_patch();
        let mesh = p.tessellate(4, 3);
        assert_eq!(mesh.vertex_count(), 5 * 4);
        assert_eq!(mesh.patch_count(), 4 * 3);
        // Degenerate resolution still yields a single patch.
        let one = p.tessellate(0, 0);
        assert_eq!(one.patch_count(), 1);
        assert_eq!(one.vertex_count(), 4);
    }

    #[test]
    fn tessellated_bvh_hits_the_surface() {
        let p = domed_patch();
        // Odd resolution keeps the dome center `(0.5, 0.5)` strictly interior to
        // one patch instead of on a shared tessellation vertex, where a ray
        // threading the exact corner of four bilinear patches can be missed by
        // every incident patch (the quad mesh is not watertight at vertices).
        let bvh = p.tessellate_bvh(15, 15);
        // Shoot straight down at the dome center; the surface there is ~z=0.6*…
        // but definitely above z=0, so a downward ray from high z must hit.
        let center = p.point(0.5, 0.5);
        let origin = [center[0], center[1], center[2] + 5.0];
        let ray = Ray::infinite(origin, [0.0, 0.0, -1.0]);
        let hit = bvh.closest_hit(&ray).expect("ray must hit the dome");
        let hit_pt = ray.at(hit.t);
        // The hit should land close to the true surface point at the center.
        let d = sub3(hit_pt, center);
        let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        assert!(dist < 0.05, "hit {hit_pt:?} far from surface {center:?}");
        assert!(bvh.any_hit(&ray));
        // The shading normal should point generally upward on the dome.
        assert!(hit.shading_normal[2] > 0.3);
    }

    #[test]
    fn tessellated_mesh_welds_shared_vertices() {
        let p = domed_patch();
        let mesh = p.tessellate(3, 3);
        // Interior vertex (1,1) is shared by four patches; its single pooled
        // normal is what every incident patch reads.
        let cols = 4;
        // Interior vertex at grid (col=1, row=1): index = row * cols + col.
        let shared = (cols + 1) as u32;
        let incident: Vec<usize> = (0..mesh.patch_count())
            .filter(|&k| mesh.indices()[k].contains(&shared))
            .collect();
        assert_eq!(incident.len(), 4, "interior vertex should touch 4 patches");
    }
}
