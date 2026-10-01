//! Bicubic Catmull-Rom surface patch that **interpolates** its control net,
//! converted to Bézier form and tessellated into an
//! [`IndexedBilinearPatchMesh`] for the `CPU` golden path.
//!
//! Where a [`super::bezier_patch::BezierPatch`] *approximates* its control net
//! (the surface passes through the four corners only), a Catmull-Rom patch
//! *interpolates* its interior grid: the surface passes through the inner 2×2
//! control points and uses the outer ring purely to define the entry/exit
//! tangents. This is what artists and solvers usually want when they author a
//! smooth surface *through* a set of sample points — terrain spines, hair/cloth
//! card ribs, trimmed-surface fitting — rather than pulling a cage toward them.
//!
//! A uniform Catmull-Rom span `[c0, c1, c2, c3]` is exactly a cubic Bézier
//! segment over `[c1, c2]` with tangents `(c2 − c0)/2` and `(c3 − c1)/2`, i.e.
//! the Bézier control points are `b0 = c1`, `b1 = c1 + (c2 − c0)/6`,
//! `b2 = c2 − (c3 − c1)/6`, `b3 = c2`. The conversion is **purely linear** (no
//! transcendental basis functions): applying it along `u` then along `v` turns
//! the 4×4 Catmull-Rom net into a 4×4 Bézier net, after which all evaluation,
//! analytic normals and tessellation are delegated to
//! [`super::bezier_patch::BezierPatch`]. The patch is therefore defined over the
//! central cell, with parameters `(u, v) ∈ [0, 1]²` spanning the inner
//! `[c1, c2]` grid in both directions.

use super::bezier_patch::BezierPatch;
use super::bvh::Aabb;
use super::indexed_bilinear_patch_mesh::{IndexedBilinearPatchMesh, IndexedBilinearPatchMeshBvh};

/// Component-wise difference `a − b`.
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Component-wise sum `a + b`.
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scales `a` by the scalar `s`.
fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Converts one uniform Catmull-Rom span `[c0, c1, c2, c3]` into the four cubic
/// Bézier control points of its central segment `[c1, c2]`.
///
/// The Catmull-Rom tangents at `c1` and `c2` are `(c2 − c0)/2` and
/// `(c3 − c1)/2`; dividing by the cubic Bézier factor of `3` gives the inner
/// control points `b1 = c1 + (c2 − c0)/6` and `b2 = c2 − (c3 − c1)/6`, while the
/// endpoints are interpolated (`b0 = c1`, `b3 = c2`). This is a pure linear
/// combination, so it introduces no transcendental functions.
fn span_to_bezier(c0: [f32; 3], c1: [f32; 3], c2: [f32; 3], c3: [f32; 3]) -> [[f32; 3]; 4] {
    let sixth = 1.0 / 6.0;
    let b1 = add3(c1, scale3(sub3(c2, c0), sixth));
    let b2 = sub3(c2, scale3(sub3(c3, c1), sixth));
    [c1, b1, b2, c2]
}

/// A bicubic Catmull-Rom surface patch defined by a 4×4 interpolating control
/// net.
///
/// The net is stored row-major (`control[row * 4 + col]`), with `col` advancing
/// along `u` and `row` advancing along `v`, matching
/// [`super::bezier_patch::BezierPatch`]. The surface interpolates the inner 2×2
/// points (`control[5]`, `control[6]`, `control[9]`, `control[10]`) at its
/// corners; the outer ring only shapes the boundary tangents.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CatmullRomPatch {
    /// The 4×4 Catmull-Rom control net, row-major (`col` along `u`, `row` along
    /// `v`).
    control: [[f32; 3]; 16],
}

impl CatmullRomPatch {
    /// Creates a patch from a 4×4 row-major control net.
    #[must_use]
    pub fn new(control: [[f32; 3]; 16]) -> Self {
        Self { control }
    }

    /// Returns a copy of the 4×4 Catmull-Rom control net.
    #[must_use]
    pub fn control(&self) -> [[f32; 3]; 16] {
        self.control
    }

    /// Converts the Catmull-Rom net into the equivalent bicubic
    /// [`super::bezier_patch::BezierPatch`] over the central cell.
    ///
    /// The tensor-product conversion runs [`span_to_bezier`] along each of the
    /// four `u` rows, then along each of the four `v` columns of the
    /// intermediate net; both passes are linear, so the result reproduces the
    /// Catmull-Rom surface exactly while reusing the Bézier evaluation path.
    #[must_use]
    pub fn to_bezier(&self) -> BezierPatch {
        // Pass 1: convert each row along u into Bézier control points.
        let mut tmp = [[0.0f32; 3]; 16];
        for row in 0..4 {
            let base = row * 4;
            let span = span_to_bezier(
                self.control[base],
                self.control[base + 1],
                self.control[base + 2],
                self.control[base + 3],
            );
            tmp[base] = span[0];
            tmp[base + 1] = span[1];
            tmp[base + 2] = span[2];
            tmp[base + 3] = span[3];
        }
        // Pass 2: convert each column along v into Bézier control points.
        let mut out = [[0.0f32; 3]; 16];
        for col in 0..4 {
            let span = span_to_bezier(
                tmp[col],
                tmp[4 + col],
                tmp[8 + col],
                tmp[12 + col],
            );
            out[col] = span[0];
            out[4 + col] = span[1];
            out[8 + col] = span[2];
            out[12 + col] = span[3];
        }
        BezierPatch::new(out)
    }

    /// Evaluates the surface position at parameters `(u, v) ∈ [0, 1]²`.
    #[must_use]
    pub fn point(&self, u: f32, v: f32) -> [f32; 3] {
        self.to_bezier().point(u, v)
    }

    /// Evaluates the unit analytic surface normal at `(u, v)`.
    #[must_use]
    pub fn normal(&self, u: f32, v: f32) -> [f32; 3] {
        self.to_bezier().normal(u, v)
    }

    /// Returns the axis-aligned bounding box of the 16 control points.
    ///
    /// This is a convenience bound on the control net, not a tight bound on the
    /// surface; traversal uses the tessellated mesh's own `BVH` for exact hits.
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
    /// `res_u × res_v` quad cells by converting to Bézier form first.
    ///
    /// Both resolutions are clamped to at least `1`; see
    /// [`super::bezier_patch::BezierPatch::tessellate`] for the exact sampling
    /// and welding behaviour.
    #[must_use]
    pub fn tessellate(&self, res_u: usize, res_v: usize) -> IndexedBilinearPatchMesh {
        self.to_bezier().tessellate(res_u, res_v)
    }

    /// Tessellates the patch and builds a traversable
    /// [`IndexedBilinearPatchMeshBvh`] in one step.
    #[must_use]
    pub fn tessellate_bvh(&self, res_u: usize, res_v: usize) -> IndexedBilinearPatchMeshBvh {
        self.to_bezier().tessellate_bvh(res_u, res_v)
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

    /// Component-wise difference, for test assertions.
    fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    /// Euclidean length of a 3-vector.
    fn len(a: [f32; 3]) -> f32 {
        (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
    }

    /// A planar `z = 0` net spanning `[-1, 2]²` with unit spacing, so the
    /// central cell covers the unit square `[0, 1]²` in `xy`.
    fn flat_patch() -> CatmullRomPatch {
        let mut control = [[0.0f32; 3]; 16];
        for row in 0..4 {
            for col in 0..4 {
                let x = col as f32 - 1.0;
                let y = row as f32 - 1.0;
                control[row * 4 + col] = [x, y, 0.0];
            }
        }
        CatmullRomPatch::new(control)
    }

    /// The flat net with the inner 2×2 points lifted in `+z` so the surface
    /// bulges upward and interpolates those lifted points at the corners.
    fn domed_patch() -> CatmullRomPatch {
        let mut control = flat_patch().control();
        for row in 1..3 {
            for col in 1..3 {
                control[row * 4 + col][2] = 0.5;
            }
        }
        CatmullRomPatch::new(control)
    }

    #[test]
    fn corners_interpolate_the_inner_grid() {
        // Catmull-Rom passes through the inner 2x2 control points at the
        // corners of the central cell.
        let p = domed_patch();
        let c = p.control();
        let eq = |a: [f32; 3], b: [f32; 3]| len(sub(a, b)) < 1e-5;
        assert!(eq(p.point(0.0, 0.0), c[5]), "corner (0,0) -> inner (1,1)");
        assert!(eq(p.point(1.0, 0.0), c[6]), "corner (1,0) -> inner (1,2)");
        assert!(eq(p.point(0.0, 1.0), c[9]), "corner (0,1) -> inner (2,1)");
        assert!(eq(p.point(1.0, 1.0), c[10]), "corner (1,1) -> inner (2,2)");
    }

    #[test]
    fn flat_patch_stays_planar_with_up_normal() {
        let p = flat_patch();
        let mut rng = Rng::new(0xCA7);
        for _ in 0..100 {
            let u = rng.range(0.0, 1.0);
            let v = rng.range(0.0, 1.0);
            let pt = p.point(u, v);
            assert!(pt[2].abs() < 1e-5, "z should be ~0, got {}", pt[2]);
            let n = p.normal(u, v);
            assert!((len(n) - 1.0).abs() < 1e-5);
            assert!(n[2].abs() > 1.0 - 1e-5);
            assert!(n[0].abs() < 1e-4 && n[1].abs() < 1e-4);
        }
    }

    #[test]
    fn flat_patch_is_affine_in_the_cell() {
        // A uniform planar grid tessellates to the exact unit square; the
        // central cell maps (u, v) linearly to (x, y) because the spacing is
        // uniform.
        let p = flat_patch();
        let mut rng = Rng::new(0x5EED);
        for _ in 0..100 {
            let u = rng.range(0.0, 1.0);
            let v = rng.range(0.0, 1.0);
            let pt = p.point(u, v);
            assert!((pt[0] - u).abs() < 1e-5, "x={} u={u}", pt[0]);
            assert!((pt[1] - v).abs() < 1e-5, "y={} v={v}", pt[1]);
        }
    }

    #[test]
    fn analytic_normal_matches_finite_difference() {
        let p = domed_patch();
        let mut rng = Rng::new(0x0CA7);
        for _ in 0..200 {
            let u = rng.range(0.1, 0.9);
            let v = rng.range(0.1, 0.9);
            let h = 1e-3;
            let du = sub(p.point(u + h, v), p.point(u - h, v));
            let dv = sub(p.point(u, v + h), p.point(u, v - h));
            let cross = [
                du[1] * dv[2] - du[2] * dv[1],
                du[2] * dv[0] - du[0] * dv[2],
                du[0] * dv[1] - du[1] * dv[0],
            ];
            let l = len(cross);
            let fd = if l > 1e-12 {
                [cross[0] / l, cross[1] / l, cross[2] / l]
            } else {
                [0.0, 0.0, 1.0]
            };
            let an = p.normal(u, v);
            let dot = fd[0] * an[0] + fd[1] * an[1] + fd[2] * an[2];
            assert!(dot.abs() > 0.999, "normal mismatch dot={dot}");
        }
    }

    #[test]
    fn to_bezier_reproduces_the_surface() {
        // The converted Bézier patch must evaluate identically to the
        // Catmull-Rom patch everywhere in the cell.
        let p = domed_patch();
        let b = p.to_bezier();
        let mut rng = Rng::new(0xB2E);
        for _ in 0..200 {
            let u = rng.range(0.0, 1.0);
            let v = rng.range(0.0, 1.0);
            assert!(len(sub(p.point(u, v), b.point(u, v))) < 1e-6);
        }
    }

    #[test]
    fn tessellation_has_expected_shape() {
        let p = domed_patch();
        let mesh = p.tessellate(4, 3);
        assert_eq!(mesh.vertex_count(), 5 * 4);
        assert_eq!(mesh.patch_count(), 4 * 3);
        let one = p.tessellate(0, 0);
        assert_eq!(one.patch_count(), 1);
        assert_eq!(one.vertex_count(), 4);
    }

    #[test]
    fn tessellated_bvh_hits_the_surface() {
        let p = domed_patch();
        // Odd resolution keeps the dome center strictly interior to one patch
        // instead of on a shared tessellation vertex (the quad mesh is not
        // watertight at vertices).
        let bvh = p.tessellate_bvh(15, 15);
        let center = p.point(0.5, 0.5);
        let origin = [center[0], center[1], center[2] + 5.0];
        let ray = Ray::infinite(origin, [0.0, 0.0, -1.0]);
        let hit = bvh.closest_hit(&ray).expect("ray must hit the dome");
        let hit_pt = ray.at(hit.t);
        let dist = len(sub(hit_pt, center));
        assert!(dist < 0.05, "hit {hit_pt:?} far from surface {center:?}");
        assert!(bvh.any_hit(&ray));
        assert!(hit.shading_normal[2] > 0.3);
    }

    #[test]
    fn control_aabb_bounds_the_net() {
        let p = domed_patch();
        let bb = p.control_aabb();
        assert_eq!(bb.min, [-1.0, -1.0, 0.0]);
        assert_eq!(bb.max, [2.0, 2.0, 0.5]);
    }
}
