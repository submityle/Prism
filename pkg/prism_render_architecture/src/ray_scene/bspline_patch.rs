//! Bicubic uniform B-spline surface patch that **approximates** its control
//! net, converted to Bézier form and tessellated into an
//! [`IndexedBilinearPatchMesh`] for the `CPU` golden path.
//!
//! This is the approximating counterpart to the interpolating
//! [`super::catmull_rom_patch::CatmullRomPatch`]. A uniform cubic B-spline
//! surface stays inside the convex hull of its 4×4 control net and is globally
//! `C²` continuous when such patches are tiled: the surface does *not* pass
//! through any control point (not even the corners), which is exactly what you
//! want for smooth deformable cages — subdivision-surface limit stand-ins,
//! cloth/skin shells, blobby organic forms — where control points are handles,
//! not samples.
//!
//! A uniform cubic B-spline span over control points `[c0, c1, c2, c3]` is the
//! cubic Bézier segment with control points `b0 = (c0 + 4·c1 + c2)/6`,
//! `b1 = (2·c1 + c2)/3`, `b2 = (c1 + 2·c2)/3`, `b3 = (c1 + 4·c2 + c3)/6`. This
//! conversion is **purely linear** (no transcendental basis functions):
//! applying it along `u` then along `v` turns the 4×4 B-spline net into a 4×4
//! Bézier net, after which all evaluation, analytic normals and tessellation
//! are delegated to [`super::bezier_patch::BezierPatch`]. The patch is defined
//! over the central span with parameters `(u, v) ∈ [0, 1]²`. Because the
//! conversion reproduces affine functions exactly, a uniformly spaced planar
//! net maps `(u, v)` linearly to the surface.

use super::bezier_patch::BezierPatch;
use super::bvh::Aabb;
use super::indexed_bilinear_patch_mesh::{IndexedBilinearPatchMesh, IndexedBilinearPatchMeshBvh};

/// Component-wise sum `a + b`.
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scales `a` by the scalar `s`.
fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Converts one uniform cubic B-spline span `[c0, c1, c2, c3]` into the four
/// cubic Bézier control points of its central segment.
///
/// Uses the standard uniform-knot conversion `b0 = (c0 + 4·c1 + c2)/6`,
/// `b1 = (2·c1 + c2)/3`, `b2 = (c1 + 2·c2)/3`, `b3 = (c1 + 4·c2 + c3)/6`. Each
/// coefficient is a convex combination, so the segment lies in the control
/// hull and the conversion introduces no transcendental functions.
fn span_to_bezier(c0: [f32; 3], c1: [f32; 3], c2: [f32; 3], c3: [f32; 3]) -> [[f32; 3]; 4] {
    let sixth = 1.0 / 6.0;
    let third = 1.0 / 3.0;
    let b0 = scale3(add3(add3(c0, scale3(c1, 4.0)), c2), sixth);
    let b1 = scale3(add3(scale3(c1, 2.0), c2), third);
    let b2 = scale3(add3(c1, scale3(c2, 2.0)), third);
    let b3 = scale3(add3(add3(c1, scale3(c2, 4.0)), c3), sixth);
    [b0, b1, b2, b3]
}

/// A bicubic uniform B-spline surface patch defined by a 4×4 approximating
/// control net.
///
/// The net is stored row-major (`control[row * 4 + col]`), with `col` advancing
/// along `u` and `row` advancing along `v`, matching
/// [`super::bezier_patch::BezierPatch`]. The surface stays inside the convex
/// hull of the net and interpolates no control point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BsplinePatch {
    /// The 4×4 B-spline control net, row-major (`col` along `u`, `row` along
    /// `v`).
    control: [[f32; 3]; 16],
}

impl BsplinePatch {
    /// Creates a patch from a 4×4 row-major control net.
    #[must_use]
    pub fn new(control: [[f32; 3]; 16]) -> Self {
        Self { control }
    }

    /// Returns a copy of the 4×4 B-spline control net.
    #[must_use]
    pub fn control(&self) -> [[f32; 3]; 16] {
        self.control
    }

    /// Converts the B-spline net into the equivalent bicubic
    /// [`super::bezier_patch::BezierPatch`] over the central span.
    ///
    /// The tensor-product conversion runs [`span_to_bezier`] along each of the
    /// four `u` rows, then along each of the four `v` columns of the
    /// intermediate net; both passes are linear, so the result reproduces the
    /// B-spline surface exactly while reusing the Bézier evaluation path.
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
            let span = span_to_bezier(tmp[col], tmp[4 + col], tmp[8 + col], tmp[12 + col]);
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
    /// Because a B-spline surface stays within the convex hull of its net, this
    /// control-point box is also a valid (loose) bound on the surface itself;
    /// traversal still uses the tessellated mesh's own `BVH` for exact hits.
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
    /// central span covers the unit square `[0, 1]²` in `xy`.
    fn flat_patch() -> BsplinePatch {
        let mut control = [[0.0f32; 3]; 16];
        for row in 0..4 {
            for col in 0..4 {
                let x = col as f32 - 1.0;
                let y = row as f32 - 1.0;
                control[row * 4 + col] = [x, y, 0.0];
            }
        }
        BsplinePatch::new(control)
    }

    /// The flat net with the inner 2×2 points lifted in `+z` so the surface
    /// bulges upward (but stays below the control height).
    fn domed_patch() -> BsplinePatch {
        let mut control = flat_patch().control();
        for row in 1..3 {
            for col in 1..3 {
                control[row * 4 + col][2] = 0.5;
            }
        }
        BsplinePatch::new(control)
    }

    #[test]
    fn flat_patch_stays_planar_with_up_normal() {
        let p = flat_patch();
        let mut rng = Rng::new(0xB59);
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
        // Cubic B-splines reproduce affine functions exactly, so a uniform
        // planar net maps (u, v) linearly onto (x, y) in the central span.
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
    fn surface_stays_inside_the_control_hull() {
        // Unlike Catmull-Rom, a B-spline does not interpolate the lifted inner
        // points: the dome peak is positive but strictly below the control
        // height of 0.5.
        let p = domed_patch();
        let peak = p.point(0.5, 0.5)[2];
        assert!(peak > 0.0, "peak should bulge up, got {peak}");
        assert!(peak < 0.5, "peak must stay below control height, got {peak}");
        // No corner interpolates its nearest inner control point.
        let c = p.control();
        assert!(len(sub(p.point(0.0, 0.0), c[5])) > 0.1);
        assert!(len(sub(p.point(1.0, 1.0), c[10])) > 0.1);
    }

    #[test]
    fn analytic_normal_matches_finite_difference() {
        let p = domed_patch();
        let mut rng = Rng::new(0x0B59);
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
