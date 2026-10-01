//! Rational bicubic Bézier (NURBS) surface patch with analytic normals,
//! tessellated into an [`IndexedBilinearPatchMesh`] for the `CPU` golden path.
//!
//! A polynomial [`super::bezier_patch::BezierPatch`] cannot represent conics
//! exactly — circles, spheres, cylinders and cones are *rational*. A rational
//! Bézier patch attaches a positive weight `w_ij` to each of the 16 control
//! points and evaluates the projective surface
//! `P(u, v) = Σ Bᵢ(u)Bⱼ(v) wᵢⱼ Pᵢⱼ / Σ Bᵢ(u)Bⱼ(v) wᵢⱼ`. This is exactly a
//! single-span `NURBS` patch (uniform clamped knots), so with the right weights
//! it reproduces spherical caps, cylindrical shells and swept conics that the
//! polynomial patch only approximates.
//!
//! Evaluation lifts each control point to homogeneous coordinates
//! `[wx, wy, wz, w]` and runs De Casteljau's algorithm there — **pure linear
//! interpolation, no transcendental basis functions** — then projects back with
//! a single division by the homogeneous weight. Analytic partials use the
//! quotient rule on the homogeneous numerator and denominator:
//! `∂P/∂u ∝ (∂H.xyz/∂u)·H.w − H.xyz·(∂H.w/∂u)`, and the surface normal is the
//! normalized cross product of the two partials (the shared positive
//! `1 / H.w²` factor cancels under normalization, so it is never formed). With
//! all weights equal to `1` the patch is bit-for-bit the polynomial Bézier
//! patch. As with the polynomial patch, the smooth surface is **tessellated**
//! into a welded [`IndexedBilinearPatchMesh`] rather than intersected
//! analytically.

use super::bvh::Aabb;
use super::indexed_bilinear_patch_mesh::{IndexedBilinearPatchMesh, IndexedBilinearPatchMeshBvh};

/// Component-wise linear interpolation `a + (b − a)·t` over a 4-vector.
fn lerp4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

/// Component-wise difference `a − b` over a 4-vector.
fn sub4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

/// Cross product of two 3-vectors.
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Returns the unit vector along `v`, or `fallback` when `v` is near zero.
fn normalize_or3(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-20 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        fallback
    }
}

/// Cubic De Casteljau point at `t` over four homogeneous control points (three
/// nested `lerp`s).
fn cubic_point4(p: [[f32; 4]; 4], t: f32) -> [f32; 4] {
    let a = lerp4(p[0], p[1], t);
    let b = lerp4(p[1], p[2], t);
    let c = lerp4(p[2], p[3], t);
    let d = lerp4(a, b, t);
    let e = lerp4(b, c, t);
    lerp4(d, e, t)
}

/// Cubic De Casteljau derivative at `t`: `3·` the quadratic De Casteljau over
/// the adjacent control-point differences, in homogeneous coordinates.
fn cubic_deriv4(p: [[f32; 4]; 4], t: f32) -> [f32; 4] {
    let d0 = sub4(p[1], p[0]);
    let d1 = sub4(p[2], p[1]);
    let d2 = sub4(p[3], p[2]);
    let a = lerp4(d0, d1, t);
    let b = lerp4(d1, d2, t);
    let q = lerp4(a, b, t);
    [q[0] * 3.0, q[1] * 3.0, q[2] * 3.0, q[3] * 3.0]
}

/// A rational bicubic Bézier (single-span `NURBS`) surface patch defined by a
/// 4×4 control net and matching positive weights.
///
/// The net is stored row-major (`control[row * 4 + col]`), with `col` advancing
/// along `u` and `row` advancing along `v`, matching
/// [`super::bezier_patch::BezierPatch`]. With all weights equal the surface is
/// the polynomial Bézier patch; unequal positive weights bend the surface
/// toward the heavier control points and let it represent conics exactly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RationalBezierPatch {
    /// The 4×4 control net, row-major (`col` along `u`, `row` along `v`).
    control: [[f32; 3]; 16],
    /// Per-control-point weights, row-major and index-aligned with `control`.
    /// Expected to be strictly positive so the surface stays in the convex
    /// hull of the net.
    weights: [f32; 16],
}

impl RationalBezierPatch {
    /// Creates a patch from a 4×4 row-major control net and matching weights.
    #[must_use]
    pub fn new(control: [[f32; 3]; 16], weights: [f32; 16]) -> Self {
        Self { control, weights }
    }

    /// Creates a patch with all weights equal to `1` (equivalent to the
    /// polynomial [`super::bezier_patch::BezierPatch`]).
    #[must_use]
    pub fn unit_weights(control: [[f32; 3]; 16]) -> Self {
        Self {
            control,
            weights: [1.0; 16],
        }
    }

    /// Returns a copy of the 4×4 control net.
    #[must_use]
    pub fn control(&self) -> [[f32; 3]; 16] {
        self.control
    }

    /// Returns a copy of the 16 control-point weights.
    #[must_use]
    pub fn weights(&self) -> [f32; 16] {
        self.weights
    }

    /// Builds the homogeneous control row `[wx, wy, wz, w]` for net row `i`.
    fn homogeneous_row(&self, i: usize) -> [[f32; 4]; 4] {
        let mut row = [[0.0f32; 4]; 4];
        for (col, slot) in row.iter_mut().enumerate() {
            let p = self.control[i * 4 + col];
            let w = self.weights[i * 4 + col];
            *slot = [p[0] * w, p[1] * w, p[2] * w, w];
        }
        row
    }

    /// Evaluates the homogeneous surface point `H(u, v) = [wx, wy, wz, w]`.
    fn homogeneous_point(&self, u: f32, v: f32) -> [f32; 4] {
        let column = [
            cubic_point4(self.homogeneous_row(0), u),
            cubic_point4(self.homogeneous_row(1), u),
            cubic_point4(self.homogeneous_row(2), u),
            cubic_point4(self.homogeneous_row(3), u),
        ];
        cubic_point4(column, v)
    }

    /// Evaluates `∂H/∂u` in homogeneous coordinates.
    fn homogeneous_partial_u(&self, u: f32, v: f32) -> [f32; 4] {
        let column = [
            cubic_deriv4(self.homogeneous_row(0), u),
            cubic_deriv4(self.homogeneous_row(1), u),
            cubic_deriv4(self.homogeneous_row(2), u),
            cubic_deriv4(self.homogeneous_row(3), u),
        ];
        cubic_point4(column, v)
    }

    /// Evaluates `∂H/∂v` in homogeneous coordinates.
    fn homogeneous_partial_v(&self, u: f32, v: f32) -> [f32; 4] {
        let column = [
            cubic_point4(self.homogeneous_row(0), u),
            cubic_point4(self.homogeneous_row(1), u),
            cubic_point4(self.homogeneous_row(2), u),
            cubic_point4(self.homogeneous_row(3), u),
        ];
        cubic_deriv4(column, v)
    }

    /// Evaluates the projected surface position at `(u, v) ∈ [0, 1]²`.
    ///
    /// Runs De Casteljau in homogeneous coordinates and divides by the
    /// homogeneous weight; falls back to the raw numerator if the weight
    /// collapses (only possible with non-positive input weights).
    #[must_use]
    pub fn point(&self, u: f32, v: f32) -> [f32; 3] {
        let h = self.homogeneous_point(u, v);
        if h[3].abs() > 1e-20 {
            [h[0] / h[3], h[1] / h[3], h[2] / h[3]]
        } else {
            [h[0], h[1], h[2]]
        }
    }

    /// The projected numerator of `∂P/∂u`, i.e. `∂P/∂u · H.w²`.
    ///
    /// By the quotient rule `∂P/∂u = (H_u.xyz · H.w − H.xyz · H_u.w) / H.w²`;
    /// the returned vector is the parenthesised numerator, which shares the
    /// positive `1 / H.w²` factor with [`Self::partial_v_numerator`] so the
    /// factor cancels under normalization.
    fn partial_u_numerator(&self, u: f32, v: f32) -> [f32; 3] {
        let h = self.homogeneous_point(u, v);
        let hu = self.homogeneous_partial_u(u, v);
        [
            hu[0] * h[3] - h[0] * hu[3],
            hu[1] * h[3] - h[1] * hu[3],
            hu[2] * h[3] - h[2] * hu[3],
        ]
    }

    /// The projected numerator of `∂P/∂v`; see [`Self::partial_u_numerator`].
    fn partial_v_numerator(&self, u: f32, v: f32) -> [f32; 3] {
        let h = self.homogeneous_point(u, v);
        let hv = self.homogeneous_partial_v(u, v);
        [
            hv[0] * h[3] - h[0] * hv[3],
            hv[1] * h[3] - h[1] * hv[3],
            hv[2] * h[3] - h[2] * hv[3],
        ]
    }

    /// Partial derivative `∂P/∂u` of the projected surface at `(u, v)`.
    #[must_use]
    pub fn partial_u(&self, u: f32, v: f32) -> [f32; 3] {
        let h = self.homogeneous_point(u, v);
        let n = self.partial_u_numerator(u, v);
        let w2 = h[3] * h[3];
        if w2 > 1e-20 {
            [n[0] / w2, n[1] / w2, n[2] / w2]
        } else {
            n
        }
    }

    /// Partial derivative `∂P/∂v` of the projected surface at `(u, v)`.
    #[must_use]
    pub fn partial_v(&self, u: f32, v: f32) -> [f32; 3] {
        let h = self.homogeneous_point(u, v);
        let n = self.partial_v_numerator(u, v);
        let w2 = h[3] * h[3];
        if w2 > 1e-20 {
            [n[0] / w2, n[1] / w2, n[2] / w2]
        } else {
            n
        }
    }

    /// Unit surface normal `∂P/∂u × ∂P/∂v` at `(u, v)`.
    ///
    /// Formed from the quotient-rule numerators, whose shared `1 / H.w²` factor
    /// cancels under normalization. When one partial collapses (a degenerate
    /// control-net point), the sample is nudged a few steps toward the patch
    /// interior; failing that it falls back to `[0, 0, 1]`.
    #[must_use]
    pub fn normal(&self, u: f32, v: f32) -> [f32; 3] {
        const FALLBACK: [f32; 3] = [0.0, 0.0, 1.0];
        for step in 0..4 {
            let eps = 1e-3 * step as f32;
            let uu = (u + eps).clamp(0.0, 1.0);
            let vv = (v + eps).clamp(0.0, 1.0);
            let du = self.partial_u_numerator(uu, vv);
            let dv = self.partial_v_numerator(uu, vv);
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
    /// With positive weights a rational Bézier patch lies within the convex
    /// hull of its control points, so the control-net `AABB` is a correct
    /// (conservative) bound of the surface.
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
    /// Samples a `(res_u + 1) × (res_v + 1)` vertex grid at the exact projected
    /// positions and exact analytic normals, with planar `UV`s equal to the
    /// patch `(u, v)` parameters. Both resolutions are clamped to at least `1`.
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
                indices.push([vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]);
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
    use crate::ray_scene::bezier_patch::BezierPatch;
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

    /// A flat `z = 0` net spanning the unit square in `xy`.
    fn flat_net() -> [[f32; 3]; 16] {
        let mut control = [[0.0f32; 3]; 16];
        for row in 0..4 {
            for col in 0..4 {
                control[row * 4 + col] = [col as f32 / 3.0, row as f32 / 3.0, 0.0];
            }
        }
        control
    }

    /// A domed net: the flat net with the inner 2×2 points lifted in `+z`.
    fn domed_net() -> [[f32; 3]; 16] {
        let mut control = flat_net();
        for row in 1..3 {
            for col in 1..3 {
                control[row * 4 + col][2] = 0.6;
            }
        }
        control
    }

    #[test]
    fn unit_weights_match_polynomial_bezier() {
        // With all weights 1 the rational patch is the polynomial Bézier patch.
        let net = domed_net();
        let rat = RationalBezierPatch::unit_weights(net);
        let poly = BezierPatch::new(net);
        let mut rng = Rng::new(0x4A71);
        for _ in 0..300 {
            let u = rng.range(0.0, 1.0);
            let v = rng.range(0.0, 1.0);
            assert!(len(sub(rat.point(u, v), poly.point(u, v))) < 1e-6, "point");
            // Normals must agree in direction too.
            let nr = rat.normal(u, v);
            let np = poly.normal(u, v);
            let dot = nr[0] * np[0] + nr[1] * np[1] + nr[2] * np[2];
            assert!(dot > 0.999, "normal dot={dot}");
        }
    }

    #[test]
    fn flat_patch_stays_planar_under_any_weights() {
        // Coplanar control points stay coplanar regardless of weights; only the
        // parameterization shifts.
        let net = flat_net();
        let mut weights = [1.0f32; 16];
        for (i, w) in weights.iter_mut().enumerate() {
            *w = 0.3 + (i % 5) as f32 * 0.4;
        }
        let p = RationalBezierPatch::new(net, weights);
        let mut rng = Rng::new(0xF1A7);
        for _ in 0..200 {
            let u = rng.range(0.0, 1.0);
            let v = rng.range(0.0, 1.0);
            let pt = p.point(u, v);
            assert!(pt[2].abs() < 1e-5, "z should be ~0, got {}", pt[2]);
            let n = p.normal(u, v);
            assert!((len(n) - 1.0).abs() < 1e-5);
            assert!(n[2].abs() > 1.0 - 1e-4);
        }
    }

    #[test]
    fn analytic_normal_matches_finite_difference() {
        // Non-uniform weights so the quotient-rule path is exercised.
        let net = domed_net();
        let mut weights = [1.0f32; 16];
        weights[5] = 3.0;
        weights[6] = 2.0;
        weights[9] = 2.0;
        weights[10] = 3.0;
        let p = RationalBezierPatch::new(net, weights);
        let mut rng = Rng::new(0x0A71);
        for _ in 0..200 {
            let u = rng.range(0.1, 0.9);
            let v = rng.range(0.1, 0.9);
            let h = 1e-3;
            let du = sub(p.point(u + h, v), p.point(u - h, v));
            let dv = sub(p.point(u, v + h), p.point(u, v - h));
            let fd = normalize_or3(cross3_f(du, dv), [0.0, 0.0, 1.0]);
            let an = p.normal(u, v);
            let dot = fd[0] * an[0] + fd[1] * an[1] + fd[2] * an[2];
            assert!(dot.abs() > 0.998, "normal mismatch dot={dot}");
        }
    }

    /// Cross product helper for the finite-difference reference.
    fn cross3_f(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    #[test]
    fn heavier_inner_weight_pulls_surface_up() {
        // Lifting the inner weights pulls the dome peak toward the lifted
        // control height (0.6); unit weights stay well below it.
        let net = domed_net();
        let unit = RationalBezierPatch::unit_weights(net);
        let mut heavy = [1.0f32; 16];
        heavy[5] = 20.0;
        heavy[6] = 20.0;
        heavy[9] = 20.0;
        heavy[10] = 20.0;
        let heavy = RationalBezierPatch::new(net, heavy);
        let peak_unit = unit.point(0.5, 0.5)[2];
        let peak_heavy = heavy.point(0.5, 0.5)[2];
        assert!(peak_heavy > peak_unit, "heavy {peak_heavy} <= unit {peak_unit}");
        assert!(peak_heavy < 0.6, "peak must stay below control height");
        assert!(peak_heavy > 0.5, "heavy weights should pull near the top");
    }

    #[test]
    fn exact_quarter_circle_profile() {
        // A rational Bézier reproduces a circular arc exactly where a
        // polynomial one cannot. Build a flat extrusion whose u-profile is the
        // exact cubic rational quarter circle of radius 1, obtained by
        // degree-elevating the exact *quadratic* rational quarter circle
        // (control (1,0),(1,1),(0,1) with middle weight cos45 = √2/2). The
        // elevated cubic has interior control points (1, 2−√2) and (2−√2, 1)
        // with weights (1, (1+√2)/3, (1+√2)/3, 1). Swept trivially in v, every
        // sampled point must lie on the unit circle.
        let s2 = 2.0f32.sqrt();
        let inner = 2.0 - s2; // ≈ 0.585786
        let wmid = (1.0 + s2) / 3.0; // ≈ 0.804738
        // u-profile control points (x, y) from (1,0) to (0,1):
        let prof = [[1.0, 0.0], [1.0, inner], [inner, 1.0], [0.0, 1.0]];
        let wrow = [1.0f32, wmid, wmid, 1.0];
        let mut control = [[0.0f32; 3]; 16];
        let mut weights = [0.0f32; 16];
        for row in 0..4 {
            let z = row as f32 / 3.0;
            for col in 0..4 {
                control[row * 4 + col] = [prof[col][0], prof[col][1], z];
                weights[row * 4 + col] = wrow[col];
            }
        }
        let p = RationalBezierPatch::new(control, weights);
        let mut rng = Rng::new(0xC18C);
        for _ in 0..200 {
            let u = rng.range(0.0, 1.0);
            let v = rng.range(0.0, 1.0);
            let pt = p.point(u, v);
            let r = (pt[0] * pt[0] + pt[1] * pt[1]).sqrt();
            assert!((r - 1.0).abs() < 1e-3, "radius {r} off unit circle at u={u}");
        }
    }

    #[test]
    fn tessellation_has_expected_shape() {
        let p = RationalBezierPatch::unit_weights(domed_net());
        let mesh = p.tessellate(4, 3);
        assert_eq!(mesh.vertex_count(), 5 * 4);
        assert_eq!(mesh.patch_count(), 4 * 3);
        let one = p.tessellate(0, 0);
        assert_eq!(one.patch_count(), 1);
        assert_eq!(one.vertex_count(), 4);
    }

    #[test]
    fn tessellated_bvh_hits_the_surface() {
        let net = domed_net();
        let mut weights = [1.0f32; 16];
        weights[5] = 2.0;
        weights[10] = 2.0;
        let p = RationalBezierPatch::new(net, weights);
        let bvh = p.tessellate_bvh(15, 15);
        let center = p.point(0.5, 0.5);
        let origin = [center[0], center[1], center[2] + 5.0];
        let ray = Ray::infinite(origin, [0.0, 0.0, -1.0]);
        let hit = bvh.closest_hit(&ray).expect("ray must hit the dome");
        let hit_pt = ray.at(hit.t);
        assert!(len(sub(hit_pt, center)) < 0.05, "hit far from surface");
        assert!(bvh.any_hit(&ray));
        assert!(hit.shading_normal[2] > 0.3);
    }
}
