//! Catmull-Rom / cardinal / uniform-B-spline round-curve primitive.
//!
//! Hair, fur, foliage, and sketched paths are often authored not as isolated
//! cubic Béziers but as *splines through shared control points*: a Catmull-Rom
//! strand interpolates every control vertex, a cardinal spline relaxes that
//! with a tension knob, and a uniform cubic B-spline approximates its hull with
//! `C²` continuity. All three are cubic polynomials, so a single segment of any
//! of them is exactly representable as a cubic Bézier through four *derived*
//! control points obtained by a fixed `4×4` basis-change (Catmull-Rom/cardinal
//! use a Hermite tangent rule; the uniform B-spline uses its blossom). This
//! module performs that basis change and then **reuses the proven cubic-Bézier
//! round-curve intersector** ([`super::curve::Curve`]) so there is exactly one
//! swept-circle ray test, one refinement scheme, and one `BVH` contract for all
//! curve flavours.
//!
//! Because every segment lowers to a Bézier, the `GPU` upload path is simply the
//! existing [`super::curve_gpu_layout`]: build a [`SplineBvh`], take its
//! [`SplineBvh::curve_bvh`], and pack it with
//! [`super::curve_gpu_layout::GpuCurveBvhBuffers`]. There is deliberately **no
//! separate `spline_gpu_layout`**: the on-device representation of a spline
//! segment *is* a Bézier curve, and duplicating the packer would only risk the
//! two drifting apart.
//!
//! The basis change is pure add/sub/mul/div — no transcendental call, no
//! normalization — so it is deterministic and bit-reproducible on the `GPU`.

use super::bvh::{Aabb, BvhBuildConfig, LinearBvhNode};
use super::curve::{Curve, CurveBvh, CurveHit};
use super::traversal::Ray;

/// Which cubic-spline basis the four control points are expressed in.
///
/// Every variant maps one segment to an equivalent cubic Bézier through
/// [`SplineBasis::bezier_control_points`]. The segment a variant describes is
/// the span between the *inner* two control points `p1 -> p2`; `p0` and `p3`
/// are the neighbouring vertices that shape the entry/exit tangents.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SplineBasis {
    /// Interpolating Catmull-Rom (the uniform cardinal spline with zero
    /// tension): the Bézier passes through `p1` at `u = 0` and `p2` at `u = 1`,
    /// with tangents `(p2 - p0)/2` and `(p3 - p1)/2`.
    CatmullRom,
    /// Interpolating cardinal spline with an explicit `tension` knob: the
    /// endpoint tangents of [`SplineBasis::CatmullRom`] are scaled by
    /// `1 - tension`, so `tension = 0` reproduces Catmull-Rom and `tension = 1`
    /// gives zero-tangent (piecewise-linear) interpolation. Values outside
    /// `[0, 1]` are permitted and produce deliberate over/undershoot; the field
    /// is stored verbatim.
    Cardinal {
        /// Tangent-relaxation factor; tangents are scaled by `1 - tension`.
        tension: f32,
    },
    /// Approximating uniform cubic B-spline: the segment stays inside the convex
    /// hull of the four control points and is `C²`-continuous with its
    /// neighbours, but does *not* in general interpolate any control vertex.
    BSpline,
}

impl SplineBasis {
    /// Converts the four spline control `points` (`p0`, `p1`, `p2`, `p3`) into
    /// the four cubic-Bézier control points of the `p1 -> p2` segment.
    ///
    /// The result can be handed straight to [`Curve::new`]: evaluating the
    /// returned Bézier over `u ∈ [0, 1]` traces exactly the spline segment this
    /// basis describes. All arithmetic is add/sub/mul/div.
    #[must_use]
    pub fn bezier_control_points(&self, points: [[f32; 3]; 4]) -> [[f32; 3]; 4] {
        let [p0, p1, p2, p3] = points;
        match *self {
            Self::CatmullRom => cardinal_bezier(p0, p1, p2, p3, 0.0),
            Self::Cardinal { tension } => cardinal_bezier(p0, p1, p2, p3, tension),
            Self::BSpline => bspline_bezier(p0, p1, p2, p3),
        }
    }
}

/// Bézier control points of the cardinal segment `p1 -> p2`.
///
/// A cardinal spline is a Hermite curve whose endpoint tangents are
/// `m1 = (1 - tension)·(p2 - p0)/2` and `m2 = (1 - tension)·(p3 - p1)/2`.
/// Converting Hermite to Bézier on the unit interval gives inner control points
/// `p1 + m1/3` and `p2 - m2/3`, i.e. a shared scale of `(1 - tension)/6` on the
/// neighbour differences.
fn cardinal_bezier(
    p0: [f32; 3],
    p1: [f32; 3],
    p2: [f32; 3],
    p3: [f32; 3],
    tension: f32,
) -> [[f32; 3]; 4] {
    let s = (1.0 - tension) / 6.0;
    let b1 = add(p1, scale(sub(p2, p0), s));
    let b2 = sub(p2, scale(sub(p3, p1), s));
    [p1, b1, b2, p2]
}

/// Bézier control points of the uniform cubic B-spline segment `p1 -> p2`.
///
/// The standard blossom of the uniform cubic B-spline basis:
/// `b0 = (p0 + 4p1 + p2)/6`, `b1 = (2p1 + p2)/3`, `b2 = (p1 + 2p2)/3`,
/// `b3 = (p1 + 4p2 + p3)/6`.
fn bspline_bezier(
    p0: [f32; 3],
    p1: [f32; 3],
    p2: [f32; 3],
    p3: [f32; 3],
) -> [[f32; 3]; 4] {
    let b0 = scale(add(add(p0, scale(p1, 4.0)), p2), 1.0 / 6.0);
    let b1 = scale(add(scale(p1, 2.0), p2), 1.0 / 3.0);
    let b2 = scale(add(p1, scale(p2, 2.0)), 1.0 / 3.0);
    let b3 = scale(add(add(p1, scale(p2, 4.0)), p3), 1.0 / 6.0);
    [b0, b1, b2, b3]
}

/// A cubic-spline round-curve primitive (Catmull-Rom / cardinal / B-spline).
///
/// The spine is one segment of the chosen [`SplineBasis`] through the four
/// `control` vertices; the swept radius is half of a width that interpolates
/// linearly from `width_start` at `u = 0` to `width_end` at `u = 1`. `primitive`
/// is the caller's stable id, reported unchanged on every hit (mirroring
/// [`Curve`]). Widths are stored non-negative. The segment lowers to a cubic
/// Bézier via [`SplineCurve::to_curve`]; intersection and bounds delegate to
/// that Bézier so the swept-circle test is shared with [`Curve`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplineCurve {
    /// The four spline control vertices (`p0`, `p1`, `p2`, `p3`).
    control: [[f32; 3]; 4],
    /// Non-negative full width at the segment start (`u = 0`, the vertex `p1`
    /// for interpolating bases).
    width_start: f32,
    /// Non-negative full width at the segment end (`u = 1`, the vertex `p2`
    /// for interpolating bases).
    width_end: f32,
    /// The spline basis the control vertices are expressed in.
    basis: SplineBasis,
    /// Caller's stable primitive id, reported unchanged on every hit.
    primitive: u32,
}

impl SplineCurve {
    /// Builds a spline segment from four `control` vertices in the given
    /// `basis`, with the start/end widths (folded to their magnitudes) and
    /// stable id `primitive`.
    #[must_use]
    pub fn new(
        control: [[f32; 3]; 4],
        width_start: f32,
        width_end: f32,
        basis: SplineBasis,
        primitive: u32,
    ) -> Self {
        Self {
            control,
            width_start: width_start.abs(),
            width_end: width_end.abs(),
            basis,
            primitive,
        }
    }

    /// The four spline control vertices.
    #[must_use]
    pub fn control(&self) -> [[f32; 3]; 4] {
        self.control
    }

    /// Non-negative full width at the segment start (`u = 0`).
    #[must_use]
    pub fn width_start(&self) -> f32 {
        self.width_start
    }

    /// Non-negative full width at the segment end (`u = 1`).
    #[must_use]
    pub fn width_end(&self) -> f32 {
        self.width_end
    }

    /// The spline basis the control vertices are expressed in.
    #[must_use]
    pub fn basis(&self) -> SplineBasis {
        self.basis
    }

    /// Caller's stable primitive id.
    #[must_use]
    pub fn primitive(&self) -> u32 {
        self.primitive
    }

    /// Lowers this segment to the equivalent cubic-Bézier [`Curve`], carrying
    /// the widths and primitive id unchanged.
    ///
    /// This is the single point where the spline basis is applied; every other
    /// query delegates through it so the Bézier intersector is the sole source
    /// of truth, and the resulting [`Curve`] is also exactly what the `GPU`
    /// path uploads via [`super::curve_gpu_layout`].
    #[must_use]
    pub fn to_curve(&self) -> Curve {
        Curve::new(
            self.basis.bezier_control_points(self.control),
            self.width_start,
            self.width_end,
            self.primitive,
        )
    }

    /// Conservative axis-aligned bounds of the swept segment.
    ///
    /// Delegates to the lowered Bézier's hull (padded by the larger half-width),
    /// which — unlike the raw control hull — actually contains an interpolating
    /// Catmull-Rom/cardinal segment that overshoots its vertices.
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        self.to_curve().aabb()
    }

    /// Nearest intersection of `ray` with the swept segment, or `None`.
    ///
    /// Delegates to [`Curve::intersect`] on the lowered Bézier; the returned
    /// [`CurveHit`] carries this segment's `primitive` id, the spine parameter
    /// `u`, and a unit normal oriented against the incident ray.
    #[must_use]
    pub fn intersect(&self, ray: &Ray) -> Option<CurveHit> {
        self.to_curve().intersect(ray)
    }
}

/// A single-level `BVH` over cubic-spline [`SplineCurve`] segments.
///
/// Each segment is lowered to a cubic Bézier [`Curve`] once at build time and
/// stored in an inner [`CurveBvh`], so traversal reuses the exact ordered slab
/// walk and swept-circle test of the Bézier curve path. [`SplineBvh::splines`]
/// preserves the authored segments in input order for inspection; the inner
/// [`SplineBvh::curve_bvh`] holds the lowered curves reordered into leaf slices
/// and is what the `GPU` upload path packs. Empty input yields an empty
/// hierarchy ([`SplineBvh::is_empty`]).
#[derive(Clone, Debug, PartialEq)]
pub struct SplineBvh {
    /// Inner Bézier `BVH` over the lowered segments (owns the node array and the
    /// reordered curves the traversal walks).
    inner: CurveBvh,
    /// The authored spline segments, in input order.
    splines: Vec<SplineCurve>,
}

impl SplineBvh {
    /// Builds a `BVH` over `splines` with [`BvhBuildConfig::default`].
    #[must_use]
    pub fn build(splines: &[SplineCurve]) -> Self {
        Self::build_with(splines, BvhBuildConfig::default())
    }

    /// Builds a `BVH` over `splines` with the given binned-`SAH` `config`.
    ///
    /// Every segment is lowered to a cubic Bézier [`Curve`] and the inner
    /// [`CurveBvh`] is built over those; the authored segments are retained in
    /// input order.
    #[must_use]
    pub fn build_with(splines: &[SplineCurve], config: BvhBuildConfig) -> Self {
        let curves: Vec<Curve> = splines.iter().map(SplineCurve::to_curve).collect();
        Self {
            inner: CurveBvh::build_with(&curves, config),
            splines: splines.to_vec(),
        }
    }

    /// Number of flattened `BVH` nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Number of spline segments in the hierarchy.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.splines.len()
    }

    /// True when the hierarchy holds no primitives.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Root bounds, or the empty box when the hierarchy is empty.
    #[must_use]
    pub fn bounds(&self) -> Aabb {
        self.inner.bounds()
    }

    /// The flattened node array.
    #[must_use]
    pub fn nodes(&self) -> &[LinearBvhNode] {
        self.inner.nodes()
    }

    /// The authored spline segments, in input order.
    #[must_use]
    pub fn splines(&self) -> &[SplineCurve] {
        &self.splines
    }

    /// The inner Bézier `BVH` over the lowered segments.
    ///
    /// This is the structure the `GPU` upload path packs via
    /// [`super::curve_gpu_layout::GpuCurveBvhBuffers`]; a spline segment's
    /// on-device form is its lowered cubic Bézier curve.
    #[must_use]
    pub fn curve_bvh(&self) -> &CurveBvh {
        &self.inner
    }

    /// Nearest intersection along `ray`, or `None` if the ray hits nothing.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<CurveHit> {
        self.inner.closest_hit(ray)
    }

    /// True when *any* segment intersects `ray` inside its interval.
    #[must_use]
    pub fn any_hit(&self, ray: &Ray) -> bool {
        self.inner.any_hit(ray)
    }
}

/// Sum of two 3-vectors.
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Difference of two 3-vectors.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Scales a 3-vector by a scalar.
fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::curve_gpu_layout::GpuCurveBvhBuffers;

    /// Small deterministic xorshift RNG (shared `ray_scene` test generator).
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
        fn point(&mut self, lo: f32, hi: f32) -> [f32; 3] {
            [self.range(lo, hi), self.range(lo, hi), self.range(lo, hi)]
        }
    }

    /// Independent Bernstein-form cubic Bézier evaluation (oracle for the
    /// lowered control points; deliberately not the curve module's de Casteljau).
    fn bezier_eval(cp: &[[f32; 3]; 4], t: f32) -> [f32; 3] {
        let u = 1.0 - t;
        let b0 = u * u * u;
        let b1 = 3.0 * u * u * t;
        let b2 = 3.0 * u * t * t;
        let b3 = t * t * t;
        let mut out = [0.0f32; 3];
        for k in 0..3 {
            out[k] = b0 * cp[0][k] + b1 * cp[1][k] + b2 * cp[2][k] + b3 * cp[3][k];
        }
        out
    }

    /// Independent cardinal/Catmull-Rom evaluation via the Hermite basis, with
    /// tangents `m1 = (1 - tension)(p2 - p0)/2`, `m2 = (1 - tension)(p3 - p1)/2`.
    fn cardinal_eval(points: &[[f32; 3]; 4], tension: f32, t: f32) -> [f32; 3] {
        let [p0, p1, p2, p3] = *points;
        let scale_tan = (1.0 - tension) * 0.5;
        let m1 = scale(sub(p2, p0), scale_tan);
        let m2 = scale(sub(p3, p1), scale_tan);
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        let mut out = [0.0f32; 3];
        for k in 0..3 {
            out[k] = h00 * p1[k] + h10 * m1[k] + h01 * p2[k] + h11 * m2[k];
        }
        out
    }

    /// Independent uniform cubic B-spline evaluation via the explicit basis
    /// polynomials (oracle for the B-spline lowering).
    fn bspline_eval(points: &[[f32; 3]; 4], t: f32) -> [f32; 3] {
        let [p0, p1, p2, p3] = *points;
        let t2 = t * t;
        let t3 = t2 * t;
        let w0 = (1.0 - t) * (1.0 - t) * (1.0 - t) / 6.0;
        let w1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0;
        let w2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0;
        let w3 = t3 / 6.0;
        let mut out = [0.0f32; 3];
        for k in 0..3 {
            out[k] = w0 * p0[k] + w1 * p1[k] + w2 * p2[k] + w3 * p3[k];
        }
        out
    }

    /// The Catmull-Rom lowering reproduces the Hermite-basis spline to within
    /// float noise over thousands of (segment, parameter) samples.
    #[test]
    fn catmull_rom_lowering_matches_hermite_oracle() {
        let mut rng = Rng::new(0xABCD_1234);
        let mut samples = 0u32;
        for _ in 0..400 {
            let points = [
                rng.point(-4.0, 4.0),
                rng.point(-4.0, 4.0),
                rng.point(-4.0, 4.0),
                rng.point(-4.0, 4.0),
            ];
            let cp = SplineBasis::CatmullRom.bezier_control_points(points);
            for step in 0..=10 {
                let t = step as f32 / 10.0;
                let got = bezier_eval(&cp, t);
                let want = cardinal_eval(&points, 0.0, t);
                for k in 0..3 {
                    assert!((got[k] - want[k]).abs() < 1e-4, "catmull mismatch");
                }
                samples += 1;
            }
        }
        assert!(samples > 2000);
    }

    /// The cardinal lowering matches the Hermite oracle for arbitrary tension.
    #[test]
    fn cardinal_lowering_matches_hermite_oracle() {
        let mut rng = Rng::new(0x5EED_F00D);
        let mut samples = 0u32;
        for _ in 0..400 {
            let points = [
                rng.point(-3.0, 3.0),
                rng.point(-3.0, 3.0),
                rng.point(-3.0, 3.0),
                rng.point(-3.0, 3.0),
            ];
            let tension = rng.range(-0.5, 1.5);
            let cp = SplineBasis::Cardinal { tension }.bezier_control_points(points);
            for step in 0..=10 {
                let t = step as f32 / 10.0;
                let got = bezier_eval(&cp, t);
                let want = cardinal_eval(&points, tension, t);
                for k in 0..3 {
                    assert!((got[k] - want[k]).abs() < 1e-4, "cardinal mismatch");
                }
                samples += 1;
            }
        }
        assert!(samples > 2000);
    }

    /// The B-spline lowering reproduces the explicit uniform-B-spline basis.
    #[test]
    fn bspline_lowering_matches_basis_oracle() {
        let mut rng = Rng::new(0x1357_9BDF);
        let mut samples = 0u32;
        for _ in 0..400 {
            let points = [
                rng.point(-5.0, 5.0),
                rng.point(-5.0, 5.0),
                rng.point(-5.0, 5.0),
                rng.point(-5.0, 5.0),
            ];
            let cp = SplineBasis::BSpline.bezier_control_points(points);
            for step in 0..=10 {
                let t = step as f32 / 10.0;
                let got = bezier_eval(&cp, t);
                let want = bspline_eval(&points, t);
                for k in 0..3 {
                    assert!((got[k] - want[k]).abs() < 1e-4, "bspline mismatch");
                }
                samples += 1;
            }
        }
        assert!(samples > 2000);
    }

    /// Catmull-Rom interpolates its inner vertices: the lowered Bézier starts at
    /// `p1` and ends at `p2`, so a ray aimed through either vertex strikes.
    #[test]
    fn catmull_rom_interpolates_inner_vertices() {
        let points = [
            [-2.0, 0.3, 0.1],
            [0.0, 0.0, 0.0],
            [1.0, 0.5, -0.2],
            [3.0, -0.4, 0.2],
        ];
        let spline = SplineCurve::new(points, 0.2, 0.2, SplineBasis::CatmullRom, 7);
        let cp = SplineBasis::CatmullRom.bezier_control_points(points);
        for k in 0..3 {
            assert!((cp[0][k] - points[1][k]).abs() < 1e-6);
            assert!((cp[3][k] - points[2][k]).abs() < 1e-6);
        }
        // Aim a ray straight down the -z axis through p1.
        let origin = [points[1][0], points[1][1], points[1][2] + 5.0];
        let ray = Ray::new(origin, [0.0, 0.0, -1.0], 0.0, 100.0);
        let hit = spline.intersect(&ray).expect("ray through p1 must hit");
        assert_eq!(hit.primitive, 7);
        let n2 = hit.normal[0] * hit.normal[0]
            + hit.normal[1] * hit.normal[1]
            + hit.normal[2] * hit.normal[2];
        assert!((n2 - 1.0).abs() < 1e-4, "normal must be unit");
        assert!(hit.front_face);
    }

    /// A ray fired at a point sampled on the spine strikes the swept segment,
    /// and the reported hit lies within the local radius of the spine (residual
    /// test over thousands of random B-spline segments and parameters).
    #[test]
    fn ray_through_spine_hits_within_radius() {
        let mut rng = Rng::new(0x0BAD_C0DE);
        let mut hits = 0u32;
        for _ in 0..2500 {
            let points = [
                rng.point(-3.0, 3.0),
                rng.point(-3.0, 3.0),
                rng.point(-3.0, 3.0),
                rng.point(-3.0, 3.0),
            ];
            let width = rng.range(0.1, 0.5);
            let spline = SplineCurve::new(points, width, width, SplineBasis::BSpline, 1);
            let cp = SplineBasis::BSpline.bezier_control_points(points);
            let u = rng.range(0.15, 0.85);
            let spine = bezier_eval(&cp, u);
            // Fire from a random direction straight at the spine point.
            let dir = {
                let d = sub(rng.point(-1.0, 1.0), [0.0, 0.0, 0.0]);
                let len2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if len2 < 1e-3 {
                    continue;
                }
                scale(d, 1.0 / len2.sqrt())
            };
            let origin = sub(spine, scale(dir, 6.0));
            let ray = Ray::new(origin, dir, 0.0, 100.0);
            if let Some(hit) = spline.intersect(&ray) {
                let p = ray.at(hit.t);
                let off = sub(p, bezier_eval(&cp, hit.u));
                let dist = (off[0] * off[0] + off[1] * off[1] + off[2] * off[2]).sqrt();
                assert!(dist <= 0.5 * width + 1e-2, "hit outside radius: {dist}");
                hits += 1;
            }
        }
        assert!(hits > 2000, "expected plenty of spine hits, got {hits}");
    }

    /// `SplineBvh::closest_hit` and `any_hit` agree with a brute-force scan over
    /// the lowered segments across many random rays.
    #[test]
    fn bvh_matches_brute_force() {
        let mut rng = Rng::new(0xFEED_BEEF);
        let splines: Vec<SplineCurve> = (0..48)
            .map(|i| {
                let base = rng.point(-6.0, 6.0);
                let points = [
                    add(base, rng.point(-1.0, 1.0)),
                    add(base, rng.point(-1.0, 1.0)),
                    add(base, rng.point(-1.0, 1.0)),
                    add(base, rng.point(-1.0, 1.0)),
                ];
                let basis = match i % 3 {
                    0 => SplineBasis::CatmullRom,
                    1 => SplineBasis::Cardinal { tension: 0.3 },
                    _ => SplineBasis::BSpline,
                };
                SplineCurve::new(points, rng.range(0.1, 0.4), rng.range(0.1, 0.4), basis, i)
            })
            .collect();
        let bvh = SplineBvh::build(&splines);
        assert_eq!(bvh.primitive_count(), splines.len());

        for _ in 0..3000 {
            let origin = rng.point(-10.0, 10.0);
            let target = rng.point(-6.0, 6.0);
            let dir = sub(target, origin);
            let len2 = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
            if len2 < 1e-3 {
                continue;
            }
            let ray = Ray::new(origin, scale(dir, 1.0 / len2.sqrt()), 0.0, 100.0);

            let mut best: Option<CurveHit> = None;
            let mut scan = Ray::new(ray.origin(), ray.direction(), ray.t_min(), ray.t_max());
            for s in &splines {
                if let Some(hit) = s.intersect(&scan) {
                    scan = Ray::new(scan.origin(), scan.direction(), scan.t_min(), hit.t);
                    best = Some(hit);
                }
            }
            let got = bvh.closest_hit(&ray);
            match (best, got) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    assert_eq!(a.primitive, b.primitive);
                    assert!((a.t - b.t).abs() < 1e-3);
                }
                _ => panic!("bvh/brute-force disagreement"),
            }
            assert_eq!(bvh.any_hit(&ray), best.is_some());
        }
    }

    /// An empty hierarchy reports empty and never hits.
    #[test]
    fn empty_bvh_never_hits() {
        let bvh = SplineBvh::build(&[]);
        assert!(bvh.is_empty());
        assert_eq!(bvh.primitive_count(), 0);
        assert_eq!(bvh.node_count(), 0);
        let ray = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0, 100.0);
        assert!(bvh.closest_hit(&ray).is_none());
        assert!(!bvh.any_hit(&ray));
    }

    /// The lowered inner `BVH` packs through the shared curve GPU layout and the
    /// packed walk reproduces the in-memory closest hit, confirming the "spline
    /// uploads as a Bézier curve" contract holds end to end.
    #[test]
    fn curve_bvh_packs_via_shared_curve_layout() {
        let mut rng = Rng::new(0x2468_ACE0);
        let splines: Vec<SplineCurve> = (0..32)
            .map(|i| {
                let base = rng.point(-5.0, 5.0);
                let points = [
                    add(base, rng.point(-1.0, 1.0)),
                    add(base, rng.point(-1.0, 1.0)),
                    add(base, rng.point(-1.0, 1.0)),
                    add(base, rng.point(-1.0, 1.0)),
                ];
                SplineCurve::new(points, 0.25, 0.25, SplineBasis::CatmullRom, i)
            })
            .collect();
        let bvh = SplineBvh::build(&splines);
        let gpu = GpuCurveBvhBuffers::from_bvh(bvh.curve_bvh());

        for _ in 0..2000 {
            let origin = rng.point(-8.0, 8.0);
            let target = rng.point(-5.0, 5.0);
            let dir = sub(target, origin);
            let len2 = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
            if len2 < 1e-3 {
                continue;
            }
            let ray = Ray::new(origin, scale(dir, 1.0 / len2.sqrt()), 0.0, 100.0);
            let cpu = bvh.closest_hit(&ray);
            let packed = gpu.closest_hit(&ray);
            match (cpu, packed) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    assert_eq!(a.primitive, b.primitive);
                    assert!((a.t - b.t).abs() < 1e-4);
                }
                _ => panic!("packed walk disagreement"),
            }
        }
    }
}
