//! Multi-segment spline strand (hair/grass/foliage) and its single-level `BVH`.
//!
//! A production hair or grass strand is not one curve segment but a *strip*: an
//! ordered run of control vertices that a cubic basis turns into a chain of
//! overlapping segments sharing endpoints. UE's Groom hair, foliage cards, and
//! sketched ribbons are all authored this way. This module expands a strip into
//! [`super::spline::SplineCurve`] segments with a sliding four-vertex window,
//! then reuses the proven spline lowering + Bézier swept-circle intersector, so
//! there is no new ray math here — only the strand topology (segment windowing,
//! per-vertex widths, and optional endpoint clamping) layered on the existing
//! curve contract.
//!
//! A strip of `n` control vertices in the uniform [`SplineBasis::CatmullRom`],
//! [`SplineBasis::Cardinal`], or [`SplineBasis::BSpline`] basis produces
//! `n - 3` segments, each spanning the inner pair of its window; the chain
//! covers vertices `v1 .. v(n-2)`. [`SplineStrip::clamped`] duplicates the first
//! and last vertex so the chain instead interpolates the endpoints and covers
//! the full vertex range (the usual "pinned root/tip" strand).
//!
//! Because every segment lowers to a cubic Bézier, a [`SplineStripBvh`] is just
//! a [`super::spline::SplineBvh`] over all segments of all strands, and uploads
//! to the `GPU` through the shared [`super::curve_gpu_layout`] with no separate
//! layout.

use super::bvh::{Aabb, BvhBuildConfig, LinearBvhNode};
use super::curve::CurveHit;
use super::spline::{SplineBasis, SplineBvh, SplineCurve};
use super::traversal::Ray;

/// A multi-segment spline strand: a run of control vertices with per-vertex
/// widths, expanded into overlapping [`SplineCurve`] segments.
///
/// `control` and `width` are parallel arrays (one width per vertex); `basis`
/// selects the cubic spline family and `primitive` is the caller's stable id,
/// reported unchanged on every hit regardless of which segment was struck.
/// Widths are stored non-negative. A strip with fewer than four vertices has no
/// segments and never intersects.
#[derive(Clone, Debug, PartialEq)]
pub struct SplineStrip {
    /// The ordered control vertices.
    control: Vec<[f32; 3]>,
    /// Per-vertex full widths (parallel to `control`), stored non-negative.
    width: Vec<f32>,
    /// The spline basis the control vertices are expressed in.
    basis: SplineBasis,
    /// Caller's stable primitive id, reported unchanged on every hit.
    primitive: u32,
}

impl SplineStrip {
    /// Builds a strand from parallel `control` / `width` arrays in `basis` with
    /// stable id `primitive`.
    ///
    /// Widths are folded to their magnitudes. The arrays are truncated to their
    /// common length so they always stay parallel (a caller passing mismatched
    /// lengths simply loses the unmatched tail).
    #[must_use]
    pub fn new(
        control: Vec<[f32; 3]>,
        width: Vec<f32>,
        basis: SplineBasis,
        primitive: u32,
    ) -> Self {
        let n = control.len().min(width.len());
        let mut control = control;
        let mut width = width;
        control.truncate(n);
        width.truncate(n);
        for w in &mut width {
            *w = w.abs();
        }
        Self {
            control,
            width,
            basis,
            primitive,
        }
    }

    /// Builds an endpoint-interpolating strand by duplicating the first and last
    /// vertex (and their widths) before windowing.
    ///
    /// With the first/last vertex repeated, the generated segment chain covers
    /// the full original vertex range `v0 .. v(n-1)` and (for the interpolating
    /// [`SplineBasis::CatmullRom`]/[`SplineBasis::Cardinal`] bases) passes
    /// through both endpoints — the standard "pinned root and tip" strand. An
    /// input with fewer than two vertices is returned unclamped.
    #[must_use]
    pub fn clamped(
        control: Vec<[f32; 3]>,
        width: Vec<f32>,
        basis: SplineBasis,
        primitive: u32,
    ) -> Self {
        let n = control.len().min(width.len());
        if n < 2 {
            return Self::new(control, width, basis, primitive);
        }
        let mut padded_control = Vec::with_capacity(n + 2);
        let mut padded_width = Vec::with_capacity(n + 2);
        padded_control.push(control[0]);
        padded_width.push(width[0]);
        padded_control.extend_from_slice(&control[..n]);
        padded_width.extend_from_slice(&width[..n]);
        padded_control.push(control[n - 1]);
        padded_width.push(width[n - 1]);
        Self::new(padded_control, padded_width, basis, primitive)
    }

    /// The ordered control vertices.
    #[must_use]
    pub fn control(&self) -> &[[f32; 3]] {
        &self.control
    }

    /// Per-vertex full widths (parallel to [`SplineStrip::control`]).
    #[must_use]
    pub fn width(&self) -> &[f32] {
        &self.width
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

    /// Number of segments the strand expands into (`control.len() - 3`, or `0`).
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.control.len().saturating_sub(3)
    }

    /// The `i`-th segment, or `None` when `i` is out of range.
    ///
    /// Segment `i` uses the window `control[i..i + 4]` and spans the inner pair
    /// `control[i + 1] -> control[i + 2]`; its start/end widths are the inner
    /// vertices' widths so the swept radius stays continuous across the chain.
    #[must_use]
    pub fn segment(&self, i: usize) -> Option<SplineCurve> {
        if i >= self.segment_count() {
            return None;
        }
        let window = [
            self.control[i],
            self.control[i + 1],
            self.control[i + 2],
            self.control[i + 3],
        ];
        Some(SplineCurve::new(
            window,
            self.width[i + 1],
            self.width[i + 2],
            self.basis,
            self.primitive,
        ))
    }

    /// All segments of the strand, in order.
    #[must_use]
    pub fn segments(&self) -> Vec<SplineCurve> {
        (0..self.segment_count())
            .filter_map(|i| self.segment(i))
            .collect()
    }

    /// Conservative axis-aligned bounds: the union of every segment's bounds.
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        let mut bounds = Aabb::empty();
        for i in 0..self.segment_count() {
            if let Some(seg) = self.segment(i) {
                bounds = bounds.union(&seg.aabb());
            }
        }
        bounds
    }

    /// Nearest intersection of `ray` with any segment of the strand, or `None`.
    ///
    /// Scans the segments while shrinking the search interval by each hit so the
    /// reported [`CurveHit`] is the closest; it carries the strand's `primitive`
    /// id and the struck segment's local spine parameter `u`.
    #[must_use]
    pub fn intersect(&self, ray: &Ray) -> Option<CurveHit> {
        let mut scan = *ray;
        let mut best: Option<CurveHit> = None;
        for i in 0..self.segment_count() {
            let Some(seg) = self.segment(i) else { continue };
            if let Some(hit) = seg.intersect(&scan) {
                scan = Ray::new(scan.origin(), scan.direction(), scan.t_min(), hit.t);
                best = Some(hit);
            }
        }
        best
    }
}

/// A single-level `BVH` over the segments of one or more [`SplineStrip`]s.
///
/// Every strand is expanded into its [`SplineCurve`] segments, all segments are
/// pooled, and an inner [`SplineBvh`] is built over them; traversal therefore
/// reuses the spline lowering and the Bézier swept-circle test. Each hit carries
/// the originating strand's `primitive` id. Empty input (or strands with fewer
/// than four vertices) yields an empty hierarchy ([`SplineStripBvh::is_empty`]).
#[derive(Clone, Debug, PartialEq)]
pub struct SplineStripBvh {
    /// Inner spline `BVH` over the pooled segments.
    inner: SplineBvh,
    /// Number of source strands (for reporting; segments are pooled in `inner`).
    strip_count: usize,
}

impl SplineStripBvh {
    /// Builds a `BVH` over `strips` with [`BvhBuildConfig::default`].
    #[must_use]
    pub fn build(strips: &[SplineStrip]) -> Self {
        Self::build_with(strips, BvhBuildConfig::default())
    }

    /// Builds a `BVH` over the pooled segments of `strips` with the given
    /// binned-`SAH` `config`.
    #[must_use]
    pub fn build_with(strips: &[SplineStrip], config: BvhBuildConfig) -> Self {
        let mut segments: Vec<SplineCurve> = Vec::new();
        for strip in strips {
            segments.extend(strip.segments());
        }
        Self {
            inner: SplineBvh::build_with(&segments, config),
            strip_count: strips.len(),
        }
    }

    /// Number of flattened `BVH` nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Number of pooled segments across all strands.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.inner.primitive_count()
    }

    /// Number of source strands.
    #[must_use]
    pub fn strip_count(&self) -> usize {
        self.strip_count
    }

    /// True when the hierarchy holds no segments.
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

    /// The inner spline `BVH` over the pooled segments (its
    /// [`SplineBvh::curve_bvh`] is the `GPU` upload source).
    #[must_use]
    pub fn spline_bvh(&self) -> &SplineBvh {
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

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Independent Bernstein-form cubic Bézier evaluation.
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

    /// Independent Bernstein-form cubic Bézier derivative (for tangent checks).
    fn bezier_deriv(cp: &[[f32; 3]; 4], t: f32) -> [f32; 3] {
        let u = 1.0 - t;
        let d0 = 3.0 * u * u;
        let d1 = 6.0 * u * t;
        let d2 = 3.0 * t * t;
        let mut out = [0.0f32; 3];
        for k in 0..3 {
            let a = cp[1][k] - cp[0][k];
            let b = cp[2][k] - cp[1][k];
            let c = cp[3][k] - cp[2][k];
            out[k] = d0 * a + d1 * b + d2 * c;
        }
        out
    }

    fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// An `n`-vertex strip yields exactly `n - 3` segments; a short strip none.
    #[test]
    fn segment_counts() {
        let verts: Vec<[f32; 3]> = (0..7).map(|i| [i as f32, 0.0, 0.0]).collect();
        let widths = vec![0.2; 7];
        let strip = SplineStrip::new(verts, widths, SplineBasis::BSpline, 0);
        assert_eq!(strip.segment_count(), 4);
        assert_eq!(strip.segments().len(), 4);

        let short = SplineStrip::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            vec![0.1; 3],
            SplineBasis::BSpline,
            1,
        );
        assert_eq!(short.segment_count(), 0);
        assert!(short.segments().is_empty());
    }

    /// Adjacent segments are `C0`- and `C1`-continuous for both the
    /// interpolating Catmull-Rom and the approximating B-spline bases: the end
    /// of segment `i` coincides with the start of segment `i + 1`, and their
    /// unit tangents are parallel.
    #[test]
    fn adjacent_segments_are_c1_continuous() {
        let mut rng = Rng::new(0xC0FF_EE42);
        for basis in [SplineBasis::CatmullRom, SplineBasis::BSpline] {
            for _ in 0..300 {
                let n = 6;
                let verts: Vec<[f32; 3]> = (0..n).map(|_| rng.point(-4.0, 4.0)).collect();
                let widths = vec![0.2; n];
                let strip = SplineStrip::new(verts, widths, basis, 0);
                let segs = strip.segments();
                for pair in segs.windows(2) {
                    let a = pair[0].to_curve().control();
                    let b = pair[1].to_curve().control();
                    let end_a = bezier_eval(&a, 1.0);
                    let start_b = bezier_eval(&b, 0.0);
                    assert!(dist(end_a, start_b) < 1e-4, "C0 break");

                    let ta = bezier_deriv(&a, 1.0);
                    let tb = bezier_deriv(&b, 0.0);
                    let la = (ta[0] * ta[0] + ta[1] * ta[1] + ta[2] * ta[2]).sqrt();
                    let lb = (tb[0] * tb[0] + tb[1] * tb[1] + tb[2] * tb[2]).sqrt();
                    if la > 1e-4 && lb > 1e-4 {
                        let cos = (ta[0] * tb[0] + ta[1] * tb[1] + ta[2] * tb[2]) / (la * lb);
                        assert!(cos > 0.999, "C1 tangent break: cos={cos}");
                    }
                }
            }
        }
    }

    /// A clamped strip interpolates its first and last authored vertex.
    #[test]
    fn clamped_strip_pins_endpoints() {
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, -1.0, 0.5],
            [3.0, 0.5, -0.5],
        ];
        let widths = vec![0.1, 0.2, 0.2, 0.1];
        let strip = SplineStrip::clamped(verts.clone(), widths, SplineBasis::CatmullRom, 9);
        let segs = strip.segments();
        assert!(!segs.is_empty());
        let first = segs[0].to_curve().control();
        let last = segs[segs.len() - 1].to_curve().control();
        assert!(dist(bezier_eval(&first, 0.0), verts[0]) < 1e-4);
        assert!(dist(bezier_eval(&last, 1.0), verts[verts.len() - 1]) < 1e-4);
    }

    /// `SplineStripBvh` closest/any-hit agree with a brute-force scan over the
    /// pooled segments across many random rays and strands.
    #[test]
    fn bvh_matches_brute_force() {
        let mut rng = Rng::new(0xBADD_CAFE);
        let strips: Vec<SplineStrip> = (0..12)
            .map(|s| {
                let n = 5 + (s as usize % 3);
                let base = rng.point(-6.0, 6.0);
                let verts: Vec<[f32; 3]> = (0..n)
                    .map(|_| {
                        [
                            base[0] + rng.range(-2.0, 2.0),
                            base[1] + rng.range(-2.0, 2.0),
                            base[2] + rng.range(-2.0, 2.0),
                        ]
                    })
                    .collect();
                let widths = vec![rng.range(0.1, 0.3); n];
                let basis = if s % 2 == 0 {
                    SplineBasis::CatmullRom
                } else {
                    SplineBasis::BSpline
                };
                SplineStrip::new(verts, widths, basis, s)
            })
            .collect();
        let bvh = SplineStripBvh::build(&strips);
        assert_eq!(bvh.strip_count(), strips.len());

        let all_segments: Vec<SplineCurve> =
            strips.iter().flat_map(SplineStrip::segments).collect();
        assert_eq!(bvh.segment_count(), all_segments.len());

        for _ in 0..3000 {
            let origin = rng.point(-10.0, 10.0);
            let target = rng.point(-6.0, 6.0);
            let dir = [
                target[0] - origin[0],
                target[1] - origin[1],
                target[2] - origin[2],
            ];
            let len2 = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
            if len2 < 1e-3 {
                continue;
            }
            let inv = 1.0 / len2.sqrt();
            let ray = Ray::new(origin, [dir[0] * inv, dir[1] * inv, dir[2] * inv], 0.0, 100.0);

            let mut best: Option<CurveHit> = None;
            let mut scan = ray;
            for seg in &all_segments {
                if let Some(hit) = seg.intersect(&scan) {
                    scan = Ray::new(scan.origin(), scan.direction(), scan.t_min(), hit.t);
                    best = Some(hit);
                }
            }
            match (best, bvh.closest_hit(&ray)) {
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
        let bvh = SplineStripBvh::build(&[]);
        assert!(bvh.is_empty());
        assert_eq!(bvh.segment_count(), 0);
        assert_eq!(bvh.strip_count(), 0);
        let ray = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0, 100.0);
        assert!(bvh.closest_hit(&ray).is_none());
        assert!(!bvh.any_hit(&ray));
    }
}
