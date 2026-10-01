//! Hardware ray-traced curve / `Linear Swept Spheres` (LSS) strand primitive
//! `BLAS` description contract (design §8.5 item 11).
//!
//! The newest ray-tracing hardware (`RTX` via `DXR` / `OptiX`) can express a
//! hair strand segment as a `Linear Swept Spheres` (LSS) / curve primitive that
//! the acceleration structure traverses directly, so hair self-shadowing and
//! in-reflection real strands no longer need a proxy mesh. This module is the
//! architecture-side contract plus a CPU golden: it turns one strand (a
//! polyline of vertices, each with a radius) into a list of LSS segments
//! (endpoints `a`/`b` with radii `radius_a`/`radius_b`) and produces each
//! segment's conservative `axis-aligned` bounding box (`AABB`) for a downstream
//! `BLAS` build.
//!
//! The real `BLAS` build and ray traversal belong to the non-portable driver
//! bucket (design §9); this module only lays down the deterministic geometry
//! contract: arrays in, arrays out, order preserved, never panics, and no
//! transcendental math. A swept-sphere endpoint box needs no square root — each
//! axis is just the center plus or minus the radius — so the whole module uses
//! only `min`/`max` and `sqrt`-free arithmetic.

use alloc::vec::Vec;

/// One vertex of a strand polyline: a world-space position and the strand
/// radius at that vertex.
///
/// A negative or non-finite radius is not a valid sweep radius; it is clamped
/// to `0.0` by [`CurveVertex::sanitized`] (and therefore by every public entry
/// point that consumes vertices). Non-finite position components are likewise
/// forced to `0.0` so a stray `NaN` can never poison an `AABB`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurveVertex {
    /// World-space position of the vertex.
    pub position: [f32; 3],
    /// Strand radius (half-thickness) at this vertex.
    pub radius: f32,
}

impl CurveVertex {
    /// Builds a vertex from a raw position and radius without sanitizing.
    #[must_use]
    pub const fn new(position: [f32; 3], radius: f32) -> Self {
        Self { position, radius }
    }

    /// Returns a copy with non-finite position components forced to `0.0` and
    /// the radius clamped to a finite, non-negative value.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            position: sanitize_point(self.position),
            radius: sanitize_radius(self.radius),
        }
    }
}

/// One `Linear Swept Spheres` (LSS) segment: a sphere of radius `radius_a` at
/// `a` linearly swept to a sphere of radius `radius_b` at `b`.
///
/// This is the per-segment primitive a downstream `BLAS` registers. Both radii
/// are non-negative and finite after [`LssSegment::sanitized`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LssSegment {
    /// First endpoint center.
    pub a: [f32; 3],
    /// Second endpoint center.
    pub b: [f32; 3],
    /// Sweep radius at `a`.
    pub radius_a: f32,
    /// Sweep radius at `b`.
    pub radius_b: f32,
}

impl LssSegment {
    /// Builds a segment from raw endpoints and radii without sanitizing.
    #[must_use]
    pub const fn new(a: [f32; 3], b: [f32; 3], radius_a: f32, radius_b: f32) -> Self {
        Self {
            a,
            b,
            radius_a,
            radius_b,
        }
    }

    /// Returns a copy with non-finite endpoint components forced to `0.0` and
    /// both radii clamped to finite, non-negative values.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            a: sanitize_point(self.a),
            b: sanitize_point(self.b),
            radius_a: sanitize_radius(self.radius_a),
            radius_b: sanitize_radius(self.radius_b),
        }
    }
}

/// A conservative `axis-aligned` bounding box (`AABB`).
///
/// The empty box is encoded as `min` greater than `max` on every axis (see
/// [`Aabb::EMPTY`]); it is the identity of [`Aabb::union`] and contains no
/// point. A box is valid (non-empty) exactly when `min[i] <= max[i]` on all
/// three axes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// Minimum corner.
    pub min: [f32; 3],
    /// Maximum corner.
    pub max: [f32; 3],
}

impl Aabb {
    /// The empty box: `min = +inf`, `max = -inf` on every axis. Unioning it
    /// with any box `x` yields `x`, and it contains no point.
    pub const EMPTY: Self = Self {
        min: [f32::INFINITY; 3],
        max: [f32::NEG_INFINITY; 3],
    };

    /// Builds a box from explicit corners without validation.
    #[must_use]
    pub const fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    /// `true` when the box is non-empty, i.e. `min[i] <= max[i]` on all axes.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.min[0] <= self.max[0] && self.min[1] <= self.max[1] && self.min[2] <= self.max[2]
    }

    /// Returns the tightest box enclosing both `self` and `other`.
    ///
    /// [`Aabb::EMPTY`] acts as the identity: `EMPTY.union(x) == x` and
    /// `x.union(EMPTY) == x`, because `min` takes a per-axis minimum (so `+inf`
    /// never wins) and `max` takes a per-axis maximum (so `-inf` never wins).
    #[must_use]
    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    /// `true` when `point` lies inside the box (inclusive on all faces). The
    /// empty box contains no point.
    #[must_use]
    pub fn contains_point(&self, point: [f32; 3]) -> bool {
        point[0] >= self.min[0]
            && point[0] <= self.max[0]
            && point[1] >= self.min[1]
            && point[1] <= self.max[1]
            && point[2] >= self.min[2]
            && point[2] <= self.max[2]
    }
}

/// Clamps a radius to a finite, non-negative value: negative or non-finite
/// (`NaN` / `inf`) radii collapse to `0.0`.
#[must_use]
fn sanitize_radius(radius: f32) -> f32 {
    if radius.is_finite() && radius >= 0.0 {
        radius
    } else {
        0.0
    }
}

/// Forces any non-finite component of a point to `0.0`, leaving finite
/// components untouched.
#[must_use]
fn sanitize_point(point: [f32; 3]) -> [f32; 3] {
    [
        sanitize_coord(point[0]),
        sanitize_coord(point[1]),
        sanitize_coord(point[2]),
    ]
}

/// Forces a single non-finite coordinate to `0.0`.
#[must_use]
fn sanitize_coord(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

/// Conservative `axis-aligned` box around a single swept-sphere endpoint:
/// the center expanded by the (already non-negative) radius on each axis.
#[must_use]
fn endpoint_aabb(center: [f32; 3], radius: f32) -> Aabb {
    Aabb {
        min: [center[0] - radius, center[1] - radius, center[2] - radius],
        max: [center[0] + radius, center[1] + radius, center[2] + radius],
    }
}

/// Converts a strand polyline into `Linear Swept Spheres` (LSS) segments.
///
/// Adjacent vertices form one segment each, so `n` vertices yield `n - 1`
/// segments in input order; fewer than two vertices yield no segments. Every
/// endpoint and radius is sanitized ([`CurveVertex::sanitized`]), so bad radii
/// collapse to `0.0` and non-finite positions collapse to the origin. The
/// mapping is deterministic and never panics.
#[must_use]
pub fn strand_to_lss(vertices: &[CurveVertex]) -> Vec<LssSegment> {
    let mut segments = Vec::new();
    if vertices.len() < 2 {
        return segments;
    }
    segments.reserve(vertices.len() - 1);
    for pair in vertices.windows(2) {
        let v0 = pair[0].sanitized();
        let v1 = pair[1].sanitized();
        segments.push(LssSegment::new(
            v0.position,
            v1.position,
            v0.radius,
            v1.radius,
        ));
    }
    segments
}

/// Conservative `axis-aligned` bounding box (`AABB`) of one LSS segment: the
/// union of the two endpoint spheres' boxes. Needs no square root — each axis
/// is the endpoint center plus or minus its radius. The segment is sanitized
/// first, so the result is always a valid, finite box.
#[must_use]
pub fn lss_segment_aabb(seg: &LssSegment) -> Aabb {
    let clean = seg.sanitized();
    let box_a = endpoint_aabb(clean.a, clean.radius_a);
    let box_b = endpoint_aabb(clean.b, clean.radius_b);
    box_a.union(&box_b)
}

/// Conservative `AABB` enclosing every LSS segment of a strand. An empty strand
/// (or a single vertex, which yields no segments) maps to [`Aabb::EMPTY`].
#[must_use]
pub fn strand_aabb(vertices: &[CurveVertex]) -> Aabb {
    let segments = strand_to_lss(vertices);
    segments_aabb(&segments)
}

/// Conservative `AABB` enclosing a set of LSS segments. An empty set maps to
/// [`Aabb::EMPTY`].
#[must_use]
pub fn segments_aabb(segs: &[LssSegment]) -> Aabb {
    let mut bounds = Aabb::EMPTY;
    for seg in segs {
        bounds = bounds.union(&lss_segment_aabb(seg));
    }
    bounds
}

/// Total number of LSS segments produced by a batch of strands, i.e. the sum of
/// `max(0, len - 1)` over each strand. Lets a downstream `BLAS` size its
/// primitive budget before any geometry is emitted.
#[must_use]
pub fn total_lss_segments(strands: &[&[CurveVertex]]) -> usize {
    let mut total = 0usize;
    for strand in strands {
        if strand.len() >= 2 {
            total += strand.len() - 1;
        }
    }
    total
}

/// Per-strand LSS segment counts, in strand order. Element `i` is the number of
/// segments strand `i` contributes (`max(0, len - 1)`), so a downstream `BLAS`
/// can bucket primitives per strand.
#[must_use]
pub fn lss_segment_counts(strands: &[&[CurveVertex]]) -> Vec<usize> {
    let mut counts = Vec::with_capacity(strands.len());
    for strand in strands {
        counts.push(if strand.len() >= 2 {
            strand.len() - 1
        } else {
            0
        });
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    fn close3(a: [f32; 3], b: [f32; 3]) -> bool {
        close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2])
    }

    #[test]
    fn three_vertices_make_two_segments() {
        let verts = [
            CurveVertex::new([0.0, 0.0, 0.0], 1.0),
            CurveVertex::new([1.0, 0.0, 0.0], 1.0),
            CurveVertex::new([2.0, 0.0, 0.0], 1.0),
        ];
        let segs = strand_to_lss(&verts);
        assert_eq!(segs.len(), 2);
        assert!(close3(segs[0].a, [0.0, 0.0, 0.0]));
        assert!(close3(segs[0].b, [1.0, 0.0, 0.0]));
        assert!(close3(segs[1].a, [1.0, 0.0, 0.0]));
        assert!(close3(segs[1].b, [2.0, 0.0, 0.0]));
    }

    #[test]
    fn single_vertex_makes_no_segments() {
        let verts = [CurveVertex::new([0.0, 0.0, 0.0], 1.0)];
        assert!(strand_to_lss(&verts).is_empty());
    }

    #[test]
    fn empty_input_makes_no_segments() {
        let verts: [CurveVertex; 0] = [];
        assert!(strand_to_lss(&verts).is_empty());
    }

    #[test]
    fn segment_aabb_contains_both_endpoint_spheres() {
        let seg = LssSegment::new([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], 1.0, 2.0);
        let bb = lss_segment_aabb(&seg);
        // Endpoint a sphere extremes.
        assert!(bb.contains_point([-1.0, 0.0, 0.0]));
        assert!(bb.contains_point([0.0, 1.0, 0.0]));
        // Endpoint b sphere extremes (radius 2 around x=4).
        assert!(bb.contains_point([6.0, 0.0, 0.0]));
        assert!(bb.contains_point([4.0, 2.0, -2.0]));
        // Tight bounds: min from a's box, max from b's box.
        assert!(close3(bb.min, [-1.0, -2.0, -2.0]));
        assert!(close3(bb.max, [6.0, 2.0, 2.0]));
    }

    #[test]
    fn strand_aabb_contains_all_segments() {
        let verts = [
            CurveVertex::new([0.0, 0.0, 0.0], 0.5),
            CurveVertex::new([0.0, 3.0, 0.0], 0.5),
            CurveVertex::new([0.0, 3.0, 5.0], 0.5),
        ];
        let bb = strand_aabb(&verts);
        assert!(bb.is_valid());
        // Must enclose every vertex sphere.
        for v in &verts {
            let s = v.sanitized();
            assert!(bb.contains_point([s.position[0] + s.radius, s.position[1], s.position[2]]));
            assert!(bb.contains_point([s.position[0] - s.radius, s.position[1], s.position[2]]));
        }
        assert!(close3(bb.min, [-0.5, -0.5, -0.5]));
        assert!(close3(bb.max, [0.5, 3.5, 5.5]));
    }

    #[test]
    fn aabb_union_is_correct() {
        let left = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let right = Aabb::new([-2.0, 0.5, 3.0], [0.5, 2.0, 4.0]);
        let u = left.union(&right);
        assert!(close3(u.min, [-2.0, 0.0, 0.0]));
        assert!(close3(u.max, [1.0, 2.0, 4.0]));
    }

    #[test]
    fn bad_radius_is_sanitized_to_zero() {
        let verts = [
            CurveVertex::new([0.0, 0.0, 0.0], -5.0),
            CurveVertex::new([1.0, 0.0, 0.0], f32::NAN),
        ];
        let segs = strand_to_lss(&verts);
        assert_eq!(segs.len(), 1);
        assert!(close(segs[0].radius_a, 0.0));
        assert!(close(segs[0].radius_b, 0.0));
        // With zero radii the box collapses to the endpoints.
        let bb = lss_segment_aabb(&segs[0]);
        assert!(close3(bb.min, [0.0, 0.0, 0.0]));
        assert!(close3(bb.max, [1.0, 0.0, 0.0]));
    }

    #[test]
    fn nan_vertex_does_not_panic() {
        let verts = [
            CurveVertex::new([f32::NAN, 0.0, f32::INFINITY], 1.0),
            CurveVertex::new([2.0, f32::NEG_INFINITY, 0.0], 1.0),
        ];
        let bb = strand_aabb(&verts);
        // Non-finite coords are forced to 0.0, so the box stays valid/finite.
        assert!(bb.is_valid());
        assert!(bb.min[0].is_finite());
        assert!(bb.max[1].is_finite());
    }

    #[test]
    fn empty_union_is_identity() {
        let x = Aabb::new([-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]);
        let left = Aabb::EMPTY.union(&x);
        let right = x.union(&Aabb::EMPTY);
        assert!(close3(left.min, x.min) && close3(left.max, x.max));
        assert!(close3(right.min, x.min) && close3(right.max, x.max));
        // EMPTY unioned with EMPTY stays empty / invalid.
        assert!(!Aabb::EMPTY.union(&Aabb::EMPTY).is_valid());
    }

    #[test]
    fn contains_point_boundary() {
        let bb = Aabb::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        // Faces and corners are inclusive.
        assert!(bb.contains_point([0.0, 0.0, 0.0]));
        assert!(bb.contains_point([2.0, 2.0, 2.0]));
        assert!(bb.contains_point([0.0, 1.0, 2.0]));
        // Just outside is excluded.
        assert!(!bb.contains_point([-EPS, 1.0, 1.0]));
        assert!(!bb.contains_point([1.0, 2.0 + 1.0, 1.0]));
        // The empty box contains nothing.
        assert!(!Aabb::EMPTY.contains_point([0.0, 0.0, 0.0]));
    }

    #[test]
    fn order_is_preserved() {
        let verts = [
            CurveVertex::new([0.0, 0.0, 0.0], 0.1),
            CurveVertex::new([1.0, 0.0, 0.0], 0.2),
            CurveVertex::new([2.0, 0.0, 0.0], 0.3),
            CurveVertex::new([3.0, 0.0, 0.0], 0.4),
        ];
        let segs = strand_to_lss(&verts);
        assert_eq!(segs.len(), 3);
        assert!(close(segs[0].radius_a, 0.1) && close(segs[0].radius_b, 0.2));
        assert!(close(segs[1].radius_a, 0.2) && close(segs[1].radius_b, 0.3));
        assert!(close(segs[2].radius_a, 0.3) && close(segs[2].radius_b, 0.4));
    }

    #[test]
    fn total_and_per_strand_counts() {
        let a = [
            CurveVertex::new([0.0, 0.0, 0.0], 1.0),
            CurveVertex::new([1.0, 0.0, 0.0], 1.0),
            CurveVertex::new([2.0, 0.0, 0.0], 1.0),
        ];
        let b = [CurveVertex::new([0.0, 0.0, 0.0], 1.0)];
        let c: [CurveVertex; 0] = [];
        let strands: [&[CurveVertex]; 3] = [&a, &b, &c];
        assert_eq!(total_lss_segments(&strands), 2);
        assert_eq!(lss_segment_counts(&strands), alloc::vec![2usize, 0, 0]);
    }

    #[test]
    fn segments_aabb_matches_strand_aabb() {
        let verts = [
            CurveVertex::new([0.0, 0.0, 0.0], 1.0),
            CurveVertex::new([5.0, 0.0, 0.0], 1.0),
        ];
        let segs = strand_to_lss(&verts);
        let from_segs = segments_aabb(&segs);
        let from_strand = strand_aabb(&verts);
        assert!(close3(from_segs.min, from_strand.min));
        assert!(close3(from_segs.max, from_strand.max));
        // Empty segment set is EMPTY.
        assert!(!segments_aabb(&[]).is_valid());
    }
}
