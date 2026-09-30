//! Closest-point geometry between two 3D line segments (design §10, §14).
//!
//! This module is the *3D segment-vs-segment* half of the subsystem's proximity
//! math. It is deliberately disjoint from its siblings:
//!
//! * [`crate::particle::segment_intersect_2d`] answers a **2D** question — do
//!   two planar segments cross, and where — using orientation predicates. It
//!   lives in the plane and has no notion of a squared distance in space.
//! * [`crate::particle::collision`] owns **collision response**: it turns an
//!   already-detected penetration into a corrected position and velocity. It
//!   consumes proximity results; it does not compute segment-to-segment closest
//!   points.
//! * **This module** answers the pure **3D geometry** question: given two line
//!   segments, what are the parameters `(s, t) ∈ [0, 1]²` of the mutually
//!   closest points, where do those points land, and what is the squared
//!   distance between them? It is the building block a capsule-vs-capsule test,
//!   a ribbon self-collision broad phase, or a trail-vs-trail proximity query
//!   would call before any response is computed.
//!
//! The algorithm is the classic closed form from Christer Ericson's
//! *Real-Time Collision Detection* (§5.1.9, `ClosestPtSegmentSegment`),
//! re-derived here rather than copied. Each segment is written
//! `P(s) = a + s · (b − a)` and `Q(t) = c + t · (d − c)`; the objective is to
//! minimize `‖P(s) − Q(t)‖²` over the unit square. Setting the gradient to zero
//! gives a 2×2 linear system whose determinant is `a·e − b²` (with `a`, `e` the
//! squared direction lengths and `b` their dot product). When that determinant
//! is near zero the directions are parallel, so the system is rank-deficient and
//! the code falls into a stable degenerate branch that fixes `s` and solves for
//! `t`. Every clamp keeps `(s, t)` inside `[0, 1]²`, and fully degenerate
//! (zero-length) segments collapse gracefully to point-segment and point-point
//! queries.
//!
//! Everything is a zero-dependency contract: the vector math is hand-rolled in
//! this file and every step uses only `+ - * /`, `f32::sqrt`, `f32::abs`,
//! `f32::min`, `f32::max`, and `f32::clamp`. No transcendental function and no
//! `==` / `!=` on a production `f32` ever appears, so this `CPU` reference
//! agrees bit for bit with a future `GPU` (`WESL`) kernel that packs the same
//! segments through the `std430` helpers in [`crate::particle::gpu_layout`].

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE, VEC4_STRIDE};

/// Epsilon used to guard divisions and to compare magnitudes without ever
/// writing an exact `==` / `!=` on a production `f32`.
///
/// A segment direction whose squared length is at or below this value is
/// treated as degenerate (a point), and a system determinant at or below it is
/// treated as parallel.
pub const CMP_EPS: f32 = 1.0e-6;

/// A hand-rolled three-component vector, kept local so the module stays a
/// zero-dependency contract and its vector math is auditable in one place.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    /// The x component.
    pub x: f32,
    /// The y component.
    pub y: f32,
    /// The z component.
    pub z: f32,
}

impl Vec3 {
    /// The zero vector.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Builds a vector from components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// A vector with all three lanes set to `s`.
    #[must_use]
    pub const fn splat(s: f32) -> Self {
        Self { x: s, y: s, z: s }
    }

    /// Component-wise sum `self + rhs`.
    ///
    /// Named `plus` (not the `Add` operator) so the whole module keeps a single
    /// uniform call-site style and never trips the operator-trait lint.
    #[must_use]
    pub fn plus(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    /// Component-wise difference `self - rhs`.
    #[must_use]
    pub fn minus(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }

    /// Uniform scale by a scalar.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    /// Euclidean dot product `self · rhs`.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Right-handed cross product `self × rhs`.
    ///
    /// Only used by tests and callers that need the mutual perpendicular of two
    /// skew directions; the closest-point solver itself never needs it.
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    /// Squared Euclidean length `self · self`.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length `√(self · self)`.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Squared Euclidean distance between two points.
    #[must_use]
    pub fn distance_squared(self, rhs: Self) -> f32 {
        self.minus(rhs).length_squared()
    }

    /// Euclidean distance between two points.
    #[must_use]
    pub fn distance(self, rhs: Self) -> f32 {
        self.distance_squared(rhs).sqrt()
    }
}

/// A directed 3D line segment from `a` to `b`.
///
/// The segment is parameterized `P(u) = a + u · (b − a)` for `u ∈ [0, 1]`, so
/// `a` is `u = 0` and `b` is `u = 1`. A segment whose endpoints coincide (its
/// direction has squared length at or below [`CMP_EPS`]) is *degenerate* and is
/// treated everywhere as the single point `a`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    /// The start endpoint (`u = 0`).
    pub a: Vec3,
    /// The end endpoint (`u = 1`).
    pub b: Vec3,
}

impl Segment {
    /// Builds a segment from its two endpoints.
    #[must_use]
    pub const fn new(a: Vec3, b: Vec3) -> Self {
        Self { a, b }
    }

    /// The direction vector `b − a` (not normalized).
    #[must_use]
    pub fn direction(&self) -> Vec3 {
        self.b.minus(self.a)
    }

    /// The squared length of the segment.
    #[must_use]
    pub fn length_squared(&self) -> f32 {
        self.direction().length_squared()
    }

    /// Whether the segment is degenerate (its endpoints coincide to within
    /// [`CMP_EPS`] in squared length), and should be treated as the point `a`.
    #[must_use]
    pub fn is_degenerate(&self) -> bool {
        self.length_squared() <= CMP_EPS
    }

    /// Evaluates the point `a + u · (b − a)` at parameter `u`.
    ///
    /// The caller is responsible for clamping `u` to `[0, 1]` when a point on
    /// the segment (rather than its supporting line) is required; the solver in
    /// this module always passes clamped parameters.
    #[must_use]
    pub fn point_at(&self, u: f32) -> Vec3 {
        self.a.plus(self.direction().scale(u))
    }
}

/// The result of a closest-point query between two segments.
///
/// `s` and `t` are the parameters along the first and second segment,
/// respectively, each already clamped to `[0, 1]`. `point_on_first` equals
/// `first.point_at(s)` and `point_on_second` equals `second.point_at(t)`;
/// `distance_squared` is the squared distance between them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClosestPoints {
    /// Parameter along the first segment, in `[0, 1]`.
    pub s: f32,
    /// Parameter along the second segment, in `[0, 1]`.
    pub t: f32,
    /// The closest point on the first segment.
    pub point_on_first: Vec3,
    /// The closest point on the second segment.
    pub point_on_second: Vec3,
    /// The squared distance between the two closest points.
    pub distance_squared: f32,
}

impl ClosestPoints {
    /// The (non-squared) distance between the two closest points.
    #[must_use]
    pub fn distance(&self) -> f32 {
        self.distance_squared.sqrt()
    }

    /// Whether the two segments touch or intersect, i.e. their closest points
    /// coincide to within [`CMP_EPS`] in squared distance.
    #[must_use]
    pub fn is_touching(&self) -> bool {
        self.distance_squared <= CMP_EPS
    }
}

/// The closest point on segment `seg` to the query point `p`, returned as the
/// clamped parameter `t ∈ [0, 1]` and the point itself.
///
/// For a degenerate segment (both endpoints coincident) the parameter is `0`
/// and the point is `seg.a`. Otherwise `t = clamp((p − a) · (b − a) / ‖b − a‖²,
/// 0, 1)` — the projection of `p` onto the supporting line, clamped back onto
/// the segment.
#[must_use]
pub fn closest_point_on_segment(seg: &Segment, p: Vec3) -> (f32, Vec3) {
    let dir = seg.direction();
    let len_sq = dir.length_squared();
    if len_sq <= CMP_EPS {
        // Degenerate segment: the whole thing is the point `a`.
        return (0.0, seg.a);
    }
    let raw = p.minus(seg.a).dot(dir) / len_sq;
    let t = raw.clamp(0.0, 1.0);
    (t, seg.point_at(t))
}

/// The squared distance from a point to a segment.
///
/// A convenience wrapper over [`closest_point_on_segment`] for callers that
/// only need the magnitude.
#[must_use]
pub fn point_segment_distance_squared(seg: &Segment, p: Vec3) -> f32 {
    let (_, q) = closest_point_on_segment(seg, p);
    p.distance_squared(q)
}

/// Computes the mutually closest points of two 3D segments.
///
/// Implements the closed-form solver from Ericson's *Real-Time Collision
/// Detection* (§5.1.9). Writing `d1 = b − a`, `d2 = d − c`, and `r = a − c`,
/// the coefficients are `a = d1·d1`, `e = d2·d2`, `f = d2·r`, `b = d1·d2`, and
/// `c = d1·r`. The unconstrained optimum solves
///
/// ```text
/// [ a  -b ] [ s ]   [ -c ]
/// [ b  -e ] [ t ] = [ -f ]
/// ```
///
/// whose determinant is `denom = a·e − b²`. Four regimes are handled:
///
/// * **Both degenerate** — both directions vanish; the answer is the
///   point-to-point distance `‖a − c‖²` at `s = t = 0`.
/// * **First degenerate** — segment one is a point; project it onto segment two.
/// * **Second degenerate** — segment two is a point; project it onto segment one.
/// * **General / parallel** — solve for `s` (clamped), then recover `t`; when
///   `denom` is at or below [`CMP_EPS`] (parallel or near-parallel) `s` is
///   pinned to `0` and `t` derived from it. A second clamp of `t` re-derives `s`
///   so both parameters end up inside `[0, 1]²`.
///
/// The returned points always satisfy `point_on_first == first.point_at(s)` and
/// `point_on_second == second.point_at(t)`, and the query is deterministic:
/// identical inputs always yield identical output.
#[must_use]
pub fn closest_points_between_segments(first: &Segment, second: &Segment) -> ClosestPoints {
    let d1 = first.direction();
    let d2 = second.direction();
    let r = first.a.minus(second.a);

    let a = d1.length_squared(); // squared length of segment 1
    let e = d2.length_squared(); // squared length of segment 2
    let f = d2.dot(r);

    let first_degenerate = a <= CMP_EPS;
    let second_degenerate = e <= CMP_EPS;

    let (s, t) = if first_degenerate && second_degenerate {
        // Both segments are points: nothing to project.
        (0.0, 0.0)
    } else if first_degenerate {
        // First segment is a point; clamp its projection onto segment two.
        // s = 0, t = clamp(f / e, 0, 1).
        let t = (f / e).clamp(0.0, 1.0);
        (0.0, t)
    } else {
        let c = d1.dot(r);
        if second_degenerate {
            // Second segment is a point; clamp its projection onto segment one.
            // t = 0, s = clamp(-c / a, 0, 1).
            let s = (-c / a).clamp(0.0, 1.0);
            (s, 0.0)
        } else {
            // The fully general non-degenerate case.
            let b = d1.dot(d2);
            let denom = a * e - b * b;

            // If the determinant is not near zero the lines are not parallel, so
            // solve for the line-line optimum along segment one; otherwise pin
            // `s` to the start and let the `t` recovery pick the offset.
            let s_line = if denom > CMP_EPS {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };

            // Recover `t` for this `s`: t = (b·s + f) / e.
            let t_line = (b * s_line + f) / e;

            // If `t` fell outside [0, 1], clamp it and recompute `s` for the
            // clamped `t` via s = (b·t − c) / a, clamped back to [0, 1].
            if t_line < 0.0 {
                let s = (-c / a).clamp(0.0, 1.0);
                (s, 0.0)
            } else if t_line > 1.0 {
                let s = ((b - c) / a).clamp(0.0, 1.0);
                (s, 1.0)
            } else {
                (s_line, t_line)
            }
        }
    };

    let point_on_first = first.point_at(s);
    let point_on_second = second.point_at(t);
    let distance_squared = point_on_first.distance_squared(point_on_second);

    ClosestPoints {
        s,
        t,
        point_on_first,
        point_on_second,
        distance_squared,
    }
}

// ---------------------------------------------------------------------------
// Optional GPU packing.
// ---------------------------------------------------------------------------

/// `std430` byte size of one packed [`Segment`]: two `vec4` slots
/// (`a.xyz + pad`, then `b.xyz + pad`). A multiple of 16 so an array of these
/// stays `vec4`-aligned on the `GPU`.
pub const SEGMENT_STD430_SIZE: usize = 2 * VEC4_STRIDE;

impl Segment {
    /// Serializes the segment to its little-endian `std430` byte image.
    ///
    /// Layout: `[a.x, a.y, a.z, 0.0]` in the first `vec4` slot, then
    /// `[b.x, b.y, b.z, 0.0]` in the second, matching how a `WESL` kernel would
    /// read two `vec4<f32>` loads.
    #[must_use]
    pub fn to_std430(&self) -> [u8; SEGMENT_STD430_SIZE] {
        let mut bytes = [0u8; SEGMENT_STD430_SIZE];
        let words = [
            self.a.x, self.a.y, self.a.z, 0.0, self.b.x, self.b.y, self.b.z, 0.0,
        ];
        for (i, word) in words.iter().enumerate() {
            let start = i * U32_STRIDE;
            bytes[start..start + U32_STRIDE].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// Total `std430` byte size for a storage buffer of `count` segments, clamped
/// up to one element so a `WebGPU` binding is never zero-sized.
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(SEGMENT_STD430_SIZE, count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Absolute tolerance for distance/parameter comparisons that are not
    /// bit-exact.
    const EPS: f32 = 1.0e-4;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < EPS, "expected {b}, got {a}");
    }

    fn approx_vec(a: Vec3, b: Vec3) {
        approx(a.x, b.x);
        approx(a.y, b.y);
        approx(a.z, b.z);
    }

    fn unit_interval(v: f32) {
        assert!((0.0..=1.0).contains(&v), "parameter {v} out of [0, 1]");
    }

    // --- Vector math ------------------------------------------------------

    #[test]
    fn vec3_plus_minus_scale_are_exact() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.plus(b), Vec3::new(5.0, 7.0, 9.0));
        assert_eq!(b.minus(a), Vec3::new(3.0, 3.0, 3.0));
        assert_eq!(a.scale(2.0), Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(Vec3::splat(2.0), Vec3::new(2.0, 2.0, 2.0));
    }

    #[test]
    fn vec3_dot_and_cross() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(x.cross(y), Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(x.dot(y), 0.0);
        assert_eq!(Vec3::new(1.0, 2.0, 3.0).dot(Vec3::new(4.0, 5.0, 6.0)), 32.0);
    }

    #[test]
    fn vec3_length_and_distance() {
        let a = Vec3::new(3.0, 4.0, 0.0);
        assert_eq!(a.length_squared(), 25.0);
        approx(a.length(), 5.0);
        assert_eq!(Vec3::ZERO.distance_squared(a), 25.0);
        approx(Vec3::ZERO.distance(a), 5.0);
    }

    #[test]
    fn segment_direction_length_and_point_at() {
        let seg = Segment::new(Vec3::new(1.0, 1.0, 1.0), Vec3::new(1.0, 5.0, 1.0));
        assert_eq!(seg.direction(), Vec3::new(0.0, 4.0, 0.0));
        assert_eq!(seg.length_squared(), 16.0);
        assert_eq!(seg.point_at(0.0), seg.a);
        assert_eq!(seg.point_at(1.0), seg.b);
        assert_eq!(seg.point_at(0.5), Vec3::new(1.0, 3.0, 1.0));
    }

    #[test]
    fn segment_degeneracy_flag() {
        let point = Segment::new(Vec3::new(2.0, 2.0, 2.0), Vec3::new(2.0, 2.0, 2.0));
        assert!(point.is_degenerate());
        let real = Segment::new(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0));
        assert!(!real.is_degenerate());
    }

    // --- Point-segment helper --------------------------------------------

    #[test]
    fn point_segment_projects_to_interior() {
        let seg = Segment::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0));
        let (t, q) = closest_point_on_segment(seg_ref(&seg), Vec3::new(3.0, 4.0, 0.0));
        approx(t, 0.3);
        approx_vec(q, Vec3::new(3.0, 0.0, 0.0));
        approx(
            point_segment_distance_squared(&seg, Vec3::new(3.0, 4.0, 0.0)),
            16.0,
        );
    }

    #[test]
    fn point_segment_clamps_before_start() {
        let seg = Segment::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0));
        let (t, q) = closest_point_on_segment(&seg, Vec3::new(-5.0, 2.0, 0.0));
        approx(t, 0.0);
        approx_vec(q, seg.a);
    }

    #[test]
    fn point_segment_clamps_past_end() {
        let seg = Segment::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0));
        let (t, q) = closest_point_on_segment(&seg, Vec3::new(20.0, -3.0, 0.0));
        approx(t, 1.0);
        approx_vec(q, seg.b);
    }

    #[test]
    fn point_segment_on_degenerate_is_point() {
        let seg = Segment::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(1.0, 2.0, 3.0));
        let (t, q) = closest_point_on_segment(&seg, Vec3::new(9.0, 9.0, 9.0));
        approx(t, 0.0);
        approx_vec(q, Vec3::new(1.0, 2.0, 3.0));
    }

    // helper so the interior-projection test can pass a reference cleanly
    fn seg_ref(seg: &Segment) -> &Segment {
        seg
    }

    // --- Segment-segment: crossing / touching ----------------------------

    #[test]
    fn crossing_segments_have_zero_distance() {
        // Two segments through the origin in the xy-plane, crossing at (0,0,0).
        let first = Segment::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let second = Segment::new(Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let r = closest_points_between_segments(&first, &second);
        approx(r.distance_squared, 0.0);
        assert!(r.is_touching());
        approx(r.s, 0.5);
        approx(r.t, 0.5);
        approx_vec(r.point_on_first, Vec3::ZERO);
        approx_vec(r.point_on_second, Vec3::ZERO);
    }

    #[test]
    fn touching_at_shared_endpoint() {
        let first = Segment::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        let second = Segment::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0));
        let r = closest_points_between_segments(&first, &second);
        approx(r.distance_squared, 0.0);
        approx(r.s, 1.0);
        approx(r.t, 0.0);
        approx_vec(r.point_on_first, Vec3::new(1.0, 0.0, 0.0));
    }

    // --- Segment-segment: skew (non-coplanar) lines ----------------------

    #[test]
    fn skew_lines_perpendicular_offset() {
        // Segment 1 along x at z=0; segment 2 along y at z=2, offset in x so the
        // mutual closest points sit at their crossing in the xy-projection.
        let first = Segment::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(5.0, 0.0, 0.0));
        let second = Segment::new(Vec3::new(0.0, -5.0, 2.0), Vec3::new(0.0, 5.0, 2.0));
        let r = closest_points_between_segments(&first, &second);
        approx(r.distance_squared, 4.0);
        approx(r.distance(), 2.0);
        approx(r.s, 0.5);
        approx(r.t, 0.5);
        approx_vec(r.point_on_first, Vec3::new(0.0, 0.0, 0.0));
        approx_vec(r.point_on_second, Vec3::new(0.0, 0.0, 2.0));
    }

    #[test]
    fn skew_lines_general_position() {
        let first = Segment::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(4.0, 0.0, 0.0));
        let second = Segment::new(Vec3::new(1.0, 3.0, 1.0), Vec3::new(1.0, 3.0, -1.0));
        let r = closest_points_between_segments(&first, &second);
        // Closest approach: (1,0,0) on first, (1,3,0) on second, distance 3.
        approx(r.distance(), 3.0);
        approx_vec(r.point_on_first, Vec3::new(1.0, 0.0, 0.0));
        approx_vec(r.point_on_second, Vec3::new(1.0, 3.0, 0.0));
        unit_interval(r.s);
        unit_interval(r.t);
    }

    // --- Segment-segment: parallel ---------------------------------------

    #[test]
    fn parallel_overlapping_offset() {
        // Two parallel x-axis segments 2 apart in y, fully overlapping in x.
        let first = Segment::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0));
        let second = Segment::new(Vec3::new(2.0, 2.0, 0.0), Vec3::new(8.0, 2.0, 0.0));
        let r = closest_points_between_segments(&first, &second);
        approx(r.distance_squared, 4.0);
        // Parallel branch pins s=0 then recovers t; both must stay in range and
        // the y-gap must be exactly 2 with no x separation.
        unit_interval(r.s);
        unit_interval(r.t);
        approx(r.point_on_first.y, 0.0);
        approx(r.point_on_second.y, 2.0);
        approx(r.point_on_first.x, r.point_on_second.x);
    }

    #[test]
    fn parallel_disjoint_along_axis() {
        // Collinear-direction parallel segments that do not overlap in x: the
        // near endpoints (10,0,0) and (14,1,0) govern the distance.
        let first = Segment::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0));
        let second = Segment::new(Vec3::new(14.0, 1.0, 0.0), Vec3::new(24.0, 1.0, 0.0));
        let r = closest_points_between_segments(&first, &second);
        // dx = 4, dy = 1 -> squared distance 17.
        approx(r.distance_squared, 17.0);
        approx_vec(r.point_on_first, Vec3::new(10.0, 0.0, 0.0));
        approx_vec(r.point_on_second, Vec3::new(14.0, 1.0, 0.0));
        approx(r.s, 1.0);
        approx(r.t, 0.0);
    }

    #[test]
    fn collinear_overlapping_touch() {
        // Same supporting line, overlapping ranges -> distance zero.
        let first = Segment::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(4.0, 0.0, 0.0));
        let second = Segment::new(Vec3::new(2.0, 0.0, 0.0), Vec3::new(6.0, 0.0, 0.0));
        let r = closest_points_between_segments(&first, &second);
        approx(r.distance_squared, 0.0);
        assert!(r.is_touching());
        unit_interval(r.s);
        unit_interval(r.t);
    }

    // --- Endpoint clamping ------------------------------------------------

    #[test]
    fn endpoint_clamp_both_ends() {
        // Two short segments whose closest approach forces both parameters to a
        // boundary. First near the origin along +x, second far along +y from a
        // high x so the optimum is first.b vs second.a.
        let first = Segment::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0));
        let second = Segment::new(Vec3::new(5.0, 3.0, 0.0), Vec3::new(5.0, 9.0, 0.0));
        let r = closest_points_between_segments(&first, &second);
        approx(r.s, 1.0);
        approx(r.t, 0.0);
        approx_vec(r.point_on_first, Vec3::new(2.0, 0.0, 0.0));
        approx_vec(r.point_on_second, Vec3::new(5.0, 3.0, 0.0));
    }

    #[test]
    fn endpoint_clamp_t_past_end_recovers_s() {
        // Force the t>1 branch: second segment's near approach is at its far end.
        let first = Segment::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0));
        let second = Segment::new(Vec3::new(20.0, 5.0, 0.0), Vec3::new(3.0, 1.0, 0.0));
        let r = closest_points_between_segments(&first, &second);
        unit_interval(r.s);
        unit_interval(r.t);
        // Point on second must equal second.point_at(t) exactly.
        approx_vec(r.point_on_second, second.point_at(r.t));
        approx_vec(r.point_on_first, first.point_at(r.s));
    }

    // --- Degenerate segments ---------------------------------------------

    #[test]
    fn both_degenerate_is_point_point() {
        let first = Segment::new(Vec3::new(1.0, 1.0, 1.0), Vec3::new(1.0, 1.0, 1.0));
        let second = Segment::new(Vec3::new(4.0, 5.0, 1.0), Vec3::new(4.0, 5.0, 1.0));
        let r = closest_points_between_segments(&first, &second);
        approx(r.s, 0.0);
        approx(r.t, 0.0);
        // (3,4,0) -> distance 5, squared 25.
        approx(r.distance_squared, 25.0);
        approx(r.distance(), 5.0);
        approx_vec(r.point_on_first, first.a);
        approx_vec(r.point_on_second, second.a);
    }

    #[test]
    fn first_degenerate_projects_onto_second() {
        let first = Segment::new(Vec3::new(3.0, 4.0, 0.0), Vec3::new(3.0, 4.0, 0.0));
        let second = Segment::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0));
        let r = closest_points_between_segments(&first, &second);
        approx(r.s, 0.0);
        approx(r.t, 0.3);
        approx_vec(r.point_on_second, Vec3::new(3.0, 0.0, 0.0));
        approx(r.distance_squared, 16.0);
    }

    #[test]
    fn first_degenerate_clamps_onto_second_endpoint() {
        let first = Segment::new(Vec3::new(-4.0, 3.0, 0.0), Vec3::new(-4.0, 3.0, 0.0));
        let second = Segment::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0));
        let r = closest_points_between_segments(&first, &second);
        approx(r.s, 0.0);
        approx(r.t, 0.0);
        approx_vec(r.point_on_second, second.a);
        approx(r.distance_squared, 25.0);
    }

    #[test]
    fn second_degenerate_projects_onto_first() {
        let first = Segment::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 10.0, 0.0));
        let second = Segment::new(Vec3::new(4.0, 6.0, 0.0), Vec3::new(4.0, 6.0, 0.0));
        let r = closest_points_between_segments(&first, &second);
        approx(r.t, 0.0);
        approx(r.s, 0.6);
        approx_vec(r.point_on_first, Vec3::new(0.0, 6.0, 0.0));
        approx(r.distance_squared, 16.0);
    }

    #[test]
    fn second_degenerate_clamps_onto_first_endpoint() {
        let first = Segment::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 10.0, 0.0));
        let second = Segment::new(Vec3::new(3.0, -4.0, 0.0), Vec3::new(3.0, -4.0, 0.0));
        let r = closest_points_between_segments(&first, &second);
        approx(r.t, 0.0);
        approx(r.s, 0.0);
        approx_vec(r.point_on_first, first.a);
        approx(r.distance_squared, 25.0);
    }

    // --- Structural invariants -------------------------------------------

    #[test]
    fn points_match_parameter_evaluation() {
        let first = Segment::new(Vec3::new(-2.0, 1.0, 3.0), Vec3::new(4.0, -1.0, 0.0));
        let second = Segment::new(Vec3::new(1.0, 5.0, -2.0), Vec3::new(-3.0, 2.0, 4.0));
        let r = closest_points_between_segments(&first, &second);
        approx_vec(r.point_on_first, first.point_at(r.s));
        approx_vec(r.point_on_second, second.point_at(r.t));
        approx(
            r.distance_squared,
            r.point_on_first.distance_squared(r.point_on_second),
        );
    }

    #[test]
    fn parameters_always_in_unit_square() {
        let cases = [
            (
                Segment::new(Vec3::new(-3.0, -3.0, -3.0), Vec3::new(3.0, 2.0, 1.0)),
                Segment::new(Vec3::new(5.0, -1.0, 2.0), Vec3::new(-2.0, 4.0, -1.0)),
            ),
            (
                Segment::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),
                Segment::new(Vec3::new(0.0, 0.0, 5.0), Vec3::new(1.0, 0.0, 5.0)),
            ),
            (
                Segment::new(Vec3::new(2.0, 2.0, 2.0), Vec3::new(2.0, 2.0, 2.0)),
                Segment::new(Vec3::new(-1.0, 4.0, 0.0), Vec3::new(3.0, 1.0, 7.0)),
            ),
        ];
        for (first, second) in cases {
            let r = closest_points_between_segments(&first, &second);
            unit_interval(r.s);
            unit_interval(r.t);
        }
    }

    #[test]
    fn symmetry_swaps_the_two_points() {
        let first = Segment::new(Vec3::new(-2.0, 0.0, 1.0), Vec3::new(5.0, 3.0, -2.0));
        let second = Segment::new(Vec3::new(0.0, 6.0, 4.0), Vec3::new(4.0, -1.0, 2.0));
        let forward = closest_points_between_segments(&first, &second);
        let backward = closest_points_between_segments(&second, &first);
        // Distance is symmetric.
        approx(forward.distance_squared, backward.distance_squared);
        // Swapping the arguments swaps the roles of the two closest points.
        approx_vec(forward.point_on_first, backward.point_on_second);
        approx_vec(forward.point_on_second, backward.point_on_first);
        approx(forward.s, backward.t);
        approx(forward.t, backward.s);
    }

    #[test]
    fn distance_is_symmetric_for_parallel() {
        let first = Segment::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(6.0, 0.0, 0.0));
        let second = Segment::new(Vec3::new(-2.0, 3.0, 0.0), Vec3::new(9.0, 3.0, 0.0));
        let forward = closest_points_between_segments(&first, &second);
        let backward = closest_points_between_segments(&second, &first);
        approx(forward.distance_squared, backward.distance_squared);
        approx(forward.distance_squared, 9.0);
    }

    #[test]
    fn determinism_repeated_queries_match() {
        let first = Segment::new(Vec3::new(1.5, -2.0, 0.5), Vec3::new(-3.0, 4.0, 2.0));
        let second = Segment::new(Vec3::new(2.0, 2.0, -1.0), Vec3::new(-1.0, -3.0, 3.0));
        let a = closest_points_between_segments(&first, &second);
        let b = closest_points_between_segments(&first, &second);
        assert_eq!(a, b);
    }

    #[test]
    fn touching_flag_reflects_distance() {
        let far = closest_points_between_segments(
            &Segment::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)),
            &Segment::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(1.0, 5.0, 0.0)),
        );
        assert!(!far.is_touching());
        let near = closest_points_between_segments(
            &Segment::new(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)),
            &Segment::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 4.0, 0.0)),
        );
        assert!(near.is_touching());
    }

    #[test]
    fn distance_never_exceeds_any_endpoint_pairing() {
        // The closest-approach distance must be <= every endpoint-endpoint gap.
        let first = Segment::new(Vec3::new(-1.0, -1.0, 0.0), Vec3::new(2.0, 3.0, 1.0));
        let second = Segment::new(Vec3::new(3.0, 0.0, 4.0), Vec3::new(-2.0, 5.0, 2.0));
        let r = closest_points_between_segments(&first, &second);
        let endpoint_pairs = [
            first.a.distance_squared(second.a),
            first.a.distance_squared(second.b),
            first.b.distance_squared(second.a),
            first.b.distance_squared(second.b),
        ];
        for gap in endpoint_pairs {
            assert!(r.distance_squared <= gap + EPS);
        }
    }

    // --- GPU packing ------------------------------------------------------

    #[test]
    fn std430_size_is_multiple_of_16() {
        assert_eq!(SEGMENT_STD430_SIZE % 16, 0);
        assert_eq!(SEGMENT_STD430_SIZE, 32);
    }

    #[test]
    fn std430_round_trips_endpoints() {
        let seg = Segment::new(Vec3::new(1.0, -2.0, 3.5), Vec3::new(-4.0, 0.25, 6.0));
        let bytes = seg.to_std430();
        assert_eq!(bytes.len(), SEGMENT_STD430_SIZE);
        let read = |off: usize| {
            f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
        };
        approx(read(0), 1.0);
        approx(read(4), -2.0);
        approx(read(8), 3.5);
        approx(read(12), 0.0);
        approx(read(16), -4.0);
        approx(read(20), 0.25);
        approx(read(24), 6.0);
        approx(read(28), 0.0);
    }

    #[test]
    fn gpu_storage_bytes_scales_and_clamps() {
        assert_eq!(gpu_storage_bytes(0), SEGMENT_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(1), SEGMENT_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(8), 8 * SEGMENT_STD430_SIZE);
    }

    #[test]
    fn packing_many_segments_stays_aligned() {
        let mut blob: Vec<u8> = Vec::new();
        for i in 0..5 {
            let f = i as f32;
            let seg = Segment::new(Vec3::splat(f), Vec3::splat(f + 1.0));
            blob.extend_from_slice(&seg.to_std430());
        }
        assert_eq!(blob.len(), 5 * SEGMENT_STD430_SIZE);
        assert_eq!(blob.len() % 16, 0);
    }
}
