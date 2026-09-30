//! Finite segment vs. triangle intersection in 3D via a segment-clamped
//! `Moller-Trumbore` solve (design §14): the analytic "does this *bounded* line
//! pierce this face, and where" contract used by particle trails, tethers, and
//! swept-point collision probes.
//!
//! A particle that moves from `start` to `end` in one step sweeps a finite
//! segment; deciding whether that step crosses a triangle of static geometry is
//! the `Moller-Trumbore` triangle test with the ray parameter *clamped to the
//! unit interval* `[0, 1]`. Solving the `3x3` system once yields the two free
//! `barycentric` weights, the third weight, the segment parameter `t`, and the
//! hit point in a single division, with every degenerate input (a segment
//! parallel to or lying in the face, and a zero-area triangle) guarded so it
//! reports a miss instead of dividing by a near-zero quantity.
//!
//! # Border with the sibling contracts
//! This module is *only* the finite-segment analytic kernel, and is
//! deliberately distinct from its neighbours:
//! * [`ray_triangle`](crate::particle::ray_triangle) tests an *infinite* half
//!   line (`t >= 0`, unbounded above); here `t` is bounded to `[0, 1]`, so a
//!   crossing that lies past `end` (`t > 1`) or behind `start` (`t < 0`) is a
//!   miss, not a hit.
//! * `tri_tri_intersect` (triangle vs. triangle, authored separately in this
//!   batch) answers whether two *faces* overlap; this module intersects a
//!   one-dimensional segment against a single face.
//! * [`barycentric_coord`](crate::particle::barycentric_coord) only converts an
//!   already-known point into weights; it performs no line/plane solve and
//!   never decides *where* along a segment a surface is met.
//! * [`point_triangle_closest_3d`](crate::particle::point_triangle_closest_3d)
//!   finds the nearest surface point to a query point (a projection /
//!   minimisation); this module finds a transversal crossing (an intersection).
//!
//! # No transcendental math
//! Every routine is polynomial plus at most one guarded reciprocal and, only in
//! the vector helpers, one `sqrt` for a length. There is no `sin`, `cos`,
//! `atan`, `exp` or `pow` anywhere, and floats are never compared with `==` or
//! `!=`: a magnitude is "zero" when it is smaller than the relevant epsilon.
//!
//! # Barycentric convention
//! A returned [`Hit`] reports `u` as the weight of `v1`, `v` as the weight of
//! `v2`, and `w = 1 - u - v` as the weight of `v0`, so the hit point equals
//! `w * v0 + u * v1 + v * v2` and the three weights sum to one.

/// Magnitude below which the solve determinant is treated as zero, so a segment
/// that runs parallel to (or inside) the triangle plane falls back to a miss
/// rather than amplifying round-off through a near-zero division.
const DET_EPS: f32 = 1.0e-8;

/// Magnitude below which a triangle's doubled area (the length of the edge
/// cross product) marks the triangle as degenerate (a point or a sliver), so it
/// reports a miss instead of solving an ill-conditioned system.
const AREA_EPS: f32 = 1.0e-6;

/// Slack applied to the `barycentric` and segment-parameter inclusion tests so
/// a hit that lands exactly on an edge, a vertex, or an endpoint is accepted
/// despite floating-point round-off, instead of being rejected as just-outside.
const EDGE_EPS: f32 = 1.0e-5;

/// A three-component vector in the right-handed space the segment endpoints and
/// triangle vertices share.
///
/// Its arithmetic methods are named `plus` / `minus` / `scale` (not the
/// operator names) to keep the closed-form algebra explicit and to avoid
/// implying a component-wise `Mul` alongside the `dot` and `cross` products.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    /// First (x) component.
    pub x: f32,
    /// Second (y) component.
    pub y: f32,
    /// Third (z) component.
    pub z: f32,
}

impl Vec3 {
    /// The additive-identity vector `(0, 0, 0)`.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Builds a vector from its three components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Component-wise sum `self + other`.
    #[must_use]
    pub fn plus(&self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    /// Component-wise difference `self - other`.
    #[must_use]
    pub fn minus(&self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    /// Uniform scale `self * scalar`.
    #[must_use]
    pub fn scale(&self, scalar: f32) -> Vec3 {
        Vec3 {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }

    /// The Euclidean dot product `self . other`.
    #[must_use]
    pub fn dot(&self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// The right-handed cross product `self x other`.
    #[must_use]
    pub fn cross(&self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// The squared length `self . self`, avoiding the `sqrt` when only relative
    /// magnitudes matter.
    #[must_use]
    pub fn length_squared(&self) -> f32 {
        self.dot(*self)
    }

    /// The Euclidean length `sqrt(self . self)`.
    #[must_use]
    pub fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }
}

/// A finite line segment between the points `start` and `end`.
///
/// The segment direction `end - start` need not be unit length; the returned
/// parameter `t` is measured as a fraction of that direction, so `t = 0` is the
/// `start`, `t = 1` is the `end`, and only `t` in `[0, 1]` lies on the segment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    /// The point the segment starts from (parameter `t = 0`).
    pub start: Vec3,
    /// The point the segment ends at (parameter `t = 1`).
    pub end: Vec3,
}

impl Segment {
    /// Builds a segment from its two endpoints.
    #[must_use]
    pub const fn new(start: Vec3, end: Vec3) -> Self {
        Self { start, end }
    }

    /// The point `start + t * (end - start)` at parameter `t` along the
    /// segment.
    #[must_use]
    pub fn point_at(&self, t: f32) -> Vec3 {
        self.start.plus(self.end.minus(self.start).scale(t))
    }
}

/// A successful segment-triangle intersection.
///
/// The `barycentric` weights follow the module convention: `u` weights `v1`,
/// `v` weights `v2`, and `w = 1 - u - v` weights `v0`, so the hit `point`
/// equals `w * v0 + u * v1 + v * v2`. The parameter `t` gives the crossing
/// position along the segment and always lies in `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    /// The intersection point in world space.
    pub point: Vec3,
    /// `barycentric` weight of the second vertex `v1`.
    pub u: f32,
    /// `barycentric` weight of the third vertex `v2`.
    pub v: f32,
    /// `barycentric` weight of the first vertex `v0`, equal to `1 - u - v`.
    pub w: f32,
    /// Fraction along the segment (`0` at `start`, `1` at `end`) of the
    /// crossing.
    pub t: f32,
}

/// Intersects the finite `segment` with the triangle `v0`, `v1`, `v2` using a
/// segment-clamped `Moller-Trumbore` solve.
///
/// Returns the [`Hit`] (crossing point, the three `barycentric` weights, and the
/// segment parameter `t`) when the segment pierces the triangle at some `t` in
/// `[0, 1]`, or `None` on a miss. It reports a miss when:
/// * the triangle is degenerate (its doubled area is below [`AREA_EPS`]),
/// * the segment is parallel to or lies in the triangle plane (the solve
///   determinant is below [`DET_EPS`] in magnitude),
/// * the crossing lies outside the triangle (the `barycentric` weights fall
///   outside `[0, 1]` beyond the [`EDGE_EPS`] slack), or
/// * the crossing lies off the segment (`t` outside `[0, 1]` beyond the slack).
#[must_use]
pub fn intersect(segment: &Segment, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<Hit> {
    let edge1 = v1.minus(v0);
    let edge2 = v2.minus(v0);

    // Reject a degenerate (zero-area) triangle before dividing.
    let face_normal = edge1.cross(edge2);
    if face_normal.length_squared() < AREA_EPS * AREA_EPS {
        return None;
    }

    let dir = segment.end.minus(segment.start);
    let pvec = dir.cross(edge2);
    let det = edge1.dot(pvec);

    // A near-zero determinant means the segment is parallel to (or lies in) the
    // triangle plane; also catches a zero-length segment (a null direction).
    if det.abs() < DET_EPS {
        return None;
    }

    let inv_det = 1.0 / det;
    let tvec = segment.start.minus(v0);

    let u = tvec.dot(pvec) * inv_det;
    if !(-EDGE_EPS..=1.0 + EDGE_EPS).contains(&u) {
        return None;
    }

    let qvec = tvec.cross(edge1);
    let v = dir.dot(qvec) * inv_det;
    if v < -EDGE_EPS || u + v > 1.0 + EDGE_EPS {
        return None;
    }

    let t = edge2.dot(qvec) * inv_det;
    if !(-EDGE_EPS..=1.0 + EDGE_EPS).contains(&t) {
        return None;
    }

    let point = segment.start.plus(dir.scale(t));
    let w = 1.0 - u - v;
    Some(Hit { point, u, v, w, t })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for scalar comparisons in the tests, wide enough to
    /// absorb `f32` round-off yet far tighter than any geometric feature used.
    const TOL: f32 = 1.0e-4;

    /// Epsilon-based scalar equality, used instead of `==` on `f32`.
    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < TOL
    }

    /// Epsilon-based vector equality, component by component.
    fn close_vec(a: Vec3, b: Vec3) -> bool {
        close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z)
    }

    /// The canonical unit right-triangle in the `z = 0` plane.
    fn unit_triangle() -> (Vec3, Vec3, Vec3) {
        (
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        )
    }

    #[test]
    fn through_interior_center_hits_midway() {
        let (v0, v1, v2) = unit_triangle();
        let seg = Segment::new(Vec3::new(0.3, 0.3, 1.0), Vec3::new(0.3, 0.3, -1.0));
        let hit = intersect(&seg, v0, v1, v2).expect("interior crossing must hit");
        assert!(close(hit.t, 0.5), "t should be halfway, got {}", hit.t);
        assert!(close(hit.u, 0.3));
        assert!(close(hit.v, 0.3));
        assert!(close(hit.w, 0.4));
        assert!(close_vec(hit.point, Vec3::new(0.3, 0.3, 0.0)));
    }

    #[test]
    fn hits_vertex_v0_with_full_w_weight() {
        let (v0, v1, v2) = unit_triangle();
        let seg = Segment::new(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = intersect(&seg, v0, v1, v2).expect("crossing at v0 must hit");
        assert!(close(hit.u, 0.0));
        assert!(close(hit.v, 0.0));
        assert!(close(hit.w, 1.0));
        assert!(close_vec(hit.point, v0));
    }

    #[test]
    fn hits_vertex_v1_with_full_u_weight() {
        let (v0, v1, v2) = unit_triangle();
        let seg = Segment::new(Vec3::new(1.0, 0.0, 1.0), Vec3::new(1.0, 0.0, -1.0));
        let hit = intersect(&seg, v0, v1, v2).expect("crossing at v1 must hit");
        assert!(close(hit.u, 1.0));
        assert!(close(hit.v, 0.0));
        assert!(close(hit.w, 0.0));
        assert!(close_vec(hit.point, v1));
    }

    #[test]
    fn hits_vertex_v2_with_full_v_weight() {
        let (v0, v1, v2) = unit_triangle();
        let seg = Segment::new(Vec3::new(0.0, 1.0, 1.0), Vec3::new(0.0, 1.0, -1.0));
        let hit = intersect(&seg, v0, v1, v2).expect("crossing at v2 must hit");
        assert!(close(hit.u, 0.0));
        assert!(close(hit.v, 1.0));
        assert!(close(hit.w, 0.0));
        assert!(close_vec(hit.point, v2));
    }

    #[test]
    fn hits_midpoint_of_edge_v1_v2() {
        let (v0, v1, v2) = unit_triangle();
        let seg = Segment::new(Vec3::new(0.5, 0.5, 1.0), Vec3::new(0.5, 0.5, -1.0));
        let hit = intersect(&seg, v0, v1, v2).expect("crossing on edge v1v2 must hit");
        assert!(close(hit.u, 0.5));
        assert!(close(hit.v, 0.5));
        assert!(
            close(hit.u + hit.v, 1.0),
            "on edge v1v2 the weights sum to one"
        );
        assert!(close(hit.w, 0.0));
    }

    #[test]
    fn grazes_edge_v0_v1_boundary() {
        let (v0, v1, v2) = unit_triangle();
        // Point (0.5, 0, 0) sits exactly on edge v0v1 (v == 0).
        let seg = Segment::new(Vec3::new(0.5, 0.0, 1.0), Vec3::new(0.5, 0.0, -1.0));
        let hit = intersect(&seg, v0, v1, v2).expect("grazing the edge counts as a hit");
        assert!(
            close(hit.v, 0.0),
            "v weight is zero on edge v0v1, got {}",
            hit.v
        );
        assert!(close(hit.u, 0.5));
        assert!(close(hit.w, 0.5));
    }

    #[test]
    fn misses_when_line_passes_outside_triangle() {
        let (v0, v1, v2) = unit_triangle();
        let seg = Segment::new(Vec3::new(2.0, 2.0, 1.0), Vec3::new(2.0, 2.0, -1.0));
        assert!(
            intersect(&seg, v0, v1, v2).is_none(),
            "a crossing far outside the face must miss"
        );
    }

    #[test]
    fn misses_when_segment_too_short_to_reach_plane() {
        let (v0, v1, v2) = unit_triangle();
        // Aimed at the interior but stops at z = 0.5, so the plane sits at
        // t = 2 (> 1) along the segment.
        let seg = Segment::new(Vec3::new(0.3, 0.3, 1.0), Vec3::new(0.3, 0.3, 0.5));
        assert!(
            intersect(&seg, v0, v1, v2).is_none(),
            "a segment ending before the plane must miss"
        );
    }

    #[test]
    fn misses_when_crossing_lies_beyond_end() {
        let (v0, v1, v2) = unit_triangle();
        // Both endpoints above the plane; the plane is only reached at t > 1.
        let seg = Segment::new(Vec3::new(0.3, 0.3, 2.0), Vec3::new(0.3, 0.3, 1.0));
        assert!(
            intersect(&seg, v0, v1, v2).is_none(),
            "a crossing past the end (t > 1) must miss"
        );
    }

    #[test]
    fn misses_when_crossing_lies_before_start() {
        let (v0, v1, v2) = unit_triangle();
        // Both endpoints below the plane; the plane is only reached at t < 0.
        let seg = Segment::new(Vec3::new(0.3, 0.3, -0.5), Vec3::new(0.3, 0.3, -1.5));
        assert!(
            intersect(&seg, v0, v1, v2).is_none(),
            "a crossing behind the start (t < 0) must miss"
        );
    }

    #[test]
    fn misses_when_parallel_to_face() {
        let (v0, v1, v2) = unit_triangle();
        // Direction lies in the z = 0.5 plane, parallel to the triangle plane.
        let seg = Segment::new(Vec3::new(0.1, 0.1, 0.5), Vec3::new(0.6, 0.1, 0.5));
        assert!(
            intersect(&seg, v0, v1, v2).is_none(),
            "a segment parallel to the face must miss"
        );
    }

    #[test]
    fn misses_when_coplanar_with_face() {
        let (v0, v1, v2) = unit_triangle();
        // Segment lies entirely in the z = 0 plane of the triangle.
        let seg = Segment::new(Vec3::new(0.1, 0.1, 0.0), Vec3::new(0.5, 0.1, 0.0));
        assert!(
            intersect(&seg, v0, v1, v2).is_none(),
            "a coplanar segment must miss (degenerate determinant)"
        );
    }

    #[test]
    fn hits_when_start_endpoint_lies_on_face() {
        let (v0, v1, v2) = unit_triangle();
        // start is on the plane inside the triangle, so the hit is at t = 0.
        let seg = Segment::new(Vec3::new(0.3, 0.3, 0.0), Vec3::new(0.3, 0.3, -1.0));
        let hit = intersect(&seg, v0, v1, v2).expect("start on the face must hit");
        assert!(
            close(hit.t, 0.0),
            "hit should be at the start, got {}",
            hit.t
        );
        assert!(close_vec(hit.point, Vec3::new(0.3, 0.3, 0.0)));
    }

    #[test]
    fn hits_when_end_endpoint_lies_on_face() {
        let (v0, v1, v2) = unit_triangle();
        // end is on the plane inside the triangle, so the hit is at t = 1.
        let seg = Segment::new(Vec3::new(0.3, 0.3, 1.0), Vec3::new(0.3, 0.3, 0.0));
        let hit = intersect(&seg, v0, v1, v2).expect("end on the face must hit");
        assert!(close(hit.t, 1.0), "hit should be at the end, got {}", hit.t);
        assert!(close_vec(hit.point, Vec3::new(0.3, 0.3, 0.0)));
    }

    #[test]
    fn barycentric_weights_sum_to_one() {
        let (v0, v1, v2) = unit_triangle();
        let seg = Segment::new(Vec3::new(0.2, 0.5, 1.0), Vec3::new(0.2, 0.5, -1.0));
        let hit = intersect(&seg, v0, v1, v2).expect("interior crossing must hit");
        assert!(
            close(hit.u + hit.v + hit.w, 1.0),
            "weights must sum to one, got {}",
            hit.u + hit.v + hit.w
        );
    }

    #[test]
    fn hit_parameter_lies_in_unit_interval() {
        let (v0, v1, v2) = unit_triangle();
        let seg = Segment::new(Vec3::new(0.25, 0.25, 3.0), Vec3::new(0.25, 0.25, -2.0));
        let hit = intersect(&seg, v0, v1, v2).expect("interior crossing must hit");
        assert!(
            (0.0..=1.0).contains(&hit.t),
            "t must be within [0, 1], got {}",
            hit.t
        );
    }

    #[test]
    fn misses_on_degenerate_collinear_triangle() {
        let v0 = Vec3::new(0.0, 0.0, 0.0);
        let v1 = Vec3::new(1.0, 0.0, 0.0);
        let v2 = Vec3::new(2.0, 0.0, 0.0);
        let seg = Segment::new(Vec3::new(0.5, 0.5, 1.0), Vec3::new(0.5, 0.5, -1.0));
        assert!(
            intersect(&seg, v0, v1, v2).is_none(),
            "a zero-area triangle must miss"
        );
    }

    #[test]
    fn is_invariant_under_translation() {
        let (v0, v1, v2) = unit_triangle();
        let seg = Segment::new(Vec3::new(0.3, 0.3, 1.0), Vec3::new(0.3, 0.3, -1.0));
        let base = intersect(&seg, v0, v1, v2).expect("base crossing must hit");

        let shift = Vec3::new(-4.0, 7.5, 2.25);
        let seg_t = Segment::new(seg.start.plus(shift), seg.end.plus(shift));
        let moved = intersect(&seg_t, v0.plus(shift), v1.plus(shift), v2.plus(shift)).expect("hit");

        assert!(close(base.u, moved.u));
        assert!(close(base.v, moved.v));
        assert!(close(base.w, moved.w));
        assert!(close(base.t, moved.t));
        assert!(close_vec(base.point.plus(shift), moved.point));
    }

    #[test]
    fn reversed_segment_still_hits() {
        let (v0, v1, v2) = unit_triangle();
        let forward = Segment::new(Vec3::new(0.3, 0.3, 1.0), Vec3::new(0.3, 0.3, -1.0));
        let backward = Segment::new(Vec3::new(0.3, 0.3, -1.0), Vec3::new(0.3, 0.3, 1.0));
        let a = intersect(&forward, v0, v1, v2).expect("forward hit");
        let b = intersect(&backward, v0, v1, v2).expect("backward hit");
        assert!(
            close_vec(a.point, b.point),
            "both directions cross the same point"
        );
        assert!(close(a.t + b.t, 1.0), "reversing complements t");
    }

    #[test]
    fn hit_point_reconstructs_from_barycentric_weights() {
        let v0 = Vec3::new(1.0, 0.0, 0.0);
        let v1 = Vec3::new(0.0, 2.0, 0.0);
        let v2 = Vec3::new(0.0, 0.0, 3.0);
        let seg = Segment::new(Vec3::new(1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, -1.0));
        let hit = intersect(&seg, v0, v1, v2).expect("oblique crossing must hit");
        let rebuilt = v0.scale(hit.w).plus(v1.scale(hit.u)).plus(v2.scale(hit.v));
        assert!(
            close_vec(rebuilt, hit.point),
            "w*v0 + u*v1 + v*v2 must equal the hit point"
        );
    }

    #[test]
    fn hit_point_lies_on_the_triangle_plane() {
        let v0 = Vec3::new(1.0, 0.0, 0.0);
        let v1 = Vec3::new(0.0, 2.0, 0.0);
        let v2 = Vec3::new(0.0, 0.0, 3.0);
        let seg = Segment::new(Vec3::new(1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, -1.0));
        let hit = intersect(&seg, v0, v1, v2).expect("oblique crossing must hit");
        let normal = v1.minus(v0).cross(v2.minus(v0));
        // The plane equation n . (p - v0) must vanish for a point on the plane.
        assert!(
            close(normal.dot(hit.point.minus(v0)), 0.0),
            "hit point must satisfy the plane equation"
        );
    }

    #[test]
    fn misses_just_outside_a_vertex() {
        let (v0, v1, v2) = unit_triangle();
        // u = v = 0.6 => u + v = 1.2, just outside the far edge.
        let seg = Segment::new(Vec3::new(0.6, 0.6, 1.0), Vec3::new(0.6, 0.6, -1.0));
        assert!(
            intersect(&seg, v0, v1, v2).is_none(),
            "a crossing just past the far edge must miss"
        );
    }

    #[test]
    fn misses_on_zero_length_segment() {
        let (v0, v1, v2) = unit_triangle();
        // A degenerate segment (start == end) has a null direction.
        let p = Vec3::new(0.3, 0.3, 0.0);
        let seg = Segment::new(p, p);
        assert!(
            intersect(&seg, v0, v1, v2).is_none(),
            "a zero-length segment has no direction and must miss"
        );
    }

    #[test]
    fn hits_with_non_unit_long_direction() {
        let (v0, v1, v2) = unit_triangle();
        // A long segment whose direction is far from unit length still resolves
        // the crossing correctly because t is a fraction of that direction.
        let seg = Segment::new(Vec3::new(0.2, 0.2, 50.0), Vec3::new(0.2, 0.2, -50.0));
        let hit = intersect(&seg, v0, v1, v2).expect("long segment must still hit");
        assert!(close(hit.t, 0.5), "t should be halfway, got {}", hit.t);
        assert!(close_vec(hit.point, Vec3::new(0.2, 0.2, 0.0)));
    }

    #[test]
    fn hits_a_large_scaled_triangle() {
        let v0 = Vec3::new(0.0, 0.0, 0.0);
        let v1 = Vec3::new(100.0, 0.0, 0.0);
        let v2 = Vec3::new(0.0, 100.0, 0.0);
        let seg = Segment::new(Vec3::new(30.0, 30.0, 10.0), Vec3::new(30.0, 30.0, -10.0));
        let hit = intersect(&seg, v0, v1, v2).expect("scaled triangle must hit");
        assert!(close(hit.u, 0.3));
        assert!(close(hit.v, 0.3));
        assert!(close_vec(hit.point, Vec3::new(30.0, 30.0, 0.0)));
    }

    #[test]
    fn oblique_hit_is_internally_consistent() {
        let v0 = Vec3::new(-1.0, -1.0, 0.5);
        let v1 = Vec3::new(2.0, 0.0, -0.5);
        let v2 = Vec3::new(0.0, 2.0, 1.0);
        let seg = Segment::new(Vec3::new(0.2, 0.2, 5.0), Vec3::new(0.3, 0.1, -5.0));
        let hit = intersect(&seg, v0, v1, v2).expect("oblique crossing must hit");
        // The reported point must agree both with the segment parameterisation
        // and with the barycentric reconstruction.
        assert!(close_vec(hit.point, seg.point_at(hit.t)));
        let rebuilt = v0.scale(hit.w).plus(v1.scale(hit.u)).plus(v2.scale(hit.v));
        assert!(close_vec(rebuilt, hit.point));
        assert!(close(hit.u + hit.v + hit.w, 1.0));
    }

    #[test]
    fn w_weight_equals_one_minus_u_minus_v() {
        let (v0, v1, v2) = unit_triangle();
        let seg = Segment::new(Vec3::new(0.15, 0.55, 1.0), Vec3::new(0.15, 0.55, -1.0));
        let hit = intersect(&seg, v0, v1, v2).expect("interior crossing must hit");
        assert!(
            close(hit.w, 1.0 - hit.u - hit.v),
            "w must equal 1 - u - v exactly (up to round-off)"
        );
    }
}
