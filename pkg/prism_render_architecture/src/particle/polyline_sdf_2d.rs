//! 2D analytic signed-distance fields for polylines and polygons used by the
//! particle spatial contracts (design §8.2, §12-§13).
//!
//! Several particle stages reason about *flat, screen-space* geometry rather
//! than the 3D primitives that [`super::sdf`] and [`super::capsule_sdf`] model:
//! a spawn mask fades emission by how far a sample lies from an authored 2D
//! outline; a screen-space kill volume pushes particles out of a polygonal
//! keep-out region by its signed boundary distance; and a debug overlay draws
//! the medial band of a stroked path. This module owns the small,
//! `CPU`-verifiable contract those stages share: the clamped point-to-segment
//! distance, the closest point on a segment, the unsigned distance to an open
//! polyline, and the signed distance to a closed polygon (negative inside,
//! positive outside).
//!
//! The polyline distance is the minimum unsigned distance over every segment of
//! the path. The polygon signed distance takes that same unsigned distance over
//! the *closed* ring and then chooses a sign from a containment test: the
//! non-zero winding number decides inside (negative) from outside (positive),
//! so the field is correct for both convex and concave (and even
//! self-intersecting) rings.
//!
//! # Strict scope
//! This module is purely 2D and purely *analytic distance*. It deliberately
//! does **not** touch the 3D sphere/box/capsule fields of [`super::sdf`] or
//! [`super::capsule_sdf`], build a hull ([`super::convex_hull_2d`]), or
//! triangulate ([`super::ear_clip_triangulate`]); it neither imports nor
//! reconstructs those contracts and keeps its own vector math and containment
//! rule rather than sharing a sibling's.
//!
//! # No transcendental math
//! Every routine here is pure `+`, `-`, `*`, `/`, comparison, `f32::clamp`,
//! `f32::abs`, `f32::min`/`f32::max`, and the single `f32::sqrt` that turns a
//! squared distance into a distance. There is no `sin`, `cos`, `atan`, `exp`,
//! `ln`, `powf`, `ceil`, `round` or any other transcendental / rounding call,
//! and no `f32` equality: near-zero magnitudes are compared against
//! [`CMP_EPS`].

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Magnitude below which a squared length, a coordinate difference, or a
/// distance is treated as zero. This is the comparison rule used throughout
/// instead of `==` on `f32`: two scalars are "equal" when their absolute
/// difference does not exceed this bound, and a segment whose squared length
/// does not exceed it is treated as a single point when projecting onto it.
pub const CMP_EPS: f32 = 1.0e-6;

/// `std430` byte size of one packed [`Segment`] (two `vec2<f32>` endpoints
/// laid out contiguously, i.e. exactly one `vec4` slot of 16 bytes).
pub const POLYLINE_SDF_2D_STD430_SIZE: usize = VEC4_STRIDE;

/// A hand-rolled 2D vector, owned by this module so it depends on no sibling
/// type. All arithmetic is exact `f32` `+`, `-`, `*`, `/`; the only irrational
/// operation is the [`Vec2::length`] square root.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    /// Horizontal component.
    pub x: f32,
    /// Vertical component.
    pub y: f32,
}

impl Vec2 {
    /// The zero vector, `(0, 0)`.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    /// Builds a vector from its two components.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Component-wise sum `self + other`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API is specified with named add/sub/neg methods for call-site uniformity, matching the sibling particle contracts; operator traits are intentionally not part of this internal type."
    )]
    pub fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }

    /// Component-wise difference `self - other`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API is specified with named add/sub/neg methods for call-site uniformity, matching the sibling particle contracts; operator traits are intentionally not part of this internal type."
    )]
    pub fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }

    /// Uniform scale `self * s`.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s)
    }

    /// Dot product `self . other`.
    #[must_use]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y
    }

    /// 2D cross product `self x other` (a scalar), i.e. the signed area of the
    /// parallelogram the two vectors span.
    #[must_use]
    pub fn cross(self, other: Self) -> f32 {
        self.x * other.y - self.y * other.x
    }

    /// Squared Euclidean length, avoiding the square root.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length. Uses the single permitted [`f32::sqrt`].
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Squared distance to `other`, avoiding the square root.
    #[must_use]
    pub fn distance_squared(self, other: Self) -> f32 {
        self.sub(other).length_squared()
    }

    /// Euclidean distance to `other`.
    #[must_use]
    pub fn distance(self, other: Self) -> f32 {
        self.sub(other).length()
    }

    /// Component-wise minimum of two vectors.
    #[must_use]
    pub fn min(self, other: Self) -> Self {
        Self::new(self.x.min(other.x), self.y.min(other.y))
    }

    /// Component-wise maximum of two vectors.
    #[must_use]
    pub fn max(self, other: Self) -> Self {
        Self::new(self.x.max(other.x), self.y.max(other.y))
    }
}

/// A directed 2D line segment from [`Segment::a`] to [`Segment::b`], the
/// natural element of a polyline and the module's `std430`-packable structure.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Segment {
    /// Start endpoint.
    pub a: Vec2,
    /// End endpoint.
    pub b: Vec2,
}

impl Segment {
    /// Builds a segment from its two endpoints.
    #[must_use]
    pub const fn new(a: Vec2, b: Vec2) -> Self {
        Self { a, b }
    }

    /// The `b - a` direction vector (not normalized).
    #[must_use]
    pub fn direction(self) -> Vec2 {
        self.b.sub(self.a)
    }

    /// Squared length of the segment.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.direction().length_squared()
    }

    /// Euclidean length of the segment.
    #[must_use]
    pub fn length(self) -> f32 {
        self.direction().length()
    }

    /// Unsigned distance from `p` to this segment, delegating to
    /// [`segment_distance`].
    #[must_use]
    pub fn distance_to(self, p: Vec2) -> f32 {
        segment_distance(p, self.a, self.b)
    }

    /// Closest point on this segment to `p`, delegating to
    /// [`closest_point_on_segment`].
    #[must_use]
    pub fn closest_point(self, p: Vec2) -> Vec2 {
        closest_point_on_segment(p, self.a, self.b)
    }

    /// Packs the segment into its `std430` block as little-endian
    /// `a.x, a.y, b.x, b.y`, spanning one `vec4` slot
    /// ([`POLYLINE_SDF_2D_STD430_SIZE`] bytes).
    #[must_use]
    pub fn to_std430(self) -> [u8; POLYLINE_SDF_2D_STD430_SIZE] {
        let mut bytes = [0u8; POLYLINE_SDF_2D_STD430_SIZE];
        bytes[0..4].copy_from_slice(&self.a.x.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.a.y.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.b.x.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.b.y.to_le_bytes());
        bytes
    }
}

/// The clamped projection parameter `t` of `p` onto the line through `a` and
/// `b`, clamped to `[0, 1]` so it names a point on the *segment*.
///
/// For a degenerate (zero-length) segment the direction vanishes and the
/// projection is undefined, so this returns `0.0`, pinning the closest point to
/// `a`. The clamp is the sole reason an interior projection collapses onto an
/// endpoint when `p` lies beyond it.
#[must_use]
pub fn segment_projection_t(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b.sub(a);
    let len2 = ab.length_squared();
    if len2 <= CMP_EPS {
        return 0.0;
    }
    let ap = p.sub(a);
    (ap.dot(ab) / len2).clamp(0.0, 1.0)
}

/// The closest point on segment `a -> b` to `p`.
///
/// The projection parameter is clamped to `[0, 1]`, so points whose foot of
/// perpendicular falls outside the segment snap to the nearer endpoint. A
/// zero-length segment returns `a`.
#[must_use]
pub fn closest_point_on_segment(p: Vec2, a: Vec2, b: Vec2) -> Vec2 {
    let t = segment_projection_t(p, a, b);
    a.add(b.sub(a).scale(t))
}

/// The unsigned Euclidean distance from `p` to segment `a -> b`.
///
/// This is `|p - c|` where `c` is [`closest_point_on_segment`]. It is `0` when
/// `p` lies on the segment (including at either endpoint), symmetric in the two
/// endpoints, and well-defined for a zero-length segment (distance from `p` to
/// the coincident point).
#[must_use]
pub fn segment_distance(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    p.distance(closest_point_on_segment(p, a, b))
}

/// The unsigned distance from `p` to an *open* polyline given by its ordered
/// vertices, i.e. the minimum [`segment_distance`] over every consecutive pair.
///
/// * An empty vertex list has no geometry, so this returns [`f32::INFINITY`].
/// * A single vertex degenerates to the distance from `p` to that point.
/// * Otherwise the vertices are treated as an open path (no closing edge from
///   the last vertex back to the first).
#[must_use]
pub fn polyline_distance(p: Vec2, vertices: &[Vec2]) -> f32 {
    match vertices.len() {
        0 => f32::INFINITY,
        1 => p.distance(vertices[0]),
        n => {
            let mut best = f32::INFINITY;
            let mut i = 0;
            while i + 1 < n {
                best = best.min(segment_distance(p, vertices[i], vertices[i + 1]));
                i += 1;
            }
            best
        }
    }
}

/// The unsigned distance from `p` to the *closed* ring of `polygon`, i.e. the
/// minimum [`segment_distance`] over every edge including the closing edge from
/// the last vertex back to the first.
///
/// An empty polygon returns [`f32::INFINITY`]; a single vertex returns the
/// distance to that point.
#[must_use]
pub fn polygon_boundary_distance(p: Vec2, polygon: &[Vec2]) -> f32 {
    let n = polygon.len();
    match n {
        0 => f32::INFINITY,
        1 => p.distance(polygon[0]),
        _ => {
            let mut best = f32::INFINITY;
            let mut i = 0;
            while i < n {
                let j = if i + 1 == n { 0 } else { i + 1 };
                best = best.min(segment_distance(p, polygon[i], polygon[j]));
                i += 1;
            }
            best
        }
    }
}

/// Signed side of the directed edge `a -> b` that `p` lies on: `(b - a) x
/// (p - a)`.
///
/// Strictly positive when `p` is to the left of `a -> b` (a counter-clockwise
/// turn), strictly negative when to the right, and zero when the three points
/// are collinear.
#[must_use]
fn is_left(a: Vec2, b: Vec2, p: Vec2) -> f32 {
    b.sub(a).cross(p.sub(a))
}

/// The winding number of a closed `polygon` ring around `p`.
///
/// This is the classic integer winding count: each edge that crosses the
/// horizontal line through `p` upward while `p` is to its left contributes
/// `+1`, and each edge that crosses downward while `p` is to its right
/// contributes `-1`. The result is `0` when `p` is outside a simple ring, and
/// its magnitude counts how many times a self-overlapping ring wraps `p`. A
/// polygon with fewer than three vertices always winds `0`.
#[must_use]
pub fn winding_number(p: Vec2, polygon: &[Vec2]) -> i32 {
    let n = polygon.len();
    if n < 3 {
        return 0;
    }
    let mut wn = 0i32;
    let mut i = 0;
    while i < n {
        let a = polygon[i];
        let b = polygon[if i + 1 == n { 0 } else { i + 1 }];
        if a.y <= p.y {
            if b.y > p.y && is_left(a, b, p) > 0.0 {
                wn += 1;
            }
        } else if b.y <= p.y && is_left(a, b, p) < 0.0 {
            wn -= 1;
        }
        i += 1;
    }
    wn
}

/// Tests whether `p` lies inside `polygon` under the non-zero winding rule
/// (the [`winding_number`] is not zero).
///
/// For a simple ring this matches the intuitive "inside"; for a
/// self-intersecting ring the overlapping core is still counted as inside,
/// which is what the signed field wants.
#[must_use]
pub fn point_in_polygon(p: Vec2, polygon: &[Vec2]) -> bool {
    winding_number(p, polygon) != 0
}

/// The signed distance from `p` to the closed `polygon` ring: negative inside,
/// positive outside, and `0` on the boundary.
///
/// The magnitude is [`polygon_boundary_distance`]; the sign comes from
/// [`point_in_polygon`] (non-zero winding). This is correct for convex and
/// concave rings alike. A degenerate polygon with fewer than three vertices
/// encloses no area, so the result is the (non-negative) boundary distance.
///
/// Samples on the boundary have a magnitude at or below [`CMP_EPS`]; those are
/// snapped to exactly `0.0` so a caller never has to reason about a boundary
/// point's sign.
#[must_use]
pub fn polygon_signed_distance(p: Vec2, polygon: &[Vec2]) -> f32 {
    let unsigned = polygon_boundary_distance(p, polygon);
    if unsigned <= CMP_EPS {
        return 0.0;
    }
    if polygon.len() < 3 {
        return unsigned;
    }
    if point_in_polygon(p, polygon) {
        -unsigned
    } else {
        unsigned
    }
}

/// Total `std430` byte size of a storage buffer holding `count` packed
/// [`Segment`]s, clamped up to a single element per the shared [`storage_bytes`]
/// rule (a `WebGPU` storage binding may not be zero-sized).
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(POLYLINE_SDF_2D_STD430_SIZE, count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Absolute tolerance for the test assertions; direct `==` on floating
    /// point is intentionally avoided.
    const TEST_EPS: f32 = 1.0e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= TEST_EPS
    }

    fn approx_vec(a: Vec2, b: Vec2) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y)
    }

    fn unit_square() -> [Vec2; 4] {
        [
            Vec2::new(-1.0, -1.0),
            Vec2::new(1.0, -1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(-1.0, 1.0),
        ]
    }

    // CCW L-shaped concave polygon occupying x in [0,2] y in [0,1] plus
    // x in [0,1] y in [1,2].
    fn l_shape() -> [Vec2; 6] {
        [
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 2.0),
            Vec2::new(0.0, 2.0),
        ]
    }

    #[test]
    fn vec2_algebra_is_exact() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(4.0, 6.0);
        assert!(approx_vec(a.add(b), Vec2::new(5.0, 8.0)));
        assert!(approx_vec(b.sub(a), Vec2::new(3.0, 4.0)));
        assert!(approx_vec(a.scale(3.0), Vec2::new(3.0, 6.0)));
        assert!(approx(a.dot(b), 16.0));
        assert!(approx(a.cross(b), 1.0 * 6.0 - 2.0 * 4.0));
    }

    #[test]
    fn vec2_length_and_distance() {
        let a = Vec2::new(3.0, 4.0);
        assert!(approx(a.length_squared(), 25.0));
        assert!(approx(a.length(), 5.0));
        assert!(approx(Vec2::ZERO.distance(a), 5.0));
        assert!(approx(Vec2::ZERO.distance_squared(a), 25.0));
    }

    #[test]
    fn vec2_min_max_are_component_wise() {
        let a = Vec2::new(1.0, -2.0);
        let b = Vec2::new(-1.0, 3.0);
        assert!(approx_vec(a.min(b), Vec2::new(-1.0, -2.0)));
        assert!(approx_vec(a.max(b), Vec2::new(1.0, 3.0)));
    }

    #[test]
    fn projection_t_clamps_before_start() {
        // p projects behind a, so t clamps to 0.
        let t = segment_projection_t(
            Vec2::new(-2.0, 0.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
        );
        assert!(approx(t, 0.0));
    }

    #[test]
    fn projection_t_clamps_after_end() {
        let t = segment_projection_t(
            Vec2::new(9.0, 0.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
        );
        assert!(approx(t, 1.0));
    }

    #[test]
    fn projection_t_interior_is_fraction() {
        let t = segment_projection_t(
            Vec2::new(1.0, 5.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
        );
        assert!(approx(t, 0.25));
    }

    #[test]
    fn segment_distance_on_segment_is_zero() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(4.0, 0.0);
        assert!(approx(segment_distance(Vec2::new(2.0, 0.0), a, b), 0.0));
    }

    #[test]
    fn segment_distance_at_endpoints_is_zero() {
        let a = Vec2::new(1.0, 1.0);
        let b = Vec2::new(4.0, 5.0);
        assert!(approx(segment_distance(a, a, b), 0.0));
        assert!(approx(segment_distance(b, a, b), 0.0));
    }

    #[test]
    fn segment_distance_perpendicular_offset() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(4.0, 0.0);
        assert!(approx(segment_distance(Vec2::new(2.0, 3.0), a, b), 3.0));
    }

    #[test]
    fn segment_distance_clamps_to_start_endpoint() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(4.0, 0.0);
        // Foot of perpendicular is behind a; nearest point is a itself.
        assert!(approx(segment_distance(Vec2::new(-3.0, 4.0), a, b), 5.0));
    }

    #[test]
    fn segment_distance_clamps_to_end_endpoint() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(4.0, 0.0);
        assert!(approx(segment_distance(Vec2::new(7.0, 4.0), a, b), 5.0));
    }

    #[test]
    fn segment_distance_is_symmetric_in_endpoints() {
        let p = Vec2::new(2.0, 3.0);
        let a = Vec2::new(-1.0, 0.0);
        let b = Vec2::new(5.0, 1.0);
        assert!(approx(segment_distance(p, a, b), segment_distance(p, b, a)));
    }

    #[test]
    fn segment_distance_degenerate_zero_length() {
        let a = Vec2::new(2.0, 2.0);
        // Zero-length segment collapses to distance from p to the point a.
        assert!(approx(segment_distance(Vec2::new(2.0, 5.0), a, a), 3.0));
    }

    #[test]
    fn closest_point_interior() {
        let c = closest_point_on_segment(
            Vec2::new(1.0, 5.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
        );
        assert!(approx_vec(c, Vec2::new(1.0, 0.0)));
    }

    #[test]
    fn closest_point_clamps_to_start() {
        let c = closest_point_on_segment(
            Vec2::new(-5.0, 2.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
        );
        assert!(approx_vec(c, Vec2::new(0.0, 0.0)));
    }

    #[test]
    fn closest_point_clamps_to_end() {
        let c = closest_point_on_segment(
            Vec2::new(9.0, 2.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
        );
        assert!(approx_vec(c, Vec2::new(4.0, 0.0)));
    }

    #[test]
    fn closest_point_degenerate_returns_start() {
        let a = Vec2::new(7.0, -3.0);
        let c = closest_point_on_segment(Vec2::new(0.0, 0.0), a, a);
        assert!(approx_vec(c, a));
    }

    #[test]
    fn polyline_distance_empty_is_infinite() {
        assert!(polyline_distance(Vec2::ZERO, &[]).is_infinite());
    }

    #[test]
    fn polyline_distance_single_vertex() {
        let v = vec![Vec2::new(3.0, 4.0)];
        assert!(approx(polyline_distance(Vec2::ZERO, &v), 5.0));
    }

    #[test]
    fn polyline_distance_picks_nearest_segment() {
        // An open "staircase" path; the query is nearest the middle segment.
        let path = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(4.0, 4.0),
            Vec2::new(8.0, 4.0),
        ];
        // Point just above the first segment.
        assert!(approx(polyline_distance(Vec2::new(2.0, 1.0), &path), 1.0));
        // Point to the right of the vertical middle segment.
        assert!(approx(polyline_distance(Vec2::new(6.0, 2.0), &path), 2.0));
    }

    #[test]
    fn polyline_distance_on_vertex_is_zero() {
        let path = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(4.0, 4.0),
        ];
        assert!(approx(polyline_distance(Vec2::new(4.0, 0.0), &path), 0.0));
    }

    #[test]
    fn polyline_distance_on_edge_is_zero() {
        let path = vec![Vec2::new(0.0, 0.0), Vec2::new(4.0, 0.0)];
        assert!(approx(polyline_distance(Vec2::new(1.5, 0.0), &path), 0.0));
    }

    #[test]
    fn polygon_signed_distance_inside_is_negative() {
        let sq = unit_square();
        let d = polygon_signed_distance(Vec2::new(0.0, 0.0), &sq);
        // Center of the 2x2 square is 1.0 from the nearest edge, inside.
        assert!(approx(d, -1.0));
    }

    #[test]
    fn polygon_signed_distance_outside_is_positive() {
        let sq = unit_square();
        let d = polygon_signed_distance(Vec2::new(3.0, 0.0), &sq);
        // 2.0 to the right edge at x = 1.
        assert!(approx(d, 2.0));
    }

    #[test]
    fn polygon_signed_distance_on_vertex_is_zero() {
        let sq = unit_square();
        assert!(approx(
            polygon_signed_distance(Vec2::new(1.0, 1.0), &sq),
            0.0
        ));
    }

    #[test]
    fn polygon_signed_distance_on_edge_is_zero() {
        let sq = unit_square();
        assert!(approx(
            polygon_signed_distance(Vec2::new(1.0, 0.3), &sq),
            0.0
        ));
    }

    #[test]
    fn polygon_signed_distance_is_symmetric_about_center() {
        let sq = unit_square();
        let right = polygon_signed_distance(Vec2::new(2.0, 0.0), &sq);
        let left = polygon_signed_distance(Vec2::new(-2.0, 0.0), &sq);
        let up = polygon_signed_distance(Vec2::new(0.0, 2.0), &sq);
        let down = polygon_signed_distance(Vec2::new(0.0, -2.0), &sq);
        assert!(approx(right, left));
        assert!(approx(up, down));
        assert!(approx(right, up));
    }

    #[test]
    fn polygon_signed_distance_convex_interior_offcenter() {
        let sq = unit_square();
        // Nearest edge is the right one at x = 1, so depth is 1 - 0.5 = 0.5.
        let d = polygon_signed_distance(Vec2::new(0.5, 0.0), &sq);
        assert!(approx(d, -0.5));
    }

    #[test]
    fn polygon_signed_distance_concave_interior_is_negative() {
        let poly = l_shape();
        let d = polygon_signed_distance(Vec2::new(0.5, 0.5), &poly);
        assert!(d < 0.0);
        // Nearest boundary is the left/bottom edge, half a unit away.
        assert!(approx(d, -0.5));
    }

    #[test]
    fn polygon_signed_distance_concave_notch_is_positive() {
        let poly = l_shape();
        // (1.5, 1.5) sits in the notch cut out of the L; it is outside.
        let d = polygon_signed_distance(Vec2::new(1.5, 1.5), &poly);
        assert!(d > 0.0);
        // Nearest edges are x = 1 (for y in [1,2]) and y = 1 (for x in [1,2]);
        // distance to either is 0.5.
        assert!(approx(d, 0.5));
    }

    #[test]
    fn polygon_boundary_distance_ignores_sign() {
        let sq = unit_square();
        let inside = polygon_boundary_distance(Vec2::new(0.0, 0.0), &sq);
        assert!(approx(inside, 1.0));
        let outside = polygon_boundary_distance(Vec2::new(3.0, 0.0), &sq);
        assert!(approx(outside, 2.0));
    }

    #[test]
    fn winding_number_counts_inside_and_outside() {
        let sq = unit_square();
        assert_eq!(winding_number(Vec2::new(0.0, 0.0), &sq), 1);
        assert_eq!(winding_number(Vec2::new(5.0, 5.0), &sq), 0);
    }

    #[test]
    fn point_in_polygon_matches_containment() {
        let poly = l_shape();
        assert!(point_in_polygon(Vec2::new(0.5, 0.5), &poly));
        assert!(point_in_polygon(Vec2::new(0.5, 1.5), &poly));
        assert!(!point_in_polygon(Vec2::new(1.5, 1.5), &poly));
        assert!(!point_in_polygon(Vec2::new(-1.0, -1.0), &poly));
    }

    #[test]
    fn degenerate_polygon_has_no_interior() {
        // Two vertices cannot enclose area; the signed distance is the
        // non-negative boundary distance.
        let seg = vec![Vec2::new(0.0, 0.0), Vec2::new(4.0, 0.0)];
        let d = polygon_signed_distance(Vec2::new(2.0, 3.0), &seg);
        assert!(d >= 0.0);
        assert!(approx(d, 3.0));
        assert_eq!(winding_number(Vec2::new(2.0, 0.5), &seg), 0);
    }

    #[test]
    fn polygon_signed_distance_is_deterministic() {
        let poly = l_shape();
        let p = Vec2::new(0.75, 1.25);
        let first = polygon_signed_distance(p, &poly);
        let second = polygon_signed_distance(p, &poly);
        assert_eq!(first.to_bits(), second.to_bits());
    }

    #[test]
    fn segment_helpers_match_free_functions() {
        let seg = Segment::new(Vec2::new(0.0, 0.0), Vec2::new(4.0, 0.0));
        let p = Vec2::new(2.0, 3.0);
        assert!(approx(
            seg.distance_to(p),
            segment_distance(p, seg.a, seg.b)
        ));
        assert!(approx_vec(
            seg.closest_point(p),
            closest_point_on_segment(p, seg.a, seg.b)
        ));
        assert!(approx(seg.length(), 4.0));
        assert!(approx(seg.length_squared(), 16.0));
        assert!(approx_vec(seg.direction(), Vec2::new(4.0, 0.0)));
    }

    #[test]
    fn segment_to_std430_round_trips_endpoints() {
        let seg = Segment::new(Vec2::new(1.5, -2.0), Vec2::new(3.25, 4.75));
        let bytes = seg.to_std430();
        assert_eq!(bytes.len(), POLYLINE_SDF_2D_STD430_SIZE);
        let ax = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let ay = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let bx = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let by = f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        assert!(approx(ax, 1.5));
        assert!(approx(ay, -2.0));
        assert!(approx(bx, 3.25));
        assert!(approx(by, 4.75));
    }

    #[test]
    fn std430_size_is_multiple_of_sixteen() {
        assert_eq!(POLYLINE_SDF_2D_STD430_SIZE, 16);
        assert_eq!(POLYLINE_SDF_2D_STD430_SIZE % 16, 0);
    }

    #[test]
    fn gpu_storage_bytes_scales_and_clamps() {
        assert_eq!(gpu_storage_bytes(0), POLYLINE_SDF_2D_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(1), POLYLINE_SDF_2D_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(8), POLYLINE_SDF_2D_STD430_SIZE * 8);
        assert_eq!(gpu_storage_bytes(8) % 16, 0);
    }
}
