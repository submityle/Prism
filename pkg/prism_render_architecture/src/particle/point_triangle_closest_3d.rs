//! Closest point on a 3D triangle to a query point, with barycentric weights
//! and squared distance (design §8.2, §10, §14).
//!
//! Several particle stages need the *nearest surface point* of a triangle to an
//! arbitrary point in space: a decal or trail footprint snapping onto a mesh
//! face, a collision proxy resolving a particle against a triangle soup, a
//! spawn position clamped back onto its source surface, or a proximity query in
//! a broad phase. This module owns that single, `CPU`-verifiable geometry
//! contract: given a point `p` and a triangle `(a, b, c)`, it returns the point
//! on the (closed, filled) triangle nearest to `p`, the barycentric weights of
//! that point, and the squared distance between them.
//!
//! # Strict scope — how this differs from its siblings
//! This is the **3D space point-to-triangle** nearest-point query and nothing
//! else. It is deliberately disjoint from the other proximity/geometry modules
//! and neither imports nor reconstructs their types:
//!
//! * [`crate::particle::ray_triangle`] answers a **ray-vs-triangle
//!   intersection** question (does a ray pierce the face, and at what `t`); it
//!   is a hit test along a direction, not a nearest-point-in-space query.
//! * [`crate::particle::segment_closest_point_3d`] answers **segment-vs-segment
//!   closest points** (two 1D primitives in 3D); this module's primitive is a
//!   2D filled triangle versus a 0D point.
//! * [`crate::particle::triangle_circumcircle`] computes a **2D circumcircle**
//!   (center/radius for Delaunay-style meshing) in the plane; it has no notion
//!   of a 3D query point or a nearest surface point.
//! * [`crate::particle::barycentric_coord`] converts a point that is assumed to
//!   lie *in the triangle's plane* into weights and interpolates attributes; it
//!   does **not** clamp an off-surface point back onto the closed triangle. This
//!   module does exactly that clamp (via Voronoi-region case analysis) and
//!   reports the squared distance the clamp introduced.
//!
//! # Algorithm
//! The implementation is the classic closed-form Voronoi-region solution from
//! Christer Ericson's *Real-Time Collision Detection* (§5.1.5,
//! `ClosestPtPointTriangle`), re-derived here rather than copied. The plane of
//! the triangle is partitioned into seven regions — three vertex regions, three
//! edge regions and the interior face — and a handful of dot-product sign tests
//! route the query into the region that owns it. Each region yields the nearest
//! point directly as a barycentric blend, so no iterative search is needed.
//!
//! A degenerate (collinear / zero-area) triangle would make the face
//! denominator vanish, so it is detected up front through the squared length of
//! the triangle normal and handled by falling back to the minimum over the three
//! edge segments (which themselves collapse gracefully to point-point queries
//! when a vertex is repeated).
//!
//! # No transcendental math, no float equality
//! Every step uses only `+ - * /`, [`f32::sqrt`] (through the squared-distance
//! helpers, only where a length is genuinely needed), [`f32::abs`],
//! [`f32::min`], [`f32::max`] and [`f32::clamp`]. No transcendental function is
//! called and no `==` / `!=` comparison on an `f32` ever appears: degeneracy and
//! division safety are decided against explicit epsilon constants, so this `CPU`
//! reference agrees bit for bit with a future `GPU` (`WESL`) kernel that packs
//! the same results through the `std430` helpers in
//! [`crate::particle::gpu_layout`].

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Squared triangle-normal length below which the triangle is treated as
/// degenerate (collinear or zero-area). The normal length is twice the triangle
/// area, so this is a squared-area threshold; below it the face branch would
/// divide by (near) zero and the edge fallback is used instead.
const AREA_EPS_SQ: f32 = 1.0e-14;

/// Magnitude below which a division denominator (an edge squared length or an
/// edge parameter denominator) is treated as zero, so the parameter collapses
/// to the region's start vertex instead of producing `NaN`/`inf`.
const DENOM_EPS: f32 = 1.0e-20;

/// A hand-rolled three-component vector.
///
/// The crate is a dependency-free contracts crate and each geometry module owns
/// its own small vector type (mirroring [`crate::particle::segment_closest_point_3d`])
/// so the math stays local, `no_std`-friendly and free of transcendental calls.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
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

    /// Uniform vector with every component set to `v`.
    #[must_use]
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
    }

    /// Component-wise sum `self + rhs`.
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

    /// Dot (inner) product.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Cross product `self × rhs`.
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    /// Squared Euclidean length.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length (the only place a `sqrt` is reachable).
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Squared distance between two points.
    #[must_use]
    pub fn distance_squared(self, rhs: Self) -> f32 {
        self.minus(rhs).length_squared()
    }

    /// Euclidean distance between two points.
    #[must_use]
    pub fn distance(self, rhs: Self) -> f32 {
        self.minus(rhs).length()
    }
}

/// The result of a point-to-triangle nearest-point query.
///
/// `bary` are the barycentric weights `(u, v, w)` of [`Self::point`] with
/// respect to the triangle corners `(a, b, c)`, so
/// `point == a·u + b·v + c·w` and, for a non-degenerate triangle, `u + v + w`
/// is (numerically) `1` with every weight in `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClosestPointOnTriangle {
    /// The point on the closed, filled triangle nearest to the query point.
    pub point: Vec3,
    /// Barycentric weights `(u, v, w)` of [`Self::point`] over `(a, b, c)`.
    pub bary: [f32; 3],
    /// Squared distance between the query point and [`Self::point`].
    pub distance_squared: f32,
}

/// Byte stride of one `std430`-packed [`ClosestPointOnTriangle`] record: the
/// nearest point (padded to a `vec4`, with the squared distance in `.w`) plus
/// the barycentric weights (padded to a `vec4`, with `0` in `.w`). Two aligned
/// `vec4`s, so the stride is a multiple of 16 and needs no extra padding.
pub const CLOSEST_STRIDE: usize = 2 * VEC4_STRIDE;

/// Total `std430` byte size of a storage buffer holding `count`
/// [`ClosestPointOnTriangle`] records, clamped up to one element (a `WebGPU`
/// storage binding may not be zero-sized).
#[must_use]
pub fn std430_bytes(count: usize) -> usize {
    storage_bytes(CLOSEST_STRIDE, count)
}

/// Closest point on the *segment* `[a, b]` to `p`, returning the clamped
/// parameter `t ∈ [0, 1]` (`point = a + t·(b − a)`) and that point. A
/// zero-length segment collapses to `a`.
fn closest_on_segment(p: Vec3, a: Vec3, b: Vec3) -> (f32, Vec3) {
    let ab = b.minus(a);
    let denom = ab.length_squared();
    if denom <= DENOM_EPS {
        return (0.0, a);
    }
    let t = (ab.dot(p.minus(a)) / denom).clamp(0.0, 1.0);
    (t, a.plus(ab.scale(t)))
}

/// Degenerate (collinear / zero-area) fallback: the triangle has no interior,
/// so the nearest point is the closest of the three edge segments. Barycentric
/// weights are taken from the winning edge (the opposite corner weight is `0`).
fn closest_on_degenerate(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> ClosestPointOnTriangle {
    let (t_ab, q_ab) = closest_on_segment(p, a, b);
    let (t_bc, q_bc) = closest_on_segment(p, b, c);
    let (t_ca, q_ca) = closest_on_segment(p, c, a);

    let d_ab = p.distance_squared(q_ab);
    let d_bc = p.distance_squared(q_bc);
    let d_ca = p.distance_squared(q_ca);

    // Strict `<` keeps the earliest edge (AB, then BC, then CA) on a tie.
    if d_ab <= d_bc && d_ab <= d_ca {
        ClosestPointOnTriangle {
            point: q_ab,
            bary: [1.0 - t_ab, t_ab, 0.0],
            distance_squared: d_ab,
        }
    } else if d_bc <= d_ca {
        ClosestPointOnTriangle {
            point: q_bc,
            bary: [0.0, 1.0 - t_bc, t_bc],
            distance_squared: d_bc,
        }
    } else {
        // CA runs from `c` to `a`: weight on `a` is `t_ca`, weight on `c` is `1 - t_ca`.
        ClosestPointOnTriangle {
            point: q_ca,
            bary: [t_ca, 0.0, 1.0 - t_ca],
            distance_squared: d_ca,
        }
    }
}

/// Builds a result from a known nearest point and its barycentric weights,
/// filling in the squared distance to the query point.
fn finish(p: Vec3, point: Vec3, u: f32, v: f32, w: f32) -> ClosestPointOnTriangle {
    ClosestPointOnTriangle {
        point,
        bary: [u, v, w],
        distance_squared: p.distance_squared(point),
    }
}

/// Returns the point on the closed, filled triangle `(a, b, c)` nearest to `p`,
/// together with its barycentric weights and the squared distance to `p`.
///
/// The seven Voronoi regions of the triangle are tested in turn (vertices `a`,
/// `b`, `c`; edges `ab`, `ca`, `bc`; then the interior face). A degenerate
/// (collinear / zero-area) triangle is detected first and routed to the
/// edge-minimum fallback so the face division never sees a (near) zero
/// denominator.
#[must_use]
pub fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> ClosestPointOnTriangle {
    let ab = b.minus(a);
    let ac = c.minus(a);

    // Degenerate guard: the normal length squared is `(2·area)²`.
    let normal = ab.cross(ac);
    if normal.length_squared() <= AREA_EPS_SQ {
        return closest_on_degenerate(p, a, b, c);
    }

    // Vertex region outside A.
    let ap = p.minus(a);
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return finish(p, a, 1.0, 0.0, 0.0);
    }

    // Vertex region outside B.
    let bp = p.minus(b);
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return finish(p, b, 0.0, 1.0, 0.0);
    }

    // Edge region of AB: projection of P onto AB.
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let denom = d1 - d3;
        let v = if denom > DENOM_EPS { d1 / denom } else { 0.0 };
        let point = a.plus(ab.scale(v));
        return finish(p, point, 1.0 - v, v, 0.0);
    }

    // Vertex region outside C.
    let cp = p.minus(c);
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return finish(p, c, 0.0, 0.0, 1.0);
    }

    // Edge region of AC: projection of P onto AC.
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let denom = d2 - d6;
        let w = if denom > DENOM_EPS { d2 / denom } else { 0.0 };
        let point = a.plus(ac.scale(w));
        return finish(p, point, 1.0 - w, 0.0, w);
    }

    // Edge region of BC: projection of P onto BC.
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let denom = (d4 - d3) + (d5 - d6);
        let w = if denom > DENOM_EPS {
            (d4 - d3) / denom
        } else {
            0.0
        };
        let point = b.plus(c.minus(b).scale(w));
        return finish(p, point, 0.0, 1.0 - w, w);
    }

    // Interior face region.
    let sum = va + vb + vc;
    if sum <= DENOM_EPS {
        // Should not happen for a non-degenerate triangle, but guard the divide.
        return closest_on_degenerate(p, a, b, c);
    }
    let denom = 1.0 / sum;
    let v = vb * denom;
    let w = vc * denom;
    let u = 1.0 - v - w;
    let point = a.plus(ab.scale(v)).plus(ac.scale(w));
    finish(p, point, u, v, w)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-5;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() <= EPS, "expected {a} ≈ {b}");
    }

    fn approx_vec(a: Vec3, b: Vec3) {
        approx(a.x, b.x);
        approx(a.y, b.y);
        approx(a.z, b.z);
    }

    /// The canonical unit right triangle in the `z = 0` plane.
    fn tri() -> (Vec3, Vec3, Vec3) {
        (
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        )
    }

    // ----- interior / face region -------------------------------------------

    #[test]
    fn interior_point_projects_onto_face() {
        let (a, b, c) = tri();
        let p = Vec3::new(0.25, 0.25, 0.5);
        let r = closest_point_on_triangle(p, a, b, c);
        approx_vec(r.point, Vec3::new(0.25, 0.25, 0.0));
        approx(r.distance_squared, 0.25);
    }

    #[test]
    fn interior_barycentric_matches_projection() {
        let (a, b, c) = tri();
        let r = closest_point_on_triangle(Vec3::new(0.25, 0.25, 0.5), a, b, c);
        approx(r.bary[0], 0.5);
        approx(r.bary[1], 0.25);
        approx(r.bary[2], 0.25);
    }

    #[test]
    fn interior_barycentric_sum_is_one() {
        let (a, b, c) = tri();
        let r = closest_point_on_triangle(Vec3::new(0.3, 0.4, -2.0), a, b, c);
        approx(r.bary[0] + r.bary[1] + r.bary[2], 1.0);
    }

    #[test]
    fn interior_barycentric_all_in_unit_range() {
        let (a, b, c) = tri();
        let r = closest_point_on_triangle(Vec3::new(0.2, 0.3, 7.0), a, b, c);
        for w in r.bary {
            assert!((0.0..=1.0).contains(&w), "weight {w} out of range");
        }
    }

    #[test]
    fn face_point_reconstructs_from_barycentric() {
        let (a, b, c) = tri();
        let r = closest_point_on_triangle(Vec3::new(0.2, 0.5, 3.0), a, b, c);
        let rebuilt = a
            .scale(r.bary[0])
            .plus(b.scale(r.bary[1]))
            .plus(c.scale(r.bary[2]));
        approx_vec(rebuilt, r.point);
    }

    #[test]
    fn closest_point_lies_in_plane_for_interior() {
        let (a, b, c) = tri();
        let r = closest_point_on_triangle(Vec3::new(0.3, 0.3, 9.0), a, b, c);
        approx(r.point.z, 0.0);
    }

    #[test]
    fn point_on_face_returns_itself() {
        let (a, b, c) = tri();
        let on_face = Vec3::new(0.25, 0.5, 0.0);
        let r = closest_point_on_triangle(on_face, a, b, c);
        approx_vec(r.point, on_face);
        approx(r.distance_squared, 0.0);
    }

    #[test]
    fn above_and_below_face_are_symmetric() {
        let (a, b, c) = tri();
        let above = closest_point_on_triangle(Vec3::new(0.25, 0.25, 0.5), a, b, c);
        let below = closest_point_on_triangle(Vec3::new(0.25, 0.25, -0.5), a, b, c);
        approx_vec(above.point, below.point);
        approx(above.distance_squared, below.distance_squared);
    }

    // ----- vertex regions -----------------------------------------------------

    #[test]
    fn closest_to_vertex_a() {
        let (a, b, c) = tri();
        let r = closest_point_on_triangle(Vec3::new(-1.0, -1.0, 0.0), a, b, c);
        approx_vec(r.point, a);
        approx(r.bary[0], 1.0);
        approx(r.distance_squared, 2.0);
    }

    #[test]
    fn closest_to_vertex_b() {
        let (a, b, c) = tri();
        let r = closest_point_on_triangle(Vec3::new(2.0, -1.0, 0.0), a, b, c);
        approx_vec(r.point, b);
        approx(r.bary[1], 1.0);
        approx(r.distance_squared, 2.0);
    }

    #[test]
    fn closest_to_vertex_c() {
        let (a, b, c) = tri();
        let r = closest_point_on_triangle(Vec3::new(-1.0, 2.0, 0.0), a, b, c);
        approx_vec(r.point, c);
        approx(r.bary[2], 1.0);
        approx(r.distance_squared, 2.0);
    }

    #[test]
    fn vertex_region_barycentric_is_unit() {
        let (a, b, c) = tri();
        let r = closest_point_on_triangle(Vec3::new(-3.0, -4.0, 1.0), a, b, c);
        approx(r.bary[0], 1.0);
        approx(r.bary[1], 0.0);
        approx(r.bary[2], 0.0);
    }

    #[test]
    fn point_exactly_on_vertex_has_zero_distance() {
        let (a, b, c) = tri();
        let r = closest_point_on_triangle(c, a, b, c);
        approx_vec(r.point, c);
        approx(r.distance_squared, 0.0);
    }

    // ----- edge regions -------------------------------------------------------

    #[test]
    fn closest_on_edge_ab() {
        let (a, b, c) = tri();
        let r = closest_point_on_triangle(Vec3::new(0.5, -1.0, 0.0), a, b, c);
        approx_vec(r.point, Vec3::new(0.5, 0.0, 0.0));
        approx(r.bary[0], 0.5);
        approx(r.bary[1], 0.5);
        approx(r.bary[2], 0.0);
        approx(r.distance_squared, 1.0);
    }

    #[test]
    fn closest_on_edge_ca() {
        let (a, b, c) = tri();
        let r = closest_point_on_triangle(Vec3::new(-1.0, 0.5, 0.0), a, b, c);
        approx_vec(r.point, Vec3::new(0.0, 0.5, 0.0));
        approx(r.bary[0], 0.5);
        approx(r.bary[1], 0.0);
        approx(r.bary[2], 0.5);
        approx(r.distance_squared, 1.0);
    }

    #[test]
    fn closest_on_edge_bc() {
        let (a, b, c) = tri();
        let r = closest_point_on_triangle(Vec3::new(1.0, 1.0, 0.0), a, b, c);
        approx_vec(r.point, Vec3::new(0.5, 0.5, 0.0));
        approx(r.bary[0], 0.0);
        approx(r.bary[1], 0.5);
        approx(r.bary[2], 0.5);
        approx(r.distance_squared, 0.5);
    }

    #[test]
    fn edge_point_reconstructs_from_barycentric() {
        let (a, b, c) = tri();
        let r = closest_point_on_triangle(Vec3::new(0.5, -2.0, 4.0), a, b, c);
        let rebuilt = a
            .scale(r.bary[0])
            .plus(b.scale(r.bary[1]))
            .plus(c.scale(r.bary[2]));
        approx_vec(rebuilt, r.point);
    }

    #[test]
    fn edge_projection_is_off_plane_symmetric() {
        let (a, b, c) = tri();
        let up = closest_point_on_triangle(Vec3::new(0.5, -1.0, 0.7), a, b, c);
        let down = closest_point_on_triangle(Vec3::new(0.5, -1.0, -0.7), a, b, c);
        approx_vec(up.point, down.point);
        approx(up.distance_squared, down.distance_squared);
    }

    // ----- distance consistency ----------------------------------------------

    #[test]
    fn distance_squared_matches_point_distance() {
        let (a, b, c) = tri();
        let p = Vec3::new(0.9, 0.9, 1.3);
        let r = closest_point_on_triangle(p, a, b, c);
        approx(r.distance_squared, p.distance_squared(r.point));
    }

    #[test]
    fn distance_squared_matches_manual_height() {
        let (a, b, c) = tri();
        // Directly above the centroid: distance is purely the z height.
        let p = Vec3::new(1.0 / 3.0, 1.0 / 3.0, 2.0);
        let r = closest_point_on_triangle(p, a, b, c);
        approx(r.distance_squared, 4.0);
    }

    #[test]
    fn distance_helper_agrees_with_squared() {
        let (a, b, c) = tri();
        let p = Vec3::new(-1.0, 0.5, 0.0);
        let r = closest_point_on_triangle(p, a, b, c);
        approx(
            p.distance(r.point) * p.distance(r.point),
            r.distance_squared,
        );
    }

    // ----- degenerate triangles ----------------------------------------------

    #[test]
    fn degenerate_collinear_falls_back_to_segment() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(2.0, 0.0, 0.0);
        let r = closest_point_on_triangle(Vec3::new(0.5, 1.0, 0.0), a, b, c);
        approx_vec(r.point, Vec3::new(0.5, 0.0, 0.0));
        approx(r.distance_squared, 1.0);
    }

    #[test]
    fn degenerate_all_vertices_coincident() {
        let a = Vec3::new(3.0, 3.0, 3.0);
        let r = closest_point_on_triangle(Vec3::ZERO, a, a, a);
        approx_vec(r.point, a);
        approx(r.distance_squared, 27.0);
    }

    #[test]
    fn degenerate_two_vertices_coincident() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = a;
        let c = Vec3::new(1.0, 0.0, 0.0);
        let r = closest_point_on_triangle(Vec3::new(0.5, 1.0, 0.0), a, b, c);
        approx_vec(r.point, Vec3::new(0.5, 0.0, 0.0));
        approx(r.distance_squared, 1.0);
    }

    #[test]
    fn degenerate_barycentric_is_finite_and_bounded() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 0.0, 0.0);
        let c = Vec3::new(4.0, 0.0, 0.0);
        let r = closest_point_on_triangle(Vec3::new(1.0, 5.0, 0.0), a, b, c);
        for w in r.bary {
            assert!(w.is_finite(), "weight {w} is not finite");
            assert!((-EPS..=1.0 + EPS).contains(&w), "weight {w} out of range");
        }
    }

    // ----- determinism --------------------------------------------------------

    #[test]
    fn deterministic_point_to_bits() {
        let (a, b, c) = tri();
        let p = Vec3::new(0.137, 0.951, 0.618);
        let r1 = closest_point_on_triangle(p, a, b, c);
        let r2 = closest_point_on_triangle(p, a, b, c);
        assert_eq!(r1.point.x.to_bits(), r2.point.x.to_bits());
        assert_eq!(r1.point.y.to_bits(), r2.point.y.to_bits());
        assert_eq!(r1.point.z.to_bits(), r2.point.z.to_bits());
    }

    #[test]
    fn deterministic_barycentric_and_distance_to_bits() {
        let (a, b, c) = tri();
        let p = Vec3::new(-0.42, 0.73, -1.9);
        let r1 = closest_point_on_triangle(p, a, b, c);
        let r2 = closest_point_on_triangle(p, a, b, c);
        assert_eq!(r1.bary[0].to_bits(), r2.bary[0].to_bits());
        assert_eq!(r1.bary[1].to_bits(), r2.bary[1].to_bits());
        assert_eq!(r1.bary[2].to_bits(), r2.bary[2].to_bits());
        assert_eq!(r1.distance_squared.to_bits(), r2.distance_squared.to_bits());
    }

    // ----- std430 layout ------------------------------------------------------

    #[test]
    fn std430_stride_is_multiple_of_16() {
        assert_eq!(CLOSEST_STRIDE % 16, 0);
        assert_eq!(CLOSEST_STRIDE, 32);
    }

    #[test]
    fn std430_bytes_scale_with_count() {
        assert_eq!(std430_bytes(1), 32);
        assert_eq!(std430_bytes(4), 128);
        assert_eq!(std430_bytes(10) % 16, 0);
    }

    #[test]
    fn std430_empty_reserves_one_element() {
        assert_eq!(std430_bytes(0), CLOSEST_STRIDE);
    }

    // ----- general placement ---------------------------------------------------

    #[test]
    fn non_axis_aligned_triangle_interior() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 2.0, 0.0);
        let c = Vec3::new(0.0, 0.0, 3.0);
        // Centroid lies on the face; querying it must return it exactly.
        let centroid = a.plus(b).plus(c).scale(1.0 / 3.0);
        let r = closest_point_on_triangle(centroid, a, b, c);
        approx_vec(r.point, centroid);
        approx(r.distance_squared, 0.0);
        approx(r.bary[0] + r.bary[1] + r.bary[2], 1.0);
    }

    #[test]
    fn nearest_point_is_no_farther_than_any_vertex() {
        let (a, b, c) = tri();
        let p = Vec3::new(0.6, -0.3, 0.4);
        let r = closest_point_on_triangle(p, a, b, c);
        let min_vertex = p
            .distance_squared(a)
            .min(p.distance_squared(b))
            .min(p.distance_squared(c));
        assert!(r.distance_squared <= min_vertex + EPS);
    }
}
