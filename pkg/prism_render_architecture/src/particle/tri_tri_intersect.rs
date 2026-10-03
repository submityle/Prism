//! Möller triangle-triangle intersection: a boolean overlap test between two
//! triangles in 3D, faithful to Möller's 1997 "A Fast Triangle-Triangle
//! Intersection Test" (design §14).
//!
//! Particle collision proxies, decal clipping, and mesh self-overlap checks all
//! need to know whether two triangles share any point in space. This module
//! owns that single `CPU`-verifiable predicate: given two triangles it returns
//! `true` when their closed (filled) faces touch or cross, and `false` when they
//! are disjoint. No intersection point, segment, or barycentric data is
//! produced — only the yes/no answer.
//!
//! # Algorithm
//! The non-coplanar path is Möller's interval-overlap test. Each triangle's
//! vertices are scored by their signed distance to the *other* triangle's
//! supporting plane. If all three vertices of either triangle lie strictly on
//! one side of the other plane, the triangles cannot meet and the test rejects
//! early. Otherwise both triangles straddle the line where the two planes
//! intersect; each triangle is projected onto that line to obtain a scalar
//! interval, and the triangles overlap exactly when the two intervals overlap.
//! When all of one triangle's signed distances vanish the triangles are
//! coplanar, and the test drops to a 2D triangle-triangle overlap in the
//! axis plane that best preserves area (edge-edge crossings plus a
//! containment test), matching Möller's `coplanar_tri_tri` fallback.
//!
//! # Strict scope — how this differs from its siblings
//! This is the **triangle-vs-triangle** boolean test and nothing else. It does
//! not import or reconstruct the other geometry contracts:
//! * [`crate::particle::ray_triangle`] answers a **ray-vs-triangle** hit test
//!   (does a directed ray pierce a face, and at what `t`); it has a ray, not a
//!   second triangle.
//! * `segment_triangle_intersect` (the sibling **segment-vs-triangle** test)
//!   clips a finite 1D segment against a face; its first primitive is a segment,
//!   not a triangle.
//! * [`crate::particle::point_triangle_closest_3d`] returns the **nearest
//!   surface point** of a triangle to a point and a squared distance; it is a
//!   proximity query, not an overlap predicate.
//! * [`crate::particle::barycentric_coord`] converts an in-plane point into
//!   **weights**; it neither takes a second triangle nor decides overlap.
//!
//! # No transcendental math, no float equality
//! Every routine is pure `+ - * /` arithmetic; the only non-arithmetic calls are
//! [`f32::abs`], [`f32::min`], and [`f32::max`]. There is no `sqrt`, `sin`,
//! `cos`, `atan`, `pow`, or any transcendental function, and no `==` / `!=`
//! comparison on an `f32`: every sign, degeneracy, and overlap decision is made
//! against the explicit [`EPS`] epsilon so this reference agrees with a future
//! `GPU` (`WESL`) kernel evaluating the same predicate. A degenerate
//! (near zero-area) triangle has no well-defined supporting plane, so the test
//! conservatively reports `false` for it rather than dividing by a near-zero
//! normal.

/// Magnitude below which a signed distance, an orientation determinant, or a
/// projected coordinate difference is treated as zero. Every sign test in this
/// module compares against this epsilon instead of using `==` on an `f32`.
pub const EPS: f32 = 1.0e-6;

/// Squared triangle-normal length below which a triangle is treated as
/// degenerate (collinear / zero-area) and reported as non-intersecting, because
/// its supporting plane — and therefore the whole interval test — is undefined.
pub const AREA_EPS_SQ: f32 = 1.0e-12;

/// A three-component vector in the right-handed space the triangle vertices are
/// expressed in.
///
/// Arithmetic methods are named `plus` / `minus` / `scale` (not the operator
/// trait names) to keep the closed-form algebra explicit and to avoid implying
/// a component-wise `Mul` on the cross / dot products.
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
    /// Builds a vector from its three components.
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Component-wise sum `self + other`.
    pub fn plus(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    /// Component-wise difference `self - other`.
    pub fn minus(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    /// Uniform scale of every component by `s`.
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    /// Euclidean dot product `self . other`.
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Right-handed cross product `self x other`.
    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Squared length `self . self`; avoids the `sqrt` a true length would need.
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Returns component `0 => x`, `1 => y`, anything else `=> z`. Used to read
    /// the axis chosen for the interval projection without a range loop.
    fn component(self, index: usize) -> f32 {
        match index {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }
}

/// A triangle given by its three corners `a`, `b`, `c`. Winding order is
/// irrelevant to the overlap predicate: flipping it only negates the supporting
/// plane normal, which cancels out of every sign test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tri {
    /// First corner.
    pub a: Vec3,
    /// Second corner.
    pub b: Vec3,
    /// Third corner.
    pub c: Vec3,
}

impl Tri {
    /// Builds a triangle from its three corners.
    pub const fn new(a: Vec3, b: Vec3, c: Vec3) -> Self {
        Self { a, b, c }
    }
}

/// Snaps a signed distance to exactly zero when it is within [`EPS`], so the
/// subsequent sign products and the coplanarity test are decided against the
/// epsilon rather than against raw round-off.
fn snap(value: f32) -> f32 {
    if value.abs() < EPS {
        0.0
    } else {
        value
    }
}

/// Returns `true` when the two closed triangles share at least one point.
///
/// Handles the general (planes crossing), coplanar, edge-sharing, and
/// vertex-touching cases. A degenerate triangle (near zero area) has no
/// supporting plane and is reported as non-intersecting.
pub fn tri_tri_intersect(t1: &Tri, t2: &Tri) -> bool {
    // Supporting plane of t2: normal n2 and offset d2 with n2 . x + d2 = 0.
    let n2 = t2.b.minus(t2.a).cross(t2.c.minus(t2.a));
    if n2.length_squared() < AREA_EPS_SQ {
        return false;
    }
    let d2 = -n2.dot(t2.a);

    // Signed distances of t1's vertices to t2's plane.
    let dv0 = snap(n2.dot(t1.a) + d2);
    let dv1 = snap(n2.dot(t1.b) + d2);
    let dv2 = snap(n2.dot(t1.c) + d2);
    let dv0dv1 = dv0 * dv1;
    let dv0dv2 = dv0 * dv2;
    if dv0dv1 > 0.0 && dv0dv2 > 0.0 {
        // All of t1 lies strictly on one side of t2's plane.
        return false;
    }

    // Supporting plane of t1.
    let n1 = t1.b.minus(t1.a).cross(t1.c.minus(t1.a));
    if n1.length_squared() < AREA_EPS_SQ {
        return false;
    }
    let d1 = -n1.dot(t1.a);

    // Signed distances of t2's vertices to t1's plane.
    let du0 = snap(n1.dot(t2.a) + d1);
    let du1 = snap(n1.dot(t2.b) + d1);
    let du2 = snap(n1.dot(t2.c) + d1);
    let du0du1 = du0 * du1;
    let du0du2 = du0 * du2;
    if du0du1 > 0.0 && du0du2 > 0.0 {
        // All of t2 lies strictly on one side of t1's plane.
        return false;
    }

    // If every t1 vertex lies in t2's plane the triangles are coplanar.
    if dv0.abs() < EPS && dv1.abs() < EPS && dv2.abs() < EPS {
        return coplanar_tri_tri(n2, t1, t2);
    }

    // Direction of the line where the two planes meet, and the axis onto which
    // projection loses the least precision (largest absolute component).
    let dir = n1.cross(n2);
    let index = largest_axis(dir);

    let (a1, b1) = compute_interval(
        t1.a.component(index),
        t1.b.component(index),
        t1.c.component(index),
        dv0,
        dv1,
        dv2,
    );
    let (a2, b2) = compute_interval(
        t2.a.component(index),
        t2.b.component(index),
        t2.c.component(index),
        du0,
        du1,
        du2,
    );

    let lo1 = a1.min(b1);
    let hi1 = a1.max(b1);
    let lo2 = a2.min(b2);
    let hi2 = a2.max(b2);

    // Intervals overlap (including a shared endpoint) unless one ends strictly
    // before the other begins.
    !(hi1 < lo2 - EPS || hi2 < lo1 - EPS)
}

/// Index of the largest-magnitude component of `dir` (`0 => x`, `1 => y`,
/// `2 => z`), chosen without a range loop.
fn largest_axis(dir: Vec3) -> usize {
    let ax = dir.x.abs();
    let ay = dir.y.abs();
    let az = dir.z.abs();
    if ax > ay {
        if ax > az {
            0
        } else {
            2
        }
    } else if ay > az {
        1
    } else {
        2
    }
}

/// Projects a triangle onto the plane-intersection line and returns its scalar
/// interval `(t_a, t_b)` (unsorted). `p0`, `p1`, `p2` are the vertices'
/// coordinates along the chosen axis and `d0`, `d1`, `d2` their signed distances
/// to the opposite plane. The interval endpoints are the two edge crossings of
/// that plane; the caller sorts the pair.
fn compute_interval(p0: f32, p1: f32, p2: f32, d0: f32, d1: f32, d2: f32) -> (f32, f32) {
    let d0d1 = d0 * d1;
    let d0d2 = d0 * d2;
    if d0d1 > 0.0 {
        // d0 and d1 share a side, so vertex 2 is the lone one.
        isect(p2, p0, p1, d2, d0, d1)
    } else if d0d2 > 0.0 {
        // d0 and d2 share a side, so vertex 1 is the lone one.
        isect(p1, p0, p2, d1, d0, d2)
    } else if d1 * d2 > 0.0 || d0.abs() > EPS {
        // Vertex 0 is the lone one (or the only one off the plane).
        isect(p0, p1, p2, d0, d1, d2)
    } else if d1.abs() > EPS {
        isect(p1, p0, p2, d1, d0, d2)
    } else {
        // Not fully coplanar (handled by the caller), so vertex 2 is off-plane.
        isect(p2, p0, p1, d2, d0, d1)
    }
}

/// Interpolates the two points where the edges leaving vertex `v0` cross the
/// opposite plane. `d0` is `v0`'s signed distance and `d1`, `d2` are the far
/// vertices'; the lone vertex convention guarantees `d0 - d1` and `d0 - d2` are
/// bounded away from zero, so neither division approaches a singularity.
fn isect(v0: f32, v1: f32, v2: f32, d0: f32, d1: f32, d2: f32) -> (f32, f32) {
    let i0 = v0 + (v1 - v0) * d0 / (d0 - d1);
    let i1 = v0 + (v2 - v0) * d0 / (d0 - d2);
    (i0, i1)
}

/// 2D orientation determinant of `(a, b, c)`: positive when the turn is
/// counter-clockwise, negative when clockwise, and near zero when collinear.
fn orient(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Whether the collinear point `p` lies within the axis-aligned bounding box of
/// segment `a`-`b` (expanded by [`EPS`]); used only after `p` is known to be
/// collinear with the segment.
fn on_seg(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> bool {
    let minx = a[0].min(b[0]) - EPS;
    let maxx = a[0].max(b[0]) + EPS;
    let miny = a[1].min(b[1]) - EPS;
    let maxy = a[1].max(b[1]) + EPS;
    (minx..=maxx).contains(&p[0]) && (miny..=maxy).contains(&p[1])
}

/// Whether the closed 2D segments `p1`-`p2` and `p3`-`p4` intersect, including
/// collinear-overlap and endpoint-touching cases decided against [`EPS`].
fn seg_seg_2d(p1: [f32; 2], p2: [f32; 2], p3: [f32; 2], p4: [f32; 2]) -> bool {
    let d1 = orient(p3, p4, p1);
    let d2 = orient(p3, p4, p2);
    let d3 = orient(p1, p2, p3);
    let d4 = orient(p1, p2, p4);

    let straddle_a = (d1 > EPS && d2 < -EPS) || (d1 < -EPS && d2 > EPS);
    let straddle_b = (d3 > EPS && d4 < -EPS) || (d3 < -EPS && d4 > EPS);
    if straddle_a && straddle_b {
        return true;
    }

    if d1.abs() <= EPS && on_seg(p3, p4, p1) {
        return true;
    }
    if d2.abs() <= EPS && on_seg(p3, p4, p2) {
        return true;
    }
    if d3.abs() <= EPS && on_seg(p1, p2, p3) {
        return true;
    }
    if d4.abs() <= EPS && on_seg(p1, p2, p4) {
        return true;
    }
    false
}

/// Whether the point `p` lies inside the closed 2D triangle `a`, `b`, `c`
/// (boundary included), regardless of winding.
fn point_in_tri_2d(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> bool {
    let d1 = orient(a, b, p);
    let d2 = orient(b, c, p);
    let d3 = orient(c, a, p);
    let has_neg = d1 < -EPS || d2 < -EPS || d3 < -EPS;
    let has_pos = d1 > EPS || d2 > EPS || d3 > EPS;
    !(has_neg && has_pos)
}

/// Fixed edge index pairs of a triangle, used to iterate edges without a range
/// loop.
const EDGES: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];

/// Coplanar fallback: projects both triangles onto the axis plane that best
/// preserves their area (dropping the largest normal component) and tests the
/// resulting 2D triangles for overlap via edge crossings plus containment.
fn coplanar_tri_tri(n: Vec3, t1: &Tri, t2: &Tri) -> bool {
    let ax = n.x.abs();
    let ay = n.y.abs();
    let az = n.z.abs();
    let (i0, i1) = if ax > ay {
        if ax > az {
            (1, 2)
        } else {
            (0, 1)
        }
    } else if az > ay {
        (0, 1)
    } else {
        (0, 2)
    };

    let t = [
        [t1.a.component(i0), t1.a.component(i1)],
        [t1.b.component(i0), t1.b.component(i1)],
        [t1.c.component(i0), t1.c.component(i1)],
    ];
    let u = [
        [t2.a.component(i0), t2.a.component(i1)],
        [t2.b.component(i0), t2.b.component(i1)],
        [t2.c.component(i0), t2.c.component(i1)],
    ];

    for &(i, j) in &EDGES {
        for &(k, l) in &EDGES {
            if seg_seg_2d(t[i], t[j], u[k], u[l]) {
                return true;
            }
        }
    }

    // No edge crossing: one triangle may still be wholly inside the other.
    point_in_tri_2d(t[0], u[0], u[1], u[2]) || point_in_tri_2d(u[0], t[0], t[1], t[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z)
    }

    fn shift(t: &Tri, o: Vec3) -> Tri {
        Tri::new(t.a.plus(o), t.b.plus(o), t.c.plus(o))
    }

    /// A unit right triangle in the z = 0 plane.
    fn base() -> Tri {
        Tri::new(v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(0.0, 1.0, 0.0))
    }

    #[test]
    fn identical_triangles_intersect() {
        let t = base();
        assert!(tri_tri_intersect(&t, &t));
    }

    #[test]
    fn far_apart_triangles_do_not_intersect() {
        let t1 = base();
        let t2 = shift(&base(), v(100.0, 100.0, 100.0));
        assert!(!tri_tri_intersect(&t1, &t2));
    }

    #[test]
    fn coplanar_far_apart_do_not_intersect() {
        let t1 = base();
        let t2 = shift(&base(), v(10.0, 0.0, 0.0));
        assert!(!tri_tri_intersect(&t1, &t2));
    }

    #[test]
    fn shared_edge_folded_intersects() {
        // t2 shares the edge (0,0,0)-(1,0,0) but folds up into the y = 0 plane.
        let t1 = base();
        let t2 = Tri::new(v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(0.0, 0.0, 1.0));
        assert!(tri_tri_intersect(&t1, &t2));
    }

    #[test]
    fn shared_single_vertex_touches() {
        // Both meet only at the origin; t2 rises in +z out of the y = 0 plane.
        let t1 = base();
        let t2 = Tri::new(v(0.0, 0.0, 0.0), v(1.0, 0.0, 1.0), v(0.0, 0.0, 1.0));
        assert!(tri_tri_intersect(&t1, &t2));
    }

    #[test]
    fn triangle_pierces_interior() {
        let t1 = Tri::new(v(-2.0, -2.0, 0.0), v(2.0, -2.0, 0.0), v(0.0, 2.0, 0.0));
        let t2 = Tri::new(v(0.0, 0.0, -1.0), v(0.0, 0.0, 1.0), v(0.0, 1.0, 0.0));
        assert!(tri_tri_intersect(&t1, &t2));
    }

    #[test]
    fn perpendicular_crossing_intersects() {
        // t1 in z = 0 straddles y = 0; t2 in y = 0 straddles z = 0; the crossing
        // segments on the x-axis overlap on x in [0.5, 1.5].
        let t1 = Tri::new(v(0.0, -1.0, 0.0), v(2.0, -1.0, 0.0), v(1.0, 1.0, 0.0));
        let t2 = Tri::new(v(0.0, 0.0, -1.0), v(2.0, 0.0, -1.0), v(1.0, 0.0, 1.0));
        assert!(tri_tri_intersect(&t1, &t2));
    }

    #[test]
    fn perpendicular_disjoint_intervals_miss() {
        // Same construction, but t2 shifted along x so the crossing segments do
        // not overlap even though both planes are straddled.
        let t1 = Tri::new(v(0.0, -1.0, 0.0), v(2.0, -1.0, 0.0), v(1.0, 1.0, 0.0));
        let t2 = Tri::new(v(5.0, 0.0, -1.0), v(7.0, 0.0, -1.0), v(6.0, 0.0, 1.0));
        assert!(!tri_tri_intersect(&t1, &t2));
    }

    #[test]
    fn coplanar_overlapping_intersect() {
        let t1 = base();
        let t2 = Tri::new(v(0.25, 0.25, 0.0), v(1.25, 0.25, 0.0), v(0.25, 1.25, 0.0));
        assert!(tri_tri_intersect(&t1, &t2));
    }

    #[test]
    fn coplanar_disjoint_miss() {
        let t1 = base();
        let t2 = Tri::new(v(2.0, 2.0, 0.0), v(3.0, 2.0, 0.0), v(2.0, 3.0, 0.0));
        assert!(!tri_tri_intersect(&t1, &t2));
    }

    #[test]
    fn coplanar_containment_intersect() {
        // A small triangle fully inside a large one, both in z = 0.
        let big = Tri::new(v(-5.0, -5.0, 0.0), v(5.0, -5.0, 0.0), v(0.0, 5.0, 0.0));
        let small = Tri::new(v(-0.5, -0.5, 0.0), v(0.5, -0.5, 0.0), v(0.0, 0.5, 0.0));
        assert!(tri_tri_intersect(&big, &small));
        assert!(tri_tri_intersect(&small, &big));
    }

    #[test]
    fn coplanar_star_of_david_edges_cross() {
        // Two coplanar triangles that share no vertex but whose edges cross.
        let up = Tri::new(v(-1.0, -0.6, 0.0), v(1.0, -0.6, 0.0), v(0.0, 1.0, 0.0));
        let down = Tri::new(v(-1.0, 0.6, 0.0), v(1.0, 0.6, 0.0), v(0.0, -1.0, 0.0));
        assert!(tri_tri_intersect(&up, &down));
    }

    #[test]
    fn coplanar_shared_vertex_intersects() {
        let t1 = base();
        // Shares only the vertex (1,0,0), extends away in +x.
        let t2 = Tri::new(v(1.0, 0.0, 0.0), v(2.0, 0.0, 0.0), v(2.0, 1.0, 0.0));
        assert!(tri_tri_intersect(&t1, &t2));
    }

    #[test]
    fn parallel_separated_planes_miss() {
        let t1 = base();
        let t2 = shift(&base(), v(0.0, 0.0, 3.0));
        assert!(!tri_tri_intersect(&t1, &t2));
    }

    #[test]
    fn parallel_touching_planes_but_offset_miss() {
        // Parallel planes one unit apart: no shared point regardless of xy.
        let t1 = base();
        let t2 = shift(&base(), v(0.25, 0.25, 1.0));
        assert!(!tri_tri_intersect(&t1, &t2));
    }

    #[test]
    fn degenerate_first_triangle_reports_false() {
        let deg = Tri::new(v(1.0, 1.0, 1.0), v(1.0, 1.0, 1.0), v(1.0, 1.0, 1.0));
        let t2 = base();
        assert!(!tri_tri_intersect(&deg, &t2));
    }

    #[test]
    fn degenerate_second_triangle_reports_false() {
        let t1 = base();
        let deg = Tri::new(v(0.2, 0.2, 0.0), v(0.4, 0.4, 0.0), v(0.6, 0.6, 0.0));
        assert!(!tri_tri_intersect(&t1, &deg));
    }

    #[test]
    fn symmetry_on_intersecting_pair() {
        let t1 = Tri::new(v(0.0, -1.0, 0.0), v(2.0, -1.0, 0.0), v(1.0, 1.0, 0.0));
        let t2 = Tri::new(v(0.0, 0.0, -1.0), v(2.0, 0.0, -1.0), v(1.0, 0.0, 1.0));
        assert_eq!(tri_tri_intersect(&t1, &t2), tri_tri_intersect(&t2, &t1));
        assert!(tri_tri_intersect(&t1, &t2));
    }

    #[test]
    fn symmetry_on_disjoint_pair() {
        let t1 = base();
        let t2 = shift(&base(), v(9.0, 9.0, 9.0));
        assert_eq!(tri_tri_intersect(&t1, &t2), tri_tri_intersect(&t2, &t1));
        assert!(!tri_tri_intersect(&t1, &t2));
    }

    #[test]
    fn translation_invariance_intersecting() {
        let t1 = Tri::new(v(0.0, -1.0, 0.0), v(2.0, -1.0, 0.0), v(1.0, 1.0, 0.0));
        let t2 = Tri::new(v(0.0, 0.0, -1.0), v(2.0, 0.0, -1.0), v(1.0, 0.0, 1.0));
        let o = v(-3.5, 12.25, 7.0);
        assert_eq!(
            tri_tri_intersect(&t1, &t2),
            tri_tri_intersect(&shift(&t1, o), &shift(&t2, o))
        );
    }

    #[test]
    fn translation_invariance_disjoint() {
        let t1 = base();
        let t2 = shift(&base(), v(10.0, 0.0, 0.0));
        let o = v(4.0, -2.0, 6.0);
        assert_eq!(
            tri_tri_intersect(&t1, &t2),
            tri_tri_intersect(&shift(&t1, o), &shift(&t2, o))
        );
    }

    #[test]
    fn far_from_origin_still_intersects() {
        let o = v(1000.0, -1000.0, 500.0);
        let t1 = shift(
            &Tri::new(v(0.0, -1.0, 0.0), v(2.0, -1.0, 0.0), v(1.0, 1.0, 0.0)),
            o,
        );
        let t2 = shift(
            &Tri::new(v(0.0, 0.0, -1.0), v(2.0, 0.0, -1.0), v(1.0, 0.0, 1.0)),
            o,
        );
        assert!(tri_tri_intersect(&t1, &t2));
    }

    #[test]
    fn tiny_scaled_triangles_intersect() {
        let s = 1.0e-2;
        let t1 = Tri::new(
            v(0.0, -s, 0.0),
            v(2.0 * s, -s, 0.0),
            v(1.0 * s, 1.0 * s, 0.0),
        );
        let t2 = Tri::new(
            v(0.0, 0.0, -s),
            v(2.0 * s, 0.0, -s),
            v(1.0 * s, 0.0, 1.0 * s),
        );
        assert!(tri_tri_intersect(&t1, &t2));
    }

    #[test]
    fn winding_reversal_is_ignored() {
        let t1 = base();
        let t2 = Tri::new(v(0.25, 0.25, 0.0), v(1.25, 0.25, 0.0), v(0.25, 1.25, 0.0));
        // Reverse t2's winding: the predicate must be unchanged.
        let t2_rev = Tri::new(t2.c, t2.b, t2.a);
        assert_eq!(tri_tri_intersect(&t1, &t2), tri_tri_intersect(&t1, &t2_rev));
        assert!(tri_tri_intersect(&t1, &t2_rev));
    }

    #[test]
    fn one_vertex_on_plane_otherwise_separated_miss() {
        // t2 touches z = 0 only at a vertex far from t1, rest above.
        let t1 = base();
        let t2 = Tri::new(v(5.0, 5.0, 0.0), v(6.0, 5.0, 1.0), v(5.0, 6.0, 1.0));
        assert!(!tri_tri_intersect(&t1, &t2));
    }

    #[test]
    fn partial_edge_overlap_intersects() {
        // t2's crossing segment on the x-axis (x in [0,1]) overlaps t1's on
        // [0.5, 1.5], sharing the sub-range [0.5, 1.0].
        let t1 = Tri::new(v(0.0, -1.0, 0.0), v(2.0, -1.0, 0.0), v(1.0, 1.0, 0.0));
        let t2 = Tri::new(v(-1.0, 0.0, -1.0), v(1.0, 0.0, -1.0), v(0.0, 0.0, 1.0));
        assert!(tri_tri_intersect(&t1, &t2));
    }

    #[test]
    fn coplanar_edge_touch_intersects() {
        // Two coplanar triangles sharing a full edge (0,0,0)-(1,0,0).
        let t1 = base();
        let t2 = Tri::new(v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(0.5, -1.0, 0.0));
        assert!(tri_tri_intersect(&t1, &t2));
    }

    #[test]
    fn near_miss_just_beyond_plane_is_false() {
        // t2 is parallel to and just above t1's plane by more than EPS.
        let t1 = base();
        let t2 = shift(&base(), v(0.0, 0.0, 0.01));
        assert!(!tri_tri_intersect(&t1, &t2));
    }
}
