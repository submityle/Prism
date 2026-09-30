//! 2D convex-polygon Boolean intersection via the `Gilbert-Johnson-Keerthi`
//! (`GJK`) simplex algorithm, for the particle spatial-query contracts
//! (design §8.2, §10).
//!
//! A handful of particle stages reduce a query to "do these two convex 2D
//! shapes share any point": a splat-vs-collider overlap gate, a footprint
//! pre-pass that rejects obviously separated pairs before the expensive
//! depenetration solve, and a screen-space cluster test that only needs a
//! yes/no answer. This module owns the small, `CPU`-verifiable contract those
//! stages share: given two convex point sets expressed as vertex rings, it
//! decides overlap by iterating the `GJK` support-mapping simplex toward the
//! origin of the Minkowski difference.
//!
//! # Algorithm
//! `GJK` never materializes the Minkowski difference `A - B`. Instead it probes
//! that set through a *support mapping*: for a search direction `d`, the support
//! point is `support(A, d) - support(B, -d)`, the vertex of `A - B` farthest
//! along `d`. Starting from an arbitrary support point, the search maintains a
//! *simplex* of one, two, or three such points that brackets the origin ever
//! more tightly. Each step folds the newest point into the simplex, discards the
//! feature (vertex or edge) that cannot face the origin, and picks the next
//! search direction as the simplex feature normal pointing at the origin. When a
//! support point fails to pass the origin the two shapes are disjoint; when the
//! simplex grows to a triangle that encloses the origin they overlap. A fixed
//! iteration bound guarantees deterministic termination even for degenerate,
//! boundary-touching, or many-vertex inputs.
//!
//! # Strict scope
//! This module only answers *convex overlap yes/no* through `GJK`. It is a
//! different algorithm from [`super::sat_collision_2d`], which decides the same
//! question through the Separating Axis Theorem and additionally reports a
//! minimum translation vector; this module reports neither penetration nor
//! separating axis. It does not construct a convex hull
//! ([`super::convex_hull_2d`]), it does not resolve analytic-primitive collision
//! response ([`super::collision`]), and it does not classify two-segment
//! crossings ([`super::segment_intersect_2d`]). It neither imports nor
//! reconstructs those contracts and keeps its own [`Vec2`] and math helpers.
//!
//! # No transcendental math
//! Support projection, the triple-product simplex normals, and origin
//! containment are pure `+`, `-`, `*` cross-product and comparison arithmetic;
//! the only irrational operation is the `f32::sqrt` used to length-normalize the
//! separation tolerance. There is no `sin`, `cos`, `tan`, `atan`, `exp`, `ln`,
//! `powf`, `ceil`, `round`, or any other transcendental/rounding call, and no
//! `f32` equality: near-zero magnitudes are compared against [`GJK_EPS`].

use alloc::vec::Vec;

/// Magnitude below which a coordinate difference or a squared length is treated
/// as zero. This is the comparison rule used throughout instead of `==` on
/// `f32`: two scalars are "equal" when their absolute difference does not exceed
/// this bound.
pub const GJK_EPS: f32 = 1.0e-6;

/// Relative tolerance on the signed support-plane distance used to classify a
/// separating direction. A support projection is treated as a real separation
/// only when it falls below `-SEP_EPS * |d|`, so that shapes touching within
/// float noise are reported as intersecting rather than disjoint.
const SEP_EPS: f32 = 1.0e-5;

/// Squared-length threshold below which a search direction has collapsed onto
/// the origin, meaning the origin lies on the current simplex feature (a
/// boundary touch) and the shapes intersect.
const DIR_EPS_SQ: f32 = 1.0e-10;

/// Squared-distance threshold below which a freshly probed support point is
/// considered a duplicate of one already in the simplex; reaching a duplicate
/// means the search cannot expand further and the origin is at or inside the
/// simplex boundary.
const DUP_EPS_SQ: f32 = 1.0e-10;

/// Hard cap on simplex iterations. 2D `GJK` converges in a number of steps
/// bounded by the combined vertex count; the cap makes termination and output
/// deterministic for every input, including degenerate rings.
const MAX_ITERS: usize = 64;

/// A hand-rolled 2D vector, local to this module so it never depends on a
/// sibling's math type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec2 {
    /// The `x` component.
    pub x: f32,
    /// The `y` component.
    pub y: f32,
}

impl Vec2 {
    /// The zero vector.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    /// Builds a vector from its components.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Component-wise sum.
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

    /// The additive inverse `-self`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API is specified with named add/sub/neg methods for call-site uniformity, matching the sibling particle contracts; operator traits are intentionally not part of this internal type."
    )]
    pub fn neg(self) -> Self {
        Self::new(-self.x, -self.y)
    }

    /// Uniformly scales both components by `s`.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s)
    }

    /// The dot product `self . other`.
    #[must_use]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y
    }

    /// The 2D scalar cross product `self x other`.
    #[must_use]
    pub fn cross(self, other: Self) -> f32 {
        self.x * other.y - self.y * other.x
    }

    /// The squared Euclidean length, free of any square root.
    #[must_use]
    pub fn length_sq(self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    /// The Euclidean length. The only irrational operation in this module.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_sq().sqrt()
    }
}

/// The vector triple product `(a x b) x c = b (a . c) - a (b . c)`, used to
/// build a simplex-edge normal that points toward the origin.
fn triple(a: Vec2, b: Vec2, c: Vec2) -> Vec2 {
    b.scale(a.dot(c)).sub(a.scale(b.dot(c)))
}

/// Returns the vertex of `vertices` farthest along `dir` (the support point of
/// the convex set in that direction). Ties keep the earliest vertex, which
/// keeps the search deterministic.
///
/// # Panics
/// Panics if `vertices` is empty; callers pass non-empty rings, and
/// [`intersects`] guards empty inputs before probing.
#[must_use]
pub fn support(vertices: &[Vec2], dir: Vec2) -> Vec2 {
    let mut best = vertices[0];
    let mut best_dot = best.dot(dir);
    for &v in &vertices[1..] {
        let projected = v.dot(dir);
        if projected > best_dot {
            best_dot = projected;
            best = v;
        }
    }
    best
}

/// The support point of the Minkowski difference `a - b` along `dir`, i.e.
/// `support(a, dir) - support(b, -dir)`. `GJK` probes the difference set only
/// through this mapping and never enumerates its vertices.
#[must_use]
pub fn minkowski_support(a: &[Vec2], b: &[Vec2], dir: Vec2) -> Vec2 {
    support(a, dir).sub(support(b, dir.neg()))
}

/// Evolves the current simplex toward the origin. Returns `true` when the
/// simplex is a triangle that encloses the origin (the shapes overlap);
/// otherwise it trims the simplex to the feature closest to the origin and
/// writes the next search direction into `dir`.
fn do_simplex(simplex: &mut Vec<Vec2>, dir: &mut Vec2) -> bool {
    if simplex.len() == 3 {
        let a = simplex[2];
        let b = simplex[1];
        let c = simplex[0];
        let ab = b.sub(a);
        let ac = c.sub(a);
        let ao = a.neg();
        let ab_perp = triple(ac, ab, ab);
        let ac_perp = triple(ab, ac, ac);
        if ab_perp.dot(ao) > 0.0 {
            simplex.clear();
            simplex.push(b);
            simplex.push(a);
            *dir = ab_perp;
            return false;
        }
        if ac_perp.dot(ao) > 0.0 {
            simplex.clear();
            simplex.push(c);
            simplex.push(a);
            *dir = ac_perp;
            return false;
        }
        return true;
    }

    // Line case: two points, `a` newest.
    let a = simplex[1];
    let b = simplex[0];
    let ab = b.sub(a);
    let ao = a.neg();
    if ab.dot(ao) > 0.0 {
        *dir = triple(ab, ao, ab);
    } else {
        simplex.clear();
        simplex.push(a);
        *dir = ao;
    }
    false
}

/// Decides whether two convex point sets overlap (sharing at least a boundary
/// point) using the `GJK` simplex search over their Minkowski difference.
///
/// Each argument is a convex set given by its vertices; a single vertex encodes
/// a point and two vertices encode a segment, both handled as degenerate convex
/// sets. Boundary contact (a shared edge, a shared vertex, or a point lying on
/// an edge) counts as an intersection. Empty input yields `false`.
#[must_use]
pub fn intersects(a: &[Vec2], b: &[Vec2]) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }

    let mut dir = Vec2::new(1.0, 0.0);
    let first = minkowski_support(a, b, dir);
    let mut simplex: Vec<Vec2> = Vec::with_capacity(3);
    simplex.push(first);
    dir = first.neg();

    for _ in 0..MAX_ITERS {
        if dir.length_sq() <= DIR_EPS_SQ {
            // The direction collapsed onto the origin: the origin sits on the
            // current simplex feature, so the sets touch.
            return true;
        }

        let probe = minkowski_support(a, b, dir);
        let dir_len = dir.length();
        let projection = probe.dot(dir);
        if projection < -SEP_EPS * dir_len {
            // The farthest point of `a - b` along `dir` fails to reach the
            // origin: `dir` is a separating direction.
            return false;
        }

        let duplicate = simplex
            .iter()
            .any(|&q| q.sub(probe).length_sq() <= DUP_EPS_SQ);
        if duplicate {
            // No new vertex can be added; a strictly separated pair would have
            // returned above, so the origin is at or inside the simplex.
            return true;
        }

        simplex.push(probe);
        if do_simplex(&mut simplex, &mut dir) {
            return true;
        }
    }

    // Converged within the iteration bound without a separating direction.
    true
}

#[cfg(test)]
mod tests {
    use super::{intersects, minkowski_support, support, Vec2};

    fn v(x: f32, y: f32) -> Vec2 {
        Vec2::new(x, y)
    }

    fn unit_square() -> [Vec2; 4] {
        [v(0.0, 0.0), v(1.0, 0.0), v(1.0, 1.0), v(0.0, 1.0)]
    }

    fn octagon(cx: f32, cy: f32) -> [Vec2; 8] {
        [
            v(cx + 2.0, cy + 1.0),
            v(cx + 1.0, cy + 2.0),
            v(cx - 1.0, cy + 2.0),
            v(cx - 2.0, cy + 1.0),
            v(cx - 2.0, cy - 1.0),
            v(cx - 1.0, cy - 2.0),
            v(cx + 1.0, cy - 2.0),
            v(cx + 2.0, cy - 1.0),
        ]
    }

    fn approx(a: Vec2, b: Vec2) -> bool {
        a.sub(b).length_sq() <= 1.0e-10
    }

    #[test]
    fn support_picks_extreme_right() {
        let s = unit_square();
        assert!(approx(support(&s, v(1.0, 0.0)), v(1.0, 0.0)));
    }

    #[test]
    fn support_picks_extreme_up() {
        let s = unit_square();
        let picked = support(&s, v(0.0, 1.0));
        assert!((picked.y - 1.0).abs() <= 1.0e-6);
    }

    #[test]
    fn support_picks_extreme_diagonal() {
        let s = unit_square();
        assert!(approx(support(&s, v(1.0, 1.0)), v(1.0, 1.0)));
    }

    #[test]
    fn support_single_vertex_is_constant() {
        let point = [v(3.0, -2.0)];
        assert!(approx(support(&point, v(1.0, 0.0)), v(3.0, -2.0)));
        assert!(approx(support(&point, v(-5.0, 7.0)), v(3.0, -2.0)));
    }

    #[test]
    fn minkowski_support_matches_manual_difference() {
        let a = unit_square();
        let b = [v(5.0, 5.0), v(6.0, 5.0), v(6.0, 6.0), v(5.0, 6.0)];
        // Farthest of A along +x is (1,0); farthest of B along -x is (5,5).
        let expected = v(1.0 - 5.0, 0.0 - 5.0);
        assert!(approx(minkowski_support(&a, &b, v(1.0, 0.0)), expected));
    }

    #[test]
    fn overlapping_unit_squares_intersect() {
        let a = unit_square();
        let b = [v(0.5, 0.0), v(1.5, 0.0), v(1.5, 1.0), v(0.5, 1.0)];
        assert!(intersects(&a, &b));
    }

    #[test]
    fn disjoint_squares_do_not_intersect() {
        let a = unit_square();
        let b = [v(2.0, 0.0), v(3.0, 0.0), v(3.0, 1.0), v(2.0, 1.0)];
        assert!(!intersects(&a, &b));
    }

    #[test]
    fn edge_touching_squares_intersect() {
        let a = unit_square();
        let b = [v(1.0, 0.0), v(2.0, 0.0), v(2.0, 1.0), v(1.0, 1.0)];
        assert!(intersects(&a, &b));
    }

    #[test]
    fn vertex_touching_squares_intersect() {
        let a = unit_square();
        let b = [v(1.0, 1.0), v(2.0, 1.0), v(2.0, 2.0), v(1.0, 2.0)];
        assert!(intersects(&a, &b));
    }

    #[test]
    fn small_square_contained_in_big_square_intersects() {
        let big = [v(-5.0, -5.0), v(5.0, -5.0), v(5.0, 5.0), v(-5.0, 5.0)];
        let small = [v(-1.0, -1.0), v(1.0, -1.0), v(1.0, 1.0), v(-1.0, 1.0)];
        assert!(intersects(&big, &small));
    }

    #[test]
    fn containment_is_symmetric_in_argument_order() {
        let big = [v(-5.0, -5.0), v(5.0, -5.0), v(5.0, 5.0), v(-5.0, 5.0)];
        let small = [v(-1.0, -1.0), v(1.0, -1.0), v(1.0, 1.0), v(-1.0, 1.0)];
        assert_eq!(intersects(&big, &small), intersects(&small, &big));
        assert!(intersects(&small, &big));
    }

    #[test]
    fn far_apart_squares_do_not_intersect() {
        let a = unit_square();
        let b = [
            v(100.0, 100.0),
            v(101.0, 100.0),
            v(101.0, 101.0),
            v(100.0, 101.0),
        ];
        assert!(!intersects(&a, &b));
    }

    #[test]
    fn overlapping_triangles_intersect() {
        let t1 = [v(0.0, 0.0), v(2.0, 0.0), v(0.0, 2.0)];
        let t2 = [v(0.5, 0.5), v(2.0, 0.5), v(0.5, 2.0)];
        assert!(intersects(&t1, &t2));
    }

    #[test]
    fn disjoint_triangles_do_not_intersect() {
        let t1 = [v(0.0, 0.0), v(1.0, 0.0), v(0.0, 1.0)];
        let t2 = [v(3.0, 3.0), v(4.0, 3.0), v(3.0, 4.0)];
        assert!(!intersects(&t1, &t2));
    }

    #[test]
    fn triangle_inside_square_intersects() {
        let square = [v(-2.0, -2.0), v(2.0, -2.0), v(2.0, 2.0), v(-2.0, 2.0)];
        let tri = [v(-0.5, -0.5), v(0.5, -0.5), v(0.0, 0.5)];
        assert!(intersects(&square, &tri));
    }

    #[test]
    fn point_inside_polygon_intersects() {
        let square = unit_square();
        let point = [v(0.5, 0.5)];
        assert!(intersects(&square, &point));
    }

    #[test]
    fn point_outside_polygon_does_not_intersect() {
        let square = unit_square();
        let point = [v(2.0, 2.0)];
        assert!(!intersects(&square, &point));
    }

    #[test]
    fn point_on_polygon_edge_touches() {
        let square = unit_square();
        let point = [v(0.5, 0.0)];
        assert!(intersects(&square, &point));
    }

    #[test]
    fn point_on_polygon_vertex_touches() {
        let square = unit_square();
        let point = [v(0.0, 0.0)];
        assert!(intersects(&square, &point));
    }

    #[test]
    fn identical_points_intersect() {
        let a = [v(1.5, -3.25)];
        let b = [v(1.5, -3.25)];
        assert!(intersects(&a, &b));
    }

    #[test]
    fn distinct_points_do_not_intersect() {
        let a = [v(0.0, 0.0)];
        let b = [v(1.0, 1.0)];
        assert!(!intersects(&a, &b));
    }

    #[test]
    fn crossing_segments_intersect() {
        let horizontal = [v(-1.0, 0.0), v(1.0, 0.0)];
        let vertical = [v(0.0, -1.0), v(0.0, 1.0)];
        assert!(intersects(&horizontal, &vertical));
    }

    #[test]
    fn parallel_segments_do_not_intersect() {
        let lower = [v(-1.0, 0.0), v(1.0, 0.0)];
        let upper = [v(-1.0, 1.0), v(1.0, 1.0)];
        assert!(!intersects(&lower, &upper));
    }

    #[test]
    fn collinear_overlapping_segments_intersect() {
        let a = [v(0.0, 0.0), v(2.0, 0.0)];
        let b = [v(1.0, 0.0), v(3.0, 0.0)];
        assert!(intersects(&a, &b));
    }

    #[test]
    fn collinear_segments_touching_at_endpoint_intersect() {
        let a = [v(0.0, 0.0), v(1.0, 0.0)];
        let b = [v(1.0, 0.0), v(2.0, 0.0)];
        assert!(intersects(&a, &b));
    }

    #[test]
    fn collinear_disjoint_segments_do_not_intersect() {
        let a = [v(0.0, 0.0), v(1.0, 0.0)];
        let b = [v(2.0, 0.0), v(3.0, 0.0)];
        assert!(!intersects(&a, &b));
    }

    #[test]
    fn many_vertex_octagons_overlap() {
        let a = octagon(0.0, 0.0);
        let b = octagon(1.0, 1.0);
        assert!(intersects(&a, &b));
    }

    #[test]
    fn many_vertex_octagons_disjoint() {
        let a = octagon(0.0, 0.0);
        let b = octagon(20.0, 0.0);
        assert!(!intersects(&a, &b));
    }

    #[test]
    fn rotated_diamond_overlaps_axis_square() {
        let diamond = [v(0.0, 1.0), v(1.0, 0.0), v(0.0, -1.0), v(-1.0, 0.0)];
        let square = [v(-0.5, -0.5), v(0.5, -0.5), v(0.5, 0.5), v(-0.5, 0.5)];
        assert!(intersects(&diamond, &square));
    }

    #[test]
    fn segment_touching_polygon_boundary_intersects() {
        let square = unit_square();
        // A segment whose right endpoint lands exactly on the square's edge.
        let segment = [v(-1.0, 0.5), v(0.0, 0.5)];
        assert!(intersects(&square, &segment));
    }

    #[test]
    fn intersection_is_symmetric_over_many_pairs() {
        let a = octagon(0.0, 0.0);
        let pairs = [
            octagon(0.5, 0.5),
            octagon(3.0, 3.0),
            octagon(30.0, 0.0),
            octagon(-1.0, 1.0),
        ];
        for b in &pairs {
            assert_eq!(intersects(&a, b), intersects(b, &a));
        }
    }

    #[test]
    fn repeated_calls_are_deterministic() {
        let a = unit_square();
        let b = [v(0.5, 0.5), v(1.5, 0.5), v(1.5, 1.5), v(0.5, 1.5)];
        let first = intersects(&a, &b);
        for _ in 0..8 {
            assert_eq!(intersects(&a, &b), first);
        }
        assert!(first);
    }

    #[test]
    fn empty_input_never_intersects() {
        let square = unit_square();
        let empty: [Vec2; 0] = [];
        assert!(!intersects(&square, &empty));
        assert!(!intersects(&empty, &square));
        assert!(!intersects(&empty, &empty));
    }
}
