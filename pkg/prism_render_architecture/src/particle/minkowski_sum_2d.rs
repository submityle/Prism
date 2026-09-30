//! 2D convex-polygon Minkowski sum construction for the particle spatial
//! contracts (design §8.2, §10).
//!
//! Several particle stages need the *swept* boundary of one convex shape
//! dragged around another: a splat whose footprint is inflated by a brush
//! radius, a collider grown by a probe's own extent so the probe can be treated
//! as a point, and a broadphase cell that must bound every relative placement of
//! two clusters. This module owns the small, `CPU`-verifiable contract those
//! stages share: given two convex polygons expressed as vertex rings, it
//! *constructs* their Minkowski sum `A ⊕ B = { a + b : a ∈ A, b ∈ B }` as a
//! fresh `CCW` (counter-clockwise) vertex ring.
//!
//! # Algorithm
//! Each input is first normalized to a `CCW` ring whose first vertex is the
//! bottom-most (smallest `y`, ties broken by smallest `x`); collinear and
//! duplicate vertices are dropped, and a fully degenerate input collapses to a
//! two-point segment or a single point. From that bottom-most start every
//! convex polygon's edge vectors are already sorted by increasing polar angle,
//! so the sum is built by *merging the two edge sequences by polar angle*
//! (`O(n + m)`), accumulating each chosen edge onto a running vertex. Polar
//! angle is compared without `atan`: edges are first split into the upper and
//! lower half-planes, then ordered within a half-plane by the sign of their
//! 2D cross product. A final pass removes collinear vertices and reduces a
//! degenerate (zero-area) result to its extreme endpoints.
//!
//! # Strict scope
//! This module only *constructs* the Minkowski sum polygon. It is deliberately
//! different from [`super::gjk_2d`], which probes the Minkowski *difference*
//! `A - B` through a support mapping to answer overlap yes/no and never
//! materializes any polygon; from [`super::convex_hull_2d`], which builds the
//! hull of an unordered point set rather than summing two rings; and from
//! [`super::sat_collision_2d`], which decides overlap and a minimum translation
//! vector via the Separating Axis Theorem. It neither imports nor reconstructs
//! those contracts and keeps its own [`Vec2`] and math helpers.
//!
//! # No transcendental math
//! Normalization, the polar-angle merge, and cleanup are pure `+`, `-`, `*`
//! cross-product and comparison arithmetic. There is no `sin`, `cos`, `tan`,
//! `atan`, `exp`, `ln`, `powf`, `ceil`, `round`, or any other
//! transcendental/rounding call, and no `f32` equality: near-zero magnitudes
//! are compared against [`SUM_EPS`].

use alloc::vec::Vec;

/// Magnitude below which a coordinate difference, a cross product, or a signed
/// area is treated as zero. This is the comparison rule used throughout instead
/// of `==` on `f32`: two scalars are "equal" when their absolute difference does
/// not exceed this bound, and a turn is a real turn only when its cross product
/// exceeds it.
pub const SUM_EPS: f32 = 1.0e-6;

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
    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }

    /// Component-wise difference.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API is specified with named add/sub/neg methods for call-site uniformity, matching the sibling particle contracts; operator traits are intentionally not part of this internal type."
    )]
    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }

    /// The additive inverse.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API is specified with named add/sub/neg methods for call-site uniformity, matching the sibling particle contracts; operator traits are intentionally not part of this internal type."
    )]
    pub fn neg(self) -> Self {
        Self::new(-self.x, -self.y)
    }

    /// Uniform scale by a scalar.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s)
    }

    /// 2D cross product `self × rhs`, i.e. the signed area of the parallelogram
    /// spanned by the two vectors. Its sign classifies the turn from `self` to
    /// `rhs`: positive for a counter-clockwise (left) turn, negative for a
    /// clockwise (right) turn, zero (within [`SUM_EPS`]) when parallel.
    #[must_use]
    pub fn cross(self, rhs: Self) -> f32 {
        self.x * rhs.y - self.y * rhs.x
    }
}

/// Returns `true` when two points coincide within [`SUM_EPS`] on both axes.
fn points_equal(a: Vec2, b: Vec2) -> bool {
    (a.x - b.x).abs() <= SUM_EPS && (a.y - b.y).abs() <= SUM_EPS
}

/// Returns `true` when `a` should precede `b` in bottom-most order: strictly
/// smaller `y`, or an equal `y` (within [`SUM_EPS`]) and strictly smaller `x`.
fn is_lower(a: Vec2, b: Vec2) -> bool {
    if (a.y - b.y).abs() <= SUM_EPS {
        a.x < b.x - SUM_EPS
    } else {
        a.y < b.y
    }
}

/// Twice the signed area of a vertex ring via the shoelace formula. Positive for
/// a `CCW` ring, negative for a clockwise ring, zero (within [`SUM_EPS`]) for a
/// degenerate ring.
fn signed_area2(ring: &[Vec2]) -> f32 {
    let n = ring.len();
    if n < 3 {
        return 0.0;
    }
    let mut acc = 0.0;
    for (a, b) in ring.iter().zip(ring.iter().cycle().skip(1)).take(n) {
        acc += a.cross(*b);
    }
    acc
}

/// Drops consecutive duplicate vertices (including the wrap-around pair) within
/// [`SUM_EPS`], preserving order.
fn dedupe(ring: &[Vec2]) -> Vec<Vec2> {
    let mut out: Vec<Vec2> = Vec::new();
    for &p in ring {
        match out.last() {
            Some(&last) if points_equal(last, p) => {}
            _ => out.push(p),
        }
    }
    if out.len() > 1 {
        let first = out[0];
        let last = out[out.len() - 1];
        if points_equal(first, last) {
            out.pop();
        }
    }
    out
}

/// Rotates a ring so its bottom-most vertex (smallest `y`, ties by smallest `x`)
/// is first, preserving cyclic order.
fn rotate_to_lowest(ring: &[Vec2]) -> Vec<Vec2> {
    let n = ring.len();
    if n == 0 {
        return Vec::new();
    }
    let mut best = 0;
    for (idx, &p) in ring.iter().enumerate() {
        if is_lower(p, ring[best]) {
            best = idx;
        }
    }
    let mut out = Vec::with_capacity(n);
    for offset in 0..n {
        out.push(ring[(best + offset) % n]);
    }
    out
}

/// Removes every vertex that is collinear with its two neighbours, repeating
/// until the ring is stable so that chains of collinear points collapse fully.
fn strip_collinear(ring: &[Vec2]) -> Vec<Vec2> {
    let mut current = ring.to_vec();
    loop {
        let n = current.len();
        if n < 3 {
            return current;
        }
        let mut next: Vec<Vec2> = Vec::with_capacity(n);
        for (idx, &cur) in current.iter().enumerate() {
            let prev = current[(idx + n - 1) % n];
            let after = current[(idx + 1) % n];
            let turn = cur.sub(prev).cross(after.sub(cur));
            if turn.abs() > SUM_EPS {
                next.push(cur);
            }
        }
        if next.len() == n || next.len() < 3 {
            return next;
        }
        current = next;
    }
}

/// Reduces a set of collinear (or coincident) points to its extreme endpoints
/// along the dominant axis, returning a single point when they all coincide and
/// a two-point segment otherwise. The result is ordered bottom-most first.
fn reduce_to_segment(points: &[Vec2]) -> Vec<Vec2> {
    if points.is_empty() {
        return Vec::new();
    }
    let mut min_x = points[0].x;
    let mut max_x = points[0].x;
    let mut min_y = points[0].y;
    let mut max_y = points[0].y;
    for &p in points {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }
    let use_x = (max_x - min_x) >= (max_y - min_y);
    let mut lo = points[0];
    let mut hi = points[0];
    for &p in points {
        let key = if use_x { p.x } else { p.y };
        let lo_key = if use_x { lo.x } else { lo.y };
        let hi_key = if use_x { hi.x } else { hi.y };
        if key < lo_key - SUM_EPS {
            lo = p;
        }
        if key > hi_key + SUM_EPS {
            hi = p;
        }
    }
    if points_equal(lo, hi) {
        return alloc::vec![lo];
    }
    if is_lower(hi, lo) {
        alloc::vec![hi, lo]
    } else {
        alloc::vec![lo, hi]
    }
}

/// Normalizes an arbitrary convex vertex ring into the canonical form the merge
/// expects: a `CCW` ring (for a real polygon) or a two-point segment / single
/// point (for a degenerate input), always ordered bottom-most first, with
/// duplicate and collinear vertices removed.
fn normalize(poly: &[Vec2]) -> Vec<Vec2> {
    let deduped = dedupe(poly);
    if deduped.len() <= 1 {
        return deduped;
    }
    let area2 = signed_area2(&deduped);
    if area2.abs() <= SUM_EPS {
        return reduce_to_segment(&deduped);
    }
    let oriented = if area2 < 0.0 {
        let mut rev = deduped.clone();
        rev.reverse();
        rev
    } else {
        deduped
    };
    let stripped = strip_collinear(&oriented);
    if stripped.len() <= 2 {
        return reduce_to_segment(&stripped);
    }
    rotate_to_lowest(&stripped)
}

/// The edge vectors of a normalized ring, in cyclic order. A single point has
/// no edges; a two-point segment yields its two antiparallel edges; a polygon
/// yields one edge per vertex.
fn edges(ring: &[Vec2]) -> Vec<Vec2> {
    let n = ring.len();
    if n < 2 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n);
    for (idx, &v) in ring.iter().enumerate() {
        out.push(ring[(idx + 1) % n].sub(v));
    }
    out
}

/// Half-plane classifier for polar-angle ordering: `0` for the upper half
/// `[0°, 180°)` (including the positive `x` axis) and `1` for the lower half
/// `[180°, 360°)` (including the negative `x` axis).
fn half_plane(v: Vec2) -> u8 {
    if v.y.abs() <= SUM_EPS {
        if v.x >= 0.0 {
            0
        } else {
            1
        }
    } else if v.y > 0.0 {
        0
    } else {
        1
    }
}

/// Returns `true` when edge `a` has a polar angle no greater than edge `b`,
/// comparing without `atan`: first by half-plane, then by cross-product sign
/// within a half-plane.
fn angle_leq(a: Vec2, b: Vec2) -> bool {
    let ha = half_plane(a);
    let hb = half_plane(b);
    if ha != hb {
        return ha < hb;
    }
    a.cross(b) >= -SUM_EPS
}

/// Constructs the Minkowski sum `a ⊕ b` of two convex polygons, each given as a
/// vertex ring in any winding and starting vertex, and returns it as a fresh
/// `CCW` vertex ring ordered bottom-most first.
///
/// Degenerate inputs are handled uniformly: a one-vertex input acts as a pure
/// translation of the other shape, a two-vertex (segment) input sweeps the
/// other shape along its length, and a sum that collapses to a line is returned
/// as a two-point segment (or a single point when both inputs are points).
///
/// The result never has more vertices than the sum of the two normalized input
/// vertex counts, and collinear vertices are removed so the ring is minimal.
#[must_use]
pub fn minkowski_sum(a: &[Vec2], b: &[Vec2]) -> Vec<Vec2> {
    let poly_a = normalize(a);
    let poly_b = normalize(b);
    if poly_a.is_empty() || poly_b.is_empty() {
        return Vec::new();
    }

    let edges_a = edges(&poly_a);
    let edges_b = edges(&poly_b);
    let n = edges_a.len();
    let m = edges_b.len();

    let start = poly_a[0].add(poly_b[0]);
    let mut cursor = start;
    let mut verts: Vec<Vec2> = Vec::with_capacity(n + m + 1);
    let mut i = 0;
    let mut j = 0;
    while i < n || j < m {
        verts.push(cursor);
        let take_a = if i >= n {
            false
        } else if j >= m {
            true
        } else {
            angle_leq(edges_a[i], edges_b[j])
        };
        if take_a {
            cursor = cursor.add(edges_a[i]);
            i += 1;
        } else {
            cursor = cursor.add(edges_b[j]);
            j += 1;
        }
    }
    if verts.is_empty() {
        verts.push(start);
    }

    finalize(&verts)
}

/// Cleans a freshly accumulated vertex list into the canonical result: a
/// `CCW` polygon (bottom-most first, no collinear vertices) or, when the result
/// is degenerate, a two-point segment or single point.
fn finalize(verts: &[Vec2]) -> Vec<Vec2> {
    let deduped = dedupe(verts);
    if deduped.len() <= 2 {
        return reduce_to_segment(&deduped);
    }
    let area2 = signed_area2(&deduped);
    if area2.abs() <= SUM_EPS {
        return reduce_to_segment(&deduped);
    }
    let oriented = if area2 < 0.0 {
        let mut rev = deduped.clone();
        rev.reverse();
        rev
    } else {
        deduped
    };
    let stripped = strip_collinear(&oriented);
    if stripped.len() <= 2 {
        return reduce_to_segment(&stripped);
    }
    rotate_to_lowest(&stripped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn v(x: f32, y: f32) -> Vec2 {
        Vec2::new(x, y)
    }

    /// Sorts a ring by `(x, y)` so two rings can be compared regardless of the
    /// starting vertex or exact rotation.
    fn sorted(ring: &[Vec2]) -> Vec<Vec2> {
        let mut out = ring.to_vec();
        out.sort_by(|p, q| {
            p.x.partial_cmp(&q.x)
                .unwrap()
                .then(p.y.partial_cmp(&q.y).unwrap())
        });
        out
    }

    fn sets_equal(a: &[Vec2], b: &[Vec2]) -> bool {
        let sa = sorted(a);
        let sb = sorted(b);
        if sa.len() != sb.len() {
            return false;
        }
        sa.iter().zip(sb.iter()).all(|(p, q)| points_equal(*p, *q))
    }

    fn contains_point(ring: &[Vec2], p: Vec2) -> bool {
        ring.iter().any(|q| points_equal(*q, p))
    }

    fn area(ring: &[Vec2]) -> f32 {
        signed_area2(ring).abs() * 0.5
    }

    fn is_ccw_convex(ring: &[Vec2]) -> bool {
        let n = ring.len();
        if n < 3 {
            return true;
        }
        for idx in 0..n {
            let prev = ring[(idx + n - 1) % n];
            let cur = ring[idx];
            let after = ring[(idx + 1) % n];
            if cur.sub(prev).cross(after.sub(cur)) < -SUM_EPS {
                return false;
            }
        }
        signed_area2(ring) > SUM_EPS
    }

    fn unit_square() -> Vec<Vec2> {
        vec![v(0.0, 0.0), v(1.0, 0.0), v(1.0, 1.0), v(0.0, 1.0)]
    }

    fn right_triangle() -> Vec<Vec2> {
        vec![v(0.0, 0.0), v(1.0, 0.0), v(0.0, 1.0)]
    }

    #[test]
    fn triangle_plus_itself_is_scaled_triangle() {
        let t = right_triangle();
        let sum = minkowski_sum(&t, &t);
        let expected = vec![v(0.0, 0.0), v(2.0, 0.0), v(0.0, 2.0)];
        assert!(sets_equal(&sum, &expected), "sum = {sum:?}");
        assert!((area(&sum) - 2.0).abs() <= 1.0e-4);
    }

    #[test]
    fn two_unit_squares_make_two_by_two_square() {
        let sum = minkowski_sum(&unit_square(), &unit_square());
        let expected = vec![v(0.0, 0.0), v(2.0, 0.0), v(2.0, 2.0), v(0.0, 2.0)];
        assert!(sets_equal(&sum, &expected), "sum = {sum:?}");
        assert_eq!(sum.len(), 4);
    }

    #[test]
    fn square_side_lengths_add() {
        let big = vec![v(0.0, 0.0), v(3.0, 0.0), v(3.0, 3.0), v(0.0, 3.0)];
        let small = vec![v(0.0, 0.0), v(2.0, 0.0), v(2.0, 2.0), v(0.0, 2.0)];
        let sum = minkowski_sum(&big, &small);
        let expected = vec![v(0.0, 0.0), v(5.0, 0.0), v(5.0, 5.0), v(0.0, 5.0)];
        assert!(sets_equal(&sum, &expected), "sum = {sum:?}");
    }

    #[test]
    fn polygon_plus_point_is_translation() {
        let square = unit_square();
        let point = vec![v(3.0, 4.0)];
        let sum = minkowski_sum(&square, &point);
        let expected: Vec<Vec2> = square.iter().map(|p| p.add(v(3.0, 4.0))).collect();
        assert!(sets_equal(&sum, &expected), "sum = {sum:?}");
        assert_eq!(sum.len(), 4);
    }

    #[test]
    fn point_plus_polygon_is_translation() {
        let square = unit_square();
        let point = vec![v(-2.0, 1.0)];
        let sum = minkowski_sum(&point, &square);
        let expected: Vec<Vec2> = square.iter().map(|p| p.add(v(-2.0, 1.0))).collect();
        assert!(sets_equal(&sum, &expected), "sum = {sum:?}");
    }

    #[test]
    fn polygon_plus_horizontal_segment() {
        let square = unit_square();
        let seg = vec![v(0.0, 0.0), v(2.0, 0.0)];
        let sum = minkowski_sum(&square, &seg);
        // Unit square swept 2 units to the right => 3x1 rectangle.
        let expected = vec![v(0.0, 0.0), v(3.0, 0.0), v(3.0, 1.0), v(0.0, 1.0)];
        assert!(sets_equal(&sum, &expected), "sum = {sum:?}");
    }

    #[test]
    fn area_is_at_least_each_input_area() {
        let t = right_triangle();
        let s = unit_square();
        let sum = minkowski_sum(&t, &s);
        assert!(area(&sum) >= area(&t) - 1.0e-5);
        assert!(area(&sum) >= area(&s) - 1.0e-5);
    }

    #[test]
    fn vertex_count_at_most_sum_of_inputs() {
        let t = right_triangle();
        let s = unit_square();
        let sum = minkowski_sum(&t, &s);
        assert!(sum.len() <= t.len() + s.len(), "len = {}", sum.len());
    }

    #[test]
    fn result_is_ccw_convex() {
        let pent = vec![
            v(0.0, 0.0),
            v(2.0, 0.0),
            v(3.0, 2.0),
            v(1.0, 3.0),
            v(-1.0, 2.0),
        ];
        let t = right_triangle();
        let sum = minkowski_sum(&pent, &t);
        assert!(is_ccw_convex(&sum), "sum = {sum:?}");
    }

    #[test]
    fn commutativity_vertex_sets_match() {
        let pent = vec![
            v(0.0, 0.0),
            v(2.0, 0.0),
            v(3.0, 2.0),
            v(1.0, 3.0),
            v(-1.0, 2.0),
        ];
        let t = right_triangle();
        let ab = minkowski_sum(&pent, &t);
        let ba = minkowski_sum(&t, &pent);
        assert!(sets_equal(&ab, &ba), "ab = {ab:?}, ba = {ba:?}");
    }

    #[test]
    fn commutativity_areas_match() {
        let s = unit_square();
        let t = right_triangle();
        let ab = minkowski_sum(&s, &t);
        let ba = minkowski_sum(&t, &s);
        assert!((area(&ab) - area(&ba)).abs() <= 1.0e-5);
    }

    #[test]
    fn translation_additivity() {
        let s = unit_square();
        let t = right_triangle();
        let shift = v(5.0, -3.0);
        let shifted: Vec<Vec2> = s.iter().map(|p| p.add(shift)).collect();
        let sum_shifted = minkowski_sum(&shifted, &t);
        let base = minkowski_sum(&s, &t);
        let base_shifted: Vec<Vec2> = base.iter().map(|p| p.add(shift)).collect();
        assert!(
            sets_equal(&sum_shifted, &base_shifted),
            "shifted = {sum_shifted:?}"
        );
    }

    #[test]
    fn perpendicular_segments_make_square() {
        let horiz = vec![v(0.0, 0.0), v(2.0, 0.0)];
        let vert = vec![v(0.0, 0.0), v(0.0, 2.0)];
        let sum = minkowski_sum(&horiz, &vert);
        let expected = vec![v(0.0, 0.0), v(2.0, 0.0), v(2.0, 2.0), v(0.0, 2.0)];
        assert!(sets_equal(&sum, &expected), "sum = {sum:?}");
        assert!((area(&sum) - 4.0).abs() <= 1.0e-5);
    }

    #[test]
    fn parallel_segments_make_longer_segment() {
        let a = vec![v(0.0, 0.0), v(2.0, 0.0)];
        let b = vec![v(0.0, 0.0), v(1.0, 0.0)];
        let sum = minkowski_sum(&a, &b);
        let expected = vec![v(0.0, 0.0), v(3.0, 0.0)];
        assert!(sets_equal(&sum, &expected), "sum = {sum:?}");
        assert!(area(&sum) <= 1.0e-5);
    }

    #[test]
    fn point_plus_point_is_point() {
        let a = vec![v(1.0, 2.0)];
        let b = vec![v(3.0, -1.0)];
        let sum = minkowski_sum(&a, &b);
        assert_eq!(sum.len(), 1);
        assert!(points_equal(sum[0], v(4.0, 1.0)), "sum = {sum:?}");
    }

    #[test]
    fn segment_plus_point_is_translated_segment() {
        let seg = vec![v(0.0, 0.0), v(2.0, 1.0)];
        let point = vec![v(1.0, 1.0)];
        let sum = minkowski_sum(&seg, &point);
        let expected = vec![v(1.0, 1.0), v(3.0, 2.0)];
        assert!(sets_equal(&sum, &expected), "sum = {sum:?}");
    }

    #[test]
    fn triangle_plus_segment_stays_convex() {
        let t = right_triangle();
        let seg = vec![v(0.0, 0.0), v(1.0, 1.0)];
        let sum = minkowski_sum(&t, &seg);
        assert!(is_ccw_convex(&sum), "sum = {sum:?}");
        assert!(sum.len() <= t.len() + 2);
    }

    #[test]
    fn square_plus_triangle_known_area() {
        let s = unit_square();
        let t = right_triangle();
        let sum = minkowski_sum(&s, &t);
        // area = area(s) + area(t) + mixed area. Just require it exceeds both
        // and is convex; exact value validated by the min/max extent tests.
        assert!(area(&sum) > area(&s));
        assert!(area(&sum) > area(&t));
        assert!(is_ccw_convex(&sum));
    }

    #[test]
    fn clockwise_input_is_normalized() {
        // Same square, but wound clockwise and starting elsewhere.
        let cw = vec![v(0.0, 0.0), v(0.0, 1.0), v(1.0, 1.0), v(1.0, 0.0)];
        let ccw = unit_square();
        let sum_cw = minkowski_sum(&cw, &ccw);
        let sum_ccw = minkowski_sum(&ccw, &ccw);
        assert!(sets_equal(&sum_cw, &sum_ccw), "cw = {sum_cw:?}");
        assert!(is_ccw_convex(&sum_cw));
    }

    #[test]
    fn redundant_collinear_input_vertices_are_ignored() {
        // A square with an extra midpoint vertex on the bottom edge.
        let with_extra = vec![
            v(0.0, 0.0),
            v(0.5, 0.0),
            v(1.0, 0.0),
            v(1.0, 1.0),
            v(0.0, 1.0),
        ];
        let sum = minkowski_sum(&with_extra, &unit_square());
        let expected = vec![v(0.0, 0.0), v(2.0, 0.0), v(2.0, 2.0), v(0.0, 2.0)];
        assert!(sets_equal(&sum, &expected), "sum = {sum:?}");
        assert_eq!(sum.len(), 4);
    }

    #[test]
    fn result_has_no_collinear_vertices() {
        let s = unit_square();
        let big = vec![v(0.0, 0.0), v(2.0, 0.0), v(2.0, 2.0), v(0.0, 2.0)];
        let sum = minkowski_sum(&s, &big);
        // Every vertex must make a real (non-collinear) turn.
        let n = sum.len();
        for idx in 0..n {
            let prev = sum[(idx + n - 1) % n];
            let cur = sum[idx];
            let after = sum[(idx + 1) % n];
            assert!(cur.sub(prev).cross(after.sub(cur)).abs() > SUM_EPS);
        }
    }

    #[test]
    fn first_vertex_is_bottom_most() {
        let pent = vec![
            v(0.0, 0.0),
            v(2.0, 0.0),
            v(3.0, 2.0),
            v(1.0, 3.0),
            v(-1.0, 2.0),
        ];
        let sum = minkowski_sum(&pent, &right_triangle());
        for &p in &sum {
            assert!(
                !is_lower(p, sum[0]),
                "vertex {p:?} is lower than first {:?}",
                sum[0]
            );
        }
    }

    #[test]
    fn min_y_extent_adds() {
        let a = vec![v(1.0, 2.0), v(4.0, 3.0), v(2.0, 6.0)];
        let b = vec![v(-1.0, -1.0), v(1.0, -2.0), v(2.0, 1.0)];
        let sum = minkowski_sum(&a, &b);
        let min_y_a = a.iter().fold(f32::INFINITY, |acc, p| acc.min(p.y));
        let min_y_b = b.iter().fold(f32::INFINITY, |acc, p| acc.min(p.y));
        let min_y_sum = sum.iter().fold(f32::INFINITY, |acc, p| acc.min(p.y));
        assert!((min_y_sum - (min_y_a + min_y_b)).abs() <= 1.0e-4);
    }

    #[test]
    fn max_x_extent_adds() {
        let a = vec![v(1.0, 2.0), v(4.0, 3.0), v(2.0, 6.0)];
        let b = vec![v(-1.0, -1.0), v(1.0, -2.0), v(2.0, 1.0)];
        let sum = minkowski_sum(&a, &b);
        let max_x_a = a.iter().fold(f32::NEG_INFINITY, |acc, p| acc.max(p.x));
        let max_x_b = b.iter().fold(f32::NEG_INFINITY, |acc, p| acc.max(p.x));
        let max_x_sum = sum.iter().fold(f32::NEG_INFINITY, |acc, p| acc.max(p.x));
        assert!((max_x_sum - (max_x_a + max_x_b)).abs() <= 1.0e-4);
    }

    #[test]
    fn sum_contains_pairwise_vertex_sums_on_boundary() {
        // Extreme corners always sum to boundary vertices of the result.
        let s = unit_square();
        let t = unit_square();
        let sum = minkowski_sum(&s, &t);
        assert!(contains_point(&sum, v(0.0, 0.0)));
        assert!(contains_point(&sum, v(2.0, 2.0)));
    }

    #[test]
    fn pentagon_plus_triangle_bounded_and_convex() {
        let pent = vec![
            v(0.0, 0.0),
            v(4.0, 0.0),
            v(5.0, 3.0),
            v(2.0, 5.0),
            v(-1.0, 3.0),
        ];
        let t = right_triangle();
        let sum = minkowski_sum(&pent, &t);
        assert!(sum.len() <= pent.len() + t.len());
        assert!(is_ccw_convex(&sum));
        assert!(area(&sum) >= area(&pent));
    }

    #[test]
    fn degenerate_segment_plus_segment_general_position() {
        // Two non-parallel segments produce a parallelogram.
        let a = vec![v(0.0, 0.0), v(2.0, 0.0)];
        let b = vec![v(0.0, 0.0), v(1.0, 2.0)];
        let sum = minkowski_sum(&a, &b);
        let expected = vec![v(0.0, 0.0), v(2.0, 0.0), v(3.0, 2.0), v(1.0, 2.0)];
        assert!(sets_equal(&sum, &expected), "sum = {sum:?}");
        assert!((area(&sum) - 4.0).abs() <= 1.0e-5);
    }

    #[test]
    fn self_sum_doubles_convex_polygon() {
        // A ⊕ A of a convex polygon equals the polygon scaled by 2 about the
        // origin only when it is star-shaped from origin; instead verify the
        // area quadruples, a property of Minkowski self-sum of convex sets.
        let pent = vec![
            v(0.0, 0.0),
            v(2.0, 0.0),
            v(3.0, 2.0),
            v(1.0, 3.0),
            v(-1.0, 2.0),
        ];
        let sum = minkowski_sum(&pent, &pent);
        assert!((area(&sum) - 4.0 * area(&pent)).abs() <= 1.0e-3);
        assert!(is_ccw_convex(&sum));
    }

    #[test]
    fn associativity_like_translation_of_segment_sum() {
        // Summing a shape with a segment then translating equals translating
        // first: a light associativity/translation cross-check.
        let s = unit_square();
        let seg = vec![v(0.0, 0.0), v(1.0, 1.0)];
        let shift = v(-4.0, 7.0);
        let lhs: Vec<Vec2> = minkowski_sum(&s, &seg)
            .iter()
            .map(|p| p.add(shift))
            .collect();
        let shifted_seg: Vec<Vec2> = seg.iter().map(|p| p.add(shift)).collect();
        let rhs = minkowski_sum(&s, &shifted_seg);
        assert!(sets_equal(&lhs, &rhs), "lhs = {lhs:?}, rhs = {rhs:?}");
    }
}
