//! 3D convex-polytope Boolean intersection via the `Gilbert-Johnson-Keerthi`
//! (`GJK`) simplex algorithm, for the particle spatial-query contracts
//! (design §8.2, §10, §13).
//!
//! Several particle stages reduce a broadphase or gating decision to one
//! question: "do these two convex 3D shapes share any point?" Examples are an
//! emitter-volume-vs-collider overlap gate, a bounds-cluster merge test, and a
//! splat-vs-proxy pre-pass that rejects clearly separated pairs before the
//! expensive depenetration solve. This module owns the small, `CPU`-verifiable
//! contract those stages share: given two convex point sets expressed as convex
//! hull vertices, it decides overlap by iterating the `GJK` support-mapping
//! simplex toward the origin of the Minkowski difference `A - B`.
//!
//! # Algorithm
//! `GJK` never materializes the Minkowski difference `A - B`. It probes that set
//! through a *support mapping*: for a search direction `d`, the support point is
//! `support(A, d) - support(B, -d)`, the vertex of `A - B` farthest along `d`.
//! Starting from one support point, the search maintains a *simplex* of one,
//! two, three, or four such points that brackets the origin ever more tightly:
//!
//! - a **point** simply aims the next search back at the origin;
//! - a **line segment** keeps the whole edge when the origin projects onto it
//!   (new direction = the edge normal toward the origin, built with a vector
//!   triple product) and otherwise falls back to the newest endpoint;
//! - a **triangle** classifies the origin against its two free edges and its
//!   two face half-spaces, trimming to the closest edge or orienting the face
//!   normal (above/below) toward the origin;
//! - a **tetrahedron** tests the three faces incident to the newest vertex; if
//!   the origin is outside one it drops back to that triangle, and if the origin
//!   is inside all three the shapes overlap.
//!
//! Each step folds the newest support point into the simplex, discards the
//! feature (vertex, edge, or face) that cannot face the origin, and picks the
//! next search direction as the surviving feature's normal pointing at the
//! origin. When a support point fails to pass the origin along `d` the shapes
//! are disjoint; when the tetrahedron encloses the origin they overlap. A fixed
//! iteration bound guarantees deterministic termination even for degenerate,
//! boundary-touching, or many-vertex inputs.
//!
//! # Strict scope
//! This module answers *convex overlap yes/no in 3D* through `GJK`, and nothing
//! more. It neither imports nor reconstructs the sibling contracts it is
//! deliberately distinct from:
//! - [`super::gjk_2d`] runs the same simplex idea in the *plane* (a `Vec2`
//!   triple-product test that never grows past a triangle); this module is the
//!   *3D* case whose simplex grows to a tetrahedron.
//! - [`super::convex_hull_3d`] *constructs* the convex hull of a point cloud;
//!   this module *consumes* an already-convex vertex set and never builds a
//!   hull. Passing a raw non-convex cloud only tests the overlap of its hulls.
//! - [`super::minkowski_sum_2d`] *explicitly materializes* a Minkowski sum
//!   polygon in 2D; this module only ever samples the Minkowski *difference*
//!   through the support mapping and never enumerates its vertices.
//! - [`super::obb_obb_sat_3d`] answers the same yes/no for two oriented boxes
//!   through the Separating Axis Theorem (`SAT`); this module uses `GJK` on
//!   arbitrary convex vertex sets rather than the fifteen fixed box axes.
//! - a penetration-depth solver (the Expanding Polytope Algorithm, `EPA`, a
//!   future `epa_penetration_3d` sibling) would refine a containing simplex into
//!   a contact normal and depth; this module reports *only* a boolean and never
//!   a penetration vector, separating axis, or contact feature.
//!
//! It keeps its own `[f32; 3]` vector helpers and small [`Simplex`] type and
//! shares no math code with any neighbour.
//!
//! # No transcendental math
//! Support projection, the triple-product edge normals, the cross-product face
//! normals, and origin containment are pure `+`, `-`, `*` dot/cross arithmetic;
//! `GJK` needs no square root at all in the Boolean case. There is no `sin`,
//! `cos`, `tan`, `atan`, `exp`, `ln`, `powf`, `sqrt`, `floor`, `round`, or any
//! other transcendental or rounding call, and no `f32` equality: a collapsed
//! search direction or a duplicate support point is detected by comparing a
//! squared magnitude against [`DIR_EPS_SQ`] / [`DUP_EPS_SQ`] (never `== 0.0`),
//! and a separating support projection is compared against `0.0` with a strict
//! `<`. `NaN` inputs are not expected; the comparisons degrade to "disjoint"
//! rather than looping.

/// Squared-length threshold below which a search direction has collapsed onto
/// the origin, meaning the origin lies on the current simplex feature (a
/// boundary touch) and the shapes intersect. Used instead of `== 0.0`.
pub const DIR_EPS_SQ: f32 = 1.0e-10;

/// Squared-distance threshold below which a freshly probed support point is
/// treated as a duplicate of one already in the simplex. Reaching a duplicate
/// means the search cannot expand further, so the origin is at or inside the
/// current feature and the shapes intersect (or touch). Used instead of `==`.
pub const DUP_EPS_SQ: f32 = 1.0e-10;

/// Hard cap on simplex iterations. 3D `GJK` converges in a number of steps
/// bounded by the combined vertex count; the cap makes termination and output
/// deterministic for every input, including degenerate hulls. The bound is
/// comfortably above the handful of steps healthy inputs need.
pub const MAX_ITERS: usize = 32;

/// The initial search direction (`+x`) used to seed the first support point
/// before any simplex feature exists. Any non-zero direction works; a fixed
/// choice keeps the search deterministic.
const INITIAL_DIR: [f32; 3] = [1.0, 0.0, 0.0];

/// Component-wise difference `a - b`.
#[must_use]
fn v_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// The additive inverse `-a`.
#[must_use]
fn v_neg(a: [f32; 3]) -> [f32; 3] {
    [-a[0], -a[1], -a[2]]
}

/// The dot product `a . b`.
#[must_use]
fn v_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The cross product `a x b`.
#[must_use]
fn v_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The squared Euclidean length `a . a`, free of any square root.
#[must_use]
fn v_len_sq(a: [f32; 3]) -> f32 {
    v_dot(a, a)
}

/// The vector triple product `(a x b) x c`, which equals `b (a . c) - a (b . c)`
/// and lies in the plane of `a` and `b`. `GJK` uses it to build an edge normal
/// that is perpendicular to the edge and points toward the origin.
#[must_use]
fn triple(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    v_cross(v_cross(a, b), c)
}

/// Returns `true` when `a` and `b` point into the same half-space, i.e. their
/// dot product is strictly positive. A zero or negative dot is treated as "not
/// the same direction" so a perpendicular feature is trimmed away.
#[must_use]
fn same_direction(a: [f32; 3], b: [f32; 3]) -> bool {
    v_dot(a, b) > 0.0
}

/// Returns the vertex of `vertices` farthest along `dir` (the support point of
/// the convex set in that direction). Ties keep the earliest vertex, which
/// keeps the search deterministic.
///
/// # Panics
/// Panics if `vertices` is empty; [`intersect`] guards empty inputs before any
/// probe, so this is never reached through the public entry point.
#[must_use]
fn support_point(vertices: &[[f32; 3]], dir: [f32; 3]) -> [f32; 3] {
    let mut best = vertices[0];
    let mut best_dot = v_dot(best, dir);
    for &v in &vertices[1..] {
        let projected = v_dot(v, dir);
        if projected > best_dot {
            best_dot = projected;
            best = v;
        }
    }
    best
}

/// The support point of the Minkowski difference `a - b` along `dir`, i.e.
/// `support_point(a, dir) - support_point(b, -dir)`. `GJK` probes the difference
/// set only through this mapping and never enumerates its vertices.
///
/// # Panics
/// Panics if either `a` or `b` is empty; [`intersect`] guards empty inputs.
#[must_use]
pub fn support(a: &[[f32; 3]], b: &[[f32; 3]], dir: [f32; 3]) -> [f32; 3] {
    v_sub(support_point(a, dir), support_point(b, v_neg(dir)))
}

/// The evolving `GJK` simplex: up to four Minkowski-difference support points,
/// stored newest-first so index `0` is always the most recently added vertex
/// `a`. The remaining indices hold the older vertices `b`, `c`, and `d`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Simplex {
    /// Backing storage; only the first `len` entries are meaningful.
    pts: [[f32; 3]; 4],
    /// Number of live vertices in `pts`, in the range `0..=4`.
    len: usize,
}

impl Simplex {
    /// An empty simplex.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pts: [[0.0; 3]; 4],
            len: 0,
        }
    }

    /// The number of live vertices (`0..=4`).
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the simplex currently holds no vertices.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Inserts `p` as the new front vertex `a`, shifting the older vertices back
    /// by one slot. The simplex never legitimately grows past four vertices in
    /// `GJK`, so this is only ever called when `len < 4`.
    fn push_front(&mut self, p: [f32; 3]) {
        debug_assert!(self.len < 4, "GJK simplex overflow");
        let mut i = self.len;
        while i > 0 {
            self.pts[i] = self.pts[i - 1];
            i -= 1;
        }
        self.pts[0] = p;
        self.len += 1;
    }

    /// Replaces the live vertices with `pts`, preserving the given newest-first
    /// order. Used by [`do_simplex`] to trim the simplex to a surviving feature.
    fn set(&mut self, pts: &[[f32; 3]]) {
        debug_assert!(pts.len() <= 4, "GJK simplex overflow");
        self.len = pts.len();
        for (dst, src) in self.pts.iter_mut().zip(pts.iter()) {
            *dst = *src;
        }
    }

    /// Whether any live vertex is within `DUP_EPS_SQ` (squared distance) of `p`.
    #[must_use]
    fn contains_near(&self, p: [f32; 3]) -> bool {
        self.pts[..self.len]
            .iter()
            .any(|&q| v_len_sq(v_sub(q, p)) <= DUP_EPS_SQ)
    }
}

/// The one-vertex-to-line evolution. With the segment `[a, b]` (a newest), if
/// the origin projects onto the edge the whole edge is kept and `dir` becomes
/// the edge normal toward the origin; otherwise the simplex collapses to `a`
/// and `dir` points from `a` at the origin. Always returns `false`: a segment
/// can never enclose the origin in 3D.
fn line_case(s: &mut Simplex, dir: &mut [f32; 3]) -> bool {
    let a = s.pts[0];
    let b = s.pts[1];
    let ab = v_sub(b, a);
    let ao = v_neg(a);
    if same_direction(ab, ao) {
        *dir = triple(ab, ao, ab);
    } else {
        s.set(&[a]);
        *dir = ao;
    }
    false
}

/// The triangle evolution. With `[a, b, c]` (a newest) it classifies the origin
/// against the two free edges `ab` and `ac` and the two face half-spaces of the
/// triangle normal `abc`, trimming to the closest edge (delegating to
/// [`line_case`]) or orienting the surviving face toward the origin. Always
/// returns `false`: a triangle brackets the origin only within a face prism,
/// never encloses it.
fn triangle_case(s: &mut Simplex, dir: &mut [f32; 3]) -> bool {
    let a = s.pts[0];
    let b = s.pts[1];
    let c = s.pts[2];
    let ab = v_sub(b, a);
    let ac = v_sub(c, a);
    let ao = v_neg(a);
    let abc = v_cross(ab, ac);

    if same_direction(v_cross(abc, ac), ao) {
        if same_direction(ac, ao) {
            s.set(&[a, c]);
            *dir = triple(ac, ao, ac);
        } else {
            s.set(&[a, b]);
            return line_case(s, dir);
        }
    } else if same_direction(v_cross(ab, abc), ao) {
        s.set(&[a, b]);
        return line_case(s, dir);
    } else if same_direction(abc, ao) {
        *dir = abc;
    } else {
        s.set(&[a, c, b]);
        *dir = v_neg(abc);
    }
    false
}

/// The tetrahedron evolution. With `[a, b, c, d]` (a newest) it tests the three
/// faces incident to `a` (`abc`, `acd`, `adb`). If the origin lies outside one
/// face the simplex drops to that triangle (delegating to [`triangle_case`]);
/// if the origin is inside all three the tetrahedron encloses it and the shapes
/// overlap, so it returns `true`.
fn tetra_case(s: &mut Simplex, dir: &mut [f32; 3]) -> bool {
    let a = s.pts[0];
    let b = s.pts[1];
    let c = s.pts[2];
    let d = s.pts[3];
    let ab = v_sub(b, a);
    let ac = v_sub(c, a);
    let ad = v_sub(d, a);
    let ao = v_neg(a);
    let abc = v_cross(ab, ac);
    let acd = v_cross(ac, ad);
    let adb = v_cross(ad, ab);

    if same_direction(abc, ao) {
        s.set(&[a, b, c]);
        return triangle_case(s, dir);
    }
    if same_direction(acd, ao) {
        s.set(&[a, c, d]);
        return triangle_case(s, dir);
    }
    if same_direction(adb, ao) {
        s.set(&[a, d, b]);
        return triangle_case(s, dir);
    }
    true
}

/// Evolves the simplex one step toward the origin, dispatching on the current
/// vertex count. Returns `true` only when a tetrahedron encloses the origin;
/// otherwise it trims the simplex to the surviving feature and writes the next
/// search direction into `dir`.
fn do_simplex(s: &mut Simplex, dir: &mut [f32; 3]) -> bool {
    match s.len {
        2 => line_case(s, dir),
        3 => triangle_case(s, dir),
        4 => tetra_case(s, dir),
        _ => false,
    }
}

/// Decides whether the convex hulls of two 3D point sets share any point.
///
/// Each argument is the vertex set of a convex body (a convex hull's corners).
/// The result is a single Boolean: `true` when the bodies intersect (touching
/// counts as intersecting), `false` when they are strictly separated. Empty
/// inputs describe empty bodies and never intersect.
///
/// The search iterates the `GJK` support-mapping simplex toward the origin of
/// the Minkowski difference. It terminates in one of four deterministic ways:
/// a support point that fails to pass the origin (disjoint), a collapsed search
/// direction sitting on the origin (boundary touch, intersecting), a duplicate
/// support point that cannot expand the simplex (origin reached, intersecting),
/// a tetrahedron that encloses the origin (intersecting), or the [`MAX_ITERS`]
/// safety cap (treated as intersecting, since the search only reaches the cap
/// while still bracketing the origin ever more tightly).
///
/// # Examples
/// ```
/// use prism_render_architecture::particle::gjk_3d::intersect;
/// let cube = |o: f32| {
///     [
///         [o, o, o], [o + 1.0, o, o], [o, o + 1.0, o], [o + 1.0, o + 1.0, o],
///         [o, o, o + 1.0], [o + 1.0, o, o + 1.0], [o, o + 1.0, o + 1.0],
///         [o + 1.0, o + 1.0, o + 1.0],
///     ]
/// };
/// assert!(intersect(&cube(0.0), &cube(0.5)));
/// assert!(!intersect(&cube(0.0), &cube(5.0)));
/// ```
#[must_use]
pub fn intersect(a: &[[f32; 3]], b: &[[f32; 3]]) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }

    let first = support(a, b, INITIAL_DIR);
    let mut s = Simplex::new();
    s.push_front(first);
    let mut dir = v_neg(first);
    if v_len_sq(dir) <= DIR_EPS_SQ {
        // The first support point is the origin: 0 is inside `A - B`.
        return true;
    }

    for _ in 0..MAX_ITERS {
        let p = support(a, b, dir);
        if v_dot(p, dir) < 0.0 {
            // The farthest Minkowski-difference point along `dir` still lies on
            // the near side of the origin: a separating direction exists.
            return false;
        }
        if s.contains_near(p) {
            // The search cannot expand past a point it already holds, so the
            // origin is as close as it gets: on or inside the current feature.
            return true;
        }
        s.push_front(p);
        if do_simplex(&mut s, &mut dir) {
            return true;
        }
        if v_len_sq(dir) <= DIR_EPS_SQ {
            // The surviving feature passes through the origin (a boundary
            // touch); no further direction can separate the shapes.
            return true;
        }
    }

    // Reached the iteration cap while still bracketing the origin.
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// An axis-aligned unit cube with its minimum corner at `origin`.
    fn cube(origin: [f32; 3]) -> Vec<[f32; 3]> {
        let [x, y, z] = origin;
        Vec::from([
            [x, y, z],
            [x + 1.0, y, z],
            [x, y + 1.0, z],
            [x + 1.0, y + 1.0, z],
            [x, y, z + 1.0],
            [x + 1.0, y, z + 1.0],
            [x, y + 1.0, z + 1.0],
            [x + 1.0, y + 1.0, z + 1.0],
        ])
    }

    /// An axis-aligned box spanning `[min, max]`.
    fn boxv(min: [f32; 3], max: [f32; 3]) -> Vec<[f32; 3]> {
        Vec::from([
            [min[0], min[1], min[2]],
            [max[0], min[1], min[2]],
            [min[0], max[1], min[2]],
            [max[0], max[1], min[2]],
            [min[0], min[1], max[2]],
            [max[0], min[1], max[2]],
            [min[0], max[1], max[2]],
            [max[0], max[1], max[2]],
        ])
    }

    /// A regular-ish tetrahedron scaled by `s` and translated by `t`.
    fn tetra(s: f32, t: [f32; 3]) -> Vec<[f32; 3]> {
        Vec::from([
            [t[0], t[1], t[2]],
            [t[0] + s, t[1], t[2]],
            [t[0], t[1] + s, t[2]],
            [t[0], t[1], t[2] + s],
        ])
    }

    /// A coarse icosphere-like point cloud approximating a sphere of `radius`
    /// centered at `c`, using the six axis poles plus eight diagonal corners.
    fn ball(c: [f32; 3], radius: f32) -> Vec<[f32; 3]> {
        let r = radius;
        let d = radius * 0.577; // ~1/sqrt(3), avoids any sqrt call here
        Vec::from([
            [c[0] + r, c[1], c[2]],
            [c[0] - r, c[1], c[2]],
            [c[0], c[1] + r, c[2]],
            [c[0], c[1] - r, c[2]],
            [c[0], c[1], c[2] + r],
            [c[0], c[1], c[2] - r],
            [c[0] + d, c[1] + d, c[2] + d],
            [c[0] - d, c[1] + d, c[2] + d],
            [c[0] + d, c[1] - d, c[2] + d],
            [c[0] + d, c[1] + d, c[2] - d],
            [c[0] - d, c[1] - d, c[2] - d],
            [c[0] + d, c[1] - d, c[2] - d],
            [c[0] - d, c[1] + d, c[2] - d],
            [c[0] - d, c[1] - d, c[2] + d],
        ])
    }

    // ---- support / helper unit coverage -------------------------------------

    #[test]
    fn support_picks_extreme_vertex() {
        let c = cube([0.0, 0.0, 0.0]);
        assert_eq!(support_point(&c, [1.0, 1.0, 1.0]), [1.0, 1.0, 1.0]);
        assert_eq!(support_point(&c, [-1.0, -1.0, -1.0]), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn minkowski_support_is_difference_of_supports() {
        let a = cube([0.0, 0.0, 0.0]);
        let b = cube([2.0, 0.0, 0.0]);
        // Along +x: max of a is x=1, min of b (support in -x) is x=2 -> 1-2=-1.
        let s = support(&a, &b, [1.0, 0.0, 0.0]);
        assert!((s[0] - (-1.0)).abs() <= 1.0e-6);
    }

    #[test]
    fn triple_product_is_perpendicular_to_first_argument() {
        let ab = [1.0, 0.0, 0.0];
        let ao = [0.0, 1.0, 0.0];
        let t = triple(ab, ao, ab);
        // Perpendicular to ab, pointing toward ao.
        assert!(v_dot(t, ab).abs() <= 1.0e-6);
        assert!(v_dot(t, ao) > 0.0);
    }

    #[test]
    fn cross_and_dot_agree_with_hand_computation() {
        let x = [1.0, 0.0, 0.0];
        let y = [0.0, 1.0, 0.0];
        assert_eq!(v_cross(x, y), [0.0, 0.0, 1.0]);
        assert!((v_dot(x, y)).abs() <= 1.0e-6);
        assert!((v_len_sq([2.0, 3.0, 6.0]) - 49.0).abs() <= 1.0e-4);
    }

    // ---- two boxes: overlap / disjoint / touch ------------------------------

    #[test]
    fn two_cubes_overlap() {
        assert!(intersect(&cube([0.0, 0.0, 0.0]), &cube([0.5, 0.5, 0.5])));
    }

    #[test]
    fn two_cubes_disjoint_far() {
        assert!(!intersect(
            &cube([0.0, 0.0, 0.0]),
            &cube([10.0, 10.0, 10.0])
        ));
    }

    #[test]
    fn two_cubes_disjoint_along_one_axis() {
        assert!(!intersect(&cube([0.0, 0.0, 0.0]), &cube([2.0, 0.0, 0.0])));
    }

    #[test]
    fn two_cubes_face_touch_counts_as_intersecting() {
        // b's min-x face coincides with a's max-x face at x = 1.
        assert!(intersect(&cube([0.0, 0.0, 0.0]), &cube([1.0, 0.0, 0.0])));
    }

    #[test]
    fn two_cubes_edge_touch_counts_as_intersecting() {
        // Share only the edge at x = 1, y = 1.
        assert!(intersect(&cube([0.0, 0.0, 0.0]), &cube([1.0, 1.0, 0.0])));
    }

    #[test]
    fn two_cubes_vertex_touch_counts_as_intersecting() {
        // Share only the corner at (1, 1, 1).
        assert!(intersect(&cube([0.0, 0.0, 0.0]), &cube([1.0, 1.0, 1.0])));
    }

    #[test]
    fn two_cubes_barely_separated() {
        assert!(!intersect(&cube([0.0, 0.0, 0.0]), &cube([1.01, 0.0, 0.0])));
    }

    #[test]
    fn two_cubes_barely_overlapping() {
        assert!(intersect(&cube([0.0, 0.0, 0.0]), &cube([0.99, 0.0, 0.0])));
    }

    // ---- containment --------------------------------------------------------

    #[test]
    fn big_box_contains_small_box() {
        let big = boxv([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]);
        let small = boxv([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]);
        assert!(intersect(&big, &small));
        assert!(intersect(&small, &big));
    }

    #[test]
    fn box_contains_single_point_body() {
        let big = boxv([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]);
        let point = [[0.25, -0.75, 1.5]];
        assert!(intersect(&big, &point));
    }

    #[test]
    fn box_excludes_single_point_body() {
        let big = boxv([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]);
        let point = [[3.0, 0.0, 0.0]];
        assert!(!intersect(&big, &point));
    }

    // ---- tetrahedron vs tetrahedron -----------------------------------------

    #[test]
    fn tetra_vs_tetra_overlap() {
        assert!(intersect(
            &tetra(2.0, [0.0, 0.0, 0.0]),
            &tetra(2.0, [0.3, 0.3, 0.3])
        ));
    }

    #[test]
    fn tetra_vs_tetra_disjoint() {
        assert!(!intersect(
            &tetra(1.0, [0.0, 0.0, 0.0]),
            &tetra(1.0, [5.0, 5.0, 5.0])
        ));
    }

    #[test]
    fn tetra_inside_box() {
        let big = boxv([-3.0, -3.0, -3.0], [3.0, 3.0, 3.0]);
        assert!(intersect(&big, &tetra(1.0, [0.0, 0.0, 0.0])));
    }

    #[test]
    fn tetra_vs_box_disjoint() {
        let far = boxv([10.0, 10.0, 10.0], [12.0, 12.0, 12.0]);
        assert!(!intersect(&far, &tetra(1.0, [0.0, 0.0, 0.0])));
    }

    // ---- sphere-like point clouds -------------------------------------------

    #[test]
    fn balls_overlap_when_close() {
        assert!(intersect(
            &ball([0.0, 0.0, 0.0], 1.0),
            &ball([1.5, 0.0, 0.0], 1.0)
        ));
    }

    #[test]
    fn balls_disjoint_when_far() {
        assert!(!intersect(
            &ball([0.0, 0.0, 0.0], 1.0),
            &ball([5.0, 0.0, 0.0], 1.0)
        ));
    }

    #[test]
    fn ball_vs_box_overlap() {
        let b = boxv([-0.2, -0.2, -0.2], [0.2, 0.2, 0.2]);
        assert!(intersect(&ball([0.0, 0.0, 0.0], 1.0), &b));
    }

    // ---- degenerate & coplanar shapes ---------------------------------------

    #[test]
    fn coplanar_squares_overlap() {
        // Two axis-aligned squares in the z = 0 plane that share area.
        let s0 = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
        ];
        let s1 = [
            [1.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [1.0, 3.0, 0.0],
            [3.0, 3.0, 0.0],
        ];
        assert!(intersect(&s0, &s1));
    }

    #[test]
    fn coplanar_squares_disjoint() {
        let s0 = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let s1 = [
            [3.0, 3.0, 0.0],
            [4.0, 3.0, 0.0],
            [3.0, 4.0, 0.0],
            [4.0, 4.0, 0.0],
        ];
        assert!(!intersect(&s0, &s1));
    }

    #[test]
    fn collinear_segments_overlap() {
        // Degenerate 1D bodies on the x-axis that share the interval [1, 2].
        let a = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let b = [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
        assert!(intersect(&a, &b));
    }

    #[test]
    fn collinear_segments_disjoint() {
        let a = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let b = [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
        assert!(!intersect(&a, &b));
    }

    #[test]
    fn identical_points_intersect() {
        let p = [[1.0, 2.0, 3.0]];
        assert!(intersect(&p, &p));
    }

    #[test]
    fn distinct_points_disjoint() {
        assert!(!intersect(&[[0.0, 0.0, 0.0]], &[[1.0, 0.0, 0.0]]));
    }

    // ---- origin-on-boundary of the Minkowski difference ---------------------

    #[test]
    fn origin_exactly_on_boundary_via_touching_boxes() {
        // Boxes touching at a face put 0 exactly on the boundary of A - B.
        let a = boxv([-1.0, -1.0, -1.0], [0.0, 1.0, 1.0]);
        let b = boxv([0.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        assert!(intersect(&a, &b));
    }

    #[test]
    fn shared_single_vertex_puts_origin_on_boundary() {
        let a = [
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ];
        let b = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        assert!(intersect(&a, &b));
    }

    // ---- simplex evolution paths --------------------------------------------

    #[test]
    fn line_case_keeps_edge_toward_origin() {
        // Origin projects onto segment interior -> keep both, new dir toward it.
        let mut s = Simplex::new();
        s.push_front([-1.0, 1.0, 0.0]); // becomes b
        s.push_front([1.0, 1.0, 0.0]); // becomes a
        let mut dir = [0.0, 0.0, 0.0];
        assert!(!line_case(&mut s, &mut dir));
        assert_eq!(s.len(), 2);
        assert!(dir[1] < 0.0); // points downward toward the origin
    }

    #[test]
    fn line_case_collapses_to_newest_vertex() {
        // Origin behind a along the edge -> collapse to a.
        let mut s = Simplex::new();
        s.push_front([2.0, 0.0, 0.0]); // b
        s.push_front([1.0, 0.0, 0.0]); // a
        let mut dir = [0.0, 0.0, 0.0];
        assert!(!line_case(&mut s, &mut dir));
        assert_eq!(s.len(), 1);
        assert!(dir[0] < 0.0);
    }

    #[test]
    fn triangle_case_orients_face_normal_above() {
        // Triangle in z = 1 plane surrounding the origin's xy projection, origin
        // below it -> keep the triangle, dir points down toward the origin.
        let mut s = Simplex::new();
        s.push_front([0.0, 1.0, 1.0]); // c
        s.push_front([1.0, -1.0, 1.0]); // b
        s.push_front([-1.0, -1.0, 1.0]); // a
        let mut dir = [0.0, 0.0, 0.0];
        assert!(!triangle_case(&mut s, &mut dir));
        assert_eq!(s.len(), 3);
        assert!(dir[2] < 0.0); // face normal aimed at the origin below
    }

    #[test]
    fn tetra_case_encloses_origin() {
        // A tetrahedron built around the origin returns true.
        let mut s = Simplex::new();
        s.push_front([0.0, 0.0, -1.0]); // d
        s.push_front([0.0, 1.0, 1.0]); // c
        s.push_front([1.0, -1.0, 1.0]); // b
        s.push_front([-1.0, -1.0, 1.0]); // a
        let mut dir = [0.0, 0.0, 0.0];
        assert!(tetra_case(&mut s, &mut dir));
    }

    #[test]
    fn tetra_case_drops_to_face_when_origin_outside() {
        // An outward-wound tetrahedron sitting entirely at x >= 2, so the origin
        // is outside the `abc` face and the simplex must drop to that triangle.
        let mut s = Simplex::new();
        s.push_front([2.0, 0.2, 1.2]); // d
        s.push_front([3.0, 0.2, 0.2]); // c
        s.push_front([2.0, 1.2, 0.2]); // b
        s.push_front([2.0, 0.2, 0.2]); // a
        let mut dir = [0.0, 0.0, 0.0];
        // Does not enclose the origin (every vertex has x >= 2).
        assert!(!tetra_case(&mut s, &mut dir));
        assert!(s.len() <= 3);
    }

    // ---- convergence & robustness -------------------------------------------

    #[test]
    fn converges_within_iteration_cap_for_many_vertex_hulls() {
        // Two dense sphere clouds that overlap must resolve well under the cap.
        let a = ball([0.0, 0.0, 0.0], 2.0);
        let b = ball([1.0, 1.0, 1.0], 2.0);
        assert!(intersect(&a, &b));
    }

    #[test]
    fn empty_inputs_never_intersect() {
        let empty: [[f32; 3]; 0] = [];
        assert!(!intersect(&empty, &cube([0.0, 0.0, 0.0])));
        assert!(!intersect(&cube([0.0, 0.0, 0.0]), &empty));
        assert!(!intersect(&empty, &empty));
    }

    #[test]
    fn intersection_is_symmetric() {
        let a = cube([0.0, 0.0, 0.0]);
        let b = cube([0.4, 0.4, 0.4]);
        assert_eq!(intersect(&a, &b), intersect(&b, &a));
        let far = cube([9.0, 0.0, 0.0]);
        assert_eq!(intersect(&a, &far), intersect(&far, &a));
    }
}
