//! 2D convex-polygon overlap detection via the Separating Axis Theorem (`SAT`),
//! with a minimum-translation-vector (`MTV`) resolution for the particle
//! collision-broadphase contracts (design §8.2, §10).
//!
//! A handful of particle stages need a cheap, exact answer to "do these two
//! convex 2D shapes overlap, and if so by how much and along which direction?":
//! a splat-vs-splat broadphase, a footprint-vs-footprint depenetration hint,
//! and a screen-space cluster-vs-cluster separation test. This module owns the
//! small, `CPU`-verifiable contract those stages share: given two convex
//! polygons expressed as `CCW` (counter-clockwise) vertex rings, it decides
//! overlap with the Separating Axis Theorem and, when they overlap, reports the
//! minimum translation vector that pushes them apart.
//!
//! # Algorithm
//! For every edge of *both* polygons the outward edge normal is a *candidate
//! separating axis*. Each polygon is projected onto the (normalized) axis to a
//! `[min, max]` interval; if any axis yields a gap between the two intervals the
//! polygons are disjoint (the theorem guarantees a separating axis exists among
//! the face normals of two convex polygons). When no axis separates them, the
//! axis of *smallest* interval overlap is the minimum translation vector: its
//! direction (oriented from `a` toward `b`) and its penetration depth are what
//! [`mtv`] returns. Moving `b` by `axis * depth` (or `a` by `-axis * depth`)
//! resolves the penetration.
//!
//! # Strict scope
//! This module only performs *convex-polygon `SAT` overlap + `MTV`*. It does not
//! clip polygons ([`super::plane_clip`] owns Sutherland-Hodgman convex
//! clipping), it does not resolve analytic-primitive collision response
//! ([`super::collision`]), and it does not build or test oriented bounding boxes
//! ([`super::decal`]). It neither imports nor reconstructs those contracts and
//! keeps its own [`Vec2`] and math helpers.
//!
//! # No transcendental math
//! Projection, interval overlap and depth selection are pure `+`, `-`, `*`
//! comparison arithmetic; the only irrational operation is the `f32::sqrt` used
//! to normalize a separating axis. There is no `sin`, `cos`, `atan`, `exp`,
//! `ln`, `powf`, `ceil`, `round` or any other transcendental/rounding call, and
//! no `f32` equality: near-equal magnitudes are compared against [`SAT_EPS`].

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE, VEC4_STRIDE};

/// Magnitude below which a projection gap, an axis length, or a penetration
/// depth is treated as zero. This is the comparison rule used throughout
/// instead of `==` on `f32`: two scalars are "equal" when their absolute
/// difference does not exceed this bound, and a gap is a real separation only
/// when it exceeds it.
pub const SAT_EPS: f32 = 1.0e-6;

/// `std430` byte size of a serialized [`Mtv`]: one `vec4<f32>` slot holding
/// `[axis.x, axis.y, depth, 0.0]`.
pub const MTV_STD430_SIZE: usize = VEC4_STRIDE;

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

    /// Uniform scale by a scalar.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s)
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

    /// Dot product.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y
    }

    /// The left perpendicular `(-y, x)`, i.e. this vector rotated by
    /// `+90` degrees. Used to turn an edge direction into its normal axis.
    #[must_use]
    pub fn perp(self) -> Self {
        Self::new(-self.y, self.x)
    }

    /// Squared Euclidean length (no `sqrt`).
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    /// Euclidean length.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Returns the unit vector along `self`, or [`Vec2::ZERO`] when `self` is
    /// shorter than [`SAT_EPS`] (a degenerate edge), so no `NaN` ever escapes.
    #[must_use]
    pub fn normalize_or_zero(self) -> Self {
        let len = self.length();
        if len <= SAT_EPS {
            Self::ZERO
        } else {
            self.scale(1.0 / len)
        }
    }

    /// Returns `true` when both components match `other` within [`SAT_EPS`].
    #[must_use]
    pub fn approx_eq(self, other: Self) -> bool {
        (self.x - other.x).abs() <= SAT_EPS && (self.y - other.y).abs() <= SAT_EPS
    }
}

/// A minimum translation vector: the shortest push that separates two
/// overlapping convex polygons.
///
/// `axis` is a unit direction oriented from polygon `a` toward polygon `b`;
/// `depth` (always `>= 0`) is how far along that axis the polygons interpenetrate.
/// Translating `b` by `axis.scale(depth)`, or `a` by `axis.scale(-depth)`,
/// resolves the overlap. A `depth` of zero (within [`SAT_EPS`]) denotes a
/// touching contact rather than interpenetration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mtv {
    /// Unit push direction oriented from `a` toward `b`.
    pub axis: Vec2,
    /// Penetration depth along `axis`, never negative.
    pub depth: f32,
}

impl Mtv {
    /// Builds a minimum translation vector from its parts.
    #[must_use]
    pub const fn new(axis: Vec2, depth: f32) -> Self {
        Self { axis, depth }
    }

    /// The vector `b` must move by to just separate from `a`.
    #[must_use]
    pub fn push_b(self) -> Vec2 {
        self.axis.scale(self.depth)
    }

    /// The vector `a` must move by to just separate from `b`.
    #[must_use]
    pub fn push_a(self) -> Vec2 {
        self.axis.scale(-self.depth)
    }

    /// Serializes the `MTV` to its little-endian `std430` byte image.
    ///
    /// Layout: a single `vec4<f32>` slot `[axis.x, axis.y, depth, 0.0]`, matching
    /// how a `WESL` kernel would read one `vec4<f32>` load.
    #[must_use]
    pub fn to_std430(self) -> [u8; MTV_STD430_SIZE] {
        let mut bytes = [0u8; MTV_STD430_SIZE];
        let words = [self.axis.x, self.axis.y, self.depth, 0.0];
        for (i, word) in words.iter().enumerate() {
            let start = i * U32_STRIDE;
            bytes[start..start + U32_STRIDE].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// Total `std430` byte size for a storage buffer of `count` [`Mtv`] records,
/// clamped up to one element so a `WebGPU` binding is never zero-sized.
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(MTV_STD430_SIZE, count)
}

/// Projects `poly` onto `axis`, returning the `[min, max]` scalar interval of
/// the vertex dot products. `axis` need not be unit length; callers normalize
/// it first so the resulting depths are comparable across axes.
///
/// Returns `None` for an empty polygon, which has no projection.
#[must_use]
fn project(poly: &[Vec2], axis: Vec2) -> Option<(f32, f32)> {
    let mut iter = poly.iter();
    let first = iter.next()?;
    let mut lo = first.dot(axis);
    let mut hi = lo;
    for v in iter {
        let d = v.dot(axis);
        if d < lo {
            lo = d;
        }
        if d > hi {
            hi = d;
        }
    }
    Some((lo, hi))
}

/// The arithmetic mean of a polygon's vertices, used only to orient the `MTV`
/// axis from `a` toward `b`. Returns [`Vec2::ZERO`] for an empty polygon.
#[must_use]
fn centroid(poly: &[Vec2]) -> Vec2 {
    if poly.is_empty() {
        return Vec2::ZERO;
    }
    let mut sum = Vec2::ZERO;
    for v in poly {
        sum = sum.add(*v);
    }
    sum.scale(1.0 / poly.len() as f32)
}

/// Collects the normalized outward edge-normal candidate axes of `poly`.
///
/// For a `CCW` ring the edge `v[i] -> v[i + 1]` has direction `e`; its normal is
/// `e.perp()`. Degenerate (zero-length) edges are skipped so no zero axis ever
/// enters the projection loop. A polygon with fewer than two vertices yields no
/// axes.
#[must_use]
fn edge_axes(poly: &[Vec2]) -> Vec<Vec2> {
    let n = poly.len();
    let mut axes = Vec::new();
    if n < 2 {
        return axes;
    }
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        let normal = b.sub(a).perp().normalize_or_zero();
        if normal.length_squared() > SAT_EPS {
            axes.push(normal);
        }
    }
    axes
}

/// Tests a single axis, returning the signed interval overlap of the two
/// projections: positive when the intervals overlap (the interpenetration along
/// this axis), zero when they merely touch, and negative when a gap separates
/// them. Returns `None` when either polygon is empty (no projection).
#[must_use]
fn axis_overlap(a: &[Vec2], b: &[Vec2], axis: Vec2) -> Option<f32> {
    let (min_a, max_a) = project(a, axis)?;
    let (min_b, max_b) = project(b, axis)?;
    // overlap = min(max_a, max_b) - max(min_a, min_b)
    let upper = if max_a < max_b { max_a } else { max_b };
    let lower = if min_a > min_b { min_a } else { min_b };
    Some(upper - lower)
}

/// Returns `true` when the two `CCW` convex polygons overlap (interpenetrate or
/// touch), and `false` when a separating axis proves them disjoint.
///
/// A shared edge or a single touching vertex counts as an overlap (a
/// zero-depth contact): only a real gap wider than [`SAT_EPS`] on some axis
/// reports disjoint. Empty or single-vertex inputs have no separating face and
/// are reported as non-overlapping.
#[must_use]
pub fn overlaps(a: &[Vec2], b: &[Vec2]) -> bool {
    if a.len() < 2 || b.len() < 2 {
        return false;
    }
    let mut axes = edge_axes(a);
    axes.extend(edge_axes(b));
    if axes.is_empty() {
        return false;
    }
    for axis in axes {
        match axis_overlap(a, b, axis) {
            Some(overlap) => {
                if overlap < -SAT_EPS {
                    return false;
                }
            }
            None => return false,
        }
    }
    true
}

/// Computes the minimum translation vector separating two `CCW` convex
/// polygons, or `None` when a separating axis proves them disjoint.
///
/// The returned tuple is `(axis, depth)`: `axis` is a unit direction oriented
/// from `a` toward `b`, and `depth >= 0` is the penetration along it. Swapping
/// the arguments negates `axis` and preserves `depth`, so
/// `mtv(a, b)` and `mtv(b, a)` describe the same separation from opposite sides.
/// A touching contact yields `Some((axis, 0.0))`; a genuine gap yields `None`.
#[must_use]
pub fn mtv(a: &[Vec2], b: &[Vec2]) -> Option<(Vec2, f32)> {
    if a.len() < 2 || b.len() < 2 {
        return None;
    }
    let mut axes = edge_axes(a);
    axes.extend(edge_axes(b));
    if axes.is_empty() {
        return None;
    }

    let mut best_axis = Vec2::ZERO;
    let mut best_depth = f32::INFINITY;
    let mut found = false;

    for axis in axes {
        let overlap = axis_overlap(a, b, axis)?;
        if overlap < -SAT_EPS {
            return None;
        }
        // Clamp a tiny negative (touching) overlap to a non-negative depth.
        let depth = if overlap < 0.0 { 0.0 } else { overlap };
        if !found || depth < best_depth - SAT_EPS {
            best_depth = depth;
            best_axis = axis;
            found = true;
        }
    }

    if !found {
        return None;
    }

    // Orient the axis from a toward b so the result is symmetric under argument
    // swap and unambiguous as a push direction.
    let dir = centroid(b).sub(centroid(a));
    let oriented = if best_axis.dot(dir) < 0.0 {
        best_axis.neg()
    } else {
        best_axis
    };

    Some((oriented, best_depth))
}

/// Convenience wrapper returning the [`Mtv`] struct instead of a bare tuple, so
/// callers that serialize to `std430` do not re-pack the fields.
#[must_use]
pub fn mtv_struct(a: &[Vec2], b: &[Vec2]) -> Option<Mtv> {
    mtv(a, b).map(|(axis, depth)| Mtv::new(axis, depth))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for float comparisons that are not bit-exact.
    const EPS: f32 = 1.0e-4;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < EPS, "expected {b}, got {a}");
    }

    /// An axis-aligned square `[cx-h, cx+h] x [cy-h, cy+h]`, wound `CCW`.
    fn square(cx: f32, cy: f32, h: f32) -> Vec<Vec2> {
        alloc::vec![
            Vec2::new(cx - h, cy - h),
            Vec2::new(cx + h, cy - h),
            Vec2::new(cx + h, cy + h),
            Vec2::new(cx - h, cy + h),
        ]
    }

    fn unit_square() -> Vec<Vec2> {
        square(0.0, 0.0, 1.0)
    }

    // ----- Vec2 math -------------------------------------------------------

    #[test]
    fn vec2_add_sub_scale_are_exact() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(4.0, 6.0);
        assert_eq!(a.add(b), Vec2::new(5.0, 8.0));
        assert_eq!(b.sub(a), Vec2::new(3.0, 4.0));
        assert_eq!(a.scale(3.0), Vec2::new(3.0, 6.0));
        assert_eq!(a.neg(), Vec2::new(-1.0, -2.0));
    }

    #[test]
    fn vec2_dot_and_perp() {
        let a = Vec2::new(3.0, 4.0);
        let b = Vec2::new(2.0, 1.0);
        assert_eq!(a.dot(b), 10.0);
        assert_eq!(a.perp(), Vec2::new(-4.0, 3.0));
        // The perpendicular is orthogonal to the original.
        assert_eq!(a.dot(a.perp()), 0.0);
    }

    #[test]
    fn vec2_length_and_length_squared() {
        let a = Vec2::new(3.0, 4.0);
        assert_eq!(a.length_squared(), 25.0);
        assert_eq!(a.length(), 5.0);
    }

    #[test]
    fn normalize_zero_vector_is_zero_not_nan() {
        assert_eq!(Vec2::ZERO.normalize_or_zero(), Vec2::ZERO);
        let n = Vec2::new(0.0, 7.0).normalize_or_zero();
        assert_eq!(n, Vec2::new(0.0, 1.0));
    }

    #[test]
    fn normalized_axis_is_unit_length() {
        let n = Vec2::new(3.0, 4.0).normalize_or_zero();
        approx(n.length(), 1.0);
    }

    #[test]
    fn approx_eq_uses_epsilon() {
        let a = Vec2::new(1.0, 1.0);
        let b = Vec2::new(1.0 + SAT_EPS * 0.5, 1.0 - SAT_EPS * 0.5);
        assert!(a.approx_eq(b));
        assert!(!a.approx_eq(Vec2::new(1.1, 1.0)));
    }

    // ----- overlaps --------------------------------------------------------

    #[test]
    fn overlapping_squares_report_overlap() {
        let a = unit_square();
        let b = square(0.5, 0.0, 1.0);
        assert!(overlaps(&a, &b));
        assert!(overlaps(&b, &a));
    }

    #[test]
    fn separated_squares_report_no_overlap() {
        let a = unit_square();
        let b = square(5.0, 0.0, 1.0);
        assert!(!overlaps(&a, &b));
        assert!(!overlaps(&b, &a));
    }

    #[test]
    fn diagonally_separated_squares_report_no_overlap() {
        let a = unit_square();
        let b = square(3.0, 3.0, 1.0);
        assert!(!overlaps(&a, &b));
    }

    #[test]
    fn touching_edge_counts_as_overlap() {
        let a = unit_square();
        // b centered at x=2.0 has its left edge at x=1.0, meeting a's right edge.
        let b = square(2.0, 0.0, 1.0);
        assert!(overlaps(&a, &b));
    }

    #[test]
    fn contained_square_reports_overlap() {
        let outer = unit_square();
        let inner = square(0.0, 0.0, 0.25);
        assert!(overlaps(&outer, &inner));
        assert!(overlaps(&inner, &outer));
    }

    #[test]
    fn degenerate_inputs_do_not_overlap() {
        let a = unit_square();
        assert!(!overlaps(&a, &[]));
        assert!(!overlaps(&[], &a));
        assert!(!overlaps(&a, &[Vec2::new(0.0, 0.0)]));
    }

    // ----- mtv -------------------------------------------------------------

    #[test]
    fn mtv_none_when_separated() {
        let a = unit_square();
        let b = square(5.0, 0.0, 1.0);
        assert!(mtv(&a, &b).is_none());
    }

    #[test]
    fn mtv_pushes_overlapping_squares_apart() {
        let a = unit_square();
        let b = square(1.5, 0.0, 1.0);
        let (axis, depth) = mtv(&a, &b).expect("overlap");
        // Overlap on x is 0.5, none needed on y, so MTV is along +x with depth 0.5.
        approx(axis.length(), 1.0);
        approx(depth, 0.5);
        assert!(axis.approx_eq(Vec2::new(1.0, 0.0)));

        // Applying the push separates them: b moved by axis*depth no longer
        // overlaps a beyond touching.
        let moved: Vec<Vec2> = b.iter().map(|v| v.add(axis.scale(depth))).collect();
        // They should now be exactly touching (depth ~0), not interpenetrating.
        let (_, d2) = mtv(&a, &moved).expect("touching");
        approx(d2, 0.0);
    }

    #[test]
    fn mtv_axis_points_from_a_toward_b() {
        let a = unit_square();
        let b = square(1.5, 0.0, 1.0);
        let (axis, _) = mtv(&a, &b).expect("overlap");
        // b is to the +x of a, so the oriented axis has positive x.
        assert!(axis.x > 0.0);
    }

    #[test]
    fn mtv_prefers_shallow_vertical_overlap() {
        // Squares overlapping more in x than y: MTV should resolve along y.
        let a = unit_square();
        let b = square(0.25, 1.75, 1.0);
        let (axis, depth) = mtv(&a, &b).expect("overlap");
        // y overlap = 0.25, x overlap = 1.75 - clamped... x overlap is larger.
        approx(depth, 0.25);
        assert!(axis.approx_eq(Vec2::new(0.0, 1.0)));
    }

    #[test]
    fn mtv_touching_has_zero_depth() {
        let a = unit_square();
        let b = square(2.0, 0.0, 1.0); // left edge of b at x=1 == right edge of a
        let (_, depth) = mtv(&a, &b).expect("touching contact");
        approx(depth, 0.0);
    }

    #[test]
    fn mtv_symmetry_swaps_axis_direction() {
        let a = unit_square();
        let b = square(1.5, 0.0, 1.0);
        let (axis_ab, depth_ab) = mtv(&a, &b).expect("overlap");
        let (axis_ba, depth_ba) = mtv(&b, &a).expect("overlap");
        approx(depth_ab, depth_ba);
        assert!(axis_ab.approx_eq(axis_ba.neg()));
    }

    #[test]
    fn mtv_containment_reports_positive_depth() {
        let outer = square(0.0, 0.0, 2.0);
        let inner = square(0.3, 0.0, 0.5);
        let (axis, depth) = mtv(&outer, &inner).expect("overlap");
        approx(axis.length(), 1.0);
        // Depth must be positive and finite for a fully contained shape.
        assert!(depth > 0.0);
        assert!(depth.is_finite());
    }

    #[test]
    fn mtv_is_deterministic() {
        let a = unit_square();
        let b = square(1.5, 0.3, 1.0);
        let first = mtv(&a, &b).expect("overlap");
        for _ in 0..8 {
            let again = mtv(&a, &b).expect("overlap");
            assert_eq!(first.0, again.0);
            assert_eq!(first.1, again.1);
        }
    }

    #[test]
    fn triangle_vs_quad_overlap_and_mtv() {
        // CCW triangle straddling the origin.
        let tri = alloc::vec![
            Vec2::new(-1.0, -1.0),
            Vec2::new(1.0, -1.0),
            Vec2::new(0.0, 1.5),
        ];
        let quad = square(0.0, 0.0, 0.75);
        assert!(overlaps(&tri, &quad));
        let (axis, depth) = mtv(&tri, &quad).expect("overlap");
        approx(axis.length(), 1.0);
        assert!(depth > 0.0);
    }

    #[test]
    fn triangle_vs_quad_separated() {
        let tri = alloc::vec![
            Vec2::new(-1.0, -1.0),
            Vec2::new(1.0, -1.0),
            Vec2::new(0.0, 1.5),
        ];
        let quad = square(10.0, 10.0, 0.75);
        assert!(!overlaps(&tri, &quad));
        assert!(mtv(&tri, &quad).is_none());
    }

    #[test]
    fn rotated_square_overlap_uses_precomputed_coords() {
        // A unit square rotated 45 degrees (a diamond), coordinates given
        // directly without any trig call.
        let diamond = alloc::vec![
            Vec2::new(0.0, -1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(-1.0, 0.0),
        ];
        let box_a = square(0.6, 0.0, 0.5);
        assert!(overlaps(&diamond, &box_a));
        let (axis, depth) = mtv(&diamond, &box_a).expect("overlap");
        approx(axis.length(), 1.0);
        assert!(depth > 0.0);
    }

    #[test]
    fn rotated_shapes_separate_along_diagonal_axis() {
        // Two diamonds separated along a diagonal; the separating axis is one
        // of the diamond faces (a 45-degree normal), found without trig.
        let d1 = alloc::vec![
            Vec2::new(0.0, -1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(-1.0, 0.0),
        ];
        let d2: Vec<Vec2> = d1.iter().map(|v| v.add(Vec2::new(3.0, 3.0))).collect();
        assert!(!overlaps(&d1, &d2));
        assert!(mtv(&d1, &d2).is_none());
    }

    #[test]
    fn resolving_mtv_removes_interpenetration() {
        let a = unit_square();
        let b = square(0.75, 0.4, 1.0);
        let (axis, depth) = mtv(&a, &b).expect("overlap");
        // Move b fully out of a along the MTV.
        let moved: Vec<Vec2> = b.iter().map(|v| v.add(axis.scale(depth))).collect();
        // After resolution the remaining penetration is at most epsilon.
        if let Some((_, d)) = mtv(&a, &moved) {
            assert!(d <= 1.0e-3, "residual penetration {d}");
        }
    }

    #[test]
    fn identical_squares_fully_overlap() {
        let a = unit_square();
        let b = unit_square();
        assert!(overlaps(&a, &b));
        let (axis, depth) = mtv(&a, &b).expect("overlap");
        approx(axis.length(), 1.0);
        // Fully coincident: penetration equals the full extent (2.0).
        approx(depth, 2.0);
    }

    // ----- Mtv struct & std430 --------------------------------------------

    #[test]
    fn mtv_struct_matches_tuple() {
        let a = unit_square();
        let b = square(1.5, 0.0, 1.0);
        let tuple = mtv(&a, &b).expect("overlap");
        let s = mtv_struct(&a, &b).expect("overlap");
        assert_eq!(s.axis, tuple.0);
        assert_eq!(s.depth, tuple.1);
    }

    #[test]
    fn mtv_push_helpers_are_opposite() {
        let m = Mtv::new(Vec2::new(1.0, 0.0), 0.5);
        assert_eq!(m.push_b(), Vec2::new(0.5, 0.0));
        assert_eq!(m.push_a(), Vec2::new(-0.5, 0.0));
    }

    #[test]
    fn std430_size_is_multiple_of_sixteen() {
        assert_eq!(MTV_STD430_SIZE % 16, 0);
        assert_eq!(MTV_STD430_SIZE, 16);
    }

    #[test]
    fn to_std430_roundtrips_components() {
        let m = Mtv::new(Vec2::new(0.5, -0.25), 1.5);
        let bytes = m.to_std430();
        assert_eq!(bytes.len(), MTV_STD430_SIZE);
        let x = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let y = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let d = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let pad = f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        approx(x, 0.5);
        approx(y, -0.25);
        approx(d, 1.5);
        approx(pad, 0.0);
    }

    #[test]
    fn gpu_storage_bytes_clamps_and_scales() {
        assert_eq!(gpu_storage_bytes(0), MTV_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(1), MTV_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(4), MTV_STD430_SIZE * 4);
    }

    // ----- projection helpers ---------------------------------------------

    #[test]
    fn project_returns_none_for_empty() {
        assert!(project(&[], Vec2::new(1.0, 0.0)).is_none());
    }

    #[test]
    fn project_interval_of_unit_square_on_x() {
        let (lo, hi) = project(&unit_square(), Vec2::new(1.0, 0.0)).expect("nonempty");
        approx(lo, -1.0);
        approx(hi, 1.0);
    }

    #[test]
    fn centroid_of_symmetric_square_is_center() {
        let c = centroid(&square(2.0, 3.0, 1.0));
        assert!(c.approx_eq(Vec2::new(2.0, 3.0)));
    }

    #[test]
    fn edge_axes_of_square_are_the_four_face_normals() {
        let axes = edge_axes(&unit_square());
        assert_eq!(axes.len(), 4);
        for ax in &axes {
            approx(ax.length(), 1.0);
        }
    }

    #[test]
    fn axis_overlap_detects_gap_and_penetration() {
        let a = unit_square();
        let far = square(5.0, 0.0, 1.0);
        let near = square(1.5, 0.0, 1.0);
        let x = Vec2::new(1.0, 0.0);
        assert!(axis_overlap(&a, &far, x).expect("some") < -SAT_EPS);
        approx(axis_overlap(&a, &near, x).expect("some"), 0.5);
    }
}
