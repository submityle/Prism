//! Triangle circumcircle construction and Delaunay in-circle testing for the
//! particle triangulation contracts (design §8.2, §12-§13).
//!
//! Several particle stages need the *circumscribed circle* of a triangle and
//! the Delaunay predicate that tests it: a 2D Delaunay/Voronoi meshing pass
//! (used to seed anisotropic splat footprints or to remesh a soft-body patch)
//! flips an edge exactly when the opposite vertex falls *inside* the current
//! triangle's circumcircle; a bounds pass wants the tight enclosing circle of a
//! three-point cluster; and a collision-broadphase proxy wants the circle that
//! passes through three contact points. This module owns the small,
//! `CPU`-verifiable contract those stages share: turning three points into
//! their circumcenter and circumradius, classifying the winding of a triangle,
//! and evaluating the robust `in-circle` determinant that drives edge flips.
//!
//! # Strict scope
//! This module only *constructs* the circumcircle of a triangle and evaluates
//! the in-circle / orientation predicates on it. It deliberately does not solve
//! barycentric coordinates ([`super::barycentric_coord`]) or build convex hulls
//! ([`super::convex_hull_2d`]); it neither imports nor reconstructs those
//! contracts, and it keeps its own [`Vec2`] and cross-product helpers.
//!
//! # No transcendental math
//! The circumcenter is the solution of two perpendicular-bisector equations,
//! solved by Cramer's rule as a pure `+`, `-`, `*`, `/` determinant ratio; the
//! in-circle test is a `3x3` determinant of the same kind; the winding is the
//! sign of a signed area. The only irrational operation is the single
//! [`f32::sqrt`] used to turn the squared circumradius into a radius. There is
//! no `sin`, `cos`, `atan`, `exp`, `ln`, `powf`, `ceil`, `round`, or any other
//! transcendental call, and no `f32` equality: near-zero magnitudes are always
//! compared against [`CMP_EPS`].

use crate::particle::gpu_layout::VEC4_STRIDE;

/// Magnitude below which a determinant, a signed area, a coordinate difference,
/// or an in-circle power value is treated as zero.
///
/// This is the comparison rule used throughout instead of `==` on `f32`: two
/// scalars are "equal" when their absolute difference does not exceed this
/// bound. A triangle whose doubled signed area is within this bound is treated
/// as degenerate (its three vertices are collinear or coincident).
pub const CMP_EPS: f32 = 1.0e-6;

/// Byte stride of one packed [`Circle`] record in a `std430` storage buffer.
///
/// A circle packs naturally into a single `vec4<f32>` slot
/// (`center.x, center.y, radius, pad`), so its stride is exactly one
/// [`VEC4_STRIDE`] and therefore a multiple of 16, as `std430` requires for a
/// `vec4`-aligned element.
pub const CIRCLE_STRIDE: usize = VEC4_STRIDE;

/// A point / vector in the 2D plane, owned by this module so it depends on no
/// sibling's vector type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec2 {
    /// The `x` (horizontal) component.
    pub x: f32,
    /// The `y` (vertical) component.
    pub y: f32,
}

impl Vec2 {
    /// The zero vector `(0, 0)`.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    /// Builds a vector from its two components.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Component-wise sum `self + rhs`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API is specified with named add/sub/neg methods for call-site uniformity, matching the sibling particle contracts; operator traits are intentionally not part of this internal type."
    )]
    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }

    /// Component-wise difference `self - rhs`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API is specified with named add/sub/neg methods for call-site uniformity, matching the sibling particle contracts; operator traits are intentionally not part of this internal type."
    )]
    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }

    /// Uniform scale of both components by `s`.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s)
    }

    /// Dot product `self · rhs`.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y
    }

    /// Scalar 2D cross product `self × rhs` (the `z` component of the 3D
    /// cross), i.e. twice the signed area of the parallelogram they span.
    #[must_use]
    pub fn cross(self, rhs: Self) -> f32 {
        self.x * rhs.y - self.y * rhs.x
    }

    /// Squared Euclidean length `|self|²` (no `sqrt`).
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length `|self|` (the module's only `sqrt`).
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Squared Euclidean distance between `self` and `other` (no `sqrt`).
    #[must_use]
    pub fn distance_squared(self, other: Self) -> f32 {
        self.sub(other).length_squared()
    }

    /// Euclidean distance between `self` and `other`.
    #[must_use]
    pub fn distance(self, other: Self) -> f32 {
        self.sub(other).length()
    }

    /// The midpoint of the segment `self`–`other`.
    #[must_use]
    pub fn midpoint(self, other: Self) -> Self {
        self.add(other).scale(0.5)
    }

    /// Returns `true` when the two points coincide within [`CMP_EPS`] on both
    /// axes.
    #[must_use]
    pub fn approx_eq(self, other: Self) -> bool {
        (self.x - other.x).abs() <= CMP_EPS && (self.y - other.y).abs() <= CMP_EPS
    }
}

/// A circle in the 2D plane, given by its `center` and non-negative `radius`.
///
/// This is the return type of [`Triangle::circumcircle`]: the unique circle
/// passing through all three vertices of a non-degenerate triangle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Circle {
    /// The center point of the circle.
    pub center: Vec2,
    /// The (non-negative) radius of the circle.
    pub radius: f32,
}

impl Circle {
    /// Builds a circle from its center and radius.
    #[must_use]
    pub const fn new(center: Vec2, radius: f32) -> Self {
        Self { center, radius }
    }

    /// Signed "power" of a point with respect to this circle:
    /// `|p - center|² - radius²`.
    ///
    /// It is negative strictly inside the circle, zero on it, and positive
    /// outside; classification against [`CMP_EPS`] is left to the caller.
    #[must_use]
    pub fn power(&self, p: Vec2) -> f32 {
        self.center.distance_squared(p) - self.radius * self.radius
    }

    /// Classifies where `p` lies relative to this circle using [`CMP_EPS`] as
    /// the on-boundary tolerance.
    #[must_use]
    pub fn classify(&self, p: Vec2) -> InCircle {
        let power = self.power(p);
        if power < -CMP_EPS {
            InCircle::Inside
        } else if power > CMP_EPS {
            InCircle::Outside
        } else {
            InCircle::OnCircle
        }
    }

    /// Packs the circle into its `std430` `vec4`-aligned word layout.
    ///
    /// Layout: `[center.x, center.y, radius, pad]` as raw `u32` words (each
    /// `f32` field via [`f32::to_bits`]), filling one `vec4` slot that matches
    /// [`CIRCLE_STRIDE`]. The trailing word is padding.
    #[must_use]
    pub fn to_std430(&self) -> [u32; 4] {
        [
            self.center.x.to_bits(),
            self.center.y.to_bits(),
            self.radius.to_bits(),
            0,
        ]
    }
}

/// The winding of an ordered triple of points, i.e. the sign of their doubled
/// signed area.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Orientation {
    /// The triple turns counter-clockwise (strictly positive signed area).
    CounterClockwise,
    /// The triple turns clockwise (strictly negative signed area).
    Clockwise,
    /// The three points are collinear (signed area within [`CMP_EPS`]).
    Collinear,
}

/// Where a query point lies relative to a triangle's circumcircle, as returned
/// by [`Triangle::in_circle`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InCircle {
    /// The point lies strictly inside the circumcircle (Delaunay-violating for
    /// the opposite triangle).
    Inside,
    /// The point lies strictly outside the circumcircle.
    Outside,
    /// The point lies on the circumcircle within [`CMP_EPS`] (a cocircular,
    /// ambiguous configuration).
    OnCircle,
}

/// Twice the signed area of triangle `a, b, c`, i.e. the 2D cross product
/// `(b - a) × (c - a)`.
///
/// The sign classifies the winding: strictly positive for counter-clockwise,
/// strictly negative for clockwise, and zero (within [`CMP_EPS`]) when the
/// three points are collinear.
#[must_use]
pub fn signed_area_doubled(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    b.sub(a).cross(c.sub(a))
}

/// The signed area of triangle `a, b, c` (half of [`signed_area_doubled`]).
#[must_use]
pub fn signed_area(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    signed_area_doubled(a, b, c) * 0.5
}

/// Classifies the winding of the ordered triple `a, b, c`.
///
/// Returns [`Orientation::Collinear`] when the doubled signed area is within
/// [`CMP_EPS`] of zero, [`Orientation::CounterClockwise`] when it is strictly
/// positive, and [`Orientation::Clockwise`] when it is strictly negative.
#[must_use]
pub fn orientation(a: Vec2, b: Vec2, c: Vec2) -> Orientation {
    let area2 = signed_area_doubled(a, b, c);
    if area2 > CMP_EPS {
        Orientation::CounterClockwise
    } else if area2 < -CMP_EPS {
        Orientation::Clockwise
    } else {
        Orientation::Collinear
    }
}

/// A triangle in the 2D plane, given by its three corners in `a`, `b`, `c`
/// order. The corner order fixes the winding reported by
/// [`Triangle::orientation`] but does not change the circumcircle, which is
/// intrinsic to the point set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Triangle {
    /// First corner.
    pub a: Vec2,
    /// Second corner.
    pub b: Vec2,
    /// Third corner.
    pub c: Vec2,
}

impl Triangle {
    /// Builds a triangle from its three corners.
    #[must_use]
    pub const fn new(a: Vec2, b: Vec2, c: Vec2) -> Self {
        Self { a, b, c }
    }

    /// Twice the signed area of this triangle (see [`signed_area_doubled`]).
    #[must_use]
    pub fn signed_area_doubled(&self) -> f32 {
        signed_area_doubled(self.a, self.b, self.c)
    }

    /// The signed area of this triangle (see [`signed_area`]).
    #[must_use]
    pub fn signed_area(&self) -> f32 {
        signed_area(self.a, self.b, self.c)
    }

    /// The winding of this triangle (see [`orientation`]).
    #[must_use]
    pub fn orientation(&self) -> Orientation {
        orientation(self.a, self.b, self.c)
    }

    /// Returns `true` when this triangle is degenerate: its three corners are
    /// collinear or coincident (doubled signed area within [`CMP_EPS`]).
    #[must_use]
    pub fn is_degenerate(&self) -> bool {
        self.signed_area_doubled().abs() <= CMP_EPS
    }

    /// The centroid (arithmetic mean of the three corners) of this triangle.
    #[must_use]
    pub fn centroid(&self) -> Vec2 {
        self.a.add(self.b).add(self.c).scale(1.0 / 3.0)
    }

    /// The circumcenter: the point equidistant from all three corners.
    ///
    /// It is the intersection of the perpendicular bisectors of two edges,
    /// solved by Cramer's rule. Returns `None` when the triangle is degenerate
    /// (collinear/coincident corners), which would otherwise divide by a
    /// near-zero determinant.
    #[must_use]
    pub fn circumcenter(&self) -> Option<Vec2> {
        // d = 2 * doubled-signed-area = 4 * area; when it vanishes the three
        // points are collinear and no finite circumcenter exists.
        let d = 2.0 * self.signed_area_doubled();
        if d.abs() <= CMP_EPS {
            return None;
        }
        let a = self.a;
        let b = self.b;
        let c = self.c;
        let asq = a.length_squared();
        let bsq = b.length_squared();
        let csq = c.length_squared();
        let inv = 1.0 / d;
        let ux = (asq * (b.y - c.y) + bsq * (c.y - a.y) + csq * (a.y - b.y)) * inv;
        let uy = (asq * (c.x - b.x) + bsq * (a.x - c.x) + csq * (b.x - a.x)) * inv;
        Some(Vec2::new(ux, uy))
    }

    /// The squared circumradius, i.e. the squared distance from the
    /// circumcenter to any corner (no `sqrt`).
    ///
    /// Returns `None` for a degenerate triangle.
    #[must_use]
    pub fn circumradius_squared(&self) -> Option<f32> {
        let center = self.circumcenter()?;
        Some(center.distance_squared(self.a))
    }

    /// The circumradius: the radius of the circle through all three corners.
    ///
    /// Returns `None` for a degenerate triangle. This is the module's only use
    /// of [`f32::sqrt`].
    #[must_use]
    pub fn circumradius(&self) -> Option<f32> {
        Some(self.circumradius_squared()?.sqrt())
    }

    /// The circumscribed circle: the unique circle passing through all three
    /// corners.
    ///
    /// Returns `None` when the triangle is degenerate (collinear/coincident
    /// corners), in which case no finite circumcircle exists.
    #[must_use]
    pub fn circumcircle(&self) -> Option<Circle> {
        let center = self.circumcenter()?;
        let radius = center.distance(self.a);
        Some(Circle::new(center, radius))
    }

    /// Delaunay in-circle test: classifies `d` relative to this triangle's
    /// circumcircle.
    ///
    /// Evaluates the `3x3` in-circle determinant on the corner offsets
    /// `a - d`, `b - d`, `c - d` (with each row's third column its squared
    /// length). For a counter-clockwise triangle a strictly positive
    /// determinant means `d` lies *inside* the circumcircle; the sign is
    /// flipped for a clockwise triangle so the result is winding-agnostic.
    /// Returns `None` when the triangle is degenerate.
    #[must_use]
    pub fn in_circle(&self, d: Vec2) -> Option<InCircle> {
        let winding = self.orientation();
        let sign = match winding {
            Orientation::CounterClockwise => 1.0,
            Orientation::Clockwise => -1.0,
            Orientation::Collinear => return None,
        };
        let av = self.a.sub(d);
        let bv = self.b.sub(d);
        let cv = self.c.sub(d);
        let asq = av.length_squared();
        let bsq = bv.length_squared();
        let csq = cv.length_squared();
        // 3x3 determinant with columns (x, y, x²+y²).
        let det = av.x * (bv.y * csq - bsq * cv.y) - av.y * (bv.x * csq - bsq * cv.x)
            + asq * (bv.x * cv.y - bv.y * cv.x);
        let oriented = det * sign;
        if oriented > CMP_EPS {
            Some(InCircle::Inside)
        } else if oriented < -CMP_EPS {
            Some(InCircle::Outside)
        } else {
            Some(InCircle::OnCircle)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SQRT3_OVER_2: f32 = 0.866_025_4;

    fn close(lhs: f32, rhs: f32) -> bool {
        (lhs - rhs).abs() <= 1.0e-4
    }

    fn close_vec(lhs: Vec2, rhs: Vec2) -> bool {
        close(lhs.x, rhs.x) && close(lhs.y, rhs.y)
    }

    #[test]
    fn vec2_basic_algebra_is_exact() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(4.0, 6.0);
        assert_eq!(a.add(b), Vec2::new(5.0, 8.0));
        assert_eq!(b.sub(a), Vec2::new(3.0, 4.0));
        assert_eq!(a.scale(2.0), Vec2::new(2.0, 4.0));
        assert_eq!(a.dot(b), 4.0 + 12.0);
    }

    #[test]
    fn vec2_cross_matches_signed_parallelogram() {
        let x = Vec2::new(1.0, 0.0);
        let y = Vec2::new(0.0, 1.0);
        assert_eq!(x.cross(y), 1.0);
        assert_eq!(y.cross(x), -1.0);
    }

    #[test]
    fn vec2_length_and_distance() {
        let a = Vec2::new(3.0, 4.0);
        assert_eq!(a.length_squared(), 25.0);
        assert_eq!(a.length(), 5.0);
        assert_eq!(Vec2::ZERO.distance(a), 5.0);
        assert_eq!(Vec2::ZERO.distance_squared(a), 25.0);
    }

    #[test]
    fn vec2_midpoint_is_average() {
        let a = Vec2::new(-2.0, 4.0);
        let b = Vec2::new(6.0, -2.0);
        assert_eq!(a.midpoint(b), Vec2::new(2.0, 1.0));
    }

    #[test]
    fn vec2_approx_eq_uses_epsilon() {
        let a = Vec2::new(1.0, 1.0);
        let b = Vec2::new(1.0 + CMP_EPS * 0.5, 1.0 - CMP_EPS * 0.5);
        assert!(a.approx_eq(b));
        assert!(!a.approx_eq(Vec2::new(1.1, 1.0)));
    }

    #[test]
    fn signed_area_doubled_is_twice_signed_area() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(4.0, 0.0);
        let c = Vec2::new(0.0, 3.0);
        assert_eq!(signed_area_doubled(a, b, c), 12.0);
        assert_eq!(signed_area(a, b, c), 6.0);
    }

    #[test]
    fn orientation_detects_counter_clockwise() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(1.0, 0.0);
        let c = Vec2::new(0.0, 1.0);
        assert_eq!(orientation(a, b, c), Orientation::CounterClockwise);
    }

    #[test]
    fn orientation_detects_clockwise() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(0.0, 1.0);
        let c = Vec2::new(1.0, 0.0);
        assert_eq!(orientation(a, b, c), Orientation::Clockwise);
    }

    #[test]
    fn orientation_detects_collinear() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(1.0, 1.0);
        let c = Vec2::new(2.0, 2.0);
        assert_eq!(orientation(a, b, c), Orientation::Collinear);
    }

    #[test]
    fn orientation_reversing_winding_flips_sign() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(2.0, 0.0);
        let c = Vec2::new(1.0, 2.0);
        assert_eq!(orientation(a, b, c), Orientation::CounterClockwise);
        assert_eq!(orientation(a, c, b), Orientation::Clockwise);
    }

    #[test]
    fn equilateral_circumcenter_is_centroid() {
        let t = Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.5, SQRT3_OVER_2),
        );
        let center = t.circumcenter().unwrap();
        assert!(close_vec(center, t.centroid()));
        assert!(close_vec(center, Vec2::new(0.5, SQRT3_OVER_2 / 3.0)));
    }

    #[test]
    fn equilateral_circumradius_matches_closed_form() {
        // For a unit-side equilateral triangle, R = 1 / sqrt(3) ≈ 0.57735.
        let t = Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.5, SQRT3_OVER_2),
        );
        assert!(close(t.circumradius().unwrap(), 0.577_350_3));
    }

    #[test]
    fn right_triangle_circumcenter_is_hypotenuse_midpoint() {
        // Right angle at the origin; hypotenuse joins (4,0) and (0,3).
        let t = Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(0.0, 3.0),
        );
        let center = t.circumcenter().unwrap();
        let hypotenuse_mid = Vec2::new(4.0, 0.0).midpoint(Vec2::new(0.0, 3.0));
        assert!(close_vec(center, hypotenuse_mid));
        assert!(close_vec(center, Vec2::new(2.0, 1.5)));
    }

    #[test]
    fn right_triangle_circumradius_is_half_hypotenuse() {
        let t = Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(0.0, 3.0),
        );
        // Hypotenuse length 5 ⇒ circumradius 2.5.
        assert!(close(t.circumradius().unwrap(), 2.5));
    }

    #[test]
    fn circumcircle_passes_through_all_three_corners() {
        let t = Triangle::new(
            Vec2::new(-1.0, 2.0),
            Vec2::new(3.0, 1.0),
            Vec2::new(0.5, -2.5),
        );
        let circle = t.circumcircle().unwrap();
        assert!(close(circle.center.distance(t.a), circle.radius));
        assert!(close(circle.center.distance(t.b), circle.radius));
        assert!(close(circle.center.distance(t.c), circle.radius));
    }

    #[test]
    fn circumradius_squared_matches_circumradius() {
        let t = Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(0.0, 3.0),
        );
        let r2 = t.circumradius_squared().unwrap();
        let r = t.circumradius().unwrap();
        assert!(close(r2, r * r));
        assert!(close(r2, 6.25));
    }

    #[test]
    fn circumcircle_is_winding_agnostic() {
        let ccw = Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(0.0, 3.0),
        );
        let cw = Triangle::new(ccw.a, ccw.c, ccw.b);
        let c1 = ccw.circumcircle().unwrap();
        let c2 = cw.circumcircle().unwrap();
        assert!(close_vec(c1.center, c2.center));
        assert!(close(c1.radius, c2.radius));
    }

    #[test]
    fn collinear_triangle_has_no_circumcenter() {
        let t = Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(2.0, 2.0),
        );
        assert!(t.circumcenter().is_none());
        assert!(t.circumcircle().is_none());
        assert!(t.circumradius().is_none());
        assert!(t.circumradius_squared().is_none());
    }

    #[test]
    fn coincident_corners_have_no_circumcircle() {
        let t = Triangle::new(
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(3.0, 5.0),
        );
        assert!(t.is_degenerate());
        assert!(t.circumcircle().is_none());
    }

    #[test]
    fn is_degenerate_flags_collinear_only() {
        let good = Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
        );
        let bad = Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(5.0, 0.0),
        );
        assert!(!good.is_degenerate());
        assert!(bad.is_degenerate());
    }

    #[test]
    fn in_circle_inside_point() {
        let t = Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
        );
        // Circumcenter (0.5, 0.5), radius ≈ 0.707; the centroid is well inside.
        assert_eq!(t.in_circle(Vec2::new(0.3, 0.3)), Some(InCircle::Inside));
    }

    #[test]
    fn in_circle_outside_point() {
        let t = Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
        );
        assert_eq!(t.in_circle(Vec2::new(2.0, 2.0)), Some(InCircle::Outside));
    }

    #[test]
    fn in_circle_boundary_point_is_on_circle() {
        let t = Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
        );
        // The fourth corner of the unit square lies exactly on the circle.
        assert_eq!(t.in_circle(Vec2::new(1.0, 1.0)), Some(InCircle::OnCircle));
    }

    #[test]
    fn in_circle_corner_is_on_circle() {
        let t = Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(0.0, 3.0),
        );
        // Each defining corner sits on the circumcircle by construction.
        assert_eq!(t.in_circle(t.a), Some(InCircle::OnCircle));
        assert_eq!(t.in_circle(t.b), Some(InCircle::OnCircle));
        assert_eq!(t.in_circle(t.c), Some(InCircle::OnCircle));
    }

    #[test]
    fn in_circle_is_winding_agnostic() {
        let ccw = Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
        );
        let cw = Triangle::new(ccw.a, ccw.c, ccw.b);
        let p = Vec2::new(0.3, 0.3);
        assert_eq!(ccw.in_circle(p), Some(InCircle::Inside));
        assert_eq!(cw.in_circle(p), Some(InCircle::Inside));
        let q = Vec2::new(5.0, 5.0);
        assert_eq!(ccw.in_circle(q), Some(InCircle::Outside));
        assert_eq!(cw.in_circle(q), Some(InCircle::Outside));
    }

    #[test]
    fn in_circle_degenerate_returns_none() {
        let t = Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(2.0, 2.0),
        );
        assert!(t.in_circle(Vec2::new(0.5, 0.0)).is_none());
    }

    #[test]
    fn in_circle_agrees_with_circle_classify() {
        let t = Triangle::new(
            Vec2::new(-2.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(0.0, 2.0),
        );
        let circle = t.circumcircle().unwrap();
        for p in [
            Vec2::new(0.0, 0.0),
            Vec2::new(0.0, 1.9),
            Vec2::new(10.0, 10.0),
            Vec2::new(-5.0, 0.0),
        ] {
            assert_eq!(t.in_circle(p), Some(circle.classify(p)));
        }
    }

    #[test]
    fn circle_power_sign_matches_inside_outside() {
        let circle = Circle::new(Vec2::new(0.0, 0.0), 2.0);
        assert!(circle.power(Vec2::new(0.0, 0.0)) < 0.0);
        assert!(close(circle.power(Vec2::new(2.0, 0.0)), 0.0));
        assert!(circle.power(Vec2::new(5.0, 0.0)) > 0.0);
    }

    #[test]
    fn circle_classify_uses_epsilon_band() {
        let circle = Circle::new(Vec2::new(1.0, 1.0), 3.0);
        assert_eq!(circle.classify(Vec2::new(1.0, 1.0)), InCircle::Inside);
        assert_eq!(circle.classify(Vec2::new(4.0, 1.0)), InCircle::OnCircle);
        assert_eq!(circle.classify(Vec2::new(10.0, 1.0)), InCircle::Outside);
    }

    #[test]
    fn in_circle_is_deterministic() {
        let t = Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(3.0, 0.0),
            Vec2::new(1.0, 2.0),
        );
        let p = Vec2::new(1.0, 0.5);
        let first = t.in_circle(p);
        for _ in 0..64 {
            assert_eq!(t.in_circle(p), first);
        }
    }

    #[test]
    fn circumcircle_is_deterministic() {
        let t = Triangle::new(
            Vec2::new(-3.0, 1.0),
            Vec2::new(2.0, 4.0),
            Vec2::new(5.0, -1.0),
        );
        let first = t.circumcircle().unwrap();
        for _ in 0..64 {
            let again = t.circumcircle().unwrap();
            assert_eq!(again.center, first.center);
            assert_eq!(again.radius, first.radius);
        }
    }

    #[test]
    fn circumcircle_is_translation_invariant_in_radius() {
        let base = Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(1.0, 3.0),
        );
        let shift = Vec2::new(100.0, -50.0);
        let moved = Triangle::new(base.a.add(shift), base.b.add(shift), base.c.add(shift));
        let c1 = base.circumcircle().unwrap();
        let c2 = moved.circumcircle().unwrap();
        assert!(close(c1.radius, c2.radius));
        assert!(close_vec(c1.center.add(shift), c2.center));
    }

    #[test]
    fn circumradius_scales_with_uniform_scale() {
        let base = Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(0.0, 3.0),
        );
        let scaled = Triangle::new(base.a.scale(2.0), base.b.scale(2.0), base.c.scale(2.0));
        assert!(close(
            scaled.circumradius().unwrap(),
            2.0 * base.circumradius().unwrap()
        ));
    }

    #[test]
    fn centroid_is_average_of_corners() {
        let t = Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(3.0, 0.0),
            Vec2::new(0.0, 6.0),
        );
        assert!(close_vec(t.centroid(), Vec2::new(1.0, 2.0)));
    }

    #[test]
    fn triangle_signed_area_helpers_agree_with_free_functions() {
        let t = Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(0.0, 3.0),
        );
        assert_eq!(t.signed_area_doubled(), 12.0);
        assert_eq!(t.signed_area(), 6.0);
        assert_eq!(t.orientation(), Orientation::CounterClockwise);
    }

    #[test]
    fn circle_stride_is_one_vec4_multiple_of_16() {
        assert_eq!(CIRCLE_STRIDE, VEC4_STRIDE);
        assert_eq!(CIRCLE_STRIDE, 16);
        assert_eq!(CIRCLE_STRIDE % 16, 0);
    }

    #[test]
    fn circle_to_std430_encodes_center_radius_and_padding() {
        let circle = Circle::new(Vec2::new(2.0, -3.0), 5.0);
        let words = circle.to_std430();
        assert_eq!(words[0], 2.0_f32.to_bits());
        assert_eq!(words[1], (-3.0_f32).to_bits());
        assert_eq!(words[2], 5.0_f32.to_bits());
        assert_eq!(words[3], 0);
        assert_eq!(words.len() * 4, CIRCLE_STRIDE);
    }

    #[test]
    fn obtuse_triangle_circumcenter_lies_outside() {
        // A flat, obtuse triangle: its circumcenter falls below the base.
        let t = Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(2.0, 0.25),
        );
        let center = t.circumcenter().unwrap();
        // Equidistant check still holds even though the center is far away.
        let circle = t.circumcircle().unwrap();
        assert!(close(circle.center.distance(t.a), circle.radius));
        assert!(close(circle.center.distance(t.c), circle.radius));
        // The apex is only slightly above the base, so the center sits well
        // below it (negative y).
        assert!(center.y < 0.0);
    }
}
