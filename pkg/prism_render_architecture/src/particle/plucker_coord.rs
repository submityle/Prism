//! Plücker line coordinates for the particle ray/triangle intersection contract
//! (design §8.2 mesh emission, §11 collision, §13 culling).
//!
//! A directed line in 3D admits a coordinate-free six-tuple representation
//! `(u, v)` where `u` is the line's direction and `v = a × b` is its *moment*
//! about the origin for any two points `a`, `b` on the line. These Plücker
//! coordinates make the *relative orientation* of two lines a single symmetric
//! bilinear form — the permuted inner product `side(l1, l2) = u1·v2 + u2·v1` —
//! whose sign tells whether one directed line passes clockwise, counter-
//! clockwise, or coplanarly with respect to another, without ever solving a
//! linear system or intersecting anything. That single sign test is exactly
//! what a Plücker ray/triangle intersection needs: build the three directed
//! edge lines of a triangle (consistently wound), take the `side` of the ray
//! against each, and if all three signs agree the ray pierces the triangle's
//! interior. The particle subsystem uses this for mesh-emission surface hits,
//! trail/decal footprint tests, and reference ray-cast culling that must agree
//! bit-for-bit with the GPU path.
//!
//! # Strict scope
//! This module owns only its local [`Vec3`] and [`Line6`] types and the pure
//! Plücker algebra over them. It does not build meshes, own a camera, or touch
//! any sibling contract type; the sole shared dependency is the `std430` stride
//! arithmetic in [`crate::particle::gpu_layout`], reused so the GPU line buffer
//! layout is defined exactly once.
//!
//! # No transcendental math
//! Every routine is `+`, `-`, `*`, and comparisons against [`PLUCKER_EPS`].
//! The only non-trivial float op is a single `sqrt` inside [`Vec3::length`],
//! which the intersection logic never calls. There is no `sin`, `atan`, `powf`,
//! `ceil`, `round`, or any transcendental. Floats are never compared with `==`;
//! a magnitude is "zero" only when it is within [`PLUCKER_EPS`].

use crate::particle::gpu_layout::VEC4_STRIDE;

/// Magnitude below which a coordinate, a moment product, or a `side` value is
/// treated as zero. This is the comparison rule used everywhere in place of
/// `==` / `!=` on `f32`.
pub const PLUCKER_EPS: f32 = 1.0e-6;

/// `std430` byte size of one [`Line6`]: two `vec4<f32>` slots (direction +
/// padding, then moment + padding), i.e. `2 * 16 = 32` bytes, a multiple of the
/// 16-byte `vec4` alignment a `WebGPU` storage buffer requires.
pub const LINE6_STD430_SIZE: usize = 2 * VEC4_STRIDE;

/// A hand-rolled 3D vector, the only vector type this module defines.
///
/// All algebra is exact `+`/`-`/`*`; no method reaches for a transcendental.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    /// The `x` component.
    pub x: f32,
    /// The `y` component.
    pub y: f32,
    /// The `z` component.
    pub z: f32,
}

impl Vec3 {
    /// The zero vector.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Constructs a vector from its components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Component-wise sum `self + rhs`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API is specified with named add/sub/neg methods for call-site uniformity, matching the sibling particle contracts; operator traits are intentionally not part of this internal type."
    )]
    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    /// Component-wise difference `self - rhs`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API is specified with named add/sub/neg methods for call-site uniformity, matching the sibling particle contracts; operator traits are intentionally not part of this internal type."
    )]
    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }

    /// Uniform scale `self * s`.
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
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    /// Squared length `self · self` (no `sqrt`).
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length. The only `sqrt` in the module; never on a hot path.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Whether every component of `self` is within `eps` of `other`.
    #[must_use]
    pub fn approx_eq(self, other: Self, eps: f32) -> bool {
        (self.x - other.x).abs() < eps
            && (self.y - other.y).abs() < eps
            && (self.z - other.z).abs() < eps
    }
}

/// The relative orientation of one directed line about another, read from the
/// sign of their [`side`] product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    /// `side > 0`: the second line passes to the right (clockwise) of the first.
    Clockwise,
    /// `side < 0`: the second line passes to the left (counter-clockwise).
    CounterClockwise,
    /// `|side| <= eps`: the two lines are coplanar (they meet or are parallel).
    Coplanar,
}

/// How a directed line (a "ray", though the test is direction-agnostic) meets a
/// triangle, decided by the three edge [`side`] signs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TriangleHit {
    /// All three edge sides share the same strict sign: the line pierces the
    /// triangle's interior.
    Inside,
    /// At least one edge side is within `eps` of zero while the rest agree in
    /// sign: the line grazes an edge or vertex.
    Boundary,
    /// The edge sides disagree in sign: the line passes outside the triangle.
    Outside,
}

/// A directed line in Plücker coordinates: direction `u` and moment `v = a × b`.
///
/// For any two distinct points `a`, `b` on the line, `u = b - a` and
/// `v = a × b`; the moment is independent of which pair of the line's points is
/// chosen. The Plücker constraint `u · v = 0` holds identically because
/// `a × b` is orthogonal to both `a` and `b`, hence to `b - a`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Line6 {
    /// Direction `b - a` (not normalized; its magnitude scales `side`).
    pub u: Vec3,
    /// Moment `a × b` about the origin.
    pub v: Vec3,
}

impl Line6 {
    /// Builds the directed line through `a` then `b`.
    ///
    /// `u = b - a` is the direction and `v = a × b` is the moment. The returned
    /// line always satisfies the Plücker relation `u · v = 0`.
    #[must_use]
    pub fn from_points(a: Vec3, b: Vec3) -> Self {
        Self {
            u: b.sub(a),
            v: a.cross(b),
        }
    }

    /// Constructs a line directly from raw direction and moment components.
    #[must_use]
    pub const fn from_raw(u: Vec3, v: Vec3) -> Self {
        Self { u, v }
    }

    /// The line's direction `u`.
    #[must_use]
    pub fn direction(&self) -> Vec3 {
        self.u
    }

    /// The line's moment `v`.
    #[must_use]
    pub fn moment(&self) -> Vec3 {
        self.v
    }

    /// The Plücker constraint residual `u · v`, which is `0` for any line built
    /// from two points (up to floating-point rounding).
    #[must_use]
    pub fn moment_orthogonality(&self) -> f32 {
        self.u.dot(self.v)
    }

    /// Whether this line is non-degenerate, i.e. its direction is longer than
    /// [`PLUCKER_EPS`] (two coincident construction points give a zero
    /// direction and no well-defined line).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.u.length_squared() > PLUCKER_EPS * PLUCKER_EPS
    }

    /// Packs the line as a `std430` block of two `vec4<f32>` slots:
    /// `[u.x, u.y, u.z, 0]` then `[v.x, v.y, v.z, 0]`, little-endian, 32 bytes.
    ///
    /// The trailing lane in each `vec4` is explicit padding so the layout obeys
    /// the 16-byte `vec4` alignment a `WebGPU` storage buffer requires.
    #[must_use]
    pub fn to_std430(&self) -> [u8; LINE6_STD430_SIZE] {
        let comps = [
            self.u.x, self.u.y, self.u.z, 0.0, self.v.x, self.v.y, self.v.z, 0.0,
        ];
        let mut out = [0u8; LINE6_STD430_SIZE];
        let mut off = 0usize;
        for c in comps {
            let bytes = c.to_le_bytes();
            out[off] = bytes[0];
            out[off + 1] = bytes[1];
            out[off + 2] = bytes[2];
            out[off + 3] = bytes[3];
            off += 4;
        }
        out
    }
}

/// The permuted (reciprocal) inner product `side(l1, l2) = u1·v2 + u2·v1`.
///
/// This symmetric bilinear form is the heart of Plücker geometry: its sign is
/// positive when `l2` winds clockwise about `l1`, negative when counter-
/// clockwise, and (within [`PLUCKER_EPS`]) zero exactly when the two lines are
/// coplanar — that is, when they intersect or are parallel.
#[must_use]
pub fn side(l1: &Line6, l2: &Line6) -> f32 {
    l1.u.dot(l2.v) + l2.u.dot(l1.v)
}

/// Whether two lines are coplanar, i.e. their [`side`] product is within
/// [`PLUCKER_EPS`] of zero.
#[must_use]
pub fn is_coplanar(l1: &Line6, l2: &Line6) -> bool {
    side(l1, l2).abs() < PLUCKER_EPS
}

/// Classifies the relative orientation of `l2` about `l1` from the sign of
/// their [`side`] product.
#[must_use]
pub fn relative_orientation(l1: &Line6, l2: &Line6) -> Orientation {
    let s = side(l1, l2);
    if s > PLUCKER_EPS {
        Orientation::Clockwise
    } else if s < -PLUCKER_EPS {
        Orientation::CounterClockwise
    } else {
        Orientation::Coplanar
    }
}

/// The three directed edge lines of triangle `(a, b, c)`, wound consistently as
/// `a→b`, `b→c`, `c→a`.
///
/// Feeding these to [`side`] against a query line yields three signs that agree
/// when the query passes through the triangle's interior.
#[must_use]
pub fn triangle_edge_lines(a: Vec3, b: Vec3, c: Vec3) -> [Line6; 3] {
    [
        Line6::from_points(a, b),
        Line6::from_points(b, c),
        Line6::from_points(c, a),
    ]
}

/// The three edge [`side`] values of `ray` against triangle `(a, b, c)`, in
/// edge order `a→b`, `b→c`, `c→a`.
#[must_use]
pub fn triangle_side_signs(ray: &Line6, a: Vec3, b: Vec3, c: Vec3) -> [f32; 3] {
    let edges = triangle_edge_lines(a, b, c);
    [
        side(ray, &edges[0]),
        side(ray, &edges[1]),
        side(ray, &edges[2]),
    ]
}

/// Whether all three `side` values share the same strict sign (all positive or
/// all negative beyond [`PLUCKER_EPS`]).
fn all_same_strict_sign(s: [f32; 3]) -> bool {
    let all_pos = s[0] > PLUCKER_EPS && s[1] > PLUCKER_EPS && s[2] > PLUCKER_EPS;
    let all_neg = s[0] < -PLUCKER_EPS && s[1] < -PLUCKER_EPS && s[2] < -PLUCKER_EPS;
    all_pos || all_neg
}

/// Classifies how the directed line `ray` meets triangle `(a, b, c)` using the
/// Plücker edge-sign test.
///
/// - [`TriangleHit::Inside`] when all three edge sides share a strict sign.
/// - [`TriangleHit::Boundary`] when some side is ~zero and the non-zero sides
///   agree (the line grazes an edge or vertex).
/// - [`TriangleHit::Outside`] otherwise (the non-zero sides disagree).
///
/// The test is direction-agnostic: reversing the ray flips all three signs
/// together, so an interior hit stays an interior hit.
#[must_use]
pub fn classify_ray_triangle(ray: &Line6, a: Vec3, b: Vec3, c: Vec3) -> TriangleHit {
    let s = triangle_side_signs(ray, a, b, c);
    if all_same_strict_sign(s) {
        return TriangleHit::Inside;
    }
    let any_zero = s[0].abs() < PLUCKER_EPS || s[1].abs() < PLUCKER_EPS || s[2].abs() < PLUCKER_EPS;
    if any_zero {
        // Collect the strict signs of the non-zero entries; a boundary hit needs
        // them to all agree (or there to be none, a fully coplanar line).
        let mut saw_pos = false;
        let mut saw_neg = false;
        for value in s {
            if value > PLUCKER_EPS {
                saw_pos = true;
            } else if value < -PLUCKER_EPS {
                saw_neg = true;
            }
        }
        if saw_pos && saw_neg {
            TriangleHit::Outside
        } else {
            TriangleHit::Boundary
        }
    } else {
        TriangleHit::Outside
    }
}

/// Whether the directed line `ray` pierces the interior of triangle
/// `(a, b, c)`, i.e. [`classify_ray_triangle`] is [`TriangleHit::Inside`].
#[must_use]
pub fn ray_hits_triangle(ray: &Line6, a: Vec3, b: Vec3, c: Vec3) -> bool {
    matches!(classify_ray_triangle(ray, a, b, c), TriangleHit::Inside)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::gpu_layout::storage_bytes;

    const TEST_EPS: f32 = 1.0e-5;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < TEST_EPS
    }

    fn decode_f32(bytes: &[u8], off: usize) -> f32 {
        f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    }

    #[test]
    fn vec3_basic_algebra_is_exact() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.add(b), Vec3::new(5.0, 7.0, 9.0));
        assert_eq!(b.sub(a), Vec3::new(3.0, 3.0, 3.0));
        assert_eq!(a.scale(2.0), Vec3::new(2.0, 4.0, 6.0));
        assert!(close(a.dot(b), 32.0));
    }

    #[test]
    fn vec3_cross_is_right_handed() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(x.cross(y), Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(y.cross(x), Vec3::new(0.0, 0.0, -1.0));
    }

    #[test]
    fn vec3_length_matches_squared() {
        let a = Vec3::new(3.0, 4.0, 0.0);
        assert!(close(a.length_squared(), 25.0));
        assert!(close(a.length(), 5.0));
    }

    #[test]
    fn vec3_approx_eq_uses_epsilon() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(1.0 + 1.0e-7, 2.0, 3.0 - 1.0e-7);
        assert!(a.approx_eq(b, PLUCKER_EPS));
        assert!(!a.approx_eq(Vec3::new(1.1, 2.0, 3.0), PLUCKER_EPS));
    }

    #[test]
    fn from_points_sets_direction() {
        let l = Line6::from_points(Vec3::new(1.0, 1.0, 1.0), Vec3::new(4.0, 1.0, 1.0));
        assert_eq!(l.direction(), Vec3::new(3.0, 0.0, 0.0));
    }

    #[test]
    fn from_points_sets_moment_as_cross() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(-2.0, 0.0, 5.0);
        let l = Line6::from_points(a, b);
        assert_eq!(l.moment(), a.cross(b));
    }

    #[test]
    fn moment_orthogonality_is_zero_for_constructed_line() {
        let l = Line6::from_points(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0));
        assert!(l.moment_orthogonality().abs() < PLUCKER_EPS);
    }

    #[test]
    fn moment_orthogonality_zero_for_many_lines() {
        let pts = [
            (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),
            (Vec3::new(-3.0, 2.0, 7.0), Vec3::new(5.0, -1.0, 2.0)),
            (Vec3::new(9.0, 9.0, -9.0), Vec3::new(-4.0, 3.0, 1.0)),
            (Vec3::new(0.5, -0.5, 0.25), Vec3::new(2.0, 8.0, -3.0)),
        ];
        for (a, b) in pts {
            let l = Line6::from_points(a, b);
            assert!(l.moment_orthogonality().abs() < PLUCKER_EPS);
        }
    }

    #[test]
    fn from_raw_preserves_components() {
        let u = Vec3::new(1.0, 0.0, 0.0);
        let v = Vec3::new(0.0, 2.0, 0.0);
        let l = Line6::from_raw(u, v);
        assert_eq!(l.direction(), u);
        assert_eq!(l.moment(), v);
    }

    #[test]
    fn side_is_symmetric() {
        let l1 = Line6::from_points(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let l2 = Line6::from_points(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 1.0, 1.0));
        assert!(close(side(&l1, &l2), side(&l2, &l1)));
    }

    #[test]
    fn skew_lines_have_negative_side() {
        // x-axis through origin vs a line offset along +z running in +y.
        let l1 = Line6::from_points(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let l2 = Line6::from_points(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 1.0, 1.0));
        assert!(close(side(&l1, &l2), -1.0));
        assert!(side(&l1, &l2) < -PLUCKER_EPS);
    }

    #[test]
    fn skew_lines_flip_sign_when_reversed() {
        let l1 = Line6::from_points(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let forward = Line6::from_points(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 1.0, 1.0));
        let reversed = Line6::from_points(Vec3::new(0.0, 1.0, 1.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(side(&l1, &forward) < -PLUCKER_EPS);
        assert!(side(&l1, &reversed) > PLUCKER_EPS);
        assert!(close(side(&l1, &forward), -side(&l1, &reversed)));
    }

    #[test]
    fn crossing_coplanar_lines_have_zero_side() {
        // Two lines that cross at (1, 0, 0) inside the z = 0 plane.
        let l1 = Line6::from_points(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0));
        let l2 = Line6::from_points(Vec3::new(1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 0.0));
        assert!(side(&l1, &l2).abs() < PLUCKER_EPS);
    }

    #[test]
    fn parallel_lines_are_coplanar() {
        let l1 = Line6::from_points(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let l2 = Line6::from_points(Vec3::new(0.0, 5.0, 0.0), Vec3::new(1.0, 5.0, 0.0));
        assert!(is_coplanar(&l1, &l2));
    }

    #[test]
    fn is_coplanar_true_for_crossing_lines() {
        let l1 = Line6::from_points(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0));
        let l2 = Line6::from_points(Vec3::new(1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 0.0));
        assert!(is_coplanar(&l1, &l2));
    }

    #[test]
    fn is_coplanar_false_for_skew_lines() {
        let l1 = Line6::from_points(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let l2 = Line6::from_points(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 1.0, 1.0));
        assert!(!is_coplanar(&l1, &l2));
    }

    #[test]
    fn lines_through_origin_meet_with_zero_side() {
        let l1 = Line6::from_points(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let l2 = Line6::from_points(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(l1.moment(), Vec3::ZERO);
        assert_eq!(l2.moment(), Vec3::ZERO);
        assert!(side(&l1, &l2).abs() < PLUCKER_EPS);
    }

    #[test]
    fn relative_orientation_classifies_all_three_cases() {
        let l1 = Line6::from_points(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let cw = Line6::from_points(Vec3::new(0.0, 1.0, 1.0), Vec3::new(0.0, 0.0, 1.0));
        let ccw = Line6::from_points(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 1.0, 1.0));
        let coplanar = Line6::from_points(Vec3::new(0.0, 5.0, 0.0), Vec3::new(1.0, 5.0, 0.0));
        assert_eq!(relative_orientation(&l1, &cw), Orientation::Clockwise);
        assert_eq!(
            relative_orientation(&l1, &ccw),
            Orientation::CounterClockwise
        );
        assert_eq!(relative_orientation(&l1, &coplanar), Orientation::Coplanar);
    }

    #[test]
    fn is_valid_rejects_degenerate_line() {
        let good = Line6::from_points(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let degenerate = Line6::from_points(Vec3::new(2.0, 2.0, 2.0), Vec3::new(2.0, 2.0, 2.0));
        assert!(good.is_valid());
        assert!(!degenerate.is_valid());
    }

    #[test]
    fn edge_lines_wind_consistently() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);
        let edges = triangle_edge_lines(a, b, c);
        assert_eq!(edges[0].direction(), b.sub(a));
        assert_eq!(edges[1].direction(), c.sub(b));
        assert_eq!(edges[2].direction(), a.sub(c));
    }

    #[test]
    fn ray_through_interior_has_all_positive_sides() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);
        // Ray along +z through the interior point (0.25, 0.25).
        let ray = Line6::from_points(Vec3::new(0.25, 0.25, -1.0), Vec3::new(0.25, 0.25, 1.0));
        let s = triangle_side_signs(&ray, a, b, c);
        assert!(s[0] > PLUCKER_EPS && s[1] > PLUCKER_EPS && s[2] > PLUCKER_EPS);
        assert!(ray_hits_triangle(&ray, a, b, c));
    }

    #[test]
    fn ray_from_other_side_has_all_negative_sides() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);
        // Same interior point but the line is oriented the opposite way.
        let ray = Line6::from_points(Vec3::new(0.25, 0.25, 1.0), Vec3::new(0.25, 0.25, -1.0));
        let s = triangle_side_signs(&ray, a, b, c);
        assert!(s[0] < -PLUCKER_EPS && s[1] < -PLUCKER_EPS && s[2] < -PLUCKER_EPS);
        assert!(ray_hits_triangle(&ray, a, b, c));
    }

    #[test]
    fn ray_outside_has_mixed_signs_and_misses() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);
        // (2, 2) is well outside the unit right triangle.
        let ray = Line6::from_points(Vec3::new(2.0, 2.0, -1.0), Vec3::new(2.0, 2.0, 1.0));
        let s = triangle_side_signs(&ray, a, b, c);
        let has_pos = s[0] > PLUCKER_EPS || s[1] > PLUCKER_EPS || s[2] > PLUCKER_EPS;
        let has_neg = s[0] < -PLUCKER_EPS || s[1] < -PLUCKER_EPS || s[2] < -PLUCKER_EPS;
        assert!(has_pos && has_neg);
        assert!(!ray_hits_triangle(&ray, a, b, c));
    }

    #[test]
    fn ray_classify_reports_inside_and_outside() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 2.0, 0.0);
        let inside = Line6::from_points(Vec3::new(0.4, 0.4, -1.0), Vec3::new(0.4, 0.4, 1.0));
        let outside = Line6::from_points(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(-1.0, -1.0, 1.0));
        assert_eq!(classify_ray_triangle(&inside, a, b, c), TriangleHit::Inside);
        assert_eq!(
            classify_ray_triangle(&outside, a, b, c),
            TriangleHit::Outside
        );
    }

    #[test]
    fn ray_through_edge_is_boundary() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 2.0, 0.0);
        // Pierce the midpoint of edge a->b at (1, 0): the a->b side is ~zero.
        let ray = Line6::from_points(Vec3::new(1.0, 0.0, -1.0), Vec3::new(1.0, 0.0, 1.0));
        let s = triangle_side_signs(&ray, a, b, c);
        assert!(s[0].abs() < PLUCKER_EPS);
        assert_eq!(classify_ray_triangle(&ray, a, b, c), TriangleHit::Boundary);
    }

    #[test]
    fn ray_through_vertex_is_boundary() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 2.0, 0.0);
        // Pierce vertex a at the origin: two adjacent edges give ~zero side.
        let ray = Line6::from_points(Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(classify_ray_triangle(&ray, a, b, c), TriangleHit::Boundary);
    }

    #[test]
    fn to_std430_size_is_thirty_two() {
        assert_eq!(LINE6_STD430_SIZE, 32);
        let l = Line6::from_points(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0));
        assert_eq!(l.to_std430().len(), 32);
    }

    #[test]
    fn to_std430_is_vec4_multiple() {
        assert_eq!(LINE6_STD430_SIZE % VEC4_STRIDE, 0);
        assert_eq!(LINE6_STD430_SIZE % 16, 0);
    }

    #[test]
    fn storage_bytes_stays_sixteen_aligned() {
        for count in [0usize, 1, 2, 7, 64, 4096] {
            let bytes = storage_bytes(LINE6_STD430_SIZE, count);
            assert_eq!(bytes % 16, 0);
        }
        assert_eq!(storage_bytes(LINE6_STD430_SIZE, 0), 32);
        assert_eq!(storage_bytes(LINE6_STD430_SIZE, 3), 96);
    }

    #[test]
    fn to_std430_roundtrips_components_with_padding() {
        let l = Line6::from_points(Vec3::new(1.0, 2.0, 3.0), Vec3::new(-4.0, 5.0, -6.0));
        let bytes = l.to_std430();
        assert!(close(decode_f32(&bytes, 0), l.u.x));
        assert!(close(decode_f32(&bytes, 4), l.u.y));
        assert!(close(decode_f32(&bytes, 8), l.u.z));
        assert!(close(decode_f32(&bytes, 12), 0.0));
        assert!(close(decode_f32(&bytes, 16), l.v.x));
        assert!(close(decode_f32(&bytes, 20), l.v.y));
        assert!(close(decode_f32(&bytes, 24), l.v.z));
        assert!(close(decode_f32(&bytes, 28), 0.0));
    }

    #[test]
    fn from_points_is_deterministic_bitwise() {
        let a = Vec3::new(1.25, -2.5, 3.75);
        let b = Vec3::new(-4.5, 5.5, -6.25);
        let first = Line6::from_points(a, b);
        let second = Line6::from_points(a, b);
        assert_eq!(first.to_std430(), second.to_std430());
    }

    #[test]
    fn side_is_deterministic() {
        let l1 = Line6::from_points(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let l2 = Line6::from_points(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 1.0, 1.0));
        let s1 = side(&l1, &l2);
        let s2 = side(&l1, &l2);
        assert_eq!(s1.to_le_bytes(), s2.to_le_bytes());
    }

    #[test]
    fn interior_hit_is_orientation_agnostic() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);
        let fwd = Line6::from_points(Vec3::new(0.2, 0.2, -1.0), Vec3::new(0.2, 0.2, 1.0));
        let rev = Line6::from_points(Vec3::new(0.2, 0.2, 1.0), Vec3::new(0.2, 0.2, -1.0));
        assert!(ray_hits_triangle(&fwd, a, b, c));
        assert!(ray_hits_triangle(&rev, a, b, c));
    }

    #[test]
    fn coplanar_ray_in_triangle_plane_is_not_interior() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 2.0, 0.0);
        // A ray lying in the z = 0 plane never pierces the interior.
        let ray = Line6::from_points(Vec3::new(-1.0, 0.5, 0.0), Vec3::new(3.0, 0.5, 0.0));
        assert!(!ray_hits_triangle(&ray, a, b, c));
    }
}
