//! Closest point on an *oriented bounding box* (`OBB`) to a query point, with
//! the distance, squared distance, and an inside/outside flag (design §8.2,
//! §10, §14).
//!
//! Many particle stages need the *nearest point of a box* to an arbitrary point
//! in space: clamping a spawn or collision proxy back onto an oriented volume,
//! resolving a soft-body/particle penetration against a rotated bound, snapping
//! a decal footprint onto an oriented brick, or answering a proximity query in a
//! broad phase whose bounds are rotated. This module owns that single,
//! `CPU`-verifiable geometry contract: given a point `p` and an `OBB`
//! (`center` + three orthonormal `axes` + non-negative `half` extents), it
//! returns the point of the closed, filled box nearest to `p`, the Euclidean
//! distance and squared distance to it, and whether `p` lies inside the box.
//!
//! # The `OBB` and the algorithm
//! An **`OBB`** is an [`Aabb`](crate::particle::sort_cull)-shaped box that has
//! been *rotated* into world space: a `center`, three mutually orthogonal
//! **unit** axes `axes[0..3]`, and a non-negative half-extent `half[i]` along
//! each axis. The axes are *assumed* orthonormal (right- or left-handed does not
//! matter); the module does not re-orthonormalize them. Because an orthonormal
//! basis is its own inverse, the signed coordinate of `p` in the box frame along
//! axis `i` is simply the dot product `(p - center) . axes[i]`. Clamping that
//! coordinate to `[-half[i], half[i]]` and rebuilding a world point from the
//! clamped coordinates yields the nearest point of the box: each local axis is
//! solved independently, so no search or iteration is required. The point is
//! *inside* exactly when every clamp was a no-op, i.e. every projection's
//! magnitude is within its half-extent.
//!
//! # Strict scope — how this differs from its siblings
//! This is the **point-to-`OBB` nearest-point** query and nothing else. It is
//! deliberately disjoint from the other proximity/geometry modules and neither
//! imports nor reconstructs their primitives:
//!
//! * [`crate::particle::ray_obb`] answers a **ray-vs-`OBB` intersection**
//!   question (does a ray pierce the box, and at what `t`) via the slab method;
//!   it is a hit test along a direction, not a nearest-point-in-space query.
//! * [`crate::particle::point_triangle_closest_3d`] clamps a point onto a
//!   filled **triangle** (a 2D simplex) through Voronoi-region case analysis;
//!   this module clamps onto a **box** (a 3D solid) through per-axis interval
//!   clamping.
//! * [`crate::particle::segment_closest_point_3d`] finds closest points between
//!   two **segments** (1D primitives); this module's other primitive is a 3D
//!   solid, not a segment.
//! * [`crate::particle::capsule_sdf`] evaluates a *signed distance field* of a
//!   **capsule** (a swept sphere) and returns a signed scalar; this module
//!   returns the explicit nearest *point* on a *box* plus an unsigned distance
//!   and a boolean containment flag, not an `SDF` sample.
//!
//! # No transcendental math, no float equality
//! Every step uses only `+ - * /`, [`f32::clamp`], [`f32::abs`], and
//! [`f32::sqrt`] (called once, only where a genuine Euclidean length is needed).
//! No transcendental function is ever called and no `==` / `!=` comparison on an
//! `f32` ever appears: containment is decided with `<=` against the half-extents
//! and the axes are trusted as unit orthonormal. This keeps the `CPU` reference
//! bit-reproducible against a future `GPU` (`WESL`) kernel that packs the same
//! results through the `std430` helpers in [`crate::particle::gpu_layout`].

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// A hand-rolled three-component vector, kept local so the module stays a
/// zero-dependency contract and its vector math is auditable in one place
/// (mirroring [`crate::particle::ray_obb`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    /// The x component.
    pub x: f32,
    /// The y component.
    pub y: f32,
    /// The z component.
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

    /// Component-wise sum `self + rhs` (named `plus` so the internal type does
    /// not implement the [`core::ops::Add`] operator trait).
    #[must_use]
    pub fn plus(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    /// Component-wise difference `self - rhs` (named `minus` so the internal
    /// type does not implement the [`core::ops::Sub`] operator trait).
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

    /// Squared Euclidean length (no `sqrt`).
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length. This is the module's only use of [`f32::sqrt`].
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Returns the components as a plain `[f32; 3]` array.
    #[must_use]
    pub const fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
}

/// An oriented bounding box: a `center`, three orthonormal `axes`, and a
/// non-negative `half` extent along each axis.
///
/// The `axes` are *assumed* to be unit-length and mutually orthogonal; the
/// solver relies on that (an orthonormal basis is its own inverse) and does not
/// re-orthonormalize. Half extents are expected to be non-negative; a zero
/// half-extent collapses that axis so the box degenerates to a face, an edge, or
/// a single point at `center`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Obb {
    /// The box center in world space.
    pub center: Vec3,
    /// Three orthonormal local axes.
    pub axes: [Vec3; 3],
    /// Non-negative half-extent along each corresponding axis.
    pub half: [f32; 3],
}

/// The result of a point-to-`OBB` nearest-point query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClosestPoint {
    /// The nearest point of the box to the query, in world space.
    pub point: Vec3,
    /// The Euclidean distance from the query point to [`Self::point`].
    pub distance: f32,
    /// The squared Euclidean distance (no `sqrt`), handy for comparisons.
    pub distance_squared: f32,
    /// Whether the query point lies inside (or on the surface of) the box.
    pub inside: bool,
}

impl Obb {
    /// Builds an `OBB` from its center, orthonormal axes, and half extents.
    #[must_use]
    pub const fn new(center: Vec3, axes: [Vec3; 3], half: [f32; 3]) -> Self {
        Self { center, axes, half }
    }

    /// Builds an axis-aligned box (`AABB`) as the degenerate `OBB` whose axes
    /// are the world basis vectors, so this solver agrees with a naive per-axis
    /// clamp.
    #[must_use]
    pub const fn from_aabb(center: Vec3, half: [f32; 3]) -> Self {
        Self::new(
            center,
            [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
            half,
        )
    }

    /// Returns the nearest point of the closed, filled box to `query`, together
    /// with the distance, squared distance, and whether `query` is inside.
    ///
    /// The point relative to `center` is projected onto each orthonormal axis,
    /// each projection is clamped to `[-half[i], half[i]]`, and the world-space
    /// nearest point is rebuilt from the clamped projections. Uses only
    /// `+ - * /`, [`f32::clamp`], and a single [`f32::sqrt`].
    #[must_use]
    pub fn closest_point(&self, query: Vec3) -> ClosestPoint {
        let rel = query.minus(self.center);
        let mut point = self.center;
        let mut inside = true;
        for (axis, &half) in self.axes.iter().zip(self.half.iter()) {
            let proj = rel.dot(*axis);
            let clamped = proj.clamp(-half, half);
            // `proj.abs() <= half` is a `<=` test, never an `f32` equality.
            if proj.abs() > half {
                inside = false;
            }
            point = point.plus(axis.scale(clamped));
        }
        let diff = point.minus(query);
        let distance_squared = diff.length_squared();
        ClosestPoint {
            point,
            distance: distance_squared.sqrt(),
            distance_squared,
            inside,
        }
    }
}

/// `std430` stride, in bytes, of one packed nearest-point record: a `vec4` for
/// the point (`xyz` + distance in `w`) laid out through
/// [`crate::particle::gpu_layout`].
pub const CLOSEST_STRIDE: usize = VEC4_STRIDE;

/// Bytes required to store `count` packed nearest-point records in a `std430`
/// storage buffer (a `WebGPU` binding may not be zero-sized, so an empty pool
/// still reserves one element).
#[must_use]
pub fn std430_bytes(count: usize) -> usize {
    storage_bytes(CLOSEST_STRIDE, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One-quarter-turn cosine/sine literal (`1 / sqrt(2)`), written as a plain
    /// constant so the tests never call a transcendental function.
    const SQRT_1_2: f32 = core::f32::consts::FRAC_1_SQRT_2;

    const EPS: f32 = 1.0e-4;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() <= EPS, "expected {a} ~= {b}");
    }

    fn approx_vec(a: Vec3, b: Vec3) {
        approx(a.x, b.x);
        approx(a.y, b.y);
        approx(a.z, b.z);
    }

    /// A unit axis-aligned box centered at the origin with half extents 1.
    fn unit_aabb() -> Obb {
        Obb::from_aabb(Vec3::ZERO, [1.0, 1.0, 1.0])
    }

    /// A box rotated 45 degrees about `z`, centered at the origin, half 1.
    fn rot45() -> Obb {
        Obb::new(
            Vec3::ZERO,
            [
                Vec3::new(SQRT_1_2, SQRT_1_2, 0.0),
                Vec3::new(-SQRT_1_2, SQRT_1_2, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
            [1.0, 1.0, 1.0],
        )
    }

    /// Naive per-component clamp for an axis-aligned box, for cross-checking.
    fn naive_aabb_clamp(p: Vec3, center: Vec3, half: [f32; 3]) -> Vec3 {
        Vec3::new(
            p.x.clamp(center.x - half[0], center.x + half[0]),
            p.y.clamp(center.y - half[1], center.y + half[1]),
            p.z.clamp(center.z - half[2], center.z + half[2]),
        )
    }

    // ----- inside cases -------------------------------------------------------

    #[test]
    fn center_is_inside_zero_distance() {
        let r = unit_aabb().closest_point(Vec3::ZERO);
        approx_vec(r.point, Vec3::ZERO);
        approx(r.distance, 0.0);
        approx(r.distance_squared, 0.0);
        assert!(r.inside);
    }

    #[test]
    fn generic_interior_point_maps_to_itself() {
        let p = Vec3::new(0.25, -0.5, 0.75);
        let r = unit_aabb().closest_point(p);
        approx_vec(r.point, p);
        approx(r.distance, 0.0);
        assert!(r.inside);
    }

    #[test]
    fn inside_flag_true_for_interior() {
        let r = unit_aabb().closest_point(Vec3::new(0.9, -0.9, 0.1));
        assert!(r.inside);
    }

    #[test]
    fn inside_flag_false_for_exterior() {
        let r = unit_aabb().closest_point(Vec3::new(1.5, 0.0, 0.0));
        assert!(!r.inside);
    }

    #[test]
    fn point_on_face_counts_as_inside() {
        // Exactly on the +x face; projection magnitude equals the half extent.
        let r = unit_aabb().closest_point(Vec3::new(1.0, 0.0, 0.0));
        approx_vec(r.point, Vec3::new(1.0, 0.0, 0.0));
        approx(r.distance, 0.0);
        assert!(r.inside);
    }

    #[test]
    fn interior_point_distance_is_zero_everywhere() {
        let b = unit_aabb();
        for p in [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-0.9, 0.2, -0.3),
        ] {
            let r = b.closest_point(p);
            approx(r.distance, 0.0);
            assert!(r.inside);
        }
    }

    // ----- outside: face / edge / corner --------------------------------------

    #[test]
    fn outside_a_face_projects_perpendicular() {
        let r = unit_aabb().closest_point(Vec3::new(3.0, 0.25, -0.5));
        approx_vec(r.point, Vec3::new(1.0, 0.25, -0.5));
        approx(r.distance, 2.0);
        approx(r.distance_squared, 4.0);
        assert!(!r.inside);
    }

    #[test]
    fn outside_an_edge_clamps_two_axes() {
        // Beyond the +x/+y edge but within the z slab.
        let r = unit_aabb().closest_point(Vec3::new(2.0, 2.0, 0.5));
        approx_vec(r.point, Vec3::new(1.0, 1.0, 0.5));
        approx(r.distance_squared, 2.0);
        approx(r.distance, 2.0_f32.sqrt());
        assert!(!r.inside);
    }

    #[test]
    fn outside_a_corner_clamps_all_axes() {
        let r = unit_aabb().closest_point(Vec3::new(2.0, 3.0, 4.0));
        approx_vec(r.point, Vec3::new(1.0, 1.0, 1.0));
        approx(r.distance_squared, 1.0 + 4.0 + 9.0);
        assert!(!r.inside);
    }

    // ----- axis-aligned degenerates to a naive clamp --------------------------

    #[test]
    fn axis_aligned_matches_naive_clamp() {
        let center = Vec3::new(1.0, -2.0, 0.5);
        let half = [2.0, 0.5, 1.5];
        let b = Obb::from_aabb(center, half);
        for p in [
            Vec3::new(5.0, 5.0, 5.0),
            Vec3::new(-3.0, -3.0, -3.0),
            Vec3::new(1.0, -2.0, 0.5),
            Vec3::new(0.0, 10.0, -4.0),
            Vec3::new(1.3, -2.4, 0.9),
        ] {
            let r = b.closest_point(p);
            approx_vec(r.point, naive_aabb_clamp(p, center, half));
        }
    }

    // ----- rotated 45-degree box ----------------------------------------------

    #[test]
    fn rotated_box_outside_along_x() {
        // p on +x, sqrt(2) from the origin corner of the rotated square.
        let r = rot45().closest_point(Vec3::new(2.0, 0.0, 0.0));
        approx_vec(r.point, Vec3::new(2.0_f32.sqrt(), 0.0, 0.0));
        approx(r.distance, 2.0 - 2.0_f32.sqrt());
        assert!(!r.inside);
    }

    #[test]
    fn rotated_box_interior_is_self() {
        let p = Vec3::new(0.3, 0.1, 0.4);
        let r = rot45().closest_point(p);
        approx_vec(r.point, p);
        approx(r.distance, 0.0);
        assert!(r.inside);
    }

    #[test]
    fn rotated_box_corner_query() {
        // Local corner (+u, +v) sits at world (0, sqrt(2), 0); query beyond it.
        let corner = Vec3::new(0.0, 2.0_f32.sqrt(), 0.0);
        let far = Vec3::new(0.0, 3.0, 0.0);
        let r = rot45().closest_point(far);
        approx_vec(r.point, corner);
        assert!(!r.inside);
    }

    #[test]
    fn rotated_box_point_just_outside_is_not_inside() {
        // Slightly past the +u face along the u direction.
        let p = Vec3::new(1.1 * SQRT_1_2, 1.1 * SQRT_1_2, 0.0);
        let r = rot45().closest_point(p);
        assert!(!r.inside);
    }

    // ----- degenerate boxes: face / line / point ------------------------------

    #[test]
    fn degenerate_flat_box_is_a_face() {
        // Half z = 0: the box is a square plate in the z = center plane.
        let b = Obb::from_aabb(Vec3::ZERO, [1.0, 1.0, 0.0]);
        let r = b.closest_point(Vec3::new(0.5, -0.5, 3.0));
        approx_vec(r.point, Vec3::new(0.5, -0.5, 0.0));
        approx(r.distance, 3.0);
        assert!(!r.inside);
    }

    #[test]
    fn degenerate_line_box_is_a_segment() {
        // Only the x axis has extent: the box is a segment on the x axis.
        let b = Obb::from_aabb(Vec3::ZERO, [2.0, 0.0, 0.0]);
        let r = b.closest_point(Vec3::new(1.0, 3.0, 4.0));
        approx_vec(r.point, Vec3::new(1.0, 0.0, 0.0));
        approx(r.distance, 5.0);
        assert!(!r.inside);
    }

    #[test]
    fn degenerate_line_box_clamps_along_length() {
        let b = Obb::from_aabb(Vec3::ZERO, [2.0, 0.0, 0.0]);
        let r = b.closest_point(Vec3::new(9.0, 0.0, 0.0));
        approx_vec(r.point, Vec3::new(2.0, 0.0, 0.0));
        approx(r.distance, 7.0);
    }

    #[test]
    fn degenerate_point_box_is_the_center() {
        let center = Vec3::new(1.0, 2.0, 3.0);
        let b = Obb::from_aabb(center, [0.0, 0.0, 0.0]);
        let r = b.closest_point(Vec3::new(4.0, 6.0, 3.0));
        approx_vec(r.point, center);
        approx(r.distance, 5.0);
        assert!(!r.inside);
    }

    #[test]
    fn degenerate_point_box_query_at_center_is_inside() {
        let center = Vec3::new(1.0, 2.0, 3.0);
        let b = Obb::from_aabb(center, [0.0, 0.0, 0.0]);
        let r = b.closest_point(center);
        approx(r.distance, 0.0);
        assert!(r.inside);
    }

    // ----- exact known distances ----------------------------------------------

    #[test]
    fn known_exact_closest_and_distance() {
        let b = Obb::from_aabb(Vec3::new(2.0, 0.0, 0.0), [1.0, 1.0, 1.0]);
        // Box spans x in [1, 3]; query at x = -2 is 3 units from the -x face.
        let r = b.closest_point(Vec3::new(-2.0, 0.0, 0.0));
        approx_vec(r.point, Vec3::new(1.0, 0.0, 0.0));
        approx(r.distance, 3.0);
        approx(r.distance_squared, 9.0);
    }

    #[test]
    fn distance_matches_sqrt_of_distance_squared() {
        let b = unit_aabb();
        let p = Vec3::new(2.0, 3.0, -4.0);
        let r = b.closest_point(p);
        approx(r.distance * r.distance, r.distance_squared);
    }

    #[test]
    fn distance_squared_never_negative() {
        let b = rot45();
        for p in [
            Vec3::new(5.0, -5.0, 2.0),
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(0.0, 0.0, 0.0),
        ] {
            assert!(b.closest_point(p).distance_squared >= 0.0);
        }
    }

    // ----- symmetry -----------------------------------------------------------

    #[test]
    fn symmetry_about_center_mirrors_point_and_keeps_distance() {
        let center = Vec3::new(0.5, -0.5, 1.0);
        let b = Obb::from_aabb(center, [1.0, 2.0, 0.5]);
        let p = Vec3::new(4.0, 3.0, -2.0);
        // Mirror of p through the center: 2*center - p.
        let mirror = center.scale(2.0).minus(p);
        let r_p = b.closest_point(p);
        let r_m = b.closest_point(mirror);
        approx(r_p.distance, r_m.distance);
        // The two nearest points are mirror images through the center.
        let mirrored_point = center.scale(2.0).minus(r_p.point);
        approx_vec(r_m.point, mirrored_point);
    }

    #[test]
    fn symmetry_rotated_box() {
        let b = rot45();
        let p = Vec3::new(2.0, 1.0, 0.5);
        let mirror = Vec3::ZERO.minus(p);
        let r_p = b.closest_point(p);
        let r_m = b.closest_point(mirror);
        approx(r_p.distance, r_m.distance);
        approx_vec(r_m.point, Vec3::ZERO.minus(r_p.point));
    }

    // ----- nearest point is a lower bound -------------------------------------

    #[test]
    fn nearest_is_no_farther_than_any_corner() {
        let b = unit_aabb();
        let p = Vec3::new(2.0, -3.0, 1.5);
        let r = b.closest_point(p);
        for sx in [-1.0_f32, 1.0] {
            for sy in [-1.0_f32, 1.0] {
                for sz in [-1.0_f32, 1.0] {
                    let corner = Vec3::new(sx, sy, sz);
                    let d = corner.minus(p).length_squared();
                    assert!(r.distance_squared <= d + EPS);
                }
            }
        }
    }

    #[test]
    fn nearest_projection_within_extents_for_rotated_box() {
        let b = rot45();
        let r = b.closest_point(Vec3::new(5.0, -3.0, 2.0));
        let rel = r.point.minus(b.center);
        for (axis, &half) in b.axes.iter().zip(b.half.iter()) {
            let proj = rel.dot(*axis);
            assert!(proj.abs() <= half + EPS, "projection {proj} exceeds {half}");
        }
    }

    // ----- determinism --------------------------------------------------------

    #[test]
    fn deterministic_to_bits() {
        let b = rot45();
        let p = Vec3::new(0.137, 0.951, 0.618);
        let r1 = b.closest_point(p);
        let r2 = b.closest_point(p);
        assert_eq!(r1.point.x.to_bits(), r2.point.x.to_bits());
        assert_eq!(r1.point.y.to_bits(), r2.point.y.to_bits());
        assert_eq!(r1.point.z.to_bits(), r2.point.z.to_bits());
        assert_eq!(r1.distance.to_bits(), r2.distance.to_bits());
        assert_eq!(r1.distance_squared.to_bits(), r2.distance_squared.to_bits());
    }

    #[test]
    fn to_array_matches_components() {
        let b = unit_aabb();
        let r = b.closest_point(Vec3::new(2.0, 0.3, -0.4));
        let a = r.point.to_array();
        approx(a[0], r.point.x);
        approx(a[1], r.point.y);
        approx(a[2], r.point.z);
    }

    // ----- std430 layout ------------------------------------------------------

    #[test]
    fn std430_stride_is_multiple_of_16() {
        assert_eq!(CLOSEST_STRIDE % 16, 0);
        assert_eq!(CLOSEST_STRIDE, 16);
    }

    #[test]
    fn std430_bytes_scale_with_count() {
        assert_eq!(std430_bytes(1), 16);
        assert_eq!(std430_bytes(4), 64);
        assert_eq!(std430_bytes(10) % 16, 0);
    }

    #[test]
    fn std430_empty_reserves_one_element() {
        assert_eq!(std430_bytes(0), CLOSEST_STRIDE);
    }
}
