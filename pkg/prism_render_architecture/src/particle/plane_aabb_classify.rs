//! Single-plane classification of an axis-aligned bounding box (`AABB`) for the
//! particle culling and slicing passes (design §12, §13).
//!
//! This module answers one narrow, purely geometric question: on which side of
//! an oriented plane does a box lie? Given a plane `n·x + d = 0` and a box, it
//! decides whether the box is entirely on the *positive* half-space
//! (`n·x + d > 0`), entirely on the *negative* half-space (`n·x + d < 0`), or
//! straddles the plane. The test is the textbook *center/projected-radius*
//! form: the box's signed distance is measured at its center,
//! `s = n·center + d`, and its extent along the plane normal is the box's
//! projected radius `r = |n_x|·h_x + |n_y|·h_y + |n_z|·h_z` — an `L1`-weighted
//! sum of the half-extents. The box is `Positive` when `s - r > 0`, `Negative`
//! when `s + r < 0`, and `Intersecting` otherwise. This is the deterministic
//! `CPU` reference a future `GPU` culling kernel reproduces bit for bit.
//!
//! # Relationship to the sibling modules (strict boundary)
//!
//! Several files in this subsystem touch planes, frusta, and boxes; this one is
//! deliberately disjoint and does exactly one thing:
//!
//! * [`crate::particle::frustum_plane_extract`] *derives* the six oriented
//!   planes of a view frustum from a combined view-projection matrix. It
//!   produces planes; this module *consumes* a single plane and never extracts
//!   one from a matrix.
//! * [`crate::particle::plane_clip`] *cuts* segment and convex-polygon geometry
//!   against a plane, emitting new vertices on the boundary. It rewrites
//!   geometry; this module only reports a side and never produces clipped
//!   vertices, intersection points, or interpolated attributes.
//! * A `frustum_aabb_cull` pass (six-plane frustum rejection) would loop this
//!   single-plane test over all six frustum planes and fold the results into an
//!   accept / reject / straddle decision. This module owns only the *one-plane*
//!   primitive that such a pass builds on; it never iterates a plane set,
//!   knows nothing about a frustum, and makes no visibility decision.
//! * [`crate::particle::sphere_aabb`] tests a *sphere* against a box; this
//!   module tests a *plane* against a box. The primitives are different.
//!
//! # Plane convention
//!
//! A [`Plane`] stores `normal = (n_x, n_y, n_z)` and `d` for the equation
//! `n·x + d = 0`. The normal **need not be unit length**: classification only
//! reads the *signs* of `s - r` and `s + r`, and scaling `normal` and `d` by a
//! common positive factor scales both `s` and `r` by that factor, leaving every
//! side decision unchanged. The magnitude of `s = n·center + d` is therefore a
//! true signed Euclidean distance *only* when `normal` is unit length; when it
//! is not, `s` is that distance times `|normal|`. The classification result is
//! invariant either way.
//!
//! # Determinism
//!
//! Everything here is a zero-dependency contract with hand-rolled vector math.
//! No floating-point primitive beyond ordinary `+ - * /` and [`f32::abs`] is
//! used; there is no [`f32::sqrt`], no normalization, and no transcendental
//! function, so the `CPU` reference is deterministic and a matching `GPU`
//! kernel produces identical sides. A `NaN` coordinate propagates through the
//! comparisons as a non-decision and falls through to [`Side::Intersecting`],
//! which is the conservative "do not cull" answer. Floating-point `==` / `!=`
//! are never written on an `f32`; boundary decisions go through [`CMP_EPS`] so
//! that a box which merely grazes the plane is reported as
//! [`Side::Intersecting`] rather than being forced onto one side.

/// Absolute tolerance used to guard the side decision without ever writing an
/// exact `==` / `!=` on a production `f32`. A box whose supporting distance
/// lands within this band of the plane is treated as touching and reported as
/// [`Side::Intersecting`].
pub const CMP_EPS: f32 = 1.0e-6;

/// Dot product of two three-component vectors, kept as a free function so the
/// module stays a zero-dependency contract and no inherent operator-like method
/// is introduced on a bare array.
#[must_use]
pub fn v_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a.iter().zip(b.iter()).map(|(lhs, rhs)| lhs * rhs).sum()
}

/// Which side of a plane a box occupies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// The whole box lies strictly on the `n·x + d > 0` half-space.
    Positive,
    /// The whole box lies strictly on the `n·x + d < 0` half-space.
    Negative,
    /// The box straddles or merely grazes the plane; this is the conservative
    /// answer whenever the box is not strictly on one side.
    Intersecting,
}

/// An oriented plane `n·x + d = 0`.
///
/// See the module docs for the sign and normalization conventions: `normal`
/// need not be unit length, and only the signs of the classification terms are
/// consulted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    /// The plane normal `(n_x, n_y, n_z)`; not required to be unit length.
    pub normal: [f32; 3],
    /// The plane constant `d` in `n·x + d = 0`.
    pub d: f32,
}

impl Plane {
    /// Builds a plane from its normal and constant.
    #[must_use]
    pub const fn new(normal: [f32; 3], d: f32) -> Self {
        Self { normal, d }
    }
}

/// An axis-aligned bounding box stored as a center and per-axis half-extents.
///
/// The half-extents are expected to be non-negative; the projected-radius math
/// takes their absolute value so a caller that passes a sign-flipped half is
/// still treated as describing the same box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// The box center `(c_x, c_y, c_z)`.
    pub center: [f32; 3],
    /// The per-axis half-extents `(h_x, h_y, h_z)`.
    pub half: [f32; 3],
}

impl Aabb {
    /// Builds a box from its center and half-extents.
    #[must_use]
    pub const fn new(center: [f32; 3], half: [f32; 3]) -> Self {
        Self { center, half }
    }

    /// Builds a box from its minimum and maximum corners.
    ///
    /// The center is the midpoint `(min + max) / 2` and the half-extents are
    /// `(max - min) / 2` per axis, taken through [`f32::abs`] so a swapped
    /// `min`/`max` pair still yields non-negative half-extents.
    #[must_use]
    pub fn from_min_max(min: [f32; 3], max: [f32; 3]) -> Self {
        let mut center = [0.0_f32; 3];
        let mut half = [0.0_f32; 3];
        for ((slot_c, slot_h), (lo, hi)) in center
            .iter_mut()
            .zip(half.iter_mut())
            .zip(min.iter().zip(max.iter()))
        {
            *slot_c = (lo + hi) * 0.5;
            *slot_h = ((hi - lo) * 0.5).abs();
        }
        Self { center, half }
    }
}

/// Signed distance of the box center to the plane, `s = n·center + d`.
///
/// This is a true Euclidean distance only when the plane normal is unit length;
/// see the module docs. It is exposed because the culling stage often wants the
/// center's side ordering directly.
#[must_use]
pub fn signed_distance_center(plane: Plane, aabb: Aabb) -> f32 {
    v_dot(plane.normal, aabb.center) + plane.d
}

/// The box's projected radius along the plane normal, the `L1`-weighted sum
/// `r = |n_x|·h_x + |n_y|·h_y + |n_z|·h_z`.
///
/// This is the box's maximum extent measured along `normal`, and it is always
/// non-negative. The three axes are folded with a `zip` so no index loop is
/// needed.
#[must_use]
pub fn projected_radius(normal: [f32; 3], half: [f32; 3]) -> f32 {
    normal
        .iter()
        .zip(half.iter())
        .map(|(n, h)| n.abs() * h.abs())
        .sum()
}

/// Classifies a box against a plane as [`Side::Positive`], [`Side::Negative`],
/// or [`Side::Intersecting`].
///
/// The decision compares the center's signed distance `s` against the box's
/// projected radius `r`: the box is strictly positive when `s - r` exceeds
/// [`CMP_EPS`], strictly negative when `s + r` is below `-CMP_EPS`, and
/// otherwise reported as intersecting. A box that only grazes the plane, or any
/// input carrying a `NaN`, therefore falls through to [`Side::Intersecting`].
#[must_use]
pub fn classify(plane: Plane, aabb: Aabb) -> Side {
    let s = signed_distance_center(plane, aabb);
    let r = projected_radius(plane.normal, aabb.half);
    if s - r > CMP_EPS {
        Side::Positive
    } else if s + r < -CMP_EPS {
        Side::Negative
    } else {
        Side::Intersecting
    }
}

#[cfg(test)]
mod tests {
    use super::{
        classify, projected_radius, signed_distance_center, v_dot, Aabb, Plane, Side, CMP_EPS,
    };

    /// Absolute tolerance for comparing two `f32` magnitudes in the tests.
    const TEST_EPS: f32 = 1.0e-4;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= TEST_EPS
    }

    /// A plane whose normal points along `+x` through the origin.
    fn plane_px() -> Plane {
        Plane::new([1.0, 0.0, 0.0], 0.0)
    }

    /// A unit-half-extent box centered at `c`.
    fn unit_box_at(c: [f32; 3]) -> Aabb {
        Aabb::new(c, [1.0, 1.0, 1.0])
    }

    #[test]
    fn box_fully_positive() {
        assert_eq!(
            classify(plane_px(), unit_box_at([3.0, 0.0, 0.0])),
            Side::Positive
        );
    }

    #[test]
    fn box_fully_negative() {
        assert_eq!(
            classify(plane_px(), unit_box_at([-3.0, 0.0, 0.0])),
            Side::Negative
        );
    }

    #[test]
    fn box_crossing_plane_is_intersecting() {
        assert_eq!(
            classify(plane_px(), unit_box_at([0.0, 0.0, 0.0])),
            Side::Intersecting
        );
    }

    #[test]
    fn box_touching_positive_boundary_is_intersecting() {
        // center at x = 1, half x = 1 => s - r = 0 exactly.
        assert_eq!(
            classify(plane_px(), unit_box_at([1.0, 0.0, 0.0])),
            Side::Intersecting
        );
    }

    #[test]
    fn box_touching_negative_boundary_is_intersecting() {
        // center at x = -1, half x = 1 => s + r = 0 exactly.
        assert_eq!(
            classify(plane_px(), unit_box_at([-1.0, 0.0, 0.0])),
            Side::Intersecting
        );
    }

    #[test]
    fn just_past_positive_boundary_is_positive() {
        // s - r = 0.01 > CMP_EPS.
        assert_eq!(
            classify(plane_px(), unit_box_at([1.01, 0.0, 0.0])),
            Side::Positive
        );
    }

    #[test]
    fn just_past_negative_boundary_is_negative() {
        assert_eq!(
            classify(plane_px(), unit_box_at([-1.01, 0.0, 0.0])),
            Side::Negative
        );
    }

    #[test]
    fn axis_plane_x_offset() {
        // Plane x = 5 (normal +x, d = -5). Box centered at x = 8, half 1.
        let plane = Plane::new([1.0, 0.0, 0.0], -5.0);
        assert_eq!(
            classify(plane, unit_box_at([8.0, 0.0, 0.0])),
            Side::Positive
        );
        assert_eq!(
            classify(plane, unit_box_at([2.0, 0.0, 0.0])),
            Side::Negative
        );
        assert_eq!(
            classify(plane, unit_box_at([5.0, 0.0, 0.0])),
            Side::Intersecting
        );
    }

    #[test]
    fn axis_plane_y() {
        let plane = Plane::new([0.0, 1.0, 0.0], 0.0);
        assert_eq!(
            classify(plane, unit_box_at([0.0, 3.0, 0.0])),
            Side::Positive
        );
        assert_eq!(
            classify(plane, unit_box_at([0.0, -3.0, 0.0])),
            Side::Negative
        );
        assert_eq!(
            classify(plane, unit_box_at([0.0, 0.0, 0.0])),
            Side::Intersecting
        );
    }

    #[test]
    fn axis_plane_z() {
        let plane = Plane::new([0.0, 0.0, 1.0], 0.0);
        assert_eq!(
            classify(plane, unit_box_at([0.0, 0.0, 4.0])),
            Side::Positive
        );
        assert_eq!(
            classify(plane, unit_box_at([0.0, 0.0, -4.0])),
            Side::Negative
        );
    }

    #[test]
    fn negative_axis_normal_x() {
        // Normal points along -x, so +x becomes the negative half-space.
        let plane = Plane::new([-1.0, 0.0, 0.0], 0.0);
        assert_eq!(
            classify(plane, unit_box_at([3.0, 0.0, 0.0])),
            Side::Negative
        );
        assert_eq!(
            classify(plane, unit_box_at([-3.0, 0.0, 0.0])),
            Side::Positive
        );
    }

    #[test]
    fn negative_axis_normal_y() {
        let plane = Plane::new([0.0, -1.0, 0.0], 0.0);
        assert_eq!(
            classify(plane, unit_box_at([0.0, 3.0, 0.0])),
            Side::Negative
        );
    }

    #[test]
    fn negative_axis_normal_z() {
        let plane = Plane::new([0.0, 0.0, -1.0], 0.0);
        assert_eq!(
            classify(plane, unit_box_at([0.0, 0.0, 3.0])),
            Side::Negative
        );
    }

    #[test]
    fn oblique_normal_positive() {
        // Normal (1,1,0) (not unit), box far along the +x/+y diagonal.
        let plane = Plane::new([1.0, 1.0, 0.0], 0.0);
        assert_eq!(
            classify(plane, unit_box_at([5.0, 5.0, 0.0])),
            Side::Positive
        );
    }

    #[test]
    fn oblique_normal_intersecting_at_origin() {
        let plane = Plane::new([1.0, 1.0, 0.0], 0.0);
        assert_eq!(
            classify(plane, unit_box_at([0.0, 0.0, 0.0])),
            Side::Intersecting
        );
    }

    #[test]
    fn oblique_normal_negative() {
        let plane = Plane::new([1.0, 1.0, 1.0], 0.0);
        assert_eq!(
            classify(plane, unit_box_at([-4.0, -4.0, -4.0])),
            Side::Negative
        );
    }

    #[test]
    fn box_center_exactly_on_plane_is_intersecting() {
        let plane = Plane::new([1.0, 2.0, 3.0], 0.0);
        // Center on the plane => s = 0, so it cannot be strictly on a side.
        assert_eq!(
            classify(plane, unit_box_at([0.0, 0.0, 0.0])),
            Side::Intersecting
        );
    }

    #[test]
    fn degenerate_thin_box_intersecting() {
        // Zero x half-extent: a slab lying across the plane through the origin.
        let plane = plane_px();
        let thin = Aabb::new([0.0, 0.0, 0.0], [0.0, 5.0, 5.0]);
        assert_eq!(classify(plane, thin), Side::Intersecting);
    }

    #[test]
    fn degenerate_thin_box_positive() {
        // Zero x half-extent placed strictly on the +x side.
        let plane = plane_px();
        let thin = Aabb::new([2.0, 0.0, 0.0], [0.0, 5.0, 5.0]);
        assert_eq!(classify(plane, thin), Side::Positive);
    }

    #[test]
    fn point_box_on_plane_is_intersecting() {
        // A zero-extent "point" box sitting exactly on the plane.
        let plane = plane_px();
        let point = Aabb::new([0.0, 4.0, -2.0], [0.0, 0.0, 0.0]);
        assert_eq!(classify(plane, point), Side::Intersecting);
    }

    #[test]
    fn point_box_off_plane_is_positive() {
        let plane = plane_px();
        let point = Aabb::new([2.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert_eq!(classify(plane, point), Side::Positive);
    }

    #[test]
    fn unnormalized_normal_matches_normalized_decision() {
        // Scaling normal and d by a common positive factor must not change the
        // side for any of a set of representative boxes.
        let base = Plane::new([1.0, 0.0, 0.0], -2.0);
        let scaled = Plane::new([3.0, 0.0, 0.0], -6.0);
        let boxes = [
            unit_box_at([5.0, 0.0, 0.0]),
            unit_box_at([-1.0, 0.0, 0.0]),
            unit_box_at([2.0, 0.0, 0.0]),
            unit_box_at([2.5, 0.0, 0.0]),
        ];
        for aabb in boxes {
            assert_eq!(classify(base, aabb), classify(scaled, aabb));
        }
    }

    #[test]
    fn unnormalized_oblique_normal_consistency() {
        let base = Plane::new([1.0, 1.0, 1.0], -3.0);
        let scaled = Plane::new([2.0, 2.0, 2.0], -6.0);
        let boxes = [
            unit_box_at([4.0, 4.0, 4.0]),
            unit_box_at([0.0, 0.0, 0.0]),
            unit_box_at([-2.0, -2.0, -2.0]),
        ];
        for aabb in boxes {
            assert_eq!(classify(base, aabb), classify(scaled, aabb));
        }
    }

    #[test]
    fn from_min_max_center_and_half() {
        let aabb = Aabb::from_min_max([0.0, 2.0, -4.0], [4.0, 6.0, 0.0]);
        assert!(approx(aabb.center[0], 2.0));
        assert!(approx(aabb.center[1], 4.0));
        assert!(approx(aabb.center[2], -2.0));
        assert!(approx(aabb.half[0], 2.0));
        assert!(approx(aabb.half[1], 2.0));
        assert!(approx(aabb.half[2], 2.0));
    }

    #[test]
    fn from_min_max_swapped_corners_stay_non_negative() {
        // Swapping min and max must still yield the same box with non-negative
        // half-extents.
        let a = Aabb::from_min_max([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]);
        let b = Aabb::from_min_max([4.0, 4.0, 4.0], [0.0, 0.0, 0.0]);
        assert!(approx(a.center[0], b.center[0]));
        assert!(b.half[0] >= 0.0);
        assert!(approx(a.half[0], b.half[0]));
    }

    #[test]
    fn from_min_max_classifies_like_center_half() {
        let plane = Plane::new([1.0, 0.0, 0.0], -1.0);
        let via_min_max = Aabb::from_min_max([2.0, -1.0, -1.0], [4.0, 1.0, 1.0]);
        let via_center = Aabb::new([3.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert_eq!(classify(plane, via_min_max), classify(plane, via_center));
        assert_eq!(classify(plane, via_min_max), Side::Positive);
    }

    #[test]
    fn signed_distance_center_value() {
        let plane = Plane::new([1.0, 2.0, 3.0], 4.0);
        let aabb = unit_box_at([1.0, 1.0, 1.0]);
        // s = 1*1 + 2*1 + 3*1 + 4 = 10.
        assert!(approx(signed_distance_center(plane, aabb), 10.0));
    }

    #[test]
    fn projected_radius_axis_aligned() {
        // Only the x lane contributes for an +x normal.
        assert!(approx(
            projected_radius([1.0, 0.0, 0.0], [2.0, 3.0, 4.0]),
            2.0
        ));
    }

    #[test]
    fn projected_radius_l1_weighted_sum() {
        // r = |1|*2 + |-2|*3 + |0.5|*4 = 2 + 6 + 2 = 10.
        assert!(approx(
            projected_radius([1.0, -2.0, 0.5], [2.0, 3.0, 4.0]),
            10.0
        ));
    }

    #[test]
    fn v_dot_value() {
        assert!(approx(v_dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0));
    }

    #[expect(
        clippy::assertions_on_constants,
        reason = "pins the comparison epsilon bounds as compile-time regression guards"
    )]
    #[test]
    fn cmp_eps_is_small_positive() {
        assert!(CMP_EPS > 0.0);
        assert!(CMP_EPS < 1.0e-3);
    }

    #[test]
    fn translation_along_plane_keeps_intersecting() {
        // Sliding a straddling box parallel to the +x plane keeps it straddling.
        let plane = plane_px();
        let base = unit_box_at([0.0, 0.0, 0.0]);
        let slid = unit_box_at([0.0, 100.0, -50.0]);
        assert_eq!(classify(plane, base), Side::Intersecting);
        assert_eq!(classify(plane, slid), Side::Intersecting);
    }
}
