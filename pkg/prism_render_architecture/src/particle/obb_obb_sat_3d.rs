//! 3D oriented-bounding-box vs oriented-bounding-box boolean intersection via
//! the Separating Axis Theorem (`SAT`), the fifteen-axis test that production
//! collision libraries use for `OBB`-`OBB` overlap queries (design §10, §13).
//!
//! Several particle stages need a cheap, exact yes/no answer to "do these two
//! oriented boxes overlap?": an emitter-volume-vs-emitter-volume broadphase, a
//! bounds-cluster overlap merge, and a light-proxy-vs-particle-bounds cull. This
//! module owns the small, `CPU`-verifiable contract they share: given two
//! oriented bounding boxes — each a center, three orthonormal axes, and three
//! half-extents — it decides whether the boxes intersect.
//!
//! # Algorithm
//! Two convex polytopes are disjoint if and only if some axis separates their
//! projections. For a pair of boxes it suffices to test fifteen candidate axes:
//! the three face normals of box `a`, the three face normals of box `b`, and the
//! nine pairwise cross products of one edge direction from each box. On every
//! candidate axis both boxes project to a symmetric interval whose half-width is
//! the sum of each half-extent times the absolute dot of that box axis with the
//! candidate axis; the centers project to a signed offset whose magnitude is
//! compared against the summed half-widths. If any axis leaves a positive gap
//! the boxes are disjoint; if no axis separates them they intersect. When two
//! edges are parallel their cross product is (near) zero and cannot separate the
//! boxes on its own — that direction is already covered by the face-normal
//! axes — so such degenerate axes are skipped rather than normalized. This
//! matches the standard Gottschalk `OBBTree` `SAT` formulation used across
//! real-time collision detection, reimplemented here without reusing any code.
//!
//! # Strict scope
//! This module performs *only* a 3D `OBB`-`OBB` boolean intersection test. It is
//! not the 2D convex-polygon overlap test ([`super::sat_collision_2d`], which
//! also returns a minimum translation vector for 2D shapes); it is not the
//! particle-vs-environment collision solver that produces push-out, restitution,
//! and friction ([`super::collision`]); and it is not the decal projection
//! volume that maps surface points into a box's local `UV` space
//! ([`super::decal`]). It reports neither a penetration depth nor a separation
//! vector — a single boolean is the entire contract. It keeps its own [`Vec3`]
//! and never imports those sibling contracts.
//!
//! # No transcendental math
//! Projection radii and center offsets are pure `+`, `-`, `*` arithmetic with
//! `abs`; the only irrational operation is the `f32::sqrt` used to normalize a
//! non-degenerate edge-cross axis. There is no `sin`, `cos`, `atan`, `exp`,
//! `ln`, `powf`, `ceil`, `round`, or any other transcendental or rounding call,
//! and no `f32` equality: near-zero and near-touching magnitudes are compared
//! against the module epsilons.

/// Squared-length threshold below which an edge-cross axis is treated as
/// degenerate (the two edges are parallel) and skipped. Compared against a
/// squared length, so it is the square of a `~1e-4` direction tolerance.
pub const DEGENERATE_AXIS_EPS: f32 = 1.0e-8;

/// Separation slack added to the summed projection radii. A pair is reported
/// disjoint only when the center offset exceeds the radius sum by more than this
/// tolerance, so an exact face/edge contact counts as an intersection rather
/// than flickering between states from rounding noise.
pub const CONTACT_EPS: f32 = 1.0e-5;

/// A minimal 3D vector with hand-rolled arithmetic; this module deliberately
/// does not depend on any shared vector type so the contract stays a leaf.
#[derive(Clone, Copy, Debug)]
pub struct Vec3 {
    /// Cartesian x component.
    pub x: f32,
    /// Cartesian y component.
    pub y: f32,
    /// Cartesian z component.
    pub z: f32,
}

impl Vec3 {
    /// Builds a vector from its three components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Component-wise sum.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API is specified with named add/sub/neg methods for call-site uniformity, matching the sibling particle contracts; operator traits are intentionally not part of this internal type."
    )]
    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    /// Component-wise difference (`self - rhs`).
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API is specified with named add/sub/neg methods for call-site uniformity, matching the sibling particle contracts; operator traits are intentionally not part of this internal type."
    )]
    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }

    /// Uniform scale by a scalar.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API is specified with named add/sub/neg methods for call-site uniformity, matching the sibling particle contracts; operator traits are intentionally not part of this internal type."
    )]
    pub fn mul(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    /// Negation (`-self`).
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API is specified with named add/sub/neg methods for call-site uniformity, matching the sibling particle contracts; operator traits are intentionally not part of this internal type."
    )]
    pub fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }

    /// Euclidean dot product.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Right-handed cross product (`self × rhs`).
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    /// Squared length; avoids the `sqrt` when only comparisons are needed.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Unit-length copy. Callers guarantee the vector is longer than
    /// [`DEGENERATE_AXIS_EPS`] (squared); a shorter vector is returned unchanged
    /// so no division by zero and no `NaN` can escape.
    #[must_use]
    pub fn normalized(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq <= DEGENERATE_AXIS_EPS {
            return self;
        }
        let inv = 1.0 / len_sq.sqrt();
        self.mul(inv)
    }
}

/// An oriented bounding box: a center, three orthonormal local axes, and the
/// three half-extents measured along those axes.
#[derive(Clone, Copy, Debug)]
pub struct Obb {
    /// World-space center of the box.
    pub center: Vec3,
    /// Local x axis (expected unit length and orthogonal to the others).
    pub axis_x: Vec3,
    /// Local y axis (expected unit length and orthogonal to the others).
    pub axis_y: Vec3,
    /// Local z axis (expected unit length and orthogonal to the others).
    pub axis_z: Vec3,
    /// Half-extent along [`Obb::axis_x`]; expected non-negative.
    pub half_x: f32,
    /// Half-extent along [`Obb::axis_y`]; expected non-negative.
    pub half_y: f32,
    /// Half-extent along [`Obb::axis_z`]; expected non-negative.
    pub half_z: f32,
}

impl Obb {
    /// Builds an oriented box from its center, three axes, and half-extents.
    #[must_use]
    pub const fn new(
        center: Vec3,
        axis_x: Vec3,
        axis_y: Vec3,
        axis_z: Vec3,
        half_x: f32,
        half_y: f32,
        half_z: f32,
    ) -> Self {
        Self {
            center,
            axis_x,
            axis_y,
            axis_z,
            half_x,
            half_y,
            half_z,
        }
    }

    /// Convenience constructor for an axis-aligned box (world basis).
    #[must_use]
    pub const fn axis_aligned(center: Vec3, half_x: f32, half_y: f32, half_z: f32) -> Self {
        Self::new(
            center,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            half_x,
            half_y,
            half_z,
        )
    }

    /// The three local axes as an array, in x/y/z order.
    #[must_use]
    pub fn axes(&self) -> [Vec3; 3] {
        [self.axis_x, self.axis_y, self.axis_z]
    }

    /// The three half-extents as an array, matching [`Obb::axes`] order.
    #[must_use]
    pub fn half_extents(&self) -> [f32; 3] {
        [self.half_x, self.half_y, self.half_z]
    }

    /// Half-width of this box's projection onto the given (unit) `axis`: the sum
    /// of each half-extent times the absolute alignment of its axis with `axis`.
    #[must_use]
    pub fn projected_radius(&self, axis: Vec3) -> f32 {
        let axes = self.axes();
        let halves = self.half_extents();
        let mut radius = 0.0_f32;
        for (edge, half) in axes.iter().zip(halves.iter()) {
            radius += half * edge.dot(axis).abs();
        }
        radius
    }
}

/// Returns `true` when the two boxes have a positive gap along `axis`, i.e.
/// `axis` is a separating axis. `axis` must be unit length for the
/// [`CONTACT_EPS`] slack to carry a consistent world-space meaning. A separating
/// axis proves the boxes are disjoint.
#[must_use]
pub fn separated_on_axis(a: &Obb, b: &Obb, axis: Vec3) -> bool {
    let center_offset = b.center.sub(a.center).dot(axis).abs();
    let radius_sum = a.projected_radius(axis) + b.projected_radius(axis);
    center_offset > radius_sum + CONTACT_EPS
}

/// Tests whether two oriented bounding boxes intersect.
///
/// Returns `true` when the boxes overlap or exactly touch, and `false` when a
/// separating axis exists among the fifteen `SAT` candidates (three face normals
/// per box plus nine edge-cross axes). Parallel edge pairs yield a degenerate
/// (near-zero) cross product that cannot separate the boxes on its own and are
/// skipped; that direction is already covered by the face-normal axes.
#[must_use]
pub fn intersects(a: &Obb, b: &Obb) -> bool {
    let a_axes = a.axes();
    let b_axes = b.axes();

    // Six face-normal axes: three from each box. These are already unit length.
    for face in a_axes.iter().chain(b_axes.iter()) {
        if separated_on_axis(a, b, *face) {
            return false;
        }
    }

    // Nine edge-cross axes: one edge direction from each box.
    for edge_a in a_axes.iter() {
        for edge_b in b_axes.iter() {
            let cross = edge_a.cross(*edge_b);
            if cross.length_squared() <= DEGENERATE_AXIS_EPS {
                // Parallel edges: this axis is degenerate and is covered by the
                // face normals above, so skip it rather than dividing by ~0.
                continue;
            }
            if separated_on_axis(a, b, cross.normalized()) {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tight tolerance for asserting on scalar geometry in the tests.
    const TEST_EPS: f32 = 1.0e-4;

    /// A 45-degree sine/cosine literal (no transcendental call at runtime).
    const HALF_SQRT2: f32 = 0.707_106_77;

    fn unit_cube_at(center: Vec3) -> Obb {
        Obb::axis_aligned(center, 1.0, 1.0, 1.0)
    }

    /// Box rotated by 45 degrees about the world z axis.
    fn rot_z_45(center: Vec3, half_x: f32, half_y: f32, half_z: f32) -> Obb {
        Obb::new(
            center,
            Vec3::new(HALF_SQRT2, HALF_SQRT2, 0.0),
            Vec3::new(-HALF_SQRT2, HALF_SQRT2, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            half_x,
            half_y,
            half_z,
        )
    }

    /// Box rotated 45 degrees about z then 45 degrees about the world x axis;
    /// yields a fully non-axis-aligned orthonormal triad (verified in a test).
    fn rot_zx_45(center: Vec3, half_x: f32, half_y: f32, half_z: f32) -> Obb {
        Obb::new(
            center,
            Vec3::new(HALF_SQRT2, 0.5, 0.5),
            Vec3::new(-HALF_SQRT2, 0.5, 0.5),
            Vec3::new(0.0, -HALF_SQRT2, HALF_SQRT2),
            half_x,
            half_y,
            half_z,
        )
    }

    #[test]
    fn axis_aligned_overlap_at_origin() {
        let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
        let b = unit_cube_at(Vec3::new(0.5, 0.0, 0.0));
        assert!(intersects(&a, &b));
    }

    #[test]
    fn identical_boxes_intersect() {
        let a = unit_cube_at(Vec3::new(1.0, -2.0, 3.0));
        assert!(intersects(&a, &a));
    }

    #[test]
    fn separated_along_x() {
        let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
        let b = unit_cube_at(Vec3::new(3.0, 0.0, 0.0));
        assert!(!intersects(&a, &b));
    }

    #[test]
    fn separated_along_y() {
        let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
        let b = unit_cube_at(Vec3::new(0.0, 5.0, 0.0));
        assert!(!intersects(&a, &b));
    }

    #[test]
    fn separated_along_z() {
        let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
        let b = unit_cube_at(Vec3::new(0.0, 0.0, 2.01));
        assert!(!intersects(&a, &b));
    }

    #[test]
    fn face_contact_touching_counts_as_intersection() {
        // Centers exactly 2 apart on x, each half-extent 1: faces just touch.
        let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
        let b = unit_cube_at(Vec3::new(2.0, 0.0, 0.0));
        assert!(intersects(&a, &b));
    }

    #[test]
    fn tiny_gap_is_separated() {
        let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
        let b = unit_cube_at(Vec3::new(2.01, 0.0, 0.0));
        assert!(!intersects(&a, &b));
    }

    #[test]
    fn containment_small_inside_large() {
        let big = Obb::axis_aligned(Vec3::new(0.0, 0.0, 0.0), 5.0, 5.0, 5.0);
        let small = Obb::axis_aligned(Vec3::new(1.0, -1.0, 0.5), 0.25, 0.25, 0.25);
        assert!(intersects(&big, &small));
        assert!(intersects(&small, &big));
    }

    #[test]
    fn diagonal_far_separation() {
        let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
        let b = unit_cube_at(Vec3::new(4.0, 4.0, 4.0));
        assert!(!intersects(&a, &b));
    }

    #[test]
    fn corner_overlap() {
        let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
        // Overlaps a small amount at the +++ corner.
        let b = unit_cube_at(Vec3::new(1.9, 1.9, 1.9));
        assert!(intersects(&a, &b));
    }

    #[test]
    fn rotated_z_45_overlap() {
        let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
        let b = rot_z_45(Vec3::new(1.0, 0.0, 0.0), 1.0, 1.0, 1.0);
        assert!(intersects(&a, &b));
    }

    #[test]
    fn rotated_z_45_separated_by_a_face_normal() {
        // A rotated unit cube's projection onto the world x axis has half-width
        // sqrt(2) ~= 1.414. Placing centers 3 apart on x leaves a clear gap that
        // is caught by box a's own x face normal.
        let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
        let b = rot_z_45(Vec3::new(3.0, 0.0, 0.0), 1.0, 1.0, 1.0);
        assert!(!intersects(&a, &b));
        // Confirm the world x axis (a's face normal) is the separating axis.
        assert!(separated_on_axis(&a, &b, Vec3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn separation_reported_by_box_b_face_normal() {
        // b is rotated about z; its local x axis is (HALF_SQRT2, HALF_SQRT2, 0).
        // Offset the pair along that axis far enough that only b's face normal
        // (not a's world axes) yields the gap.
        let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
        let dir = Vec3::new(HALF_SQRT2, HALF_SQRT2, 0.0);
        let center = dir.mul(3.0);
        let b = rot_z_45(center, 1.0, 1.0, 1.0);
        assert!(!intersects(&a, &b));
        assert!(separated_on_axis(&a, &b, dir));
    }

    #[test]
    fn projected_radius_of_axis_aligned_cube() {
        let cube = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
        let r = cube.projected_radius(Vec3::new(1.0, 0.0, 0.0));
        assert!((r - 1.0).abs() < TEST_EPS);
    }

    #[test]
    fn projected_radius_of_rotated_cube_on_world_x() {
        // A z-rotated unit cube projects onto world x with half-width sqrt(2).
        let cube = rot_z_45(Vec3::new(0.0, 0.0, 0.0), 1.0, 1.0, 1.0);
        let r = cube.projected_radius(Vec3::new(1.0, 0.0, 0.0));
        assert!((r - 2.0_f32.sqrt()).abs() < TEST_EPS);
    }

    #[test]
    fn rotated_triad_is_orthonormal() {
        let b = rot_zx_45(Vec3::new(0.0, 0.0, 0.0), 1.0, 1.0, 1.0);
        let axes = b.axes();
        for edge in axes.iter() {
            assert!((edge.length_squared() - 1.0).abs() < TEST_EPS);
        }
        assert!(axes[0].dot(axes[1]).abs() < TEST_EPS);
        assert!(axes[0].dot(axes[2]).abs() < TEST_EPS);
        assert!(axes[1].dot(axes[2]).abs() < TEST_EPS);
    }

    #[test]
    fn edge_cross_axis_separates_skewed_boxes() {
        // Classic edge-edge ("skew insert") configuration: an axis-aligned unit
        // cube and a fully non-axis-aligned unit cube positioned so that none of
        // the six face normals separate them, yet an edge-cross axis does. The
        // separating axis here is a.axis_z x b.axis_y (found by SAT), which is
        // oblique to every face of both boxes.
        let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
        let b = rot_zx_45(Vec3::new(-2.4, -2.7, 0.7), 1.0, 1.0, 1.0);

        // No face normal of either box separates the pair.
        for face in a.axes().iter().chain(b.axes().iter()) {
            assert!(
                !separated_on_axis(&a, &b, *face),
                "unexpected face-normal separation"
            );
        }

        // The boxes are nonetheless disjoint: an edge-cross axis separates them.
        assert!(!intersects(&a, &b));

        // Explicitly exhibit the separating edge-cross axis.
        let cross = a.axis_z.cross(b.axis_y).normalized();
        assert!(separated_on_axis(&a, &b, cross));
    }

    #[test]
    fn skewed_boxes_pulled_together_intersect() {
        // Same skew orientation but the second box slid toward the first along
        // the same line: now no axis separates them and they intersect.
        let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
        let b = rot_zx_45(Vec3::new(-1.68, -1.89, 0.49), 1.0, 1.0, 1.0);
        assert!(intersects(&a, &b));
    }

    #[test]
    fn parallel_boxes_axis_swapped_still_correct() {
        // b uses the same axes as a but permuted (x<->y), so several edge crosses
        // are exactly parallel and must be skipped as degenerate.
        let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
        let b = Obb::new(
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
            1.0,
            1.0,
        );
        assert!(intersects(&a, &b));
    }

    #[test]
    fn parallel_boxes_separated_still_correct() {
        let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
        let b = Obb::new(
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
            1.0,
            1.0,
        );
        assert!(!intersects(&a, &b));
    }

    #[test]
    fn degenerate_cross_skipped_shared_z_axis() {
        // Both boxes share the world z axis (rotated only in xy). All A_i x B_j
        // crosses involving the two z axes are degenerate and must be skipped.
        let a = rot_z_45(Vec3::new(0.0, 0.0, 0.0), 1.0, 1.0, 3.0);
        let b = unit_cube_at(Vec3::new(0.8, 0.0, 0.0));
        assert!(intersects(&a, &b));
    }

    #[test]
    fn symmetry_on_overlap() {
        let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
        let b = rot_z_45(Vec3::new(1.2, 0.3, 0.0), 1.0, 1.0, 1.0);
        assert_eq!(intersects(&a, &b), intersects(&b, &a));
        assert!(intersects(&a, &b));
    }

    #[test]
    fn symmetry_on_separation() {
        let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
        let b = rot_zx_45(Vec3::new(-2.4, -2.7, 0.7), 1.0, 1.0, 1.0);
        assert_eq!(intersects(&a, &b), intersects(&b, &a));
        assert!(!intersects(&a, &b));
    }

    #[test]
    fn translation_invariance_overlap() {
        let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
        let b = rot_z_45(Vec3::new(1.0, 0.0, 0.0), 1.0, 1.0, 1.0);
        let shift = Vec3::new(-7.0, 12.0, 4.0);
        let a2 = Obb::new(
            a.center.add(shift),
            a.axis_x,
            a.axis_y,
            a.axis_z,
            a.half_x,
            a.half_y,
            a.half_z,
        );
        let b2 = Obb::new(
            b.center.add(shift),
            b.axis_x,
            b.axis_y,
            b.axis_z,
            b.half_x,
            b.half_y,
            b.half_z,
        );
        assert_eq!(intersects(&a, &b), intersects(&a2, &b2));
        assert!(intersects(&a2, &b2));
    }

    #[test]
    fn translation_invariance_separation() {
        let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
        let b = unit_cube_at(Vec3::new(3.0, 0.0, 0.0));
        let shift = Vec3::new(2.0, -3.5, 9.0);
        let a2 = Obb::axis_aligned(a.center.add(shift), 1.0, 1.0, 1.0);
        let b2 = Obb::axis_aligned(b.center.add(shift), 1.0, 1.0, 1.0);
        assert_eq!(intersects(&a, &b), intersects(&a2, &b2));
        assert!(!intersects(&a2, &b2));
    }

    #[test]
    fn rotation_invariance_both_boxes() {
        // Rotating both boxes by the same rotation (here rot_z_45 applied to the
        // axes) must not change the intersection result.
        let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
        let b = unit_cube_at(Vec3::new(1.5, 0.0, 0.0));
        let overlap_before = intersects(&a, &b);

        let a_rot = rot_z_45(Vec3::new(0.0, 0.0, 0.0), 1.0, 1.0, 1.0);
        // Rotate b's center by the same 45-degree z rotation and give it the
        // same rotated basis.
        let rc = Vec3::new(1.5 * HALF_SQRT2, 1.5 * HALF_SQRT2, 0.0);
        let b_rot = rot_z_45(rc, 1.0, 1.0, 1.0);
        assert_eq!(overlap_before, intersects(&a_rot, &b_rot));
        assert!(intersects(&a_rot, &b_rot));
    }

    #[test]
    fn rotation_invariance_separation() {
        let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
        let b = unit_cube_at(Vec3::new(3.0, 0.0, 0.0));
        let sep_before = intersects(&a, &b);

        let a_rot = rot_z_45(Vec3::new(0.0, 0.0, 0.0), 1.0, 1.0, 1.0);
        let rc = Vec3::new(3.0 * HALF_SQRT2, 3.0 * HALF_SQRT2, 0.0);
        let b_rot = rot_z_45(rc, 1.0, 1.0, 1.0);
        assert_eq!(sep_before, intersects(&a_rot, &b_rot));
        assert!(!intersects(&a_rot, &b_rot));
    }

    #[test]
    fn thin_slab_overlap() {
        let a = Obb::axis_aligned(Vec3::new(0.0, 0.0, 0.0), 5.0, 5.0, 0.05);
        let b = Obb::axis_aligned(Vec3::new(0.0, 0.0, 0.04), 0.5, 0.5, 0.05);
        assert!(intersects(&a, &b));
    }

    #[test]
    fn thin_slab_separated() {
        let a = Obb::axis_aligned(Vec3::new(0.0, 0.0, 0.0), 5.0, 5.0, 0.05);
        let b = Obb::axis_aligned(Vec3::new(0.0, 0.0, 0.5), 0.5, 0.5, 0.05);
        assert!(!intersects(&a, &b));
    }

    #[test]
    fn rotated_about_x_overlap() {
        let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
        let b = Obb::new(
            Vec3::new(0.0, 1.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, HALF_SQRT2, HALF_SQRT2),
            Vec3::new(0.0, -HALF_SQRT2, HALF_SQRT2),
            1.0,
            1.0,
            1.0,
        );
        assert!(intersects(&a, &b));
    }

    #[test]
    fn rotated_about_y_separated() {
        let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
        let b = Obb::new(
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(HALF_SQRT2, 0.0, -HALF_SQRT2),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(HALF_SQRT2, 0.0, HALF_SQRT2),
            1.0,
            1.0,
            1.0,
        );
        assert!(!intersects(&a, &b));
    }

    #[test]
    fn negate_vector_flips_offset_direction() {
        let v = Vec3::new(1.0, -2.0, 3.0);
        let n = v.neg();
        assert!((n.x + 1.0).abs() < TEST_EPS);
        assert!((n.y - 2.0).abs() < TEST_EPS);
        assert!((n.z + 3.0).abs() < TEST_EPS);
    }

    #[test]
    fn degenerate_axis_normalized_is_unchanged() {
        let tiny = Vec3::new(0.0, 0.0, 0.0);
        let out = tiny.normalized();
        assert!(out.length_squared() < TEST_EPS);
    }

    #[test]
    fn asymmetric_half_extents_reach_across() {
        // A very long box on x reaches a distant small box; the elongation, not
        // proximity, is what makes them intersect.
        let long = Obb::axis_aligned(Vec3::new(0.0, 0.0, 0.0), 10.0, 0.5, 0.5);
        let small = Obb::axis_aligned(Vec3::new(9.5, 0.0, 0.0), 0.5, 0.5, 0.5);
        assert!(intersects(&long, &small));
        let far = Obb::axis_aligned(Vec3::new(11.5, 0.0, 0.0), 0.5, 0.5, 0.5);
        assert!(!intersects(&long, &far));
    }
}
