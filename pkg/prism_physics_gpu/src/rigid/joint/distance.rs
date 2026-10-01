//! The distance (limit) rigid-body joint.
//!
//! A [`DistanceJoint`] constrains the world-space distance between a body-local
//! anchor on body `a` and a body-local anchor on body `b` to lie within a
//! closed range `[min_distance, max_distance]`. It generalises three classic
//! mechanisms with one definition:
//!
//! * **Rigid rod / fixed-length link** when `min_distance == max_distance`: the
//!   two anchors are held at exactly that separation, a bilateral constraint.
//! * **Rope / cable limit** when `min_distance == 0` and `max_distance > 0`:
//!   the anchors may approach freely but never separate past the maximum, a
//!   unilateral upper limit.
//! * **Minimum-separation strut** when `min_distance > 0`: the anchors may move
//!   apart freely but are pushed back out when they come closer than the
//!   minimum, a unilateral lower limit.
//!
//! Between the two bounds the joint is inactive and applies no correction: the
//! bodies move as if unconstrained. This dead-zone-with-one-sided-limits form is
//! the distance joint shared by `PhysX`, `Box2D`, and the distance degree of freedom
//! of a configurable `6`-DOF joint.
//!
//! The anchors are stored in each body's own local frame, so they rotate with
//! the body for free: the world-space anchor on body `a` is
//! `position_a + rotate(orientation_a, anchor_a)`, and likewise for `b`. The
//! quantity the solver limits is the length of their world-space separation,
//! `|p_a - p_b|`.
//!
//! # Compliance
//!
//! [`compliance`](DistanceJoint::compliance) is the inverse stiffness
//! (metres per newton) of the `XPBD` limit constraint. A compliance of zero is
//! the rigid limit — the solver drives the violated bound shut as hard as the
//! substep allows — while a positive compliance yields a soft, springy limit
//! whose stiffness is `1 / compliance`. The solver divides it by the squared
//! substep time (`alpha_tilde = compliance / h^2`) to form the
//! time-step-independent `XPBD` regularisation term, so the same compliance
//! behaves consistently across substep counts.
//!
//! Provenance: the point-to-point distance constraint and its `XPBD` positional
//! handling (Müller et al., "Detailed Rigid Body Simulation with XPBD"), with
//! the one-sided limit / dead-zone treatment standard to distance joints, over
//! the world-space inverse inertia and quaternion kinematics of Baraff &
//! Witkin. No Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use super::coloring::JointBodies;

/// A distance joint limiting the world-space separation of a body-local anchor
/// on body `a` and a body-local anchor on body `b` to the range
/// `[min_distance, max_distance]`.
///
/// Either body may be static (zero inverse mass and inertia); a joint between a
/// dynamic body and a static one tethers the dynamic body to a fixed world
/// point at the configured distance range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistanceJoint {
    /// Index of the first body in the [`RigidBodyState`](super::super::RigidBodyState).
    pub body_a: u32,
    /// Index of the second body in the [`RigidBodyState`](super::super::RigidBodyState).
    pub body_b: u32,
    /// Anchor point on body `a`, in body `a`'s local frame (relative to its
    /// centre of mass).
    pub anchor_a: Vec3,
    /// Anchor point on body `b`, in body `b`'s local frame (relative to its
    /// centre of mass).
    pub anchor_b: Vec3,
    /// Lower bound (metres) on the anchor separation. The joint pushes the
    /// anchors apart when they come closer than this. Must be non-negative and
    /// no greater than [`max_distance`](DistanceJoint::max_distance).
    pub min_distance: f32,
    /// Upper bound (metres) on the anchor separation. The joint pulls the
    /// anchors together when they separate past this. Must be no less than
    /// [`min_distance`](DistanceJoint::min_distance).
    pub max_distance: f32,
    /// Inverse stiffness (metres per newton) of the limit constraint. Zero is
    /// the rigid limit; a positive value yields a soft limit of stiffness
    /// `1 / compliance`.
    pub compliance: f32,
}

impl DistanceJoint {
    /// Creates a distance joint between `body_a` and `body_b` with the given
    /// body-local anchors, separation range, and compliance.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the range is malformed (`min_distance`
    /// negative or greater than `max_distance`); callers are expected to pass a
    /// well-ordered, non-negative range.
    #[must_use]
    pub fn new(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        min_distance: f32,
        max_distance: f32,
        compliance: f32,
    ) -> DistanceJoint {
        debug_assert!(
            min_distance >= 0.0 && min_distance <= max_distance,
            "distance joint range must satisfy 0 <= min_distance <= max_distance"
        );
        DistanceJoint {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            min_distance,
            max_distance,
            compliance,
        }
    }

    /// Creates a rigid fixed-length link: a bilateral constraint holding the two
    /// anchors at exactly `length` metres apart.
    #[must_use]
    pub fn rigid_rod(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        length: f32,
    ) -> DistanceJoint {
        DistanceJoint::new(body_a, body_b, anchor_a, anchor_b, length, length, 0.0)
    }

    /// Creates a rope / cable limit: the anchors may approach freely but never
    /// separate past `max_length` metres.
    #[must_use]
    pub fn rope(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        max_length: f32,
    ) -> DistanceJoint {
        DistanceJoint::new(body_a, body_b, anchor_a, anchor_b, 0.0, max_length, 0.0)
    }

    /// The two body indices this joint couples, in `(a, b)` order.
    #[must_use]
    pub fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }

    /// Packs the joint into its `GPU` storage-buffer representation.
    #[must_use]
    pub(crate) fn to_gpu(self) -> GpuDistanceJoint {
        GpuDistanceJoint {
            anchor_a: [self.anchor_a.x, self.anchor_a.y, self.anchor_a.z, 0.0],
            anchor_b: [self.anchor_b.x, self.anchor_b.y, self.anchor_b.z, 0.0],
            body_a: self.body_a,
            body_b: self.body_b,
            compliance: self.compliance,
            min_distance: self.min_distance,
            max_distance: self.max_distance,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }
}

impl JointBodies for DistanceJoint {
    fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }
}

/// Device-packed [`DistanceJoint`]; layout matches `Joint` in
/// `shaders/rigid_joint_distance.wgsl` (`64` bytes). The two anchors are padded
/// to `vec4` so each starts on the `16`-byte boundary the storage layout
/// requires; the trailing scalars fill the final two `16`-byte rows.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuDistanceJoint {
    /// Body-local anchor on body `a` in `xyz`; `w` unused.
    anchor_a: [f32; 4],
    /// Body-local anchor on body `b` in `xyz`; `w` unused.
    anchor_b: [f32; 4],
    /// Index of body `a`.
    body_a: u32,
    /// Index of body `b`.
    body_b: u32,
    /// Inverse stiffness (metres per newton).
    compliance: f32,
    /// Lower bound (metres) on the anchor separation.
    min_distance: f32,
    /// Upper bound (metres) on the anchor separation.
    max_distance: f32,
    /// Padding to a `64`-byte multiple of the `16`-byte anchor alignment.
    _pad0: f32,
    /// Padding lane.
    _pad1: f32,
    /// Padding lane.
    _pad2: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_packing_round_trips_fields() {
        let joint = DistanceJoint::new(
            3,
            7,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-4.0, 5.0, -6.0),
            0.5,
            1.5,
            0.001,
        );
        let gpu = joint.to_gpu();
        assert_eq!(gpu.body_a, 3);
        assert_eq!(gpu.body_b, 7);
        assert_eq!(gpu.anchor_a, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(gpu.anchor_b, [-4.0, 5.0, -6.0, 0.0]);
        assert_eq!(gpu.compliance, 0.001);
        assert_eq!(gpu.min_distance, 0.5);
        assert_eq!(gpu.max_distance, 1.5);
    }

    #[test]
    fn gpu_struct_is_64_bytes() {
        assert_eq!(size_of::<GpuDistanceJoint>(), 64);
    }

    #[test]
    fn bodies_returns_pair_in_order() {
        let joint = DistanceJoint::new(5, 2, Vec3::ZERO, Vec3::ZERO, 0.0, 1.0, 0.0);
        assert_eq!(joint.bodies(), (5, 2));
    }

    #[test]
    fn rigid_rod_sets_equal_bounds_and_zero_compliance() {
        let joint = DistanceJoint::rigid_rod(0, 1, Vec3::ZERO, Vec3::ZERO, 2.0);
        assert_eq!(joint.min_distance, 2.0);
        assert_eq!(joint.max_distance, 2.0);
        assert_eq!(joint.compliance, 0.0);
    }

    #[test]
    fn rope_sets_zero_floor_and_finite_ceiling() {
        let joint = DistanceJoint::rope(0, 1, Vec3::ZERO, Vec3::ZERO, 3.0);
        assert_eq!(joint.min_distance, 0.0);
        assert_eq!(joint.max_distance, 3.0);
        assert_eq!(joint.compliance, 0.0);
    }
}
