//! The fixed (weld) rigid-body joint.
//!
//! A [`FixedJoint`] rigidly welds body `b` to body `a`: it removes **all six**
//! relative degrees of freedom, holding both the relative position and the
//! relative orientation of the pair at a fixed rest offset. It is the limiting
//! case shared by every other joint in this module — the
//! [`SphericalJoint`](super::SphericalJoint) with its orientation also locked,
//! the [`RevoluteJoint`](super::RevoluteJoint) with its hinge axis frozen, and
//! the [`PrismaticJoint`](super::PrismaticJoint) with its slide clamped. It is
//! the structural weld, the bolted bracket, and the compound rigid body
//! assembled from parts.
//!
//! # The two constraints
//!
//! * **Angular lock** (angular): the relative orientation of body `b` in body
//!   `a`'s frame is driven to the fixed
//!   [`rest_rotation`](FixedJoint::rest_rotation). The target world orientation
//!   of body `b` is `q_a * rest_rotation`, and the world-space error rotation
//!   `target * conj(q_b)` is cancelled, locking all three relative rotational
//!   degrees of freedom.
//! * **Point-to-point weld** (positional): the full world-space anchor
//!   separation `dx = p_a - p_b` (with `p = position + rotate(orientation, anchor)`)
//!   is driven to zero, locking all three relative translational degrees of
//!   freedom. Unlike the prismatic joint there is no free direction: the entire
//!   separation vector is cancelled.
//!
//! The angular lock is projected first each sweep, then the positional weld, so
//! the anchors are pulled together against an already-oriented frame.
//!
//! # Compliance
//!
//! [`compliance`](FixedJoint::compliance) is the inverse stiffness of the
//! positional weld (metres per newton) and
//! [`angular_compliance`](FixedJoint::angular_compliance) the inverse stiffness
//! of the angular lock (radians per newton-metre). Zero is the rigid limit for
//! each; a positive value yields a soft, springy weld of stiffness
//! `1 / compliance`. Each is divided by the squared substep time to form the
//! time-step-independent `XPBD` regularisation term, so the same compliance
//! behaves consistently across substep counts.
//!
//! Provenance: the point-to-point (ball-socket) constraint and the
//! relative-orientation lock with their substep `XPBD` handling (Müller et al.,
//! "Detailed Rigid Body Simulation with XPBD"), over the world-space inverse
//! inertia and quaternion kinematics of Baraff & Witkin. No Unreal Engine source
//! or derived code.

use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};

use super::coloring::JointBodies;

/// A fixed (weld) joint rigidly binding body `b` to body `a`, holding both the
/// relative position of two body-local anchors and the relative orientation at a
/// fixed rest offset.
///
/// The joint constrains the two world-space anchor points to coincide *and* the
/// relative orientation to equal a fixed rest offset, removing all six relative
/// degrees of freedom. Either body may be static (zero inverse mass and inverse
/// inertia); a joint between a dynamic body and a static one pins the dynamic
/// body rigidly to a fixed world pose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedJoint {
    /// Index of the first body in the [`RigidBodyState`](super::super::RigidBodyState).
    pub body_a: u32,
    /// Index of the second body in the [`RigidBodyState`](super::super::RigidBodyState).
    pub body_b: u32,
    /// Anchor point on body `a`, in body `a`'s local frame (relative to its
    /// centre of mass).
    pub anchor_a: Vec3,
    /// Anchor point on body `b`, in body `b`'s local frame.
    pub anchor_b: Vec3,
    /// Rest relative orientation of body `b` in body `a`'s frame — the fixed
    /// offset the angular lock holds, typically `conj(q_a0) * q_b0` captured at
    /// construction so the pair starts un-stressed.
    pub rest_rotation: Quat,
    /// Inverse stiffness (metres per newton) of the positional weld. Zero is the
    /// rigid limit.
    pub compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the angular lock. Zero is
    /// the rigid limit.
    pub angular_compliance: f32,
}

impl FixedJoint {
    /// Creates a fixed joint between `body_a` and `body_b` with the given
    /// body-local anchors, rest relative orientation, and compliances.
    #[must_use]
    pub fn new(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        rest_rotation: Quat,
        compliance: f32,
        angular_compliance: f32,
    ) -> FixedJoint {
        FixedJoint {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            rest_rotation,
            compliance,
            angular_compliance,
        }
    }

    /// The two body indices this joint couples, in `(a, b)` order.
    #[must_use]
    pub fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }

    /// Packs the joint into its `GPU` storage-buffer representation.
    #[must_use]
    pub(crate) fn to_gpu(self) -> GpuFixedJoint {
        GpuFixedJoint {
            anchor_a: [self.anchor_a.x, self.anchor_a.y, self.anchor_a.z, 0.0],
            anchor_b: [self.anchor_b.x, self.anchor_b.y, self.anchor_b.z, 0.0],
            rest_rotation: [
                self.rest_rotation.x,
                self.rest_rotation.y,
                self.rest_rotation.z,
                self.rest_rotation.w,
            ],
            body_a: self.body_a,
            body_b: self.body_b,
            compliance: self.compliance,
            angular_compliance: self.angular_compliance,
        }
    }
}

impl JointBodies for FixedJoint {
    fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }
}

/// Device-packed [`FixedJoint`]; layout matches `Joint` in
/// `shaders/rigid_joint_fixed.wgsl` (`64` bytes). The two anchors are padded to
/// `vec4` so each starts on the `16`-byte boundary the storage layout requires
/// (the two `w` lanes are unused); the rest-rotation quaternion fills a third
/// `vec4` as `(x, y, z, w)`. The two indices and two compliances fill the
/// trailing `16`-byte block.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuFixedJoint {
    /// Body-local anchor on body `a` in `xyz`; `w` unused.
    anchor_a: [f32; 4],
    /// Body-local anchor on body `b` in `xyz`; `w` unused.
    anchor_b: [f32; 4],
    /// Rest relative orientation of body `b` in body `a`'s frame as `(x, y, z, w)`.
    rest_rotation: [f32; 4],
    /// Index of body `a`.
    body_a: u32,
    /// Index of body `b`.
    body_b: u32,
    /// Inverse stiffness (metres per newton) of the positional weld.
    compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the angular lock.
    angular_compliance: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_packing_round_trips_fields() {
        let joint = FixedJoint::new(
            3,
            8,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-4.0, 5.0, -6.0),
            Quat::from_xyzw(0.1, 0.2, 0.3, 0.4),
            0.002,
            0.001,
        );
        let gpu = joint.to_gpu();
        assert_eq!(gpu.body_a, 3);
        assert_eq!(gpu.body_b, 8);
        assert_eq!(gpu.anchor_a, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(gpu.anchor_b, [-4.0, 5.0, -6.0, 0.0]);
        assert_eq!(gpu.rest_rotation, [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(gpu.compliance, 0.002);
        assert_eq!(gpu.angular_compliance, 0.001);
    }

    #[test]
    fn gpu_struct_is_64_bytes() {
        assert_eq!(size_of::<GpuFixedJoint>(), 64);
    }

    #[test]
    fn bodies_returns_pair_in_order() {
        let joint = FixedJoint::new(6, 1, Vec3::ZERO, Vec3::ZERO, Quat::IDENTITY, 0.0, 0.0);
        assert_eq!(joint.bodies(), (6, 1));
        assert_eq!(JointBodies::bodies(&joint), (6, 1));
    }
}
