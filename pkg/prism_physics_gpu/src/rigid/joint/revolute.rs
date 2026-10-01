//! The revolute (hinge) rigid-body joint.
//!
//! A [`RevoluteJoint`] is the ball-socket positional weld of a
//! [`SphericalJoint`](super::SphericalJoint) with one angular restriction added
//! on top: it pins a body-local anchor on body `a` to a body-local anchor on
//! body `b` *and* forces a body-local hinge axis on body `a` to stay parallel to
//! a body-local hinge axis on body `b`. Together these remove the three
//! translational and two of the three rotational degrees of freedom, leaving
//! exactly one free rotation — the spin about the shared hinge axis. It is the
//! door hinge, the elbow, and the wheel bearing of an articulated mechanism.
//!
//! # The two constraints
//!
//! * **Point-to-point** (positional): the same world-space anchor coincidence
//!   the spherical joint drives to zero, `p_a - p_b -> 0`, where
//!   `p = position + rotate(orientation, anchor)`.
//! * **Axis alignment** (angular): the world-space hinge axes
//!   `u_a = rotate(orientation_a, axis_a)` and `u_b = rotate(orientation_b,
//!   axis_b)` are driven parallel by cancelling their cross product
//!   `u_a x u_b -> 0`, which locks the two rotational degrees of freedom
//!   perpendicular to the hinge while leaving the spin about it free.
//!
//! The angular restriction is projected first each sweep, then the positional
//! weld, so the hinge axis is realigned before the anchors are pulled together.
//!
//! # Compliance
//!
//! [`compliance`](RevoluteJoint::compliance) is the inverse stiffness of the
//! positional weld (metres per newton) and
//! [`angular_compliance`](RevoluteJoint::angular_compliance) the inverse
//! stiffness of the axis-alignment constraint (radians per newton-metre). Zero
//! is the rigid limit for each; a positive value yields a soft, springy limit of
//! stiffness `1 / compliance`. Each is divided by the squared substep time to
//! form the time-step-independent `XPBD` regularisation term, so the same
//! compliance behaves consistently across substep counts.
//!
//! Provenance: the point-to-point and the hinge axis-alignment constraints with
//! their substep `XPBD` handling (Müller et al., "Detailed Rigid Body Simulation
//! with XPBD"), over the world-space inverse inertia and quaternion kinematics of
//! Baraff & Witkin. No Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use super::coloring::JointBodies;

/// A revolute (hinge) joint welding a body-local anchor on body `a` to a
/// body-local anchor on body `b` while keeping their body-local hinge axes
/// parallel.
///
/// The joint constrains the two world-space anchor points to coincide and the
/// two world-space hinge axes to stay parallel, permitting only the relative
/// spin about the shared axis. Either body may be static (zero inverse mass and
/// inverse inertia); a joint between a dynamic body and a static one hinges the
/// dynamic body about a fixed world axis through a fixed world point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RevoluteJoint {
    /// Index of the first body in the [`RigidBodyState`](super::super::RigidBodyState).
    pub body_a: u32,
    /// Index of the second body in the [`RigidBodyState`](super::super::RigidBodyState).
    pub body_b: u32,
    /// Anchor point on body `a`, in body `a`'s local frame (relative to its
    /// centre of mass).
    pub anchor_a: Vec3,
    /// Anchor point on body `b`, in body `b`'s local frame.
    pub anchor_b: Vec3,
    /// Hinge axis on body `a`, in body `a`'s local frame. Need not be
    /// unit-length; only its direction matters.
    pub axis_a: Vec3,
    /// Hinge axis on body `b`, in body `b`'s local frame.
    pub axis_b: Vec3,
    /// Inverse stiffness (metres per newton) of the positional weld. Zero is the
    /// rigid limit.
    pub compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the axis-alignment
    /// constraint. Zero is the rigid limit.
    pub angular_compliance: f32,
}

impl RevoluteJoint {
    /// Creates a revolute joint between `body_a` and `body_b` with the given
    /// body-local anchors, body-local hinge axes, and compliances.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "a hinge is defined by two anchors, two axes, and two \
                  compliances across the two bodies it couples"
    )]
    pub fn new(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis_a: Vec3,
        axis_b: Vec3,
        compliance: f32,
        angular_compliance: f32,
    ) -> RevoluteJoint {
        RevoluteJoint {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            axis_b,
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
    pub(crate) fn to_gpu(self) -> GpuRevoluteJoint {
        GpuRevoluteJoint {
            anchor_a: [self.anchor_a.x, self.anchor_a.y, self.anchor_a.z, 0.0],
            anchor_b: [self.anchor_b.x, self.anchor_b.y, self.anchor_b.z, 0.0],
            axis_a: [self.axis_a.x, self.axis_a.y, self.axis_a.z, 0.0],
            axis_b: [self.axis_b.x, self.axis_b.y, self.axis_b.z, 0.0],
            body_a: self.body_a,
            body_b: self.body_b,
            compliance: self.compliance,
            angular_compliance: self.angular_compliance,
        }
    }
}

impl JointBodies for RevoluteJoint {
    fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }
}

/// Device-packed [`RevoluteJoint`]; layout matches `Joint` in
/// `shaders/rigid_joint_revolute.wgsl` (`80` bytes). The two anchors and two
/// hinge axes are padded to `vec4` so each starts on the `16`-byte boundary the
/// storage layout requires; the four `w` lanes are unused. The two indices and
/// two compliances fill the trailing `16`-byte block.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuRevoluteJoint {
    /// Body-local anchor on body `a` in `xyz`; `w` unused.
    anchor_a: [f32; 4],
    /// Body-local anchor on body `b` in `xyz`; `w` unused.
    anchor_b: [f32; 4],
    /// Body-local hinge axis on body `a` in `xyz`; `w` unused.
    axis_a: [f32; 4],
    /// Body-local hinge axis on body `b` in `xyz`; `w` unused.
    axis_b: [f32; 4],
    /// Index of body `a`.
    body_a: u32,
    /// Index of body `b`.
    body_b: u32,
    /// Inverse stiffness (metres per newton) of the positional weld.
    compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the axis alignment.
    angular_compliance: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_packing_round_trips_fields() {
        let joint = RevoluteJoint::new(
            2,
            9,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-4.0, 5.0, -6.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.002,
            0.001,
        );
        let gpu = joint.to_gpu();
        assert_eq!(gpu.body_a, 2);
        assert_eq!(gpu.body_b, 9);
        assert_eq!(gpu.anchor_a, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(gpu.anchor_b, [-4.0, 5.0, -6.0, 0.0]);
        assert_eq!(gpu.axis_a, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(gpu.axis_b, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(gpu.compliance, 0.002);
        assert_eq!(gpu.angular_compliance, 0.001);
    }

    #[test]
    fn gpu_struct_is_80_bytes() {
        assert_eq!(size_of::<GpuRevoluteJoint>(), 80);
    }

    #[test]
    fn bodies_returns_pair_in_order() {
        let joint = RevoluteJoint::new(6, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y, 0.0, 0.0);
        assert_eq!(joint.bodies(), (6, 1));
        assert_eq!(JointBodies::bodies(&joint), (6, 1));
    }
}
