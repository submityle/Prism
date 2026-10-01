//! The prismatic (slider) rigid-body joint.
//!
//! A [`PrismaticJoint`] is the mirror image of the
//! [`RevoluteJoint`](super::RevoluteJoint): where the hinge leaves one
//! *rotational* degree of freedom free and locks the rest, the slider leaves one
//! *translational* degree of freedom free and locks the rest. It pins the
//! relative orientation of body `b` to body `a` to a fixed rest offset *and*
//! confines a body-local anchor on body `b` to the line through a body-local
//! anchor on body `a` directed along a body-local slide axis on body `a`.
//! Together these remove all three rotational and two of the three translational
//! degrees of freedom, leaving exactly one free translation — the slide along
//! the shared axis. It is the piston, the drawer runner, and the linear actuator
//! of an articulated mechanism.
//!
//! # The two constraints
//!
//! * **Angular lock** (angular): the relative orientation of body `b` in body
//!   `a`'s frame is driven to the fixed [`rest_rotation`](PrismaticJoint::rest_rotation).
//!   The target world orientation of body `b` is
//!   `q_a * rest_rotation`, and the world-space error rotation
//!   `q_b * conj(q_a * rest_rotation)` is cancelled, locking all three relative
//!   rotational degrees of freedom.
//! * **Perpendicular weld** (positional): the world-space anchor separation
//!   `dx = p_a - p_b` (with `p = position + rotate(orientation, anchor)`) is
//!   driven to zero *only in the plane perpendicular to the slide axis*. The
//!   component of `dx` along the world-space slide axis
//!   `rotate(orientation_a, axis_a)` is left untouched, which is exactly the one
//!   free translational degree of freedom.
//!
//! The angular lock is projected first each sweep, then the perpendicular weld,
//! so the slide axis is re-oriented before the anchors are pulled onto the line.
//!
//! # Compliance
//!
//! [`compliance`](PrismaticJoint::compliance) is the inverse stiffness of the
//! perpendicular weld (metres per newton) and
//! [`angular_compliance`](PrismaticJoint::angular_compliance) the inverse
//! stiffness of the angular lock (radians per newton-metre). Zero is the rigid
//! limit for each; a positive value yields a soft, springy limit of stiffness
//! `1 / compliance`. Each is divided by the squared substep time to form the
//! time-step-independent `XPBD` regularisation term, so the same compliance
//! behaves consistently across substep counts.
//!
//! Provenance: the point-to-point (ball-socket) constraint restricted to the
//! plane perpendicular to the slide axis and the relative-orientation lock with
//! their substep `XPBD` handling (Müller et al., "Detailed Rigid Body Simulation
//! with XPBD"), over the world-space inverse inertia and quaternion kinematics of
//! Baraff & Witkin. No Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};

use super::coloring::JointBodies;

/// A prismatic (slider) joint confining a body-local anchor on body `b` to the
/// line through a body-local anchor on body `a` along a body-local slide axis on
/// body `a`, while holding body `b`'s orientation at a fixed offset from body
/// `a`'s.
///
/// The joint constrains the two world-space anchor points to coincide *in the
/// plane perpendicular to the slide axis* and the relative orientation to equal
/// a fixed rest offset, permitting only the relative translation along the
/// shared axis. Either body may be static (zero inverse mass and inverse
/// inertia); a joint between a dynamic body and a static one slides the dynamic
/// body along a fixed world axis through a fixed world point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrismaticJoint {
    /// Index of the first body in the [`RigidBodyState`](super::super::RigidBodyState).
    pub body_a: u32,
    /// Index of the second body in the [`RigidBodyState`](super::super::RigidBodyState).
    pub body_b: u32,
    /// Anchor point on body `a`, in body `a`'s local frame (relative to its
    /// centre of mass).
    pub anchor_a: Vec3,
    /// Anchor point on body `b`, in body `b`'s local frame.
    pub anchor_b: Vec3,
    /// Slide axis on body `a`, in body `a`'s local frame. Need not be
    /// unit-length; only its direction matters.
    pub axis_a: Vec3,
    /// Rest relative orientation of body `b` in body `a`'s frame — the fixed
    /// offset the angular lock holds, typically `conj(q_a0) * q_b0` captured at
    /// construction so the pair starts un-stressed.
    pub rest_rotation: Quat,
    /// Inverse stiffness (metres per newton) of the perpendicular weld. Zero is
    /// the rigid limit.
    pub compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the angular lock. Zero is
    /// the rigid limit.
    pub angular_compliance: f32,
}

impl PrismaticJoint {
    /// Creates a prismatic joint between `body_a` and `body_b` with the given
    /// body-local anchors, body-local slide axis, rest relative orientation, and
    /// compliances.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "a slider is defined by two anchors, a slide axis, a rest \
                  orientation, and two compliances across the two bodies it couples"
    )]
    pub fn new(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis_a: Vec3,
        rest_rotation: Quat,
        compliance: f32,
        angular_compliance: f32,
    ) -> PrismaticJoint {
        PrismaticJoint {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
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
    pub(crate) fn to_gpu(self) -> GpuPrismaticJoint {
        GpuPrismaticJoint {
            anchor_a: [self.anchor_a.x, self.anchor_a.y, self.anchor_a.z, 0.0],
            anchor_b: [self.anchor_b.x, self.anchor_b.y, self.anchor_b.z, 0.0],
            axis_a: [self.axis_a.x, self.axis_a.y, self.axis_a.z, 0.0],
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

impl JointBodies for PrismaticJoint {
    fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }
}

/// Device-packed [`PrismaticJoint`]; layout matches `Joint` in
/// `shaders/rigid_joint_prismatic.wgsl` (`80` bytes). The two anchors and the
/// slide axis are padded to `vec4` so each starts on the `16`-byte boundary the
/// storage layout requires (the three `w` lanes are unused); the rest-rotation
/// quaternion fills a fourth `vec4` as `(x, y, z, w)`. The two indices and two
/// compliances fill the trailing `16`-byte block.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuPrismaticJoint {
    /// Body-local anchor on body `a` in `xyz`; `w` unused.
    anchor_a: [f32; 4],
    /// Body-local anchor on body `b` in `xyz`; `w` unused.
    anchor_b: [f32; 4],
    /// Body-local slide axis on body `a` in `xyz`; `w` unused.
    axis_a: [f32; 4],
    /// Rest relative orientation of body `b` in body `a`'s frame as `(x, y, z, w)`.
    rest_rotation: [f32; 4],
    /// Index of body `a`.
    body_a: u32,
    /// Index of body `b`.
    body_b: u32,
    /// Inverse stiffness (metres per newton) of the perpendicular weld.
    compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the angular lock.
    angular_compliance: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_packing_round_trips_fields() {
        let joint = PrismaticJoint::new(
            2,
            9,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-4.0, 5.0, -6.0),
            Vec3::new(0.0, 1.0, 0.0),
            Quat::from_xyzw(0.1, 0.2, 0.3, 0.4),
            0.002,
            0.001,
        );
        let gpu = joint.to_gpu();
        assert_eq!(gpu.body_a, 2);
        assert_eq!(gpu.body_b, 9);
        assert_eq!(gpu.anchor_a, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(gpu.anchor_b, [-4.0, 5.0, -6.0, 0.0]);
        assert_eq!(gpu.axis_a, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(gpu.rest_rotation, [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(gpu.compliance, 0.002);
        assert_eq!(gpu.angular_compliance, 0.001);
    }

    #[test]
    fn gpu_struct_is_80_bytes() {
        assert_eq!(size_of::<GpuPrismaticJoint>(), 80);
    }

    #[test]
    fn bodies_returns_pair_in_order() {
        let joint = PrismaticJoint::new(
            6,
            1,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Y,
            Quat::IDENTITY,
            0.0,
            0.0,
        );
        assert_eq!(joint.bodies(), (6, 1));
        assert_eq!(JointBodies::bodies(&joint), (6, 1));
    }
}
