//! The spherical (ball-and-socket) rigid-body joint.
//!
//! A [`SphericalJoint`] welds a body-local anchor on body `a` to a body-local
//! anchor on body `b`, removing the three translational degrees of freedom
//! between the two anchor points while leaving all three rotational degrees of
//! freedom free. It is the point-to-point constraint at the base of every
//! articulated mechanism: a revolute hinge, a prismatic slider, and a fully
//! configurable `6`-DOF joint are each this positional weld plus one or more
//! angular restrictions layered on top.
//!
//! The anchors are stored in each body's own local frame, so they rotate with
//! the body for free: the world-space anchor on body `a` is
//! `position_a + rotate(orientation_a, anchor_a)`, and likewise for `b`. The
//! constraint the solver drives to zero is the world-space separation of those
//! two points, `p_a - p_b`.
//!
//! # Compliance
//!
//! [`compliance`](SphericalJoint::compliance) is the inverse stiffness
//! (metres per newton) of the `XPBD` constraint. A compliance of zero is the
//! rigid limit — the solver pulls the anchors together as hard as the substep
//! allows — while a positive compliance yields a soft, springy attachment whose
//! stiffness is `1 / compliance`. The solver divides it by the squared substep
//! time (`alpha_tilde = compliance / h^2`) to form the time-step-independent
//! `XPBD` regularisation term, so the same compliance behaves consistently
//! across substep counts.
//!
//! Provenance: the point-to-point (ball-socket) constraint and its `XPBD`
//! positional handling (Müller et al., "Detailed Rigid Body Simulation with
//! XPBD"), over the world-space inverse inertia and quaternion kinematics of
//! Baraff & Witkin. No Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// A ball-and-socket joint welding a body-local anchor on body `a` to a
/// body-local anchor on body `b`.
///
/// The joint constrains the two world-space anchor points to coincide,
/// permitting arbitrary relative rotation about the shared point. Either body
/// may be static (zero inverse mass and inverse inertia); a joint between a
/// dynamic body and a static one pins the dynamic body's anchor to a fixed
/// world point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphericalJoint {
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
    /// Inverse stiffness (metres per newton) of the constraint. Zero is the
    /// rigid limit; a positive value yields a soft attachment of stiffness
    /// `1 / compliance`.
    pub compliance: f32,
}

impl SphericalJoint {
    /// Creates a spherical joint between `body_a` and `body_b` with the given
    /// body-local anchors and compliance.
    #[must_use]
    pub fn new(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        compliance: f32,
    ) -> SphericalJoint {
        SphericalJoint {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            compliance,
        }
    }

    /// The two body indices this joint couples, in `(a, b)` order.
    #[must_use]
    pub fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }

    /// Packs the joint into its `GPU` storage-buffer representation.
    #[must_use]
    pub(crate) fn to_gpu(self) -> GpuSphericalJoint {
        GpuSphericalJoint {
            anchor_a: [self.anchor_a.x, self.anchor_a.y, self.anchor_a.z, 0.0],
            anchor_b: [self.anchor_b.x, self.anchor_b.y, self.anchor_b.z, 0.0],
            body_a: self.body_a,
            body_b: self.body_b,
            compliance: self.compliance,
            _pad: 0.0,
        }
    }
}

/// Device-packed [`SphericalJoint`]; layout matches `Joint` in
/// `shaders/rigid_joint_spherical.wgsl` (`48` bytes). The two anchors are padded
/// to `vec4` so each starts on the `16`-byte boundary the storage layout
/// requires; the `w` lanes are unused.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuSphericalJoint {
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
    /// Padding to a `48`-byte multiple of the `16`-byte anchor alignment.
    _pad: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_packing_round_trips_fields() {
        let joint = SphericalJoint::new(
            3,
            7,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-4.0, 5.0, -6.0),
            0.001,
        );
        let gpu = joint.to_gpu();
        assert_eq!(gpu.body_a, 3);
        assert_eq!(gpu.body_b, 7);
        assert_eq!(gpu.anchor_a, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(gpu.anchor_b, [-4.0, 5.0, -6.0, 0.0]);
        assert_eq!(gpu.compliance, 0.001);
    }

    #[test]
    fn gpu_struct_is_48_bytes() {
        assert_eq!(size_of::<GpuSphericalJoint>(), 48);
    }

    #[test]
    fn bodies_returns_pair_in_order() {
        let joint = SphericalJoint::new(5, 2, Vec3::ZERO, Vec3::ZERO, 0.0);
        assert_eq!(joint.bodies(), (5, 2));
    }
}
