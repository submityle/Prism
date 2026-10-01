//! The swing-twist (cone-twist) rigid-body joint.
//!
//! A [`SwingTwistJoint`] is the ragdoll joint: a point-to-point positional weld
//! (the ball-and-socket of a [`SphericalJoint`](super::SphericalJoint)) with two
//! angular restrictions layered on, decomposed about a body-local *twist axis*:
//!
//! * a **swing cone**, which bounds how far the twist axis of body `b` may tilt
//!   away from the twist axis of body `a` — the half-angle of a cone the limb
//!   may sweep within, free in its interior and hard-stopped on its rim; and
//! * a **twist limit**, which bounds the signed rotation *about* that twist axis
//!   into a closed range `[twist_min, twist_max]`, free in its interior.
//!
//! It is the shoulder, the hip, and the neck of a ragdoll — the socket joint
//! that lets a limb cone around freely up to a rim while independently limiting
//! how far it may spin on its own axis. The spherical joint welds the two
//! bodies' anchors; the cone bounds the lateral swing; the twist limit bounds
//! the axial spin.
//!
//! # The three constraints
//!
//! * **Point-to-point** (positional): the world-space anchor coincidence the
//!   spherical joint drives to zero, `p_a - p_b -> 0`.
//! * **Swing cone** (angular, one-sided): with
//!   `u_a = rotate(orientation_a, twist_axis_a)` and
//!   `u_b = rotate(orientation_b, twist_axis_b)` the two unit twist axes, the
//!   swing angle `swing = acos(clamp(u_a . u_b, -1, 1))` is the true angle
//!   between them. While `swing <= swing_limit` the cone is inactive; past it
//!   the violation `swing - swing_limit` is driven shut by rotating about
//!   `n = normalize(u_a x u_b)`, the axis that closes the gap between the two
//!   twist axes. The angle is taken with `acos` rather than the cross-product
//!   magnitude so the cone half-angle may exceed `90` degrees.
//! * **Twist limit** (angular, one-sided): the signed rotation about the twist
//!   axis, measured between the two bodies' reference directions exactly as the
//!   hinge limit measures its hinge angle, is clamped into
//!   `[twist_min, twist_max]` with a free dead zone inside the range.
//!
//! Each sweep projects the swing cone first, then the twist limit (about the
//! twist axis), then the positional weld.
//!
//! # Measuring the twist angle
//!
//! The twist limit needs a signed angle, which needs a zero reference. Each body
//! carries a body-local reference direction, [`ref_a`](SwingTwistJoint::ref_a)
//! and [`ref_b`](SwingTwistJoint::ref_b), nominally perpendicular to its twist
//! axis. Each sweep both are rotated to world space, projected onto the plane
//! perpendicular to the (unit) twist axis `u = rotate(orientation_a,
//! twist_axis_a)`, and normalised; the signed angle from `a`'s projection to
//! `b`'s projection about `u` is `theta = atan2((p_a x p_b) . u, p_a . p_b)`.
//! The references need not be exactly perpendicular to the twist axis — only
//! non-parallel to it — because the projection removes the axial component
//! before the angle is taken.
//!
//! # Compliance
//!
//! [`compliance`](SwingTwistJoint::compliance) is the inverse stiffness of the
//! positional weld (metres per newton),
//! [`swing_compliance`](SwingTwistJoint::swing_compliance) the inverse stiffness
//! of the swing cone (radians per newton-metre), and
//! [`twist_compliance`](SwingTwistJoint::twist_compliance) the inverse stiffness
//! of the twist limit (radians per newton-metre). Zero is the rigid limit for
//! each; a positive value yields a soft, springy stop of stiffness
//! `1 / compliance`. Each is divided by the squared substep time to form the
//! time-step-independent `XPBD` regularisation term.
//!
//! # Scope
//!
//! The swing cone here is *circular* (a single [`swing_limit`](
//! SwingTwistJoint::swing_limit) half-angle). An *elliptical* cone — distinct
//! swing limits about the two axes perpendicular to the twist axis, as some
//! shoulder and hip models use — is a genuine extension of the constraint rather
//! than a reparametrisation of it, and is intentionally left as future work
//! rather than faked here. The twist limit is already a general asymmetric range
//! `[twist_min, twist_max]`.
//!
//! Provenance: the point-to-point (ball-socket) constraint, the signed angular
//! limit shared with the hinge limit, and the cone-swing limit with their
//! substep `XPBD` handling (Müller et al., "Detailed Rigid Body Simulation with
//! XPBD"), over the world-space inverse inertia and quaternion kinematics of
//! Baraff & Witkin. No Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use super::coloring::JointBodies;

/// A swing-twist (cone-twist) joint: a point-to-point weld plus a one-sided
/// swing-cone limit on the lateral tilt of the twist axis and a one-sided twist
/// limit on the axial spin.
///
/// The interior of both the cone and the twist range is free; only a violated
/// bound exerts a torque. Either body may be static (zero inverse mass and
/// inertia); a joint between a dynamic body and a static one sockets the dynamic
/// body about a fixed world anchor with a fixed cone and twist range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwingTwistJoint {
    /// Index of the first body in the [`RigidBodyState`](super::super::RigidBodyState).
    pub body_a: u32,
    /// Index of the second body in the [`RigidBodyState`](super::super::RigidBodyState).
    pub body_b: u32,
    /// Anchor point on body `a`, in body `a`'s local frame (relative to its
    /// centre of mass).
    pub anchor_a: Vec3,
    /// Anchor point on body `b`, in body `b`'s local frame.
    pub anchor_b: Vec3,
    /// Twist axis on body `a`, in body `a`'s local frame. Need not be
    /// unit-length; only its direction matters. The cone bounds how far
    /// [`twist_axis_b`](Self::twist_axis_b) may tilt from this axis, and the
    /// twist limit bounds the signed spin about it.
    pub twist_axis_a: Vec3,
    /// Twist axis on body `b`, in body `b`'s local frame.
    pub twist_axis_b: Vec3,
    /// Zero-angle reference direction on body `a`, in body `a`'s local frame.
    /// Must be non-parallel to [`twist_axis_a`](Self::twist_axis_a); its
    /// component along the twist axis is projected out before the twist angle is
    /// measured.
    pub ref_a: Vec3,
    /// Zero-angle reference direction on body `b`, in body `b`'s local frame.
    /// Must be non-parallel to [`twist_axis_b`](Self::twist_axis_b).
    pub ref_b: Vec3,
    /// Half-angle of the swing cone, in radians: the maximum angle
    /// [`twist_axis_b`](Self::twist_axis_b) may tilt from
    /// [`twist_axis_a`](Self::twist_axis_a) before the cone resists. Must be
    /// non-negative.
    pub swing_limit: f32,
    /// Lower bound of the free twist range, in radians.
    pub twist_min: f32,
    /// Upper bound of the free twist range, in radians.
    pub twist_max: f32,
    /// Inverse stiffness (metres per newton) of the positional weld. Zero is the
    /// rigid limit.
    pub compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the swing cone. Zero is
    /// the rigid limit (a hard rim).
    pub swing_compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the twist limit. Zero is
    /// the rigid limit (a hard stop).
    pub twist_compliance: f32,
}

impl SwingTwistJoint {
    /// Creates a swing-twist joint from its full specification.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the twist range is malformed
    /// (`twist_min > twist_max`) or the swing limit is negative; callers are
    /// expected to pass a well-ordered range and a non-negative cone.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "a cone-twist joint is defined by two anchors, two twist axes, \
                  two angle references, a swing cone, a twist range, and three \
                  compliances across the two bodies it couples"
    )]
    pub fn new(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        twist_axis_a: Vec3,
        twist_axis_b: Vec3,
        ref_a: Vec3,
        ref_b: Vec3,
        swing_limit: f32,
        twist_min: f32,
        twist_max: f32,
        compliance: f32,
        swing_compliance: f32,
        twist_compliance: f32,
    ) -> SwingTwistJoint {
        debug_assert!(
            swing_limit >= 0.0,
            "swing cone half-angle must be non-negative"
        );
        debug_assert!(
            twist_min <= twist_max,
            "twist range must satisfy twist_min <= twist_max"
        );
        SwingTwistJoint {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            twist_axis_a,
            twist_axis_b,
            ref_a,
            ref_b,
            swing_limit,
            twist_min,
            twist_max,
            compliance,
            swing_compliance,
            twist_compliance,
        }
    }

    /// Creates a rigid (zero-compliance) cone-twist joint with a circular cone
    /// and a twist range symmetric about zero: the limb may swing freely within
    /// a cone of half-angle `swing_limit` and twist freely within
    /// `[-twist_limit, +twist_limit]`, hard-stopped outside either bound.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the socket geometry alone needs two anchors, two twist axes, \
                  and two references across the two coupled bodies"
    )]
    pub fn symmetric_cone(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        twist_axis_a: Vec3,
        twist_axis_b: Vec3,
        ref_a: Vec3,
        ref_b: Vec3,
        swing_limit: f32,
        twist_limit: f32,
    ) -> SwingTwistJoint {
        SwingTwistJoint::new(
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            twist_axis_a,
            twist_axis_b,
            ref_a,
            ref_b,
            swing_limit,
            -twist_limit,
            twist_limit,
            0.0,
            0.0,
            0.0,
        )
    }

    /// The two body indices this joint couples, in `(a, b)` order.
    #[must_use]
    pub fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }

    /// Packs the joint into its `GPU` storage-buffer representation.
    #[must_use]
    pub(crate) fn to_gpu(self) -> GpuSwingTwistJoint {
        GpuSwingTwistJoint {
            anchor_a: [self.anchor_a.x, self.anchor_a.y, self.anchor_a.z, 0.0],
            anchor_b: [self.anchor_b.x, self.anchor_b.y, self.anchor_b.z, 0.0],
            twist_axis_a: [
                self.twist_axis_a.x,
                self.twist_axis_a.y,
                self.twist_axis_a.z,
                0.0,
            ],
            twist_axis_b: [
                self.twist_axis_b.x,
                self.twist_axis_b.y,
                self.twist_axis_b.z,
                0.0,
            ],
            ref_a: [self.ref_a.x, self.ref_a.y, self.ref_a.z, 0.0],
            ref_b: [self.ref_b.x, self.ref_b.y, self.ref_b.z, 0.0],
            body_a: self.body_a,
            body_b: self.body_b,
            swing_limit: self.swing_limit,
            twist_min: self.twist_min,
            twist_max: self.twist_max,
            compliance: self.compliance,
            swing_compliance: self.swing_compliance,
            twist_compliance: self.twist_compliance,
        }
    }
}

impl JointBodies for SwingTwistJoint {
    fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }
}

/// Device-packed [`SwingTwistJoint`]; layout matches `Joint` in
/// `shaders/rigid_joint_swing_twist.wgsl` (`128` bytes). The two anchors, two
/// twist axes, and two angle references are padded to `vec4` so each starts on
/// the `16`-byte boundary the storage layout requires; the six `w` lanes are
/// unused. The two indices, swing limit, and twist lower bound fill the next
/// `16`-byte block, and the twist upper bound and three compliances fill the
/// last.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuSwingTwistJoint {
    /// Body-local anchor on body `a` in `xyz`; `w` unused.
    anchor_a: [f32; 4],
    /// Body-local anchor on body `b` in `xyz`; `w` unused.
    anchor_b: [f32; 4],
    /// Body-local twist axis on body `a` in `xyz`; `w` unused.
    twist_axis_a: [f32; 4],
    /// Body-local twist axis on body `b` in `xyz`; `w` unused.
    twist_axis_b: [f32; 4],
    /// Body-local zero-angle reference on body `a` in `xyz`; `w` unused.
    ref_a: [f32; 4],
    /// Body-local zero-angle reference on body `b` in `xyz`; `w` unused.
    ref_b: [f32; 4],
    /// Index of body `a`.
    body_a: u32,
    /// Index of body `b`.
    body_b: u32,
    /// Half-angle of the swing cone, in radians.
    swing_limit: f32,
    /// Lower bound of the free twist range, in radians.
    twist_min: f32,
    /// Upper bound of the free twist range, in radians.
    twist_max: f32,
    /// Inverse stiffness (metres per newton) of the positional weld.
    compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the swing cone.
    swing_compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the twist limit.
    twist_compliance: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn gpu_packing_round_trips_fields() {
        let joint = SwingTwistJoint::new(
            2,
            9,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-4.0, 5.0, -6.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.7,
            -0.5,
            0.75,
            0.002,
            0.001,
            0.003,
        );
        let gpu = joint.to_gpu();
        assert_eq!(gpu.body_a, 2);
        assert_eq!(gpu.body_b, 9);
        assert_eq!(gpu.anchor_a, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(gpu.anchor_b, [-4.0, 5.0, -6.0, 0.0]);
        assert_eq!(gpu.twist_axis_a, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(gpu.twist_axis_b, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(gpu.ref_a, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(gpu.ref_b, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(gpu.swing_limit, 0.7);
        assert_eq!(gpu.twist_min, -0.5);
        assert_eq!(gpu.twist_max, 0.75);
        assert_eq!(gpu.compliance, 0.002);
        assert_eq!(gpu.swing_compliance, 0.001);
        assert_eq!(gpu.twist_compliance, 0.003);
    }

    #[test]
    fn gpu_struct_is_128_bytes() {
        assert_eq!(size_of::<GpuSwingTwistJoint>(), 128);
    }

    #[test]
    fn symmetric_cone_builds_a_centred_rigid_socket() {
        let joint = SwingTwistJoint::symmetric_cone(
            0,
            1,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Y,
            Vec3::Y,
            Vec3::X,
            Vec3::X,
            FRAC_PI_2,
            0.5,
        );
        assert_eq!(joint.swing_limit, FRAC_PI_2);
        assert_eq!(joint.twist_min, -0.5);
        assert_eq!(joint.twist_max, 0.5);
        assert_eq!(joint.compliance, 0.0);
        assert_eq!(joint.swing_compliance, 0.0);
        assert_eq!(joint.twist_compliance, 0.0);
    }

    #[test]
    fn bodies_returns_pair_in_order() {
        let joint = SwingTwistJoint::symmetric_cone(
            6,
            1,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Y,
            Vec3::Y,
            Vec3::X,
            Vec3::X,
            0.5,
            0.3,
        );
        assert_eq!(joint.bodies(), (6, 1));
        assert_eq!(JointBodies::bodies(&joint), (6, 1));
    }
}
