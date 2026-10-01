//! The configurable six-degree-of-freedom (`D6`) rigid-body joint.
//!
//! A [`D6Joint`] is the general-purpose joint that subsumes every other joint in
//! this crate: it exposes all six relative degrees of freedom — three linear
//! (`x`, `y`, `z`) and three angular (`twist`, `swing1`, `swing2`) — and lets
//! each one independently be [`Locked`](D6Motion::Locked),
//! [`Limited`](D6Motion::Limited), or [`Free`](D6Motion::Free). This is the model
//! Unreal Engine's `FConstraintInstance` and `PhysX`'s `PxD6Joint` are built on:
//! a single joint type whose per-axis motion flags recover a weld, a hinge, a
//! prismatic slider, a ball-socket, a cone-twist ragdoll, and everything in
//! between, purely by configuration.
//!
//! # Joint frames
//!
//! Each body carries a body-local *joint frame*: an anchor point
//! ([`anchor_a`](D6Joint::anchor_a) / [`anchor_b`](D6Joint::anchor_b), relative
//! to the centre of mass) and an orthonormal basis orientation
//! ([`basis_a`](D6Joint::basis_a) / [`basis_b`](D6Joint::basis_b), a unit
//! quaternion). The joint measures every degree of freedom as the relative
//! motion of body `b`'s frame with respect to body `a`'s frame. The frame's
//! columns are the degree-of-freedom axes: its `x` column is the linear `x` and
//! the angular twist axis, and its `y` / `z` columns are the linear `y` / `z`
//! and the swing axes. At rest the two frames coincide, so every free coordinate
//! reads zero.
//!
//! # The six degrees of freedom
//!
//! * **Linear `x` / `y` / `z`** ([`linear_x`](D6Joint::linear_x) …): the
//!   components, along body `a`'s frame axes, of the separation
//!   `p_b_world - p_a_world` between the two anchors. A `Locked` component is
//!   driven to zero (a positional weld along that axis); a `Limited` component is
//!   clamped to the symmetric slab `[-linear_limit, +linear_limit]` and is free
//!   inside it; a `Free` component exerts no force.
//! * **Angular twist** ([`twist`](D6Joint::twist)): the signed rotation of body
//!   `b`'s frame about body `a`'s frame `x` axis. `Locked` pins it to zero;
//!   `Limited` clamps it to `[twist_min, twist_max]` with a free dead zone
//!   inside; `Free` leaves it unconstrained.
//! * **Angular swing1 / swing2** ([`swing1`](D6Joint::swing1) /
//!   [`swing2`](D6Joint::swing2)): the tilt of body `b`'s frame `x` axis away
//!   from body `a`'s frame `x` axis, resolved onto the frame `y` (swing1) and `z`
//!   (swing2) axes. Each `Locked` swing pins its tilt to zero; a `Limited` pair
//!   bounds the tilt inside the elliptical cone whose half-angles are
//!   [`swing1_limit`](D6Joint::swing1_limit) and
//!   [`swing2_limit`](D6Joint::swing2_limit) (matching the circular
//!   [`SwingTwistJoint`](super::SwingTwistJoint) and the
//!   [`EllipticalConeTwistJoint`](super::EllipticalConeTwistJoint)); a `Free`
//!   swing is unconstrained.
//!
//! # Compliance
//!
//! The joint carries three inverse stiffnesses, each divided by the squared
//! substep time to form the time-step-independent `XPBD` regularisation term:
//! [`compliance`](D6Joint::compliance) softens every `Locked` axis (metres per
//! newton for the linear welds, radians per newton-metre for the angular pins),
//! [`linear_limit_compliance`](D6Joint::linear_limit_compliance) softens the
//! linear slab stops, and
//! [`angular_limit_compliance`](D6Joint::angular_limit_compliance) softens the
//! twist and swing cone stops. Zero is the rigid limit for each.
//!
//! # Degeneration to the specialised joints
//!
//! Setting all six axes `Locked` recovers the [`FixedJoint`](super::FixedJoint);
//! locking the three linear axes and the two swings while leaving twist `Free`
//! recovers the [`RevoluteJoint`](super::RevoluteJoint) about the frame `x`
//! axis; locking two linear axes and all three angular axes while leaving linear
//! `x` `Free` recovers the [`PrismaticJoint`](super::PrismaticJoint); locking the
//! three linear axes while leaving all three angular axes `Free` recovers the
//! [`SphericalJoint`](super::SphericalJoint). The specialised joints are kept as
//! dedicated kernels because their fixed configuration admits a smaller, tighter
//! solver; the `D6` joint is the escape hatch for configurations they do not
//! cover.
//!
//! Provenance: the per-axis configurable degree-of-freedom model of a
//! general-purpose constraint (Unreal Engine's `FConstraintInstance`, `PhysX`'s
//! `PxD6Joint`), realised over the point-to-point weld, the signed angular
//! limit, and the elliptical swing cone already used by this crate's specialised
//! joints, with their substep `XPBD` handling (Müller et al., "Detailed Rigid
//! Body Simulation with XPBD") over the world-space inverse inertia and
//! quaternion kinematics of Baraff & Witkin. No Unreal Engine source or derived
//! code: only the public per-axis configuration semantics are mirrored.

use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};

use super::coloring::JointBodies;

/// The motion allowed along or about a single `D6` degree of freedom.
///
/// The discriminants are the stable on-device encoding read by the shader twin,
/// so they must not be reordered: `0` is [`Locked`](Self::Locked), `1` is
/// [`Limited`](Self::Limited), `2` is [`Free`](Self::Free).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum D6Motion {
    /// The degree of freedom is pinned to zero (a rigid weld along or about the
    /// axis, softened by the joint's weld compliance).
    #[default]
    Locked,
    /// The degree of freedom is free inside its limit and clamped at the bound
    /// (the linear slab, the twist range, or the elliptical swing cone).
    Limited,
    /// The degree of freedom is unconstrained; it exerts no force or torque.
    Free,
}

impl D6Motion {
    /// The stable `u32` encoding handed to the `GPU` twin: `Locked` is `0`,
    /// `Limited` is `1`, `Free` is `2`.
    #[must_use]
    pub const fn code(self) -> u32 {
        match self {
            D6Motion::Locked => 0,
            D6Motion::Limited => 1,
            D6Motion::Free => 2,
        }
    }
}

/// A configurable six-degree-of-freedom joint: a per-axis combination of welds,
/// limits, and free axes between two bodies' joint frames.
///
/// Either body may be static (zero inverse mass and inertia); a joint between a
/// dynamic body and a static one anchors the dynamic body to a fixed world frame
/// with the configured per-axis freedoms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct D6Joint {
    /// Index of the first body in the [`RigidBodyState`](super::super::RigidBodyState).
    pub body_a: u32,
    /// Index of the second body in the [`RigidBodyState`](super::super::RigidBodyState).
    pub body_b: u32,
    /// Anchor point on body `a`, in body `a`'s local frame (relative to its
    /// centre of mass).
    pub anchor_a: Vec3,
    /// Anchor point on body `b`, in body `b`'s local frame.
    pub anchor_b: Vec3,
    /// Orientation of the joint frame on body `a`, in body `a`'s local frame.
    /// Its `x` / `y` / `z` columns are the degree-of-freedom axes. Must be a
    /// unit quaternion.
    pub basis_a: Quat,
    /// Orientation of the joint frame on body `b`, in body `b`'s local frame.
    /// Must be a unit quaternion.
    pub basis_b: Quat,
    /// Motion allowed along the frame `x` axis.
    pub linear_x: D6Motion,
    /// Motion allowed along the frame `y` axis.
    pub linear_y: D6Motion,
    /// Motion allowed along the frame `z` axis.
    pub linear_z: D6Motion,
    /// Motion allowed about the frame `x` axis (the twist).
    pub twist: D6Motion,
    /// Motion allowed in the tilt toward the frame `y` axis (swing1).
    pub swing1: D6Motion,
    /// Motion allowed in the tilt toward the frame `z` axis (swing2).
    pub swing2: D6Motion,
    /// Half-extent of the symmetric slab `[-linear_limit, +linear_limit]` that
    /// bounds every [`Limited`](D6Motion::Limited) linear axis, in metres. Must
    /// be non-negative.
    pub linear_limit: f32,
    /// Lower bound of the free twist range, in radians.
    pub twist_min: f32,
    /// Upper bound of the free twist range, in radians.
    pub twist_max: f32,
    /// Swing half-angle toward the `+/-` frame `y` axis, in radians. Must be
    /// positive when [`swing1`](Self::swing1) is [`Limited`](D6Motion::Limited).
    pub swing1_limit: f32,
    /// Swing half-angle toward the `+/-` frame `z` axis, in radians. Must be
    /// positive when [`swing2`](Self::swing2) is [`Limited`](D6Motion::Limited).
    pub swing2_limit: f32,
    /// Inverse stiffness of every [`Locked`](D6Motion::Locked) axis (metres per
    /// newton for linear welds, radians per newton-metre for angular pins). Zero
    /// is the rigid limit.
    pub compliance: f32,
    /// Inverse stiffness (metres per newton) of the linear slab stops. Zero is
    /// the rigid limit.
    pub linear_limit_compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the twist and swing cone
    /// stops. Zero is the rigid limit.
    pub angular_limit_compliance: f32,
}

impl D6Joint {
    /// Creates a `D6` joint with every field given explicitly.
    ///
    /// `anchor_a` / `anchor_b` are the body-local anchors and `basis_a` /
    /// `basis_b` the body-local joint-frame orientations. The six `D6Motion`
    /// flags select the per-axis freedoms, `linear_limit` is the symmetric slab
    /// half-extent for limited linear axes, `[twist_min, twist_max]` the twist
    /// range, and `swing1_limit` / `swing2_limit` the elliptical cone
    /// half-angles. The three compliances soften the welds, the linear stops,
    /// and the angular stops respectively.
    #[expect(
        clippy::too_many_arguments,
        reason = "the D6 joint is intrinsically wide: six motion flags plus frames, limits, and compliances"
    )]
    #[must_use]
    pub fn new(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        basis_a: Quat,
        basis_b: Quat,
        linear: [D6Motion; 3],
        angular: [D6Motion; 3],
        linear_limit: f32,
        twist_range: (f32, f32),
        swing_limits: (f32, f32),
        compliances: (f32, f32, f32),
    ) -> D6Joint {
        D6Joint {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            basis_a,
            basis_b,
            linear_x: linear[0],
            linear_y: linear[1],
            linear_z: linear[2],
            twist: angular[0],
            swing1: angular[1],
            swing2: angular[2],
            linear_limit,
            twist_min: twist_range.0,
            twist_max: twist_range.1,
            swing1_limit: swing_limits.0,
            swing2_limit: swing_limits.1,
            compliance: compliances.0,
            linear_limit_compliance: compliances.1,
            angular_limit_compliance: compliances.2,
        }
    }

    /// A rigid weld: all six axes [`Locked`](D6Motion::Locked). Reproduces the
    /// [`FixedJoint`](super::FixedJoint) between the two joint frames.
    #[must_use]
    pub fn fixed(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        basis_a: Quat,
        basis_b: Quat,
    ) -> D6Joint {
        D6Joint::new(
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            basis_a,
            basis_b,
            [D6Motion::Locked; 3],
            [D6Motion::Locked; 3],
            0.0,
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
    }

    /// A fully free pair: all six axes [`Free`](D6Motion::Free). The two bodies
    /// are not coupled at all; useful as a configuration baseline.
    #[must_use]
    pub fn free(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        basis_a: Quat,
        basis_b: Quat,
    ) -> D6Joint {
        D6Joint::new(
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            basis_a,
            basis_b,
            [D6Motion::Free; 3],
            [D6Motion::Free; 3],
            0.0,
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
    }

    /// The `(body_a, body_b)` endpoints, in order.
    #[must_use]
    pub const fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }

    /// Packs the joint into its `GPU` storage-buffer representation.
    #[must_use]
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "consumed by the d6_gpu twin landing in a later slice; already exercised by the round-trip unit test under cfg(test)"
        )
    )]
    pub(crate) fn to_gpu(self) -> GpuD6Joint {
        GpuD6Joint {
            anchor_a: [self.anchor_a.x, self.anchor_a.y, self.anchor_a.z, 0.0],
            anchor_b: [self.anchor_b.x, self.anchor_b.y, self.anchor_b.z, 0.0],
            basis_a: [
                self.basis_a.x,
                self.basis_a.y,
                self.basis_a.z,
                self.basis_a.w,
            ],
            basis_b: [
                self.basis_b.x,
                self.basis_b.y,
                self.basis_b.z,
                self.basis_b.w,
            ],
            body_a: self.body_a,
            body_b: self.body_b,
            linear_x: self.linear_x.code(),
            linear_y: self.linear_y.code(),
            linear_z: self.linear_z.code(),
            twist: self.twist.code(),
            swing1: self.swing1.code(),
            swing2: self.swing2.code(),
            linear_limit: self.linear_limit,
            twist_min: self.twist_min,
            twist_max: self.twist_max,
            swing1_limit: self.swing1_limit,
            swing2_limit: self.swing2_limit,
            compliance: self.compliance,
            linear_limit_compliance: self.linear_limit_compliance,
            angular_limit_compliance: self.angular_limit_compliance,
        }
    }
}

impl JointBodies for D6Joint {
    fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }
}

/// Device-packed [`D6Joint`]; layout matches `Joint` in
/// `shaders/rigid_joint_d6.wgsl` (`128` bytes). The two anchors and two joint-
/// frame quaternions fill four `vec4` blocks; the two body indices and six
/// per-axis motion codes fill the next two `16`-byte blocks; the linear limit,
/// twist range, and swing half-angles the next; the three compliances plus one
/// pad word close the final block.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuD6Joint {
    /// Body-local anchor on body `a` in `xyz`; `w` unused.
    anchor_a: [f32; 4],
    /// Body-local anchor on body `b` in `xyz`; `w` unused.
    anchor_b: [f32; 4],
    /// Body-local joint-frame orientation on body `a` as a quaternion.
    basis_a: [f32; 4],
    /// Body-local joint-frame orientation on body `b` as a quaternion.
    basis_b: [f32; 4],
    /// Index of body `a`.
    body_a: u32,
    /// Index of body `b`.
    body_b: u32,
    /// Motion code for the linear `x` axis (`0` locked, `1` limited, `2` free).
    linear_x: u32,
    /// Motion code for the linear `y` axis.
    linear_y: u32,
    /// Motion code for the linear `z` axis.
    linear_z: u32,
    /// Motion code for the twist axis.
    twist: u32,
    /// Motion code for the swing1 axis.
    swing1: u32,
    /// Motion code for the swing2 axis.
    swing2: u32,
    /// Symmetric slab half-extent for limited linear axes, in metres.
    linear_limit: f32,
    /// Lower bound of the free twist range, in radians.
    twist_min: f32,
    /// Upper bound of the free twist range, in radians.
    twist_max: f32,
    /// Swing half-angle toward the frame `y` axis, in radians.
    swing1_limit: f32,
    /// Swing half-angle toward the frame `z` axis, in radians.
    swing2_limit: f32,
    /// Inverse stiffness of the locked-axis welds.
    compliance: f32,
    /// Inverse stiffness of the linear slab stops.
    linear_limit_compliance: f32,
    /// Inverse stiffness of the angular stops.
    angular_limit_compliance: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_6};

    #[test]
    fn motion_codes_are_stable() {
        assert_eq!(D6Motion::Locked.code(), 0);
        assert_eq!(D6Motion::Limited.code(), 1);
        assert_eq!(D6Motion::Free.code(), 2);
        assert_eq!(D6Motion::default(), D6Motion::Locked);
    }

    #[test]
    fn gpu_struct_is_128_bytes() {
        assert_eq!(size_of::<GpuD6Joint>(), 128);
    }

    #[test]
    fn gpu_packing_round_trips_fields() {
        let joint = D6Joint::new(
            2,
            9,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-4.0, 5.0, -6.0),
            Quat::from_axis_angle(Vec3::Y, FRAC_PI_2),
            Quat::from_axis_angle(Vec3::X, FRAC_PI_6),
            [D6Motion::Locked, D6Motion::Limited, D6Motion::Free],
            [D6Motion::Free, D6Motion::Limited, D6Motion::Locked],
            0.25,
            (-0.5, 0.75),
            (0.7, 0.4),
            (0.002, 0.001, 0.004),
        );
        let gpu = joint.to_gpu();
        assert_eq!(gpu.anchor_a, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(gpu.anchor_b, [-4.0, 5.0, -6.0, 0.0]);
        let qa = Quat::from_axis_angle(Vec3::Y, FRAC_PI_2);
        let qb = Quat::from_axis_angle(Vec3::X, FRAC_PI_6);
        assert_eq!(gpu.basis_a, [qa.x, qa.y, qa.z, qa.w]);
        assert_eq!(gpu.basis_b, [qb.x, qb.y, qb.z, qb.w]);
        assert_eq!(gpu.body_a, 2);
        assert_eq!(gpu.body_b, 9);
        assert_eq!(gpu.linear_x, 0);
        assert_eq!(gpu.linear_y, 1);
        assert_eq!(gpu.linear_z, 2);
        assert_eq!(gpu.twist, 2);
        assert_eq!(gpu.swing1, 1);
        assert_eq!(gpu.swing2, 0);
        assert_eq!(gpu.linear_limit, 0.25);
        assert_eq!(gpu.twist_min, -0.5);
        assert_eq!(gpu.twist_max, 0.75);
        assert_eq!(gpu.swing1_limit, 0.7);
        assert_eq!(gpu.swing2_limit, 0.4);
        assert_eq!(gpu.compliance, 0.002);
        assert_eq!(gpu.linear_limit_compliance, 0.001);
        assert_eq!(gpu.angular_limit_compliance, 0.004);
    }

    #[test]
    fn fixed_locks_every_axis() {
        let joint = D6Joint::fixed(
            0,
            1,
            Vec3::ZERO,
            Vec3::new(0.0, -1.0, 0.0),
            Quat::IDENTITY,
            Quat::IDENTITY,
        );
        assert_eq!(joint.linear_x, D6Motion::Locked);
        assert_eq!(joint.linear_y, D6Motion::Locked);
        assert_eq!(joint.linear_z, D6Motion::Locked);
        assert_eq!(joint.twist, D6Motion::Locked);
        assert_eq!(joint.swing1, D6Motion::Locked);
        assert_eq!(joint.swing2, D6Motion::Locked);
    }

    #[test]
    fn free_unlocks_every_axis() {
        let joint = D6Joint::free(3, 4, Vec3::ZERO, Vec3::ZERO, Quat::IDENTITY, Quat::IDENTITY);
        assert_eq!(joint.linear_x, D6Motion::Free);
        assert_eq!(joint.twist, D6Motion::Free);
        assert_eq!(joint.swing2, D6Motion::Free);
    }

    #[test]
    fn bodies_returns_pair_in_order() {
        let joint = D6Joint::fixed(4, 7, Vec3::ZERO, Vec3::ZERO, Quat::IDENTITY, Quat::IDENTITY);
        assert_eq!(joint.bodies(), (4, 7));
        assert_eq!(JointBodies::bodies(&joint), (4, 7));
    }
}
