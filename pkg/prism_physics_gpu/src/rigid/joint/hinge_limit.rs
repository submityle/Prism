//! The hinge angular-limit rigid-body joint.
//!
//! A [`HingeLimitJoint`] is a revolute (hinge) joint — the point-to-point
//! positional weld plus the hinge axis-alignment of a
//! [`RevoluteJoint`](super::RevoluteJoint) — with one more restriction layered
//! on: a *swing limit* about the hinge axis. The free spin a bare hinge permits
//! is clamped to the closed angular range `[min_angle, max_angle]`, with the
//! interior of that range left completely free. It is the elbow that cannot
//! bend backwards, the door that stops against its frame, and the knee of a
//! ragdoll — the single hinge degree of freedom bounded on one or both sides.
//!
//! # The three constraints
//!
//! * **Point-to-point** (positional): the same world-space anchor coincidence
//!   the spherical joint drives to zero, `p_a - p_b -> 0`.
//! * **Axis alignment** (angular): the world-space hinge axes
//!   `u_a = rotate(orientation_a, axis_a)` and `u_b = rotate(orientation_b,
//!   axis_b)` are driven parallel, locking the two rotational degrees of freedom
//!   perpendicular to the hinge.
//! * **Angular limit** (angular, one-sided): the signed hinge angle `theta`
//!   between the two bodies' reference directions, measured about the shared
//!   hinge axis, is clamped into `[min_angle, max_angle]`. When
//!   `min_angle <= theta <= max_angle` the limit is inactive and exerts no
//!   torque — the hinge spins freely inside the range; outside it the violated
//!   bound is driven shut with a signed correction about the hinge axis.
//!
//! Each sweep projects the axis alignment first (realigning the hinge), then the
//! angular limit (about the freshly aligned axis), then the positional weld.
//!
//! # Measuring the hinge angle
//!
//! The limit needs a signed angle, which needs a zero reference. Each body
//! carries a body-local reference direction, [`ref_a`](HingeLimitJoint::ref_a)
//! and [`ref_b`](HingeLimitJoint::ref_b), nominally perpendicular to its hinge
//! axis. Each sweep both are rotated to world space, projected onto the plane
//! perpendicular to the (unit) hinge axis `u`, and normalised; the signed angle
//! from `ref_a`'s projection to `ref_b`'s projection about `u` is
//! `theta = atan2((p_a x p_b) . u, p_a . p_b)`. The references need not be
//! exactly perpendicular to the axis — only non-parallel to it — because the
//! projection removes the axial component before the angle is taken.
//!
//! # Compliance
//!
//! [`compliance`](HingeLimitJoint::compliance) is the inverse stiffness of the
//! positional weld (metres per newton),
//! [`angular_compliance`](HingeLimitJoint::angular_compliance) the inverse
//! stiffness of the axis alignment (radians per newton-metre), and
//! [`limit_compliance`](HingeLimitJoint::limit_compliance) the inverse stiffness
//! of the angular limit (radians per newton-metre). Zero is the rigid limit for
//! each; a positive value yields a soft, springy stop of stiffness
//! `1 / compliance`. Each is divided by the squared substep time to form the
//! time-step-independent `XPBD` regularisation term.
//!
//! Provenance: the point-to-point and hinge axis-alignment constraints and the
//! one-sided angular limit with their substep `XPBD` handling (Müller et al.,
//! "Detailed Rigid Body Simulation with XPBD"), over the world-space inverse
//! inertia and quaternion kinematics of Baraff & Witkin. No Unreal Engine source
//! or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use super::coloring::JointBodies;

/// A hinge joint with a swing limit: a point-to-point weld plus hinge
/// axis-alignment plus a one-sided clamp of the hinge angle into
/// `[min_angle, max_angle]`.
///
/// The interior of the angular range is free; only a violated bound exerts a
/// torque. Either body may be static (zero inverse mass and inertia); a joint
/// between a dynamic body and a static one hinges the dynamic body about a fixed
/// world axis with a fixed angular range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HingeLimitJoint {
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
    /// Zero-angle reference direction on body `a`, in body `a`'s local frame.
    /// Must be non-parallel to [`axis_a`](Self::axis_a); its component along the
    /// hinge axis is projected out before the angle is measured.
    pub ref_a: Vec3,
    /// Zero-angle reference direction on body `b`, in body `b`'s local frame.
    /// Must be non-parallel to [`axis_b`](Self::axis_b).
    pub ref_b: Vec3,
    /// Lower bound of the free hinge range, in radians. Must satisfy
    /// `min_angle <= max_angle`.
    pub min_angle: f32,
    /// Upper bound of the free hinge range, in radians.
    pub max_angle: f32,
    /// Inverse stiffness (metres per newton) of the positional weld. Zero is the
    /// rigid limit.
    pub compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the axis-alignment
    /// constraint. Zero is the rigid limit.
    pub angular_compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the angular limit. Zero
    /// is the rigid limit (a hard stop).
    pub limit_compliance: f32,
}

impl HingeLimitJoint {
    /// Creates a hinge-limit joint from its full specification.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the angular range is malformed
    /// (`min_angle > max_angle`); callers are expected to pass a well-ordered
    /// range.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "a hinge limit is defined by two anchors, two hinge axes, two \
                  angle references, an angular range, and three compliances \
                  across the two bodies it couples"
    )]
    pub fn new(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis_a: Vec3,
        axis_b: Vec3,
        ref_a: Vec3,
        ref_b: Vec3,
        min_angle: f32,
        max_angle: f32,
        compliance: f32,
        angular_compliance: f32,
        limit_compliance: f32,
    ) -> HingeLimitJoint {
        debug_assert!(
            min_angle <= max_angle,
            "hinge limit range must satisfy min_angle <= max_angle"
        );
        HingeLimitJoint {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            axis_b,
            ref_a,
            ref_b,
            min_angle,
            max_angle,
            compliance,
            angular_compliance,
            limit_compliance,
        }
    }

    /// Creates a rigid (zero-compliance) hinge limit symmetric about the zero
    /// angle: the hinge is free within `[-half_range, +half_range]` radians and
    /// hard-stopped outside it.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the hinge geometry alone needs two anchors, two axes, and two \
                  references across the two coupled bodies"
    )]
    pub fn symmetric(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis_a: Vec3,
        axis_b: Vec3,
        ref_a: Vec3,
        ref_b: Vec3,
        half_range: f32,
    ) -> HingeLimitJoint {
        HingeLimitJoint::new(
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            axis_b,
            ref_a,
            ref_b,
            -half_range,
            half_range,
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
    pub(crate) fn to_gpu(self) -> GpuHingeLimitJoint {
        GpuHingeLimitJoint {
            anchor_a: [self.anchor_a.x, self.anchor_a.y, self.anchor_a.z, 0.0],
            anchor_b: [self.anchor_b.x, self.anchor_b.y, self.anchor_b.z, 0.0],
            axis_a: [self.axis_a.x, self.axis_a.y, self.axis_a.z, 0.0],
            axis_b: [self.axis_b.x, self.axis_b.y, self.axis_b.z, 0.0],
            ref_a: [self.ref_a.x, self.ref_a.y, self.ref_a.z, 0.0],
            ref_b: [self.ref_b.x, self.ref_b.y, self.ref_b.z, 0.0],
            body_a: self.body_a,
            body_b: self.body_b,
            compliance: self.compliance,
            angular_compliance: self.angular_compliance,
            limit_compliance: self.limit_compliance,
            min_angle: self.min_angle,
            max_angle: self.max_angle,
            _pad0: 0.0,
        }
    }
}

impl JointBodies for HingeLimitJoint {
    fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }
}

/// Device-packed [`HingeLimitJoint`]; layout matches `Joint` in
/// `shaders/rigid_joint_hinge_limit.wgsl` (`128` bytes). The two anchors, two
/// hinge axes, and two angle references are padded to `vec4` so each starts on
/// the `16`-byte boundary the storage layout requires; the six `w` lanes are
/// unused. The two indices, three compliances, and the two angle bounds fill the
/// trailing two `16`-byte blocks.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuHingeLimitJoint {
    /// Body-local anchor on body `a` in `xyz`; `w` unused.
    anchor_a: [f32; 4],
    /// Body-local anchor on body `b` in `xyz`; `w` unused.
    anchor_b: [f32; 4],
    /// Body-local hinge axis on body `a` in `xyz`; `w` unused.
    axis_a: [f32; 4],
    /// Body-local hinge axis on body `b` in `xyz`; `w` unused.
    axis_b: [f32; 4],
    /// Body-local zero-angle reference on body `a` in `xyz`; `w` unused.
    ref_a: [f32; 4],
    /// Body-local zero-angle reference on body `b` in `xyz`; `w` unused.
    ref_b: [f32; 4],
    /// Index of body `a`.
    body_a: u32,
    /// Index of body `b`.
    body_b: u32,
    /// Inverse stiffness (metres per newton) of the positional weld.
    compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the axis alignment.
    angular_compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the angular limit.
    limit_compliance: f32,
    /// Lower bound of the free hinge range, in radians.
    min_angle: f32,
    /// Upper bound of the free hinge range, in radians.
    max_angle: f32,
    /// Padding to the `16`-byte boundary; always zero.
    _pad0: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn gpu_packing_round_trips_fields() {
        let joint = HingeLimitJoint::new(
            2,
            9,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-4.0, 5.0, -6.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
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
        assert_eq!(gpu.axis_a, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(gpu.axis_b, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(gpu.ref_a, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(gpu.ref_b, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(gpu.min_angle, -0.5);
        assert_eq!(gpu.max_angle, 0.75);
        assert_eq!(gpu.compliance, 0.002);
        assert_eq!(gpu.angular_compliance, 0.001);
        assert_eq!(gpu.limit_compliance, 0.003);
        assert_eq!(gpu._pad0, 0.0);
    }

    #[test]
    fn gpu_struct_is_128_bytes() {
        assert_eq!(size_of::<GpuHingeLimitJoint>(), 128);
    }

    #[test]
    fn symmetric_builds_a_centred_rigid_range() {
        let joint = HingeLimitJoint::symmetric(
            0,
            1,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Y,
            Vec3::Y,
            Vec3::X,
            Vec3::X,
            FRAC_PI_2,
        );
        assert_eq!(joint.min_angle, -FRAC_PI_2);
        assert_eq!(joint.max_angle, FRAC_PI_2);
        assert_eq!(joint.compliance, 0.0);
        assert_eq!(joint.angular_compliance, 0.0);
        assert_eq!(joint.limit_compliance, 0.0);
    }

    #[test]
    fn bodies_returns_pair_in_order() {
        let joint = HingeLimitJoint::symmetric(
            6,
            1,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Y,
            Vec3::Y,
            Vec3::X,
            Vec3::X,
            0.5,
        );
        assert_eq!(joint.bodies(), (6, 1));
        assert_eq!(JointBodies::bodies(&joint), (6, 1));
    }
}
