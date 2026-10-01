//! The prismatic (slider) joint with a travel limit along its slide axis.
//!
//! A [`PrismaticLimitJoint`] is a prismatic (slider) joint — the relative
//! orientation lock plus the perpendicular weld of a
//! [`PrismaticJoint`](super::PrismaticJoint) — with one more restriction layered
//! on: a *travel limit* along the slide axis. The free slide a bare prismatic
//! joint permits is clamped to the closed interval `[min_distance,
//! max_distance]`, with the interior of that range left completely free. It is
//! the drawer that stops against its runner's ends, the piston bounded by its
//! cylinder, and the linear actuator with a bounded stroke — the single sliding
//! degree of freedom bounded on one or both sides.
//!
//! # The three constraints
//!
//! * **Angular lock** (angular): the relative orientation of body `b` in body
//!   `a`'s frame is driven to the fixed
//!   [`rest_rotation`](PrismaticLimitJoint::rest_rotation), locking all three
//!   relative rotational degrees of freedom.
//! * **Perpendicular weld** (positional): the world-space anchor separation
//!   `dx = p_a - p_b` (with `p = position + rotate(orientation, anchor)`) is
//!   driven to zero *only in the plane perpendicular to the slide axis*, leaving
//!   the along-axis separation free.
//! * **Travel limit** (positional, one-sided): the signed along-axis separation
//!   `s = dx . axis_world` is clamped into `[min_distance, max_distance]`. When
//!   `min_distance <= s <= max_distance` the limit is inactive and exerts no
//!   force — the body slides freely inside the range; outside it the violated
//!   bound is driven shut with a signed correction along the slide axis.
//!
//! Each sweep projects the angular lock first (realigning the slide axis), then
//! the perpendicular weld (onto the freshly oriented axis), then the travel
//! limit (along it).
//!
//! # Measuring the slide position
//!
//! The limit needs a signed scalar position along the axis. The world-space
//! slide axis is `axis = rotate(orientation_a, axis_a)`, normalised to `u`; the
//! signed separation is `s = ((p_a + r_a) - (p_b + r_b)) . u`, exactly the
//! along-axis component the perpendicular weld leaves free. A larger `s` means
//! body `a`'s anchor sits further along `+u` from body `b`'s anchor.
//!
//! # Compliance
//!
//! [`compliance`](PrismaticLimitJoint::compliance) is the inverse stiffness of
//! the perpendicular weld (metres per newton),
//! [`angular_compliance`](PrismaticLimitJoint::angular_compliance) the inverse
//! stiffness of the angular lock (radians per newton-metre), and
//! [`limit_compliance`](PrismaticLimitJoint::limit_compliance) the inverse
//! stiffness of the travel limit (metres per newton). Zero is the rigid limit
//! for each; a positive value yields a soft, springy stop of stiffness
//! `1 / compliance`. Each is divided by the squared substep time to form the
//! time-step-independent `XPBD` regularisation term, so the same compliance
//! behaves consistently across substep counts.
//!
//! Provenance: the point-to-point (ball-socket) constraint restricted to the
//! plane perpendicular to the slide axis, the relative-orientation lock, and the
//! one-sided along-axis limit with their substep `XPBD` handling (Müller et al.,
//! "Detailed Rigid Body Simulation with XPBD"), over the world-space inverse
//! inertia and quaternion kinematics of Baraff & Witkin. No Unreal Engine source
//! or derived code.

use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};

use super::coloring::JointBodies;

/// A prismatic (slider) joint with a travel limit: a relative-orientation lock
/// plus a perpendicular weld plus a one-sided clamp of the along-axis separation
/// into `[min_distance, max_distance]`.
///
/// The interior of the travel range is free; only a violated bound exerts a
/// force. Either body may be static (zero inverse mass and inertia); a joint
/// between a dynamic body and a static one slides the dynamic body along a fixed
/// world axis through a fixed world point, bounded to a fixed stroke.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrismaticLimitJoint {
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
    /// Lower bound of the free travel range, in metres along the slide axis.
    pub min_distance: f32,
    /// Upper bound of the free travel range, in metres along the slide axis.
    pub max_distance: f32,
    /// Inverse stiffness (metres per newton) of the perpendicular weld. Zero is
    /// the rigid limit.
    pub compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the angular lock. Zero is
    /// the rigid limit.
    pub angular_compliance: f32,
    /// Inverse stiffness (metres per newton) of the travel limit. Zero is the
    /// rigid limit.
    pub limit_compliance: f32,
}

impl PrismaticLimitJoint {
    /// Creates a prismatic travel-limit joint between `body_a` and `body_b` with
    /// the given body-local anchors, body-local slide axis, rest relative
    /// orientation, travel bounds, and compliances.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "a limited slider is defined by two anchors, a slide axis, a \
                  rest orientation, two travel bounds, and three compliances \
                  across the two bodies it couples"
    )]
    pub fn new(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis_a: Vec3,
        rest_rotation: Quat,
        min_distance: f32,
        max_distance: f32,
        compliance: f32,
        angular_compliance: f32,
        limit_compliance: f32,
    ) -> PrismaticLimitJoint {
        PrismaticLimitJoint {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            rest_rotation,
            min_distance,
            max_distance,
            compliance,
            angular_compliance,
            limit_compliance,
        }
    }

    /// Creates a rigid slider centred on the construction configuration with a
    /// symmetric travel range `[-half_travel, half_travel]` and identity rest
    /// rotation — the common case of a stroke-bounded actuator starting at the
    /// middle of its range.
    #[must_use]
    pub fn symmetric(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis_a: Vec3,
        half_travel: f32,
    ) -> PrismaticLimitJoint {
        PrismaticLimitJoint::new(
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            Quat::IDENTITY,
            -half_travel,
            half_travel,
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
    pub(crate) fn to_gpu(self) -> GpuPrismaticLimitJoint {
        GpuPrismaticLimitJoint {
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
            limit_compliance: self.limit_compliance,
            min_distance: self.min_distance,
            max_distance: self.max_distance,
            _pad0: 0.0,
        }
    }
}

impl JointBodies for PrismaticLimitJoint {
    fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }
}

/// Device-packed [`PrismaticLimitJoint`]; layout matches `Joint` in
/// `shaders/rigid_joint_prismatic_limit.wgsl` (`96` bytes). The two anchors and
/// the slide axis are padded to `vec4` so each starts on the `16`-byte boundary
/// the storage layout requires (the three `w` lanes are unused); the
/// rest-rotation quaternion fills a fourth `vec4` as `(x, y, z, w)`. The two
/// indices, three compliances, and the two travel bounds fill the trailing two
/// `16`-byte blocks.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuPrismaticLimitJoint {
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
    /// Inverse stiffness (metres per newton) of the travel limit.
    limit_compliance: f32,
    /// Lower bound of the free travel range, in metres.
    min_distance: f32,
    /// Upper bound of the free travel range, in metres.
    max_distance: f32,
    /// Padding to the `16`-byte boundary; always zero.
    _pad0: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_packing_round_trips_fields() {
        let joint = PrismaticLimitJoint::new(
            2,
            9,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-4.0, 5.0, -6.0),
            Vec3::new(0.0, 1.0, 0.0),
            Quat::from_xyzw(0.1, 0.2, 0.3, 0.4),
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
        assert_eq!(gpu.rest_rotation, [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(gpu.min_distance, -0.5);
        assert_eq!(gpu.max_distance, 0.75);
        assert_eq!(gpu.compliance, 0.002);
        assert_eq!(gpu.angular_compliance, 0.001);
        assert_eq!(gpu.limit_compliance, 0.003);
        assert_eq!(gpu._pad0, 0.0);
    }

    #[test]
    fn gpu_struct_is_96_bytes() {
        assert_eq!(size_of::<GpuPrismaticLimitJoint>(), 96);
    }

    #[test]
    fn symmetric_builds_a_centred_rigid_range() {
        let joint = PrismaticLimitJoint::symmetric(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Y, 0.5);
        assert_eq!(joint.min_distance, -0.5);
        assert_eq!(joint.max_distance, 0.5);
        assert_eq!(joint.rest_rotation, Quat::IDENTITY);
        assert_eq!(joint.compliance, 0.0);
        assert_eq!(joint.angular_compliance, 0.0);
        assert_eq!(joint.limit_compliance, 0.0);
    }

    #[test]
    fn bodies_returns_pair_in_order() {
        let joint = PrismaticLimitJoint::symmetric(6, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Y, 0.5);
        assert_eq!(joint.bodies(), (6, 1));
        assert_eq!(JointBodies::bodies(&joint), (6, 1));
    }
}
