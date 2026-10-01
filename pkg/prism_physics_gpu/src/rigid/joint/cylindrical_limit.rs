//! The cylindrical rigid-body joint with a travel limit along its slide axis.
//!
//! A [`CylindricalLimitJoint`] is a [`CylindricalJoint`](super::CylindricalJoint)
//! — the axis-alignment angular lock plus the perpendicular point-on-line weld
//! that free the slide along and the spin about a shared axis — with one more
//! restriction layered on: a *travel limit* along the slide axis. The free slide
//! a bare cylindrical joint permits is clamped to the closed interval
//! `[min_distance, max_distance]`, with the interior of that range left
//! completely free; the spin about the axis stays free throughout. It is the
//! telescoping shaft that bottoms out against its stops yet still spins, the
//! sleeve-over-a-rod bounded to a finite stroke, the piston in a bore with a
//! mechanical end-stop — a combined slide-and-spin coupling whose slide alone is
//! bounded.
//!
//! It is exactly the along-axis stop the bare [`CylindricalJoint`] documents as a
//! genuine extension: it keeps that joint's two constraints unchanged and adds a
//! third, the same one-sided clamp the
//! [`PrismaticLimitJoint`](super::PrismaticLimitJoint) layers onto the slider.
//!
//! # The three constraints
//!
//! * **Axis alignment** (angular): with `u_a = rotate(orientation_a, axis_a)`
//!   and `u_b = rotate(orientation_b, axis_b)` the two unit axes, their cross
//!   product `u_a x u_b` (whose magnitude is `sin` of the angle between them) is
//!   driven to zero, holding `b`'s axis parallel to `a`'s and locking the two
//!   rotational freedoms perpendicular to the axis while leaving the spin about
//!   it free. This is the same alignment the
//!   [`RevoluteJoint`](super::RevoluteJoint) uses.
//! * **Point-on-line** (positional): with `dx = p_a - p_b` the world-space anchor
//!   separation (`p = position + rotate(orientation, anchor)`), only the
//!   component of `dx` *perpendicular* to the world-space axis is cancelled; the
//!   along-axis component is left free, so the two anchors are pinned to a common
//!   line rather than a common point. This is the same perpendicular weld the
//!   [`PrismaticJoint`](super::PrismaticJoint) uses.
//! * **Travel limit** (positional, one-sided): the signed along-axis separation
//!   `s = dx . axis_world` is clamped into `[min_distance, max_distance]`. When
//!   `min_distance <= s <= max_distance` the limit is inactive and exerts no
//!   force — the sleeve slides freely inside the range; outside it the violated
//!   bound is driven shut with a signed correction along the slide axis.
//!
//! Each sweep projects the axis alignment first (realigning the slide axis), then
//! the point-on-line weld (onto the freshly oriented axis), then the travel
//! limit (along it).
//!
//! # Measuring the slide position
//!
//! The limit needs a signed scalar position along the axis. The world-space
//! slide axis is `axis = rotate(orientation_a, axis_a)`, normalised to `u`; the
//! signed separation is `s = ((p_a + r_a) - (p_b + r_b)) . u`, exactly the
//! along-axis component the point-on-line weld leaves free. A larger `s` means
//! body `a`'s anchor sits further along `+u` from body `b`'s anchor.
//!
//! # Compliance
//!
//! [`compliance`](CylindricalLimitJoint::compliance) is the inverse stiffness of
//! the perpendicular point-on-line weld (metres per newton),
//! [`angular_compliance`](CylindricalLimitJoint::angular_compliance) the inverse
//! stiffness of the axis alignment (radians per newton-metre), and
//! [`limit_compliance`](CylindricalLimitJoint::limit_compliance) the inverse
//! stiffness of the travel limit (metres per newton). Zero is the rigid limit
//! for each; a positive value yields a soft, springy stop of stiffness
//! `1 / compliance`. Each is divided by the squared substep time to form the
//! time-step-independent `XPBD` regularisation term, so the same compliance
//! behaves consistently across substep counts.
//!
//! # Scope
//!
//! The joint bounds the slide but leaves the spin about the axis completely free;
//! a spin limit or spin drive is a genuine extension rather than a
//! reparametrisation and is left as future work, as is a soft spring-back inside
//! the travel range.
//!
//! Provenance: the axis-alignment (orthogonality) angular constraint shared with
//! the revolute hinge, the perpendicular point-on-line positional constraint
//! shared with the prismatic slider, and the one-sided along-axis limit shared
//! with the prismatic limit, with their substep `XPBD` handling (Müller et al.,
//! "Detailed Rigid Body Simulation with XPBD"), over the world-space inverse
//! inertia and quaternion kinematics of Baraff & Witkin. No Unreal Engine source
//! or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use super::coloring::JointBodies;

/// A cylindrical joint with a travel limit: an axis-alignment angular lock plus
/// a perpendicular point-on-line weld plus a one-sided clamp of the along-axis
/// separation into `[min_distance, max_distance]`.
///
/// The interior of the travel range is free and the spin about the axis is
/// always free; only a violated travel bound exerts a force. Either body may be
/// static (zero inverse mass and inverse inertia); a joint between a dynamic body
/// and a static one slides and spins the dynamic sleeve along a fixed rod,
/// bounded to a fixed stroke.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CylindricalLimitJoint {
    /// Index of the first body in the [`RigidBodyState`](super::super::RigidBodyState).
    pub body_a: u32,
    /// Index of the second body in the [`RigidBodyState`](super::super::RigidBodyState).
    pub body_b: u32,
    /// Anchor on body `a`, in body `a`'s local frame (relative to its centre of
    /// mass). The shared axis line passes through this point.
    pub anchor_a: Vec3,
    /// Anchor on body `b`, in body `b`'s local frame.
    pub anchor_b: Vec3,
    /// Slide-and-spin axis fixed in body `a`'s local frame; the free
    /// translational and rotational direction. Need not be unit length — it is
    /// normalised each sweep.
    pub axis_a: Vec3,
    /// Spin axis fixed in body `b`'s local frame; held parallel to
    /// [`axis_a`](CylindricalLimitJoint::axis_a). Need not be unit length — it is
    /// normalised each sweep.
    pub axis_b: Vec3,
    /// Lower bound of the free travel range, in metres. Signed along `+axis_a`.
    pub min_distance: f32,
    /// Upper bound of the free travel range, in metres. Signed along `+axis_a`.
    pub max_distance: f32,
    /// Inverse stiffness (metres per newton) of the perpendicular point-on-line
    /// weld. Zero is the rigid limit.
    pub compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the axis alignment. Zero
    /// is the rigid limit.
    pub angular_compliance: f32,
    /// Inverse stiffness (metres per newton) of the travel limit. Zero is the
    /// rigid limit.
    pub limit_compliance: f32,
}

impl CylindricalLimitJoint {
    /// Creates a cylindrical limit joint between `body_a` and `body_b` with the
    /// given body-local anchors, body-local axes, travel bounds, and
    /// compliances.
    ///
    /// The two axes should be chosen so they start parallel in the bodies'
    /// initial poses; the alignment constraint drives any later deviation back to
    /// parallel. They need not be unit length — each is normalised every sweep.
    /// `min_distance` must not exceed `max_distance`; a degenerate range
    /// (`min == max`) pins the slide to a single point like a rigid weld along
    /// the axis.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "a cylindrical limit joint is defined by its two bodies, two \
                  anchors, two axes, two travel bounds, and three compliances; \
                  grouping them into sub-structs would obscure the device packing"
    )]
    pub fn new(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis_a: Vec3,
        axis_b: Vec3,
        min_distance: f32,
        max_distance: f32,
        compliance: f32,
        angular_compliance: f32,
        limit_compliance: f32,
    ) -> CylindricalLimitJoint {
        CylindricalLimitJoint {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            axis_b,
            min_distance,
            max_distance,
            compliance,
            angular_compliance,
            limit_compliance,
        }
    }

    /// Creates a rigid cylindrical limit joint with a travel range centred on the
    /// anchors' initial along-axis separation: `[-half_travel, +half_travel]`.
    ///
    /// All three compliances are zero (the rigid limit). This is the common
    /// symmetric stop — a sleeve free to slide `half_travel` either way from its
    /// start and to spin without bound.
    #[must_use]
    pub fn symmetric(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis_a: Vec3,
        axis_b: Vec3,
        half_travel: f32,
    ) -> CylindricalLimitJoint {
        CylindricalLimitJoint::new(
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            axis_b,
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
    pub(crate) fn to_gpu(self) -> GpuCylindricalLimitJoint {
        GpuCylindricalLimitJoint {
            anchor_a: [self.anchor_a.x, self.anchor_a.y, self.anchor_a.z, 0.0],
            anchor_b: [self.anchor_b.x, self.anchor_b.y, self.anchor_b.z, 0.0],
            axis_a: [self.axis_a.x, self.axis_a.y, self.axis_a.z, 0.0],
            axis_b: [self.axis_b.x, self.axis_b.y, self.axis_b.z, 0.0],
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

impl JointBodies for CylindricalLimitJoint {
    fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }
}

/// Device-packed [`CylindricalLimitJoint`]; layout matches `Joint` in
/// `shaders/rigid_joint_cylindrical_limit.wgsl` (`96` bytes). The two anchors and
/// two axes are padded to `vec4` so each starts on the `16`-byte boundary the
/// storage layout requires (the four `w` lanes are unused). The two indices,
/// three compliances, and the two travel bounds fill the trailing two `16`-byte
/// blocks.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuCylindricalLimitJoint {
    /// Body-local anchor on body `a` in `xyz`; `w` unused.
    anchor_a: [f32; 4],
    /// Body-local anchor on body `b` in `xyz`; `w` unused.
    anchor_b: [f32; 4],
    /// Body-local slide-and-spin axis on body `a` in `xyz`; `w` unused.
    axis_a: [f32; 4],
    /// Body-local spin axis on body `b` in `xyz`; `w` unused.
    axis_b: [f32; 4],
    /// Index of body `a`.
    body_a: u32,
    /// Index of body `b`.
    body_b: u32,
    /// Inverse stiffness (metres per newton) of the perpendicular point-on-line
    /// weld.
    compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the axis alignment.
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
        let joint = CylindricalLimitJoint::new(
            2,
            7,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-4.0, 5.0, -6.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            -0.5,
            0.75,
            0.003,
            0.0015,
            0.002,
        );
        let gpu = joint.to_gpu();
        assert_eq!(gpu.body_a, 2);
        assert_eq!(gpu.body_b, 7);
        assert_eq!(gpu.anchor_a, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(gpu.anchor_b, [-4.0, 5.0, -6.0, 0.0]);
        assert_eq!(gpu.axis_a, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(gpu.axis_b, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(gpu.min_distance, -0.5);
        assert_eq!(gpu.max_distance, 0.75);
        assert_eq!(gpu.compliance, 0.003);
        assert_eq!(gpu.angular_compliance, 0.0015);
        assert_eq!(gpu.limit_compliance, 0.002);
        assert_eq!(gpu._pad0, 0.0);
    }

    #[test]
    fn gpu_struct_is_96_bytes() {
        assert_eq!(size_of::<GpuCylindricalLimitJoint>(), 96);
    }

    #[test]
    fn symmetric_builds_a_centred_rigid_range() {
        let joint =
            CylindricalLimitJoint::symmetric(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y, 0.5);
        assert_eq!(joint.min_distance, -0.5);
        assert_eq!(joint.max_distance, 0.5);
        assert_eq!(joint.compliance, 0.0);
        assert_eq!(joint.angular_compliance, 0.0);
        assert_eq!(joint.limit_compliance, 0.0);
    }

    #[test]
    fn bodies_returns_pair_in_order() {
        let joint =
            CylindricalLimitJoint::symmetric(4, 9, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y, 0.5);
        assert_eq!(joint.bodies(), (4, 9));
        assert_eq!(JointBodies::bodies(&joint), (4, 9));
    }
}
