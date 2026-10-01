//! The cylindrical rigid-body joint.
//!
//! A [`CylindricalJoint`] couples body `b` to body `a` along a shared axis,
//! leaving exactly two relative degrees of freedom: body `b` may **slide** along
//! the axis and **spin** about it, but may not translate off the axis line or
//! tilt its own axis away from the shared one. It is the sleeve-over-a-rod
//! coupling — a piston in a bore, a drawer slide that is also free to roll, a
//! telescoping shaft that transmits no bending — the mechanism that permits a
//! single combined translate-and-twist along one line.
//!
//! It is the union of the two single-freedom sliders of the family: it keeps the
//! point-on-line positional restriction of the [`PrismaticJoint`](super::PrismaticJoint)
//! (which frees translation only) and the axis-alignment angular restriction of
//! the [`RevoluteJoint`](super::RevoluteJoint) (which frees spin only), so the
//! two freedoms coexist on the same axis. Locking the slide collapses it to a
//! hinge; locking the spin collapses it to a slider; locking both collapses it
//! to the [`FixedJoint`](super::FixedJoint).
//!
//! # The two constraints
//!
//! * **Axis alignment** (angular): with `u_a = rotate(orientation_a, axis_a)`
//!   and `u_b = rotate(orientation_b, axis_b)` the two unit axes, their cross
//!   product `u_a x u_b` (whose magnitude is `sin` of the angle between them) is
//!   driven to zero, holding `b`'s axis parallel to `a`'s and locking the two
//!   rotational freedoms perpendicular to the axis while leaving the spin about
//!   it free. This is the same alignment the [`RevoluteJoint`](super::RevoluteJoint)
//!   uses.
//! * **Point-on-line** (positional): with `dx = p_a - p_b` the world-space anchor
//!   separation (`p = position + rotate(orientation, anchor)`), only the
//!   component of `dx` *perpendicular* to the world-space axis is cancelled; the
//!   along-axis component is left free, so the two anchors are pinned to a common
//!   line rather than a common point. This is the same perpendicular weld the
//!   [`PrismaticJoint`](super::PrismaticJoint) uses.
//!
//! Each sweep projects the axis alignment first, then the point-on-line weld, so
//! the positional pass cancels the perpendicular offset against an
//! already-aligned axis.
//!
//! # Compliance
//!
//! [`compliance`](CylindricalJoint::compliance) is the inverse stiffness of the
//! perpendicular point-on-line weld (metres per newton) and
//! [`angular_compliance`](CylindricalJoint::angular_compliance) the inverse
//! stiffness of the axis alignment (radians per newton-metre). Zero is the rigid
//! limit for each; a positive value yields a soft, springy coupling of stiffness
//! `1 / compliance`. Each is divided by the squared substep time to form the
//! time-step-independent `XPBD` regularisation term, so the same compliance
//! behaves consistently across substep counts.
//!
//! # Scope
//!
//! The joint frees the slide and the spin but imposes no travel limit on either;
//! a one-sided along-axis stop (as the [`PrismaticLimitJoint`](super::PrismaticLimitJoint)
//! adds to the slider) and a spin limit or drive are genuine extensions rather
//! than reparametrisations and are left as future work.
//!
//! Provenance: the axis-alignment (orthogonality) angular constraint shared with
//! the revolute hinge and the perpendicular point-on-line positional constraint
//! shared with the prismatic slider, with their substep `XPBD` handling (Müller
//! et al., "Detailed Rigid Body Simulation with XPBD"), over the world-space
//! inverse inertia and quaternion kinematics of Baraff & Witkin. No Unreal Engine
//! source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use super::coloring::JointBodies;

/// A cylindrical joint coupling body `b` to body `a` along a shared axis,
/// leaving the slide along and the spin about that axis free while locking the
/// other four degrees of freedom.
///
/// The joint constrains the two world-space axes to stay parallel *and* the two
/// world-space anchor points to share a common line, removing two rotational
/// freedoms and two translational freedoms (the slide along the axis and the
/// spin about it remain free). Either body may be static (zero inverse mass and
/// inverse inertia); a joint between a dynamic body and a static one slides and
/// spins the dynamic sleeve along a fixed rod.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CylindricalJoint {
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
    /// [`axis_a`](CylindricalJoint::axis_a). Need not be unit length — it is
    /// normalised each sweep.
    pub axis_b: Vec3,
    /// Inverse stiffness (metres per newton) of the perpendicular point-on-line
    /// weld. Zero is the rigid limit.
    pub compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the axis alignment. Zero
    /// is the rigid limit.
    pub angular_compliance: f32,
}

impl CylindricalJoint {
    /// Creates a cylindrical joint between `body_a` and `body_b` with the given
    /// body-local anchors, body-local axes, and compliances.
    ///
    /// The two axes should be chosen so they start parallel in the bodies'
    /// initial poses; the alignment constraint drives any later deviation back to
    /// parallel. They need not be unit length — each is normalised every sweep.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "a cylindrical joint is defined by its two bodies, two anchors, \
                  two axes, and two compliances; grouping them into sub-structs \
                  would obscure the device packing"
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
    ) -> CylindricalJoint {
        CylindricalJoint {
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
    pub(crate) fn to_gpu(self) -> GpuCylindricalJoint {
        GpuCylindricalJoint {
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

impl JointBodies for CylindricalJoint {
    fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }
}

/// Device-packed [`CylindricalJoint`]; layout matches `Joint` in
/// `shaders/rigid_joint_cylindrical.wgsl` (`80` bytes). The two anchors and two
/// axes are padded to `vec4` so each starts on the `16`-byte boundary the storage
/// layout requires (the four `w` lanes are unused). The two indices and two
/// compliances fill the trailing `16`-byte block.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuCylindricalJoint {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_packing_round_trips_fields() {
        let joint = CylindricalJoint::new(
            2,
            7,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-4.0, 5.0, -6.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            0.003,
            0.0015,
        );
        let gpu = joint.to_gpu();
        assert_eq!(gpu.body_a, 2);
        assert_eq!(gpu.body_b, 7);
        assert_eq!(gpu.anchor_a, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(gpu.anchor_b, [-4.0, 5.0, -6.0, 0.0]);
        assert_eq!(gpu.axis_a, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(gpu.axis_b, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(gpu.compliance, 0.003);
        assert_eq!(gpu.angular_compliance, 0.0015);
    }

    #[test]
    fn gpu_struct_is_80_bytes() {
        assert_eq!(size_of::<GpuCylindricalJoint>(), 80);
    }

    #[test]
    fn bodies_returns_pair_in_order() {
        let joint = CylindricalJoint::new(4, 9, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y, 0.0, 0.0);
        assert_eq!(joint.bodies(), (4, 9));
        assert_eq!(JointBodies::bodies(&joint), (4, 9));
    }
}
