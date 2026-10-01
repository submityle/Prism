//! The universal (Cardan / Hooke) rigid-body joint.
//!
//! A [`UniversalJoint`] couples body `b` to body `a` through a cross-shaped
//! gimbal: it pins the two bodies at a shared pivot and forces one body-fixed
//! axis on each body to stay mutually perpendicular, transmitting rotation
//! between two shafts whose axes meet at an angle while still permitting each
//! shaft to spin about its own axis. It is the drive-shaft coupling of a
//! vehicle, the wrist of a gimbal, and the Cardan joint of a steering column —
//! the mechanism that carries torque across a bend.
//!
//! It is the mechanical dual of the [`RevoluteJoint`](super::RevoluteJoint): a
//! hinge *locks* the relative swing and leaves one twist free, whereas a
//! universal joint keeps the two drive axes orthogonal and leaves **both**
//! shafts free to spin about themselves. Removing one of its two rotational
//! freedoms collapses it toward a hinge; adding an orientation lock collapses it
//! toward the [`FixedJoint`](super::FixedJoint). It is also the natural stepping
//! stone toward a configurable `6`-DOF joint.
//!
//! # The two constraints
//!
//! * **Perpendicularity** (angular): with
//!   `u_a = rotate(orientation_a, axis_a)` and
//!   `u_b = rotate(orientation_b, axis_b)` the two unit drive axes, the angle
//!   between them `phi = acos(clamp(u_a . u_b, -1, 1))` is driven to a right
//!   angle. The *signed* violation `c = phi - pi/2` (negative when the axes are
//!   closer than perpendicular, positive when they have opened past it) is
//!   cancelled by rotating about `n = normalize(u_a x u_b)`, with the gradient
//!   `-n` on body `a` and `+n` on body `b` — the same antisymmetric pattern the
//!   swing cone of the [`SwingTwistJoint`](super::SwingTwistJoint) uses, since
//!   `u_b` is `u_a` rotated by `+phi` about `n`. The constraint is two-sided: it
//!   pulls the axes apart when they close and together when they open, holding
//!   the right angle from either side.
//! * **Point-to-point weld** (positional): the full world-space anchor
//!   separation `dx = p_a - p_b` (with `p = position + rotate(orientation, anchor)`)
//!   is driven to zero, pinning the two bodies at the shared gimbal centre and
//!   locking all three relative translational degrees of freedom. This is the
//!   same ball-socket weld the [`SphericalJoint`](super::SphericalJoint) uses.
//!
//! Each sweep projects the perpendicularity constraint first, then the
//! positional weld, so the anchors are pulled together against an already-gimbaled
//! frame.
//!
//! # Compliance
//!
//! [`compliance`](UniversalJoint::compliance) is the inverse stiffness of the
//! positional weld (metres per newton) and
//! [`angular_compliance`](UniversalJoint::angular_compliance) the inverse
//! stiffness of the perpendicularity constraint (radians per newton-metre). Zero
//! is the rigid limit for each; a positive value yields a soft, springy coupling
//! of stiffness `1 / compliance`. Each is divided by the squared substep time to
//! form the time-step-independent `XPBD` regularisation term, so the same
//! compliance behaves consistently across substep counts.
//!
//! # Scope
//!
//! The gimbal axes `axis_a` and `axis_b` are the two shafts' drive axes; the
//! constraint keeps them orthogonal but does **not** itself model the
//! characteristic non-constant velocity ratio of a single Cardan joint (the
//! reason real drive-trains use a double Cardan or a constant-velocity joint).
//! That velocity ratio is an emergent property of the kinematics the two bodies
//! settle into, not a separate constraint, so nothing here fakes a
//! constant-velocity coupling. A gear-ratio coupling between the two shafts'
//! spins is a genuine extension rather than a reparametrisation and is left as
//! future work.
//!
//! Provenance: the point-to-point (ball-socket) constraint and the
//! perpendicularity (orthogonality) constraint over two body-fixed axes with
//! their substep `XPBD` handling (Müller et al., "Detailed Rigid Body Simulation
//! with XPBD"), over the world-space inverse inertia and quaternion kinematics of
//! Baraff & Witkin. No Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use super::coloring::JointBodies;

/// A universal (Cardan / Hooke) joint pinning body `b` to body `a` at a shared
/// pivot and holding one body-fixed drive axis on each body mutually
/// perpendicular.
///
/// The joint constrains the two world-space anchor points to coincide *and* the
/// two world-space drive axes to stay orthogonal, removing the three
/// translational freedoms and one of the three rotational freedoms (the two
/// shafts remain free to spin about themselves and to flex at the gimbal).
/// Either body may be static (zero inverse mass and inverse inertia); a joint
/// between a dynamic body and a static one gimbals the dynamic shaft off a fixed
/// mount.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UniversalJoint {
    /// Index of the first body in the [`RigidBodyState`](super::super::RigidBodyState).
    pub body_a: u32,
    /// Index of the second body in the [`RigidBodyState`](super::super::RigidBodyState).
    pub body_b: u32,
    /// Anchor (gimbal centre) on body `a`, in body `a`'s local frame (relative
    /// to its centre of mass).
    pub anchor_a: Vec3,
    /// Anchor (gimbal centre) on body `b`, in body `b`'s local frame.
    pub anchor_b: Vec3,
    /// Drive axis fixed in body `a`'s local frame; held perpendicular to
    /// [`axis_b`](UniversalJoint::axis_b). Need not be unit length — it is
    /// normalised each sweep.
    pub axis_a: Vec3,
    /// Drive axis fixed in body `b`'s local frame; held perpendicular to
    /// [`axis_a`](UniversalJoint::axis_a). Need not be unit length — it is
    /// normalised each sweep.
    pub axis_b: Vec3,
    /// Inverse stiffness (metres per newton) of the positional weld. Zero is the
    /// rigid limit.
    pub compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the perpendicularity
    /// constraint. Zero is the rigid limit.
    pub angular_compliance: f32,
}

impl UniversalJoint {
    /// Creates a universal joint between `body_a` and `body_b` with the given
    /// body-local anchors, body-local drive axes, and compliances.
    ///
    /// The two axes should be chosen so they start mutually perpendicular in the
    /// bodies' initial poses; the constraint drives any later deviation back to a
    /// right angle. They need not be unit length — each is normalised every
    /// sweep.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "a universal joint is defined by its two bodies, two \
                  anchors, two drive axes, and two compliances; grouping \
                  them into sub-structs would obscure the device packing"
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
    ) -> UniversalJoint {
        UniversalJoint {
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
    pub(crate) fn to_gpu(self) -> GpuUniversalJoint {
        GpuUniversalJoint {
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

impl JointBodies for UniversalJoint {
    fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }
}

/// Device-packed [`UniversalJoint`]; layout matches `Joint` in
/// `shaders/rigid_joint_universal.wgsl` (`80` bytes). The two anchors and two
/// drive axes are padded to `vec4` so each starts on the `16`-byte boundary the
/// storage layout requires (the four `w` lanes are unused). The two indices and
/// two compliances fill the trailing `16`-byte block.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuUniversalJoint {
    /// Body-local anchor on body `a` in `xyz`; `w` unused.
    anchor_a: [f32; 4],
    /// Body-local anchor on body `b` in `xyz`; `w` unused.
    anchor_b: [f32; 4],
    /// Body-local drive axis on body `a` in `xyz`; `w` unused.
    axis_a: [f32; 4],
    /// Body-local drive axis on body `b` in `xyz`; `w` unused.
    axis_b: [f32; 4],
    /// Index of body `a`.
    body_a: u32,
    /// Index of body `b`.
    body_b: u32,
    /// Inverse stiffness (metres per newton) of the positional weld.
    compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the perpendicularity
    /// constraint.
    angular_compliance: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_packing_round_trips_fields() {
        let joint = UniversalJoint::new(
            3,
            8,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-4.0, 5.0, -6.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.002,
            0.001,
        );
        let gpu = joint.to_gpu();
        assert_eq!(gpu.body_a, 3);
        assert_eq!(gpu.body_b, 8);
        assert_eq!(gpu.anchor_a, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(gpu.anchor_b, [-4.0, 5.0, -6.0, 0.0]);
        assert_eq!(gpu.axis_a, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(gpu.axis_b, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(gpu.compliance, 0.002);
        assert_eq!(gpu.angular_compliance, 0.001);
    }

    #[test]
    fn gpu_struct_is_80_bytes() {
        assert_eq!(size_of::<GpuUniversalJoint>(), 80);
    }

    #[test]
    fn bodies_returns_pair_in_order() {
        let joint = UniversalJoint::new(6, 1, Vec3::ZERO, Vec3::ZERO, Vec3::X, Vec3::Z, 0.0, 0.0);
        assert_eq!(joint.bodies(), (6, 1));
        assert_eq!(JointBodies::bodies(&joint), (6, 1));
    }
}
