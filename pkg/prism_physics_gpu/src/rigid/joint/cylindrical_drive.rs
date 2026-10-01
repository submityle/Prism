//! The cylindrical joint with a linear drive (motor) along its slide axis.
//!
//! A [`CylindricalDriveJoint`] is a [`CylindricalJoint`](super::CylindricalJoint)
//! — the axis-alignment angular restriction plus the point-on-line positional
//! weld, leaving the slide along and the spin about the shared axis free — with
//! a *drive* layered on: an active spring-damper that pulls the free along-axis
//! separation toward a commanded
//! [`target_position`](CylindricalDriveJoint::target_position). The spin about
//! the axis stays free; only the slide is actively servoed. It is the powered
//! sleeve on a rod that may still roll — a motorised telescoping shaft, a piston
//! commanded to a stroke that transmits no torque, a quill drive.
//!
//! It differs from the [`PrismaticDriveJoint`](super::PrismaticDriveJoint) in
//! exactly the degree of freedom the base joint frees: the prismatic drive locks
//! *all three* relative rotations (a slider that may not roll) and drives the
//! single slide, whereas the cylindrical drive locks only the two rotations
//! *perpendicular* to the axis (keeping the axes parallel) and leaves the spin
//! about the axis free while driving the slide.
//!
//! # The three constraints
//!
//! * **Axis alignment** (angular): with `u_a = rotate(orientation_a, axis_a)`
//!   and `u_b = rotate(orientation_b, axis_b)` the two unit axes, their cross
//!   product `u_a x u_b` is driven to zero, holding `b`'s axis parallel to `a`'s
//!   and locking the two rotational freedoms perpendicular to the axis while
//!   leaving the spin about it free. This is the same alignment the
//!   [`RevoluteJoint`](super::RevoluteJoint) uses.
//! * **Point-on-line** (positional): with `dx = p_a - p_b` the world-space anchor
//!   separation (`p = position + rotate(orientation, anchor)`), only the
//!   component of `dx` *perpendicular* to the world-space axis is cancelled; the
//!   along-axis component is left free for the drive to command. This is the same
//!   perpendicular weld the [`PrismaticJoint`](super::PrismaticJoint) uses.
//! * **Linear drive** (positional, bilateral spring-damper): the signed
//!   along-axis separation `s = dx . axis_world` is actively driven toward
//!   [`target_position`](CylindricalDriveJoint::target_position). Unlike a travel
//!   limit, the drive is always active and pulls from either side — it is a
//!   two-sided servo, not a one-sided stop.
//!
//! Each sweep projects the axis alignment first (realigning the slide axis), then
//! the point-on-line weld (onto the freshly aligned axis), then the drive (along
//! it).
//!
//! # The drive model
//!
//! The drive is the `XPBD` compliant-and-damped equality constraint
//! `C = s - target_position`, projected along the world slide axis. Two tunables
//! shape it:
//!
//! * [`drive_compliance`](CylindricalDriveJoint::drive_compliance) is the inverse
//!   drive stiffness (metres per newton). Zero yields a *rigid position servo*:
//!   each substep snaps `s` exactly onto the target. A positive value yields a
//!   spring of stiffness `1 / drive_compliance` pulling toward the target.
//! * [`drive_damping`](CylindricalDriveJoint::drive_damping) is the drive damping
//!   coefficient (newton-seconds per metre), which resists the along-axis slide
//!   rate so the servo settles without ringing. Following the `XPBD` damping
//!   formulation its influence is scaled by `drive_compliance`
//!   (`gamma = drive_compliance * drive_damping / h`), so damping shapes the
//!   approach of a *soft* drive (positive compliance); a rigid servo
//!   (`drive_compliance == 0`) needs none and ignores it.
//!
//! This reproduces the position-drive mode of the `PhysX`/`Chaos` articulation
//! drives — a target position with spring and damping gains — within the
//! substep-`XPBD` scheme, applied to the cylindrical joint's single slide
//! freedom. A free-spinning velocity motor (zero stiffness, a pure velocity
//! target) and a drive on the *spin* freedom are velocity-level and angular
//! drives respectively and are intentionally out of scope for this positional
//! linear-drive stepper.
//!
//! # Measuring the slide position
//!
//! The drive needs a signed scalar position along the axis. The world-space
//! slide axis is `axis = rotate(orientation_a, axis_a)`, normalised to `u`; the
//! signed separation is `s = ((p_a + r_a) - (p_b + r_b)) . u`, exactly the
//! along-axis component the point-on-line weld leaves free. A larger `s` means
//! body `a`'s anchor sits further along `+u` from body `b`'s anchor.
//!
//! # Compliance
//!
//! [`compliance`](CylindricalDriveJoint::compliance) is the inverse stiffness of
//! the perpendicular point-on-line weld (metres per newton) and
//! [`angular_compliance`](CylindricalDriveJoint::angular_compliance) the inverse
//! stiffness of the axis alignment (radians per newton-metre). Zero is the rigid
//! limit for each; a positive value yields a soft, springy coupling of stiffness
//! `1 / compliance`. Each is divided by the squared substep time to form the
//! time-step-independent `XPBD` regularisation term, so the same compliance
//! behaves consistently across substep counts.
//!
//! Provenance: the axis-alignment (orthogonality) angular constraint shared with
//! the revolute hinge, the perpendicular point-on-line positional constraint
//! shared with the prismatic slider, and the bilateral along-axis drive shared
//! with the prismatic drive, with their substep compliant-and-damped `XPBD`
//! handling (Müller et al., "Detailed Rigid Body Simulation with XPBD"; Macklin
//! et al., "XPBD: Position-Based Simulation of Compliant Constrained Dynamics"),
//! over the world-space inverse inertia and quaternion kinematics of Baraff &
//! Witkin. No Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use super::coloring::JointBodies;

/// A cylindrical joint with a linear drive: the axis-alignment angular
/// restriction plus the perpendicular point-on-line weld of a
/// [`CylindricalJoint`](super::CylindricalJoint), plus a bilateral spring-damper
/// that servos the along-axis slide toward a commanded target.
///
/// The along-axis slide is actively driven rather than free; the spin about the
/// axis remains free. The drive pulls from either side toward
/// [`target_position`](CylindricalDriveJoint::target_position). Either body may
/// be static (zero inverse mass and inertia); a joint between a dynamic body and
/// a static one servos the dynamic sleeve along a fixed rod to a commanded stroke
/// while leaving it free to roll.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CylindricalDriveJoint {
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
    /// [`axis_a`](CylindricalDriveJoint::axis_a). Need not be unit length — it is
    /// normalised each sweep.
    pub axis_b: Vec3,
    /// Commanded along-axis separation the drive servos toward, in metres (the
    /// signed slide position `s` the drive pulls `s` onto).
    pub target_position: f32,
    /// Inverse stiffness (metres per newton) of the perpendicular point-on-line
    /// weld. Zero is the rigid limit.
    pub compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the axis alignment. Zero
    /// is the rigid limit.
    pub angular_compliance: f32,
    /// Inverse stiffness (metres per newton) of the drive. Zero is a rigid
    /// position servo; a positive value is a spring of stiffness
    /// `1 / drive_compliance` toward the target.
    pub drive_compliance: f32,
    /// Drive damping coefficient (newton-seconds per metre), resisting the
    /// along-axis slide rate. Scaled by [`drive_compliance`](Self::drive_compliance)
    /// per the `XPBD` damping formulation; ignored by a rigid servo.
    pub drive_damping: f32,
}

impl CylindricalDriveJoint {
    /// Creates a cylindrical drive joint from its full parameter set.
    ///
    /// The two axes should be chosen so they start parallel in the bodies'
    /// initial poses; the alignment constraint drives any later deviation back to
    /// parallel. They need not be unit length — each is normalised every sweep.
    /// `target_position` is the commanded along-axis separation. The compliances
    /// and the drive damping are the inverse stiffnesses and dashpot gain
    /// described on the fields.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "a cylindrical drive joint is defined by its two bodies, two \
                  anchors, two axes, target, and four solver gains; grouping them \
                  into sub-structs would obscure the device packing"
    )]
    pub fn new(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis_a: Vec3,
        axis_b: Vec3,
        target_position: f32,
        compliance: f32,
        angular_compliance: f32,
        drive_compliance: f32,
        drive_damping: f32,
    ) -> CylindricalDriveJoint {
        CylindricalDriveJoint {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            axis_b,
            target_position,
            compliance,
            angular_compliance,
            drive_compliance,
            drive_damping,
        }
    }

    /// Creates a rigid position servo: a drive with zero compliance everywhere,
    /// so each substep snaps the slide position exactly onto `target_position`
    /// while the sleeve stays free to spin and the axes stay rigidly aligned.
    #[must_use]
    pub fn servo(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis_a: Vec3,
        axis_b: Vec3,
        target_position: f32,
    ) -> CylindricalDriveJoint {
        CylindricalDriveJoint::new(
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            axis_b,
            target_position,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    }

    /// Creates a soft spring-damper drive: rigid alignment and perpendicular
    /// welds, a drive of the given stiffness (`drive_compliance =
    /// 1 / drive_stiffness`), and the given drive damping.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the spring constructor still names both bodies, both anchors, \
                  both axes, the target, and the two drive gains"
    )]
    pub fn spring(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis_a: Vec3,
        axis_b: Vec3,
        target_position: f32,
        drive_stiffness: f32,
        drive_damping: f32,
    ) -> CylindricalDriveJoint {
        let drive_compliance = if drive_stiffness > 0.0 {
            1.0 / drive_stiffness
        } else {
            0.0
        };
        CylindricalDriveJoint::new(
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            axis_b,
            target_position,
            0.0,
            0.0,
            drive_compliance,
            drive_damping,
        )
    }

    /// The two body indices this joint couples, in `(a, b)` order.
    #[must_use]
    pub fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }

    /// Packs the joint into its `GPU` storage-buffer representation.
    #[must_use]
    pub(crate) fn to_gpu(self) -> GpuCylindricalDriveJoint {
        GpuCylindricalDriveJoint {
            anchor_a: [self.anchor_a.x, self.anchor_a.y, self.anchor_a.z, 0.0],
            anchor_b: [self.anchor_b.x, self.anchor_b.y, self.anchor_b.z, 0.0],
            axis_a: [self.axis_a.x, self.axis_a.y, self.axis_a.z, 0.0],
            axis_b: [self.axis_b.x, self.axis_b.y, self.axis_b.z, 0.0],
            body_a: self.body_a,
            body_b: self.body_b,
            compliance: self.compliance,
            angular_compliance: self.angular_compliance,
            drive_compliance: self.drive_compliance,
            drive_damping: self.drive_damping,
            target_position: self.target_position,
            _pad0: 0.0,
        }
    }
}

impl JointBodies for CylindricalDriveJoint {
    fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }
}

/// Device-packed [`CylindricalDriveJoint`]; layout matches `Joint` in
/// `shaders/rigid_joint_cylindrical_drive.wgsl` (`96` bytes). The two anchors and
/// two axes are padded to `vec4` so each starts on the `16`-byte boundary the
/// storage layout requires (the four `w` lanes are unused). The two indices and
/// the two structural compliances fill the fifth `16`-byte block; the drive
/// compliance, drive damping, and target position fill the trailing one.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuCylindricalDriveJoint {
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
    /// Inverse stiffness (metres per newton) of the drive.
    drive_compliance: f32,
    /// Drive damping coefficient (newton-seconds per metre).
    drive_damping: f32,
    /// Commanded along-axis separation the drive servos toward, in metres.
    target_position: f32,
    /// Padding to the `16`-byte boundary; always zero.
    _pad0: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_packing_round_trips_fields() {
        let joint = CylindricalDriveJoint::new(
            2,
            7,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-4.0, 5.0, -6.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            0.5,
            0.003,
            0.0015,
            0.002,
            0.004,
        );
        let gpu = joint.to_gpu();
        assert_eq!(gpu.body_a, 2);
        assert_eq!(gpu.body_b, 7);
        assert_eq!(gpu.anchor_a, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(gpu.anchor_b, [-4.0, 5.0, -6.0, 0.0]);
        assert_eq!(gpu.axis_a, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(gpu.axis_b, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(gpu.target_position, 0.5);
        assert_eq!(gpu.compliance, 0.003);
        assert_eq!(gpu.angular_compliance, 0.0015);
        assert_eq!(gpu.drive_compliance, 0.002);
        assert_eq!(gpu.drive_damping, 0.004);
        assert_eq!(gpu._pad0, 0.0);
    }

    #[test]
    fn gpu_struct_is_96_bytes() {
        assert_eq!(size_of::<GpuCylindricalDriveJoint>(), 96);
    }

    #[test]
    fn servo_builds_a_rigid_zero_compliance_drive() {
        let joint =
            CylindricalDriveJoint::servo(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y, 0.25);
        assert_eq!(joint.target_position, 0.25);
        assert_eq!(joint.compliance, 0.0);
        assert_eq!(joint.angular_compliance, 0.0);
        assert_eq!(joint.drive_compliance, 0.0);
        assert_eq!(joint.drive_damping, 0.0);
    }

    #[test]
    fn spring_inverts_stiffness_into_compliance() {
        let joint = CylindricalDriveJoint::spring(
            0,
            1,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Y,
            Vec3::Y,
            0.4,
            500.0,
            10.0,
        );
        assert!((joint.drive_compliance - 1.0 / 500.0).abs() < 1.0e-9);
        assert_eq!(joint.drive_damping, 10.0);
        assert_eq!(joint.target_position, 0.4);
    }

    #[test]
    fn spring_with_zero_stiffness_is_rigid() {
        let joint = CylindricalDriveJoint::spring(
            0,
            1,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Y,
            Vec3::Y,
            0.0,
            0.0,
            5.0,
        );
        assert_eq!(joint.drive_compliance, 0.0);
    }

    #[test]
    fn bodies_returns_pair_in_order() {
        let joint =
            CylindricalDriveJoint::servo(4, 9, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y, 0.1);
        assert_eq!(joint.bodies(), (4, 9));
        assert_eq!(JointBodies::bodies(&joint), (4, 9));
    }
}
