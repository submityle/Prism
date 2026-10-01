//! The prismatic (slider) joint with a linear drive (motor) along its slide
//! axis.
//!
//! A [`PrismaticDriveJoint`] is a prismatic (slider) joint — the relative
//! orientation lock plus the perpendicular weld of a
//! [`PrismaticJoint`](super::PrismaticJoint) — with a *drive* layered on: an
//! active spring-damper that pulls the free along-axis separation toward a
//! commanded [`target_position`](PrismaticDriveJoint::target_position). It is
//! the powered linear actuator, the motorised drawer, and the hydraulic ram
//! commanded to a stroke — the single sliding degree of freedom actively
//! servoed rather than merely free.
//!
//! # The three constraints
//!
//! * **Angular lock** (angular): the relative orientation of body `b` in body
//!   `a`'s frame is driven to the fixed
//!   [`rest_rotation`](PrismaticDriveJoint::rest_rotation), locking all three
//!   relative rotational degrees of freedom.
//! * **Perpendicular weld** (positional): the world-space anchor separation
//!   `dx = p_a - p_b` (with `p = position + rotate(orientation, anchor)`) is
//!   driven to zero *only in the plane perpendicular to the slide axis*, leaving
//!   the along-axis separation free for the drive to command.
//! * **Linear drive** (positional, bilateral spring-damper): the signed
//!   along-axis separation `s = dx . axis_world` is actively driven toward
//!   [`target_position`](PrismaticDriveJoint::target_position). Unlike a travel
//!   limit, the drive is always active and pulls from either side — it is a
//!   two-sided servo, not a one-sided stop.
//!
//! Each sweep projects the angular lock first (realigning the slide axis), then
//! the perpendicular weld (onto the freshly oriented axis), then the drive
//! (along it).
//!
//! # The drive model
//!
//! The drive is the `XPBD` compliant-and-damped equality constraint
//! `C = s - target_position`, projected along the world slide axis. Two tunables
//! shape it:
//!
//! * [`drive_compliance`](PrismaticDriveJoint::drive_compliance) is the inverse
//!   drive stiffness (metres per newton). Zero yields a *rigid position servo*:
//!   each substep snaps `s` exactly onto the target. A positive value yields a
//!   spring of stiffness `1 / drive_compliance` pulling toward the target.
//! * [`drive_damping`](PrismaticDriveJoint::drive_damping) is the drive damping
//!   coefficient (newton-seconds per metre), which resists the along-axis slide
//!   rate so the servo settles without ringing. Following the `XPBD` damping
//!   formulation its influence is scaled by `drive_compliance`
//!   (`gamma = drive_compliance * drive_damping / h`), so damping shapes the
//!   approach of a *soft* drive (positive compliance); a rigid servo
//!   (`drive_compliance == 0`) needs none and ignores it.
//!
//! This reproduces the position-drive mode of the `PhysX`/`Chaos` articulation
//! drives — a target position with spring and damping gains — within the
//! substep-`XPBD` scheme. A free-spinning velocity motor (zero stiffness, a pure
//! velocity target) is a velocity-level drive and is intentionally out of scope
//! for this positional stepper.
//!
//! # Measuring the slide position
//!
//! The drive needs a signed scalar position along the axis. The world-space
//! slide axis is `axis = rotate(orientation_a, axis_a)`, normalised to `u`; the
//! signed separation is `s = ((p_a + r_a) - (p_b + r_b)) . u`, exactly the
//! along-axis component the perpendicular weld leaves free. A larger `s` means
//! body `a`'s anchor sits further along `+u` from body `b`'s anchor.
//!
//! # Compliance
//!
//! [`compliance`](PrismaticDriveJoint::compliance) is the inverse stiffness of
//! the perpendicular weld (metres per newton) and
//! [`angular_compliance`](PrismaticDriveJoint::angular_compliance) the inverse
//! stiffness of the angular lock (radians per newton-metre). Zero is the rigid
//! limit for each; a positive value yields a soft constraint of stiffness
//! `1 / compliance`. Each compliance is divided by the squared substep time to
//! form the time-step-independent `XPBD` regularisation term, so the same
//! compliance behaves consistently across substep counts.
//!
//! Provenance: the point-to-point (ball-socket) constraint restricted to the
//! plane perpendicular to the slide axis, the relative-orientation lock, and the
//! bilateral along-axis drive with their substep compliant-and-damped `XPBD`
//! handling (Müller et al., "Detailed Rigid Body Simulation with XPBD"; Macklin
//! et al., "XPBD: Position-Based Simulation of Compliant Constrained Dynamics"),
//! over the world-space inverse inertia and quaternion kinematics of Baraff &
//! Witkin. No Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};

use super::coloring::JointBodies;

/// A prismatic (slider) joint with a linear drive: a relative-orientation lock
/// plus a perpendicular weld plus a bilateral spring-damper that servos the
/// along-axis separation toward a commanded target.
///
/// The along-axis slide is actively driven rather than free; the drive pulls
/// from either side toward [`target_position`](PrismaticDriveJoint::target_position).
/// Either body may be static (zero inverse mass and inertia); a joint between a
/// dynamic body and a static one servos the dynamic body along a fixed world
/// axis through a fixed world point to a commanded stroke.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrismaticDriveJoint {
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
    /// Commanded along-axis separation the drive servos toward, in metres (the
    /// signed slide position `s` the drive pulls `s` onto).
    pub target_position: f32,
    /// Inverse stiffness (metres per newton) of the perpendicular weld. Zero is
    /// rigid.
    pub compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the angular lock. Zero is
    /// rigid.
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

impl PrismaticDriveJoint {
    /// Creates a prismatic drive joint from its full parameter set.
    ///
    /// `axis_a` is the slide axis in body `a`'s local frame (its direction is
    /// all that matters). `rest_rotation` is the relative orientation the
    /// angular lock holds. `target_position` is the commanded along-axis
    /// separation. The three compliances and the drive damping are the inverse
    /// stiffnesses and dashpot gain described on the fields.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "a drive joint is defined by its two bodies, two anchors, slide \
                  axis, rest rotation, target, and four solver gains; grouping \
                  them into sub-structs would obscure the device packing"
    )]
    pub fn new(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis_a: Vec3,
        rest_rotation: Quat,
        target_position: f32,
        compliance: f32,
        angular_compliance: f32,
        drive_compliance: f32,
        drive_damping: f32,
    ) -> PrismaticDriveJoint {
        PrismaticDriveJoint {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            rest_rotation,
            target_position,
            compliance,
            angular_compliance,
            drive_compliance,
            drive_damping,
        }
    }

    /// Creates a rigid position servo: a drive with zero compliance everywhere
    /// and an identity rest rotation, so each substep snaps the slide position
    /// exactly onto `target_position`.
    #[must_use]
    pub fn servo(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis_a: Vec3,
        target_position: f32,
    ) -> PrismaticDriveJoint {
        PrismaticDriveJoint::new(
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            Quat::IDENTITY,
            target_position,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    }

    /// Creates a soft spring-damper drive: an identity rest rotation with rigid
    /// orientation and perpendicular welds, a drive of the given stiffness
    /// (`drive_compliance = 1 / drive_stiffness`), and the given drive damping.
    #[must_use]
    pub fn spring(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis_a: Vec3,
        target_position: f32,
        drive_stiffness: f32,
        drive_damping: f32,
    ) -> PrismaticDriveJoint {
        let drive_compliance = if drive_stiffness > 0.0 {
            1.0 / drive_stiffness
        } else {
            0.0
        };
        PrismaticDriveJoint::new(
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            Quat::IDENTITY,
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
    pub(crate) fn to_gpu(self) -> GpuPrismaticDriveJoint {
        GpuPrismaticDriveJoint {
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
            drive_compliance: self.drive_compliance,
            drive_damping: self.drive_damping,
            target_position: self.target_position,
            _pad0: 0.0,
        }
    }
}

impl JointBodies for PrismaticDriveJoint {
    fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }
}

/// Device-packed [`PrismaticDriveJoint`]; layout matches `Joint` in
/// `shaders/rigid_joint_prismatic_drive.wgsl` (`96` bytes). The two anchors and
/// the slide axis are padded to `vec4` so each starts on the `16`-byte boundary
/// the storage layout requires (the three `w` lanes are unused); the
/// rest-rotation quaternion fills a fourth `vec4` as `(x, y, z, w)`. The two
/// indices and the two structural compliances fill the fifth `16`-byte block;
/// the drive compliance, drive damping, and target position fill the trailing
/// one.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuPrismaticDriveJoint {
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
        let joint = PrismaticDriveJoint::new(
            2,
            9,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-4.0, 5.0, -6.0),
            Vec3::new(0.0, 1.0, 0.0),
            Quat::from_xyzw(0.1, 0.2, 0.3, 0.4),
            0.5,
            0.002,
            0.001,
            0.003,
            0.004,
        );
        let gpu = joint.to_gpu();
        assert_eq!(gpu.body_a, 2);
        assert_eq!(gpu.body_b, 9);
        assert_eq!(gpu.anchor_a, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(gpu.anchor_b, [-4.0, 5.0, -6.0, 0.0]);
        assert_eq!(gpu.axis_a, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(gpu.rest_rotation, [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(gpu.target_position, 0.5);
        assert_eq!(gpu.compliance, 0.002);
        assert_eq!(gpu.angular_compliance, 0.001);
        assert_eq!(gpu.drive_compliance, 0.003);
        assert_eq!(gpu.drive_damping, 0.004);
        assert_eq!(gpu._pad0, 0.0);
    }

    #[test]
    fn gpu_struct_is_96_bytes() {
        assert_eq!(size_of::<GpuPrismaticDriveJoint>(), 96);
    }

    #[test]
    fn servo_builds_a_rigid_zero_compliance_drive() {
        let joint = PrismaticDriveJoint::servo(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Y, 0.25);
        assert_eq!(joint.target_position, 0.25);
        assert_eq!(joint.rest_rotation, Quat::IDENTITY);
        assert_eq!(joint.compliance, 0.0);
        assert_eq!(joint.angular_compliance, 0.0);
        assert_eq!(joint.drive_compliance, 0.0);
        assert_eq!(joint.drive_damping, 0.0);
    }

    #[test]
    fn spring_inverts_stiffness_into_compliance() {
        let joint =
            PrismaticDriveJoint::spring(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Y, 0.4, 500.0, 10.0);
        assert!((joint.drive_compliance - 1.0 / 500.0).abs() < 1.0e-9);
        assert_eq!(joint.drive_damping, 10.0);
        assert_eq!(joint.target_position, 0.4);
    }

    #[test]
    fn spring_with_zero_stiffness_is_rigid() {
        let joint =
            PrismaticDriveJoint::spring(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Y, 0.0, 0.0, 5.0);
        assert_eq!(joint.drive_compliance, 0.0);
    }

    #[test]
    fn bodies_returns_pair_in_order() {
        let joint = PrismaticDriveJoint::servo(6, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Y, 0.1);
        assert_eq!(joint.bodies(), (6, 1));
        assert_eq!(JointBodies::bodies(&joint), (6, 1));
    }
}
