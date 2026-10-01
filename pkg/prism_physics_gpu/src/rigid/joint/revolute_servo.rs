//! The revolute (hinge) torque-saturated angular servo rigid-body joint.
//!
//! A [`RevoluteServoJoint`] is a [`RevoluteJoint`](super::RevoluteJoint) — the
//! point-to-point positional weld plus hinge axis-alignment that leaves exactly
//! the spin about the shared hinge axis free — with a *torque-limited angular
//! drive* layered on top. Like [`RevoluteDriveJoint`](super::RevoluteDriveJoint)
//! it servos the free spin toward a commanded
//! [`target_angle`](RevoluteServoJoint::target_angle), as a rigid position servo
//! or a soft compliant-and-damped spring; **unlike** the drive it caps the drive
//! torque at [`max_torque`](RevoluteServoJoint::max_torque), so the actuator can
//! only pull as hard as a real motor and stalls — rather than snapping — against
//! a load that exceeds its rating. It is the complete model of a powered hinge:
//! the stiffness/damping/force-limit triple of a servo motor, a landing-gear
//! actuator, or a torque-limited robot joint.
//!
//! # The three constraints
//!
//! * **Point-to-point** (positional): the world-space anchor coincidence the
//!   spherical joint drives to zero, `p_a - p_b -> 0`.
//! * **Axis alignment** (angular): the world-space hinge axes
//!   `u_a = rotate(orientation_a, axis_a)` and `u_b = rotate(orientation_b,
//!   axis_b)` are driven parallel by cancelling their cross product, locking the
//!   two rotational degrees of freedom perpendicular to the hinge.
//! * **Torque-saturated angular drive** (angular, bilateral): the signed hinge
//!   angle `theta` between the two bodies' reference directions, measured about
//!   the shared hinge axis, is driven onto `target_angle` with a
//!   compliant-and-damped `XPBD` equality `C = theta - target_angle`, exactly as
//!   the plain drive — but the accumulated drive multiplier is clamped each
//!   sweep to `+/- max_torque * h`, the largest angular impulse a torque of
//!   `max_torque` can deliver over a substep of length `h`. Beyond that cap the
//!   drive saturates: it applies its maximum torque and no more, so an
//!   over-ranged target no longer snaps the hinge rigidly onto its set point.
//!
//! Each sweep projects the axis alignment first (realigning the hinge), then the
//! torque-saturated drive (about the freshly aligned axis), then the positional
//! weld.
//!
//! # Torque saturation
//!
//! The angular drive's Lagrange multiplier `lambda` is the net angular impulse
//! (newton-metre-seconds) the drive has applied about the hinge axis this
//! substep; the torque it represents is `lambda / h`. Capping the torque at
//! `max_torque` therefore clamps the accumulated multiplier to the box
//! `[-max_torque * h, +max_torque * h]` after each `XPBD` update, and only the
//! change in the *clamped* multiplier is applied to the bodies — the standard
//! box-limited projected-Gauss-Seidel step shared with every force-limited
//! constraint in the crate. A non-positive `max_torque` disables the cap, so the
//! joint degenerates exactly to [`RevoluteDriveJoint`](super::RevoluteDriveJoint)
//! and the servo can pull arbitrarily hard.
//!
//! # Compliance and damping
//!
//! [`compliance`](RevoluteServoJoint::compliance) is the inverse stiffness of
//! the positional weld (metres per newton),
//! [`angular_compliance`](RevoluteServoJoint::angular_compliance) the inverse
//! stiffness of the axis alignment (radians per newton-metre), and
//! [`drive_compliance`](RevoluteServoJoint::drive_compliance) the inverse
//! stiffness of the angular drive (radians per newton-metre). Zero drive
//! compliance is a rigid position servo that would snap the spin onto the target
//! each sweep were the torque unbounded; a positive value softens it into an
//! angular spring of stiffness `1 / drive_compliance`.
//!
//! [`drive_damping`](RevoluteServoJoint::drive_damping) is the drive's dashpot
//! gain. Following Macklin et al. the damping enters the `XPBD` update through
//! `gamma = drive_compliance * drive_damping / h`, so it is **coupled to the
//! drive compliance**: a perfectly rigid servo (`drive_compliance = 0`) carries
//! no damping term. Give the drive a non-zero compliance (a spring) to make the
//! damping take effect.
//!
//! # Scope
//!
//! This joint is the force-limited superset of
//! [`RevoluteDriveJoint`](super::RevoluteDriveJoint): with `max_torque <= 0` the
//! two are numerically identical. For a free-spinning powered hinge with no
//! positional stiffness — a wheel or turntable driven to a *rate* rather than an
//! *angle* — use the velocity dual [`RevoluteMotorJoint`](super::RevoluteMotorJoint).
//!
//! Provenance: the point-to-point and hinge axis-alignment constraints (Müller
//! et al., "Detailed Rigid Body Simulation with XPBD"), the signed hinge-angle
//! measurement shared with the hinge limit, the bilateral compliant-and-damped
//! angular drive with Macklin-style damping regularisation (Macklin et al.,
//! "XPBD: Position-Based Simulation of Compliant Constrained Dynamics"), and the
//! box-limited (projected) multiplier clamp that realises the torque saturation,
//! over the world-space inverse inertia and quaternion kinematics of Baraff &
//! Witkin. No Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use super::coloring::JointBodies;

/// A revolute (hinge) joint whose single free spin is servoed onto a commanded
/// angle by a torque-limited drive: a point-to-point weld plus hinge
/// axis-alignment plus a bilateral compliant-and-damped angular drive whose
/// torque is capped at [`max_torque`](Self::max_torque).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RevoluteServoJoint {
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
    /// Commanded hinge angle the drive servos toward, in radians.
    pub target_angle: f32,
    /// Inverse stiffness (metres per newton) of the positional weld. Zero is the
    /// rigid limit.
    pub compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the axis-alignment
    /// constraint. Zero is the rigid limit.
    pub angular_compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the angular drive. Zero
    /// is a rigid position servo; a positive value is an angular spring of
    /// stiffness `1 / drive_compliance`.
    pub drive_compliance: f32,
    /// Dashpot gain of the drive. Enters the `XPBD` update through
    /// `gamma = drive_compliance * drive_damping / h`, so it is coupled to the
    /// drive compliance and has no effect on a rigid (`drive_compliance = 0`)
    /// servo.
    pub drive_damping: f32,
    /// Maximum magnitude of the drive torque, in newton-metres. The accumulated
    /// drive impulse is clamped each sweep to `+/- max_torque * h`. A
    /// non-positive value disables the cap, making the drive unbounded (then the
    /// joint is numerically identical to [`RevoluteDriveJoint`](super::RevoluteDriveJoint)).
    pub max_torque: f32,
}

impl RevoluteServoJoint {
    /// Creates a revolute servo joint from its full parameter set.
    ///
    /// `axis_a` / `axis_b` are the hinge axes in each body's local frame (only
    /// their direction matters). `ref_a` / `ref_b` are the zero-angle reference
    /// directions the signed hinge angle is measured between. `target_angle` is
    /// the commanded spin. The three compliances and the drive damping are the
    /// inverse stiffnesses and dashpot gain described on the fields, and
    /// `max_torque` is the drive's torque cap (non-positive disables it).
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "a torque-limited hinge drive is defined by its two bodies, two \
                  anchors, two hinge axes, two angle references, target angle, \
                  four solver gains, and a torque cap; grouping them into \
                  sub-structs would obscure the device packing"
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
        target_angle: f32,
        compliance: f32,
        angular_compliance: f32,
        drive_compliance: f32,
        drive_damping: f32,
        max_torque: f32,
    ) -> RevoluteServoJoint {
        RevoluteServoJoint {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            axis_b,
            ref_a,
            ref_b,
            target_angle,
            compliance,
            angular_compliance,
            drive_compliance,
            drive_damping,
            max_torque,
        }
    }

    /// Creates a rigid torque-limited position servo: zero compliance
    /// everywhere, so each substep drives the hinge angle onto `target_angle`
    /// with up to `max_torque` of drive torque while the weld and axis alignment
    /// stay rigid. A non-positive `max_torque` leaves the servo unbounded.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the servo still needs both anchors, both hinge axes, both \
                  angle references, the target angle, and its torque cap to be \
                  fully specified"
    )]
    pub fn servo(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis_a: Vec3,
        axis_b: Vec3,
        ref_a: Vec3,
        ref_b: Vec3,
        target_angle: f32,
        max_torque: f32,
    ) -> RevoluteServoJoint {
        RevoluteServoJoint::new(
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            axis_b,
            ref_a,
            ref_b,
            target_angle,
            0.0,
            0.0,
            0.0,
            0.0,
            max_torque,
        )
    }

    /// Creates a torque-limited angular spring-damper drive — the full
    /// stiffness / damping / force-limit triple of a servo motor. The weld and
    /// axis alignment stay rigid; the drive has the given stiffness
    /// (`drive_compliance = 1 / drive_stiffness`), the given drive damping, and
    /// a torque capped at `max_torque` (non-positive disables the cap).
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the drive still needs both anchors, both hinge axes, both \
                  angle references, the target angle, its stiffness, its \
                  damping, and its torque cap to be fully specified"
    )]
    pub fn drive(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis_a: Vec3,
        axis_b: Vec3,
        ref_a: Vec3,
        ref_b: Vec3,
        target_angle: f32,
        drive_stiffness: f32,
        drive_damping: f32,
        max_torque: f32,
    ) -> RevoluteServoJoint {
        let drive_compliance = if drive_stiffness > 0.0 {
            1.0 / drive_stiffness
        } else {
            0.0
        };
        RevoluteServoJoint::new(
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            axis_b,
            ref_a,
            ref_b,
            target_angle,
            0.0,
            0.0,
            drive_compliance,
            drive_damping,
            max_torque,
        )
    }

    /// The two body indices this joint couples, in `(a, b)` order.
    #[must_use]
    pub fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }

    /// Packs the joint into its `GPU` storage-buffer representation.
    #[must_use]
    pub(crate) fn to_gpu(self) -> GpuRevoluteServoJoint {
        GpuRevoluteServoJoint {
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
            drive_compliance: self.drive_compliance,
            drive_damping: self.drive_damping,
            target_angle: self.target_angle,
            max_torque: self.max_torque,
        }
    }
}

impl JointBodies for RevoluteServoJoint {
    fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }
}

/// Device-packed [`RevoluteServoJoint`]; layout matches `Joint` in
/// `shaders/rigid_joint_revolute_servo.wgsl` (`128` bytes). The two anchors, two
/// hinge axes, and two angle references are padded to `vec4` so each starts on
/// the `16`-byte boundary the storage layout requires; the six `w` lanes are
/// unused. The two indices and the weld / axis compliances fill the next
/// `16`-byte block, and the drive compliance, drive damping, target angle, and
/// torque cap fill the last — the layout of
/// [`GpuRevoluteDriveJoint`](super::RevoluteDriveJoint) with its trailing pad
/// word repurposed as `max_torque`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuRevoluteServoJoint {
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
    /// Inverse stiffness (radians per newton-metre) of the angular drive.
    drive_compliance: f32,
    /// Dashpot gain of the angular drive.
    drive_damping: f32,
    /// Commanded hinge angle, in radians.
    target_angle: f32,
    /// Maximum drive torque, in newton-metres; non-positive disables the cap.
    max_torque: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::FRAC_PI_4;

    #[test]
    fn gpu_packing_round_trips_fields() {
        let joint = RevoluteServoJoint::new(
            2,
            9,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-4.0, 5.0, -6.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            FRAC_PI_4,
            0.002,
            0.001,
            0.004,
            0.5,
            25.0,
        );
        let gpu = joint.to_gpu();
        assert_eq!(gpu.body_a, 2);
        assert_eq!(gpu.body_b, 9);
        assert_eq!(gpu.anchor_a, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(gpu.anchor_b, [-4.0, 5.0, -6.0, 0.0]);
        assert_eq!(gpu.axis_a, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(gpu.axis_b, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(gpu.ref_a, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(gpu.ref_b, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(gpu.compliance, 0.002);
        assert_eq!(gpu.angular_compliance, 0.001);
        assert_eq!(gpu.drive_compliance, 0.004);
        assert_eq!(gpu.drive_damping, 0.5);
        assert_eq!(gpu.target_angle, FRAC_PI_4);
        assert_eq!(gpu.max_torque, 25.0);
    }

    #[test]
    fn gpu_struct_is_128_bytes() {
        assert_eq!(size_of::<GpuRevoluteServoJoint>(), 128);
    }

    #[test]
    fn servo_builds_a_rigid_zero_compliance_drive_with_torque_cap() {
        let joint = RevoluteServoJoint::servo(
            0,
            1,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Z,
            Vec3::Z,
            Vec3::X,
            Vec3::X,
            FRAC_PI_4,
            12.0,
        );
        assert_eq!(joint.compliance, 0.0);
        assert_eq!(joint.angular_compliance, 0.0);
        assert_eq!(joint.drive_compliance, 0.0);
        assert_eq!(joint.drive_damping, 0.0);
        assert_eq!(joint.target_angle, FRAC_PI_4);
        assert_eq!(joint.max_torque, 12.0);
    }

    #[test]
    fn drive_inverts_stiffness_into_compliance_and_keeps_cap() {
        let joint = RevoluteServoJoint::drive(
            0,
            1,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Z,
            Vec3::Z,
            Vec3::X,
            Vec3::X,
            FRAC_PI_4,
            200.0,
            5.0,
            40.0,
        );
        assert!((joint.drive_compliance - 1.0 / 200.0).abs() < 1e-9);
        assert_eq!(joint.drive_damping, 5.0);
        assert_eq!(joint.max_torque, 40.0);
        assert_eq!(joint.compliance, 0.0);
        assert_eq!(joint.angular_compliance, 0.0);
    }

    #[test]
    fn drive_with_zero_stiffness_is_rigid() {
        let joint = RevoluteServoJoint::drive(
            0,
            1,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Z,
            Vec3::Z,
            Vec3::X,
            Vec3::X,
            0.0,
            0.0,
            3.0,
            10.0,
        );
        assert_eq!(joint.drive_compliance, 0.0);
    }

    #[test]
    fn non_positive_torque_cap_is_preserved_as_unbounded() {
        let joint = RevoluteServoJoint::servo(
            0,
            1,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Z,
            Vec3::Z,
            Vec3::X,
            Vec3::X,
            0.0,
            0.0,
        );
        assert_eq!(joint.max_torque, 0.0);
    }

    #[test]
    fn bodies_returns_pair_in_order() {
        let joint = RevoluteServoJoint::servo(
            6,
            1,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Y,
            Vec3::Y,
            Vec3::X,
            Vec3::X,
            0.0,
            5.0,
        );
        assert_eq!(joint.bodies(), (6, 1));
        assert_eq!(JointBodies::bodies(&joint), (6, 1));
    }
}
