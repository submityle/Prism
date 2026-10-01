//! The angular `SLERP` drive rigid-body joint — a pure 3-DOF soft
//! orientation servo.
//!
//! An [`AngularSlerpDriveJoint`] drives body `b`'s orientation, relative to body
//! `a`, toward a commanded target relative rotation
//! [`target_rotation`](AngularSlerpDriveJoint::target_rotation) along the
//! geodesic (shortest-arc) path on the unit-quaternion sphere — the rotational
//! analogue of a point-to-point positional drive. It is the isolated angular
//! `SLERP` drive of a configurable joint: unlike the hinge drives
//! ([`RevoluteDriveJoint`](super::RevoluteDriveJoint),
//! [`RevoluteServoJoint`](super::RevoluteServoJoint)) that servo a single signed
//! angle about one axis, this drive acts on all three relative rotational
//! degrees of freedom at once, pulling the full relative orientation onto its
//! target without privileging any axis. It couples only the orientations: the
//! two bodies' positions are left entirely free, so the joint is a reaction
//! controller — a thruster-stabilised attitude hold, a camera gimbal servo, a
//! ragdoll pose drive — rather than a mechanical linkage.
//!
//! # The single constraint
//!
//! The joint owns one angular constraint. Body `b`'s target world orientation
//! is `q_a * target_rotation`; the world-space error rotation that carries `b`
//! onto it is `error = (q_a * target_rotation) * conj(q_b)`. Its rotation vector
//! `theta * n` — twice the imaginary part of the hemisphere-corrected `error`
//! quaternion — is the geodesic axis `n` and angle `theta` the drive works to
//! cancel. Driving the single scalar `theta` to zero along `n` slides the
//! relative orientation along the shortest arc toward the target, exactly the
//! `SLERP` trajectory.
//!
//! # Compliance and damping
//!
//! [`drive_compliance`](AngularSlerpDriveJoint::drive_compliance) is the inverse
//! stiffness of the drive (radians per newton-metre). Zero is a rigid servo
//! that would snap the orientation onto the target each sweep were the torque
//! unbounded; a positive value softens it into an angular spring of stiffness
//! `1 / drive_compliance`.
//!
//! [`drive_damping`](AngularSlerpDriveJoint::drive_damping) is the drive's
//! dashpot gain. Following Macklin et al. the damping enters the `XPBD` update
//! through `gamma = drive_compliance * drive_damping / h`, so it is **coupled to
//! the drive compliance**: a perfectly rigid servo (`drive_compliance = 0`)
//! carries no damping term. Give the drive a non-zero compliance (a spring) to
//! make the damping take effect.
//!
//! # Torque saturation
//!
//! The drive's Lagrange multiplier `lambda` is the net angular impulse
//! (newton-metre-seconds) applied about the geodesic axis this substep; the
//! torque it represents is `lambda / h`. Capping the torque at
//! [`max_torque`](AngularSlerpDriveJoint::max_torque) therefore clamps the
//! accumulated multiplier to the box `[-max_torque * h, +max_torque * h]` after
//! each `XPBD` update, and only the change in the *clamped* multiplier is
//! applied to the bodies — the standard box-limited projected-Gauss-Seidel step
//! shared with every force-limited constraint in the crate. A non-positive
//! `max_torque` disables the cap, so the drive can pull arbitrarily hard.
//!
//! # Scope
//!
//! This is a bilateral drive: it is always active, pulling the orientation onto
//! the target from either side. For a rigid full-six-DOF weld (orientation lock
//! plus positional weld, no compliance control) use
//! [`FixedJoint`](super::FixedJoint); for a single-axis angular drive use the
//! revolute drives. The three-DOF geodesic measurement here is the one the
//! `FixedJoint`'s angular lock uses, lifted into a compliant, damped,
//! torque-limited drive.
//!
//! Provenance: the relative-orientation geodesic measurement and its substep
//! `XPBD` angular correction (Müller et al., "Detailed Rigid Body Simulation
//! with XPBD"), with the bilateral compliant-and-damped drive and its
//! Macklin-style damping regularisation (Macklin et al., "XPBD: Position-Based
//! Simulation of Compliant Constrained Dynamics") and the box-limited multiplier
//! clamp realising the torque saturation, over the world-space inverse inertia
//! and quaternion kinematics of Baraff & Witkin. No Unreal Engine source or
//! derived code.

use bytemuck::{Pod, Zeroable};
use glam::Quat;

use super::coloring::JointBodies;

/// A pure 3-DOF angular `SLERP` drive binding body `b`'s orientation toward a
/// target relative rotation in body `a`'s frame.
///
/// The joint constrains only the relative orientation — it servos `b` toward
/// `q_a * target_rotation` along the geodesic — and leaves both bodies'
/// positions free. Either body may be static (zero inverse mass and inverse
/// inertia); a drive between a dynamic body and a static one servos the dynamic
/// body's orientation toward a fixed world attitude.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AngularSlerpDriveJoint {
    /// Index of the first body in the [`RigidBodyState`](super::super::RigidBodyState).
    pub body_a: u32,
    /// Index of the second body in the [`RigidBodyState`](super::super::RigidBodyState).
    pub body_b: u32,
    /// Target relative orientation of body `b` in body `a`'s frame — the drive
    /// servos `b`'s world orientation toward `q_a * target_rotation`.
    pub target_rotation: Quat,
    /// Inverse stiffness (radians per newton-metre) of the drive. Zero is a
    /// rigid position servo; a positive value is an angular spring of stiffness
    /// `1 / drive_compliance`.
    pub drive_compliance: f32,
    /// Dashpot gain of the drive. Enters the `XPBD` update through
    /// `gamma = drive_compliance * drive_damping / h`, so it is coupled to the
    /// drive compliance and has no effect on a rigid (`drive_compliance = 0`)
    /// servo.
    pub drive_damping: f32,
    /// Maximum magnitude of the drive torque, in newton-metres. The accumulated
    /// drive impulse is clamped each sweep to `+/- max_torque * h`. A
    /// non-positive value disables the cap, making the drive unbounded.
    pub max_torque: f32,
}

impl AngularSlerpDriveJoint {
    /// Creates an angular `SLERP` drive from its full parameter set.
    ///
    /// `target_rotation` is the commanded relative orientation of `b` in `a`'s
    /// frame. `drive_compliance` and `drive_damping` are the inverse stiffness
    /// and dashpot gain described on the fields, and `max_torque` is the drive's
    /// torque cap (non-positive disables it).
    #[must_use]
    pub fn new(
        body_a: u32,
        body_b: u32,
        target_rotation: Quat,
        drive_compliance: f32,
        drive_damping: f32,
        max_torque: f32,
    ) -> AngularSlerpDriveJoint {
        AngularSlerpDriveJoint {
            body_a,
            body_b,
            target_rotation,
            drive_compliance,
            drive_damping,
            max_torque,
        }
    }

    /// Creates a rigid torque-limited attitude servo: zero compliance, so each
    /// substep drives the relative orientation onto `target_rotation` with up
    /// to `max_torque` of drive torque. A non-positive `max_torque` leaves the
    /// servo unbounded.
    #[must_use]
    pub fn servo(
        body_a: u32,
        body_b: u32,
        target_rotation: Quat,
        max_torque: f32,
    ) -> AngularSlerpDriveJoint {
        AngularSlerpDriveJoint::new(body_a, body_b, target_rotation, 0.0, 0.0, max_torque)
    }

    /// Creates an angular spring-damper drive — the full stiffness / damping /
    /// force-limit triple. The drive has the given stiffness
    /// (`drive_compliance = 1 / drive_stiffness`), the given drive damping, and
    /// a torque capped at `max_torque` (non-positive disables the cap). A
    /// non-positive `drive_stiffness` is the rigid limit.
    #[must_use]
    pub fn drive(
        body_a: u32,
        body_b: u32,
        target_rotation: Quat,
        drive_stiffness: f32,
        drive_damping: f32,
        max_torque: f32,
    ) -> AngularSlerpDriveJoint {
        let drive_compliance = if drive_stiffness > 0.0 {
            1.0 / drive_stiffness
        } else {
            0.0
        };
        AngularSlerpDriveJoint::new(
            body_a,
            body_b,
            target_rotation,
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
    pub(crate) fn to_gpu(self) -> GpuAngularSlerpDriveJoint {
        GpuAngularSlerpDriveJoint {
            target_rotation: [
                self.target_rotation.x,
                self.target_rotation.y,
                self.target_rotation.z,
                self.target_rotation.w,
            ],
            body_a: self.body_a,
            body_b: self.body_b,
            drive_compliance: self.drive_compliance,
            drive_damping: self.drive_damping,
            max_torque: self.max_torque,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }
}

impl JointBodies for AngularSlerpDriveJoint {
    fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }
}

/// Device-packed [`AngularSlerpDriveJoint`]; layout matches `Joint` in
/// `shaders/rigid_joint_angular_slerp_drive.wgsl` (`48` bytes). The target
/// rotation fills the leading `vec4` as `(x, y, z, w)`; the two indices and
/// two gains fill the second `16`-byte block, and the torque cap plus three
/// padding lanes fill the trailing block so the whole element is `16`-byte
/// aligned.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuAngularSlerpDriveJoint {
    /// Target relative orientation of body `b` in body `a`'s frame as `(x, y, z, w)`.
    target_rotation: [f32; 4],
    /// Index of body `a`.
    body_a: u32,
    /// Index of body `b`.
    body_b: u32,
    /// Inverse stiffness (radians per newton-metre) of the angular drive.
    drive_compliance: f32,
    /// Dashpot gain of the angular drive.
    drive_damping: f32,
    /// Maximum drive torque, in newton-metres; non-positive disables the cap.
    max_torque: f32,
    /// Padding to the `16`-byte boundary; unused.
    _pad0: f32,
    /// Padding to the `16`-byte boundary; unused.
    _pad1: f32,
    /// Padding to the `16`-byte boundary; unused.
    _pad2: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn gpu_packing_round_trips_fields() {
        let q = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.5);
        let joint = AngularSlerpDriveJoint::new(2, 9, q, 0.004, 0.5, 25.0);
        let gpu = joint.to_gpu();
        assert_eq!(gpu.body_a, 2);
        assert_eq!(gpu.body_b, 9);
        assert_eq!(gpu.target_rotation, [q.x, q.y, q.z, q.w]);
        assert_eq!(gpu.drive_compliance, 0.004);
        assert_eq!(gpu.drive_damping, 0.5);
        assert_eq!(gpu.max_torque, 25.0);
        assert_eq!(gpu._pad0, 0.0);
        assert_eq!(gpu._pad1, 0.0);
        assert_eq!(gpu._pad2, 0.0);
    }

    #[test]
    fn gpu_struct_is_48_bytes() {
        assert_eq!(size_of::<GpuAngularSlerpDriveJoint>(), 48);
    }

    #[test]
    fn servo_builds_a_rigid_zero_compliance_drive_with_torque_cap() {
        let q = Quat::from_axis_angle(Vec3::Z, 0.3);
        let joint = AngularSlerpDriveJoint::servo(0, 1, q, 12.0);
        assert_eq!(joint.drive_compliance, 0.0);
        assert_eq!(joint.drive_damping, 0.0);
        assert_eq!(joint.target_rotation, q);
        assert_eq!(joint.max_torque, 12.0);
    }

    #[test]
    fn drive_inverts_stiffness_into_compliance_and_keeps_cap() {
        let q = Quat::from_axis_angle(Vec3::X, 0.2);
        let joint = AngularSlerpDriveJoint::drive(0, 1, q, 200.0, 5.0, 40.0);
        assert!((joint.drive_compliance - 1.0 / 200.0).abs() < 1e-9);
        assert_eq!(joint.drive_damping, 5.0);
        assert_eq!(joint.max_torque, 40.0);
    }

    #[test]
    fn drive_with_zero_stiffness_is_rigid() {
        let joint = AngularSlerpDriveJoint::drive(0, 1, Quat::IDENTITY, 0.0, 3.0, 10.0);
        assert_eq!(joint.drive_compliance, 0.0);
    }

    #[test]
    fn bodies_returns_pair_in_order() {
        let joint = AngularSlerpDriveJoint::servo(6, 1, Quat::IDENTITY, 5.0);
        assert_eq!(joint.bodies(), (6, 1));
        assert_eq!(JointBodies::bodies(&joint), (6, 1));
    }
}
