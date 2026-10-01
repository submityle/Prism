//! The revolute (hinge) velocity-motor rigid-body joint.
//!
//! A [`RevoluteMotorJoint`] is a [`RevoluteJoint`](super::RevoluteJoint) — the
//! point-to-point positional weld plus hinge axis-alignment that leaves exactly
//! the spin about the shared hinge axis free — with a *velocity motor* layered
//! on top: a powered hinge that drives the free spin's *rate* toward a commanded
//! [`target_velocity`](RevoluteMotorJoint::target_velocity), with no set point
//! and no positional stiffness about the axis. It is the free-spinning powered
//! hinge of a wheel, a turntable, a conveyor roller, or a fan — the single hinge
//! degree of freedom driven to an angular *speed* rather than to an angle.
//!
//! This is the velocity counterpart of
//! [`RevoluteDriveJoint`](super::RevoluteDriveJoint): the drive is a *position
//! servo* that pulls the hinge angle onto a commanded set point, whereas the
//! motor is a *velocity servo* that pulls the hinge rate onto a commanded
//! angular velocity. Because a velocity target has no zero-angle reference, the
//! motor carries no `ref_a` / `ref_b` reference directions and never measures an
//! absolute hinge angle; it reads only the relative angular rate about the
//! shared axis accumulated over the substep.
//!
//! # The three constraints
//!
//! * **Point-to-point** (positional): the world-space anchor coincidence the
//!   spherical joint drives to zero, `p_a - p_b -> 0`.
//! * **Axis alignment** (angular): the world-space hinge axes
//!   `u_a = rotate(orientation_a, axis_a)` and `u_b = rotate(orientation_b,
//!   axis_b)` are driven parallel by cancelling their cross product, locking the
//!   two rotational degrees of freedom perpendicular to the hinge.
//! * **Velocity motor** (angular, bilateral): the relative angular rate about
//!   the shared hinge axis `u` is driven onto `target_velocity`. Expressed in
//!   the shared position-based stepper this is the per-substep equality
//!   `C = u . (dphi_b - dphi_a) - target_velocity * h`, where `dphi` is each
//!   body's angular displacement since the substep snapshot and `h` the substep
//!   duration: forcing the relative displacement over the substep to equal
//!   `target_velocity * h` forces the recovered relative angular velocity onto
//!   `target_velocity`.
//!
//! Each sweep projects the axis alignment first (realigning the hinge), then the
//! velocity motor (about the freshly aligned axis), then the positional weld.
//!
//! # Compliance
//!
//! [`compliance`](RevoluteMotorJoint::compliance) is the inverse stiffness of
//! the positional weld (metres per newton),
//! [`angular_compliance`](RevoluteMotorJoint::angular_compliance) the inverse
//! stiffness of the axis alignment (radians per newton-metre), and
//! [`motor_compliance`](RevoluteMotorJoint::motor_compliance) the inverse
//! torque gain of the velocity motor (reciprocal newton-metre-seconds per
//! radian). Zero motor compliance is a *rigid* motor that forces the relative
//! hinge rate exactly onto `target_velocity` each substep (unbounded torque); a
//! positive value is a *soft* motor that applies a finite torque proportional to
//! the rate error, so the hinge approaches the commanded speed asymptotically —
//! the model of a real motor with limited stall torque.
//!
//! Unlike the position drive the motor carries no explicit dashpot: a velocity
//! motor *is* the damping term of an angular drive used on its own, so a
//! separate damping gain would be redundant. The motor compliance alone sets how
//! hard it pulls the rate toward the target.
//!
//! # Scope
//!
//! This completes the pair the position drive deliberately left open: the
//! position drive is a servo onto an *angle*, this motor is a servo onto an
//! *angular velocity*. A hinge that is both bounded and powered (a limit plus a
//! motor) is a composition of this family with
//! [`HingeLimitJoint`](super::HingeLimitJoint) rather than a single joint, and a
//! motor with a torque *ceiling* (a saturating motor) would need a clamped
//! Lagrange multiplier the shared bilateral stepper does not expose, and is left
//! as future work rather than faked here.
//!
//! Provenance: the point-to-point and hinge axis-alignment constraints (Müller
//! et al., "Detailed Rigid Body Simulation with XPBD"), with the velocity-level
//! motor expressed as a per-substep compliant equality on the relative angular
//! displacement about the hinge axis (Macklin et al., "XPBD: Position-Based
//! Simulation of Compliant Constrained Dynamics"), over the world-space inverse
//! inertia and quaternion kinematics of Baraff & Witkin. No Unreal Engine source
//! or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use super::coloring::JointBodies;

/// A revolute (hinge) joint whose single free spin is driven to a commanded
/// *angular velocity*: a point-to-point weld plus hinge axis-alignment plus a
/// bilateral velocity motor about the hinge axis.
///
/// Either body may be static (zero inverse mass and inertia); a motor between a
/// dynamic rotor and a static housing spins the rotor up to `target_velocity`
/// and holds it there against load. The motor reads only the relative angular
/// rate about the shared axis, so it needs no zero-angle reference and never
/// drifts with absolute orientation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RevoluteMotorJoint {
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
    /// Commanded relative angular velocity about the hinge axis the motor drives
    /// toward, in radians per second. The sign is `u . (omega_b - omega_a)`:
    /// positive spins body `b` ahead of body `a` about the shared axis.
    pub target_velocity: f32,
    /// Inverse stiffness (metres per newton) of the positional weld. Zero is the
    /// rigid limit.
    pub compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the axis-alignment
    /// constraint. Zero is the rigid limit.
    pub angular_compliance: f32,
    /// Inverse torque gain of the velocity motor. Zero is a rigid motor that
    /// forces the relative hinge rate exactly onto `target_velocity` each
    /// substep; a positive value is a soft motor of torque gain
    /// `1 / motor_compliance` that approaches the commanded speed with a finite
    /// torque.
    pub motor_compliance: f32,
}

impl RevoluteMotorJoint {
    /// Creates a revolute velocity-motor joint from its full parameter set.
    ///
    /// `axis_a` / `axis_b` are the hinge axes in each body's local frame (only
    /// their direction matters). `target_velocity` is the commanded relative
    /// angular velocity about the shared axis. The three compliances are the
    /// inverse stiffnesses described on the fields.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "a hinge motor is defined by its two bodies, two anchors, two \
                  hinge axes, target velocity, and three solver gains; grouping \
                  them into sub-structs would obscure the device packing"
    )]
    pub fn new(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis_a: Vec3,
        axis_b: Vec3,
        target_velocity: f32,
        compliance: f32,
        angular_compliance: f32,
        motor_compliance: f32,
    ) -> RevoluteMotorJoint {
        RevoluteMotorJoint {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            axis_b,
            target_velocity,
            compliance,
            angular_compliance,
            motor_compliance,
        }
    }

    /// Creates a rigid velocity motor: a motor with zero compliance everywhere,
    /// so each substep forces the relative hinge rate exactly onto
    /// `target_velocity` (unbounded torque) while the weld and axis alignment
    /// stay rigid.
    #[must_use]
    pub fn rigid(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis_a: Vec3,
        axis_b: Vec3,
        target_velocity: f32,
    ) -> RevoluteMotorJoint {
        RevoluteMotorJoint::new(
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            axis_b,
            target_velocity,
            0.0,
            0.0,
            0.0,
        )
    }

    /// Creates a soft velocity motor: a rigid weld and axis alignment with a
    /// motor of the given torque stiffness
    /// (`motor_compliance = 1 / motor_stiffness`), so the hinge approaches
    /// `target_velocity` with a finite torque rather than instantly. A
    /// non-positive stiffness collapses to the rigid motor.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the soft motor still needs both anchors, both hinge axes, the \
                  target velocity, and its stiffness to be fully specified"
    )]
    pub fn soft(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis_a: Vec3,
        axis_b: Vec3,
        target_velocity: f32,
        motor_stiffness: f32,
    ) -> RevoluteMotorJoint {
        let motor_compliance = if motor_stiffness > 0.0 {
            1.0 / motor_stiffness
        } else {
            0.0
        };
        RevoluteMotorJoint::new(
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            axis_b,
            target_velocity,
            0.0,
            0.0,
            motor_compliance,
        )
    }

    /// The two body indices this joint couples, in `(a, b)` order.
    #[must_use]
    pub fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }

    /// Packs the joint into its `GPU` storage-buffer representation.
    #[must_use]
    pub(crate) fn to_gpu(self) -> GpuRevoluteMotorJoint {
        GpuRevoluteMotorJoint {
            anchor_a: [self.anchor_a.x, self.anchor_a.y, self.anchor_a.z, 0.0],
            anchor_b: [self.anchor_b.x, self.anchor_b.y, self.anchor_b.z, 0.0],
            axis_a: [self.axis_a.x, self.axis_a.y, self.axis_a.z, 0.0],
            axis_b: [self.axis_b.x, self.axis_b.y, self.axis_b.z, 0.0],
            body_a: self.body_a,
            body_b: self.body_b,
            compliance: self.compliance,
            angular_compliance: self.angular_compliance,
            motor_compliance: self.motor_compliance,
            target_velocity: self.target_velocity,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }
}

impl JointBodies for RevoluteMotorJoint {
    fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }
}

/// Device-packed [`RevoluteMotorJoint`]; layout matches `Joint` in
/// `shaders/rigid_joint_revolute_motor.wgsl` (`96` bytes). The two anchors and
/// two hinge axes are padded to `vec4` so each starts on the `16`-byte boundary
/// the storage layout requires; the four `w` lanes are unused. The two indices
/// and the weld / axis compliances fill the next `16`-byte block, and the motor
/// compliance, target velocity, and two pad words fill the last.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuRevoluteMotorJoint {
    /// Body-local anchor on body `a` in `xyz`; `w` unused.
    anchor_a: [f32; 4],
    /// Body-local anchor on body `b` in `xyz`; `w` unused.
    anchor_b: [f32; 4],
    /// Body-local hinge axis on body `a` in `xyz`; `w` unused.
    axis_a: [f32; 4],
    /// Body-local hinge axis on body `b` in `xyz`; `w` unused.
    axis_b: [f32; 4],
    /// Index of body `a`.
    body_a: u32,
    /// Index of body `b`.
    body_b: u32,
    /// Inverse stiffness (metres per newton) of the positional weld.
    compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the axis alignment.
    angular_compliance: f32,
    /// Inverse torque gain of the velocity motor.
    motor_compliance: f32,
    /// Commanded relative angular velocity about the hinge axis, in radians per
    /// second.
    target_velocity: f32,
    /// Padding to the `16`-byte boundary; always zero.
    _pad0: f32,
    /// Padding to the `16`-byte boundary; always zero.
    _pad1: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_packing_round_trips_fields() {
        let joint = RevoluteMotorJoint::new(
            2,
            9,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-4.0, 5.0, -6.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            7.5,
            0.002,
            0.001,
            0.004,
        );
        let gpu = joint.to_gpu();
        assert_eq!(gpu.body_a, 2);
        assert_eq!(gpu.body_b, 9);
        assert_eq!(gpu.anchor_a, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(gpu.anchor_b, [-4.0, 5.0, -6.0, 0.0]);
        assert_eq!(gpu.axis_a, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(gpu.axis_b, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(gpu.compliance, 0.002);
        assert_eq!(gpu.angular_compliance, 0.001);
        assert_eq!(gpu.motor_compliance, 0.004);
        assert_eq!(gpu.target_velocity, 7.5);
        assert_eq!(gpu._pad0, 0.0);
        assert_eq!(gpu._pad1, 0.0);
    }

    #[test]
    fn gpu_struct_is_96_bytes() {
        assert_eq!(size_of::<GpuRevoluteMotorJoint>(), 96);
    }

    #[test]
    fn rigid_builds_a_zero_compliance_motor() {
        let joint = RevoluteMotorJoint::rigid(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Z, Vec3::Z, 3.0);
        assert_eq!(joint.compliance, 0.0);
        assert_eq!(joint.angular_compliance, 0.0);
        assert_eq!(joint.motor_compliance, 0.0);
        assert_eq!(joint.target_velocity, 3.0);
    }

    #[test]
    fn soft_inverts_stiffness_into_compliance() {
        let joint =
            RevoluteMotorJoint::soft(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Z, Vec3::Z, 2.0, 200.0);
        assert!((joint.motor_compliance - 1.0 / 200.0).abs() < 1e-9);
        assert_eq!(joint.target_velocity, 2.0);
        assert_eq!(joint.compliance, 0.0);
        assert_eq!(joint.angular_compliance, 0.0);
    }

    #[test]
    fn soft_with_zero_stiffness_is_rigid() {
        let joint =
            RevoluteMotorJoint::soft(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Z, Vec3::Z, 1.0, 0.0);
        assert_eq!(joint.motor_compliance, 0.0);
    }

    #[test]
    fn bodies_returns_pair_in_order() {
        let joint = RevoluteMotorJoint::rigid(6, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y, 0.0);
        assert_eq!(joint.bodies(), (6, 1));
        assert_eq!(JointBodies::bodies(&joint), (6, 1));
    }
}
