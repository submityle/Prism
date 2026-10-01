//! The revolute (hinge) position-drive rigid-body joint.
//!
//! A [`RevoluteDriveJoint`] is a [`RevoluteJoint`](super::RevoluteJoint) — the
//! point-to-point positional weld plus hinge axis-alignment that leaves exactly
//! the spin about the shared hinge axis free — with an *angular drive* layered
//! on top: a motor that servos that free spin toward a commanded
//! [`target_angle`](RevoluteDriveJoint::target_angle). It is the powered joint
//! of a robot arm, the servo of a landing gear, and the spring-loaded return of
//! a lid — the single hinge degree of freedom driven to a set point rather than
//! left free or merely bounded.
//!
//! # The three constraints
//!
//! * **Point-to-point** (positional): the world-space anchor coincidence the
//!   spherical joint drives to zero, `p_a - p_b -> 0`.
//! * **Axis alignment** (angular): the world-space hinge axes
//!   `u_a = rotate(orientation_a, axis_a)` and `u_b = rotate(orientation_b,
//!   axis_b)` are driven parallel by cancelling their cross product, locking the
//!   two rotational degrees of freedom perpendicular to the hinge.
//! * **Angular drive** (angular, bilateral): the signed hinge angle `theta`
//!   between the two bodies' reference directions, measured about the shared
//!   hinge axis, is driven onto `target_angle` with a compliant-and-damped
//!   `XPBD` equality `C = theta - target_angle`. Unlike the one-sided limit of
//!   [`HingeLimitJoint`](super::HingeLimitJoint) the drive is always active,
//!   pulling the spin toward its set point from either side.
//!
//! Each sweep projects the axis alignment first (realigning the hinge), then the
//! angular drive (about the freshly aligned axis), then the positional weld.
//!
//! # Measuring the hinge angle
//!
//! The drive needs a signed angle, which needs a zero reference. Each body
//! carries a body-local reference direction, [`ref_a`](RevoluteDriveJoint::ref_a)
//! and [`ref_b`](RevoluteDriveJoint::ref_b), nominally perpendicular to its
//! hinge axis. Each sweep both are rotated to world space, projected onto the
//! plane perpendicular to the (unit) hinge axis `u`, and normalised; the signed
//! angle from `ref_a`'s projection to `ref_b`'s projection about `u` is
//! `theta = atan2((p_a x p_b) . u, p_a . p_b)`. The references need not be
//! exactly perpendicular to the axis — only non-parallel to it — because the
//! projection removes the axial component before the angle is taken.
//!
//! # Compliance and damping
//!
//! [`compliance`](RevoluteDriveJoint::compliance) is the inverse stiffness of
//! the positional weld (metres per newton),
//! [`angular_compliance`](RevoluteDriveJoint::angular_compliance) the inverse
//! stiffness of the axis alignment (radians per newton-metre), and
//! [`drive_compliance`](RevoluteDriveJoint::drive_compliance) the inverse
//! stiffness of the angular drive (radians per newton-metre). Zero drive
//! compliance is a rigid *position servo* that snaps the spin exactly onto the
//! target each sweep; a positive value softens it into an angular spring of
//! stiffness `1 / drive_compliance`.
//!
//! [`drive_damping`](RevoluteDriveJoint::drive_damping) is the drive's dashpot
//! gain, resisting the hinge's closing rate so the approach settles without
//! ringing. Following Macklin et al. the damping enters the `XPBD` update
//! through `gamma = drive_compliance * drive_damping / h`, so it is **coupled to
//! the drive compliance**: a perfectly rigid servo (`drive_compliance = 0`)
//! carries no damping term, which is the correct behaviour — a hard servo needs
//! no dashpot. Give the drive a non-zero compliance (a spring) to make the
//! damping take effect.
//!
//! # Scope
//!
//! This is a *position-target* angular drive: a rigid servo or a soft
//! spring-damper to a commanded angle. A pure *velocity* motor — a free-spinning
//! powered hinge with zero positional stiffness, as a wheel or turntable — is a
//! velocity-level constraint the shared position-based stepper does not expose,
//! and is intentionally left as future work rather than faked here.
//!
//! Provenance: the point-to-point and hinge axis-alignment constraints (Müller
//! et al., "Detailed Rigid Body Simulation with XPBD"), the signed hinge-angle
//! measurement shared with the hinge limit, and the bilateral
//! compliant-and-damped angular drive with Macklin-style damping regularisation
//! (Macklin et al., "XPBD: Position-Based Simulation of Compliant Constrained
//! Dynamics"), over the world-space inverse inertia and quaternion kinematics of
//! Baraff & Witkin. No Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use super::coloring::JointBodies;

/// A revolute (hinge) joint whose single free spin is servoed onto a commanded
/// angle: a point-to-point weld plus hinge axis-alignment plus a bilateral
/// compliant-and-damped angular drive about the hinge axis.
///
/// Either body may be static (zero inverse mass and inertia); a joint between a
/// dynamic body and a static one drives the dynamic body's hinge angle about a
/// fixed world axis through a fixed world point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RevoluteDriveJoint {
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
}

impl RevoluteDriveJoint {
    /// Creates a revolute drive joint from its full parameter set.
    ///
    /// `axis_a` / `axis_b` are the hinge axes in each body's local frame (only
    /// their direction matters). `ref_a` / `ref_b` are the zero-angle reference
    /// directions the signed hinge angle is measured between. `target_angle` is
    /// the commanded spin. The three compliances and the drive damping are the
    /// inverse stiffnesses and dashpot gain described on the fields.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "a hinge drive is defined by its two bodies, two anchors, two \
                  hinge axes, two angle references, target angle, and four solver \
                  gains; grouping them into sub-structs would obscure the device \
                  packing"
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
    ) -> RevoluteDriveJoint {
        RevoluteDriveJoint {
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
        }
    }

    /// Creates a rigid angular position servo: a drive with zero compliance
    /// everywhere, so each substep snaps the hinge angle exactly onto
    /// `target_angle` while the weld and axis alignment stay rigid.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the servo still needs both anchors, both hinge axes, both \
                  angle references, and the target angle to be fully specified"
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
    ) -> RevoluteDriveJoint {
        RevoluteDriveJoint::new(
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
        )
    }

    /// Creates a soft angular spring-damper drive: a rigid weld and axis
    /// alignment with a drive of the given stiffness
    /// (`drive_compliance = 1 / drive_stiffness`) and the given drive damping.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the spring still needs both anchors, both hinge axes, both \
                  angle references, the target angle, and its stiffness and \
                  damping to be fully specified"
    )]
    pub fn spring(
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
    ) -> RevoluteDriveJoint {
        let drive_compliance = if drive_stiffness > 0.0 {
            1.0 / drive_stiffness
        } else {
            0.0
        };
        RevoluteDriveJoint::new(
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
        )
    }

    /// The two body indices this joint couples, in `(a, b)` order.
    #[must_use]
    pub fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }

    /// Packs the joint into its `GPU` storage-buffer representation.
    #[must_use]
    pub(crate) fn to_gpu(self) -> GpuRevoluteDriveJoint {
        GpuRevoluteDriveJoint {
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
            _pad0: 0.0,
        }
    }
}

impl JointBodies for RevoluteDriveJoint {
    fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }
}

/// Device-packed [`RevoluteDriveJoint`]; layout matches `Joint` in
/// `shaders/rigid_joint_revolute_drive.wgsl` (`128` bytes). The two anchors, two
/// hinge axes, and two angle references are padded to `vec4` so each starts on
/// the `16`-byte boundary the storage layout requires; the six `w` lanes are
/// unused. The two indices and the weld / axis compliances fill the next
/// `16`-byte block, and the drive compliance, drive damping, target angle, and a
/// pad word fill the last.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuRevoluteDriveJoint {
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
    /// Padding to the `16`-byte boundary; always zero.
    _pad0: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::FRAC_PI_4;

    #[test]
    fn gpu_packing_round_trips_fields() {
        let joint = RevoluteDriveJoint::new(
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
        assert_eq!(gpu._pad0, 0.0);
    }

    #[test]
    fn gpu_struct_is_128_bytes() {
        assert_eq!(size_of::<GpuRevoluteDriveJoint>(), 128);
    }

    #[test]
    fn servo_builds_a_rigid_zero_compliance_drive() {
        let joint = RevoluteDriveJoint::servo(
            0,
            1,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Z,
            Vec3::Z,
            Vec3::X,
            Vec3::X,
            FRAC_PI_4,
        );
        assert_eq!(joint.compliance, 0.0);
        assert_eq!(joint.angular_compliance, 0.0);
        assert_eq!(joint.drive_compliance, 0.0);
        assert_eq!(joint.drive_damping, 0.0);
        assert_eq!(joint.target_angle, FRAC_PI_4);
    }

    #[test]
    fn spring_inverts_stiffness_into_compliance() {
        let joint = RevoluteDriveJoint::spring(
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
        );
        assert!((joint.drive_compliance - 1.0 / 200.0).abs() < 1e-9);
        assert_eq!(joint.drive_damping, 5.0);
        assert_eq!(joint.compliance, 0.0);
        assert_eq!(joint.angular_compliance, 0.0);
    }

    #[test]
    fn spring_with_zero_stiffness_is_rigid() {
        let joint = RevoluteDriveJoint::spring(
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
        );
        assert_eq!(joint.drive_compliance, 0.0);
    }

    #[test]
    fn bodies_returns_pair_in_order() {
        let joint = RevoluteDriveJoint::servo(
            6,
            1,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Y,
            Vec3::Y,
            Vec3::X,
            Vec3::X,
            0.0,
        );
        assert_eq!(joint.bodies(), (6, 1));
        assert_eq!(JointBodies::bodies(&joint), (6, 1));
    }
}
