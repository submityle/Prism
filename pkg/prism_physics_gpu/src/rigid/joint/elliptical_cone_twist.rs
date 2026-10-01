//! The elliptical cone-twist rigid-body joint.
//!
//! An [`EllipticalConeTwistJoint`] generalises the circular
//! [`SwingTwistJoint`](super::SwingTwistJoint) ragdoll joint by replacing its
//! single [`swing_limit`](super::SwingTwistJoint::swing_limit) half-angle with
//! two *independent* swing half-angles, so the twist axis of body `b` is bounded
//! within an **elliptical** cone rather than a circular one. This is the
//! shoulder / hip model used by production ragdolls (Unreal Engine's
//! `FConstraintInstance` exposes distinct `Swing1LimitDegrees` and
//! `Swing2LimitDegrees`, `PhysX`'s `PxJointLimitCone` distinct `yAngle`/`zAngle`):
//! a limb that may lean much further forward/back than side to side, swept
//! inside an elliptical rather than circular rim.
//!
//! It is the strict superset of the circular cone-twist: setting both swing
//! half-angles equal recovers the circular [`SwingTwistJoint`](
//! super::SwingTwistJoint) exactly (see the degeneration parity scene).
//!
//! # The three constraints
//!
//! * **Point-to-point** (positional): the world-space anchor coincidence the
//!   spherical joint drives to zero, `p_a - p_b -> 0`.
//! * **Elliptical swing cone** (angular, one-sided): with
//!   `t = rotate(orientation_a, twist_axis_a)` body `a`'s unit twist axis and
//!   `u_b = rotate(orientation_b, twist_axis_b)` body `b`'s, the tilt of `u_b`
//!   from `t` is decomposed in the swing plane spanned by two body-local-`a`
//!   axes: `s1`, the component of [`ref_a`](EllipticalConeTwistJoint::ref_a)
//!   perpendicular to `t`, and `s2 = t x s1`. The tilt magnitude is
//!   `phi = acos(clamp(t . u_b, -1, 1))` at azimuth `psi` within the `(s1, s2)`
//!   plane; the elliptical rim at that azimuth is
//!   `phi_max = 1 / sqrt((cos psi / swing1_limit)^2 + (sin psi / swing2_limit)^2)`.
//!   While `phi <= phi_max` the cone is inactive; past it the violation
//!   `phi - phi_max` is driven shut by a rotation about
//!   `n = normalize(t x u_b)` — the axis that closes the swing radially while
//!   leaving the azimuth (and hence `phi_max`) fixed — exactly as the circular
//!   cone closes its rim.
//! * **Twist limit** (angular, one-sided): the signed rotation about the twist
//!   axis, measured between the two bodies' reference directions exactly as the
//!   circular cone-twist and the hinge limit measure it, is clamped into
//!   `[twist_min, twist_max]` with a free dead zone inside the range.
//!
//! Each sweep projects the elliptical swing cone first, then the twist limit
//! (about the twist axis), then the positional weld.
//!
//! # Why the radial azimuth projection is exact for the swing component
//!
//! Rotating `u_b` about `n = normalize(t x u_b)` toward `t` reduces the tilt
//! `phi` while keeping `u_b` in the plane spanned by `(t, u_b)`; its projection
//! onto the `(s1, s2)` swing plane keeps its *direction* (the azimuth `psi`) and
//! only shrinks in magnitude. Because `phi_max(psi)` depends on the azimuth
//! alone, it is invariant under this correction, so the constraint
//! `C = phi - phi_max(psi)` is reduced purely through its `phi` term — the same
//! stable, single-axis angular projection the circular cone uses, now with an
//! azimuth-dependent rim.
//!
//! # The ellipse orientation
//!
//! The swing plane's first axis `s1` is [`ref_a`](EllipticalConeTwistJoint::ref_a)
//! projected perpendicular to the twist axis — the same reference the twist
//! limit measures its zero angle from — so `swing1_limit` bounds the tilt toward
//! `+/- s1` and `swing2_limit` the tilt toward `+/- s2 = t x s1`. Reusing
//! `ref_a` ties the ellipse's major/minor axes to the joint's twist reference,
//! so no extra body-local frame is needed. Both swing limits must be positive;
//! a locked swing axis (zero limit) belongs to a general 6-DOF joint rather than
//! an elliptical cone.
//!
//! # Compliance
//!
//! [`compliance`](EllipticalConeTwistJoint::compliance) is the inverse stiffness
//! of the positional weld (metres per newton),
//! [`swing_compliance`](EllipticalConeTwistJoint::swing_compliance) the inverse
//! stiffness of the elliptical swing rim (radians per newton-metre), and
//! [`twist_compliance`](EllipticalConeTwistJoint::twist_compliance) the inverse
//! stiffness of the twist limit (radians per newton-metre). Zero is the rigid
//! limit for each; a positive value yields a soft, springy stop. Each is divided
//! by the squared substep time to form the time-step-independent `XPBD`
//! regularisation term.
//!
//! Provenance: the point-to-point (ball-socket) constraint, the signed angular
//! limit shared with the hinge limit, and the swing-cone limit generalised to an
//! elliptical rim, with their substep `XPBD` handling (Müller et al., "Detailed
//! Rigid Body Simulation with XPBD"), over the world-space inverse inertia and
//! quaternion kinematics of Baraff & Witkin. The elliptical two-angle swing
//! parametrisation follows the standard cone-twist of production engines. No
//! Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use super::coloring::JointBodies;

/// An elliptical cone-twist joint: a point-to-point weld plus a one-sided
/// elliptical swing-cone limit (independent half-angles about the two axes
/// perpendicular to the twist axis) and a one-sided twist limit on the axial
/// spin.
///
/// The interior of both the elliptical cone and the twist range is free; only a
/// violated bound exerts a torque. Either body may be static (zero inverse mass
/// and inertia); a joint between a dynamic body and a static one sockets the
/// dynamic body about a fixed world anchor with a fixed elliptical cone and
/// twist range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EllipticalConeTwistJoint {
    /// Index of the first body in the [`RigidBodyState`](super::super::RigidBodyState).
    pub body_a: u32,
    /// Index of the second body in the [`RigidBodyState`](super::super::RigidBodyState).
    pub body_b: u32,
    /// Anchor point on body `a`, in body `a`'s local frame (relative to its
    /// centre of mass).
    pub anchor_a: Vec3,
    /// Anchor point on body `b`, in body `b`'s local frame.
    pub anchor_b: Vec3,
    /// Twist axis on body `a`, in body `a`'s local frame. Need not be
    /// unit-length; only its direction matters.
    pub twist_axis_a: Vec3,
    /// Twist axis on body `b`, in body `b`'s local frame.
    pub twist_axis_b: Vec3,
    /// Zero-angle reference direction on body `a`, in body `a`'s local frame.
    /// Must be non-parallel to [`twist_axis_a`](Self::twist_axis_a); its
    /// perpendicular component is the ellipse's first swing axis `s1` and the
    /// twist limit's zero reference.
    pub ref_a: Vec3,
    /// Zero-angle reference direction on body `b`, in body `b`'s local frame.
    /// Must be non-parallel to [`twist_axis_b`](Self::twist_axis_b).
    pub ref_b: Vec3,
    /// Swing half-angle toward the `+/- s1` axis (the perpendicular component of
    /// [`ref_a`](Self::ref_a)), in radians. Must be positive.
    pub swing1_limit: f32,
    /// Swing half-angle toward the `+/- s2 = t x s1` axis, in radians. Must be
    /// positive.
    pub swing2_limit: f32,
    /// Lower bound of the free twist range, in radians.
    pub twist_min: f32,
    /// Upper bound of the free twist range, in radians.
    pub twist_max: f32,
    /// Inverse stiffness (metres per newton) of the positional weld. Zero is the
    /// rigid limit.
    pub compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the elliptical swing rim.
    /// Zero is the rigid limit (a hard rim).
    pub swing_compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the twist limit. Zero is
    /// the rigid limit (a hard stop).
    pub twist_compliance: f32,
}

impl EllipticalConeTwistJoint {
    /// Creates an elliptical cone-twist joint from its full specification.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the twist range is malformed
    /// (`twist_min > twist_max`) or either swing half-angle is non-positive;
    /// callers are expected to pass a well-ordered range and positive cone
    /// half-angles.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "an elliptical cone-twist joint is defined by two anchors, two \
                  twist axes, two angle references, two swing half-angles, a \
                  twist range, and three compliances across the two bodies it \
                  couples"
    )]
    pub fn new(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        twist_axis_a: Vec3,
        twist_axis_b: Vec3,
        ref_a: Vec3,
        ref_b: Vec3,
        swing1_limit: f32,
        swing2_limit: f32,
        twist_min: f32,
        twist_max: f32,
        compliance: f32,
        swing_compliance: f32,
        twist_compliance: f32,
    ) -> EllipticalConeTwistJoint {
        debug_assert!(
            swing1_limit > 0.0 && swing2_limit > 0.0,
            "elliptical swing half-angles must be positive"
        );
        debug_assert!(
            twist_min <= twist_max,
            "twist range must satisfy twist_min <= twist_max"
        );
        EllipticalConeTwistJoint {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            twist_axis_a,
            twist_axis_b,
            ref_a,
            ref_b,
            swing1_limit,
            swing2_limit,
            twist_min,
            twist_max,
            compliance,
            swing_compliance,
            twist_compliance,
        }
    }

    /// Creates a rigid (zero-compliance) elliptical cone-twist joint with a twist
    /// range symmetric about zero: the limb may swing freely within the
    /// elliptical cone of half-angles `swing1_limit` and `swing2_limit` and
    /// twist freely within `[-twist_limit, +twist_limit]`, hard-stopped outside
    /// either bound.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the socket geometry alone needs two anchors, two twist axes, \
                  and two references across the two coupled bodies"
    )]
    pub fn symmetric_cone(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        twist_axis_a: Vec3,
        twist_axis_b: Vec3,
        ref_a: Vec3,
        ref_b: Vec3,
        swing1_limit: f32,
        swing2_limit: f32,
        twist_limit: f32,
    ) -> EllipticalConeTwistJoint {
        EllipticalConeTwistJoint::new(
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            twist_axis_a,
            twist_axis_b,
            ref_a,
            ref_b,
            swing1_limit,
            swing2_limit,
            -twist_limit,
            twist_limit,
            0.0,
            0.0,
            0.0,
        )
    }

    /// Returns the `(body_a, body_b)` index pair this joint couples.
    #[must_use]
    pub fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }

    /// Packs the joint into its `GPU` storage-buffer representation.
    #[must_use]
    pub(crate) fn to_gpu(self) -> GpuEllipticalConeTwistJoint {
        GpuEllipticalConeTwistJoint {
            anchor_a: [self.anchor_a.x, self.anchor_a.y, self.anchor_a.z, 0.0],
            anchor_b: [self.anchor_b.x, self.anchor_b.y, self.anchor_b.z, 0.0],
            twist_axis_a: [
                self.twist_axis_a.x,
                self.twist_axis_a.y,
                self.twist_axis_a.z,
                0.0,
            ],
            twist_axis_b: [
                self.twist_axis_b.x,
                self.twist_axis_b.y,
                self.twist_axis_b.z,
                0.0,
            ],
            ref_a: [self.ref_a.x, self.ref_a.y, self.ref_a.z, 0.0],
            ref_b: [self.ref_b.x, self.ref_b.y, self.ref_b.z, 0.0],
            body_a: self.body_a,
            body_b: self.body_b,
            swing1_limit: self.swing1_limit,
            swing2_limit: self.swing2_limit,
            twist_min: self.twist_min,
            twist_max: self.twist_max,
            compliance: self.compliance,
            swing_compliance: self.swing_compliance,
            twist_compliance: self.twist_compliance,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }
}

impl JointBodies for EllipticalConeTwistJoint {
    fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }
}

/// Device-packed [`EllipticalConeTwistJoint`]; layout matches `Joint` in
/// `shaders/rigid_joint_elliptical_cone_twist.wgsl` (`144` bytes). The two
/// anchors, two twist axes, and two angle references are padded to `vec4` so each
/// starts on the `16`-byte boundary the storage layout requires; the six `w`
/// lanes are unused. The two indices and two swing half-angles fill the next
/// `16`-byte block; the twist range and the weld/swing compliances the one
/// after; the twist compliance plus three pad words close the final block.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuEllipticalConeTwistJoint {
    /// Body-local anchor on body `a` in `xyz`; `w` unused.
    anchor_a: [f32; 4],
    /// Body-local anchor on body `b` in `xyz`; `w` unused.
    anchor_b: [f32; 4],
    /// Body-local twist axis on body `a` in `xyz`; `w` unused.
    twist_axis_a: [f32; 4],
    /// Body-local twist axis on body `b` in `xyz`; `w` unused.
    twist_axis_b: [f32; 4],
    /// Body-local zero-angle reference on body `a` in `xyz`; `w` unused.
    ref_a: [f32; 4],
    /// Body-local zero-angle reference on body `b` in `xyz`; `w` unused.
    ref_b: [f32; 4],
    /// Index of body `a`.
    body_a: u32,
    /// Index of body `b`.
    body_b: u32,
    /// Swing half-angle toward `+/- s1`, in radians.
    swing1_limit: f32,
    /// Swing half-angle toward `+/- s2`, in radians.
    swing2_limit: f32,
    /// Lower bound of the free twist range, in radians.
    twist_min: f32,
    /// Upper bound of the free twist range, in radians.
    twist_max: f32,
    /// Inverse stiffness (metres per newton) of the positional weld.
    compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the elliptical swing rim.
    swing_compliance: f32,
    /// Inverse stiffness (radians per newton-metre) of the twist limit.
    twist_compliance: f32,
    /// Padding to the `16`-byte storage boundary; always zero.
    _pad0: f32,
    /// Padding to the `16`-byte storage boundary; always zero.
    _pad1: f32,
    /// Padding to the `16`-byte storage boundary; always zero.
    _pad2: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_6};

    #[test]
    fn gpu_struct_is_144_bytes() {
        assert_eq!(size_of::<GpuEllipticalConeTwistJoint>(), 144);
    }

    #[test]
    fn gpu_packing_round_trips_fields() {
        let joint = EllipticalConeTwistJoint::new(
            2,
            9,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-4.0, 5.0, -6.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.7,
            0.4,
            -0.5,
            0.75,
            0.002,
            0.001,
            0.004,
        );
        let gpu = joint.to_gpu();
        assert_eq!(gpu.anchor_a, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(gpu.anchor_b, [-4.0, 5.0, -6.0, 0.0]);
        assert_eq!(gpu.twist_axis_a, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(gpu.twist_axis_b, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(gpu.ref_a, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(gpu.ref_b, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(gpu.body_a, 2);
        assert_eq!(gpu.body_b, 9);
        assert_eq!(gpu.swing1_limit, 0.7);
        assert_eq!(gpu.swing2_limit, 0.4);
        assert_eq!(gpu.twist_min, -0.5);
        assert_eq!(gpu.twist_max, 0.75);
        assert_eq!(gpu.compliance, 0.002);
        assert_eq!(gpu.swing_compliance, 0.001);
        assert_eq!(gpu.twist_compliance, 0.004);
        assert_eq!([gpu._pad0, gpu._pad1, gpu._pad2], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn symmetric_cone_builds_rigid_elliptical_cone() {
        let joint = EllipticalConeTwistJoint::symmetric_cone(
            0,
            1,
            Vec3::ZERO,
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::Y,
            Vec3::Y,
            Vec3::X,
            Vec3::X,
            FRAC_PI_2,
            FRAC_PI_6,
            FRAC_PI_6,
        );
        assert_eq!(joint.swing1_limit, FRAC_PI_2);
        assert_eq!(joint.swing2_limit, FRAC_PI_6);
        assert_eq!(joint.twist_min, -FRAC_PI_6);
        assert_eq!(joint.twist_max, FRAC_PI_6);
        assert_eq!(joint.compliance, 0.0);
        assert_eq!(joint.swing_compliance, 0.0);
        assert_eq!(joint.twist_compliance, 0.0);
    }

    #[test]
    fn bodies_returns_pair_in_order() {
        let joint = EllipticalConeTwistJoint::symmetric_cone(
            4,
            7,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Y,
            Vec3::Y,
            Vec3::X,
            Vec3::X,
            0.5,
            0.3,
            0.2,
        );
        assert_eq!(joint.bodies(), (4, 7));
        assert_eq!(JointBodies::bodies(&joint), (4, 7));
    }
}
