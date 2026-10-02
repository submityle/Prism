//! Per-degree-of-freedom *drives* (actuators) for the configurable six-degree-
//! of-freedom (`D6`) joint.
//!
//! A [`D6Joint`](super::d6::D6Joint) decides, per axis, whether a degree of
//! freedom is welded, limited, or free. A *drive* is the orthogonal concept: an
//! actuator that pushes a degree of freedom toward a target. Where the joint's
//! motion flags are passive (they only resist motion past a bound), a drive is
//! active — it injects force or torque every substep to servo the axis to a
//! commanded position and/or velocity. This is the model Unreal Engine's
//! `FConstraintDrive` and `PhysX`'s `PxD6JointDrive` expose: a spring-damper per
//! axis that turns a passive constraint into a motor.
//!
//! # The spring-damper per axis
//!
//! Each [`D6Drive`] is a parallel spring and damper on one scalar degree of
//! freedom. The spring pulls the axis coordinate `q` toward
//! [`target_position`](D6Drive::target_position) with gain
//! [`stiffness`](D6Drive::stiffness) (newtons per metre for the linear axes,
//! newton-metres per radian for the angular axes); the damper pulls the axis
//! rate toward [`target_velocity`](D6Drive::target_velocity) with gain
//! [`damping`](D6Drive::damping) (newton-seconds per metre, or newton-metre-
//! seconds per radian). Either gain may be zero: a pure position servo sets only
//! the stiffness, a pure velocity motor sets only the damping, and a drive with
//! both gains zero is inert and skipped entirely.
//!
//! # Relationship to the joint's motion flags
//!
//! A drive is independent of the axis's [`D6Motion`](super::d6::D6Motion) flag,
//! and the two compose: a `Free` axis with an active drive is a pure motor; a
//! `Limited` axis with a drive is a motor that still respects its mechanical
//! stop; a `Locked` axis's weld dominates any drive on the same axis, so driving
//! a locked axis is a no-op in practice. The caller is free to configure any
//! combination; the stepper applies the weld, the limit, and the drive each in
//! their own `XPBD` projection.
//!
//! # `XPBD` realisation
//!
//! A drive is projected as a compliant `XPBD` constraint whose compliance is the
//! inverse of the stiffness, `alpha = 1 / stiffness`, so an infinitely stiff
//! drive is a hard servo and a soft drive yields under load. The velocity target
//! is applied through the standard `XPBD` constraint-damping term (Müller et
//! al.), which couples the correction to the per-substep motion of the axis so
//! the drive bleeds the axis rate toward [`target_velocity`](D6Drive::target_velocity)
//! without an explicit force integration. The arithmetic lives in the drive
//! stepper and its `GPU` twin; this module only carries the data.
//!
//! Provenance: the per-axis spring-damper actuator of a general-purpose
//! constraint (Unreal Engine's `FConstraintDrive`, `PhysX`'s `PxD6JointDrive`),
//! realised as a compliant, velocity-damped `XPBD` constraint (Müller et al.,
//! "Detailed Rigid Body Simulation with XPBD"). No Unreal Engine source or
//! derived code: only the public spring-damper drive semantics are mirrored.

use bytemuck::{Pod, Zeroable};

/// A spring-damper actuator on one `D6` degree of freedom.
///
/// The drive adds, every substep, a servo toward [`target_position`](Self::target_position)
/// with gain [`stiffness`](Self::stiffness) and a damper toward
/// [`target_velocity`](Self::target_velocity) with gain [`damping`](Self::damping).
/// A drive with both gains zero (the [`Default`], also [`OFF`](Self::OFF)) is
/// inert and skipped.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct D6Drive {
    /// Spring gain pulling the axis coordinate toward
    /// [`target_position`](Self::target_position): newtons per metre on the
    /// linear axes, newton-metres per radian on the angular axes. Must be
    /// non-negative; zero disables the position servo.
    pub stiffness: f32,
    /// Damper gain pulling the axis rate toward
    /// [`target_velocity`](Self::target_velocity): newton-seconds per metre on
    /// the linear axes, newton-metre-seconds per radian on the angular axes.
    /// Must be non-negative; zero disables the velocity motor.
    pub damping: f32,
    /// The commanded axis coordinate the spring servos toward: metres of anchor
    /// separation along the frame axis for a linear drive, radians of twist or
    /// swing for an angular drive.
    pub target_position: f32,
    /// The commanded axis rate the damper servos toward: metres per second for a
    /// linear drive, radians per second for an angular drive.
    pub target_velocity: f32,
}

impl D6Drive {
    /// An inert drive: both gains zero, so the stepper skips it. Equivalent to
    /// [`D6Drive::default`].
    pub const OFF: D6Drive = D6Drive {
        stiffness: 0.0,
        damping: 0.0,
        target_position: 0.0,
        target_velocity: 0.0,
    };

    /// Creates a drive with every field given explicitly.
    #[must_use]
    pub const fn new(
        stiffness: f32,
        damping: f32,
        target_position: f32,
        target_velocity: f32,
    ) -> D6Drive {
        D6Drive {
            stiffness,
            damping,
            target_position,
            target_velocity,
        }
    }

    /// A pure position servo: a spring of the given `stiffness` toward
    /// `target_position`, with no velocity damping and a zero velocity target.
    #[must_use]
    pub const fn position(stiffness: f32, target_position: f32) -> D6Drive {
        D6Drive {
            stiffness,
            damping: 0.0,
            target_position,
            target_velocity: 0.0,
        }
    }

    /// A pure velocity motor: a damper of the given `damping` toward
    /// `target_velocity`, with no position spring and a zero position target.
    #[must_use]
    pub const fn velocity(damping: f32, target_velocity: f32) -> D6Drive {
        D6Drive {
            stiffness: 0.0,
            damping,
            target_position: 0.0,
            target_velocity,
        }
    }

    /// Whether the drive exerts any force this step: true when either gain is
    /// strictly positive. An inactive drive is skipped by the stepper.
    #[must_use]
    pub fn is_active(self) -> bool {
        self.stiffness > 0.0 || self.damping > 0.0
    }

    /// The `XPBD` compliance of the position spring, `1 / stiffness`, in metres
    /// per newton (linear) or radians per newton-metre (angular). Returns
    /// [`f32::INFINITY`] when the stiffness is zero (no position servo), which
    /// the stepper reads as "apply no positional correction".
    #[must_use]
    pub fn compliance(self) -> f32 {
        if self.stiffness > 0.0 {
            1.0 / self.stiffness
        } else {
            f32::INFINITY
        }
    }

    /// Packs the drive into its `GPU` storage-buffer representation.
    #[must_use]
    pub(crate) fn to_gpu(self) -> GpuD6Drive {
        GpuD6Drive {
            stiffness: self.stiffness,
            damping: self.damping,
            target_position: self.target_position,
            target_velocity: self.target_velocity,
        }
    }
}

/// The six per-axis drives attached to one [`D6Joint`](super::d6::D6Joint): one
/// [`D6Drive`] for each degree of freedom, in the same axis order the joint and
/// its multiplier layout use (linear `x` / `y` / `z`, then twist, swing1,
/// swing2).
///
/// A drive set is carried alongside the joint array rather than inside
/// [`D6Joint`](super::d6::D6Joint) so a scene that uses no motors pays nothing
/// for them and the joint's device layout stays unchanged. The [`rest`](Self::rest)
/// set has every axis inert.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct D6DriveSet {
    /// Drive on the linear `x` axis (anchor separation along the frame `x`).
    pub linear_x: D6Drive,
    /// Drive on the linear `y` axis.
    pub linear_y: D6Drive,
    /// Drive on the linear `z` axis.
    pub linear_z: D6Drive,
    /// Drive on the twist axis (rotation about the frame `x`).
    pub twist: D6Drive,
    /// Drive on the swing1 axis (tilt toward the frame `y`).
    pub swing1: D6Drive,
    /// Drive on the swing2 axis (tilt toward the frame `z`).
    pub swing2: D6Drive,
}

impl D6DriveSet {
    /// A drive set with every axis inert: no motor on any degree of freedom.
    /// Equivalent to [`D6DriveSet::default`].
    pub const REST: D6DriveSet = D6DriveSet {
        linear_x: D6Drive::OFF,
        linear_y: D6Drive::OFF,
        linear_z: D6Drive::OFF,
        twist: D6Drive::OFF,
        swing1: D6Drive::OFF,
        swing2: D6Drive::OFF,
    };

    /// Creates a drive set from the three linear and three angular drives, each
    /// in frame-axis order (`[x, y, z]` and `[twist, swing1, swing2]`).
    #[must_use]
    pub const fn new(linear: [D6Drive; 3], angular: [D6Drive; 3]) -> D6DriveSet {
        D6DriveSet {
            linear_x: linear[0],
            linear_y: linear[1],
            linear_z: linear[2],
            twist: angular[0],
            swing1: angular[1],
            swing2: angular[2],
        }
    }

    /// An all-inert drive set. Equivalent to [`D6DriveSet::REST`].
    #[must_use]
    pub const fn rest() -> D6DriveSet {
        D6DriveSet::REST
    }

    /// Whether any of the six axes carries an active drive. A set for which this
    /// is false contributes nothing and the stepper may skip it wholesale.
    #[must_use]
    pub fn any_active(self) -> bool {
        self.linear_x.is_active()
            || self.linear_y.is_active()
            || self.linear_z.is_active()
            || self.twist.is_active()
            || self.swing1.is_active()
            || self.swing2.is_active()
    }

    /// Packs the drive set into its `GPU` storage-buffer representation.
    #[must_use]
    pub(crate) fn to_gpu(self) -> GpuD6DriveSet {
        GpuD6DriveSet {
            linear_x: self.linear_x.to_gpu(),
            linear_y: self.linear_y.to_gpu(),
            linear_z: self.linear_z.to_gpu(),
            twist: self.twist.to_gpu(),
            swing1: self.swing1.to_gpu(),
            swing2: self.swing2.to_gpu(),
        }
    }
}

/// Device-packed [`D6Drive`]: four floats (`16` bytes), matching one `Drive`
/// element in `shaders/rigid_joint_d6_drive.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuD6Drive {
    /// Position-spring gain; zero disables the position servo.
    stiffness: f32,
    /// Velocity-damper gain; zero disables the velocity motor.
    damping: f32,
    /// Target axis coordinate (metres or radians).
    target_position: f32,
    /// Target axis rate (metres per second or radians per second).
    target_velocity: f32,
}

/// Device-packed [`D6DriveSet`]: six [`GpuD6Drive`] in axis order (`96` bytes,
/// a multiple of the `16`-byte storage alignment with no padding).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuD6DriveSet {
    /// Drive on the linear `x` axis.
    linear_x: GpuD6Drive,
    /// Drive on the linear `y` axis.
    linear_y: GpuD6Drive,
    /// Drive on the linear `z` axis.
    linear_z: GpuD6Drive,
    /// Drive on the twist axis.
    twist: GpuD6Drive,
    /// Drive on the swing1 axis.
    swing1: GpuD6Drive,
    /// Drive on the swing2 axis.
    swing2: GpuD6Drive,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_drive_is_inert() {
        assert!(!D6Drive::OFF.is_active());
        assert_eq!(D6Drive::default(), D6Drive::OFF);
        assert_eq!(D6Drive::OFF.compliance(), f32::INFINITY);
    }

    #[test]
    fn position_drive_sets_only_the_spring() {
        let drive = D6Drive::position(400.0, 0.25);
        assert_eq!(drive.stiffness, 400.0);
        assert_eq!(drive.damping, 0.0);
        assert_eq!(drive.target_position, 0.25);
        assert_eq!(drive.target_velocity, 0.0);
        assert!(drive.is_active());
        assert_eq!(drive.compliance(), 1.0 / 400.0);
    }

    #[test]
    fn velocity_drive_sets_only_the_damper() {
        let drive = D6Drive::velocity(12.0, -1.5);
        assert_eq!(drive.stiffness, 0.0);
        assert_eq!(drive.damping, 12.0);
        assert_eq!(drive.target_position, 0.0);
        assert_eq!(drive.target_velocity, -1.5);
        assert!(drive.is_active());
        // No spring, so the position compliance is infinite (servo disabled).
        assert_eq!(drive.compliance(), f32::INFINITY);
    }

    #[test]
    fn active_requires_a_positive_gain() {
        assert!(!D6Drive::new(0.0, 0.0, 1.0, 1.0).is_active());
        assert!(D6Drive::new(1.0, 0.0, 0.0, 0.0).is_active());
        assert!(D6Drive::new(0.0, 1.0, 0.0, 0.0).is_active());
    }

    #[test]
    fn drive_set_axis_order_matches_the_constructor() {
        let set = D6DriveSet::new(
            [
                D6Drive::position(1.0, 0.1),
                D6Drive::position(2.0, 0.2),
                D6Drive::position(3.0, 0.3),
            ],
            [
                D6Drive::velocity(4.0, 0.4),
                D6Drive::velocity(5.0, 0.5),
                D6Drive::velocity(6.0, 0.6),
            ],
        );
        assert_eq!(set.linear_x, D6Drive::position(1.0, 0.1));
        assert_eq!(set.linear_y, D6Drive::position(2.0, 0.2));
        assert_eq!(set.linear_z, D6Drive::position(3.0, 0.3));
        assert_eq!(set.twist, D6Drive::velocity(4.0, 0.4));
        assert_eq!(set.swing1, D6Drive::velocity(5.0, 0.5));
        assert_eq!(set.swing2, D6Drive::velocity(6.0, 0.6));
    }

    #[test]
    fn rest_set_is_fully_inert() {
        let rest = D6DriveSet::rest();
        assert_eq!(rest, D6DriveSet::REST);
        assert_eq!(rest, D6DriveSet::default());
        assert!(!rest.any_active());
    }

    #[test]
    fn any_active_detects_a_single_live_axis() {
        let mut set = D6DriveSet::rest();
        assert!(!set.any_active());
        set.swing2 = D6Drive::position(10.0, 0.0);
        assert!(set.any_active());
    }

    #[test]
    fn gpu_drive_is_16_bytes_and_set_is_96() {
        assert_eq!(size_of::<GpuD6Drive>(), 16);
        assert_eq!(size_of::<GpuD6DriveSet>(), 96);
    }

    #[test]
    fn gpu_packing_round_trips_fields() {
        let set = D6DriveSet::new(
            [
                D6Drive::new(1.0, 2.0, 3.0, 4.0),
                D6Drive::OFF,
                D6Drive::position(7.0, 0.5),
            ],
            [
                D6Drive::velocity(9.0, -0.25),
                D6Drive::OFF,
                D6Drive::new(11.0, 12.0, 13.0, 14.0),
            ],
        );
        let gpu = set.to_gpu();
        assert_eq!(gpu.linear_x.stiffness, 1.0);
        assert_eq!(gpu.linear_x.damping, 2.0);
        assert_eq!(gpu.linear_x.target_position, 3.0);
        assert_eq!(gpu.linear_x.target_velocity, 4.0);
        assert_eq!(gpu.linear_z.stiffness, 7.0);
        assert_eq!(gpu.linear_z.target_position, 0.5);
        assert_eq!(gpu.twist.damping, 9.0);
        assert_eq!(gpu.twist.target_velocity, -0.25);
        assert_eq!(gpu.swing2.stiffness, 11.0);
        assert_eq!(gpu.swing2.target_velocity, 14.0);
    }
}
