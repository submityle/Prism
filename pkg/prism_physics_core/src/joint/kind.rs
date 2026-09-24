//! Joint kinds and their per-kind parameters.
//!
//! [`JointKind`] enumerates the constraint families Prism's M2 milestone
//! supports. Each variant carries a small parameter struct describing the free
//! degrees of freedom, optional limits, motor, and compliance:
//!
//! - [`FixedJoint`] welds two bodies rigidly (0 free DOF).
//! - [`DistanceJoint`] keeps the anchor separation inside a length range
//!   (rope / rigid rod).
//! - [`SphericalJoint`] is a ball-and-socket: anchors coincide, rotation free.
//! - [`RevoluteJoint`] is a hinge: one free rotation about an axis, with an
//!   optional angle limit and motor.
//! - [`PrismaticJoint`] is a slider: one free translation along an axis, with
//!   an optional linear limit and motor.
//!
//! # Provenance
//!
//! Original data types describing standard mechanical-joint families. They
//! contain **no Unreal Engine source or derived code**.

use crate::joint::motor::{AngleLimit, LinearLimit, Motor};
use glam::Vec3;

/// A rigid weld between two bodies: the anchor points coincide and the
/// reference frames stay aligned. No relative motion is permitted.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct FixedJoint {
    /// Constraint compliance (inverse stiffness). `0` is a perfectly rigid
    /// weld; small positive values make it springy.
    pub compliance: f32,
}

impl Default for FixedJoint {
    fn default() -> Self {
        FixedJoint { compliance: 0.0 }
    }
}

/// Keeps the distance between the two anchor points within
/// `[min_length, max_length]`. Set both bounds equal for a rigid rod, or
/// `min_length = 0` for a rope that only resists stretching.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DistanceJoint {
    /// Minimum allowed anchor separation, in metres.
    pub min_length: f32,
    /// Maximum allowed anchor separation, in metres.
    pub max_length: f32,
    /// Constraint compliance (inverse stiffness). `0` is a rigid link.
    pub compliance: f32,
}

impl DistanceJoint {
    /// Creates a rigid rod of fixed `length`.
    #[must_use]
    pub const fn rigid(length: f32) -> DistanceJoint {
        DistanceJoint {
            min_length: length,
            max_length: length,
            compliance: 0.0,
        }
    }

    /// Creates a link that keeps the separation within `[min, max]`.
    #[must_use]
    pub const fn range(min: f32, max: f32) -> DistanceJoint {
        DistanceJoint {
            min_length: min,
            max_length: max,
            compliance: 0.0,
        }
    }

    /// Sets the compliance and returns the modified joint.
    #[must_use]
    pub const fn with_compliance(mut self, compliance: f32) -> DistanceJoint {
        self.compliance = compliance;
        self
    }
}

/// A ball-and-socket joint: the two anchor points are held coincident while
/// relative rotation stays completely free.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SphericalJoint {
    /// Constraint compliance (inverse stiffness) of the point-coincidence
    /// constraint. `0` is rigid.
    pub compliance: f32,
}

impl Default for SphericalJoint {
    fn default() -> Self {
        SphericalJoint { compliance: 0.0 }
    }
}

/// A hinge joint: the anchor points coincide and the two reference frames may
/// only differ by a rotation about `axis` (expressed in the anchor's local
/// frame). Optional [`AngleLimit`] and [`Motor`] act on that free rotation.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct RevoluteJoint {
    /// Hinge axis in the anchor reference frame. Normalised on use.
    pub axis: Vec3,
    /// Optional angular range for the free rotation.
    pub limit: Option<AngleLimit>,
    /// Optional actuator driving the free rotation.
    pub motor: Option<Motor>,
    /// Constraint compliance (inverse stiffness) of the coincidence and
    /// axis-alignment constraints. `0` is rigid.
    pub compliance: f32,
}

impl RevoluteJoint {
    /// Creates a plain hinge about `axis` with no limit or motor.
    #[must_use]
    pub const fn new(axis: Vec3) -> RevoluteJoint {
        RevoluteJoint {
            axis,
            limit: None,
            motor: None,
            compliance: 0.0,
        }
    }

    /// Sets the angle limit and returns the modified joint.
    #[must_use]
    pub const fn with_limit(mut self, limit: AngleLimit) -> RevoluteJoint {
        self.limit = Some(limit);
        self
    }

    /// Sets the motor and returns the modified joint.
    #[must_use]
    pub const fn with_motor(mut self, motor: Motor) -> RevoluteJoint {
        self.motor = Some(motor);
        self
    }

    /// Sets the compliance and returns the modified joint.
    #[must_use]
    pub const fn with_compliance(mut self, compliance: f32) -> RevoluteJoint {
        self.compliance = compliance;
        self
    }
}

/// A slider joint: relative rotation is fully locked and translation is
/// permitted only along `axis` (expressed in the anchor reference frame).
/// Optional [`LinearLimit`] and [`Motor`] act on that free translation.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PrismaticJoint {
    /// Slide axis in the anchor reference frame. Normalised on use.
    pub axis: Vec3,
    /// Optional linear range for the free translation.
    pub limit: Option<LinearLimit>,
    /// Optional actuator driving the free translation.
    pub motor: Option<Motor>,
    /// Constraint compliance (inverse stiffness) of the perpendicular-lock and
    /// rotation-lock constraints. `0` is rigid.
    pub compliance: f32,
}

impl PrismaticJoint {
    /// Creates a plain slider along `axis` with no limit or motor.
    #[must_use]
    pub const fn new(axis: Vec3) -> PrismaticJoint {
        PrismaticJoint {
            axis,
            limit: None,
            motor: None,
            compliance: 0.0,
        }
    }

    /// Sets the linear limit and returns the modified joint.
    #[must_use]
    pub const fn with_limit(mut self, limit: LinearLimit) -> PrismaticJoint {
        self.limit = Some(limit);
        self
    }

    /// Sets the motor and returns the modified joint.
    #[must_use]
    pub const fn with_motor(mut self, motor: Motor) -> PrismaticJoint {
        self.motor = Some(motor);
        self
    }

    /// Sets the compliance and returns the modified joint.
    #[must_use]
    pub const fn with_compliance(mut self, compliance: f32) -> PrismaticJoint {
        self.compliance = compliance;
        self
    }
}

/// The constraint family of a joint, tagged with its parameters.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum JointKind {
    /// A rigid weld (see [`FixedJoint`]).
    Fixed(FixedJoint),
    /// A length-limited link (see [`DistanceJoint`]).
    Distance(DistanceJoint),
    /// A ball-and-socket joint (see [`SphericalJoint`]).
    Spherical(SphericalJoint),
    /// A hinge joint (see [`RevoluteJoint`]).
    Revolute(RevoluteJoint),
    /// A slider joint (see [`PrismaticJoint`]).
    Prismatic(PrismaticJoint),
}

impl JointKind {
    /// Returns the compliance of the joint's primary coupling constraints.
    #[must_use]
    pub fn compliance(&self) -> f32 {
        match self {
            JointKind::Fixed(j) => j.compliance,
            JointKind::Distance(j) => j.compliance,
            JointKind::Spherical(j) => j.compliance,
            JointKind::Revolute(j) => j.compliance,
            JointKind::Prismatic(j) => j.compliance,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_rigid_has_equal_bounds() {
        let d = DistanceJoint::rigid(2.0);
        assert_eq!(d.min_length, d.max_length);
        assert_eq!(d.min_length, 2.0);
    }

    #[test]
    fn revolute_builder_chains() {
        let j = RevoluteJoint::new(Vec3::Y)
            .with_limit(AngleLimit::symmetric(1.0))
            .with_motor(Motor::velocity(1.0, 5.0))
            .with_compliance(0.001);
        assert!(j.limit.is_some());
        assert!(j.motor.is_some());
        assert!((j.compliance - 0.001).abs() < 1e-9);
    }

    #[test]
    fn kind_reports_compliance() {
        let k = JointKind::Spherical(SphericalJoint { compliance: 0.5 });
        assert!((k.compliance() - 0.5).abs() < 1e-9);
    }
}
