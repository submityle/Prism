//! Joint limits and motors.
//!
//! These value types drive the optional extra behaviour on the axial degree of
//! freedom of a hinge (revolute) or slider (prismatic) joint:
//!
//! - [`AngleLimit`] / [`LinearLimit`] clamp the free coordinate to a range.
//! - [`Motor`] actively drives the free coordinate toward a target velocity or
//!   target position, subject to a maximum force/torque budget.
//!
//! Motors are resolved in the position domain: a velocity target is converted
//! into a per-sub-step positional goal, and the force budget bounds the impulse
//! applied each sub-step. This keeps motors consistent with the rest of the
//! XPBD solve rather than bolting on a separate velocity pass.
//!
//! # Provenance
//!
//! Original data types expressing standard joint-limit and joint-motor
//! concepts. They contain **no Unreal Engine source or derived code**.

/// An inclusive angular range, in radians, for a hinge's free rotation.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AngleLimit {
    /// Lower bound of the allowed angle, in radians.
    pub min: f32,
    /// Upper bound of the allowed angle, in radians.
    pub max: f32,
}

impl AngleLimit {
    /// Creates an angular limit from an explicit `min`/`max` pair (radians).
    #[must_use]
    pub const fn new(min: f32, max: f32) -> AngleLimit {
        AngleLimit { min, max }
    }

    /// Creates a symmetric limit `[-half, half]` (radians).
    #[must_use]
    pub const fn symmetric(half: f32) -> AngleLimit {
        AngleLimit {
            min: -half,
            max: half,
        }
    }

    /// Clamps `value` into the limit range.
    #[must_use]
    pub fn clamp(&self, value: f32) -> f32 {
        value.clamp(self.min, self.max)
    }
}

/// An inclusive linear range, in metres, for a slider's free translation.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct LinearLimit {
    /// Lower bound of the allowed offset along the slide axis, in metres.
    pub min: f32,
    /// Upper bound of the allowed offset along the slide axis, in metres.
    pub max: f32,
}

impl LinearLimit {
    /// Creates a linear limit from an explicit `min`/`max` pair (metres).
    #[must_use]
    pub const fn new(min: f32, max: f32) -> LinearLimit {
        LinearLimit { min, max }
    }

    /// Clamps `value` into the limit range.
    #[must_use]
    pub fn clamp(&self, value: f32) -> f32 {
        value.clamp(self.min, self.max)
    }
}

/// What a [`Motor`] drives its joint's free coordinate toward.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum MotorTarget {
    /// Drive the free coordinate at a constant rate (radians per second for a
    /// hinge, metres per second for a slider).
    Velocity(f32),
    /// Drive the free coordinate toward a fixed position (radians for a hinge,
    /// metres for a slider).
    Position(f32),
}

/// An actuator on a joint's free coordinate.
///
/// A motor adds a compliant driving constraint on the hinge angle or slide
/// offset. `max_force` bounds the corrective impulse applied each sub-step so
/// the motor cannot inject unbounded energy; `compliance` softens the drive
/// (`0` is a perfectly stiff servo).
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Motor {
    /// The coordinate goal (target velocity or target position).
    pub target: MotorTarget,
    /// Maximum force (slider) or torque (hinge) the motor can exert. Bounds the
    /// per-sub-step impulse to `max_force * h^2`.
    pub max_force: f32,
    /// Drive compliance (inverse stiffness). `0` is a rigid servo.
    pub compliance: f32,
}

impl Motor {
    /// Creates a velocity-target motor with the given force/torque budget.
    #[must_use]
    pub const fn velocity(target: f32, max_force: f32) -> Motor {
        Motor {
            target: MotorTarget::Velocity(target),
            max_force,
            compliance: 0.0,
        }
    }

    /// Creates a position-target motor with the given force/torque budget.
    #[must_use]
    pub const fn position(target: f32, max_force: f32) -> Motor {
        Motor {
            target: MotorTarget::Position(target),
            max_force,
            compliance: 0.0,
        }
    }

    /// Sets the drive compliance and returns the modified motor.
    #[must_use]
    pub const fn with_compliance(mut self, compliance: f32) -> Motor {
        self.compliance = compliance;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angle_limit_clamps() {
        let l = AngleLimit::symmetric(1.0);
        assert!((l.clamp(2.0) - 1.0).abs() < 1e-6);
        assert!((l.clamp(-2.0) + 1.0).abs() < 1e-6);
        assert!((l.clamp(0.25) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn linear_limit_clamps() {
        let l = LinearLimit::new(-0.5, 0.5);
        assert!((l.clamp(1.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn motor_builders_set_target() {
        let v = Motor::velocity(2.0, 10.0);
        assert_eq!(v.target, MotorTarget::Velocity(2.0));
        let p = Motor::position(0.3, 5.0).with_compliance(0.01);
        assert_eq!(p.target, MotorTarget::Position(0.3));
        assert!((p.compliance - 0.01).abs() < 1e-9);
    }
}
