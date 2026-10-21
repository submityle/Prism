//! Scalar generalized-force actuation, separate from an XPBD position motor.
//!
//! Both targets participate in the same motor effort law. The cap bounds motor
//! effort; optional independent viscous effort is uncapped. Neither is a
//! position correction or an iteration-local multiplier. A solver
//! samples this law once before prediction and applies `effort * h` once per
//! accepted substep, independently of its position-iteration count.

/// A validated scalar proportional-derivative actuator for a hinge or slider.
///
/// Hinge coordinates are continuous radians, not principal angles; slider
/// coordinates are metres. Velocity targets are in the corresponding units
/// per second. Hard joint limits are independent constraints: this law never
/// clamps the target to the allowed coordinate interval.
///
/// This is an explicit force law, not an implicit/stable-PD integrator. A force
/// cap bounds actuation but does not guarantee stability for arbitrary gains
/// and timesteps. `INFINITY` explicitly opts out of the finite effort cap.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serialize",
    serde(try_from = "BoundedPdParameters", into = "BoundedPdParameters")
)]
pub struct BoundedPd {
    target_position: f64,
    target_velocity: f64,
    stiffness: f64,
    damping: f64,
    joint_damping: f64,
    max_effort: f64,
}

/// Invalid authored parameters or an unevaluable actuator sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum BoundedPdError {
    /// Position and velocity targets must both be finite.
    NonFiniteTarget,
    /// Feedback gains must be finite and nonnegative.
    InvalidGain,
    /// The effort limit must not be NaN; its absolute value is used.
    InvalidEffortLimit,
    /// Independent joint damping must be finite and nonnegative.
    InvalidJointDamping,
    /// The sampled coordinate and generalized velocity must be finite.
    NonFiniteState,
    /// Arithmetic did not produce a finite effort, even after capping.
    NonFiniteEffort,
}

/// Observable per-joint failure of the last enabled substep sample.
/// Failures do not silently substitute a different actuation model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum BoundedPdFailure {
    /// Only scalar revolute and prismatic coordinates are supported.
    UnsupportedFamily,
    /// A compliant XPBD motor cannot also own the same coordinate.
    ConflictingMotor,
    /// An endpoint is stale, missing, or aliases the other endpoint.
    InvalidEndpoints,
    /// A frame, axis, coordinate, or twist gradient is undefined.
    InvalidGeometry,
    /// Feedback sampling failed; no impulse was applied.
    Sample(BoundedPdError),
    /// Effort times dt, its wrenches, or resulting velocities cannot be
    /// represented as finite body-state values. Neither endpoint is changed.
    NonFiniteImpulse,
}

impl core::fmt::Display for BoundedPdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::NonFiniteTarget => "bounded-PD targets must be finite",
            Self::InvalidGain => "bounded-PD gains must be finite and nonnegative",
            Self::InvalidEffortLimit => "bounded-PD effort limit must not be NaN",
            Self::InvalidJointDamping => {
                "bounded-PD independent joint damping must be finite and nonnegative"
            }
            Self::NonFiniteState => "bounded-PD coordinate and speed must be finite",
            Self::NonFiniteEffort => "bounded-PD effort is not finite after capping",
        })
    }
}

impl core::error::Error for BoundedPdError {}

impl BoundedPd {
    /// Constructs a policy. Negative caps are canonicalized to their absolute
    /// value, matching reduced-coordinate motor semantics. Zero disables motor
    /// effort, not independent viscous damping added with `with_joint_damping`;
    /// infinite caps are allowed deliberately.
    pub fn new(
        target_position: f64,
        target_velocity: f64,
        stiffness: f64,
        damping: f64,
        max_effort: f64,
    ) -> Result<Self, BoundedPdError> {
        if !target_position.is_finite() || !target_velocity.is_finite() {
            return Err(BoundedPdError::NonFiniteTarget);
        }
        if !stiffness.is_finite() || stiffness < 0.0 || !damping.is_finite() || damping < 0.0 {
            return Err(BoundedPdError::InvalidGain);
        }
        if max_effort.is_nan() {
            return Err(BoundedPdError::InvalidEffortLimit);
        }
        Ok(Self {
            target_position,
            target_velocity,
            stiffness,
            damping,
            joint_damping: 0.0,
            max_effort: max_effort.abs(),
        })
    }

    /// Adds viscous damping owned by the generalized joint coordinate.
    ///
    /// This term is applied after the bounded motor effort and is therefore
    /// not limited by `max_effort`. The coefficient multiplies speed to produce
    /// physical effort (torque for a hinge, force for a slider), so the resulting
    /// acceleration depends on inverse inertia/mass. This is not equivalent to
    /// reduced articulation's mass-independent implicit generalized-velocity
    /// decay `qd_next = (qd + dt * qdd) / (1 + dt * damping)`.
    pub fn with_joint_damping(mut self, joint_damping: f64) -> Result<Self, BoundedPdError> {
        if !joint_damping.is_finite() || joint_damping < 0.0 {
            return Err(BoundedPdError::InvalidJointDamping);
        }
        self.joint_damping = joint_damping;
        Ok(self)
    }

    #[must_use]
    pub const fn target_position(self) -> f64 {
        self.target_position
    }

    #[must_use]
    pub const fn target_velocity(self) -> f64 {
        self.target_velocity
    }

    #[must_use]
    pub const fn stiffness(self) -> f64 {
        self.stiffness
    }

    #[must_use]
    pub const fn damping(self) -> f64 {
        self.damping
    }

    #[must_use]
    pub const fn joint_damping(self) -> f64 {
        self.joint_damping
    }

    #[must_use]
    pub const fn max_effort(self) -> f64 {
        self.max_effort
    }

    /// `clamp(k * (q_target - q) + c * (v_target - v), -cap, cap)`
    /// `- joint_damping * v`.
    ///
    /// Computing in f64 preserves the continuous hinge branch and avoids
    /// overflow in ordinary f32-authored gains before applying a finite cap.
    /// Invalid samples are errors, not silently converted to zero force.
    pub fn effort(self, coordinate: f64, speed: f64) -> Result<f64, BoundedPdError> {
        if !coordinate.is_finite() || !speed.is_finite() {
            return Err(BoundedPdError::NonFiniteState);
        }
        let motor_effort = if self.max_effort == 0.0 {
            0.0
        } else {
            // Zero gains must not turn an overflowing, unused error into
            // `0 * inf`.
            let position = if self.stiffness == 0.0 {
                0.0
            } else {
                self.stiffness * (self.target_position - coordinate)
            };
            let velocity = if self.damping == 0.0 {
                0.0
            } else {
                self.damping * (self.target_velocity - speed)
            };
            (position + velocity).clamp(-self.max_effort, self.max_effort)
        };
        let joint_damping_effort = if self.joint_damping == 0.0 {
            0.0
        } else {
            self.joint_damping * speed
        };
        let effort = motor_effort - joint_damping_effort;
        if effort.is_finite() {
            Ok(effort)
        } else {
            Err(BoundedPdError::NonFiniteEffort)
        }
    }
}

// The wire representation goes through the same validation as authored data.
// Private runtime fields alone do not protect invariants during deserialization.
#[cfg(feature = "serialize")]
#[derive(serde::Serialize, serde::Deserialize)]
struct BoundedPdParameters {
    target_position: f64,
    target_velocity: f64,
    stiffness: f64,
    damping: f64,
    #[serde(default)]
    joint_damping: f64,
    max_effort: f64,
}

#[cfg(feature = "serialize")]
impl TryFrom<BoundedPdParameters> for BoundedPd {
    type Error = BoundedPdError;

    fn try_from(value: BoundedPdParameters) -> Result<Self, Self::Error> {
        Self::new(
            value.target_position,
            value.target_velocity,
            value.stiffness,
            value.damping,
            value.max_effort,
        )
        .and_then(|policy| policy.with_joint_damping(value.joint_damping))
    }
}

#[cfg(feature = "serialize")]
impl From<BoundedPd> for BoundedPdParameters {
    fn from(value: BoundedPd) -> Self {
        Self {
            target_position: value.target_position,
            target_velocity: value.target_velocity,
            stiffness: value.stiffness,
            damping: value.damping,
            joint_damping: value.joint_damping,
            max_effort: value.max_effort,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_and_velocity_targets_contribute_simultaneously() {
        let policy = BoundedPd::new(2.0, -3.0, 7.0, 5.0, 100.0).unwrap();
        assert_eq!(policy.effort(1.0, 1.0), Ok(-13.0));
        assert_eq!(policy.effort(2.0, -3.0), Ok(0.0));
    }

    #[test]
    fn effort_not_each_feedback_term_is_capped() {
        let policy = BoundedPd::new(10.0, -9.0, 1.0, 1.0, 2.0).unwrap();
        assert_eq!(policy.effort(0.0, 0.0), Ok(1.0));
        assert_eq!(policy.effort(-10.0, 0.0), Ok(2.0));
        assert_eq!(policy.effort(20.0, 0.0), Ok(-2.0));
    }

    #[test]
    fn independent_joint_damping_is_not_limited_by_motor_cap() {
        let policy = BoundedPd::new(0.0, 0.0, 0.0, 0.0, 1.0)
            .unwrap()
            .with_joint_damping(3.0)
            .unwrap();
        assert_eq!(policy.effort(0.0, 2.0), Ok(-6.0));

        let pure_damping = BoundedPd::new(0.0, 0.0, 0.0, 0.0, 0.0)
            .unwrap()
            .with_joint_damping(3.0)
            .unwrap();
        assert_eq!(pure_damping.effort(0.0, 2.0), Ok(-6.0));
    }

    #[test]
    fn caps_are_absolute_and_zero_is_inert() {
        let positive = BoundedPd::new(10.0, 0.0, 8.0, 0.0, 3.0).unwrap();
        let negative = BoundedPd::new(10.0, 0.0, 8.0, 0.0, -3.0).unwrap();
        assert_eq!(positive, negative);
        let zero = BoundedPd::new(f64::MAX, 0.0, f64::MAX, 0.0, -0.0).unwrap();
        assert_eq!(zero.max_effort().to_bits(), 0.0_f64.to_bits());
        assert_eq!(zero.effort(-f64::MAX, 0.0), Ok(0.0));
    }

    #[test]
    fn unbounded_cap_is_explicit_and_finite_caps_survive_large_efforts() {
        let unbounded = BoundedPd::new(3.0, 0.0, 2.0, 0.0, -f64::INFINITY).unwrap();
        assert_eq!(unbounded.effort(0.0, 0.0), Ok(6.0));
        let bounded = BoundedPd::new(f64::MAX, 0.0, 2.0, 0.0, 4.0).unwrap();
        assert_eq!(bounded.effort(-f64::MAX, 0.0), Ok(4.0));
        let overflow = BoundedPd::new(f64::MAX, 0.0, 2.0, 0.0, f64::INFINITY).unwrap();
        assert_eq!(
            overflow.effort(-f64::MAX, 0.0),
            Err(BoundedPdError::NonFiniteEffort)
        );
    }

    #[test]
    fn continuous_hinge_errors_are_never_wrapped_or_rounded_to_f32() {
        let q = core::f64::consts::TAU * 10_000_000.0;
        let policy = BoundedPd::new(q + 0.25, 0.0, 4.0, 0.0, 100.0).unwrap();
        assert_eq!(policy.effort(q, 0.0), Ok(1.0));
        let full_turn = BoundedPd::new(core::f64::consts::TAU, 0.0, 1.0, 0.0, 100.0).unwrap();
        assert_eq!(full_turn.effort(0.0, 0.0), Ok(core::f64::consts::TAU));
    }

    #[test]
    fn zero_gain_does_not_evaluate_unused_overflowing_error() {
        let policy = BoundedPd::new(f64::MAX, 2.0, 0.0, 3.0, 100.0).unwrap();
        assert_eq!(policy.effort(-f64::MAX, 1.0), Ok(3.0));
    }

    #[test]
    fn invalid_parameters_and_samples_are_errors() {
        assert_eq!(
            BoundedPd::new(f64::NAN, 0.0, 1.0, 1.0, 1.0),
            Err(BoundedPdError::NonFiniteTarget)
        );
        assert_eq!(
            BoundedPd::new(0.0, f64::INFINITY, 1.0, 1.0, 1.0),
            Err(BoundedPdError::NonFiniteTarget)
        );
        for gain in [-1.0, f64::INFINITY, f64::NAN] {
            assert_eq!(
                BoundedPd::new(0.0, 0.0, gain, 1.0, 1.0),
                Err(BoundedPdError::InvalidGain)
            );
            assert_eq!(
                BoundedPd::new(0.0, 0.0, 1.0, gain, 1.0),
                Err(BoundedPdError::InvalidGain)
            );
        }
        for damping in [-1.0, f64::INFINITY, f64::NAN] {
            assert_eq!(
                BoundedPd::new(0.0, 0.0, 1.0, 1.0, 1.0)
                    .unwrap()
                    .with_joint_damping(damping),
                Err(BoundedPdError::InvalidJointDamping)
            );
        }
        assert_eq!(
            BoundedPd::new(0.0, 0.0, 1.0, 1.0, f64::NAN),
            Err(BoundedPdError::InvalidEffortLimit)
        );
        let policy = BoundedPd::new(0.0, 0.0, 1.0, 1.0, 1.0).unwrap();
        assert_eq!(
            policy.effort(f64::NAN, 0.0),
            Err(BoundedPdError::NonFiniteState)
        );
        assert_eq!(
            policy.effort(0.0, f64::INFINITY),
            Err(BoundedPdError::NonFiniteState)
        );
    }

    #[cfg(feature = "serialize")]
    #[test]
    fn deserialize_validates_the_wire_representation() {
        use serde::de::value::{Error, MapDeserializer};
        use serde::Deserialize;

        let deserialize = |gain| {
            let fields = [
                ("target_position", 1.0),
                ("target_velocity", 2.0),
                ("stiffness", gain),
                ("damping", 3.0),
                ("joint_damping", 6.0),
                ("max_effort", -4.0),
            ];
            BoundedPd::deserialize(MapDeserializer::<_, Error>::new(fields.into_iter()))
        };
        assert_eq!(
            deserialize(5.0).unwrap(),
            BoundedPd::new(1.0, 2.0, 5.0, 3.0, 4.0)
                .unwrap()
                .with_joint_damping(6.0)
                .unwrap()
        );
        assert!(deserialize(-1.0).is_err());
        assert!(deserialize(f64::NAN).is_err());
    }
}
