//! Stable angular actuation and passive-response value types.
//!
//! The types in this module deliberately do not own a world or a solver. They
//! describe bounded, deterministic response laws which a solver can evaluate
//! from a relative pose and relative angular velocity. Keeping the response
//! law separate from storage makes it possible to use the same M13 contract
//! from a CPU solver, a replay driver, or a backend adapter.

use glam::{Quat, Vec3};

/// A finite, non-negative compliant angular limit response.
///
/// `compliance` is used by the position projection (`0` is a hard stop), while
/// `stiffness` and `damping` are used by the velocity/force response. The
/// optional torque cap prevents a soft limit from injecting unbounded energy.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SoftLimit {
    /// XPBD-style angular compliance. Zero means a hard position stop.
    pub compliance: f32,
    /// Spring stiffness in torque per radian.
    pub stiffness: f32,
    /// Viscous damping in torque per radian per second.
    pub damping: f32,
    /// Optional absolute torque cap.
    pub max_torque: Option<f32>,
}

impl SoftLimit {
    /// Creates a validated soft limit response.
    #[must_use]
    pub fn new(
        compliance: f32,
        stiffness: f32,
        damping: f32,
        max_torque: Option<f32>,
    ) -> Option<Self> {
        if compliance.is_finite()
            && compliance >= 0.0
            && stiffness.is_finite()
            && stiffness >= 0.0
            && damping.is_finite()
            && damping >= 0.0
            && max_torque.is_none_or(|value| value.is_finite() && value >= 0.0)
        {
            Some(Self {
                compliance,
                stiffness,
                damping,
                max_torque,
            })
        } else {
            None
        }
    }

    /// A hard positional stop with no force response of its own.
    #[must_use]
    pub const fn hard() -> Self {
        Self {
            compliance: 0.0,
            stiffness: 0.0,
            damping: 0.0,
            max_torque: None,
        }
    }

    /// A critically damped response for a given natural frequency and
    /// effective inertia.
    #[must_use]
    pub fn critically_damped(natural_frequency: f32, effective_inertia: f32) -> Option<Self> {
        if natural_frequency.is_finite()
            && natural_frequency > 0.0
            && effective_inertia.is_finite()
            && effective_inertia > 0.0
        {
            let stiffness = effective_inertia * natural_frequency * natural_frequency;
            let damping = 2.0 * effective_inertia * natural_frequency;
            Self::new(0.0, stiffness, damping, None)
        } else {
            None
        }
    }

    /// Returns the positional correction fraction for a timestep.
    #[must_use]
    pub fn correction_fraction(&self, dt: f32) -> f32 {
        if !dt.is_finite() || dt <= 0.0 {
            return 0.0;
        }
        let dt2 = dt * dt;
        (dt2 / (dt2 + self.compliance)).clamp(0.0, 1.0)
    }

    /// Evaluates a signed restoring torque for an angular error and speed.
    #[must_use]
    pub fn torque(&self, error: f32, speed: f32) -> f32 {
        let mut value = self.stiffness * error - self.damping * speed;
        if let Some(limit) = self.max_torque {
            value = value.clamp(-limit, limit);
        }
        if value.is_finite() {
            value
        } else {
            0.0
        }
    }

    /// Evaluates the signed torque that restores an out-of-range coordinate.
    ///
    /// The error points from the allowed interval toward the current
    /// coordinate, so the restoring response has the opposite sign. This is
    /// the form consumed by the XPBD velocity pass.
    #[must_use]
    pub fn restoring_torque(&self, error: f32, speed: f32) -> f32 {
        if !error.is_finite() || !speed.is_finite() {
            return 0.0;
        }
        let mut value = -(self.stiffness * error + self.damping * speed);
        if let Some(limit) = self.max_torque {
            value = value.clamp(-limit, limit);
        }
        if value.is_finite() {
            value
        } else {
            0.0
        }
    }
}

impl Default for SoftLimit {
    fn default() -> Self {
        Self::hard()
    }
}

/// A bounded viscous-plus-Coulomb angular friction law.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AngularFriction {
    /// Viscous coefficient in torque per radian per second.
    pub viscous: f32,
    /// Coulomb torque at sustained slip.
    pub coulomb: f32,
    /// Speeds below this value use a smooth static-friction ramp.
    pub static_speed: f32,
}

impl AngularFriction {
    /// Creates a validated friction law.
    #[must_use]
    pub fn new(viscous: f32, coulomb: f32, static_speed: f32) -> Option<Self> {
        if viscous.is_finite()
            && viscous >= 0.0
            && coulomb.is_finite()
            && coulomb >= 0.0
            && static_speed.is_finite()
            && static_speed > 0.0
        {
            Some(Self {
                viscous,
                coulomb,
                static_speed,
            })
        } else {
            None
        }
    }

    /// Returns the torque opposing the supplied relative angular speed.
    #[must_use]
    pub fn torque(&self, speed: f32) -> f32 {
        if !speed.is_finite() {
            return 0.0;
        }
        let magnitude = self.viscous * speed.abs()
            + self.coulomb * (speed.abs() / self.static_speed).clamp(0.0, 1.0);
        -speed.signum() * magnitude
    }
}

impl Default for AngularFriction {
    fn default() -> Self {
        Self {
            viscous: 0.0,
            coulomb: 0.0,
            static_speed: 1.0e-3,
        }
    }
}

/// A pose servo in the joint's reference frame.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PoseDrive {
    /// Desired relative orientation from frame `a` to frame `b`.
    pub target: Quat,
    /// Proportional torque gain.
    pub stiffness: f32,
    /// Relative angular velocity gain.
    pub damping: f32,
    /// Maximum active torque. This cap is mandatory for bounded actuation.
    pub max_torque: f32,
}

impl PoseDrive {
    /// Creates a validated pose drive.
    #[must_use]
    pub fn new(target: Quat, stiffness: f32, damping: f32, max_torque: f32) -> Option<Self> {
        if target.is_finite()
            && target.length_squared() > 1.0e-12
            && stiffness.is_finite()
            && stiffness >= 0.0
            && damping.is_finite()
            && damping >= 0.0
            && max_torque.is_finite()
            && max_torque >= 0.0
        {
            Some(Self {
                target: target.normalize(),
                stiffness,
                damping,
                max_torque,
            })
        } else {
            None
        }
    }

    /// Returns the torque magnitude after applying the active cap.
    #[must_use]
    pub fn clamp_torque(&self, torque: f32) -> f32 {
        torque.clamp(-self.max_torque, self.max_torque)
    }
}

/// Relative weighting of active drive and passive limit/friction response.
///
/// The constructor normalises the two weights. Consequently the blend cannot
/// amplify a response merely because both contributors were authored with
/// arbitrary, non-normalised weights. The individual contributors remain
/// independently capped before blending.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ActuationBlend {
    /// Normalised active contribution in `[0, 1]`.
    pub active: f32,
    /// Normalised passive contribution in `[0, 1]`.
    pub passive: f32,
}

impl ActuationBlend {
    /// Fully passive response.
    pub const PASSIVE: Self = Self {
        active: 0.0,
        passive: 1.0,
    };

    /// Fully active response.
    pub const ACTIVE: Self = Self {
        active: 1.0,
        passive: 0.0,
    };

    /// Creates and normalises a blend. Both weights cannot be zero.
    #[must_use]
    pub fn new(active: f32, passive: f32) -> Option<Self> {
        if !active.is_finite() || !passive.is_finite() || active < 0.0 || passive < 0.0 {
            return None;
        }
        let total = active + passive;
        if total <= f32::EPSILON {
            None
        } else {
            Some(Self {
                active: active / total,
                passive: passive / total,
            })
        }
    }

    /// Blends independently bounded active and passive torques.
    #[must_use]
    pub fn mix(&self, active: f32, passive: f32) -> f32 {
        self.active * active + self.passive * passive
    }

    /// Blends two independently bounded vector responses component-wise with
    /// the same normalised active/passive weights.
    #[must_use]
    pub fn mix_vec3(&self, active: Vec3, passive: Vec3) -> Vec3 {
        active * self.active + passive * self.passive
    }
}

impl Default for ActuationBlend {
    fn default() -> Self {
        Self::PASSIVE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compliant_limit_is_time_step_independent_in_the_expected_direction() {
        let limit = SoftLimit::new(0.01, 100.0, 0.0, Some(20.0)).unwrap();
        assert!(limit.correction_fraction(1.0 / 60.0) < limit.correction_fraction(1.0 / 30.0));
        assert_eq!(limit.torque(1.0, 0.0), 20.0);
    }

    #[test]
    fn friction_is_odd_and_bounded_at_low_speed() {
        let friction = AngularFriction::new(2.0, 3.0, 0.5).unwrap();
        let positive = friction.torque(0.25);
        let negative = friction.torque(-0.25);
        assert!(positive < 0.0);
        assert!((positive + negative).abs() < 1.0e-6);
        assert!(positive.abs() <= 3.5 + 1.0e-6);
    }

    #[test]
    fn blend_normalises_weights() {
        let blend = ActuationBlend::new(2.0, 1.0).unwrap();
        assert!((blend.active - 2.0 / 3.0).abs() < 1.0e-6);
        assert!((blend.mix(3.0, -3.0) - 1.0).abs() < 1.0e-6);
        assert_eq!(
            blend.mix_vec3(Vec3::splat(3.0), Vec3::splat(-3.0)),
            Vec3::splat(1.0)
        );
    }

    #[test]
    fn restoring_torque_opposes_limit_error_and_speed() {
        let limit = SoftLimit::new(0.0, 10.0, 2.0, None).unwrap();
        assert!(limit.restoring_torque(0.5, 0.25) < 0.0);
        assert!(limit.restoring_torque(-0.5, -0.25) > 0.0);
    }
}
