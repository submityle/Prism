//! Sleeping (deactivation) configuration for the rigid-body solver.
//!
//! A dynamic body whose linear and angular speed both stay below the
//! configured thresholds for a continuous [`SleepConfig::time_to_sleep`] window
//! is allowed to *sleep*: the solver then skips its prediction and constraint
//! work until something wakes it again. Sleeping is the primary culling
//! mechanism that keeps large-scale (ten-thousand body) scenes real-time once
//! most bodies have settled.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! velocity-threshold plus dwell-time deactivation policy is a standard,
//! publicly documented rigid-body sleeping technique.

use glam::Vec3;

/// Thresholds and timing that govern when a dynamic body may sleep.
///
/// A body accumulates idle time while both its linear and angular speeds are
/// below the respective thresholds; once that accumulated time reaches
/// [`time_to_sleep`](SleepConfig::time_to_sleep) the body is eligible to sleep.
/// Any motion above threshold resets the accumulator to zero.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SleepConfig {
    /// Whether sleeping is enabled at all. When `false`, bodies never sleep and
    /// [`ready_to_sleep`](crate::sleep::ready_to_sleep) always returns `false`.
    pub enabled: bool,
    /// Linear speed (metres per second) at or below which a body counts as
    /// idle for sleeping purposes.
    pub linear_velocity_threshold: f32,
    /// Angular speed (radians per second) at or below which a body counts as
    /// idle for sleeping purposes.
    pub angular_velocity_threshold: f32,
    /// Continuous idle time (seconds) a body must accumulate before it becomes
    /// eligible to sleep.
    pub time_to_sleep: f32,
}

impl SleepConfig {
    /// Default linear speed threshold in metres per second.
    pub const DEFAULT_LINEAR_VELOCITY_THRESHOLD: f32 = 0.05;
    /// Default angular speed threshold in radians per second.
    pub const DEFAULT_ANGULAR_VELOCITY_THRESHOLD: f32 = 0.05;
    /// Default idle dwell time in seconds before a body may sleep.
    pub const DEFAULT_TIME_TO_SLEEP: f32 = 0.5;

    /// Returns `true` when both the linear and angular velocities are at or
    /// below their respective idle thresholds.
    ///
    /// The comparison is done on squared magnitudes to avoid a square root.
    #[must_use]
    pub fn is_below_thresholds(&self, linear_velocity: Vec3, angular_velocity: Vec3) -> bool {
        let lin_limit = self.linear_velocity_threshold * self.linear_velocity_threshold;
        let ang_limit = self.angular_velocity_threshold * self.angular_velocity_threshold;
        linear_velocity.length_squared() <= lin_limit
            && angular_velocity.length_squared() <= ang_limit
    }
}

impl Default for SleepConfig {
    fn default() -> Self {
        SleepConfig {
            enabled: true,
            linear_velocity_threshold: Self::DEFAULT_LINEAR_VELOCITY_THRESHOLD,
            angular_velocity_threshold: Self::DEFAULT_ANGULAR_VELOCITY_THRESHOLD,
            time_to_sleep: Self::DEFAULT_TIME_TO_SLEEP,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_spec() {
        let c = SleepConfig::default();
        assert!(c.enabled);
        assert_eq!(c.linear_velocity_threshold, 0.05);
        assert_eq!(c.angular_velocity_threshold, 0.05);
        assert_eq!(c.time_to_sleep, 0.5);
    }

    #[test]
    fn below_thresholds_detects_idle() {
        let c = SleepConfig::default();
        assert!(c.is_below_thresholds(Vec3::ZERO, Vec3::ZERO));
        assert!(c.is_below_thresholds(Vec3::new(0.01, 0.0, 0.0), Vec3::new(0.0, 0.01, 0.0)));
    }

    #[test]
    fn above_linear_threshold_is_not_idle() {
        let c = SleepConfig::default();
        assert!(!c.is_below_thresholds(Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO));
    }

    #[test]
    fn above_angular_threshold_is_not_idle() {
        let c = SleepConfig::default();
        assert!(!c.is_below_thresholds(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0)));
    }
}
