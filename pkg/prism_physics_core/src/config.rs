//! World-level simulation configuration.

use glam::Vec3;

/// Global configuration for a [`PhysicsWorld`](crate::world::PhysicsWorld).
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct WorldConfig {
    /// Uniform gravitational acceleration applied to dynamic bodies.
    pub gravity: Vec3,
    /// Default number of solver sub-steps per full step. Clamped by callers to
    /// a sensible range (1..=8); the default is `4`.
    pub default_substeps: u32,
}

impl WorldConfig {
    /// The default number of solver sub-steps.
    pub const DEFAULT_SUBSTEPS: u32 = 4;

    /// Creates a configuration with the given gravity and the default sub-step
    /// count.
    #[must_use]
    pub fn with_gravity(gravity: Vec3) -> WorldConfig {
        WorldConfig {
            gravity,
            default_substeps: Self::DEFAULT_SUBSTEPS,
        }
    }
}

impl Default for WorldConfig {
    fn default() -> Self {
        WorldConfig {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            default_substeps: Self::DEFAULT_SUBSTEPS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_spec() {
        let c = WorldConfig::default();
        assert_eq!(c.gravity, Vec3::new(0.0, -9.81, 0.0));
        assert_eq!(c.default_substeps, 4);
    }

    #[test]
    fn with_gravity_keeps_default_substeps() {
        let c = WorldConfig::with_gravity(Vec3::ZERO);
        assert_eq!(c.gravity, Vec3::ZERO);
        assert_eq!(c.default_substeps, WorldConfig::DEFAULT_SUBSTEPS);
    }
}
