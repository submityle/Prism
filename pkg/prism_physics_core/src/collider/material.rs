//! Surface contact material parameters.
//!
//! A [`PhysicsMaterial`] carries the scalar surface properties the contact
//! solver consumes: a Coulomb friction coefficient and a restitution
//! (bounciness) coefficient. Materials are plain value types so that they can
//! be stored per body in the Structure-of-Arrays storage without indirection.
//!
//! These are standard, publicly documented contact-response parameters and are
//! not derived from Unreal Engine source.

/// Surface response parameters used by the contact solver.
///
/// - `friction` is a dimensionless Coulomb friction coefficient shared by the
///   static and dynamic friction models. `0.0` is frictionless; typical solids
///   sit around `0.4..=1.0`.
/// - `restitution` is the coefficient of restitution in `0.0..=1.0`, where
///   `0.0` is a fully inelastic (non-bouncing) contact and `1.0` conserves the
///   normal closing speed (a perfect bounce).
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PhysicsMaterial {
    /// Coulomb friction coefficient (shared static/dynamic), clamped to be
    /// non-negative.
    pub friction: f32,
    /// Coefficient of restitution in `0.0..=1.0`.
    pub restitution: f32,
}

impl PhysicsMaterial {
    /// A sensible general-purpose default: moderate friction, no bounce.
    pub const DEFAULT: PhysicsMaterial = PhysicsMaterial {
        friction: 0.5,
        restitution: 0.0,
    };

    /// A perfectly frictionless, non-bouncing material.
    pub const FRICTIONLESS: PhysicsMaterial = PhysicsMaterial {
        friction: 0.0,
        restitution: 0.0,
    };

    /// Creates a material from explicit coefficients.
    ///
    /// `friction` is clamped to be non-negative and `restitution` is clamped to
    /// `0.0..=1.0`, so callers cannot construct an energy-gaining contact.
    #[must_use]
    pub fn new(friction: f32, restitution: f32) -> PhysicsMaterial {
        PhysicsMaterial {
            friction: friction.max(0.0),
            restitution: restitution.clamp(0.0, 1.0),
        }
    }

    /// Combines two contacting materials into the effective pair parameters.
    ///
    /// Friction is combined with the geometric mean (a common, stable choice
    /// that avoids one very sticky surface dominating), and restitution is
    /// combined with the maximum (so a bouncy surface keeps its bounce). Both
    /// combination rules are standard engine conventions.
    #[must_use]
    pub fn combine(self, other: PhysicsMaterial) -> PhysicsMaterial {
        PhysicsMaterial {
            friction: (self.friction * other.friction).max(0.0).sqrt(),
            restitution: self.restitution.max(other.restitution),
        }
    }
}

impl Default for PhysicsMaterial {
    fn default() -> Self {
        PhysicsMaterial::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_clamps_out_of_range_inputs() {
        let m = PhysicsMaterial::new(-1.0, 5.0);
        assert_eq!(m.friction, 0.0);
        assert_eq!(m.restitution, 1.0);
    }

    #[test]
    fn combine_uses_geometric_mean_friction_and_max_restitution() {
        let a = PhysicsMaterial::new(0.16, 0.2);
        let b = PhysicsMaterial::new(0.64, 0.8);
        let c = a.combine(b);
        // sqrt(0.16 * 0.64) = sqrt(0.1024) = 0.32.
        assert!((c.friction - 0.32).abs() < 1e-6);
        assert!((c.restitution - 0.8).abs() < 1e-6);
    }

    #[test]
    fn default_is_moderate_friction_no_bounce() {
        let m = PhysicsMaterial::default();
        assert!((m.friction - 0.5).abs() < 1e-6);
        assert_eq!(m.restitution, 0.0);
    }
}
