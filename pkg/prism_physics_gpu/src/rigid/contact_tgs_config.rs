//! Tunables for the soft-constraint Temporal Gauss-Seidel (`TGS`) rigid-body
//! contact solver.
//!
//! The `TGS`-soft solver substeps the frame and, within each substep, resolves
//! contacts as damped springs rather than with a raw Baumgarte bias: the
//! penetration error is pulled out at a target frequency `contact_hertz` and
//! damping ratio `contact_damping_ratio`, and the bodies are integrated and the
//! contacts re-linearised between substeps. That removes the Baumgarte energy
//! leak of the velocity-only solver and lets a tall stack settle without jitter.
//!
//! The substep count, gravity, and damping come from the shared
//! [`IntegratorConfig`](super::IntegratorConfig); this type carries only the
//! contact-specific coefficients so the two configurations compose without
//! duplicating the integrator's fields.
//!
//! Provenance: the soft-constraint `TGS` contact parametrisation of Catto
//! ("Soft Constraints", GDC 2011) and the substepping solver loop it feeds
//! (`Box2D` TGS Soft). No Unreal Engine source or derived code.

use super::config::RigidError;

/// Contact-specific coefficients for the `TGS`-soft solver.
///
/// Build with [`TgsContactConfig::new`] or take [`TgsContactConfig::DEFAULT`].
/// All fields are validated by [`TgsContactConfig::validate`] before a solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TgsContactConfig {
    /// Target angular frequency of the contact spring, in hertz. Higher is
    /// stiffer (less penetration, more like the rigid limit); a non-positive
    /// value collapses to the rigid Baumgarte limit at the substep size.
    pub contact_hertz: f32,
    /// Dimensionless damping ratio of the contact spring. `1` is critical
    /// damping; contacts are usually over-damped (`>= 2`) so recovery does not
    /// overshoot into a bounce.
    pub contact_damping_ratio: f32,
    /// Penetration (metres) left uncorrected so resting contacts do not
    /// jitter around exact touching.
    pub slop: f32,
    /// Approach speed below which restitution is suppressed, so resting
    /// contacts do not acquire a spurious bounce.
    pub restitution_threshold: f32,
    /// Number of bias-free relaxation sweeps run after the biased solve and
    /// position integration of each substep. At least one removes the bias
    /// velocity the biased pass injected.
    pub relax_iterations: u32,
}

impl TgsContactConfig {
    /// Default contact spring frequency (hertz).
    pub const DEFAULT_CONTACT_HERTZ: f32 = 30.0;
    /// Default contact spring damping ratio (over-damped to avoid bounce).
    pub const DEFAULT_CONTACT_DAMPING_RATIO: f32 = 10.0;
    /// Default penetration slop (metres).
    pub const DEFAULT_SLOP: f32 = 0.005;
    /// Default restitution threshold (metres per second).
    pub const DEFAULT_RESTITUTION_THRESHOLD: f32 = 0.5;
    /// Default relaxation sweep count.
    pub const DEFAULT_RELAX_ITERATIONS: u32 = 1;

    /// The default configuration: a `30 Hz`, heavily over-damped contact
    /// spring with a `5 mm` slop and one relaxation sweep.
    pub const DEFAULT: TgsContactConfig = TgsContactConfig {
        contact_hertz: Self::DEFAULT_CONTACT_HERTZ,
        contact_damping_ratio: Self::DEFAULT_CONTACT_DAMPING_RATIO,
        slop: Self::DEFAULT_SLOP,
        restitution_threshold: Self::DEFAULT_RESTITUTION_THRESHOLD,
        relax_iterations: Self::DEFAULT_RELAX_ITERATIONS,
    };

    /// Creates a `TGS`-soft contact configuration.
    #[must_use]
    pub fn new(
        contact_hertz: f32,
        contact_damping_ratio: f32,
        slop: f32,
        restitution_threshold: f32,
        relax_iterations: u32,
    ) -> TgsContactConfig {
        TgsContactConfig {
            contact_hertz,
            contact_damping_ratio,
            slop,
            restitution_threshold,
            relax_iterations,
        }
    }

    /// Returns the effective relaxation sweep count (at least `1`).
    #[must_use]
    pub fn effective_relax_iterations(&self) -> u32 {
        self.relax_iterations.max(1)
    }

    /// Validates the configuration, returning the first violated invariant.
    ///
    /// # Errors
    ///
    /// Returns [`RigidError::InvalidConfig`] when any field is not finite, when
    /// `contact_damping_ratio`, `slop`, or `restitution_threshold` is negative.
    /// A non-positive `contact_hertz` is allowed and means "as stiff as the
    /// substep permits" (the rigid limit).
    pub fn validate(&self) -> Result<(), RigidError> {
        if !self.contact_hertz.is_finite() {
            return Err(RigidError::InvalidConfig("contact_hertz must be finite"));
        }
        if !self.contact_damping_ratio.is_finite() || self.contact_damping_ratio < 0.0 {
            return Err(RigidError::InvalidConfig(
                "contact_damping_ratio must be finite and non-negative",
            ));
        }
        if !self.slop.is_finite() || self.slop < 0.0 {
            return Err(RigidError::InvalidConfig(
                "slop must be finite and non-negative",
            ));
        }
        if !self.restitution_threshold.is_finite() || self.restitution_threshold < 0.0 {
            return Err(RigidError::InvalidConfig(
                "restitution_threshold must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

impl Default for TgsContactConfig {
    fn default() -> TgsContactConfig {
        TgsContactConfig::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_valid() {
        assert!(TgsContactConfig::DEFAULT.validate().is_ok());
        assert_eq!(TgsContactConfig::default(), TgsContactConfig::DEFAULT);
    }

    #[test]
    fn relax_iterations_floor_is_one() {
        let c = TgsContactConfig::new(30.0, 10.0, 0.005, 0.5, 0);
        assert_eq!(c.effective_relax_iterations(), 1);
    }

    #[test]
    fn non_positive_hertz_is_allowed() {
        let c = TgsContactConfig::new(0.0, 10.0, 0.005, 0.5, 1);
        assert!(c.validate().is_ok());
        let c = TgsContactConfig::new(-5.0, 10.0, 0.005, 0.5, 1);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn negative_damping_is_rejected() {
        let c = TgsContactConfig::new(30.0, -1.0, 0.005, 0.5, 1);
        assert!(c.validate().is_err());
    }

    #[test]
    fn non_finite_fields_are_rejected() {
        let c = TgsContactConfig::new(f32::NAN, 10.0, 0.005, 0.5, 1);
        assert!(c.validate().is_err());
        let c = TgsContactConfig::new(30.0, 10.0, f32::INFINITY, 0.5, 1);
        assert!(c.validate().is_err());
    }
}
