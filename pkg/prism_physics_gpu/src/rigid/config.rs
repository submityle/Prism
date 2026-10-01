//! Integrator configuration and error type for the `GPU` 6-DOF rigid-body
//! integrator.
//!
//! [`IntegratorConfig`] carries the tunables that control how a frame's time
//! step is advanced: a uniform gravitational acceleration, the number of equal
//! substeps the frame is divided into, and independent linear and angular
//! velocity damping coefficients. Each substep integrates every body's linear
//! and angular motion once, so a larger substep count trades throughput for the
//! smaller time step a stiff or fast-tumbling body needs.
//!
//! Damping is applied as an explicit per-substep velocity scale
//! `(1 - damping * h).max(0)`, matching the convention of the `XPBD` solver in
//! this crate ([`XpbdConfig`](crate::XpbdConfig)); the clamp keeps an
//! over-large coefficient from flipping the velocity sign instead of merely
//! removing it.
//!
//! Provenance: standard substep rigid-body integration parameters (Baraff &
//! Witkin). No Unreal Engine source or derived code.

use glam::Vec3;

/// Tunables controlling how the rigid-body state is advanced each frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntegratorConfig {
    /// Uniform acceleration (metres per second squared) applied to every
    /// non-static body each substep, typically gravity.
    pub gravity: Vec3,
    /// Number of equal substeps the frame time step is divided into. Clamped to
    /// at least `1` when stepping.
    pub substeps: u32,
    /// Linear velocity damping coefficient (per second). Each substep scales
    /// linear velocity by `(1 - linear_damping * h).max(0)`.
    pub linear_damping: f32,
    /// Angular velocity damping coefficient (per second). Each substep scales
    /// angular velocity by `(1 - angular_damping * h).max(0)`.
    pub angular_damping: f32,
}

impl IntegratorConfig {
    /// Default gravity (Earth-like, downward along `-Y`).
    pub const DEFAULT_GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);
    /// Default substep count.
    pub const DEFAULT_SUBSTEPS: u32 = 4;
    /// Default linear velocity damping (per second).
    pub const DEFAULT_LINEAR_DAMPING: f32 = 0.0;
    /// Default angular velocity damping (per second).
    pub const DEFAULT_ANGULAR_DAMPING: f32 = 0.0;

    /// Creates an integrator configuration.
    #[must_use]
    pub fn new(
        gravity: Vec3,
        substeps: u32,
        linear_damping: f32,
        angular_damping: f32,
    ) -> IntegratorConfig {
        IntegratorConfig {
            gravity,
            substeps,
            linear_damping,
            angular_damping,
        }
    }

    /// Returns the effective substep count (at least `1`).
    #[must_use]
    pub fn effective_substeps(&self) -> u32 {
        self.substeps.max(1)
    }

    /// Validates the configuration, returning the first violated invariant.
    ///
    /// # Errors
    ///
    /// Returns [`RigidError::InvalidConfig`] when `gravity`, `linear_damping`,
    /// or `angular_damping` is not finite. A zero substep count is permitted
    /// and clamped to `1` at step time.
    pub fn validate(&self) -> Result<(), RigidError> {
        if !self.gravity.is_finite() {
            return Err(RigidError::InvalidConfig("gravity must be finite"));
        }
        if !self.linear_damping.is_finite() {
            return Err(RigidError::InvalidConfig("linear_damping must be finite"));
        }
        if !self.angular_damping.is_finite() {
            return Err(RigidError::InvalidConfig("angular_damping must be finite"));
        }
        Ok(())
    }
}

impl Default for IntegratorConfig {
    fn default() -> Self {
        IntegratorConfig {
            gravity: Self::DEFAULT_GRAVITY,
            substeps: Self::DEFAULT_SUBSTEPS,
            linear_damping: Self::DEFAULT_LINEAR_DAMPING,
            angular_damping: Self::DEFAULT_ANGULAR_DAMPING,
        }
    }
}

/// Tunables controlling the velocity-level rigid-body contact solver.
///
/// The solver runs `iterations` sequential-impulse sweeps per frame. Penetration
/// is removed with a Baumgarte position bias `baumgarte / dt * max(penetration -
/// slop, 0)`: `baumgarte` in `[0, 1]` sets how aggressively overlap is pushed out
/// (a fraction of the error removed per frame) and `slop` is the small allowed
/// penetration that is left uncorrected so resting contacts do not jitter.
/// `restitution_threshold` is the approach speed below which restitution is
/// suppressed, so bodies merely resting (small, gravity-driven closing speed) do
/// not bounce and gain energy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactSolverConfig {
    /// Number of sequential-impulse sweeps over the contact set per frame.
    /// Clamped to at least `1` when solving.
    pub iterations: u32,
    /// Baumgarte position-bias factor in `[0, 1]`: the fraction of the
    /// (slop-reduced) penetration converted into a separating bias velocity
    /// each frame.
    pub baumgarte: f32,
    /// Allowed penetration (metres) left uncorrected so resting stacks settle
    /// without jitter.
    pub slop: f32,
    /// Approach speed (metres per second) below which restitution is
    /// suppressed, preventing resting contacts from bouncing.
    pub restitution_threshold: f32,
}

impl ContactSolverConfig {
    /// Default sweep count.
    pub const DEFAULT_ITERATIONS: u32 = 8;
    /// Default Baumgarte position-bias factor.
    pub const DEFAULT_BAUMGARTE: f32 = 0.2;
    /// Default penetration slop (metres).
    pub const DEFAULT_SLOP: f32 = 0.005;
    /// Default restitution suppression threshold (metres per second).
    pub const DEFAULT_RESTITUTION_THRESHOLD: f32 = 0.5;

    /// Creates a contact-solver configuration.
    #[must_use]
    pub fn new(
        iterations: u32,
        baumgarte: f32,
        slop: f32,
        restitution_threshold: f32,
    ) -> ContactSolverConfig {
        ContactSolverConfig {
            iterations,
            baumgarte,
            slop,
            restitution_threshold,
        }
    }

    /// Returns the effective sweep count (at least `1`).
    #[must_use]
    pub fn effective_iterations(&self) -> u32 {
        self.iterations.max(1)
    }

    /// Validates the configuration, returning the first violated invariant.
    ///
    /// # Errors
    ///
    /// Returns [`RigidError::InvalidConfig`] when `baumgarte` is outside
    /// `[0, 1]`, when `slop` or `restitution_threshold` is negative, or when any
    /// field is not finite.
    pub fn validate(&self) -> Result<(), RigidError> {
        if !self.baumgarte.is_finite() || !(0.0..=1.0).contains(&self.baumgarte) {
            return Err(RigidError::InvalidConfig("baumgarte must lie in [0, 1]"));
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

impl Default for ContactSolverConfig {
    fn default() -> Self {
        ContactSolverConfig {
            iterations: Self::DEFAULT_ITERATIONS,
            baumgarte: Self::DEFAULT_BAUMGARTE,
            slop: Self::DEFAULT_SLOP,
            restitution_threshold: Self::DEFAULT_RESTITUTION_THRESHOLD,
        }
    }
}

/// Errors the `GPU` rigid-body integrator can report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RigidError {
    /// A configuration invariant was violated; carries a static reason.
    InvalidConfig(&'static str),
    /// The per-body state arrays were not all the same length, or a supplied
    /// force or torque slice did not match the body count.
    InconsistentState {
        /// A human-readable description of which array disagreed.
        reason: &'static str,
    },
    /// The contact graph needed more parallel batches (colours) than the
    /// solver's fixed ceiling supports.
    TooManyContactBatches {
        /// The batch index that overflowed the ceiling.
        batches: u32,
        /// The maximum number of batches supported.
        maximum: u32,
    },
    /// The joint graph needed more parallel batches (colours) than the joint
    /// solver's fixed ceiling supports.
    TooManyJointBatches {
        /// The batch index that overflowed the ceiling.
        batches: u32,
        /// The maximum number of batches supported.
        maximum: u32,
    },
}

impl core::fmt::Display for RigidError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RigidError::InvalidConfig(reason) => {
                write!(f, "invalid integrator config: {reason}")
            }
            RigidError::InconsistentState { reason } => {
                write!(f, "inconsistent rigid-body state: {reason}")
            }
            RigidError::TooManyContactBatches { batches, maximum } => write!(
                f,
                "contact graph needs {batches} batches, exceeding the maximum of {maximum}"
            ),
            RigidError::TooManyJointBatches { batches, maximum } => write!(
                f,
                "joint graph needs {batches} batches, exceeding the maximum of {maximum}"
            ),
        }
    }
}

impl core::error::Error for RigidError {}
