//! Configuration for the position-based (`XPBD`) rigid-body joint stepper.
//!
//! [`JointSolverConfig`] carries the single tunable the joint stepper owns on
//! top of the shared [`IntegratorConfig`](super::super::config::IntegratorConfig):
//! the number of constraint-projection sweeps run per substep. Everything else
//! that governs how a frame is advanced — gravity, the substep count, and the
//! linear and angular velocity damping — is read from the integrator config, so
//! the joint stepper integrates the bodies on exactly the same schedule as the
//! rest of the crate.
//!
//! # Why a separate iteration count
//!
//! The substep `XPBD` scheme converges by taking many small time steps rather
//! than many solver iterations: a single projection sweep per substep is the
//! canonical choice (Müller et al., "Detailed Rigid Body Simulation with
//! XPBD"). The iteration count is kept configurable so a stiff articulated body
//! can trade a few extra sweeps for fewer substeps, but it clamps to at least
//! `1` so a zero never silently skips the solve.
//!
//! Provenance: substep `XPBD` with per-substep constraint projection (Müller et
//! al.). No Unreal Engine source or derived code.

use super::super::config::RigidError;

/// Tunables controlling the position-based rigid-body joint solver.
///
/// The joint stepper is a *full stepper*: within each of the integrator's
/// substeps it predicts the bodies forward under gravity, projects the joint
/// constraints [`position_iterations`](Self::position_iterations) times in
/// colour order, then recovers the velocities from the net motion. The gravity,
/// substep count, and damping are taken from the
/// [`IntegratorConfig`](super::super::config::IntegratorConfig); this struct adds
/// only the per-substep projection-sweep count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JointSolverConfig {
    /// Number of constraint-projection sweeps run per substep. Clamped to at
    /// least `1` when solving.
    pub position_iterations: u32,
}

impl JointSolverConfig {
    /// Default per-substep projection-sweep count. One sweep per substep is the
    /// canonical substep-`XPBD` choice.
    pub const DEFAULT_POSITION_ITERATIONS: u32 = 1;

    /// Creates a joint-solver configuration.
    #[must_use]
    pub fn new(position_iterations: u32) -> JointSolverConfig {
        JointSolverConfig {
            position_iterations,
        }
    }

    /// Returns the effective projection-sweep count (at least `1`).
    #[must_use]
    pub fn effective_position_iterations(&self) -> u32 {
        self.position_iterations.max(1)
    }

    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Currently infallible — a zero `position_iterations` is permitted and
    /// clamped to `1` at solve time — but returns a [`Result`] so callers can
    /// treat it uniformly with the other configs and so future invariants can be
    /// added without changing the signature.
    pub fn validate(&self) -> Result<(), RigidError> {
        Ok(())
    }
}

impl Default for JointSolverConfig {
    fn default() -> Self {
        JointSolverConfig {
            position_iterations: Self::DEFAULT_POSITION_ITERATIONS,
        }
    }
}
