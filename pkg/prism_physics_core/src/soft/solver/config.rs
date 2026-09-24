//! Configuration for the substep XPBD soft-body solver.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Substep
//! counts, solver-iteration counts, and linear velocity damping are standard,
//! publicly documented position-based-dynamics parameters.

use glam::Vec3;

use crate::math::scalar::Real;

/// Tunables controlling how a soft body is advanced each frame.
///
/// The frame time step `dt` is divided into [`substeps`](Self::substeps) equal
/// substeps; each substep predicts positions under gravity, then runs
/// [`iterations`](Self::iterations) Gauss-Seidel projection sweeps over the
/// constraints. More substeps stiffen the effective response (the recommended
/// XPBD knob), while more iterations improve convergence within a substep.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SoftSolverConfig {
    /// Uniform acceleration (metres per second squared) applied to every
    /// non-pinned particle, typically gravity.
    pub gravity: Vec3,
    /// Number of equal substeps the frame time step is divided into. Clamped to
    /// at least `1` when stepping.
    pub substeps: u32,
    /// Number of Gauss-Seidel constraint-projection sweeps per substep. Clamped
    /// to at least `1` when stepping.
    pub iterations: u32,
    /// Linear velocity damping coefficient (per second). Each substep scales
    /// velocity by `(1 - damping * h).max(0)`, bleeding off energy so cloth and
    /// rope settle instead of oscillating forever.
    pub damping: Real,
}

impl SoftSolverConfig {
    /// Default gravity (Earth-like, downward along `-Y`) in metres per second
    /// squared.
    pub const DEFAULT_GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);
    /// Default substep count.
    pub const DEFAULT_SUBSTEPS: u32 = 8;
    /// Default projection iterations per substep.
    pub const DEFAULT_ITERATIONS: u32 = 1;
    /// Default linear velocity damping (per second).
    pub const DEFAULT_DAMPING: Real = 0.5;
}

impl Default for SoftSolverConfig {
    fn default() -> Self {
        SoftSolverConfig {
            gravity: Self::DEFAULT_GRAVITY,
            substeps: Self::DEFAULT_SUBSTEPS,
            iterations: Self::DEFAULT_ITERATIONS,
            damping: Self::DEFAULT_DAMPING,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_spec() {
        let c = SoftSolverConfig::default();
        assert_eq!(c.gravity, Vec3::new(0.0, -9.81, 0.0));
        assert_eq!(c.substeps, 8);
        assert_eq!(c.iterations, 1);
        assert_eq!(c.damping, 0.5);
    }
}
