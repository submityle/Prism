//! Configuration for the substep XPBD soft-body solver.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Substep
//! counts, solver-iteration counts, and linear velocity damping are standard,
//! publicly documented position-based-dynamics parameters.

use glam::Vec3;

use crate::math::scalar::Real;
use crate::soft::collision::{CcdParams, SelfCcdParams};

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
    /// Optional self-collision contact pass run once per substep after
    /// constraint projection. `None` (the default) disables it entirely, so a
    /// garment that never folds onto itself pays no cost. When set, the solver
    /// pushes interpenetrating particle pairs apart with the deterministic
    /// spatial-hash resolver in [`crate::soft::collision`].
    pub self_collision: Option<SelfCollisionParams>,
    /// Optional continuous self-collision (self-CCD) pass run once per substep
    /// after the discrete self-collision pass. `None` (the default) disables
    /// it. When set to an *enabled* [`SelfCcdParams`], the solver sweeps each
    /// particle's substep motion segment against every other and clamps
    /// particles to their first time of impact, closing the thin-sheet tunneling
    /// gap the discrete pass can miss under fast motion. A disabled params value
    /// is a no-op.
    pub self_ccd: Option<SelfCcdParams>,
    /// Optional continuous body collision (CCD) pass run once per substep after
    /// the discrete body-collision pass, sweeping each particle's substep motion
    /// against the per-frame [`SoftContacts`](crate::soft::solver::SoftContacts)
    /// body colliders so a fast particle cannot tunnel through a thin or quickly
    /// moving proxy. `None` (the default) disables it; a disabled
    /// [`CcdParams`] value is also a no-op. The body-contact friction is taken
    /// from the same `SoftContacts` bundle as the discrete pass.
    pub ccd: Option<CcdParams>,
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
            self_collision: None,
            self_ccd: None,
            ccd: None,
        }
    }
}

/// Parameters for the solver's optional per-substep self-collision pass.
///
/// The pass buckets particles into a uniform spatial hash of side
/// [`cell_size`](Self::cell_size) and pushes any pair closer than
/// [`thickness`](Self::thickness) apart, split by inverse mass. A positive
/// [`friction`](Self::friction) additionally rubs each separated pair's
/// tangential slide with position-level Coulomb friction so stacked layers grip
/// instead of shearing freely. See [`crate::soft::collision`] for the exact
/// math and determinism guarantees.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SelfCollisionParams {
    /// Uniform spatial-hash cell side length, in metres. Should be on the order
    /// of the particle spacing (or the `thickness`); a non-positive value makes
    /// the pass a no-op.
    pub cell_size: Real,
    /// Contact thickness: the minimum separation enforced between any two
    /// particles, in metres. A non-positive value makes the pass a no-op.
    pub thickness: Real,
    /// Coulomb friction coefficient, clamped to `0..=1` by the resolver; `0`
    /// gives a frictionless (pure normal) separation.
    pub friction: Real,
}

impl SelfCollisionParams {
    /// Creates frictionless self-collision parameters with the given spatial-
    /// hash `cell_size` and contact `thickness`.
    #[must_use]
    pub const fn new(cell_size: Real, thickness: Real) -> SelfCollisionParams {
        SelfCollisionParams {
            cell_size,
            thickness,
            friction: 0.0,
        }
    }

    /// Returns a copy of these parameters with the given Coulomb `friction`
    /// coefficient (clamped to `0..=1` by the resolver at use time).
    #[must_use]
    pub const fn with_friction(mut self, friction: Real) -> SelfCollisionParams {
        self.friction = friction;
        self
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
