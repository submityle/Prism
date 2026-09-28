//! Configuration for reduced-order (modal subspace) soft-body dynamics.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Modal
//! truncation counts and Rayleigh damping are standard, publicly documented
//! structural-dynamics parameters.

use glam::Vec3;

use crate::math::scalar::Real;

/// Tunables controlling how a reduced-order body is built and advanced.
///
/// A reduced model keeps only the [`num_modes`](Self::num_modes) lowest-frequency
/// vibration modes of a soft body and integrates one scalar generalised
/// coordinate per mode. Because the modes are decoupled and integrated
/// implicitly, the cost per step is `O(num_modes)` regardless of how many
/// vertices the original mesh has, which is what makes large deformable props
/// cheap to simulate at a distance.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ReducedConfig {
    /// Uniform acceleration (metres per second squared) applied to every vertex,
    /// typically gravity.
    pub gravity: Vec3,
    /// Number of low-frequency modes to retain. Clamped to the number of
    /// available modes when the model is built.
    pub num_modes: usize,
    /// Mass-proportional Rayleigh damping coefficient (per second).
    pub rayleigh_alpha: Real,
    /// Stiffness-proportional Rayleigh damping coefficient (seconds).
    pub rayleigh_beta: Real,
    /// Number of equal substeps the frame time step is divided into. Clamped to
    /// at least `1` when stepping.
    pub substeps: u32,
}

impl ReducedConfig {
    /// Default gravity (Earth-like, downward along `-Y`).
    pub const DEFAULT_GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);
    /// Default retained-mode count.
    pub const DEFAULT_NUM_MODES: usize = 12;
    /// Default mass-proportional damping.
    pub const DEFAULT_RAYLEIGH_ALPHA: Real = 0.5;
    /// Default stiffness-proportional damping.
    pub const DEFAULT_RAYLEIGH_BETA: Real = 0.01;
    /// Default substep count.
    pub const DEFAULT_SUBSTEPS: u32 = 1;
}

impl Default for ReducedConfig {
    fn default() -> Self {
        ReducedConfig {
            gravity: Self::DEFAULT_GRAVITY,
            num_modes: Self::DEFAULT_NUM_MODES,
            rayleigh_alpha: Self::DEFAULT_RAYLEIGH_ALPHA,
            rayleigh_beta: Self::DEFAULT_RAYLEIGH_BETA,
            substeps: Self::DEFAULT_SUBSTEPS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_spec() {
        let c = ReducedConfig::default();
        assert_eq!(c.gravity, Vec3::new(0.0, -9.81, 0.0));
        assert_eq!(c.num_modes, 12);
        assert_eq!(c.substeps, 1);
        assert!((c.rayleigh_alpha - 0.5).abs() < 1e-6);
        assert!((c.rayleigh_beta - 0.01).abs() < 1e-6);
    }
}
