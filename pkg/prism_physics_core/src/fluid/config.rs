//! Configuration for the FLIP/APIC free-surface fluid solver.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! PIC/FLIP blend and pressure-projection controls follow Bridson, *Fluid
//! Simulation for Computer Graphics*, and Zhu & Bridson 2005.

use glam::Vec3;

use crate::math::scalar::Real;

/// How grid velocities are transferred back to particles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum TransferMode {
    /// PIC/FLIP blend controlled by [`FluidConfig::flip_blend`].
    PicFlip,
    /// Affine particle-in-cell (APIC) transfer (low dissipation, stable).
    Apic,
}

/// Parameters controlling one fluid step.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct FluidConfig {
    /// Time step in seconds.
    pub dt: Real,
    /// Constant body acceleration (gravity).
    pub gravity: Vec3,
    /// Fraction of FLIP in the PIC/FLIP blend (`1.0` = pure FLIP, `0.0` = pure
    /// PIC). Ignored when [`TransferMode::Apic`] is selected.
    pub flip_blend: Real,
    /// Number of Gauss–Seidel sweeps for the pressure Poisson solve.
    pub pressure_iterations: usize,
    /// Successive over-relaxation factor for the Poisson solve, in `[1, 2)`.
    pub over_relaxation: Real,
    /// Rest fluid density (used to scale the pressure solve; velocity result
    /// is density-independent for a single-phase liquid).
    pub density: Real,
    /// The particle-to-grid / grid-to-particle transfer scheme.
    pub transfer: TransferMode,
    /// Number of velocity-extrapolation sweeps into air cells.
    pub extrapolation_iterations: usize,
}

impl FluidConfig {
    /// Creates a configuration with the given time step and gravity, using a
    /// 0.95 FLIP blend and 60 Gauss–Seidel sweeps.
    #[must_use]
    pub fn new(dt: Real, gravity: Vec3) -> FluidConfig {
        FluidConfig {
            dt,
            gravity,
            flip_blend: 0.95,
            pressure_iterations: 60,
            over_relaxation: 1.4,
            density: 1.0e3,
            transfer: TransferMode::PicFlip,
            extrapolation_iterations: 4,
        }
    }
}

impl Default for FluidConfig {
    fn default() -> FluidConfig {
        FluidConfig::new(1.0e-2, Vec3::new(0.0, -9.81, 0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let c = FluidConfig::default();
        assert!(c.dt > 0.0);
        assert!(c.flip_blend > 0.9 && c.flip_blend <= 1.0);
        assert!(c.over_relaxation >= 1.0 && c.over_relaxation < 2.0);
        assert_eq!(c.transfer, TransferMode::PicFlip);
    }
}
