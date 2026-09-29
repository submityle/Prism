//! Configuration for the `GPU` `FLIP`/`APIC` free-surface fluid solver.
//!
//! [`FluidConfig`] mirrors the tunables of the `CPU` reference in
//! [`prism_physics_core`](prism_physics_core::fluid::config): the time step,
//! body acceleration, the `PIC`/`FLIP` blend, the pressure-projection controls,
//! and the transfer mode. The `GPU` solver splits the same step into the same
//! canonical pipeline (`P2G` -> save -> gravity -> solid -> project ->
//! extrapolate -> `G2P` -> advect), so the parameters carry the same meaning.
//!
//! # Provenance
//!
//! This module contains no Unreal Engine source or derived code. The
//! `PIC`/`FLIP` blend and pressure-projection controls follow Bridson, *Fluid
//! Simulation for Computer Graphics*, and Zhu and Bridson 2005.

use glam::Vec3;

/// How grid velocities are transferred back to particles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferMode {
    /// `PIC`/`FLIP` blend controlled by [`FluidConfig::flip_blend`].
    PicFlip,
    /// Affine particle-in-cell (`APIC`) transfer (low dissipation, stable).
    Apic,
}

/// Parameters controlling one fluid step on the `GPU`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FluidConfig {
    /// Time step in seconds.
    pub dt: f32,
    /// Constant body acceleration (gravity).
    pub gravity: Vec3,
    /// Fraction of `FLIP` in the `PIC`/`FLIP` blend (`1.0` = pure `FLIP`,
    /// `0.0` = pure `PIC`). Ignored when [`TransferMode::Apic`] is selected.
    pub flip_blend: f32,
    /// Number of red-black Gauss-Seidel sweeps for the pressure Poisson solve.
    pub pressure_iterations: u32,
    /// Successive over-relaxation factor for the Poisson solve, in `[1, 2)`.
    pub over_relaxation: f32,
    /// Rest fluid density (scales the pressure solve; the velocity result is
    /// density-independent for a single-phase liquid).
    pub density: f32,
    /// The particle-to-grid / grid-to-particle transfer scheme.
    pub transfer: TransferMode,
    /// Number of velocity-extrapolation sweeps into air cells.
    pub extrapolation_iterations: u32,
}

impl FluidConfig {
    /// Default `FLIP` fraction in the `PIC`/`FLIP` blend.
    pub const DEFAULT_FLIP_BLEND: f32 = 0.95;
    /// Default number of pressure Gauss-Seidel sweeps.
    pub const DEFAULT_PRESSURE_ITERATIONS: u32 = 60;
    /// Default successive over-relaxation factor.
    pub const DEFAULT_OVER_RELAXATION: f32 = 1.4;
    /// Default rest fluid density.
    pub const DEFAULT_DENSITY: f32 = 1.0e3;
    /// Default number of velocity-extrapolation sweeps.
    pub const DEFAULT_EXTRAPOLATION_ITERATIONS: u32 = 4;

    /// Creates a configuration with the given time step and gravity, using a
    /// `0.95` `FLIP` blend and 60 Gauss-Seidel sweeps.
    #[must_use]
    pub fn new(dt: f32, gravity: Vec3) -> FluidConfig {
        FluidConfig {
            dt,
            gravity,
            flip_blend: Self::DEFAULT_FLIP_BLEND,
            pressure_iterations: Self::DEFAULT_PRESSURE_ITERATIONS,
            over_relaxation: Self::DEFAULT_OVER_RELAXATION,
            density: Self::DEFAULT_DENSITY,
            transfer: TransferMode::PicFlip,
            extrapolation_iterations: Self::DEFAULT_EXTRAPOLATION_ITERATIONS,
        }
    }

    /// The `FLIP` blend clamped to `[0, 1]`.
    #[must_use]
    pub fn effective_flip_blend(&self) -> f32 {
        self.flip_blend.clamp(0.0, 1.0)
    }

    /// The over-relaxation factor clamped to `[1.0, 1.99]`.
    #[must_use]
    pub fn effective_over_relaxation(&self) -> f32 {
        self.over_relaxation.clamp(1.0, 1.99)
    }

    /// Validates the configuration, returning the first violated invariant.
    ///
    /// # Errors
    ///
    /// Returns [`FluidError::InvalidConfig`] when a scalar parameter is not
    /// finite or the density or time step is non-positive.
    pub fn validate(&self) -> Result<(), FluidError> {
        if !(self.dt.is_finite() && self.dt > 0.0) {
            return Err(FluidError::InvalidConfig("dt must be finite and positive"));
        }
        if !self.gravity.is_finite() {
            return Err(FluidError::InvalidConfig("gravity must be finite"));
        }
        if !self.flip_blend.is_finite() {
            return Err(FluidError::InvalidConfig("flip_blend must be finite"));
        }
        if !self.over_relaxation.is_finite() {
            return Err(FluidError::InvalidConfig("over_relaxation must be finite"));
        }
        if !(self.density.is_finite() && self.density > 0.0) {
            return Err(FluidError::InvalidConfig(
                "density must be finite and positive",
            ));
        }
        Ok(())
    }
}

impl Default for FluidConfig {
    fn default() -> FluidConfig {
        FluidConfig::new(1.0e-2, Vec3::new(0.0, -9.81, 0.0))
    }
}

/// Errors the `GPU` fluid solver can report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FluidError {
    /// A configuration invariant was violated; carries a static reason.
    InvalidConfig(&'static str),
    /// A grid dimension was zero; the solver needs at least one cell per axis.
    EmptyGrid,
    /// The particle Structure-of-Arrays columns had mismatched lengths.
    InconsistentParticles,
}

impl core::fmt::Display for FluidError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FluidError::InvalidConfig(reason) => write!(f, "invalid fluid config: {reason}"),
            FluidError::EmptyGrid => write!(f, "fluid grid must have at least one cell per axis"),
            FluidError::InconsistentParticles => {
                write!(f, "fluid particle columns must have equal length")
            }
        }
    }
}

impl core::error::Error for FluidError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let c = FluidConfig::default();
        assert!(c.dt > 0.0);
        assert!(c.flip_blend > 0.9 && c.flip_blend <= 1.0);
        assert!(c.effective_over_relaxation() >= 1.0 && c.effective_over_relaxation() < 2.0);
        assert_eq!(c.transfer, TransferMode::PicFlip);
        c.validate().expect("defaults valid");
    }

    #[test]
    fn rejects_non_finite() {
        let c = FluidConfig {
            dt: 0.0,
            ..Default::default()
        };
        assert!(matches!(c.validate(), Err(FluidError::InvalidConfig(_))));
    }
}
