//! Material and simulation parameters for the MPM solver.
//!
//! These plain-data structures configure the constitutive model
//! (fixed-corotated elasticity with optional snow plasticity) and the global
//! simulation controls (time step, gravity, wall boundary behaviour).
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! elastic/plastic parameterization follows Stomakhin et al. 2013 and the
//! MLS-MPM formulation of Hu et al. 2018.

use glam::Vec3;

use crate::math::scalar::Real;

/// Elastic material parameters for a fixed-corotated continuum.
///
/// The Young's modulus and Poisson ratio determine the Lamé parameters used by
/// the constitutive model; the reference density converts a particle's tracked
/// volume into its mass.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MpmMaterial {
    /// Young's modulus `E` (stiffness), in pascals-equivalent units.
    pub youngs_modulus: Real,
    /// Poisson ratio `ν` in the open interval `(-1, 0.5)`.
    pub poisson_ratio: Real,
    /// Reference (rest) mass density used to derive particle masses.
    pub density: Real,
}

impl MpmMaterial {
    /// Creates a material from its Young's modulus, Poisson ratio and density.
    #[must_use]
    pub const fn new(youngs_modulus: Real, poisson_ratio: Real, density: Real) -> MpmMaterial {
        MpmMaterial {
            youngs_modulus,
            poisson_ratio,
            density,
        }
    }

    /// Returns the first and second Lamé parameters `(λ, μ)` for this material.
    ///
    /// `μ = E / (2(1+ν))` and `λ = Eν / ((1+ν)(1−2ν))`.
    #[must_use]
    pub fn lame(&self) -> (Real, Real) {
        let e = self.youngs_modulus;
        let nu = self.poisson_ratio;
        let mu = e / (2.0 * (1.0 + nu));
        let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        (lambda, mu)
    }
}

impl Default for MpmMaterial {
    fn default() -> MpmMaterial {
        // A soft, jelly-like elastic solid.
        MpmMaterial::new(1.0e4, 0.2, 1.0e3)
    }
}

/// Snow-style plasticity parameters (Stomakhin et al. 2013).
///
/// After the elastic trial deformation, each singular value of the deformation
/// gradient is clamped into `[1 − critical_compression, 1 + critical_stretch]`;
/// the clamped-off part is folded into the plastic determinant, and the
/// hardening coefficient stiffens the material as it compacts.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SnowPlasticity {
    /// Critical compression `θ_c`: singular values are clamped from below by
    /// `1 − θ_c`.
    pub critical_compression: Real,
    /// Critical stretch `θ_s`: singular values are clamped from above by
    /// `1 + θ_s`.
    pub critical_stretch: Real,
    /// Hardening coefficient `ξ`. Zero disables hardening (the Lamé parameters
    /// stay constant); positive values stiffen compacted material.
    pub hardening: Real,
}

impl SnowPlasticity {
    /// Creates a plasticity configuration.
    #[must_use]
    pub const fn new(
        critical_compression: Real,
        critical_stretch: Real,
        hardening: Real,
    ) -> SnowPlasticity {
        SnowPlasticity {
            critical_compression,
            critical_stretch,
            hardening,
        }
    }
}

impl Default for SnowPlasticity {
    fn default() -> SnowPlasticity {
        // The canonical snow parameters from Stomakhin et al. 2013.
        SnowPlasticity::new(2.5e-2, 7.5e-3, 10.0)
    }
}

/// How the solver treats the domain walls when updating grid-node velocities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum BoundaryCondition {
    /// Zero the full velocity of boundary nodes (no-slip, fully clamped).
    Sticky,
    /// Zero only the wall-normal velocity component (free tangential slip).
    Slip,
    /// Zero the wall-normal component only when it points *into* the wall
    /// (one-way / separating boundary).
    Separate,
}

/// Global simulation controls for a single MPM step.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MpmConfig {
    /// Time step in seconds.
    pub dt: Real,
    /// Constant body acceleration (gravity), in world units per second².
    pub gravity: Vec3,
    /// Wall boundary behaviour.
    pub boundary: BoundaryCondition,
    /// Thickness of the boundary layer, in grid cells, that the boundary
    /// condition is applied to.
    pub boundary_thickness: usize,
    /// Whether snow plasticity is applied after the elastic update.
    pub plastic: bool,
    /// The snow-plasticity parameters (only used when `plastic` is `true`).
    pub plasticity: SnowPlasticity,
}

impl MpmConfig {
    /// Creates a configuration with the given time step and gravity, using
    /// slip walls two cells thick and elasticity only (no plasticity).
    #[must_use]
    pub fn new(dt: Real, gravity: Vec3) -> MpmConfig {
        MpmConfig {
            dt,
            gravity,
            boundary: BoundaryCondition::Slip,
            boundary_thickness: 2,
            plastic: false,
            plasticity: SnowPlasticity::default(),
        }
    }
}

impl Default for MpmConfig {
    fn default() -> MpmConfig {
        MpmConfig::new(1.0e-3, Vec3::new(0.0, -9.81, 0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lame_parameters_are_positive_for_typical_material() {
        let m = MpmMaterial::new(1.0e5, 0.3, 1.0e3);
        let (lambda, mu) = m.lame();
        assert!(mu > 0.0);
        assert!(lambda > 0.0);
        // Known closed form checks.
        assert!((mu - 1.0e5 / 2.6).abs() < 1.0);
    }

    #[test]
    fn defaults_are_sane() {
        let c = MpmConfig::default();
        assert!(c.dt > 0.0);
        assert!(c.gravity.y < 0.0);
        assert_eq!(c.boundary, BoundaryCondition::Slip);
    }
}
