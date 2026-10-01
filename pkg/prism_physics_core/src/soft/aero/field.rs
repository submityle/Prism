//! Wind field and per-triangle aerodynamic coefficients.
//!
//! These are the sanitizable inputs to the aero pass: a [`WindField`] describes
//! the ambient airflow (and optional turbulence) every face feels, and
//! [`AeroParams`] carries the drag/lift coefficients and the optional fluid
//! density that selects the linear or quadratic force model. Both types expose
//! a [`sanitized`](WindField::sanitized) copy that is finite and range-clamped,
//! so a caller can never inject a `NaN` or a negative coefficient into the
//! simulation.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! per-triangle drag/lift decomposition and the optional quadratic
//! dynamic-pressure term are the standard, publicly documented cloth
//! aerodynamics model (described for e.g. UE5 `Chaos` Cloth and NVIDIA
//! `NvCloth`).

use glam::Vec3;

use crate::math::scalar::Real;

use super::sanitize::{sanitize_non_negative, sanitize_unit, sanitize_vec};

/// Ambient wind sampled as a single world-space velocity plus a turbulence
/// strength.
///
/// `velocity` is the steady airflow every face feels; `turbulence` in `0..=1`
/// scales a small, deterministic per-triangle jitter added to that airflow so
/// the garment does not move as one rigid clump. A default (zero velocity, zero
/// turbulence) field exerts no force, so callers may always run the pass
/// unconditionally.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct WindField {
    /// Steady wind velocity in world units per second.
    pub velocity: Vec3,
    /// Turbulence strength, clamped to `0..=1`; scales the per-triangle jitter.
    pub turbulence: Real,
}

impl WindField {
    /// Builds a wind field from a steady velocity and turbulence strength.
    ///
    /// The inputs are stored verbatim; call [`WindField::sanitized`] to obtain a
    /// finite, range-clamped copy before simulating.
    #[must_use]
    pub const fn new(velocity: Vec3, turbulence: Real) -> Self {
        WindField {
            velocity,
            turbulence,
        }
    }

    /// Returns a copy with any non-finite velocity component replaced by zero
    /// and `turbulence` clamped to `0..=1` (a `NaN` becomes `0`).
    ///
    /// This guarantees the field can never inject a `NaN` or a negative
    /// turbulence into the simulation.
    #[must_use]
    pub fn sanitized(self) -> Self {
        WindField {
            velocity: sanitize_vec(self.velocity),
            turbulence: sanitize_unit(self.turbulence),
        }
    }
}

/// Per-triangle aerodynamic coefficients.
///
/// `drag` scales the force along the face normal (resisting the airflow) and
/// `lift` scales the in-plane force (the sideways push that makes fabric
/// flutter). Both are dimensionless multipliers on `area * relative_wind`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AeroParams {
    /// Normal-direction (drag) coefficient; clamped non-negative.
    pub drag: Real,
    /// In-plane (lift) coefficient; clamped non-negative.
    pub lift: Real,
    /// Fluid (air) density scaling the quadratic drag/lift term, clamped
    /// non-negative. A non-positive density selects the linear
    /// `area * relative_wind` model; a positive density selects the UE5
    /// `Chaos`-style quadratic model, where the per-face force additionally
    /// scales by `0.5 * air_density * relative_wind_magnitude`, so it grows
    /// with the square of the airspeed the way real aerodynamic drag does.
    pub air_density: Real,
}

impl AeroParams {
    /// Builds coefficients from a drag and lift value, leaving the fluid
    /// density at zero so the linear aerodynamic model is used.
    ///
    /// The inputs are stored verbatim; call [`AeroParams::sanitized`] for a
    /// finite, non-negative copy before simulating. Use
    /// [`AeroParams::with_air_density`] to opt into the quadratic model.
    #[must_use]
    pub const fn new(drag: Real, lift: Real) -> Self {
        AeroParams {
            drag,
            lift,
            air_density: 0.0,
        }
    }

    /// Returns a copy with the fluid density set to `air_density`, opting the
    /// coefficients into the quadratic (airspeed-squared) aerodynamic model.
    ///
    /// A non-positive `air_density` keeps the linear model; the value is stored
    /// verbatim and clamped non-negative by [`AeroParams::sanitized`].
    #[must_use]
    pub const fn with_air_density(self, air_density: Real) -> Self {
        AeroParams {
            drag: self.drag,
            lift: self.lift,
            air_density,
        }
    }

    /// Returns a copy with every coefficient made finite and non-negative (a
    /// `NaN` or negative value becomes `0`).
    #[must_use]
    pub fn sanitized(self) -> Self {
        AeroParams {
            drag: sanitize_non_negative(self.drag),
            lift: sanitize_non_negative(self.lift),
            air_density: sanitize_non_negative(self.air_density),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windfield_default_is_calm() {
        let field = WindField::default();
        assert_eq!(field.velocity, Vec3::ZERO);
        assert!(field.turbulence.abs() < 1.0e-6);
    }

    #[test]
    fn windfield_sanitize_clamps_turbulence_and_nan() {
        let dirty = WindField {
            velocity: Vec3::new(Real::NAN, 1.0, Real::INFINITY),
            turbulence: Real::NAN,
        };
        let clean = dirty.sanitized();
        assert_eq!(clean.velocity, Vec3::new(0.0, 1.0, 0.0));
        assert!(clean.turbulence.abs() < 1.0e-6);

        let over = WindField::new(Vec3::ZERO, 5.0).sanitized();
        assert!((over.turbulence - 1.0).abs() < 1.0e-6);
        let under = WindField::new(Vec3::ZERO, -2.0).sanitized();
        assert!(under.turbulence.abs() < 1.0e-6);
    }

    #[test]
    fn aeroparams_sanitize_clamps_negative_and_nan() {
        let clean = AeroParams::new(-3.0, Real::NAN).sanitized();
        assert!(clean.drag.abs() < 1.0e-6);
        assert!(clean.lift.abs() < 1.0e-6);
        let kept = AeroParams::new(0.5, 2.0).sanitized();
        assert!((kept.drag - 0.5).abs() < 1.0e-6);
        assert!((kept.lift - 2.0).abs() < 1.0e-6);
    }

    #[test]
    fn with_air_density_opts_into_quadratic_model() {
        let p = AeroParams::new(1.0, 1.0).with_air_density(1.225);
        assert!((p.air_density - 1.225).abs() < 1.0e-6);
        let neg = AeroParams::new(1.0, 1.0).with_air_density(-5.0).sanitized();
        assert!(neg.air_density.abs() < 1.0e-6);
    }
}
