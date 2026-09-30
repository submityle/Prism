//! Tunables and the stable time-step formula for the `GPU` `CFL` reducer.
//!
//! An adaptive simulation caps its time step by the Courant-Friedrichs-Lewy
//! (`CFL`) condition: no particle may cross more than a fraction of a grid cell
//! in one step, or the advection and grid transfer become unstable. Given the
//! fastest particle speed in the scene, [`CflConfig::suggest_dt`] turns the
//! `CFL` bound into a concrete, clamped time step.
//!
//! # The formula
//!
//! ```text
//! dt = clamp(cfl_number * cell_size / max_speed, dt_min, dt_max)
//! ```
//!
//! A larger `max_speed` shrinks the step; a still scene (`max_speed <= 0`)
//! takes the largest permitted step, `dt_max`. The `dt_min` floor keeps a
//! single very fast particle from stalling the whole simulation, trading a
//! little stability for forward progress.
//!
//! # Provenance
//!
//! The `CFL` condition is a classical, openly published stability criterion for
//! explicit advection schemes. This module contains no Unreal Engine source or
//! derived code.

/// Parameters controlling the adaptive time-step suggestion.
///
/// The defaults suit a unit-cell fluid grid stepping at up to `1/60 s`; a finer
/// grid should lower [`CflConfig::cell_size`] and a stiffer scene should lower
/// [`CflConfig::cfl_number`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct CflConfig {
    /// The Courant number: the fraction of a cell the fastest particle may
    /// cross in one step. Must be positive; smaller is more stable.
    pub cfl_number: f32,
    /// The grid cell size the `CFL` bound is measured against.
    pub cell_size: f32,
    /// The smallest time step the suggestion may return, so one fast particle
    /// cannot stall the simulation.
    pub dt_min: f32,
    /// The largest time step the suggestion may return, used directly when the
    /// scene is at rest.
    pub dt_max: f32,
}

impl CflConfig {
    /// Default Courant number (`0.5`), a conservative choice for explicit
    /// advection.
    pub const DEFAULT_CFL_NUMBER: f32 = 0.5;
    /// Default grid cell size (`1.0`).
    pub const DEFAULT_CELL_SIZE: f32 = 1.0;
    /// Default lower time-step bound (`1e-4 s`).
    pub const DEFAULT_DT_MIN: f32 = 1.0e-4;
    /// Default upper time-step bound (`1/60 s`).
    pub const DEFAULT_DT_MAX: f32 = 1.0 / 60.0;

    /// Suggests a stable time step for a scene whose fastest particle moves at
    /// `max_speed`.
    ///
    /// Returns [`CflConfig::dt_max`] when the scene is at rest
    /// (`max_speed <= 0`) or when `max_speed` is `NaN`, and otherwise the
    /// `CFL`-bounded step clamped into `[dt_min, dt_max]`.
    #[must_use]
    pub fn suggest_dt(&self, max_speed: f32) -> f32 {
        if max_speed.is_nan() || max_speed <= 0.0 {
            return self.dt_max;
        }
        let dt = self.cfl_number * self.cell_size / max_speed;
        dt.clamp(self.dt_min, self.dt_max)
    }
}

impl Default for CflConfig {
    fn default() -> CflConfig {
        CflConfig {
            cfl_number: Self::DEFAULT_CFL_NUMBER,
            cell_size: Self::DEFAULT_CELL_SIZE,
            dt_min: Self::DEFAULT_DT_MIN,
            dt_max: Self::DEFAULT_DT_MAX,
        }
    }
}
