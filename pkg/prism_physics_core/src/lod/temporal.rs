//! CFL-based adaptive temporal level-of-detail.
//!
//! The stability and accuracy of an explicit-ish solver degrade when a feature
//! (a particle, a wave front) crosses more than a fraction of a characteristic
//! length in a single step. The Courant–Friedrichs–Lewy (CFL) condition bounds
//! that fraction: a stable substep satisfies
//!
//! ```text
//! h_stable = cfl_target * cell_size / max_speed,
//! ```
//!
//! so the number of substeps needed to advance a frame `dt` is
//! `ceil(dt / h_stable)`, clamped to a configured range. Fast motion is stepped
//! finely; slow or static motion collapses to the minimum substep count and
//! costs almost nothing.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The CFL
//! condition and adaptive substepping are standard, publicly documented
//! numerical-integration results.

use crate::math::scalar::Real;

/// Chooses a per-frame substep count from a CFL target and the fastest motion
/// currently in the simulation.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TemporalController {
    /// Target Courant number (fraction of a cell a feature may cross per
    /// substep). Smaller is more accurate and more expensive.
    cfl_target: Real,
    /// Characteristic length used in the CFL bound (e.g. grid cell size or the
    /// shortest edge in a mesh).
    cell_size: Real,
    /// Lower bound on the substep count (always at least one).
    min_substeps: u32,
    /// Upper bound on the substep count.
    max_substeps: u32,
}

impl TemporalController {
    /// Creates a controller.
    ///
    /// `cfl_target` and `cell_size` are clamped to a tiny positive value to keep
    /// the CFL division well defined; `min_substeps` is raised to at least one
    /// and `max_substeps` to at least `min_substeps`.
    #[must_use]
    pub fn new(cfl_target: Real, cell_size: Real, min_substeps: u32, max_substeps: u32) -> Self {
        let min = min_substeps.max(1);
        TemporalController {
            cfl_target: cfl_target.max(Real::MIN_POSITIVE),
            cell_size: cell_size.max(Real::MIN_POSITIVE),
            min_substeps: min,
            max_substeps: max_substeps.max(min),
        }
    }

    /// Returns the target Courant number.
    #[must_use]
    pub fn cfl_target(&self) -> Real {
        self.cfl_target
    }

    /// Returns the characteristic length.
    #[must_use]
    pub fn cell_size(&self) -> Real {
        self.cell_size
    }

    /// Returns the substep-count bounds as `(min, max)`.
    #[must_use]
    pub fn bounds(&self) -> (u32, u32) {
        (self.min_substeps, self.max_substeps)
    }

    /// Number of substeps needed to advance a frame of `dt` seconds when the
    /// fastest feature moves at `max_speed` (metres per second).
    ///
    /// A non-positive `dt` or `max_speed` needs no refinement and returns the
    /// minimum substep count.
    #[must_use]
    pub fn substeps(&self, dt: Real, max_speed: Real) -> u32 {
        if dt <= 0.0 || max_speed <= 0.0 {
            return self.min_substeps;
        }
        // ceil(dt * max_speed / (cfl_target * cell_size)) without a transcendental
        // ceil call: take the integer part and bump it when there is a remainder.
        let ratio = dt * max_speed / (self.cfl_target * self.cell_size);
        let floor = ratio as u32;
        let needed = if (floor as Real) < ratio {
            floor + 1
        } else {
            floor
        };
        needed.clamp(self.min_substeps, self.max_substeps)
    }

    /// Stable substep duration for the frame: `dt` divided by
    /// [`substeps`](Self::substeps).
    #[must_use]
    pub fn substep_dt(&self, dt: Real, max_speed: Real) -> Real {
        let n = self.substeps(dt, max_speed).max(1);
        dt / n as Real
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slow_motion_uses_minimum_substeps() {
        let c = TemporalController::new(0.5, 0.1, 1, 32);
        assert_eq!(c.substeps(1.0 / 60.0, 0.0), 1);
        assert_eq!(c.substeps(1.0 / 60.0, 1e-6), 1);
    }

    #[test]
    fn fast_motion_refines_and_clamps() {
        let c = TemporalController::new(0.5, 0.1, 1, 8);
        // h_stable = 0.5 * 0.1 / 100 = 5e-4; dt/h = (1/60)/5e-4 ~= 33.3 -> ceil 34,
        // clamped to the max of 8.
        assert_eq!(c.substeps(1.0 / 60.0, 100.0), 8);
    }

    #[test]
    fn moderate_motion_rounds_up() {
        let c = TemporalController::new(0.5, 1.0, 1, 64);
        // h_stable = 0.5; dt = 1.2 -> ratio 2.4 -> ceil 3.
        assert_eq!(c.substeps(1.2, 1.0), 3);
    }

    #[test]
    fn substep_dt_matches_count() {
        let c = TemporalController::new(0.5, 1.0, 1, 64);
        let dt = 1.2;
        let n = c.substeps(dt, 1.0);
        assert!((c.substep_dt(dt, 1.0) - dt / n as Real).abs() < 1e-6);
    }

    #[test]
    fn degenerate_parameters_are_sanitised() {
        let c = TemporalController::new(-1.0, 0.0, 0, 0);
        let (min, max) = c.bounds();
        assert_eq!(min, 1);
        assert!(max >= min);
        assert!(c.cfl_target() > 0.0);
        assert!(c.cell_size() > 0.0);
    }
}
