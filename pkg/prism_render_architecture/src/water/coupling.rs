//! Two-way fluid/rigid coupling: buoyancy, drag, and sub-step scheduling.
//!
//! Coupling is bidirectional. The fluid exposes height, velocity, and pressure
//! queries so `physics_core` can compute the `Archimedes` buoyancy, drag, and
//! added-mass reaction on floating bodies; the bodies write momentum sources
//! back into the fluid so their motion pushes the water. To stop the two
//! integrators from jittering against each other, the exchange runs a small
//! fixed number of sub-steps per frame, and the field queries are batched into
//! a bounded read-back rather than a per-frame full read of the GPU field.
//!
//! This module owns the pure, deterministic force and scheduling math. Forces
//! are non-negative magnitudes with an explicit sense in their docs, and only
//! `sqrt` is used. There are no `f32` equality tests and no AI/ML.

use super::EPS;

/// `Archimedes` buoyancy force magnitude on a submerged volume.
///
/// Returns `fluid_density * submerged_volume * GRAVITY`, the weight of the
/// displaced fluid, directed upward. It is zero for a body clear of the water
/// and rises linearly with the submerged volume, so a body sinking deeper feels
/// more lift until it is fully submerged. The magnitude is non-negative.
#[must_use]
pub fn buoyancy_force(fluid_density: f32, submerged_volume: f32, gravity: f32) -> f32 {
    fluid_density.max(0.0) * submerged_volume.max(0.0) * gravity.max(0.0)
}

/// Quadratic hydrodynamic drag force magnitude opposing relative motion.
///
/// Returns `0.5 * drag_coeff * fluid_density * area * rel_speed^2`, the standard
/// form-drag law, always directed against the body's velocity relative to the
/// fluid. It grows with the square of the relative speed, so it is the dominant
/// resistance at speed. The magnitude is non-negative.
#[must_use]
pub fn drag_force(drag_coeff: f32, fluid_density: f32, area: f32, rel_speed: f32) -> f32 {
    let v = rel_speed.max(0.0);
    0.5 * drag_coeff.max(0.0) * fluid_density.max(0.0) * area.max(0.0) * v * v
}

/// Added-mass reaction from the fluid a body must accelerate with itself.
///
/// Returns `added_mass_coeff * fluid_density * displaced_volume`, the effective
/// extra inertia the surrounding fluid contributes. Folding this into the
/// body's mass in the sub-step is what keeps the coupled system stable when a
/// light body accelerates in dense water. The value is non-negative.
#[must_use]
pub fn added_mass(added_mass_coeff: f32, fluid_density: f32, displaced_volume: f32) -> f32 {
    added_mass_coeff.max(0.0) * fluid_density.max(0.0) * displaced_volume.max(0.0)
}

/// Fraction of a body's momentum written back into the fluid, by submersion.
///
/// Returns `submerged_volume / total_volume`, clamped to `0..=1`: a body barely
/// touching the surface injects little momentum, a fully submerged body injects
/// all of it. A degenerate zero total volume yields zero. This weights the
/// source term the body stamps into the velocity field.
#[must_use]
pub fn source_writeback_fraction(submerged_volume: f32, total_volume: f32) -> f32 {
    let total = total_volume.max(0.0);
    if total <= EPS {
        return 0.0;
    }
    (submerged_volume.max(0.0) / total).clamp(0.0, 1.0)
}

/// The per-frame coupling schedule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CouplingPlan {
    /// Number of fixed-point sub-steps to run this frame.
    pub substeps: u32,
    /// Number of field queries read back from the GPU this frame.
    pub readback_batch: u32,
}

/// Chooses the sub-step count and read-back batch for one coupling frame.
///
/// Faster relative motion needs more sub-steps to converge without jitter, so
/// the count rises with `max_rel_speed` (scaled by the time step and cell size)
/// and is clamped to `max_substeps` with a floor of one. The read-back batch is
/// the query count capped at `max_readback`, honouring the rule that coupling
/// never reads the whole field back per frame. Fully determined by its inputs.
#[must_use]
pub fn plan_coupling(
    query_count: u32,
    max_rel_speed: f32,
    dt: f32,
    dx: f32,
    max_substeps: u32,
    max_readback: u32,
) -> CouplingPlan {
    let cap = max_substeps.max(1);
    let cell = dx.max(EPS);
    // Courant-like count: how many cells the fastest body crosses per frame.
    let crossings = max_rel_speed.max(0.0) * dt.max(0.0) / cell;
    let needed = 1 + crossings as u32;
    let substeps = needed.min(cap);
    let readback_batch = query_count.min(max_readback);
    CouplingPlan {
        substeps,
        readback_batch,
    }
}

#[cfg(test)]
mod tests {
    use super::super::GRAVITY;
    use super::*;

    #[test]
    fn buoyancy_rises_with_submerged_volume() {
        let shallow = buoyancy_force(1000.0, 0.1, GRAVITY);
        let deep = buoyancy_force(1000.0, 0.5, GRAVITY);
        assert!(deep > shallow, "more displacement lifts more");
        assert!(shallow >= 0.0);
        // Clear of the water: no lift.
        assert!(buoyancy_force(1000.0, 0.0, GRAVITY).abs() < EPS);
    }

    #[test]
    fn drag_grows_with_speed_squared() {
        let slow = drag_force(1.0, 1000.0, 2.0, 1.0);
        let fast = drag_force(1.0, 1000.0, 2.0, 2.0);
        // Doubling speed quadruples drag.
        assert!((fast - 4.0 * slow).abs() < 1e-2 * fast.max(1.0));
        assert!(slow >= 0.0);
        assert!(drag_force(1.0, 1000.0, 2.0, 0.0).abs() < EPS);
    }

    #[test]
    fn added_mass_scales_with_density_and_volume() {
        let light = added_mass(0.5, 1000.0, 0.2);
        let heavy = added_mass(0.5, 1000.0, 1.0);
        assert!(heavy > light);
        assert!(light >= 0.0);
    }

    #[test]
    fn writeback_fraction_tracks_submersion() {
        assert!(source_writeback_fraction(0.0, 1.0).abs() < EPS);
        assert!((source_writeback_fraction(1.0, 1.0) - 1.0).abs() < EPS);
        assert!((source_writeback_fraction(0.5, 1.0) - 0.5).abs() < EPS);
        // Over-submersion clamps to one; degenerate total is inert.
        assert!((source_writeback_fraction(2.0, 1.0) - 1.0).abs() < EPS);
        assert!(source_writeback_fraction(1.0, 0.0).abs() < EPS);
    }

    #[test]
    fn plan_scales_substeps_with_speed_and_caps_readback() {
        let calm = plan_coupling(100, 1.0, 0.016, 1.0, 16, 32);
        let churning = plan_coupling(100, 100.0, 0.016, 1.0, 16, 32);
        assert!(
            churning.substeps >= calm.substeps,
            "faster flow needs more sub-steps"
        );
        assert!(churning.substeps <= 16, "sub-steps respect the cap");
        assert!(calm.substeps >= 1, "at least one sub-step");
        // Read-back never exceeds the cap, honouring the no-full-readback rule.
        assert_eq!(calm.readback_batch, 32);
        assert_eq!(
            plan_coupling(10, 1.0, 0.016, 1.0, 16, 32).readback_batch,
            10
        );
    }

    #[test]
    fn plan_is_deterministic() {
        let a = plan_coupling(50, 10.0, 0.016, 0.5, 8, 64);
        let b = plan_coupling(50, 10.0, 0.016, 0.5, 8, 64);
        assert_eq!(a, b);
    }
}
