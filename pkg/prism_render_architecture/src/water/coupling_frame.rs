//! Per-frame two-way coupling assembly: forces plus the sub-step schedule.
//!
//! This module composes the pure coupling primitives from [`super::coupling`]
//! into a single deterministic per-frame plan. Given a body/fluid `profile` and
//! the current frame `inputs`, it evaluates the `Archimedes` buoyancy, quadratic
//! drag, and added-mass reaction, decides how many `SWE`/`PBF` sub-steps to run
//! and how large a bounded `GPU` read-back batch to issue, and reports the
//! fraction of body momentum stamped back into the fluid.
//!
//! All math here is pure and deterministic: the same inputs always yield the
//! same [`CouplingFramePlan`]. Only `sqrt` may appear among float operations
//! (none is needed here), and there are no `f32` equality tests and no AI/ML.

use super::coupling::{
    added_mass, buoyancy_force, drag_force, plan_coupling, source_writeback_fraction, CouplingPlan,
};
use super::GRAVITY;

/// Static body/fluid coupling parameters that persist across frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CouplingProfile {
    /// Density of the surrounding fluid, in mass per unit volume.
    pub fluid_density: f32,
    /// Dimensionless quadratic form-drag coefficient.
    pub drag_coeff: f32,
    /// Dimensionless added-mass coefficient for the body shape.
    pub added_mass_coeff: f32,
    /// Upper bound on coupling sub-steps run per frame.
    pub max_substeps: u32,
    /// Upper bound on field queries read back from the `GPU` per frame.
    pub max_readback: u32,
}

/// Per-frame coupling inputs sampled from the body and fluid state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CouplingInputs {
    /// Number of field queries requested this frame.
    pub query_count: u32,
    /// Fastest relative body/fluid speed this frame, for scheduling.
    pub max_rel_speed: f32,
    /// Frame time step.
    pub frame_dt: f32,
    /// Fluid grid cell size.
    pub cell_size: f32,
    /// Volume of the body currently below the surface.
    pub submerged_volume: f32,
    /// Total volume of the body.
    pub total_volume: f32,
    /// Cross-sectional area presented to the flow, for drag.
    pub cross_section: f32,
    /// Relative body/fluid speed used to evaluate drag.
    pub rel_speed: f32,
}

/// The assembled forces and schedule for one coupling frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CouplingFramePlan {
    /// Sub-step and read-back schedule for the frame.
    pub plan: CouplingPlan,
    /// `Archimedes` buoyancy force magnitude, directed upward.
    pub buoyancy: f32,
    /// Quadratic drag force magnitude, opposing relative motion.
    pub drag: f32,
    /// Added-mass reaction magnitude from the entrained fluid.
    pub added_mass: f32,
    /// Fraction of body momentum written back into the fluid, in `0..=1`.
    pub writeback_fraction: f32,
}

/// Assembles the forces and schedule for a single coupling frame.
///
/// Combines [`plan_coupling`], [`buoyancy_force`], [`drag_force`],
/// [`added_mass`], and [`source_writeback_fraction`] into one
/// [`CouplingFramePlan`]. The result is fully determined by `profile` and
/// `inputs`, so replaying the same arguments reproduces the same plan.
#[must_use]
pub fn plan_coupling_frame(profile: CouplingProfile, inputs: CouplingInputs) -> CouplingFramePlan {
    let plan = plan_coupling(
        inputs.query_count,
        inputs.max_rel_speed,
        inputs.frame_dt,
        inputs.cell_size,
        profile.max_substeps,
        profile.max_readback,
    );
    let buoyancy = buoyancy_force(profile.fluid_density, inputs.submerged_volume, GRAVITY);
    let drag = drag_force(
        profile.drag_coeff,
        profile.fluid_density,
        inputs.cross_section,
        inputs.rel_speed,
    );
    let added_mass_val = added_mass(
        profile.added_mass_coeff,
        profile.fluid_density,
        inputs.submerged_volume,
    );
    let writeback_fraction =
        source_writeback_fraction(inputs.submerged_volume, inputs.total_volume);
    CouplingFramePlan {
        plan,
        buoyancy,
        drag,
        added_mass: added_mass_val,
        writeback_fraction,
    }
}

#[cfg(test)]
mod tests {
    use super::super::EPS;
    use super::*;

    /// Fixed body/fluid profile shared across the tests.
    const PROFILE: CouplingProfile = CouplingProfile {
        fluid_density: 1000.0,
        drag_coeff: 1.0,
        added_mass_coeff: 0.5,
        max_substeps: 16,
        max_readback: 32,
    };

    /// Baseline inputs; individual tests override the fields they exercise.
    fn base_inputs() -> CouplingInputs {
        CouplingInputs {
            query_count: 8,
            max_rel_speed: 1.0,
            frame_dt: 1.0 / 60.0,
            cell_size: 0.25,
            submerged_volume: 0.3,
            total_volume: 1.0,
            cross_section: 2.0,
            rel_speed: 1.0,
        }
    }

    #[test]
    fn buoyancy_rises_with_submerged_volume() {
        let mut shallow_inputs = base_inputs();
        shallow_inputs.submerged_volume = 0.1;
        let mut deep_inputs = base_inputs();
        deep_inputs.submerged_volume = 0.5;
        let shallow = plan_coupling_frame(PROFILE, shallow_inputs).buoyancy;
        let deep = plan_coupling_frame(PROFILE, deep_inputs).buoyancy;
        assert!(deep > shallow, "more submerged volume lifts more");
        assert!(shallow >= 0.0);
    }

    #[test]
    fn drag_quadruples_when_speed_doubles() {
        let mut slow_inputs = base_inputs();
        slow_inputs.rel_speed = 1.0;
        let mut fast_inputs = base_inputs();
        fast_inputs.rel_speed = 2.0;
        let slow = plan_coupling_frame(PROFILE, slow_inputs).drag;
        let fast = plan_coupling_frame(PROFILE, fast_inputs).drag;
        assert!(
            (fast - 4.0 * slow).abs() < 1e-2 * fast.max(1.0),
            "quadratic drag quadruples when speed doubles",
        );
        assert!(slow >= 0.0);
    }

    #[test]
    fn readback_batch_is_query_count_capped() {
        // Under the cap: batch tracks the query count.
        let mut small = base_inputs();
        small.query_count = 8;
        let batch_small = plan_coupling_frame(PROFILE, small).plan.readback_batch;
        assert_eq!(batch_small, small.query_count.min(PROFILE.max_readback));
        assert_eq!(batch_small, 8);
        // Over the cap: batch is clamped to max_readback.
        let mut big = base_inputs();
        big.query_count = 1000;
        let batch_big = plan_coupling_frame(PROFILE, big).plan.readback_batch;
        assert_eq!(batch_big, big.query_count.min(PROFILE.max_readback));
        assert_eq!(batch_big, PROFILE.max_readback);
    }

    #[test]
    fn faster_speed_yields_more_substeps_capped() {
        let mut slow = base_inputs();
        slow.max_rel_speed = 1.0;
        slow.frame_dt = 0.1;
        slow.cell_size = 0.1;
        let mut fast = base_inputs();
        fast.max_rel_speed = 20.0;
        fast.frame_dt = 0.1;
        fast.cell_size = 0.1;
        let slow_steps = plan_coupling_frame(PROFILE, slow).plan.substeps;
        let fast_steps = plan_coupling_frame(PROFILE, fast).plan.substeps;
        assert!(
            fast_steps > slow_steps,
            "faster motion needs more sub-steps"
        );
        assert!(fast_steps <= PROFILE.max_substeps, "sub-steps stay capped");
        assert!(slow_steps >= 1, "at least one sub-step");
    }

    #[test]
    fn writeback_tracks_submersion_and_clamps() {
        let mut half = base_inputs();
        half.submerged_volume = 0.25;
        half.total_volume = 1.0;
        let mut full = base_inputs();
        full.submerged_volume = 0.75;
        full.total_volume = 1.0;
        let frac_half = plan_coupling_frame(PROFILE, half).writeback_fraction;
        let frac_full = plan_coupling_frame(PROFILE, full).writeback_fraction;
        assert!(frac_full > frac_half, "deeper bodies inject more momentum");
        // Over-submerged (submerged exceeds total) clamps to one.
        let mut over = base_inputs();
        over.submerged_volume = 5.0;
        over.total_volume = 1.0;
        let frac_over = plan_coupling_frame(PROFILE, over).writeback_fraction;
        assert!((frac_over - 1.0).abs() < EPS, "fraction clamps to one");
        assert!(frac_over <= 1.0);
    }

    #[test]
    fn plan_is_deterministic() {
        let inputs = base_inputs();
        let first = plan_coupling_frame(PROFILE, inputs);
        let second = plan_coupling_frame(PROFILE, inputs);
        assert_eq!(first, second, "same inputs reproduce the same plan");
    }
}
