//! Deterministic, budgeted, engine-neutral world simulation primitives.
//!
//! This crate owns bounded integer algorithms for local erosion, destructive
//! voxel deltas, fragment deltas, ecological state transitions, and discrete
//! hydrology. It does not model complete rigid-body, fluid, terrain, or
//! material physics. Every step is finite, deterministic, and returns the
//! changes and dirty regions needed by an external world owner.
#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

extern crate alloc;

mod codec;
mod ecology;
mod hydrology;
#[cfg(feature = "persist")]
mod persist;
mod sim;
mod types;
mod voxel;

pub use codec::{
    decode_delta, decode_delta_with_limits, encode_delta, CodecError, DeltaDecodeLimits,
};
pub use ecology::{EcologyCell, EcologyConfig, EcologyDelta, EcologyField};
pub use hydrology::{FlowDelta, HydrologyCell, HydrologyField};
#[cfg(feature = "persist")]
pub use persist::{
    simulation_delta_to_transaction, transaction_to_simulation_delta, SimulationPersistError,
    SIMULATION_DELTA_FIELD,
};
pub use sim::{
    BudgetUsage, SimulationApplyError, SimulationDelta, SimulationError, TickBudget, TickReport,
    WorldSim, WorldSimConfig,
};
pub use types::{CellCoord, DirtyRegion, Region};
pub use voxel::{stable_fragment_key, FragmentDelta, VoxelCell, VoxelDelta, VoxelField};

#[cfg(test)]
mod tests {
    use super::*;

    fn populated_sim() -> WorldSim {
        let mut sim = WorldSim::new(WorldSimConfig::default()).unwrap();
        let source = CellCoord::new(0, 0, 0);
        let sink = CellCoord::new(1, 0, 0);
        sim.hydrology
            .set(source, HydrologyCell::new(2, 40, 100, u16::MAX));
        sim.hydrology
            .set(sink, HydrologyCell::new(0, 0, 100, u16::MAX));
        sim.voxels.set(source, VoxelCell::new(7, 5, 0).unwrap());
        sim.ecology
            .set(source, EcologyCell::new(4, 65_535, 100).unwrap());
        sim
    }

    #[test]
    fn one_tick_conserves_water_except_for_erosion_consumption() {
        let mut sim = populated_sim();
        let initial_water = sim.hydrology.water_at(CellCoord::new(0, 0, 0))
            + sim.hydrology.water_at(CellCoord::new(1, 0, 0));
        let report = sim.step(TickBudget::new(64, 64, 16)).unwrap();
        let final_water = sim.hydrology.water_at(CellCoord::new(0, 0, 0))
            + sim.hydrology.water_at(CellCoord::new(1, 0, 0));
        let consumed = report
            .delta
            .voxel_deltas
            .iter()
            .map(|delta| delta.removed_mass * 8)
            .sum::<u32>();
        let moved = report
            .delta
            .flow_deltas
            .iter()
            .map(|delta| delta.amount)
            .sum::<u32>();
        assert_eq!(moved, 10);
        assert_eq!(initial_water, final_water + consumed);
        assert_eq!(report.delta.voxel_deltas[0].removed_mass, 1);
    }

    #[test]
    fn identical_inputs_produce_identical_tick_reports() {
        let mut left = populated_sim();
        let mut right = populated_sim();
        let budget = TickBudget::new(64, 64, 16);
        assert_eq!(left.step(budget), right.step(budget));
        assert_eq!(left, right);
    }

    #[test]
    fn shared_budget_caps_work_and_output() {
        let mut sim = populated_sim();
        let report = sim.step(TickBudget::new(1, 1, 1)).unwrap();
        assert!(report.budget.work_units <= 1);
        assert!(report.budget.output_records <= 1);
        assert!(report.delta.voxel_deltas.is_empty());
        assert!(report.delta.fragment_deltas.is_empty());
    }

    #[test]
    fn dirty_regions_merge_adjacent_phase_outputs() {
        let mut sim = populated_sim();
        let report = sim.step(TickBudget::new(64, 64, 16)).unwrap();
        assert_eq!(report.dirty_regions.len(), 1);
        assert_eq!(
            report.dirty_regions[0].region,
            Region::new(CellCoord::new(-1, -1, -1), CellCoord::new(1, 1, 1)).unwrap()
        );
    }

    #[test]
    fn stable_fragment_keys_are_replayable_and_coordinate_sensitive() {
        let origin = CellCoord::new(-3, 2, 8);
        assert_eq!(
            stable_fragment_key(9, origin, 4),
            stable_fragment_key(9, origin, 4)
        );
        assert_ne!(
            stable_fragment_key(9, origin, 4),
            stable_fragment_key(10, origin, 4)
        );
        assert_ne!(
            stable_fragment_key(9, origin, 4),
            stable_fragment_key(9, CellCoord::new(-2, 2, 8), 4)
        );
    }

    #[test]
    fn extreme_elevation_difference_routes_without_signed_overflow() {
        let mut sim = WorldSim::new(WorldSimConfig::default()).unwrap();
        let source = CellCoord::new(0, 0, 0);
        let sink = CellCoord::new(1, 0, 0);
        sim.hydrology
            .set(source, HydrologyCell::new(i32::MAX, 40, 100, u16::MAX));
        sim.hydrology
            .set(sink, HydrologyCell::new(i32::MIN, 0, 100, u16::MAX));
        let report = sim.step(TickBudget::new(8, 8, 4)).unwrap();
        assert_eq!(report.delta.flow_deltas[0].amount, 10);
    }

    #[test]
    fn emitted_delta_replays_exactly_without_rerunning_simulation() {
        let mut source = populated_sim();
        let mut replay = source.clone();
        let report = source.step(TickBudget::new(64, 64, 16)).unwrap();

        replay.apply_delta(&report.delta).unwrap();

        assert_eq!(replay, source);
        assert_eq!(replay.current_tick(), report.delta.tick);
    }

    #[test]
    fn invalid_delta_is_rejected_before_any_state_mutation() {
        let mut sim = populated_sim();
        let before = sim.clone();
        let delta = SimulationDelta {
            tick: 1,
            flow_deltas: alloc::vec![FlowDelta {
                from: CellCoord::new(0, 0, 0),
                to: CellCoord::new(1, 0, 0),
                amount: 101,
            }],
            ..SimulationDelta::default()
        };

        assert!(matches!(
            sim.apply_delta(&delta),
            Err(SimulationApplyError::FlowTransitionMismatch { .. })
                | Err(SimulationApplyError::WaterCapacityExceeded { .. })
                | Err(SimulationApplyError::WaterUnderflow { .. })
        ));
        assert_eq!(sim, before);
    }

    #[test]
    fn replay_rejects_ecology_metadata_or_transition_tampering() {
        let mut source = populated_sim();
        let mut replay = source.clone();
        let report = source.step(TickBudget::new(64, 64, 16)).unwrap();
        let mut delta = report.delta.clone();
        let original_stress = delta
            .ecology_deltas
            .first()
            .expect("fixture must emit an ecology delta")
            .stress;
        delta.ecology_deltas[0].stress = original_stress.wrapping_add(1);

        let before = replay.clone();
        assert!(matches!(
            replay.apply_delta(&delta),
            Err(SimulationApplyError::EcologyStressMismatch { .. })
        ));
        assert_eq!(replay, before);

        let ecology = &mut delta.ecology_deltas[0];
        ecology.stress = report.delta.ecology_deltas[0].stress;
        ecology.after_biomass = ecology.after_biomass.saturating_add(1);
        assert!(matches!(
            replay.apply_delta(&delta),
            Err(SimulationApplyError::EcologyTransitionMismatch { .. })
        ));
        assert_eq!(replay, before);
    }

    #[test]
    fn replay_rejects_flow_and_erosion_tampering_without_mutation() {
        let mut source = populated_sim();
        let mut replay = source.clone();
        let report = source.step(TickBudget::new(64, 64, 16)).unwrap();

        let mut flow_tampered = report.delta.clone();
        flow_tampered.flow_deltas[0].amount -= 1;
        let before = replay.clone();
        assert!(matches!(
            replay.apply_delta(&flow_tampered),
            Err(SimulationApplyError::FlowTransitionMismatch { .. })
        ));
        assert_eq!(replay, before);

        let mut erosion_tampered = report.delta.clone();
        erosion_tampered.voxel_deltas[0].removed_mass += 1;
        erosion_tampered.voxel_deltas[0].after_mass -= 1;
        assert!(matches!(
            replay.apply_delta(&erosion_tampered),
            Err(SimulationApplyError::VoxelRemovalMismatch { .. })
        ));
        assert_eq!(replay, before);
    }

    #[test]
    fn replay_accepts_ecology_without_a_hydrology_cell() {
        let mut source = WorldSim::new(WorldSimConfig::default()).unwrap();
        let coord = CellCoord::new(0, 0, 0);
        source
            .ecology
            .set(coord, EcologyCell::new(4, 65_535, 100).unwrap());
        let mut replay = source.clone();
        let report = source.step(TickBudget::new(64, 64, 16)).unwrap();

        replay.apply_delta(&report.delta).unwrap();
        assert_eq!(replay, source);
    }
}
