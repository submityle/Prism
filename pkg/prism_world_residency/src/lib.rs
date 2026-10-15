//! Deterministic world-residency primitives for Prism's world system.
//!
//! The crate is the allocation-aware core for the M2 residency/materialization,
//! M4 HLOD, and M7 sparse-SDF milestones. It deliberately owns only policy and
//! derived data: scene truth, rendering, ECS storage, persistence, and GPU
//! submission remain outside this crate.
//!
//! All public decisions use integer or fixed-point values so a recorded input
//! stream can be replayed consistently without a platform floating-point
//! ordering dependency. The crate has no operating-system or engine-runtime
//! dependency and can be used from a `no_std` target with `alloc`.
#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

extern crate alloc;

mod hlod;
mod pcg;
mod qos;
mod residency;
mod scheduler;
mod sdf;

pub use hlod::{HlodCluster, HlodError, HlodLevel, LodTransition};
pub use pcg::{
    derive_cache_key, pcg_hash_u32, DependencySet, DerivedCacheKey, DirtySector, NodeId, NodeInput,
    NodeKind, Pcg32, PcgEvaluation, PcgGraph, PcgGraphError, PcgNode, SectorCoord,
};
pub use qos::{QosConfig, QosDecision, QosGovernor, QualityTier};
pub use residency::{
    MaterializationBudget, PolicyError, ResidencyPolicy, ResidencyRecord, ResidencyState,
    ResidencyTier,
};
pub use scheduler::{ResidencyAction, ResidencyKeepReason, ResidencyPlan, ResidencyScheduler};
pub use sdf::{BrickCoord, SdfError, SparseSdfBrickGrid};

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn residency_state_machine_allows_direct_materialization_penetration() {
        let states = [
            ResidencyState::TierA,
            ResidencyState::TierB,
            ResidencyState::TierC,
        ];
        for from in states {
            for to in states {
                assert!(from.can_transition_to(to));
                assert_eq!(from.transition_to(to), Some(to));
            }
        }
        assert!(ResidencyState::TierC.is_upgrade_to(ResidencyState::TierA));
        assert!(ResidencyState::TierA.is_downgrade_to(ResidencyState::TierC));
        assert_eq!(ResidencyTier::A.detail_rank(), 2);
        assert_eq!(ResidencyTier::C.detail_rank(), 0);
    }

    #[test]
    fn budget_reserves_both_dimensions_without_overflow() {
        let mut budget = MaterializationBudget::new(10, 100);
        assert!(budget.try_reserve(4, 40));
        assert!(!budget.try_reserve(7, 1));
        assert!(!budget.try_reserve(1, 61));
        assert_eq!(budget.used(), (4, 40));
        assert_eq!(budget.remaining(), (6, 60));
    }

    #[test]
    fn scheduler_applies_hysteresis_budget_and_deterministic_order() {
        let policy = ResidencyPolicy::new(10, 20, 50, 70, 3).unwrap();
        let scheduler = ResidencyScheduler::new(policy);
        let records = [
            ResidencyRecord::new(30, ResidencyState::TierC, 5, 1, 4, 40),
            ResidencyRecord::new(10, ResidencyState::TierC, 5, 9, 4, 40),
            ResidencyRecord::new(20, ResidencyState::TierA, 100, 2, 1, 10),
            ResidencyRecord::new(40, ResidencyState::TierB, 5, 5, 1, 10)
                .with_last_transition_frame(9),
        ];
        let plan = scheduler.plan(10, &records, MaterializationBudget::new(4, 40));
        assert_eq!(plan.actions.len(), 4);
        assert_eq!(plan.actions[0].id(), 10);
        assert!(matches!(
            plan.actions[0],
            ResidencyAction::Promote {
                from: ResidencyState::TierC,
                to: ResidencyState::TierA,
                ..
            }
        ));
        assert!(matches!(
            plan.actions[1],
            ResidencyAction::Keep {
                id: 30,
                reason: ResidencyKeepReason::Budget,
                ..
            }
        ));
        assert!(matches!(
            plan.actions[2],
            ResidencyAction::Demote {
                id: 20,
                from: ResidencyState::TierA,
                to: ResidencyState::TierC,
            }
        ));
        assert!(matches!(
            plan.actions[3],
            ResidencyAction::Keep {
                id: 40,
                reason: ResidencyKeepReason::MinimumResidence,
                ..
            }
        ));
        assert_eq!(plan.budget.used(), (4, 40));
    }

    #[test]
    fn hlod_alpha_is_continuous_and_dither_is_stable() {
        let cluster = HlodCluster::new(
            7,
            vec![11, 12],
            vec![
                HlodLevel {
                    asset_key: 1,
                    enter_distance: 0,
                    transition_width: 10,
                },
                HlodLevel {
                    asset_key: 2,
                    enter_distance: 100,
                    transition_width: 10,
                },
            ],
        )
        .unwrap();
        assert_eq!(cluster.transition_alpha(90, 0, 1).unwrap(), 0);
        assert!(cluster.transition_alpha(100, 0, 1).unwrap() > 0);
        assert!(cluster.transition_alpha(110, 0, 1).unwrap() >= 65_534);
        assert_eq!(
            cluster.transition_alpha(100, 1, 0).unwrap(),
            65_535 - cluster.transition_alpha(100, 0, 1).unwrap()
        );
        assert_eq!(cluster.dither_seed(0, 1), cluster.dither_seed(0, 1));
        assert_eq!(cluster.transition_target(105, 0).unwrap(), 0);
        assert_eq!(cluster.transition_target(111, 0).unwrap(), 1);
    }

    #[test]
    fn qos_downshifts_after_lag_and_does_not_ping_pong() {
        let config = QosConfig::new(100, 10, 10, 2, 2, 2).unwrap();
        let mut governor = QosGovernor::new(config, QualityTier::High);
        assert_eq!(governor.observe(111).current, QualityTier::High);
        assert_eq!(governor.observe(111).current, QualityTier::Medium);
        assert_eq!(governor.observe(1).current, QualityTier::Medium);
        assert_eq!(governor.observe(1).current, QualityTier::Medium);
        assert_eq!(governor.observe(1).current, QualityTier::Medium);
        assert_eq!(governor.observe(1).current, QualityTier::High);
        assert_eq!(governor.tier().work_rate_milli(), 1_000);
    }

    #[test]
    fn sparse_sdf_handles_negative_coordinates_dirty_regions_and_neighbors() {
        let mut grid = SparseSdfBrickGrid::new(2, 2).unwrap();
        let mut samples = vec![0_i16; 8];
        samples[0] = -7;
        grid.insert_brick(BrickCoord::new(-1, 0, 0, 0), samples)
            .unwrap();
        assert_eq!(grid.sample([-2, 0, 0], 0).unwrap(), Some(-7));
        assert_eq!(grid.sample([0, 0, 0], 0).unwrap(), None);
        grid.insert_brick(BrickCoord::new(0, 0, 0, 0), vec![1; 8])
            .unwrap();
        grid.insert_brick(BrickCoord::new(1, 0, 0, 0), vec![2; 8])
            .unwrap();
        assert_eq!(
            grid.mip_neighborhood(BrickCoord::new(0, 0, 0, 0), 1)
                .unwrap(),
            vec![
                BrickCoord::new(-1, 0, 0, 0),
                BrickCoord::new(0, 0, 0, 0),
                BrickCoord::new(1, 0, 0, 0),
            ]
        );
        assert_eq!(
            grid.mark_dirty_region([-1, -1, -1], [2, 1, 1], 0).unwrap(),
            12
        );
        assert_eq!(grid.take_dirty().len(), 12);
        assert_eq!(grid.dirty_count(), 0);
    }

    #[test]
    fn pcg_and_derived_cache_keys_are_reproducible() {
        let mut first = Pcg32::new(42, 9);
        let mut second = Pcg32::new(42, 9);
        for _ in 0..16 {
            assert_eq!(first.next_u32(), second.next_u32());
        }
        assert_eq!(pcg_hash_u32(99), pcg_hash_u32(99));
        assert_ne!(pcg_hash_u32(99), pcg_hash_u32(100));
        let sector = SectorCoord::new(0, 0, 0);
        assert_eq!(
            derive_cache_key(1, 2, 3, sector, 4),
            derive_cache_key(1, 2, 3, sector, 4)
        );
        assert_ne!(
            derive_cache_key(1, 2, 3, sector, 4),
            derive_cache_key(1, 2, 4, sector, 4)
        );
    }
}
