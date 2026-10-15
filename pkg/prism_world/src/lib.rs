//! Prism world-system facade.
//!
//! [`WorldRuntime`] composes the three policy cores without taking ownership
//! of scene truth, ECS storage, IO, GPU resources, or network authority. A
//! host applies the returned frame plan through its scene/render providers and
//! appends authoritative override transactions through [`WorldRuntime::append`].
#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

extern crate alloc;

#[cfg(feature = "scene_host")]
pub mod scene_host;

use prism_world_persist::{AppendOnlyLog, LogError, Transaction};
use prism_world_residency::{
    MaterializationBudget, ResidencyPlan, ResidencyRecord, ResidencyScheduler,
};
use prism_world_stream::{
    CellManifest, DemandSource, SectorId, StreamAction, StreamBudget, StreamPlan, StreamScheduler,
};

#[cfg(feature = "simulation")]
use prism_world_persist::{ActorId, LogicalClock};
#[cfg(feature = "simulation")]
use prism_world_sim::{simulation_delta_to_transaction, SimulationDelta, SimulationPersistError};

#[cfg(feature = "scene_host")]
pub use scene_host::{
    ResidencyBinding, SceneEcsHost, SceneEcsHostConfigError, SceneEcsHostError, SceneEcsProvider,
};

/// Per-frame budgets for the two hot policy paths.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorldBudget {
    /// Streaming IO/action budget.
    pub stream: StreamBudget,
    /// Residency materialization budget.
    pub materialization: MaterializationBudget,
}

/// The deterministic policy result for one world frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldFramePlan {
    /// Cell load/prefetch/unload operations.
    pub stream: StreamPlan,
    /// Tier promotion/demotion/keep operations.
    pub residency: ResidencyPlan,
}

/// Host boundary used to apply a policy plan to the authoritative scene.
pub trait SceneHost {
    /// Host-specific failure type.
    type Error;

    /// Loads the authoritative cell scene for a required or prefetched sector.
    fn load_cell(&mut self, sector: SectorId) -> Result<(), Self::Error>;

    /// Removes a cell scene after its persistence handoff is complete.
    fn unload_cell(&mut self, sector: SectorId) -> Result<(), Self::Error>;

    /// Promotes one stable item into its interactive representation.
    fn promote(&mut self, id: u64) -> Result<(), Self::Error>;

    /// Demotes one stable item into its lower-cost representation.
    fn demote(&mut self, id: u64) -> Result<(), Self::Error>;
}

/// Failure while applying a generated plan to a scene host.
#[derive(Debug, Eq, PartialEq)]
pub enum ApplyError<E> {
    /// The host rejected an operation.
    Host(E),
}

/// Failure while converting and appending a simulation result.
#[cfg(feature = "simulation")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SimulationAppendError {
    /// The simulation delta could not be encoded as a persistence transaction.
    Conversion(SimulationPersistError),
    /// The world log rejected the converted transaction.
    Persistence(LogError),
}

/// World policy runtime with one owner for each responsibility.
pub struct WorldRuntime {
    stream: StreamScheduler,
    residency: ResidencyScheduler,
    persist: AppendOnlyLog,
}

impl WorldRuntime {
    /// Creates a runtime from a residency policy.
    pub const fn new(residency: ResidencyScheduler) -> Self {
        Self {
            stream: StreamScheduler::new(),
            residency,
            persist: AppendOnlyLog::new(),
        }
    }

    /// Plans streaming and residency independently but atomically as one frame.
    pub fn tick(
        &mut self,
        demands: &[DemandSource],
        manifests: &[CellManifest],
        records: &[ResidencyRecord],
        frame: u64,
        budget: WorldBudget,
    ) -> WorldFramePlan {
        WorldFramePlan {
            stream: self.stream.plan(demands, manifests, budget.stream),
            residency: self.residency.plan(frame, records, budget.materialization),
        }
    }

    /// Applies a plan in deterministic stream-then-tier order.
    pub fn apply_plan<H: SceneHost>(
        &mut self,
        plan: &WorldFramePlan,
        host: &mut H,
    ) -> Result<(), ApplyError<H::Error>> {
        for action in &plan.stream.actions {
            match action {
                StreamAction::Load(request) | StreamAction::Prefetch(request) => {
                    host.load_cell(request.sector).map_err(ApplyError::Host)?;
                }
                StreamAction::Unload(sector) => {
                    host.unload_cell(*sector).map_err(ApplyError::Host)?;
                }
            }
        }
        for action in &plan.residency.actions {
            match action {
                prism_world_residency::ResidencyAction::Promote { id, .. } => {
                    host.promote(*id).map_err(ApplyError::Host)?;
                }
                prism_world_residency::ResidencyAction::Demote { id, .. } => {
                    host.demote(*id).map_err(ApplyError::Host)?;
                }
                prism_world_residency::ResidencyAction::Keep { .. } => {}
            }
        }
        Ok(())
    }

    /// Commits an IO completion without mutating scene truth.
    pub fn mark_resident(&mut self, sector: SectorId) {
        self.stream.mark_resident(sector);
    }

    /// Commits an eviction completion.
    pub fn mark_unloaded(&mut self, sector: SectorId) {
        self.stream.mark_unloaded(sector);
    }

    /// Appends one validated persistent override transaction.
    pub fn append(&mut self, transaction: Transaction) -> Result<(), LogError> {
        self.persist.append(transaction)
    }

    /// Converts one simulation delta into the world persistence format and
    /// appends it at the log's next sequence number.
    #[cfg(feature = "simulation")]
    pub fn append_simulation_delta(
        &mut self,
        transaction_id: prism_world_persist::TransactionId,
        actor: ActorId,
        clock: LogicalClock,
        object: prism_world_persist::PersistentId,
        delta: &SimulationDelta,
    ) -> Result<(), SimulationAppendError> {
        let transaction = simulation_delta_to_transaction(
            transaction_id,
            actor,
            clock,
            self.persist.next_sequence(),
            object,
            delta,
        )
        .map_err(SimulationAppendError::Conversion)?;
        self.persist
            .append(transaction)
            .map_err(SimulationAppendError::Persistence)
    }

    /// Returns the append-only persistence log.
    pub const fn persistence(&self) -> &AppendOnlyLog {
        &self.persist
    }

    /// Returns the current stream scheduler frame.
    pub const fn stream_frame(&self) -> u64 {
        self.stream.frame()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_world_residency::{ResidencyPolicy, ResidencyState};
    use prism_world_stream::{AabbI64, GlobalPos, SectorId};

    #[test]
    fn facade_runs_stream_and_residency_from_one_tick() {
        let policy = ResidencyPolicy::new(1, 2, 3, 4, 0).unwrap();
        let mut world = WorldRuntime::new(ResidencyScheduler::new(policy));
        let sector = SectorId::new(0, 0, 0, 0);
        let manifest = CellManifest {
            sector,
            bounds: AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(1, 1, 1)).unwrap(),
            scene_key: 1,
            content_version: 1,
            derived_hash: 2,
            io_bytes: 4,
        };
        let demand = DemandSource::new(sector, 0, 1, 10).unwrap();
        let record = ResidencyRecord::new(1, ResidencyState::TierC, 0, 1, 1, 1);
        let plan = world.tick(
            &[demand],
            &[manifest],
            &[record],
            1,
            WorldBudget {
                stream: StreamBudget {
                    io_bytes: 8,
                    max_actions: 2,
                },
                materialization: MaterializationBudget::new(2, 2),
            },
        );
        assert_eq!(plan.stream.admitted_bytes, 4);
        assert_eq!(plan.residency.actions.len(), 1);
        assert_eq!(world.stream_frame(), 1);
    }

    #[test]
    fn facade_applies_plan_through_scene_boundary() {
        struct Host {
            calls: alloc::vec::Vec<u64>,
        }

        impl SceneHost for Host {
            type Error = ();

            fn load_cell(&mut self, sector: SectorId) -> Result<(), Self::Error> {
                self.calls.push(sector.x as u64);
                Ok(())
            }

            fn unload_cell(&mut self, sector: SectorId) -> Result<(), Self::Error> {
                self.calls.push(100 + sector.x as u64);
                Ok(())
            }

            fn promote(&mut self, id: u64) -> Result<(), Self::Error> {
                self.calls.push(1_000 + id);
                Ok(())
            }

            fn demote(&mut self, id: u64) -> Result<(), Self::Error> {
                self.calls.push(2_000 + id);
                Ok(())
            }
        }

        let policy = ResidencyPolicy::new(1, 2, 3, 4, 0).unwrap();
        let mut world = WorldRuntime::new(ResidencyScheduler::new(policy));
        let sector = SectorId::new(0, 0, 0, 0);
        let bounds = AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(1, 1, 1)).unwrap();
        let plan = world.tick(
            &[DemandSource::new(sector, 0, 1, 10).unwrap()],
            &[CellManifest {
                sector,
                bounds,
                scene_key: 1,
                content_version: 1,
                derived_hash: 2,
                io_bytes: 1,
            }],
            &[ResidencyRecord::new(7, ResidencyState::TierC, 0, 1, 1, 1)],
            1,
            WorldBudget {
                stream: StreamBudget {
                    io_bytes: 2,
                    max_actions: 2,
                },
                materialization: MaterializationBudget::new(2, 2),
            },
        );
        let mut host = Host {
            calls: alloc::vec::Vec::new(),
        };
        world.apply_plan(&plan, &mut host).unwrap();
        assert_eq!(host.calls, alloc::vec![0, 1_007]);
    }

    #[cfg(feature = "simulation")]
    #[test]
    fn simulation_delta_enters_the_world_persistence_log() {
        let policy = ResidencyPolicy::new(1, 2, 3, 4, 0).unwrap();
        let mut world = WorldRuntime::new(ResidencyScheduler::new(policy));
        world
            .append_simulation_delta(
                prism_world_persist::TransactionId(10),
                ActorId(11),
                LogicalClock(12),
                prism_world_persist::PersistentId(13),
                &SimulationDelta::default(),
            )
            .unwrap();
        assert_eq!(world.persistence().len(), 1);
        let transaction = world.persistence().get(0).unwrap();
        assert_eq!(transaction.sequence(), 0);
        assert_eq!(transaction.object(), prism_world_persist::PersistentId(13));
        assert_eq!(
            prism_world_sim::transaction_to_simulation_delta(transaction).unwrap(),
            SimulationDelta::default()
        );
    }
}
