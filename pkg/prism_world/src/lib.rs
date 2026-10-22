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
#[cfg(test)]
extern crate std;

#[cfg(feature = "scene_host")]
pub mod scene_host;

use prism_world_persist::{AppendOnlyLog, LogError, Transaction};
use prism_world_residency::{
    MaterializationBudget, ResidencyPlan, ResidencyRecord, ResidencyScheduler,
};
use prism_world_stream::{
    validate_manifests, CellManifest, DemandSource, ManifestSpatialIndex, ManifestValidationError,
    SectorId, StreamAction, StreamBudget, StreamPlan, StreamScheduler,
};

#[cfg(feature = "simulation")]
use prism_world_persist::{ActorId, LogicalClock};
#[cfg(feature = "simulation")]
use prism_world_sim::{simulation_delta_to_transaction, SimulationDelta, SimulationPersistError};

#[cfg(feature = "scene_host")]
pub use scene_host::{
    CellPersistenceState, ResidencyBinding, SceneEcsHost, SceneEcsHostConfigError,
    SceneEcsHostError, SceneEcsProvider,
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

/// Invalid frame input rejected before either policy scheduler advances.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorldTickError {
    /// The manifest slice contains duplicate sector identities.
    DuplicateManifest(ManifestValidationError),
}

impl core::fmt::Display for WorldTickError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DuplicateManifest(error) => write!(formatter, "{error}"),
        }
    }
}

impl core::error::Error for WorldTickError {}

impl From<ManifestValidationError> for WorldTickError {
    fn from(error: ManifestValidationError) -> Self {
        Self::DuplicateManifest(error)
    }
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
    pub fn try_tick(
        &mut self,
        demands: &[DemandSource],
        manifests: &[CellManifest],
        records: &[ResidencyRecord],
        frame: u64,
        budget: WorldBudget,
    ) -> Result<WorldFramePlan, WorldTickError> {
        validate_manifests(manifests)?;
        Ok(WorldFramePlan {
            stream: self
                .stream
                .try_plan(demands, manifests, budget.stream)
                .expect("manifest validation completed before stream planning"),
            residency: self.residency.plan(frame, records, budget.materialization),
        })
    }

    /// Plans a frame using a prebuilt hierarchical manifest index.
    ///
    /// The index must represent the immutable manifest snapshot used by the
    /// caller. The policy output and residency plan are otherwise the same as
    /// the linear planner, while load/prefetch candidate generation avoids
    /// scanning every manifest for every demand source.
    pub fn try_tick_indexed(
        &mut self,
        demands: &[DemandSource],
        index: &ManifestSpatialIndex,
        records: &[ResidencyRecord],
        frame: u64,
        budget: WorldBudget,
    ) -> WorldFramePlan {
        WorldFramePlan {
            stream: self.stream.try_plan_indexed(demands, index, budget.stream),
            residency: self.residency.plan(frame, records, budget.materialization),
        }
    }

    /// Plans a frame and panics on duplicate manifest input.
    ///
    /// New callers that receive external or dynamically assembled manifests
    /// should use [`Self::try_tick`] to propagate validation errors.
    pub fn tick(
        &mut self,
        demands: &[DemandSource],
        manifests: &[CellManifest],
        records: &[ResidencyRecord],
        frame: u64,
        budget: WorldBudget,
    ) -> WorldFramePlan {
        self.try_tick(demands, manifests, records, frame, budget)
            .expect("world manifests must contain unique sectors")
    }

    /// Applies a plan in deterministic stream-then-tier order.
    pub fn apply_plan<H: SceneHost>(
        &mut self,
        plan: &WorldFramePlan,
        host: &mut H,
    ) -> Result<(), ApplyError<H::Error>> {
        apply_plan_to(plan, host)
    }

    /// Commits an IO completion without mutating scene truth.
    pub fn mark_resident(&mut self, sector: SectorId) {
        self.stream.mark_resident(sector);
    }

    /// Marks a sector resident with the exact manifest revision committed by
    /// its host, enabling later revision changes to be distinguished as reloads.
    pub fn mark_resident_manifest(&mut self, manifest: CellManifest) {
        self.stream.mark_resident_manifest(manifest);
    }

    /// Requests a budgeted reload while keeping the resident revision intact
    /// until a future host transaction succeeds.
    pub fn request_reload(&mut self, sector: SectorId) -> bool {
        self.stream.request_reload(sector)
    }

    /// Restores the last committed resident revision after replacement failure.
    pub fn restore_resident_manifest(&mut self, manifest: CellManifest) {
        self.stream.restore_resident_manifest(manifest);
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

pub(crate) fn apply_plan_to<H: SceneHost>(
    plan: &WorldFramePlan,
    host: &mut H,
) -> Result<(), ApplyError<H::Error>> {
    for action in &plan.stream.actions {
        match action {
            StreamAction::Load(request)
            | StreamAction::Reload(request)
            | StreamAction::Prefetch(request) => {
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
            data_layers: prism_world_stream::DataLayerMask::EMPTY,
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
    fn indexed_facade_matches_linear_stream_plan() {
        let policy = ResidencyPolicy::new(1, 2, 3, 4, 0).unwrap();
        let sector = SectorId::new(0, 0, 0, 0);
        let manifest = CellManifest {
            sector,
            bounds: AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(1, 1, 1)).unwrap(),
            scene_key: 1,
            content_version: 1,
            derived_hash: 2,
            io_bytes: 4,
            data_layers: prism_world_stream::DataLayerMask::EMPTY,
        };
        let demand = DemandSource::new(sector, 0, 1, 10).unwrap();
        let budget = WorldBudget {
            stream: StreamBudget {
                io_bytes: 8,
                max_actions: 2,
            },
            materialization: MaterializationBudget::new(2, 2),
        };
        let index = ManifestSpatialIndex::build(&[manifest]).unwrap();
        let mut linear = WorldRuntime::new(ResidencyScheduler::new(policy));
        let mut indexed = WorldRuntime::new(ResidencyScheduler::new(policy));
        let linear_plan = linear
            .try_tick(&[demand], &[manifest], &[], 1, budget)
            .unwrap();
        let indexed_plan = indexed.try_tick_indexed(&[demand], &index, &[], 1, budget);
        assert_eq!(indexed_plan, linear_plan);
    }

    #[test]
    fn duplicate_manifests_fail_before_stream_or_residency_advances() {
        let policy = ResidencyPolicy::new(1, 2, 3, 4, 0).unwrap();
        let mut world = WorldRuntime::new(ResidencyScheduler::new(policy));
        let sector = SectorId::new(0, 0, 0, 0);
        let bounds = AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(1, 1, 1)).unwrap();
        let first = CellManifest {
            sector,
            bounds,
            scene_key: 1,
            content_version: 1,
            derived_hash: 2,
            io_bytes: 4,
            data_layers: prism_world_stream::DataLayerMask::EMPTY,
        };
        let duplicate = CellManifest {
            scene_key: 2,
            io_bytes: 64,
            ..first
        };
        let demand = DemandSource::new(sector, 0, 1, 10).unwrap();
        let error = world
            .try_tick(
                &[demand],
                &[first, duplicate],
                &[],
                1,
                WorldBudget {
                    stream: StreamBudget {
                        io_bytes: 64,
                        max_actions: 2,
                    },
                    materialization: MaterializationBudget::new(2, 2),
                },
            )
            .unwrap_err();
        assert_eq!(
            error,
            WorldTickError::DuplicateManifest(ManifestValidationError::DuplicateSector {
                sector,
                first_index: 0,
                duplicate_index: 1,
            })
        );
        assert_eq!(world.stream_frame(), 0);
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
                data_layers: prism_world_stream::DataLayerMask::EMPTY,
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
