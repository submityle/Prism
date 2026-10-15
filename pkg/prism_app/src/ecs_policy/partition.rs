//! Host-driven consumption of the existing ECS partition implementations.

use alloc::sync::Arc;
use alloc::vec::Vec;
use hashbrown::HashMap;
use prism_ecs::entity::Entity;
use prism_ecs::partition::cell::{CellCoord, CellState};
use prism_ecs::partition::dormant::DormancySet;
use prism_ecs::partition::driver::StreamResult;
use prism_ecs::partition::floating_origin::WorldPos;
use prism_ecs::partition::lod::{distance_sq, LodDecision};
use prism_ecs::partition::processor::{EntityLodProcessor, LodTickResult, PhasePolicy};
use prism_ecs::partition::world_partition::WorldPartition;
use prism_ecs::policy::{EcsPolicy, StreamingProfile};
use prism_ecs::world::{World, WorldId};

use super::{AppliedEcsPolicy, EcsPolicyError};
use crate::App;

// The existing CellCoord arithmetic uses signed i32 differences. Keep both
// interests AND resident cells inside this interval so subtraction/abs cannot
// overflow, and reserve the load radius before enumerating each load cube.
const CELL_BOUND: i32 = i32::MAX / 2;

/// An opaque completion ticket for one cell load in one main World.
///
/// Clones identify the same request. A cell evicted and subsequently requested
/// again receives a new identity, even at the same coordinates. There is no
/// wrapping counter or reliance on entity-handle provenance.
#[derive(Clone, Debug)]
pub struct EcsCellLoad {
    cell: CellCoord,
    world: WorldId,
    identity: Arc<()>,
}

impl EcsCellLoad {
    /// Requested streaming-grid coordinate.
    pub fn cell(&self) -> CellCoord {
        self.cell
    }

    /// The authoritative main World that issued this request.
    pub fn world_id(&self) -> WorldId {
        self.world
    }
}

/// Loads to satisfy and evictions already applied to the main World.
#[derive(Clone, Debug, Default)]
pub struct EcsStreamingResult {
    /// All still-pending tickets, sorted by cell coordinate. Repeated results
    /// carry the same ticket until settled or evicted, not duplicate I/O work.
    pub to_load: Vec<EcsCellLoad>,
    /// Cells evicted by this operation, sorted by coordinate.
    pub unloaded_cells: Vec<CellCoord>,
    /// Live entities actually despawned, in deterministic cell/handle order.
    pub despawned: Vec<Entity>,
    /// Registered entities already absent from the World.
    pub already_gone: Vec<Entity>,
}

pub(super) struct PolicyPartition {
    lod: Option<EntityLodProcessor>,
    dormancy: DormancySet,
    phase: PhasePolicy,
    streaming: Option<WorldPartition>,
    pending: HashMap<CellCoord, Arc<()>>,
}

pub(super) fn validate_reconfiguration(
    old: Option<&AppliedEcsPolicy>,
    policy: &EcsPolicy,
) -> Result<(), EcsPolicyError> {
    if policy
        .profile()
        .streaming
        .is_some_and(|s| s.load_radius > CELL_BOUND as u32)
    {
        return Err(EcsPolicyError::InvalidPosition);
    }
    if let Some(old) = old
        && old.policy.profile().streaming != policy.profile().streaming
        && old
            .partition
            .streaming
            .as_ref()
            .is_some_and(|p| p.driver().streamer().tracked_count() != 0 || !p.index().is_empty())
    {
        return Err(EcsPolicyError::StreamingBusy);
    }
    Ok(())
}

impl PolicyPartition {
    pub(super) fn reconfigure(old: Option<AppliedEcsPolicy>, policy: &EcsPolicy) -> Self {
        if let Some(old) = old {
            let mut runtime = old.partition;
            if old.policy.profile().lod != policy.profile().lod {
                runtime.lod = policy.lod_processor(runtime.phase);
                runtime.dormancy = DormancySet::new();
            }
            if old.policy.profile().streaming != policy.profile().streaming {
                runtime.streaming = policy.world_partition();
                runtime.pending.clear();
            }
            runtime
        } else {
            Self {
                lod: policy.lod_processor(PhasePolicy::Synchronized),
                dormancy: DormancySet::new(),
                phase: PhasePolicy::Synchronized,
                streaming: policy.world_partition(),
                pending: HashMap::new(),
            }
        }
    }

    fn finish_stream(&mut self, world: WorldId, result: StreamResult) -> EcsStreamingResult {
        for cell in &result.unloaded_cells {
            self.pending.remove(cell);
        }
        for cell in result.to_load {
            self.pending.entry(cell).or_insert_with(|| Arc::new(()));
        }
        let mut to_load: Vec<_> = self
            .pending
            .iter()
            .map(|(&cell, identity)| EcsCellLoad {
                cell,
                world,
                identity: identity.clone(),
            })
            .collect();
        to_load.sort_unstable_by_key(|ticket| {
            let c = ticket.cell;
            (c.x, c.y, c.z)
        });
        EcsStreamingResult {
            to_load,
            unloaded_cells: result.unloaded_cells,
            despawned: result.despawned,
            already_gone: result.already_gone,
        }
    }
}

fn validate_population(world: &World, entities: &[Entity]) -> Result<(), EcsPolicyError> {
    let mut sorted = Vec::with_capacity(entities.len());
    for &entity in entities {
        if !world.contains(entity) {
            return Err(EcsPolicyError::EntityNotLive(entity));
        }
        sorted.push(entity);
    }
    sorted.sort_unstable_by_key(|e| e.to_bits());
    if let Some(pair) = sorted.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(EcsPolicyError::DuplicateEntity(pair[0]));
    }
    Ok(())
}

fn validate_position(profile: StreamingProfile, position: WorldPos) -> Result<(), EcsPolicyError> {
    let lo = -f64::from(CELL_BOUND) + f64::from(profile.load_radius);
    let hi = f64::from(CELL_BOUND) - f64::from(profile.load_radius) + 1.0;
    for axis in [position.x, position.y, position.z] {
        let q = axis / profile.cell_size;
        if !q.is_finite() || q < lo || q >= hi {
            return Err(EcsPolicyError::InvalidPosition);
        }
    }
    Ok(())
}

impl App {
    /// Select deterministic synchronized or handle-distributed LOD cadence.
    pub fn set_ecs_lod_phase(&mut self, phase: PhasePolicy) -> Result<&mut Self, EcsPolicyError> {
        self.applied_ecs_policy()?;
        let runtime = &mut self
            .ecs_policy
            .as_mut()
            .expect("validated policy")
            .partition;
        let lod = runtime.lod.as_ref().ok_or(EcsPolicyError::LodDisabled)?;
        runtime.lod = Some(lod.clone().with_phase_policy(phase));
        runtime.phase = phase;
        Ok(self)
    }

    /// Whether the explicit LOD driver currently considers an entity dormant.
    /// This is not a global query/schedule filter and does not add ECS tags.
    pub fn is_ecs_dormant(&self, entity: Entity) -> Result<bool, EcsPolicyError> {
        let applied = self.applied_ecs_policy()?;
        if applied.partition.lod.is_none() {
            return Err(EcsPolicyError::LodDisabled);
        }
        Ok(self.world().contains(entity) && applied.partition.dormancy.is_dormant(entity))
    }

    /// Execute project work only for live entities selected by the policy LOD.
    ///
    /// The host supplies positions in a common local coordinate frame and an
    /// explicit frame number; repeated calls intentionally execute repeated work.
    /// Samples must be unique live handles in the main World (raw Entity values
    /// themselves carry no World provenance). Invalid input changes no dormancy
    /// state and invokes no callbacks. Missing samples retain their sleep state;
    /// dead sleep entries are pruned on the next valid call.
    ///
    /// Callbacks run in sorted handle order and receive band/quality decisions.
    /// A callback may despawn a later entity; that entity is skipped. The returned
    /// `to_tick` is the evaluated selection, not a callback completion receipt.
    /// This does not suppress unrelated App schedules; host-owned work must be
    /// routed through this driver to receive LOD cadence. Callback panic propagates;
    /// already completed work and the evaluated dormancy transitions remain valid.
    pub fn drive_ecs_lod(
        &mut self,
        viewpoint: [f32; 3],
        population: &[(Entity, [f32; 3])],
        frame: u64,
        mut tick: impl FnMut(&mut World, Entity, LodDecision),
    ) -> Result<LodTickResult, EcsPolicyError> {
        let applied = self.applied_ecs_policy()?;
        if applied.partition.lod.is_none() {
            return Err(EcsPolicyError::LodDisabled);
        }
        if !viewpoint.iter().all(|v| v.is_finite())
            || population.iter().any(|(_, position)| {
                !position.iter().all(|v| v.is_finite())
                    || !distance_sq(viewpoint, *position).is_finite()
            })
        {
            return Err(EcsPolicyError::InvalidPosition);
        }
        validate_population(
            self.world(),
            &population.iter().map(|(e, _)| *e).collect::<Vec<_>>(),
        )?;
        let (state, world) = self.ecs_policy_parts();
        let runtime = &mut state.as_mut().expect("validated policy").partition;
        let dead: Vec<_> = runtime
            .dormancy
            .iter()
            .filter(|&e| !world.contains(e))
            .collect();
        for entity in dead {
            runtime.dormancy.wake(entity);
        }
        let result = runtime.lod.as_ref().expect("validated LOD").drive(
            viewpoint,
            population,
            frame,
            &mut runtime.dormancy,
        );
        // The result already owns this pass's wake events; do not accumulate a
        // second, unconsumed log in the persistent DormancySet.
        runtime.dormancy.drain_woken();
        let mut decisions = result.decisions.clone();
        decisions.sort_unstable_by_key(|(e, _)| e.to_bits());
        for (entity, decision) in decisions {
            if decision.tick_this_frame && world.contains(entity) {
                tick(world, entity, decision);
            }
        }
        Ok(result)
    }

    /// Inspect the real partition without bypassing completion-ticket validation.
    pub fn ecs_world_partition(&self) -> Result<&WorldPartition, EcsPolicyError> {
        self.applied_ecs_policy()?
            .partition
            .streaming
            .as_ref()
            .ok_or(EcsPolicyError::StreamingDisabled)
    }

    /// Advance real streaming and immediately despawn entities in evicted cells.
    ///
    /// All positions are validated before any mutation. The adapter restricts
    /// cell coordinates to `[-i32::MAX/2, i32::MAX/2]`, reserving the load radius
    /// at both edges; this avoids saturation and overflow in the existing grid.
    /// Extra interests and hysteresis retain cells through the real ECS driver.
    /// All pending loads are returned with stable completion tickets. Use
    /// explicit admission/spawn and then `settle_ecs_cell`; no I/O is performed.
    ///
    /// Partition bookkeeping is cloned before eviction and committed on success.
    /// If a destructor/hook panics, already applied World mutations are not
    /// rolled back, but the original cell/entity index is retained for retry
    /// (including still-live entities the driver had drained). This costs a copy
    /// proportional to tracked partition metadata, not component payloads.
    /// Retrying requires the World to remain usable after the originating panic;
    /// this adapter cannot repair a component-storage destructor-unwind failure.
    pub fn advance_ecs_streaming(
        &mut self,
        camera: WorldPos,
        extra: &[WorldPos],
    ) -> Result<EcsStreamingResult, EcsPolicyError> {
        let applied = self.applied_ecs_policy()?;
        let profile = applied
            .policy
            .profile()
            .streaming
            .ok_or(EcsPolicyError::StreamingDisabled)?;
        validate_position(profile, camera)?;
        for &position in extra {
            validate_position(profile, position)?;
        }
        let (state, world) = self.ecs_policy_parts();
        let runtime = &mut state.as_mut().expect("validated policy").partition;
        let mut next = runtime.streaming.as_ref().expect("configured grid").clone();
        let result = next.advance(world, camera, extra);
        runtime.streaming = Some(next);
        Ok(runtime.finish_stream(world.id(), result))
    }

    /// Evict all tracked cells and cancel their pending tickets before replacing
    /// a streaming profile. No interests remain; the next advance starts loading
    /// again. Panic/retry semantics match `advance_ecs_streaming`.
    pub fn drain_ecs_streaming(&mut self) -> Result<EcsStreamingResult, EcsPolicyError> {
        self.ecs_world_partition()?;
        let (state, world) = self.ecs_policy_parts();
        let runtime = &mut state.as_mut().expect("validated policy").partition;
        let mut next = runtime.streaming.as_ref().expect("configured grid").clone();
        let result = next.driver_mut().stream(world, &[]);
        runtime.streaming = Some(next);
        Ok(runtime.finish_stream(world.id(), result))
    }

    /// Confirm one still-current cell load after the host has spawned its payload.
    ///
    /// Every handle must be live in this App and not owned by another cell.
    /// Checks complete before bookkeeping mutation. Handles themselves cannot
    /// prove cross-World provenance; the ticket proves request ownership only.
    /// This checks current counts, not a second allocation: admitting the payload
    /// belongs before spawn. Rejected completions leave spawned entities under
    /// host ownership; they are not silently despawned or rolled back.
    pub fn settle_ecs_cell(
        &mut self,
        request: &EcsCellLoad,
        entities: &[Entity],
    ) -> Result<(), EcsPolicyError> {
        let applied = self.applied_ecs_policy()?;
        let runtime = &applied.partition;
        let grid = runtime
            .streaming
            .as_ref()
            .ok_or(EcsPolicyError::StreamingDisabled)?;
        if request.world != applied.world
            || grid.driver().streamer().state(request.cell) != CellState::Loading
            || !runtime
                .pending
                .get(&request.cell)
                .is_some_and(|id| Arc::ptr_eq(id, &request.identity))
        {
            return Err(EcsPolicyError::InvalidLoad);
        }
        validate_population(self.world(), entities)?;
        self.check_entity_admission(0)?;
        for &entity in entities {
            if grid
                .index()
                .cell_of(entity)
                .is_some_and(|cell| cell != request.cell)
            {
                return Err(EcsPolicyError::EntityAlreadyStreamed(entity));
            }
        }
        let runtime = &mut self
            .ecs_policy
            .as_mut()
            .expect("validated policy")
            .partition;
        runtime
            .streaming
            .as_mut()
            .expect("configured grid")
            .settle_loaded(request.cell, entities);
        runtime.pending.remove(&request.cell);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
