//! ECS assembly for the deterministic `StateTree` semantic kernel.
//!
//! A [`StateTreeAsset`] shares an immutable compiled [`Plan`] between spawned
//! instances. Each [`StateTreeInstance`] owns its independent semantic cursor.
//! Input events are queued first and only become transitions when
//! [`tick_entity`] is called at an exclusive-world/tick boundary.
#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

extern crate alloc;

mod lifecycle;
mod schedule;

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::mem;

use prism_ecs::{
    component::Component,
    entity::{Entity, EntityLocation},
    query::With,
    world::World,
};
use prism_state_tree_core::{
    ContextRef, Event, EventQueueError, Guard, GuardId, Instance, MicrostepLimit, Plan,
    RuntimeError, Schema, SchemaError, StateId, StepReport,
};

use lifecycle::LifecycleCollector;
pub use lifecycle::{StateTreeLifecycleEvent, StateTreeLifecycleNotification};

/// A host context validated against a concrete core [`Schema`].
///
/// The context is borrowed from the integration's world and is therefore not
/// an ECS component. This wrapper is the explicit boundary an ECS adapter can
/// pass to future binding/evaluator layers; the current ST1 runtime does not
/// inspect or mutate it.
pub struct ValidatedContext<'w, S>
where
    S: Schema + 'w,
{
    context: S::Context<'w>,
    schema: PhantomData<fn() -> S>,
}

impl<'w, S> ValidatedContext<'w, S>
where
    S: Schema + 'w,
{
    /// Validates a borrowed host context and retains it without taking world
    /// ownership.
    pub fn new(context: S::Context<'w>) -> Result<Self, SchemaError> {
        S::validate_context(&context)?;
        Ok(Self {
            context,
            schema: PhantomData,
        })
    }

    /// Returns the validated read-only host context.
    pub fn context(&self) -> &S::Context<'w> {
        &self.context
    }

    /// Returns the context to the integration layer.
    pub fn into_inner(self) -> S::Context<'w> {
        self.context
    }
}

impl<'w, S> ContextRef<'w> for ValidatedContext<'w, S>
where
    S: Schema + 'w,
{
    fn schema_id(&self) -> prism_state_tree_core::SchemaId {
        self.context.schema_id()
    }
}

/// Validates a host context at the ECS adapter boundary without gathering a
/// world implicitly. ECS integrations own the actual `World`/`Entity` query
/// and pass its borrowed result here.
pub fn validate_context<S: Schema>(context: &S::Context<'_>) -> Result<(), SchemaError> {
    S::validate_context(context)
}

/// Shared immutable compiled tree asset.
#[derive(Clone, Debug)]
pub struct StateTreeAsset(Arc<Plan>);

impl StateTreeAsset {
    /// Wraps a validated plan so many instances can refer to one shared asset.
    pub fn new(plan: Plan) -> Self {
        Self(Arc::new(plan))
    }

    /// Returns the shared immutable plan.
    pub fn plan(&self) -> &Plan {
        &self.0
    }

    /// Returns another handle to the same immutable plan allocation.
    pub fn clone_shared(&self) -> Self {
        self.clone()
    }
}

/// Per-entity `StateTree` runtime component.
pub struct StateTreeInstance {
    asset: StateTreeAsset,
    runtime: Instance,
    lifecycle_events: Vec<StateTreeLifecycleEvent>,
}

pub use schedule::{install_tick_system, StateTreeTickOutcome, StateTreeTickReport};

impl Component for StateTreeInstance {}

impl StateTreeInstance {
    /// Creates an independent runtime cursor for a shared tree asset.
    pub fn new(
        asset: StateTreeAsset,
        event_capacity: usize,
        microstep_limit: MicrostepLimit,
    ) -> Result<Self, RuntimeError> {
        if event_capacity == 0 {
            return Err(RuntimeError::EventQueue(EventQueueError::ZeroCapacity));
        }
        let runtime = Instance::new_shared(Arc::clone(&asset.0), event_capacity, microstep_limit)?;
        Ok(Self {
            asset,
            runtime,
            lifecycle_events: Vec::new(),
        })
    }

    /// Returns this instance's shared immutable asset handle.
    pub fn asset(&self) -> &StateTreeAsset {
        &self.asset
    }

    /// Returns the runtime's active configuration in deterministic entry order.
    pub fn active_states(&self) -> &[StateId] {
        self.runtime.active_states()
    }

    /// Returns the number of queued events that have not been consumed.
    pub fn queued_events(&self) -> usize {
        self.runtime.queued_events()
    }

    /// Returns lifecycle observations that have not yet been drained.
    #[must_use]
    pub fn lifecycle_events(&self) -> &[StateTreeLifecycleEvent] {
        &self.lifecycle_events
    }

    /// Removes and returns lifecycle observations in core callback order.
    pub fn drain_lifecycle_events(&mut self) -> Vec<StateTreeLifecycleEvent> {
        mem::take(&mut self.lifecycle_events)
    }

    /// Queues an input event. It cannot cause a transition until the next tick.
    pub fn request_event(&mut self, event: Event) -> Result<(), EventQueueError> {
        self.runtime.send(event)
    }

    /// Installs or replaces an immutable transition guard for this entity.
    ///
    /// Guard state must be thread-safe (`Send + Sync`) because the owning
    /// runtime is stored as an ECS component. Guard evaluation happens during
    /// [`Self::step`] and receives only the core's read-only `GuardContext`.
    pub fn set_guard<G: Guard + 'static>(&mut self, id: GuardId, guard: G) {
        self.runtime.set_guard(id, guard);
    }

    /// Executes one run-to-completion tick, including initial entry if needed.
    pub fn step(&mut self) -> Result<StepReport, RuntimeError> {
        let mut collector = LifecycleCollector::new(&mut self.lifecycle_events);
        self.runtime.run(&mut collector)
    }
}

/// Queues a deferred spawn carrying a fresh `StateTree` runtime instance.
///
/// The returned entity handle is reserved immediately; its components become
/// visible after [`CommandQueue::apply`].
pub fn queue_spawn(
    commands: &mut prism_ecs::command::Commands<'_, '_>,
    asset: StateTreeAsset,
    event_capacity: usize,
    microstep_limit: MicrostepLimit,
) -> Result<Entity, RuntimeError> {
    let instance = StateTreeInstance::new(asset, event_capacity, microstep_limit)?;
    Ok(commands.spawn(instance))
}

/// Runs one entity's instance at an exclusive-world tick boundary.
///
/// Any queued events are consumed by the core runtime here. No ECS structure is
/// mutated during transition evaluation; component mutation is bounded to this
/// explicit tick call. Returns `None` if the entity has no `StateTree` component.
pub fn tick_entity(world: &mut World, entity: Entity) -> Option<Result<StepReport, RuntimeError>> {
    world
        .get_mut::<StateTreeInstance>(entity)
        .map(|mut instance| instance.step())
}

/// The outcome of one entry in an explicit [`tick_entities`] request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchTickFailure {
    /// The runtime error returned by the failed tick.
    pub error: RuntimeError,
    /// Active configuration after the error. A failed tick is not transactional
    /// and may have already applied transitions before reporting the error.
    pub active_states: Vec<StateId>,
    /// Events still queued after the failed tick.
    pub queued_events: usize,
}

/// Reason an explicit batch tick could not find a runnable state-tree
/// component.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchTickMissingReason {
    /// The handle is stale, freed, or belongs to a different generation.
    StaleEntity,
    /// The handle is live but has not been placed in an archetype yet.
    NotMaterialized,
    /// The live, materialized entity does not carry `StateTreeInstance`.
    MissingComponent,
}

/// Failure while reserving the explicit entity gather buffer for [`tick_all`].
///
/// This covers the adapter-owned gather vector only. Allocations performed by
/// the ECS query cache or by the existing report/runtime paths remain governed
/// by their respective host APIs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchTickError {
    /// The caller-provided world contents could not fit in the gather buffer.
    AllocationFailed,
}

/// Outcome of one entry in an explicit [`tick_entities`] request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BatchTickOutcome {
    /// The entity had a `StateTreeInstance` and its tick completed.
    Completed(StepReport),
    /// The entity was not runnable. No world state was changed for this entry.
    Missing(BatchTickMissingReason),
    /// The entity had a state-tree instance, but its tick returned a runtime
    /// error. The error is scoped to this entry; later entries are still run.
    /// The failure snapshot describes the post-failure state because core
    /// runtime errors do not roll back transitions already applied.
    Failed(BatchTickFailure),
}

/// One ordered entity/outcome pair produced by [`tick_entities`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchTickEntry {
    /// The entity handle supplied by the caller.
    pub entity: Entity,
    /// The outcome of ticking `entity`.
    pub outcome: BatchTickOutcome,
}

/// Deterministic result of an explicit batch tick.
///
/// Entries have exactly the same order, length, and multiplicity as the input
/// slice. In particular, duplicate handles are processed once per occurrence;
/// the function does not sort, deduplicate, query, or schedule them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchTickReport {
    entries: Vec<BatchTickEntry>,
    completed: usize,
    missing: usize,
    failed: usize,
}

impl BatchTickReport {
    /// Returns the per-entity results in caller-supplied order.
    pub fn entries(&self) -> &[BatchTickEntry] {
        &self.entries
    }

    /// Returns the number of input entries that completed successfully.
    pub const fn completed_count(&self) -> usize {
        self.completed
    }

    /// Returns the number of input entries that were safe no-ops.
    pub const fn missing_count(&self) -> usize {
        self.missing
    }

    /// Returns the number of input entries whose runtime tick failed.
    pub const fn failed_count(&self) -> usize {
        self.failed
    }

    /// Returns the number of input entities, including duplicates.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether the input slice was empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Ticks the requested entities exactly once each, in the supplied order.
///
/// This is an explicit exclusive-world operation. It does not discover
/// entities through a query, schedule work, invoke observers, or deduplicate
/// handles. A stale/dead entity or an entity without [`StateTreeInstance`] is
/// reported as [`BatchTickOutcome::Missing`] and is otherwise a no-op. Runtime
/// errors are retained per entry so one failed entity cannot suppress later
/// requested ticks. A [`BatchTickOutcome::Failed`] entry does not imply a
/// rollback: inspect its post-failure snapshot to observe state changes that
/// happened before the core returned the error.
pub fn tick_entities(world: &mut World, entities: &[Entity]) -> BatchTickReport {
    let mut report = BatchTickReport {
        entries: Vec::with_capacity(entities.len()),
        completed: 0,
        missing: 0,
        failed: 0,
    };

    for &entity in entities {
        let outcome = match tick_entity(world, entity) {
            Some(Ok(step)) => {
                report.completed += 1;
                BatchTickOutcome::Completed(step)
            }
            Some(Err(error)) => {
                report.failed += 1;
                let failure = world
                    .get::<StateTreeInstance>(entity)
                    .map(|instance| BatchTickFailure {
                        error,
                        active_states: instance.active_states().to_vec(),
                        queued_events: instance.queued_events(),
                    })
                    .unwrap_or(BatchTickFailure {
                        error,
                        active_states: Vec::new(),
                        queued_events: 0,
                    });
                BatchTickOutcome::Failed(failure)
            }
            None => {
                report.missing += 1;
                let reason = if !world.contains(entity) {
                    BatchTickMissingReason::StaleEntity
                } else if world
                    .entities()
                    .location(entity)
                    .is_some_and(|location| location == EntityLocation::EMPTY)
                {
                    BatchTickMissingReason::NotMaterialized
                } else {
                    BatchTickMissingReason::MissingComponent
                };
                BatchTickOutcome::Missing(reason)
            }
        };
        report.entries.push(BatchTickEntry { entity, outcome });
    }

    report
}

/// Gathers every materialized [`StateTreeInstance`] and ticks them once.
///
/// The gather is an actual ECS query, not a scan of guessed entity ids. Results
/// are sorted by the generational [`Entity`] handle before execution, making
/// the batch order independent of archetype insertion order. This function is
/// still an explicit exclusive-world operation: it does not register itself
/// with a schedule or run automatically each frame.
pub fn tick_all(world: &mut World) -> Result<BatchTickReport, BatchTickError> {
    let query = world.query_filtered::<Entity, With<StateTreeInstance>>();
    let count = query.iter(world).count();
    let mut entities = Vec::new();
    entities
        .try_reserve_exact(count)
        .map_err(|_| BatchTickError::AllocationFailed)?;
    for entity in query.iter(world) {
        entities.push(entity);
    }
    entities.sort_unstable();
    Ok(tick_entities(world, &entities))
}

#[cfg(test)]
mod tests;
