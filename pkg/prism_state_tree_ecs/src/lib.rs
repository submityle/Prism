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

use alloc::sync::Arc;
use core::marker::PhantomData;

use prism_ecs::{component::Component, entity::Entity, world::World};
use prism_state_tree_core::{
    ContextRef, Event, EventQueueError, Guard, GuardId, Instance, MicrostepLimit, NoopLifecycle,
    Plan, RuntimeError, Schema, SchemaError, StateId, StepReport,
};

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
}

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
        Ok(Self { asset, runtime })
    }

    /// Returns this instance's shared immutable asset handle.
    pub fn asset(&self) -> &StateTreeAsset {
        &self.asset
    }

    /// Returns the runtime's active configuration in deterministic entry order.
    pub fn active_states(&self) -> &[StateId] {
        self.runtime.active_states()
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
        self.runtime.run(&mut NoopLifecycle)
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

#[cfg(test)]
mod tests;
