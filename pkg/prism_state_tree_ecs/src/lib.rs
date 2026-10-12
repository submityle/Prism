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

use prism_ecs::{component::Component, entity::Entity, world::World};
use prism_state_tree_core::{
    Event, EventQueueError, Instance, MicrostepLimit, NoopLifecycle, Plan, RuntimeError, StateId,
    StepReport,
};

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
