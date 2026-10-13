//! ECS-owned lifecycle observations emitted by the core runtime.

use alloc::vec::Vec;

use prism_ecs::entity::Entity;
use prism_ecs::event::Event;
use prism_state_tree_core::{LifecycleCallback, StateId, TransitionTarget};

/// A lifecycle event produced by one `StateTreeInstance` tick.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StateTreeLifecycleEvent {
    /// A state was inserted into the active configuration.
    Entered(StateId),
    /// A state was removed from the active configuration.
    Exited(StateId),
    /// A transition completed its exit and entry work.
    Transitioned {
        /// State whose transition declaration was selected.
        source: StateId,
        /// Authored transition target, preserving `SelfState`/`History`.
        target: TransitionTarget,
    },
}

/// Deferred, entity-addressed lifecycle notification emitted by a scheduled
/// A deferred, entity-addressed lifecycle notification from a scheduled
/// `StateTree` tick.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateTreeLifecycleNotification {
    /// ECS entity owning the `StateTree` instance.
    pub entity: Entity,
    /// Core lifecycle payload, in callback order.
    pub event: StateTreeLifecycleEvent,
}

impl Event for StateTreeLifecycleNotification {}

/// Core callback that appends observations to the owning ECS instance.
pub(crate) struct LifecycleCollector<'a> {
    events: &'a mut Vec<StateTreeLifecycleEvent>,
}

impl<'a> LifecycleCollector<'a> {
    /// Creates a collector for the instance-owned event queue.
    pub(crate) const fn new(events: &'a mut Vec<StateTreeLifecycleEvent>) -> Self {
        Self { events }
    }
}

impl LifecycleCallback for LifecycleCollector<'_> {
    fn on_enter(&mut self, state: StateId) {
        self.events.push(StateTreeLifecycleEvent::Entered(state));
    }

    fn on_exit(&mut self, state: StateId) {
        self.events.push(StateTreeLifecycleEvent::Exited(state));
    }

    fn on_transition(&mut self, source: StateId, target: TransitionTarget) {
        self.events
            .push(StateTreeLifecycleEvent::Transitioned { source, target });
    }
}
