//! Engine-agnostic observers and an observer registry.
//!
//! An [`Observer`] is any callback that reacts to [`PhysicsEvent`]s. The
//! [`ObserverRegistry`] fans a batch of events out to every registered observer
//! and simultaneously buffers them in an internal queue, so integrators can
//! either push (callbacks) or pull (drain the queue) as they prefer. Nothing in
//! this module depends on a specific game engine or ECS.
//!
//! # Provenance
//!
//! The observer/registry pattern is a standard, publicly documented software
//! design pattern. This file contains no Unreal Engine source or derived code.

use crate::events::event::PhysicsEvent;

/// A reactive sink for [`PhysicsEvent`]s.
///
/// A blanket implementation is provided for every `FnMut(&PhysicsEvent)`, so a
/// plain closure can be registered directly as an observer.
pub trait Observer {
    /// Handles a single physics event.
    fn on_event(&mut self, event: &PhysicsEvent);
}

impl<F> Observer for F
where
    F: FnMut(&PhysicsEvent),
{
    fn on_event(&mut self, event: &PhysicsEvent) {
        self(event);
    }
}

/// A registry that dispatches physics events to observers and queues them.
///
/// Registered observers are invoked in registration order for every event
/// passed to [`ObserverRegistry::dispatch`]. The same events are appended to an
/// internal queue that can be inspected with [`ObserverRegistry::queued`] or
/// consumed with [`ObserverRegistry::drain_queue`].
#[derive(Default)]
pub struct ObserverRegistry {
    observers: Vec<Box<dyn Observer>>,
    queue: Vec<PhysicsEvent>,
}

impl ObserverRegistry {
    /// Creates an empty registry with no observers and an empty queue.
    #[must_use]
    pub fn new() -> ObserverRegistry {
        ObserverRegistry::default()
    }

    /// Registers an observer, which will receive every event dispatched from
    /// now on.
    pub fn register(&mut self, observer: impl Observer + 'static) {
        self.observers.push(Box::new(observer));
    }

    /// Returns the number of registered observers.
    #[must_use]
    pub fn observer_count(&self) -> usize {
        self.observers.len()
    }

    /// Dispatches a batch of events: every observer is invoked for each event,
    /// and each event is appended to the internal queue.
    pub fn dispatch(&mut self, events: &[PhysicsEvent]) {
        for event in events {
            for observer in &mut self.observers {
                observer.on_event(event);
            }
            self.queue.push(*event);
        }
    }

    /// Returns the currently queued events without consuming them.
    #[must_use]
    pub fn queued(&self) -> &[PhysicsEvent] {
        &self.queue
    }

    /// Removes and returns all queued events, leaving the queue empty.
    #[must_use]
    pub fn drain_queue(&mut self) -> Vec<PhysicsEvent> {
        core::mem::take(&mut self.queue)
    }

    /// Clears the internal event queue without returning its contents.
    pub fn clear_queue(&mut self) {
        self.queue.clear();
    }
}
