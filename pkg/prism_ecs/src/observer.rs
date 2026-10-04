//! Observers (design §12): high-level, dynamically registered, event-driven
//! reactions layered on top of the low-level component [`hooks`] layer.
//!
//! Where a [`ComponentHook`](crate::component_hooks::ComponentHook) is a single
//! stateless `fn` pointer bound to a component *at registration time*,
//! an **observer** is a stateful boxed closure that can be added (and removed)
//! at any time, with **many** observers watching the same event/component pair.
//! Observers are the engine's reactive bus (design §24.2): spatial-index
//! sync, GPU-resource (de)registration, UI/game event handling, and relation
//! cleanup all subscribe here instead of polling.
//!
//! # Trigger model
//!
//! Observers fire at exactly the same structural-change sites as component
//! hooks, immediately after the matching hook batch, so they observe a fully
//! consistent [`World`]:
//!
//! | Transition | Events (in order) |
//! |---|---|
//! | component newly added     | [`Add`](LifecycleEvent::Add) then [`Insert`](LifecycleEvent::Insert) |
//! | existing value overwritten | [`Replace`](LifecycleEvent::Replace) then [`Insert`](LifecycleEvent::Insert) |
//! | component removed / despawn | [`Replace`](LifecycleEvent::Replace) then [`Remove`](LifecycleEvent::Remove) |
//!
//! [`Add`](LifecycleEvent::Add) / [`Insert`](LifecycleEvent::Insert) fire
//! after the value is in place; [`Replace`](LifecycleEvent::Replace) /
//! [`Remove`](LifecycleEvent::Remove) fire before it is overwritten or
//! dropped, so a callback can still read the outgoing value.
//!
//! # Bubbling
//!
//! A custom event raised with [`World::trigger`](crate::world::World::trigger)
//! can **bubble** up a relation chain (default [`ChildOf`]-style ancestry): the
//! event is delivered to observers of the target entity, then to observers of
//! each ancestor, until a callback stops propagation or the chain ends. This
//! powers UI / gameplay event routing (design §12).
//!
//! # Re-entrancy & borrow model
//!
//! A callback receives `&mut `[`World`] and may freely issue structural
//! changes, trigger further events, or register new observers. To hand the
//! callback `&mut World` while the observer list lives *inside* that world, the
//! firing path temporarily moves each matching callback out of its slot, runs
//! it, then restores it. Consequently a callback may remove **other**
//! observers synchronously, but a request to remove the **currently firing**
//! observer takes effect only after that callback returns.
//!
//! [`hooks`]: crate::component_hooks
//! [`World`]: crate::world::World
//! [`ChildOf`]: crate::relation

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::component::ComponentId;
use crate::entity::Entity;
use crate::world::World;

/// The lifecycle transition an observer watches (design §12).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LifecycleEvent {
    /// Fired when a component is newly added to an entity.
    Add,
    /// Fired on every write of a component value (add *and* overwrite).
    Insert,
    /// Fired just before an existing value is overwritten or removed.
    Replace,
    /// Fired just before a component is removed (including on despawn).
    Remove,
}

/// A stable, process-unique identifier for a registered observer, returned by
/// the `observe*` registration methods and accepted by
/// [`World::remove_observer`](crate::world::World::remove_observer).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ObserverId(u64);

impl ObserverId {
    /// The raw monotonically assigned index backing this id.
    #[inline]
    pub const fn index(self) -> u64 {
        self.0
    }
}

/// The context handed to a lifecycle observer callback when it fires.
///
/// It names the triggering entity, component, and [`LifecycleEvent`], and
/// grants mutable access to the owning [`World`] so the callback can read the
/// value (e.g. `ctx.world.get::<T>(ctx.entity)`), maintain derived state, or
/// perform further structural changes.
pub struct ObserverContext<'w> {
    /// The world in which the triggering change occurred, fully consistent.
    pub world: &'w mut World,
    /// The entity whose component set changed.
    pub entity: Entity,
    /// The component that triggered this observer.
    pub component: ComponentId,
    /// The lifecycle transition that fired.
    pub event: LifecycleEvent,
}

/// A boxed, stateful observer callback.
///
/// `Send + Sync` keeps the registry movable across threads (observers run only
/// at exclusive structural-change sites, so they are never called concurrently).
pub type ObserverCallback = Box<dyn FnMut(&mut ObserverContext<'_>) + Send + Sync>;

/// The context handed to a **custom-event** observer, raised explicitly via
/// [`World::trigger`](crate::world::World::trigger) rather than by a structural
/// change. Supports [`stop_propagation`](EventContext::stop_propagation) to
/// halt bubbling up the ancestry chain.
pub struct EventContext<'w> {
    /// The world in which the event was raised, fully consistent.
    pub world: &'w mut World,
    /// The entity the event is currently being delivered to (the original
    /// target first, then successive ancestors while bubbling).
    pub entity: Entity,
    /// The entity the event was originally raised against.
    pub target: Entity,
    stop: bool,
}

impl EventContext<'_> {
    /// Stop the event from bubbling further up the relation chain. Observers
    /// already scheduled on the current entity still run; no ancestor is
    /// visited afterwards.
    #[inline]
    pub fn stop_propagation(&mut self) {
        self.stop = true;
    }

    /// Whether propagation has been stopped.
    #[inline]
    pub fn is_propagation_stopped(&self) -> bool {
        self.stop
    }
}

/// A boxed custom-event observer callback.
pub type EventCallback = Box<dyn FnMut(&mut EventContext<'_>) + Send + Sync>;

/// One registered lifecycle observer. Stored in an `Option` slot so a firing
/// callback can be moved out and restored without disturbing other slots.
struct LifecycleEntry {
    id: ObserverId,
    event: LifecycleEvent,
    component: ComponentId,
    callback: ObserverCallback,
}

/// One registered custom-event observer, keyed by the event-type
/// [`ComponentId`] used as the event marker.
struct EventEntry {
    id: ObserverId,
    event: ComponentId,
    callback: EventCallback,
}

/// The observer registry owned by the [`World`] (design §12).
///
/// Holds lifecycle observers (keyed by [`LifecycleEvent`] + component) and
/// custom-event observers (keyed by an event marker [`ComponentId`]). Empty by
/// default, so a world with no observers pays nothing at structural-change
/// sites beyond a single emptiness check.
#[derive(Default)]
pub struct Observers {
    lifecycle: Vec<Option<LifecycleEntry>>,
    events: Vec<Option<EventEntry>>,
}

impl Observers {
    /// Create an empty registry.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether no observers of any kind are registered.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.lifecycle.iter().all(Option::is_none) && self.events.iter().all(Option::is_none)
    }

    /// Whether any lifecycle observer watches any component in `ids`. Used as a
    /// cheap gate to decide whether a structural path must compute id sets for
    /// observer dispatch.
    pub(crate) fn watches_any_lifecycle(&self, ids: &[ComponentId]) -> bool {
        self.lifecycle
            .iter()
            .flatten()
            .any(|e| ids.contains(&e.component))
    }

    /// Register a lifecycle observer. The caller supplies the pre-allocated
    /// [`ObserverId`]; returns it for convenience.
    pub(crate) fn add_lifecycle(
        &mut self,
        id: ObserverId,
        event: LifecycleEvent,
        component: ComponentId,
        callback: ObserverCallback,
    ) -> ObserverId {
        self.lifecycle.push(Some(LifecycleEntry {
            id,
            event,
            component,
            callback,
        }));
        id
    }

    /// Register a custom-event observer keyed by event marker `event`.
    pub(crate) fn add_event(
        &mut self,
        id: ObserverId,
        event: ComponentId,
        callback: EventCallback,
    ) -> ObserverId {
        self.events.push(Some(EventEntry {
            id,
            event,
            callback,
        }));
        id
    }

    /// Remove the observer with `id` from either registry. Returns whether it
    /// was found. A no-op for the currently-firing observer (see module docs).
    pub(crate) fn remove(&mut self, id: ObserverId) -> bool {
        for slot in &mut self.lifecycle {
            if slot.as_ref().is_some_and(|e| e.id == id) {
                *slot = None;
                return true;
            }
        }
        for slot in &mut self.events {
            if slot.as_ref().is_some_and(|e| e.id == id) {
                *slot = None;
                return true;
            }
        }
        false
    }
}

impl World {
    /// Register a lifecycle observer for component `T` and transition `event`
    /// (design §12). Returns a stable [`ObserverId`] that
    /// [`remove_observer`](World::remove_observer) accepts.
    ///
    /// `T` is registered as a component on demand. Many observers may watch the
    /// same `(event, T)`; they fire in registration order, immediately after
    /// the corresponding component [`hook`](crate::component_hooks).
    pub fn observe<T, F>(&mut self, event: LifecycleEvent, callback: F) -> ObserverId
    where
        T: crate::component::Component,
        F: FnMut(&mut ObserverContext<'_>) + Send + Sync + 'static,
    {
        let component = self.components_mut().register::<T>();
        let id = self.alloc_observer_id();
        self.observers_mut()
            .add_lifecycle(id, event, component, Box::new(callback))
    }

    /// Register a **custom-event** observer keyed by marker type `E` (design
    /// §12). The callback fires when [`trigger::<E>`](World::trigger) delivers
    /// an event to an entity it is watching (including via bubbling).
    pub fn observe_event<E, F>(&mut self, callback: F) -> ObserverId
    where
        E: crate::component::Component,
        F: FnMut(&mut EventContext<'_>) + Send + Sync + 'static,
    {
        let event = self.components_mut().register::<E>();
        let id = self.alloc_observer_id();
        self.observers_mut()
            .add_event(id, event, Box::new(callback))
    }

    /// Remove a previously registered observer (lifecycle or custom-event).
    /// Returns whether an observer with that id existed.
    pub fn remove_observer(&mut self, id: ObserverId) -> bool {
        self.observers_mut().remove(id)
    }

    /// Raise a custom event of marker type `E` against `target` (design §12),
    /// delivering it to every matching custom-event observer. If `bubble` is
    /// true the event then propagates to each ancestor reachable by following
    /// `R` relations from `target` (e.g. `ChildOf`), stopping early if a
    /// callback calls [`EventContext::stop_propagation`].
    ///
    /// `R` is the relation whose edges define "parent": the ancestry walk
    /// follows `entity --R--> parent`. Pass the same relation type used to
    /// build the hierarchy. The walk is depth-first over the first target of
    /// each entity and is cycle-safe.
    pub fn trigger<E, R>(&mut self, target: Entity, bubble: bool)
    where
        E: crate::component::Component,
        R: crate::component::Component,
    {
        let Some(event) = self.components().id_of::<E>() else {
            return;
        };
        if self.observers.events.iter().all(Option::is_none) {
            return;
        }
        let relation = self.components().id_of::<R>();

        let mut current = Some(target);
        let mut visited: Vec<Entity> = Vec::new();
        while let Some(entity) = current {
            if visited.contains(&entity) {
                break;
            }
            visited.push(entity);

            let stopped = self.fire_event_observers(event, target, entity);
            if stopped || !bubble {
                break;
            }
            // Follow the first `R` target as the parent, if any.
            current = relation.and_then(|rel| {
                self.relations()
                    .index()
                    .targets(rel, entity)
                    .first()
                    .copied()
            });
        }
    }

    /// Mutable access to the observer registry.
    fn observers_mut(&mut self) -> &mut Observers {
        &mut self.observers
    }

    /// Allocate the next process-unique observer id.
    fn alloc_observer_id(&mut self) -> ObserverId {
        let id = ObserverId(self.next_observer_id);
        self.next_observer_id += 1;
        id
    }

    /// Fire every lifecycle observer watching `event` for any component in
    /// `ids`, in registration order, immediately after the matching hook batch.
    ///
    /// Each callback is moved out of its slot for the duration of its run so it
    /// can receive `&mut World`, then restored unless it was removed. Dispatch
    /// stops early if a callback despawns `entity`.
    pub(crate) fn fire_lifecycle_observers(
        &mut self,
        event: LifecycleEvent,
        entity: Entity,
        ids: &[ComponentId],
    ) {
        if self.observers.lifecycle.is_empty() {
            return;
        }
        let mut i = 0;
        while i < self.observers.lifecycle.len() {
            let matches = self.observers.lifecycle[i]
                .as_ref()
                .is_some_and(|e| e.event == event && ids.contains(&e.component));
            if matches {
                // Move the entry out so the callback can borrow `&mut World`.
                let mut entry = self.observers.lifecycle[i]
                    .take()
                    .expect("slot checked present");
                let component = entry.component;
                let mut ctx = ObserverContext {
                    world: self,
                    entity,
                    component,
                    event,
                };
                (entry.callback)(&mut ctx);
                // Restore unless a nested call removed this observer.
                if self.observers.lifecycle[i].is_none() {
                    self.observers.lifecycle[i] = Some(entry);
                }
                // A callback may have despawned the entity; stop delivering.
                if !self.contains(entity) {
                    break;
                }
            }
            i += 1;
        }
    }

    /// Fire every custom-event observer for marker `event` against `entity`
    /// (the current bubbling position; `target` is the original). Returns
    /// whether a callback stopped propagation.
    fn fire_event_observers(&mut self, event: ComponentId, target: Entity, entity: Entity) -> bool {
        let mut stopped = false;
        let mut i = 0;
        while i < self.observers.events.len() {
            let matches = self.observers.events[i]
                .as_ref()
                .is_some_and(|e| e.event == event);
            if matches {
                let mut entry = self.observers.events[i]
                    .take()
                    .expect("slot checked present");
                let mut ctx = EventContext {
                    world: self,
                    entity,
                    target,
                    stop: false,
                };
                (entry.callback)(&mut ctx);
                if ctx.stop {
                    stopped = true;
                }
                if self.observers.events[i].is_none() {
                    self.observers.events[i] = Some(entry);
                }
                if stopped {
                    break;
                }
            }
            i += 1;
        }
        stopped
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::LifecycleEvent;
    use crate::component::Component;
    use crate::relation::RelationKind;
    use crate::resource::Resource;
    use crate::world::World;

    /// A table component the observers watch.
    #[derive(Debug, PartialEq)]
    struct Watched(i32);
    impl Component for Watched {}

    /// A second component used to prove observers are keyed per component type.
    struct Other;
    impl Component for Other {}

    /// Marker type used as a custom bubbling event.
    struct Ping;
    impl Component for Ping {}

    /// Marker relation standing in for a `ChildOf` hierarchy.
    struct ChildOf;
    impl Component for ChildOf {}

    /// Records every observer firing in call order so tests can assert both the
    /// set and the ordering of deliveries. Routed through a world resource so
    /// `Send + Sync` closures never need shared interior mutability.
    #[derive(Default)]
    struct Log {
        /// One label per observer firing, in the order the world fired them.
        events: Vec<&'static str>,
    }
    impl Resource for Log {}

    fn log_push(w: &mut World, label: &'static str) {
        w.resource_mut::<Log>().events.push(label);
    }

    #[test]
    fn lifecycle_add_insert_fire_on_spawn() {
        let mut w = World::new();
        w.init_resource::<Log>();
        w.observe::<Watched, _>(LifecycleEvent::Add, |ctx| log_push(ctx.world, "add"));
        w.observe::<Watched, _>(LifecycleEvent::Insert, |ctx| log_push(ctx.world, "insert"));
        w.observe::<Watched, _>(LifecycleEvent::Replace, |ctx| {
            log_push(ctx.world, "replace");
        });
        w.observe::<Watched, _>(LifecycleEvent::Remove, |ctx| log_push(ctx.world, "remove"));

        w.spawn(Watched(1));
        assert_eq!(w.resource::<Log>().events, ["add", "insert"]);
    }

    #[test]
    fn observer_reads_value_through_world() {
        let mut w = World::new();
        w.init_resource::<Log>();
        // The callback reads the just-written value to prove the world is
        // fully consistent when the Insert observer fires.
        w.observe::<Watched, _>(LifecycleEvent::Insert, |ctx| {
            let v = ctx
                .world
                .get::<Watched>(ctx.entity)
                .map(|c| c.0)
                .unwrap_or(-1);
            ctx.world
                .resource_mut::<Log>()
                .events
                .push(if v == 7 { "seven" } else { "other" });
        });
        w.spawn(Watched(7));
        assert_eq!(w.resource::<Log>().events, ["seven"]);
    }

    #[test]
    fn overwrite_insert_fires_replace_then_insert() {
        let mut w = World::new();
        w.init_resource::<Log>();
        w.observe::<Watched, _>(LifecycleEvent::Add, |ctx| log_push(ctx.world, "add"));
        w.observe::<Watched, _>(LifecycleEvent::Insert, |ctx| log_push(ctx.world, "insert"));
        w.observe::<Watched, _>(LifecycleEvent::Replace, |ctx| {
            log_push(ctx.world, "replace");
        });

        let e = w.spawn(Watched(1));
        w.resource_mut::<Log>().events.clear();
        // Overwriting an existing value: Replace (before) then Insert (after),
        // no Add because the component was already present.
        w.insert(e, Watched(2));
        assert_eq!(w.resource::<Log>().events, ["replace", "insert"]);
    }

    #[test]
    fn remove_fires_replace_then_remove() {
        let mut w = World::new();
        w.init_resource::<Log>();
        w.observe::<Watched, _>(LifecycleEvent::Replace, |ctx| {
            log_push(ctx.world, "replace");
        });
        w.observe::<Watched, _>(LifecycleEvent::Remove, |ctx| log_push(ctx.world, "remove"));

        let e = w.spawn(Watched(1));
        w.resource_mut::<Log>().events.clear();
        w.remove::<Watched>(e);
        assert_eq!(w.resource::<Log>().events, ["replace", "remove"]);
    }

    #[test]
    fn despawn_fires_replace_then_remove() {
        let mut w = World::new();
        w.init_resource::<Log>();
        w.observe::<Watched, _>(LifecycleEvent::Replace, |ctx| {
            log_push(ctx.world, "replace");
        });
        w.observe::<Watched, _>(LifecycleEvent::Remove, |ctx| log_push(ctx.world, "remove"));

        let e = w.spawn(Watched(1));
        w.resource_mut::<Log>().events.clear();
        w.despawn(e);
        assert_eq!(w.resource::<Log>().events, ["replace", "remove"]);
    }

    #[test]
    fn observers_keyed_per_component() {
        let mut w = World::new();
        w.init_resource::<Log>();
        w.observe::<Watched, _>(LifecycleEvent::Add, |ctx| log_push(ctx.world, "watched"));
        w.observe::<Other, _>(LifecycleEvent::Add, |ctx| log_push(ctx.world, "other"));

        w.spawn(Other);
        assert_eq!(w.resource::<Log>().events, ["other"]);
    }

    #[test]
    fn multiple_observers_fire_in_registration_order() {
        let mut w = World::new();
        w.init_resource::<Log>();
        w.observe::<Watched, _>(LifecycleEvent::Add, |ctx| log_push(ctx.world, "first"));
        w.observe::<Watched, _>(LifecycleEvent::Add, |ctx| log_push(ctx.world, "second"));
        w.observe::<Watched, _>(LifecycleEvent::Add, |ctx| log_push(ctx.world, "third"));

        w.spawn(Watched(1));
        assert_eq!(w.resource::<Log>().events, ["first", "second", "third"]);
    }

    #[test]
    fn remove_observer_stops_firing() {
        let mut w = World::new();
        w.init_resource::<Log>();
        let id = w.observe::<Watched, _>(LifecycleEvent::Add, |ctx| log_push(ctx.world, "add"));

        w.spawn(Watched(1));
        assert_eq!(w.resource::<Log>().events, ["add"]);

        assert!(w.remove_observer(id));
        assert!(!w.remove_observer(id), "second removal reports not-found");

        w.resource_mut::<Log>().events.clear();
        w.spawn(Watched(2));
        assert!(w.resource::<Log>().events.is_empty());
    }

    #[test]
    fn hook_free_world_without_observers_is_inert() {
        // No observers and no hooks: the structural paths must stay on the
        // fast path and record nothing.
        let mut w = World::new();
        w.init_resource::<Log>();
        let e = w.spawn(Watched(1));
        w.insert(e, Watched(2));
        w.remove::<Watched>(e);
        w.despawn(e);
        assert!(w.resource::<Log>().events.is_empty());
    }

    #[test]
    fn custom_event_delivered_to_target() {
        let mut w = World::new();
        w.init_resource::<Log>();
        w.observe_event::<Ping, _>(|ctx| ctx.world.resource_mut::<Log>().events.push("ping"));

        let e = w.spawn(());
        w.trigger::<Ping, ChildOf>(e, false);
        assert_eq!(w.resource::<Log>().events, ["ping"]);
    }

    #[test]
    fn custom_event_bubbles_up_relation_chain() {
        let mut w = World::new();
        w.init_resource::<Log>();
        w.register_relation::<ChildOf>(RelationKind::new());

        let parent = w.spawn(());
        let child = w.spawn(());
        w.add_relation::<ChildOf>(child, parent);

        // One observer records which entity it was delivered to.
        w.observe_event::<Ping, _>(|ctx| {
            let is_target = ctx.entity == ctx.target;
            ctx.world.resource_mut::<Log>().events.push(if is_target {
                "target"
            } else {
                "ancestor"
            });
        });

        w.trigger::<Ping, ChildOf>(child, true);
        // Delivered to the child (target) then bubbled to the parent (ancestor).
        assert_eq!(w.resource::<Log>().events, ["target", "ancestor"]);
    }

    #[test]
    fn bubbling_does_not_climb_when_disabled() {
        let mut w = World::new();
        w.init_resource::<Log>();
        w.register_relation::<ChildOf>(RelationKind::new());

        let parent = w.spawn(());
        let child = w.spawn(());
        w.add_relation::<ChildOf>(child, parent);
        w.observe_event::<Ping, _>(|ctx| log_push(ctx.world, "hit"));

        w.trigger::<Ping, ChildOf>(child, false);
        assert_eq!(w.resource::<Log>().events, ["hit"]);
    }

    #[test]
    fn stop_propagation_halts_bubbling() {
        let mut w = World::new();
        w.init_resource::<Log>();
        w.register_relation::<ChildOf>(RelationKind::new());

        let parent = w.spawn(());
        let child = w.spawn(());
        w.add_relation::<ChildOf>(child, parent);

        // Stop as soon as the event reaches the target; the parent must not see it.
        w.observe_event::<Ping, _>(|ctx| {
            ctx.world.resource_mut::<Log>().events.push("seen");
            ctx.stop_propagation();
        });

        w.trigger::<Ping, ChildOf>(child, true);
        assert_eq!(w.resource::<Log>().events, ["seen"]);
    }
}
