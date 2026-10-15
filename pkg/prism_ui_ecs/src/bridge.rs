//! Observer-driven field binding registry, with explicit full-pass fallbacks.
//!
//! [`EcsBridge`] owns a collection of type-erased [`SyncBinding`]s and drives
//! them together: [`EcsBridge::pull_changed`] refreshes notified bindings and
//! [`EcsBridge::push_changed`] writes UI-side changes back. Explicit
//! [`EcsBridge::pull_all`] / [`EcsBridge::push_all`] remain available.
//!
//! # Feedback-loop protection
//!
//! Both directions are guarded by equality checks, so an `ECS -> signal -> ECS`
//! round trip settles instead of oscillating:
//!
//! * **Pull** uses [`Signal::set_if_changed`](prism_ui_reactive::Signal::set_if_changed):
//!   a component value already matching the signal produces no signal update.
//! * **Push** compares the projected field to the signal value *before*
//!   mutating, so a component already equal to the signal is left untouched and
//!   its change tick is not advanced.
//!
//! Consequently, after a `push_all` the next `pull_all` sees no net change (the
//! value just written equals the signal), and after a `pull_all` the next
//! `push_all` finds the component already equal to the signal.

use std::any::TypeId;
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

use prism_ecs::prelude::{Component, Entity, World};
use prism_ecs::world::WorldId;
use prism_ui_reactive::{Signal, SignalChangeId, SignalChangeQueue, SignalSubscription};

use crate::binding::FieldBinding;
use crate::entity_binding::{EntityBinding, SyncBinding};
use crate::invalidation::{cleanup_bridge_observers, lock, Subscription};

/// Process-unique registration token. A token from another bridge never aliases
/// a binding here, even after removal or bridge replacement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BindingId(u64);

/// The bridge is still attached to a different World. Detach it explicitly
/// before moving it; entity handles alone do not identify a World.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BridgeWorldMismatch;

impl std::fmt::Display for BridgeWorldMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("EcsBridge belongs to a different World; detach before reattaching")
    }
}

impl std::error::Error for BridgeWorldMismatch {}

struct BindingEntry {
    entity: Entity,
    component: TypeId,
    binding: Box<dyn SyncBinding>,
    signal_subscription: Option<SignalSubscription>,
}

/// A registry of entity/field bindings with frame-level sync passes.
#[derive(Default)]
pub struct EcsBridge {
    bindings: BTreeMap<BindingId, BindingEntry>,
    subscriptions: BTreeMap<TypeId, Subscription>,
    world: Cell<Option<(WorldId, u64)>>,
    signal_changes: SignalChangeQueue,
    signal_bindings: BTreeMap<SignalChangeId, BindingId>,
}

impl EcsBridge {
    /// Creates an empty bridge.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a one-way binding from `entity`'s component field into
    /// `signal`.
    pub fn bind<C, T>(
        &mut self,
        entity: Entity,
        signal: Signal<T>,
        reader: impl Fn(&C) -> T + 'static,
    ) -> BindingId
    where
        C: Component,
        T: Clone + PartialEq + 'static,
    {
        let binding = FieldBinding::read_only(signal, reader);
        self.register::<C>(entity, Box::new(EntityBinding::new(entity, binding)), None)
    }

    /// Registers a two-way binding between `entity`'s component field and
    /// `signal`.
    pub fn bind_two_way<C, T>(
        &mut self,
        entity: Entity,
        signal: Signal<T>,
        reader: impl Fn(&C) -> T + 'static,
        writer: impl Fn(&mut C, &T) + 'static,
    ) -> BindingId
    where
        C: Component,
        T: Clone + PartialEq + 'static,
    {
        let subscription = signal.subscribe_changes(&self.signal_changes);
        let binding = FieldBinding::read_write(signal, reader, writer);
        self.register::<C>(
            entity,
            Box::new(EntityBinding::new(entity, binding)),
            Some(subscription),
        )
    }

    fn register<C: Component>(
        &mut self,
        entity: Entity,
        binding: Box<dyn SyncBinding>,
        signal_subscription: Option<SignalSubscription>,
    ) -> BindingId {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let id = BindingId(
            NEXT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
                .expect("binding identity exhausted"),
        );
        let component = TypeId::of::<C>();
        let subscription = self
            .subscriptions
            .entry(component)
            .or_insert_with(Subscription::new::<C>);
        let mut routes = lock(&subscription.routes);
        routes.entities.entry(entity).or_default().insert(id);
        routes.dirty.insert(id);
        if let Some(subscription) = &signal_subscription {
            self.signal_bindings.insert(subscription.id(), id);
        }
        self.bindings.insert(
            id,
            BindingEntry {
                entity,
                component,
                binding,
                signal_subscription,
            },
        );
        id
    }

    /// Remove one registration and immediately release its observers when the
    /// last binding of that component type is removed. Other bridges and World
    /// observers are unaffected. The supplied World is used only for cleanup.
    pub fn unbind(&mut self, world: &mut World, id: BindingId) -> bool {
        let Some(entry) = self.bindings.remove(&id) else {
            return false;
        };
        if let Some(subscription) = &entry.signal_subscription {
            self.signal_bindings.remove(&subscription.id());
        }
        let subscription = self
            .subscriptions
            .get(&entry.component)
            .expect("binding subscription");
        let empty = {
            let mut routes = lock(&subscription.routes);
            routes.dirty.remove(&id);
            if let Some(ids) = routes.entities.get_mut(&entry.entity) {
                ids.remove(&id);
                if ids.is_empty() {
                    routes.entities.remove(&entry.entity);
                }
            }
            routes.entities.is_empty()
        };
        if empty {
            self.subscriptions.remove(&entry.component);
        }
        cleanup_bridge_observers(world);
        true
    }

    /// Release observer leases but keep bindings for later reattachment. If the
    /// old World is not available, its expired observers are harmless and can
    /// be reclaimed by [`cleanup_bridge_observers`] on that World later.
    pub fn detach(&mut self, world: &mut World) {
        for subscription in self.subscriptions.values_mut() {
            subscription.reset();
        }
        self.world.set(None);
        cleanup_bridge_observers(world);
    }

    fn attach(&mut self, world: &mut World) -> Result<(), BridgeWorldMismatch> {
        let stamp = (world.id(), world.archetype_generation());
        if self.world.get().is_some_and(|(id, _)| id != stamp.0) {
            return Err(BridgeWorldMismatch);
        }
        if self.world.get() != Some(stamp) {
            for subscription in self.subscriptions.values_mut() {
                subscription.reset();
            }
            self.world.set(Some(stamp));
        }
        cleanup_bridge_observers(world);
        for subscription in self.subscriptions.values_mut() {
            subscription.install(world);
        }
        Ok(())
    }

    /// Refresh only bindings invalidated by changed/insert/remove observers.
    /// First attachment and snapshot restoration refresh every binding once.
    /// Repeated notifications coalesce; refreshes run in registration order.
    /// Missing components preserve the signal until insertion/restoration.
    ///
    /// This flushes the World's change bus (including unrelated callbacks).
    /// Callback writes follow that bus's next-flush semantics. A reader panic
    /// leaves its notification pending for retry. A World replaced by a callback
    /// is rejected before reading any of its entities.
    pub fn pull_changed(&mut self, world: &mut World) -> Result<usize, BridgeWorldMismatch> {
        self.attach(world)?;
        world.flush_changed_observers();
        // Flush callbacks can restore or replace the World.
        self.attach(world)?;
        let pending: BTreeSet<_> = self
            .subscriptions
            .values()
            .flat_map(|subscription| lock(&subscription.routes).dirty.clone())
            .collect();
        let mut changed = 0;
        for id in pending {
            let entry = self.bindings.get_mut(&id).expect("live invalidation route");
            changed += usize::from(entry.binding.refresh(world));
            lock(&self.subscriptions[&entry.component].routes)
                .dirty
                .remove(&id);
        }
        Ok(changed)
    }

    /// Process only notified two-way signals, coalesced in registration order.
    /// A clean pass does not visit bindings or read/clone signal values. A
    /// panicking reader/writer leaves its entry and the remaining batch pending;
    /// already completed entries stay acknowledged. This is retry, not rollback
    /// of partial component writes or external side effects. Signal writes made
    /// during a push are retained for the next pass, even for the current entry.
    /// With the scheduled pull-before-push order, attachment/restoration imports
    /// the ECS baseline first and equality checks suppress its signal echoes.
    pub fn push_changed(&self, world: &mut World) -> Result<usize, BridgeWorldMismatch> {
        if self.world.get().is_some_and(|(id, _)| id != world.id()) {
            return Err(BridgeWorldMismatch);
        }
        // Even a push-first bridge must not accept an aliased entity in a
        // second World. Pull installs the actual ECS observer leases later.
        if self.world.get().is_none() {
            self.world
                .set(Some((world.id(), world.archetype_generation())));
        }
        let mut pending = self.signal_changes.take_pending();
        let mut written = 0;
        while let Some(signal_id) = pending.first() {
            if let Some(id) = self.signal_bindings.get(&signal_id) {
                written += usize::from(self.bindings[id].binding.push_changed(world));
            }
            pending.acknowledge(signal_id);
        }
        Ok(written)
    }

    /// Number of registered bindings.
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Whether no bindings are registered.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Pulls every binding, returning how many signals changed.
    pub fn pull_all(&mut self, world: &World) -> usize {
        let mut changed = 0;
        for entry in self.bindings.values_mut() {
            if entry.binding.pull(world) {
                changed += 1;
            }
        }
        changed
    }

    /// Pushes every binding, returning how many components were mutated.
    pub fn push_all(&self, world: &mut World) -> usize {
        let mut written = 0;
        for entry in self.bindings.values() {
            if entry.binding.push(world) {
                written += 1;
            }
        }
        written
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui_reactive::Runtime;

    #[derive(Component)]
    struct Position {
        x: i32,
        y: i32,
    }

    #[test]
    fn new_bridge_is_empty() {
        let bridge = EcsBridge::new();
        assert!(bridge.is_empty());
        assert_eq!(bridge.len(), 0);
    }

    #[test]
    fn bind_registers_binding() {
        let mut world = World::new();
        let entity = world.spawn(Position { x: 1, y: 2 });
        let rt = Runtime::new();
        let mut bridge = EcsBridge::new();
        bridge.bind::<Position, i32>(entity, rt.signal(0), |p| p.x);
        assert_eq!(bridge.len(), 1);
        assert!(!bridge.is_empty());
    }

    #[test]
    fn pull_all_counts_changed_bindings() {
        let mut world = World::new();
        let entity = world.spawn(Position { x: 10, y: 20 });
        let rt = Runtime::new();
        let x = rt.signal(0i32);
        let y = rt.signal(0i32);
        let mut bridge = EcsBridge::new();
        bridge.bind::<Position, i32>(entity, x.clone(), |p| p.x);
        bridge.bind::<Position, i32>(entity, y.clone(), |p| p.y);

        assert_eq!(bridge.pull_all(&world), 2);
        assert_eq!(x.get_untracked(), 10);
        assert_eq!(y.get_untracked(), 20);
        // Nothing changed on the component, so a second pull reports zero.
        assert_eq!(bridge.pull_all(&world), 0);
    }

    #[test]
    fn push_all_counts_written_bindings() {
        let mut world = World::new();
        let entity = world.spawn(Position { x: 0, y: 0 });
        let rt = Runtime::new();
        let x = rt.signal(3i32);
        let y = rt.signal(4i32);
        let mut bridge = EcsBridge::new();
        bridge.bind_two_way::<Position, i32>(entity, x, |p| p.x, |p, v| p.x = *v);
        bridge.bind_two_way::<Position, i32>(entity, y, |p| p.y, |p, v| p.y = *v);

        assert_eq!(bridge.push_all(&mut world), 2);
        let pos = world.get::<Position>(entity).unwrap();
        assert_eq!((pos.x, pos.y), (3, 4));
        // Values already match now, so a second push writes nothing.
        assert_eq!(bridge.push_all(&mut world), 0);
    }

    #[test]
    fn read_only_bindings_do_not_push() {
        let mut world = World::new();
        let entity = world.spawn(Position { x: 1, y: 2 });
        let rt = Runtime::new();
        let mut bridge = EcsBridge::new();
        bridge.bind::<Position, i32>(entity, rt.signal(99), |p| p.x);

        assert_eq!(bridge.push_all(&mut world), 0);
        assert_eq!(world.get::<Position>(entity).unwrap().x, 1);
    }

    #[test]
    fn multiple_entities_sync_independently() {
        let mut world = World::new();
        let a = world.spawn(Position { x: 1, y: 0 });
        let b = world.spawn(Position { x: 2, y: 0 });
        let rt = Runtime::new();
        let sa = rt.signal(0i32);
        let sb = rt.signal(0i32);
        let mut bridge = EcsBridge::new();
        bridge.bind::<Position, i32>(a, sa.clone(), |p| p.x);
        bridge.bind::<Position, i32>(b, sb.clone(), |p| p.x);

        assert_eq!(bridge.pull_all(&world), 2);
        assert_eq!(sa.get_untracked(), 1);
        assert_eq!(sb.get_untracked(), 2);
    }

    #[test]
    fn two_way_round_trip_does_not_oscillate() {
        let mut world = World::new();
        let entity = world.spawn(Position { x: 0, y: 0 });
        let rt = Runtime::new();
        let x = rt.signal(0i32);
        let mut bridge = EcsBridge::new();
        bridge.bind_two_way::<Position, i32>(entity, x.clone(), |p| p.x, |p, v| p.x = *v);

        // ECS -> signal.
        world.get_mut::<Position>(entity).unwrap().x = 5;
        assert_eq!(bridge.pull_all(&world), 1);
        assert_eq!(x.get_untracked(), 5);
        // signal already equals component: push is a no-op, no oscillation.
        assert_eq!(bridge.push_all(&mut world), 0);

        // signal -> ECS.
        x.set(9);
        assert_eq!(bridge.push_all(&mut world), 1);
        assert_eq!(world.get::<Position>(entity).unwrap().x, 9);
        // component now equals signal: pull reports no net change.
        assert_eq!(bridge.pull_all(&world), 0);
        assert_eq!(x.get_untracked(), 9);
    }
}
