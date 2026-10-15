//! Change-driven observers (design §24.2): a coalesced, deterministic
//! derived-update bus.
//!
//! [`World::observe_changed`] subscribes a callback to *change detection* (as
//! opposed to the structural lifecycle triggers of [`crate::observer`]): when
//! [`World::flush_changed_observers`] runs, every registered observer is invoked
//! **at most once per flush**, carrying the coalesced set of entities whose
//! component changed since that observer last fired.
//!
//! This is the ECS half of §24.2: spatial-index sync, GPU-resource
//! invalidation, audio re-localisation and shadow invalidation all subscribe
//! here instead of polling, and the ECS guarantees the two properties the doc
//! requires of that bus:
//!
//! - **Coalescing.** Ten writes to one entity between two flushes yield a
//!   single delivery naming that entity once.
//! - **Deterministic replay** (配合 §14). Observers fire in registration order,
//!   and each delivery's entity set is sorted ascending by entity index, so a
//!   given input produces a bit-identical delivery sequence across runs and
//!   across thread counts.
//!
//! [`Schedule::run`](crate::schedule::Schedule::run) flushes automatically at the
//! end of a run, so registering an observer is enough for the common case; call
//! [`World::flush_changed_observers`] directly to control the cadence yourself.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::change::Tick;
use crate::component::Component;
use crate::component::ComponentId;
use crate::entity::Entity;
use crate::observer::ObserverId;
use crate::world::{World, WorldId};

/// The context handed to a change observer when it fires.
pub struct ChangeContext<'w> {
    /// The world the observed changes happened in, fully consistent and mutable
    /// so the callback can read the values or perform further changes.
    pub world: &'w mut World,
    /// The entities whose observed component changed since the previous flush,
    /// sorted ascending by entity index and deduplicated.
    changed: &'w [Entity],
}

impl<'w> ChangeContext<'w> {
    /// The coalesced, deterministically ordered changed-entity set.
    #[inline]
    pub fn changed(&self) -> &[Entity] {
        self.changed
    }

    /// Number of changed entities in this delivery.
    #[inline]
    pub fn len(&self) -> usize {
        self.changed.len()
    }

    /// Whether this delivery would be empty (never handed to a callback, but
    /// useful for symmetry with [`len`](Self::len)).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.changed.is_empty()
    }
}

/// Type-erased change-observer behavior so the registry can hold observers for
/// many component types in one ordered list.
trait ChangeObserver: Send + Sync {
    fn component_id(&self) -> ComponentId;
    /// Append entities from the exact write-side index.
    fn collect(
        &self,
        world: &World,
        cursor: &mut crate::change_index::ChangeCursor,
        last_run: Tick,
        this_run: Tick,
        out: &mut Vec<Entity>,
    );
    /// Invoke the user callback.
    fn fire(&mut self, ctx: &mut ChangeContext<'_>);
}

/// A change observer for one component type `T` and callback `C`.
struct TypedChangeObserver<T: Component, C> {
    callback: C,
    component: ComponentId,
    _marker: PhantomData<fn() -> T>,
}

impl<T, C> ChangeObserver for TypedChangeObserver<T, C>
where
    T: Component,
    C: FnMut(&mut ChangeContext<'_>) + Send + Sync + 'static,
{
    fn component_id(&self) -> ComponentId {
        self.component
    }

    fn collect(
        &self,
        world: &World,
        cursor: &mut crate::change_index::ChangeCursor,
        last_run: Tick,
        this_run: Tick,
        out: &mut Vec<Entity>,
    ) {
        world
            .change_tracker
            .collect(self.component, cursor, last_run, this_run, out);
        out.retain(|&entity| {
            world
                .get_ticks::<T>(entity)
                .is_some_and(|ticks| ticks.is_changed(last_run, this_run))
        });
    }

    fn fire(&mut self, ctx: &mut ChangeContext<'_>) {
        (self.callback)(ctx);
    }
}

/// One registered change observer plus its change-detection window.
struct ChangeSlot {
    id: ObserverId,
    /// End of the previous window: the observer fires for changes in
    /// `(last_run, this_run]` and then advances this to `this_run`.
    last_run: Tick,
    observer: Box<dyn ChangeObserver>,
    cursor: crate::change_index::ChangeCursor,
    pending: Vec<Entity>,
}

struct ChangeEntry {
    id: ObserverId,
    component: ComponentId,
    active: bool,
    slot: Option<ChangeSlot>,
}

/// The ordered change-observer registry (design §24.2).
///
/// Slots are never compacted: removal clears a slot in place so the *registration
/// order* — and therefore deterministic firing order — is preserved across
/// remove/re-add churn, and so an index captured while a callback runs stays
/// valid even if that callback registers or removes observers.
pub(crate) struct ChangeObservers {
    slots: Vec<ChangeEntry>,
    active: usize,
    pub(crate) flushing: bool,
    restore_generation: u64,
    restore_tick: Tick,
    restore_cursor: crate::change_index::ChangeCursor,
}

impl Default for ChangeObservers {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            active: 0,
            flushing: false,
            restore_generation: 0,
            restore_tick: Tick::ZERO,
            restore_cursor: crate::change_index::ChangeCursor::default(),
        }
    }
}

impl ChangeObservers {
    /// Whether no observers are currently registered.
    pub(crate) fn is_empty(&self) -> bool {
        self.active == 0
    }

    pub(crate) fn is_flushing(&self) -> bool {
        self.flushing
    }

    /// Number of slots ever occupied (including cleared ones).
    fn slot_count(&self) -> usize {
        self.slots.len()
    }

    fn push(&mut self, slot: ChangeSlot) {
        self.slots.push(ChangeEntry {
            id: slot.id,
            component: slot.observer.component_id(),
            active: true,
            slot: Some(slot),
        });
        self.active += 1;
    }

    fn take(&mut self, index: usize) -> Option<ChangeSlot> {
        let entry = self.slots.get_mut(index)?;
        if entry.active {
            entry.slot.take()
        } else {
            None
        }
    }

    fn put_back(&mut self, index: usize, slot: ChangeSlot) {
        let Some(entry) = self.slots.get_mut(index) else {
            return;
        };
        if entry.active && entry.id == slot.id && entry.slot.is_none() {
            entry.slot = Some(slot);
        }
    }

    pub(crate) fn remove(&mut self, id: ObserverId) -> bool {
        for entry in &mut self.slots {
            if entry.active && entry.id == id {
                entry.active = false;
                entry.slot = None;
                self.active -= 1;
                return true;
            }
        }
        false
    }

    pub(crate) fn component_for(&self, id: ObserverId) -> Option<ComponentId> {
        self.slots
            .iter()
            .find(|entry| entry.active && entry.id == id)
            .map(|entry| entry.component)
    }

    /// Invalidate delivery state after an authoritative storage restore.
    /// Batches from the abandoned timeline must not be replayed, while writes
    /// made after the restored baseline remain visible to the next flush.
    pub(crate) fn reset_after_restore(
        &mut self,
        tick: Tick,
        cursor: crate::change_index::ChangeCursor,
    ) {
        self.restore_generation = self.restore_generation.wrapping_add(1);
        self.restore_tick = tick;
        self.restore_cursor = cursor;
        // Restore keeps the snapshot's exact change tick. A write made
        // immediately afterwards is stamped with that same tick, so the
        // delivery window starts one logical tick earlier. The tracker was
        // reset above, so this cannot replay restored values.
        let baseline = Tick::new(tick.get().wrapping_sub(1));
        for entry in &mut self.slots {
            if let Some(slot) = entry.slot.as_mut() {
                slot.pending.clear();
                slot.last_run = baseline;
                slot.cursor = cursor;
            }
        }
    }
}

impl World {
    /// Subscribe `callback` to changes of component `T` (design §24.2).
    ///
    /// The callback fires only from [`flush_changed_observers`](World::flush_changed_observers)
    /// (or a [`Schedule`](crate::schedule::Schedule) run, which flushes for you),
    /// at most once per flush, with the entities whose `T` changed since the
    /// observer's previous delivery. `T` is registered on demand; the window is
    /// baselined to the current tick, then registration advances the tick so
    /// earlier writes are excluded and immediate subsequent writes are included.
    /// Returns a stable [`ObserverId`] accepted by both
    /// [`remove_change_observer`](World::remove_change_observer) and the shared
    /// [`remove_observer`](World::remove_observer).
    pub fn observe_changed<T, C>(&mut self, callback: C) -> ObserverId
    where
        T: Component,
        C: FnMut(&mut ChangeContext<'_>) + Send + Sync + 'static,
    {
        // Register `T` so `Changed<T>` resolves even for a world that never
        // touched it yet.
        self.components_mut().register::<T>();
        let component = self
            .components()
            .id_of::<T>()
            .expect("component registered");
        let cursor = self.change_tracker.activate(component);
        let id = self.alloc_observer_id();
        let last_run = self.change_tick();
        self.change_observers.push(ChangeSlot {
            id,
            last_run,
            observer: Box::new(TypedChangeObserver::<T, C> {
                callback,
                component,
                _marker: PhantomData,
            }),
            cursor,
            pending: Vec::new(),
        });
        self.increment_change_tick();
        id
    }

    /// Remove a change observer registered with
    /// [`observe_changed`](World::observe_changed). Returns whether it existed.
    ///
    /// Removing the observer that is *currently firing* takes effect after its
    /// callback returns, mirroring [`remove_observer`](World::remove_observer).
    pub fn remove_change_observer(&mut self, id: ObserverId) -> bool {
        let component = self.change_observers.component_for(id);
        let removed = self.change_observers.remove(id);
        if removed && let Some(component) = component {
            self.change_tracker.deactivate(component);
        }
        removed
    }

    /// Flush every change observer (design §24.2).
    ///
    /// Observers fire in registration order; each is invoked at most once, with
    /// its changed-entity set sorted ascending by entity index and deduplicated.
    /// Returns how many observers actually received a non-empty batch. A change
    /// made *by* a callback lands in the next flush (every batch is collected
    /// before callbacks run), so a callback cannot feed itself within one flush.
    /// A later callback may receive an entity whose component an earlier
    /// callback has removed; re-check current state before reading that value.
    /// If a callback panics, its batch and callbacks not yet run are retained
    /// for retry; successful callbacks are not replayed without new changes.
    /// Replacing the World inside a callback ends this delivery; the replacement
    /// retains its own observers and is not automatically flushed by this call.
    pub fn flush_changed_observers(&mut self) -> usize {
        // Effects may explicitly request a flush. Their writes belong to the
        // next delivery window and must not recursively re-enter this bus.
        if self.reactions.is_flushing() || self.change_observers.flushing {
            return 0;
        }
        if self.change_observers.is_empty() {
            self.flush_reactions();
            return 0;
        }
        let count = self.change_observers.slot_count();
        let world_id = self.id();
        let restore_generation = self.change_observers.restore_generation;
        self.change_observers.flushing = true;
        let flush = ChangeFlush {
            world: self,
            world_id,
        };
        let this_run = flush.world.change_tick();
        let mut fired = 0;
        for index in 0..count {
            let Some(slot) = flush.world.change_observers.take(index) else {
                continue;
            };
            let mut guard = ChangeSlotGuard {
                world: &mut *flush.world,
                world_id,
                index,
                slot: Some(slot),
                restore_generation,
            };
            let slot = guard.slot.as_mut().expect("owned change observer");
            // A previous flush may have left an undelivered batch here after a
            // callback panic. Collect only the new window into a separate
            // buffer so retry state is never re-filtered against a later
            // `last_run` and accidentally discarded.
            let mut fresh = Vec::new();
            slot.observer.collect(
                guard.world,
                &mut slot.cursor,
                slot.last_run,
                this_run,
                &mut fresh,
            );
            fresh.sort_unstable_by_key(|entity| (entity.index(), entity.generation()));
            fresh.dedup();
            slot.pending.extend(fresh);
            slot.pending
                .sort_unstable_by_key(|entity| (entity.index(), entity.generation()));
            slot.pending.dedup();
            slot.last_run = this_run;
        }
        // Snapshot every observer's batch before executing any callback. A
        // callback may remove components or write values, but cannot change the
        // already-collected delivery of a later observer in this flush.
        flush.world.increment_change_tick();
        for index in 0..count {
            let Some(slot) = flush.world.change_observers.take(index) else {
                continue;
            };
            let mut guard = ChangeSlotGuard {
                world: &mut *flush.world,
                world_id,
                index,
                slot: Some(slot),
                restore_generation,
            };
            let slot = guard.slot.as_mut().expect("owned change observer");
            if !slot.pending.is_empty() {
                let mut ctx = ChangeContext {
                    world: &mut *guard.world,
                    changed: &slot.pending,
                };
                slot.observer.fire(&mut ctx);
                // Preserve this batch if the callback unwinds, for an explicit
                // retry. Successful callbacks are never replayed on retry.
                slot.pending.clear();
                fired += 1;
                if guard.world.id() != world_id
                    || guard.world.change_observers.restore_generation != restore_generation
                {
                    return fired;
                }
            }
        }
        // Release the delivery barrier only after every source callback has
        // forwarded its batch. Explicit flushes inside callbacks are deferred.
        drop(flush);
        self.flush_reactions();
        fired
    }
}

struct ChangeSlotGuard<'w> {
    world: &'w mut World,
    world_id: WorldId,
    index: usize,
    slot: Option<ChangeSlot>,
    restore_generation: u64,
}

impl Drop for ChangeSlotGuard<'_> {
    fn drop(&mut self) {
        if self.world.id() == self.world_id
            && let Some(slot) = self.slot.take()
        {
            let mut slot = slot;
            if self.world.change_observers.restore_generation != self.restore_generation {
                slot.pending.clear();
                slot.last_run = Tick::new(
                    self.world
                        .change_observers
                        .restore_tick
                        .get()
                        .wrapping_sub(1),
                );
                slot.cursor = self.world.change_observers.restore_cursor;
            }
            self.world.change_observers.put_back(self.index, slot);
        }
    }
}

struct ChangeFlush<'w> {
    world: &'w mut World,
    world_id: WorldId,
}

impl Drop for ChangeFlush<'_> {
    fn drop(&mut self) {
        if self.world.id() == self.world_id {
            self.world.change_observers.flushing = false;
        }
    }
}
