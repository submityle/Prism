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
use alloc::collections::BTreeMap;
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
    ) -> crate::change_index::ChangeIndexStats;
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
    ) -> crate::change_index::ChangeIndexStats {
        let stats = world
            .change_tracker
            .collect(self.component, cursor, last_run, this_run, out);
        out.retain(|&entity| {
            world
                .get_ticks::<T>(entity)
                .is_some_and(|ticks| ticks.is_changed(last_run, this_run))
        });
        stats
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
    registration_order: u64,
    component: ComponentId,
    active: bool,
    slot: Option<ChangeSlot>,
}

/// Test-only operation counts for candidate selection and delivery.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct FlushStats {
    revision_checks: usize,
    slot_visits: usize,
    collections: usize,
    candidate_slots: usize,
    log_batches: usize,
    log_events: usize,
    /// Auditable upper bound: current collection calls `get_ticks` once per
    /// event passed through the retain filter; derived from indexed events.
    retain_tick_lookup_upper_bound: usize,
    idle_fast_path: bool,
}

/// The ordered change-observer registry (design §24.2).
///
/// Empty slots are reused outside an in-progress flush. Slots removed by a
/// callback are deferred until the flush guard drops, so candidate indices
/// remain stable for the entire delivery. Registration order is stored
/// separately from the slot index and remains deterministic across reuse.
pub(crate) struct ChangeObservers {
    slots: Vec<ChangeEntry>,
    free_indices: Vec<usize>,
    deferred_free_indices: Vec<usize>,
    next_registration_order: u64,
    /// Active slots in registration order. Unlike `slots`, this never contains
    /// removed historical entries and is the source for a flush's candidate
    /// snapshot.
    active_indices: Vec<usize>,
    /// Active slots grouped by the component they observe. Component ids are
    /// dense within a World, so a vector avoids map lookup in the common path.
    active_by_component: Vec<Vec<usize>>,
    /// Active slots with an already-collected batch that still needs delivery
    /// after a callback panic. Kept sorted by registration slot so retry order
    /// remains deterministic without scanning every active observer.
    pending_indices: Vec<usize>,
    /// Observer ids share one namespace with lifecycle observers and are not a
    /// dense change-observer index.
    slot_by_id: BTreeMap<ObserverId, usize>,
    active: usize,
    pub(crate) flushing: bool,
    restore_generation: u64,
    restore_tick: Tick,
    restore_cursor: crate::change_index::ChangeCursor,
    /// Published before callbacks, valid only after a fully successful delivery.
    /// Registration/removal/restore and unwind invalidate it. Callback writes
    /// are newer than this snapshot and therefore cannot be acknowledged here.
    clean_revision: Option<u64>,
    /// Last skipped collection window and the old slots it applies to. New
    /// registrations retain their own (later) baseline instead of inheriting it.
    idle_baseline: Option<(Tick, u64)>,
    #[cfg(test)]
    stats: FlushStats,
}

impl Default for ChangeObservers {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free_indices: Vec::new(),
            deferred_free_indices: Vec::new(),
            next_registration_order: 0,
            active_indices: Vec::new(),
            active_by_component: Vec::new(),
            pending_indices: Vec::new(),
            slot_by_id: BTreeMap::new(),
            active: 0,
            flushing: false,
            restore_generation: 0,
            restore_tick: Tick::ZERO,
            restore_cursor: crate::change_index::ChangeCursor::default(),
            clean_revision: None,
            idle_baseline: None,
            #[cfg(test)]
            stats: FlushStats::default(),
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

    fn push(&mut self, slot: ChangeSlot) {
        if !self.flushing {
            self.compact_if_sparse();
        }
        let registration_order = self.next_registration_order;
        self.next_registration_order = self
            .next_registration_order
            .checked_add(1)
            .expect("change observer registration order exhausted");
        let id = slot.id;
        let component = slot.observer.component_id();
        let entry = ChangeEntry {
            id,
            registration_order,
            component,
            active: true,
            slot: Some(slot),
        };
        let reusable = self.free_indices.pop().or_else(|| {
            (!self.flushing)
                .then(|| self.deferred_free_indices.pop())
                .flatten()
        });
        let index = if let Some(index) = reusable {
            self.slots[index] = entry;
            index
        } else {
            let index = self.slots.len();
            self.slots.push(entry);
            index
        };
        self.active_indices.push(index);
        let component_index = component.index() as usize;
        if self.active_by_component.len() <= component_index {
            self.active_by_component
                .resize_with(component_index + 1, Vec::new);
        }
        self.active_by_component[component_index].push(index);
        let previous = self.slot_by_id.insert(id, index);
        debug_assert!(previous.is_none(), "observer id registered twice");
        self.active += 1;
        self.clean_revision = None;
    }

    fn active_candidates(&self, dirty_components: &[ComponentId]) -> Vec<usize> {
        debug_assert_eq!(self.active_indices.len(), self.active);
        debug_assert_eq!(
            self.active_by_component.iter().map(Vec::len).sum::<usize>(),
            self.active
        );
        let mut candidates = self.pending_indices.clone();
        for component in dirty_components {
            if let Some(indices) = self.active_by_component.get(component.index() as usize) {
                candidates.extend_from_slice(indices);
            }
        }
        candidates.sort_unstable_by_key(|&index| self.slots[index].registration_order);
        candidates.dedup();
        candidates
    }

    fn mark_pending(&mut self, index: usize) {
        let order = self.slots[index].registration_order;
        match self
            .pending_indices
            .binary_search_by_key(&order, |&pending| self.slots[pending].registration_order)
        {
            Ok(_) => {}
            Err(position) => self.pending_indices.insert(position, index),
        }
    }

    fn clear_pending(&mut self, index: usize) {
        let order = self.slots[index].registration_order;
        if let Ok(position) = self
            .pending_indices
            .binary_search_by_key(&order, |&pending| self.slots[pending].registration_order)
        {
            self.pending_indices.remove(position);
        }
    }

    fn compact_if_sparse(&mut self) {
        debug_assert!(!self.flushing);
        let retain_limit = self.active.saturating_mul(2).saturating_add(64);
        if self.slots.len() <= retain_limit {
            return;
        }

        let old_slots = core::mem::take(&mut self.slots);
        let mut old_slots: Vec<_> = old_slots.into_iter().map(Some).collect();
        let mut remap = alloc::vec![usize::MAX; old_slots.len()];
        let mut compacted = Vec::with_capacity(self.active);
        for old_index in self.active_indices.iter().copied() {
            let entry = old_slots[old_index]
                .take()
                .expect("active observer slot must be unique");
            debug_assert!(entry.active);
            remap[old_index] = compacted.len();
            compacted.push(entry);
        }

        self.active_indices.clear();
        self.active_indices.extend(0..compacted.len());
        for indices in &mut self.active_by_component {
            indices.clear();
        }
        for (index, entry) in compacted.iter().enumerate() {
            self.active_by_component[entry.component.index() as usize].push(index);
        }
        for index in self.slot_by_id.values_mut() {
            *index = remap[*index];
        }
        for index in &mut self.pending_indices {
            *index = remap[*index];
        }

        self.slots = compacted;
        self.free_indices.clear();
        self.deferred_free_indices.clear();
        self.slots.shrink_to_fit();
        self.free_indices.shrink_to_fit();
        self.deferred_free_indices.shrink_to_fit();
        self.active_indices.shrink_to_fit();
        self.pending_indices.shrink_to_fit();
        for indices in &mut self.active_by_component {
            indices.shrink_to_fit();
        }
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
        let Some(index) = self.slot_by_id.remove(&id) else {
            return false;
        };
        let (component, registration_order) = {
            let entry = self
                .slots
                .get_mut(index)
                .expect("active observer index must name a slot");
            debug_assert!(entry.active && entry.id == id);
            let component = entry.component;
            let registration_order = entry.registration_order;
            entry.active = false;
            entry.slot = None;
            (component, registration_order)
        };

        let active_position = self
            .active_indices
            .binary_search_by_key(&registration_order, |&active_index| {
                self.slots[active_index].registration_order
            })
            .expect("active observer missing from registration-order index");
        self.active_indices.remove(active_position);
        let component_index = component.index() as usize;
        let component_position = self.active_by_component[component_index]
            .binary_search_by_key(&registration_order, |&component_index| {
                self.slots[component_index].registration_order
            })
            .expect("active observer missing from component index");
        self.active_by_component[component_index].remove(component_position);
        debug_assert_eq!(registration_order, self.slots[index].registration_order);
        self.clear_pending(index);
        if self.flushing {
            self.deferred_free_indices.push(index);
        } else {
            self.free_indices.push(index);
        }

        self.active -= 1;
        self.clean_revision = None;
        if !self.flushing {
            self.compact_if_sparse();
        }
        true
    }

    pub(crate) fn component_for(&self, id: ObserverId) -> Option<ComponentId> {
        let index = *self.slot_by_id.get(&id)?;
        let entry = self.slots.get(index)?;
        (entry.active && entry.id == id).then_some(entry.component)
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
        self.clean_revision = None;
        self.idle_baseline = None;
        self.pending_indices.clear();
        // Restore keeps the snapshot's exact change tick. A write made
        // immediately afterwards is stamped with that same tick, so the
        // delivery window starts one logical tick earlier. The tracker was
        // reset above, so this cannot replay restored values.
        let baseline = Tick::new(tick.get().wrapping_sub(1));
        for &index in &self.active_indices {
            if let Some(slot) = self.slots[index].slot.as_mut() {
                slot.pending.clear();
                slot.last_run = baseline;
                slot.cursor = cursor;
            }
        }
    }

    fn finish_flush(&mut self, compact: bool) {
        debug_assert!(self.flushing);
        self.flushing = false;
        if compact {
            self.compact_if_sparse();
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
    /// After a successful delivery, an unchanged exact write-index revision
    /// skips both slot passes in O(1). Removed slots are reused or compacted
    /// without changing registration order. This still advances the delivery
    /// tick and settles pending reaction effects.
    pub fn flush_changed_observers(&mut self) -> usize {
        // Effects may explicitly request a flush. Their writes belong to the
        // next delivery window and must not recursively re-enter this bus.
        if self.reactions.is_flushing() || self.change_observers.flushing {
            return 0;
        }
        #[cfg(test)]
        {
            self.change_observers.stats = FlushStats::default();
        }
        self.change_observers.compact_if_sparse();
        if self.change_observers.is_empty() {
            self.flush_reactions();
            return 0;
        }
        let registration_cutoff = self.change_observers.next_registration_order;
        let revision = self.change_tracker.revision();
        #[cfg(test)]
        {
            self.change_observers.stats.revision_checks += 1;
        }
        if revision.is_some() && revision == self.change_observers.clean_revision {
            self.change_observers.idle_baseline = Some((self.change_tick(), registration_cutoff));
            #[cfg(test)]
            {
                self.change_observers.stats.idle_fast_path = true;
            }
            // Preserve the old window boundary even without collecting slots:
            // subsequent writes must start a new tick batch in ChangeTracker.
            self.increment_change_tick();
            self.flush_reactions();
            return 0;
        }
        self.change_observers.clean_revision = None;
        let dirty_components = self.change_tracker.take_dirty_components();
        let candidates = self.change_observers.active_candidates(&dirty_components);
        #[cfg(test)]
        {
            self.change_observers.stats.candidate_slots = candidates.len();
        }
        let idle_baseline = self.change_observers.idle_baseline;
        let world_id = self.id();
        let restore_generation = self.change_observers.restore_generation;
        self.change_observers.flushing = true;
        let mut flush = ChangeFlush {
            world: self,
            world_id,
            completed: false,
        };
        let this_run = flush.world.change_tick();
        let mut fired = 0;
        for &index in &candidates {
            #[cfg(test)]
            {
                flush.world.change_observers.stats.slot_visits += 1;
            }
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
            if let Some((tick, old_cutoff)) = idle_baseline
                && guard.world.change_observers.slots[index].registration_order < old_cutoff
            {
                slot.last_run = tick;
            }
            #[cfg(test)]
            {
                guard.world.change_observers.stats.collections += 1;
            }
            // A previous flush may have left an undelivered batch here after a
            // callback panic. Collect only the new window into a separate
            // buffer so retry state is never re-filtered against a later
            // `last_run` and accidentally discarded.
            let mut fresh = Vec::new();
            let _collect_stats = slot.observer.collect(
                guard.world,
                &mut slot.cursor,
                slot.last_run,
                this_run,
                &mut fresh,
            );
            #[cfg(test)]
            {
                guard.world.change_observers.stats.log_batches += _collect_stats.batches;
                guard.world.change_observers.stats.log_events += _collect_stats.events;
                guard
                    .world
                    .change_observers
                    .stats
                    .retain_tick_lookup_upper_bound += _collect_stats.events;
            }
            fresh.sort_unstable_by_key(|entity| (entity.index(), entity.generation()));
            fresh.dedup();
            slot.pending.extend(fresh);
            slot.pending
                .sort_unstable_by_key(|entity| (entity.index(), entity.generation()));
            slot.pending.dedup();
            if !slot.pending.is_empty() {
                guard.world.change_observers.mark_pending(index);
            }
            slot.last_run = this_run;
        }
        flush.world.change_observers.idle_baseline = None;
        // Publish only the pre-callback revision. Mutations of the registry in
        // callbacks invalidate it; writes change the tracker revision. Unwind
        // clears it so pending deliveries retry even without any new write.
        flush.world.change_observers.clean_revision = revision;
        // Snapshot every observer's batch before executing any callback. A
        // callback may remove components or write values, but cannot change the
        // already-collected delivery of a later observer in this flush.
        flush.world.increment_change_tick();
        for &index in &candidates {
            #[cfg(test)]
            {
                flush.world.change_observers.stats.slot_visits += 1;
            }
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
                // A callback may replace or restore the World. In that case
                // the world no longer refers to the timeline from which this
                // slot was collected; never clear a pending index in the
                // replacement registry using the old slot index.
                fired += 1;
                if guard.world.id() != world_id
                    || guard.world.change_observers.restore_generation != restore_generation
                {
                    return fired;
                }
                slot.pending.clear();
                guard.world.change_observers.clear_pending(index);
            }
        }
        // Release the delivery barrier only after every source callback has
        // forwarded its batch. Explicit flushes inside callbacks are deferred.
        flush.completed = true;
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
    completed: bool,
}

impl Drop for ChangeFlush<'_> {
    fn drop(&mut self) {
        if self.world.id() == self.world_id {
            if !self.completed {
                self.world.change_observers.clean_revision = None;
            }
            self.world.change_observers.finish_flush(self.completed);
        }
    }
}

#[cfg(test)]
#[path = "change_observers_fast_path_tests.rs"]
mod fast_path_tests;
