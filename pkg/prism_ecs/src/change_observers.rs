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
use crate::entity::Entity;
use crate::observer::ObserverId;
use crate::query::{Changed, QueryState};
use crate::world::World;

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
    /// Build the per-observer change query lazily (needs `&mut World`).
    fn ensure_state(&mut self, world: &mut World);
    /// Append the entities whose component changed in `(last_run, this_run]`.
    fn collect(&self, world: &World, last_run: Tick, this_run: Tick, out: &mut Vec<Entity>);
    /// Invoke the user callback.
    fn fire(&mut self, ctx: &mut ChangeContext<'_>);
}

/// A change observer for one component type `T` and callback `C`.
struct TypedChangeObserver<T: Component, C> {
    callback: C,
    /// `Entity` rows whose `T` changed in the observer's current window.
    /// Built lazily on the first flush so registration needs no world borrow.
    state: Option<QueryState<Entity, Changed<T>>>,
    _marker: PhantomData<fn() -> T>,
}

impl<T, C> ChangeObserver for TypedChangeObserver<T, C>
where
    T: Component,
    C: FnMut(&mut ChangeContext<'_>) + Send + Sync + 'static,
{
    fn ensure_state(&mut self, world: &mut World) {
        if self.state.is_none() {
            self.state = Some(world.query_filtered::<Entity, Changed<T>>());
        }
    }

    fn collect(&self, world: &World, last_run: Tick, this_run: Tick, out: &mut Vec<Entity>) {
        let Some(state) = &self.state else {
            return;
        };
        let world_ptr = (world as *const World).cast_mut();
        // SAFETY: `Entity` is a read-only data term, so no `&mut` fetch is
        // formed and the shared `world` borrow is sufficient. `state` was built
        // from this same world (the `observe_changed` call site), and the
        // `WorldId` guard inside `QueryState` would reject any other.
        let iter = unsafe { state.iter_from_ptr(world_ptr, last_run, this_run) };
        out.extend(iter);
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
}

/// The ordered change-observer registry (design §24.2).
///
/// Slots are never compacted: removal clears a slot in place so the *registration
/// order* — and therefore deterministic firing order — is preserved across
/// remove/re-add churn, and so an index captured while a callback runs stays
/// valid even if that callback registers or removes observers.
#[derive(Default)]
pub(crate) struct ChangeObservers {
    slots: Vec<Option<ChangeSlot>>,
}

impl ChangeObservers {
    /// Whether no observers are currently registered.
    pub(crate) fn is_empty(&self) -> bool {
        self.slots.iter().all(Option::is_none)
    }

    /// Number of slots ever occupied (including cleared ones).
    fn slot_count(&self) -> usize {
        self.slots.len()
    }

    fn push(&mut self, slot: ChangeSlot) {
        self.slots.push(Some(slot));
    }

    fn take(&mut self, index: usize) -> Option<ChangeSlot> {
        self.slots.get_mut(index)?.take()
    }

    fn put_back(&mut self, index: usize, slot: ChangeSlot) {
        self.slots[index] = Some(slot);
    }

    pub(crate) fn remove(&mut self, id: ObserverId) -> bool {
        for slot in &mut self.slots {
            if slot.as_ref().is_some_and(|s| s.id == id) {
                *slot = None;
                return true;
            }
        }
        false
    }
}

impl World {
    /// Subscribe `callback` to changes of component `T` (design §24.2).
    ///
    /// The callback fires only from [`flush_changed_observers`](World::flush_changed_observers)
    /// (or a [`Schedule`](crate::schedule::Schedule) run, which flushes for you),
    /// at most once per flush, with the entities whose `T` changed since the
    /// observer's previous delivery. `T` is registered on demand; the window is
    /// baselined to the current tick so changes made *before* registration do
    /// not fire it. Returns a stable [`ObserverId`] accepted by both
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
        let id = self.alloc_observer_id();
        let last_run = self.change_tick();
        self.change_observers.push(ChangeSlot {
            id,
            last_run,
            observer: Box::new(TypedChangeObserver::<T, C> {
                callback,
                state: None,
                _marker: PhantomData,
            }),
        });
        id
    }

    /// Remove a change observer registered with
    /// [`observe_changed`](World::observe_changed). Returns whether it existed.
    ///
    /// Removing the observer that is *currently firing* takes effect after its
    /// callback returns, mirroring [`remove_observer`](World::remove_observer).
    pub fn remove_change_observer(&mut self, id: ObserverId) -> bool {
        self.change_observers.remove(id)
    }

    /// Flush every change observer (design §24.2).
    ///
    /// Observers fire in registration order; each is invoked at most once, with
    /// its changed-entity set sorted ascending by entity index and deduplicated.
    /// Returns how many observers actually received a non-empty batch. A change
    /// made *by* a callback lands in the next flush (the window advances before
    /// the callback runs), so a callback cannot feed itself within one flush.
    pub fn flush_changed_observers(&mut self) -> usize {
        let count = self.change_observers.slot_count();
        if count == 0 {
            return 0;
        }
        let this_run = self.change_tick();
        let mut batch: Vec<Entity> = Vec::new();
        let mut fired = 0;
        for index in 0..count {
            // Move the slot out so the callback can take `&mut World` while its
            // own observer list lives inside that world; restore it afterwards.
            let Some(mut slot) = self.change_observers.take(index) else {
                continue;
            };
            slot.observer.ensure_state(self);

            batch.clear();
            slot.observer
                .collect(&*self, slot.last_run, this_run, &mut batch);
            // Deterministic delivery: ascending entity index, no duplicates even
            // if a term could match a row twice.
            batch.sort_unstable_by_key(|e| e.index());
            batch.dedup();
            slot.last_run = this_run;

            if !batch.is_empty() {
                // Advance the tick before the callback runs so any component it
                // writes is stamped *strictly later* than the window we just
                // consumed. Without this a callback that touches the component
                // it observes would write at exactly `this_run`, which the next
                // window `(this_run, ..]` would then shadow — losing the change
                // permanently. With it, callback writes are delivered next flush.
                self.increment_change_tick();
                let mut ctx = ChangeContext {
                    world: &mut *self,
                    changed: &batch,
                };
                slot.observer.fire(&mut ctx);
                fired += 1;
            }
            self.change_observers.put_back(index, slot);
        }
        fired
    }
}
