//! The generational dense arena mapping [`AssetId`] to stored values.

use crate::error::AssetErrorId;
use crate::event::AssetEvent;
use crate::handle::{Handle, HandleInner};
use crate::id::{AssetId, AssetIndex, UntypedAssetId};
use crate::load_state::LoadState;
use alloc::sync::{Arc, Weak};
use alloc::vec::Vec;

/// A generational arena storing assets of type `A`.
///
/// Each slot is minted either eagerly with [`Assets::insert`] (data already in
/// hand) or ahead of time with [`Assets::reserve`] (a handle is needed *now*
/// while an async load is still in flight). Every mint hands back a strong
/// [`Handle`] and records a weak reference, so the arena knows when the last
/// external owner has dropped. Freed slots are recycled with a bumped
/// generation, which keeps ids dense without ever aliasing a stale
/// [`AssetId`] onto a new occupant.
///
/// A reserved slot carries a [`Payload::Pending`] until the load resolves to
/// either [`Assets::fulfill`] (success) or [`Assets::fail`] (error). Lifecycle
/// transitions enqueue [`AssetEvent`]s that downstream systems drain each frame
/// via [`Assets::drain_events`].
pub struct Assets<A> {
    slots: Vec<Slot<A>>,
    free: Vec<u32>,
    len: usize,
    events: Vec<AssetEvent<A>>,
}

/// One arena slot: a reuse counter plus an optional live entry.
struct Slot<A> {
    generation: u32,
    entry: Option<Entry<A>>,
}

/// A live arena entry: its current payload plus the weak reference used to
/// detect abandonment once the last strong [`Handle`] drops.
struct Entry<A> {
    payload: Payload<A>,
    handle: Weak<HandleInner>,
}

/// The data state of an occupied slot.
enum Payload<A> {
    /// Reserved for an in-flight load; no value yet.
    Pending,
    /// Loaded and ready to read.
    Ready(A),
    /// The load failed; the id resolves to the recorded reason in an
    /// [`ErrorRegistry`](crate::ErrorRegistry).
    Failed(AssetErrorId),
}

impl<A> Assets<A> {
    /// Creates an empty arena.
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            len: 0,
            events: Vec::new(),
        }
    }

    /// Inserts `value`, returning a strong [`Handle`] that keeps it alive and
    /// enqueuing an [`AssetEvent::Added`].
    pub fn insert(&mut self, value: A) -> Handle<A> {
        self.mint(Payload::Ready(value), true)
    }

    /// Reserves a slot for an asset that is still loading, returning a strong
    /// [`Handle`] with a stable id *before* the data exists. The slot starts in
    /// [`Payload::Pending`]; no event is emitted until it is resolved with
    /// [`Assets::fulfill`] or [`Assets::fail`].
    pub fn reserve(&mut self) -> Handle<A> {
        self.mint(Payload::Pending, false)
    }

    /// Mints a new occupied slot with `payload`, optionally emitting
    /// [`AssetEvent::Added`], and returns a strong handle to it.
    fn mint(&mut self, payload: Payload<A>, emit_added: bool) -> Handle<A> {
        let index = match self.free.pop() {
            Some(index) => index,
            None => {
                let index = u32::try_from(self.slots.len()).expect("slot index fits u32");
                self.slots.push(Slot {
                    generation: 0,
                    entry: None,
                });
                index
            }
        };

        let slot = &mut self.slots[index as usize];
        let asset_index = AssetIndex::from_parts(index, slot.generation);
        let untyped = UntypedAssetId::new(asset_index);
        let arc = HandleInner::new_arc(untyped);
        slot.entry = Some(Entry {
            payload,
            handle: Arc::downgrade(&arc),
        });
        self.len += 1;

        if emit_added {
            self.events.push(AssetEvent::Added {
                id: AssetId::new(asset_index),
            });
        }
        Handle::from_arc(arc)
    }

    /// Supplies the loaded `value` for a previously [`reserve`](Assets::reserve)d
    /// (or already-present) slot and marks it ready. Emits
    /// [`AssetEvent::Added`] when the slot was pending or failed, or
    /// [`AssetEvent::Modified`] when it already held a ready value (a reload).
    /// Returns `false` if `id` is stale.
    pub fn fulfill(&mut self, id: AssetId<A>, value: A) -> bool {
        let Some(slot_index) = self.resolve(id.index()) else {
            return false;
        };
        let entry = self.slots[slot_index]
            .entry
            .as_mut()
            .expect("resolved slot is occupied");
        let was_ready = matches!(entry.payload, Payload::Ready(_));
        entry.payload = Payload::Ready(value);
        self.events.push(if was_ready {
            AssetEvent::Modified { id }
        } else {
            AssetEvent::Added { id }
        });
        true
    }

    /// Marks a previously [`reserve`](Assets::reserve)d (or present) slot as
    /// failed, referencing a recorded `error`, and enqueues
    /// [`AssetEvent::Failed`]. The slot stays occupied (the handle remains
    /// valid) so the failure is observable until the handle drops or the slot
    /// is removed. Returns `false` if `id` is stale.
    pub fn fail(&mut self, id: AssetId<A>, error: AssetErrorId) -> bool {
        let Some(slot_index) = self.resolve(id.index()) else {
            return false;
        };
        let entry = self.slots[slot_index]
            .entry
            .as_mut()
            .expect("resolved slot is occupied");
        entry.payload = Payload::Failed(error);
        self.events.push(AssetEvent::Failed { id, error });
        true
    }

    /// Returns a shared reference to the asset `id` points at, or `None` if the
    /// id is stale, pending, or failed (only [`Payload::Ready`] yields a value).
    #[must_use]
    pub fn get(&self, id: AssetId<A>) -> Option<&A> {
        let slot_index = self.resolve(id.index())?;
        match &self.slots[slot_index].entry.as_ref()?.payload {
            Payload::Ready(value) => Some(value),
            Payload::Pending | Payload::Failed(_) => None,
        }
    }

    /// Returns a mutable reference to the ready asset `id` points at and
    /// enqueues an [`AssetEvent::Modified`], or `None` if the id is stale,
    /// pending, or failed.
    pub fn get_mut(&mut self, id: AssetId<A>) -> Option<&mut A> {
        let slot_index = self.resolve(id.index())?;
        // Confirm readiness before taking the long-lived &mut, so the event
        // push below does not overlap the returned borrow.
        let is_ready = matches!(
            self.slots[slot_index].entry.as_ref()?.payload,
            Payload::Ready(_)
        );
        if !is_ready {
            return None;
        }
        self.events.push(AssetEvent::Modified { id });
        match &mut self.slots[slot_index].entry.as_mut()?.payload {
            Payload::Ready(value) => Some(value),
            Payload::Pending | Payload::Failed(_) => None,
        }
    }

    /// Whether the slot `id` points at is occupied (pending, ready, or failed).
    #[must_use]
    pub fn contains(&self, id: AssetId<A>) -> bool {
        self.resolve(id.index()).is_some()
    }

    /// Whether the asset `id` points at is loaded and ready to read.
    #[must_use]
    pub fn is_ready(&self, id: AssetId<A>) -> bool {
        self.get(id).is_some()
    }

    /// The [`LoadState`] of the slot `id` points at: `NotLoaded` if stale,
    /// `Loading` if pending, `Loaded` if ready, or `Failed` with the recorded
    /// error id.
    #[must_use]
    pub fn load_state(&self, id: AssetId<A>) -> LoadState {
        let Some(slot_index) = self.resolve(id.index()) else {
            return LoadState::NotLoaded;
        };
        match &self.slots[slot_index]
            .entry
            .as_ref()
            .expect("resolved slot is occupied")
            .payload
        {
            Payload::Pending => LoadState::Loading,
            Payload::Ready(_) => LoadState::Loaded,
            Payload::Failed(error) => LoadState::Failed(*error),
        }
    }

    /// Removes the slot `id` points at and enqueues an [`AssetEvent::Removed`],
    /// returning the stored value if it was [`Payload::Ready`] (pending/failed
    /// slots are freed and return `None`). Returns `None` if `id` is stale.
    pub fn remove(&mut self, id: AssetId<A>) -> Option<A> {
        let slot_index = self.resolve(id.index())?;
        let payload = self.take_slot(slot_index)?;
        self.events.push(AssetEvent::Removed { id });
        match payload {
            Payload::Ready(value) => Some(value),
            Payload::Pending | Payload::Failed(_) => None,
        }
    }

    /// Removes every slot whose last strong [`Handle`] has dropped — regardless
    /// of payload state, so abandoned in-flight loads are reclaimed too —
    /// enqueuing an [`AssetEvent::Removed`] for each, and returns how many were
    /// reclaimed.
    pub fn remove_unused(&mut self) -> usize {
        let mut reclaimed = 0;
        for index in 0..self.slots.len() {
            let abandoned = self.slots[index]
                .entry
                .as_ref()
                .is_some_and(|entry| entry.handle.strong_count() == 0);
            if abandoned {
                let generation = self.slots[index].generation;
                let slot_index = u32::try_from(index).expect("slot index fits u32");
                let asset_index = AssetIndex::from_parts(slot_index, generation);
                if self.take_slot(index).is_some() {
                    self.events.push(AssetEvent::Removed {
                        id: AssetId::new(asset_index),
                    });
                    reclaimed += 1;
                }
            }
        }
        reclaimed
    }

    /// The number of occupied slots (pending, ready, and failed combined).
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the arena holds no occupied slots.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Iterates over `(id, &value)` pairs for every *ready* asset in slot order.
    /// Pending and failed slots are skipped (they have no value to yield).
    pub fn iter(&self) -> impl Iterator<Item = (AssetId<A>, &A)> {
        self.slots.iter().enumerate().filter_map(|(index, slot)| {
            let entry = slot.entry.as_ref()?;
            let Payload::Ready(value) = &entry.payload else {
                return None;
            };
            let slot_index = u32::try_from(index).expect("slot index fits u32");
            let asset_index = AssetIndex::from_parts(slot_index, slot.generation);
            Some((AssetId::new(asset_index), value))
        })
    }

    /// Takes all queued [`AssetEvent`]s, leaving the queue empty.
    pub fn drain_events(&mut self) -> Vec<AssetEvent<A>> {
        core::mem::take(&mut self.events)
    }

    /// The number of events currently queued (not yet drained).
    #[must_use]
    pub fn pending_event_count(&self) -> usize {
        self.events.len()
    }

    /// Resolves an [`AssetIndex`] to a live slot position, checking that the
    /// generation matches and the slot still holds an entry.
    fn resolve(&self, index: AssetIndex) -> Option<usize> {
        let slot_index = index.index() as usize;
        let slot = self.slots.get(slot_index)?;
        if slot.generation == index.generation() && slot.entry.is_some() {
            Some(slot_index)
        } else {
            None
        }
    }

    /// Empties a slot, bumps its generation, records it as free, and returns the
    /// evicted payload.
    fn take_slot(&mut self, slot_index: usize) -> Option<Payload<A>> {
        let slot = self.slots.get_mut(slot_index)?;
        let entry = slot.entry.take()?;
        slot.generation = slot.generation.wrapping_add(1);
        self.free
            .push(u32::try_from(slot_index).expect("slot index fits u32"));
        self.len -= 1;
        Some(entry.payload)
    }
}

impl<A> Default for Assets<A> {
    fn default() -> Self {
        Self::new()
    }
}
