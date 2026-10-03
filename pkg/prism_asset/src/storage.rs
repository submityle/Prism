//! The generational dense arena mapping [`AssetId`] to stored values.

use crate::event::AssetEvent;
use crate::handle::{Handle, HandleInner};
use crate::id::{AssetId, AssetIndex, UntypedAssetId};
use alloc::sync::{Arc, Weak};
use alloc::vec::Vec;

/// A generational arena storing assets of type `A`.
///
/// Each insert hands back a strong [`Handle`] and records a weak reference, so
/// the arena knows when the last external owner has dropped. Freed slots are
/// recycled with a bumped generation, which keeps ids dense without ever
/// aliasing a stale [`AssetId`] onto a new occupant. Mutations and lifecycle
/// transitions enqueue [`AssetEvent`]s that downstream systems drain each
/// frame via [`Assets::drain_events`].
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

/// A live asset together with the weak reference used to detect abandonment.
struct Entry<A> {
    value: A,
    handle: Weak<HandleInner>,
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
            value,
            handle: Arc::downgrade(&arc),
        });
        self.len += 1;

        self.events.push(AssetEvent::Added {
            id: AssetId::new(asset_index),
        });
        Handle::from_arc(arc)
    }

    /// Returns a shared reference to the asset `id` points at, or `None` if the
    /// id is stale or was removed.
    #[must_use]
    pub fn get(&self, id: AssetId<A>) -> Option<&A> {
        let slot_index = self.resolve(id.index())?;
        self.slots[slot_index]
            .entry
            .as_ref()
            .map(|entry| &entry.value)
    }

    /// Returns a mutable reference to the asset `id` points at and enqueues an
    /// [`AssetEvent::Modified`], or `None` if the id is stale or was removed.
    pub fn get_mut(&mut self, id: AssetId<A>) -> Option<&mut A> {
        let slot_index = self.resolve(id.index())?;
        let value = self.slots[slot_index]
            .entry
            .as_mut()
            .map(|entry| &mut entry.value)?;
        self.events.push(AssetEvent::Modified { id });
        Some(value)
    }

    /// Whether a live asset exists for `id`.
    #[must_use]
    pub fn contains(&self, id: AssetId<A>) -> bool {
        self.resolve(id.index()).is_some()
    }

    /// Removes the asset `id` points at, returning its value and enqueuing an
    /// [`AssetEvent::Removed`]. Returns `None` if the id is stale.
    pub fn remove(&mut self, id: AssetId<A>) -> Option<A> {
        let slot_index = self.resolve(id.index())?;
        let value = self.take_slot(slot_index)?;
        self.events.push(AssetEvent::Removed { id });
        Some(value)
    }

    /// Removes every asset whose last strong [`Handle`] has dropped, enqueuing
    /// an [`AssetEvent::Removed`] for each, and returns how many were reclaimed.
    pub fn remove_unused(&mut self) -> usize {
        let mut reclaimed = 0;
        for index in 0..self.slots.len() {
            let abandoned = self.slots[index]
                .entry
                .as_ref()
                .is_some_and(|entry| entry.handle.strong_count() == 0);
            if abandoned {
                let generation = self.slots[index].generation;
                let slot_index =
                    u32::try_from(index).expect("slot index fits u32");
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

    /// The number of live assets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the arena holds no live assets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Iterates over `(id, &value)` pairs for every live asset in slot order.
    pub fn iter(&self) -> impl Iterator<Item = (AssetId<A>, &A)> {
        self.slots.iter().enumerate().filter_map(|(index, slot)| {
            slot.entry.as_ref().map(|entry| {
                let slot_index = u32::try_from(index).expect("slot index fits u32");
                let asset_index = AssetIndex::from_parts(slot_index, slot.generation);
                (AssetId::new(asset_index), &entry.value)
            })
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
    /// evicted value.
    fn take_slot(&mut self, slot_index: usize) -> Option<A> {
        let slot = self.slots.get_mut(slot_index)?;
        let entry = slot.entry.take()?;
        slot.generation = slot.generation.wrapping_add(1);
        self.free
            .push(u32::try_from(slot_index).expect("slot index fits u32"));
        self.len -= 1;
        Some(entry.value)
    }
}

impl<A> Default for Assets<A> {
    fn default() -> Self {
        Self::new()
    }
}
