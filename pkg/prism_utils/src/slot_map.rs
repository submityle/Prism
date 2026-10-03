//! A generational slot map (`SlotMap`): stable, safely-invalidating handles.
//!
//! Each insertion returns a [`SlotKey`] carrying both a slot `index` and a
//! `generation` counter. Removing an element bumps the slot's generation, so a
//! stale [`SlotKey`] held across a remove+reuse cycle no longer matches and
//! lookups return `None` instead of aliasing the new occupant. This is the
//! "代数失效" (generational invalidation) guarantee that underpins safe asset
//! and entity handles. The implementation is fully safe.

extern crate alloc;

use alloc::vec::Vec;

/// Sentinel meaning "no next free slot" in the vacant free list.
const NO_FREE: u32 = u32::MAX;

/// A stable, copyable handle into a [`SlotMap`].
///
/// A key stays valid only while its slot holds the matching `generation`. Once
/// the referenced element is removed, the slot's generation advances and this
/// key becomes permanently stale.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SlotKey {
    index: u32,
    generation: u32,
}

impl SlotKey {
    /// Raw slot index component of the handle.
    pub const fn index(self) -> u32 {
        self.index
    }

    /// Generation component of the handle.
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

/// Per-slot payload: either a live value or a link in the vacant free list.
enum Content<T> {
    /// Slot currently holds a value.
    Occupied(T),
    /// Slot is free; `next_free` points at the next vacant slot (or [`NO_FREE`]).
    Vacant { next_free: u32 },
}

/// One slot: a generation stamp plus its current content.
struct Slot<T> {
    generation: u32,
    content: Content<T>,
}

/// A generational slot map storing values of type `T`.
///
/// Freed slots are recycled through an internal free list, and each reuse bumps
/// the slot generation so previously handed-out [`SlotKey`]s cannot alias the
/// new occupant. Iteration order follows ascending slot index and is stable
/// for a given sequence of operations.
pub struct SlotMap<T> {
    slots: Vec<Slot<T>>,
    free_head: u32,
    len: usize,
}

impl<T> SlotMap<T> {
    /// Create an empty slot map.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_head: NO_FREE,
            len: 0,
        }
    }

    /// Create an empty slot map with pre-reserved slot capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            slots: Vec::with_capacity(cap),
            free_head: NO_FREE,
            len: 0,
        }
    }

    /// Insert `value`, returning a fresh [`SlotKey`] that refers to it.
    pub fn insert(&mut self, value: T) -> SlotKey {
        if self.free_head != NO_FREE {
            let index = self.free_head;
            let slot = &mut self.slots[index as usize];
            let next_free = match slot.content {
                Content::Vacant { next_free } => next_free,
                Content::Occupied(_) => unreachable_occupied_free_slot(),
            };
            self.free_head = next_free;
            slot.content = Content::Occupied(value);
            self.len += 1;
            SlotKey {
                index,
                generation: slot.generation,
            }
        } else {
            let index = self.slots.len() as u32;
            self.slots.push(Slot {
                generation: 0,
                content: Content::Occupied(value),
            });
            self.len += 1;
            SlotKey {
                index,
                generation: 0,
            }
        }
    }

    /// Borrow the value referenced by `key`, or `None` if the key is stale.
    pub fn get(&self, key: SlotKey) -> Option<&T> {
        let slot = self.slots.get(key.index as usize)?;
        if slot.generation != key.generation {
            return None;
        }
        match &slot.content {
            Content::Occupied(value) => Some(value),
            Content::Vacant { .. } => None,
        }
    }

    /// Mutably borrow the value referenced by `key`, or `None` if stale.
    pub fn get_mut(&mut self, key: SlotKey) -> Option<&mut T> {
        let slot = self.slots.get_mut(key.index as usize)?;
        if slot.generation != key.generation {
            return None;
        }
        match &mut slot.content {
            Content::Occupied(value) => Some(value),
            Content::Vacant { .. } => None,
        }
    }

    /// Remove and return the value referenced by `key`.
    ///
    /// On success the slot's generation is advanced, permanently invalidating
    /// `key` and any copies of it.
    pub fn remove(&mut self, key: SlotKey) -> Option<T> {
        let slot = self.slots.get_mut(key.index as usize)?;
        if slot.generation != key.generation {
            return None;
        }
        if matches!(slot.content, Content::Vacant { .. }) {
            return None;
        }
        let previous = core::mem::replace(
            &mut slot.content,
            Content::Vacant {
                next_free: self.free_head,
            },
        );
        slot.generation = slot.generation.wrapping_add(1);
        self.free_head = key.index;
        self.len -= 1;
        match previous {
            Content::Occupied(value) => Some(value),
            Content::Vacant { .. } => None,
        }
    }

    /// Whether `key` currently refers to a live value.
    pub fn contains_key(&self, key: SlotKey) -> bool {
        self.get(key).is_some()
    }

    /// Number of live elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the map holds no live elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of allocated slots (live plus recycled), i.e. the backing
    /// capacity used so far.
    pub fn capacity(&self) -> usize {
        self.slots.capacity()
    }

    /// Remove every element, invalidating all outstanding keys while keeping
    /// the allocated slot capacity.
    pub fn clear(&mut self) {
        self.free_head = NO_FREE;
        self.len = 0;
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if matches!(slot.content, Content::Occupied(_)) {
                slot.generation = slot.generation.wrapping_add(1);
            }
            slot.content = Content::Vacant {
                next_free: self.free_head,
            };
            self.free_head = i as u32;
        }
    }

    /// Iterate over `(SlotKey, &value)` pairs in ascending slot order.
    pub fn iter(&self) -> impl Iterator<Item = (SlotKey, &T)> {
        self.slots.iter().enumerate().filter_map(|(i, slot)| {
            let generation = slot.generation;
            match &slot.content {
                Content::Occupied(value) => Some((
                    SlotKey {
                        index: i as u32,
                        generation,
                    },
                    value,
                )),
                Content::Vacant { .. } => None,
            }
        })
    }

    /// Iterate over `(SlotKey, &mut value)` pairs in ascending slot order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (SlotKey, &mut T)> {
        self.slots.iter_mut().enumerate().filter_map(|(i, slot)| {
            let generation = slot.generation;
            match &mut slot.content {
                Content::Occupied(value) => Some((
                    SlotKey {
                        index: i as u32,
                        generation,
                    },
                    value,
                )),
                Content::Vacant { .. } => None,
            }
        })
    }

    /// Iterate over the live [`SlotKey`]s in ascending slot order.
    pub fn keys(&self) -> impl Iterator<Item = SlotKey> + '_ {
        self.iter().map(|(key, _)| key)
    }

    /// Iterate over shared references to the live values.
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.iter().map(|(_, value)| value)
    }

    /// Iterate over mutable references to the live values.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.iter_mut().map(|(_, value)| value)
    }
}

impl<T> Default for SlotMap<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Diverging helper for the logically-impossible "free list points at an
/// occupied slot" case. Kept as a cold `unreachable!` so the hot path stays
/// branch-light while the invariant is still documented.
#[cold]
#[inline(never)]
fn unreachable_occupied_free_slot() -> ! {
    unreachable!("free list referenced an occupied slot: SlotMap invariant broken")
}
