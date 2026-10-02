//! A small generational arena.
//!
//! Nodes are stored in a contiguous `Vec` and addressed by [`NodeId`], which
//! pairs a slot index with a generation counter. Reusing a freed slot bumps its
//! generation, so a stale [`NodeId`] from a previous occupant fails to resolve
//! instead of silently aliasing an unrelated node. This keeps the retained tree
//! cheap (dense storage, O(1) access) while staying memory-safe without any
//! `unsafe`.

use alloc::vec::Vec;

/// A stable handle into an [`Arena`].
///
/// The handle is only valid while the slot it points at still holds the same
/// generation. Removing a value invalidates every outstanding handle to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId {
    index: u32,
    generation: u32,
}

impl NodeId {
    /// The raw slot index. Exposed mainly for debugging and stable ordering.
    #[inline]
    pub fn index(self) -> u32 {
        self.index
    }

    /// The generation this handle was minted with.
    #[inline]
    pub fn generation(self) -> u32 {
        self.generation
    }
}

#[derive(Debug)]
struct Slot<T> {
    generation: u32,
    entry: Entry<T>,
}

#[derive(Debug)]
enum Entry<T> {
    Occupied(T),
    Free { next_free: Option<u32> },
}

/// A generational arena holding values of type `T`.
#[derive(Debug)]
pub struct Arena<T> {
    slots: Vec<Slot<T>>,
    free_head: Option<u32>,
    len: usize,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Arena<T> {
    /// Creates an empty arena.
    #[inline]
    pub const fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_head: None,
            len: 0,
        }
    }

    /// Number of live values.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the arena holds no live values.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Inserts `value`, returning a fresh handle to it.
    pub fn insert(&mut self, value: T) -> NodeId {
        // Try to reuse a slot from the free list, but only when the head
        // actually points at a free slot. The free list only ever threads
        // through free slots; if the head resolves to a missing or occupied
        // slot the list is inconsistent, so we ignore it rather than trust a
        // corrupt pointer, then fall through to appending a fresh slot.
        let reuse = self
            .free_head
            .and_then(|index| match self.slots.get(index as usize) {
                Some(slot) => match slot.entry {
                    Entry::Free { next_free } => Some((index, next_free, slot.generation)),
                    Entry::Occupied(_) => None,
                },
                None => None,
            });

        if let Some((index, next_free, generation)) = reuse {
            self.free_head = next_free;
            self.slots[index as usize].entry = Entry::Occupied(value);
            self.len += 1;
            return NodeId { index, generation };
        }

        // Empty or inconsistent free list: discard any stale head and append a
        // new slot at the end of the backing storage.
        self.free_head = None;
        let index =
            u32::try_from(self.slots.len()).expect("arena capacity exceeded u32::MAX slots");
        self.slots.push(Slot {
            generation: 0,
            entry: Entry::Occupied(value),
        });
        self.len += 1;
        NodeId {
            index,
            generation: 0,
        }
    }

    /// Returns `true` if `id` still resolves to a live value.
    pub fn contains(&self, id: NodeId) -> bool {
        self.get(id).is_some()
    }

    /// Borrows the value behind `id`, if it is still live.
    pub fn get(&self, id: NodeId) -> Option<&T> {
        let slot = self.slots.get(id.index as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        match &slot.entry {
            Entry::Occupied(value) => Some(value),
            Entry::Free { .. } => None,
        }
    }

    /// Mutably borrows the value behind `id`, if it is still live.
    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut T> {
        let slot = self.slots.get_mut(id.index as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        match &mut slot.entry {
            Entry::Occupied(value) => Some(value),
            Entry::Free { .. } => None,
        }
    }

    /// Removes and returns the value behind `id`, invalidating the handle.
    pub fn remove(&mut self, id: NodeId) -> Option<T> {
        let slot = self.slots.get_mut(id.index as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        if matches!(slot.entry, Entry::Free { .. }) {
            return None;
        }
        slot.generation = slot.generation.wrapping_add(1);
        let taken = core::mem::replace(
            &mut slot.entry,
            Entry::Free {
                next_free: self.free_head,
            },
        );
        self.free_head = Some(id.index);
        self.len -= 1;
        match taken {
            Entry::Occupied(value) => Some(value),
            Entry::Free { .. } => None,
        }
    }

    /// Iterates over every live `(NodeId, &T)` pair in slot order.
    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &T)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| match &slot.entry {
                Entry::Occupied(value) => Some((
                    NodeId {
                        index: index as u32,
                        generation: slot.generation,
                    },
                    value,
                )),
                Entry::Free { .. } => None,
            })
    }
}
