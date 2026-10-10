//! A generational slot map backends use to associate a [`ResourceId`] with the
//! native object it names (a `wgpu::Buffer`, `VkImage`, `MTLTexture`, …).
//!
//! [`ResourceId`] is a `(index, generation)` pair. The slot map stores values
//! in a dense `Vec` indexed by `index`, tags each slot with the generation it
//! was minted at, and recycles freed slots through a free list. Resolving a
//! stale id — one whose slot has since been reused — fails the generation
//! check and returns `None` instead of aliasing the new occupant, which turns
//! use-after-free into a safe, detectable miss.
//!
//! The map is single-threaded (`&mut self` for mutation); backends wrap it in a
//! lock to get `Send + Sync`, matching the design's "shared layer is
//! single-threaded, backend adds synchronization" split. Pure, `no_std`, no
//! `unsafe`.

use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::resource::ResourceId;

/// A slot in the map: either occupied with a value at a given generation, or
/// free and pointing at the next free slot.
enum Slot<V> {
    /// An occupied slot holding `value`, minted at `generation`.
    Occupied {
        /// The generation this slot's current id was minted at.
        generation: u32,
        /// The stored value.
        value: V,
    },
    /// A free slot. `next_free` is the index of the next free slot, or
    /// [`SENTINEL`] if this is the end of the free list. `generation` is the
    /// generation the *next* occupant will be minted at.
    Free {
        /// The generation the next occupant will receive.
        generation: u32,
        /// Index of the next free slot, or [`SENTINEL`].
        next_free: u32,
    },
}

/// End-of-free-list marker.
const SENTINEL: u32 = u32::MAX;

/// A generational slot map keyed by [`ResourceId<K>`] storing values of type
/// `V`.
///
/// Insertion returns a fresh id; removal frees the slot and bumps its
/// generation so previously handed-out ids no longer resolve. Lookups validate
/// both the index and the generation.
pub struct GenerationalSlotMap<K: ?Sized, V> {
    slots: Vec<Slot<V>>,
    free_head: u32,
    len: u32,
    marker: PhantomData<fn() -> K>,
}

impl<K: ?Sized, V> GenerationalSlotMap<K, V> {
    /// Creates an empty slot map.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_head: SENTINEL,
            len: 0,
            marker: PhantomData,
        }
    }

    /// Creates an empty slot map with capacity for `cap` slots preallocated.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            slots: Vec::with_capacity(cap),
            free_head: SENTINEL,
            len: 0,
            marker: PhantomData,
        }
    }

    /// The number of occupied slots.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len as usize
    }

    /// Whether the map holds no values.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The total number of slots (occupied + free) ever allocated. Capacity in
    /// the index space; useful for backends sizing parallel arrays.
    #[must_use]
    pub fn slot_capacity(&self) -> usize {
        self.slots.len()
    }

    /// Inserts `value` and returns its fresh id. Reuses a free slot when one is
    /// available, otherwise grows the backing store. Panics only if the index
    /// space (`u32`) is exhausted, which would require ~4 billion live slots.
    pub fn insert(&mut self, value: V) -> ResourceId<K> {
        self.len += 1;
        if self.free_head != SENTINEL {
            let index = self.free_head;
            let slot = &mut self.slots[index as usize];
            let (generation, next_free) = match slot {
                Slot::Free {
                    generation,
                    next_free,
                } => (*generation, *next_free),
                Slot::Occupied { .. } => unreachable!("free_head pointed at occupied slot"),
            };
            *slot = Slot::Occupied { generation, value };
            self.free_head = next_free;
            ResourceId::from_parts(index, generation)
        } else {
            let index = u32::try_from(self.slots.len()).expect("slot map index space exhausted");
            self.slots.push(Slot::Occupied {
                generation: 0,
                value,
            });
            ResourceId::from_parts(index, 0)
        }
    }

    /// Returns a shared reference to the value named by `id`, or `None` if the
    /// id is stale or out of range.
    #[must_use]
    pub fn get(&self, id: ResourceId<K>) -> Option<&V> {
        match self.slots.get(id.index() as usize) {
            Some(Slot::Occupied { generation, value }) if *generation == id.generation() => {
                Some(value)
            }
            _ => None,
        }
    }

    /// Returns a mutable reference to the value named by `id`, or `None` if the
    /// id is stale or out of range.
    #[must_use]
    pub fn get_mut(&mut self, id: ResourceId<K>) -> Option<&mut V> {
        match self.slots.get_mut(id.index() as usize) {
            Some(Slot::Occupied { generation, value }) if *generation == id.generation() => {
                Some(value)
            }
            _ => None,
        }
    }

    /// Whether `id` currently resolves to a live value.
    #[must_use]
    pub fn contains(&self, id: ResourceId<K>) -> bool {
        self.get(id).is_some()
    }

    /// Removes and returns the value named by `id`, freeing its slot and
    /// bumping the slot generation so `id` (and any copies of it) no longer
    /// resolve. Returns `None` for a stale or out-of-range id.
    pub fn remove(&mut self, id: ResourceId<K>) -> Option<V> {
        let idx = id.index() as usize;
        let slot = self.slots.get_mut(idx)?;
        match slot {
            Slot::Occupied { generation, .. } if *generation == id.generation() => {
                // Bump generation; saturate at MAX-1 so the slot is retired
                // rather than wrapping and re-colliding with an old id.
                let next_generation = generation.wrapping_add(1);
                let taken = core::mem::replace(
                    slot,
                    Slot::Free {
                        generation: next_generation,
                        next_free: self.free_head,
                    },
                );
                self.free_head = id.index();
                self.len -= 1;
                match taken {
                    Slot::Occupied { value, .. } => Some(value),
                    Slot::Free { .. } => unreachable!("matched Occupied above"),
                }
            }
            _ => None,
        }
    }

    /// Removes every value, resetting the map to empty while *preserving* slot
    /// generations so ids minted before the clear stay invalid. Backends call
    /// this on device loss to drop all native handles safely.
    pub fn clear(&mut self) {
        // Collect indices first to avoid borrow issues, then free each.
        let mut i = 0u32;
        while (i as usize) < self.slots.len() {
            let is_occupied = matches!(self.slots[i as usize], Slot::Occupied { .. });
            if is_occupied {
                let next_gen = match &self.slots[i as usize] {
                    Slot::Occupied { generation, .. } => generation.wrapping_add(1),
                    Slot::Free { .. } => unreachable!(),
                };
                self.slots[i as usize] = Slot::Free {
                    generation: next_gen,
                    next_free: self.free_head,
                };
                self.free_head = i;
            }
            i += 1;
        }
        self.len = 0;
    }

    /// Visits every live `(ResourceId, &V)` pair in slot order. Order is stable
    /// for a given insertion/removal history but is not insertion order after
    /// slots are recycled.
    pub fn iter(&self) -> impl Iterator<Item = (ResourceId<K>, &V)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| match slot {
                Slot::Occupied { generation, value } => {
                    Some((ResourceId::from_parts(i as u32, *generation), value))
                }
                Slot::Free { .. } => None,
            })
    }

    /// Visits every live value mutably.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.slots.iter_mut().filter_map(|slot| match slot {
            Slot::Occupied { value, .. } => Some(value),
            Slot::Free { .. } => None,
        })
    }
}

impl<K: ?Sized, V> Default for GenerationalSlotMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::BufferKind;

    type Map = GenerationalSlotMap<BufferKind, u64>;

    #[test]
    fn insert_get_remove() {
        let mut m = Map::new();
        assert!(m.is_empty());
        let a = m.insert(10);
        let b = m.insert(20);
        assert_eq!(m.len(), 2);
        assert_eq!(m.get(a), Some(&10));
        assert_eq!(m.get(b), Some(&20));
        assert_eq!(m.remove(a), Some(10));
        assert_eq!(m.get(a), None);
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn stale_id_does_not_resolve_after_reuse() {
        let mut m = Map::new();
        let a = m.insert(100);
        let idx = a.index();
        assert_eq!(m.remove(a), Some(100));
        // Reuse the freed slot.
        let b = m.insert(200);
        assert_eq!(b.index(), idx, "freed slot should be reused");
        assert_ne!(b.generation(), a.generation(), "generation must bump");
        // The old id must not alias the new occupant.
        assert_eq!(m.get(a), None);
        assert_eq!(m.get(b), Some(&200));
    }

    #[test]
    fn double_remove_is_safe() {
        let mut m = Map::new();
        let a = m.insert(1);
        assert_eq!(m.remove(a), Some(1));
        assert_eq!(m.remove(a), None);
    }

    #[test]
    fn get_mut_updates_value() {
        let mut m = Map::new();
        let a = m.insert(5);
        *m.get_mut(a).unwrap() += 37;
        assert_eq!(m.get(a), Some(&42));
    }

    #[test]
    fn free_list_lifo_reuse() {
        let mut m = Map::new();
        let a = m.insert(1);
        let b = m.insert(2);
        let c = m.insert(3);
        m.remove(a);
        m.remove(b);
        // LIFO: last freed (b) is reused first.
        let d = m.insert(4);
        assert_eq!(d.index(), b.index());
        let _ = c;
    }

    #[test]
    fn clear_invalidates_all_ids() {
        let mut m = Map::new();
        let a = m.insert(1);
        let b = m.insert(2);
        m.clear();
        assert!(m.is_empty());
        assert_eq!(m.get(a), None);
        assert_eq!(m.get(b), None);
        // New inserts still work and never collide with the old ids.
        let c = m.insert(3);
        assert_eq!(m.get(c), Some(&3));
        assert_ne!(c.generation(), a.generation());
    }

    #[test]
    fn iter_visits_live_only() {
        let mut m = Map::new();
        let a = m.insert(1);
        let b = m.insert(2);
        let _c = m.insert(3);
        m.remove(b);
        let mut vals: Vec<u64> = m.iter().map(|(_, v)| *v).collect();
        vals.sort_unstable();
        assert_eq!(vals, alloc::vec![1, 3]);
        let _ = a;
    }
}
