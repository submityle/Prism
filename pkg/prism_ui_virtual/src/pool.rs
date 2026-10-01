//! A recycle pool that maps live item indices to reusable render slots.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// A stable identity for a recyclable render slot.
///
/// Slot ids are handed out by a [`RecyclePool`] and are reused as items scroll
/// out of and back into view, which lets a backend keep a bounded set of
/// retained nodes instead of one per logical item.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SlotId(pub usize);

/// Recycles a bounded set of render slots across a changing visible window.
///
/// Each live item index is mapped to exactly one [`SlotId`]. Releasing an item
/// returns its slot to a free list; the next [`RecyclePool::acquire`] of a new
/// item reuses the most recently freed slot before allocating a fresh one. Slot
/// reuse is therefore deterministic: identical acquire/release sequences always
/// produce identical slot assignments.
#[derive(Clone, Debug, Default)]
pub struct RecyclePool {
    active: BTreeMap<usize, SlotId>,
    free: Vec<SlotId>,
    next: usize,
}

impl RecyclePool {
    /// Creates an empty pool.
    #[must_use]
    pub fn new() -> Self {
        Self {
            active: BTreeMap::new(),
            free: Vec::new(),
            next: 0,
        }
    }

    /// Returns the slot bound to `item_index`, allocating or recycling one if
    /// the item is not already active.
    ///
    /// Acquiring an already-active item is idempotent and returns its existing
    /// slot. Otherwise the most recently freed slot is reused; if none are
    /// free, a brand-new slot id is allocated.
    pub fn acquire(&mut self, item_index: usize) -> SlotId {
        if let Some(&slot) = self.active.get(&item_index) {
            return slot;
        }
        let slot = match self.free.pop() {
            Some(reused) => reused,
            None => {
                let fresh = SlotId(self.next);
                self.next += 1;
                fresh
            }
        };
        self.active.insert(item_index, slot);
        slot
    }

    /// Releases the slot bound to `item_index`, returning it to the free list.
    ///
    /// Returns the freed [`SlotId`], or `None` if the item was not active.
    /// Releasing an item twice is a no-op on the second call.
    pub fn release(&mut self, item_index: usize) -> Option<SlotId> {
        let slot = self.active.remove(&item_index)?;
        self.free.push(slot);
        Some(slot)
    }

    /// Returns the slot currently bound to `item_index`, if any.
    #[must_use]
    pub fn slot_of(&self, item_index: usize) -> Option<SlotId> {
        self.active.get(&item_index).copied()
    }

    /// Whether `item_index` currently holds a slot.
    #[must_use]
    pub fn is_active(&self, item_index: usize) -> bool {
        self.active.contains_key(&item_index)
    }

    /// The number of items currently holding a slot.
    #[must_use]
    pub fn active_len(&self) -> usize {
        self.active.len()
    }

    /// The number of slots sitting idle in the free list.
    #[must_use]
    pub fn free_len(&self) -> usize {
        self.free.len()
    }

    /// The total number of distinct slots this pool has ever allocated.
    ///
    /// This is the high-water mark of concurrently live slots and bounds the
    /// number of retained backend nodes.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.next
    }

    /// Iterates over the active `(item_index, slot)` bindings in ascending item
    /// order.
    pub fn active_bindings(&self) -> impl Iterator<Item = (usize, SlotId)> + '_ {
        self.active.iter().map(|(&index, &slot)| (index, slot))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_allocation_counts_up() {
        let mut pool = RecyclePool::new();
        assert_eq!(pool.acquire(0), SlotId(0));
        assert_eq!(pool.acquire(1), SlotId(1));
        assert_eq!(pool.acquire(2), SlotId(2));
        assert_eq!(pool.capacity(), 3);
        assert_eq!(pool.active_len(), 3);
        assert_eq!(pool.free_len(), 0);
    }

    #[test]
    fn acquire_is_idempotent_for_active_items() {
        let mut pool = RecyclePool::new();
        let first = pool.acquire(7);
        assert_eq!(pool.acquire(7), first);
        assert_eq!(pool.active_len(), 1);
        assert_eq!(pool.capacity(), 1);
    }

    #[test]
    fn released_slots_are_reused_lifo() {
        let mut pool = RecyclePool::new();
        let s0 = pool.acquire(0);
        let s1 = pool.acquire(1);
        pool.acquire(2);

        // Release 0 then 1; the free list is LIFO so 1's slot is reused first.
        assert_eq!(pool.release(0), Some(s0));
        assert_eq!(pool.release(1), Some(s1));
        assert_eq!(pool.free_len(), 2);

        // No new slot is allocated while the free list is non-empty.
        let reused_a = pool.acquire(10);
        assert_eq!(reused_a, s1);
        let reused_b = pool.acquire(11);
        assert_eq!(reused_b, s0);
        assert_eq!(pool.capacity(), 3);
        assert_eq!(pool.free_len(), 0);
    }

    #[test]
    fn active_mapping_is_tracked() {
        let mut pool = RecyclePool::new();
        pool.acquire(5);
        pool.acquire(9);
        assert!(pool.is_active(5));
        assert_eq!(pool.slot_of(5), Some(SlotId(0)));
        assert_eq!(pool.slot_of(9), Some(SlotId(1)));
        assert_eq!(pool.slot_of(42), None);

        let bindings: Vec<(usize, SlotId)> = pool.active_bindings().collect();
        assert_eq!(bindings, alloc::vec![(5, SlotId(0)), (9, SlotId(1))]);
    }

    #[test]
    fn releasing_unknown_or_twice_is_safe() {
        let mut pool = RecyclePool::new();
        pool.acquire(3);
        assert_eq!(pool.release(3), Some(SlotId(0)));
        assert_eq!(pool.release(3), None);
        assert_eq!(pool.release(99), None);
    }

    #[test]
    fn deterministic_sequence() {
        // Two identical acquire/release scripts yield identical assignments.
        fn run() -> Vec<SlotId> {
            let mut pool = RecyclePool::new();
            let mut out = Vec::new();
            out.push(pool.acquire(0));
            out.push(pool.acquire(1));
            pool.release(0);
            out.push(pool.acquire(2));
            pool.release(1);
            out.push(pool.acquire(3));
            out
        }
        assert_eq!(run(), run());
    }
}
