//! Physical slot allocation for resident virtualized pages.
//!
//! [`ResidencyTable`](crate::paging::ResidencyTable) decides *which* pages must
//! be resident; it holds no storage. This layer decides *where* each resident
//! page lives: it hands every resident key a slot in a bounded, fixed-size
//! physical pool and reclaims that slot when the page is evicted, so a backend
//! can key its real GPU page buffer on the slot index. Like the residency
//! table it is GPU-independent and deterministic - allocation always takes the
//! lowest free slot and every export iterates in key order - so the slot map a
//! given request/eviction sequence produces is reproducible and can be diffed
//! bit-for-bit against a GPU twin that resolves page keys to slots on-device.
//!
//! The pool is parameterized over the page key `K` exactly as the residency
//! table is, so geometry pages, shadow clip pages and any future virtualized
//! stream share one allocator.

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;

/// Sentinel slot index meaning "this key is not resident in the pool".
///
/// A GPU resolve kernel writes this value for a query key that misses the
/// pool, so the CPU golden [`PagePool::slot_of`] returning [`None`] and the
/// device writing [`UNMAPPED_SLOT`] denote the same outcome.
pub const UNMAPPED_SLOT: u32 = u32::MAX;

/// Why a physical-slot allocation could not be satisfied.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PagePoolError {
    /// Every slot in the pool is occupied and the key was not already mapped,
    /// so the caller must evict a resident page before this one can be admitted.
    PoolFull {
        /// Fixed number of physical slots the pool was created with.
        capacity: u32,
    },
}

impl core::fmt::Display for PagePoolError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PagePoolError::PoolFull { capacity } => {
                write!(f, "physical page pool is full ({capacity} slots occupied)")
            }
        }
    }
}

/// A bounded pool of fixed-size physical page slots, mapping resident keys to
/// slot indices.
///
/// The pool owns `capacity` slots numbered `0..capacity`. [`allocate`] admits a
/// key by handing it the lowest currently free slot; [`free`] returns a key's
/// slot to the free set for reuse. Allocation is idempotent - re-admitting an
/// already-resident key returns its existing slot without consuming another -
/// so a frame can call [`allocate`] for every key its residency table reports
/// resident without tracking which were admitted last frame.
///
/// [`allocate`]: PagePool::allocate
/// [`free`]: PagePool::free
#[derive(Clone, Debug)]
pub struct PagePool<K: Copy + Ord> {
    capacity: u32,
    /// Resident key -> physical slot. Iterates in key order for a deterministic
    /// export.
    slots: BTreeMap<K, u32>,
    /// Freed slots available for reuse, lowest index first.
    freed: BinaryHeap<Reverse<u32>>,
    /// Lowest slot index never yet handed out; the free set below this is
    /// exactly `freed`.
    high_water: u32,
}

impl<K: Copy + Ord> PagePool<K> {
    /// Creates an empty pool with `capacity` physical slots.
    #[must_use]
    pub fn new(capacity: u32) -> Self {
        Self {
            capacity,
            slots: BTreeMap::new(),
            freed: BinaryHeap::new(),
            high_water: 0,
        }
    }

    /// Number of physical slots the pool was created with.
    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Number of slots currently occupied.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether the pool holds no resident pages.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Whether every physical slot is occupied.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.slots.len() as u32 >= self.capacity
    }

    /// Physical slot backing `key`, or [`None`] if the key is not resident.
    #[must_use]
    pub fn slot_of(&self, key: K) -> Option<u32> {
        self.slots.get(&key).copied()
    }

    /// Admits `key` into the pool, returning the slot that now backs it.
    ///
    /// If `key` is already resident its existing slot is returned unchanged.
    /// Otherwise the lowest free slot is assigned - a previously freed slot in
    /// preference to growing the high-water mark, so slot indices stay dense
    /// and reuse is deterministic. Returns [`PagePoolError::PoolFull`] when the
    /// key is new and no slot is free.
    ///
    /// # Errors
    ///
    /// [`PagePoolError::PoolFull`] if the pool is full and `key` is not already
    /// mapped.
    pub fn allocate(&mut self, key: K) -> Result<u32, PagePoolError> {
        if let Some(&slot) = self.slots.get(&key) {
            return Ok(slot);
        }
        let slot = if let Some(Reverse(slot)) = self.freed.pop() {
            slot
        } else if self.high_water < self.capacity {
            let slot = self.high_water;
            self.high_water += 1;
            slot
        } else {
            return Err(PagePoolError::PoolFull {
                capacity: self.capacity,
            });
        };
        self.slots.insert(key, slot);
        Ok(slot)
    }

    /// Releases the slot backing `key` for reuse, returning it.
    ///
    /// Returns the freed slot, or [`None`] when `key` was not resident. The
    /// slot rejoins the free set and the next [`allocate`](Self::allocate) of a
    /// new key may reuse it.
    pub fn free(&mut self, key: K) -> Option<u32> {
        let slot = self.slots.remove(&key)?;
        self.freed.push(Reverse(slot));
        Some(slot)
    }

    /// Frees each key in `victims`, ignoring any that are not resident.
    ///
    /// A convenience for applying a whole
    /// [`select_evictions`](crate::paging::ResidencyTable::select_evictions)
    /// result in one call.
    pub fn free_all(&mut self, victims: &[K]) {
        for &key in victims {
            self.free(key);
        }
    }

    /// The resident slot map as a key-ordered `(key, slot)` list.
    ///
    /// This is the exact table a GPU resolve kernel binary-searches: because
    /// [`BTreeMap`] iterates in ascending key order the list is sorted, so the
    /// device and this reference agree on both contents and order.
    #[must_use]
    pub fn entries(&self) -> Vec<(K, u32)> {
        self.slots.iter().map(|(&k, &s)| (k, s)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_lowest_free_slot_densely() {
        let mut pool: PagePool<u32> = PagePool::new(4);
        assert_eq!(pool.allocate(10), Ok(0));
        assert_eq!(pool.allocate(20), Ok(1));
        assert_eq!(pool.allocate(30), Ok(2));
        assert_eq!(pool.len(), 3);
        assert_eq!(pool.slot_of(20), Some(1));
        assert_eq!(pool.slot_of(99), None);
    }

    #[test]
    fn allocation_is_idempotent() {
        let mut pool: PagePool<u32> = PagePool::new(2);
        assert_eq!(pool.allocate(7), Ok(0));
        // Re-admitting the same key returns its slot without consuming another.
        assert_eq!(pool.allocate(7), Ok(0));
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn freed_slot_is_reused_before_growing() {
        let mut pool: PagePool<u32> = PagePool::new(4);
        pool.allocate(10).unwrap();
        pool.allocate(20).unwrap();
        pool.allocate(30).unwrap();
        // Free the middle slot; the next new key must reclaim slot 1, not 3.
        assert_eq!(pool.free(20), Some(1));
        assert_eq!(pool.slot_of(20), None);
        assert_eq!(pool.allocate(40), Ok(1));
    }

    #[test]
    fn lowest_freed_slot_wins() {
        let mut pool: PagePool<u32> = PagePool::new(4);
        for k in [10u32, 20, 30, 40] {
            pool.allocate(k).unwrap();
        }
        // Free 30 (slot 2) then 20 (slot 1); the min-heap hands back 1 first.
        pool.free(30);
        pool.free(20);
        assert_eq!(pool.allocate(50), Ok(1));
        assert_eq!(pool.allocate(60), Ok(2));
    }

    #[test]
    fn full_pool_rejects_new_keys_but_admits_resident() {
        let mut pool: PagePool<u32> = PagePool::new(2);
        pool.allocate(10).unwrap();
        pool.allocate(20).unwrap();
        assert!(pool.is_full());
        assert_eq!(pool.allocate(30), Err(PagePoolError::PoolFull { capacity: 2 }));
        // A resident key is still fine even when full.
        assert_eq!(pool.allocate(10), Ok(0));
    }

    #[test]
    fn free_all_applies_an_eviction_batch() {
        let mut pool: PagePool<u32> = PagePool::new(4);
        for k in [10u32, 20, 30] {
            pool.allocate(k).unwrap();
        }
        pool.free_all(&[20, 30, 999]);
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.slot_of(10), Some(0));
    }

    #[test]
    fn entries_are_sorted_by_key() {
        let mut pool: PagePool<u32> = PagePool::new(8);
        for k in [50u32, 10, 30, 20] {
            pool.allocate(k).unwrap();
        }
        let entries = pool.entries();
        let keys: Vec<u32> = entries.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, alloc::vec![10, 20, 30, 50]);
    }
}
