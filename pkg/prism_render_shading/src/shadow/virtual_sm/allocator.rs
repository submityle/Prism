//! Physical shadow-page pool with least-recently-used (LRU) eviction.
//!
//! The virtual shadow map can address far more pages than fit in memory, so a
//! fixed pool of `physical_pages` (the [`VirtualShadowSettings`] budget) backs
//! whichever virtual pages are hottest.  This allocator owns that pool: each
//! frame the driver *requests* a physical page for every virtual page it needs;
//! a request either reuses the page already mapped to that key or claims a free
//! slot, and when the pool is full the least-recently-used occupant is evicted
//! to make room.
//!
//! Recency is the frame index last passed to [`PhysicalPageAllocator::request`]
//! or [`PhysicalPageAllocator::touch`], so a page used every frame is never the
//! eviction victim while a page the camera has panned away from ages out.  Ties
//! break on the physical index, keeping eviction fully deterministic for the
//! golden test.
//!
//! The pool mirrors the GPU physical-page atlas: the physical indices handed
//! out here are the atlas slots `shadow.wesl` would sample, so the CPU eviction
//! order predicts the on-device residency set exactly.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::shadow::virtual_sm::page_table::{page_order, PageOrder};
use prism_render_architecture::virtual_shadow::ShadowPageKey;

/// What one occupied physical slot is holding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Occupant {
    key: ShadowPageKey,
    last_used_frame: u64,
}

/// Outcome of a single [`PhysicalPageAllocator::request`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Allocation {
    /// Physical page index now mapped to the requested key.
    pub physical_page: u32,
    /// Whether the key was already resident (a reuse rather than a new claim).
    pub was_resident: bool,
    /// The key evicted to make room, when the pool was full and the request was
    /// a fresh claim.
    pub evicted: Option<ShadowPageKey>,
}

/// Cumulative pool statistics for budget reporting.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AllocatorStats {
    /// Total physical slots in the pool.
    pub capacity: u32,
    /// Requests that claimed a fresh slot (free or via eviction).
    pub allocations: u64,
    /// Requests served by reusing the key's existing mapping.
    pub reuses: u64,
    /// Occupants dropped to make room for a fresh claim.
    pub evictions: u64,
    /// Physical slots currently occupied.
    pub live_pages: u32,
}

impl AllocatorStats {
    /// Fraction of requests served without a fresh claim, in `[0, 1]`; `0.0`
    /// before any request.
    pub fn reuse_rate(&self) -> f32 {
        let total = self.allocations + self.reuses;
        if total == 0 {
            0.0
        } else {
            self.reuses as f32 / total as f32
        }
    }
}

/// Fixed-capacity physical page pool with LRU eviction.
#[derive(Clone, Debug)]
pub struct PhysicalPageAllocator {
    capacity: u32,
    free: Vec<u32>,
    occupied: BTreeMap<u32, Occupant>,
    by_key: BTreeMap<PageOrder, u32>,
    allocations: u64,
    reuses: u64,
    evictions: u64,
}

impl PhysicalPageAllocator {
    /// Creates a pool of `capacity` physical pages, all initially free.
    pub fn new(capacity: u32) -> Self {
        // Hand out low indices first for deterministic assignment.
        let free = (0..capacity).rev().collect();
        Self {
            capacity,
            free,
            occupied: BTreeMap::new(),
            by_key: BTreeMap::new(),
            allocations: 0,
            reuses: 0,
            evictions: 0,
        }
    }

    /// Total physical slots in the pool.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Physical slots currently occupied.
    pub fn live_pages(&self) -> u32 {
        self.occupied.len() as u32
    }

    /// The physical page currently mapped to `key`, if any.
    pub fn physical_page(&self, key: &ShadowPageKey) -> Option<u32> {
        self.by_key.get(&page_order(key)).copied()
    }

    /// Refreshes the recency of `key` if it is resident, returning whether it
    /// was present.
    pub fn touch(&mut self, key: &ShadowPageKey, frame: u64) -> bool {
        if let Some(&physical) = self.by_key.get(&page_order(key)) {
            if let Some(occupant) = self.occupied.get_mut(&physical) {
                occupant.last_used_frame = frame;
            }
            true
        } else {
            false
        }
    }

    /// Requests a physical page for `key` at `frame`.
    ///
    /// * If `key` is already resident its recency is refreshed and the same
    ///   slot is returned (`was_resident = true`).
    /// * Otherwise a free slot is taken, or — when the pool is full — the
    ///   least-recently-used occupant is evicted and its slot reused.
    ///
    /// Returns `None` only for a zero-capacity pool, which can never satisfy a
    /// request.
    pub fn request(&mut self, key: &ShadowPageKey, frame: u64) -> Option<Allocation> {
        if self.capacity == 0 {
            return None;
        }
        let order = page_order(key);
        if let Some(&physical) = self.by_key.get(&order) {
            if let Some(occupant) = self.occupied.get_mut(&physical) {
                occupant.last_used_frame = frame;
            }
            self.reuses += 1;
            return Some(Allocation {
                physical_page: physical,
                was_resident: true,
                evicted: None,
            });
        }

        let (physical, evicted) = match self.free.pop() {
            Some(slot) => (slot, None),
            None => {
                let victim = self.lru_victim();
                let victim_occupant = self
                    .occupied
                    .remove(&victim)
                    .expect("lru victim is occupied");
                self.by_key.remove(&page_order(&victim_occupant.key));
                self.evictions += 1;
                (victim, Some(victim_occupant.key))
            }
        };

        self.occupied.insert(
            physical,
            Occupant {
                key: *key,
                last_used_frame: frame,
            },
        );
        self.by_key.insert(order, physical);
        self.allocations += 1;
        Some(Allocation {
            physical_page: physical,
            was_resident: false,
            evicted,
        })
    }

    /// Releases `key`'s physical page back to the free list, returning the
    /// freed physical index if the key was resident.
    pub fn release(&mut self, key: &ShadowPageKey) -> Option<u32> {
        let physical = self.by_key.remove(&page_order(key))?;
        self.occupied.remove(&physical);
        self.free.push(physical);
        Some(physical)
    }

    /// Physical index of the least-recently-used occupant, breaking ties on the
    /// lower physical index for determinism.
    fn lru_victim(&self) -> u32 {
        self.occupied
            .iter()
            .min_by(|a, b| {
                a.1.last_used_frame
                    .cmp(&b.1.last_used_frame)
                    .then_with(|| a.0.cmp(b.0))
            })
            .map(|(physical, _)| *physical)
            .expect("a full pool has at least one occupant")
    }

    /// Snapshot of the cumulative pool statistics.
    pub fn stats(&self) -> AllocatorStats {
        AllocatorStats {
            capacity: self.capacity,
            allocations: self.allocations,
            reuses: self.reuses,
            evictions: self.evictions,
            live_pages: self.live_pages(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(x: u16) -> ShadowPageKey {
        ShadowPageKey {
            light: 0,
            level: 0,
            x,
            y: 0,
        }
    }

    /// Fresh claims take free slots in ascending order; re-requesting a live
    /// key reuses its slot without a new allocation.
    #[test]
    fn claims_are_ascending_and_reuse_is_free() {
        let mut pool = PhysicalPageAllocator::new(4);
        let a = pool.request(&key(0), 1).unwrap();
        let b = pool.request(&key(1), 1).unwrap();
        assert_eq!((a.physical_page, b.physical_page), (0, 1));
        assert!(!a.was_resident && !b.was_resident);

        let a_again = pool.request(&key(0), 2).unwrap();
        assert_eq!(a_again.physical_page, 0);
        assert!(a_again.was_resident);
        assert_eq!(a_again.evicted, None);

        let stats = pool.stats();
        assert_eq!(stats.allocations, 2);
        assert_eq!(stats.reuses, 1);
        assert_eq!(stats.live_pages, 2);
    }

    /// A full pool evicts the least-recently-used page, and touching a page
    /// keeps it warm so it is not the victim.
    #[test]
    fn full_pool_evicts_least_recently_used() {
        let mut pool = PhysicalPageAllocator::new(2);
        pool.request(&key(0), 1).unwrap(); // slot 0, frame 1
        pool.request(&key(1), 2).unwrap(); // slot 1, frame 2
                                           // Refresh key 0 so key 1 becomes the LRU victim.
        assert!(pool.touch(&key(0), 3));

        let evicting = pool.request(&key(2), 4).unwrap();
        assert!(!evicting.was_resident);
        assert_eq!(evicting.evicted, Some(key(1)));
        // key 2 took key 1's freed slot.
        assert_eq!(evicting.physical_page, 1);
        assert_eq!(pool.physical_page(&key(1)), None);
        assert_eq!(pool.stats().evictions, 1);
    }

    /// Releasing a key frees its slot for the next claim.
    #[test]
    fn release_returns_slot_to_the_pool() {
        let mut pool = PhysicalPageAllocator::new(1);
        let a = pool.request(&key(0), 1).unwrap();
        assert_eq!(pool.release(&key(0)), Some(a.physical_page));
        assert_eq!(pool.live_pages(), 0);
        // The freed slot is reused with no eviction.
        let b = pool.request(&key(1), 2).unwrap();
        assert_eq!(b.physical_page, a.physical_page);
        assert_eq!(b.evicted, None);
    }

    /// A zero-capacity pool can never satisfy a request.
    #[test]
    fn zero_capacity_pool_never_allocates() {
        let mut pool = PhysicalPageAllocator::new(0);
        assert_eq!(pool.request(&key(0), 1), None);
        assert_eq!(pool.live_pages(), 0);
    }

    /// Tie-broken LRU is deterministic: with equal recency the lower physical
    /// index is evicted first.
    #[test]
    fn lru_ties_break_on_physical_index() {
        let mut pool = PhysicalPageAllocator::new(2);
        pool.request(&key(0), 5).unwrap(); // slot 0
        pool.request(&key(1), 5).unwrap(); // slot 1, same frame
        let evicting = pool.request(&key(2), 6).unwrap();
        // Equal recency -> lowest physical index (0, holding key 0) evicted.
        assert_eq!(evicting.evicted, Some(key(0)));
        assert_eq!(evicting.physical_page, 0);
    }
}
