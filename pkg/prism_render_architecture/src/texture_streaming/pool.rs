//! Physical page pool: the `GPU` backend's slot allocator over streamed tiles.
//!
//! The residency table ([`super::residency`]) and scheduler
//! ([`super::scheduler`]) decide *which* [`TexturePageKey`] pages should be
//! resident this frame; this module decides *where* each resident page physically
//! lives. A virtual-texture physical pool is a fixed array of equal-size tile
//! slots (one slot per page) carved out of a backing texture array; this pool
//! owns the deterministic slot bookkeeping over that array:
//!
//! * every virtual page that becomes resident is bound to exactly one physical
//!   slot index, and
//! * every eviction frees its slot for the next admission.
//!
//! The pool holds no `GPU` handle: it maps [`TexturePageKey`] to a `u32` slot
//! index, and the device-side copy dispatch (staging upload into the slot's
//! atlas tile) consumes [`PageUpload`] records against that index. Slot
//! allocation always picks the lowest free index, so an identical plan sequence
//! always yields an identical slot layout and the resulting uploads are
//! reproducible.

use super::scheduler::StreamingPlan;
use super::TexturePageKey;
use crate::paging::PagePool;
use alloc::vec;
use alloc::vec::Vec;

/// One copy the backend must record this frame: upload `key`'s tile into the
/// physical pool's slot `slot`.
///
/// Produced by [`PhysicalPagePool::apply_plan`] in the scheduler's load order
/// (most urgent first). The device records a staging-buffer-to-texture copy into
/// the atlas tile the slot addresses; the pool itself never touches the device.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PageUpload {
    /// Virtual page being made resident.
    pub key: TexturePageKey,
    /// Physical slot index the page's tile is uploaded into.
    pub slot: u32,
}

/// Fixed-capacity physical page pool mapping virtual pages to physical slots.
///
/// Keyed on [`TexturePageKey`] and sized to `capacity` equal tile slots. The
/// caller sizes `capacity` to the physical pool byte budget divided by the fixed
/// tile byte cost so a slot-bounded pool and the byte-bounded
/// [`schedule`](super::scheduler::schedule) agree on how many pages fit.
#[derive(Clone, Debug)]
pub struct PhysicalPagePool {
    /// Key -> slot bookkeeping and lowest-free-slot allocation, delegated to the
    /// generic [`PagePool`] that every virtualized stream (geometry / shadow
    /// pages) shares, so this crate keeps exactly one slot allocator.
    slots: PagePool<TexturePageKey>,
    /// `occupants[slot]` is the page bound to that slot, or `None` when free.
    /// The reverse slot -> key map is the texture-streaming-specific addition the
    /// generic pool does not carry; the indirection builder relies on it.
    occupants: Vec<Option<TexturePageKey>>,
}

impl PartialEq for PhysicalPagePool {
    /// Two pools are equal iff the same page occupies each physical slot. The
    /// generic pool's internal free-list is fully determined by that mapping, so
    /// comparing occupants alone is both necessary and sufficient.
    fn eq(&self, other: &Self) -> bool {
        self.occupants == other.occupants
    }
}

impl Eq for PhysicalPagePool {}

impl PhysicalPagePool {
    /// Creates an empty pool of `capacity` physical tile slots.
    #[must_use]
    pub fn new(capacity: u32) -> Self {
        Self {
            slots: PagePool::new(capacity),
            occupants: vec![None; capacity as usize],
        }
    }

    /// Total number of physical slots.
    #[must_use]
    pub fn capacity(&self) -> u32 {
        self.slots.capacity()
    }

    /// Number of slots currently bound to a resident page.
    #[must_use]
    pub fn resident_count(&self) -> u32 {
        self.slots.len() as u32
    }

    /// Number of free slots available for admission.
    #[must_use]
    pub fn free_count(&self) -> u32 {
        self.capacity() - self.resident_count()
    }

    /// Whether every slot is bound to a resident page.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.slots.is_full()
    }

    /// Whether the pool backs no pages at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Physical slot `key` is bound to, if resident.
    #[must_use]
    pub fn slot_of(&self, key: TexturePageKey) -> Option<u32> {
        self.slots.slot_of(key)
    }

    /// Page currently bound to physical slot `slot`, if any.
    #[must_use]
    pub fn occupant(&self, slot: u32) -> Option<TexturePageKey> {
        self.occupants.get(slot as usize).copied().flatten()
    }

    /// Whether `key` is currently backed by a physical slot.
    #[must_use]
    pub fn contains(&self, key: TexturePageKey) -> bool {
        self.slots.slot_of(key).is_some()
    }

    /// Binds a not-yet-resident page to the lowest free slot and returns it.
    ///
    /// A page already resident keeps its slot (returned unchanged, no new slot
    /// consumed); a full pool returns `None`. Admission always takes the lowest
    /// free index so the slot layout is a deterministic function of the
    /// admit/evict history.
    pub fn admit(&mut self, key: TexturePageKey) -> Option<u32> {
        let slot = self.slots.allocate(key).ok()?;
        self.occupants[slot as usize] = Some(key);
        Some(slot)
    }

    /// Drops a page's physical backing, freeing its slot, and returns the freed
    /// slot index. A no-op (returning `None`) for a page that is not resident.
    pub fn evict(&mut self, key: TexturePageKey) -> Option<u32> {
        let slot = self.slots.free(key)?;
        self.occupants[slot as usize] = None;
        Some(slot)
    }

    /// Drops every resident page, returning the pool to fully free.
    pub fn clear(&mut self) {
        self.slots = PagePool::new(self.capacity());
        for occupant in &mut self.occupants {
            *occupant = None;
        }
    }

    /// Applies a scheduler [`StreamingPlan`]: frees each evicted page's slot,
    /// then binds each loaded page to a slot, returning the uploads the backend
    /// must record this frame in the plan's load order.
    ///
    /// Evictions run first so their freed slots are available to the frame's
    /// loads, matching the scheduler's own budget accounting (the chosen
    /// resident set it sized always fits once this frame's evictions are
    /// applied). A load already resident is skipped (no redundant upload); a load
    /// that cannot be placed because the pool is full is skipped rather than
    /// fatal, mirroring the scheduler's "skip, do not abort" overflow policy.
    pub fn apply_plan(&mut self, plan: &StreamingPlan) -> Vec<PageUpload> {
        for &key in &plan.evicts {
            self.evict(key);
        }
        let mut uploads = Vec::with_capacity(plan.loads.len());
        for &key in &plan.loads {
            if self.contains(key) {
                continue;
            }
            if let Some(slot) = self.admit(key) {
                uploads.push(PageUpload { key, slot });
            }
        }
        uploads
    }

    /// Iterates resident `(page, slot)` bindings in ascending page-key order.
    ///
    /// Deterministic by construction; the `GPU` indirection builder in
    /// [`super::indirection`] relies on this ordering to emit a sorted,
    /// binary-searchable page table.
    pub fn iter(&self) -> impl Iterator<Item = (TexturePageKey, u32)> + '_ {
        self.slots.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture_streaming::scheduler::StreamingPlan;

    fn key(mip: u8, x: u16, y: u16) -> TexturePageKey {
        TexturePageKey {
            texture: 5,
            mip,
            layer: 0,
            x,
            y,
        }
    }

    #[test]
    fn new_pool_is_empty_with_full_free_list() {
        let pool = PhysicalPagePool::new(4);
        assert_eq!(pool.capacity(), 4);
        assert_eq!(pool.free_count(), 4);
        assert_eq!(pool.resident_count(), 0);
        assert!(pool.is_empty());
        assert!(!pool.is_full());
    }

    #[test]
    fn admit_takes_lowest_free_slot() {
        let mut pool = PhysicalPagePool::new(3);
        assert_eq!(pool.admit(key(0, 0, 0)), Some(0));
        assert_eq!(pool.admit(key(0, 0, 1)), Some(1));
        assert_eq!(pool.admit(key(0, 0, 2)), Some(2));
        assert!(pool.is_full());
        assert_eq!(pool.resident_count(), 3);
    }

    #[test]
    fn admit_resident_page_keeps_its_slot() {
        let mut pool = PhysicalPagePool::new(2);
        assert_eq!(pool.admit(key(0, 0, 0)), Some(0));
        // Re-admitting the same page does not consume a new slot.
        assert_eq!(pool.admit(key(0, 0, 0)), Some(0));
        assert_eq!(pool.resident_count(), 1);
        assert_eq!(pool.free_count(), 1);
    }

    #[test]
    fn full_pool_admit_returns_none() {
        let mut pool = PhysicalPagePool::new(1);
        assert_eq!(pool.admit(key(0, 0, 0)), Some(0));
        assert_eq!(pool.admit(key(0, 0, 1)), None);
        assert!(pool.is_full());
    }

    #[test]
    fn evict_frees_the_lowest_slot_for_reuse() {
        let mut pool = PhysicalPagePool::new(2);
        pool.admit(key(0, 0, 0));
        pool.admit(key(0, 0, 1));
        assert_eq!(pool.evict(key(0, 0, 0)), Some(0));
        assert!(!pool.contains(key(0, 0, 0)));
        assert_eq!(pool.occupant(0), None);
        // The freed low slot is reused ahead of any higher free slot.
        assert_eq!(pool.admit(key(0, 0, 2)), Some(0));
    }

    #[test]
    fn evict_unknown_page_is_noop() {
        let mut pool = PhysicalPagePool::new(2);
        pool.admit(key(0, 0, 0));
        assert_eq!(pool.evict(key(9, 9, 9)), None);
        assert_eq!(pool.resident_count(), 1);
    }

    #[test]
    fn slot_of_and_occupant_are_inverse() {
        let mut pool = PhysicalPagePool::new(4);
        let slot = pool.admit(key(1, 2, 3)).expect("admitted");
        assert_eq!(pool.slot_of(key(1, 2, 3)), Some(slot));
        assert_eq!(pool.occupant(slot), Some(key(1, 2, 3)));
        assert_eq!(pool.slot_of(key(0, 0, 0)), None);
    }

    #[test]
    fn clear_releases_every_slot() {
        let mut pool = PhysicalPagePool::new(3);
        pool.admit(key(0, 0, 0));
        pool.admit(key(0, 0, 1));
        pool.clear();
        assert!(pool.is_empty());
        assert_eq!(pool.free_count(), 3);
        assert_eq!(pool.admit(key(0, 0, 9)), Some(0));
    }

    #[test]
    fn apply_plan_evicts_then_loads_into_freed_slots() {
        let mut pool = PhysicalPagePool::new(2);
        pool.admit(key(0, 0, 0));
        pool.admit(key(0, 0, 1));
        // Evict both and load two fresh pages: they take the freed low slots.
        let plan = StreamingPlan {
            loads: alloc::vec![key(1, 0, 0), key(1, 0, 1)],
            evicts: alloc::vec![key(0, 0, 0), key(0, 0, 1)],
            resident_bytes: 0,
        };
        let uploads = pool.apply_plan(&plan);
        assert_eq!(
            uploads,
            alloc::vec![
                PageUpload {
                    key: key(1, 0, 0),
                    slot: 0,
                },
                PageUpload {
                    key: key(1, 0, 1),
                    slot: 1,
                },
            ]
        );
        assert_eq!(pool.resident_count(), 2);
        assert!(!pool.contains(key(0, 0, 0)));
    }

    #[test]
    fn apply_plan_skips_already_resident_load() {
        let mut pool = PhysicalPagePool::new(2);
        pool.admit(key(0, 0, 0));
        let plan = StreamingPlan {
            loads: alloc::vec![key(0, 0, 0), key(0, 0, 1)],
            evicts: alloc::vec![],
            resident_bytes: 0,
        };
        let uploads = pool.apply_plan(&plan);
        // The already-resident page produces no upload; only the new one does.
        assert_eq!(
            uploads,
            alloc::vec![PageUpload {
                key: key(0, 0, 1),
                slot: 1,
            }]
        );
    }

    #[test]
    fn apply_plan_skips_loads_that_overflow_the_pool() {
        let mut pool = PhysicalPagePool::new(1);
        let plan = StreamingPlan {
            loads: alloc::vec![key(0, 0, 0), key(0, 0, 1)],
            evicts: alloc::vec![],
            resident_bytes: 0,
        };
        let uploads = pool.apply_plan(&plan);
        // Only the first fits; the overflow load is skipped, not fatal.
        assert_eq!(
            uploads,
            alloc::vec![PageUpload {
                key: key(0, 0, 0),
                slot: 0,
            }]
        );
        assert!(pool.is_full());
    }

    #[test]
    fn iter_is_key_ordered() {
        let mut pool = PhysicalPagePool::new(4);
        pool.admit(key(2, 0, 0));
        pool.admit(key(0, 5, 9));
        pool.admit(key(0, 1, 0));
        let keys: Vec<TexturePageKey> = pool.iter().map(|(k, _)| k).collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
    }

    #[test]
    fn slot_layout_is_deterministic_across_identical_histories() {
        let build = || {
            let mut pool = PhysicalPagePool::new(3);
            pool.admit(key(0, 0, 0));
            pool.admit(key(0, 0, 1));
            pool.evict(key(0, 0, 0));
            pool.admit(key(0, 0, 2));
            pool
        };
        assert_eq!(build(), build());
    }
}
