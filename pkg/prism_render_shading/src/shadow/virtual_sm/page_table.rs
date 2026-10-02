//! Virtual-to-physical page mapping table and per-page residency tracking.
//!
//! The virtual shadow map (VSM) addresses shadow depth through a sparse grid of
//! fixed-size pages identified by [`ShadowPageKey`].  Only a bounded working set
//! of those pages is ever backed by physical memory, so this table records, for
//! every virtual page the renderer has touched, whether it is currently
//! *resident* (its depth is valid), *pending* (a physical slot is reserved but
//! the depth still has to be rasterised) or *evicted* (the mapping was dropped
//! to make room for a hotter page).
//!
//! The table is the CPU golden twin of the GPU page-table buffer that
//! `shadow.wesl` samples: a lookup that misses on the CPU is exactly the lookup
//! that would sample an unmapped page on the GPU, so the hit/miss statistics
//! recorded here predict the on-device page-request feedback.
//!
//! [`ShadowPageKey`] deliberately derives `Hash`/`Eq` but not `Ord`, and the
//! crate is `no_std`, so the table keys a deterministic [`BTreeMap`] on the
//! packed tuple returned by [`page_order`] rather than reaching for a hasher.

use alloc::collections::BTreeMap;
use prism_render_architecture::virtual_shadow::ShadowPageKey;

/// Packed total order for a [`ShadowPageKey`] (`light`, then `level`, then row
/// `y`, then column `x`).  [`ShadowPageKey`] is `Hash + Eq` but not `Ord`, so
/// this tuple is what the ordered [`BTreeMap`] is actually keyed on.
pub type PageOrder = (u32, u16, u16, u16);

/// Packs a [`ShadowPageKey`] into its [`PageOrder`] sort key.
pub fn page_order(key: &ShadowPageKey) -> PageOrder {
    (key.light, key.level, key.y, key.x)
}

/// Rebuilds a [`ShadowPageKey`] from its [`PageOrder`] sort key (the inverse of
/// [`page_order`]).
pub fn key_from_order(order: PageOrder) -> ShadowPageKey {
    let (light, level, y, x) = order;
    ShadowPageKey { light, level, x, y }
}

/// Residency state of a single virtual shadow page.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Residency {
    /// Mapped to a physical page whose depth is valid this frame.
    Resident {
        /// Index of the backing physical page.
        physical_page: u32,
    },
    /// A physical page is reserved but its depth still has to be rasterised.
    Pending {
        /// Index of the reserved physical page.
        physical_page: u32,
    },
    /// The mapping was dropped to reclaim its physical page; no backing exists.
    Evicted,
}

impl Residency {
    /// The backing physical page for a resident or pending mapping, or `None`
    /// when the page has been evicted.
    pub fn physical_page(self) -> Option<u32> {
        match self {
            Residency::Resident { physical_page } | Residency::Pending { physical_page } => {
                Some(physical_page)
            }
            Residency::Evicted => None,
        }
    }

    /// Whether the page's depth is valid and can be sampled this frame.
    pub fn is_resident(self) -> bool {
        matches!(self, Residency::Resident { .. })
    }

    /// Whether a physical page is reserved but still awaiting rasterisation.
    pub fn is_pending(self) -> bool {
        matches!(self, Residency::Pending { .. })
    }
}

/// One tracked virtual page: its residency plus the last frame it was accessed
/// (the recency the physical-page allocator uses for its LRU eviction).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PageEntry {
    residency: Residency,
    last_used_frame: u64,
}

/// Cumulative counters describing how well the working set fits the budget.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PageTableStats {
    /// Residency queries that found a resident page (a cache hit).
    pub hits: u64,
    /// Residency queries that found no resident page (a cache miss).
    pub misses: u64,
    /// Number of virtual pages currently tracked (any residency state).
    pub tracked: usize,
    /// Number of tracked pages that are currently resident.
    pub resident: usize,
    /// Number of tracked pages that are currently pending.
    pub pending: usize,
}

impl PageTableStats {
    /// Fraction of residency queries that hit, in `[0, 1]`; `0.0` when no query
    /// has been issued yet.
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f32 / total as f32
        }
    }
}

/// Sparse virtual-page directory mapping [`ShadowPageKey`]s to their residency.
#[derive(Clone, Debug, Default)]
pub struct VirtualPageTable {
    entries: BTreeMap<PageOrder, PageEntry>,
    hits: u64,
    misses: u64,
}

impl VirtualPageTable {
    /// Creates an empty page table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the residency of `key` without touching the hit/miss counters.
    pub fn get(&self, key: &ShadowPageKey) -> Option<Residency> {
        self.entries
            .get(&page_order(key))
            .map(|entry| entry.residency)
    }

    /// Whether `key` is tracked (in any residency state).
    pub fn contains(&self, key: &ShadowPageKey) -> bool {
        self.entries.contains_key(&page_order(key))
    }

    /// Residency query used by the frame driver: returns the backing physical
    /// page when `key` is *resident*, counting a hit, and otherwise counts a
    /// miss and returns `None`.  A present-but-not-resident page still has its
    /// recency refreshed so a follow-up allocation keeps it warm.
    pub fn query(&mut self, key: &ShadowPageKey, frame: u64) -> Option<u32> {
        match self.entries.get_mut(&page_order(key)) {
            Some(entry) => {
                entry.last_used_frame = frame;
                if let Residency::Resident { physical_page } = entry.residency {
                    self.hits += 1;
                    Some(physical_page)
                } else {
                    self.misses += 1;
                    None
                }
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Refreshes the recency of `key` if it is tracked, returning whether it
    /// was present.  Does not affect the hit/miss counters.
    pub fn touch(&mut self, key: &ShadowPageKey, frame: u64) -> bool {
        if let Some(entry) = self.entries.get_mut(&page_order(key)) {
            entry.last_used_frame = frame;
            true
        } else {
            false
        }
    }

    /// Reserves `physical_page` for `key` and marks it pending (a physical slot
    /// exists but the depth still has to be rasterised).
    pub fn insert_pending(&mut self, key: &ShadowPageKey, physical_page: u32, frame: u64) {
        self.entries.insert(
            page_order(key),
            PageEntry {
                residency: Residency::Pending { physical_page },
                last_used_frame: frame,
            },
        );
    }

    /// Promotes `key` to resident once its depth has been rasterised.  A page
    /// with no reserved physical slot is left unchanged.
    pub fn mark_resident(&mut self, key: &ShadowPageKey, frame: u64) {
        if let Some(entry) = self.entries.get_mut(&page_order(key))
            && let Some(physical_page) = entry.residency.physical_page()
        {
            entry.residency = Residency::Resident { physical_page };
            entry.last_used_frame = frame;
        }
    }

    /// Marks `key` evicted, retaining the entry so a later re-request is a
    /// *known* miss (matching the GPU feedback of an unmapped page).
    pub fn mark_evicted(&mut self, key: &ShadowPageKey) {
        if let Some(entry) = self.entries.get_mut(&page_order(key)) {
            entry.residency = Residency::Evicted;
        }
    }

    /// Removes `key` from the table entirely, returning its prior residency.
    pub fn remove(&mut self, key: &ShadowPageKey) -> Option<Residency> {
        self.entries
            .remove(&page_order(key))
            .map(|entry| entry.residency)
    }

    /// Drops every retained *evicted* entry, returning how many were pruned.
    pub fn prune_evicted(&mut self) -> usize {
        let before = self.entries.len();
        self.entries
            .retain(|_, entry| entry.residency != Residency::Evicted);
        before - self.entries.len()
    }

    /// Number of tracked pages in any residency state.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table tracks no pages.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterates tracked `(key, residency)` pairs in deterministic key order.
    pub fn iter(&self) -> impl Iterator<Item = (ShadowPageKey, Residency)> + '_ {
        self.entries
            .iter()
            .map(|(order, entry)| (key_from_order(*order), entry.residency))
    }

    /// Snapshot of the cumulative hit/miss counters and current occupancy.
    pub fn stats(&self) -> PageTableStats {
        let mut resident = 0;
        let mut pending = 0;
        for entry in self.entries.values() {
            match entry.residency {
                Residency::Resident { .. } => resident += 1,
                Residency::Pending { .. } => pending += 1,
                Residency::Evicted => {}
            }
        }
        PageTableStats {
            hits: self.hits,
            misses: self.misses,
            tracked: self.entries.len(),
            resident,
            pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(light: u32, level: u16, x: u16, y: u16) -> ShadowPageKey {
        ShadowPageKey { light, level, x, y }
    }

    /// `page_order` and `key_from_order` must round-trip every field.
    #[test]
    fn order_round_trips_the_key() {
        let k = key(3, 2, 17, 42);
        assert_eq!(key_from_order(page_order(&k)), k);
    }

    /// A pending page is not yet sampleable but carries its reserved slot; once
    /// marked resident it becomes a hit on the next query.
    #[test]
    fn pending_then_resident_flips_query_result() {
        let mut table = VirtualPageTable::new();
        let k = key(0, 0, 1, 1);
        table.insert_pending(&k, 5, 10);
        assert_eq!(table.get(&k), Some(Residency::Pending { physical_page: 5 }));
        // A pending page is a miss for sampling purposes.
        assert_eq!(table.query(&k, 11), None);
        table.mark_resident(&k, 12);
        assert_eq!(table.query(&k, 13), Some(5));
    }

    /// Hit/miss counters and the derived hit rate track query outcomes.
    #[test]
    fn stats_track_hits_and_misses() {
        let mut table = VirtualPageTable::new();
        let hot = key(1, 0, 2, 2);
        table.insert_pending(&hot, 0, 0);
        table.mark_resident(&hot, 0);

        assert_eq!(table.query(&hot, 1), Some(0)); // hit
        assert_eq!(table.query(&key(1, 0, 9, 9), 1), None); // miss (absent)
        let stats = table.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 0.5).abs() < 1.0e-6);
        assert_eq!(stats.resident, 1);
    }

    /// Eviction retains the entry as a known miss until pruned.
    #[test]
    fn evicted_entry_is_a_known_miss_until_pruned() {
        let mut table = VirtualPageTable::new();
        let k = key(2, 1, 3, 4);
        table.insert_pending(&k, 7, 0);
        table.mark_resident(&k, 0);
        table.mark_evicted(&k);
        assert_eq!(table.get(&k), Some(Residency::Evicted));
        assert_eq!(table.query(&k, 1), None);
        assert_eq!(table.prune_evicted(), 1);
        assert!(!table.contains(&k));
    }

    /// Iteration is ordered by `(light, level, y, x)` regardless of insertion
    /// order, keeping the golden output deterministic.
    #[test]
    fn iteration_is_deterministically_ordered() {
        let mut table = VirtualPageTable::new();
        table.insert_pending(&key(0, 1, 5, 0), 0, 0);
        table.insert_pending(&key(0, 0, 9, 9), 1, 0);
        table.insert_pending(&key(0, 0, 0, 1), 2, 0);
        let keys: Vec<_> = table.iter().map(|(k, _)| page_order(&k)).collect();
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        assert_eq!(keys, sorted);
    }
}
