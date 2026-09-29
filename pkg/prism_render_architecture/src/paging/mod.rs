//! Generic paged-residency bookkeeping shared by virtualized subsystems.
//!
//! Virtualized geometry and virtual shadow maps both stream fixed-size pages
//! into a bounded physical pool: each frame names the pages it needs at some
//! screen-importance priority, the backend uploads them, and when the pool
//! overflows the least-important idle pages are evicted. The lifecycle and the
//! budget policy are identical regardless of what a page *contains*, so they
//! live here once, parameterized over the page key `K`. A subsystem supplies
//! its own key type (a geometry page address, a shadow clip page, ...) and gets
//! the whole residency machine for free.
//!
//! The layer is GPU-independent and deterministic: it holds no GPU handles, the
//! backend keys its physical store on `K` and consults this table to decide
//! what to upload and drop, and every container iterates in `K` order so
//! request and eviction sequences are reproducible.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

pub mod page_pool;
pub mod page_storage;

pub use page_pool::{PagePool, PagePoolError, UNMAPPED_SLOT};
pub use page_storage::{PageStorage, PageStorageError};

/// Lifecycle state of a tracked page.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PageResidency {
    /// Known to the table but neither requested nor resident.
    #[default]
    Unloaded,
    /// A streaming request is outstanding.
    Requested,
    /// Backed by physical storage and ready to use.
    Resident,
}

/// Per-page bookkeeping record.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PageEntry {
    /// Current lifecycle state.
    pub residency: PageResidency,
    /// Screen importance; higher survives eviction longer.
    pub priority: f32,
    /// Frame index the page was last requested or touched.
    pub last_used_frame: u64,
}

/// Session-lifetime residency table over virtualized pages keyed on `K`.
#[derive(Clone, Debug)]
pub struct ResidencyTable<K: Copy + Ord> {
    entries: BTreeMap<K, PageEntry>,
}

impl<K: Copy + Ord> Default for ResidencyTable<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Copy + Ord> ResidencyTable<K> {
    /// Creates an empty table.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Records that `key` is needed this frame at the given screen `priority`.
    ///
    /// An unknown page becomes [`PageResidency::Requested`]; a known page keeps
    /// its residency but raises its priority to the max seen and advances its
    /// recency. An `Unloaded` page transitions back to `Requested`.
    pub fn request(&mut self, key: K, priority: f32, frame: u64) {
        let entry = self.entries.entry(key).or_insert(PageEntry {
            residency: PageResidency::Requested,
            priority,
            last_used_frame: frame,
        });
        entry.priority = entry.priority.max(priority);
        entry.last_used_frame = entry.last_used_frame.max(frame);
        if entry.residency == PageResidency::Unloaded {
            entry.residency = PageResidency::Requested;
        }
    }

    /// Marks a tracked page resident once the backend reports its upload done.
    pub fn mark_resident(&mut self, key: K) {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.residency = PageResidency::Resident;
        }
    }

    /// Refreshes recency for a page still in use without changing priority.
    pub fn touch(&mut self, key: K, frame: u64) {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_used_frame = entry.last_used_frame.max(frame);
        }
    }

    /// Drops a page from the table entirely (physical storage freed).
    pub fn evict(&mut self, key: K) {
        self.entries.remove(&key);
    }

    /// Current residency state of `key`.
    #[must_use]
    pub fn residency(&self, key: K) -> PageResidency {
        self.entries
            .get(&key)
            .map_or(PageResidency::Unloaded, |e| e.residency)
    }

    /// Number of pages currently resident.
    #[must_use]
    pub fn resident_count(&self) -> usize {
        self.entries
            .values()
            .filter(|e| e.residency == PageResidency::Resident)
            .count()
    }

    /// Total number of tracked pages.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table tracks no pages.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Chooses resident pages to evict so at most `resident_budget` remain.
    ///
    /// Pages whose `last_used_frame` is at or after `protect_frame` are treated
    /// as in-use this frame and are never returned. Among the rest, the lowest
    /// priority is evicted first, breaking ties by oldest use. Returns the keys
    /// to drop, best-effort: if protection prevents reaching the budget, fewer
    /// than the ideal count are returned.
    #[must_use]
    pub fn select_evictions(&self, resident_budget: usize, protect_frame: u64) -> Vec<K> {
        let mut resident: Vec<(K, PageEntry)> = self
            .entries
            .iter()
            .filter(|(_, e)| e.residency == PageResidency::Resident)
            .map(|(k, e)| (*k, *e))
            .collect();
        if resident.len() <= resident_budget {
            return Vec::new();
        }
        let excess = resident.len() - resident_budget;
        resident.sort_by(|a, b| {
            a.1.priority
                .total_cmp(&b.1.priority)
                .then(a.1.last_used_frame.cmp(&b.1.last_used_frame))
        });
        let mut victims = Vec::new();
        for (key, entry) in resident {
            if victims.len() >= excess {
                break;
            }
            if entry.last_used_frame >= protect_frame {
                continue;
            }
            victims.push(key);
        }
        victims
    }
}

/// Accumulates page references for one frame, deduplicated by page key.
///
/// A single physical page routinely backs many drawn primitives, so a frame
/// names the same key repeatedly. This batch collapses those into one request
/// per unique key, keeping the highest priority any reference reported, then
/// flushes them to a [`ResidencyTable`] in a single deterministic pass. Reuse
/// it across frames via [`clear`](RequestBatch::clear).
#[derive(Clone, Debug)]
pub struct RequestBatch<K: Copy + Ord> {
    priorities: BTreeMap<K, f32>,
}

impl<K: Copy + Ord> Default for RequestBatch<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Copy + Ord> RequestBatch<K> {
    /// Builds an empty batch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            priorities: BTreeMap::new(),
        }
    }

    /// Records one reference to `key` at screen-importance `priority`.
    ///
    /// If the page was already referenced this frame, the higher priority wins
    /// so a page pulled by any high-importance primitive streams in with that
    /// urgency regardless of visit order.
    pub fn record(&mut self, key: K, priority: f32) {
        let slot = self.priorities.entry(key).or_insert(priority);
        if *slot < priority {
            *slot = priority;
        }
    }

    /// Number of distinct pages referenced this frame.
    #[must_use]
    pub fn len(&self) -> usize {
        self.priorities.len()
    }

    /// Returns `true` when no page has been referenced.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.priorities.is_empty()
    }

    /// Highest priority recorded for `key`, or `None` if it was never referenced.
    #[must_use]
    pub fn priority(&self, key: K) -> Option<f32> {
        self.priorities.get(&key).copied()
    }

    /// Issues one request per unique page to `table`, tagged with `frame`.
    ///
    /// Iteration follows key order, so the table sees a stable request sequence
    /// every frame. The batch is left intact; call [`clear`](Self::clear) to
    /// reuse it for the next frame.
    pub fn flush(&self, table: &mut ResidencyTable<K>, frame: u64) {
        for (&key, &priority) in &self.priorities {
            table.request(key, priority, frame);
        }
    }

    /// Drops every recorded reference so the batch can be reused next frame.
    pub fn clear(&mut self) {
        self.priorities.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_table_request_and_evict() {
        let mut table: ResidencyTable<u32> = ResidencyTable::new();
        assert_eq!(table.residency(1), PageResidency::Unloaded);
        table.request(1, 2.0, 5);
        table.request(1, 1.0, 9);
        table.mark_resident(1);
        assert_eq!(table.residency(1), PageResidency::Resident);
        // Lower re-request never lowers survival priority; recency advanced.
        assert!(table.select_evictions(0, 9).is_empty());
        assert_eq!(table.select_evictions(0, 10), alloc::vec![1]);
    }

    #[test]
    fn generic_batch_coalesces_then_flushes() {
        let mut batch: RequestBatch<u32> = RequestBatch::new();
        batch.record(1, 3.0);
        batch.record(1, 7.0);
        batch.record(2, 1.0);
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.priority(1), Some(7.0));
        let mut table = ResidencyTable::new();
        batch.flush(&mut table, 4);
        assert_eq!(table.len(), 2);
        assert_eq!(table.residency(1), PageResidency::Requested);
    }
}
