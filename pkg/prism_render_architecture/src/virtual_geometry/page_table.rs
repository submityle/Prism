//! Cluster page residency bookkeeping and budget-driven eviction.
//!
//! The table tracks every page the selector has asked about this session and
//! its lifecycle state, from a streaming request through residency. It carries
//! no GPU handles; the backend keys its physical page store on
//! [`GeometryPageKey`] and consults this table to decide what to upload and
//! what to drop. Eviction is priority-first (least screen-important pages go
//! first) with a recency tie-break, and pages touched in the current frame are
//! protected so an in-view page is never evicted out from under the raster.

use super::GeometryPageKey;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// Lifecycle state of a tracked geometry page.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PageResidency {
    /// Known to the table but neither requested nor resident.
    #[default]
    Unloaded,
    /// A streaming request is outstanding.
    Requested,
    /// Backed by physical storage and ready to raster.
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

/// Session-lifetime residency table over virtualized geometry pages.
#[derive(Clone, Debug, Default)]
pub struct GeometryPageTable {
    entries: BTreeMap<GeometryPageKey, PageEntry>,
}

impl GeometryPageTable {
    /// Creates an empty table.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Records that `key` is needed this frame at the given screen `priority`.
    ///
    /// An unknown page becomes [`PageResidency::Requested`]; a known page keeps
    /// its residency but raises its priority to the max seen and advances its
    /// recency. An `Unloaded` page transitions back to `Requested`.
    pub fn request(&mut self, key: GeometryPageKey, priority: f32, frame: u64) {
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
    pub fn mark_resident(&mut self, key: GeometryPageKey) {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.residency = PageResidency::Resident;
        }
    }

    /// Refreshes recency for a page still in view without changing priority.
    pub fn touch(&mut self, key: GeometryPageKey, frame: u64) {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_used_frame = entry.last_used_frame.max(frame);
        }
    }

    /// Drops a page from the table entirely (physical storage freed).
    pub fn evict(&mut self, key: GeometryPageKey) {
        self.entries.remove(&key);
    }

    /// Current residency state of `key`.
    #[must_use]
    pub fn residency(&self, key: GeometryPageKey) -> PageResidency {
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
    pub fn select_evictions(
        &self,
        resident_budget: usize,
        protect_frame: u64,
    ) -> Vec<GeometryPageKey> {
        let mut resident: Vec<(GeometryPageKey, PageEntry)> = self
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

#[cfg(test)]
mod tests {
    use super::*;

    fn key(page: u32) -> GeometryPageKey {
        GeometryPageKey::new(7, page)
    }

    #[test]
    fn request_then_resident_transitions() {
        let mut table = GeometryPageTable::new();
        assert_eq!(table.residency(key(0)), PageResidency::Unloaded);
        table.request(key(0), 1.0, 10);
        assert_eq!(table.residency(key(0)), PageResidency::Requested);
        table.mark_resident(key(0));
        assert_eq!(table.residency(key(0)), PageResidency::Resident);
        assert_eq!(table.resident_count(), 1);
    }

    #[test]
    fn request_keeps_max_priority_and_latest_frame() {
        let mut table = GeometryPageTable::new();
        table.request(key(0), 2.0, 5);
        table.request(key(0), 1.0, 9);
        table.mark_resident(key(0));
        // A lower-priority re-request must not lower the survival priority, and
        // the recency must advance to the newest frame.
        let victims = table.select_evictions(0, 9);
        assert!(victims.is_empty(), "page touched this frame is protected");
        let victims = table.select_evictions(0, 10);
        assert_eq!(victims, alloc::vec![key(0)]);
    }

    #[test]
    fn eviction_drops_lowest_priority_first() {
        let mut table = GeometryPageTable::new();
        for (page, prio) in [(0u32, 3.0f32), (1, 1.0), (2, 2.0)] {
            table.request(key(page), prio, 1);
            table.mark_resident(key(page));
        }
        // Budget of 1 with nothing protected this frame => evict the two
        // lowest-priority pages (1.0 then 2.0), keeping the 3.0 page.
        let victims = table.select_evictions(1, 100);
        assert_eq!(victims, alloc::vec![key(1), key(2)]);
    }

    #[test]
    fn eviction_respects_budget_and_protection() {
        let mut table = GeometryPageTable::new();
        table.request(key(0), 1.0, 50);
        table.mark_resident(key(0));
        table.request(key(1), 1.0, 1);
        table.mark_resident(key(1));
        // Both resident, budget 1. key(0) was used at frame 50 and is protected
        // at protect_frame 50, so only key(1) is eligible.
        let victims = table.select_evictions(1, 50);
        assert_eq!(victims, alloc::vec![key(1)]);
    }
}
