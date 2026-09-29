//! Residency table and lifecycle state machine for streamed texture pages.
//!
//! A virtual-texture system streams fixed-size tiles (pages) of individual mip
//! levels into a bounded physical pool. Each frame the `GPU` feedback pass names
//! the pages a view needs; the backend uploads them; when the pool overflows the
//! least valuable idle pages are dropped. This module owns the `CPU`-side
//! bookkeeping half of that loop: it records, per [`TexturePageKey`], the page's
//! lifecycle state, the streaming priority last computed for it, its byte cost,
//! and the frame it was last touched. It holds no `GPU` handles and every
//! container iterates in key order, so request and eviction sequences are
//! deterministic and reproducible.
//!
//! The scheduler in [`crate::texture_streaming::scheduler`] consumes this table
//! to pick a budget-bounded resident set, and the priority stored per page is
//! produced by [`crate::texture_streaming::feedback`].

use super::TexturePageKey;
use alloc::collections::BTreeMap;

/// Lifecycle state of a tracked texture page.
///
/// A page climbs `NotResident -> Requested -> Resident` as the feedback pass
/// asks for it and the backend confirms the upload, and falls back to
/// `NotResident` when evicted. Pages the table has never seen are reported as
/// `NotResident`.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PageResidency {
    /// Tracked but not backed by physical storage and not currently requested.
    #[default]
    NotResident,
    /// A streaming upload is outstanding for this page.
    Requested,
    /// Backed by physical storage and ready to sample.
    Resident,
}

impl PageResidency {
    /// Whether the page currently occupies physical storage.
    #[must_use]
    pub const fn is_resident(self) -> bool {
        matches!(self, Self::Resident)
    }

    /// Whether the page is either resident or has an upload in flight.
    ///
    /// Such pages are the scheduler's candidates: they are the ones a view has
    /// asked for or already paid to keep, as opposed to fully dropped pages.
    #[must_use]
    pub const fn is_live(self) -> bool {
        matches!(self, Self::Requested | Self::Resident)
    }
}

/// Per-page bookkeeping record held by the residency table.
///
/// The record is pure integer state so ordering and equality are exact; the
/// streaming priority is the fixed-point score computed by
/// [`crate::texture_streaming::feedback`], never a floating-point value.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PageRecord {
    /// Current lifecycle state.
    pub residency: PageResidency,
    /// Fixed-point streaming priority; higher survives eviction longer.
    pub priority: u64,
    /// Physical byte cost of this page once resident.
    pub byte_cost: u64,
    /// Frame index the page was last requested or touched (for `LRU` order).
    pub last_used_frame: u64,
}

impl PageRecord {
    /// Builds a freshly requested record.
    #[must_use]
    const fn requested(priority: u64, byte_cost: u64, frame: u64) -> Self {
        Self {
            residency: PageResidency::Requested,
            priority,
            byte_cost,
            last_used_frame: frame,
        }
    }
}

/// Session-lifetime residency table over streamed texture pages.
///
/// Keyed on [`TexturePageKey`] and ordered by it, the table drives the page
/// lifecycle and feeds the budget scheduler. It is intentionally free of any
/// `GPU` handle: the backend keys its own physical store on [`TexturePageKey`]
/// and consults this table to decide what to upload and drop.
#[derive(Clone, Debug)]
pub struct TextureResidencyTable {
    entries: BTreeMap<TexturePageKey, PageRecord>,
}

impl Default for TextureResidencyTable {
    fn default() -> Self {
        Self::new()
    }
}

impl TextureResidencyTable {
    /// Creates an empty table.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Records that `key` is needed this frame at the given streaming inputs.
    ///
    /// An unknown page becomes [`PageResidency::Requested`]. A known page keeps
    /// its residency but raises its priority to the max seen this frame,
    /// advances its recency, and adopts the latest byte cost. A page that had
    /// been evicted to [`PageResidency::NotResident`] transitions back to
    /// `Requested` so the scheduler can reconsider it.
    pub fn request(&mut self, key: TexturePageKey, priority: u64, byte_cost: u64, frame: u64) {
        let entry = self
            .entries
            .entry(key)
            .or_insert_with(|| PageRecord::requested(priority, byte_cost, frame));
        entry.priority = entry.priority.max(priority);
        entry.last_used_frame = entry.last_used_frame.max(frame);
        entry.byte_cost = byte_cost;
        if entry.residency == PageResidency::NotResident {
            entry.residency = PageResidency::Requested;
        }
    }

    /// Marks a tracked page resident once the backend reports its upload done.
    ///
    /// No-op for an unknown page: only pages the table already tracks can be
    /// promoted, matching the request-before-upload lifecycle.
    pub fn mark_resident(&mut self, key: TexturePageKey) {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.residency = PageResidency::Resident;
        }
    }

    /// Refreshes recency for a page still in use without changing priority.
    ///
    /// Advances `last_used_frame` monotonically so a stale value can never
    /// overwrite a fresher one out of order.
    pub fn touch(&mut self, key: TexturePageKey, frame: u64) {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_used_frame = entry.last_used_frame.max(frame);
        }
    }

    /// Drops a page's physical backing, returning it to `NotResident`.
    ///
    /// The record is retained (priority and recency history survive) so the
    /// page falls out of the scheduler's live candidate set until something
    /// requests it again. Use [`forget`](Self::forget) to remove it entirely.
    pub fn evict(&mut self, key: TexturePageKey) {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.residency = PageResidency::NotResident;
        }
    }

    /// Removes a page from the table entirely, discarding its history.
    pub fn forget(&mut self, key: TexturePageKey) {
        self.entries.remove(&key);
    }

    /// Current residency state of `key`.
    #[must_use]
    pub fn residency(&self, key: TexturePageKey) -> PageResidency {
        self.entries
            .get(&key)
            .map_or(PageResidency::NotResident, |e| e.residency)
    }

    /// The full record for `key`, if tracked.
    #[must_use]
    pub fn record(&self, key: TexturePageKey) -> Option<PageRecord> {
        self.entries.get(&key).copied()
    }

    /// Number of pages currently resident.
    #[must_use]
    pub fn resident_count(&self) -> usize {
        self.entries
            .values()
            .filter(|e| e.residency.is_resident())
            .count()
    }

    /// Sum of byte costs of all currently resident pages.
    #[must_use]
    pub fn resident_bytes(&self) -> u64 {
        self.entries
            .values()
            .filter(|e| e.residency.is_resident())
            .map(|e| e.byte_cost)
            .sum()
    }

    /// Total number of tracked pages, regardless of state.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table tracks no pages at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterates all tracked pages in ascending key order.
    ///
    /// Deterministic by construction; the scheduler relies on this ordering to
    /// produce reproducible load and evict lists.
    pub fn iter(&self) -> impl Iterator<Item = (TexturePageKey, PageRecord)> + '_ {
        self.entries.iter().map(|(k, r)| (*k, *r))
    }

    /// Iterates the scheduler's candidate pages (requested or resident).
    pub fn live_pages(&self) -> impl Iterator<Item = (TexturePageKey, PageRecord)> + '_ {
        self.iter().filter(|(_, r)| r.residency.is_live())
    }

    /// Applies a scheduler plan: promotes loads to resident, drops evicts.
    ///
    /// This collapses the asynchronous upload into a single step and is meant
    /// for `CPU`-side simulation and tests; a live runtime instead issues the
    /// plan's uploads and calls [`mark_resident`](Self::mark_resident) as each
    /// completes, pending the `GPU` backend.
    pub fn apply_plan(&mut self, plan: &super::scheduler::StreamingPlan) {
        for &key in &plan.evicts {
            self.evict(key);
        }
        for &key in &plan.loads {
            self.mark_resident(key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn key(mip: u8, x: u16, y: u16) -> TexturePageKey {
        TexturePageKey {
            texture: 7,
            mip,
            layer: 0,
            x,
            y,
        }
    }

    #[test]
    fn unknown_page_reports_not_resident() {
        let table = TextureResidencyTable::new();
        assert_eq!(table.residency(key(0, 0, 0)), PageResidency::NotResident);
        assert!(table.is_empty());
        assert_eq!(table.record(key(0, 0, 0)), None);
    }

    #[test]
    fn request_then_resident_transitions() {
        let mut table = TextureResidencyTable::new();
        table.request(key(0, 0, 0), 100, 4096, 10);
        assert_eq!(table.residency(key(0, 0, 0)), PageResidency::Requested);
        table.mark_resident(key(0, 0, 0));
        assert_eq!(table.residency(key(0, 0, 0)), PageResidency::Resident);
        assert_eq!(table.resident_count(), 1);
        assert_eq!(table.resident_bytes(), 4096);
    }

    #[test]
    fn request_keeps_max_priority_and_recency_and_latest_cost() {
        let mut table = TextureResidencyTable::new();
        table.request(key(0, 0, 0), 100, 4096, 10);
        table.request(key(0, 0, 0), 50, 8192, 5);
        let rec = table.record(key(0, 0, 0)).expect("tracked");
        assert_eq!(rec.priority, 100, "priority keeps the max seen");
        assert_eq!(rec.last_used_frame, 10, "recency never regresses");
        assert_eq!(rec.byte_cost, 8192, "byte cost adopts the latest report");
    }

    #[test]
    fn mark_resident_ignores_unknown_page() {
        let mut table = TextureResidencyTable::new();
        table.mark_resident(key(0, 0, 0));
        assert_eq!(table.residency(key(0, 0, 0)), PageResidency::NotResident);
        assert!(table.is_empty());
    }

    #[test]
    fn evict_returns_to_not_resident_but_retains_history() {
        let mut table = TextureResidencyTable::new();
        table.request(key(0, 0, 0), 100, 4096, 10);
        table.mark_resident(key(0, 0, 0));
        table.evict(key(0, 0, 0));
        assert_eq!(table.residency(key(0, 0, 0)), PageResidency::NotResident);
        assert_eq!(table.resident_count(), 0);
        assert_eq!(table.len(), 1, "record retained after eviction");
        let rec = table.record(key(0, 0, 0)).expect("retained");
        assert_eq!(rec.priority, 100);
    }

    #[test]
    fn re_request_after_evict_becomes_requested() {
        let mut table = TextureResidencyTable::new();
        table.request(key(0, 0, 0), 100, 4096, 10);
        table.mark_resident(key(0, 0, 0));
        table.evict(key(0, 0, 0));
        table.request(key(0, 0, 0), 40, 4096, 20);
        assert_eq!(table.residency(key(0, 0, 0)), PageResidency::Requested);
    }

    #[test]
    fn forget_removes_record_entirely() {
        let mut table = TextureResidencyTable::new();
        table.request(key(0, 0, 0), 100, 4096, 10);
        table.forget(key(0, 0, 0));
        assert!(table.is_empty());
        assert_eq!(table.record(key(0, 0, 0)), None);
    }

    #[test]
    fn touch_advances_recency_monotonically() {
        let mut table = TextureResidencyTable::new();
        table.request(key(0, 0, 0), 100, 4096, 10);
        table.touch(key(0, 0, 0), 5);
        assert_eq!(
            table.record(key(0, 0, 0)).expect("tracked").last_used_frame,
            10
        );
        table.touch(key(0, 0, 0), 25);
        assert_eq!(
            table.record(key(0, 0, 0)).expect("tracked").last_used_frame,
            25
        );
    }

    #[test]
    fn iteration_is_key_ordered_and_deterministic() {
        let mut table = TextureResidencyTable::new();
        table.request(key(2, 0, 0), 1, 1, 1);
        table.request(key(0, 5, 9), 1, 1, 1);
        table.request(key(0, 1, 0), 1, 1, 1);
        let keys: Vec<TexturePageKey> = table.iter().map(|(k, _)| k).collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted, "iteration must be ascending key order");
    }

    #[test]
    fn live_pages_excludes_evicted() {
        let mut table = TextureResidencyTable::new();
        table.request(key(0, 0, 0), 100, 1, 1);
        table.mark_resident(key(0, 0, 0));
        table.request(key(0, 0, 1), 100, 1, 1);
        table.evict(key(0, 0, 1));
        let live: Vec<TexturePageKey> = table.live_pages().map(|(k, _)| k).collect();
        assert_eq!(live, alloc::vec![key(0, 0, 0)]);
    }
}
