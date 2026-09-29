//! Coalescing of per-cluster page requests before they hit the residency table.
//!
//! A single cluster page backs many clusters, so a selected cut routinely names
//! the same [`GeometryPageKey`] from dozens of drawn clusters in one frame.
//! Forwarding each of those to [`GeometryPageTable::request`] individually is
//! wasteful and lets whichever cluster happens to be visited last dictate the
//! page's streaming priority. This batch coalesces a frame's page references
//! into one request per unique page, keeping the highest screen-importance
//! priority any referencing cluster reported, then flushes them to the table in
//! a single deterministic pass. The batch is GPU-independent and reusable across
//! frames via [`clear`](PageRequestBatch::clear).

use super::page_table::GeometryPageTable;
use super::GeometryPageKey;
use alloc::collections::BTreeMap;

/// Accumulates page references for one frame, deduplicated by page key.
///
/// Entries are keyed on [`GeometryPageKey`], whose ordering gives the batch a
/// deterministic iteration and flush order independent of the cut's visit
/// order. Recording the same page more than once keeps the maximum priority.
#[derive(Clone, Debug, Default)]
pub struct PageRequestBatch {
    priorities: BTreeMap<GeometryPageKey, f32>,
}

impl PageRequestBatch {
    /// Builds an empty batch.
    #[must_use]
    pub fn new() -> Self {
        Self {
            priorities: BTreeMap::new(),
        }
    }

    /// Records one cluster's reference to `key` at screen-importance `priority`.
    ///
    /// If the page was already referenced this frame, the higher priority wins
    /// so a page pulled by any high-coverage cluster streams in with that
    /// urgency regardless of visit order.
    pub fn record(&mut self, key: GeometryPageKey, priority: f32) {
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
    pub fn priority(&self, key: GeometryPageKey) -> Option<f32> {
        self.priorities.get(&key).copied()
    }

    /// Issues one request per unique page to `table`, tagged with `frame`.
    ///
    /// Iteration follows [`GeometryPageKey`] order, so the table sees a stable
    /// request sequence every frame. The batch is left intact; call
    /// [`clear`](Self::clear) to reuse it for the next frame.
    pub fn flush(&self, table: &mut GeometryPageTable, frame: u64) {
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
    use super::super::page_table::PageResidency;
    use super::*;

    fn key(page: u32) -> GeometryPageKey {
        GeometryPageKey::new(0, page)
    }

    #[test]
    fn dedups_pages_keeping_max_priority() {
        let mut batch = PageRequestBatch::new();
        batch.record(key(1), 2.0);
        batch.record(key(1), 9.0);
        batch.record(key(1), 4.0);
        assert_eq!(batch.len(), 1);
        assert_eq!(batch.priority(key(1)), Some(9.0));
    }

    #[test]
    fn tracks_distinct_pages_separately() {
        let mut batch = PageRequestBatch::new();
        batch.record(key(1), 1.0);
        batch.record(key(2), 5.0);
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.priority(key(2)), Some(5.0));
        assert!(batch.priority(key(3)).is_none());
    }

    #[test]
    fn flush_issues_one_request_per_page() {
        let mut batch = PageRequestBatch::new();
        batch.record(key(1), 3.0);
        batch.record(key(1), 7.0);
        batch.record(key(2), 1.0);
        let mut table = GeometryPageTable::new();
        batch.flush(&mut table, 10);
        // Two distinct pages requested; the batch collapsed three references.
        assert_eq!(table.len(), 2);
        assert_eq!(table.residency(key(1)), PageResidency::Requested);
        assert_eq!(table.residency(key(2)), PageResidency::Requested);
    }

    #[test]
    fn clear_resets_the_batch() {
        let mut batch = PageRequestBatch::new();
        batch.record(key(1), 1.0);
        batch.clear();
        assert!(batch.is_empty());
        assert!(batch.priority(key(1)).is_none());
    }

    #[test]
    fn empty_batch_flush_is_a_noop() {
        let batch = PageRequestBatch::new();
        let mut table = GeometryPageTable::new();
        batch.flush(&mut table, 0);
        assert!(table.is_empty());
    }
}
