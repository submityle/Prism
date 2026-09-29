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

use super::GeometryPageKey;

/// Per-frame coalescing of cluster page requests keyed on [`GeometryPageKey`].
///
/// This is the shared [`crate::paging::RequestBatch`] instantiated on the
/// geometry page key: it collapses a cut's many references to the same page
/// into one request at the highest screen-importance priority seen, then
/// flushes them to a [`super::page_table::GeometryPageTable`] in one stable,
/// key-ordered pass.
pub type PageRequestBatch = crate::paging::RequestBatch<GeometryPageKey>;

#[cfg(test)]
mod tests {
    use super::super::page_table::{GeometryPageTable, PageResidency};
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
