//! Page residency and per-frame request coalescing for virtual shadow maps.
//!
//! A virtual shadow map streams fixed-size depth pages into a bounded physical
//! pool exactly like virtualized geometry does: each frame names the pages a
//! light needs at some screen-importance priority, the backend uploads them,
//! and the pool evicts the least-important idle pages when it overflows. That
//! lifecycle is subsystem-agnostic, so it is the shared [`crate::paging`] machine
//! instantiated on [`ShadowPageKey`] rather than a second hand-written copy.
//!
//! Callers drive it as they do the geometry table: accumulate a frame's page
//! references into a [`ShadowRequestBatch`], flush the deduplicated set into the
//! [`ShadowResidencyTable`], mark pages resident as the backend confirms their
//! uploads, and each frame ask the table which resident pages to evict against
//! the physical-page budget.

use super::ShadowPageKey;

/// Session-lifetime residency table over virtual shadow-map pages.
///
/// The [`crate::paging::ResidencyTable`] machine keyed on [`ShadowPageKey`]:
/// lifecycle, priority policy and budget-driven eviction are shared with every
/// other virtualized subsystem.
pub type ShadowResidencyTable = crate::paging::ResidencyTable<ShadowPageKey>;

/// Per-frame coalescing of shadow page requests keyed on [`ShadowPageKey`].
///
/// The shared [`crate::paging::RequestBatch`] instantiated on the shadow page
/// key: it collapses the many receivers that pull the same clip page into one
/// request at the highest priority seen, then flushes them to a
/// [`ShadowResidencyTable`] in one stable, key-ordered pass.
pub type ShadowRequestBatch = crate::paging::RequestBatch<ShadowPageKey>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paging::PageResidency;

    fn key(level: u16, x: u16, y: u16) -> ShadowPageKey {
        ShadowPageKey {
            light: 0,
            level,
            x,
            y,
        }
    }

    #[test]
    fn request_then_resident_transitions() {
        let mut table = ShadowResidencyTable::new();
        assert_eq!(table.residency(key(0, 0, 0)), PageResidency::Unloaded);
        table.request(key(0, 0, 0), 1.0, 10);
        assert_eq!(table.residency(key(0, 0, 0)), PageResidency::Requested);
        table.mark_resident(key(0, 0, 0));
        assert_eq!(table.residency(key(0, 0, 0)), PageResidency::Resident);
        assert_eq!(table.resident_count(), 1);
    }

    #[test]
    fn batch_dedups_keeping_max_priority_then_flushes() {
        let mut batch = ShadowRequestBatch::new();
        batch.record(key(1, 2, 3), 2.0);
        batch.record(key(1, 2, 3), 9.0);
        batch.record(key(1, 2, 4), 1.0);
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.priority(key(1, 2, 3)), Some(9.0));
        let mut table = ShadowResidencyTable::new();
        batch.flush(&mut table, 4);
        assert_eq!(table.len(), 2);
        assert_eq!(table.residency(key(1, 2, 3)), PageResidency::Requested);
    }

    #[test]
    fn eviction_drops_lowest_priority_first() {
        let mut table = ShadowResidencyTable::new();
        for (y, prio) in [(0u16, 3.0f32), (1, 1.0), (2, 2.0)] {
            table.request(key(0, 0, y), prio, 1);
            table.mark_resident(key(0, 0, y));
        }
        // Budget 1, nothing protected this frame => evict the two lowest
        // priorities (1.0 then 2.0), keeping the 3.0 page.
        let victims = table.select_evictions(1, 100);
        assert_eq!(victims, alloc::vec![key(0, 0, 1), key(0, 0, 2)]);
    }

    #[test]
    fn eviction_protects_pages_used_this_frame() {
        let mut table = ShadowResidencyTable::new();
        table.request(key(0, 0, 0), 1.0, 50);
        table.mark_resident(key(0, 0, 0));
        table.request(key(0, 0, 1), 1.0, 1);
        table.mark_resident(key(0, 0, 1));
        // Both resident, budget 1. The page touched at frame 50 is protected at
        // protect_frame 50, so only the stale page is eligible.
        let victims = table.select_evictions(1, 50);
        assert_eq!(victims, alloc::vec![key(0, 0, 1)]);
    }
}
