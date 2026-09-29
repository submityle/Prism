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

pub use crate::paging::{PageEntry, PageResidency};

/// Session-lifetime residency table over virtualized geometry pages.
///
/// This is the [`crate::paging::ResidencyTable`] machine instantiated on
/// [`GeometryPageKey`]; the lifecycle, priority policy and budget-driven
/// eviction are shared with every other virtualized subsystem.
pub type GeometryPageTable = crate::paging::ResidencyTable<GeometryPageKey>;

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
