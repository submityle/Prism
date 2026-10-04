//! Archetype / chunk fragmentation diagnostics (design §16.6 / §22 risk #5).
//!
//! Two distinct fragmentation pressures degrade a chunked-archetype ECS, and
//! this module quantifies both from a single read-only pass:
//!
//! * **Archetype fragmentation** — live entities spread thin across many
//!   archetypes (the "archetype explosion" that fragmenting relations risk,
//!   design §11 / §22 risk #5). Its smell is a high ratio of *singleton*
//!   archetypes (one live entity each): iteration pays per-archetype setup cost
//!   for almost no payload.
//! * **Chunk internal fragmentation** — allocated 16 KiB chunks (design §5.3)
//!   left mostly empty, wasting resident memory. Its smell is low chunk
//!   occupancy and whole chunks that could be reclaimed if rows were compacted.
//!
//! The report builds on [`WorldReport`](super::inspector::WorldReport), reusing
//! its per-archetype occupancy snapshot, and folds it into fragmentation-centric
//! rollups plus a worst-first ranking an editor panel can surface. Only
//! archetypes that actually own storage (at least one allocated chunk) appear;
//! a pristine empty archetype has no footprint to fragment.
//!
//! Capture is `O(archetypes)`, read-only, and deterministic: entries are ranked
//! by ascending occupancy with an archetype-id tie-break.

use alloc::vec::Vec;

use crate::archetype::ArchetypeId;
use crate::diagnostics::inspector::WorldReport;
use crate::world::World;

/// Fragmentation facts for a single storage-owning archetype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchetypeFragmentEntry {
    /// The archetype's stable id.
    pub id: ArchetypeId,
    /// Number of component types in the archetype's identity.
    pub component_count: usize,
    /// Live entity rows currently stored.
    pub live_rows: usize,
    /// Allocated 16 KiB chunks backing this archetype.
    pub chunk_count: usize,
    /// Row capacity of a single chunk (`N` in design §5.3).
    pub rows_per_chunk: usize,
}

impl ArchetypeFragmentEntry {
    /// Total addressable rows across every allocated chunk.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.chunk_count * self.rows_per_chunk
    }

    /// Allocated-but-unoccupied rows (`capacity - live_rows`).
    #[inline]
    pub fn free_slots(&self) -> usize {
        self.capacity().saturating_sub(self.live_rows)
    }

    /// Fraction of allocated capacity that is live, in `[0.0, 1.0]`; `0.0` when
    /// nothing is allocated.
    #[inline]
    pub fn occupancy(&self) -> f32 {
        let capacity = self.capacity();
        if capacity == 0 {
            0.0
        } else {
            self.live_rows as f32 / capacity as f32
        }
    }

    /// Whether this archetype owns chunks but holds no live rows — pure wasted
    /// resident memory, the strongest reclamation candidate.
    #[inline]
    pub fn is_empty_allocated(&self) -> bool {
        self.chunk_count > 0 && self.live_rows == 0
    }

    /// Whether this archetype holds exactly one entity — the archetype-explosion
    /// smell (design §22 risk #5).
    #[inline]
    pub fn is_singleton(&self) -> bool {
        self.live_rows == 1
    }

    /// Minimum chunks required to hold the live rows if they were compacted.
    #[inline]
    pub fn min_chunks_needed(&self) -> usize {
        if self.rows_per_chunk == 0 {
            0
        } else {
            self.live_rows.div_ceil(self.rows_per_chunk)
        }
    }

    /// Whole chunks reclaimable by compacting live rows (`chunk_count -
    /// min_chunks_needed`). Zero for a tightly packed archetype.
    #[inline]
    pub fn reclaimable_chunks(&self) -> usize {
        self.chunk_count.saturating_sub(self.min_chunks_needed())
    }
}

/// A whole-world fragmentation summary with a worst-first archetype ranking
/// (design §16.6).
///
/// Produced by [`ArchetypeFragmentationReport::capture`].
/// [`entries`](Self::entries) holds every storage-owning archetype sorted by
/// ascending [`occupancy`](ArchetypeFragmentEntry::occupancy) (worst first),
/// with an ascending archetype-id tie-break for determinism.
#[derive(Debug, Clone, Default)]
pub struct ArchetypeFragmentationReport {
    /// Storage-owning archetypes, worst occupancy first.
    pub entries: Vec<ArchetypeFragmentEntry>,
    /// Total archetypes in the world, including empty / storage-free ones.
    pub archetype_count: usize,
    /// Archetypes holding at least one live row.
    pub populated_archetype_count: usize,
    /// Archetypes holding exactly one live row.
    pub singleton_archetype_count: usize,
    /// Archetypes that own chunks but hold no live rows.
    pub empty_allocated_count: usize,
    /// Total live rows across every archetype.
    pub total_live_rows: usize,
    /// Total addressable rows across every allocated chunk.
    pub total_capacity: usize,
    /// Total allocated chunks.
    pub total_chunks: usize,
    /// Total whole chunks reclaimable by compaction across the world.
    pub reclaimable_chunks: usize,
}

impl ArchetypeFragmentationReport {
    /// Capture a fragmentation summary of `world`.
    #[inline]
    pub fn capture(world: &World) -> Self {
        Self::from_world_report(&WorldReport::capture(world))
    }

    /// Fold an existing [`WorldReport`] into a fragmentation summary, avoiding a
    /// second walk of the world when a structural snapshot is already in hand.
    pub fn from_world_report(report: &WorldReport) -> Self {
        let mut entries = Vec::new();
        let mut populated_archetype_count = 0;
        let mut singleton_archetype_count = 0;
        let mut empty_allocated_count = 0;
        let mut total_live_rows = 0;
        let mut total_capacity = 0;
        let mut total_chunks = 0;
        let mut reclaimable_chunks = 0;

        for a in &report.archetypes {
            let occ = &a.occupancy;
            total_live_rows += occ.live_rows;
            if occ.live_rows > 0 {
                populated_archetype_count += 1;
            }
            if occ.live_rows == 1 {
                singleton_archetype_count += 1;
            }

            // Only archetypes that own chunks can fragment.
            if occ.chunk_count == 0 {
                continue;
            }
            let entry = ArchetypeFragmentEntry {
                id: a.id,
                component_count: a.components.len(),
                live_rows: occ.live_rows,
                chunk_count: occ.chunk_count,
                rows_per_chunk: occ.rows_per_chunk,
            };
            total_capacity += entry.capacity();
            total_chunks += entry.chunk_count;
            reclaimable_chunks += entry.reclaimable_chunks();
            if entry.is_empty_allocated() {
                empty_allocated_count += 1;
            }
            entries.push(entry);
        }

        // Worst occupancy first; archetype id breaks ties deterministically.
        entries.sort_by(|a, b| {
            a.occupancy()
                .total_cmp(&b.occupancy())
                .then_with(|| a.id.cmp(&b.id))
        });

        Self {
            entries,
            archetype_count: report.archetype_count,
            populated_archetype_count,
            singleton_archetype_count,
            empty_allocated_count,
            total_live_rows,
            total_capacity,
            total_chunks,
            reclaimable_chunks,
        }
    }

    /// Whether no storage-owning archetype exists (nothing to analyze).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of storage-owning archetypes analyzed.
    #[inline]
    pub fn analyzed_count(&self) -> usize {
        self.entries.len()
    }

    /// World-wide chunk occupancy: live rows over allocated capacity, in
    /// `[0.0, 1.0]`; `0.0` when nothing is allocated.
    #[inline]
    pub fn occupancy(&self) -> f32 {
        if self.total_capacity == 0 {
            0.0
        } else {
            self.total_live_rows as f32 / self.total_capacity as f32
        }
    }

    /// World-wide internal fragmentation: the unused fraction of allocated
    /// capacity (`1.0 - occupancy`).
    #[inline]
    pub fn internal_fragmentation(&self) -> f32 {
        1.0 - self.occupancy()
    }

    /// Fraction of populated archetypes that hold a single entity, in
    /// `[0.0, 1.0]`; the archetype-explosion indicator (design §22 risk #5).
    /// `0.0` when nothing is populated.
    #[inline]
    pub fn singleton_ratio(&self) -> f32 {
        if self.populated_archetype_count == 0 {
            0.0
        } else {
            self.singleton_archetype_count as f32 / self.populated_archetype_count as f32
        }
    }

    /// The most fragmented storage-owning archetype (lowest occupancy), or
    /// `None` when none own storage. Ties resolve to the lowest archetype id.
    #[inline]
    pub fn worst(&self) -> Option<&ArchetypeFragmentEntry> {
        self.entries.first()
    }

    /// The entry for `id`, if that archetype owns storage.
    pub fn entry(&self, id: ArchetypeId) -> Option<&ArchetypeFragmentEntry> {
        self.entries.iter().find(|e| e.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;
    use crate::world::World;

    struct A;
    impl Component for A {}
    struct B;
    impl Component for B {}
    struct C;
    impl Component for C {}

    #[test]
    fn empty_world_has_no_storage_to_analyze() {
        let world = World::new();
        let report = ArchetypeFragmentationReport::capture(&world);
        assert!(report.is_empty());
        assert_eq!(report.analyzed_count(), 0);
        assert_eq!(report.total_live_rows, 0);
        assert_eq!(report.total_capacity, 0);
        assert_eq!(report.occupancy(), 0.0);
        assert_eq!(report.singleton_ratio(), 0.0);
        assert!(report.worst().is_none());
    }

    #[test]
    fn distinct_component_sets_fragment_into_singletons() {
        let mut world = World::new();
        // Three distinct archetypes, one entity each — maximal archetype
        // fragmentation.
        world.spawn(A);
        world.spawn(B);
        world.spawn((A, B));

        let report = ArchetypeFragmentationReport::capture(&world);

        assert_eq!(report.populated_archetype_count, 3);
        assert_eq!(report.singleton_archetype_count, 3);
        assert_eq!(report.total_live_rows, 3);
        assert_eq!(report.singleton_ratio(), 1.0, "every populated archetype is a singleton");
        assert_eq!(report.analyzed_count(), 3, "each singleton owns one chunk");

        // A singleton leaves its chunk almost entirely empty.
        let worst = report.worst().unwrap();
        assert!(worst.is_singleton());
        assert!(worst.occupancy() > 0.0 && worst.occupancy() < 1.0);
        assert!(report.internal_fragmentation() > 0.0);
        // A singleton still needs its one chunk — nothing whole to reclaim.
        assert_eq!(report.reclaimable_chunks, 0);
    }

    #[test]
    fn entries_are_ranked_worst_occupancy_first() {
        let mut world = World::new();
        // One archetype with three rows, two archetypes with one row each.
        world.spawn(A);
        world.spawn(A);
        world.spawn(A);
        world.spawn(B);
        world.spawn(C);

        let report = ArchetypeFragmentationReport::capture(&world);
        assert_eq!(report.populated_archetype_count, 3);
        assert_eq!(report.singleton_archetype_count, 2, "B and C are singletons");

        // Ascending occupancy, archetype-id tie-break.
        for pair in report.entries.windows(2) {
            let (lo, hi) = (&pair[0], &pair[1]);
            assert!(
                lo.occupancy() < hi.occupancy()
                    || (lo.occupancy().to_bits() == hi.occupancy().to_bits()
                        && lo.id <= hi.id),
                "entries must be worst-occupancy-first, id-ascending on ties"
            );
        }

        // The {A} archetype holds the most rows, so it is the least fragmented
        // and must not be the worst entry.
        let worst = report.worst().unwrap();
        assert_eq!(worst.live_rows, 1, "a singleton is the most fragmented");
    }

    #[test]
    fn from_world_report_matches_direct_capture() {
        let mut world = World::new();
        world.spawn(A);
        world.spawn((A, B));

        let via_world = ArchetypeFragmentationReport::capture(&world);
        let via_report =
            ArchetypeFragmentationReport::from_world_report(&WorldReport::capture(&world));

        assert_eq!(via_world.entries, via_report.entries);
        assert_eq!(via_world.total_live_rows, via_report.total_live_rows);
        assert_eq!(via_world.total_capacity, via_report.total_capacity);
        assert_eq!(
            via_world.singleton_archetype_count,
            via_report.singleton_archetype_count
        );
    }
}
