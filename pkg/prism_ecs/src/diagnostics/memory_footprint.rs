//! Byte-level resident-memory accounting for Table-backed storage
//! (design §16.6 / §5.3 / §17).
//!
//! Where [`archetype_fragmentation`](super::archetype_fragmentation) measures
//! fragmentation in *rows* (how full the allocated chunk capacity is), this
//! module converts that same geometry into *bytes*, the unit an editor memory
//! panel and the design's "内存回收" budgeting (design §17) actually care
//! about. Two archetypes can share an identical row occupancy yet differ by
//! orders of magnitude in resident bytes depending on how wide their component
//! rows are, so the row view alone cannot rank what to reclaim first.
//!
//! # Honest byte model
//! The report mirrors exactly how [`Table`](crate::storage::table::Table) lays
//! out storage (design §6), so every figure is a real reserved-byte count, not
//! an estimate:
//!
//! * `bytes_per_row` is the sum of each component's
//!   [`Layout::size`](core::alloc::Layout::size) across the archetype's
//!   identity — the same sum the table uses to pick its chunk geometry
//!   (`Table::new`). It is pure element size with **no per-row padding**,
//!   matching the columnar `BlobVec` layout where each column packs
//!   same-type elements back to back.
//! * `allocated_bytes` is the payload the archetype's column `BlobVec`s
//!   actually reserve: `capacity * bytes_per_row`, where `capacity =
//!   chunk_count * rows_per_chunk`. This is reserved resident memory, live or
//!   not.
//! * `live_bytes` is `live_rows * bytes_per_row`: the payload backing entities
//!   that currently exist.
//! * `wasted_bytes` is `free_slots * bytes_per_row`: reserved payload behind
//!   empty row slots — the reclamation target.
//!
//! The 16 KiB chunk size ([`TARGET_CHUNK_BYTES`](crate::storage::TARGET_CHUNK_BYTES))
//! is a *sizing target*, not a physical per-chunk allocation: a column of
//! `k`-byte elements reserves `rows_per_chunk * k` bytes per chunk, which the
//! flooring in [`rows_per_chunk`](crate::storage::rows_per_chunk) keeps at or
//! below the target. [`chunk_fill_ratio`](ArchetypeMemoryEntry::chunk_fill_ratio)
//! exposes how much of that target a full chunk's payload actually uses, so the
//! geometric slack from flooring is visible without being misreported as
//! allocated bytes.
//!
//! # Scope
//! Figures describe the chunked Table-backed storage only (design §6), mirroring
//! [`WorldReport`]. `SparseSet` components are not laid out in archetype chunks
//! and contribute no bytes here. Zero-sized (marker) components add nothing to
//! `bytes_per_row`, so a ZST-only archetype reports zero resident payload even
//! though it owns chunk bookkeeping.
//!
//! Capture is `O(archetypes * components_per_archetype)`, read-only, and
//! deterministic: entries are ranked by descending `wasted_bytes` with an
//! ascending archetype-id tie-break.

use alloc::vec::Vec;

use crate::archetype::ArchetypeId;
use crate::component::Components;
use crate::diagnostics::inspector::WorldReport;
use crate::storage::TARGET_CHUNK_BYTES;
use crate::world::World;

/// Byte-level memory facts for a single storage-owning archetype.
///
/// All byte figures are reserved-payload counts derived from the archetype's
/// chunk geometry and row width; see the [module docs](self#honest-byte-model)
/// for the exact model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchetypeMemoryEntry {
    /// The archetype's stable id.
    pub id: ArchetypeId,
    /// Number of component types in the archetype's identity.
    pub component_count: usize,
    /// Sum of component element sizes in bytes (no per-row padding).
    pub bytes_per_row: usize,
    /// Row capacity of a single chunk (`N` in design §5.3).
    pub rows_per_chunk: usize,
    /// Allocated 16 KiB-targeted chunks backing this archetype.
    pub chunk_count: usize,
    /// Live entity rows currently stored.
    pub live_rows: usize,
}

impl ArchetypeMemoryEntry {
    /// Total addressable rows across every allocated chunk
    /// (`chunk_count * rows_per_chunk`).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.chunk_count * self.rows_per_chunk
    }

    /// Allocated-but-unoccupied rows (`capacity - live_rows`).
    #[inline]
    pub fn free_slots(&self) -> usize {
        self.capacity().saturating_sub(self.live_rows)
    }

    /// Reserved column payload backing this archetype
    /// (`capacity * bytes_per_row`), live or not.
    #[inline]
    pub fn allocated_bytes(&self) -> usize {
        self.capacity() * self.bytes_per_row
    }

    /// Payload backing the live entity rows (`live_rows * bytes_per_row`).
    #[inline]
    pub fn live_bytes(&self) -> usize {
        self.live_rows * self.bytes_per_row
    }

    /// Reserved payload behind empty row slots (`free_slots * bytes_per_row`) —
    /// the resident memory reclaimable by compacting this archetype.
    #[inline]
    pub fn wasted_bytes(&self) -> usize {
        self.free_slots() * self.bytes_per_row
    }

    /// Payload a single fully packed chunk holds
    /// (`rows_per_chunk * bytes_per_row`); always at or below
    /// [`TARGET_CHUNK_BYTES`] by the chunk-sizing floor.
    #[inline]
    pub fn chunk_payload_bytes(&self) -> usize {
        self.rows_per_chunk * self.bytes_per_row
    }

    /// Fraction of the 16 KiB chunk target a full chunk's payload uses, in
    /// `[0.0, 1.0]`. Below 1.0 reflects the geometric slack from flooring
    /// `TARGET_CHUNK_BYTES / bytes_per_row`; `0.0` for a zero-sized row.
    #[inline]
    pub fn chunk_fill_ratio(&self) -> f32 {
        self.chunk_payload_bytes() as f32 / TARGET_CHUNK_BYTES as f32
    }

    /// Fraction of allocated payload that is live, in `[0.0, 1.0]`; `0.0` when
    /// nothing is allocated or the row is zero-sized.
    #[inline]
    pub fn byte_occupancy(&self) -> f32 {
        let allocated = self.allocated_bytes();
        if allocated == 0 {
            0.0
        } else {
            self.live_bytes() as f32 / allocated as f32
        }
    }

    /// Whether this archetype reserves payload but holds no live rows — pure
    /// wasted resident bytes, the strongest reclamation candidate.
    #[inline]
    pub fn is_empty_allocated(&self) -> bool {
        self.allocated_bytes() > 0 && self.live_rows == 0
    }
}

/// A whole-world byte-level memory summary with a wasted-first archetype
/// ranking (design §16.6 / §17).
///
/// Produced by [`MemoryFootprintReport::capture`].
/// [`entries`](Self::entries) holds every archetype that reserves column
/// payload, sorted by descending [`wasted_bytes`](ArchetypeMemoryEntry::wasted_bytes)
/// (biggest reclamation win first), with an ascending archetype-id tie-break
/// for determinism.
#[derive(Debug, Clone, Default)]
pub struct MemoryFootprintReport {
    /// Payload-reserving archetypes, most wasted bytes first.
    pub entries: Vec<ArchetypeMemoryEntry>,
    /// Total archetypes in the world, including empty / storage-free ones.
    pub archetype_count: usize,
    /// Archetypes that reserve at least one byte of column payload.
    pub payload_archetype_count: usize,
    /// Total live-entity payload across the world.
    pub total_live_bytes: usize,
    /// Total reserved column payload across the world (live and free).
    pub total_allocated_bytes: usize,
    /// Total reserved payload behind empty slots across the world.
    pub total_wasted_bytes: usize,
}

impl MemoryFootprintReport {
    /// Capture a byte-level memory summary of `world`.
    #[inline]
    pub fn capture(world: &World) -> Self {
        Self::from_world_report(&WorldReport::capture(world), world.components())
    }

    /// Fold an existing [`WorldReport`] into a memory summary, reading row
    /// widths from `components`. Use this to avoid a second world walk when a
    /// structural snapshot is already in hand; `components` must be the same
    /// registry the report was captured from so component ids resolve.
    pub fn from_world_report(report: &WorldReport, components: &Components) -> Self {
        let mut entries = Vec::new();
        let mut payload_archetype_count = 0;
        let mut total_live_bytes = 0;
        let mut total_allocated_bytes = 0;
        let mut total_wasted_bytes = 0;

        for a in &report.archetypes {
            // Row width = sum of component element sizes. An id missing from the
            // registry contributes zero rather than aborting the report.
            let bytes_per_row: usize = a
                .components
                .iter()
                .map(|&id| components.info(id).map_or(0, |info| info.layout().size()))
                .sum();

            let entry = ArchetypeMemoryEntry {
                id: a.id,
                component_count: a.components.len(),
                bytes_per_row,
                rows_per_chunk: a.occupancy.rows_per_chunk,
                chunk_count: a.occupancy.chunk_count,
                live_rows: a.occupancy.live_rows,
            };

            // Only archetypes that reserve real payload bytes are memory-relevant.
            if entry.allocated_bytes() == 0 {
                continue;
            }
            payload_archetype_count += 1;
            total_live_bytes += entry.live_bytes();
            total_allocated_bytes += entry.allocated_bytes();
            total_wasted_bytes += entry.wasted_bytes();
            entries.push(entry);
        }

        // Biggest reclamation win first; archetype id breaks ties deterministically.
        entries.sort_by(|a, b| {
            b.wasted_bytes()
                .cmp(&a.wasted_bytes())
                .then_with(|| a.id.cmp(&b.id))
        });

        Self {
            entries,
            archetype_count: report.archetype_count,
            payload_archetype_count,
            total_live_bytes,
            total_allocated_bytes,
            total_wasted_bytes,
        }
    }

    /// Whether no archetype reserves column payload (nothing to analyze).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of payload-reserving archetypes analyzed.
    #[inline]
    pub fn analyzed_count(&self) -> usize {
        self.entries.len()
    }

    /// World-wide byte occupancy: live payload over allocated payload, in
    /// `[0.0, 1.0]`; `0.0` when nothing is allocated.
    #[inline]
    pub fn byte_occupancy(&self) -> f32 {
        if self.total_allocated_bytes == 0 {
            0.0
        } else {
            self.total_live_bytes as f32 / self.total_allocated_bytes as f32
        }
    }

    /// World-wide internal fragmentation in bytes: the unused fraction of
    /// allocated payload (`1.0 - byte_occupancy`).
    #[inline]
    pub fn internal_fragmentation(&self) -> f32 {
        1.0 - self.byte_occupancy()
    }

    /// The archetype wasting the most resident bytes, or `None` when none
    /// reserve payload. Ties resolve to the lowest archetype id.
    #[inline]
    pub fn worst(&self) -> Option<&ArchetypeMemoryEntry> {
        self.entries.first()
    }

    /// The entry for `id`, if that archetype reserves payload.
    pub fn entry(&self, id: ArchetypeId) -> Option<&ArchetypeMemoryEntry> {
        self.entries.iter().find(|e| e.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;
    use crate::world::World;

    // Distinct row widths so byte accounting is distinguishable from row counts.
    #[derive(Debug)]
    struct Wide {
        _a: u64,
        _b: u64,
        _c: u64,
        _d: u64,
    }
    impl Component for Wide {}

    #[derive(Debug)]
    struct Narrow {
        _a: u8,
    }
    impl Component for Narrow {}

    // Zero-sized marker: contributes bookkeeping but no payload bytes.
    #[derive(Debug)]
    struct Marker;
    impl Component for Marker {}

    #[test]
    fn empty_world_has_no_payload() {
        let world = World::new();
        let report = MemoryFootprintReport::capture(&world);
        assert!(report.is_empty());
        assert_eq!(report.analyzed_count(), 0);
        assert_eq!(report.total_live_bytes, 0);
        assert_eq!(report.total_allocated_bytes, 0);
        assert_eq!(report.total_wasted_bytes, 0);
        assert_eq!(report.byte_occupancy(), 0.0);
        assert!(report.worst().is_none());
    }

    #[test]
    fn bytes_track_row_width_and_count() {
        let mut world = World::new();
        for _ in 0..4u32 {
            world.spawn(Wide {
                _a: 0,
                _b: 0,
                _c: 0,
                _d: 0,
            });
        }
        let report = MemoryFootprintReport::capture(&world);
        assert_eq!(report.analyzed_count(), 1);
        let e = report.worst().unwrap();

        // Wide is 4 * u64 = 32 bytes/row, no padding between like-typed fields.
        assert_eq!(e.bytes_per_row, 32);
        assert_eq!(e.live_rows, 4);
        assert_eq!(e.live_bytes(), 4 * 32);
        assert_eq!(e.allocated_bytes(), e.capacity() * 32);
        assert_eq!(e.wasted_bytes(), e.free_slots() * 32);
        assert_eq!(
            e.allocated_bytes(),
            e.live_bytes() + e.wasted_bytes()
        );
        assert_eq!(report.total_live_bytes, 4 * 32);

        // A 32-byte row fits 512 rows in a 16 KiB target chunk, fully packed.
        assert_eq!(e.rows_per_chunk, TARGET_CHUNK_BYTES / 32);
        assert_eq!(e.chunk_payload_bytes(), TARGET_CHUNK_BYTES);
        assert_eq!(e.chunk_fill_ratio(), 1.0);
    }

    #[test]
    fn zero_sized_archetype_reserves_no_bytes() {
        let mut world = World::new();
        world.spawn(Marker);
        let report = MemoryFootprintReport::capture(&world);
        // A ZST-only archetype has zero payload, so it is excluded entirely.
        assert!(report.is_empty());
        assert_eq!(report.total_allocated_bytes, 0);
    }

    #[test]
    fn entries_rank_by_wasted_bytes_descending() {
        let mut world = World::new();
        // Two single-row archetypes of different widths. Both allocate one chunk
        // with mostly-free slots; the ranking is purely by wasted bytes, and the
        // report must stay sorted descending regardless of which width that is.
        world.spawn(Wide {
            _a: 0,
            _b: 0,
            _c: 0,
            _d: 0,
        });
        world.spawn(Narrow { _a: 0 });

        let report = MemoryFootprintReport::capture(&world);
        assert_eq!(report.analyzed_count(), 2);

        // Deterministic descending-wasted ordering (ties broken by ascending id).
        assert!(report.entries[0].wasted_bytes() >= report.entries[1].wasted_bytes());
        assert_eq!(report.worst().unwrap().id, report.entries[0].id);

        // Both widths are represented with their true element sizes, and live
        // payload tracks width even though wasted payload is slot-count driven.
        let wide = report.entries.iter().find(|e| e.bytes_per_row == 32).unwrap();
        let narrow = report.entries.iter().find(|e| e.bytes_per_row == 1).unwrap();
        assert_eq!(wide.live_bytes(), 32);
        assert_eq!(narrow.live_bytes(), 1);

        // Totals are the sum of the per-archetype figures.
        let sum_wasted: usize = report.entries.iter().map(|e| e.wasted_bytes()).sum();
        assert_eq!(report.total_wasted_bytes, sum_wasted);
        let sum_live: usize = report.entries.iter().map(|e| e.live_bytes()).sum();
        assert_eq!(report.total_live_bytes, sum_live);
        assert!(report.internal_fragmentation() > 0.0);
    }

    #[test]
    fn entry_lookup_and_from_world_report_match_capture() {
        let mut world = World::new();
        world.spawn(Wide {
            _a: 1,
            _b: 2,
            _c: 3,
            _d: 4,
        });

        let direct = MemoryFootprintReport::capture(&world);
        let folded = MemoryFootprintReport::from_world_report(
            &WorldReport::capture(&world),
            world.components(),
        );
        assert_eq!(direct.entries, folded.entries);
        assert_eq!(direct.total_allocated_bytes, folded.total_allocated_bytes);

        let id = direct.worst().unwrap().id;
        assert!(direct.entry(id).is_some());
    }
}
