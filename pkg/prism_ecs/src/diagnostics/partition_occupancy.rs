//! World-partition cell occupancy / streaming census (design §13.1 / §16.6).
//!
//! The streaming kernel keeps two independent bookkeeping structures per
//! partitioned world (design §13.1):
//!
//! * the [`CellStreamer`](crate::partition::cell::CellStreamer) — which cells
//!   are tracked and in what lifecycle [`CellState`] (loading / loaded /
//!   unloading), and
//! * the [`CellEntityIndex`](crate::partition::streaming::CellEntityIndex) —
//!   which entities currently live in which cell.
//!
//! Neither alone answers the question a streaming devtools panel actually
//! asks: *are the cells I streamed in carrying the content I expect, and is
//! any content stranded in a cell the streamer no longer owns?* This module
//! joins the two read-only, per cell, and derives the consistency signals that
//! fall out of the join:
//!
//! * **Occupancy** — per cell: its [`CellState`] and how many entities the
//!   index assigns to it, so an editor can see hot cells, and *empty loaded*
//!   cells (streamed in but carrying nothing — wasted residency).
//! * **Non-resident residents** — entities the index assigns to a tracked cell
//!   that is **not** [`Loaded`](CellState::Loaded) (still loading, or
//!   mid-eviction). Simulating entities in a cell that is not resident is a
//!   streaming-ordering bug (content placed before load settles, or not
//!   drained before unload completes).
//! * **Untracked residents** — entities the index assigns to a cell the
//!   streamer does **not** track at all (its [`CellState`] is effectively
//!   [`Unloaded`](CellState::Unloaded)). These are entities stranded by an
//!   eviction that forgot to drain them, or placed into a cell that was never
//!   loaded — a leak the streamer can no longer reach to clean up.
//!
//! Capture is read-only and deterministic: cells are enumerated from the
//! streamer and sorted by `(x, y, z)`, and the index is only queried, never
//! mutated. The two totals the index reports
//! ([`entity_count`](crate::partition::streaming::CellEntityIndex::entity_count)
//! / [`cell_count`](crate::partition::streaming::CellEntityIndex::cell_count))
//! are differenced against the per-tracked-cell sums to recover the untracked
//! residents and the entity-bearing cells that lie outside the streamer,
//! without needing to enumerate the index's own cell set.
//!
//! The optional data-layer dimension ([`from_partition`](PartitionOccupancyReport::from_partition))
//! records, per cell, whether its [`DataLayers`](crate::partition::data_layer::DataLayers)
//! currently permit streaming — a tracked-but-not-streamable cell is a layer
//! toggle the streamer has not yet reacted to.

use alloc::vec::Vec;

use crate::partition::cell::{CellCoord, CellState};
use crate::partition::driver::StreamDriver;
use crate::partition::world_partition::WorldPartition;

/// Per-cell occupancy record: a tracked cell, its streaming
/// [`CellState`], the number of entities the index assigns to it, and
/// (optionally) whether its data layers permit streaming.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CellOccupancyEntry {
    /// The cell's integer grid coordinate.
    pub coord: CellCoord,
    /// The cell's current streaming lifecycle state.
    pub state: CellState,
    /// Number of entities the [`CellEntityIndex`](crate::partition::streaming::CellEntityIndex)
    /// currently assigns to this cell.
    pub entity_count: usize,
    /// Whether this cell's data layers currently permit streaming, or `None`
    /// when the report was built from a [`StreamDriver`] alone (no data-layer
    /// context). See [`from_partition`](PartitionOccupancyReport::from_partition).
    pub streamable: Option<bool>,
}

impl CellOccupancyEntry {
    /// Whether the cell is fully resident ([`CellState::Loaded`]).
    #[inline]
    pub fn is_resident(&self) -> bool {
        self.state == CellState::Loaded
    }

    /// Whether the cell carries no entities.
    #[inline]
    pub fn is_vacant(&self) -> bool {
        self.entity_count == 0
    }

    /// Whether the cell carries at least one entity.
    #[inline]
    pub fn is_populated(&self) -> bool {
        self.entity_count != 0
    }

    /// A loaded cell that carries no entities — streamed in but holding no
    /// content, so its residency is currently paying memory for nothing.
    #[inline]
    pub fn is_empty_loaded(&self) -> bool {
        self.is_resident() && self.is_vacant()
    }

    /// A populated cell that is **not** resident (still loading, or
    /// mid-eviction) — entities being simulated in a non-resident cell, a
    /// streaming-ordering bug.
    #[inline]
    pub fn is_non_resident_resident(&self) -> bool {
        self.is_populated() && !self.is_resident()
    }

    /// A tracked cell whose data layers currently forbid streaming — a layer
    /// toggle the streamer has not yet acted on. Always `false` when no
    /// data-layer context was captured.
    #[inline]
    pub fn is_blocked_but_tracked(&self) -> bool {
        matches!(self.streamable, Some(false))
    }
}

/// A read-only snapshot of world-partition cell occupancy and the streaming
/// consistency signals derived from joining the streamer with the entity index
/// (design §13.1 / §16.6).
///
/// Build it from a [`StreamDriver`] with [`from_driver`](Self::from_driver), or
/// from a whole [`WorldPartition`] with [`from_partition`](Self::from_partition)
/// to also capture per-cell data-layer streamability.
#[derive(Clone, Debug, Default)]
pub struct PartitionOccupancyReport {
    /// Per tracked cell, sorted by `(x, y, z)`.
    cells: Vec<CellOccupancyEntry>,
    /// Total entities the index tracks across *all* cells, including cells the
    /// streamer does not track.
    index_entity_count: usize,
    /// Number of distinct cells the index holds at least one entity in,
    /// including cells the streamer does not track.
    index_cell_count: usize,
    /// Sum of `entity_count` over tracked cells only.
    tracked_resident_entities: usize,
    /// Number of tracked cells carrying at least one entity.
    tracked_populated_cells: usize,
}

impl PartitionOccupancyReport {
    /// Captures occupancy from a [`StreamDriver`]'s streamer and entity index.
    ///
    /// Entries carry no data-layer context ([`streamable`](CellOccupancyEntry::streamable)
    /// is `None`); use [`from_partition`](Self::from_partition) for that.
    pub fn from_driver(driver: &StreamDriver) -> Self {
        Self::build(driver, None)
    }

    /// Captures occupancy from a whole [`WorldPartition`], additionally
    /// recording per-cell data-layer streamability
    /// ([`DataLayers::is_cell_streamable`](crate::partition::data_layer::DataLayers::is_cell_streamable)).
    pub fn from_partition(partition: &WorldPartition) -> Self {
        Self::build(partition.driver(), Some(partition))
    }

    fn build(driver: &StreamDriver, partition: Option<&WorldPartition>) -> Self {
        let streamer = driver.streamer();
        let index = driver.index();

        let mut cells: Vec<CellOccupancyEntry> = streamer
            .iter()
            .map(|(coord, state)| CellOccupancyEntry {
                coord,
                state,
                entity_count: index.entities_in(coord).len(),
                streamable: partition.map(|p| p.data_layers().is_cell_streamable(coord)),
            })
            .collect();
        cells.sort_unstable_by(|a, b| {
            a.coord
                .x
                .cmp(&b.coord.x)
                .then(a.coord.y.cmp(&b.coord.y))
                .then(a.coord.z.cmp(&b.coord.z))
        });

        let tracked_resident_entities = cells.iter().map(|c| c.entity_count).sum();
        let tracked_populated_cells = cells.iter().filter(|c| c.is_populated()).count();

        Self {
            cells,
            index_entity_count: index.entity_count(),
            index_cell_count: index.cell_count(),
            tracked_resident_entities,
            tracked_populated_cells,
        }
    }

    /// Whether the partition tracks no cells and the index holds no entities.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty() && self.index_entity_count == 0
    }

    /// Every tracked cell's occupancy record, sorted by `(x, y, z)`.
    #[inline]
    pub fn cells(&self) -> &[CellOccupancyEntry] {
        &self.cells
    }

    /// Number of cells the streamer tracks (anything not
    /// [`CellState::Unloaded`]).
    #[inline]
    pub fn tracked_cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Number of distinct cells the entity index holds entities in, including
    /// cells the streamer does not track.
    #[inline]
    pub fn index_cell_count(&self) -> usize {
        self.index_cell_count
    }

    /// Total entities the index tracks across all cells.
    #[inline]
    pub fn total_entities(&self) -> usize {
        self.index_entity_count
    }

    /// Sum of entities assigned to tracked cells only.
    #[inline]
    pub fn tracked_resident_entities(&self) -> usize {
        self.tracked_resident_entities
    }

    /// Number of tracked cells currently in the given [`CellState`].
    pub fn state_count(&self, state: CellState) -> usize {
        self.cells.iter().filter(|c| c.state == state).count()
    }

    /// Number of tracked cells that are fully resident
    /// ([`CellState::Loaded`]).
    #[inline]
    pub fn loaded_cell_count(&self) -> usize {
        self.state_count(CellState::Loaded)
    }

    /// Number of tracked cells whose load I/O is still in flight
    /// ([`CellState::Loading`]).
    #[inline]
    pub fn loading_cell_count(&self) -> usize {
        self.state_count(CellState::Loading)
    }

    /// Number of tracked cells being evicted ([`CellState::Unloading`]).
    #[inline]
    pub fn unloading_cell_count(&self) -> usize {
        self.state_count(CellState::Unloading)
    }

    /// Number of tracked cells carrying at least one entity.
    #[inline]
    pub fn populated_cell_count(&self) -> usize {
        self.tracked_populated_cells
    }

    /// Number of tracked cells carrying no entities.
    #[inline]
    pub fn vacant_cell_count(&self) -> usize {
        self.cells.len() - self.tracked_populated_cells
    }

    /// Number of loaded cells carrying no entities — resident cells paying
    /// memory for no content (design §13.1 / §17).
    pub fn empty_loaded_cell_count(&self) -> usize {
        self.cells.iter().filter(|c| c.is_empty_loaded()).count()
    }

    /// Total entities assigned to tracked cells that are **not** resident
    /// (loading or unloading) — entities simulated in a non-resident cell, a
    /// streaming-ordering bug (design §13.1).
    pub fn non_resident_residents(&self) -> usize {
        self.cells
            .iter()
            .filter(|c| !c.is_resident())
            .map(|c| c.entity_count)
            .sum()
    }

    /// Whether any entity is assigned to a non-resident tracked cell.
    #[inline]
    pub fn has_non_resident_residents(&self) -> bool {
        self.non_resident_residents() != 0
    }

    /// Entities the index assigns to cells the streamer does **not** track —
    /// content stranded by an eviction that forgot to drain it, or placed into
    /// a never-loaded cell (design §13.1). This is the index's total minus the
    /// entities accounted for by tracked cells.
    #[inline]
    pub fn untracked_cell_residents(&self) -> usize {
        self.index_entity_count
            .saturating_sub(self.tracked_resident_entities)
    }

    /// Number of entity-bearing cells that lie outside the streamer's tracked
    /// set — the cell-level dual of [`untracked_cell_residents`](Self::untracked_cell_residents).
    #[inline]
    pub fn untracked_populated_cells(&self) -> usize {
        self.index_cell_count
            .saturating_sub(self.tracked_populated_cells)
    }

    /// Whether any entity is stranded in an untracked cell.
    #[inline]
    pub fn has_untracked_residents(&self) -> bool {
        self.untracked_cell_residents() != 0
    }

    /// Number of tracked cells whose data layers currently forbid streaming,
    /// or `None` when the report carries no data-layer context (built from a
    /// bare [`StreamDriver`]).
    pub fn blocked_cell_count(&self) -> Option<usize> {
        // Any captured data-layer context (even on one cell) means the report
        // was built from a `WorldPartition`; a report with no entries at all
        // likewise carries no context.
        if self.cells.iter().all(|c| c.streamable.is_none()) {
            return None;
        }
        Some(self.cells.iter().filter(|c| c.is_blocked_but_tracked()).count())
    }

    /// The most-populated tracked cell, ties broken by lowest `(x, y, z)`, or
    /// `None` when no cells are tracked.
    pub fn hottest_cell(&self) -> Option<&CellOccupancyEntry> {
        self.cells.iter().max_by(|a, b| {
            a.entity_count.cmp(&b.entity_count).then_with(|| {
                // Lower coordinate wins the tie, so invert the comparison.
                b.coord
                    .x
                    .cmp(&a.coord.x)
                    .then(b.coord.y.cmp(&a.coord.y))
                    .then(b.coord.z.cmp(&a.coord.z))
            })
        })
    }

    /// The largest entity count held by any single tracked cell.
    #[inline]
    pub fn max_entities_in_cell(&self) -> usize {
        self.hottest_cell().map_or(0, |c| c.entity_count)
    }

    /// The occupancy record for a specific cell, or `None` if the streamer does
    /// not track it.
    pub fn cell(&self, coord: CellCoord) -> Option<&CellOccupancyEntry> {
        self.cells.iter().find(|c| c.coord == coord)
    }

    /// Fraction of tracked cells that carry at least one entity, in per-mille
    /// (0–1000). `0` when no cells are tracked. A low value means the streamer
    /// is holding many cells resident that carry no content (design §17).
    pub fn populated_permille(&self) -> u32 {
        let total = self.cells.len();
        if total == 0 {
            return 0;
        }
        ((self.tracked_populated_cells as u64 * 1000) / total as u64) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partition::cell::CellStreamer;
    use crate::partition::data_layer::{DataLayerId, DataLayerState};
    use crate::partition::floating_origin::FloatingOrigin;
    use crate::world::World;

    fn c(x: i32, y: i32, z: i32) -> CellCoord {
        CellCoord::new(x, y, z)
    }

    /// Builds a driver with cells A(0,0,0) & B(1,0,0) resident, C(2,0,0)
    /// loading, plus entities: 2 in A, 0 in B, 1 in C (non-resident resident),
    /// and 1 in untracked cell D(9,0,0) (untracked resident).
    fn fixture() -> (StreamDriver, [CellCoord; 4]) {
        let (a, b, cc, d) = (c(0, 0, 0), c(1, 0, 0), c(2, 0, 0), c(9, 0, 0));
        let mut world = World::new();

        let mut driver = StreamDriver::new(0, 0);
        // C enters Loading via the streaming cycle (empty map first => no
        // eviction side effects), then A and B are forced resident.
        let delta = driver.streamer_mut().update(&[cc]);
        assert!(delta.to_load.contains(&cc));
        driver.streamer_mut().mark_resident(a);
        driver.streamer_mut().mark_resident(b);

        let ea1 = world.spawn(());
        let ea2 = world.spawn(());
        let ec1 = world.spawn(());
        let ed1 = world.spawn(());
        driver.register(ea1, a);
        driver.register(ea2, a);
        driver.register(ec1, cc);
        driver.register(ed1, d);

        (driver, [a, b, cc, d])
    }

    #[test]
    fn empty_report_is_empty() {
        let driver = StreamDriver::new(0, 0);
        let report = PartitionOccupancyReport::from_driver(&driver);
        assert!(report.is_empty());
        assert_eq!(report.tracked_cell_count(), 0);
        assert_eq!(report.total_entities(), 0);
        assert_eq!(report.hottest_cell(), None);
        assert_eq!(report.populated_permille(), 0);
        assert_eq!(report.blocked_cell_count(), None);
    }

    #[test]
    fn per_cell_states_and_counts() {
        let (driver, [a, b, cc, _d]) = fixture();
        let report = PartitionOccupancyReport::from_driver(&driver);

        assert!(!report.is_empty());
        // Three tracked cells, deterministically sorted by (x, y, z).
        assert_eq!(report.tracked_cell_count(), 3);
        let coords: Vec<CellCoord> = report.cells().iter().map(|e| e.coord).collect();
        assert_eq!(coords, [a, b, cc]);

        assert_eq!(report.loaded_cell_count(), 2); // A, B
        assert_eq!(report.loading_cell_count(), 1); // C
        assert_eq!(report.unloading_cell_count(), 0);

        assert_eq!(report.cell(a).unwrap().entity_count, 2);
        assert_eq!(report.cell(b).unwrap().entity_count, 0);
        assert_eq!(report.cell(cc).unwrap().entity_count, 1);
        // Driver-only capture carries no data-layer context.
        assert_eq!(report.cell(a).unwrap().streamable, None);
    }

    #[test]
    fn occupancy_and_density() {
        let (driver, [a, b, _cc, _d]) = fixture();
        let report = PartitionOccupancyReport::from_driver(&driver);

        assert_eq!(report.populated_cell_count(), 2); // A, C
        assert_eq!(report.vacant_cell_count(), 1); // B
        assert_eq!(report.empty_loaded_cell_count(), 1); // B is loaded + vacant
        assert_eq!(report.max_entities_in_cell(), 2);
        assert_eq!(report.hottest_cell().unwrap().coord, a);
        // 2 populated of 3 tracked => 666 per-mille.
        assert_eq!(report.populated_permille(), 666);
        let _ = b;
    }

    #[test]
    fn consistency_gaps_surface() {
        let (driver, _) = fixture();
        let report = PartitionOccupancyReport::from_driver(&driver);

        // The entity in loading cell C is a resident of a non-resident cell.
        assert_eq!(report.non_resident_residents(), 1);
        assert!(report.has_non_resident_residents());

        // Totals: index holds 4 entities across A, C, D; tracked cells account
        // for A(2) + B(0) + C(1) = 3, leaving D's single entity untracked.
        assert_eq!(report.total_entities(), 4);
        assert_eq!(report.tracked_resident_entities(), 3);
        assert_eq!(report.untracked_cell_residents(), 1);
        assert!(report.has_untracked_residents());

        // Index touches 3 cells (A, C, D); 2 of them (A, C) are tracked and
        // populated, so exactly one entity-bearing cell lies outside tracking.
        assert_eq!(report.index_cell_count(), 3);
        assert_eq!(report.untracked_populated_cells(), 1);
    }

    #[test]
    fn hottest_cell_breaks_ties_by_lowest_coord() {
        let mut world = World::new();
        let mut driver = StreamDriver::new(0, 0);
        let (lo, hi) = (c(1, 0, 0), c(5, 0, 0));
        driver.streamer_mut().mark_resident(lo);
        driver.streamer_mut().mark_resident(hi);
        let e0 = world.spawn(());
        let e1 = world.spawn(());
        driver.register(e0, hi);
        driver.register(e1, lo);

        let report = PartitionOccupancyReport::from_driver(&driver);
        // Both cells hold one entity; the lower coordinate wins the tie.
        assert_eq!(report.max_entities_in_cell(), 1);
        assert_eq!(report.hottest_cell().unwrap().coord, lo);
    }

    #[test]
    fn data_layer_context_flags_blocked_cells() {
        let mut world = World::new();
        let mut partition = WorldPartition::from_parts(
            FloatingOrigin::new(100.0),
            CellStreamer::new(0, 0),
        );
        let (open, blocked) = (c(0, 0, 0), c(3, 0, 0));
        partition.driver_mut().streamer_mut().mark_resident(open);
        partition.driver_mut().streamer_mut().mark_resident(blocked);

        // A data layer gating `blocked`, left unloaded, forbids its streaming.
        let layer = DataLayerId(7);
        partition.data_layers_mut().register_layer(layer);
        partition.data_layers_mut().assign(blocked, layer);
        partition
            .data_layers_mut()
            .set_state(layer, DataLayerState::Unloaded);

        let e = world.spawn(());
        partition.driver_mut().register(e, open);

        let report = PartitionOccupancyReport::from_partition(&partition);
        assert_eq!(report.blocked_cell_count(), Some(1));
        assert_eq!(report.cell(open).unwrap().streamable, Some(true));
        assert_eq!(report.cell(blocked).unwrap().streamable, Some(false));
        assert!(report.cell(blocked).unwrap().is_blocked_but_tracked());
    }
}
