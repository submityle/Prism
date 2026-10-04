//! Change-volume / dirty-chunk accounting (design §10 / §16.6).
//!
//! The design headline is "成本 ∝ 变化量" (cost proportional to the amount of
//! change, not the total entity count). This module quantifies that change
//! volume between a reference tick and the world's current change tick, so a
//! devtools panel or a performance regression gate can observe how much work a
//! frame actually dirtied.
//!
//! Two granularities are reported, mirroring the two-layer change detection in
//! design §10:
//! * **dirty chunks** — the coarse layer: a chunk is dirty when any of its
//!   columns bumped its `chunk_version` past the reference tick. This is the
//!   quantity that `Changed<T>` iteration can skip wholesale.
//! * **changed / added cells** — the fine layer: per-component ticks, counting
//!   individual rows whose value was changed or whose component was added in
//!   the `(reference, this_run]` window.
//!
//! # Scope
//! Only Table-backed storage carries chunk versioning, so these figures cover
//! the chunked columnar path (design §6). The accounting is read-only.

use alloc::vec::Vec;

use crate::archetype::ArchetypeId;
use crate::change::Tick;
use crate::world::World;

/// Change volume for a single archetype's Table storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchetypeChangeReport {
    /// The archetype's stable id.
    pub id: ArchetypeId,
    /// Chunks with at least one column newer than the reference tick.
    pub dirty_chunks: usize,
    /// Total allocated chunks in the archetype's table.
    pub total_chunks: usize,
    /// Cells (component values) whose change tick falls in the window.
    pub changed_cells: usize,
    /// Cells whose add tick falls in the window (newly inserted components).
    pub added_cells: usize,
}

/// Whole-world change volume between a reference tick and the current tick.
///
/// Produced by [`ChangeReport::since`]. The window is `(reference, this_run]`
/// where `this_run` is the world's change tick at capture time. The top-level
/// counters are the sums of the per-archetype entries.
#[derive(Debug, Clone)]
pub struct ChangeReport {
    /// The lower bound of the observed window (exclusive).
    pub reference: Tick,
    /// The world's change tick at capture time (upper bound, inclusive).
    pub this_run: Tick,
    /// Total dirty chunks across all archetypes.
    pub dirty_chunks: usize,
    /// Total allocated chunks across all archetypes.
    pub total_chunks: usize,
    /// Total changed cells across all archetypes.
    pub changed_cells: usize,
    /// Total added cells across all archetypes.
    pub added_cells: usize,
    /// One entry per archetype, in native archetype order.
    pub per_archetype: Vec<ArchetypeChangeReport>,
}

impl ChangeReport {
    /// Measure the change volume accumulated since `reference`.
    ///
    /// A chunk counts as dirty when *any* of its columns reports a
    /// `chunk_version` newer than `reference` relative to the current tick. A
    /// cell counts as changed/added when its per-row changed/added tick is
    /// newer than `reference`. Read-only.
    pub fn since(world: &World, reference: Tick) -> Self {
        let this_run = world.change_tick();
        let archetypes = world.archetypes();
        let mut per_archetype = Vec::with_capacity(archetypes.len());

        let mut total_dirty_chunks = 0;
        let mut total_chunks = 0;
        let mut total_changed = 0;
        let mut total_added = 0;

        for archetype in archetypes.iter() {
            let table = archetype.table();
            let chunk_count = table.chunk_count();
            let component_ids = archetype.components().ids();

            // Coarse layer: a chunk is dirty if any column version is newer.
            let mut dirty_chunks = 0;
            for chunk in 0..chunk_count {
                let mut chunk_dirty = false;
                for &id in component_ids {
                    if let Some(column) = table.column(id)
                        && chunk < column.chunk_count()
                        && column
                            .chunk_version(chunk)
                            .is_newer_than(reference, this_run)
                    {
                        chunk_dirty = true;
                        break;
                    }
                }
                if chunk_dirty {
                    dirty_chunks += 1;
                }
            }

            // Fine layer: per-row changed / added ticks across every column.
            let mut changed_cells = 0;
            let mut added_cells = 0;
            for &id in component_ids {
                if let Some(column) = table.column(id) {
                    for row in 0..column.len() {
                        if column.changed_tick(row).is_newer_than(reference, this_run) {
                            changed_cells += 1;
                        }
                        if column.added_tick(row).is_newer_than(reference, this_run) {
                            added_cells += 1;
                        }
                    }
                }
            }

            total_dirty_chunks += dirty_chunks;
            total_chunks += chunk_count;
            total_changed += changed_cells;
            total_added += added_cells;

            per_archetype.push(ArchetypeChangeReport {
                id: archetype.id(),
                dirty_chunks,
                total_chunks: chunk_count,
                changed_cells,
                added_cells,
            });
        }

        Self {
            reference,
            this_run,
            dirty_chunks: total_dirty_chunks,
            total_chunks,
            changed_cells: total_changed,
            added_cells: total_added,
            per_archetype,
        }
    }

    /// Fraction of chunks that are dirty, in `[0.0, 1.0]`. Returns `0.0` when no
    /// chunks are allocated.
    pub fn dirty_chunk_ratio(&self) -> f32 {
        if self.total_chunks == 0 {
            0.0
        } else {
            self.dirty_chunks as f32 / self.total_chunks as f32
        }
    }

    /// Whether any change (chunk, cell, or add) was observed in the window.
    pub fn is_quiescent(&self) -> bool {
        self.dirty_chunks == 0 && self.changed_cells == 0 && self.added_cells == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;
    use crate::world::World;

    #[derive(Debug, PartialEq)]
    struct Position(f32, f32);
    impl Component for Position {}
    #[derive(Debug, PartialEq)]
    struct Velocity(f32, f32);
    impl Component for Velocity {}

    #[test]
    fn fresh_spawns_count_as_added() {
        let mut world = World::new();
        let reference = world.change_tick();
        // Writes made after `reference` land in the observed window.
        world.increment_change_tick();
        for i in 0..5u32 {
            world.spawn(Position(i as f32, 0.0));
        }
        let report = ChangeReport::since(&world, reference);
        // Five new Position cells added.
        assert_eq!(report.added_cells, 5);
        // Added also stamps the changed tick for a brand-new value.
        assert!(report.changed_cells >= 5);
        assert!(report.dirty_chunks >= 1);
        assert!(!report.is_quiescent());
    }

    #[test]
    fn no_changes_after_reference_is_quiescent() {
        let mut world = World::new();
        for i in 0..5u32 {
            world.spawn(Position(i as f32, 0.0));
        }
        // Advance past the spawn tick, then observe from the newer reference.
        world.increment_change_tick();
        let reference = world.change_tick();
        let report = ChangeReport::since(&world, reference);
        assert_eq!(report.changed_cells, 0);
        assert_eq!(report.added_cells, 0);
        assert_eq!(report.dirty_chunks, 0);
        assert!(report.is_quiescent());
        assert_eq!(report.dirty_chunk_ratio(), 0.0);
    }

    #[test]
    fn mutation_counts_as_changed_not_added() {
        let mut world = World::new();
        let entities: Vec<_> = (0..5u32)
            .map(|i| world.spawn(Position(i as f32, 0.0)))
            .collect();
        // Reference sits after the spawn tick but before the mutation tick.
        world.increment_change_tick();
        let reference = world.change_tick();
        world.increment_change_tick();
        // Mutate three of the five positions.
        for &e in entities.iter().take(3) {
            world.get_mut::<Position>(e).unwrap().0 += 1.0;
        }
        let report = ChangeReport::since(&world, reference);
        assert_eq!(report.changed_cells, 3);
        assert_eq!(report.added_cells, 0);
        assert!(report.dirty_chunks >= 1);
    }

    #[test]
    fn per_archetype_sums_match_totals() {
        let mut world = World::new();
        let reference = world.change_tick();
        world.increment_change_tick();
        world.spawn(Position(0.0, 0.0));
        world.spawn((Position(1.0, 1.0), Velocity(1.0, 1.0)));
        let report = ChangeReport::since(&world, reference);
        let sum_changed: usize = report.per_archetype.iter().map(|a| a.changed_cells).sum();
        let sum_added: usize = report.per_archetype.iter().map(|a| a.added_cells).sum();
        let sum_dirty: usize = report.per_archetype.iter().map(|a| a.dirty_chunks).sum();
        assert_eq!(sum_changed, report.changed_cells);
        assert_eq!(sum_added, report.added_cells);
        assert_eq!(sum_dirty, report.dirty_chunks);
        // Position(1) + Position+Velocity(2) = 3 added cells.
        assert_eq!(report.added_cells, 3);
    }
}
