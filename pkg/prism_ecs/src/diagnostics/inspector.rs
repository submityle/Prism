//! ECS inspector: archetype / chunk occupancy snapshots (design §16.6).
//!
//! This module provides read-only, allocation-light structural introspection
//! over the world's Table-backed storage. It is the data source behind an
//! external editor inspector / devtools panel (design §16.6 对接
//! `prism_ui_devtools` / `prism_ui_inspector`).
//!
//! # Scope
//! The reports here describe the chunked columnar [`Table`](crate::storage::table::Table)
//! storage that every archetype owns. Components stored in `SparseSet` form are
//! *not* laid out in archetype chunks, so chunk occupancy numbers describe the
//! Table-backed portion only. This matches the design headline that the hot,
//! cache-friendly path is the chunked Table (design §6).
//!
//! Capturing a report is `O(archetypes)` and performs no interior mutation of
//! the world; it is safe to call from a diagnostics system or an exclusive
//! inspection pass.

use alloc::vec::Vec;

use crate::archetype::ArchetypeId;
use crate::change::Tick;
use crate::component::ComponentId;
use crate::world::World;

/// Chunk occupancy for a single archetype's Table-backed storage.
///
/// All figures are expressed in rows (one row == one entity slot). `capacity`
/// is derived from the chunk geometry, so `live_rows <= capacity` always holds
/// for a well-formed table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OccupancyStats {
    /// Number of allocated 16 KiB chunks backing this archetype's table.
    pub chunk_count: usize,
    /// Row capacity of a single chunk (`N` in design §5.3).
    pub rows_per_chunk: usize,
    /// Number of live entity rows currently stored.
    pub live_rows: usize,
}

impl OccupancyStats {
    /// Total addressable rows across every allocated chunk
    /// (`chunk_count * rows_per_chunk`).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.chunk_count * self.rows_per_chunk
    }

    /// Rows that are allocated but unoccupied (`capacity - live_rows`).
    #[inline]
    pub fn free_slots(&self) -> usize {
        self.capacity().saturating_sub(self.live_rows)
    }

    /// Fraction of allocated row capacity that is live, in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when no capacity is allocated so callers never divide by
    /// zero.
    #[inline]
    pub fn occupancy(&self) -> f32 {
        let capacity = self.capacity();
        if capacity == 0 {
            0.0
        } else {
            self.live_rows as f32 / capacity as f32
        }
    }
}

/// Per stable-ID slot snapshot: live archetype identity (or an explicitly
/// marked evicted tombstone) and chunk occupancy.
#[derive(Debug, Clone)]
pub struct ArchetypeReport {
    /// The archetype's stable id.
    pub id: ArchetypeId,
    /// Whether this stable-ID slot is a permanently evicted tombstone.
    /// Tombstones remain in reports for ID continuity, but their released
    /// component identity is intentionally empty.
    pub evicted: bool,
    /// Component ids that make up a live archetype's identity. Empty for an
    /// evicted tombstone because its identity metadata has been reclaimed.
    pub components: Vec<ComponentId>,
    /// Chunk occupancy for the archetype's Table storage.
    pub occupancy: OccupancyStats,
}

/// A whole-world structural snapshot: counts plus every archetype report.
///
/// Produced by [`WorldReport::capture`]. The archetype list is in the world's
/// native archetype order (stable within a run), making diffs between two
/// captures meaningful.
#[derive(Debug, Clone)]
pub struct WorldReport {
    /// Number of live entities in the world.
    pub entity_count: u32,
    /// Number of addressable archetype slots, including the empty archetype
    /// and permanently evicted tombstones.
    pub archetype_count: usize,
    /// Number of live archetypes, including the live empty archetype but
    /// excluding permanently evicted tombstones.
    pub live_archetype_count: usize,
    /// Number of registered component types.
    pub component_count: usize,
    /// The world's current change tick at capture time.
    pub change_tick: Tick,
    /// One report per stable-ID slot, in native archetype order; tombstones
    /// are retained and marked by [`ArchetypeReport::evicted`].
    pub archetypes: Vec<ArchetypeReport>,
}

impl WorldReport {
    /// Capture a structural snapshot of `world`.
    ///
    /// Read-only: walks the archetype table once and records chunk geometry,
    /// live-row counts, and tombstone state. Evicted slots remain in the list
    /// for stable-ID continuity and have empty component identity. Does not
    /// touch component values.
    pub fn capture(world: &World) -> Self {
        let archetypes = world.archetypes();
        let mut reports = Vec::with_capacity(archetypes.len());
        let mut live_archetype_count = 0;
        for archetype in archetypes.iter() {
            live_archetype_count += usize::from(!archetype.is_evicted());
            let table = archetype.table();
            let occupancy = OccupancyStats {
                chunk_count: table.chunk_count(),
                rows_per_chunk: table.rows_per_chunk(),
                live_rows: table.len(),
            };
            reports.push(ArchetypeReport {
                id: archetype.id(),
                evicted: archetype.is_evicted(),
                components: archetype.components().ids().to_vec(),
                occupancy,
            });
        }
        Self {
            entity_count: world.entity_count(),
            archetype_count: archetypes.len(),
            live_archetype_count,
            component_count: world.components().len(),
            change_tick: world.change_tick(),
            archetypes: reports,
        }
    }

    /// Total addressable rows across every archetype's allocated chunks.
    pub fn total_capacity(&self) -> usize {
        self.archetypes.iter().map(|a| a.occupancy.capacity()).sum()
    }

    /// Total live entity rows across every archetype's Table storage.
    pub fn total_live_rows(&self) -> usize {
        self.archetypes.iter().map(|a| a.occupancy.live_rows).sum()
    }

    /// Total allocated-but-unoccupied rows across every archetype.
    pub fn total_free_slots(&self) -> usize {
        self.archetypes
            .iter()
            .map(|a| a.occupancy.free_slots())
            .sum()
    }

    /// Total allocated chunks across every archetype.
    pub fn total_chunks(&self) -> usize {
        self.archetypes
            .iter()
            .map(|a| a.occupancy.chunk_count)
            .sum()
    }

    /// Fraction of allocated row capacity that is live across the whole world,
    /// in `[0.0, 1.0]`. Returns `0.0` when nothing is allocated.
    pub fn occupancy(&self) -> f32 {
        let capacity = self.total_capacity();
        if capacity == 0 {
            0.0
        } else {
            self.total_live_rows() as f32 / capacity as f32
        }
    }

    /// Number of archetypes that currently hold at least one live entity.
    pub fn non_empty_archetypes(&self) -> usize {
        self.archetypes
            .iter()
            .filter(|a| !a.evicted && a.occupancy.live_rows > 0)
            .count()
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
    #[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
    struct Batch(u32);
    impl Component for Batch {
        const STORAGE: crate::component::StorageType = crate::component::StorageType::Shared;
        fn install_storage_glue(components: &mut crate::component::Components, id: ComponentId) {
            components.set_shared_box(id, crate::component::shared_box_of::<Self>());
        }
    }

    #[test]
    fn empty_world_report() {
        let world = World::new();
        let report = WorldReport::capture(&world);
        assert_eq!(report.entity_count, 0);
        // The empty archetype always exists.
        assert!(report.archetype_count >= 1);
        assert_eq!(report.total_live_rows(), 0);
        assert_eq!(report.occupancy(), 0.0);
        assert_eq!(report.non_empty_archetypes(), 0);
    }

    #[test]
    fn occupancy_tracks_spawns() {
        let mut world = World::new();
        for i in 0..10u32 {
            world.spawn((Position(i as f32, 0.0), Velocity(0.0, 0.0)));
        }
        let report = WorldReport::capture(&world);
        assert_eq!(report.entity_count, 10);
        assert_eq!(report.total_live_rows(), 10);

        // Exactly one archetype should hold the 10 (Position, Velocity) rows.
        let populated: Vec<&ArchetypeReport> = report
            .archetypes
            .iter()
            .filter(|a| a.occupancy.live_rows > 0)
            .collect();
        assert_eq!(populated.len(), 1);
        let arch = populated[0];
        assert_eq!(arch.occupancy.live_rows, 10);
        assert_eq!(arch.components.len(), 2);
        assert!(arch.occupancy.chunk_count >= 1);
        assert!(arch.occupancy.rows_per_chunk >= 10);
        assert!(arch.occupancy.capacity() >= arch.occupancy.live_rows);
        assert_eq!(
            arch.occupancy.free_slots(),
            arch.occupancy.capacity() - arch.occupancy.live_rows
        );
        assert!(arch.occupancy.occupancy() > 0.0 && arch.occupancy.occupancy() <= 1.0);
        assert_eq!(report.non_empty_archetypes(), 1);
    }

    #[test]
    fn distinct_archetypes_are_separated() {
        let mut world = World::new();
        world.spawn(Position(1.0, 1.0));
        world.spawn((Position(2.0, 2.0), Velocity(1.0, 1.0)));
        let report = WorldReport::capture(&world);
        assert_eq!(report.entity_count, 2);
        assert_eq!(report.total_live_rows(), 2);
        assert_eq!(report.non_empty_archetypes(), 2);
    }

    #[test]
    fn report_marks_evicted_slots_and_separates_live_archetype_count() {
        let mut world = World::new();
        let entity = world.spawn((Position(1.0, 1.0), Batch(17)));
        let evicted_id = world.entities().location(entity).unwrap().archetype_id;
        assert!(world.despawn(entity));

        let report = WorldReport::capture(&world);
        assert_eq!(report.archetype_count, report.archetypes.len());
        assert_eq!(report.live_archetype_count + 1, report.archetype_count);
        let tombstone = report
            .archetypes
            .iter()
            .find(|item| item.id == evicted_id)
            .unwrap();
        assert!(tombstone.evicted);
        assert!(tombstone.components.is_empty());
        assert_eq!(tombstone.occupancy.live_rows, 0);
    }
}
