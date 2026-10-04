//! End-to-end cell streaming driver (design §13.1; §14; §22 risk 4).
//!
//! [`CellStreamer`](crate::partition::cell::CellStreamer) decides *which cells*
//! should load or evict, and [`CellEntityIndex`](crate::partition::streaming::CellEntityIndex)
//! tracks *which entities* live in each cell — but neither touches a [`World`]
//! or closes the loop between the two. [`StreamDriver`] is that closure: it owns
//! both pieces and turns one streaming tick into real world mutation.
//!
//! A single [`stream`](StreamDriver::stream) call:
//!
//! 1. runs the streamer against the current interest set, producing a
//!    [`StreamingDelta`](crate::partition::cell::StreamingDelta);
//! 2. for every cell in `to_unload`, drains exactly that cell's entities from
//!    the index — in a deterministic `(x, y, z)`-then-handle order (design §14)
//!    — despawns each from the authoritative [`World`], and settles the cell to
//!    `Unloaded` via [`CellStreamer::mark_unloaded`];
//! 3. returns the still-pending `to_load` requests plus the exact set of
//!    despawned entities, so the owner can satisfy loads and
//!    [`prune`](crate::partition::streaming::WeakRefs::prune) any cross-cell
//!    [`WeakRefs`](crate::partition::streaming::WeakRefs) that pointed at the
//!    evicted entities.
//!
//! What the driver deliberately does *not* do is disk I/O. Reading a cell's
//! payload off disk and spawning its entities belongs to the scene crate
//! (`prism_scene`), not the ECS kernel. The driver surfaces `to_load` as a
//! request and offers [`settle_loaded`](StreamDriver::settle_loaded) as the
//! callback the owner invokes once those entities exist. This keeps the kernel
//! focused on the unload closure and reference safety, with no fake I/O.

use alloc::vec::Vec;

use crate::entity::Entity;
use crate::partition::cell::{CellCoord, CellState, CellStreamer, StreamingDelta};
use crate::partition::streaming::{CellEntityIndex, WeakRefs};
use crate::world::World;

/// The outcome of one [`StreamDriver::stream`] tick (design §13.1).
///
/// The unload half is already applied to the [`World`] by the time this is
/// returned; `despawned` reports exactly which entities were removed, in the
/// deterministic order they were evicted. The load half is *not* applied — the
/// kernel does no disk I/O — so `to_load` is handed back for the owner to
/// satisfy and then confirm via [`StreamDriver::settle_loaded`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StreamResult {
    /// Cells the streamer wants resident but that are not loaded yet. The owner
    /// reads their payload, spawns their entities, and calls
    /// [`StreamDriver::settle_loaded`] for each. Sorted by `(x, y, z)`.
    pub to_load: Vec<CellCoord>,
    /// Cells that were evicted this tick (settled to `Unloaded`). Sorted by
    /// `(x, y, z)`.
    pub unloaded_cells: Vec<CellCoord>,
    /// Entities despawned this tick because their cell was evicted, in the
    /// deterministic eviction order (design §14): grouped by cell in
    /// `(x, y, z)` order, by handle within each cell.
    pub despawned: Vec<Entity>,
    /// Entities that were tracked in an evicted cell but were already absent
    /// from the [`World`] (e.g. despawned out of band). Reported for honesty so
    /// callers can detect bookkeeping drift; normally empty.
    pub already_gone: Vec<Entity>,
}

impl StreamResult {
    /// Whether this tick produced no load requests and evicted nothing.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.to_load.is_empty() && self.unloaded_cells.is_empty() && self.despawned.is_empty()
    }

    /// Number of entities actually despawned from the world this tick.
    #[inline]
    pub fn despawned_count(&self) -> usize {
        self.despawned.len()
    }
}

/// Drives world-partition streaming end to end (design §13.1).
///
/// Owns a [`CellStreamer`] (the schedule) and a [`CellEntityIndex`] (the
/// per-cell entity bookkeeping), and applies the streamer's eviction decisions
/// to a [`World`]. Keeping the two sub-objects behind one driver gives the
/// owner a single `stream` entry point while preserving the clean split
/// between `World`-independent accounting and `World` mutation.
#[derive(Clone, Debug)]
pub struct StreamDriver {
    streamer: CellStreamer,
    index: CellEntityIndex,
}

impl StreamDriver {
    /// Builds a driver with a fresh streamer of the given radii and an empty
    /// entity index. See [`CellStreamer::new`] for the radius contract
    /// (`unload_radius` must be `>= load_radius`).
    #[inline]
    pub fn new(load_radius: u32, unload_radius: u32) -> Self {
        Self {
            streamer: CellStreamer::new(load_radius, unload_radius),
            index: CellEntityIndex::new(),
        }
    }

    /// Builds a driver around an already-configured [`CellStreamer`], starting
    /// with an empty entity index.
    #[inline]
    pub fn with_streamer(streamer: CellStreamer) -> Self {
        Self {
            streamer,
            index: CellEntityIndex::new(),
        }
    }

    /// Shared access to the underlying streamer (cell states, radii).
    #[inline]
    pub fn streamer(&self) -> &CellStreamer {
        &self.streamer
    }

    /// Mutable access to the underlying streamer, for owners that drive cell
    /// state transitions directly.
    #[inline]
    pub fn streamer_mut(&mut self) -> &mut CellStreamer {
        &mut self.streamer
    }

    /// Shared access to the underlying cell↔entity index.
    #[inline]
    pub fn index(&self) -> &CellEntityIndex {
        &self.index
    }

    /// Mutable access to the underlying cell↔entity index.
    #[inline]
    pub fn index_mut(&mut self) -> &mut CellEntityIndex {
        &mut self.index
    }

    /// Registers a freshly spawned `entity` as resident in `cell`.
    ///
    /// This is the index-side bookkeeping half of a spawn; the caller is
    /// responsible for having spawned the entity in the [`World`] already. A
    /// thin wrapper over [`CellEntityIndex::assign`] so callers never reach past
    /// the driver into the index for the common case. Returns the entity's
    /// previous cell if it was already tracked.
    #[inline]
    pub fn register(&mut self, entity: Entity, cell: CellCoord) -> Option<CellCoord> {
        self.index.assign(entity, cell)
    }

    /// Unregisters `entity` from the index (e.g. a non-streaming despawn),
    /// returning the cell it was in. Does **not** touch the [`World`].
    #[inline]
    pub fn unregister(&mut self, entity: Entity) -> Option<CellCoord> {
        self.index.remove(entity)
    }

    /// Confirms that a `to_load` cell's payload has been spawned by the owner.
    ///
    /// Registers each of `entities` as resident in `cell` and promotes the cell
    /// to [`CellState::Loaded`]. This is the settle callback for the load half
    /// the kernel intentionally does not perform itself.
    pub fn settle_loaded(&mut self, cell: CellCoord, entities: &[Entity]) {
        for &entity in entities {
            self.index.assign(entity, cell);
        }
        self.streamer.mark_resident(cell);
    }

    /// Runs one streaming tick against `interests` and applies all evictions to
    /// `world` (design §13.1).
    ///
    /// Loads are *not* applied — the returned [`StreamResult::to_load`] lists
    /// the cells the owner must still read and spawn, confirming each with
    /// [`settle_loaded`](Self::settle_loaded). Every cell in the streamer's
    /// `to_unload` is drained from the index, its entities despawned from
    /// `world` in deterministic order, and the cell settled to `Unloaded`.
    pub fn stream(&mut self, world: &mut World, interests: &[CellCoord]) -> StreamResult {
        let StreamingDelta { to_load, to_unload } = self.streamer.update(interests);

        // Drain every evicted cell's entities in one deterministic pass
        // (grouped by cell order, by handle within a cell — design §14), then
        // despawn each from the authoritative world.
        let drained = self.index.take_cells(&to_unload);
        let mut despawned = Vec::with_capacity(drained.len());
        let mut already_gone = Vec::new();
        for entity in drained {
            if world.despawn(entity) {
                despawned.push(entity);
            } else {
                already_gone.push(entity);
            }
        }

        // Settle every evicted cell to `Unloaded`. `mark_unloaded` is a no-op
        // for any cell not currently `Unloading`, so this is safe even if a
        // cell held no entities.
        for &cell in &to_unload {
            self.streamer.mark_unloaded(cell);
        }

        StreamResult {
            to_load,
            unloaded_cells: to_unload,
            despawned,
            already_gone,
        }
    }

    /// Force-evicts a single `cell` immediately, despawning its entities and
    /// settling it to `Unloaded`, independent of the interest-driven schedule.
    ///
    /// Useful for teardown and editor operations. The cell is first moved to
    /// [`CellState::Unloading`] if it was resident so [`CellStreamer::mark_unloaded`]
    /// can settle it. Returns the despawned entities in deterministic handle
    /// order.
    pub fn evict_cell(&mut self, world: &mut World, cell: CellCoord) -> Vec<Entity> {
        let drained = self.index.take_cell(cell);
        let mut despawned = Vec::with_capacity(drained.len());
        for entity in drained {
            if world.despawn(entity) {
                despawned.push(entity);
            }
        }
        // Drop the cell from the streamer regardless of its prior state so a
        // forced eviction always lands on `Unloaded`.
        if self.streamer.state(cell) != CellState::Unloaded {
            self.streamer.force_unload(cell);
        }
        despawned
    }

    /// Prunes `refs` against the current `world`, dropping every weak reference
    /// whose target has been despawned (e.g. by a prior [`stream`](Self::stream)
    /// eviction). Returns the number of references removed.
    ///
    /// A convenience wrapper over [`WeakRefs::prune`] so cross-cell reference
    /// hygiene reads naturally alongside the stream loop that invalidated them.
    #[inline]
    pub fn prune_refs(&self, world: &World, refs: &mut WeakRefs) -> usize {
        refs.prune(world)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partition::streaming::{WeakEntity, WeakRefs};

    fn cell(x: i32, y: i32, z: i32) -> CellCoord {
        CellCoord::new(x, y, z)
    }

    /// Spawns `n` entities, registers them all into `cell`, and marks the cell
    /// loaded so the streamer tracks it as resident.
    fn populate(driver: &mut StreamDriver, world: &mut World, c: CellCoord, n: usize) -> Vec<Entity> {
        let mut out = Vec::new();
        for _ in 0..n {
            out.push(world.spawn(()));
        }
        driver.settle_loaded(c, &out);
        out
    }

    #[test]
    fn register_and_settle_track_residency() {
        let mut world = World::new();
        let mut driver = StreamDriver::new(1, 2);

        let e = world.spawn(());
        assert_eq!(driver.register(e, cell(0, 0, 0)), None);
        assert_eq!(driver.index().cell_of(e), Some(cell(0, 0, 0)));

        let f = world.spawn(());
        driver.settle_loaded(cell(1, 0, 0), &[f]);
        assert_eq!(driver.index().cell_of(f), Some(cell(1, 0, 0)));
        assert!(driver.streamer().is_loaded(cell(1, 0, 0)));
        assert_eq!(driver.index().entity_count(), 2);
    }

    #[test]
    fn stream_requests_loads_without_spawning() {
        let mut world = World::new();
        let mut driver = StreamDriver::new(1, 2);

        let result = driver.stream(&mut world, &[CellCoord::ORIGIN]);
        // A radius-1 Chebyshev ball = 27 cells, all requested for load.
        assert_eq!(result.to_load.len(), 27);
        assert!(result.unloaded_cells.is_empty());
        assert!(result.despawned.is_empty());
        // The kernel performed no spawns: load is the owner's job.
        assert_eq!(driver.index().entity_count(), 0);
        // Every requested cell is mid-load until settled.
        for &c in &result.to_load {
            assert_eq!(driver.streamer().state(c), CellState::Loading);
        }
    }

    #[test]
    fn stream_evicts_cell_and_despawns_its_entities() {
        let mut world = World::new();
        let mut driver = StreamDriver::new(1, 1);

        // A far-away cell, resident with three entities.
        let far = cell(100, 0, 0);
        let ents = populate(&mut driver, &mut world, far, 3);
        for &e in &ents {
            assert!(world.contains(e));
        }

        // Streaming around the origin leaves `far` outside the unload radius, so
        // it is evicted and its entities despawned.
        let result = driver.stream(&mut world, &[CellCoord::ORIGIN]);
        assert!(result.unloaded_cells.contains(&far));
        // Deterministic order: sorted by handle within the single cell.
        let mut expected = ents.clone();
        expected.sort();
        assert_eq!(result.despawned, expected);
        assert!(result.already_gone.is_empty());

        // The entities are really gone from the world and from the index.
        for &e in &ents {
            assert!(!world.contains(e));
        }
        assert!(driver.index().entities_in(far).is_empty());
        assert_eq!(driver.streamer().state(far), CellState::Unloaded);
    }

    #[test]
    fn end_to_end_eviction_prunes_cross_cell_weak_refs() {
        let mut world = World::new();
        let mut driver = StreamDriver::new(1, 1);

        // A survivor near the origin keeps weak references into a far cell.
        let near = cell(0, 0, 0);
        let survivor = world.spawn(());
        driver.settle_loaded(near, &[survivor]);

        let far = cell(50, 50, 50);
        let far_ents = populate(&mut driver, &mut world, far, 2);

        let mut refs = WeakRefs::new();
        refs.push(WeakEntity::new(survivor));
        for &e in &far_ents {
            refs.push(WeakEntity::new(e));
        }
        assert_eq!(refs.len(), 3);

        // Stream around the origin: `far` is evicted, its entities despawned.
        let result = driver.stream(&mut world, &[near]);
        assert_eq!(result.despawned_count(), 2);

        // The cross-cell references to the evicted entities now resolve to
        // nothing, and pruning drops exactly those (design §22 risk 4).
        let dropped = driver.prune_refs(&world, &mut refs);
        assert_eq!(dropped, 2);
        assert_eq!(refs.len(), 1);
        let remaining: Vec<Entity> = refs.iter_alive(&world).collect();
        assert_eq!(remaining, alloc::vec![survivor]);
        assert!(world.contains(survivor));
    }

    #[test]
    fn stream_reports_entities_already_despawned_out_of_band() {
        let mut world = World::new();
        let mut driver = StreamDriver::new(1, 1);

        let far = cell(100, 0, 0);
        let ents = populate(&mut driver, &mut world, far, 2);

        // One entity is despawned out of band; the index still tracks it.
        assert!(world.despawn(ents[0]));

        let result = driver.stream(&mut world, &[CellCoord::ORIGIN]);
        // The live one is despawned; the stale one is reported, not double-counted.
        assert_eq!(result.despawned, alloc::vec![ents[1]]);
        assert_eq!(result.already_gone, alloc::vec![ents[0]]);
        assert!(!world.contains(ents[1]));
    }

    #[test]
    fn settle_loaded_after_request_completes_the_load() {
        let mut world = World::new();
        let mut driver = StreamDriver::new(1, 2);

        let result = driver.stream(&mut world, &[CellCoord::ORIGIN]);
        let target = result.to_load[0];
        assert_eq!(driver.streamer().state(target), CellState::Loading);

        // Owner reads the cell off disk, spawns its entities, confirms the load.
        let spawned = alloc::vec![world.spawn(()), world.spawn(())];
        driver.settle_loaded(target, &spawned);

        assert!(driver.streamer().is_loaded(target));
        assert_eq!(driver.index().entities_in(target), {
            let mut s = spawned.clone();
            s.sort();
            s
        });
    }

    #[test]
    fn evict_cell_forces_immediate_teardown() {
        let mut world = World::new();
        let mut driver = StreamDriver::new(2, 4);

        let c = cell(1, 2, 3);
        let ents = populate(&mut driver, &mut world, c, 3);
        assert!(driver.streamer().is_loaded(c));

        let despawned = driver.evict_cell(&mut world, c);
        let mut expected = ents.clone();
        expected.sort();
        assert_eq!(despawned, expected);
        for &e in &ents {
            assert!(!world.contains(e));
        }
        assert_eq!(driver.streamer().state(c), CellState::Unloaded);
        assert!(driver.index().is_empty());
    }
}
