//! Big-world streaming subsystem that unifies the §13 pieces (design §13.1 +
//! §13.3).
//!
//! The individual partition primitives each own one concern:
//!
//! - [`FloatingOrigin`] keeps a camera-local `f32` frame so distant geometry
//!   never jitters (§13.3);
//! - [`InterestGrid`] maps continuous [`WorldPos`] interest sources onto the
//!   integer streaming lattice (§13.1 bridge);
//! - [`StreamDriver`] turns a cell interest set into real [`World`] load/evict
//!   mutation (§13.1).
//!
//! Wiring those three together correctly every frame — recenter the origin on
//! the camera, project every interest onto the *same* lattice, then drive the
//! streamer — is exactly the glue an owner would otherwise re-implement by
//! hand. [`WorldPartition`] is that subsystem (the UE5 *World Partition
//! subsystem* form): one [`advance`](WorldPartition::advance) call per frame
//! runs the whole loop, with the grids guaranteed to stay aligned because the
//! interest grid is built by [`InterestGrid::matching`] from the same origin.
//!
//! It still does **no disk I/O**: like [`StreamDriver`], loads are surfaced as
//! requests in the returned [`StreamResult`] and confirmed by the owner through
//! [`settle_loaded`](WorldPartition::settle_loaded) once the scene crate has
//! spawned the cell's entities. The kernel owns the bookkeeping and the evict
//! closure; the scene owns the bytes.
//!
//! ```
//! use prism_ecs::partition::floating_origin::WorldPos;
//! use prism_ecs::partition::world_partition::WorldPartition;
//! use prism_ecs::world::World;
//!
//! let mut world = World::new();
//! // 1 km cells, load a 1-cell ball, evict beyond a 2-cell ball.
//! let mut partition = WorldPartition::new(1000.0, 1, 2);
//!
//! // Frame N: camera sits in cell (1, 0, 0); stream its surroundings.
//! let camera = WorldPos::new(1500.0, 20.0, 0.0);
//! let result = partition.advance(&mut world, camera, &[]);
//! assert!(!result.to_load.is_empty()); // owner reads these off disk
//! // The origin now tracks the camera cell so rebased offsets stay small.
//! assert_eq!(partition.origin().origin().x, 1);
//! ```

use alloc::vec::Vec;

use crate::partition::cell::{CellCoord, CellStreamer};
use crate::partition::data_layer::DataLayers;
use crate::partition::driver::{StreamDriver, StreamResult};
use crate::partition::floating_origin::{FloatingOrigin, GridCell, LocalPos, WorldPos};
use crate::partition::interest::InterestGrid;
use crate::partition::streaming::{CellEntityIndex, WeakRefs};
use crate::world::World;

/// The UE5-style World Partition subsystem: a per-frame big-world streaming
/// driver that keeps a [`FloatingOrigin`], an [`InterestGrid`] and a
/// [`StreamDriver`] in lockstep (design §13.1 + §13.3).
///
/// Construct with [`new`](Self::new) (which keeps all three grids aligned by
/// construction) and call [`advance`](Self::advance) once per frame with the
/// camera position and any extra interest sources. The subsystem recenters the
/// origin on the camera, projects every interest onto the shared lattice, and
/// drives the streamer — returning the owner's to-do list ([`StreamResult`]).
#[derive(Clone, Debug)]
pub struct WorldPartition {
    origin: FloatingOrigin,
    interest: InterestGrid,
    driver: StreamDriver,
    data_layers: DataLayers,
}

impl WorldPartition {
    /// Builds a subsystem with a cubic cell of `cell_size` metres and a
    /// streamer loading within `load_radius` cells / evicting beyond
    /// `unload_radius` cells.
    ///
    /// The interest grid is derived from the origin via
    /// [`InterestGrid::matching`], so the streaming lattice mirrors the
    /// floating-origin [`GridCell`] grid exactly — an entity is rebased against
    /// and streamed on the same cells.
    ///
    /// # Panics
    /// Panics if `cell_size` is not finite and `> 0` (see [`FloatingOrigin::new`])
    /// or if `unload_radius < load_radius` (see [`CellStreamer::new`]).
    pub fn new(cell_size: f64, load_radius: u32, unload_radius: u32) -> Self {
        let origin = FloatingOrigin::new(cell_size);
        let interest = InterestGrid::matching(&origin);
        Self {
            origin,
            interest,
            driver: StreamDriver::new(load_radius, unload_radius),
            data_layers: DataLayers::new(),
        }
    }

    /// Builds a subsystem around an existing [`FloatingOrigin`] and
    /// [`CellStreamer`], deriving the interest grid from the origin so the two
    /// lattices stay aligned.
    ///
    /// Use this to preserve an already-recentered origin or a streamer with
    /// resident cells across a subsystem rebuild.
    pub fn from_parts(origin: FloatingOrigin, streamer: CellStreamer) -> Self {
        let interest = InterestGrid::matching(&origin);
        Self {
            origin,
            interest,
            driver: StreamDriver::with_streamer(streamer),
            data_layers: DataLayers::new(),
        }
    }

    /// The floating origin (shared ref).
    #[inline]
    pub fn origin(&self) -> &FloatingOrigin {
        &self.origin
    }

    /// The interest grid (shared ref).
    #[inline]
    pub fn interest_grid(&self) -> &InterestGrid {
        &self.interest
    }

    /// The underlying stream driver (shared ref).
    #[inline]
    pub fn driver(&self) -> &StreamDriver {
        &self.driver
    }

    /// The underlying stream driver (mutable ref), for registering entities or
    /// forcing an eviction outside the per-frame [`advance`](Self::advance).
    #[inline]
    pub fn driver_mut(&mut self) -> &mut StreamDriver {
        &mut self.driver
    }

    /// The data-layer registry (shared ref). Empty by default, in which case
    /// every cell is streamable and [`advance`](Self::advance) behaves like
    /// plain proximity streaming (design §13.1).
    #[inline]
    pub fn data_layers(&self) -> &DataLayers {
        &self.data_layers
    }

    /// The data-layer registry (mutable ref): register layers, toggle their
    /// [`DataLayerState`](crate::partition::data_layer::DataLayerState), and
    /// assign cells to them. Changes take effect on the next
    /// [`advance`](Self::advance) — unloading a layer evicts its resident cells,
    /// (re)loading it lets them stream back in (design §13.1).
    #[inline]
    pub fn data_layers_mut(&mut self) -> &mut DataLayers {
        &mut self.data_layers
    }

    /// Advances streaming for one frame.
    ///
    /// 1. recenters the floating origin onto the cell containing `camera`, so
    ///    rebased render/physics offsets stay near zero (§13.3); the camera's
    ///    `(cell, local)` split is returned for the owner's own rebasing;
    /// 2. projects `camera` plus every `extra` interest source onto the shared
    ///    lattice via the interest grid — sorted and de-duplicated (§14);
    /// 3. drives the [`StreamDriver`], evicting cells that left every interest's
    ///    unload ball (despawning their entities) and surfacing the still-
    ///    pending loads. The eviction/load set is gated by the subsystem's
    ///    [`DataLayers`](Self::data_layers): a cell whose data layer is unloaded
    ///    is neither loaded nor kept resident (§13.1).
    ///
    /// Returns the resulting [`StreamResult`]: the unload half is already
    /// applied to `world`, the load half is the owner's to satisfy. The
    /// camera's rebased split is available via [`camera_split`](Self::camera_split)
    /// or by calling [`FloatingOrigin::quantize`] on [`origin`](Self::origin).
    pub fn advance(
        &mut self,
        world: &mut World,
        camera: WorldPos,
        extra: &[WorldPos],
    ) -> StreamResult {
        // Track the camera so subsequent rebasing keeps full f32 precision. The
        // cell size itself is the recenter hysteresis (the origin only moves
        // when the camera crosses a cell boundary).
        self.origin.recenter_to(camera);

        // Build the interest set from the camera and every extra source on the
        // one shared lattice. Collecting into a small scratch Vec keeps the
        // determinism (sort + dedup) in `InterestGrid::interests`.
        let interests = self.interests_for(camera, extra);
        let data_layers = &self.data_layers;
        self.driver
            .stream_filtered(world, &interests, |cell| data_layers.is_cell_streamable(cell))
    }

    /// Projects `camera` plus `extra` onto the streaming lattice, returning the
    /// sorted, de-duplicated interest cells that [`advance`](Self::advance)
    /// feeds the driver — exposed for callers that want to inspect or stream
    /// manually without recentering.
    pub fn interests_for(&self, camera: WorldPos, extra: &[WorldPos]) -> Vec<CellCoord> {
        if extra.is_empty() {
            // Fast path: a lone camera interest needs no scratch buffer.
            return self.interest.interests(&[camera]);
        }
        let mut sources = Vec::with_capacity(extra.len() + 1);
        sources.push(camera);
        sources.extend_from_slice(extra);
        self.interest.interests(&sources)
    }

    /// Splits a world position into this subsystem's current `(cell, local)`
    /// on the floating-origin grid — e.g. to rebase the camera after an
    /// [`advance`](Self::advance).
    #[inline]
    pub fn camera_split(&self, pos: WorldPos) -> (GridCell, LocalPos) {
        self.origin.quantize(pos)
    }

    /// Confirms that a requested load finished: records the spawned entities
    /// against `cell` and settles it to `Loaded`. Forwards to
    /// [`StreamDriver::settle_loaded`]; call it for each cell in
    /// [`StreamResult::to_load`] after the scene spawns its entities.
    #[inline]
    pub fn settle_loaded(&mut self, cell: CellCoord, entities: &[crate::entity::Entity]) {
        self.driver.settle_loaded(cell, entities);
    }

    /// Prunes a cross-cell [`WeakRefs`] set against `world`, dropping references
    /// to entities evicted by a prior [`advance`](Self::advance). Forwards to
    /// [`StreamDriver::prune_refs`]; returns how many were dropped (design §22
    /// risk 4).
    #[inline]
    pub fn prune_refs(&self, world: &World, refs: &mut WeakRefs) -> usize {
        self.driver.prune_refs(world, refs)
    }

    /// The per-cell entity index (shared ref).
    #[inline]
    pub fn index(&self) -> &CellEntityIndex {
        self.driver.index()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::Entity;
    use crate::partition::cell::CellState;

    /// Spawns `n` entities into `cell` and registers them resident, mirroring a
    /// completed load.
    fn populate(
        partition: &mut WorldPartition,
        world: &mut World,
        cell: CellCoord,
        n: usize,
    ) -> Vec<Entity> {
        let ents: Vec<Entity> = (0..n).map(|_| world.spawn(())).collect();
        partition.settle_loaded(cell, &ents);
        ents
    }

    #[test]
    fn new_aligns_interest_grid_with_origin() {
        let p = WorldPartition::new(1000.0, 1, 2);
        assert_eq!(p.interest_grid().cell_size(), p.origin().cell_size());
    }

    #[test]
    fn advance_recenters_origin_onto_camera_cell() {
        let mut world = World::new();
        let mut p = WorldPartition::new(1000.0, 1, 2);
        let camera = WorldPos::new(2500.0, -1500.0, 10.0); // cell (2, -2, 0)
        p.advance(&mut world, camera, &[]);
        assert_eq!(p.origin().origin(), GridCell::new(2, -2, 0));
    }

    #[test]
    fn advance_requests_loads_around_camera_without_spawning() {
        let mut world = World::new();
        let mut p = WorldPartition::new(100.0, 1, 2);
        let result = p.advance(&mut world, WorldPos::new(50.0, 50.0, 50.0), &[]);
        // A radius-1 Chebyshev ball around the camera cell = 27 cells.
        assert_eq!(result.to_load.len(), 27);
        assert!(result.unloaded_cells.is_empty());
        assert!(result.despawned.is_empty());
        // The kernel spawned nothing: loading is the owner's job.
        assert_eq!(p.index().entity_count(), 0);
        for &c in &result.to_load {
            assert_eq!(p.driver().streamer().state(c), CellState::Loading);
        }
    }

    #[test]
    fn moving_camera_far_evicts_old_cells_and_despawns_entities() {
        let mut world = World::new();
        let mut p = WorldPartition::new(100.0, 1, 1);

        // A cell resident near the origin with two entities.
        let home = CellCoord::new(0, 0, 0);
        let ents = populate(&mut p, &mut world, home, 2);
        for &e in &ents {
            assert!(world.contains(e));
        }

        // Camera jumps far away: `home` leaves the unload ball and is evicted.
        let result = p.advance(&mut world, WorldPos::new(100_000.0, 0.0, 0.0), &[]);
        assert!(result.unloaded_cells.contains(&home));
        let mut expected = ents.clone();
        expected.sort();
        assert_eq!(result.despawned, expected);
        for &e in &ents {
            assert!(!world.contains(e));
        }
        assert!(p.index().entities_in(home).is_empty());
    }

    #[test]
    fn extra_interest_keeps_its_neighbourhood_resident() {
        let mut world = World::new();
        let mut p = WorldPartition::new(100.0, 1, 1);

        // An entity far from the camera but near a mission anchor.
        let anchor_cell = CellCoord::new(500, 0, 0);
        let ents = populate(&mut p, &mut world, anchor_cell, 1);

        // Camera is at the origin; the anchor's own WorldPos keeps its cell
        // inside an interest ball, so nothing there is evicted.
        let anchor_pos = WorldPos::new(50_050.0, 50.0, 50.0); // cell (500, 0, 0)
        let result = p.advance(&mut world, WorldPos::new(50.0, 50.0, 50.0), &[anchor_pos]);
        assert!(!result.unloaded_cells.contains(&anchor_cell));
        assert!(world.contains(ents[0]));
    }

    #[test]
    fn interests_for_dedups_camera_and_colocated_extra() {
        let p = WorldPartition::new(100.0, 1, 2);
        let camera = WorldPos::new(150.0, 0.0, 0.0); // cell (1, 0, 0)
        let ally = WorldPos::new(199.0, 0.0, 0.0); //   cell (1, 0, 0) — same
        let far = WorldPos::new(-10.0, 0.0, 0.0); //    cell (-1, 0, 0)
        let interests = p.interests_for(camera, &[ally, far]);
        assert_eq!(
            interests,
            [CellCoord::new(-1, 0, 0), CellCoord::new(1, 0, 0)]
        );
    }

    #[test]
    fn interests_for_lone_camera_matches_grid() {
        let p = WorldPartition::new(100.0, 1, 2);
        let camera = WorldPos::new(150.0, 250.0, -50.0);
        assert_eq!(
            p.interests_for(camera, &[]),
            alloc::vec![p.interest_grid().cell_of(camera)]
        );
    }

    #[test]
    fn data_layer_gates_streaming_and_eviction() {
        use crate::partition::data_layer::{DataLayerId, DataLayerState};

        let mut world = World::new();
        let mut p = WorldPartition::new(100.0, 1, 1);

        // Tag the camera's own cell with a data layer and leave it unloaded.
        let home = CellCoord::new(0, 0, 0);
        let layer = DataLayerId::new(1);
        p.data_layers_mut().assign(home, layer);

        // Layer unloaded -> the home cell must NOT be requested.
        let camera = WorldPos::new(50.0, 50.0, 50.0); // cell (0,0,0)
        let result = p.advance(&mut world, camera, &[]);
        assert!(!result.to_load.contains(&home));
        assert_eq!(p.driver().streamer().state(home), CellState::Unloaded);

        // Activate the layer -> the home cell streams in on the next tick.
        p.data_layers_mut().set_state(layer, DataLayerState::Activated);
        let result = p.advance(&mut world, camera, &[]);
        assert!(result.to_load.contains(&home));
        assert_eq!(p.driver().streamer().state(home), CellState::Loading);
    }

    #[test]
    fn unloading_data_layer_evicts_resident_cell() {
        use crate::partition::data_layer::DataLayerId;

        let mut world = World::new();
        let mut p = WorldPartition::new(100.0, 1, 1);

        let home = CellCoord::new(0, 0, 0);
        let layer = DataLayerId::new(1);
        p.data_layers_mut().assign(home, layer);
        p.data_layers_mut().activate(layer);

        // Resident cell with an entity while the layer is active.
        let ents = populate(&mut p, &mut world, home, 1);
        assert!(world.contains(ents[0]));

        // Camera stays on `home`, but unloading the layer evicts it.
        p.data_layers_mut().unload(layer);
        let camera = WorldPos::new(50.0, 50.0, 50.0);
        let result = p.advance(&mut world, camera, &[]);
        assert!(result.unloaded_cells.contains(&home));
        assert!(!world.contains(ents[0]));
    }

    #[test]
    fn empty_data_layers_impose_no_gate() {
        // With no layers assigned, advance is identical to plain proximity
        // streaming: a radius-1 ball around the camera cell = 27 cells.
        let mut world = World::new();
        let mut p = WorldPartition::new(100.0, 1, 2);
        let result = p.advance(&mut world, WorldPos::new(50.0, 50.0, 50.0), &[]);
        assert_eq!(result.to_load.len(), 27);
    }
}
