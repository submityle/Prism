//! World Partition cell streaming (design §13.1).
//!
//! A large world is diced into a regular 3D grid of **cells**. Only cells near
//! something the game cares about — the camera, a player, an active mission
//! anchor — need their entities, meshes and physics resident in memory; the
//! rest stay on disk. This module is the CPU-side *scheduler* that decides,
//! each frame, **which cells should be loaded** and which should be evicted. It
//! is the bookkeeping half of UE5-style World Partition streaming: the actual
//! disk serialization and GPU upload live in the render/scene crates, which
//! consume the [`StreamingDelta`] this scheduler produces.
//!
//! # Interest-driven streaming with hysteresis
//!
//! Streaming is driven by a set of **interest sources** — the cells that
//! "want" their surroundings loaded. Around each interest we load a Chebyshev
//! (cube) ball of radius `load_radius` cells. To avoid *thrashing* — a cell on
//! the exact boundary flickering load/unload as an interest jitters back and
//! forth across the edge — eviction uses a strictly larger `unload_radius`.
//! A cell is only unloaded once it leaves the (larger) unload ball of *every*
//! interest. The band between the two radii is the hysteresis margin: cells
//! there are kept resident but are not (re)loaded.
//!
//! # Async completion
//!
//! [`CellStreamer::update`] never performs I/O: it only *requests* loads and
//! unloads, moving cells into the transient [`CellState::Loading`] /
//! [`CellState::Unloading`] states. When the owner finishes the disk work it
//! calls [`CellStreamer::mark_loaded`] / [`CellStreamer::mark_unloaded`] to
//! settle the cell into [`CellState::Loaded`] or drop it entirely.
//!
//! # Determinism
//!
//! Both vectors in the returned [`StreamingDelta`] are sorted by `(x, y, z)`,
//! so the schedule is frame-stable and independent of hash-map iteration order
//! (design §14).

use alloc::vec::Vec;

use crate::collections::HashMap;
use crate::component::Component;

/// Integer index of a world-partition cell on the regular streaming grid
/// (design §13.1).
///
/// Each axis counts whole cells from the world origin; the mapping between a
/// continuous world position and its cell is defined by the owning scene (it
/// typically mirrors the floating-origin
/// [`GridCell`](crate::partition::floating_origin::GridCell) grid). The type is
/// cheap, `Copy`, and hashable so it can key the streamer's map and tag
/// entities as a [`WorldPartitionCell`] component.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CellCoord {
    /// Cell index along X.
    pub x: i32,
    /// Cell index along Y.
    pub y: i32,
    /// Cell index along Z.
    pub z: i32,
}

impl CellCoord {
    /// The origin cell `(0, 0, 0)`.
    pub const ORIGIN: Self = Self { x: 0, y: 0, z: 0 };

    /// Creates a cell coordinate from its three integer axes.
    #[inline]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Chebyshev (chessboard / ring) distance to `other`: the maximum of the
    /// per-axis absolute differences.
    ///
    /// This is the metric streaming uses, because a Chebyshev ball of radius
    /// `r` is exactly the cube of cells within `r` steps on every axis — the
    /// natural "load everything around me" region.
    #[inline]
    pub const fn ring_distance(self, other: Self) -> i32 {
        let dx = (self.x - other.x).abs();
        let dy = (self.y - other.y).abs();
        let dz = (self.z - other.z).abs();
        let m = if dx > dy { dx } else { dy };
        if m > dz { m } else { dz }
    }

    /// Manhattan (taxicab) distance to `other`: the sum of the per-axis
    /// absolute differences. Occasionally useful for cost heuristics; the
    /// scheduler itself streams on [`ring_distance`](Self::ring_distance).
    #[inline]
    pub const fn manhattan(self, other: Self) -> i32 {
        (self.x - other.x).abs() + (self.y - other.y).abs() + (self.z - other.z).abs()
    }
}

/// Component tagging the world-partition cell an entity belongs to
/// (design §13.1).
///
/// Spawning systems attach this so the scene can bucket entities per cell and
/// (de)serialize them together when the owning cell streams in or out.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct WorldPartitionCell {
    /// The cell this entity currently lives in.
    pub coord: CellCoord,
}

impl WorldPartitionCell {
    /// Creates the tag for a given cell.
    #[inline]
    pub const fn new(coord: CellCoord) -> Self {
        Self { coord }
    }
}

impl Component for WorldPartitionCell {}

/// Lifecycle state of a single streamed cell.
///
/// A cell starts [`Unloaded`](Self::Unloaded) (not tracked at all), is promoted
/// through [`Loading`](Self::Loading) while its disk payload is read, reaches
/// [`Loaded`](Self::Loaded) when resident, and passes through
/// [`Unloading`](Self::Unloading) while it is being evicted before returning to
/// `Unloaded`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CellState {
    /// Not resident and not tracked by the streamer. This is the default.
    #[default]
    Unloaded,
    /// A load has been requested; disk I/O is in flight.
    Loading,
    /// Fully resident and ready for simulation/rendering.
    Loaded,
    /// An unload has been requested; eviction I/O is in flight.
    Unloading,
}

/// The set of load/unload requests produced by one
/// [`CellStreamer::update`] (design §13.1).
///
/// Both vectors are sorted by `(x, y, z)` so the schedule is deterministic and
/// frame-stable regardless of internal hash-map ordering. The owner turns
/// `to_load` into disk reads and `to_unload` into evictions, then reports
/// completion through [`CellStreamer::mark_loaded`] /
/// [`CellStreamer::mark_unloaded`].
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct StreamingDelta {
    /// Cells that newly entered a load ball and should be read from disk.
    pub to_load: Vec<CellCoord>,
    /// Cells that left every unload ball and should be evicted.
    pub to_unload: Vec<CellCoord>,
}

impl StreamingDelta {
    /// Whether this delta requests no work at all.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.to_load.is_empty() && self.to_unload.is_empty()
    }
}

/// The World Partition cell-streaming scheduler (design §13.1).
///
/// Tracks every cell that is not [`CellState::Unloaded`] in a map and, each
/// frame, reconciles that set against the Chebyshev balls around the current
/// interest sources, emitting a [`StreamingDelta`]. The `unload_radius`
/// hysteresis margin (`>= load_radius`) keeps boundary cells from thrashing.
///
/// The streamer is pure and `World`-independent: it only shuffles
/// [`CellCoord`]/[`CellState`] bookkeeping, which makes it fully CPU-testable
/// and deterministic.
pub struct CellStreamer {
    /// All cells currently in a non-`Unloaded` state, keyed by coordinate.
    cells: HashMap<CellCoord, CellState>,
    /// Chebyshev radius, in cells, of the ball loaded around each interest.
    load_radius: u32,
    /// Chebyshev radius, in cells, beyond which a tracked cell is evicted.
    /// Always `>= load_radius`; the gap is the anti-thrash hysteresis band.
    unload_radius: u32,
}

impl CellStreamer {
    /// Creates a streamer with the given Chebyshev load and unload radii, both
    /// counted in cells.
    ///
    /// # Contract
    /// `unload_radius` must be `>= load_radius`; this method panics otherwise,
    /// because an unload radius smaller than the load radius would evict cells
    /// the same frame they are loaded.
    pub fn new(load_radius: u32, unload_radius: u32) -> Self {
        assert!(
            unload_radius >= load_radius,
            "unload_radius ({unload_radius}) must be >= load_radius ({load_radius})"
        );
        Self {
            cells: HashMap::new(),
            load_radius,
            unload_radius,
        }
    }

    /// The Chebyshev load radius, in cells.
    #[inline]
    pub fn load_radius(&self) -> u32 {
        self.load_radius
    }

    /// The Chebyshev unload (hysteresis) radius, in cells.
    #[inline]
    pub fn unload_radius(&self) -> u32 {
        self.unload_radius
    }

    /// Reconciles the tracked set against the current interest sources and
    /// returns the resulting [`StreamingDelta`].
    ///
    /// Deterministically:
    /// * the *desired* set is the union of the `load_radius` Chebyshev balls
    ///   around every interest;
    /// * each desired cell that is not already resident/loading transitions to
    ///   [`CellState::Loading`] and is emitted in `to_load` (a cell mid-eviction
    ///   is revived rather than left to die);
    /// * each tracked cell outside the `unload_radius` ball of *every* interest
    ///   transitions to [`CellState::Unloading`] and is emitted in `to_unload`;
    /// * cells inside the hysteresis band (outside `load_radius` but within
    ///   `unload_radius`) are left untouched.
    ///
    /// Both result vectors are sorted by `(x, y, z)`. With no interests the
    /// desired set is empty, so every tracked cell is scheduled for unload.
    pub fn update(&mut self, interests: &[CellCoord]) -> StreamingDelta {
        let load_r = self.load_radius as i32;
        let unload_r = self.unload_radius as i32;

        let mut to_load = Vec::new();
        let mut to_unload = Vec::new();

        // 1. Desired set = union of load balls. A HashMap keyed by coord acts
        //    as a dedup'd set (several interests overlap in dense scenes).
        let mut desired: HashMap<CellCoord, ()> = HashMap::new();
        for interest in interests {
            for dz in -load_r..=load_r {
                for dy in -load_r..=load_r {
                    for dx in -load_r..=load_r {
                        let coord = CellCoord::new(
                            interest.x + dx,
                            interest.y + dy,
                            interest.z + dz,
                        );
                        desired.insert(coord, ());
                    }
                }
            }
        }

        // 2. Load pass: promote every desired cell that is not yet resident or
        //    already loading. A cell caught mid-unload is revived to Loading.
        for (&coord, _) in desired.iter() {
            match self.cells.get(&coord).copied() {
                Some(CellState::Loaded) | Some(CellState::Loading) => {}
                _ => {
                    self.cells.insert(coord, CellState::Loading);
                    to_load.push(coord);
                }
            }
        }

        // 3. Unload pass: any tracked cell outside the unload ball of *every*
        //    interest is evicted. Collect first to avoid mutating while reading.
        let mut evicting = Vec::new();
        for (&coord, &state) in self.cells.iter() {
            if state == CellState::Unloading {
                continue; // already in flight; don't re-emit.
            }
            let kept = interests
                .iter()
                .any(|i| i.ring_distance(coord) <= unload_r);
            if !kept {
                evicting.push(coord);
            }
        }
        for coord in evicting {
            self.cells.insert(coord, CellState::Unloading);
            to_unload.push(coord);
        }

        sort_coords(&mut to_load);
        sort_coords(&mut to_unload);

        StreamingDelta { to_load, to_unload }
    }

    /// Settles a cell whose load I/O just finished: [`CellState::Loading`] ->
    /// [`CellState::Loaded`]. A no-op for cells in any other state.
    pub fn mark_loaded(&mut self, coord: CellCoord) {
        if let Some(state) = self.cells.get_mut(&coord)
            && *state == CellState::Loading
        {
            *state = CellState::Loaded;
        }
    }

    /// Settles a cell whose eviction I/O just finished: [`CellState::Unloading`]
    /// -> removed from the map (returning to [`CellState::Unloaded`]). A no-op
    /// for cells in any other state.
    pub fn mark_unloaded(&mut self, coord: CellCoord) {
        if matches!(self.cells.get(&coord), Some(CellState::Unloading)) {
            self.cells.remove(&coord);
        }
    }

    /// The current [`CellState`] of `coord`. Untracked cells report
    /// [`CellState::Unloaded`].
    #[inline]
    pub fn state(&self, coord: CellCoord) -> CellState {
        self.cells.get(&coord).copied().unwrap_or_default()
    }

    /// Whether `coord` is fully resident ([`CellState::Loaded`]).
    #[inline]
    pub fn is_loaded(&self, coord: CellCoord) -> bool {
        self.state(coord) == CellState::Loaded
    }

    /// The number of cells currently in [`CellState::Loaded`].
    pub fn loaded_count(&self) -> usize {
        self.cells
            .values()
            .filter(|&&s| s == CellState::Loaded)
            .count()
    }

    /// The number of cells currently tracked (anything not
    /// [`CellState::Unloaded`]).
    #[inline]
    pub fn tracked_count(&self) -> usize {
        self.cells.len()
    }

    /// Iterates over every tracked cell and its state, in unspecified order.
    pub fn iter(&self) -> impl Iterator<Item = (CellCoord, CellState)> + '_ {
        self.cells.iter().map(|(&coord, &state)| (coord, state))
    }
}

/// Sorts a coordinate list by `(x, y, z)` for deterministic deltas.
#[inline]
fn sort_coords(coords: &mut [CellCoord]) {
    coords.sort_unstable_by(|a, b| {
        a.x.cmp(&b.x)
            .then(a.y.cmp(&b.y))
            .then(a.z.cmp(&b.z))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Number of cells in a Chebyshev ball of the given radius: `(2r + 1)^3`.
    fn ball_size(r: i32) -> usize {
        let side = (2 * r + 1) as usize;
        side * side * side
    }

    #[test]
    fn ring_distance_is_chebyshev() {
        let a = CellCoord::new(0, 0, 0);
        assert_eq!(a.ring_distance(CellCoord::new(2, 1, -3)), 3);
        assert_eq!(a.ring_distance(CellCoord::new(-4, 0, 1)), 4);
        assert_eq!(a.ring_distance(a), 0);
    }

    #[test]
    fn manhattan_sums_axes() {
        let a = CellCoord::new(1, 1, 1);
        assert_eq!(a.manhattan(CellCoord::new(2, 3, -1)), 1 + 2 + 2);
    }

    #[test]
    fn component_tag_carries_coord() {
        let tag = WorldPartitionCell::new(CellCoord::new(3, -2, 5));
        assert_eq!(tag.coord, CellCoord::new(3, -2, 5));
        assert_eq!(WorldPartitionCell::default().coord, CellCoord::ORIGIN);
    }

    #[test]
    fn default_state_is_unloaded() {
        assert_eq!(CellState::default(), CellState::Unloaded);
    }

    #[test]
    #[should_panic]
    fn unload_radius_below_load_radius_panics() {
        let _ = CellStreamer::new(3, 2);
    }

    #[test]
    fn first_update_loads_the_whole_ball() {
        let mut s = CellStreamer::new(1, 2);
        let delta = s.update(&[CellCoord::ORIGIN]);

        assert_eq!(delta.to_load.len(), ball_size(1)); // 27
        assert!(delta.to_unload.is_empty());

        // Every ball cell is now Loading and tracked.
        for &c in &delta.to_load {
            assert_eq!(s.state(c), CellState::Loading);
            assert!(c.ring_distance(CellCoord::ORIGIN) <= 1);
        }
        assert_eq!(s.tracked_count(), ball_size(1));
        assert_eq!(s.loaded_count(), 0);
    }

    #[test]
    fn deltas_are_sorted_by_xyz() {
        let mut s = CellStreamer::new(2, 3);
        let delta = s.update(&[CellCoord::ORIGIN]);
        assert_eq!(delta.to_load.len(), ball_size(2)); // 125

        let mut prev: Option<CellCoord> = None;
        for &c in &delta.to_load {
            if let Some(p) = prev {
                let ordered = (p.x, p.y, p.z) <= (c.x, c.y, c.z);
                assert!(ordered, "not sorted: {p:?} then {c:?}");
            }
            prev = Some(c);
        }
    }

    #[test]
    fn second_update_same_interest_is_noop() {
        let mut s = CellStreamer::new(1, 2);
        s.update(&[CellCoord::ORIGIN]);
        let delta = s.update(&[CellCoord::ORIGIN]);
        assert!(delta.is_empty());
    }

    #[test]
    fn mark_loaded_and_unloaded_state_machine() {
        let mut s = CellStreamer::new(1, 1);
        let c = CellCoord::ORIGIN;
        s.update(&[c]);
        assert_eq!(s.state(c), CellState::Loading);

        // mark_loaded only acts on Loading cells.
        s.mark_loaded(c);
        assert_eq!(s.state(c), CellState::Loaded);
        assert!(s.is_loaded(c));
        assert_eq!(s.loaded_count(), 1);

        // mark_unloaded is a no-op while Loaded (not yet Unloading).
        s.mark_unloaded(c);
        assert_eq!(s.state(c), CellState::Loaded);

        // Leaving the interest schedules an unload.
        let far = CellCoord::new(100, 0, 0);
        let delta = s.update(&[far]);
        assert!(delta.to_unload.contains(&c));
        assert_eq!(s.state(c), CellState::Unloading);

        // Now mark_unloaded drops it entirely from the tracked map.
        let before = s.tracked_count();
        s.mark_unloaded(c);
        assert_eq!(s.state(c), CellState::Unloaded);
        assert_eq!(s.tracked_count(), before - 1);
    }

    #[test]
    fn mark_loaded_ignores_non_loading() {
        let mut s = CellStreamer::new(1, 1);
        let c = CellCoord::new(7, 7, 7); // untracked
        s.mark_loaded(c);
        assert_eq!(s.state(c), CellState::Unloaded);
    }

    #[test]
    fn hysteresis_keeps_band_cells_resident() {
        // load 1, unload 2: a cell at Chebyshev distance 2 from the new
        // interest is outside the load ball but inside the unload ball, so it
        // must NOT be unloaded (anti-thrash).
        let mut s = CellStreamer::new(1, 2);
        s.update(&[CellCoord::ORIGIN]);
        // Fully load everything so we can observe survival as `Loaded`.
        let loaded: Vec<CellCoord> = s.iter().map(|(c, _)| c).collect();
        for c in loaded {
            s.mark_loaded(c);
        }

        let origin = CellCoord::ORIGIN;
        assert!(s.is_loaded(origin));

        // Move interest to (2,0,0): origin is now at ring distance 2.
        let interest = CellCoord::new(2, 0, 0);
        assert_eq!(interest.ring_distance(origin), 2); // == unload_radius
        let delta = s.update(&[interest]);

        // Origin is in the hysteresis band: not reloaded, not unloaded.
        assert!(!delta.to_unload.contains(&origin));
        assert_eq!(s.state(origin), CellState::Loaded);

        // But a cell at distance 3 (e.g. (-1,0,0)) leaves the unload ball.
        let gone = CellCoord::new(-1, 0, 0);
        assert_eq!(interest.ring_distance(gone), 3);
        assert!(delta.to_unload.contains(&gone));
        assert_eq!(s.state(gone), CellState::Unloading);
    }

    #[test]
    fn unload_when_interest_leaves_entirely() {
        let mut s = CellStreamer::new(1, 2);
        let first = s.update(&[CellCoord::ORIGIN]);
        let n = first.to_load.len();

        // Jump far away: every previously tracked cell is now beyond unload.
        let delta = s.update(&[CellCoord::new(1000, 1000, 1000)]);
        assert_eq!(delta.to_unload.len(), n);
        for &c in &delta.to_unload {
            assert_eq!(s.state(c), CellState::Unloading);
        }
        // And the new ball was requested.
        assert_eq!(delta.to_load.len(), ball_size(1));
    }

    #[test]
    fn revives_cell_caught_mid_unload() {
        let mut s = CellStreamer::new(1, 1);
        let c = CellCoord::ORIGIN;
        s.update(&[c]);
        s.mark_loaded(c);

        // Leave -> schedule unload.
        s.update(&[CellCoord::new(50, 0, 0)]);
        assert_eq!(s.state(c), CellState::Unloading);

        // Return before the eviction completes: the cell is re-requested.
        let delta = s.update(&[c]);
        assert!(delta.to_load.contains(&c));
        assert_eq!(s.state(c), CellState::Loading);
    }

    #[test]
    fn multiple_interests_union_their_balls() {
        let mut s = CellStreamer::new(1, 1);
        let a = CellCoord::new(0, 0, 0);
        let b = CellCoord::new(5, 0, 0); // far enough that balls don't overlap
        let delta = s.update(&[a, b]);

        // Disjoint balls -> exactly two full balls' worth of loads.
        assert_eq!(delta.to_load.len(), 2 * ball_size(1));
        assert!(delta.to_load.contains(&a));
        assert!(delta.to_load.contains(&b));

        // Overlapping interests dedup to a single union with no double-counts.
        let mut s2 = CellStreamer::new(1, 1);
        let delta2 = s2.update(&[a, CellCoord::new(1, 0, 0)]);
        // Union of two radius-1 cubes offset by 1 on X: 4 x 3 x 3 = 36 cells.
        assert_eq!(delta2.to_load.len(), 36);
        // No coordinate appears twice.
        let mut seen = delta2.to_load.clone();
        seen.sort_unstable_by(|p, q| {
            p.x.cmp(&q.x).then(p.y.cmp(&q.y)).then(p.z.cmp(&q.z))
        });
        seen.dedup();
        assert_eq!(seen.len(), delta2.to_load.len());
    }

    #[test]
    fn empty_interests_unload_everything() {
        let mut s = CellStreamer::new(1, 2);
        let first = s.update(&[CellCoord::ORIGIN]);
        let delta = s.update(&[]);
        assert!(delta.to_load.is_empty());
        assert_eq!(delta.to_unload.len(), first.to_load.len());
    }

    #[test]
    fn update_output_is_deterministic_across_runs() {
        let interests = [
            CellCoord::new(0, 0, 0),
            CellCoord::new(3, -1, 2),
            CellCoord::new(-2, 4, 0),
        ];
        let mut a = CellStreamer::new(2, 3);
        let mut b = CellStreamer::new(2, 3);
        assert_eq!(a.update(&interests), b.update(&interests));
    }
}
