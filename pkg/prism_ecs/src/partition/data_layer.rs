//! World Partition **data layers** (design §13.1).
//!
//! Cell streaming ([`cell`](crate::partition::cell)) decides *where* to stream
//! based purely on spatial proximity to interest sources. Data layers add the
//! orthogonal, *content*-driven axis UE5 calls **Data Layers**: a named group
//! of cells that can be loaded and activated independently of the camera, so a
//! game can gate whole slices of the world on state rather than distance.
//!
//! Typical uses mirror UE5's: a day/night variant of a region, a mission that
//! only materialises its props once accepted, a destruction state that swaps
//! the pristine building set for the rubble set. In every case the geometry is
//! co-located with ordinary streamed cells, but whether those cells are allowed
//! to stream is decided by a layer toggle, not by walking closer.
//!
//! # Three states (UE5 form)
//!
//! Each layer is in one of three [states](DataLayerState):
//!
//! * [`Unloaded`](DataLayerState::Unloaded) — not in memory; its cells are
//!   *blocked* from streaming even when the camera is right on top of them.
//! * [`Loaded`](DataLayerState::Loaded) — resident in memory but logically
//!   inactive (e.g. pre-warmed, or present-but-hidden); its cells may stream.
//! * [`Activated`](DataLayerState::Activated) — resident *and* live; its cells
//!   stream exactly as [`Loaded`](DataLayerState::Loaded) does. The distinction
//!   between loaded and activated is gameplay-visible (systems can treat an
//!   activated layer as "running") but does not change streaming eligibility.
//!
//! The three states form a total order
//! `Unloaded < Loaded < Activated`, so the streaming gate is simply
//! "state ≥ [`Loaded`](DataLayerState::Loaded)".
//!
//! # Eligibility gate
//!
//! A cell may carry zero or more layer memberships ([`assign`](DataLayers::assign)).
//! A cell is **streamable** iff *every* layer it belongs to is at least
//! [`Loaded`](DataLayerState::Loaded) ([`is_cell_streamable`](DataLayers::is_cell_streamable));
//! a single [`Unloaded`](DataLayerState::Unloaded) layer blocks the whole cell.
//! A cell with no memberships is unconditionally streamable, so the default
//! (empty) registry imposes no gate at all — plain proximity streaming.
//!
//! This type is pure CPU bookkeeping: it owns no [`World`](crate::world::World)
//! state and performs no I/O. It produces the eligibility predicate that
//! [`CellStreamer::update_filtered`](crate::partition::cell::CellStreamer::update_filtered)
//! and [`StreamDriver::stream_filtered`](crate::partition::driver::StreamDriver::stream_filtered)
//! consume to gate the spatial schedule.
//!
//! # Determinism
//!
//! Per-cell membership lists are kept sorted by [`DataLayerId`], so iteration
//! order and the eligibility decision are independent of insertion order
//! (design §14).

use alloc::vec::Vec;

use crate::collections::HashMap;
use crate::partition::cell::CellCoord;

/// Stable identifier of a world-partition data layer (design §13.1).
///
/// Assigned by the owner (scene/editor) and used to key the layer's state and
/// to tag the cells that belong to it. Cheap, `Copy`, and hashable.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct DataLayerId(pub u32);

impl DataLayerId {
    /// Creates a data-layer id from its raw value.
    #[inline]
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    /// The raw integer value.
    #[inline]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Lifecycle state of a single data layer (design §13.1, UE5 three-state form).
///
/// The variants are ordered `Unloaded < Loaded < Activated`; the streaming gate
/// in [`DataLayers::is_cell_streamable`] is the single comparison
/// `state >= DataLayerState::Loaded`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub enum DataLayerState {
    /// Not in memory. The layer's cells are blocked from streaming.
    #[default]
    Unloaded,
    /// Resident in memory but logically inactive. The layer's cells may stream.
    Loaded,
    /// Resident and live. Streams exactly as [`Loaded`](Self::Loaded); the
    /// distinction is gameplay-visible, not streaming-visible.
    Activated,
}

impl DataLayerState {
    /// Whether a layer in this state permits its cells to stream, i.e. the
    /// state is at least [`Loaded`](Self::Loaded).
    #[inline]
    pub const fn permits_streaming(self) -> bool {
        matches!(self, DataLayerState::Loaded | DataLayerState::Activated)
    }

    /// Whether the layer is live ([`Activated`](Self::Activated)).
    #[inline]
    pub const fn is_activated(self) -> bool {
        matches!(self, DataLayerState::Activated)
    }
}

/// Registry of world-partition data layers and the cells that belong to them
/// (design §13.1).
///
/// Holds two maps: each layer's [`DataLayerState`], and each cell's sorted set
/// of layer memberships. From these it derives the per-cell streaming
/// eligibility ([`is_cell_streamable`](Self::is_cell_streamable)) that gates the
/// spatial streamer. It is `World`-independent and does no I/O.
#[derive(Clone, Debug, Default)]
pub struct DataLayers {
    /// Current state of every registered layer. Absent ids read as
    /// [`DataLayerState::Unloaded`].
    states: HashMap<DataLayerId, DataLayerState>,
    /// Per-cell layer memberships, each list kept sorted and de-duplicated by
    /// [`DataLayerId`] for deterministic iteration (design §14).
    cell_layers: HashMap<CellCoord, Vec<DataLayerId>>,
}

impl DataLayers {
    /// Creates an empty registry. With no layers registered or assigned, every
    /// cell is streamable, so this imposes no gate.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `layer` with an initial state of
    /// [`DataLayerState::Unloaded`], if it is not already registered.
    ///
    /// Returns whether the layer was newly registered. Registering is optional:
    /// an unregistered layer reads as [`DataLayerState::Unloaded`] anyway, so
    /// this exists mainly to enumerate known layers and to make an explicit
    /// "exists but unloaded" declaration.
    pub fn register_layer(&mut self, layer: DataLayerId) -> bool {
        if self.states.contains_key(&layer) {
            false
        } else {
            self.states.insert(layer, DataLayerState::Unloaded);
            true
        }
    }

    /// Sets `layer` to `state`, registering it if necessary, and returns the
    /// layer's previous (effective) state.
    ///
    /// An unregistered layer is treated as [`DataLayerState::Unloaded`] before
    /// the change, so toggling an unknown layer to
    /// [`DataLayerState::Loaded`] both registers it and reports
    /// [`DataLayerState::Unloaded`] as its prior state.
    pub fn set_state(&mut self, layer: DataLayerId, state: DataLayerState) -> DataLayerState {
        let prev = self.states.insert(layer, state);
        prev.unwrap_or_default()
    }

    /// The current state of `layer`. Unregistered layers report
    /// [`DataLayerState::Unloaded`].
    #[inline]
    pub fn state(&self, layer: DataLayerId) -> DataLayerState {
        self.states.get(&layer).copied().unwrap_or_default()
    }

    /// Whether `layer` has been explicitly registered.
    #[inline]
    pub fn is_registered(&self, layer: DataLayerId) -> bool {
        self.states.contains_key(&layer)
    }

    /// Number of registered layers.
    #[inline]
    pub fn layer_count(&self) -> usize {
        self.states.len()
    }

    /// Assigns `cell` to `layer`, so the cell's streaming is gated on that
    /// layer being at least [`Loaded`](DataLayerState::Loaded).
    ///
    /// The membership is inserted into the cell's sorted list; a redundant
    /// assignment is a no-op. Returns whether the membership was newly added.
    /// The layer itself does not need to be registered first — but note that
    /// until it is set to [`Loaded`](DataLayerState::Loaded)/[`Activated`](DataLayerState::Activated)
    /// it reads as [`Unloaded`](DataLayerState::Unloaded) and therefore blocks
    /// the cell.
    pub fn assign(&mut self, cell: CellCoord, layer: DataLayerId) -> bool {
        let layers = self.cell_layers.entry(cell).or_default();
        match layers.binary_search(&layer) {
            Ok(_) => false,
            Err(pos) => {
                layers.insert(pos, layer);
                true
            }
        }
    }

    /// Removes `cell`'s membership in `layer`, returning whether a membership
    /// was present. Drops the cell's entry entirely once its last membership is
    /// removed so an unassigned cell becomes unconditionally streamable again.
    pub fn unassign(&mut self, cell: CellCoord, layer: DataLayerId) -> bool {
        if let Some(layers) = self.cell_layers.get_mut(&cell)
            && let Ok(pos) = layers.binary_search(&layer)
        {
            layers.remove(pos);
            if layers.is_empty() {
                self.cell_layers.remove(&cell);
            }
            return true;
        }
        false
    }

    /// The sorted layer memberships of `cell` (empty if the cell carries none).
    #[inline]
    pub fn layers_of(&self, cell: CellCoord) -> &[DataLayerId] {
        self.cell_layers
            .get(&cell)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Whether `cell` is allowed to stream right now (design §13.1).
    ///
    /// `true` iff *every* layer the cell belongs to permits streaming
    /// (state ≥ [`Loaded`](DataLayerState::Loaded)). A single
    /// [`Unloaded`](DataLayerState::Unloaded) membership blocks the cell. A cell
    /// with no memberships is unconditionally streamable.
    pub fn is_cell_streamable(&self, cell: CellCoord) -> bool {
        match self.cell_layers.get(&cell) {
            None => true,
            Some(layers) => layers
                .iter()
                .all(|&id| self.state(id).permits_streaming()),
        }
    }

    /// Convenience: sets `layer` to [`DataLayerState::Loaded`], returning the
    /// previous state. Shorthand for [`set_state`](Self::set_state).
    #[inline]
    pub fn load(&mut self, layer: DataLayerId) -> DataLayerState {
        self.set_state(layer, DataLayerState::Loaded)
    }

    /// Convenience: sets `layer` to [`DataLayerState::Activated`], returning the
    /// previous state.
    #[inline]
    pub fn activate(&mut self, layer: DataLayerId) -> DataLayerState {
        self.set_state(layer, DataLayerState::Activated)
    }

    /// Convenience: sets `layer` to [`DataLayerState::Unloaded`], returning the
    /// previous state. Its cells become blocked from streaming on the next tick
    /// and any already-resident ones are evicted by the gated streamer.
    #[inline]
    pub fn unload(&mut self, layer: DataLayerId) -> DataLayerState {
        self.set_state(layer, DataLayerState::Unloaded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(x: i32, y: i32, z: i32) -> CellCoord {
        CellCoord::new(x, y, z)
    }

    #[test]
    fn state_default_is_unloaded_and_ordered() {
        assert_eq!(DataLayerState::default(), DataLayerState::Unloaded);
        assert!(DataLayerState::Unloaded < DataLayerState::Loaded);
        assert!(DataLayerState::Loaded < DataLayerState::Activated);
        assert!(!DataLayerState::Unloaded.permits_streaming());
        assert!(DataLayerState::Loaded.permits_streaming());
        assert!(DataLayerState::Activated.permits_streaming());
        assert!(DataLayerState::Activated.is_activated());
        assert!(!DataLayerState::Loaded.is_activated());
    }

    #[test]
    fn id_roundtrips() {
        let id = DataLayerId::new(42);
        assert_eq!(id.get(), 42);
        assert_eq!(DataLayerId::default(), DataLayerId::new(0));
    }

    #[test]
    fn register_then_set_state_reports_previous() {
        let mut dl = DataLayers::new();
        let a = DataLayerId::new(1);

        assert!(dl.register_layer(a));
        assert!(!dl.register_layer(a)); // second register is a no-op
        assert!(dl.is_registered(a));
        assert_eq!(dl.state(a), DataLayerState::Unloaded);

        assert_eq!(dl.set_state(a, DataLayerState::Loaded), DataLayerState::Unloaded);
        assert_eq!(dl.state(a), DataLayerState::Loaded);
        assert_eq!(dl.activate(a), DataLayerState::Loaded);
        assert_eq!(dl.state(a), DataLayerState::Activated);
        assert_eq!(dl.layer_count(), 1);
    }

    #[test]
    fn set_state_autoregisters_unknown_layer() {
        let mut dl = DataLayers::new();
        let a = DataLayerId::new(7);
        assert!(!dl.is_registered(a));
        // Toggling an unknown layer reports its prior effective state (Unloaded)
        // and registers it.
        assert_eq!(dl.set_state(a, DataLayerState::Loaded), DataLayerState::Unloaded);
        assert!(dl.is_registered(a));
    }

    #[test]
    fn assign_is_sorted_and_dedups() {
        let mut dl = DataLayers::new();
        let c = cell(1, 2, 3);
        assert!(dl.assign(c, DataLayerId::new(5)));
        assert!(dl.assign(c, DataLayerId::new(1)));
        assert!(dl.assign(c, DataLayerId::new(3)));
        assert!(!dl.assign(c, DataLayerId::new(3))); // redundant
        assert_eq!(
            dl.layers_of(c),
            &[DataLayerId::new(1), DataLayerId::new(3), DataLayerId::new(5)]
        );
    }

    #[test]
    fn unassign_removes_and_clears_empty_cell() {
        let mut dl = DataLayers::new();
        let c = cell(0, 0, 0);
        dl.assign(c, DataLayerId::new(1));
        dl.assign(c, DataLayerId::new(2));

        assert!(dl.unassign(c, DataLayerId::new(1)));
        assert!(!dl.unassign(c, DataLayerId::new(1))); // already gone
        assert_eq!(dl.layers_of(c), &[DataLayerId::new(2)]);

        assert!(dl.unassign(c, DataLayerId::new(2)));
        // Last membership gone: cell is unconditionally streamable again.
        assert!(dl.layers_of(c).is_empty());
        assert!(dl.is_cell_streamable(c));
    }

    #[test]
    fn cell_without_memberships_is_always_streamable() {
        let dl = DataLayers::new();
        assert!(dl.is_cell_streamable(cell(9, 9, 9)));
    }

    #[test]
    fn streamable_requires_all_layers_loaded() {
        let mut dl = DataLayers::new();
        let c = cell(4, 0, 0);
        let day = DataLayerId::new(1);
        let mission = DataLayerId::new(2);
        dl.assign(c, day);
        dl.assign(c, mission);

        // Both start Unloaded -> blocked.
        assert!(!dl.is_cell_streamable(c));

        // One loaded is still not enough.
        dl.load(day);
        assert!(!dl.is_cell_streamable(c));

        // Both >= Loaded -> streamable. Activated also counts.
        dl.activate(mission);
        assert!(dl.is_cell_streamable(c));

        // Unloading either one re-blocks the cell.
        dl.unload(day);
        assert!(!dl.is_cell_streamable(c));
    }
}
