//! A generational registry of per-view temporal history.
//!
//! Every camera, reflection probe, and shadow view that reuses results across
//! frames needs a stable identity for its history buffers plus a record of the
//! epochs it last integrated and any invalidation reasons that have piled up
//! since. [`ViewHistoryRegistry`] owns that bookkeeping on the `CPU`.
//!
//! Slots are recycled through a free list, and each recycle bumps the slot's
//! generation so a [`ViewHistoryId`](crate::history::ViewHistoryId) minted for a
//! released view never resolves against the view that later reuses the slot.
//! Resolving a view against the current [`HistoryEpochs`] folds the cached-epoch
//! diff into any pending invalidation, adopts the new epochs, and reports the
//! effective invalidation for the frame — all as deterministic integer work.

use crate::abi::GenerationalHandle;
use crate::history::epochs::HistoryEpochs;
use crate::history::invalidation::InvalidationMask;

/// A stable, generational identity for one view's temporal history.
pub type ViewHistoryId = GenerationalHandle;

/// The per-view state tracked by the registry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ViewHistoryState {
    /// The epochs this view last integrated.
    pub epochs: HistoryEpochs,
    /// Invalidation reasons accumulated but not yet resolved.
    pub pending: InvalidationMask,
    /// The frame index at which this view was created.
    pub created_frame: u64,
    /// The most recent frame index that touched this view.
    pub last_touched_frame: u64,
    /// Whether the view has been resolved at least once since creation.
    pub resolved_once: bool,
}

impl ViewHistoryState {
    fn new(epochs: HistoryEpochs, frame: u64) -> Self {
        Self {
            epochs,
            pending: InvalidationMask::EMPTY,
            created_frame: frame,
            last_touched_frame: frame,
            resolved_once: false,
        }
    }
}

/// The outcome of resolving a view against the current epochs for one frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedHistory {
    /// The invalidation reasons the consumer must honor this frame.
    pub invalidation: InvalidationMask,
    /// Whether this is the first resolve since the view was created, meaning
    /// there is no prior history at all.
    pub first_frame: bool,
    /// Whether the previous frame's result may be reprojected this frame.
    ///
    /// `false` when this is the first frame or when a hard-reset reason (camera
    /// cut, resolution change, shader-version change) is present.
    pub reprojectable: bool,
}

#[derive(Clone, Copy, Debug)]
struct Slot {
    generation: u32,
    occupied: bool,
    state: ViewHistoryState,
}

/// Tracks temporal history for a set of views, keyed by generational handle.
#[derive(Clone, Debug, Default)]
pub struct ViewHistoryRegistry {
    slots: Vec<Slot>,
    free: Vec<u32>,
    live: u32,
}

impl ViewHistoryRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of currently live views.
    #[must_use]
    pub fn live_count(&self) -> u32 {
        self.live
    }

    /// Returns `true` when no view is live.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// The number of allocated slots, live or free.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Registers a new view seeded with `epochs`, created at `frame`.
    ///
    /// Reuses a freed slot when one is available; otherwise appends a new slot.
    pub fn register(&mut self, epochs: HistoryEpochs, frame: u64) -> ViewHistoryId {
        let state = ViewHistoryState::new(epochs, frame);
        self.live += 1;
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.occupied = true;
            slot.state = state;
            ViewHistoryId::new(index, slot.generation)
        } else {
            let index = self.slots.len() as u32;
            self.slots.push(Slot {
                generation: 0,
                occupied: true,
                state,
            });
            ViewHistoryId::new(index, 0)
        }
    }

    /// Resolves the slot index for a live `id`, or `None` when the handle is
    /// stale, invalid, or out of range.
    fn live_index(&self, id: ViewHistoryId) -> Option<usize> {
        if !id.is_valid() {
            return None;
        }
        let index = id.index as usize;
        let slot = self.slots.get(index)?;
        if slot.occupied && slot.generation == id.generation {
            Some(index)
        } else {
            None
        }
    }

    /// Returns `true` when `id` refers to a currently live view.
    #[must_use]
    pub fn is_live(&self, id: ViewHistoryId) -> bool {
        self.live_index(id).is_some()
    }

    /// Borrows the state behind a live `id`.
    #[must_use]
    pub fn get(&self, id: ViewHistoryId) -> Option<&ViewHistoryState> {
        self.live_index(id).map(|i| &self.slots[i].state)
    }

    /// Mutably borrows the state behind a live `id`.
    pub fn get_mut(&mut self, id: ViewHistoryId) -> Option<&mut ViewHistoryState> {
        self.live_index(id).map(|i| &mut self.slots[i].state)
    }

    /// Releases a view, bumping its slot generation so stale handles fail.
    ///
    /// Returns `true` when the handle was live and is now released.
    pub fn release(&mut self, id: ViewHistoryId) -> bool {
        let Some(index) = self.live_index(id) else {
            return false;
        };
        let slot = &mut self.slots[index];
        slot.occupied = false;
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(index as u32);
        self.live -= 1;
        true
    }

    /// Records that `frame` touched the view without changing epochs.
    ///
    /// Returns `true` when the handle was live.
    pub fn touch(&mut self, id: ViewHistoryId, frame: u64) -> bool {
        match self.get_mut(id) {
            Some(state) => {
                state.last_touched_frame = frame;
                true
            }
            None => false,
        }
    }

    /// Accumulates external invalidation reasons for a live view.
    ///
    /// Returns `true` when the handle was live.
    pub fn invalidate(&mut self, id: ViewHistoryId, mask: InvalidationMask) -> bool {
        match self.get_mut(id) {
            Some(state) => {
                state.pending.insert(mask);
                true
            }
            None => false,
        }
    }

    /// Resolves a view for the current frame against `current_epochs`.
    ///
    /// Folds the cached-epoch diff into any pending invalidation, adopts the
    /// new epochs, clears the pending set, and stamps `frame`. The first resolve
    /// after creation reports a full reset because no prior history exists.
    /// Returns `None` when the handle is stale.
    pub fn resolve(
        &mut self,
        id: ViewHistoryId,
        current_epochs: HistoryEpochs,
        frame: u64,
    ) -> Option<ResolvedHistory> {
        let index = self.live_index(id)?;
        let slot = &mut self.slots[index];
        let state = &mut slot.state;

        let first_frame = !state.resolved_once;
        let mut invalidation = state.pending.union(state.epochs.diff(current_epochs));
        if first_frame {
            invalidation = InvalidationMask::ALL;
        }

        state.epochs = current_epochs;
        state.pending = InvalidationMask::EMPTY;
        state.last_touched_frame = frame;
        state.resolved_once = true;

        let reprojectable = !first_frame && !invalidation.forces_full_reset();
        Some(ResolvedHistory {
            invalidation,
            first_frame,
            reprojectable,
        })
    }

    /// Reports whether a view is stale: released, or untouched for longer than
    /// `max_age` frames as of `current_frame`.
    #[must_use]
    pub fn is_stale(&self, id: ViewHistoryId, current_frame: u64, max_age: u64) -> bool {
        match self.get(id) {
            Some(state) => current_frame.saturating_sub(state.last_touched_frame) > max_age,
            None => true,
        }
    }

    /// Releases every live view untouched for longer than `max_age` frames.
    ///
    /// Returns the handles that were reclaimed, in ascending slot order, so the
    /// caller can drop the matching `GPU` history buffers deterministically.
    pub fn prune_stale(&mut self, current_frame: u64, max_age: u64) -> Vec<ViewHistoryId> {
        let mut reclaimed = Vec::new();
        for index in 0..self.slots.len() {
            let slot = &self.slots[index];
            if !slot.occupied {
                continue;
            }
            if current_frame.saturating_sub(slot.state.last_touched_frame) > max_age {
                reclaimed.push(ViewHistoryId::new(index as u32, slot.generation));
            }
        }
        for &id in &reclaimed {
            self.release(id);
        }
        reclaimed
    }

    /// Collects the handles of every live view in ascending slot order.
    #[must_use]
    pub fn live_ids(&self) -> Vec<ViewHistoryId> {
        let mut out = Vec::new();
        for (index, slot) in self.slots.iter().enumerate() {
            if slot.occupied {
                out.push(ViewHistoryId::new(index as u32, slot.generation));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::epochs::EpochCategory;

    #[test]
    fn register_and_lookup_roundtrips() {
        let mut reg = ViewHistoryRegistry::new();
        assert!(reg.is_empty());
        let id = reg.register(HistoryEpochs::ZERO, 10);
        assert!(reg.is_live(id));
        assert_eq!(reg.live_count(), 1);
        let state = reg.get(id).unwrap();
        assert_eq!(state.created_frame, 10);
        assert_eq!(state.last_touched_frame, 10);
        assert!(!state.resolved_once);
    }

    #[test]
    fn invalid_handle_never_resolves() {
        let reg = ViewHistoryRegistry::new();
        assert!(!reg.is_live(ViewHistoryId::INVALID));
        assert!(reg.get(ViewHistoryId::INVALID).is_none());
    }

    #[test]
    fn released_slot_recycles_with_new_generation() {
        let mut reg = ViewHistoryRegistry::new();
        let a = reg.register(HistoryEpochs::ZERO, 0);
        assert!(reg.release(a));
        assert!(!reg.is_live(a));
        assert!(!reg.release(a), "double release must be rejected");

        let b = reg.register(HistoryEpochs::ZERO, 1);
        // Same slot index, but a fresh generation, so the stale handle `a`
        // still fails while `b` resolves.
        assert_eq!(b.index, a.index);
        assert_ne!(b.generation, a.generation);
        assert!(reg.is_live(b));
        assert!(!reg.is_live(a));
        assert_eq!(reg.capacity(), 1);
    }

    #[test]
    fn first_resolve_is_a_full_reset() {
        let mut reg = ViewHistoryRegistry::new();
        let id = reg.register(HistoryEpochs::ZERO, 0);
        let r = reg.resolve(id, HistoryEpochs::ZERO, 1).unwrap();
        assert!(r.first_frame);
        assert!(!r.reprojectable);
        assert_eq!(r.invalidation, InvalidationMask::ALL);
        assert!(reg.get(id).unwrap().resolved_once);
    }

    #[test]
    fn steady_state_resolve_is_reprojectable() {
        let mut reg = ViewHistoryRegistry::new();
        let id = reg.register(HistoryEpochs::ZERO, 0);
        let _ = reg.resolve(id, HistoryEpochs::ZERO, 1);
        let r = reg.resolve(id, HistoryEpochs::ZERO, 2).unwrap();
        assert!(!r.first_frame);
        assert!(r.reprojectable);
        assert!(r.invalidation.is_empty());
    }

    #[test]
    fn epoch_change_surfaces_as_invalidation() {
        let mut reg = ViewHistoryRegistry::new();
        let id = reg.register(HistoryEpochs::ZERO, 0);
        let _ = reg.resolve(id, HistoryEpochs::ZERO, 1);
        let next = HistoryEpochs::ZERO.bumped(EpochCategory::Lighting);
        let r = reg.resolve(id, next, 2).unwrap();
        assert_eq!(r.invalidation, InvalidationMask::LIGHTING);
        assert!(
            r.reprojectable,
            "a lighting change still reprojects geometry"
        );
        // Epochs were adopted, so the following frame is clean.
        let r2 = reg.resolve(id, next, 3).unwrap();
        assert!(r2.invalidation.is_empty());
    }

    #[test]
    fn external_invalidation_folds_into_resolve() {
        let mut reg = ViewHistoryRegistry::new();
        let id = reg.register(HistoryEpochs::ZERO, 0);
        let _ = reg.resolve(id, HistoryEpochs::ZERO, 1);
        assert!(reg.invalidate(id, InvalidationMask::CAMERA_CUT));
        let r = reg.resolve(id, HistoryEpochs::ZERO, 2).unwrap();
        assert!(r.invalidation.contains(InvalidationMask::CAMERA_CUT));
        assert!(!r.reprojectable);
        // Pending is consumed after resolve.
        let r2 = reg.resolve(id, HistoryEpochs::ZERO, 3).unwrap();
        assert!(r2.invalidation.is_empty());
    }

    #[test]
    fn staleness_and_pruning() {
        let mut reg = ViewHistoryRegistry::new();
        let fresh = reg.register(HistoryEpochs::ZERO, 100);
        let old = reg.register(HistoryEpochs::ZERO, 10);
        reg.touch(fresh, 120);

        assert!(!reg.is_stale(fresh, 121, 30));
        assert!(reg.is_stale(old, 121, 30));

        let reclaimed = reg.prune_stale(121, 30);
        assert_eq!(reclaimed, alloc::vec![old]);
        assert!(!reg.is_live(old));
        assert!(reg.is_live(fresh));
        assert_eq!(reg.live_count(), 1);
    }

    #[test]
    fn stale_handle_is_treated_as_stale() {
        let mut reg = ViewHistoryRegistry::new();
        let id = reg.register(HistoryEpochs::ZERO, 0);
        reg.release(id);
        assert!(reg.is_stale(id, 0, 1_000));
        assert!(reg.resolve(id, HistoryEpochs::ZERO, 1).is_none());
        assert!(!reg.invalidate(id, InvalidationMask::SCENE));
        assert!(!reg.touch(id, 5));
    }

    #[test]
    fn live_ids_are_sorted_by_slot() {
        let mut reg = ViewHistoryRegistry::new();
        let a = reg.register(HistoryEpochs::ZERO, 0);
        let b = reg.register(HistoryEpochs::ZERO, 0);
        let c = reg.register(HistoryEpochs::ZERO, 0);
        reg.release(b);
        let ids = reg.live_ids();
        assert_eq!(ids, alloc::vec![a, c]);
    }
}
