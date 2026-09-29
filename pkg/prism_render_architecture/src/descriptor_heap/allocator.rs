//! Generational slot allocator with free-list recycling.
//!
//! A bindless descriptor segment owns a fixed range of physical slots. This
//! allocator hands those slots out as stable [`GenerationalHandle`]s: each slot
//! carries a monotonically increasing generation, so a handle stays valid only
//! while its generation matches the slot's current one. Recycling a slot bumps
//! its generation, which permanently invalidates every handle that referred to
//! the previous occupant — the classic stale-handle guard for bindless indices.
//!
//! Retirement is two-phase to stay safe against in-flight GPU work. Retiring a
//! slot bumps its generation immediately (so the `CPU` can no longer resolve the
//! old handle) but does *not* return the index to the free-list. A separate
//! reclaim step, driven by the caller's frame pacing, returns the index once the
//! GPU can no longer be reading it. Between those two events the slot is
//! *retiring*: unresolvable, yet not reusable.
//!
//! The allocator is `GPU`-independent and deterministic. The free-list is a LIFO
//! stack of `u32` indices, so a fixed request sequence always reproduces the
//! same slot assignments, which keeps golden tests and cross-run captures
//! stable. Actual descriptor writes are pending the GPU backend; this layer only
//! tracks index ownership.

use alloc::vec;
use alloc::vec::Vec;

use crate::abi::GenerationalHandle;

/// Lifecycle state of a single slot inside a segment.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum SlotState {
    /// Available for allocation (in the free-list or never yet handed out).
    Free,
    /// Currently owned by a live handle.
    Live,
    /// Retired but not yet reclaimed; the GPU may still be reading it.
    Retiring,
}

/// Why an [`GenerationalAllocator::allocate`] request could not be satisfied.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AllocError {
    /// Every slot in the segment is live or retiring.
    Exhausted,
}

/// Fixed-capacity generational slot allocator for one descriptor segment.
///
/// Slots are numbered `0..capacity` and mapped onto physical descriptor indices
/// by the owning heap. The allocator tracks each slot's generation and
/// lifecycle state and never panics on overflow: a full segment yields
/// [`AllocError::Exhausted`].
#[derive(Clone, Debug)]
pub struct GenerationalAllocator {
    /// Current generation for each slot; bumped on retirement.
    generations: Vec<u32>,
    /// Lifecycle state for each slot.
    states: Vec<SlotState>,
    /// LIFO free-list of reclaimed slot indices.
    free: Vec<u32>,
    /// Highest slot index not yet handed out for the first time.
    high_water: u32,
    /// Total number of slots in the segment.
    capacity: u32,
    /// Count of slots currently in [`SlotState::Live`].
    live_count: u32,
    /// Count of slots currently in [`SlotState::Retiring`].
    retiring_count: u32,
}

impl GenerationalAllocator {
    /// Creates an allocator that owns `capacity` slots, all initially free.
    #[must_use]
    pub fn new(capacity: u32) -> Self {
        let len = capacity as usize;
        Self {
            generations: vec![0u32; len],
            states: vec![SlotState::Free; len],
            free: Vec::new(),
            high_water: 0,
            capacity,
            live_count: 0,
            retiring_count: 0,
        }
    }

    /// Total number of slots the segment can hold.
    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Number of slots currently owned by a live handle.
    #[must_use]
    pub const fn live_count(&self) -> u32 {
        self.live_count
    }

    /// Number of slots retired but not yet reclaimed.
    #[must_use]
    pub const fn retiring_count(&self) -> u32 {
        self.retiring_count
    }

    /// Number of slots immediately available for allocation.
    #[must_use]
    pub const fn free_count(&self) -> u32 {
        self.capacity - self.live_count - self.retiring_count
    }

    /// Allocates the next slot, returning a handle bound to its generation.
    ///
    /// Reclaimed slots are reused before fresh ones (LIFO), keeping the mapping
    /// deterministic. Returns [`AllocError::Exhausted`] when no slot is free.
    pub fn allocate(&mut self) -> Result<GenerationalHandle, AllocError> {
        let index = if let Some(index) = self.free.pop() {
            index
        } else if self.high_water < self.capacity {
            let index = self.high_water;
            self.high_water += 1;
            index
        } else {
            return Err(AllocError::Exhausted);
        };
        let slot = index as usize;
        self.states[slot] = SlotState::Live;
        self.live_count += 1;
        Ok(GenerationalHandle {
            index,
            generation: self.generations[slot],
        })
    }

    /// Returns `true` when `handle` names the current live occupant of its slot.
    #[must_use]
    pub fn is_live(&self, handle: GenerationalHandle) -> bool {
        let Some(slot) = self.slot_of(handle) else {
            return false;
        };
        self.states[slot] == SlotState::Live
    }

    /// Resolves a live handle to its slot index, or `None` if stale.
    ///
    /// A handle is stale once its slot has been retired (generation bumped) or
    /// reclaimed and reused, so this is the sole gate the heap consults before
    /// forming a physical descriptor index.
    #[must_use]
    pub fn resolve(&self, handle: GenerationalHandle) -> Option<u32> {
        let slot = self.slot_of(handle)?;
        (self.states[slot] == SlotState::Live).then_some(handle.index)
    }

    /// Retires a live handle, bumping its slot generation immediately.
    ///
    /// The slot moves to *retiring*: the old handle can no longer resolve, but
    /// the index is withheld from the free-list until [`Self::reclaim_slot`] so
    /// the GPU can finish reading it. Returns `true` if the handle was live;
    /// retiring an already stale or unknown handle is a no-op returning `false`.
    pub fn retire(&mut self, handle: GenerationalHandle) -> bool {
        let Some(slot) = self.slot_of(handle) else {
            return false;
        };
        if self.states[slot] != SlotState::Live {
            return false;
        }
        self.states[slot] = SlotState::Retiring;
        self.generations[slot] = self.generations[slot].wrapping_add(1);
        self.live_count -= 1;
        self.retiring_count += 1;
        true
    }

    /// Returns a retired slot to the free-list, making it reusable.
    ///
    /// Called by the deferred-retirement machinery once the GPU can no longer be
    /// reading `index`. The generation was already advanced at retirement, so a
    /// subsequent allocation of this slot yields a fresh, distinct handle.
    /// Returns `true` when the slot was retiring; any other state is ignored.
    pub fn reclaim_slot(&mut self, index: u32) -> bool {
        if index >= self.capacity {
            return false;
        }
        let slot = index as usize;
        if self.states[slot] != SlotState::Retiring {
            return false;
        }
        self.states[slot] = SlotState::Free;
        self.retiring_count -= 1;
        self.free.push(index);
        true
    }

    /// Maps a handle to its slot index when the index is in range and the
    /// generation matches the slot's current generation.
    fn slot_of(&self, handle: GenerationalHandle) -> Option<usize> {
        if handle.index >= self.capacity {
            return None;
        }
        let slot = handle.index as usize;
        (self.generations[slot] == handle.generation).then_some(slot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_sequential_slots_then_reports_usage() {
        let mut alloc = GenerationalAllocator::new(3);
        assert_eq!(alloc.capacity(), 3);
        assert_eq!(alloc.free_count(), 3);
        let a = alloc.allocate().unwrap();
        let b = alloc.allocate().unwrap();
        assert_eq!(a.index, 0);
        assert_eq!(b.index, 1);
        assert_eq!(a.generation, 0);
        assert_eq!(alloc.live_count(), 2);
        assert_eq!(alloc.free_count(), 1);
    }

    #[test]
    fn exhaustion_returns_error_without_panicking() {
        let mut alloc = GenerationalAllocator::new(1);
        assert!(alloc.allocate().is_ok());
        assert_eq!(alloc.allocate(), Err(AllocError::Exhausted));
    }

    #[test]
    fn resolve_tracks_live_state() {
        let mut alloc = GenerationalAllocator::new(2);
        let handle = alloc.allocate().unwrap();
        assert_eq!(alloc.resolve(handle), Some(0));
        assert!(alloc.is_live(handle));
    }

    #[test]
    fn retire_invalidates_handle_immediately_but_withholds_slot() {
        let mut alloc = GenerationalAllocator::new(2);
        let handle = alloc.allocate().unwrap();
        assert!(alloc.retire(handle));
        // Old handle is stale the instant it retires.
        assert_eq!(alloc.resolve(handle), None);
        assert!(!alloc.is_live(handle));
        // Slot is retiring, not free: it cannot be reallocated yet.
        assert_eq!(alloc.retiring_count(), 1);
        assert_eq!(alloc.free_count(), 1);
        // A fresh allocation must take the untouched high-water slot, not the
        // retiring one.
        let next = alloc.allocate().unwrap();
        assert_eq!(next.index, 1);
    }

    #[test]
    fn double_retire_is_a_noop() {
        let mut alloc = GenerationalAllocator::new(1);
        let handle = alloc.allocate().unwrap();
        assert!(alloc.retire(handle));
        assert!(!alloc.retire(handle));
        assert_eq!(alloc.retiring_count(), 1);
    }

    #[test]
    fn reclaim_recycles_slot_with_bumped_generation() {
        let mut alloc = GenerationalAllocator::new(1);
        let first = alloc.allocate().unwrap();
        assert!(alloc.retire(first));
        assert!(alloc.reclaim_slot(first.index));
        assert_eq!(alloc.retiring_count(), 0);
        assert_eq!(alloc.free_count(), 1);
        let second = alloc.allocate().unwrap();
        // Same physical slot, new generation, so the old handle stays dead.
        assert_eq!(second.index, first.index);
        assert_eq!(second.generation, first.generation + 1);
        assert_eq!(alloc.resolve(first), None);
        assert_eq!(alloc.resolve(second), Some(0));
    }

    #[test]
    fn reclaim_ignores_non_retiring_slots() {
        let mut alloc = GenerationalAllocator::new(2);
        // Never allocated: still free, cannot be reclaimed.
        assert!(!alloc.reclaim_slot(0));
        // Out of range.
        assert!(!alloc.reclaim_slot(99));
        let live = alloc.allocate().unwrap();
        // Live, not retiring: reclaim is a no-op.
        assert!(!alloc.reclaim_slot(live.index));
        assert!(alloc.is_live(live));
    }

    #[test]
    fn stale_generation_never_resolves() {
        let mut alloc = GenerationalAllocator::new(1);
        let stale = GenerationalHandle {
            index: 0,
            generation: 7,
        };
        assert_eq!(alloc.resolve(stale), None);
        let live = alloc.allocate().unwrap();
        assert_ne!(live.generation, 7);
        assert_eq!(alloc.resolve(stale), None);
    }

    #[test]
    fn free_list_reuse_is_lifo_deterministic() {
        let mut alloc = GenerationalAllocator::new(3);
        let a = alloc.allocate().unwrap();
        let b = alloc.allocate().unwrap();
        let c = alloc.allocate().unwrap();
        for h in [a, b, c] {
            assert!(alloc.retire(h));
            assert!(alloc.reclaim_slot(h.index));
        }
        // Reclaimed in order 0,1,2 => LIFO free-list pops 2,1,0.
        assert_eq!(alloc.allocate().unwrap().index, 2);
        assert_eq!(alloc.allocate().unwrap().index, 1);
        assert_eq!(alloc.allocate().unwrap().index, 0);
    }
}
