//! Address-independent deterministic ID allocator (§24.7).
//!
//! A classic generational slot allocator (slotmap / handle pool) hands out
//! `(index, generation)` handles and recycles freed slots. The usual
//! implementation keeps a **LIFO stack** of free indices, so the next index it
//! reuses depends on the *order* in which slots were freed — and that order is,
//! in a parallel or record/replay engine, nondeterministic (it depends on which
//! job finished first, which in turn depends on timing and memory addresses).
//! Two runs that free the *same set* of slots in a different order therefore
//! diverge, which shows up as a replay desync.
//!
//! [`DeterministicIdAllocator`] fixes this by recycling the **lowest free index
//! first** via a min-ordered free pool. The index returned by the next
//! [`alloc`](DeterministicIdAllocator::alloc) is then a pure function of the
//! *set* of currently-free indices — never of the order in which they were
//! freed, never of a pointer address, never of a clock or RNG. Combined with
//! per-slot generation counters (which depend only on how many times a given
//! slot has been freed), the entire stream of handles produced by a sequence of
//! `alloc`/`free` calls is bit-identical across runs, builds, and target
//! architectures, as long as the `alloc` calls happen in the same order. Frees
//! may be reordered freely.
//!
//! The handles are plain integers, so they are **address-independent**: they
//! survive serialization, relocation (see [`reloc`](crate::reloc)), and being
//! sent across a network, which is exactly what cross-machine state
//! reproduction for desync debugging needs.
//!
//! Everything here is **pure safe code** — no `unsafe`, no clock, no RNG, no
//! address dependence — and uses only `core`/`alloc`.
//!
//! ## Honest boundary (design doc §24.8)
//! - Determinism holds across *free* reordering, not across *alloc* reordering:
//!   `alloc` is inherently stateful (it consumes the lowest free index), so two
//!   runs must issue their `alloc` calls in the same logical order to agree.
//!   The determinism win is that the *free* side — the part a parallel engine
//!   cannot order — no longer affects the result.
//! - Generations wrap on overflow, skipping `0`; a slot freed `2^32 - 1` times
//!   can alias an old handle. That matches every production slotmap and is far
//!   outside normal lifetimes.
//! - This allocates integer handles, not memory. It is the deterministic
//!   analogue of a slot/arena index allocator, not a replacement for the real
//!   bump/scope allocators in [`alloc_`](crate::alloc_).

#![forbid(unsafe_code)]

extern crate alloc;

use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::Reverse;

/// A stable, address-independent handle produced by
/// [`DeterministicIdAllocator`].
///
/// It pairs a dense slot `index` with the `generation` that was live when the
/// handle was minted. A handle is only valid while the allocator's current
/// generation for that slot matches; once the slot is freed (and its generation
/// bumped) the old handle becomes stale and every query rejects it, catching
/// use-after-free / dangling-handle bugs.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DetId {
    index: u32,
    generation: u32,
}

impl DetId {
    /// The dense slot index this handle refers to.
    #[inline]
    #[must_use]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// The generation stamped into this handle when it was minted.
    #[inline]
    #[must_use]
    pub const fn generation(self) -> u32 {
        self.generation
    }

    /// Pack the handle into a single `u64` (`generation << 32 | index`) for
    /// compact, address-independent storage or wire transfer. Round-trips with
    /// [`from_bits`](DetId::from_bits).
    #[inline]
    #[must_use]
    pub const fn to_bits(self) -> u64 {
        ((self.generation as u64) << 32) | self.index as u64
    }

    /// Reconstruct a handle from its [`to_bits`](DetId::to_bits) encoding.
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u64) -> Self {
        Self {
            index: bits as u32,
            generation: (bits >> 32) as u32,
        }
    }
}

/// Why a [`free`](DeterministicIdAllocator::free) call was rejected.
///
/// The three variants distinguish the classic handle-lifetime bugs so a caller
/// (or [`prism_diagnostic`](crate::det)) can attribute them precisely.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DetIdError {
    /// The handle's index is outside the allocator's slot range — it was never
    /// minted by this allocator (or came from a different one).
    Dangling,
    /// The slot exists but is already free: this is a double-free of a handle
    /// whose slot has not yet been recycled.
    DoubleFree,
    /// The slot is live but under a newer generation: the handle is stale
    /// (use-after-free of a slot that has since been reallocated).
    Stale,
}

/// A deterministic, address-independent generational ID allocator.
///
/// Reuses the lowest free slot index first, so the handle stream is independent
/// of free ordering. See the [module docs](self) for the full contract.
#[derive(Clone, Debug)]
pub struct DeterministicIdAllocator {
    /// Current generation per slot; the low bit of liveness is tracked
    /// separately in `live` so a generation is never spent to encode state.
    generations: Vec<u32>,
    /// Whether each slot is currently handed out.
    live: Vec<bool>,
    /// Min-ordered pool of free slot indices (lowest reused first).
    free: BinaryHeap<Reverse<u32>>,
    /// Count of currently-live slots.
    live_count: usize,
}

impl DeterministicIdAllocator {
    /// Create an empty allocator.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            generations: Vec::new(),
            live: Vec::new(),
            free: BinaryHeap::new(),
            live_count: 0,
        }
    }

    /// Create an empty allocator with room for `cap` slots preallocated.
    #[inline]
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            generations: Vec::with_capacity(cap),
            live: Vec::with_capacity(cap),
            free: BinaryHeap::with_capacity(cap),
            live_count: 0,
        }
    }

    /// Allocate a fresh handle, recycling the lowest free slot if any.
    ///
    /// # Panics
    /// Panics if the slot count would exceed `u32::MAX`.
    #[inline]
    pub fn alloc(&mut self) -> DetId {
        self.live_count += 1;
        if let Some(Reverse(index)) = self.free.pop() {
            let i = index as usize;
            self.live[i] = true;
            return DetId {
                index,
                generation: self.generations[i],
            };
        }
        let index = u32::try_from(self.generations.len())
            .expect("DeterministicIdAllocator: slot count exceeds u32::MAX");
        // Generation starts at 1 so that a zeroed/default handle (generation 0)
        // never matches a live slot.
        self.generations.push(1);
        self.live.push(true);
        DetId {
            index,
            generation: 1,
        }
    }

    /// Free a previously allocated handle.
    ///
    /// Returns [`DetIdError`] if the handle is dangling, already freed, or stale
    /// (use-after-free), leaving the allocator unchanged in every error case.
    #[inline]
    pub fn free(&mut self, id: DetId) -> Result<(), DetIdError> {
        let i = id.index as usize;
        if i >= self.generations.len() {
            return Err(DetIdError::Dangling);
        }
        if !self.live[i] {
            return Err(DetIdError::DoubleFree);
        }
        if self.generations[i] != id.generation {
            return Err(DetIdError::Stale);
        }
        self.live[i] = false;
        // Bump generation (skip 0) so every outstanding handle to this slot is
        // now stale. This depends only on how many times the slot was freed —
        // not on timing — preserving determinism.
        let next = self.generations[i].wrapping_add(1);
        self.generations[i] = if next == 0 { 1 } else { next };
        self.free.push(Reverse(id.index));
        self.live_count -= 1;
        Ok(())
    }

    /// Whether `id` is currently live (matching index, in range, right
    /// generation).
    #[inline]
    #[must_use]
    pub fn is_live(&self, id: DetId) -> bool {
        let i = id.index as usize;
        i < self.generations.len() && self.live[i] && self.generations[i] == id.generation
    }

    /// The number of currently-live handles.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.live_count
    }

    /// Whether no handle is currently live.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.live_count == 0
    }

    /// The number of slots ever created (live + free). Monotonic until
    /// [`clear`](DeterministicIdAllocator::clear).
    #[inline]
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.generations.len()
    }

    /// Iterate the live handles in ascending slot-index order — a deterministic
    /// traversal independent of allocation/free history.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = DetId> + '_ {
        self.live
            .iter()
            .enumerate()
            .filter(|&(_, &alive)| alive)
            .map(|(i, _)| DetId {
                index: i as u32,
                generation: self.generations[i],
            })
    }

    /// Reset to the empty state, dropping all slots and the free pool.
    ///
    /// Generations are not preserved, so handles minted before a `clear` must
    /// not be reused afterwards (same as recreating the allocator).
    #[inline]
    pub fn clear(&mut self) {
        self.generations.clear();
        self.live.clear();
        self.free.clear();
        self.live_count = 0;
    }
}

impl Default for DeterministicIdAllocator {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
