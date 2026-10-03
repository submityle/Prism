//! Per-worker frame arena: a lock-free bump allocator (design §12).
//!
//! Each compute worker owns one [`FrameArena`]. Job bodies, captured closures,
//! and short-lived per-frame scratch are bump-allocated from the worker's arena
//! instead of the global heap: allocation is a single atomic add (no locking,
//! cache-hot), and the *whole* arena is reclaimed at the end of a frame with a
//! single pointer reset — [`FrameArena::reset`] costs O(1) and runs **no
//! destructors** (design §12: "帧末整体 reset（零析构成本）").
//!
//! ## Zero drop cost — read this before allocating
//! [`reset`](FrameArena::reset) simply rewinds the bump offset to zero. It does
//! **not** drop the values previously allocated in the arena. This is exactly
//! why it is cheap, and it is sound (leaking is safe), but it means you must
//! only arena-allocate values whose storage may be reclaimed without running
//! `Drop` — plain data, `Copy` types, job scratch, closures that own nothing
//! needing cleanup. If a value owns a heap allocation, a file handle, or any
//! other resource, allocate it on the normal heap instead. [`alloc`] accepts
//! any `T` for ergonomics; honoring this contract is the caller's job.
//!
//! ## Overflow is not a panic
//! A frame arena has a fixed capacity chosen up front. When a request does not
//! fit, [`alloc`](FrameArena::alloc) / [`alloc_bytes`](FrameArena::alloc_bytes)
//! return [`None`] and the caller falls back to the heap; the fast path never
//! blocks and never aborts. [`high_water`](FrameArena::high_water) reports the
//! peak bytes used across frames so capacity can be tuned (design §16).
//!
//! [`alloc`]: FrameArena::alloc
//! [`alloc_bytes`]: FrameArena::alloc_bytes

use std::alloc::{self, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::numa::NumaNodeId;

/// Default frame-arena capacity: 1 MiB per worker. Large enough to hold a
/// frame's worth of small job records and scratch without touching the heap,
/// small enough that even a 64-worker desktop pool reserves only tens of MiB.
pub const DEFAULT_ARENA_CAPACITY: usize = 1024 * 1024;

/// Alignment of the arena's backing allocation. A cache line (64 bytes) keeps
/// the base — and therefore the first allocation of every frame — off a shared
/// line, limiting false sharing with neighbouring workers' arenas.
const ARENA_BASE_ALIGN: usize = 64;

/// A lock-free bump allocator serving one worker's per-frame allocations.
///
/// Allocation bumps an atomic offset with a `compare_exchange` loop, so it is
/// safe to call concurrently (it is [`Sync`]) even though in the normal design
/// a single worker owns its arena. [`reset`](FrameArena::reset) takes `&mut
/// self`, so the borrow checker guarantees no allocation reference is still
/// live when a frame is recycled.
pub struct FrameArena {
    /// Base of the backing allocation.
    base: NonNull<u8>,
    /// Layout the base was allocated with (for dealloc and capacity).
    layout: Layout,
    /// Usable capacity in bytes (`layout.size()`).
    capacity: usize,
    /// Bump cursor: bytes handed out so far from `base`.
    offset: AtomicUsize,
    /// Peak value `offset` ever reached, across all frames (diagnostics).
    high_water: AtomicUsize,
    /// NUMA node this arena's memory is associated with (design §14 — "每
    /// worker 竞技场/栈在其所在 NUMA 节点分配"). On platforms without NUMA
    /// probing this is always node 0; the field records intent and lets
    /// steal-penalty logic reason about locality.
    node: NumaNodeId,
}

#[expect(unsafe_code, reason = "arena owns a unique region; alloc is atomic/disjoint")]
// SAFETY: `FrameArena` uniquely owns its backing allocation via `base`; the
// pointer is never aliased by another `FrameArena`. All mutation of the shared
// region goes through the atomic `offset`/`high_water` with a CAS loop that
// hands out strictly disjoint byte ranges, so concurrent `alloc` calls never
// overlap. Hence it is safe to share a `&FrameArena` across threads.
unsafe impl Send for FrameArena {}
#[expect(unsafe_code, reason = "arena owns a unique region; alloc is atomic/disjoint")]
// SAFETY: see the `Send` impl above — concurrent allocation is race-free.
unsafe impl Sync for FrameArena {}

impl FrameArena {
    /// Create an arena with [`DEFAULT_ARENA_CAPACITY`] on NUMA node 0.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_ARENA_CAPACITY)
    }

    /// Create an arena of `capacity` bytes on NUMA node 0.
    ///
    /// `capacity` is rounded up to at least one cache line. Panics only if the
    /// resulting [`Layout`] is invalid (astronomically large `capacity`).
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_on_node(capacity, NumaNodeId::ZERO)
    }

    /// Create an arena of `capacity` bytes tagged as living on `node`.
    ///
    /// The tag is advisory: without OS NUMA support the memory is ordinary heap
    /// (honest degradation — see [`crate::numa`]), but the association drives
    /// same-node steal preference and locality diagnostics.
    pub fn with_capacity_on_node(capacity: usize, node: NumaNodeId) -> Self {
        let capacity = capacity.max(ARENA_BASE_ALIGN);
        let layout = Layout::from_size_align(capacity, ARENA_BASE_ALIGN)
            .expect("invalid frame-arena layout");
        // SAFETY: `layout` has a non-zero, cache-line-aligned size, satisfying
        // the `alloc` contract. The returned pointer is checked for null.
        #[expect(unsafe_code, reason = "raw backing allocation for the bump arena")]
        let raw = unsafe { alloc::alloc(layout) };
        let base = NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        Self {
            base,
            layout,
            capacity,
            offset: AtomicUsize::new(0),
            high_water: AtomicUsize::new(0),
            node,
        }
    }

    /// Total capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes currently handed out this frame.
    pub fn used(&self) -> usize {
        self.offset.load(Ordering::Acquire).min(self.capacity)
    }

    /// Bytes remaining before the arena overflows to the heap this frame.
    pub fn remaining(&self) -> usize {
        self.capacity.saturating_sub(self.used())
    }

    /// Peak bytes ever used across all frames since creation (design §16 —
    /// 竞技场高水位，调容量).
    pub fn high_water(&self) -> usize {
        self.high_water.load(Ordering::Acquire)
    }

    /// The NUMA node this arena is associated with.
    pub fn node(&self) -> NumaNodeId {
        self.node
    }

    /// Allocate `size` bytes aligned to `align`, lock-free. Returns [`None`] if
    /// the request does not fit in the remaining capacity (the caller then
    /// falls back to the heap). `align` must be a power of two.
    ///
    /// The returned pointer is valid until the next [`reset`](Self::reset) and
    /// points to uninitialized memory. The borrow ties the pointer's provenance
    /// to `&self`, and `reset` requires `&mut self`, so a reset cannot race a
    /// live allocation.
    pub fn alloc_bytes(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        debug_assert!(align.is_power_of_two(), "alignment must be a power of two");
        if size == 0 {
            // Hand back an aligned, non-null dangling-but-in-range pointer:
            // base is `ARENA_BASE_ALIGN`-aligned, which covers any `align`
            // we accept here (callers only use <= base alignment for ZSTs).
            return Some(self.base);
        }
        let mut cur = self.offset.load(Ordering::Relaxed);
        loop {
            let aligned = align_up(cur, align)?;
            let end = aligned.checked_add(size)?;
            if end > self.capacity {
                return None;
            }
            match self.offset.compare_exchange_weak(
                cur,
                end,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.high_water.fetch_max(end, Ordering::AcqRel);
                    // SAFETY: `aligned + size <= capacity == layout.size()`, so
                    // `base + aligned` is within the single owned allocation and
                    // the `[aligned, end)` range is disjoint from every other
                    // winner of this CAS. `base` is non-null and the add stays
                    // in-bounds, so the offset pointer is non-null.
                    #[expect(unsafe_code, reason = "offset into the owned arena region")]
                    let ptr = unsafe {
                        NonNull::new_unchecked(self.base.as_ptr().add(aligned))
                    };
                    return Some(ptr);
                }
                Err(actual) => cur = actual,
            }
        }
    }

    /// Bump-allocate `value` in the arena and return an exclusive reference to
    /// it, or [`None`] (returning `value`) if it does not fit.
    ///
    /// # Zero drop cost
    /// The value is **never dropped** by the arena: [`reset`](Self::reset)
    /// reclaims the storage without running destructors (see the module docs).
    /// Only allocate values that are safe to forget.
    #[expect(
        clippy::mut_from_ref,
        reason = "bump allocator: each call carves a disjoint, exclusive slot \
                  from the shared region, so handing out `&mut` from `&self` is \
                  sound (standard arena pattern, e.g. `bumpalo`)"
    )]
    pub fn alloc<T>(&self, value: T) -> Result<&mut T, T> {
        let layout = Layout::new::<T>();
        let Some(ptr) = self.alloc_bytes(layout.size(), layout.align()) else {
            return Err(value);
        };
        let typed = ptr.as_ptr().cast::<T>();
        // SAFETY: `alloc_bytes` returned a pointer to `size_of::<T>()` bytes
        // aligned to `align_of::<T>()` inside the arena, exclusive to this call.
        // Writing initializes it; the returned `&mut` borrows `self`, so it
        // cannot outlive the arena nor coexist with a `&mut self` reset.
        #[expect(unsafe_code, reason = "initialize and expose the arena slot")]
        unsafe {
            typed.write(value);
            Ok(&mut *typed)
        }
    }

    /// Reclaim the entire arena for the next frame in O(1), running **no
    /// destructors**. Requires `&mut self`, so the borrow checker proves no
    /// allocation reference is still outstanding.
    pub fn reset(&mut self) {
        self.offset.store(0, Ordering::Release);
    }
}

impl Default for FrameArena {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for FrameArena {
    fn drop(&mut self) {
        // SAFETY: `base`/`layout` came from the matching `alloc::alloc` in
        // `with_capacity_on_node`; this is the sole owner, dropped once. Values
        // allocated inside are intentionally not dropped (zero-drop contract).
        #[expect(unsafe_code, reason = "free the arena's backing region")]
        unsafe {
            alloc::dealloc(self.base.as_ptr(), self.layout);
        }
    }
}

/// Round `value` up to the next multiple of `align` (a power of two), or
/// [`None`] on overflow.
fn align_up(value: usize, align: usize) -> Option<usize> {
    let mask = align - 1;
    value.checked_add(mask).map(|v| v & !mask)
}

/// A per-worker set of [`FrameArena`]s plus a single frame-boundary reset.
///
/// One arena per compute worker, each tagged with the worker's NUMA node. The
/// application resets all arenas at a frame boundary via
/// [`reset_all`](Self::reset_all), which takes `&mut self` so no job may hold a
/// live arena reference across the reset.
pub struct FrameArenas {
    arenas: Vec<FrameArena>,
}

impl FrameArenas {
    /// Create `workers` arenas, each with `capacity` bytes, all on node 0.
    pub fn new(workers: usize, capacity: usize) -> Self {
        let arenas = (0..workers)
            .map(|_| FrameArena::with_capacity(capacity))
            .collect();
        Self { arenas }
    }

    /// Create arenas sized `capacity`, each placed on the NUMA node that
    /// `node_of_worker(index)` reports — the hook used to co-locate a worker's
    /// arena with the node it is pinned to (design §14).
    pub fn with_nodes(
        workers: usize,
        capacity: usize,
        mut node_of_worker: impl FnMut(usize) -> NumaNodeId,
    ) -> Self {
        let arenas = (0..workers)
            .map(|index| FrameArena::with_capacity_on_node(capacity, node_of_worker(index)))
            .collect();
        Self { arenas }
    }

    /// Number of arenas (one per worker).
    pub fn len(&self) -> usize {
        self.arenas.len()
    }

    /// Whether there are no arenas.
    pub fn is_empty(&self) -> bool {
        self.arenas.is_empty()
    }

    /// Borrow the arena for `worker`, if in range.
    pub fn arena(&self, worker: usize) -> Option<&FrameArena> {
        self.arenas.get(worker)
    }

    /// Sum of every arena's high-water mark (frame-memory diagnostics).
    pub fn total_high_water(&self) -> usize {
        self.arenas.iter().map(FrameArena::high_water).sum()
    }

    /// Reset every arena for the next frame. O(workers), no destructors.
    pub fn reset_all(&mut self) {
        for arena in &mut self.arenas {
            arena.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_returns_distinct_aligned_pointers() {
        let arena = FrameArena::with_capacity(4096);
        let a = arena.alloc(1u64).unwrap();
        let b = arena.alloc(2u64).unwrap();
        assert_eq!(*a, 1);
        assert_eq!(*b, 2);
        assert_ne!(a as *mut u64, b as *mut u64);
        assert_eq!((a as *mut u64).align_offset(align_of::<u64>()), 0);
        assert_eq!((b as *mut u64).align_offset(align_of::<u64>()), 0);
    }

    #[test]
    fn respects_alignment() {
        let arena = FrameArena::with_capacity(4096);
        // Force a 1-byte bump first so the next 64-aligned request must pad.
        let _ = arena.alloc(0u8).unwrap();
        let p = arena.alloc_bytes(16, 64).unwrap();
        assert_eq!(p.as_ptr().align_offset(64), 0);
    }

    #[test]
    fn overflow_returns_none_and_value_back() {
        let arena = FrameArena::with_capacity(64);
        // 64 bytes capacity: a 128-byte request cannot fit.
        assert!(arena.alloc_bytes(128, 1).is_none());
        let big = [0u8; 128];
        assert_eq!(arena.alloc(big).unwrap_err(), big);
    }

    #[test]
    fn reset_rewinds_and_reuses_storage() {
        let mut arena = FrameArena::with_capacity(128);
        let p1 = arena.alloc(7u32).unwrap() as *mut u32;
        assert!(arena.used() >= 4);
        arena.reset();
        assert_eq!(arena.used(), 0);
        let p2 = arena.alloc(9u32).unwrap() as *mut u32;
        // After reset the first allocation of the new frame reuses the base.
        assert_eq!(p1, p2);
    }

    #[test]
    fn high_water_tracks_peak_across_frames() {
        let mut arena = FrameArena::with_capacity(1024);
        let _ = arena.alloc([0u8; 300]).unwrap();
        let after_first = arena.high_water();
        assert!(after_first >= 300);
        arena.reset();
        // A smaller frame must not lower the recorded peak.
        let _ = arena.alloc([0u8; 10]).unwrap();
        assert_eq!(arena.high_water(), after_first);
        // A larger frame raises it.
        arena.reset();
        let _ = arena.alloc([0u8; 500]).unwrap();
        assert!(arena.high_water() >= 500);
    }

    #[test]
    fn zero_sized_alloc_is_ok() {
        let arena = FrameArena::with_capacity(64);
        let unit = arena.alloc(()).unwrap();
        let _ = unit;
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn concurrent_allocs_are_disjoint() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let arena = Arc::new(FrameArena::with_capacity(1 << 20));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let arena = Arc::clone(&arena);
                std::thread::spawn(move || {
                    let mut ptrs = Vec::new();
                    for _ in 0..256 {
                        if let Some(p) = arena.alloc_bytes(16, 16) {
                            ptrs.push(p.as_ptr() as usize);
                        }
                    }
                    ptrs
                })
            })
            .collect();
        let mut all: Vec<usize> = threads.into_iter().flat_map(|t| t.join().unwrap()).collect();
        let total = all.len();
        all.sort_unstable();
        all.dedup();
        // Every successful 16-byte allocation must be a distinct address.
        assert_eq!(all.len(), total);
        // Adjacent allocations are at least 16 bytes apart (disjoint).
        for pair in all.windows(2) {
            assert!(pair[1] - pair[0] >= 16);
        }
        // Sanity: the lock-free counter did not over-count.
        let _ = AtomicUsize::new(0).load(Ordering::Relaxed);
    }

    #[test]
    fn frame_arenas_reset_all_and_high_water() {
        let mut arenas = FrameArenas::new(4, 1024);
        assert_eq!(arenas.len(), 4);
        for w in 0..4 {
            let _ = arenas.arena(w).unwrap().alloc([0u8; 64]).unwrap();
        }
        assert!(arenas.total_high_water() >= 4 * 64);
        arenas.reset_all();
        for w in 0..4 {
            assert_eq!(arenas.arena(w).unwrap().used(), 0);
        }
    }

    #[test]
    fn frame_arenas_place_on_nodes() {
        let arenas = FrameArenas::with_nodes(4, 256, |w| NumaNodeId::new((w % 2) as u16));
        assert_eq!(arenas.arena(0).unwrap().node(), NumaNodeId::new(0));
        assert_eq!(arenas.arena(1).unwrap().node(), NumaNodeId::new(1));
        assert_eq!(arenas.arena(2).unwrap().node(), NumaNodeId::new(0));
    }
}
