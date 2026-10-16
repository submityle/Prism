//! Chunk-block arena (design §17: 大页竞技场分配 / NUMA 就近分配).
//!
//! Every chunk block of one [`Table`](crate::storage::Table) has the same
//! layout, so instead of one global allocation per chunk the table draws its
//! blocks from a [`ChunkArena`]:
//!
//! * **default (`no_std`) backend** — one global `alloc`/`dealloc` per block,
//!   identical to the pre-arena behaviour;
//! * **`vm` feature backend** — the arena reserves large, page-aligned
//!   segments from `prism_platform`'s VM layer and bump-carves blocks out of
//!   them. Where the platform supports large/huge pages
//!   ([`Reservation::reserve_huge`](prism_platform::vm::Reservation)) the whole
//!   segment is huge-page backed, so a table's chunks cost a handful of TLB
//!   entries instead of hundreds; otherwise the segment is an ordinary aligned
//!   reservation. Freed blocks are reused within their segment. Empty segments
//!   are returned to the OS immediately, keeping at most one warm segment to
//!   avoid mapping churn for short-lived tables.
//!
//! # NUMA
//!
//! The default [`ChunkArena::new`] path is deliberately **first-touch**: a
//! chunk block is physically backed by the memory node of whichever thread
//! first writes it. This is a valid portable default, but it is not evidence
//! that a particular node was selected. With the `vm` feature, callers that
//! already have a worker/node placement can use [`ChunkArena::new_on_node`]
//! to request an explicit node policy. The returned diagnostic records
//! whether the platform bound the mapping, used an explicit ordinary-memory
//! fallback, or has not allocated the first segment yet. No path claims node
//! locality from a request alone.

use alloc::alloc::{alloc, dealloc, handle_alloc_error};
#[cfg(feature = "vm")]
use alloc::collections::BTreeMap;
#[cfg(feature = "vm")]
use alloc::vec::Vec;
use core::alloc::Layout;
use core::ptr::NonNull;

#[cfg(feature = "vm")]
use prism_platform::vm::{
    NumaAllocation, NumaAllocationError, NumaPlacement, Protection, Reservation,
};

/// Placement evidence for an ECS chunk arena.
///
/// `FirstTouch` and `Global` are policies, not hardware proofs. `Bound` is
/// reported only after the platform accepted a node-memory policy for a
/// segment. `Fallback` is explicit ordinary memory and must not be counted as
/// NUMA-local. `Mixed` means different segments did not produce the same
/// result, so callers must not treat the arena as uniformly node-bound.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChunkArenaPlacement {
    /// The non-VM allocator is used; no node policy was requested.
    Global,
    /// VM reservation with first-touch placement.
    FirstTouch,
    /// An explicit node request has not allocated its first segment yet.
    #[allow(dead_code)]
    #[cfg(feature = "vm")]
    Requested { node: u16 },
    /// The OS accepted the requested node policy for all observed segments.
    #[cfg(feature = "vm")]
    Bound { node: u16 },
    /// At least one segment used explicit ordinary-memory fallback.
    #[cfg(feature = "vm")]
    Fallback {
        requested_node: u16,
        reason: NumaAllocationError,
    },
    /// Segment placement evidence disagreed; locality is not uniform.
    #[cfg(feature = "vm")]
    Mixed { requested_node: u16 },
}

impl ChunkArenaPlacement {
    /// Returns true only for an arena whose observed segments are all bound to
    /// the requested node.
    #[allow(dead_code)]
    #[cfg(feature = "vm")]
    #[must_use]
    pub(crate) const fn is_node_bound(self) -> bool {
        matches!(self, Self::Bound { .. })
    }

    /// The non-VM allocator has no node-local placement evidence.
    #[allow(dead_code)]
    #[cfg(not(feature = "vm"))]
    #[must_use]
    pub(crate) const fn is_node_bound(self) -> bool {
        false
    }
}

/// Bytes reserved per VM segment (rounded up to the page/large-page size).
#[cfg(feature = "vm")]
const SEGMENT_BYTES: usize = 2 * 1024 * 1024;

/// Allocator for the fixed-layout chunk blocks of one table.
pub(crate) struct ChunkArena {
    layout: Layout,
    /// Distance between consecutive block slots: the block size rounded up to
    /// the block alignment, so every slot is correctly aligned.
    #[cfg_attr(not(feature = "vm"), allow(dead_code))]
    stride: usize,
    backend: Backend,
}

enum Backend {
    /// One global allocation per block (the `no_std` default).
    Global,
    /// Bump-carved blocks inside VM segments (the `vm` feature).
    #[cfg(feature = "vm")]
    Vm(VmArena),
}

// SAFETY: `ChunkArena` owns raw block pointers, but every method that touches
// them (`alloc`/`free`) takes `&mut self`, and the bytes it hands out hold the
// same component data the surrounding `Table`/`Column` types already declare
// `Send + Sync` under the unique-per-row discipline. Moving the arena across
// threads just moves the owning handles, exactly like a `Box`.
unsafe impl Send for ChunkArena {}
// SAFETY: see the `Send` impl above — `&ChunkArena` exposes no mutation; only
// the `&mut self` methods touch the pointers.
unsafe impl Sync for ChunkArena {}

#[cfg(feature = "vm")]
struct Segment {
    #[allow(dead_code)]
    storage: SegmentStorage,
    base: NonNull<u8>,
    /// Index of the next never-handed-out block slot.
    bump: usize,
    /// Number of block slots in this segment.
    slots: usize,
    live: usize,
    free: Vec<NonNull<u8>>,
    /// Position in `VmArena::available`, or `None` when full.
    available: Option<usize>,
}

#[cfg(feature = "vm")]
enum SegmentStorage {
    Reservation(Reservation),
    Numa(NumaAllocation),
}

#[cfg(feature = "vm")]
impl Segment {
    /// Reserve (and commit) a fresh segment sized for at least one `stride`-byte
    /// block, preferring huge pages where the platform offers them.
    fn reserve(stride: usize, align: usize) -> Option<Segment> {
        let page = prism_platform::vm::page_size().max(1);
        // Huge pages first: one reservation commit, few TLB entries for the
        // whole table's chunk set.
        if prism_platform::vm::huge_pages_supported()
            && let Some(large) = prism_platform::vm::large_page_size()
        {
            let bytes = round_up(SEGMENT_BYTES.max(stride).max(large), large)?;
            if let Ok(reservation) = Reservation::reserve_huge(bytes) {
                // A huge-page base need not meet a component's still larger
                // alignment. Drop an unsuitable mapping and use aligned VM.
                if reservation.as_mut_ptr().addr().is_multiple_of(align) {
                    return Some(Self::new(SegmentStorage::Reservation(reservation), stride));
                }
            }
        }
        // Ordinary aligned reservation, committed read-write in one call.
        let bytes = round_up(SEGMENT_BYTES.max(stride), page)?;
        let reservation = Reservation::reserve_aligned(bytes, page.max(align)).ok()?;
        reservation
            .commit(0, reservation.len(), Protection::ReadWrite)
            .ok()?;
        Some(Self::new(SegmentStorage::Reservation(reservation), stride))
    }

    /// Reserve a segment through the platform NUMA contract. Explicit node
    /// requests intentionally use `NumaAllocation` rather than pretending
    /// that a plain VM reservation is node-bound. The platform may return an
    /// explicit ordinary-memory fallback when policy binding is unavailable.
    fn reserve_on_node(
        stride: usize,
        align: usize,
        node: u16,
    ) -> Result<(Segment, NumaPlacement), NumaAllocationError> {
        let page = prism_platform::vm::page_size().max(1);
        if align > page {
            return Err(NumaAllocationError::InvalidArgument);
        }
        let bytes = round_up(SEGMENT_BYTES.max(stride), page)
            .ok_or(NumaAllocationError::InvalidArgument)?;
        let allocation = NumaAllocation::on_node_or_fallback(bytes, page.max(align), node)?;
        let placement = allocation.placement();
        Ok((
            Self::new(SegmentStorage::Numa(allocation), stride),
            placement,
        ))
    }

    fn new(mut storage: SegmentStorage, stride: usize) -> Self {
        let (base, len) = match &mut storage {
            SegmentStorage::Reservation(reservation) => (
                NonNull::new(reservation.as_mut_ptr()).expect("VM reservation has a base"),
                reservation.len(),
            ),
            SegmentStorage::Numa(allocation) => (
                NonNull::new(allocation.as_mut_ptr()).expect("NUMA allocation has a base"),
                allocation.len(),
            ),
        };
        let slots = len / stride;
        debug_assert!(slots > 0);
        Self {
            storage,
            base,
            bump: 0,
            slots,
            live: 0,
            free: Vec::new(),
            available: None,
        }
    }

    /// Base pointer of block slot `slot`.
    fn slot_ptr(&self, stride: usize, slot: usize) -> NonNull<u8> {
        // SAFETY: `slot < slots` and the segment holds `slots * stride` bytes,
        // so the offset stays inside the owned mapping/allocation.
        unsafe { NonNull::new_unchecked(self.base.as_ptr().add(slot * stride)) }
    }
}

/// Address-ordered ownership permits O(log segments) frees without scanning
/// all blocks or retaining pointers into released reservations. The available
/// stack selects a non-full segment without a linear search.
#[cfg(feature = "vm")]
#[derive(Default)]
struct VmArena {
    segments: BTreeMap<usize, Segment>,
    available: Vec<usize>,
    empty: Option<usize>,
    requested_node: Option<u16>,
    placement: Option<NumaPlacement>,
    mixed: bool,
}

#[cfg(feature = "vm")]
impl VmArena {
    fn first_touch() -> Self {
        Self {
            segments: BTreeMap::new(),
            available: Vec::new(),
            empty: None,
            requested_node: None,
            placement: None,
            mixed: false,
        }
    }

    #[allow(dead_code)]
    fn on_node(node: u16) -> Self {
        Self {
            segments: BTreeMap::new(),
            available: Vec::new(),
            empty: None,
            requested_node: Some(node),
            placement: None,
            mixed: false,
        }
    }

    fn alloc(&mut self, stride: usize, layout: Layout) -> NonNull<u8> {
        if self.available.is_empty() {
            let (mut segment, placement): (Segment, Option<NumaPlacement>) = if let Some(node) =
                self.requested_node
            {
                let (segment, placement) = Segment::reserve_on_node(stride, layout.align(), node)
                    .unwrap_or_else(|_| handle_alloc_error(layout));
                (segment, Some(placement))
            } else {
                // The first-touch path does not have a node-policy result. Its
                // diagnostic remains `FirstTouch` at the arena level.
                (
                    Segment::reserve(stride, layout.align())
                        .unwrap_or_else(|| handle_alloc_error(layout)),
                    None,
                )
            };
            if let Some(placement) = placement {
                match self.placement {
                    None => self.placement = Some(placement),
                    Some(previous) if previous == placement => {}
                    Some(_) => self.mixed = true,
                }
            }
            let base = segment.base.as_ptr().addr();
            segment.available = Some(0);
            self.segments.insert(base, segment);
            self.available.push(base);
        }
        let base = *self.available.last().expect("available segment ensured");
        if self.empty == Some(base) {
            self.empty = None;
        }
        let segment = self.segments.get_mut(&base).expect("owned segment");
        let ptr = segment.free.pop().unwrap_or_else(|| {
            let ptr = segment.slot_ptr(stride, segment.bump);
            segment.bump += 1;
            ptr
        });
        segment.live += 1;
        if segment.live == segment.slots {
            self.available.pop();
            segment.available = None;
        }
        ptr
    }

    fn placement(&self) -> Option<NumaPlacement> {
        self.placement
    }

    /// Caller has ended all accesses to this arena's live block.
    ///
    /// # Safety
    /// `ptr` must identify a live allocation belonging to this arena, and
    /// `stride` must be the stride used when that allocation was acquired.
    unsafe fn free(&mut self, ptr: NonNull<u8>, stride: usize) {
        let addr = ptr.as_ptr().addr();
        let (&base, segment) = self
            .segments
            .range_mut(..=addr)
            .next_back()
            .expect("block must belong to this arena");
        debug_assert!(addr - base < segment.bump * stride);
        debug_assert!((addr - base).is_multiple_of(stride));
        debug_assert!(segment.live > 0);
        segment.live -= 1;
        segment.free.push(ptr);
        if segment.available.is_none() {
            segment.available = Some(self.available.len());
            self.available.push(base);
        }
        if segment.live == 0 {
            if self.empty.is_none() {
                self.empty = Some(base);
            } else {
                // Keep one warm empty segment; all other empty mappings are
                // released now. No pointer in another segment refers to this
                // segment's blocks.
                let index = segment.available.expect("empty segment is available");
                self.available.swap_remove(index);
                if let Some(&moved) = self.available.get(index) {
                    self.segments
                        .get_mut(&moved)
                        .expect("owned segment")
                        .available = Some(index);
                }
                self.segments.remove(&base);
            }
        }
    }
}

impl ChunkArena {
    /// Build an arena for blocks of `block_layout`.
    pub(crate) fn new(block_layout: Layout) -> Self {
        let layout =
            Layout::from_size_align(block_layout.size().max(1), block_layout.align().max(64))
                .unwrap_or_else(|_| handle_alloc_error(block_layout));
        let stride = layout.pad_to_align().size();
        #[cfg(feature = "vm")]
        let backend = if prism_platform::vm::virtual_memory_supported() {
            Backend::Vm(VmArena::first_touch())
        } else {
            Backend::Global
        };
        #[cfg(not(feature = "vm"))]
        let backend = Backend::Global;
        Self {
            layout,
            stride,
            backend,
        }
    }

    /// Build an arena whose VM segments request a specific NUMA node.
    ///
    /// This constructor is intentionally separate from [`Self::new`]: the
    /// table's default remains portable first-touch placement, while a
    /// placement-aware caller must opt in with a worker/node decision already
    /// made by its scheduler. The first allocation is lazy; before it, the
    /// diagnostic is [`ChunkArenaPlacement::Requested`].
    #[cfg(feature = "vm")]
    #[allow(dead_code)]
    pub(crate) fn new_on_node(
        block_layout: Layout,
        node: u16,
    ) -> Result<Self, NumaAllocationError> {
        if !prism_platform::vm::virtual_memory_supported() {
            return Err(NumaAllocationError::Unsupported);
        }
        let alignment = block_layout.align().max(64);
        if alignment > prism_platform::vm::page_size() {
            return Err(NumaAllocationError::InvalidArgument);
        }
        let layout = Layout::from_size_align(block_layout.size().max(1), alignment)
            .map_err(|_| NumaAllocationError::InvalidArgument)?;
        let stride = layout.pad_to_align().size();
        Ok(Self {
            layout,
            stride,
            backend: Backend::Vm(VmArena::on_node(node)),
        })
    }

    /// Placement evidence for this arena.
    #[allow(dead_code)]
    #[must_use]
    pub(crate) fn placement(&self) -> ChunkArenaPlacement {
        match &self.backend {
            Backend::Global => ChunkArenaPlacement::Global,
            #[cfg(feature = "vm")]
            Backend::Vm(vm) => {
                let Some(node) = vm.requested_node else {
                    return ChunkArenaPlacement::FirstTouch;
                };
                if vm.mixed {
                    return ChunkArenaPlacement::Mixed {
                        requested_node: node,
                    };
                }
                match vm.placement() {
                    None => ChunkArenaPlacement::Requested { node },
                    Some(NumaPlacement::Bound { node: bound }) if bound == node => {
                        ChunkArenaPlacement::Bound { node }
                    }
                    Some(NumaPlacement::Fallback {
                        requested_node,
                        reason,
                    }) => ChunkArenaPlacement::Fallback {
                        requested_node,
                        reason,
                    },
                    Some(_) => ChunkArenaPlacement::Mixed {
                        requested_node: node,
                    },
                }
            }
        }
    }

    /// Hand out a block. The returned pointer is aligned for `block_layout`.
    pub(crate) fn alloc(&mut self) -> NonNull<u8> {
        match &mut self.backend {
            Backend::Global => {
                // SAFETY: the layout comes from the owning table and has a
                // non-zero size (`stride >= 1`), as `alloc` requires.
                let ptr = unsafe { alloc(self.layout) };
                NonNull::new(ptr).unwrap_or_else(|| handle_alloc_error(self.layout))
            }
            #[cfg(feature = "vm")]
            Backend::Vm(arena) => arena.alloc(self.stride, self.layout),
        }
    }

    /// Size in bytes of one block's allocation layout.
    #[allow(dead_code)] // used by the table's tests / future diagnostics
    #[inline]
    pub(crate) fn block_size(&self) -> usize {
        self.layout.size()
    }

    /// Whether this arena carves blocks out of VM segments rather than the
    /// global allocator (i.e. the `vm` feature is on **and** the platform has a
    /// VM backend).
    #[allow(dead_code)] // used by the arena's tests / future diagnostics
    #[inline]
    pub(crate) fn is_vm_backed(&self) -> bool {
        #[cfg(feature = "vm")]
        {
            matches!(self.backend, Backend::Vm(_))
        }
        #[cfg(not(feature = "vm"))]
        {
            false
        }
    }

    /// Return a block previously handed out by [`alloc`](ChunkArena::alloc).
    ///
    /// # Safety
    /// `ptr` must be a block from this arena that has not already been freed,
    /// and no live reference into it may remain.
    pub(crate) unsafe fn free(&mut self, ptr: NonNull<u8>) {
        match &mut self.backend {
            Backend::Global => {
                // SAFETY: forwarded contract — `ptr` came from `alloc(self.layout)`.
                unsafe { dealloc(ptr.as_ptr(), self.layout) };
            }
            #[cfg(feature = "vm")]
            Backend::Vm(arena) => {
                // SAFETY: forwarded contract: this block belongs to the arena
                // and all accesses have ended.
                unsafe { arena.free(ptr, self.stride) };
            }
        }
    }
}

/// Round `n` up to a multiple of `align` (`align` must be non-zero).
#[cfg(feature = "vm")]
fn round_up(n: usize, align: usize) -> Option<usize> {
    debug_assert!(align.is_power_of_two());
    n.checked_add(align - 1).map(|n| n & !(align - 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn arena_for(size: usize, align: usize) -> ChunkArena {
        ChunkArena::new(Layout::from_size_align(size, align).unwrap())
    }

    #[test]
    fn blocks_are_aligned_distinct_and_writable() {
        let mut arena = arena_for(16 * 1024, 64);
        let mut blocks = Vec::new();
        for i in 0..64u64 {
            let ptr = arena.alloc();
            assert_eq!(ptr.as_ptr() as usize % 64, 0, "blocks must be 64-aligned");
            assert!(!blocks.contains(&ptr), "fresh blocks must be distinct");
            // Write across the whole block to prove it is committed and usable.
            // SAFETY: `ptr` owns a block of at least 16 KiB and this write is the
            // only access.
            unsafe { core::ptr::write_bytes(ptr.as_ptr(), i as u8, 16 * 1024) };
            blocks.push(ptr);
        }
        for (i, ptr) in blocks.iter().enumerate() {
            // SAFETY: the block is live and was filled with `i as u8`.
            let first = unsafe { *ptr.as_ptr() };
            assert_eq!(first, i as u8);
        }
    }

    #[test]
    fn freed_blocks_come_back() {
        let mut arena = arena_for(4096, 64);
        let a = arena.alloc();
        let b = arena.alloc();
        // SAFETY: both blocks came from this arena and are not aliased.
        unsafe {
            arena.free(a);
            arena.free(b);
        }
        // The next two allocations reuse the freed blocks (no new reservation /
        // global allocation is needed for them).
        let c = arena.alloc();
        let d = arena.alloc();
        let mut reused = [c, d];
        reused.sort_by_key(|p| p.as_ptr() as usize);
        let mut freed = [a, b];
        freed.sort_by_key(|p| p.as_ptr() as usize);
        assert_eq!(reused, freed, "freed blocks must be handed back out");
    }

    #[test]
    fn arena_spanning_multiple_segments() {
        // A block big enough that a 2 MiB segment holds only a few of them, so a
        // modest loop crosses segment boundaries.
        let block = 512 * 1024;
        let mut arena = arena_for(block, 64);
        let mut blocks = Vec::new();
        for _ in 0..8 {
            let ptr = arena.alloc();
            // SAFETY: the block is live and this is its only access.
            unsafe { core::ptr::write_bytes(ptr.as_ptr(), 0xAB, block) };
            blocks.push(ptr);
        }
        for ptr in blocks {
            // SAFETY: the block is live and was filled with 0xAB.
            assert_eq!(unsafe { *ptr.as_ptr() }, 0xAB);
        }
    }

    /// The backend that actually got selected must match the platform's VM
    /// support: with the `vm` feature on a platform that has a VM layer the
    /// arena carves from reservations, otherwise it uses the global allocator.
    #[cfg(feature = "vm")]
    #[test]
    fn vm_feature_selects_the_vm_backend_when_supported() {
        let arena = arena_for(16 * 1024, 64);
        assert_eq!(
            arena.is_vm_backed(),
            prism_platform::vm::virtual_memory_supported(),
            "the arena backend must follow the platform's VM support",
        );
    }

    /// Blocks carved from a segment share one reservation: consecutive fresh
    /// blocks are separated by the (fixed) stride, not scattered by the global
    /// allocator.
    #[cfg(feature = "vm")]
    #[test]
    fn vm_blocks_share_one_segment() {
        let mut arena = arena_for(16 * 1024, 64);
        if !arena.is_vm_backed() {
            return; // no VM backend on this platform
        }
        let a = arena.alloc();
        let b = arena.alloc();
        let delta = b.as_ptr() as usize - a.as_ptr() as usize;
        // The stride is the 16 KiB block rounded up to 64 bytes, so the two
        // blocks sit adjacent inside one segment.
        assert_eq!(
            delta,
            16 * 1024,
            "blocks must be stride-spaced in a segment"
        );
    }

    #[cfg(feature = "vm")]
    #[test]
    fn vm_accepts_blocks_larger_than_a_segment_and_overaligned_blocks() {
        for (size, align) in [
            (3 * 1024 * 1024, 64),
            (4096, 4 * 1024 * 1024),
            (3 * 1024 * 1024, 4 * 1024 * 1024),
        ] {
            let mut arena = arena_for(size, align);
            let mut blocks = Vec::new();
            for value in 1..=3 {
                let block = arena.alloc();
                assert_eq!(block.as_ptr().addr() % align, 0);
                assert!(!blocks.contains(&block));
                // SAFETY: the allocation owns `size` writable bytes.
                unsafe { core::ptr::write_bytes(block.as_ptr(), value, size) };
                blocks.push(block);
            }
            for (index, block) in blocks.into_iter().enumerate() {
                // SAFETY: every block is live, distinct, and filled above.
                unsafe {
                    assert_eq!(*block.as_ptr(), (index + 1) as u8);
                    assert_eq!(*block.as_ptr().add(size - 1), (index + 1) as u8);
                    arena.free(block);
                }
            }
        }
    }

    #[cfg(feature = "vm")]
    #[test]
    fn vm_churn_releases_empty_segments_and_preserves_live_blocks() {
        let mut arena = arena_for(512 * 1024, 64);
        if !arena.is_vm_backed() {
            return;
        }
        for cycle in 0..16 {
            let mut blocks = Vec::new();
            for value in 0..32u8 {
                let block = arena.alloc();
                assert!(!blocks.contains(&block));
                // SAFETY: this is a fresh, exclusively owned writable block.
                unsafe { core::ptr::write_bytes(block.as_ptr(), value, 512 * 1024) };
                blocks.push(block);
            }
            // Interleave releases across segments and leave odd blocks alive
            // while even ones are returned. This exercises available-stack
            // swap removal, rather than only freeing in allocation order.
            for parity in [cycle % 2, 1 - cycle % 2] {
                for index in (parity..blocks.len()).step_by(2) {
                    let block = blocks[index];
                    // SAFETY: each index is checked and freed exactly once;
                    // every other live block must remain mapped and unchanged.
                    unsafe {
                        assert_eq!(*block.as_ptr(), index as u8);
                        assert_eq!(*block.as_ptr().add(512 * 1024 - 1), index as u8);
                        arena.free(block);
                    }
                }
            }
            let Backend::Vm(vm) = &arena.backend else {
                unreachable!()
            };
            assert_eq!(vm.segments.len(), 1, "only one warm mapping may remain");
            assert_eq!(vm.available.len(), 1);
            let (&base, segment) = vm.segments.first_key_value().unwrap();
            assert_eq!(vm.empty, Some(base));
            assert_eq!(segment.live, 0);
            assert_eq!(segment.free.len(), segment.bump);
            assert_eq!(segment.available, Some(0));
        }
    }

    #[cfg(feature = "vm")]
    #[test]
    fn vm_reuses_holes_without_new_segments() {
        let mut arena = arena_for(512 * 1024, 64);
        if !arena.is_vm_backed() {
            return;
        }
        let blocks: Vec<_> = (0..20).map(|_| arena.alloc()).collect();
        let before = match &arena.backend {
            Backend::Vm(vm) => vm.segments.len(),
            _ => unreachable!(),
        };
        for &block in blocks.iter().step_by(2) {
            // SAFETY: every even block is live and is freed once here.
            unsafe { arena.free(block) };
        }
        let replacements: Vec<_> = (0..10).map(|_| arena.alloc()).collect();
        let Backend::Vm(vm) = &arena.backend else {
            unreachable!()
        };
        assert_eq!(vm.segments.len(), before);
        for &block in &replacements {
            assert!(blocks.iter().step_by(2).any(|&old| old == block));
        }
        for block in replacements
            .into_iter()
            .chain(blocks.into_iter().skip(1).step_by(2))
        {
            // SAFETY: every remaining allocation is returned exactly once.
            unsafe { arena.free(block) };
        }
    }

    #[test]
    fn zero_sized_layout_is_normalized_for_global_allocation() {
        let mut arena = arena_for(0, 1);
        assert!(arena.layout.size() > 0);
        let block = arena.alloc();
        assert!(block.as_ptr().addr().is_multiple_of(64));
        // SAFETY: the normalized allocation has at least one writable byte.
        unsafe {
            block.as_ptr().write(42);
            assert_eq!(block.as_ptr().read(), 42);
            arena.free(block);
        }
    }

    #[test]
    fn default_arena_reports_policy_without_fabricating_node_locality() {
        let arena = arena_for(4096, 64);
        let placement = arena.placement();
        #[cfg(feature = "vm")]
        assert_eq!(placement, ChunkArenaPlacement::FirstTouch);
        #[cfg(not(feature = "vm"))]
        assert_eq!(placement, ChunkArenaPlacement::Global);
        assert!(!placement.is_node_bound());
    }

    #[cfg(feature = "vm")]
    #[test]
    fn explicit_node_arena_exposes_binding_or_explicit_fallback() {
        let layout = Layout::from_size_align(4096, 64).unwrap();
        let Ok(mut arena) = ChunkArena::new_on_node(layout, 0) else {
            // A target without a VM mapping backend has no safe explicit-node
            // path; the constructor must reject it instead of falling back to
            // the unrelated global arena.
            return;
        };
        assert_eq!(
            arena.placement(),
            ChunkArenaPlacement::Requested { node: 0 }
        );
        let block = arena.alloc();
        match arena.placement() {
            ChunkArenaPlacement::Bound { node } => assert_eq!(node, 0),
            ChunkArenaPlacement::Fallback { requested_node, .. } => assert_eq!(requested_node, 0),
            other => panic!("unexpected explicit-node placement: {other:?}"),
        }
        let placement = arena.placement();
        assert!(
            !placement.is_node_bound()
                || matches!(placement, ChunkArenaPlacement::Bound { node: 0 })
        );
        // SAFETY: `block` is the only live allocation from this arena.
        unsafe { arena.free(block) };
    }

    #[cfg(feature = "vm")]
    #[test]
    fn explicit_node_arena_rejects_alignment_that_cannot_be_node_bound() {
        let page = prism_platform::vm::page_size();
        let layout = Layout::from_size_align(4096, page * 2).unwrap();
        assert!(matches!(
            ChunkArena::new_on_node(layout, 0),
            Err(NumaAllocationError::InvalidArgument)
        ));
    }
}
