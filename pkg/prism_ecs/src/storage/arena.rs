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
//!   reservation. Freed blocks go on a free list and are reused before any new
//!   segment is reserved.
//!
//! # NUMA
//!
//! Placement is **first-touch**: a chunk block is physically backed by the
//! memory node of whichever thread first writes it, which is the classic
//! default NUMA policy and keeps an archetype's hot chunks local to the worker
//! that streams them in. Forcing a *specific* node (`mbind`/`set_mempolicy`) is
//! not something `prism_platform` exposes yet; this module says so rather than
//! pretending to bind, and a future explicit-placement backend can slot in
//! behind [`ChunkArena`] without touching the table.

use alloc::alloc::{alloc, dealloc, handle_alloc_error};
#[cfg(feature = "vm")]
use alloc::vec::Vec;
use core::alloc::Layout;
use core::ptr::NonNull;

#[cfg(feature = "vm")]
use prism_platform::vm::{Reservation, Protection};

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
    Vm {
        segments: Vec<Segment>,
        /// Blocks returned by `free`, reused before reserving a new segment.
        free: Vec<NonNull<u8>>,
    },
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
    reservation: Reservation,
    /// Index of the next never-handed-out block slot.
    bump: u32,
    /// Number of block slots in this segment.
    slots: u32,
}

#[cfg(feature = "vm")]
impl Segment {
    /// Reserve (and commit) a fresh segment sized for at least one `stride`-byte
    /// block, preferring huge pages where the platform offers them.
    fn reserve(stride: usize) -> Option<Segment> {
        let page = prism_platform::vm::page_size().max(1);
        // Huge pages first: one reservation commit, few TLB entries for the
        // whole table's chunk set.
        if prism_platform::vm::huge_pages_supported()
            && let Some(large) = prism_platform::vm::large_page_size()
        {
            let bytes = round_up(SEGMENT_BYTES.max(large), large);
            if let Ok(reservation) = Reservation::reserve_huge(bytes) {
                let slots = (reservation.len() / stride) as u32;
                if slots > 0 {
                    return Some(Segment {
                        reservation,
                        bump: 0,
                        slots,
                    });
                }
            }
        }
        // Ordinary aligned reservation, committed read-write in one call.
        let bytes = round_up(SEGMENT_BYTES, page);
        let reservation = Reservation::reserve_aligned(bytes, page).ok()?;
        reservation
            .commit(0, reservation.len(), Protection::ReadWrite)
            .ok()?;
        let slots = (reservation.len() / stride) as u32;
        if slots == 0 {
            return None;
        }
        Some(Segment {
            reservation,
            bump: 0,
            slots,
        })
    }

    /// Base pointer of block slot `slot`.
    fn slot_ptr(&self, stride: usize, slot: u32) -> NonNull<u8> {
        // SAFETY: `slot < slots` and the reservation holds `slots * stride`
        // bytes, so the offset stays in the committed region.
        unsafe {
            NonNull::new_unchecked(self.reservation.as_mut_ptr().add(slot as usize * stride))
        }
    }
}

impl ChunkArena {
    /// Build an arena for blocks of `block_layout`.
    pub(crate) fn new(block_layout: Layout) -> Self {
        let align = block_layout.align().max(64);
        let stride = round_up(block_layout.size().max(1), align);
        #[cfg(feature = "vm")]
        let backend = if prism_platform::vm::virtual_memory_supported() {
            Backend::Vm {
                segments: Vec::new(),
                free: Vec::new(),
            }
        } else {
            Backend::Global
        };
        #[cfg(not(feature = "vm"))]
        let backend = Backend::Global;
        Self {
            layout: block_layout,
            stride,
            backend,
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
            Backend::Vm { segments, free } => {
                if let Some(ptr) = free.pop() {
                    return ptr;
                }
                if segments.last().is_none_or(|s| s.bump >= s.slots) {
                    let segment = Segment::reserve(self.stride)
                        .unwrap_or_else(|| handle_alloc_error(self.layout));
                    segments.push(segment);
                }
                let slot = {
                    let segment = segments.last_mut().expect("segment just ensured");
                    let slot = segment.bump;
                    segment.bump += 1;
                    slot
                };
                segments.last().expect("segment just ensured").slot_ptr(self.stride, slot)
            }
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
            matches!(self.backend, Backend::Vm { .. })
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
            Backend::Vm { free, .. } => {
                // Retained for reuse; the segment (and thus the block) is
                // released to the OS only when the arena drops.
                free.push(ptr);
            }
        }
    }
}

/// Round `n` up to a multiple of `align` (`align` must be non-zero).
fn round_up(n: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (n + align - 1) & !(align - 1)
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
        assert_eq!(delta, 16 * 1024, "blocks must be stride-spaced in a segment");
    }
}
