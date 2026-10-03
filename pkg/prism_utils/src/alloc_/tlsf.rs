//! A `TLSF` (Two-Level Segregated Fit) sub-allocator over an abstract region:
//! [`TlsfAllocator`].
//!
//! # What it is
//! `TLSF` is a real-time general allocator with `O(1)` allocate and free. Free
//! blocks are bucketed by size into a two-level segregated structure: a
//! first-level index picks a power-of-two size class (the position of the size's
//! most significant bit), and a second-level index linearly subdivides that
//! class into [`SL_COUNT`] ranges. Two bitmaps (one first-level, one
//! second-level per first-level class) let a request jump straight to a
//! non-empty list that is guaranteed to fit, using only `trailing_zeros`, so
//! there is no search loop. Freed blocks are coalesced with their physical
//! neighbours via boundary tags, bounding fragmentation.
//!
//! # Offsets, not pointers
//! Like [`BuddyAllocator`](crate::alloc_::buddy::BuddyAllocator), this manages
//! *byte offsets* into a caller-owned region (host RAM, a `GPU` heap, a
//! streaming ring) and never dereferences memory. All structure is held in an
//! index-based block pool with explicit physical and free-list links, so the
//! module is safe code with no `unsafe`. [`allocate`](TlsfAllocator::allocate)
//! returns an offset (optionally aligned), or `None` on exhaustion.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

/// Base allocation granularity / minimum block size in bytes (power of two).
const ALIGN: usize = 8;
/// `log2(ALIGN)`.
const ALIGN_LOG2: usize = 3;
/// `log2` of the second-level subdivision count.
const SL_LOG2: usize = 4;
/// Number of second-level lists per first-level class.
pub const SL_COUNT: usize = 1 << SL_LOG2;
/// Blocks smaller than `1 << FL_SHIFT` share first-level class 0.
const FL_SHIFT: usize = SL_LOG2 + ALIGN_LOG2;
/// The small-block threshold.
const SMALL: usize = 1 << FL_SHIFT;
/// Number of first-level classes. Class `i` covers sizes near
/// `1 << (i + FL_SHIFT - 1)`, so this supports regions up to roughly
/// `2^38` bytes.
const FL_COUNT: usize = 32;

/// Index of the most significant set bit of a non-zero value.
#[inline]
fn fls(x: usize) -> usize {
    (usize::BITS - 1 - x.leading_zeros()) as usize
}

/// Round `x` up to a multiple of the power-of-two `align`.
#[inline]
fn round_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}

/// A physical block in the region. Blocks tile the region with no gaps; the
/// `prev_phys` / `next_phys` links form the boundary-tag chain used for
/// coalescing, and `prev_free` / `next_free` thread the segregated free lists.
#[derive(Clone, Debug)]
struct Block {
    offset: usize,
    size: usize,
    free: bool,
    prev_phys: Option<usize>,
    next_phys: Option<usize>,
    prev_free: Option<usize>,
    next_free: Option<usize>,
}

/// A `TLSF` allocator over a region of `size` bytes.
#[derive(Debug)]
pub struct TlsfAllocator {
    size: usize,
    /// Block pool. Slots are recycled through `recycled` when blocks merge.
    blocks: Vec<Block>,
    /// Free block-pool slots available for reuse.
    recycled: Vec<usize>,
    /// Allocated blocks, keyed by their byte offset, for `O(log n)` free.
    allocated: BTreeMap<usize, usize>,
    /// First-level occupancy bitmap: bit `fl` set iff some second-level list of
    /// class `fl` is non-empty.
    fl_bitmap: u32,
    /// Second-level occupancy bitmaps, one per first-level class.
    sl_bitmap: [u16; FL_COUNT],
    /// Free-list heads indexed `[fl][sl]`.
    heads: Vec<[Option<usize>; SL_COUNT]>,
    /// Bytes currently handed out.
    used: usize,
}

impl TlsfAllocator {
    /// Create an allocator managing a region of `size` bytes.
    ///
    /// `size` is rounded down to a multiple of the 8-byte granularity.
    ///
    /// # Panics
    /// Panics if the rounded size is zero or too large for the first-level
    /// index range (roughly `2^38` bytes).
    #[must_use]
    pub fn new(size: usize) -> Self {
        let size = size & !(ALIGN - 1);
        assert!(size >= ALIGN, "TLSF region must be at least 8 bytes");
        let (fl, _) = mapping_insert(size);
        assert!(fl < FL_COUNT, "TLSF region too large for first-level range");
        let root = Block {
            offset: 0,
            size,
            free: false,
            prev_phys: None,
            next_phys: None,
            prev_free: None,
            next_free: None,
        };
        let mut me = Self {
            size,
            blocks: vec![root],
            recycled: Vec::new(),
            allocated: BTreeMap::new(),
            fl_bitmap: 0,
            sl_bitmap: [0; FL_COUNT],
            heads: vec![[None; SL_COUNT]; FL_COUNT],
            used: 0,
        };
        me.insert_free(0);
        me
    }

    /// Total managed size in bytes.
    #[must_use]
    #[inline]
    pub fn capacity(&self) -> usize {
        self.size
    }

    /// Bytes currently allocated (rounded up to block sizes).
    #[must_use]
    #[inline]
    pub fn allocated_bytes(&self) -> usize {
        self.used
    }

    /// Bytes not currently allocated (may be fragmented).
    #[must_use]
    #[inline]
    pub fn free_bytes(&self) -> usize {
        self.size - self.used
    }

    /// Allocate `size` bytes with 8-byte alignment, returning a byte offset or
    /// `None` on exhaustion.
    pub fn allocate(&mut self, size: usize) -> Option<usize> {
        self.allocate_aligned(size, ALIGN)
    }

    /// Allocate `size` bytes aligned to `align` (a power of two), returning a
    /// byte offset or `None` on exhaustion.
    ///
    /// # Panics
    /// Panics if `align` is not a power of two.
    pub fn allocate_aligned(&mut self, size: usize, align: usize) -> Option<usize> {
        if size == 0 {
            return None;
        }
        assert!(align.is_power_of_two(), "align must be a power of two");
        let align = align.max(ALIGN);
        let adjusted = round_up(size, ALIGN).max(ALIGN);
        // Over-request so a block always contains an aligned sub-block.
        let search = if align > ALIGN {
            adjusted + align - ALIGN
        } else {
            adjusted
        };

        let mut bi = self.find_suitable(search)?;
        self.remove_free(bi);

        // Trim front padding so the returned offset is aligned. `offset` and
        // `aligned` are both multiples of ALIGN, so `front` is 0 or >= ALIGN.
        let base = self.blocks[bi].offset;
        let aligned = round_up(base, align);
        let front = aligned - base;
        if front > 0 {
            let back = self.split_block(bi, front);
            self.insert_free(bi); // front padding returns to the pool
            bi = back;
        }

        // Trim trailing slack into its own free block if it is big enough.
        if self.blocks[bi].size >= adjusted + ALIGN {
            let tail = self.split_block(bi, adjusted);
            self.insert_free(tail);
        }

        self.blocks[bi].free = false;
        self.used += self.blocks[bi].size;
        let offset = self.blocks[bi].offset;
        self.allocated.insert(offset, bi);
        Some(offset)
    }

    /// Free a block previously returned by [`allocate`](Self::allocate) or
    /// [`allocate_aligned`](Self::allocate_aligned). Returns `true` if `offset`
    /// named a live allocation.
    pub fn deallocate(&mut self, offset: usize) -> bool {
        let Some(bi) = self.allocated.remove(&offset) else {
            return false;
        };
        self.used -= self.blocks[bi].size;
        self.blocks[bi].free = true;

        let mut cur = bi;
        // Merge with the following physical block if it is free.
        if let Some(next) = self.blocks[cur].next_phys
            && self.blocks[next].free
        {
            self.remove_free(next);
            self.merge_next(cur);
        }
        // Merge with the preceding physical block if it is free.
        if let Some(prev) = self.blocks[cur].prev_phys
            && self.blocks[prev].free
        {
            self.remove_free(prev);
            self.merge_next(prev);
            cur = prev;
        }
        self.insert_free(cur);
        true
    }

    // --- internal bookkeeping ---------------------------------------------

    /// Allocate a block-pool slot, recycling a freed one when possible.
    fn alloc_slot(&mut self, block: Block) -> usize {
        if let Some(idx) = self.recycled.pop() {
            self.blocks[idx] = block;
            idx
        } else {
            self.blocks.push(block);
            self.blocks.len() - 1
        }
    }

    /// Split block `bi` (size `S`) into `bi` of `first_size` and a new trailing
    /// block of `S - first_size`, preserving the physical chain; returns the
    /// new trailing block index.
    fn split_block(&mut self, bi: usize, first_size: usize) -> usize {
        let total = self.blocks[bi].size;
        debug_assert!(first_size > 0 && first_size < total);
        let offset = self.blocks[bi].offset + first_size;
        let next_phys = self.blocks[bi].next_phys;
        let new = self.alloc_slot(Block {
            offset,
            size: total - first_size,
            free: false,
            prev_phys: Some(bi),
            next_phys,
            prev_free: None,
            next_free: None,
        });
        self.blocks[bi].size = first_size;
        self.blocks[bi].next_phys = Some(new);
        if let Some(n) = next_phys {
            self.blocks[n].prev_phys = Some(new);
        }
        new
    }

    /// Merge `bi`'s immediate physical successor into `bi` and recycle the
    /// successor's slot. The successor must already be unlinked from any free
    /// list.
    fn merge_next(&mut self, bi: usize) {
        let next = self.blocks[bi].next_phys.expect("merge_next without successor");
        self.blocks[bi].size += self.blocks[next].size;
        let after = self.blocks[next].next_phys;
        self.blocks[bi].next_phys = after;
        if let Some(a) = after {
            self.blocks[a].prev_phys = Some(bi);
        }
        self.recycled.push(next);
    }

    /// Push block `bi` onto the segregated free list for its size and mark the
    /// occupancy bitmaps.
    fn insert_free(&mut self, bi: usize) {
        let (fl, sl) = mapping_insert(self.blocks[bi].size);
        debug_assert!(fl < FL_COUNT);
        let head = self.heads[fl][sl];
        self.blocks[bi].free = true;
        self.blocks[bi].prev_free = None;
        self.blocks[bi].next_free = head;
        if let Some(h) = head {
            self.blocks[h].prev_free = Some(bi);
        }
        self.heads[fl][sl] = Some(bi);
        self.fl_bitmap |= 1 << fl;
        self.sl_bitmap[fl] |= 1 << sl;
    }

    /// Unlink block `bi` from its segregated free list, clearing bitmaps when a
    /// list becomes empty.
    fn remove_free(&mut self, bi: usize) {
        let (fl, sl) = mapping_insert(self.blocks[bi].size);
        let prev = self.blocks[bi].prev_free;
        let next = self.blocks[bi].next_free;
        if let Some(p) = prev {
            self.blocks[p].next_free = next;
        } else {
            self.heads[fl][sl] = next;
        }
        if let Some(n) = next {
            self.blocks[n].prev_free = prev;
        }
        if self.heads[fl][sl].is_none() {
            self.sl_bitmap[fl] &= !(1u16 << sl);
            if self.sl_bitmap[fl] == 0 {
                self.fl_bitmap &= !(1u32 << fl);
            }
        }
        self.blocks[bi].prev_free = None;
        self.blocks[bi].next_free = None;
        self.blocks[bi].free = false;
    }

    /// Find a free block guaranteed to fit `size`, using the two-level bitmaps,
    /// or `None` if the region is exhausted.
    fn find_suitable(&self, size: usize) -> Option<usize> {
        let (mut fl, sl) = mapping_search(size);
        if fl >= FL_COUNT {
            return None;
        }
        // Second-level lists in class `fl` at or above `sl`.
        let sl_map = self.sl_bitmap[fl] & ((u16::MAX).wrapping_shl(sl as u32));
        let sl_final = if sl_map != 0 {
            sl_map.trailing_zeros() as usize
        } else {
            // Jump to the next non-empty first-level class.
            let fl_map = self.fl_bitmap & (u32::MAX.checked_shl((fl + 1) as u32).unwrap_or(0));
            if fl_map == 0 {
                return None;
            }
            fl = fl_map.trailing_zeros() as usize;
            self.sl_bitmap[fl].trailing_zeros() as usize
        };
        self.heads[fl][sl_final]
    }
}

/// Map a block size to its `(first_level, second_level)` free-list indices.
fn mapping_insert(size: usize) -> (usize, usize) {
    if size < SMALL {
        (0, size >> ALIGN_LOG2)
    } else {
        let f = fls(size);
        let sl = (size >> (f - SL_LOG2)) - SL_COUNT;
        let fl = f - (FL_SHIFT - 1);
        (fl, sl)
    }
}

/// Map a request size to the `(first_level, second_level)` indices of the first
/// list whose smallest member is guaranteed to fit, rounding up within the
/// size class.
fn mapping_search(size: usize) -> (usize, usize) {
    if size >= SMALL {
        let f = fls(size);
        let round = (1 << (f - SL_LOG2)) - 1;
        mapping_insert(size + round)
    } else {
        mapping_insert(size)
    }
}
