//! A type-erased, growable column of equally-sized values stored in fixed-size
//! **blocks** — the physical layer of the columnar (Structure-of-Arrays) storage
//! described in design §5.3.
//!
//! # Block layout
//!
//! Unlike a `Vec<T>`, a [`BlobVec`] does not keep its elements in one
//! ever-reallocating contiguous buffer. It allocates them in **blocks of a
//! fixed number of element slots** ([`BlobVec::rows_per_block`]). Block `b`
//! holds rows `[b * rows_per_block, (b + 1) * rows_per_block)`; within a block
//! the values stay contiguous, so each block is an independent, cache/GPU
//! friendly SoA run that can be allocated, freed and handed out on its own —
//! exactly the "16 KiB chunk" of the design, with the row-window size chosen by
//! the owning [`Table`](crate::storage::Table) so one chunk's worth of a row
//! spans all of its columns.
//!
//! Consequences the callers rely on:
//!
//! * **Stable addresses.** Growing a column never reallocates or moves the
//!   blocks already handed out: appending a row that crosses a block boundary
//!   merely allocates a fresh block, leaving every existing value at the same
//!   address (the property GPU resident columns §15 and interior-mutability
//!   change stamping depend on).
//! * **Chunk-local slices.** [`as_slice_range`](BlobVec::as_slice_range) hands
//!   out a `&[T]` only when `[start, end)` lies inside a *single* block — the
//!   slice *is* the SIMD kernel input (§7/§17). The row-by-row query fetch does
//!   **not** go through a block lookup at all: it resolves the block base once
//!   per chunk (see `TableFetch` in `query/fetch.rs`). Only whole-column
//!   ([`as_slice`](BlobVec::as_slice)) and per-range access touch the block
//!   table, and those are per-chunk / per-snapshot, not per-row.
//!
//! All element access is `unsafe` because the type is erased — callers must
//! pass pointers to values of the exact type the column was created for.

use alloc::alloc::{alloc, dealloc, handle_alloc_error};
use alloc::vec::Vec;
use core::alloc::Layout;
use core::fmt;
use core::ptr::NonNull;

use crate::component::DropFn;
use crate::storage::chunk::TARGET_CHUNK_BYTES;

/// Minimum alignment of every block, matching a common SIMD register / cache
/// line width so the kernels can use aligned loads where it helps.
const BLOCK_ALIGN: usize = 64;

/// Errors reported by the fallible `BlobVec` plumbing helpers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlobVecError {
    /// The requested length cannot be represented by `usize`.
    CapacityOverflow,
    /// The configured element size and row window cannot form one block.
    BlockLayoutOverflow,
    /// A borrowed column was asked to grow beyond blocks owned by its table.
    BorrowedCapacity {
        /// Capacity currently owned by the borrowed column.
        capacity: usize,
        /// Capacity required by the requested operation.
        required: usize,
    },
    /// A borrowed block attachment was null or incorrectly aligned.
    InvalidBlockPointer,
    /// Detaching a block would violate the borrowed-column lifecycle.
    NoAttachedBlock,
}

impl fmt::Display for BlobVecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CapacityOverflow => f.write_str("BlobVec capacity overflow"),
            Self::BlockLayoutOverflow => f.write_str("BlobVec block layout overflow"),
            Self::BorrowedCapacity { capacity, required } => write!(
                f,
                "borrowed BlobVec capacity {capacity} is smaller than required {required}"
            ),
            Self::InvalidBlockPointer => f.write_str("invalid BlobVec block pointer"),
            Self::NoAttachedBlock => f.write_str("no attached BlobVec block to detach"),
        }
    }
}

/// A type-erased, owning, growable array of `item_layout`-sized values stored in
/// fixed-slot blocks (design §5.3 physical chunks).
pub struct BlobVec {
    item_layout: Layout,
    /// Element slots per block. `usize::MAX` for zero-sized items (no blocks).
    rows_per_block: usize,
    len: usize,
    /// The allocated blocks, in row order. Block `b` covers rows
    /// `[b * rows_per_block, (b + 1) * rows_per_block)`.
    blocks: Vec<Block>,
    drop: Option<DropFn>,
    /// Whether this column owns (allocates and frees) its blocks.
    ///
    /// A standalone column owns them. A column that is one facet of a
    /// [`Table`](crate::storage::Table) **does not**: the table allocates one
    /// physical chunk block holding every column of a row window (design §5.3
    /// DOTS chunk) and *attaches* each column's slice of it, so growing or
    /// shrinking is driven by the table and this column never calls the global
    /// allocator.
    owns_blocks: bool,
}

/// A single allocated block: `rows_per_block` element slots, contiguous.
struct Block {
    ptr: NonNull<u8>,
}

// SAFETY: `BlobVec` only stores component data for `Component: Send + Sync`
// types (or dynamically-registered data the caller promises is `Send + Sync`),
// so transferring/aliasing the column across threads is as safe as for the
// underlying values. The raw block pointers are owning allocations, not shared
// state.
unsafe impl Send for BlobVec {}
// SAFETY: see the `Send` impl above.
unsafe impl Sync for BlobVec {}

impl BlobVec {
    /// Create an empty column for values described by `item_layout`, dropped by
    /// `drop` (or none if the type needs no drop).
    ///
    /// The block slot count is derived from the element size so one block holds
    /// about [`TARGET_CHUNK_BYTES`] of one column; a whole-table chunk uses
    /// [`new_with_rows_per_block`](BlobVec::new_with_rows_per_block) so every
    /// column shares the table's row window.
    pub fn new(item_layout: Layout, drop: Option<DropFn>) -> Self {
        let rows_per_block = match TARGET_CHUNK_BYTES.checked_div(item_layout.size()) {
            None => usize::MAX, // zero-sized item: no block is ever allocated
            Some(0) => 1,       // element larger than a target block: one per block
            Some(rows) => rows,
        };
        Self::new_with_rows_per_block(item_layout, drop, rows_per_block)
    }

    /// Create an empty column whose blocks each hold `rows_per_block` element
    /// slots.
    ///
    /// The owning table passes its shared `rows_per_chunk` so a column block is
    /// exactly one logical chunk window of that table (design §5.3).
    pub fn new_with_rows_per_block(
        item_layout: Layout,
        drop: Option<DropFn>,
        rows_per_block: usize,
    ) -> Self {
        let rows_per_block = if item_layout.size() == 0 {
            usize::MAX
        } else {
            rows_per_block.max(1)
        };
        Self {
            item_layout,
            rows_per_block,
            len: 0,
            blocks: Vec::new(),
            drop,
            owns_blocks: true,
        }
    }

    /// Create an empty column whose blocks are **attached by an external owner**
    /// (a [`Table`](crate::storage::Table) chunk block) rather than allocated
    /// here.
    ///
    /// In this mode [`reserve`](BlobVec::reserve) and the swap-remove shrink path
    /// never touch the global allocator — the owner grows and shrinks the chunk
    /// blocks and calls [`attach_block`](BlobVec::attach_block) /
    /// [`detach_block`](BlobVec::detach_block) — and dropping the column frees
    /// nothing.
    pub fn new_borrowed(item_layout: Layout, drop: Option<DropFn>, rows_per_block: usize) -> Self {
        let mut this = Self::new_with_rows_per_block(item_layout, drop, rows_per_block);
        this.owns_blocks = false;
        this
    }

    /// Whether this column owns its block allocations.
    #[inline]
    pub fn owns_blocks(&self) -> bool {
        self.owns_blocks
    }

    /// Attach the base pointer of this column's slice of a new chunk block.
    ///
    /// Only valid for a borrowed column (see [`new_borrowed`](BlobVec::new_borrowed));
    /// the owner must pass the slice base for the chunk it is appending, so the
    /// block list stays in row-window order. Zero-sized columns hold no bytes
    /// and ignore the pointer.
    ///
    /// # Safety
    /// `ptr` must be non-null and aligned for this column's element type, and
    /// must point at a live slice of at least `rows_per_block` slots inside a
    /// chunk block the owner keeps alive for this column's lifetime.
    pub unsafe fn attach_block(&mut self, ptr: *mut u8) {
        // SAFETY: forwarded from this function's caller contract.
        unsafe { self.try_attach_block(ptr) }
            .unwrap_or_else(|error| panic!("BlobVec block attachment failed: {error}"));
    }

    /// Fallible form of [`attach_block`](Self::attach_block).
    ///
    /// The caller still owns the unsafe lifetime/alising contract; this method
    /// checks the pointer invariants before it records the block.
    ///
    /// # Safety
    /// `ptr` must satisfy the same lifetime and ownership contract as
    /// [`Self::attach_block`].
    pub unsafe fn try_attach_block(&mut self, ptr: *mut u8) -> Result<(), BlobVecError> {
        if self.owns_blocks {
            return Err(BlobVecError::InvalidBlockPointer);
        }
        if self.item_layout.size() == 0 {
            return Ok(());
        }
        if ptr.is_null() || !(ptr as usize).is_multiple_of(self.item_layout.align()) {
            return Err(BlobVecError::InvalidBlockPointer);
        }
        self.blocks.push(Block {
            // SAFETY: the caller guarantees `ptr` is non-null and aligned (a
            // slice base inside a chunk block it owns); it is never dereferenced
            // past the live rows.
            ptr: unsafe { NonNull::new_unchecked(ptr) },
        });
        Ok(())
    }

    /// Detach the trailing chunk block (its storage is owned by the table, so it
    /// is not freed here) after the block's last live row was removed.
    pub fn detach_block(&mut self) {
        self.try_detach_block()
            .unwrap_or_else(|error| panic!("BlobVec block detachment failed: {error}"));
    }

    /// Fallible form of [`detach_block`](Self::detach_block).
    pub fn try_detach_block(&mut self) -> Result<(), BlobVecError> {
        if self.owns_blocks {
            return Err(BlobVecError::NoAttachedBlock);
        }
        if self.item_layout.size() == 0 {
            return Ok(());
        }
        self.blocks
            .pop()
            .map(|_| ())
            .ok_or(BlobVecError::NoAttachedBlock)
    }

    /// Number of live elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the column holds no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The per-element layout.
    #[inline]
    pub fn item_layout(&self) -> Layout {
        self.item_layout
    }

    /// The fixed number of element slots per block (the table's chunk window).
    ///
    /// [`usize::MAX`] for a zero-sized item, which allocates no blocks.
    #[inline]
    pub fn rows_per_block(&self) -> usize {
        self.rows_per_block
    }

    /// Number of blocks currently allocated.
    #[inline]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Elements the backing blocks can hold before another block must be
    /// allocated.
    ///
    /// Zero-sized element types report [`usize::MAX`]: they never allocate.
    #[inline]
    pub fn capacity(&self) -> usize {
        if self.item_layout.size() == 0 {
            return usize::MAX;
        }
        self.blocks.len().saturating_mul(self.rows_per_block)
    }

    /// The block index owning `row`, and the row-within-block offset.
    #[inline]
    fn split(&self, row: usize) -> (usize, usize) {
        (row / self.rows_per_block, row % self.rows_per_block)
    }

    /// The layout of one block (its slot count times the element size), with the
    /// block alignment.
    fn block_layout(&self) -> Layout {
        self.try_block_layout()
            .expect("BlobVec block layout invariant violated")
    }

    /// Compute one block's allocation layout without panicking.
    fn try_block_layout(&self) -> Result<Layout, BlobVecError> {
        let size = self
            .item_layout
            .size()
            .checked_mul(self.rows_per_block)
            .ok_or(BlobVecError::BlockLayoutOverflow)?;
        let align = self.item_layout.align().max(BLOCK_ALIGN);
        Layout::from_size_align(size, align).map_err(|_| BlobVecError::BlockLayoutOverflow)
    }

    /// Free and drop the trailing block (used when a removal empties it).
    fn pop_block(&mut self) {
        let layout = self.block_layout();
        let block = self.blocks.pop().expect("pop_block on an empty column");
        // SAFETY: `block.ptr` was allocated with this exact layout by
        // `push_block`, and no value in the block is live (the caller only calls
        // this after the block's last row was removed).
        unsafe { dealloc(block.ptr.as_ptr(), layout) };
    }

    /// Ensure there is capacity for at least `additional` more elements.
    pub fn reserve(&mut self, additional: usize) {
        self.try_reserve(additional)
            .unwrap_or_else(|error| panic!("BlobVec reserve failed: {error}"));
    }

    /// Fallible reserve path used by callers that need to reject malformed
    /// lengths or a borrowed-column ownership violation without entering an
    /// unsafe write path.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), BlobVecError> {
        if self.item_layout.size() == 0 {
            return Ok(()); // no blocks are ever needed for a ZST
        }
        let required = self
            .len
            .checked_add(additional)
            .ok_or(BlobVecError::CapacityOverflow)?;
        if !self.owns_blocks {
            // The owning table grew the chunk blocks before filling. Refuse an
            // under-provisioned borrowed column in release builds too.
            let capacity = self.capacity();
            return if capacity >= required {
                Ok(())
            } else {
                Err(BlobVecError::BorrowedCapacity { capacity, required })
            };
        }
        let needed_blocks = required.div_ceil(self.rows_per_block);
        while self.blocks.len() < needed_blocks {
            let layout = self.try_block_layout()?;
            // SAFETY: the checked layout is non-zero for a non-ZST item and has
            // a valid alignment for the global allocator.
            let ptr = unsafe { alloc(layout) };
            let ptr = NonNull::new(ptr).unwrap_or_else(|| handle_alloc_error(layout));
            self.blocks.push(Block { ptr });
        }
        Ok(())
    }

    /// Free every trailing block that no longer holds a live row.
    #[inline]
    fn shrink_blocks_to_len(&mut self) {
        if self.item_layout.size() == 0 || !self.owns_blocks {
            return; // borrowed blocks are detached by the owning table
        }
        let needed = self.len.div_ceil(self.rows_per_block);
        while self.blocks.len() > needed {
            self.pop_block();
        }
    }

    /// Pointer to the start of element `index`.
    ///
    /// # Safety
    /// `index` must be `< capacity()` (callers writing a freshly-reserved slot
    /// call [`reserve`](BlobVec::reserve) first). The returned pointer must only
    /// be used for `item_layout`-typed accesses, and only indexes `< len()` are
    /// dereferenced.
    #[inline]
    pub unsafe fn get_ptr(&self, index: usize) -> *mut u8 {
        if self.item_layout.size() == 0 {
            // A dangling-but-aligned base is valid for zero-sized reads/writes.
            return dangling_aligned(self.item_layout.align()).as_ptr();
        }
        let (block, offset) = self.split(index);
        if block >= self.blocks.len() {
            // Only reachable when `index` is past the last live row of an empty
            // column (e.g. an empty `as_slice`); the base is never dereferenced.
            return dangling_aligned(self.item_layout.align().max(BLOCK_ALIGN)).as_ptr();
        }
        // SAFETY: `block < blocks.len()` was just checked; the offset stays
        // within the block because `offset < rows_per_block`.
        unsafe {
            let base = self.blocks.get_unchecked(block).ptr.as_ptr();
            base.add(offset * self.item_layout.size())
        }
    }

    /// Borrow the live elements as a typed slice `&[T]`.
    ///
    /// A block-based column is only a single contiguous run while it fits in one
    /// block, so this is valid **only for single-block columns**; iterate
    /// [`as_slice_range`](BlobVec::as_slice_range) block by block otherwise (the
    /// `simd_iter` / `par_chunks` drivers already do).
    ///
    /// # Safety
    /// `T` must be the exact type this column stores (so `item_layout` matches
    /// `Layout::new::<T>()`), and the caller must hold shared access to every
    /// live element for the slice's lifetime.
    ///
    /// # Panics
    /// Panics if the column spans more than one block.
    #[inline]
    pub unsafe fn as_slice<T>(&self) -> &[T] {
        assert!(
            self.blocks.len() <= 1,
            "whole-column as_slice requires a single block; this column spans \
             {} blocks — use as_slice_range per block (simd_iter / par_chunks)",
            self.blocks.len(),
        );
        // SAFETY: the forwarded type contract makes the base a valid `*const T`;
        // at most one block holds all `len` live elements, contiguous from base.
        unsafe { core::slice::from_raw_parts(self.get_ptr(0).cast::<T>(), self.len) }
    }

    /// Borrow the live elements as a typed mutable slice `&mut [T]`.
    ///
    /// See [`as_slice`](BlobVec::as_slice): valid only for single-block columns.
    ///
    /// # Safety
    /// `T` must be the exact type this column stores, and the caller must hold
    /// unique access to every live element for the slice's lifetime.
    ///
    /// # Panics
    /// Panics if the column spans more than one block.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn as_mut_slice<T>(&self) -> &mut [T] {
        assert!(
            self.blocks.len() <= 1,
            "whole-column as_mut_slice requires a single block; this column spans \
             {} blocks — use as_mut_slice_range per block (simd_iter / par_chunks)",
            self.blocks.len(),
        );
        // SAFETY: as `as_slice`, plus the caller's unique-access guarantee makes
        // forming a `&mut [T]` non-aliasing.
        unsafe { core::slice::from_raw_parts_mut(self.get_ptr(0).cast::<T>(), self.len) }
    }

    /// Borrow the half-open sub-range `[start, end)` as a typed slice `&[T]`.
    ///
    /// The range must lie **within a single block** (the design's chunk-local
    /// slice; the SIMD kernels run one block at a time). Forming it directly
    /// from the offset element base avoids transiently borrowing a whole-column
    /// slice, so disjoint ranges of one column can go to different threads.
    ///
    /// # Safety
    /// `T` must be the exact type this column stores; `start <= end <= len()`;
    /// and the caller must hold shared access to every element in the range for
    /// the slice's lifetime.
    ///
    /// # Panics
    /// Panics if `[start, end)` straddles a block boundary.
    #[inline]
    pub unsafe fn as_slice_range<T>(&self, start: usize, end: usize) -> &[T] {
        debug_assert!(start <= end && end <= self.len);
        self.assert_single_block_range(start, end);
        // SAFETY: the forwarded type contract makes the base a valid `*const T`;
        // the range is within one block, so `start .. end` is contiguous.
        unsafe { core::slice::from_raw_parts(self.get_ptr(start).cast::<T>(), end - start) }
    }

    /// Borrow the half-open sub-range `[start, end)` as a typed mutable slice
    /// `&mut [T]`, within a single block.
    ///
    /// See [`as_slice_range`](BlobVec::as_slice_range).
    ///
    /// # Safety
    /// `T` must be the exact type this column stores; `start <= end <= len()`;
    /// and the caller must hold unique access to every element in the range for
    /// the slice's lifetime (disjoint ranges on the same column may be lent to
    /// different threads).
    ///
    /// # Panics
    /// Panics if `[start, end)` straddles a block boundary.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn as_mut_slice_range<T>(&self, start: usize, end: usize) -> &mut [T] {
        debug_assert!(start <= end && end <= self.len);
        self.assert_single_block_range(start, end);
        // SAFETY: as `as_slice_range`, plus the caller's unique-access guarantee
        // over `[start, end)` makes the `&mut [T]` non-aliasing even when
        // sibling ranges of the same column are borrowed concurrently.
        unsafe { core::slice::from_raw_parts_mut(self.get_ptr(start).cast::<T>(), end - start) }
    }

    /// Assert `[start, end)` does not cross a block boundary (or is empty).
    #[inline]
    fn assert_single_block_range(&self, start: usize, end: usize) {
        if self.item_layout.size() == 0 || start == end {
            return; // ZSTs have a single logical block; empty ranges are fine
        }
        assert_eq!(
            start / self.rows_per_block,
            (end - 1) / self.rows_per_block,
            "column slice [{start}, {end}) straddles a block boundary \
             (rows_per_block {}); slices must be block-local",
            self.rows_per_block,
        );
    }

    /// Append a value by copying `item_layout.size()` bytes from `value`.
    ///
    /// Ownership of the value is transferred into the column (a bitwise move);
    /// the caller must not drop the source afterwards.
    ///
    /// # Safety
    /// `value` must point to a valid, initialized value of the exact type this
    /// column was created for.
    pub unsafe fn push(&mut self, value: *const u8) {
        self.reserve(1);
        let index = self.len;
        // SAFETY: we just reserved space, so `index < capacity`; `value` is a
        // valid source of `size` bytes per the caller's contract, and the dest
        // slot is uninitialized so a bitwise copy establishes a valid value.
        unsafe {
            let dst = self.get_ptr(index);
            core::ptr::copy_nonoverlapping(value, dst, self.item_layout.size());
        }
        self.len = index + 1;
    }

    /// Remove the element at `index` by swapping the last element into its
    /// place, dropping the removed element.
    ///
    /// # Safety
    /// `index < len`.
    pub unsafe fn swap_remove_and_drop(&mut self, index: usize) {
        debug_assert!(index < self.len);
        let size = self.item_layout.size();
        // SAFETY: `index < len`, so the pointer is in-bounds.
        let removed = unsafe { self.get_ptr(index) };
        if let Some(drop) = self.drop {
            // SAFETY: `removed` points at a valid, initialized element; dropping
            // it exactly once (and not using it afterwards) is sound.
            unsafe { drop(removed) };
        }
        let last = self.len - 1;
        if index != last {
            // SAFETY: both `index` and `last` are in-bounds (possibly in
            // different blocks); the source slot is being logically removed so
            // moving its bytes over the (already dropped) `index` slot leaves
            // exactly one owner.
            unsafe {
                let src = self.get_ptr(last);
                core::ptr::copy_nonoverlapping(src, removed, size);
            }
        }
        self.len = last;
        self.shrink_blocks_to_len();
    }

    /// Remove the element at `index` by swapping the last element into its
    /// place, copying the removed element's bytes to `dst` (transferring
    /// ownership to the caller) instead of dropping it.
    ///
    /// This is how a component value is moved from one archetype's column to
    /// another's during a structural change.
    ///
    /// # Safety
    /// `index < len`, and `dst` must be valid for writes of
    /// `item_layout.size()` bytes and properly aligned. The caller takes
    /// ownership of the value written to `dst`.
    pub unsafe fn swap_remove_and_copy_out(&mut self, index: usize, dst: *mut u8) {
        debug_assert!(index < self.len);
        let size = self.item_layout.size();
        // SAFETY: `index < len`, so in-bounds.
        let removed = unsafe { self.get_ptr(index) };
        // SAFETY: `removed` is a valid initialized element and `dst` is valid
        // for `size` bytes per the caller's contract; this moves the value out.
        unsafe { core::ptr::copy_nonoverlapping(removed, dst, size) };
        let last = self.len - 1;
        if index != last {
            // SAFETY: see `swap_remove_and_drop`; both slots are in-bounds.
            unsafe {
                let src = self.get_ptr(last);
                core::ptr::copy_nonoverlapping(src, removed, size);
            }
        }
        self.len = last;
        self.shrink_blocks_to_len();
    }

    /// Drop all live elements, leaving the column empty (blocks retained).
    pub fn clear(&mut self) {
        if let Some(drop) = self.drop {
            for i in 0..self.len {
                // SAFETY: `i < len`, so each pointer is a valid, initialized
                // element dropped exactly once.
                unsafe { drop(self.get_ptr(i)) };
            }
        }
        self.len = 0;
    }

    /// Relocate the element at `src_index` out of `src` and append it to
    /// `self`. The value is bitwise-copied, transferring ownership into `self`;
    /// the source slot is **not** removed or shrunk here, so after this call
    /// `src` holds a logically moved-out duplicate at `src_index` that must be
    /// reconciled by the caller via [`BlobVec::swap_remove_forget`] (never
    /// dropped). Used to relocate a component value from one archetype column
    /// to another during a structural change.
    ///
    /// # Safety
    /// `self` and `src` must have identical item layouts, and
    /// `src_index < src.len()`.
    pub unsafe fn push_from(&mut self, src: &BlobVec, src_index: usize) {
        debug_assert_eq!(self.item_layout.size(), src.item_layout.size());
        debug_assert_eq!(self.item_layout.align(), src.item_layout.align());
        debug_assert!(src_index < src.len);
        self.reserve(1);
        // SAFETY: we reserved a slot, so `self.len` is a valid write target;
        // `src_index < src.len()` per the contract, and the layouts match so
        // the copied bytes form a valid value in `self`. The source slot keeps
        // its bits (a moved-out duplicate) for the caller to forget later.
        unsafe {
            let dst = self.get_ptr(self.len);
            let src_ptr = src.get_ptr(src_index);
            core::ptr::copy_nonoverlapping(src_ptr, dst, self.item_layout.size());
        }
        self.len += 1;
    }

    /// Overwrite the element at `index` with the value at `value`, dropping
    /// the previous element. Used for last-wins component insertion.
    ///
    /// # Safety
    /// `index < len()`, and `value` points to a valid, initialized value of
    /// this column's type whose ownership transfers into the column.
    pub unsafe fn replace(&mut self, index: usize, value: *const u8) {
        debug_assert!(index < self.len);
        // SAFETY: `index < len`, so the slot is a valid initialized element.
        let slot = unsafe { self.get_ptr(index) };
        if let Some(drop) = self.drop {
            // SAFETY: `slot` is a valid initialized element dropped exactly once.
            unsafe { drop(slot) };
        }
        // SAFETY: `value` is a valid source of `size` bytes; the slot is now
        // logically uninitialized, so the copy establishes a new valid value.
        unsafe { core::ptr::copy_nonoverlapping(value, slot, self.item_layout.size()) };
    }

    /// Remove the element at `index` by moving the last element over it,
    /// WITHOUT dropping the element at `index`.
    ///
    /// Used after the value at `index` was already moved out via
    /// [`BlobVec::swap_remove_and_copy_out`] or [`BlobVec::push_from`]; running
    /// drop glue again would be a double free.
    ///
    /// # Safety
    /// `index < len()`, and the element at `index` must already have been
    /// moved out (its ownership transferred elsewhere).
    pub unsafe fn swap_remove_forget(&mut self, index: usize) {
        debug_assert!(index < self.len);
        let last = self.len - 1;
        if index != last {
            // SAFETY: both `index` and `last` are in-bounds; the `index` slot is
            // logically uninitialized (moved out), so overwriting it with the
            // last element and shrinking leaves exactly one owner.
            unsafe {
                let src = self.get_ptr(last);
                let dst = self.get_ptr(index);
                core::ptr::copy_nonoverlapping(src, dst, self.item_layout.size());
            }
        }
        self.len = last;
        self.shrink_blocks_to_len();
    }
}

impl Drop for BlobVec {
    fn drop(&mut self) {
        self.clear();
        if self.item_layout.size() == 0 || !self.owns_blocks {
            // Borrowed blocks are owned (and freed) by the table.
            return;
        }
        if self.blocks.is_empty() {
            return;
        }
        let layout = self.block_layout();
        while let Some(block) = self.blocks.pop() {
            // SAFETY: every block was allocated with `layout` by `push_block`
            // and has not been freed; all live values were dropped by `clear`.
            unsafe { dealloc(block.ptr.as_ptr(), layout) };
        }
    }
}

/// A dangling pointer with the given alignment, valid as an empty/ZST base.
fn dangling_aligned(align: usize) -> NonNull<u8> {
    debug_assert!(align.is_power_of_two());
    // SAFETY: a power-of-two alignment is a non-zero address, valid for a
    // dangling `NonNull` that is never dereferenced for a real read/write.
    unsafe { NonNull::new_unchecked(align as *mut u8) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::rc::Rc;
    use alloc::vec::Vec;
    use core::cell::Cell;

    fn layout_of<T>() -> Layout {
        Layout::new::<T>()
    }

    /// Drop glue used by the column tests.
    ///
    /// # Safety
    /// `p` must point at a valid, initialized `T` that is dropped exactly once.
    unsafe fn drop_t<T>(p: *mut u8) {
        // SAFETY: tests only push valid `T` values.
        unsafe { core::ptr::drop_in_place(p.cast::<T>()) }
    }

    #[test]
    fn push_and_read_u32() {
        let mut v = BlobVec::new(layout_of::<u32>(), None);
        for i in 0u32..10 {
            // SAFETY: pushing a valid u32.
            unsafe { v.push((&i as *const u32).cast::<u8>()) };
        }
        assert_eq!(v.len(), 10);
        for i in 0..10usize {
            // SAFETY: index in-bounds, column holds u32.
            let val = unsafe { *v.get_ptr(i).cast::<u32>() };
            assert_eq!(val, i as u32);
        }
    }

    #[test]
    fn swap_remove_moves_last_into_hole() {
        let mut v = BlobVec::new(layout_of::<u32>(), None);
        for i in 0u32..5 {
            // SAFETY: valid u32.
            unsafe { v.push((&i as *const u32).cast::<u8>()) };
        }
        // Remove index 1 (value 1); last (value 4) should move into slot 1.
        // SAFETY: index < len.
        unsafe { v.swap_remove_and_drop(1) };
        assert_eq!(v.len(), 4);
        let got: Vec<u32> = (0..4)
            // SAFETY: in-bounds u32 reads.
            .map(|i| unsafe { *v.get_ptr(i).cast::<u32>() })
            .collect();
        assert_eq!(got, alloc::vec![0, 4, 2, 3]);
    }

    #[test]
    fn drop_glue_runs_for_every_element() {
        let counter = Rc::new(Cell::new(0usize));
        struct Guard(Rc<Cell<usize>>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        {
            let mut v = BlobVec::new(layout_of::<Guard>(), Some(drop_t::<Guard> as DropFn));
            for _ in 0..3 {
                let g = core::mem::ManuallyDrop::new(Guard(counter.clone()));
                // SAFETY: moving a valid Guard into the column; source is
                // wrapped in ManuallyDrop so it is not double-dropped.
                unsafe { v.push((&*g as *const Guard).cast::<u8>()) };
            }
            // SAFETY: index < len. Dropping element 0 should run Guard::drop.
            unsafe { v.swap_remove_and_drop(0) };
            assert_eq!(counter.get(), 1);
        }
        // Dropping the column drops the remaining 2 guards.
        assert_eq!(counter.get(), 3);
    }

    #[test]
    fn copy_out_transfers_ownership_without_drop() {
        let counter = Rc::new(Cell::new(0usize));
        struct Guard(#[allow(dead_code)] Rc<Cell<usize>>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        let mut v = BlobVec::new(layout_of::<Guard>(), Some(drop_t::<Guard> as DropFn));
        let g = core::mem::ManuallyDrop::new(Guard(counter.clone()));
        // SAFETY: valid Guard, not double-dropped (ManuallyDrop source).
        unsafe { v.push((&*g as *const Guard).cast::<u8>()) };

        let mut out = core::mem::MaybeUninit::<Guard>::uninit();
        // SAFETY: index 0 < len; `out` is valid for writes of a Guard.
        unsafe { v.swap_remove_and_copy_out(0, out.as_mut_ptr().cast::<u8>()) };
        assert_eq!(v.len(), 0);
        // No drop happened during copy_out.
        assert_eq!(counter.get(), 0);
        // SAFETY: `out` was initialized by copy_out.
        let owned = unsafe { out.assume_init() };
        drop(owned);
        assert_eq!(counter.get(), 1);
    }

    #[test]
    fn zero_sized_items() {
        struct Zst;
        let mut v = BlobVec::new(layout_of::<Zst>(), None);
        for _ in 0..1000 {
            let z = Zst;
            // SAFETY: ZST push copies 0 bytes.
            unsafe { v.push((&z as *const Zst).cast::<u8>()) };
        }
        assert_eq!(v.len(), 1000);
        assert_eq!(v.block_count(), 0, "a ZST column never allocates blocks");
        // SAFETY: index < len.
        unsafe { v.swap_remove_and_drop(0) };
        assert_eq!(v.len(), 999);
    }

    #[test]
    fn borrowed_column_rejects_growth_without_attached_capacity() {
        let mut v = BlobVec::new_borrowed(layout_of::<u32>(), None, 4);
        assert_eq!(
            v.try_reserve(1),
            Err(BlobVecError::BorrowedCapacity {
                capacity: 0,
                required: 1,
            })
        );
    }

    #[test]
    fn borrowed_block_lifecycle_reports_invalid_operations() {
        let mut v = BlobVec::new_borrowed(layout_of::<u32>(), None, 4);
        // SAFETY: null is intentionally supplied to verify the checked error
        // path; no pointer is recorded on failure.
        let result = unsafe { v.try_attach_block(core::ptr::null_mut()) };
        assert_eq!(result, Err(BlobVecError::InvalidBlockPointer));
        assert_eq!(v.try_detach_block(), Err(BlobVecError::NoAttachedBlock));
    }

    #[test]
    fn block_layout_overflow_is_rejected_before_allocation() {
        let layout = Layout::from_size_align(isize::MAX as usize, 1).expect("valid layout");
        let mut v = BlobVec::new_with_rows_per_block(layout, None, 2);
        assert_eq!(v.try_reserve(1), Err(BlobVecError::BlockLayoutOverflow));
        assert_eq!(v.block_count(), 0);
    }

    // --- block (physical chunk) behaviour -----------------------------------

    #[test]
    fn rows_fill_blocks_and_blocks_grow_on_demand() {
        // 8 slots per block so the boundary is easy to cross.
        let mut v = BlobVec::new_with_rows_per_block(layout_of::<u64>(), None, 8);
        assert_eq!(v.rows_per_block(), 8);
        assert_eq!(v.block_count(), 0);
        assert_eq!(v.capacity(), 0);

        for i in 0u64..8 {
            // SAFETY: valid u64.
            unsafe { v.push((&i as *const u64).cast::<u8>()) };
        }
        assert_eq!(v.block_count(), 1);
        assert_eq!(v.capacity(), 8);

        // The 9th push crosses into a second block.
        let nine = 8u64;
        // SAFETY: valid u64.
        unsafe { v.push((&nine as *const u64).cast::<u8>()) };
        assert_eq!(v.block_count(), 2);
        assert_eq!(v.capacity(), 16);
        // Every value is still readable through row addressing.
        for i in 0..9usize {
            // SAFETY: i < len, column holds u64.
            assert_eq!(unsafe { *v.get_ptr(i).cast::<u64>() }, i as u64);
        }
    }

    #[test]
    fn cross_block_swap_remove_and_block_reclaim() {
        let mut v = BlobVec::new_with_rows_per_block(layout_of::<u32>(), None, 4);
        for i in 0u32..6 {
            // SAFETY: valid u32.
            unsafe { v.push((&i as *const u32).cast::<u8>()) };
        }
        assert_eq!(v.block_count(), 2); // rows 0..4 in block 0, 4..6 in block 1

        // Remove row 0: the last row (value 5, block 1) swaps into block 0.
        // SAFETY: 0 < len.
        unsafe { v.swap_remove_and_drop(0) };
        assert_eq!(v.len(), 5);
        // SAFETY: reads of the 5 live u32 rows.
        let got: Vec<u32> = (0..5)
            .map(|i| unsafe { *v.get_ptr(i).cast::<u32>() })
            .collect();
        assert_eq!(got, alloc::vec![5, 1, 2, 3, 4]);
        assert_eq!(v.block_count(), 2, "row 4 still lives in block 1");

        // Drain down to 3 rows -> block 1 empties and is reclaimed.
        for _ in 0..2 {
            // SAFETY: removing the current last row is always in-bounds.
            unsafe { v.swap_remove_and_drop(v.len() - 1) };
        }
        assert_eq!(v.len(), 3);
        assert_eq!(
            v.block_count(),
            1,
            "the second block is freed when it empties"
        );
    }

    #[test]
    fn slice_range_is_block_local() {
        let mut v = BlobVec::new_with_rows_per_block(layout_of::<u32>(), None, 4);
        for i in 0u32..6 {
            // SAFETY: valid u32.
            unsafe { v.push((&i as *const u32).cast::<u8>()) };
        }
        // SAFETY: [1, 4) is inside block 0.
        let inside = unsafe { v.as_slice_range::<u32>(1, 4) };
        assert_eq!(inside, [1, 2, 3]);
        // SAFETY: [4, 6) is inside block 1.
        let second = unsafe { v.as_slice_range::<u32>(4, 6) };
        assert_eq!(second, [4, 5]);
    }

    #[test]
    #[should_panic(expected = "straddles a block boundary")]
    fn slice_range_across_blocks_is_rejected() {
        let mut v = BlobVec::new_with_rows_per_block(layout_of::<u32>(), None, 4);
        for i in 0u32..6 {
            // SAFETY: valid u32.
            unsafe { v.push((&i as *const u32).cast::<u8>()) };
        }
        // [2, 5) crosses the 4-row block boundary.
        // SAFETY: in-bounds range (the panic is the block-boundary check).
        let _ = unsafe { v.as_slice_range::<u32>(2, 5) };
    }

    #[test]
    fn single_block_column_keeps_whole_slice() {
        let mut v = BlobVec::new_with_rows_per_block(layout_of::<u32>(), None, 8);
        for i in 0u32..3 {
            // SAFETY: valid u32.
            unsafe { v.push((&i as *const u32).cast::<u8>()) };
        }
        // SAFETY: a single-block column may be borrowed whole.
        let all = unsafe { v.as_slice::<u32>() };
        assert_eq!(all, [0, 1, 2]);
    }

    #[test]
    fn blocks_give_stable_addresses_on_growth() {
        let mut v = BlobVec::new_with_rows_per_block(layout_of::<u32>(), None, 4);
        for i in 0u32..4 {
            // SAFETY: valid u32.
            unsafe { v.push((&i as *const u32).cast::<u8>()) };
        }
        // SAFETY: row 0 is live.
        let before = unsafe { v.get_ptr(0) };
        // Crossing into a new block must not move the first block's values.
        let four = 4u32;
        // SAFETY: valid u32.
        unsafe { v.push((&four as *const u32).cast::<u8>()) };
        // SAFETY: row 0 is still live.
        let after = unsafe { v.get_ptr(0) };
        assert_eq!(
            before, after,
            "growing a column must not move existing blocks"
        );
        // SAFETY: row 0 holds a live u32.
        assert_eq!(unsafe { *after.cast::<u32>() }, 0);
    }

    /// Deterministic differential test: a long pseudo-random sequence of pushes,
    /// swap-removes and replaces must leave the block column byte-for-byte equal
    /// to a plain `Vec` reference, with the block count always exactly
    /// `ceil(len / rows_per_block)`. This exercises every block boundary,
    /// cross-block moves and block reclamation.
    #[test]
    fn randomized_ops_match_a_reference_vec() {
        // A tiny block (4 slots) makes boundaries and reclaims frequent.
        const RPB: usize = 4;
        let mut col = BlobVec::new_with_rows_per_block(layout_of::<u64>(), None, RPB);
        let mut reference: Vec<u64> = Vec::new();
        let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            rng >> 33
        };
        let mut counter: u64 = 0;

        for _ in 0..3_000 {
            match next() % 3 {
                // Push a fresh value.
                0 | 1 => {
                    counter += 1;
                    // SAFETY: pushing a valid u64.
                    unsafe { col.push((&counter as *const u64).cast::<u8>()) };
                    reference.push(counter);
                }
                // Swap-remove a random live row.
                _ => {
                    if reference.is_empty() {
                        continue;
                    }
                    let idx = (next() as usize) % reference.len();
                    // SAFETY: `idx < len`.
                    unsafe { col.swap_remove_and_drop(idx) };
                    reference.swap_remove(idx);
                }
            }

            assert_eq!(col.len(), reference.len());
            assert_eq!(
                col.block_count(),
                reference.len().div_ceil(RPB),
                "block count must track the live row count",
            );
            for (row, expected) in reference.iter().enumerate() {
                // SAFETY: `row < len` and the column holds u64.
                let got = unsafe { *col.get_ptr(row).cast::<u64>() };
                assert_eq!(
                    got,
                    *expected,
                    "row {row} diverged at len {}",
                    reference.len()
                );
            }
        }
    }

    /// Differential test for `push_from` + `swap_remove_forget` (the archetype
    /// relocation pair): moving rows between two block columns must reproduce
    /// the same values as moving them between two plain `Vec`s.
    #[test]
    fn randomized_relocations_match_a_reference() {
        const RPB: usize = 4;
        let mut src = BlobVec::new_with_rows_per_block(layout_of::<u64>(), None, RPB);
        let mut dst = BlobVec::new_with_rows_per_block(layout_of::<u64>(), None, RPB);
        let mut src_ref: Vec<u64> = Vec::new();
        let mut dst_ref: Vec<u64> = Vec::new();
        let mut rng: u64 = 0xD1B5_4A32_D192_ED03;
        let mut next = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            rng >> 33
        };
        for i in 0..40u64 {
            // SAFETY: valid u64.
            unsafe { src.push((&i as *const u64).cast::<u8>()) };
            src_ref.push(i);
        }

        for _ in 0..500 {
            if src_ref.is_empty() {
                break;
            }
            let idx = (next() as usize) % src_ref.len();
            // Move `src[idx]` into `dst` (copy out, then forget the hole).
            // SAFETY: identical layouts and `idx < src.len()`.
            unsafe { dst.push_from(&src, idx) };
            dst_ref.push(src_ref[idx]);
            // SAFETY: `idx < src.len()` and the value was just moved out.
            unsafe { src.swap_remove_forget(idx) };
            src_ref.swap_remove(idx);

            assert_eq!(src.len(), src_ref.len());
            assert_eq!(dst.len(), dst_ref.len());
            for (row, expected) in src_ref.iter().enumerate() {
                // SAFETY: `row < len`.
                assert_eq!(unsafe { *src.get_ptr(row).cast::<u64>() }, *expected);
            }
            for (row, expected) in dst_ref.iter().enumerate() {
                // SAFETY: `row < len`.
                assert_eq!(unsafe { *dst.get_ptr(row).cast::<u64>() }, *expected);
            }
        }
    }
}
