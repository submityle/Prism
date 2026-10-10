//! [`AoSoa`]: an array-of-structures-of-arrays tiled columnar container.
//!
//! Flat [`SoaVec`](crate::soa::SoaVec) stores every field as one long column,
//! which vectorises perfectly but, once a batch pass reads several fields of
//! the same row, scatters those fields across far-apart cache lines. Pure
//! array-of-structs has the opposite problem. `AoSoA` is the hybrid both
//! AAA engines and SIMD math libraries converge on: rows are grouped into
//! fixed-width **blocks** (tiles) of `LANE` rows, each block stores its rows in
//! structure-of-arrays form, and the blocks themselves form an array. A kernel
//! therefore walks block by block — one block's columns are a dense, aligned,
//! `LANE`-wide `SIMD` lane for every field, and all of a block's fields stay in
//! the same local window of memory.
//!
//! [`AoSoa`] composes one [`SoaVec`](crate::soa::SoaVec) per block, so it is
//! derive-free, entirely safe code, and `no_std` + `alloc` compatible like the
//! rest of the [`layout`](crate::layout) module. `LANE` is a const generic (the
//! `SIMD` width, e.g. 4 / 8 / 16) and must be non-zero.
//!
//! ```
//! use prism_utils::layout::AoSoa;
//!
//! // Four rows per block; two columns (id, weight).
//! let mut v: AoSoa<(u32, f32), 4> = AoSoa::new();
//! for i in 0..10u32 {
//!     v.push((i, i as f32 * 0.5));
//! }
//! assert_eq!(v.len(), 10);
//! // 10 rows / 4 per block => 3 blocks (4 + 4 + 2).
//! assert_eq!(v.block_count(), 3);
//! assert_eq!(v.lane_width(), 4);
//!
//! // Block-wise batch pass: each block's columns are a dense SIMD lane.
//! let mut sum = 0.0f32;
//! for block in v.blocks() {
//!     let (_ids, weights) = block.columns();
//!     for &w in weights {
//!         sum += w;
//!     }
//! }
//! assert_eq!(sum, (0..10).map(|i| i as f32 * 0.5).sum());
//!
//! // Random access still resolves to (block, offset).
//! assert_eq!(v.get(5), Some((&5, &2.5)));
//! ```

extern crate alloc;

use alloc::vec::Vec;

use crate::soa::{Soa, SoaVec};

use super::plan::ColumnShapes;

/// An array-of-structures-of-arrays container: rows tiled into fixed-width
/// `LANE`-row blocks, each block held in structure-of-arrays form.
///
/// Every block except possibly the last holds exactly `LANE` rows; the last
/// block holds `1..=LANE` rows. Row `i` lives at offset `i % LANE` of block
/// `i / LANE`. [`push`](Self::push) / [`swap_remove`](Self::swap_remove) keep
/// that invariant, so the logical length is always well defined.
pub struct AoSoa<T: Soa, const LANE: usize> {
    blocks: Vec<SoaVec<T>>,
    len: usize,
}

impl<T: Soa, const LANE: usize> core::fmt::Debug for AoSoa<T, LANE> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AoSoa")
            .field("len", &self.len)
            .field("lane", &LANE)
            .field("blocks", &self.blocks.len())
            .finish_non_exhaustive()
    }
}

impl<T: Soa, const LANE: usize> Default for AoSoa<T, LANE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Soa, const LANE: usize> AoSoa<T, LANE> {
    /// Create an empty container. Panics at compile time if `LANE == 0`.
    #[must_use]
    pub fn new() -> Self {
        const { assert!(LANE > 0, "AoSoa LANE width must be non-zero") };
        Self {
            blocks: Vec::new(),
            len: 0,
        }
    }

    /// Create an empty container pre-sized to hold `rows` rows without
    /// reallocating the block directory. Panics at compile time if
    /// `LANE == 0`.
    #[must_use]
    pub fn with_capacity(rows: usize) -> Self {
        const { assert!(LANE > 0, "AoSoa LANE width must be non-zero") };
        Self {
            blocks: Vec::with_capacity(rows.div_ceil(LANE)),
            len: 0,
        }
    }

    /// The number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The block (tile) width in rows, i.e. the `LANE` const generic.
    #[must_use]
    pub fn lane_width(&self) -> usize {
        LANE
    }

    /// The number of allocated blocks (`ceil(len / LANE)`).
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// The number of blocks the block directory can hold before it reallocates,
    /// i.e. the capacity reserved by [`with_capacity`](Self::with_capacity).
    #[must_use]
    pub fn block_capacity(&self) -> usize {
        self.blocks.capacity()
    }

    /// Append a row, opening a fresh `LANE`-capacity block when the current
    /// tail block is full.
    pub fn push(&mut self, value: T) {
        let needs_block = match self.blocks.last() {
            Some(last) => last.len() == LANE,
            None => true,
        };
        if needs_block {
            self.blocks.push(SoaVec::with_capacity(LANE));
        }
        self.blocks
            .last_mut()
            .expect("a block was just ensured")
            .push(value);
        self.len += 1;
    }

    /// Remove row `index` by moving the last row into its place, returning the
    /// removed row, or `None` if `index` is out of bounds.
    ///
    /// Like [`Vec::swap_remove`] this does not preserve order; it keeps every
    /// block except the last fully packed in `O(1)`.
    pub fn swap_remove(&mut self, index: usize) -> Option<T> {
        if index >= self.len {
            return None;
        }
        let last = self.len - 1;
        // Pop the global last row (always the tail of the tail block).
        let tail = self.blocks.last_mut().expect("len > 0 implies a block");
        let tail_offset = tail.len() - 1;
        let last_row = tail
            .swap_remove(tail_offset)
            .expect("tail offset is in bounds");
        self.len -= 1;
        if tail.is_empty() {
            self.blocks.pop();
        }
        if index == last {
            return Some(last_row);
        }
        // Overwrite the target with the moved last row, returning its old value.
        let (b, off) = (index / LANE, index % LANE);
        let old = self.blocks[b]
            .replace(off, last_row)
            .expect("target offset is in bounds");
        Some(old)
    }

    /// Borrow row `index` as a tuple of shared references, or `None` if out of
    /// bounds.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<T::Ref<'_>> {
        if index >= self.len {
            return None;
        }
        self.blocks[index / LANE].get(index % LANE)
    }

    /// Borrow row `index` as a tuple of exclusive references, or `None` if out
    /// of bounds.
    pub fn get_mut(&mut self, index: usize) -> Option<T::RefMut<'_>> {
        if index >= self.len {
            return None;
        }
        self.blocks[index / LANE].get_mut(index % LANE)
    }

    /// Borrow the blocks as a slice, for a block-wise batch pass: each block's
    /// [`columns`](crate::soa::SoaVec::columns) is a dense `SIMD`-friendly lane
    /// of up to `LANE` rows.
    #[must_use]
    pub fn blocks(&self) -> &[SoaVec<T>] {
        &self.blocks
    }

    /// Exclusively borrow the blocks as a slice for an in-place batch pass.
    pub fn blocks_mut(&mut self) -> &mut [SoaVec<T>] {
        &mut self.blocks
    }

    /// Iterate every row as a tuple of shared references, block by block, in
    /// logical order.
    pub fn iter(&self) -> impl Iterator<Item = T::Ref<'_>> {
        self.blocks.iter().flat_map(SoaVec::iter)
    }

    /// Remove every row, keeping the allocated block directory for reuse.
    pub fn clear(&mut self) {
        self.blocks.clear();
        self.len = 0;
    }
}

impl<T: Soa + ColumnShapes, const LANE: usize> AoSoa<T, LANE> {
    /// The byte size of one fully packed block when each column is forced to at
    /// least `lane_align` (e.g. a `SIMD` lane width): the sum of every column's
    /// aligned element stride times `LANE`. Independent of the current row
    /// count; useful for pre-sizing arena backing storage.
    #[must_use]
    pub fn block_bytes(lane_align: usize) -> usize {
        let per_row: usize = T::column_shapes()
            .iter()
            .map(|s| s.stride_for(lane_align))
            .sum();
        per_row * LANE
    }
}
