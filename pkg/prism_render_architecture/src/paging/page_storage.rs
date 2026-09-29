//! Physical page-data placement: the flat byte layout streamed page contents
//! occupy in the bounded pool.
//!
//! [`PagePool`](crate::paging::PagePool) decides *which slot* a resident page
//! occupies; this layer decides *where in memory* that slot's bytes live and
//! moves page contents in and out. The pool is a flat array of `capacity`
//! fixed-size slots, each `page_words` 32-bit words, so slot `s`'s data is the
//! contiguous span `[s * page_words, (s + 1) * page_words)`. Streaming a page in
//! writes one slot's span ([`upload`]); evicting zeroes it ([`clear_slot`]); the
//! rasterizer reads a single word by slot and offset ([`fetch`]).
//!
//! Like the rest of the paging layer it is GPU-independent and deterministic:
//! the layout is a pure function of `(slot, word)`, so a GPU twin that scatters
//! the same uploads into a storage buffer and gathers the same reads produces
//! bit-identical words and can be diffed against this reference with no
//! tolerance.
//!
//! [`upload`]: PageStorage::upload
//! [`clear_slot`]: PageStorage::clear_slot
//! [`fetch`]: PageStorage::fetch

use alloc::vec;
use alloc::vec::Vec;

/// Why a page-storage placement could not be satisfied.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PageStorageError {
    /// The target slot index is at or beyond the pool's slot capacity.
    SlotOutOfRange {
        /// The offending slot index.
        slot: u32,
        /// Number of slots the pool holds.
        capacity: u32,
    },
    /// The uploaded page's word count does not match the pool's fixed page size.
    PageSizeMismatch {
        /// Words supplied by the caller.
        got: usize,
        /// Words one slot holds.
        expected: u32,
    },
}

impl core::fmt::Display for PageStorageError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PageStorageError::SlotOutOfRange { slot, capacity } => {
                write!(f, "slot {slot} is out of range for a {capacity}-slot pool")
            }
            PageStorageError::PageSizeMismatch { got, expected } => {
                write!(f, "uploaded page has {got} words but the pool page size is {expected}")
            }
        }
    }
}

/// A flat, fixed-stride physical page buffer: `capacity` slots of `page_words`
/// 32-bit words each.
///
/// The backing store starts zeroed - a freshly created pool holds no page data,
/// exactly as a GPU buffer `wgpu` zero-initialises does - and every operation is
/// addressed purely by `(slot, word)`, so the byte a slot occupies is a
/// deterministic function of the slot index alone.
#[derive(Clone, Debug)]
pub struct PageStorage {
    capacity: u32,
    page_words: u32,
    words: Vec<u32>,
}

impl PageStorage {
    /// Creates a zeroed pool of `capacity` slots, each `page_words` words wide.
    #[must_use]
    pub fn new(capacity: u32, page_words: u32) -> Self {
        let total = (capacity as usize) * (page_words as usize);
        Self {
            capacity,
            page_words,
            words: vec![0; total],
        }
    }

    /// Number of physical slots the pool holds.
    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Number of 32-bit words one slot holds.
    #[must_use]
    pub const fn page_words(&self) -> u32 {
        self.page_words
    }

    /// Writes one page's contents into `slot`.
    ///
    /// # Errors
    ///
    /// [`PageStorageError::SlotOutOfRange`] when `slot >= capacity`, and
    /// [`PageStorageError::PageSizeMismatch`] when `page.len() != page_words`.
    pub fn upload(&mut self, slot: u32, page: &[u32]) -> Result<(), PageStorageError> {
        if slot >= self.capacity {
            return Err(PageStorageError::SlotOutOfRange {
                slot,
                capacity: self.capacity,
            });
        }
        if page.len() != self.page_words as usize {
            return Err(PageStorageError::PageSizeMismatch {
                got: page.len(),
                expected: self.page_words,
            });
        }
        let base = (slot as usize) * (self.page_words as usize);
        self.words[base..base + page.len()].copy_from_slice(page);
        Ok(())
    }

    /// Zeroes `slot`'s span, modelling the page being evicted from the pool.
    ///
    /// Out-of-range slots are ignored so a whole eviction batch can be applied
    /// without pre-filtering.
    pub fn clear_slot(&mut self, slot: u32) {
        if slot >= self.capacity {
            return;
        }
        let base = (slot as usize) * (self.page_words as usize);
        for w in &mut self.words[base..base + self.page_words as usize] {
            *w = 0;
        }
    }

    /// Reads word `word` of `slot`, or [`None`] when either index is out of
    /// range.
    #[must_use]
    pub fn fetch(&self, slot: u32, word: u32) -> Option<u32> {
        if slot >= self.capacity || word >= self.page_words {
            return None;
        }
        let idx = (slot as usize) * (self.page_words as usize) + (word as usize);
        Some(self.words[idx])
    }

    /// Borrows `slot`'s full word span, or [`None`] when the slot is out of
    /// range.
    #[must_use]
    pub fn slot_words(&self, slot: u32) -> Option<&[u32]> {
        if slot >= self.capacity {
            return None;
        }
        let base = (slot as usize) * (self.page_words as usize);
        Some(&self.words[base..base + self.page_words as usize])
    }

    /// Borrows the whole flat backing store, slot-major.
    ///
    /// This is exactly the buffer image a GPU twin's pool storage buffer holds
    /// after applying the same uploads, so the two can be compared word-for-word.
    #[must_use]
    pub fn as_words(&self) -> &[u32] {
        &self.words
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_pool_is_zeroed() {
        let pool = PageStorage::new(3, 4);
        assert_eq!(pool.as_words(), &[0; 12]);
        assert_eq!(pool.fetch(2, 3), Some(0));
    }

    #[test]
    fn upload_places_page_at_slot_span() {
        let mut pool = PageStorage::new(3, 4);
        pool.upload(1, &[10, 11, 12, 13]).unwrap();
        assert_eq!(pool.slot_words(1), Some(&[10, 11, 12, 13][..]));
        // Neighbouring slots stay zeroed.
        assert_eq!(pool.slot_words(0), Some(&[0, 0, 0, 0][..]));
        assert_eq!(pool.fetch(1, 2), Some(12));
    }

    #[test]
    fn clear_slot_zeroes_only_that_span() {
        let mut pool = PageStorage::new(2, 3);
        pool.upload(0, &[1, 2, 3]).unwrap();
        pool.upload(1, &[4, 5, 6]).unwrap();
        pool.clear_slot(0);
        assert_eq!(pool.slot_words(0), Some(&[0, 0, 0][..]));
        assert_eq!(pool.slot_words(1), Some(&[4, 5, 6][..]));
    }

    #[test]
    fn upload_rejects_bad_slot_and_size() {
        let mut pool = PageStorage::new(2, 4);
        assert_eq!(
            pool.upload(2, &[0; 4]),
            Err(PageStorageError::SlotOutOfRange {
                slot: 2,
                capacity: 2
            })
        );
        assert_eq!(
            pool.upload(0, &[0; 3]),
            Err(PageStorageError::PageSizeMismatch {
                got: 3,
                expected: 4
            })
        );
    }

    #[test]
    fn fetch_out_of_range_is_none() {
        let pool = PageStorage::new(2, 4);
        assert_eq!(pool.fetch(2, 0), None);
        assert_eq!(pool.fetch(0, 4), None);
    }
}
