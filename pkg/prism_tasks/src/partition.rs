//! Fixed / stable work partitioning (design §13 固定分配, §24.7).
//!
//! Work-stealing splits work adaptively by worker count, so the *same* input
//! can be chopped differently from run to run. That is fine for an associative
//! combine, but it defeats bit-for-bit determinism: with a non-associative
//! combine (floating-point addition, say) a different split yields a different
//! result. [`FixedPartition`] removes that source of nondeterminism by
//! splitting a range into chunks whose boundaries depend **only** on the input
//! length and an explicit grain — never on the worker count, the steal order,
//! or wall-clock timing.
//!
//! The partition is the shared backbone of the deterministic reduce
//! ([`crate::reduce`]) and the replay log ([`crate::replay`]): record the
//! `(len, grain)` pair and you can reproduce the exact same split, and thus the
//! exact same result, on any machine with any number of workers.

use core::ops::Range;

/// Default number of chunks a [`FixedPartition::balanced`] aims for when the
/// caller does not pin an explicit grain. Chosen to give the scheduler several
/// independent chunks to balance across workers while keeping the reduction
/// tree shallow. It is a *fixed* constant precisely so the split never depends
/// on the machine's core count.
pub const DEFAULT_TARGET_CHUNKS: usize = 64;

/// A deterministic split of `0..len` into fixed-size chunks.
///
/// Every chunk has exactly `grain` elements except possibly the last, which
/// holds the remainder. Because both fields are explicit and the chunk
/// boundaries are a pure function of them, two [`FixedPartition`]s with equal
/// `len` and `grain` iterate identical ranges in identical order regardless of
/// how many workers consume them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedPartition {
    len: usize,
    grain: usize,
}

impl FixedPartition {
    /// Partition `0..len` into chunks of exactly `grain` elements (the final
    /// chunk takes the remainder). `grain` is clamped to at least `1`.
    ///
    /// # Examples
    /// ```
    /// use prism_tasks::FixedPartition;
    /// let p = FixedPartition::with_grain(10, 4);
    /// let chunks: Vec<_> = p.chunks().collect();
    /// assert_eq!(chunks, vec![0..4, 4..8, 8..10]);
    /// ```
    #[must_use]
    pub fn with_grain(len: usize, grain: usize) -> Self {
        Self {
            len,
            grain: grain.max(1),
        }
    }

    /// Partition `0..len` into at most `target_chunks` evenly sized chunks.
    ///
    /// The grain is derived as `len.div_ceil(target_chunks)`, so the number of
    /// chunks never exceeds `target_chunks` and the split depends only on `len`
    /// and `target_chunks`. `target_chunks` is clamped to at least `1`.
    #[must_use]
    pub fn with_target_chunks(len: usize, target_chunks: usize) -> Self {
        if len == 0 {
            return Self { len: 0, grain: 1 };
        }
        let grain = len.div_ceil(target_chunks.max(1));
        Self::with_grain(len, grain)
    }

    /// Partition `0..len` aiming for [`DEFAULT_TARGET_CHUNKS`] chunks.
    ///
    /// This is the default deterministic split: worker-count independent by
    /// construction, so a reduction over it is reproducible everywhere.
    #[must_use]
    pub fn balanced(len: usize) -> Self {
        Self::with_target_chunks(len, DEFAULT_TARGET_CHUNKS)
    }

    /// Total number of elements covered (`len`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the covered range is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The fixed chunk size (always at least `1`).
    #[must_use]
    pub fn grain(&self) -> usize {
        self.grain
    }

    /// Number of chunks the split produces.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        if self.len == 0 {
            0
        } else {
            self.len.div_ceil(self.grain)
        }
    }

    /// The half-open range of chunk `index`, or `None` if out of bounds.
    #[must_use]
    pub fn chunk(&self, index: usize) -> Option<Range<usize>> {
        if index >= self.chunk_count() {
            return None;
        }
        let start = index * self.grain;
        let end = (start + self.grain).min(self.len);
        Some(start..end)
    }

    /// Iterate the chunk ranges in index order (`0..grain`, `grain..2*grain`,
    /// ...). The order is fixed, so it is safe to zip with an index-keyed
    /// result buffer.
    pub fn chunks(&self) -> impl Iterator<Item = Range<usize>> + '_ {
        (0..self.chunk_count()).map(move |i| {
            let start = i * self.grain;
            let end = (start + self.grain).min(self.len);
            start..end
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_TARGET_CHUNKS, FixedPartition};

    #[test]
    fn grain_split_is_exact_with_remainder() {
        let p = FixedPartition::with_grain(10, 4);
        assert_eq!(p.chunk_count(), 3);
        assert_eq!(p.chunk(0), Some(0..4));
        assert_eq!(p.chunk(1), Some(4..8));
        assert_eq!(p.chunk(2), Some(8..10));
        assert_eq!(p.chunk(3), None);
    }

    #[test]
    fn empty_partition_has_no_chunks() {
        let p = FixedPartition::balanced(0);
        assert!(p.is_empty());
        assert_eq!(p.chunk_count(), 0);
        assert_eq!(p.chunks().count(), 0);
    }

    #[test]
    fn target_chunks_bounds_the_count() {
        let p = FixedPartition::with_target_chunks(1000, 8);
        assert!(p.chunk_count() <= 8);
        // Covers the whole range with no gaps or overlaps.
        let covered: usize = p.chunks().map(|r| r.len()).sum();
        assert_eq!(covered, 1000);
    }

    #[test]
    fn split_is_independent_of_everything_but_len_and_grain() {
        let a = FixedPartition::with_grain(777, 32);
        let b = FixedPartition::with_grain(777, 32);
        let ca: Vec<_> = a.chunks().collect();
        let cb: Vec<_> = b.chunks().collect();
        assert_eq!(ca, cb);
    }

    #[test]
    fn balanced_uses_the_fixed_default_target() {
        let p = FixedPartition::balanced(DEFAULT_TARGET_CHUNKS * 10);
        assert!(p.chunk_count() <= DEFAULT_TARGET_CHUNKS);
    }

    #[test]
    fn chunks_tile_the_range_in_order() {
        let p = FixedPartition::with_grain(100, 7);
        let mut expected = 0;
        for r in p.chunks() {
            assert_eq!(r.start, expected);
            expected = r.end;
        }
        assert_eq!(expected, 100);
    }
}
