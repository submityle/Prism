//! Incrementally-updatable variable-size list metrics backed by a Fenwick tree.
//!
//! [`VariableList`](crate::VariableList) stores an immutable prefix-sum table:
//! queries are fast, but changing a single item's size costs `O(n)` to rebuild
//! the whole table. Real virtual scrollers measure item sizes lazily as rows
//! are rendered, so sizes change constantly. [`DynamicVariableList`] solves that
//! with a [Fenwick tree](https://en.wikipedia.org/wiki/Fenwick_tree) (binary
//! indexed tree): building is `O(n)`, and thereafter both resizing one item
//! ([`set`](DynamicVariableList::set)) and every query are `O(log n)`.
//!
//! It mirrors [`VariableList`](crate::VariableList)'s query semantics exactly —
//! [`offset_of`](DynamicVariableList::offset_of),
//! [`index_at`](DynamicVariableList::index_at) and
//! [`visible_range`](DynamicVariableList::visible_range) all agree with the
//! immutable version for the same sizes — so it is a drop-in replacement when
//! sizes are not fixed up front.

use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use crate::viewport::Viewport;

/// Variable-size list metrics supporting `O(log n)` incremental size updates.
///
/// Each item has a non-negative main-axis size. A Fenwick tree over those sizes
/// answers prefix-sum (offset) and inverse (pixel-to-index) queries in
/// logarithmic time, while [`set`](Self::set) updates a single size in
/// logarithmic time without rebuilding.
#[derive(Clone, Debug, PartialEq)]
pub struct DynamicVariableList {
    /// Per-item sizes, for `O(1)` [`size_of`](Self::size_of) and delta updates.
    sizes: Vec<f32>,
    /// One-based Fenwick tree; `tree[0]` is unused. `tree[i]` holds the sum of
    /// the `lowbit(i)` sizes ending at item `i`.
    tree: Vec<f32>,
}

/// The lowest set bit of `i` (its Fenwick block span). `i` must be non-zero.
#[inline]
fn lowbit(i: usize) -> usize {
    i & i.wrapping_neg()
}

impl DynamicVariableList {
    /// Builds a list from per-item main-axis sizes in `O(n)`.
    ///
    /// Sizes are expected to be non-negative so the cumulative offsets are
    /// monotonically non-decreasing, which [`index_at`](Self::index_at) relies
    /// on.
    #[must_use]
    pub fn from_sizes<I>(sizes: I) -> Self
    where
        I: IntoIterator<Item = f32>,
    {
        let sizes: Vec<f32> = sizes.into_iter().collect();
        let n = sizes.len();
        let mut tree = vec![0.0_f32; n + 1];
        // Linear build: seed each node with its own size, then fold it into its
        // Fenwick parent.
        for (i, &size) in sizes.iter().enumerate() {
            let node = i + 1;
            tree[node] += size;
            let parent = node + lowbit(node);
            if parent <= n {
                let carried = tree[node];
                tree[parent] += carried;
            }
        }
        Self { sizes, tree }
    }

    /// Number of items in the list.
    #[must_use]
    pub fn item_count(&self) -> usize {
        self.sizes.len()
    }

    /// Returns `true` when the list holds no items.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sizes.is_empty()
    }

    /// The main-axis size of the item at `index`.
    #[must_use]
    pub fn size_of(&self, index: usize) -> f32 {
        self.sizes[index]
    }

    /// Sets the size of the item at `index`, returning its previous size.
    ///
    /// Runs in `O(log n)` by applying the size delta along the Fenwick tree,
    /// never rebuilding the table.
    ///
    /// # Panics
    ///
    /// Panics when `index` is out of bounds.
    pub fn set(&mut self, index: usize, size: f32) -> f32 {
        assert!(index < self.sizes.len(), "index out of bounds");
        let old = self.sizes[index];
        let delta = size - old;
        self.sizes[index] = size;
        let mut node = index + 1;
        let n = self.sizes.len();
        while node <= n {
            self.tree[node] += delta;
            node += lowbit(node);
        }
        old
    }

    /// Cumulative size of the first `count` items (`count` in `0..=item_count`).
    fn prefix(&self, count: usize) -> f32 {
        let mut node = count;
        let mut sum = 0.0_f32;
        while node > 0 {
            sum += self.tree[node];
            node -= lowbit(node);
        }
        sum
    }

    /// The total main-axis extent of the whole list.
    #[must_use]
    pub fn total_size(&self) -> f32 {
        self.prefix(self.sizes.len())
    }

    /// The main-axis offset of the start of the item at `index`.
    ///
    /// Defined for `index` in `0..=item_count`; `offset_of(item_count)` returns
    /// the total size. Indices past the end clamp to the total size.
    #[must_use]
    pub fn offset_of(&self, index: usize) -> f32 {
        self.prefix(index.min(self.sizes.len()))
    }

    /// The index of the item containing `offset`.
    ///
    /// Offsets at or below `0` map to the first item; offsets at or beyond the
    /// total size map to the last item; a boundary offset belongs to the item
    /// starting there. An empty list always returns `0`. Matches
    /// [`VariableList::index_at`](crate::VariableList::index_at).
    #[must_use]
    pub fn index_at(&self, offset: f32) -> usize {
        let n = self.sizes.len();
        if n == 0 {
            return 0;
        }
        // Fenwick lower-bound walk: find the largest `pos` in `0..=n` whose
        // cumulative size `prefix(pos)` is `<= offset`. Non-negative sizes make
        // the cumulative sequence monotonic, so this is well defined.
        let mut pos = 0usize;
        let mut remaining = offset;
        let mut step = 1usize << (usize::BITS - 1 - n.leading_zeros());
        while step > 0 {
            let next = pos + step;
            if next <= n && self.tree[next] <= remaining {
                remaining -= self.tree[next];
                pos = next;
            }
            step >>= 1;
        }
        pos.min(n - 1)
    }

    /// The half-open range of item indices intersecting `viewport`, widened by
    /// the viewport's overscan and clamped to `0..item_count`.
    ///
    /// A viewport with a non-positive [`Viewport::length`] yields an empty range
    /// anchored at the clamped first visible index. Matches
    /// [`VariableList::visible_range`](crate::VariableList::visible_range).
    #[must_use]
    pub fn visible_range(&self, viewport: &Viewport) -> Range<usize> {
        let count = self.sizes.len();
        if count == 0 {
            return 0..0;
        }
        let first = self.index_at(viewport.start());
        let start = first.saturating_sub(viewport.overscan);
        if viewport.length <= 0.0 {
            return start..start;
        }
        let last = self.index_at(viewport.end());
        let end = (last + 1 + viewport.overscan).min(count);
        let start = start.min(end);
        start..end
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variable::VariableList;
    use alloc::vec;
    use alloc::vec::Vec;

    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next_u64() % n
        }
    }

    fn sample() -> DynamicVariableList {
        // sizes: 10, 15, 5, 20  => offsets: 0, 10, 25, 30, 50
        DynamicVariableList::from_sizes(vec![10.0, 15.0, 5.0, 20.0])
    }

    #[test]
    fn offsets_and_sizes_match_golden() {
        let list = sample();
        assert_eq!(list.item_count(), 4);
        assert_eq!(list.total_size(), 50.0);
        assert_eq!(list.offset_of(0), 0.0);
        assert_eq!(list.offset_of(1), 10.0);
        assert_eq!(list.offset_of(2), 25.0);
        assert_eq!(list.offset_of(3), 30.0);
        assert_eq!(list.offset_of(4), 50.0);
        assert_eq!(list.size_of(1), 15.0);
        assert_eq!(list.size_of(3), 20.0);
    }

    #[test]
    fn index_at_boundaries_match_golden() {
        let list = sample();
        assert_eq!(list.index_at(0.0), 0);
        assert_eq!(list.index_at(10.0), 1);
        assert_eq!(list.index_at(25.0), 2);
        assert_eq!(list.index_at(30.0), 3);
        assert_eq!(list.index_at(9.999), 0);
        assert_eq!(list.index_at(24.0), 1);
        assert_eq!(list.index_at(29.0), 2);
        assert_eq!(list.index_at(-5.0), 0);
        assert_eq!(list.index_at(50.0), 3);
        assert_eq!(list.index_at(1000.0), 3);
    }

    #[test]
    fn set_returns_old_and_updates_offsets() {
        let mut list = sample();
        let old = list.set(1, 100.0);
        assert_eq!(old, 15.0);
        assert_eq!(list.size_of(1), 100.0);
        // offsets: 0, 10, 110, 115, 135
        assert_eq!(list.offset_of(2), 110.0);
        assert_eq!(list.total_size(), 135.0);
    }

    #[test]
    fn empty_list_behaviour() {
        let list = DynamicVariableList::from_sizes(Vec::<f32>::new());
        assert_eq!(list.item_count(), 0);
        assert!(list.is_empty());
        assert_eq!(list.total_size(), 0.0);
        assert_eq!(list.index_at(0.0), 0);
        assert_eq!(list.index_at(100.0), 0);
        assert!(list.visible_range(&Viewport::new(0.0, 10.0)).is_empty());
    }

    #[test]
    fn set_agrees_with_rebuild() {
        // After an incremental update, the Fenwick state must equal a fresh
        // build from the same sizes.
        let mut list = sample();
        list.set(0, 7.0);
        list.set(3, 1.0);
        let rebuilt = DynamicVariableList::from_sizes(vec![7.0, 15.0, 5.0, 1.0]);
        assert_eq!(list, rebuilt);
    }

    #[test]
    fn matches_variable_list_under_random_updates() {
        // Cross-check every query against the independent immutable
        // `VariableList` oracle, after a sequence of random resizes. Integer
        // sizes keep all f32 sums exact, so equality comparison is sound.
        let mut rng = SplitMix64(0xDEAD_BEEF_1234_5678);
        for _ in 0..300 {
            let n = (rng.below(12) + 1) as usize;
            let mut sizes: Vec<f32> = (0..n)
                .map(|_| rng.below(6) as f32) // 0..=5, includes zero-size items
                .collect();

            let mut dynamic = DynamicVariableList::from_sizes(sizes.clone());

            // Apply a handful of random single-item resizes.
            let edits = rng.below(10);
            for _ in 0..edits {
                let idx = rng.below(n as u64) as usize;
                let new_size = rng.below(9) as f32;
                sizes[idx] = new_size;
                dynamic.set(idx, new_size);
            }

            let oracle = VariableList::from_sizes(sizes.clone());

            // Offsets and total.
            assert_eq!(dynamic.item_count(), oracle.item_count());
            assert_eq!(dynamic.total_size(), oracle.total_size());
            for k in 0..=n {
                assert_eq!(dynamic.offset_of(k), oracle.offset_of(k), "offset_of {k}");
            }

            // index_at across boundaries, interiors and out-of-range offsets.
            let total = oracle.total_size();
            for probe in 0..24 {
                let offset = -3.0 + (probe as f32) * (total + 6.0) / 20.0;
                assert_eq!(
                    dynamic.index_at(offset),
                    oracle.index_at(offset),
                    "index_at {offset} sizes={sizes:?}"
                );
            }

            // visible_range for a few random viewports.
            for _ in 0..6 {
                let start = rng.below(((total as u64) + 4).max(1)) as f32 - 1.0;
                let length = rng.below(((total as u64) + 4).max(1)) as f32;
                let overscan = rng.below(3) as usize;
                let vp = Viewport::new(start, length).with_overscan(overscan);
                assert_eq!(
                    dynamic.visible_range(&vp),
                    oracle.visible_range(&vp),
                    "visible_range start={start} length={length} overscan={overscan}"
                );
            }
        }
    }
}
