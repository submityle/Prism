//! Deterministic shader permutation counting and selection.
//!
//! Shader variants are described as a mixed-radix space: each *axis* is a
//! compile-time feature switch with a fixed number of options (for example a
//! quality tier with three settings, or an on/off toggle with two). The total
//! number of permutations is the product of the per-axis option counts.
//!
//! A concrete variant is a [`PermutationKey`] — one chosen option per axis.
//! Keys map to a dense linear index in row-major order via
//! [`PermutationSpace::index_of`], and back again via
//! [`PermutationSpace::key_from_index`], so build systems can address variants
//! by a single integer while keeping selection deterministic.

use alloc::vec::Vec;

/// A mixed-radix description of a shader's permutation space.
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct PermutationSpace {
    /// Number of options on each axis; every entry is at least 1.
    axes: Vec<u32>,
}

impl PermutationSpace {
    /// Builds a space from per-axis option counts.
    ///
    /// Each count is clamped to a minimum of 1: an axis always has at least the
    /// default option, so an empty axis cannot silently zero the total.
    #[must_use]
    pub fn new(axis_option_counts: impl IntoIterator<Item = u32>) -> Self {
        let axes = axis_option_counts
            .into_iter()
            .map(|count| count.max(1))
            .collect();
        Self { axes }
    }

    /// Number of axes (feature switches) in the space.
    #[must_use]
    pub fn axis_count(&self) -> usize {
        self.axes.len()
    }

    /// Option count for a given axis, if it exists.
    #[must_use]
    pub fn axis_options(&self, axis: usize) -> Option<u32> {
        self.axes.get(axis).copied()
    }

    /// Total number of permutations (product of per-axis option counts).
    ///
    /// An empty space has exactly one permutation (the trivial variant). The
    /// product saturates at [`u64::MAX`] rather than overflowing.
    #[must_use]
    pub fn total_permutations(&self) -> u64 {
        let mut total: u64 = 1;
        for &count in &self.axes {
            total = total.saturating_mul(u64::from(count));
        }
        total
    }

    /// Computes the dense linear index of `key` within this space.
    ///
    /// Uses row-major (last axis varies fastest) ordering.
    ///
    /// # Errors
    ///
    /// Returns `None` when the key has the wrong number of coordinates or any
    /// coordinate is out of range for its axis.
    #[must_use]
    pub fn index_of(&self, key: &PermutationKey) -> Option<u64> {
        if key.coords.len() != self.axes.len() {
            return None;
        }
        let mut index: u64 = 0;
        for (coord, &count) in key.coords.iter().zip(&self.axes) {
            if *coord >= count {
                return None;
            }
            index = index.saturating_mul(u64::from(count)) + u64::from(*coord);
        }
        Some(index)
    }

    /// Reconstructs the [`PermutationKey`] at a dense linear `index`.
    ///
    /// # Errors
    ///
    /// Returns `None` when `index` is not less than [`Self::total_permutations`].
    #[must_use]
    pub fn key_from_index(&self, index: u64) -> Option<PermutationKey> {
        if index >= self.total_permutations() {
            return None;
        }
        let mut coords = alloc::vec![0u32; self.axes.len()];
        let mut remainder = index;
        // Decode from the fastest-varying (last) axis back to the first.
        for (slot, &count) in coords.iter_mut().rev().zip(self.axes.iter().rev()) {
            let count = u64::from(count);
            *slot = (remainder % count) as u32;
            remainder /= count;
        }
        Some(PermutationKey { coords })
    }
}

/// A concrete selection of one option per axis.
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct PermutationKey {
    coords: Vec<u32>,
}

impl PermutationKey {
    /// Builds a key from explicit per-axis option indices.
    #[must_use]
    pub fn new(coords: impl IntoIterator<Item = u32>) -> Self {
        Self {
            coords: coords.into_iter().collect(),
        }
    }

    /// Borrows the per-axis option indices.
    #[must_use]
    pub fn coords(&self) -> &[u32] {
        &self.coords
    }
}

/// Incrementally builds a [`PermutationKey`] one axis at a time.
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct PermutationSelector {
    coords: Vec<u32>,
}

impl PermutationSelector {
    /// Creates an empty selector.
    #[must_use]
    pub fn new() -> Self {
        Self { coords: Vec::new() }
    }

    /// Appends the chosen option for the next axis (builder style).
    #[must_use]
    pub fn select(mut self, option: u32) -> Self {
        self.coords.push(option);
        self
    }

    /// Appends the chosen option for the next axis, mutating in place.
    pub fn push(&mut self, option: u32) {
        self.coords.push(option);
    }

    /// Finalizes the accumulated selection into a [`PermutationKey`].
    #[must_use]
    pub fn finish(self) -> PermutationKey {
        PermutationKey {
            coords: self.coords,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_is_product_of_axes() {
        let space = PermutationSpace::new([2, 3, 4]);
        assert_eq!(space.axis_count(), 3);
        assert_eq!(space.total_permutations(), 24);
    }

    #[test]
    fn empty_space_has_single_permutation() {
        let space = PermutationSpace::new([]);
        assert_eq!(space.total_permutations(), 1);
        assert_eq!(space.key_from_index(0), Some(PermutationKey::new([])));
        assert!(space.key_from_index(1).is_none());
    }

    #[test]
    fn zero_option_axis_is_clamped_to_one() {
        let space = PermutationSpace::new([0, 5]);
        assert_eq!(space.axis_options(0), Some(1));
        assert_eq!(space.total_permutations(), 5);
    }

    #[test]
    fn index_key_round_trip_covers_whole_space() {
        let space = PermutationSpace::new([2, 3, 4]);
        let total = space.total_permutations();
        for index in 0..total {
            let key = space.key_from_index(index).unwrap();
            assert_eq!(space.index_of(&key), Some(index));
        }
    }

    #[test]
    fn index_of_rejects_out_of_range_and_wrong_arity() {
        let space = PermutationSpace::new([2, 3]);
        // Coordinate out of range on the second axis.
        assert!(space.index_of(&PermutationKey::new([1, 3])).is_none());
        // Wrong number of coordinates.
        assert!(space.index_of(&PermutationKey::new([0])).is_none());
    }

    #[test]
    fn selector_builds_expected_key() {
        let space = PermutationSpace::new([2, 3, 4]);
        let key = PermutationSelector::new()
            .select(1)
            .select(2)
            .select(3)
            .finish();
        assert_eq!(key.coords(), [1, 2, 3]);
        // Row-major: ((1*3 + 2) * 4 + 3) = 23, the last permutation.
        assert_eq!(space.index_of(&key), Some(23));
    }

    #[test]
    fn selector_push_matches_builder() {
        let mut selector = PermutationSelector::new();
        selector.push(0);
        selector.push(1);
        assert_eq!(selector.finish(), PermutationKey::new([0, 1]));
    }

    #[test]
    fn total_saturates_instead_of_overflowing() {
        let space = PermutationSpace::new([u32::MAX, u32::MAX, u32::MAX]);
        assert_eq!(space.total_permutations(), u64::MAX);
    }
}
