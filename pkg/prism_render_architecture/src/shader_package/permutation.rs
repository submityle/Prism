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

/// Checked dense index, not a packed mask. Decode against the owning space.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PermutationIndex(u64);

impl PermutationIndex {
    pub const ZERO: Self = Self(0);
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PermutationError {
    EmptyAxis(usize),
    Overflow,
    WrongArity,
    CoordinateOutOfRange(usize),
    IndexOutOfRange(u64),
}

/// A mixed-radix description of a shader's permutation space.
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct PermutationSpace {
    /// Number of options on each axis; every entry is at least 1.
    axes: Vec<u32>,
}

impl PermutationSpace {
    /// New runtime declarations reject zero axes and any u64 product overflow.
    pub fn try_new(
        axis_option_counts: impl IntoIterator<Item = u32>,
    ) -> Result<Self, PermutationError> {
        let axes: Vec<_> = axis_option_counts.into_iter().collect();
        if let Some(axis) = axes.iter().position(|count| *count == 0) {
            return Err(PermutationError::EmptyAxis(axis));
        }
        let space = Self { axes };
        space.checked_total()?;
        Ok(space)
    }

    pub fn checked_total(&self) -> Result<u64, PermutationError> {
        self.axes.iter().try_fold(1u64, |total, count| {
            total
                .checked_mul(u64::from(*count))
                .ok_or(PermutationError::Overflow)
        })
    }

    pub fn check_index(&self, raw: u64) -> Result<PermutationIndex, PermutationError> {
        if raw >= self.checked_total()? {
            return Err(PermutationError::IndexOutOfRange(raw));
        }
        Ok(PermutationIndex(raw))
    }

    pub fn checked_index_of(
        &self,
        key: &PermutationKey,
    ) -> Result<PermutationIndex, PermutationError> {
        self.checked_total()?;
        if key.coords.len() != self.axes.len() {
            return Err(PermutationError::WrongArity);
        }
        let mut index = 0u64;
        for (axis, (&coord, &count)) in key.coords.iter().zip(&self.axes).enumerate() {
            if coord >= count {
                return Err(PermutationError::CoordinateOutOfRange(axis));
            }
            index = index
                .checked_mul(u64::from(count))
                .and_then(|index| index.checked_add(u64::from(coord)))
                .ok_or(PermutationError::Overflow)?;
        }
        Ok(PermutationIndex(index))
    }
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
        self.checked_index_of(key).ok().map(PermutationIndex::get)
    }

    /// Reconstructs the [`PermutationKey`] at a dense linear `index`.
    ///
    /// # Errors
    ///
    /// Returns `None` when `index` is not less than [`Self::total_permutations`].
    #[must_use]
    pub fn key_from_index(&self, index: u64) -> Option<PermutationKey> {
        self.check_index(index).ok()?;
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
    fn checked_dense_indices_have_no_lobe_bit_window() {
        let space = PermutationSpace::try_new([2; 40]).unwrap();
        let key = PermutationKey::new([1; 40]);
        let index = space.checked_index_of(&key).unwrap();
        assert_eq!(index.get(), (1u64 << 40) - 1);
        assert_eq!(space.key_from_index(index.get()), Some(key));
        assert_eq!(
            space.check_index(1u64 << 40),
            Err(PermutationError::IndexOutOfRange(1u64 << 40))
        );
    }

    #[test]
    fn checked_space_rejects_overflow_even_for_zero_index() {
        assert_eq!(
            PermutationSpace::try_new([0, 2]),
            Err(PermutationError::EmptyAxis(0))
        );
        assert_eq!(
            PermutationSpace::try_new([u32::MAX; 3]),
            Err(PermutationError::Overflow)
        );
        let legacy = PermutationSpace::new([u32::MAX; 3]);
        assert_eq!(
            legacy.checked_index_of(&PermutationKey::new([0; 3])),
            Err(PermutationError::Overflow)
        );
        assert_eq!(legacy.index_of(&PermutationKey::new([0; 3])), None);
        assert_eq!(legacy.key_from_index(0), None);
        let space = PermutationSpace::try_new([2, 3]).unwrap();
        assert_eq!(
            space.checked_index_of(&PermutationKey::new([0])),
            Err(PermutationError::WrongArity)
        );
        assert_eq!(
            space.checked_index_of(&PermutationKey::new([0, 3])),
            Err(PermutationError::CoordinateOutOfRange(1))
        );
    }

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
