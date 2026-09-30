//! Half-open binary-search range queries over an already-sorted `u32` slice:
//! locate insertion points and equal-value runs in `O(log n)` comparisons
//! (design §12 sort ordering, ordered lookup).
//!
//! A `GPU`-driven `VFX` pipeline repeatedly *consumes* a sequence that some
//! earlier stage already produced in sorted order — quantized view-depth keys,
//! sort-key buckets, spatial-hash cell ranges, or event timestamps — and then
//! asks positional questions about it: "where would this key insert", "how many
//! particles share this exact key", "is this key present at all". This module
//! owns the serial reference for those questions. Every entry point assumes the
//! caller's precondition holds: `data` is sorted in non-decreasing order by the
//! projected `u32` key. When that holds, each query costs `O(log n)`
//! comparisons; when it does not, the results are unspecified but still memory
//! safe (no panics, no out-of-bounds).
//!
//! The vocabulary is the classic half-open one. [`lower_bound`] returns the
//! index of the first element `>= target`, which is also the left-most position
//! at which `target` can be inserted while keeping the slice sorted.
//! [`upper_bound`] returns the index of the first element `> target`, the
//! right-most such insertion point. [`equal_range`] returns the pair
//! `(lower_bound, upper_bound)`, a half-open interval `[lo, hi)` naming exactly
//! the run of elements equal to `target`; the run is empty precisely when
//! `lo == hi`. [`contains`] reports whether that run is non-empty. The
//! `by_key` variants — [`lower_bound_by_key`], [`upper_bound_by_key`], and
//! [`equal_range_by_key`] — apply the same logic to a slice of arbitrary `T`
//! through a `Fn(&T) -> u32` projection, so a Structure-of-Arrays record can be
//! searched on one integer field without copying.
//!
//! Every search uses the overflow-safe half-open loop `lo = 0`, `hi = len`,
//! `while lo < hi { mid = lo + (hi - lo) / 2; ... }`. The midpoint is computed
//! as `lo + (hi - lo) / 2` rather than `(lo + hi) / 2` so the intermediate sum
//! can never overflow `usize`, and the invariant that `lo <= hi <= len` is
//! preserved on every iteration, which is why the returned index is always a
//! valid insertion point in `0..=len`.
//!
//! This is intentionally distinct from its neighbours and must not be conflated
//! with them. [`super::quickselect_u32`] finds the `k`-th order statistic on
//! *unsorted* data by partitioning; it neither requires nor preserves sorted
//! order. [`super::radix_sort_u32`], [`super::merge_sort_stable`], and
//! [`super::heap_sort_u32`] each *produce* a sorted sequence. This module sits
//! downstream of all of them: it *consumes* an already-sorted sequence and only
//! *locates* positions within it, never reordering, never moving, never
//! selecting. Sorting produces order; this module reads it.
//!
//! Everything here is pure integer logic. Comparisons are `u32` `<`/`>=` on the
//! projected key and `usize` bookkeeping on the indices; no element is moved, no
//! key is divided, nothing casts to floating point, and no transcendental
//! functions appear anywhere in this module. Empty and single-element slices
//! are valid inputs and are handled without special-casing.

/// Return the index of the first element that is `>= target`, or `data.len()`
/// when every element is `< target`.
///
/// This is the left-most position at which `target` could be inserted while
/// keeping `data` sorted in non-decreasing order. Assumes `data` is already
/// sorted; the result is unspecified (but safe) otherwise.
#[must_use]
pub fn lower_bound(data: &[u32], target: u32) -> usize {
    let mut lo = 0usize;
    let mut hi = data.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if data[mid] < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Return the index of the first element that is `> target`, or `data.len()`
/// when every element is `<= target`.
///
/// This is the right-most position at which `target` could be inserted while
/// keeping `data` sorted in non-decreasing order. Assumes `data` is already
/// sorted; the result is unspecified (but safe) otherwise.
#[must_use]
pub fn upper_bound(data: &[u32], target: u32) -> usize {
    let mut lo = 0usize;
    let mut hi = data.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if data[mid] <= target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Return the half-open interval `[lo, hi)` of all elements equal to `target`,
/// as the pair `(lower_bound(data, target), upper_bound(data, target))`.
///
/// The interval is empty exactly when the two returned indices are equal, in
/// which case they both name the insertion point for `target`. Assumes `data`
/// is already sorted; the result is unspecified (but safe) otherwise.
#[must_use]
pub fn equal_range(data: &[u32], target: u32) -> (usize, usize) {
    let lo = lower_bound(data, target);
    // Search only the suffix `data[lo..]` for the upper bound: elements before
    // `lo` are strictly less than `target`, so they can never be `> target`
    // candidates and skipping them keeps the two halves balanced.
    let hi = lo + upper_bound(&data[lo..], target);
    (lo, hi)
}

/// Report whether `target` occurs anywhere in the sorted slice `data`.
///
/// Assumes `data` is already sorted; the result is unspecified (but safe)
/// otherwise.
#[must_use]
pub fn contains(data: &[u32], target: u32) -> bool {
    let i = lower_bound(data, target);
    i < data.len() && data[i] == target
}

/// Return the index of the first element whose projected key is `>= target`,
/// or `data.len()` when every projected key is `< target`.
///
/// Generalizes [`lower_bound`] to a slice of arbitrary `T` searched through the
/// `key` projection. Assumes `data` is sorted in non-decreasing order *by the
/// projected key*; the result is unspecified (but safe) otherwise.
pub fn lower_bound_by_key<T, F: Fn(&T) -> u32>(data: &[T], target: u32, key: F) -> usize {
    let mut lo = 0usize;
    let mut hi = data.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if key(&data[mid]) < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Return the index of the first element whose projected key is `> target`,
/// or `data.len()` when every projected key is `<= target`.
///
/// Generalizes [`upper_bound`] to a slice of arbitrary `T` searched through the
/// `key` projection. Assumes `data` is sorted in non-decreasing order *by the
/// projected key*; the result is unspecified (but safe) otherwise.
pub fn upper_bound_by_key<T, F: Fn(&T) -> u32>(data: &[T], target: u32, key: F) -> usize {
    let mut lo = 0usize;
    let mut hi = data.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if key(&data[mid]) <= target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Return the half-open interval `[lo, hi)` of all elements whose projected key
/// equals `target`, as `(lower_bound_by_key, upper_bound_by_key)`.
///
/// Generalizes [`equal_range`] to a slice of arbitrary `T` searched through the
/// `key` projection. Assumes `data` is sorted in non-decreasing order *by the
/// projected key*; the result is unspecified (but safe) otherwise.
pub fn equal_range_by_key<T, F: Fn(&T) -> u32>(data: &[T], target: u32, key: F) -> (usize, usize) {
    let lo = lower_bound_by_key(data, target, &key);
    let hi = lo + upper_bound_by_key(&data[lo..], target, &key);
    (lo, hi)
}

/// Report whether any element's projected key equals `target`.
///
/// Generalizes [`contains`] to a slice of arbitrary `T` searched through the
/// `key` projection. Assumes `data` is sorted in non-decreasing order *by the
/// projected key*; the result is unspecified (but safe) otherwise.
pub fn contains_by_key<T, F: Fn(&T) -> u32>(data: &[T], target: u32, key: F) -> bool {
    let i = lower_bound_by_key(data, target, &key);
    i < data.len() && key(&data[i]) == target
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Naive linear reference for the first index `>= target`.
    fn naive_lower(data: &[u32], target: u32) -> usize {
        let mut i = 0usize;
        while i < data.len() && data[i] < target {
            i += 1;
        }
        i
    }

    /// Naive linear reference for the first index `> target`.
    fn naive_upper(data: &[u32], target: u32) -> usize {
        let mut i = 0usize;
        while i < data.len() && data[i] <= target {
            i += 1;
        }
        i
    }

    /// A small deterministic linear-congruential generator so the large-array
    /// tests need no external randomness. Constants are the Numerical Recipes
    /// values.
    struct Lcg {
        state: u32,
    }

    impl Lcg {
        fn new(seed: u32) -> Self {
            Lcg { state: seed }
        }

        fn next_u32(&mut self) -> u32 {
            self.state = self
                .state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            self.state
        }
    }

    #[test]
    fn lower_bound_on_empty_is_zero() {
        assert_eq!(lower_bound(&[], 7), 0);
    }

    #[test]
    fn upper_bound_on_empty_is_zero() {
        assert_eq!(upper_bound(&[], 7), 0);
    }

    #[test]
    fn equal_range_on_empty_is_zero_zero() {
        assert_eq!(equal_range(&[], 7), (0, 0));
    }

    #[test]
    fn contains_on_empty_is_false() {
        assert!(!contains(&[], 7));
    }

    #[test]
    fn single_element_present() {
        let data = [5u32];
        assert_eq!(lower_bound(&data, 5), 0);
        assert_eq!(upper_bound(&data, 5), 1);
        assert_eq!(equal_range(&data, 5), (0, 1));
        assert!(contains(&data, 5));
    }

    #[test]
    fn single_element_target_below() {
        let data = [5u32];
        assert_eq!(lower_bound(&data, 3), 0);
        assert_eq!(upper_bound(&data, 3), 0);
        assert_eq!(equal_range(&data, 3), (0, 0));
        assert!(!contains(&data, 3));
    }

    #[test]
    fn single_element_target_above() {
        let data = [5u32];
        assert_eq!(lower_bound(&data, 9), 1);
        assert_eq!(upper_bound(&data, 9), 1);
        assert_eq!(equal_range(&data, 9), (1, 1));
        assert!(!contains(&data, 9));
    }

    #[test]
    fn all_identical_full_range() {
        let data = [4u32, 4, 4, 4, 4];
        assert_eq!(lower_bound(&data, 4), 0);
        assert_eq!(upper_bound(&data, 4), 5);
        assert_eq!(equal_range(&data, 4), (0, 5));
        assert!(contains(&data, 4));
    }

    #[test]
    fn all_identical_target_below() {
        let data = [4u32, 4, 4, 4, 4];
        assert_eq!(lower_bound(&data, 2), 0);
        assert_eq!(upper_bound(&data, 2), 0);
        assert_eq!(equal_range(&data, 2), (0, 0));
        assert!(!contains(&data, 2));
    }

    #[test]
    fn all_identical_target_above() {
        let data = [4u32, 4, 4, 4, 4];
        assert_eq!(lower_bound(&data, 6), 5);
        assert_eq!(upper_bound(&data, 6), 5);
        assert_eq!(equal_range(&data, 6), (5, 5));
        assert!(!contains(&data, 6));
    }

    #[test]
    fn target_less_than_all_elements() {
        let data = [10u32, 20, 30, 40];
        assert_eq!(lower_bound(&data, 1), 0);
        assert_eq!(upper_bound(&data, 1), 0);
        assert_eq!(equal_range(&data, 1), (0, 0));
        assert!(!contains(&data, 1));
    }

    #[test]
    fn target_greater_than_all_elements() {
        let data = [10u32, 20, 30, 40];
        assert_eq!(lower_bound(&data, 99), 4);
        assert_eq!(upper_bound(&data, 99), 4);
        assert_eq!(equal_range(&data, 99), (4, 4));
        assert!(!contains(&data, 99));
    }

    #[test]
    fn target_in_the_middle_present() {
        let data = [10u32, 20, 30, 40, 50];
        assert_eq!(lower_bound(&data, 30), 2);
        assert_eq!(upper_bound(&data, 30), 3);
        assert_eq!(equal_range(&data, 30), (2, 3));
        assert!(contains(&data, 30));
    }

    #[test]
    fn target_in_a_gap_returns_insertion_point() {
        let data = [10u32, 20, 30, 40, 50];
        assert_eq!(lower_bound(&data, 35), 3);
        assert_eq!(upper_bound(&data, 35), 3);
        assert_eq!(equal_range(&data, 35), (3, 3));
        assert!(!contains(&data, 35));
    }

    #[test]
    fn duplicate_run_in_the_middle() {
        let data = [1u32, 2, 2, 2, 3, 4];
        assert_eq!(lower_bound(&data, 2), 1);
        assert_eq!(upper_bound(&data, 2), 4);
        assert_eq!(equal_range(&data, 2), (1, 4));
        assert!(contains(&data, 2));
    }

    #[test]
    fn duplicate_run_at_the_front() {
        let data = [7u32, 7, 7, 8, 9];
        assert_eq!(lower_bound(&data, 7), 0);
        assert_eq!(upper_bound(&data, 7), 3);
        assert_eq!(equal_range(&data, 7), (0, 3));
        assert!(contains(&data, 7));
    }

    #[test]
    fn duplicate_run_at_the_back() {
        let data = [1u32, 2, 3, 9, 9, 9];
        assert_eq!(lower_bound(&data, 9), 3);
        assert_eq!(upper_bound(&data, 9), 6);
        assert_eq!(equal_range(&data, 9), (3, 6));
        assert!(contains(&data, 9));
    }

    #[test]
    fn equal_range_matches_lower_and_upper_everywhere() {
        let data = [0u32, 0, 1, 1, 1, 3, 5, 5, 8, 8, 8, 8, 12];
        for target in 0u32..=13 {
            let lo = lower_bound(&data, target);
            let hi = upper_bound(&data, target);
            assert_eq!(equal_range(&data, target), (lo, hi));
            assert!(lo <= hi);
        }
    }

    #[test]
    fn contains_agrees_with_equal_range_nonempty() {
        let data = [2u32, 4, 4, 6, 8, 8, 8, 10];
        for target in 0u32..=12 {
            let (lo, hi) = equal_range(&data, target);
            assert_eq!(contains(&data, target), lo < hi);
        }
    }

    #[test]
    fn bounds_match_naive_linear_scan() {
        let data = [1u32, 3, 3, 5, 7, 7, 7, 9, 11, 11];
        for target in 0u32..=13 {
            assert_eq!(lower_bound(&data, target), naive_lower(&data, target));
            assert_eq!(upper_bound(&data, target), naive_upper(&data, target));
        }
    }

    #[test]
    fn insertion_point_keeps_slice_sorted() {
        let data = [2u32, 4, 6, 8];
        for target in 0u32..=10 {
            let idx = lower_bound(&data, target);
            let mut extended = Vec::from(data);
            extended.insert(idx, target);
            let mut i = 1usize;
            while i < extended.len() {
                assert!(extended[i - 1] <= extended[i]);
                i += 1;
            }
        }
    }

    #[test]
    fn min_and_max_u32_boundaries() {
        let data = [0u32, 0, 1, u32::MAX - 1, u32::MAX, u32::MAX];
        assert_eq!(equal_range(&data, 0), (0, 2));
        assert_eq!(equal_range(&data, u32::MAX), (4, 6));
        assert!(contains(&data, u32::MAX));
        assert!(contains(&data, 0));
        assert_eq!(lower_bound(&data, u32::MAX - 1), 3);
    }

    #[derive(Clone, Copy, Debug)]
    struct Particle {
        depth_key: u32,
        payload: u16,
    }

    #[test]
    fn by_key_lower_upper_on_struct() {
        let data = [
            Particle {
                depth_key: 10,
                payload: 1,
            },
            Particle {
                depth_key: 20,
                payload: 2,
            },
            Particle {
                depth_key: 20,
                payload: 3,
            },
            Particle {
                depth_key: 30,
                payload: 4,
            },
        ];
        assert_eq!(lower_bound_by_key(&data, 20, |p| p.depth_key), 1);
        assert_eq!(upper_bound_by_key(&data, 20, |p| p.depth_key), 3);
        assert_eq!(equal_range_by_key(&data, 20, |p| p.depth_key), (1, 3));
        assert!(contains_by_key(&data, 30, |p| p.depth_key));
        assert!(!contains_by_key(&data, 25, |p| p.depth_key));
    }

    #[test]
    fn by_key_empty_and_singleton() {
        let empty: [Particle; 0] = [];
        assert_eq!(lower_bound_by_key(&empty, 5, |p| p.depth_key), 0);
        assert_eq!(equal_range_by_key(&empty, 5, |p| p.depth_key), (0, 0));
        assert!(!contains_by_key(&empty, 5, |p| p.depth_key));

        let one = [Particle {
            depth_key: 42,
            payload: 9,
        }];
        assert_eq!(lower_bound_by_key(&one, 42, |p| p.depth_key), 0);
        assert_eq!(upper_bound_by_key(&one, 42, |p| p.depth_key), 1);
        assert!(contains_by_key(&one, 42, |p| p.depth_key));
    }

    #[test]
    fn by_key_matches_scalar_bounds() {
        let keys = [1u32, 1, 4, 4, 4, 9, 12, 12];
        let recs: Vec<Particle> = keys
            .iter()
            .enumerate()
            .map(|(i, &k)| Particle {
                depth_key: k,
                payload: i as u16,
            })
            .collect();
        for target in 0u32..=14 {
            assert_eq!(
                lower_bound_by_key(&recs, target, |p| p.depth_key),
                lower_bound(&keys, target),
            );
            assert_eq!(
                upper_bound_by_key(&recs, target, |p| p.depth_key),
                upper_bound(&keys, target),
            );
            assert_eq!(
                equal_range_by_key(&recs, target, |p| p.depth_key),
                equal_range(&keys, target),
            );
        }
    }

    #[test]
    fn large_random_array_bounds_match_naive() {
        let mut rng = Lcg::new(0x1234_5678);
        let mut data: Vec<u32> = (0..2048).map(|_| rng.next_u32() % 500).collect();
        data.sort_unstable();
        let mut probe = Lcg::new(0x9abc_def0);
        for _ in 0..300 {
            let target = probe.next_u32() % 520;
            assert_eq!(lower_bound(&data, target), naive_lower(&data, target));
            assert_eq!(upper_bound(&data, target), naive_upper(&data, target));
            let (lo, hi) = equal_range(&data, target);
            assert_eq!(
                (lo, hi),
                (naive_lower(&data, target), naive_upper(&data, target))
            );
        }
    }

    #[test]
    fn large_array_every_present_key_is_found() {
        let mut rng = Lcg::new(0x0f0f_0f0f);
        let mut data: Vec<u32> = (0..1024).map(|_| rng.next_u32() % 200).collect();
        data.sort_unstable();
        for &k in &data {
            assert!(contains(&data, k));
            let (lo, hi) = equal_range(&data, k);
            assert!(lo < hi);
            let mut i = lo;
            while i < hi {
                assert_eq!(data[i], k);
                i += 1;
            }
        }
    }

    #[test]
    fn large_array_run_lengths_sum_to_len() {
        let mut rng = Lcg::new(0xdead_beef);
        let mut data: Vec<u32> = (0..1500).map(|_| rng.next_u32() % 64).collect();
        data.sort_unstable();
        let mut total = 0usize;
        for target in 0u32..64 {
            let (lo, hi) = equal_range(&data, target);
            total += hi - lo;
        }
        assert_eq!(total, data.len());
    }

    #[test]
    fn bounds_are_monotonic_in_target() {
        let data = [3u32, 3, 5, 8, 8, 8, 13, 21];
        let mut prev_lo = 0usize;
        let mut prev_up = 0usize;
        for target in 0u32..=25 {
            let lo = lower_bound(&data, target);
            let up = upper_bound(&data, target);
            assert!(lo >= prev_lo);
            assert!(up >= prev_up);
            assert!(lo <= up);
            prev_lo = lo;
            prev_up = up;
        }
    }

    #[test]
    fn two_element_slice_all_targets() {
        let data = [4u32, 8];
        assert_eq!(equal_range(&data, 3), (0, 0));
        assert_eq!(equal_range(&data, 4), (0, 1));
        assert_eq!(equal_range(&data, 5), (1, 1));
        assert_eq!(equal_range(&data, 8), (1, 2));
        assert_eq!(equal_range(&data, 9), (2, 2));
    }

    #[test]
    fn contains_by_key_on_large_projection() {
        let mut rng = Lcg::new(0x5555_aaaa);
        let mut keys: Vec<u32> = (0..600).map(|_| rng.next_u32() % 300).collect();
        keys.sort_unstable();
        let recs: Vec<Particle> = keys
            .iter()
            .map(|&k| Particle {
                depth_key: k,
                payload: 0,
            })
            .collect();
        for target in 0u32..=305 {
            assert_eq!(
                contains_by_key(&recs, target, |p| p.depth_key),
                contains(&keys, target),
            );
        }
    }
}
