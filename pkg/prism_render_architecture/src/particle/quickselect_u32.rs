//! In-place Hoare selection over `u32` keys: reorder a slice so the element
//! that would land at rank `k` in fully sorted order is placed at index `k`,
//! with every element to its left no greater and every element to its right no
//! smaller, in average linear time (design §12 sort ordering, order
//! statistics).
//!
//! A production `GPU` `VFX` pipeline frequently needs a single order statistic
//! rather than a total order: the median particle depth for a soft-cutoff, the
//! `k`-th nearest neighbour distance for a density estimate, or a percentile
//! threshold for adaptive culling. Fully sorting the buffer to read one value
//! wastes work. Quickselect answers "what is the `k`-th smallest key" by
//! repeatedly partitioning the live sub-range around a pivot and recursing only
//! into the side that still contains rank `k`. Because one side is discarded
//! every step, the expected cost is `O(n)` rather than the `O(n log n)` of a
//! full sort. This file owns the serial reference so an eventual device kernel
//! can be checked value for value.
//!
//! Pivot choice is deterministic: [`median_of_three`] inspects the first,
//! middle, and last keys of the active range and returns the index of their
//! median, which sidesteps the quadratic blow-up on already-sorted or reversed
//! input without any random dependency. Partitioning uses the Lomuto scheme in
//! [`partition`], walking a single write cursor with a `while` loop so the
//! index arithmetic stays explicit. Runs are fully reproducible: the same slice
//! and `k` always visit the same pivots.
//!
//! Two entry points are exposed. [`quickselect`] reorders `data` in place and
//! returns the `k`-th smallest key (`0`-based), or `None` when `k` is out of
//! range. [`median`] returns the lower median for the middle element: for odd
//! lengths it is the unique middle key, and for even lengths it is the lower of
//! the two central keys (index `len / 2 - 1` in sorted order), chosen because
//! it stays inside the `u32` domain with no rounding and no risk of overflow
//! that an averaging rule would introduce.
//!
//! This is intentionally distinct from its neighbours and must not be conflated
//! with them: [`super::heap_sort_u32`], [`super::radix_sort_u32`], and
//! [`super::merge_sort_stable`] each produce a *fully* ordered result, whereas
//! this module only guarantees the single rank `k` is in place and the two
//! partitions straddle it — it never promises a total order and does less work
//! for that reason. [`super::reservoir_sample`] draws a *random* subset of
//! elements; this module makes no random choice at all and computes an *exact*
//! order statistic over every input key. Selection, not sorting; partial, not
//! total.
//!
//! Everything is pure integer arithmetic: comparisons are `u32` `<=`/`>=`,
//! reordering is [`slice::swap`], and the pivot median is decided by three
//! comparisons. Nothing divides a key, nothing casts to floating point, and no
//! transcendental functions appear anywhere in this module. Empty and
//! single-element inputs are valid and handled without recursion.

/// Return the index of the median of the three keys at `lo`, the midpoint of
/// `lo..=hi`, and `hi`. The result is always one of those three indices, so a
/// caller can use it directly as a pivot position. Requires `lo <= hi` and
/// both to be valid indices of `data`.
///
/// Only three comparisons are performed and no element is moved; this is a pure
/// index computation used by [`quickselect`] to keep pivots well-centred on
/// sorted or reversed input.
pub fn median_of_three(data: &[u32], lo: usize, hi: usize) -> usize {
    let mid = lo + (hi - lo) / 2;
    let a = data[lo];
    let b = data[mid];
    let c = data[hi];
    // Return the index whose value is the middle of the three.
    if a <= b {
        if b <= c {
            mid
        } else if a <= c {
            hi
        } else {
            lo
        }
    } else if a <= c {
        lo
    } else if b <= c {
        hi
    } else {
        mid
    }
}

/// Lomuto partition of `data[lo..=hi]` around the pivot initially stored at
/// `pivot`. The pivot key is first swapped to `hi`, then a single write cursor
/// sweeps the range moving every key no greater than the pivot to the front;
/// finally the pivot is swapped into the cursor position. Returns the final
/// resting index of the pivot, at which point every key in `data[lo..p]` is
/// `<=` the pivot and every key in `data[p + 1..=hi]` is `>=` it.
///
/// Requires `lo <= pivot <= hi` with `hi` a valid index of `data`.
pub fn partition(data: &mut [u32], lo: usize, hi: usize, pivot: usize) -> usize {
    data.swap(pivot, hi);
    let pivot_key = data[hi];
    let mut store = lo;
    let mut scan = lo;
    // Walk the range with an explicit cursor; a range-for would trip
    // `needless_range_loop` because both `scan` and `store` advance here.
    while scan < hi {
        if data[scan] <= pivot_key {
            data.swap(store, scan);
            store += 1;
        }
        scan += 1;
    }
    data.swap(store, hi);
    store
}

/// Reorder `data` in place so the `k`-th smallest key (`0`-based) sits at index
/// `k`, and return that key. Returns `None` when `data` is empty or `k` is not
/// a valid index (`k >= data.len()`).
///
/// The active range is narrowed iteratively: each step picks a
/// [`median_of_three`] pivot, partitions with [`partition`], and then keeps
/// only the side that still contains rank `k`. Because one side is dropped
/// every iteration the expected work is linear. After the call the element at
/// every index below `k` is `<=` the returned key and every index above `k` is
/// `>=` it, though neither side is otherwise ordered.
pub fn quickselect(data: &mut [u32], k: usize) -> Option<u32> {
    let len = data.len();
    if k >= len {
        return None;
    }
    let mut lo = 0usize;
    let mut hi = len - 1;
    loop {
        if lo == hi {
            return Some(data[lo]);
        }
        let pivot = median_of_three(data, lo, hi);
        let p = partition(data, lo, hi, pivot);
        if p == k {
            return Some(data[p]);
        } else if k < p {
            // Target lies strictly left of the pivot; drop the right side.
            hi = p - 1;
        } else {
            // Target lies strictly right of the pivot; drop the left side.
            lo = p + 1;
        }
    }
}

/// Return the lower median of `data`, reordering it in place, or `None` when
/// `data` is empty. For odd lengths this is the unique middle key; for even
/// lengths it is the lower of the two central keys — the element at sorted
/// index `len / 2 - 1` — chosen so the result is always an exact input key
/// inside the `u32` domain, with no rounding and no averaging overflow.
pub fn median(data: &mut [u32]) -> Option<u32> {
    let len = data.len();
    if len == 0 {
        return None;
    }
    let k = if len.is_multiple_of(2) {
        len / 2 - 1
    } else {
        len / 2
    };
    quickselect(data, k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// A small deterministic linear-congruential generator so the property
    /// tests need no external crate. Returns a fresh `u32` each call.
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u32(&mut self) -> u32 {
            // Numerical Recipes constants; full-period over `u64`.
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.state >> 32) as u32
        }
    }

    /// The `k`-th smallest via a full standard-library sort, for comparison.
    fn sorted_kth(keys: &[u32], k: usize) -> u32 {
        let mut v = keys.to_vec();
        v.sort_unstable();
        v[k]
    }

    /// Assert the post-condition of [`quickselect`]: everything left of `k` is
    /// `<=` the pivot key and everything right of `k` is `>=` it.
    fn assert_partitioned(data: &[u32], k: usize) {
        let key = data[k];
        for &x in &data[..k] {
            assert!(x <= key);
        }
        for &x in &data[k + 1..] {
            assert!(x >= key);
        }
    }

    #[test]
    fn empty_returns_none() {
        let mut data: Vec<u32> = Vec::new();
        assert_eq!(quickselect(&mut data, 0), None);
    }

    #[test]
    fn single_element_k0() {
        let mut data = [42u32];
        assert_eq!(quickselect(&mut data, 0), Some(42));
    }

    #[test]
    fn single_element_k_out_of_range() {
        let mut data = [42u32];
        assert_eq!(quickselect(&mut data, 1), None);
    }

    #[test]
    fn k0_is_minimum() {
        let mut data = [7u32, 3, 9, 1, 8, 4];
        assert_eq!(quickselect(&mut data, 0), Some(1));
        assert_eq!(data[0], 1);
    }

    #[test]
    fn k_last_is_maximum() {
        let mut data = [7u32, 3, 9, 1, 8, 4];
        let n = data.len();
        assert_eq!(quickselect(&mut data, n - 1), Some(9));
        assert_eq!(data[n - 1], 9);
    }

    #[test]
    fn middle_is_median_position() {
        let mut data = [5u32, 2, 8, 1, 9, 3, 7];
        // Sorted: 1 2 3 5 7 8 9; index 3 -> 5.
        assert_eq!(quickselect(&mut data, 3), Some(5));
    }

    #[test]
    fn k_equal_len_returns_none() {
        let mut data = [1u32, 2, 3];
        assert_eq!(quickselect(&mut data, 3), None);
    }

    #[test]
    fn k_far_out_of_range_returns_none() {
        let mut data = [1u32, 2, 3];
        assert_eq!(quickselect(&mut data, 999), None);
    }

    #[test]
    fn every_rank_matches_full_sort() {
        let base = [11u32, 4, 4, 9, 1, 7, 2, 8, 4, 6];
        for k in 0..base.len() {
            let mut data = base;
            let got = quickselect(&mut data, k).unwrap();
            assert_eq!(got, sorted_kth(&base, k));
            assert_eq!(data[k], got);
            assert_partitioned(&data, k);
        }
    }

    #[test]
    fn duplicates_present() {
        let mut data = [5u32, 5, 1, 5, 2, 5, 5];
        // Sorted: 1 2 5 5 5 5 5.
        assert_eq!(quickselect(&mut data, 0), Some(1));
        let mut d2 = [5u32, 5, 1, 5, 2, 5, 5];
        assert_eq!(quickselect(&mut d2, 1), Some(2));
        let mut d3 = [5u32, 5, 1, 5, 2, 5, 5];
        assert_eq!(quickselect(&mut d3, 4), Some(5));
    }

    #[test]
    fn all_equal_any_k() {
        let base = [8u32; 9];
        for k in 0..base.len() {
            let mut data = base;
            assert_eq!(quickselect(&mut data, k), Some(8));
        }
    }

    #[test]
    fn reversed_input_all_ranks() {
        let base: Vec<u32> = (0..64u32).rev().collect();
        for k in 0..base.len() {
            let mut data = base.clone();
            assert_eq!(quickselect(&mut data, k), Some(k as u32));
        }
    }

    #[test]
    fn already_sorted_input_all_ranks() {
        let base: Vec<u32> = (0..64u32).collect();
        for k in 0..base.len() {
            let mut data = base.clone();
            assert_eq!(quickselect(&mut data, k), Some(k as u32));
        }
    }

    #[test]
    fn two_elements_both_orders() {
        let mut a = [2u32, 1];
        assert_eq!(quickselect(&mut a, 0), Some(1));
        let mut b = [2u32, 1];
        assert_eq!(quickselect(&mut b, 1), Some(2));
    }

    #[test]
    fn u32_max_boundary() {
        let mut data = [u32::MAX, 0u32, u32::MAX - 1, 1u32];
        assert_eq!(quickselect(&mut data, 0), Some(0));
        let mut d2 = [u32::MAX, 0u32, u32::MAX - 1, 1u32];
        assert_eq!(quickselect(&mut d2, 3), Some(u32::MAX));
        let mut d3 = [u32::MAX, 0u32, u32::MAX - 1, 1u32];
        assert_eq!(quickselect(&mut d3, 2), Some(u32::MAX - 1));
    }

    #[test]
    fn u32_max_only() {
        let mut data = [u32::MAX; 5];
        assert_eq!(quickselect(&mut data, 2), Some(u32::MAX));
    }

    #[test]
    fn large_random_matches_full_sort_spot_ranks() {
        let mut rng = Lcg::new(0xDEAD_BEEF);
        let base: Vec<u32> = (0..5_000).map(|_| rng.next_u32()).collect();
        let mut reference = base.clone();
        reference.sort_unstable();
        for &k in &[0usize, 1, 1234, 2500, 3999, 4998, 4999] {
            let mut data = base.clone();
            let got = quickselect(&mut data, k).unwrap();
            assert_eq!(got, reference[k]);
            assert_eq!(data[k], got);
            assert_partitioned(&data, k);
        }
    }

    #[expect(
        clippy::needless_range_loop,
        reason = "loop index is the selection rank passed to quickselect, not merely an array cursor"
    )]
    #[test]
    fn large_random_every_rank_small() {
        let mut rng = Lcg::new(0x1234_5678);
        let base: Vec<u32> = (0..257).map(|_| rng.next_u32() % 40).collect();
        let mut reference = base.clone();
        reference.sort_unstable();
        for k in 0..base.len() {
            let mut data = base.clone();
            let got = quickselect(&mut data, k).unwrap();
            assert_eq!(got, reference[k]);
        }
    }

    #[test]
    fn large_random_bounded_domain() {
        let mut rng = Lcg::new(0x00C0_FFEE);
        let base: Vec<u32> = (0..3_000).map(|_| rng.next_u32() % 7).collect();
        let mut reference = base.clone();
        reference.sort_unstable();
        for &k in &[0usize, 500, 1500, 2999] {
            let mut data = base.clone();
            assert_eq!(quickselect(&mut data, k), Some(reference[k]));
        }
    }

    #[expect(
        clippy::needless_range_loop,
        reason = "loop index is the selection rank passed to quickselect, not merely an array cursor"
    )]
    #[test]
    fn multiple_ks_same_source_independent() {
        let base = [30u32, 10, 20, 50, 40, 60, 0, 70];
        let mut reference = base;
        reference.sort_unstable();
        for k in 0..base.len() {
            let mut data = base;
            assert_eq!(quickselect(&mut data, k), Some(reference[k]));
        }
    }

    #[test]
    fn median_odd_length() {
        let mut data = [5u32, 1, 9, 3, 7];
        // Sorted: 1 3 5 7 9; middle -> 5.
        assert_eq!(median(&mut data), Some(5));
    }

    #[test]
    fn median_even_length_is_lower_middle() {
        let mut data = [4u32, 1, 3, 2];
        // Sorted: 1 2 3 4; lower median is index 1 -> 2.
        assert_eq!(median(&mut data), Some(2));
    }

    #[test]
    fn median_single_element() {
        let mut data = [99u32];
        assert_eq!(median(&mut data), Some(99));
    }

    #[test]
    fn median_empty_is_none() {
        let mut data: Vec<u32> = Vec::new();
        assert_eq!(median(&mut data), None);
    }

    #[test]
    fn median_even_two_elements() {
        let mut data = [9u32, 4];
        // Sorted: 4 9; lower median index 0 -> 4.
        assert_eq!(median(&mut data), Some(4));
    }

    #[test]
    fn median_matches_sorted_definition_random() {
        let mut rng = Lcg::new(0x5EED_5EED);
        for len in [1usize, 2, 3, 8, 15, 16, 101, 256] {
            let base: Vec<u32> = (0..len).map(|_| rng.next_u32()).collect();
            let mut reference = base.clone();
            reference.sort_unstable();
            let expected = if len.is_multiple_of(2) {
                reference[len / 2 - 1]
            } else {
                reference[len / 2]
            };
            let mut data = base;
            assert_eq!(median(&mut data), Some(expected));
        }
    }

    #[test]
    fn median_of_three_picks_middle_value() {
        // Values 3, 1, 2 at indices 0, 1, 2; median value is 2 at index 2.
        let data = [3u32, 1, 2];
        assert_eq!(median_of_three(&data, 0, 2), 2);
    }

    #[test]
    fn median_of_three_all_orderings() {
        // Every permutation of distinct values must return the index of value 2.
        let perms = [
            [1u32, 2, 3],
            [1, 3, 2],
            [2, 1, 3],
            [2, 3, 1],
            [3, 1, 2],
            [3, 2, 1],
        ];
        for p in &perms {
            let idx = median_of_three(p, 0, 2);
            assert_eq!(p[idx], 2);
        }
    }

    #[test]
    fn partition_post_condition_holds() {
        let mut data = [7u32, 2, 9, 1, 5, 8, 3];
        let hi = data.len() - 1;
        let p = partition(&mut data, 0, hi, 3);
        let key = data[p];
        for &x in &data[..p] {
            assert!(x <= key);
        }
        for &x in &data[p + 1..] {
            assert!(x >= key);
        }
    }

    #[test]
    fn quickselect_does_not_lose_elements() {
        let mut rng = Lcg::new(0xABCD_0001);
        let base: Vec<u32> = (0..500).map(|_| rng.next_u32() % 100).collect();
        let mut reference = base.clone();
        reference.sort_unstable();
        let mut data = base;
        let _ = quickselect(&mut data, 123);
        data.sort_unstable();
        assert_eq!(data, reference);
    }
}
