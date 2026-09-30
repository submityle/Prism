//! In-place binary max-heap sort over `u32` keys: the comparison-based serial
//! reference for ordering a particle key slice ascending with no auxiliary
//! buffer (design §12 sort ordering).
//!
//! Heapsort treats the slice as an implicit complete binary tree where node
//! `i` has children `2 * i + 1` and `2 * i + 2`. It runs in two phases. First
//! `build_max_heap` walks the internal nodes from the last parent down to the
//! root, calling `sift_down` on each so every subtree obeys the max-heap
//! property (each parent is at least as large as both children). Second, the
//! sort repeatedly swaps the root (the current maximum) to the end of the live
//! heap, shrinks the heap by one, and sifts the new root back down to restore
//! the property. After `n` such extractions the slice is sorted ascending in
//! place. The algorithm is not stable — equal keys may be reordered — but it
//! needs no extra allocation and always costs `O(n log n)` comparisons in the
//! worst case, which is its reason for existing alongside the average-case
//! faster but auxiliary-hungry alternatives.
//!
//! Three entry points are exposed. [`heap_sort`] sorts a `&mut [u32]` ascending
//! in place. [`heap_sort_by_key`] sorts a generic `&mut [T]` in place by a
//! caller-supplied `u32` key projection, so payloads move with their keys.
//! [`is_max_heap`] validates that a slice already satisfies the max-heap
//! property and is exposed both as a building block and for direct unit
//! testing. All child/parent index arithmetic uses `usize`.
//!
//! This is intentionally distinct from its sorting neighbours and must not be
//! conflated with them: [`super::radix_sort_u32`] is a non-comparison,
//! *stable*, multi-pass least-significant-`digit` (`LSD`) `radix` sort that
//! needs an auxiliary scatter buffer; [`super::counting_sort_u16`] is a
//! *stable* counting sort restricted to the 16-bit value domain; and
//! [`super::bitonic_sort`] is a `GPU`-friendly sorting network with a fixed,
//! data-independent compare-exchange schedule. This module is the in-place,
//! comparison-based `CPU` heapsort over the full `u32` key range.
//!
//! Everything is pure integer arithmetic: indices are `usize` shifts and adds,
//! comparisons are ordinary integer `<`, and element moves are
//! [`slice::swap`]. Empty and single-element inputs are valid no-ops. No
//! floating point and no transcendental functions appear anywhere in this
//! module.

/// Restore the max-heap property for the subtree rooted at `root`, assuming
/// both child subtrees are already valid max-heaps. Only the first `heap_len`
/// elements of `data` are considered part of the heap; anything at or beyond
/// `heap_len` is the already-sorted tail and is never touched. The node sinks
/// down toward the leaves, swapping with its larger child whenever a child is
/// strictly greater, until it dominates both children or becomes a leaf.
fn sift_down_by<T, F: Fn(&T) -> u32>(data: &mut [T], mut root: usize, heap_len: usize, key: &F) {
    loop {
        let left = 2 * root + 1;
        // A missing left child means `root` is a leaf within the live heap.
        if left >= heap_len {
            break;
        }
        // Pick the larger of the two children (right child only if present).
        let right = left + 1;
        let mut largest = left;
        if right < heap_len && key(&data[right]) > key(&data[left]) {
            largest = right;
        }
        // If the parent already dominates its larger child, the subtree is a
        // valid max-heap and the node has reached its resting place.
        if key(&data[root]) >= key(&data[largest]) {
            break;
        }
        data.swap(root, largest);
        root = largest;
    }
}

/// Transform `data` into a max-heap in place by sifting down every internal
/// node from the last parent up to the root. A slice of length `n` has its last
/// parent at index `n / 2 - 1`; empty and single-element slices are already
/// heaps and this returns immediately.
fn build_max_heap_by<T, F: Fn(&T) -> u32>(data: &mut [T], key: &F) {
    let len = data.len();
    if len < 2 {
        return;
    }
    // Iterate internal nodes high index to low so each `sift_down` sees valid
    // child heaps. A plain index walk is used because heapsort's node order is
    // an inherent index sequence, not a slice iteration.
    let mut node = len / 2;
    while node > 0 {
        node -= 1;
        sift_down_by(data, node, len, key);
    }
}

/// Sort `data` ascending in place by the `u32` produced by `key`, using an
/// in-place binary max-heap. Payloads move together with their keys. The sort
/// is not stable. Empty and single-element slices are left unchanged.
///
/// After [`build_max_heap_by`] the largest key sits at index `0`; each
/// iteration swaps it to the current end of the live heap, shrinks the heap,
/// and sifts the new root down to restore the max-heap property, so the sorted
/// suffix grows from the back until the whole slice is ordered.
pub fn heap_sort_by_key<T, F: Fn(&T) -> u32>(data: &mut [T], key: F) {
    let len = data.len();
    if len < 2 {
        return;
    }
    build_max_heap_by(data, &key);
    let mut heap_len = len;
    while heap_len > 1 {
        heap_len -= 1;
        // Move the current maximum to the front of the sorted suffix, then
        // repair the now-shrunken heap rooted at index `0`.
        data.swap(0, heap_len);
        sift_down_by(data, 0, heap_len, &key);
    }
}

/// Sort a `u32` slice ascending in place with an in-place binary max-heap.
/// This is the concrete `u32` specialization of [`heap_sort_by_key`] using the
/// identity key projection. The sort is not stable but needs no extra
/// allocation. Empty and single-element slices are left unchanged.
pub fn heap_sort(data: &mut [u32]) {
    heap_sort_by_key(data, |value| *value);
}

/// Report whether the first `data.len()` elements form a valid max-heap under
/// the `u32` key projection: every node's key is at least as large as each of
/// its present children's keys. Empty and single-element slices are trivially
/// max-heaps. This mirrors the invariant [`build_max_heap_by`] establishes and
/// [`sift_down_by`] maintains.
pub fn is_max_heap_by<T, F: Fn(&T) -> u32>(data: &[T], key: F) -> bool {
    let len = data.len();
    // Only internal nodes can violate the property; leaves have no children.
    // The last parent is at `len / 2 - 1`, so parents are exactly `0..len / 2`.
    for parent in 0..len / 2 {
        let left = 2 * parent + 1;
        let right = left + 1;
        if left < len && key(&data[left]) > key(&data[parent]) {
            return false;
        }
        if right < len && key(&data[right]) > key(&data[parent]) {
            return false;
        }
    }
    true
}

/// Report whether a `u32` slice forms a valid max-heap: the identity-key
/// specialization of [`is_max_heap_by`]. Empty and single-element slices are
/// trivially max-heaps.
pub fn is_max_heap(data: &[u32]) -> bool {
    is_max_heap_by(data, |value| *value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// A small deterministic linear-congruential generator so the property
    /// tests need no external crate. Returns a fresh integer each call.
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            // Numerical Recipes constants; full-period over `u64`.
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.state
        }

        fn next_u32(&mut self) -> u32 {
            (self.next_u64() >> 32) as u32
        }
    }

    /// Reference ascending sort via the standard library, for value-for-value
    /// comparison against the heapsort output.
    fn std_sorted(values: &[u32]) -> Vec<u32> {
        let mut out = values.to_vec();
        out.sort_unstable();
        out
    }

    /// True when `data` is non-decreasing front to back.
    fn is_ascending(data: &[u32]) -> bool {
        data.windows(2).all(|pair| pair[0] <= pair[1])
    }

    /// Multiset-equality check: same elements with same multiplicities,
    /// order-independent, so a sort that only permutes is detected.
    fn same_multiset(a: &[u32], b: &[u32]) -> bool {
        let mut sa = a.to_vec();
        let mut sb = b.to_vec();
        sa.sort_unstable();
        sb.sort_unstable();
        sa == sb
    }

    #[test]
    fn sort_empty_is_noop() {
        let mut data: [u32; 0] = [];
        heap_sort(&mut data);
        assert_eq!(data, [] as [u32; 0]);
    }

    #[test]
    fn sort_single_element() {
        let mut data = [42u32];
        heap_sort(&mut data);
        assert_eq!(data, [42]);
    }

    #[test]
    fn sort_two_elements_unordered() {
        let mut data = [9u32, 4];
        heap_sort(&mut data);
        assert_eq!(data, [4, 9]);
    }

    #[test]
    fn sort_two_elements_ordered() {
        let mut data = [4u32, 9];
        heap_sort(&mut data);
        assert_eq!(data, [4, 9]);
    }

    #[test]
    fn sort_two_equal_elements() {
        let mut data = [7u32, 7];
        heap_sort(&mut data);
        assert_eq!(data, [7, 7]);
    }

    #[test]
    fn sort_already_sorted() {
        let mut data = [1u32, 2, 3, 4, 5, 6, 7, 8];
        heap_sort(&mut data);
        assert_eq!(data, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn sort_reverse_sorted() {
        let mut data = [8u32, 7, 6, 5, 4, 3, 2, 1];
        heap_sort(&mut data);
        assert_eq!(data, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn sort_with_duplicates() {
        let mut data = [5u32, 1, 5, 3, 1, 3, 5, 2];
        heap_sort(&mut data);
        assert_eq!(data, [1, 1, 2, 3, 3, 5, 5, 5]);
    }

    #[test]
    fn sort_all_identical() {
        let mut data = [3u32; 16];
        heap_sort(&mut data);
        assert_eq!(data, [3u32; 16]);
    }

    #[test]
    fn sort_small_known_permutation() {
        let mut data = [3u32, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        heap_sort(&mut data);
        assert_eq!(data, [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]);
    }

    #[test]
    fn sort_odd_length() {
        let mut data = [10u32, 30, 20, 50, 40, 70, 60];
        heap_sort(&mut data);
        assert_eq!(data, [10, 20, 30, 40, 50, 60, 70]);
    }

    #[test]
    fn sort_even_length() {
        let mut data = [10u32, 30, 20, 50, 40, 70, 60, 80];
        heap_sort(&mut data);
        assert_eq!(data, [10, 20, 30, 40, 50, 60, 70, 80]);
    }

    #[test]
    fn sort_includes_zero() {
        let mut data = [4u32, 0, 2, 0, 1];
        heap_sort(&mut data);
        assert_eq!(data, [0, 0, 1, 2, 4]);
    }

    #[test]
    fn sort_u32_max_boundary() {
        let mut data = [u32::MAX, 0, u32::MAX, 1, u32::MAX - 1];
        heap_sort(&mut data);
        assert_eq!(data, [0, 1, u32::MAX - 1, u32::MAX, u32::MAX]);
    }

    #[test]
    fn sort_extremes_only() {
        let mut data = [u32::MAX, u32::MIN, u32::MAX, u32::MIN];
        heap_sort(&mut data);
        assert_eq!(data, [u32::MIN, u32::MIN, u32::MAX, u32::MAX]);
    }

    #[test]
    fn sort_matches_std_random_small() {
        let mut rng = Lcg::new(0x1234_5678);
        let data: Vec<u32> = (0..37).map(|_| rng.next_u32() % 100).collect();
        let mut sorted = data.clone();
        heap_sort(&mut sorted);
        assert_eq!(sorted, std_sorted(&data));
    }

    #[test]
    fn sort_matches_std_random_large() {
        let mut rng = Lcg::new(0xDEAD_BEEF);
        let data: Vec<u32> = (0..5_000).map(|_| rng.next_u32()).collect();
        let mut sorted = data.clone();
        heap_sort(&mut sorted);
        assert_eq!(sorted, std_sorted(&data));
    }

    #[test]
    fn sort_random_full_u32_range_is_ordered_permutation() {
        let mut rng = Lcg::new(0x0BAD_F00D);
        let data: Vec<u32> = (0..4_096).map(|_| rng.next_u32()).collect();
        let mut sorted = data.clone();
        heap_sort(&mut sorted);
        assert!(is_ascending(&sorted));
        assert!(same_multiset(&data, &sorted));
    }

    #[test]
    fn sort_random_narrow_domain_has_many_duplicates() {
        let mut rng = Lcg::new(0x5151_5151);
        let data: Vec<u32> = (0..3_000).map(|_| rng.next_u32() % 7).collect();
        let mut sorted = data.clone();
        heap_sort(&mut sorted);
        assert!(is_ascending(&sorted));
        assert!(same_multiset(&data, &sorted));
    }

    #[test]
    fn sort_is_idempotent() {
        let mut rng = Lcg::new(0x9999_1111);
        let data: Vec<u32> = (0..1_500).map(|_| rng.next_u32() % 500).collect();
        let mut once = data.clone();
        heap_sort(&mut once);
        let mut twice = once.clone();
        heap_sort(&mut twice);
        assert_eq!(once, twice);
    }

    #[test]
    fn sort_all_lengths_up_to_65() {
        // Exercise every small length including the parity boundaries and the
        // `len / 2` last-parent edge cases across a range of sizes.
        let mut rng = Lcg::new(0xABCD_EF01);
        for len in 0..=65 {
            let data: Vec<u32> = (0..len).map(|_| rng.next_u32() % 1_000).collect();
            let mut sorted = data.clone();
            heap_sort(&mut sorted);
            assert_eq!(sorted, std_sorted(&data), "length {len} failed");
        }
    }

    #[test]
    fn by_key_sorts_generic_payloads() {
        // Sort `(name, priority)` pairs by the `u32` priority key; payloads must
        // travel with their keys.
        let mut items = [('a', 30u32), ('b', 10), ('c', 20), ('d', 5)];
        heap_sort_by_key(&mut items, |&(_, priority)| priority);
        assert_eq!(items, [('d', 5), ('b', 10), ('c', 20), ('a', 30)]);
    }

    #[test]
    fn by_key_descending_projection_reverses() {
        // Keying on `u32::MAX - value` yields descending value order.
        let mut data = [1u32, 5, 3, 2, 4];
        heap_sort_by_key(&mut data, |&value| u32::MAX - value);
        assert_eq!(data, [5, 4, 3, 2, 1]);
    }

    #[test]
    fn by_key_empty_and_single() {
        let mut empty: [(u32, u32); 0] = [];
        heap_sort_by_key(&mut empty, |&(k, _)| k);
        assert_eq!(empty, [] as [(u32, u32); 0]);

        let mut single = [(99u32, 7u32)];
        heap_sort_by_key(&mut single, |&(k, _)| k);
        assert_eq!(single, [(99, 7)]);
    }

    #[test]
    fn by_key_matches_std_random() {
        let mut rng = Lcg::new(0x0F0F_0F0F);
        let mut items: Vec<(u32, u32)> = (0..2_000).map(|i| (rng.next_u32(), i)).collect();
        let mut reference = items.clone();
        reference.sort_by_key(|&(key, _)| key);
        heap_sort_by_key(&mut items, |&(key, _)| key);
        // Compare only the keys: heapsort is unstable, so tie-broken payload
        // order may differ, but the key sequence must match exactly.
        let got: Vec<u32> = items.iter().map(|&(key, _)| key).collect();
        let want: Vec<u32> = reference.iter().map(|&(key, _)| key).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn is_max_heap_accepts_empty_and_single() {
        assert!(is_max_heap(&[]));
        assert!(is_max_heap(&[123]));
    }

    #[test]
    fn is_max_heap_accepts_valid_heap() {
        // A hand-built valid max-heap: each parent dominates its children.
        assert!(is_max_heap(&[9u32, 7, 8, 4, 6, 1, 3]));
    }

    #[test]
    fn is_max_heap_rejects_violating_child() {
        // Index 1 (value 2) has child index 3 (value 5) that is larger.
        assert!(!is_max_heap(&[9u32, 2, 8, 5, 6, 1, 3]));
    }

    #[test]
    fn build_produces_valid_heap_and_root_is_max() {
        let mut rng = Lcg::new(0x2468_1357);
        let mut data: Vec<u32> = (0..257).map(|_| rng.next_u32()).collect();
        let expected_max = *data.iter().max().unwrap();
        build_max_heap_by(&mut data, &|value: &u32| *value);
        assert!(is_max_heap(&data));
        assert_eq!(data[0], expected_max);
    }

    #[test]
    fn sorted_output_is_not_a_max_heap_unless_trivial() {
        // A strictly ascending multi-element slice violates the max-heap
        // property at the root, confirming `is_max_heap` is discriminating.
        let data = [1u32, 2, 3, 4, 5];
        assert!(!is_max_heap(&data));
    }
}
