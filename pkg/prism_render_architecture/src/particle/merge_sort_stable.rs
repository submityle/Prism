//! Stable merge sort over `u32` keys: the serial reference that orders
//! particles by an integer sort key while *preserving the original relative
//! order of equal keys* (design §12 sort ordering).
//!
//! A production `GPU` VFX stack often needs a *stable* ordering: when two
//! particles share the same quantized depth or draw-order key, the one that
//! appeared earlier in the pool must still appear earlier after sorting, so
//! that tie-broken layering, additive blending order, and deterministic
//! `CPU`/`GPU` cross-checks all stay reproducible frame to frame. Merge sort is
//! the classic comparison sort that guarantees this: it recursively (here,
//! iteratively bottom-up) splits the slice into runs, then merges adjacent runs
//! with a single rule — when the two front elements compare equal, take the one
//! from the *left* (earlier) run first. That single `left <= right` tie rule is
//! the entire stability guarantee, and this file owns the serial gold standard
//! so an eventual device kernel can be checked value for value.
//!
//! Two entry points are exposed. [`merge_sort_u32`] sorts a slice of raw `u32`
//! keys in place. [`merge_sort_by_key`] sorts a slice of arbitrary clonable
//! payloads by a caller-supplied `u32` key projection, carrying each payload
//! alongside its key and keeping equal-key payloads in their original order.
//! The bottom-up driver [`merge_runs`] and the two-run merge step [`merge`] are
//! exposed for direct unit testing. A single reusable auxiliary buffer (an
//! [`alloc::vec::Vec`]) holds the merged output of each pass; the algorithm
//! ping-pongs between the source slice and that buffer, so the total scratch
//! space is one extra copy of the input regardless of run size.
//!
//! This is intentionally distinct from its neighbours and must not be conflated
//! with them: [`super::heap_sort_u32`] is an *in-place, unstable* selection
//! sort built on a binary heap — it uses no auxiliary buffer but reorders equal
//! keys arbitrarily; [`super::radix_sort_u32`] is a *non-comparison* multi-pass
//! `LSD` bucket sort that reads key digits instead of comparing whole keys;
//! and [`super::bitonic_sort`] is a fixed `GPU` sorting *network* whose compare
//! -and-swap schedule is data-independent for device parallelism. Where those
//! trade stability for in-place operation, skip comparisons entirely, or target
//! the `GPU`, this module is the direct `CPU` comparison sort whose defining
//! property is *stability*: equal keys keep their original relative order.
//!
//! Everything is pure integer arithmetic: keys are compared with `<=` on `u32`,
//! indices are `usize` additions bounded by the input length, and run widths
//! double by a left shift each pass. Nothing divides, and empty or
//! single-element inputs are valid no-ops. No floating point and no
//! transcendental functions appear anywhere in this module.

use alloc::vec::Vec;

/// Merge two already-sorted adjacent runs of `keyed`, described by their keys,
/// into `scratch` starting at `lo`. The left run spans `lo..mid` and the right
/// run spans `mid..hi`; both index ranges must lie within `keyed`, `keys`, and
/// `scratch`, and `keys[i]` must be the key of `keyed[i]`.
///
/// Stability rule: when the two front elements compare equal the element from
/// the left run is emitted first (`left <= right` takes the left), so equal
/// keys never cross each other. `scratch[lo..hi]` receives the merged run; the
/// rest of `scratch` is untouched.
pub fn merge<T: Clone>(
    keyed: &[T],
    keys: &[u32],
    lo: usize,
    mid: usize,
    hi: usize,
    scratch: &mut [T],
) {
    let mut left = lo;
    let mut right = mid;
    let mut out = lo;
    while left < mid && right < hi {
        // `<=` keeps the left (earlier) element first on ties: this is the
        // stability guarantee.
        if keys[left] <= keys[right] {
            scratch[out] = keyed[left].clone();
            left += 1;
        } else {
            scratch[out] = keyed[right].clone();
            right += 1;
        }
        out += 1;
    }
    while left < mid {
        scratch[out] = keyed[left].clone();
        left += 1;
        out += 1;
    }
    while right < hi {
        scratch[out] = keyed[right].clone();
        right += 1;
        out += 1;
    }
}

/// Bottom-up stable merge sort driver. Sorts `data` in place by the parallel
/// `keys` slice (`keys[i]` is the key of `data[i]`), keeping equal keys in
/// their original relative order. `data` and `keys` must have the same length.
///
/// The pass structure is iterative rather than recursive: runs of width `1`,
/// then `2`, `4`, `8`, ... are merged pairwise until a single run covers the
/// whole slice. Each pass merges into a scratch buffer and then copies the
/// result back, so after every pass `data`/`keys` again hold the partially
/// sorted sequence. Inputs of length `0` or `1` are already sorted and return
/// untouched.
pub fn merge_runs<T: Clone>(data: &mut [T], keys: &mut [u32]) {
    let len = data.len();
    if len < 2 {
        return;
    }
    let mut scratch: Vec<T> = data.to_vec();
    let mut key_scratch: Vec<u32> = keys.to_vec();
    let mut width = 1usize;
    while width < len {
        let step = width << 1;
        let mut lo = 0usize;
        while lo < len {
            let mid = core::cmp::min(lo + width, len);
            let hi = core::cmp::min(lo + step, len);
            merge(data, keys, lo, mid, hi, &mut scratch);
            merge(keys, keys, lo, mid, hi, &mut key_scratch);
            lo += step;
        }
        data.clone_from_slice(&scratch);
        keys.clone_from_slice(&key_scratch);
        width = step;
    }
}

/// Sort `data` in ascending order by the `u32` key produced by `key`, keeping
/// payloads with equal keys in their original relative order (a *stable*
/// sort). The key projection is evaluated exactly once per element up front and
/// cached, so the comparison work is pure integer `<=` on the cached keys.
///
/// Empty and single-element slices are already sorted and return untouched.
/// This is deliberately a free function rather than a `Sort` trait method so it
/// does not shadow any standard-library ordering trait.
pub fn merge_sort_by_key<T: Clone, F: Fn(&T) -> u32>(data: &mut [T], key: F) {
    let len = data.len();
    if len < 2 {
        return;
    }
    let mut keys: Vec<u32> = data.iter().map(&key).collect();
    merge_runs(data, &mut keys);
}

/// Sort a slice of raw `u32` keys in ascending order, in place. Stability is
/// invisible for bare integers (equal keys are indistinguishable) but the same
/// stable engine is used, so the ordering matches [`merge_sort_by_key`] with an
/// identity projection. Empty and single-element slices return untouched.
pub fn merge_sort_u32(data: &mut [u32]) {
    let len = data.len();
    if len < 2 {
        return;
    }
    let mut keys: Vec<u32> = data.to_vec();
    merge_runs(data, &mut keys);
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

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

    /// True when `data` is non-decreasing.
    fn is_sorted(data: &[u32]) -> bool {
        data.windows(2).all(|w| w[0] <= w[1])
    }

    #[test]
    fn empty_is_noop() {
        let mut data: Vec<u32> = Vec::new();
        merge_sort_u32(&mut data);
        assert!(data.is_empty());
    }

    #[test]
    fn single_element_is_noop() {
        let mut data = vec![42u32];
        merge_sort_u32(&mut data);
        assert_eq!(data, vec![42u32]);
    }

    #[test]
    fn two_elements_sorted_stay() {
        let mut data = vec![1u32, 2u32];
        merge_sort_u32(&mut data);
        assert_eq!(data, vec![1u32, 2u32]);
    }

    #[test]
    fn two_elements_swap() {
        let mut data = vec![2u32, 1u32];
        merge_sort_u32(&mut data);
        assert_eq!(data, vec![1u32, 2u32]);
    }

    #[test]
    fn already_sorted_unchanged() {
        let mut data = vec![0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        merge_sort_u32(&mut data);
        assert_eq!(data, vec![0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn reversed_input() {
        let mut data = vec![9u32, 8, 7, 6, 5, 4, 3, 2, 1, 0];
        merge_sort_u32(&mut data);
        assert_eq!(data, vec![0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn with_duplicates() {
        let mut data = vec![3u32, 1, 3, 2, 1, 3, 2];
        merge_sort_u32(&mut data);
        assert_eq!(data, vec![1u32, 1, 2, 2, 3, 3, 3]);
    }

    #[test]
    fn all_equal() {
        let mut data = vec![7u32; 32];
        merge_sort_u32(&mut data);
        assert_eq!(data, vec![7u32; 32]);
    }

    #[test]
    fn odd_length_run_handling() {
        let mut data = vec![5u32, 3, 8, 1, 9, 2, 7];
        merge_sort_u32(&mut data);
        assert_eq!(data, vec![1u32, 2, 3, 5, 7, 8, 9]);
    }

    #[test]
    fn power_of_two_length() {
        let mut data = vec![8u32, 4, 6, 2, 7, 3, 5, 1];
        merge_sort_u32(&mut data);
        assert_eq!(data, vec![1u32, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn non_power_of_two_length() {
        let mut data = vec![8u32, 4, 6, 2, 7, 3, 5, 1, 0];
        merge_sort_u32(&mut data);
        assert_eq!(data, vec![0u32, 1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn contains_zero_and_max() {
        let mut data = vec![u32::MAX, 0u32, u32::MAX, 1u32, 0u32];
        merge_sort_u32(&mut data);
        assert_eq!(data, vec![0u32, 0, 1, u32::MAX, u32::MAX]);
    }

    #[test]
    fn u32_max_sorts_last() {
        let mut data = vec![u32::MAX, 0u32, 100u32, u32::MAX - 1];
        merge_sort_u32(&mut data);
        assert_eq!(data, vec![0u32, 100, u32::MAX - 1, u32::MAX]);
    }

    #[test]
    fn boundary_values_full_range() {
        let mut data = vec![u32::MAX, 0u32, 2_147_483_648u32, 2_147_483_647u32, 1u32];
        merge_sort_u32(&mut data);
        assert_eq!(data, vec![0u32, 1, 2_147_483_647, 2_147_483_648, u32::MAX]);
    }

    #[test]
    fn idempotent_on_sorted() {
        let mut data = vec![5u32, 2, 9, 1, 5, 6, 2];
        merge_sort_u32(&mut data);
        let once = data.clone();
        merge_sort_u32(&mut data);
        assert_eq!(data, once);
    }

    #[test]
    fn idempotent_random() {
        let mut rng = Lcg::new(0x1357_9BDF);
        let mut data: Vec<u32> = (0..500).map(|_| rng.next_u32() % 50).collect();
        merge_sort_u32(&mut data);
        let once = data.clone();
        merge_sort_u32(&mut data);
        assert_eq!(data, once);
    }

    #[test]
    fn large_random_is_sorted() {
        let mut rng = Lcg::new(0xDEAD_BEEF);
        let mut data: Vec<u32> = (0..5_000).map(|_| rng.next_u32()).collect();
        merge_sort_u32(&mut data);
        assert!(is_sorted(&data));
    }

    #[test]
    fn large_random_matches_std() {
        let mut rng = Lcg::new(0xABCD_EF01);
        let source: Vec<u32> = (0..4_000).map(|_| rng.next_u32() % 1_000).collect();
        let mut data = source.clone();
        merge_sort_u32(&mut data);
        let mut reference = source.clone();
        reference.sort();
        assert_eq!(data, reference);
    }

    #[test]
    fn large_random_preserves_multiset() {
        let mut rng = Lcg::new(0x0F0F_0F0F);
        let source: Vec<u32> = (0..3_000).map(|_| rng.next_u32() % 200).collect();
        let mut data = source.clone();
        merge_sort_u32(&mut data);
        let mut histogram_in = [0usize; 200];
        let mut histogram_out = [0usize; 200];
        for &value in &source {
            histogram_in[value as usize] += 1;
        }
        for &value in &data {
            histogram_out[value as usize] += 1;
        }
        assert_eq!(histogram_in, histogram_out);
    }

    #[test]
    fn stability_equal_keys_keep_order() {
        // Pair each element with its original index; keys have many ties.
        let source: Vec<(u32, usize)> =
            vec![(2, 0), (1, 1), (2, 2), (1, 3), (2, 4), (0, 5), (1, 6)];
        let mut data = source.clone();
        merge_sort_by_key(&mut data, |&(key, _)| key);
        let expected: Vec<(u32, usize)> =
            vec![(0, 5), (1, 1), (1, 3), (1, 6), (2, 0), (2, 2), (2, 4)];
        assert_eq!(data, expected);
    }

    #[test]
    fn stability_all_equal_keys_preserve_index() {
        let source: Vec<(u32, usize)> = (0..64).map(|i| (9u32, i)).collect();
        let mut data = source.clone();
        merge_sort_by_key(&mut data, |&(key, _)| key);
        assert_eq!(data, source);
    }

    #[test]
    fn stability_matches_std_stable_sort() {
        let mut rng = Lcg::new(0x0BAD_F00D);
        // Small key domain forces many ties; original index rides along.
        let source: Vec<(u32, usize)> = (0..4_000).map(|i| (rng.next_u32() % 17, i)).collect();
        let mut data = source.clone();
        merge_sort_by_key(&mut data, |&(key, _)| key);
        let mut reference = source.clone();
        reference.sort_by_key(|&(key, _)| key);
        assert_eq!(data, reference);
    }

    #[test]
    fn stability_ties_are_stable_by_index() {
        let mut rng = Lcg::new(0x2468_ACE0);
        let source: Vec<(u32, usize)> = (0..2_000).map(|i| (rng.next_u32() % 8, i)).collect();
        let mut data = source.clone();
        merge_sort_by_key(&mut data, |&(key, _)| key);
        // Within each equal-key run the original indices must be increasing.
        for pair in data.windows(2) {
            if pair[0].0 == pair[1].0 {
                assert!(pair[0].1 < pair[1].1);
            }
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    struct Particle {
        depth_key: u32,
        id: u32,
    }

    #[test]
    fn by_key_struct_sort() {
        let mut data = vec![
            Particle {
                depth_key: 30,
                id: 0,
            },
            Particle {
                depth_key: 10,
                id: 1,
            },
            Particle {
                depth_key: 20,
                id: 2,
            },
            Particle {
                depth_key: 10,
                id: 3,
            },
        ];
        merge_sort_by_key(&mut data, |particle| particle.depth_key);
        let expected = vec![
            Particle {
                depth_key: 10,
                id: 1,
            },
            Particle {
                depth_key: 10,
                id: 3,
            },
            Particle {
                depth_key: 20,
                id: 2,
            },
            Particle {
                depth_key: 30,
                id: 0,
            },
        ];
        assert_eq!(data, expected);
    }

    #[test]
    fn by_key_empty_and_single() {
        let mut empty: Vec<Particle> = Vec::new();
        merge_sort_by_key(&mut empty, |particle| particle.depth_key);
        assert!(empty.is_empty());

        let mut single = vec![Particle {
            depth_key: 5,
            id: 9,
        }];
        merge_sort_by_key(&mut single, |particle| particle.depth_key);
        assert_eq!(
            single,
            vec![Particle {
                depth_key: 5,
                id: 9
            }]
        );
    }

    #[test]
    fn merge_step_combines_two_runs() {
        let keyed = vec![1u32, 4, 6, 2, 3, 5];
        let keys = keyed.clone();
        let mut scratch = vec![0u32; keyed.len()];
        merge(&keyed, &keys, 0, 3, 6, &mut scratch);
        assert_eq!(scratch, vec![1u32, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn merge_step_is_stable_on_ties() {
        // Two runs sharing key 5; left-run indices must precede right-run ones.
        let keyed = vec![
            (5u32, 0usize),
            (5u32, 1usize),
            (5u32, 2usize),
            (5u32, 3usize),
        ];
        let keys = vec![5u32, 5, 5, 5];
        let mut scratch = keyed.clone();
        merge(&keyed, &keys, 0, 2, 4, &mut scratch);
        assert_eq!(
            scratch,
            vec![
                (5u32, 0usize),
                (5u32, 1usize),
                (5u32, 2usize),
                (5u32, 3usize)
            ]
        );
    }

    #[test]
    fn merge_runs_directly_sorts() {
        let mut data = vec![4u32, 2, 5, 1, 3];
        let mut keys = data.clone();
        merge_runs(&mut data, &mut keys);
        assert_eq!(data, vec![1u32, 2, 3, 4, 5]);
        assert_eq!(keys, vec![1u32, 2, 3, 4, 5]);
    }

    #[test]
    fn negative_range_of_keys_wraps_correctly() {
        // Values near both ends of the u32 range must not be reordered by any
        // signed interpretation; comparison is purely unsigned.
        let mut data = vec![4_000_000_000u32, 5, 3_000_000_000, 1, u32::MAX];
        merge_sort_u32(&mut data);
        assert_eq!(data, vec![1u32, 5, 3_000_000_000, 4_000_000_000, u32::MAX]);
    }

    #[test]
    fn small_random_sizes_all_sort() {
        let mut rng = Lcg::new(0x9E37_79B9);
        for size in 0..40usize {
            let mut data: Vec<u32> = (0..size).map(|_| rng.next_u32() % 100).collect();
            let mut reference = data.clone();
            reference.sort();
            merge_sort_u32(&mut data);
            assert_eq!(data, reference);
        }
    }
}
