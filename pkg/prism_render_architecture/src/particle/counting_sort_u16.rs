//! Deterministic single-pass counting sort over `u16` keys: the serial gold
//! standard for stably ordering particles by a 16-bit quantized sort key
//! (design §12 sort ordering).
//!
//! A production `GPU` VFX stack quantizes view-depth or draw-order keys into a
//! small integer domain and then bucket-sorts millions of particles per frame.
//! When that domain fits in 16 bits, a *single* counting pass sorts the whole
//! key space directly, with no multi-pass digit loop: build a `65536`-bucket
//! `histogram` of the keys, turn that `histogram` into per-bucket start offsets
//! with an exclusive prefix sum, then `scatter` every element to its bucket in
//! input order. Because the `scatter` walks the source front to back and each
//! bucket cursor only advances, elements sharing a key keep their relative
//! order — the sort is *stable*. This file owns the serial reference so an
//! eventual device kernel can be checked value for value.
//!
//! Three entry points are exposed. [`sort`] returns a sorted copy of the keys.
//! [`argsort`] returns a stable permutation of indices `0..len` such that
//! indexing the original keys by that permutation yields sorted order, ties
//! broken by original position. [`sort_by_key`] stably reorders
//! `(key, payload)` pairs, carrying each payload alongside its key. The
//! building blocks [`histogram`] and [`exclusive_prefix_sum`] are exposed for
//! direct unit testing. To stay self-contained this module carries its own tiny
//! prefix sum rather than importing a general `scan`.
//!
//! This is intentionally distinct from its neighbours and must not be conflated
//! with them: [`super::radix_sort_u32`] is a *multi-pass* least-significant-
//! `digit` (`LSD`) `radix` sort over the wider `u32` key space, where each pass
//! is itself a counting sort; [`super::bitonic_sort`] is a comparison sorting
//! network, a different algorithm class entirely; and
//! [`super::gpu_radix_histogram`] owns only the `GPU` `histogram`-construction
//! primitive, not a full sort. Where those handle wider keys, multiple passes,
//! or device primitives, this module is the direct one-pass `CPU` counting sort
//! for the exact `u16` key domain.
//!
//! Everything is pure integer arithmetic: bucketing is a widening cast to
//! `usize`, counting is `usize` addition bounded by the input length, and the
//! prefix sum is a running `usize` total. Nothing divides, and empty or
//! single-element inputs are valid no-ops. No floating point and no
//! transcendental functions appear anywhere in this module.

use alloc::vec;
use alloc::vec::Vec;

/// Number of distinct `u16` keys, hence the bucket count of the `histogram`.
pub const BUCKET_COUNT: usize = 1 << 16;

/// Build the `65536`-bucket key `histogram`: `histogram(keys)[k]` is the number
/// of elements in `keys` equal to `k as u16`. The returned vector always has
/// length [`BUCKET_COUNT`]; an empty input yields all zeros.
pub fn histogram(keys: &[u16]) -> Vec<usize> {
    let mut counts = vec![0usize; BUCKET_COUNT];
    for &key in keys {
        counts[key as usize] += 1;
    }
    counts
}

/// Exclusive prefix sum: `out[i]` is the sum of `counts[..i]`, so `out[0]` is
/// always `0` and `out[i]` is the start offset of bucket `i` in the sorted
/// output. The returned vector matches `counts` in length.
pub fn exclusive_prefix_sum(counts: &[usize]) -> Vec<usize> {
    let mut offsets = vec![0usize; counts.len()];
    let mut running = 0usize;
    for (offset, &count) in offsets.iter_mut().zip(counts.iter()) {
        *offset = running;
        running += count;
    }
    offsets
}

/// Return a sorted copy of `keys` in ascending order. Stable, deterministic,
/// and allocation-bounded by two `Vec`s plus the bucket table. Empty and
/// single-element inputs return an equal copy.
pub fn sort(keys: &[u16]) -> Vec<u16> {
    let mut cursors = exclusive_prefix_sum(&histogram(keys));
    let mut out = vec![0u16; keys.len()];
    for &key in keys {
        let bucket = key as usize;
        out[cursors[bucket]] = key;
        cursors[bucket] += 1;
    }
    out
}

/// Return a stable permutation `perm` of `0..keys.len()` such that
/// `keys[perm[0]] <= keys[perm[1]] <= ...`, with equal keys keeping their
/// original relative order.
pub fn argsort(keys: &[u16]) -> Vec<usize> {
    let mut cursors = exclusive_prefix_sum(&histogram(keys));
    let mut out = vec![0usize; keys.len()];
    for (index, &key) in keys.iter().enumerate() {
        let bucket = key as usize;
        out[cursors[bucket]] = index;
        cursors[bucket] += 1;
    }
    out
}

/// Stably sort `(key, payload)` pairs by their `u16` key, carrying each payload
/// alongside its key and preserving the original order of equal keys.
pub fn sort_by_key<P: Clone>(items: &[(u16, P)]) -> Vec<(u16, P)> {
    let keys: Vec<u16> = items.iter().map(|(key, _)| *key).collect();
    let perm = argsort(&keys);
    perm.iter().map(|&index| items[index].clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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

        fn next_u16(&mut self) -> u16 {
            (self.next_u64() >> 48) as u16
        }
    }

    /// Reference sort using the standard library, for comparison.
    fn std_sorted(keys: &[u16]) -> Vec<u16> {
        let mut v = keys.to_vec();
        v.sort_unstable();
        v
    }

    /// True when `perm` is exactly a rearrangement of `0..len`.
    fn is_permutation(perm: &[usize], len: usize) -> bool {
        let mut seen = perm.to_vec();
        seen.sort_unstable();
        seen == (0..len).collect::<Vec<usize>>()
    }

    #[test]
    fn histogram_length_is_bucket_count() {
        assert_eq!(histogram(&[]).len(), BUCKET_COUNT);
        assert_eq!(histogram(&[1, 2, 3]).len(), BUCKET_COUNT);
    }

    #[test]
    fn histogram_empty_all_zero() {
        let counts = histogram(&[]);
        assert!(counts.iter().all(|&c| c == 0));
    }

    #[test]
    fn histogram_counts_correct() {
        let counts = histogram(&[3, 3, 3, 7, 0, 7]);
        assert_eq!(counts[0], 1);
        assert_eq!(counts[3], 3);
        assert_eq!(counts[7], 2);
        assert_eq!(counts[1], 0);
        let total: usize = counts.iter().sum();
        assert_eq!(total, 6);
    }

    #[test]
    fn histogram_zero_and_max_buckets() {
        let counts = histogram(&[0, u16::MAX, 0, u16::MAX, u16::MAX]);
        assert_eq!(counts[0], 2);
        assert_eq!(counts[u16::MAX as usize], 3);
    }

    #[test]
    fn prefix_sum_empty() {
        assert_eq!(exclusive_prefix_sum(&[]), Vec::<usize>::new());
    }

    #[test]
    fn prefix_sum_single() {
        assert_eq!(exclusive_prefix_sum(&[9]), vec![0]);
    }

    #[test]
    fn prefix_sum_exclusive_basic() {
        assert_eq!(exclusive_prefix_sum(&[2, 0, 3, 1]), vec![0, 2, 2, 5]);
    }

    #[test]
    fn prefix_sum_matches_running_total() {
        let counts = [4usize, 1, 0, 7, 2, 5];
        let offsets = exclusive_prefix_sum(&counts);
        let mut running = 0usize;
        for (offset, &count) in offsets.iter().zip(counts.iter()) {
            assert_eq!(*offset, running);
            running += count;
        }
    }

    #[test]
    fn sort_empty() {
        assert_eq!(sort(&[]), Vec::<u16>::new());
    }

    #[test]
    fn sort_single() {
        assert_eq!(sort(&[42]), vec![42]);
    }

    #[test]
    fn sort_pair() {
        assert_eq!(sort(&[7, 3]), vec![3, 7]);
    }

    #[test]
    fn sort_all_equal() {
        assert_eq!(sort(&[5, 5, 5, 5]), vec![5, 5, 5, 5]);
    }

    #[test]
    fn sort_already_sorted() {
        let input = [0u16, 1, 2, 3, 4, 5];
        assert_eq!(sort(&input), input.to_vec());
    }

    #[test]
    fn sort_reverse() {
        let input = [9u16, 8, 7, 6, 5, 4, 3, 2, 1, 0];
        assert_eq!(sort(&input), vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn sort_zero_and_max() {
        let input = [u16::MAX, 0, 1, u16::MAX, 0];
        assert_eq!(sort(&input), vec![0, 0, 1, u16::MAX, u16::MAX]);
    }

    #[test]
    fn sort_idempotent() {
        let input = [4u16, 1, 9, 1, 4, 0, 7];
        let once = sort(&input);
        let twice = sort(&once);
        assert_eq!(once, twice);
    }

    #[test]
    fn sort_random_matches_std() {
        let mut rng = Lcg::new(0x1234_5678);
        let input: Vec<u16> = (0..2_000).map(|_| rng.next_u16()).collect();
        assert_eq!(sort(&input), std_sorted(&input));
    }

    #[test]
    fn sort_many_duplicates_matches_std() {
        let mut rng = Lcg::new(0xABCD_EF01);
        let input: Vec<u16> = (0..5_000).map(|_| rng.next_u16() % 11).collect();
        assert_eq!(sort(&input), std_sorted(&input));
    }

    #[test]
    fn sort_clustered_matches_std() {
        let mut rng = Lcg::new(0x0F0F_0F0F);
        let input: Vec<u16> = (0..3_000).map(|_| rng.next_u16() & 0xFF00).collect();
        assert_eq!(sort(&input), std_sorted(&input));
    }

    #[test]
    fn sort_large_random_matches_std() {
        let mut rng = Lcg::new(0xDEAD_BEEF);
        let input: Vec<u16> = (0..20_000).map(|_| rng.next_u16()).collect();
        assert_eq!(sort(&input), std_sorted(&input));
    }

    #[test]
    fn argsort_empty() {
        assert_eq!(argsort(&[]), Vec::<usize>::new());
    }

    #[test]
    fn argsort_single() {
        assert_eq!(argsort(&[42]), vec![0]);
    }

    #[test]
    fn argsort_pair() {
        assert_eq!(argsort(&[7, 3]), vec![1, 0]);
    }

    #[test]
    fn argsort_is_permutation() {
        let mut rng = Lcg::new(0x5151_5151);
        let keys: Vec<u16> = (0..4_000).map(|_| rng.next_u16()).collect();
        let perm = argsort(&keys);
        assert!(is_permutation(&perm, keys.len()));
    }

    #[test]
    fn argsort_yields_sorted() {
        let mut rng = Lcg::new(0x9999_1111);
        let keys: Vec<u16> = (0..4_000).map(|_| rng.next_u16()).collect();
        let perm = argsort(&keys);
        let via_perm: Vec<u16> = perm.iter().map(|&i| keys[i]).collect();
        assert_eq!(via_perm, sort(&keys));
    }

    #[test]
    fn argsort_stability_equal_keys() {
        // Every key collapses to a handful of buckets, so ties are frequent;
        // among equal keys the original indices must stay strictly ascending.
        let mut rng = Lcg::new(0x2468_1357);
        let keys: Vec<u16> = (0..3_000).map(|_| rng.next_u16() % 5).collect();
        let perm = argsort(&keys);
        for window in perm.windows(2) {
            let (a, b) = (window[0], window[1]);
            if keys[a] == keys[b] {
                assert!(a < b, "stability violated: {a} before {b}");
            } else {
                assert!(keys[a] < keys[b]);
            }
        }
    }

    #[test]
    fn argsort_reverse() {
        let keys = [4u16, 3, 2, 1, 0];
        assert_eq!(argsort(&keys), vec![4, 3, 2, 1, 0]);
    }

    #[test]
    fn argsort_matches_sort_via_perm() {
        let mut rng = Lcg::new(0x7C7C_7C7C);
        let keys: Vec<u16> = (0..2_500).map(|_| rng.next_u16() % 97).collect();
        let perm = argsort(&keys);
        let via_perm: Vec<u16> = perm.iter().map(|&i| keys[i]).collect();
        assert_eq!(via_perm, std_sorted(&keys));
    }

    #[test]
    fn sort_by_key_empty() {
        let items: Vec<(u16, u32)> = Vec::new();
        assert_eq!(sort_by_key(&items), items);
    }

    #[test]
    fn sort_by_key_single() {
        assert_eq!(sort_by_key(&[(7u16, 99u32)]), vec![(7, 99)]);
    }

    #[test]
    fn sort_by_key_payload_follows_key() {
        let items = [(3u16, 'c'), (1u16, 'a'), (2u16, 'b')];
        assert_eq!(sort_by_key(&items), vec![(1, 'a'), (2, 'b'), (3, 'c')]);
    }

    #[test]
    fn sort_by_key_stable_on_duplicates() {
        // Equal keys with distinct payloads inserted in ascending payload order;
        // stability requires the payloads to stay in that order after sorting.
        let items = [
            (2u16, 0u32),
            (1u16, 1u32),
            (2u16, 2u32),
            (1u16, 3u32),
            (2u16, 4u32),
            (0u16, 5u32),
            (1u16, 6u32),
        ];
        let out = sort_by_key(&items);
        let expected = vec![
            (0u16, 5u32),
            (1u16, 1u32),
            (1u16, 3u32),
            (1u16, 6u32),
            (2u16, 0u32),
            (2u16, 2u32),
            (2u16, 4u32),
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn sort_by_key_matches_std_stable() {
        let mut rng = Lcg::new(0x0BAD_F00D);
        let items: Vec<(u16, u32)> = (0..4_000)
            .map(|i| (rng.next_u16() % 13, i as u32))
            .collect();
        let out = sort_by_key(&items);
        let mut reference = items.clone();
        reference.sort_by_key(|(key, _)| *key);
        assert_eq!(out, reference);
    }
}
