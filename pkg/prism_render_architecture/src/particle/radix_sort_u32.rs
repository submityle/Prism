//! Deterministic `CPU` least-significant-`digit` (`LSD`) `radix` sort over
//! `u32` keys: the bit-exact gold standard the particle sort pass validates its
//! `GPU` bucket sort against (design §12 sort ordering).
//!
//! A production `GPU` VFX stack sorts millions of quantized view-depth keys per
//! frame with a multi-pass `LSD` `radix` sort. This module owns the serial
//! reference of that scheme so the eventual device kernel can be checked byte
//! for byte. Each pass consumes one 8-bit `digit` of the 32-bit key, so four
//! passes (bytes `0`, `1`, `2`, `3` from least to most significant) sort the
//! whole key space. A single pass is a counting sort: build a 256-bucket
//! `histogram` of the current `digit`, turn that `histogram` into per-bucket
//! start offsets with an exclusive prefix sum, then `scatter` every element to
//! its bucket in input order. Because the `scatter` walks the source front to
//! back and each bucket's cursor only advances, elements sharing a `digit` keep
//! their relative order — the pass is stable, and stability across all four
//! passes is what makes the least-significant-first ordering sort correctly.
//!
//! Two entry points are exposed. [`sort`] returns a sorted copy of the keys.
//! [`argsort`] returns a stable permutation of indices `0..len` such that
//! indexing the original keys by that permutation yields the sorted order, with
//! ties broken by original position. Both share the same four counting passes;
//! `argsort` carries indices through the `scatter` instead of keys.
//!
//! This is intentionally distinct from two neighbours and must not be conflated
//! with them: [`super::gpu_radix_histogram`] owns only the `GPU` `histogram`
//! construction primitive (a single count step, configurable `digit` width),
//! not the full multi-pass sort; and [`super::bitonic_sort`] is a comparison
//! sorting network, a different algorithm class entirely. To stay self-contained
//! this file carries its own tiny exclusive prefix sum rather than importing a
//! general `scan`.
//!
//! Everything is pure integer arithmetic: `digit` extraction is a shift and a
//! mask, counting is `usize` addition bounded by the input length, and nothing
//! panics or divides by zero. An empty or single-element input is a valid
//! no-op. No floating point and no transcendental functions appear here.

use alloc::vec;
use alloc::vec::Vec;

/// Number of key bits consumed per counting pass.
const DIGIT_BITS: u32 = 8;

/// Number of `histogram` buckets in one pass (`1 << DIGIT_BITS` = 256).
const BUCKETS: usize = 1 << DIGIT_BITS;

/// Bit mask isolating one `digit` (`0xFF` for 8-bit digits).
const DIGIT_MASK: u32 = (BUCKETS as u32) - 1;

/// Number of passes needed to consume every bit of a `u32` key
/// (`32 / DIGIT_BITS` = 4).
const PASSES: u32 = u32::BITS / DIGIT_BITS;

/// Extract the `pass`-th 8-bit `digit` of `key`, least significant first.
///
/// `pass` must be in `0..PASSES`; the shift is `pass * DIGIT_BITS` and stays
/// within `u32::BITS`.
fn digit_of(key: u32, pass: u32) -> usize {
    ((key >> (pass * DIGIT_BITS)) & DIGIT_MASK) as usize
}

/// Convert a per-bucket count `histogram` into per-bucket start offsets with an
/// in-place exclusive prefix sum.
///
/// After the sweep `counts[b]` holds the index at which bucket `b`'s first
/// element is scattered, and the running total equals the element count.
fn exclusive_prefix_sum(counts: &mut [usize; BUCKETS]) {
    let mut running = 0usize;
    for slot in counts.iter_mut() {
        let here = *slot;
        *slot = running;
        running += here;
    }
}

/// Return a sorted copy of `keys` in ascending order.
///
/// Runs four stable 8-bit counting passes over a working buffer, ping-ponging
/// between two `Vec`s. Empty and single-element inputs return an equal copy
/// without doing any work.
#[must_use]
pub fn sort(keys: &[u32]) -> Vec<u32> {
    let len = keys.len();
    let mut src: Vec<u32> = keys.to_vec();
    if len < 2 {
        return src;
    }
    let mut dst: Vec<u32> = vec![0u32; len];

    for pass in 0..PASSES {
        let mut counts = [0usize; BUCKETS];
        for &key in &src {
            counts[digit_of(key, pass)] += 1;
        }
        exclusive_prefix_sum(&mut counts);
        for &key in &src {
            let bucket = digit_of(key, pass);
            let target = counts[bucket];
            dst[target] = key;
            counts[bucket] = target + 1;
        }
        core::mem::swap(&mut src, &mut dst);
    }

    // `PASSES` is even (4), so after the final swap the fully sorted data is
    // back in `src`.
    src
}

/// Return a stable permutation of `0..keys.len()` that orders `keys` ascending.
///
/// Indexing `keys` by the returned permutation yields the same sequence as
/// [`sort`]. Keys that compare equal keep their original relative order because
/// every pass scatters indices front to back into monotonically advancing
/// bucket cursors. Empty and single-element inputs return the identity
/// permutation.
#[must_use]
pub fn argsort(keys: &[u32]) -> Vec<usize> {
    let len = keys.len();
    let mut src: Vec<usize> = (0..len).collect();
    if len < 2 {
        return src;
    }
    let mut dst: Vec<usize> = vec![0usize; len];

    for pass in 0..PASSES {
        let mut counts = [0usize; BUCKETS];
        for &idx in &src {
            counts[digit_of(keys[idx], pass)] += 1;
        }
        exclusive_prefix_sum(&mut counts);
        for &idx in &src {
            let bucket = digit_of(keys[idx], pass);
            let target = counts[bucket];
            dst[target] = idx;
            counts[bucket] = target + 1;
        }
        core::mem::swap(&mut src, &mut dst);
    }

    src
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Reference sort using the standard library, for comparison.
    fn std_sorted(keys: &[u32]) -> Vec<u32> {
        let mut v = keys.to_vec();
        v.sort_unstable();
        v
    }

    #[test]
    fn sort_empty_is_empty() {
        let out = sort(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn argsort_empty_is_empty() {
        let out = argsort(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn sort_single_element_returns_copy() {
        assert_eq!(sort(&[42]), vec![42]);
    }

    #[test]
    fn argsort_single_element_is_identity() {
        assert_eq!(argsort(&[42]), vec![0]);
    }

    #[test]
    fn sort_two_elements_ascending() {
        assert_eq!(sort(&[7, 3]), vec![3, 7]);
    }

    #[test]
    fn sort_two_equal_elements() {
        assert_eq!(sort(&[5, 5]), vec![5, 5]);
    }

    #[test]
    fn sort_already_sorted_unchanged() {
        let input = [0u32, 1, 2, 3, 4, 5, 100, 1000];
        assert_eq!(sort(&input), input.to_vec());
    }

    #[test]
    fn sort_reverse_sorted() {
        let input = [9u32, 8, 7, 6, 5, 4, 3, 2, 1, 0];
        assert_eq!(sort(&input), vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn sort_all_equal_keys() {
        let input = [17u32; 32];
        assert_eq!(sort(&input), input.to_vec());
    }

    #[test]
    fn sort_contains_zero_and_max() {
        let input = [u32::MAX, 0, u32::MAX, 0, 1];
        assert_eq!(sort(&input), vec![0, 0, 1, u32::MAX, u32::MAX]);
    }

    #[test]
    fn sort_only_zero_and_max_boundary() {
        let input = [u32::MAX, 0];
        assert_eq!(sort(&input), vec![0, u32::MAX]);
    }

    #[test]
    fn sort_values_straddling_byte_boundaries() {
        // Values chosen to exercise carry across all four digit passes.
        let input = [
            0x0000_00FFu32,
            0x0000_FF00,
            0x00FF_0000,
            0xFF00_0000,
            0x0100_0100,
        ];
        assert_eq!(sort(&input), std_sorted(&input));
    }

    #[test]
    fn sort_matches_std_on_small_fixed_set() {
        let input = [300u32, 1, 256, 255, 257, 65_535, 65_536, 2];
        assert_eq!(sort(&input), std_sorted(&input));
    }

    #[test]
    fn sort_handles_many_duplicate_keys() {
        let input = [3u32, 1, 3, 1, 2, 2, 3, 1, 2, 3, 1, 2];
        assert_eq!(sort(&input), std_sorted(&input));
    }

    #[test]
    fn sort_large_random_matches_std() {
        let mut rng = Lcg::new(0xDEAD_BEEF);
        let input: Vec<u32> = (0..10_000).map(|_| rng.next_u32()).collect();
        assert_eq!(sort(&input), std_sorted(&input));
    }

    #[test]
    fn sort_random_with_restricted_range_matches_std() {
        // Heavy duplication: keys clustered in a tiny range.
        let mut rng = Lcg::new(0x1234_5678);
        let input: Vec<u32> = (0..5_000).map(|_| rng.next_u32() % 7).collect();
        assert_eq!(sort(&input), std_sorted(&input));
    }

    #[test]
    fn sort_random_high_bytes_only_matches_std() {
        // Only the most-significant digit varies, exercising the last pass.
        let mut rng = Lcg::new(0xABCD_0001);
        let input: Vec<u32> = (0..3_000).map(|_| rng.next_u32() & 0xFF00_0000).collect();
        assert_eq!(sort(&input), std_sorted(&input));
    }

    #[test]
    fn sort_output_is_nondecreasing() {
        let mut rng = Lcg::new(0x00C0_FFEE);
        let input: Vec<u32> = (0..4_096).map(|_| rng.next_u32()).collect();
        let out = sort(&input);
        assert!(out.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn sort_preserves_multiset() {
        let mut rng = Lcg::new(0x5EED_5EED);
        let input: Vec<u32> = (0..2_048).map(|_| rng.next_u32() % 97).collect();
        let mut sorted_out = sort(&input);
        let mut sorted_in = input.clone();
        sorted_in.sort_unstable();
        sorted_out.sort_unstable();
        assert_eq!(sorted_out, sorted_in);
    }

    #[test]
    fn argsort_two_elements() {
        // keys [7, 3] -> ascending order picks index 1 then 0.
        assert_eq!(argsort(&[7, 3]), vec![1, 0]);
    }

    #[test]
    fn argsort_permutation_reproduces_sorted() {
        let mut rng = Lcg::new(0x0BAD_F00D);
        let keys: Vec<u32> = (0..5_000).map(|_| rng.next_u32()).collect();
        let perm = argsort(&keys);
        let via_perm: Vec<u32> = perm.iter().map(|&i| keys[i]).collect();
        assert_eq!(via_perm, sort(&keys));
    }

    #[test]
    fn argsort_is_a_valid_permutation() {
        let mut rng = Lcg::new(0xFEED_FACE);
        let keys: Vec<u32> = (0..3_333).map(|_| rng.next_u32()).collect();
        let perm = argsort(&keys);
        let mut seen = perm.clone();
        seen.sort_unstable();
        let identity: Vec<usize> = (0..keys.len()).collect();
        assert_eq!(seen, identity);
    }

    #[test]
    fn argsort_is_stable_on_equal_keys() {
        // Every key equal: the permutation must be the identity (original order).
        let keys = [4u32; 64];
        let perm = argsort(&keys);
        let identity: Vec<usize> = (0..keys.len()).collect();
        assert_eq!(perm, identity);
    }

    #[test]
    fn argsort_ties_keep_original_order() {
        // Keys with duplicates: for each distinct key the source indices that
        // carry it must appear in ascending (original) order within the perm.
        let keys = [2u32, 1, 2, 1, 2, 0, 1, 0];
        let perm = argsort(&keys);
        // Group source indices by key in perm order and check monotonicity.
        for target in [0u32, 1, 2] {
            let indices: Vec<usize> = perm
                .iter()
                .copied()
                .filter(|&i| keys[i] == target)
                .collect();
            assert!(indices.windows(2).all(|w| w[0] < w[1]));
        }
    }

    #[test]
    fn argsort_stability_with_payload_tracking() {
        // Attach a running tag to duplicate keys and confirm tags come out
        // ascending within each key group after the permutation.
        let keys = [5u32, 5, 1, 5, 1, 5, 1, 9, 1];
        let perm = argsort(&keys);
        let sorted_keys: Vec<u32> = perm.iter().map(|&i| keys[i]).collect();
        assert_eq!(sorted_keys, sort(&keys));
        // Within the run of key == 5, the source indices ascend.
        let fives: Vec<usize> = perm.iter().copied().filter(|&i| keys[i] == 5).collect();
        assert_eq!(fives, vec![0, 1, 3, 5]);
    }

    #[test]
    fn argsort_with_zero_and_max() {
        let keys = [u32::MAX, 0, 3, u32::MAX, 0];
        let perm = argsort(&keys);
        let via_perm: Vec<u32> = perm.iter().map(|&i| keys[i]).collect();
        assert_eq!(via_perm, vec![0, 0, 3, u32::MAX, u32::MAX]);
        // Stability: the two zeros keep source order 1 then 4.
        let zeros: Vec<usize> = perm.iter().copied().filter(|&i| keys[i] == 0).collect();
        assert_eq!(zeros, vec![1, 4]);
    }

    #[test]
    fn argsort_matches_sort_on_restricted_range() {
        let mut rng = Lcg::new(0x2468_ACE0);
        let keys: Vec<u32> = (0..6_000).map(|_| rng.next_u32() % 13).collect();
        let perm = argsort(&keys);
        let via_perm: Vec<u32> = perm.iter().map(|&i| keys[i]).collect();
        assert_eq!(via_perm, std_sorted(&keys));
    }

    #[test]
    fn digit_extraction_covers_all_four_bytes() {
        let key = 0x0403_0201u32;
        assert_eq!(digit_of(key, 0), 0x01);
        assert_eq!(digit_of(key, 1), 0x02);
        assert_eq!(digit_of(key, 2), 0x03);
        assert_eq!(digit_of(key, 3), 0x04);
    }

    #[test]
    fn prefix_sum_produces_start_offsets() {
        let mut counts = [0usize; BUCKETS];
        counts[0] = 2;
        counts[1] = 3;
        counts[5] = 1;
        exclusive_prefix_sum(&mut counts);
        assert_eq!(counts[0], 0);
        assert_eq!(counts[1], 2);
        assert_eq!(counts[2], 5);
        assert_eq!(counts[5], 5);
    }
}
