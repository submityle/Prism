//! Least-significant-`digit` (`LSD`) `radix` sort front half: the
//! `CPU`-verifiable gold standard for the particle subsystem's `GPU`
//! `histogram` / bucket-offset / stable-`scatter` sort passes (design §12 sort,
//! §13 cull ordering).
//!
//! Production `GPU` VFX stacks sort millions of quantized view-depth keys per
//! frame with a multi-pass `LSD` `radix` sort: for each `R`-bit `digit`,
//! extract that `digit` from every key, build a per-bucket `histogram`, turn the
//! `histogram` into per-bucket start offsets with an exclusive prefix `scan`,
//! and `scatter` every key to its bucket while preserving input order within a
//! bucket (a stable pass). Repeating over every `digit` from the least- to the
//! most-significant sorts the whole 32-bit key space. On the device the count
//! step is tiled: each `workgroup` scans one block of keys into a private
//! `histogram`, and those per-block `histogram`s are merged and scanned into
//! global offsets.
//!
//! This module owns only the deterministic `CPU` reference of that scheme so the
//! eventual `GPU` build can be validated bit for bit. It is intentionally
//! distinct from two neighbours that must not be reused or re-derived here:
//! [`super::bitonic_sort`] (a comparison sorting network, not a counting sort)
//! and [`super::gpu_prefix_scan`] (the general block-decomposed Blelloch
//! `scan`). To stay self-contained this file carries its own small exclusive
//! `scan` helper rather than importing the general one, because a `radix`
//! `histogram` is short (`1 << bits` buckets) and needs only a plain serial
//! sweep.
//!
//! Everything is pure integer arithmetic: `digit` extraction is a shift and a
//! mask, counting is saturating-free `u32` addition bounded by the input length,
//! and nothing panics on empty input or divides by zero.

use alloc::vec;
use alloc::vec::Vec;

use super::gpu_layout;

/// Configuration for a `radix` sort: the number of key bits consumed per
/// `digit` pass.
///
/// A `bits` value of `1`, `2`, `4`, or `8` divides the 32-bit key space into a
/// whole number of passes with `1 << bits` buckets each. The value is clamped
/// into the inclusive range `1..=8` so a bucket array never explodes and a
/// degenerate zero can never yield a single-bucket no-op.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RadixConfig {
    /// Bits consumed per `digit` pass (clamped to `1..=8`).
    pub bits: u32,
}

impl RadixConfig {
    /// Builds a config with the given per-pass bit width, clamped to `1..=8`.
    #[must_use]
    pub fn new(bits: u32) -> Self {
        Self {
            bits: bits.clamp(1, 8),
        }
    }

    /// Number of buckets a single pass distributes keys into (`1 << bits`).
    #[must_use]
    pub fn bucket_count(self) -> usize {
        1usize << self.bits
    }

    /// Number of `digit` passes needed to cover all 32 key bits at this width.
    ///
    /// Uses integer `div_ceil` so a non-divisor width still covers the top
    /// bits (the final pass simply sees zero-padded high `digit`s).
    #[must_use]
    pub fn pass_count(self) -> u32 {
        32u32.div_ceil(self.bits)
    }

    /// Byte size of the `std430` storage buffer holding one pass `histogram`
    /// (one `u32` per bucket), via the shared [`gpu_layout`] stride rule.
    #[must_use]
    pub fn histogram_bytes(self) -> usize {
        gpu_layout::storage_bytes(gpu_layout::U32_STRIDE, self.bucket_count())
    }

    /// Byte size of the `std430` storage buffer holding the per-bucket start
    /// offsets (one `u32` per bucket); identical in shape to the `histogram`.
    #[must_use]
    pub fn offsets_bytes(self) -> usize {
        gpu_layout::storage_bytes(gpu_layout::U32_STRIDE, self.bucket_count())
    }

    /// Runs a full multi-pass `LSD` `radix` sort for this configuration.
    #[must_use]
    pub fn sort(self, keys: &[u32]) -> Vec<u32> {
        radix_sort_u32(keys, self.bits)
    }
}

/// Clamps a raw `bits` request into the supported `1..=8` per-pass range.
fn clamp_bits(bits: u32) -> u32 {
    bits.clamp(1, 8)
}

/// Bucket mask for a `bits`-wide `digit`: the low `bits` bits set (`(1 << bits) - 1`).
fn digit_mask(bits: u32) -> u32 {
    (1u32 << bits) - 1
}

/// Extracts the `bits`-wide `digit` at `pass` from `key`.
///
/// Shifts the key right by `pass * bits` (so pass `0` is the least-significant
/// `digit`) and masks off the low `bits` bits. A shift that would reach or pass
/// bit 32 yields `0`, matching the zero-padded high `digit`s a real device sees
/// on its final pass.
#[must_use]
pub fn extract_digit(key: u32, pass: u32, bits: u32) -> u32 {
    let bits = clamp_bits(bits);
    let shift = pass.saturating_mul(bits);
    if shift >= u32::BITS {
        return 0;
    }
    (key >> shift) & digit_mask(bits)
}

/// Builds the single-pass `histogram`: bucket `d` holds the number of keys
/// whose `pass` `digit` equals `d`.
///
/// The returned vector always has `1 << bits` entries and its elements sum to
/// `keys.len()`.
#[must_use]
pub fn histogram(keys: &[u32], pass: u32, bits: u32) -> Vec<u32> {
    let bits = clamp_bits(bits);
    let mut hist = vec![0u32; 1usize << bits];
    for &key in keys {
        let d = extract_digit(key, pass, bits) as usize;
        hist[d] = hist[d].wrapping_add(1);
    }
    hist
}

/// Serial exclusive prefix `scan` of a `histogram` into per-bucket start
/// offsets: entry `d` holds the sum of every earlier bucket's count.
///
/// This is the module's self-contained `scan` helper (distinct from the general
/// [`super::gpu_prefix_scan`]); a `histogram` is short, so a plain serial sweep
/// is the natural reference. Accumulation wraps to match a wrapping device
/// prefix `scan` exactly.
#[must_use]
pub fn bucket_offsets(hist: &[u32]) -> Vec<u32> {
    let mut offsets = Vec::with_capacity(hist.len());
    let mut acc = 0u32;
    for &count in hist {
        offsets.push(acc);
        acc = acc.wrapping_add(count);
    }
    offsets
}

/// Stable single-pass `scatter`: reorders `keys` by their `pass` `digit`,
/// preserving the input order of keys that share a `digit`.
///
/// Counts the `pass` `digit`s into a `histogram`, scans that into per-bucket
/// write cursors, then walks the input in order writing each key to its
/// bucket's next slot. Because the walk is front-to-back and every bucket keeps
/// a monotonically advancing cursor, keys with equal `digit`s keep their
/// relative order — the stability an `LSD` `radix` sort depends on.
#[must_use]
pub fn radix_pass(keys: &[u32], pass: u32, bits: u32) -> Vec<u32> {
    let bits = clamp_bits(bits);
    let hist = histogram(keys, pass, bits);
    let mut cursors = bucket_offsets(&hist);
    let mut out = vec![0u32; keys.len()];
    for &key in keys {
        let d = extract_digit(key, pass, bits) as usize;
        let slot = cursors[d] as usize;
        out[slot] = key;
        cursors[d] += 1;
    }
    out
}

/// Full multi-pass `LSD` `radix` sort of `keys` into ascending order.
///
/// Runs one stable [`radix_pass`] per `digit` from least- to most-significant
/// until all 32 bits are covered, feeding each pass's output into the next. The
/// composition of stable passes yields a fully sorted, stable result that
/// matches a naive comparison sort.
#[must_use]
pub fn radix_sort_u32(keys: &[u32], bits: u32) -> Vec<u32> {
    let bits = clamp_bits(bits);
    let passes = 32u32.div_ceil(bits);
    let mut current: Vec<u32> = keys.to_vec();
    for pass in 0..passes {
        current = radix_pass(&current, pass, bits);
    }
    current
}

/// Per-block local `histogram`s, the `CPU` twin of the tiled `GPU` count step.
///
/// Splits `keys` into `block_size`-element blocks (the last block may be short)
/// and builds an independent `histogram` for each block, exactly as one
/// `workgroup` per block would in shared memory. Summing the blocks
/// element-wise reproduces the single-block [`histogram`] (see
/// [`merge_histograms`]).
///
/// `block_size` is clamped to at least one so a degenerate zero can never cause
/// an empty-chunk divide-by-zero; an empty `keys` slice yields no blocks.
#[must_use]
pub fn blocked_histogram(keys: &[u32], pass: u32, bits: u32, block_size: usize) -> Vec<Vec<u32>> {
    let bits = clamp_bits(bits);
    let bs = block_size.max(1);
    let mut blocks: Vec<Vec<u32>> = Vec::with_capacity(keys.len().div_ceil(bs));
    for chunk in keys.chunks(bs) {
        blocks.push(histogram(chunk, pass, bits));
    }
    blocks
}

/// Merges per-block `histogram`s into one global `histogram` by summing them
/// bucket-wise, reproducing the single-block [`histogram`].
///
/// The bucket count is inferred from the first block; an empty block list (no
/// keys) yields an empty merged `histogram`, mirroring `blocked_histogram` on
/// empty input.
#[must_use]
pub fn merge_histograms(blocks: &[Vec<u32>]) -> Vec<u32> {
    let bucket_count = blocks.first().map_or(0, Vec::len);
    let mut merged = vec![0u32; bucket_count];
    for block in blocks {
        for (slot, &count) in merged.iter_mut().zip(block.iter()) {
            *slot = slot.wrapping_add(count);
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic 32-bit linear congruential generator (`LCG`); no real
    /// randomness so the tests reproduce bit for bit. Constants are the
    /// Numerical Recipes `LCG` multiplier/increment.
    struct Lcg {
        state: u32,
    }

    impl Lcg {
        fn new(seed: u32) -> Self {
            Self { state: seed }
        }

        fn next_u32(&mut self) -> u32 {
            self.state = self
                .state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            self.state
        }

        fn fill(&mut self, count: usize) -> Vec<u32> {
            let mut out = Vec::with_capacity(count);
            for _ in 0..count {
                out.push(self.next_u32());
            }
            out
        }
    }

    /// Naive comparison sort used as the golden reference.
    fn naive_sort(keys: &[u32]) -> Vec<u32> {
        let mut v = keys.to_vec();
        v.sort_unstable();
        v
    }

    #[test]
    fn config_clamps_bits_and_derives_shape() {
        assert_eq!(RadixConfig::new(0).bits, 1);
        assert_eq!(RadixConfig::new(99).bits, 8);
        let cfg = RadixConfig::new(4);
        assert_eq!(cfg.bucket_count(), 16);
        assert_eq!(cfg.pass_count(), 8);
        assert_eq!(RadixConfig::new(8).pass_count(), 4);
        assert_eq!(RadixConfig::new(3).pass_count(), 11);
    }

    #[test]
    fn extract_digit_shifts_and_masks() {
        // 0xABCD with 4-bit digits: pass 0 = 0xD, 1 = 0xC, 2 = 0xB, 3 = 0xA.
        assert_eq!(extract_digit(0xABCD, 0, 4), 0xD);
        assert_eq!(extract_digit(0xABCD, 1, 4), 0xC);
        assert_eq!(extract_digit(0xABCD, 2, 4), 0xB);
        assert_eq!(extract_digit(0xABCD, 3, 4), 0xA);
        // 8-bit digits partition into bytes.
        assert_eq!(extract_digit(0x1234_5678, 0, 8), 0x78);
        assert_eq!(extract_digit(0x1234_5678, 3, 8), 0x12);
        // 1-bit digits read individual bits.
        assert_eq!(extract_digit(0b1010, 0, 1), 0);
        assert_eq!(extract_digit(0b1010, 1, 1), 1);
    }

    #[test]
    fn extract_digit_past_word_is_zero() {
        // A pass whose shift reaches or passes bit 32 sees zero-padded highs.
        assert_eq!(extract_digit(u32::MAX, 8, 4), 0);
        assert_eq!(extract_digit(u32::MAX, 32, 1), 0);
        assert_eq!(extract_digit(u32::MAX, 4, 8), 0);
    }

    #[test]
    fn histogram_sum_equals_len() {
        let mut lcg = Lcg::new(0x1234_5678);
        let keys = lcg.fill(500);
        for bits in [1u32, 2, 4, 8] {
            let passes = 32u32.div_ceil(bits);
            for pass in 0..passes {
                let hist = histogram(&keys, pass, bits);
                assert_eq!(hist.len(), 1usize << bits);
                let sum: u32 = hist.iter().sum();
                assert_eq!(sum as usize, keys.len());
            }
        }
    }

    #[test]
    fn histogram_counts_specific_digits() {
        let keys = [0x00u32, 0x10, 0x20, 0x21, 0x21];
        // 4-bit pass 1 (the second nibble): 0,1,2,2,2.
        let hist = histogram(&keys, 1, 4);
        assert_eq!(hist[0], 1);
        assert_eq!(hist[1], 1);
        assert_eq!(hist[2], 3);
        assert_eq!(hist[3], 0);
    }

    #[test]
    fn bucket_offsets_are_exclusive_scan() {
        let hist = [3u32, 0, 5, 2];
        let offsets = bucket_offsets(&hist);
        assert_eq!(offsets, vec![0, 3, 3, 8]);
    }

    #[test]
    fn bucket_offsets_empty_is_empty() {
        assert_eq!(bucket_offsets(&[]), Vec::<u32>::new());
    }

    #[test]
    fn radix_pass_is_stable_within_a_digit() {
        // All keys share pass-0 4-bit digit 0x1, so a stable pass must keep
        // their input order (distinct high bits make order observable).
        let keys = [0x31u32, 0x11, 0x51, 0x21, 0x41];
        let out = radix_pass(&keys, 0, 4);
        assert_eq!(out, keys.to_vec());
    }

    #[test]
    fn radix_pass_partitions_by_digit_then_preserves_order() {
        // Pass-0 4-bit digits: 2,1,2,1,3 -> buckets [1,1 | 2,2 | 3] keeping
        // relative order inside each bucket.
        let keys = [0xA2u32, 0xB1, 0xC2, 0xD1, 0xE3];
        let out = radix_pass(&keys, 0, 4);
        assert_eq!(out, vec![0xB1, 0xD1, 0xA2, 0xC2, 0xE3]);
    }

    #[test]
    fn radix_sort_matches_naive_across_bit_widths() {
        let mut lcg = Lcg::new(0xC0FF_EE01);
        for &len in &[0usize, 1, 2, 7, 63, 256, 1000] {
            let keys = lcg.fill(len);
            let expected = naive_sort(&keys);
            for bits in [1u32, 2, 4, 8] {
                assert_eq!(
                    radix_sort_u32(&keys, bits),
                    expected,
                    "len {len} bits {bits}"
                );
            }
        }
    }

    #[test]
    fn radix_sort_handles_boundaries_and_duplicates() {
        let keys = [
            u32::MAX,
            0,
            u32::MAX,
            1,
            0,
            0x8000_0000,
            0x7FFF_FFFF,
            42,
            42,
        ];
        let expected = naive_sort(&keys);
        for bits in [1u32, 2, 4, 8] {
            assert_eq!(radix_sort_u32(&keys, bits), expected);
        }
    }

    #[test]
    fn radix_sort_empty_and_single() {
        assert_eq!(radix_sort_u32(&[], 8), Vec::<u32>::new());
        assert_eq!(radix_sort_u32(&[7], 4), vec![7]);
    }

    #[test]
    fn radix_config_sort_matches_free_function() {
        let mut lcg = Lcg::new(0x0BAD_F00D);
        let keys = lcg.fill(300);
        let cfg = RadixConfig::new(4);
        assert_eq!(cfg.sort(&keys), radix_sort_u32(&keys, 4));
    }

    #[test]
    fn blocked_histogram_merges_to_single_block() {
        let mut lcg = Lcg::new(0xFEED_BEEF);
        let keys = lcg.fill(777);
        for bits in [1u32, 2, 4, 8] {
            for &block_size in &[1usize, 8, 64, 256, 2000] {
                for pass in [0u32, 1, 3] {
                    let blocks = blocked_histogram(&keys, pass, bits, block_size);
                    let merged = merge_histograms(&blocks);
                    assert_eq!(merged, histogram(&keys, pass, bits));
                }
            }
        }
    }

    #[test]
    fn blocked_histogram_block_count_and_widths() {
        let keys = [1u32, 2, 3, 4, 5];
        let blocks = blocked_histogram(&keys, 0, 4, 2);
        // ceil(5 / 2) == 3 blocks, each a full-width histogram.
        assert_eq!(blocks.len(), 3);
        for block in &blocks {
            assert_eq!(block.len(), 16);
        }
    }

    #[test]
    fn blocked_histogram_empty_yields_no_blocks() {
        let blocks = blocked_histogram(&[], 0, 8, 64);
        assert!(blocks.is_empty());
        assert_eq!(merge_histograms(&blocks), Vec::<u32>::new());
    }

    #[test]
    fn blocked_histogram_clamps_zero_block_size() {
        let keys = [1u32, 2, 3];
        // A zero block size clamps to one, so one block per key.
        let blocks = blocked_histogram(&keys, 0, 4, 0);
        assert_eq!(blocks.len(), 3);
    }

    #[test]
    fn std430_buffer_bytes_align() {
        let cfg = RadixConfig::new(8);
        // 256 buckets * 4 bytes each.
        assert_eq!(cfg.histogram_bytes(), 256 * gpu_layout::U32_STRIDE);
        assert_eq!(cfg.offsets_bytes(), 256 * gpu_layout::U32_STRIDE);
        assert_eq!(cfg.histogram_bytes() % gpu_layout::U32_STRIDE, 0);
        // A 1-bit config still yields a non-empty, aligned buffer.
        let tiny = RadixConfig::new(1);
        assert_eq!(tiny.histogram_bytes(), 2 * gpu_layout::U32_STRIDE);
        assert_eq!(tiny.offsets_bytes() % gpu_layout::U32_STRIDE, 0);
    }
}
