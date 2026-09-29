//! Hierarchical multi-block Blelloch prefix scan: the `CPU`-verifiable gold
//! standard for the particle subsystem's `GPU` work-efficient exclusive/inclusive
//! `scan` primitive (design §5.2 compaction, §11 counters, §12 sort offsets).
//!
//! Production `GPU` VFX stacks (Unreal `Niagara`'s spawn/compaction counters,
//! `Frostbite`'s FX bucket offsets, and every radix-sort digit histogram) turn a
//! per-element count array into per-element start offsets with a *prefix scan*.
//! On the device this cannot be a single serial sweep: the array is split into
//! fixed-size blocks, one `workgroup` scans each block in shared memory with the
//! work-efficient Blelloch algorithm (an `up-sweep` reduction followed by a
//! `down-sweep`), the per-block totals are collected into a `block_sums` array,
//! that array is itself scanned (recursively, block by block), and finally each
//! block's exclusive offset is added back to its elements. The result is a
//! work-efficient `O(n)` scan that composes across arbitrarily many `workgroup`s.
//!
//! This module owns only the deterministic `CPU` reference of that layered
//! scheme so the eventual `GPU` build can be validated bit for bit. It is
//! distinct from two neighbours that must not be reused or re-derived here:
//! [`super::spatial_hash`]'s single-array `prefix_sum` (one serial exclusive
//! sweep of per-cell counts) and [`super::pool`]'s stream compaction. This file
//! is the general, block-decomposed scan that both of those are special cases
//! of, and it supports any input length and any power-of-two block size.
//!
//! Everything is pure integer arithmetic with `wrapping_add` accumulation, so it
//! matches a wrapping serial reference bit for bit, never panics on empty or
//! ragged input, and never divides by zero.

use alloc::vec;
use alloc::vec::Vec;

use super::gpu_layout;

/// Result of a hierarchical multi-block exclusive [`scan`](exclusive_scan_blocked).
///
/// Mirrors the three `GPU` storage buffers the layered scan binds: the scanned
/// element offsets, the per-block totals (the intermediate `block_sums` buffer a
/// second `scan` pass consumes), and the grand total of every input element.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScanResult {
    /// Exclusive prefix offset of every input element, in input order.
    pub scanned: Vec<u32>,
    /// Per-block totals, one entry per block, before the block-sum `scan`.
    pub block_sums: Vec<u32>,
    /// Grand total of all input elements (wrapping), i.e. the exclusive offset
    /// just past the final element.
    pub total: u32,
}

/// Configuration for a hierarchical [`scan`](ScanConfig::scan): the block size
/// each simulated `workgroup` scans.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScanConfig {
    /// Elements scanned per block. Should be a power of two; a non-power-of-two
    /// value is padded up internally so the block-local Blelloch scan stays
    /// correct.
    pub block_size: usize,
}

impl ScanConfig {
    /// Builds a config with the given block size, clamped to at least one so a
    /// degenerate zero can never cause a divide-by-zero.
    #[must_use]
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size: block_size.max(1),
        }
    }

    /// Whether the configured block size is a power of two (the natural
    /// `workgroup` width for a shared-memory Blelloch scan).
    #[must_use]
    pub fn is_power_of_two_block(self) -> bool {
        self.block_size.is_power_of_two()
    }

    /// Number of blocks needed to cover `len` elements at this block size.
    #[must_use]
    pub fn num_blocks(self, len: usize) -> usize {
        len.div_ceil(self.block_size.max(1))
    }

    /// Byte size of the `std430` storage buffer holding `len` scanned `u32`
    /// offsets, via the shared [`gpu_layout`] stride rule.
    #[must_use]
    pub fn scanned_bytes(self, len: usize) -> usize {
        gpu_layout::storage_bytes(gpu_layout::U32_STRIDE, len)
    }

    /// Byte size of the `std430` storage buffer holding the per-block
    /// `block_sums` (`u32` each) for `len` input elements.
    #[must_use]
    pub fn block_sums_bytes(self, len: usize) -> usize {
        gpu_layout::storage_bytes(gpu_layout::U32_STRIDE, self.num_blocks(len))
    }

    /// Runs the hierarchical exclusive scan for this configuration.
    #[must_use]
    pub fn scan(self, input: &[u32]) -> ScanResult {
        exclusive_scan_blocked(input, self.block_size)
    }
}

/// Smallest power of two greater than or equal to `n` (at least one).
///
/// Pure integer doubling via left shifts; no floating-point `log2`/`powf`.
fn next_power_of_two(n: usize) -> usize {
    let mut p: usize = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// In-place work-efficient Blelloch exclusive scan of a slice whose length is a
/// power of two (or zero / one).
///
/// The `up-sweep` phase builds a reduction tree in place (stride-doubling
/// partial sums); the root is then cleared and the `down-sweep` phase walks the
/// tree back down, swapping each node into its left child and adding, which
/// yields the exclusive prefix sum. Accumulation wraps so the result matches a
/// wrapping serial reference exactly.
///
/// # Panics
///
/// Debug-asserts that the length is zero or a power of two; callers that scan
/// ragged input must pad to a power of two first (see
/// [`exclusive_scan_padded`]).
fn blelloch_exclusive(data: &mut [u32]) {
    let n = data.len();
    debug_assert!(
        n == 0 || n.is_power_of_two(),
        "Blelloch scan requires a power-of-two length"
    );
    if n == 0 {
        return;
    }

    // Up-sweep (reduce): stride `d` doubles until it spans the whole array.
    let mut d = 1usize;
    while d < n {
        let step = d * 2;
        let mut i = step - 1;
        while i < n {
            data[i] = data[i].wrapping_add(data[i - d]);
            i += step;
        }
        d = step;
    }

    // Clear the root so the down-sweep produces an *exclusive* scan.
    data[n - 1] = 0;

    // Down-sweep: stride `d` halves back to one, swapping and adding.
    let mut d = n / 2;
    while d >= 1 {
        let step = d * 2;
        let mut i = step - 1;
        while i < n {
            let left = data[i - d];
            data[i - d] = data[i];
            data[i] = data[i].wrapping_add(left);
            i += step;
        }
        d /= 2;
    }
}

/// Exclusive scan of an arbitrary-length slice as a *single* block: pad up to
/// the next power of two, run one Blelloch pass, and return the scanned prefix
/// (truncated to the real length) together with the wrapping grand total.
fn exclusive_scan_padded(input: &[u32]) -> (Vec<u32>, u32) {
    let n = input.len();
    if n == 0 {
        return (Vec::new(), 0);
    }
    let total = input.iter().fold(0u32, |acc, &x| acc.wrapping_add(x));
    let mut buf = vec![0u32; next_power_of_two(n)];
    buf[..n].copy_from_slice(input);
    blelloch_exclusive(&mut buf);
    buf.truncate(n);
    (buf, total)
}

/// Exclusive scan of the per-block totals, choosing the simulated device path.
///
/// A `block_sums` array that already fits in one block (or a degenerate
/// block size below two, which can never shrink under recursion) is scanned as a
/// single Blelloch block; a larger one recurses through the full hierarchical
/// [`exclusive_scan_blocked`], exactly as a real multi-level `GPU` scan would.
fn scan_block_offsets(sums: &[u32], block_size: usize) -> Vec<u32> {
    if block_size < 2 || sums.len() <= block_size {
        exclusive_scan_padded(sums).0
    } else {
        exclusive_scan_blocked(sums, block_size).scanned
    }
}

/// Hierarchical multi-block exclusive prefix scan of `input`.
///
/// Splits `input` into `block_size`-element blocks, scans each block locally
/// with [`blelloch_exclusive`], collects the per-block totals into
/// `block_sums`, scans those into per-block offsets (recursively for very long
/// inputs), and adds each block's offset back to its elements. The block size is
/// clamped to at least one and padded up to a power of two internally so the
/// block-local scan is always valid.
#[must_use]
pub fn exclusive_scan_blocked(input: &[u32], block_size: usize) -> ScanResult {
    let bs = block_size.max(1);
    let pad = next_power_of_two(bs);
    let mut scanned: Vec<u32> = Vec::with_capacity(input.len());
    let mut block_sums: Vec<u32> = Vec::with_capacity(input.len().div_ceil(bs));

    for block in input.chunks(bs) {
        let block_total = block.iter().fold(0u32, |acc, &x| acc.wrapping_add(x));
        let mut buf = vec![0u32; pad];
        buf[..block.len()].copy_from_slice(block);
        blelloch_exclusive(&mut buf);
        scanned.extend_from_slice(&buf[..block.len()]);
        block_sums.push(block_total);
    }

    let offsets = scan_block_offsets(&block_sums, bs);
    for (block_out, &offset) in scanned.chunks_mut(bs).zip(offsets.iter()) {
        for value in block_out.iter_mut() {
            *value = value.wrapping_add(offset);
        }
    }

    let total = match (offsets.last(), block_sums.last()) {
        (Some(&last_offset), Some(&last_sum)) => last_offset.wrapping_add(last_sum),
        _ => 0,
    };

    ScanResult {
        scanned,
        block_sums,
        total,
    }
}

/// Inclusive prefix scan of `input`: the exclusive scan plus each element in
/// place, so entry `i` holds the wrapping sum of `input[0..=i]`.
#[must_use]
pub fn inclusive_scan(input: &[u32]) -> Vec<u32> {
    let (mut out, _) = exclusive_scan_padded(input);
    for (slot, &value) in out.iter_mut().zip(input.iter()) {
        *slot = slot.wrapping_add(value);
    }
    out
}

/// Straightforward serial exclusive scan, the golden reference the block-based
/// [`exclusive_scan_blocked`] is validated against.
#[must_use]
pub fn naive_exclusive(input: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(input.len());
    let mut acc = 0u32;
    for &x in input {
        out.push(acc);
        acc = acc.wrapping_add(x);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic 64-bit linear congruential generator (`LCG`) so tests never
    /// rely on real randomness; the constants are the well-known `PCG`/`MMIX`
    /// multiplier and increment.
    fn lcg_next(state: &mut u64) -> u32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // The top 31 bits of the 64-bit state always fit a `u32`.
        u32::try_from(*state >> 33).expect("31-bit value fits in u32")
    }

    fn lcg_vec(seed: u64, len: usize, modulo: u32) -> Vec<u32> {
        let mut state = seed;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(lcg_next(&mut state) % modulo);
        }
        out
    }

    #[test]
    fn single_block_blelloch_matches_naive() {
        for &bits in &[1usize, 2, 4, 8, 16, 32] {
            let input = lcg_vec(0x1234_5678_9abc_def0, bits, 1000);
            let result = exclusive_scan_blocked(&input, bits);
            assert_eq!(result.scanned, naive_exclusive(&input));
        }
    }

    #[test]
    fn blelloch_exclusive_in_place_matches_naive() {
        let input = [3u32, 1, 7, 0, 4, 1, 6, 3];
        let mut buf = input;
        blelloch_exclusive(&mut buf);
        assert_eq!(buf.to_vec(), naive_exclusive(&input));
    }

    #[test]
    fn empty_input_scans_to_empty() {
        let result = exclusive_scan_blocked(&[], 4);
        assert!(result.scanned.is_empty());
        assert!(result.block_sums.is_empty());
        assert_eq!(result.total, 0);
        assert_eq!(inclusive_scan(&[]), Vec::<u32>::new());
        assert_eq!(naive_exclusive(&[]), Vec::<u32>::new());
    }

    #[test]
    fn single_element_scans_to_zero() {
        let input = [42u32];
        let result = exclusive_scan_blocked(&input, 4);
        assert_eq!(result.scanned, vec![0]);
        assert_eq!(result.block_sums, vec![42]);
        assert_eq!(result.total, 42);
        assert_eq!(inclusive_scan(&input), vec![42]);
    }

    #[test]
    fn multi_block_ragged_lengths_match_naive() {
        // Lengths that are not multiples of the block size exercise the ragged
        // final block and the block-sum scan.
        for &len in &[3usize, 5, 7, 9, 13, 31, 33, 100, 257] {
            let input = lcg_vec(0xdead_beef_0000_0001 ^ (len as u64), len, 500);
            for &bs in &[1usize, 2, 4, 8, 16] {
                let result = exclusive_scan_blocked(&input, bs);
                assert_eq!(
                    result.scanned,
                    naive_exclusive(&input),
                    "len={len} block_size={bs}"
                );
            }
        }
    }

    #[test]
    fn total_equals_wrapping_sum() {
        let input = lcg_vec(0x0f0f_0f0f_1111_2222, 129, 1000);
        let expected = input.iter().fold(0u32, |acc, &x| acc.wrapping_add(x));
        for &bs in &[1usize, 2, 4, 8, 16, 32, 64] {
            let result = exclusive_scan_blocked(&input, bs);
            assert_eq!(result.total, expected, "block_size={bs}");
        }
    }

    #[test]
    fn block_sums_are_per_block_totals() {
        let input = [1u32, 2, 3, 4, 5, 6, 7]; // 7 elements, block size 4
        let result = exclusive_scan_blocked(&input, 4);
        // block 0 = 1+2+3+4 = 10, block 1 = 5+6+7 = 18
        assert_eq!(result.block_sums, vec![10, 18]);
        assert_eq!(result.total, 28);
    }

    #[test]
    fn inclusive_equals_exclusive_plus_input() {
        for &len in &[0usize, 1, 2, 3, 8, 17, 64, 130] {
            let input = lcg_vec(0xabcd_1234_5678_9999 ^ (len as u64), len, 777);
            let exclusive = naive_exclusive(&input);
            let inclusive = inclusive_scan(&input);
            let expected: Vec<u32> = exclusive
                .iter()
                .zip(input.iter())
                .map(|(&e, &v)| e.wrapping_add(v))
                .collect();
            assert_eq!(inclusive, expected, "len={len}");
        }
    }

    #[test]
    fn large_deterministic_array_matches_naive_across_block_sizes() {
        let input = lcg_vec(0x5a5a_5a5a_a5a5_a5a5, 4096, 4096);
        let expected = naive_exclusive(&input);
        for &bs in &[1usize, 2, 4, 8, 16, 32, 64, 128, 256, 1024] {
            let result = exclusive_scan_blocked(&input, bs);
            assert_eq!(result.scanned, expected, "block_size={bs}");
        }
    }

    #[test]
    fn deeply_recursive_block_sum_scan() {
        // Many small blocks force the block-sum scan itself to be multi-block.
        let input = lcg_vec(0x1357_9bdf_2468_ace0, 1000, 100);
        let result = exclusive_scan_blocked(&input, 2);
        assert_eq!(result.scanned, naive_exclusive(&input));
        assert_eq!(result.block_sums.len(), 500);
    }

    #[test]
    fn wrapping_accumulation_matches_serial_reference() {
        // Values large enough to overflow u32 in aggregate; both paths wrap.
        let input = vec![u32::MAX; 10];
        let result = exclusive_scan_blocked(&input, 4);
        assert_eq!(result.scanned, naive_exclusive(&input));
        assert_eq!(result.total, u32::MAX.wrapping_mul(10));
    }

    #[test]
    fn config_helpers_and_std430_bytes() {
        let config = ScanConfig::new(8);
        assert!(config.is_power_of_two_block());
        assert_eq!(config.num_blocks(0), 0);
        assert_eq!(config.num_blocks(8), 1);
        assert_eq!(config.num_blocks(9), 2);

        // std430: scanned offsets and block sums are tightly packed u32 buffers,
        // clamped up to a single element for an empty pool.
        assert_eq!(config.scanned_bytes(0), gpu_layout::U32_STRIDE);
        assert_eq!(config.scanned_bytes(10), 10 * gpu_layout::U32_STRIDE);
        assert_eq!(config.block_sums_bytes(0), gpu_layout::U32_STRIDE);
        assert_eq!(config.block_sums_bytes(17), 3 * gpu_layout::U32_STRIDE);
    }

    #[test]
    fn config_clamps_zero_block_size_and_scans() {
        let config = ScanConfig::new(0);
        assert_eq!(config.block_size, 1);
        assert!(config.is_power_of_two_block());
        let input = [4u32, 8, 15, 16, 23, 42];
        assert_eq!(config.scan(&input).scanned, naive_exclusive(&input));
    }

    #[test]
    fn non_power_of_two_block_size_is_padded_and_correct() {
        let input = lcg_vec(0x2222_3333_4444_5555, 40, 250);
        for &bs in &[3usize, 5, 6, 7, 10] {
            let result = exclusive_scan_blocked(&input, bs);
            assert_eq!(result.scanned, naive_exclusive(&input), "block_size={bs}");
        }
    }

    #[test]
    fn next_power_of_two_is_integer_only() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(1024), 1024);
        assert_eq!(next_power_of_two(1025), 2048);
    }
}
