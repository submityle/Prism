//! Block-decomposed *segmented* prefix scan: the `CPU`-verifiable gold standard
//! for the particle subsystem's `GPU` segmented exclusive/inclusive `scan`
//! primitive (design §5.2 compaction, §12 sort offsets, §11 counters).
//!
//! A *segmented* scan is the per-segment cousin of the plain scan in
//! [`super::gpu_prefix_scan`]: alongside the value array it consumes a
//! *head-flag* array, where a set flag marks the first element of a new segment.
//! Wherever a head flag is set the running accumulator resets to the neutral
//! element, so the scan restarts independently inside every segment. Production
//! `GPU` VFX and sort stacks use this to offset many independent runs at once —
//! per-emitter compaction runs, per-digit radix buckets, per-tile particle
//! lists — in a single dispatch, rather than launching one plain scan per run.
//!
//! On the device this cannot be a single serial sweep: the array is split into
//! fixed-size blocks, one `workgroup` segment-scans each block in shared memory,
//! each block publishes both its trailing *open* segment sum (the tail after its
//! last head flag) and whether it contains any head flag at all, a second pass
//! turns those per-block summaries into a carry that flows into the *leading*
//! open segment of the following block, and a final pass adds that carry back —
//! but only to the elements ahead of the block's first head flag, because a head
//! flag severs the carry so no sum ever leaks across a segment boundary.
//!
//! This module owns only the deterministic `CPU` reference of that scheme so the
//! eventual `GPU` build can be validated bit for bit. It is deliberately
//! distinct from [`super::gpu_prefix_scan`], which is the *non-segmented*
//! block-decomposed scan (`ScanConfig` / `exclusive_scan_blocked` /
//! `inclusive_scan` / `naive_exclusive`) with no head flags and no per-segment
//! reset: a segmented scan whose flag array is all-zero degenerates exactly into
//! that plain scan. Every name here carries the `segmented_` prefix and pairs
//! with [`SegmentedScanConfig`] so the two contracts never collide.
//!
//! Everything is pure integer arithmetic with `wrapping_add` accumulation, so it
//! matches a wrapping serial reference bit for bit, never panics on empty or
//! ragged input, tolerates a flag array shorter than the value array (missing
//! flags read as "not a head"), and never divides by zero.

use alloc::vec;
use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE};

/// Result of a block-decomposed segmented [`scan`](SegmentedScanConfig::exclusive).
///
/// Mirrors the two `GPU` storage buffers the layered segmented scan binds: the
/// per-element scanned offsets and the per-block *carry-in* array (the
/// open-segment sum entering each block, the intermediate a second pass
/// consumes — the segmented analogue of the plain scan's `block_sums`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentedScanResult {
    /// Per-element prefix within its segment, in input order (exclusive or
    /// inclusive depending on which entry point produced it).
    pub scanned: Vec<u32>,
    /// Per-block carry-in: the open-segment sum flowing into each block from the
    /// blocks before it, one entry per block. A block whose first element is a
    /// head flag ignores its carry-in.
    pub block_carry_ins: Vec<u32>,
}

/// Configuration for a block-decomposed segmented [`scan`](SegmentedScanConfig::exclusive):
/// the block size each simulated `workgroup` segment-scans.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SegmentedScanConfig {
    /// Elements segment-scanned per block. Any positive value is valid; the
    /// block-local reference sweeps the block directly, so a non-power-of-two
    /// size needs no padding.
    pub block_size: usize,
}

impl SegmentedScanConfig {
    /// Builds a config with the given block size, clamped to at least one so a
    /// degenerate zero can never cause a divide-by-zero.
    #[must_use]
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size: block_size.max(1),
        }
    }

    /// Whether the configured block size is a power of two (the natural
    /// `workgroup` width for a shared-memory segmented scan).
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
    /// offsets, via the shared [`gpu_layout`](crate::particle::gpu_layout)
    /// stride rule.
    #[must_use]
    pub fn scanned_bytes(self, len: usize) -> usize {
        storage_bytes(U32_STRIDE, len)
    }

    /// Byte size of the `std430` storage buffer holding the `len` head flags
    /// (one `u32` each) the segmented scan reads alongside its values.
    #[must_use]
    pub fn flag_bytes(self, len: usize) -> usize {
        storage_bytes(U32_STRIDE, len)
    }

    /// Byte size of the `std430` storage buffer holding the per-block carry-in
    /// (`u32` each) for `len` input elements.
    #[must_use]
    pub fn block_carry_bytes(self, len: usize) -> usize {
        storage_bytes(U32_STRIDE, self.num_blocks(len))
    }

    /// Runs the block-decomposed segmented *exclusive* scan for this config.
    #[must_use]
    pub fn exclusive(self, values: &[u32], flags: &[u32]) -> SegmentedScanResult {
        segmented_exclusive_scan(values, flags, self.block_size)
    }

    /// Runs the block-decomposed segmented *inclusive* scan for this config.
    #[must_use]
    pub fn inclusive(self, values: &[u32], flags: &[u32]) -> SegmentedScanResult {
        segmented_inclusive_scan(values, flags, self.block_size)
    }
}

/// Whether position `index` opens a new segment, i.e. its head flag is set.
///
/// A flag array shorter than the value array reads as "not a head" past its end,
/// so ragged input never panics. Only the plain integer "is non-zero" test is
/// used; there is no floating-point comparison anywhere in this module.
fn is_head(flags: &[u32], index: usize) -> bool {
    flags.get(index).copied().unwrap_or(0) != 0
}

/// Block-decomposed segmented exclusive scan of `values` under `flags`, using
/// blocks of `block_size` elements.
///
/// The exclusive prefix at each position is the `wrapping_add` sum of the
/// earlier values *in the same segment*; the first element of every segment
/// scans to the neutral element `0`. Passing an all-zero flag array reproduces a
/// plain exclusive scan exactly.
#[must_use]
pub fn segmented_exclusive_scan(
    values: &[u32],
    flags: &[u32],
    block_size: usize,
) -> SegmentedScanResult {
    segmented_scan_blocked(values, flags, block_size, false)
}

/// Block-decomposed segmented inclusive scan of `values` under `flags`, using
/// blocks of `block_size` elements.
///
/// The inclusive prefix at each position is the `wrapping_add` sum of the
/// earlier values in the same segment plus the element itself, so it equals the
/// exclusive prefix plus the value at every position.
#[must_use]
pub fn segmented_inclusive_scan(
    values: &[u32],
    flags: &[u32],
    block_size: usize,
) -> SegmentedScanResult {
    segmented_scan_blocked(values, flags, block_size, true)
}

/// Shared three-pass engine behind the exclusive/inclusive entry points.
///
/// Pass one segment-scans each block independently (accumulator from `0`) and
/// records, per block, whether it holds any head flag and its trailing open-sum
/// (the tail after its last head flag, or the whole block when it holds none).
/// Pass two turns those summaries into each block's carry-in: a block with a
/// head flag resets and passes only its trailing open-sum forward, while a block
/// with none extends the inherited open segment. Pass three adds the carry-in to
/// the leading elements of each block up to its first head flag, which severs the
/// carry so no segment ever inherits a sum from a prior segment.
fn segmented_scan_blocked(
    values: &[u32],
    flags: &[u32],
    block_size: usize,
    inclusive: bool,
) -> SegmentedScanResult {
    let len = values.len();
    let block = block_size.max(1);
    let num_blocks = len.div_ceil(block);

    let mut scanned = vec![0u32; len];
    let mut block_has_head = vec![false; num_blocks];
    let mut block_open_sum = vec![0u32; num_blocks];

    // Pass 1: independent block-local exclusive segment scan (accumulator from
    // the neutral element), recording each block's head presence and open-sum.
    for (block_idx, (((scanned_chunk, value_chunk), has_head_slot), open_slot)) in scanned
        .chunks_mut(block)
        .zip(values.chunks(block))
        .zip(block_has_head.iter_mut())
        .zip(block_open_sum.iter_mut())
        .enumerate()
    {
        let start = block_idx * block;
        let mut acc = 0u32;
        let mut has_head = false;
        for (offset, (slot, &value)) in scanned_chunk.iter_mut().zip(value_chunk.iter()).enumerate()
        {
            if is_head(flags, start + offset) {
                acc = 0;
                has_head = true;
            }
            *slot = acc;
            acc = acc.wrapping_add(value);
        }
        *has_head_slot = has_head;
        *open_slot = acc;
    }

    // Pass 2: sequentially thread the open-segment carry across blocks. A block
    // that holds a head flag restarts the carry from its own trailing open-sum;
    // a block with none extends the inherited open segment.
    let mut block_carry_ins = vec![0u32; num_blocks];
    let mut carry = 0u32;
    for ((carry_slot, &has_head), &open_sum) in block_carry_ins
        .iter_mut()
        .zip(block_has_head.iter())
        .zip(block_open_sum.iter())
    {
        *carry_slot = carry;
        if has_head {
            carry = open_sum;
        } else {
            carry = carry.wrapping_add(open_sum);
        }
    }

    // Pass 3: fold each block's carry-in into its leading open segment only, up
    // to (but never past) the block's first head flag.
    for (block_idx, (scanned_chunk, &carry_in)) in scanned
        .chunks_mut(block)
        .zip(block_carry_ins.iter())
        .enumerate()
    {
        let start = block_idx * block;
        for (offset, slot) in scanned_chunk.iter_mut().enumerate() {
            if is_head(flags, start + offset) {
                break;
            }
            *slot = slot.wrapping_add(carry_in);
        }
    }

    // The exclusive result is complete; the inclusive result adds each element
    // to its own exclusive prefix (still per-segment, since the reset already
    // shaped the exclusive prefixes).
    if inclusive {
        for (slot, &value) in scanned.iter_mut().zip(values.iter()) {
            *slot = slot.wrapping_add(value);
        }
    }

    SegmentedScanResult {
        scanned,
        block_carry_ins,
    }
}

/// Serial one-sweep reference segmented *exclusive* scan, the naive oracle the
/// block-decomposed path is checked against.
///
/// Walks the array once, resetting the accumulator to the neutral element at
/// every head flag; the value written before adding the current element is the
/// exclusive per-segment prefix.
#[must_use]
pub fn naive_segmented_exclusive(values: &[u32], flags: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(values.len());
    let mut acc = 0u32;
    for (i, &value) in values.iter().enumerate() {
        if is_head(flags, i) {
            acc = 0;
        }
        out.push(acc);
        acc = acc.wrapping_add(value);
    }
    out
}

/// Serial one-sweep reference segmented *inclusive* scan, the naive oracle the
/// block-decomposed path is checked against.
///
/// Identical to [`naive_segmented_exclusive`] except the current element is
/// folded in before the value is written, giving the inclusive per-segment
/// prefix.
#[must_use]
pub fn naive_segmented_inclusive(values: &[u32], flags: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(values.len());
    let mut acc = 0u32;
    for (i, &value) in values.iter().enumerate() {
        if is_head(flags, i) {
            acc = 0;
        }
        acc = acc.wrapping_add(value);
        out.push(acc);
    }
    out
}

/// Recovers the segment start indices implied by a head-flag array: the sorted
/// positions where a head flag is set.
///
/// This is the inverse view of the flag encoding — given the flags a segmented
/// scan consumes, it reports where each segment begins, which callers use to
/// bound per-segment work without re-deriving segment boundaries by hand.
#[must_use]
pub fn segment_start_indices(flags: &[u32]) -> Vec<usize> {
    flags
        .iter()
        .enumerate()
        .filter_map(|(index, &flag)| (flag != 0).then_some(index))
        .collect()
}

/// Number of segments a head-flag array encodes, i.e. the count of set flags.
///
/// An array with no set flags still describes a single implicit segment starting
/// at index zero, so this returns the count of *explicit* segment heads.
#[must_use]
pub fn segment_head_count(flags: &[u32]) -> usize {
    flags.iter().filter(|&&flag| flag != 0).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic linear-congruential generator for reproducible value
    /// and flag arrays; keeps the tests free of any external `rand` dependency.
    fn lcg_vec(seed: u64, len: usize, modulo: u32) -> Vec<u32> {
        let mut state = seed | 1;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let sample = u32::try_from((state >> 33) & 0xffff_ffff).expect("33-bit shift fits u32");
            out.push(if modulo == 0 { sample } else { sample % modulo });
        }
        out
    }

    /// Deterministic flag array: `1` roughly once every `period` positions plus a
    /// hash-scattered sprinkling, so segments have varied, uneven lengths.
    fn lcg_flags(seed: u64, len: usize, period: usize) -> Vec<u32> {
        let period = period.max(1);
        let mut state = seed | 1;
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            state = state
                .wrapping_mul(2_862_933_555_777_941_757)
                .wrapping_add(3_037_000_493);
            let periodic = i % period == 0;
            let scattered = (state >> 40) & 0x7 == 0;
            out.push(u32::from(periodic || scattered));
        }
        out
    }

    /// Plain exclusive scan with no segment resets, for the degenerate case.
    fn plain_exclusive(values: &[u32]) -> Vec<u32> {
        let mut out = Vec::with_capacity(values.len());
        let mut acc = 0u32;
        for &value in values {
            out.push(acc);
            acc = acc.wrapping_add(value);
        }
        out
    }

    #[test]
    fn all_zero_flags_degenerate_to_plain_scan() {
        let values = lcg_vec(0x1111_2222_3333_4444, 40, 100);
        let flags = vec![0u32; values.len()];
        let expected = plain_exclusive(&values);
        for &bs in &[1usize, 2, 3, 4, 8, 16, 64] {
            let result = segmented_exclusive_scan(&values, &flags, bs);
            assert_eq!(result.scanned, expected, "block_size={bs}");
            assert_eq!(
                result.scanned,
                naive_segmented_exclusive(&values, &flags),
                "block_size={bs}"
            );
        }
    }

    #[test]
    fn every_element_is_a_head() {
        let values = lcg_vec(0xaaaa_bbbb_cccc_dddd, 33, 500);
        let flags = vec![1u32; values.len()];
        for &bs in &[1usize, 2, 4, 7, 16] {
            let exclusive = segmented_exclusive_scan(&values, &flags, bs);
            // Each element is its own singleton segment: exclusive prefixes are
            // all the neutral element, inclusive prefixes are the values.
            assert_eq!(
                exclusive.scanned,
                vec![0u32; values.len()],
                "block_size={bs}"
            );
            let inclusive = segmented_inclusive_scan(&values, &flags, bs);
            assert_eq!(inclusive.scanned, values, "block_size={bs}");
        }
    }

    #[test]
    fn multiple_segments_reset_independently() {
        // Segments: [1,2,3] [4,5] [6] with heads at 0, 3, 5.
        let values = [1u32, 2, 3, 4, 5, 6];
        let flags = [1u32, 0, 0, 1, 0, 1];
        let exclusive = segmented_exclusive_scan(&values, &flags, 2);
        assert_eq!(exclusive.scanned, vec![0, 1, 3, 0, 4, 0]);
        let inclusive = segmented_inclusive_scan(&values, &flags, 2);
        assert_eq!(inclusive.scanned, vec![1, 3, 6, 4, 9, 6]);
        assert_eq!(
            exclusive.scanned,
            naive_segmented_exclusive(&values, &flags)
        );
        assert_eq!(
            inclusive.scanned,
            naive_segmented_inclusive(&values, &flags)
        );
    }

    #[test]
    fn exclusive_plus_value_equals_inclusive() {
        for &len in &[0usize, 1, 2, 3, 8, 17, 64, 130] {
            let values = lcg_vec(0xfeed_face_0000_0001 ^ (len as u64), len, 777);
            let flags = lcg_flags(0x0bad_c0de_0000_0001 ^ (len as u64), len, 5);
            for &bs in &[1usize, 3, 8, 32] {
                let exclusive = segmented_exclusive_scan(&values, &flags, bs);
                let inclusive = segmented_inclusive_scan(&values, &flags, bs);
                let expected: Vec<u32> = exclusive
                    .scanned
                    .iter()
                    .zip(values.iter())
                    .map(|(&e, &v)| e.wrapping_add(v))
                    .collect();
                assert_eq!(inclusive.scanned, expected, "len={len} block_size={bs}");
            }
        }
    }

    #[test]
    fn carry_flows_across_block_boundary_inside_a_segment() {
        // One long segment (no interior heads) spanning many blocks must carry
        // the running sum across every block boundary, matching the plain scan.
        let values = lcg_vec(0x5151_5151_2626_2626, 100, 50);
        let mut flags = vec![0u32; values.len()];
        flags[0] = 1;
        let expected = plain_exclusive(&values);
        for &bs in &[1usize, 2, 3, 8, 16, 32] {
            let result = segmented_exclusive_scan(&values, &flags, bs);
            assert_eq!(result.scanned, expected, "block_size={bs}");
        }
    }

    #[test]
    fn head_at_block_start_severs_the_carry() {
        // Block size 4: a head exactly on the second block's first element must
        // ignore the carry from block 0 and restart at the neutral element.
        let values = [10u32, 20, 30, 40, 5, 6, 7, 8];
        let mut flags = vec![0u32; values.len()];
        flags[0] = 1;
        flags[4] = 1;
        let result = segmented_exclusive_scan(&values, &flags, 4);
        assert_eq!(result.scanned, vec![0, 10, 30, 60, 0, 5, 11, 18]);
        assert_eq!(result.scanned, naive_segmented_exclusive(&values, &flags));
    }

    #[test]
    fn matches_naive_on_random_arrays_across_block_sizes() {
        for &len in &[5usize, 7, 13, 31, 33, 100, 257, 512] {
            let values = lcg_vec(0xdead_beef_0000_0001 ^ (len as u64), len, 1000);
            let flags = lcg_flags(0xcafe_f00d_0000_0001 ^ (len as u64), len, 6);
            let expected_ex = naive_segmented_exclusive(&values, &flags);
            let expected_in = naive_segmented_inclusive(&values, &flags);
            for &bs in &[1usize, 2, 4, 5, 8, 16, 32, 64] {
                let exclusive = segmented_exclusive_scan(&values, &flags, bs);
                let inclusive = segmented_inclusive_scan(&values, &flags, bs);
                assert_eq!(exclusive.scanned, expected_ex, "len={len} block_size={bs}");
                assert_eq!(inclusive.scanned, expected_in, "len={len} block_size={bs}");
            }
        }
    }

    #[test]
    fn empty_input_is_empty() {
        let result = segmented_exclusive_scan(&[], &[], 8);
        assert!(result.scanned.is_empty());
        assert!(result.block_carry_ins.is_empty());
        let inclusive = segmented_inclusive_scan(&[], &[], 8);
        assert!(inclusive.scanned.is_empty());
        assert_eq!(SegmentedScanConfig::new(8).num_blocks(0), 0);
    }

    #[test]
    fn single_element_input() {
        let result = segmented_exclusive_scan(&[42u32], &[1u32], 4);
        assert_eq!(result.scanned, vec![0]);
        let inclusive = segmented_inclusive_scan(&[42u32], &[0u32], 4);
        assert_eq!(inclusive.scanned, vec![42]);
    }

    #[test]
    fn first_head_may_be_absent() {
        // No flag at index 0: the array still opens an implicit first segment.
        let values = [3u32, 4, 5, 6];
        let flags = [0u32, 0, 1, 0];
        let result = segmented_exclusive_scan(&values, &flags, 2);
        assert_eq!(result.scanned, vec![0, 3, 0, 5]);
        assert_eq!(result.scanned, naive_segmented_exclusive(&values, &flags));
    }

    #[test]
    fn wrapping_accumulation_within_segment() {
        // Values that overflow u32 aggregate; both paths wrap identically.
        let values = vec![u32::MAX; 10];
        let flags = vec![0u32; 10];
        let result = segmented_exclusive_scan(&values, &flags, 4);
        assert_eq!(result.scanned, naive_segmented_exclusive(&values, &flags));
        assert_eq!(result.scanned[9], u32::MAX.wrapping_mul(9));
    }

    #[test]
    fn ragged_flag_array_reads_missing_as_non_head() {
        // Flags shorter than values: positions past the flag end are not heads.
        let values = [1u32, 2, 3, 4, 5];
        let flags = [1u32, 0]; // only two flags provided
        let result = segmented_exclusive_scan(&values, &flags, 2);
        assert_eq!(result.scanned, vec![0, 1, 3, 6, 10]);
        assert_eq!(result.scanned, naive_segmented_exclusive(&values, &flags));
    }

    #[test]
    fn segment_start_indices_and_head_count() {
        let flags = [1u32, 0, 0, 1, 0, 1, 0];
        assert_eq!(segment_start_indices(&flags), vec![0, 3, 5]);
        assert_eq!(segment_head_count(&flags), 3);
        assert_eq!(segment_start_indices(&[0u32, 0, 0]), Vec::<usize>::new());
        assert_eq!(segment_head_count(&[0u32, 0, 0]), 0);
    }

    #[test]
    fn config_helpers_and_std430_bytes() {
        let config = SegmentedScanConfig::new(8);
        assert!(config.is_power_of_two_block());
        assert_eq!(config.num_blocks(0), 0);
        assert_eq!(config.num_blocks(8), 1);
        assert_eq!(config.num_blocks(9), 2);

        // std430: scanned offsets, head flags, and per-block carries are tightly
        // packed u32 buffers, clamped up to a single element for an empty pool.
        assert_eq!(config.scanned_bytes(0), U32_STRIDE);
        assert_eq!(config.scanned_bytes(10), 10 * U32_STRIDE);
        assert_eq!(config.flag_bytes(0), U32_STRIDE);
        assert_eq!(config.flag_bytes(10), 10 * U32_STRIDE);
        assert_eq!(config.block_carry_bytes(0), U32_STRIDE);
        assert_eq!(config.block_carry_bytes(17), 3 * U32_STRIDE);
    }

    #[test]
    fn config_clamps_zero_block_size_and_scans() {
        let config = SegmentedScanConfig::new(0);
        assert_eq!(config.block_size, 1);
        assert!(config.is_power_of_two_block());
        let values = [4u32, 8, 15, 16, 23, 42];
        let flags = [1u32, 0, 1, 0, 0, 1];
        assert_eq!(
            config.exclusive(&values, &flags).scanned,
            naive_segmented_exclusive(&values, &flags)
        );
        assert_eq!(
            config.inclusive(&values, &flags).scanned,
            naive_segmented_inclusive(&values, &flags)
        );
    }

    #[test]
    fn per_block_carry_ins_track_open_segment() {
        // One open segment across two blocks of size 4: block 0 carry-in is the
        // neutral element, block 1 carry-in is the sum of block 0.
        let values = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let flags = vec![0u32; values.len()];
        let result = segmented_exclusive_scan(&values, &flags, 4);
        assert_eq!(result.block_carry_ins, vec![0, 10]);
    }
}
