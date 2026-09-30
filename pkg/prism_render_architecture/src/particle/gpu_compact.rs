//! Predicate-driven *stream compaction*: the `CPU`-verifiable gold standard for
//! the particle subsystem's `GPU` two-pass compaction primitive (design §5.2
//! compaction, §9 `Compaction` pass, §11 counters).
//!
//! *Stream compaction* takes a per-element predicate mask (`keep = 1` /
//! `drop = 0`) and packs every kept element down into a dense prefix of the
//! output, discarding the holes the dropped elements leave behind. Production
//! `GPU` VFX stacks lean on it constantly: Unreal `Niagara` compacts its live
//! particle list every frame so dead slots stop consuming lanes, `Frostbite`'s
//! FX stack packs survivors after culling, and every indirect-draw path packs a
//! visible-instance list before filling draw arguments. On the device this is
//! never a single serial sweep — it is the classic two-pass `scatter`:
//!
//! 1. **Block-local `scan`** — the mask is split into fixed-size blocks and one
//!    simulated `workgroup` runs an *exclusive prefix sum* over its block's keep
//!    bits in shared memory, yielding each kept element's slot *within its
//!    block* and the block's total survivor count (the `block_sums` entry).
//! 2. **Block-offset `scan`** — the per-block survivor totals are themselves
//!    exclusive-scanned into per-block base offsets (`block_bases`): the first
//!    compact slot each block owns.
//! 3. **Global `scatter`** — every kept element's destination is its block base
//!    plus its block-local offset, and the original index is scattered there.
//!
//! The exclusive prefix sum is *hand-rolled here* (a serial running-accumulator
//! sweep, [`exclusive_prefix_sum`]); this module deliberately does not import or
//! re-derive any neighbour's `scan`. Everything is pure integer arithmetic with
//! `wrapping_add` accumulation, so it matches a wrapping serial reference bit for
//! bit, never panics on empty or ragged input, and never divides by zero.
//!
//! **Deliberately out of scope (no overlap with siblings):** this file owns only
//! the `predicate → compacted-index scatter` semantics. It does not recycle
//! object-pool slots (that is [`super::pool`]'s free-list `Compaction`); it does
//! not run a *segmented* head-flag `scan` (that is
//! [`super::gpu_scan_segmented`]); it does not build radix digit histograms
//! (that is [`super::gpu_radix_histogram`]); and it is not the general-purpose
//! block-decomposed `scan` primitive (that is [`super::gpu_prefix_scan`]). Its
//! only dependency is the shared [`super::gpu_layout`] `std430` stride rule.

use alloc::vec;
use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE};

/// Result of the block-local first pass of [`exclusive_scan_blocks`].
///
/// Mirrors the two `GPU` storage buffers the compaction's first pass binds: the
/// per-element block-local destination offset and the per-block survivor totals
/// (the intermediate `block_sums` buffer the block-offset `scan` consumes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScanBlocks {
    /// Exclusive prefix sum of keep bits *within each block*, in input order.
    ///
    /// Entry `i` is the compact slot element `i` would take *relative to the
    /// start of its own block*; it resets to zero at every block boundary and
    /// is only a valid destination once the block base offset is added back.
    pub block_local: Vec<u32>,
    /// Survivor count of every block, one entry per block, before the
    /// block-offset `scan`.
    pub block_sums: Vec<u32>,
}

/// Configuration for a two-pass stream [`compaction`](CompactConfig::compact):
/// the block size each simulated `workgroup` compacts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CompactConfig {
    /// Elements compacted per block (per simulated `workgroup`). Clamped to at
    /// least one so a degenerate zero can never cause a divide-by-zero.
    pub block_size: usize,
}

impl CompactConfig {
    /// Builds a config with the given block size, clamped to at least one.
    #[must_use]
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size: block_size.max(1),
        }
    }

    /// Whether the configured block size is a power of two (the natural
    /// `workgroup` width for a shared-memory `scan`).
    #[must_use]
    pub fn is_power_of_two_block(self) -> bool {
        self.block_size.is_power_of_two()
    }

    /// Number of blocks (simulated `workgroup`s) needed to cover `len` elements.
    #[must_use]
    pub fn num_blocks(self, len: usize) -> usize {
        len.div_ceil(self.block_size.max(1))
    }

    /// Byte size of the `std430` predicate-flag buffer holding `len` `u32` keep
    /// bits, via the shared [`super::gpu_layout`] stride rule.
    #[must_use]
    pub fn flag_bytes(self, len: usize) -> usize {
        storage_bytes(U32_STRIDE, len)
    }

    /// Byte size of the `std430` buffer holding the `len` per-element `scatter`
    /// destination offsets (`u32` each).
    #[must_use]
    pub fn scanned_bytes(self, len: usize) -> usize {
        storage_bytes(U32_STRIDE, len)
    }

    /// Byte size of the `std430` `block_sums` buffer (one `u32` per block) for
    /// `len` input elements.
    #[must_use]
    pub fn block_sums_bytes(self, len: usize) -> usize {
        storage_bytes(U32_STRIDE, self.num_blocks(len))
    }

    /// Byte size of the `std430` compacted-index output buffer holding `count`
    /// survivors (`u32` each).
    #[must_use]
    pub fn compacted_bytes(self, count: usize) -> usize {
        storage_bytes(U32_STRIDE, count)
    }

    /// Runs the block-local first pass for this configuration.
    #[must_use]
    pub fn scan(self, flags: &[u32]) -> ScanBlocks {
        exclusive_scan_blocks(flags, self.block_size)
    }

    /// Computes the global `scatter` destination of every element for this
    /// configuration.
    #[must_use]
    pub fn scatter(self, flags: &[u32]) -> Vec<u32> {
        scatter_offsets(flags, self.block_size)
    }

    /// Runs the full two-pass compaction, returning the dense list of surviving
    /// original indices for this configuration.
    #[must_use]
    pub fn compact(self, flags: &[u32]) -> Vec<u32> {
        compact_indices(flags, self.block_size)
    }
}

/// Normalizes one predicate flag to a single keep bit (`0` or `1`).
///
/// The `GPU` mask is conceptually `keep = 1` / `drop = 0`, but any non-zero
/// value is treated as "keep" so an upstream pass may write a raw count or a
/// packed flag without changing the compaction result.
fn keep_bit(flag: u32) -> u32 {
    u32::from(flag != 0)
}

/// Widens an input index into the `u32` a `GPU` index buffer stores, saturating
/// at [`u32::MAX`] rather than panicking on an astronomically long input.
fn index_to_u32(index: usize) -> u32 {
    u32::try_from(index).unwrap_or(u32::MAX)
}

/// Hand-rolled *exclusive* prefix sum of a keep-bit slice.
///
/// This is the module's own serial reference `scan` (a single running-accumulator
/// sweep), deliberately not delegated to any neighbouring `scan` module. It
/// returns the exclusive prefix at every position plus the grand total (the
/// exclusive offset just past the final element). Accumulation wraps so the
/// result matches a wrapping serial reference bit for bit.
fn exclusive_prefix_sum(keeps: &[u32]) -> (Vec<u32>, u32) {
    let mut scanned = Vec::with_capacity(keeps.len());
    let mut running: u32 = 0;
    for &bit in keeps {
        scanned.push(running);
        running = running.wrapping_add(bit);
    }
    (scanned, running)
}

/// First pass: an *exclusive* prefix sum of the predicate keep bits restricted
/// to each `block_size`-wide block, plus the per-block survivor totals.
///
/// This mirrors the on-device step where each `workgroup` scans its own block in
/// shared memory: [`ScanBlocks::block_local`] holds each element's slot relative
/// to its block start, and [`ScanBlocks::block_sums`] holds how many survivors
/// each block contributes (the input to the block-offset `scan`).
#[must_use]
pub fn exclusive_scan_blocks(flags: &[u32], block_size: usize) -> ScanBlocks {
    let block_size = block_size.max(1);
    let num_blocks = flags.len().div_ceil(block_size);
    let mut block_local = Vec::with_capacity(flags.len());
    let mut block_sums = Vec::with_capacity(num_blocks);
    for block in flags.chunks(block_size) {
        let keeps: Vec<u32> = block.iter().map(|&flag| keep_bit(flag)).collect();
        let (scanned, total) = exclusive_prefix_sum(&keeps);
        block_local.extend(scanned);
        block_sums.push(total);
    }
    ScanBlocks {
        block_local,
        block_sums,
    }
}

/// Global `scatter` destination of every element: block base offset plus the
/// element's block-local offset.
///
/// The second pass exclusive-scans the per-block survivor totals into per-block
/// base offsets, and the third pass adds each block's base back to its
/// block-local offsets. For a kept element the result is its final compact slot;
/// for a dropped element it is the slot the next survivor would take (the count
/// of survivors strictly before it), exactly as the on-device global scan
/// produces. Survivor destinations are therefore strictly increasing and dense
/// over `0..compacted_count`.
#[must_use]
pub fn scatter_offsets(flags: &[u32], block_size: usize) -> Vec<u32> {
    let block_size = block_size.max(1);
    let blocks = exclusive_scan_blocks(flags, block_size);
    let (block_bases, _total) = exclusive_prefix_sum(&blocks.block_sums);
    let mut destinations = Vec::with_capacity(blocks.block_local.len());
    for (index, &local) in blocks.block_local.iter().enumerate() {
        let block = index / block_size;
        destinations.push(block_bases[block].wrapping_add(local));
    }
    destinations
}

/// Number of survivors the predicate keeps (the population count of its keep
/// bits), which is exactly the length of [`compact_indices`]'s output.
#[must_use]
pub fn compacted_count(flags: &[u32]) -> usize {
    flags.iter().filter(|&&flag| flag != 0).count()
}

/// Full two-pass stream compaction: the dense list of *original indices* of the
/// kept elements, in ascending input order.
///
/// Each survivor is scattered to its [`scatter_offsets`] destination, so the
/// output has no holes: `output[compact_slot] = original_index`. Because the
/// destinations are strictly increasing the `scatter` is stable and the result
/// preserves input order without any sort.
#[must_use]
pub fn compact_indices(flags: &[u32], block_size: usize) -> Vec<u32> {
    let destinations = scatter_offsets(flags, block_size);
    let count = compacted_count(flags);
    let mut compacted = vec![0u32; count];
    for (index, &flag) in flags.iter().enumerate() {
        if flag != 0 {
            let slot = usize::try_from(destinations[index]).unwrap_or(0);
            compacted[slot] = index_to_u32(index);
        }
    }
    compacted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keep_bit_normalizes_any_nonzero_to_one() {
        assert_eq!(keep_bit(0), 0);
        assert_eq!(keep_bit(1), 1);
        assert_eq!(keep_bit(7), 1);
        assert_eq!(keep_bit(u32::MAX), 1);
    }

    #[test]
    fn all_kept_returns_identity_indices() {
        let flags = [1u32, 1, 1, 1, 1];
        let compacted = compact_indices(&flags, 2);
        assert_eq!(compacted, [0, 1, 2, 3, 4]);
        assert_eq!(compacted_count(&flags), 5);
    }

    #[test]
    fn all_dropped_returns_empty() {
        let flags = [0u32, 0, 0, 0];
        assert_eq!(compacted_count(&flags), 0);
        assert!(compact_indices(&flags, 2).is_empty());
        assert!(scatter_offsets(&flags, 2).iter().all(|&d| d == 0));
    }

    #[test]
    fn alternating_predicate_packs_survivors() {
        let flags = [1u32, 0, 1, 0, 1, 0, 1];
        assert_eq!(compact_indices(&flags, 3), [0, 2, 4, 6]);
        assert_eq!(compacted_count(&flags), 4);
    }

    #[test]
    fn nonzero_flag_counts_as_keep() {
        let flags = [5u32, 0, 9, 0, 42];
        assert_eq!(compact_indices(&flags, 4), [0, 2, 4]);
        assert_eq!(compacted_count(&flags), 3);
    }

    #[test]
    fn empty_input_is_degenerate_but_valid() {
        let flags: [u32; 0] = [];
        assert_eq!(compacted_count(&flags), 0);
        assert!(compact_indices(&flags, 8).is_empty());
        assert!(scatter_offsets(&flags, 8).is_empty());
        let blocks = exclusive_scan_blocks(&flags, 8);
        assert!(blocks.block_local.is_empty());
        assert!(blocks.block_sums.is_empty());
    }

    #[test]
    fn block_local_offsets_reset_each_block() {
        // block_size 3 over 7 elements: blocks [1,1,0] [1,0,1] [1].
        let flags = [1u32, 1, 0, 1, 0, 1, 1];
        let blocks = exclusive_scan_blocks(&flags, 3);
        // Exclusive within-block prefix of keep bits.
        assert_eq!(blocks.block_local, [0, 1, 2, 0, 1, 1, 0]);
    }

    #[test]
    fn block_sums_equal_per_block_survivor_counts() {
        let flags = [1u32, 1, 0, 1, 0, 1, 1];
        let blocks = exclusive_scan_blocks(&flags, 3);
        // block 0 has 2 keeps, block 1 has 2 keeps, block 2 has 1 keep.
        assert_eq!(blocks.block_sums, [2, 2, 1]);
        let total: u32 = blocks.block_sums.iter().sum();
        assert_eq!(usize::try_from(total).unwrap(), compacted_count(&flags));
    }

    #[test]
    fn non_divisible_length_covers_ragged_last_block() {
        // 10 elements, block_size 4 -> 3 blocks, last block has 2 elements.
        let flags = [1u32, 0, 1, 1, 0, 0, 1, 0, 1, 1];
        let cfg = CompactConfig::new(4);
        assert_eq!(cfg.num_blocks(flags.len()), 3);
        assert_eq!(cfg.compact(&flags), [0, 2, 3, 6, 8, 9]);
        assert_eq!(cfg.scan(&flags).block_sums, [3, 1, 2]);
    }

    #[test]
    fn scatter_destinations_are_unique_and_ordered_for_survivors() {
        let flags = [0u32, 1, 1, 0, 1, 1, 0, 1];
        let destinations = scatter_offsets(&flags, 3);
        let mut survivor_slots: Vec<u32> = flags
            .iter()
            .zip(destinations.iter())
            .filter_map(|(&flag, &dst)| if flag != 0 { Some(dst) } else { None })
            .collect();
        let ordered = survivor_slots.clone();
        survivor_slots.sort_unstable();
        survivor_slots.dedup();
        // Strictly increasing (unique) and identical to the natural order.
        assert_eq!(survivor_slots, ordered);
        // Dense over 0..count.
        let count = compacted_count(&flags);
        let expected: Vec<u32> = (0..index_to_u32(count)).collect();
        assert_eq!(ordered, expected);
    }

    #[test]
    fn scatter_offset_of_dropped_element_is_next_survivor_slot() {
        let flags = [1u32, 0, 0, 1, 1];
        let destinations = scatter_offsets(&flags, 2);
        // index 1 and 2 are dropped; both point at slot 1 (survivors before
        // them: just index 0).
        assert_eq!(destinations, [0, 1, 1, 1, 2]);
    }

    #[test]
    fn compacted_output_places_index_at_its_scatter_slot() {
        let flags = [0u32, 1, 0, 1, 1, 0, 1];
        let destinations = scatter_offsets(&flags, 4);
        let compacted = compact_indices(&flags, 4);
        for (index, &flag) in flags.iter().enumerate() {
            if flag != 0 {
                let slot = usize::try_from(destinations[index]).unwrap();
                assert_eq!(compacted[slot], index_to_u32(index));
            }
        }
    }

    #[test]
    fn config_clamps_zero_block_size() {
        let cfg = CompactConfig::new(0);
        assert_eq!(cfg.block_size, 1);
        assert_eq!(cfg.num_blocks(5), 5);
        assert!(cfg.is_power_of_two_block());
    }

    #[test]
    fn config_power_of_two_detection() {
        assert!(CompactConfig::new(64).is_power_of_two_block());
        assert!(!CompactConfig::new(48).is_power_of_two_block());
    }

    #[test]
    fn std430_flag_and_scanned_sizes_match_element_stride() {
        let cfg = CompactConfig::new(256);
        assert_eq!(cfg.flag_bytes(10), U32_STRIDE * 10);
        assert_eq!(cfg.scanned_bytes(10), U32_STRIDE * 10);
        assert_eq!(cfg.compacted_bytes(7), U32_STRIDE * 7);
    }

    #[test]
    fn std430_block_sums_size_counts_blocks() {
        let cfg = CompactConfig::new(64);
        // 200 elements -> 4 blocks of block_sums.
        assert_eq!(cfg.num_blocks(200), 4);
        assert_eq!(cfg.block_sums_bytes(200), U32_STRIDE * 4);
    }

    #[test]
    fn std430_empty_buffers_reserve_one_element() {
        let cfg = CompactConfig::new(64);
        // A zero-sized storage binding is illegal, so each clamps to one stride.
        assert_eq!(cfg.flag_bytes(0), U32_STRIDE);
        assert_eq!(cfg.scanned_bytes(0), U32_STRIDE);
        assert_eq!(cfg.block_sums_bytes(0), U32_STRIDE);
        assert_eq!(cfg.compacted_bytes(0), U32_STRIDE);
    }

    #[test]
    fn config_scatter_matches_free_function() {
        let flags = [1u32, 0, 1, 1, 0, 1];
        let cfg = CompactConfig::new(2);
        assert_eq!(cfg.scatter(&flags), scatter_offsets(&flags, 2));
        assert_eq!(cfg.compact(&flags), compact_indices(&flags, 2));
    }

    #[test]
    fn single_block_covers_whole_input_when_block_exceeds_len() {
        let flags = [0u32, 1, 1, 0, 1];
        let cfg = CompactConfig::new(1024);
        assert_eq!(cfg.num_blocks(flags.len()), 1);
        let blocks = cfg.scan(&flags);
        assert_eq!(blocks.block_sums, [3]);
        assert_eq!(cfg.compact(&flags), [1, 2, 4]);
    }
}
