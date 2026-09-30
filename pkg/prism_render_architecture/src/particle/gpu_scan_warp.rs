//! `Warp`/`subgroup`-level prefix `scan` primitives: the `CPU`-verifiable gold
//! standard for the particle subsystem's lowest-level `GPU` `scan` building
//! block (design §5.2 compaction, §11 counters, §12 sort offsets).
//!
//! A *warp* (also called a *subgroup* or *wave*) is the fixed bundle of `SIMD`
//! lanes a `GPU` executes in lockstep — typically 32 lanes on `NVIDIA`, 32 or 64
//! on `AMD`. This module models the intra-warp `scan` operations that a device
//! implements with lane-to-lane *shuffle* instructions, entirely on the `CPU`,
//! so the eventual device kernel can be validated bit for bit:
//!
//! 1. **`Hillis-Steele` inclusive/exclusive `scan`** — the log-step
//!    shuffle-up sweep every lane runs against its neighbours, `offset` doubling
//!    from `1` until it covers the warp width. It works for any lane count, not
//!    just powers of two, because the sweep simply stops once `offset` reaches
//!    the width.
//! 2. **`Warp` reduction** — the sum of all active lanes, the terminal value of
//!    the inclusive sweep, matching a `subgroupAdd`.
//! 3. **`Warp`-aggregated allocation** — the atomic-append pattern where each
//!    warp exclusive-scans its per-lane counts, reduces them to one warp-local
//!    sum, and a single cross-warp `scan` turns those sums into a global base
//!    offset per warp. Each lane's global slot is then its warp base plus its
//!    within-warp exclusive prefix, which reproduces the plain global exclusive
//!    `scan` while modelling the *one atomic per warp* cost of a real device.
//!
//! This module is deliberately distinct from its siblings and never reaches into
//! their territory. Unlike [`super::gpu_prefix_scan`] it performs no `block`- or
//! `grid`-level multi-pass decomposition — its horizon is a single warp plus the
//! one cross-warp `scan` a `warp`-aggregate needs. Unlike
//! [`super::gpu_scan_segmented`] it carries no head-flag array and never resets
//! mid-sweep. It touches no sorting network or histogram. Every public name
//! carries a `warp` flavour and pairs with [`WarpConfig`] so the contracts never
//! collide.
//!
//! Everything is pure integer arithmetic with `wrapping_add` accumulation, so it
//! matches a wrapping serial reference bit for bit, never panics on empty or
//! ragged input, and never divides by zero. There is no floating-point value
//! anywhere in this module, hence no floating-point comparison.

use alloc::vec;
use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE};

/// Result of a [`warp`-aggregated allocation](WarpConfig::aggregate_alloc).
///
/// Mirrors the storage a `warp`-aggregate append binds on the device: the
/// per-lane destination slot, the per-warp local sum a single atomic would add,
/// the per-warp global base the cross-warp `scan` produced, and the grand total
/// the append reserved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WarpAppend {
    /// Per-lane global base slot, in input order: the start of the run this lane
    /// owns in the compacted output. Equal to the plain global exclusive prefix
    /// sum of the input counts.
    pub lane_slots: Vec<u32>,
    /// Per-warp local sum: the `wrapping_add` reduction of each warp's lane
    /// counts, one entry per warp. This is the value a real kernel would push
    /// through a single atomic add.
    pub warp_local_sums: Vec<u32>,
    /// Per-warp global base offset: the exclusive `scan` of `warp_local_sums`,
    /// one entry per warp. The cross-warp `scan` a single warp performs over the
    /// warp sums.
    pub warp_base_offsets: Vec<u32>,
    /// Grand total reserved by the append: the `wrapping_add` sum of every
    /// warp-local sum.
    pub total: u32,
}

/// Configuration for the `warp`/`subgroup` `scan` primitives: the lane count of
/// one warp.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WarpConfig {
    /// Lanes per warp. Any positive value is valid; the reference sweeps the
    /// lanes directly, so a non-power-of-two width needs no padding.
    pub lane_count: usize,
}

impl Default for WarpConfig {
    /// The canonical 32-lane warp width shared by most `GPU` vendors.
    fn default() -> Self {
        Self::warp32()
    }
}

impl WarpConfig {
    /// The canonical warp width used by most current `GPU` architectures.
    pub const DEFAULT_LANE_COUNT: usize = 32;

    /// Builds a config with the given lane count, clamped to at least one so a
    /// degenerate zero can never cause a divide-by-zero.
    #[must_use]
    pub fn new(lane_count: usize) -> Self {
        Self {
            lane_count: lane_count.max(1),
        }
    }

    /// Builds the canonical 32-lane warp config.
    #[must_use]
    pub fn warp32() -> Self {
        Self {
            lane_count: Self::DEFAULT_LANE_COUNT,
        }
    }

    /// The effective lane count, always at least one.
    #[must_use]
    pub fn lane_count(self) -> usize {
        self.lane_count.max(1)
    }

    /// Whether the configured width is a power of two, the natural shuffle width
    /// for a `subgroup` `scan`.
    #[must_use]
    pub fn is_power_of_two_width(self) -> bool {
        self.lane_count().is_power_of_two()
    }

    /// Number of warps needed to cover `len` lanes at this width.
    #[must_use]
    pub fn num_warps(self, len: usize) -> usize {
        len.div_ceil(self.lane_count())
    }

    /// Byte size of the `std430` storage buffer holding `len` scanned `u32`
    /// values, via the shared [`gpu_layout`](crate::particle::gpu_layout) stride
    /// rule.
    #[must_use]
    pub fn lane_bytes(self, len: usize) -> usize {
        storage_bytes(U32_STRIDE, len)
    }

    /// Byte size of the `std430` storage buffer holding the per-warp local sums
    /// (`u32` each) for `len` input lanes.
    #[must_use]
    pub fn warp_sum_bytes(self, len: usize) -> usize {
        storage_bytes(U32_STRIDE, self.num_warps(len))
    }

    /// Runs the `Hillis-Steele` *inclusive* warp `scan` over `lanes`.
    #[must_use]
    pub fn inclusive(self, lanes: &[u32]) -> Vec<u32> {
        inclusive_scan_warp(lanes)
    }

    /// Runs the `Hillis-Steele` *exclusive* warp `scan` over `lanes`.
    #[must_use]
    pub fn exclusive(self, lanes: &[u32]) -> Vec<u32> {
        exclusive_scan_warp(lanes)
    }

    /// Reduces `lanes` to their `wrapping_add` sum, matching a `subgroupAdd`.
    #[must_use]
    pub fn reduce(self, lanes: &[u32]) -> u32 {
        warp_reduce(lanes)
    }

    /// Runs the `warp`-aggregated allocation over `counts`, partitioning them
    /// into warps of this config's width.
    #[must_use]
    pub fn aggregate_alloc(self, counts: &[u32]) -> WarpAppend {
        warp_aggregate_alloc(counts, self.lane_count())
    }
}

/// `Hillis-Steele` inclusive `scan` of one warp's `lanes`.
///
/// Models the shuffle-up sweep: starting at `offset` `1` and doubling until it
/// reaches the lane count, every lane at or beyond `offset` adds the value from
/// the lane `offset` positions earlier. The result at each lane is the
/// `wrapping_add` sum of every earlier lane plus itself, so the final lane holds
/// the whole-warp reduction. Works for any width, powers of two or not.
#[must_use]
pub fn inclusive_scan_warp(lanes: &[u32]) -> Vec<u32> {
    hillis_steele_inclusive(lanes)
}

/// `Hillis-Steele` exclusive `scan` of one warp's `lanes`.
///
/// The exclusive prefix at lane `i` is the `wrapping_add` sum of the strictly
/// earlier lanes; lane `0` scans to the neutral element `0`. Derived from the
/// inclusive sweep by shifting it one lane, so it equals the inclusive prefix of
/// the previous lane at every position.
#[must_use]
pub fn exclusive_scan_warp(lanes: &[u32]) -> Vec<u32> {
    let inclusive = hillis_steele_inclusive(lanes);
    let len = inclusive.len();
    let mut exclusive = vec![0u32; len];
    if len > 1 {
        exclusive[1..len].copy_from_slice(&inclusive[..len - 1]);
    }
    exclusive
}

/// `Warp` reduction: the `wrapping_add` sum of every lane, matching a
/// `subgroupAdd`. Returns `0` for an empty warp.
#[must_use]
pub fn warp_reduce(lanes: &[u32]) -> u32 {
    lanes.iter().fold(0u32, |acc, &lane| acc.wrapping_add(lane))
}

/// Naive serial inclusive prefix sum, the independent reference the warp sweep
/// is checked against.
#[must_use]
pub fn naive_inclusive(lanes: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(lanes.len());
    let mut running = 0u32;
    for &lane in lanes {
        running = running.wrapping_add(lane);
        out.push(running);
    }
    out
}

/// Naive serial exclusive prefix sum, the independent reference the warp sweep
/// and the `warp`-aggregate global slots are checked against.
#[must_use]
pub fn naive_exclusive(lanes: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(lanes.len());
    let mut running = 0u32;
    for &lane in lanes {
        out.push(running);
        running = running.wrapping_add(lane);
    }
    out
}

/// `Warp`-aggregated allocation over `counts`, partitioned into warps of
/// `lane_count` lanes.
///
/// Each warp exclusive-scans its own lane counts and reduces them to one
/// warp-local sum; a single cross-warp exclusive `scan` (the pattern a lone warp
/// runs over the warp sums) turns those into a per-warp global base; each lane's
/// global slot is its warp base plus its within-warp exclusive prefix. The
/// resulting `lane_slots` reproduce the plain global exclusive prefix sum, while
/// the structure records the *one atomic per warp* the device path would issue.
///
/// This gold standard assumes the number of warps fits one cross-warp `scan`,
/// which matches a `workgroup` of at most `lane_count` warps; the arithmetic
/// stays correct for any warp count regardless.
#[must_use]
pub fn warp_aggregate_alloc(counts: &[u32], lane_count: usize) -> WarpAppend {
    let width = lane_count.max(1);

    let mut warp_local_sums = Vec::with_capacity(counts.len().div_ceil(width));
    for warp in counts.chunks(width) {
        warp_local_sums.push(warp_reduce(warp));
    }

    let warp_base_offsets = exclusive_scan_warp(&warp_local_sums);

    let mut lane_slots = Vec::with_capacity(counts.len());
    for (warp_index, warp) in counts.chunks(width).enumerate() {
        let base = warp_base_offsets.get(warp_index).copied().unwrap_or(0);
        for within in exclusive_scan_warp(warp) {
            lane_slots.push(base.wrapping_add(within));
        }
    }

    let total = warp_reduce(&warp_local_sums);

    WarpAppend {
        lane_slots,
        warp_local_sums,
        warp_base_offsets,
        total,
    }
}

/// Packs `values` into a little-endian `std430` `u32` byte buffer, the exact
/// layout a `GPU` storage binding reads. Each value contributes
/// [`U32_STRIDE`](crate::particle::gpu_layout::U32_STRIDE) bytes.
#[must_use]
pub fn pack_std430(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * U32_STRIDE);
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// The shared `Hillis-Steele` inclusive sweep behind the public entry points.
fn hillis_steele_inclusive(lanes: &[u32]) -> Vec<u32> {
    let mut current = lanes.to_vec();
    let width = current.len();
    let mut offset = 1usize;
    while offset < width {
        let previous = current.clone();
        for i in offset..width {
            current[i] = previous[i].wrapping_add(previous[i - offset]);
        }
        offset <<= 1;
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic pseudo-random lane generator (integer `LCG`), so the
    /// tests exercise real spreads without any floating point.
    fn lcg_vec(seed: u64, len: usize, modulo: u32) -> Vec<u32> {
        let mut state = seed | 1;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let top = u32::try_from(state >> 33).unwrap_or(0);
            out.push(top % modulo.max(1));
        }
        out
    }

    #[test]
    fn inclusive_matches_naive_reference() {
        let lanes = lcg_vec(0x1111_2222_3333_4444, 32, 500);
        assert_eq!(inclusive_scan_warp(&lanes), naive_inclusive(&lanes));
    }

    #[test]
    fn exclusive_matches_naive_reference() {
        let lanes = lcg_vec(0x5555_6666_7777_8888, 32, 500);
        assert_eq!(exclusive_scan_warp(&lanes), naive_exclusive(&lanes));
    }

    #[test]
    fn exclusive_is_inclusive_shifted_by_one_lane() {
        let lanes = lcg_vec(0x9999_aaaa_bbbb_cccc, 24, 300);
        let inclusive = inclusive_scan_warp(&lanes);
        let exclusive = exclusive_scan_warp(&lanes);
        assert_eq!(exclusive[0], 0);
        for i in 1..lanes.len() {
            assert_eq!(exclusive[i], inclusive[i - 1]);
        }
    }

    #[test]
    fn exclusive_plus_value_equals_inclusive() {
        let lanes = lcg_vec(0x0f0f_0f0f_1234_5678, 32, 1000);
        let inclusive = inclusive_scan_warp(&lanes);
        let exclusive = exclusive_scan_warp(&lanes);
        for i in 0..lanes.len() {
            assert_eq!(exclusive[i].wrapping_add(lanes[i]), inclusive[i]);
        }
    }

    #[test]
    fn non_power_of_two_lane_counts_are_correct() {
        for &width in &[3usize, 5, 6, 7, 11, 17, 31] {
            let lanes = lcg_vec(0xdead_beef_0000_0000 ^ (width as u64), width, 200);
            assert_eq!(
                inclusive_scan_warp(&lanes),
                naive_inclusive(&lanes),
                "inclusive width={width}"
            );
            assert_eq!(
                exclusive_scan_warp(&lanes),
                naive_exclusive(&lanes),
                "exclusive width={width}"
            );
        }
    }

    #[test]
    fn reduce_equals_last_inclusive_lane() {
        let lanes = lcg_vec(0x2468_ace0_1357_9bdf, 32, 777);
        let inclusive = inclusive_scan_warp(&lanes);
        assert_eq!(warp_reduce(&lanes), *inclusive.last().unwrap());
    }

    #[test]
    fn reduce_equals_naive_sum() {
        let lanes = lcg_vec(0x1357_9bdf_2468_ace0, 19, 640);
        let expected = lanes.iter().fold(0u32, |a, &b| a.wrapping_add(b));
        assert_eq!(warp_reduce(&lanes), expected);
    }

    #[test]
    fn reduce_empty_is_zero() {
        let lanes: [u32; 0] = [];
        assert_eq!(warp_reduce(&lanes), 0);
    }

    #[test]
    fn reduce_single_lane_is_the_lane() {
        assert_eq!(warp_reduce(&[42u32]), 42);
    }

    #[test]
    fn empty_scans_are_empty() {
        let lanes: [u32; 0] = [];
        assert!(inclusive_scan_warp(&lanes).is_empty());
        assert!(exclusive_scan_warp(&lanes).is_empty());
    }

    #[test]
    fn single_lane_scans() {
        assert_eq!(inclusive_scan_warp(&[7u32]), vec![7]);
        assert_eq!(exclusive_scan_warp(&[7u32]), vec![0]);
    }

    #[test]
    fn wrapping_overflow_is_bit_exact() {
        let lanes = [u32::MAX, 1, u32::MAX, 2];
        assert_eq!(inclusive_scan_warp(&lanes), naive_inclusive(&lanes));
        assert_eq!(exclusive_scan_warp(&lanes), naive_exclusive(&lanes));
    }

    #[test]
    fn aggregate_lane_slots_equal_global_exclusive_scan() {
        let counts = lcg_vec(0xa1b2_c3d4_e5f6_0718, 200, 8);
        let append = warp_aggregate_alloc(&counts, 32);
        assert_eq!(append.lane_slots, naive_exclusive(&counts));
    }

    #[test]
    fn aggregate_total_equals_global_sum() {
        let counts = lcg_vec(0x0718_e5f6_c3d4_a1b2, 137, 5);
        let append = warp_aggregate_alloc(&counts, 32);
        let expected = counts.iter().fold(0u32, |a, &b| a.wrapping_add(b));
        assert_eq!(append.total, expected);
    }

    #[test]
    fn aggregate_bases_are_exclusive_scan_of_local_sums() {
        let counts = lcg_vec(0x3141_5926_5358_9793, 100, 6);
        let append = warp_aggregate_alloc(&counts, 32);
        assert_eq!(
            append.warp_base_offsets,
            naive_exclusive(&append.warp_local_sums)
        );
    }

    #[test]
    fn aggregate_local_sums_match_per_warp_reduce() {
        let counts = lcg_vec(0x2718_2818_2845_9045, 100, 9);
        let width = 32usize;
        let append = warp_aggregate_alloc(&counts, width);
        let expected: Vec<u32> = counts.chunks(width).map(warp_reduce).collect();
        assert_eq!(append.warp_local_sums, expected);
    }

    #[test]
    fn aggregate_handles_partial_last_warp() {
        // 70 lanes over a width of 32 -> three warps, the last only six lanes.
        let counts = lcg_vec(0x1618_0339_8874_9895, 70, 4);
        let append = warp_aggregate_alloc(&counts, 32);
        assert_eq!(append.warp_local_sums.len(), 3);
        assert_eq!(append.lane_slots, naive_exclusive(&counts));
    }

    #[test]
    fn aggregate_single_warp_equals_plain_exclusive() {
        let counts = lcg_vec(0x4142_4344_4546_4748, 20, 10);
        let append = warp_aggregate_alloc(&counts, 32);
        assert_eq!(append.warp_local_sums.len(), 1);
        assert_eq!(append.lane_slots, exclusive_scan_warp(&counts));
    }

    #[test]
    fn aggregate_empty_input_is_empty() {
        let counts: [u32; 0] = [];
        let append = warp_aggregate_alloc(&counts, 32);
        assert!(append.lane_slots.is_empty());
        assert!(append.warp_local_sums.is_empty());
        assert!(append.warp_base_offsets.is_empty());
        assert_eq!(append.total, 0);
    }

    #[test]
    fn config_clamps_zero_width_and_scans() {
        let config = WarpConfig::new(0);
        assert_eq!(config.lane_count(), 1);
        assert!(config.is_power_of_two_width());
        let lanes = [4u32, 8, 15, 16, 23, 42];
        assert_eq!(config.exclusive(&lanes), naive_exclusive(&lanes));
        assert_eq!(config.inclusive(&lanes), naive_inclusive(&lanes));
        assert_eq!(config.reduce(&lanes), warp_reduce(&lanes));
    }

    #[test]
    fn config_defaults_and_constructors_agree() {
        assert_eq!(WarpConfig::warp32().lane_count(), 32);
        assert_eq!(WarpConfig::default().lane_count(), 32);
        assert!(WarpConfig::warp32().is_power_of_two_width());
    }

    #[test]
    fn config_num_warps_is_div_ceil() {
        let config = WarpConfig::new(32);
        assert_eq!(config.num_warps(0), 0);
        assert_eq!(config.num_warps(1), 1);
        assert_eq!(config.num_warps(32), 1);
        assert_eq!(config.num_warps(33), 2);
        assert_eq!(config.num_warps(64), 2);
    }

    #[test]
    fn config_aggregate_alloc_matches_free_function() {
        let counts = lcg_vec(0x7a7a_5b5b_3c3c_1d1d, 80, 7);
        let config = WarpConfig::warp32();
        assert_eq!(
            config.aggregate_alloc(&counts),
            warp_aggregate_alloc(&counts, 32)
        );
    }

    #[test]
    fn std430_sizes_follow_stride_rule() {
        let config = WarpConfig::new(32);
        // Empty buffers still reserve one element per the shared stride rule.
        assert_eq!(config.lane_bytes(0), U32_STRIDE);
        assert_eq!(config.lane_bytes(10), 10 * U32_STRIDE);
        assert_eq!(config.warp_sum_bytes(0), U32_STRIDE);
        assert_eq!(config.warp_sum_bytes(33), 2 * U32_STRIDE);
    }

    #[test]
    fn pack_std430_is_little_endian() {
        let bytes = pack_std430(&[1u32, 0x0403_0201]);
        assert_eq!(bytes.len(), 2 * U32_STRIDE);
        assert_eq!(&bytes[0..4], &[1, 0, 0, 0]);
        assert_eq!(&bytes[4..8], &[0x01, 0x02, 0x03, 0x04]);
    }
}
