//! Real-device parity for the warp-`scan` twin:
//! [`GpuScanWarp`](prism_volumetric_gpu::gpu_scan_warp::GpuScanWarp) must
//! reproduce the `CPU` golden
//! [`gpu_scan_warp`](prism_render_architecture::particle::gpu_scan_warp) across
//! an empty input (host short-circuit), single-lane and single-warp inputs,
//! non-power-of-two warp widths, wrapping overflow, multi-warp inputs that
//! exercise the host coarse `scan`, and a large pseudo-random `warp`-aggregated
//! allocation over several lane counts with a ragged last warp.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL` plus `workgroupBarrier`, so it needs no
//! optional device feature.
//!
//! # Parity criterion
//!
//! Every `scan` output is a `u32`: a prefix sum, a warp reduction or a
//! destination slot. Parity is therefore asserted with **exact `==`**, not a
//! float tolerance. There is no float math anywhere in the kernel — the sweep
//! is a shared-memory `Hillis-Steele` of integer additions — so the comparison
//! is bit-exact by construction and has no `ULP`-boundary degenerate region to
//! avoid.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_scan_warp`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::gpu_scan_warp::{
    exclusive_scan_warp, inclusive_scan_warp, warp_aggregate_alloc, warp_reduce,
};
use prism_volumetric_gpu::gpu_scan_warp::{
    GpuScanWarp, GpuWarpAggregateQuery, GpuWarpConfig, GpuWarpScanQuery,
};
use prism_volumetric_gpu::GpuContext;

/// A tiny integer linear-congruential generator; only integer arithmetic, so no
/// transcendental appears. Returns the full 32-bit high word of the state.
struct Lcg {
    state: u64,
}

impl Lcg {
    /// Seeds the generator; a non-zero seed keeps the stream non-degenerate.
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    /// Advances the state and returns its high 32 bits.
    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 32) as u32
    }

    /// Fills a length-`len` vector of counts in `0..modulo`, well away from the
    /// `u32` wrap boundary so the aggregate stays easy to reason about.
    fn fill_counts(&mut self, len: usize, modulo: u32) -> Vec<u32> {
        (0..len).map(|_| self.next_u32() % modulo.max(1)).collect()
    }
}

/// Runs the `GPU` `scan` and asserts exact parity against the `CPU` golden:
/// inclusive, exclusive and the whole-input reduction.
fn check_scan(ctx: &GpuContext, gpu: &GpuScanWarp, lanes: &[u32]) {
    let query = GpuWarpScanQuery {
        lanes: lanes.to_vec(),
    };
    let result = gpu.scan(ctx, &query);

    assert_eq!(
        result.inclusive,
        inclusive_scan_warp(lanes),
        "inclusive scan must match the golden (len {})",
        lanes.len()
    );
    assert_eq!(
        result.exclusive,
        exclusive_scan_warp(lanes),
        "exclusive scan must match the golden (len {})",
        lanes.len()
    );
    assert_eq!(
        result.total,
        warp_reduce(lanes),
        "warp reduction must match the golden (len {})",
        lanes.len()
    );
}

/// Runs the `GPU` `warp`-aggregated allocation and asserts exact parity against
/// the `CPU` golden across every field.
fn check_aggregate(ctx: &GpuContext, gpu: &GpuScanWarp, counts: &[u32], lane_count: u32) {
    let query = GpuWarpAggregateQuery {
        counts: counts.to_vec(),
        config: GpuWarpConfig::new(lane_count),
    };
    let append = gpu.aggregate_alloc(ctx, &query);
    let want = warp_aggregate_alloc(counts, lane_count as usize);

    assert_eq!(
        append.lane_slots, want.lane_slots,
        "lane slots must match the golden (lane_count {lane_count})"
    );
    assert_eq!(
        append.warp_local_sums, want.warp_local_sums,
        "warp local sums must match the golden (lane_count {lane_count})"
    );
    assert_eq!(
        append.warp_base_offsets, want.warp_base_offsets,
        "warp base offsets must match the golden (lane_count {lane_count})"
    );
    assert_eq!(
        append.total, want.total,
        "aggregate total must match the golden (lane_count {lane_count})"
    );
}

#[test]
fn empty_input_is_degenerate_but_valid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuScanWarp::new(&ctx);
    let query = GpuWarpScanQuery { lanes: Vec::new() };
    let result = gpu.scan(&ctx, &query);
    assert!(result.inclusive.is_empty(), "empty input scans to nothing");
    assert!(result.exclusive.is_empty(), "empty input scans to nothing");
    assert_eq!(result.total, 0, "empty reduction is zero");
}

#[test]
fn single_lane_scans() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuScanWarp::new(&ctx);
    check_scan(&ctx, &gpu, &[7u32]);
    check_scan(&ctx, &gpu, &[0u32]);
}

#[test]
fn single_full_warp_scan() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuScanWarp::new(&ctx);
    let mut lcg = Lcg::new(0x1111_2222_3333_4444);
    let lanes = lcg.fill_counts(32, 500);
    check_scan(&ctx, &gpu, &lanes);
}

#[test]
fn non_power_of_two_lengths_scan() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuScanWarp::new(&ctx);
    let mut lcg = Lcg::new(0x9999_aaaa_bbbb_cccc);
    for len in [3usize, 5, 6, 7, 11, 17, 31, 33, 63] {
        let lanes = lcg.fill_counts(len, 300);
        check_scan(&ctx, &gpu, &lanes);
    }
}

#[test]
fn wrapping_overflow_is_bit_exact() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuScanWarp::new(&ctx);
    // The running sum wraps past u32::MAX; the device `+` wraps identically.
    let lanes = [u32::MAX, 1, u32::MAX, 2, u32::MAX - 3, 10];
    check_scan(&ctx, &gpu, &lanes);
}

#[test]
fn multi_warp_scan_exercises_host_coarse_scan() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuScanWarp::new(&ctx);
    let mut lcg = Lcg::new(0x2468_ace0_1357_9bdf);
    // Lengths beyond one 64-lane workgroup force the host per-warp base scan.
    for len in [64usize, 65, 200, 777, 4096] {
        let lanes = lcg.fill_counts(len, 1000);
        check_scan(&ctx, &gpu, &lanes);
    }
}

#[test]
fn aggregate_single_warp_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuScanWarp::new(&ctx);
    let mut lcg = Lcg::new(0x4142_4344_4546_4748);
    let counts = lcg.fill_counts(20, 10);
    check_aggregate(&ctx, &gpu, &counts, 32);
}

#[test]
fn aggregate_partial_last_warp_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuScanWarp::new(&ctx);
    let mut lcg = Lcg::new(0x1618_0339_8874_9895);
    // 70 lanes over width 32 -> three warps, the last only six lanes.
    let counts = lcg.fill_counts(70, 8);
    check_aggregate(&ctx, &gpu, &counts, 32);
}

#[test]
fn aggregate_varied_lane_counts_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuScanWarp::new(&ctx);
    let mut lcg = Lcg::new(0x7a7a_5b5b_3c3c_1d1d);
    let counts = lcg.fill_counts(200, 8);
    for lane_count in [1u32, 2, 4, 8, 16, 32, 64] {
        check_aggregate(&ctx, &gpu, &counts, lane_count);
    }
}

#[test]
fn aggregate_empty_input_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuScanWarp::new(&ctx);
    let query = GpuWarpAggregateQuery {
        counts: Vec::new(),
        config: GpuWarpConfig::new(32),
    };
    let append = gpu.aggregate_alloc(&ctx, &query);
    assert!(append.lane_slots.is_empty(), "empty aggregate has no slots");
    assert!(
        append.warp_local_sums.is_empty(),
        "empty aggregate has no warp sums"
    );
    assert!(
        append.warp_base_offsets.is_empty(),
        "empty aggregate has no warp bases"
    );
    assert_eq!(append.total, 0, "empty aggregate reserves nothing");
}
