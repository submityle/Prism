//! Real-device parity for the hierarchical prefix-scan twin:
//! [`GpuPrefixScan`](prism_volumetric_gpu::gpu_prefix_scan::GpuPrefixScan) must
//! reproduce the `CPU` golden
//! [`gpu_prefix_scan`](prism_render_architecture::particle::gpu_prefix_scan)
//! across an empty input (host short-circuit), a single element, a block-sum
//! fixture with a known per-block tiling, ragged and block-aligned lengths over
//! many block sizes, the inclusive mode, a large pseudo-random array, and a
//! `u32::MAX` fixture that forces wrapping overflow.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every scan output is a `u32`: a scanned offset, a per-block total or the
//! grand total. Parity is therefore asserted with **exact `==`**, not a float
//! tolerance. There is no float math anywhere in the kernel — the scan is a
//! bounded block-local prefix sum plus an integer base add — so the comparison
//! is bit-exact by construction and has no `ULP`-boundary degenerate region to
//! avoid.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_prefix_scan`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::gpu_prefix_scan::{
    exclusive_scan_blocked, inclusive_scan, naive_exclusive,
};
use prism_volumetric_gpu::gpu_prefix_scan::{GpuPrefixScan, GpuPrefixScanQuery};
use prism_volumetric_gpu::GpuContext;

/// A tiny integer linear-congruential generator; only integer arithmetic, so no
/// transcendental appears. The constants are the well-known `PCG`/`MMIX`
/// multiplier and increment.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // The top 32 bits of the 64-bit state make a well-mixed draw.
        (self.state >> 32) as u32
    }

    /// Fills a length-`len` vector of values each reduced modulo `modulo`, so
    /// the running prefix stays well inside `u32` for the small fixtures.
    fn fill(&mut self, len: usize, modulo: u32) -> Vec<u32> {
        (0..len).map(|_| self.next_u32() % modulo).collect()
    }
}

/// Runs the `GPU` exclusive scan and asserts exact parity against the `CPU`
/// golden: every scanned offset, every per-block total and the grand total must
/// match [`exclusive_scan_blocked`], and the scanned offsets must also match the
/// serial [`naive_exclusive`] reference.
fn check_scan(ctx: &GpuContext, gpu: &GpuPrefixScan, values: &[u32], block_size: u32) {
    let query = GpuPrefixScanQuery {
        values: values.to_vec(),
        block_size,
    };
    let got = gpu.scan(ctx, &query);
    let want = exclusive_scan_blocked(values, block_size as usize);

    assert_eq!(
        got.scanned, want.scanned,
        "scanned offsets must match the golden (block_size {block_size})"
    );
    assert_eq!(
        got.block_sums, want.block_sums,
        "per-block totals must match the golden (block_size {block_size})"
    );
    assert_eq!(
        got.total, want.total,
        "grand total must match the golden (block_size {block_size})"
    );
    assert_eq!(
        got.scanned,
        naive_exclusive(values),
        "scanned offsets must match the serial reference (block_size {block_size})"
    );
}

/// Runs the `GPU` inclusive scan and asserts exact parity against the golden
/// [`inclusive_scan`].
fn check_inclusive(ctx: &GpuContext, gpu: &GpuPrefixScan, values: &[u32], block_size: u32) {
    let query = GpuPrefixScanQuery {
        values: values.to_vec(),
        block_size,
    };
    let got = gpu.inclusive(ctx, &query);
    assert_eq!(
        got,
        inclusive_scan(values),
        "inclusive scan must match the golden (block_size {block_size})"
    );
}

#[test]
fn empty_input_is_degenerate_but_valid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrefixScan::new(&ctx);
    // No element issues no dispatch; the result is empty and matches the golden.
    let query = GpuPrefixScanQuery {
        values: Vec::new(),
        block_size: 8,
    };
    let got = gpu.scan(&ctx, &query);
    assert!(got.scanned.is_empty(), "empty input scans to nothing");
    assert!(got.block_sums.is_empty(), "no block means no block total");
    assert_eq!(got.total, 0, "empty input has a zero grand total");
    assert!(
        gpu.inclusive(&ctx, &query).is_empty(),
        "empty input has an empty inclusive scan"
    );
}

#[test]
fn single_element_scans_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrefixScan::new(&ctx);
    check_scan(&ctx, &gpu, &[42u32], 4);
    let query = GpuPrefixScanQuery {
        values: vec![42u32],
        block_size: 4,
    };
    let got = gpu.scan(&ctx, &query);
    assert_eq!(got.scanned, vec![0], "the sole element has a zero offset");
    assert_eq!(got.block_sums, vec![42], "the one block totals its element");
    assert_eq!(got.total, 42, "the grand total is the sole element");
}

#[test]
fn block_sums_are_per_block_totals() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrefixScan::new(&ctx);
    // 7 elements, block size 4 -> block 0 = 1+2+3+4 = 10, block 1 = 5+6+7 = 18.
    let values = [1u32, 2, 3, 4, 5, 6, 7];
    check_scan(&ctx, &gpu, &values, 4);
    let query = GpuPrefixScanQuery {
        values: values.to_vec(),
        block_size: 4,
    };
    let got = gpu.scan(&ctx, &query);
    assert_eq!(got.block_sums, vec![10, 18], "device reduces each block");
    assert_eq!(got.total, 28, "grand total of 1..=7");
}

#[test]
fn ragged_and_aligned_lengths_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrefixScan::new(&ctx);
    for &len in &[2usize, 3, 5, 7, 8, 9, 13, 16, 31, 33, 100, 257] {
        let mut rng = Lcg::new(0xdead_beef_0000_0001 ^ (len as u64));
        let values = rng.fill(len, 500);
        for &bs in &[1u32, 2, 4, 8, 16, 64] {
            check_scan(&ctx, &gpu, &values, bs);
        }
    }
}

#[test]
fn block_size_zero_clamps_to_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrefixScan::new(&ctx);
    // A degenerate zero block size clamps to one on both paths.
    let values = [4u32, 8, 15, 16, 23, 42];
    check_scan(&ctx, &gpu, &values, 0);
}

#[test]
fn inclusive_matches_golden_across_lengths() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrefixScan::new(&ctx);
    for &len in &[1usize, 2, 3, 8, 17, 64, 130] {
        let mut rng = Lcg::new(0xabcd_1234_5678_9999 ^ (len as u64));
        let values = rng.fill(len, 777);
        for &bs in &[1u32, 4, 16, 64] {
            check_inclusive(&ctx, &gpu, &values, bs);
        }
    }
}

#[test]
fn total_equals_wrapping_sum_across_block_sizes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrefixScan::new(&ctx);
    let mut rng = Lcg::new(0x0f0f_0f0f_1111_2222);
    let values = rng.fill(129, 1000);
    let expected = values.iter().fold(0u32, |acc, &x| acc.wrapping_add(x));
    for &bs in &[1u32, 2, 4, 8, 16, 32, 64] {
        let query = GpuPrefixScanQuery {
            values: values.clone(),
            block_size: bs,
        };
        assert_eq!(
            gpu.scan(&ctx, &query).total,
            expected,
            "grand total is block-size-independent (block_size {bs})"
        );
    }
}

#[test]
fn large_array_matches_golden_across_block_sizes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrefixScan::new(&ctx);
    let mut rng = Lcg::new(0x5a5a_5a5a_a5a5_a5a5);
    let values = rng.fill(4096, 4096);
    for &bs in &[1u32, 2, 4, 8, 16, 32, 64, 128, 256, 1024] {
        check_scan(&ctx, &gpu, &values, bs);
    }
}

#[test]
fn wrapping_overflow_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrefixScan::new(&ctx);
    // Values large enough to overflow u32 in aggregate; both paths wrap.
    let values = vec![u32::MAX; 10];
    check_scan(&ctx, &gpu, &values, 4);
    let query = GpuPrefixScanQuery {
        values: values.clone(),
        block_size: 4,
    };
    assert_eq!(
        gpu.scan(&ctx, &query).total,
        u32::MAX.wrapping_mul(10),
        "the grand total wraps exactly like the serial reference"
    );
}
