//! Real-device parity for the `de Bruijn` bit-scan twin:
//! [`GpuDeBruijnLog2`](prism_volumetric_gpu::de_bruijn_log2::GpuDeBruijnLog2)
//! must reproduce the `CPU` golden
//! [`particle::de_bruijn_log2`](prism_render_architecture::particle::de_bruijn_log2)
//! across its five `u32` routines — `floor_log2`, `ceil_log2`,
//! `is_power_of_two`, `next_power_of_two` and `trailing_zero_index` — on known
//! small values, every single bit, dense ranges, byte boundaries and a large
//! random batch spanning many workgroups.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every routine is pure `u32` bit algebra with no rounding anywhere on the
//! path, so the outputs are bit-identical and asserted with exact `==` and no
//! tolerance. Several scenarios additionally assert a non-trivial mix of
//! power-of-two flags and nonzero logarithms, so a degenerate all-zero kernel
//! could not pass.
//!
//! The golden `u64` siblings are intentionally not twinned (`WGSL` has no
//! `u64`), so this suite covers only the `u32` domain.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::de_bruijn_log2`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::de_bruijn_log2::{
    ceil_log2_u32, floor_log2_u32, is_power_of_two_u32, next_power_of_two_u32,
    trailing_zero_index_u32,
};
use prism_volumetric_gpu::de_bruijn_log2::{DeBruijnLog2Result, GpuDeBruijnLog2};
use prism_volumetric_gpu::GpuContext;

/// Largest input for which the golden `next_power_of_two_u32` is defined without
/// overflowing `u32`; `2^31` is itself a power of two, so its next power of two
/// is itself. Fixtures stay at or below this so every routine is in-contract.
const MAX_NPOT_INPUT: u32 = 1u32 << 31;

/// Asserts the device result for `x` matches every golden routine bit for bit.
fn assert_matches_golden(x: u32, r: DeBruijnLog2Result) {
    assert_eq!(
        r.floor_log2,
        floor_log2_u32(x),
        "floor_log2 mismatch at x = {x}"
    );
    assert_eq!(
        r.ceil_log2,
        ceil_log2_u32(x),
        "ceil_log2 mismatch at x = {x}"
    );
    assert_eq!(
        r.is_power_of_two,
        is_power_of_two_u32(x),
        "is_power_of_two mismatch at x = {x}"
    );
    assert_eq!(
        r.next_power_of_two,
        next_power_of_two_u32(x),
        "next_power_of_two mismatch at x = {x}"
    );
    assert_eq!(
        r.trailing_zero_index,
        trailing_zero_index_u32(x),
        "trailing_zero_index mismatch at x = {x}"
    );
}

/// Runs the twin over `xs` and asserts per-element parity against the golden,
/// returning the device results for extra assertions.
fn check(ctx: &GpuContext, gpu: &GpuDeBruijnLog2, xs: &[u32]) -> Vec<DeBruijnLog2Result> {
    let results = gpu.eval(ctx, xs);
    assert_eq!(results.len(), xs.len(), "one result per input");
    for (&x, &r) in xs.iter().zip(results.iter()) {
        assert_matches_golden(x, r);
    }
    results
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns a raw `u64` state word.
fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// Draws a `u32` in `[0, 2^31]` so the golden `next_power_of_two_u32` stays
/// in-contract. The high bits of the `LCG` state are the well-mixed ones.
fn draw_input(state: &mut u64) -> u32 {
    let bits = (lcg(state) >> 32) as u32;
    bits & (MAX_NPOT_INPUT - 1)
}

/// Builds a deterministic batch of in-contract inputs: an exhaustive small
/// range, every single bit and its neighbors, byte boundaries, and the maximum
/// in-contract value.
fn structured_inputs() -> Vec<u32> {
    let mut xs = Vec::new();
    // Exhaustive small range, including 0 and 1.
    for x in 0u32..=4096 {
        xs.push(x);
    }
    // Every single set bit 2^k, plus its neighbors (kept in-contract).
    for k in 0u32..=31 {
        let p = 1u32 << k;
        xs.push(p);
        if p > 2 {
            xs.push(p - 1);
        }
        // p + 1 must not exceed the next-power-of-two contract.
        if p < MAX_NPOT_INPUT {
            xs.push(p + 1);
        }
    }
    // Byte boundaries.
    for &x in &[255u32, 256, 257, 511, 512, 65_535, 65_536, 65_537] {
        xs.push(x);
    }
    xs.push(MAX_NPOT_INPUT);
    xs
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_structured_inputs() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping de-bruijn-log2 parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuDeBruijnLog2::new(&ctx);
    let xs = structured_inputs();
    let results = check(&ctx, &gpu, &xs);

    // Non-trivial: at least a handful of powers of two and a spread of distinct
    // floor-log2 values, so a degenerate constant kernel could not pass.
    let pow2_count = results.iter().filter(|r| r.is_power_of_two).count();
    assert!(
        pow2_count >= 16,
        "expected many powers of two in the fixture, got {pow2_count}"
    );
    let max_floor = results.iter().map(|r| r.floor_log2).max().unwrap_or(0);
    assert_eq!(
        max_floor, 31,
        "the 2^31 fixture should drive floor_log2 to 31"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_every_single_bit() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping de-bruijn-log2 parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuDeBruijnLog2::new(&ctx);

    // Each 2^k has floor_log2 == ceil_log2 == trailing_zero_index == k and is a
    // power of two; assert that chain explicitly as well as against the golden.
    let xs: Vec<u32> = (0u32..=31).map(|k| 1u32 << k).collect();
    let results = check(&ctx, &gpu, &xs);
    for (k, r) in results.iter().enumerate() {
        let k = k as u32;
        assert_eq!(r.floor_log2, k, "floor_log2 of 2^{k}");
        assert_eq!(r.ceil_log2, k, "ceil_log2 of 2^{k}");
        assert_eq!(r.trailing_zero_index, k, "trailing_zero_index of 2^{k}");
        assert!(r.is_power_of_two, "2^{k} is a power of two");
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_large_random_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping de-bruijn-log2 parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuDeBruijnLog2::new(&ctx);

    // Several thousand in-contract inputs spanning many workgroups.
    let mut state = 0x1234_5678_9ABC_DEF0u64;
    let count = 8192usize;
    let mut xs = Vec::with_capacity(count);
    for _ in 0..count {
        xs.push(draw_input(&mut state));
    }
    let results = check(&ctx, &gpu, &xs);

    // The random batch is dominated by non-powers of two, so a degenerate
    // all-true power-of-two kernel could not pass.
    let any_non_pow2 = results.iter().any(|r| !r.is_power_of_two);
    assert!(
        any_non_pow2,
        "random batch should contain non-powers of two"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_degenerate_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping de-bruijn-log2 parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuDeBruijnLog2::new(&ctx);

    // x == 0 exercises every saturating/convention branch at once.
    let results = check(&ctx, &gpu, &[0]);
    let r = results[0];
    assert_eq!(r.floor_log2, 0, "floor_log2(0) == 0 by convention");
    assert_eq!(r.ceil_log2, 0, "ceil_log2(0) == 0 by convention");
    assert!(!r.is_power_of_two, "zero is not a power of two");
    assert_eq!(r.next_power_of_two, 1, "next_power_of_two(0) == 1");
    assert_eq!(
        r.trailing_zero_index, 32,
        "trailing_zero_index(0) == 32 by convention"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_yields_empty_output() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping de-bruijn-log2 parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuDeBruijnLog2::new(&ctx);
    let results = gpu.eval(&ctx, &[]);
    assert!(results.is_empty(), "empty input must yield empty output");
}
