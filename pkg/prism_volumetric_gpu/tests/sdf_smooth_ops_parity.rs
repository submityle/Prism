//! Real-device parity for the stateless `SDF` combinator twin:
//! [`GpuSdfSmoothOps`](prism_volumetric_gpu::sdf_smooth_ops::GpuSdfSmoothOps)
//! must reproduce the six scalar combinators of the `CPU` golden
//! [`capsule_sdf`](prism_render_architecture::particle::capsule_sdf) — the hard
//! union/subtraction/intersection and their polynomial-smooth variants — across
//! hand-picked fixtures (hard fallback, normal blend, equal distances, large
//! `k`, both `h` clamp edges, negative distances, near-threshold continuity, a
//! mixed batch) and a randomized sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The six golden combinators are public pure functions, so the expected values
//! come straight from calling
//! [`op_union`](prism_render_architecture::particle::capsule_sdf::op_union),
//! [`op_subtract`](prism_render_architecture::particle::capsule_sdf::op_subtract),
//! [`op_intersect`](prism_render_architecture::particle::capsule_sdf::op_intersect),
//! [`op_smooth_union`](prism_render_architecture::particle::capsule_sdf::op_smooth_union),
//! [`op_smooth_subtract`](prism_render_architecture::particle::capsule_sdf::op_smooth_subtract)
//! and
//! [`op_smooth_intersect`](prism_render_architecture::particle::capsule_sdf::op_smooth_intersect)
//! directly, so a `GPU == oracle` pass is a `GPU == golden` pass with no
//! reconstruction gap.
//!
//! # Parity criterion
//!
//! Every output is a continuous distance threaded through a subtract, a divide
//! by `k`, a `clamp` and a `mix`, so a `GPU` fused multiply-add or divide may
//! land a few units in the last place from the scalar reference; each of the six
//! outputs is asserted within `abs_diff <= 1e-5` or `rel_diff <= 1e-4`.
//!
//! # Conditioning
//!
//! The `k <= CMP_EPS` fallback is a magnitude comparison, so every random
//! fixture keeps `k` well clear of `CMP_EPS = 1.0e-6` (random `k` is drawn in
//! `[0.1, 10.0]`) and the hard-fallback branch is exercised separately with
//! `k = 0.0` and `k = CMP_EPS * 0.5`, so `CPU` and `GPU` always take the same
//! branch.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::capsule_sdf`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::capsule_sdf::{
    op_intersect, op_smooth_intersect, op_smooth_subtract, op_smooth_union, op_subtract, op_union,
};
use prism_volumetric_gpu::sdf_smooth_ops::{
    GpuSdfSmoothOps, SdfSmoothOpsQuery, SdfSmoothOpsResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a folded distance. A `GPU` divide or fused
/// multiply-add may land a few units in the last place from the scalar
/// reference; `1e-5` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-5;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-4;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Evaluates the six golden combinators directly: the faithful oracle the `GPU`
/// is pinned against.
fn oracle(d1: f32, d2: f32, k: f32) -> SdfSmoothOpsResult {
    SdfSmoothOpsResult {
        hard_union: op_union(d1, d2),
        hard_subtract: op_subtract(d1, d2),
        hard_intersect: op_intersect(d1, d2),
        smooth_union: op_smooth_union(d1, d2, k),
        smooth_subtract: op_smooth_subtract(d1, d2, k),
        smooth_intersect: op_smooth_intersect(d1, d2, k),
    }
}

/// Pins one `GPU` combinator result against the oracle: all six outputs within
/// tolerance.
fn check_one(idx: usize, got: &SdfSmoothOpsResult, want: &SdfSmoothOpsResult) {
    assert!(
        close(got.hard_union, want.hard_union),
        "query {idx} hard_union: gpu {} vs cpu {}",
        got.hard_union,
        want.hard_union
    );
    assert!(
        close(got.hard_subtract, want.hard_subtract),
        "query {idx} hard_subtract: gpu {} vs cpu {}",
        got.hard_subtract,
        want.hard_subtract
    );
    assert!(
        close(got.hard_intersect, want.hard_intersect),
        "query {idx} hard_intersect: gpu {} vs cpu {}",
        got.hard_intersect,
        want.hard_intersect
    );
    assert!(
        close(got.smooth_union, want.smooth_union),
        "query {idx} smooth_union: gpu {} vs cpu {}",
        got.smooth_union,
        want.smooth_union
    );
    assert!(
        close(got.smooth_subtract, want.smooth_subtract),
        "query {idx} smooth_subtract: gpu {} vs cpu {}",
        got.smooth_subtract,
        want.smooth_subtract
    );
    assert!(
        close(got.smooth_intersect, want.smooth_intersect),
        "query {idx} smooth_intersect: gpu {} vs cpu {}",
        got.smooth_intersect,
        want.smooth_intersect
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSdfSmoothOps, queries: &[SdfSmoothOpsQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q.d1, q.d2, q.k);
        check_one(idx, result, &want);
    }
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws a signed distance in `[-5.0, 5.0]` at milli resolution from `state`.
fn distance(state: &mut u64) -> f32 {
    -5.0 + (lcg(state) % 10_001) as f32 / 1000.0
}

/// Draws a blend radius in `[0.1, 10.0]` at milli resolution from `state`, kept
/// well clear of `CMP_EPS` so the hard-fallback branch is never ambiguous.
fn blend_radius(state: &mut u64) -> f32 {
    0.1 + (lcg(state) % 9_901) as f32 / 1000.0
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sdf_smooth_ops parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSdfSmoothOps::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn hard_fallback_zero_k_matches_hard_ops() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSmoothOps::new(&ctx);
    // k == 0.0 is at or below CMP_EPS, so the smooth ops must equal the hard ops.
    check(&ctx, &gpu, &[SdfSmoothOpsQuery::new(1.5, -0.75, 0.0)]);
}

#[test]
fn hard_fallback_sub_epsilon_k_matches_hard_ops() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSmoothOps::new(&ctx);
    // k = CMP_EPS * 0.5 is below the fallback threshold.
    check(&ctx, &gpu, &[SdfSmoothOpsQuery::new(2.0, -1.0, 0.5e-6)]);
}

#[test]
fn normal_blend_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSmoothOps::new(&ctx);
    check(&ctx, &gpu, &[SdfSmoothOpsQuery::new(0.8, -0.6, 0.5)]);
}

#[test]
fn equal_distances_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSmoothOps::new(&ctx);
    // d1 == d2 drives h to the interpolant midpoint for the symmetric ops.
    check(&ctx, &gpu, &[SdfSmoothOpsQuery::new(1.0, 1.0, 0.75)]);
}

#[test]
fn large_blend_radius_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSmoothOps::new(&ctx);
    check(&ctx, &gpu, &[SdfSmoothOpsQuery::new(0.3, -0.4, 100.0)]);
}

#[test]
fn h_clamped_to_lower_edge_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSmoothOps::new(&ctx);
    // A strongly negative (d2 - d1) with a small k drives the smooth-union
    // interpolant below 0, so h clamps to 0.
    check(&ctx, &gpu, &[SdfSmoothOpsQuery::new(4.0, -4.0, 0.2)]);
}

#[test]
fn h_clamped_to_upper_edge_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSmoothOps::new(&ctx);
    // A strongly positive (d2 - d1) with a small k drives the smooth-union
    // interpolant above 1, so h clamps to 1.
    check(&ctx, &gpu, &[SdfSmoothOpsQuery::new(-4.0, 4.0, 0.2)]);
}

#[test]
fn negative_distances_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSmoothOps::new(&ctx);
    check(&ctx, &gpu, &[SdfSmoothOpsQuery::new(-2.5, -3.5, 1.25)]);
}

#[test]
fn near_threshold_continuity_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSmoothOps::new(&ctx);
    // k just above CMP_EPS: the smooth ops are continuous with the hard ops.
    check(&ctx, &gpu, &[SdfSmoothOpsQuery::new(1.0, -1.0, 1.0e-3)]);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSmoothOps::new(&ctx);
    let queries = [
        SdfSmoothOpsQuery::new(1.5, -0.75, 0.0),
        SdfSmoothOpsQuery::new(0.8, -0.6, 0.5),
        SdfSmoothOpsQuery::new(1.0, 1.0, 0.75),
        SdfSmoothOpsQuery::new(0.3, -0.4, 100.0),
        SdfSmoothOpsQuery::new(4.0, -4.0, 0.2),
        SdfSmoothOpsQuery::new(-4.0, 4.0, 0.2),
        SdfSmoothOpsQuery::new(-2.5, -3.5, 1.25),
        SdfSmoothOpsQuery::new(1.0, -1.0, 1.0e-3),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn randomized_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSmoothOps::new(&ctx);
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let queries: Vec<SdfSmoothOpsQuery> = (0..512)
        .map(|_| {
            let d1 = distance(&mut state);
            let d2 = distance(&mut state);
            let k = blend_radius(&mut state);
            SdfSmoothOpsQuery::new(d1, d2, k)
        })
        .collect();
    check(&ctx, &gpu, &queries);
}
