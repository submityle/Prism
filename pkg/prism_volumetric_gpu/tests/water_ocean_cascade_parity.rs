//! Real-device parity for the ocean spectral-cascade fade twin:
//! [`GpuWaterOceanCascade`](prism_volumetric_gpu::water_ocean_cascade::GpuWaterOceanCascade)
//! must reproduce the per-cascade distance weighting of the `CPU` golden
//! [`ocean_lod`](prism_render_architecture::water::ocean_lod) —
//! [`cascade_weights_into`](prism_render_architecture::water::ocean_lod::cascade_weights_into)
//! and its kernel
//! [`cascade_distance_weight`](prism_render_architecture::water::ocean_lod::cascade_distance_weight)
//! — across hand-chosen configs, boundary distances, and a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden
//! [`cascade_weights_into`](prism_render_architecture::water::ocean_lod::cascade_weights_into)
//! is public and pure, so the expected result is built in-host by calling it
//! directly into a stack buffer of the query's `out_len`. A `GPU == oracle`
//! pass is therefore directly a `GPU == golden` pass.
//!
//! # Parity criterion
//!
//! Each cascade weight threads through a subtract, a divide and a `clamp`, so a
//! `GPU` divide may land a few units in the last place from the scalar
//! reference; each is asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//! The returned `count` is pure integer arithmetic and is asserted exactly.
//!
//! # Conditioning
//!
//! The random sweep keeps `fade_range` well above the `EPS` degenerate
//! threshold and rejects any distance within a margin of a cascade's
//! `fade_begin`, so the `distance <= begin` branch never flips between `CPU`
//! and `GPU` on a last-place difference. The degenerate `fade_range <= EPS` and
//! boundary-distance cases are exercised separately with exact-zero or
//! wide-margin fixtures.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::ocean_lod`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::ocean_lod::{cascade_weights_into, OceanCascadeConfig};
use prism_volumetric_gpu::water_ocean_cascade::{
    GpuWaterOceanCascade, WaterOceanCascadeQuery, WaterOceanCascadeResult, MAX_CASCADES,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a cascade weight. A `GPU` divide may land a few
/// units in the last place from the scalar reference; `1e-4` admits that legal
/// slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Builds the in-host oracle for one query by calling the golden
/// `cascade_weights_into` into a stack buffer of the query's `out_len`. The
/// `GPU` is pinned against this exact closed form.
fn oracle(q: &WaterOceanCascadeQuery) -> WaterOceanCascadeResult {
    let cfg = OceanCascadeConfig {
        cascade_count: q.cascade_count,
        fade_start: q.fade_start,
        fade_range: q.fade_range,
        reach_per_cascade: q.reach_per_cascade,
    };
    let len = (q.out_len as usize).min(MAX_CASCADES);
    let mut buf = [0.0_f32; MAX_CASCADES];
    let count = cascade_weights_into(q.distance, cfg, &mut buf[..len]);
    let mut weights = [0.0_f32; MAX_CASCADES];
    weights[..count].copy_from_slice(&buf[..count]);
    WaterOceanCascadeResult {
        weights,
        count: count as u32,
    }
}

/// Pins one `GPU` result against the in-host oracle: `count` exactly, and every
/// valid weight within tolerance.
fn check_body(idx: usize, got: &WaterOceanCascadeResult, want: &WaterOceanCascadeResult) {
    assert_eq!(
        got.count, want.count,
        "body {idx} count: gpu {} vs cpu {}",
        got.count, want.count
    );
    for c in 0..(want.count as usize) {
        assert!(
            close(got.weights[c], want.weights[c]),
            "body {idx} weight {c}: gpu {} vs cpu {}",
            got.weights[c],
            want.weights[c]
        );
    }
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuWaterOceanCascade, queries: &[WaterOceanCascadeQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_body(idx, result, &want);
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

/// Draws a scalar in `[lo, hi]` at milli resolution from `state`.
fn draw(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let span = ((hi - lo) * 1000.0) as u32;
    lo + (lcg(state) % (span + 1)) as f32 / 1000.0
}

/// Whether `distance` keeps a comfortable margin from every cascade's
/// `fade_begin`, so the `distance <= begin` branch is stable across `CPU` and
/// `GPU`. Uses only arithmetic and comparisons, never an `f32` `==`.
fn well_conditioned(q: &WaterOceanCascadeQuery) -> bool {
    let cap = (q.cascade_count as usize).min(MAX_CASCADES);
    for c in 0..cap {
        let begin = q.fade_start + (c as f32) * q.reach_per_cascade;
        if (q.distance - begin).abs() < 1.0 {
            return false;
        }
    }
    true
}

/// Draws one well-conditioned random query: non-negative fields, `fade_range`
/// well above the `EPS` degenerate threshold, and a distance kept clear of
/// every cascade's `fade_begin`.
fn random_query(state: &mut u64) -> WaterOceanCascadeQuery {
    loop {
        let q = WaterOceanCascadeQuery {
            distance: draw(state, 0.0, 1600.0),
            fade_start: draw(state, 20.0, 400.0),
            fade_range: draw(state, 10.0, 300.0),
            reach_per_cascade: draw(state, 40.0, 400.0),
            cascade_count: lcg(state) % ((MAX_CASCADES as u32) + 1),
            out_len: lcg(state) % ((MAX_CASCADES as u32) + 1),
        };
        if well_conditioned(&q) {
            return q;
        }
    }
}

/// The deterministic hand-chosen fixtures, each exercising a distinct branch and
/// kept clear of every discrete tie.
fn fixture_queries() -> Vec<WaterOceanCascadeQuery> {
    vec![
        // Near camera: every cascade at full weight (distance below fade_start).
        WaterOceanCascadeQuery {
            distance: 10.0,
            fade_start: 100.0,
            fade_range: 50.0,
            reach_per_cascade: 200.0,
            cascade_count: 4,
            out_len: 4,
        },
        // Mid distance: finest cascade partway down its ramp, coarser ones full.
        WaterOceanCascadeQuery {
            distance: 125.0,
            fade_start: 100.0,
            fade_range: 50.0,
            reach_per_cascade: 200.0,
            cascade_count: 4,
            out_len: 4,
        },
        // Far distance: finest cascade fully faded (clamped to 0), swell remains.
        WaterOceanCascadeQuery {
            distance: 900.0,
            fade_start: 100.0,
            fade_range: 50.0,
            reach_per_cascade: 200.0,
            cascade_count: 4,
            out_len: 4,
        },
        // Full eight cascades, mid-range distance.
        WaterOceanCascadeQuery {
            distance: 640.0,
            fade_start: 80.0,
            fade_range: 70.0,
            reach_per_cascade: 150.0,
            cascade_count: 8,
            out_len: 8,
        },
        // Short buffer: count clamps to out_len below cascade_count.
        WaterOceanCascadeQuery {
            distance: 300.0,
            fade_start: 90.0,
            fade_range: 60.0,
            reach_per_cascade: 180.0,
            cascade_count: 6,
            out_len: 2,
        },
        // Oversized request: out_len above cascade_count, count clamps to the
        // cascade count.
        WaterOceanCascadeQuery {
            distance: 220.0,
            fade_start: 70.0,
            fade_range: 40.0,
            reach_per_cascade: 160.0,
            cascade_count: 3,
            out_len: 8,
        },
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_ocean_cascade parity: no wgpu adapter");
        return;
    };
    let gpu = GpuWaterOceanCascade::new(&ctx);
    // The host short-circuits an empty batch (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn fixture_bodies_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOceanCascade::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn zero_cascade_count_writes_nothing() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOceanCascade::new(&ctx);
    // cascade_count = 0: count is 0 and no weight is written.
    let q = WaterOceanCascadeQuery {
        distance: 150.0,
        fade_start: 100.0,
        fade_range: 50.0,
        reach_per_cascade: 200.0,
        cascade_count: 0,
        out_len: 4,
    };
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].count, 0, "zero cascades write no weights");
    check_body(0, &got[0], &oracle(&q));
}

#[test]
fn zero_out_len_writes_nothing() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOceanCascade::new(&ctx);
    // out_len = 0: a zero-length destination means count is 0 regardless of the
    // cascade count.
    let q = WaterOceanCascadeQuery {
        distance: 150.0,
        fade_start: 100.0,
        fade_range: 50.0,
        reach_per_cascade: 200.0,
        cascade_count: 5,
        out_len: 0,
    };
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].count, 0, "zero-length buffer writes no weights");
    check_body(0, &got[0], &oracle(&q));
}

#[test]
fn degenerate_fade_range_drops_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOceanCascade::new(&ctx);
    // fade_range exactly zero (below EPS): past fade_begin the weight collapses
    // to 0 with no ramp. Distances kept a wide margin from each fade_begin so
    // the branch is unambiguous.
    let q = WaterOceanCascadeQuery {
        distance: 500.0,
        fade_start: 100.0,
        fade_range: 0.0,
        reach_per_cascade: 120.0,
        cascade_count: 4,
        out_len: 4,
    };
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    // Cascade 0 (begin 100) and 1 (begin 220) and 2 (begin 340) are past the
    // distance 500 with zero range -> 0; cascade 3 (begin 460) is also below
    // 500 -> 0. All four collapse to zero.
    for c in 0..(want.count as usize) {
        assert!(
            want.weights[c].abs() < EPS,
            "degenerate range must drop cascade {c} to zero"
        );
    }
    check_body(0, &got[0], &want);
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOceanCascade::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random bodies pin every output across a wide
    // span of configs, distances and buffer lengths.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
