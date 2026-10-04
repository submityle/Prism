//! Real-device parity for the hair scatter-LOD twin:
//! [`GpuHairScatterLod`](prism_volumetric_gpu::hair_scatter_lod::GpuHairScatterLod)
//! must reproduce, for one query per thread, the three closed forms of the
//! golden `prism_render_architecture::hair::scatter_lod` module: the continuous
//! far-field `scatter_blend`, the discrete `scatter_regime` classified from it,
//! and the `far_field_roughness_gain`.
//!
//! # Independent oracle
//!
//! This suite does not depend on the reference crate. The host [`oracle`] is an
//! independent `f32` reimplementation of the same closed form documented on the
//! twin, evaluated in the same arithmetic order as the kernel. The golden
//! sanitisation routes every input through `sanitize_nonneg` (non-finite or
//! negative becomes `0`) or `clamp01` (non-finite becomes `0`, else clamp to
//! `[0, 1]`); the oracle uses `f32::is_finite` and the same ordered compares,
//! so a negative, infinite or `NaN` input takes the same branch as the kernel's
//! `(x == x) && abs(x) < 3.4e38` guard. A passing run is therefore evidence
//! that the `WGSL` kernel and an independent `CPU` evaluation of the same maps
//! agree, not merely that the shader compiles.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of absolute values, ordered
//! compares, one divide and one multiply-add, so the two evaluations compute
//! the same closed form in the same order. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar host leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on the
//! continuous `blend` and `roughness_gain` and an exact `==` on the discrete
//! `regime` and `valid`.
//!
//! # Conditioning
//!
//! `scatter_regime` has hard thresholds at `blend <= EPS` (Near) and `blend >=
//! 1 - EPS` (Far), so a blend a few `ULP` either side of a knee could send the
//! two evaluators to different discrete regimes. The random sweep is therefore
//! rejection-sampled so the blend stays comfortably inside one regime, clear of
//! both knees, and the thresholds keep `span = near - far` well above `EPS` so
//! neither evaluator takes the degenerate-band fall-through. The exact-boundary
//! cases (`w >= near`, `w <= far`, `near == far`) are instead pinned by named
//! fixtures, where both evaluators take the same branch exactly.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::hair::scatter_lod`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::hair_scatter_lod::{
    GpuHairScatterLod, HairScatterLodQuery, HairScatterLodResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity floor on every continuous output.
const ABS_TOL: f32 = 1.0e-4;
/// Relative parity slope on every continuous output.
const REL_TOL: f32 = 1.0e-3;
/// Relative-tolerance floor so near-zero magnitudes stay meaningful.
const REL_FLOOR: f32 = 1.0e-6;
/// The reference epsilon bracketing the blend knees and the degenerate band,
/// matching the golden `EPS`.
const EPS_LOCAL: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the documented parity bound: an
/// absolute floor or a relative term keeping large-magnitude values meaningful.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    diff <= ABS_TOL || diff <= REL_TOL * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// `if x.is_finite() { x.max(0.0) } else { 0.0 }`: an independent `f32`
/// reimplementation of the golden `sanitize_nonneg`.
fn sanitize_nonneg(x: f32) -> f32 {
    if x.is_finite() {
        x.max(0.0)
    } else {
        0.0
    }
}

/// `if x.is_finite() { x.clamp(0.0, 1.0) } else { 0.0 }`: an independent `f32`
/// reimplementation of the golden `clamp01`.
fn clamp01(x: f32) -> f32 {
    if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Independent `f32` reimplementation of `scatter_blend`, in the same
/// arithmetic order as the kernel, with the sanitized thresholds and the
/// degenerate-band fall-through.
fn scatter_blend(fiber_width_px: f32, near_px: f32, far_px: f32) -> f32 {
    let far = sanitize_nonneg(far_px);
    let near = sanitize_nonneg(near_px).max(far);
    let w = sanitize_nonneg(fiber_width_px);
    if w >= near {
        return 0.0;
    }
    if w <= far {
        return 1.0;
    }
    let span = near - far;
    if span <= EPS_LOCAL {
        return 1.0;
    }
    ((near - w) / span).clamp(0.0, 1.0)
}

/// Independent `f32` reimplementation of `scatter_regime`, returning the regime
/// code (`0` Near, `1` Far, `2` Blended) classified from the blend knees.
fn scatter_regime(blend: f32) -> u32 {
    if blend <= EPS_LOCAL {
        0
    } else if blend >= 1.0 - EPS_LOCAL {
        1
    } else {
        2
    }
}

/// Independent `f32` reimplementation of the full result for one query.
fn oracle(query: &HairScatterLodQuery) -> HairScatterLodResult {
    let blend = scatter_blend(query.fiber_width_px, query.near_px, query.far_px);
    let regime = scatter_regime(blend);
    let roughness_gain = 1.0 + clamp01(blend) * sanitize_nonneg(query.max_gain);
    HairScatterLodResult {
        blend,
        regime,
        roughness_gain,
        valid: 1,
    }
}

/// Pins one `GPU` result against the independent host oracle: both continuous
/// outputs within the parity bound and the discrete `regime` / `valid` flags
/// exactly.
fn pin(idx: usize, query: &HairScatterLodQuery, result: &HairScatterLodResult) {
    let want = oracle(query);
    assert!(
        close(result.blend, want.blend),
        "query {idx}: blend gpu={} oracle={}",
        result.blend,
        want.blend
    );
    assert!(
        close(result.roughness_gain, want.roughness_gain),
        "query {idx}: roughness_gain gpu={} oracle={}",
        result.roughness_gain,
        want.roughness_gain
    );
    assert_eq!(
        result.regime, want.regime,
        "query {idx}: regime gpu={} oracle={}",
        result.regime, want.regime
    );
    assert_eq!(
        result.valid, want.valid,
        "query {idx}: valid gpu={} oracle={}",
        result.valid, want.valid
    );
}

/// Evaluates `queries` on-device and pins every result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuHairScatterLod, queries: &[HairScatterLodQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

/// 64-bit linear-congruential step (`Knuth`/`PCG` constants), returning the
/// high word so the stream has good spread without any transcendental math.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 32) as u32
}

/// A deterministic pseudo-random `f32` in `[0, 1]`.
fn unit01(state: &mut u64) -> f32 {
    lcg(state) as f32 / u32::MAX as f32
}

/// A deterministic, well-conditioned random query: a finite non-negative near
/// threshold with a strictly smaller far threshold (so `span` is comfortably
/// above `EPS`), a width rejection-sampled so the blend stays clear of both the
/// `EPS` and `1 - EPS` knees, and a non-negative `max_gain` in `[0, 8]`.
fn rand_query(state: &mut u64) -> HairScatterLodQuery {
    // near in [1, 9], far in [0.1, near - 0.5] so span >= ~0.5 >> EPS.
    let near = 1.0 + unit01(state) * 8.0;
    let far = 0.1 + unit01(state) * (near - 0.6);
    let span = near - far;
    let width = loop {
        // Width spread across and beyond the band so Near/Far/Blended all occur.
        let w = unit01(state) * (near + 1.0);
        let blend = scatter_blend(w, near, far);
        // Keep the blend a safe margin clear of both regime knees so the
        // discrete classification cannot flip between CPU and GPU. The margin
        // (0.01) is many orders of magnitude above EPS, well inside one regime.
        if (blend <= 0.0 || blend >= 1.0 || (blend > 0.01 && blend < 0.99)) && span > 0.5 {
            break w;
        }
    };
    let max_gain = unit01(state) * 8.0;
    HairScatterLodQuery::new(width, near, far, max_gain)
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairScatterLod::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn width_at_or_above_near_is_near() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairScatterLod::new(&ctx);
    // Width exactly at (and above) the near threshold is pure near-field:
    // blend 0, regime Near = 0, gain 1 (clamp01(0) * g = 0).
    let queries = [
        HairScatterLodQuery::new(1.0, 1.0, 0.25, 4.0),
        HairScatterLodQuery::new(3.0, 1.0, 0.25, 4.0),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len());
    for (idx, (q, r)) in queries.iter().zip(got.iter()).enumerate() {
        assert!(close(r.blend, 0.0), "blend should be 0, got {}", r.blend);
        assert_eq!(r.regime, 0, "regime should be Near");
        assert!(
            close(r.roughness_gain, 1.0),
            "gain should be 1, got {}",
            r.roughness_gain
        );
        pin(idx, q, r);
    }
}

#[test]
fn width_at_or_below_far_is_far() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairScatterLod::new(&ctx);
    // Width at (and below) the far threshold is pure far-field: blend 1, regime
    // Far = 1, gain 1 + max_gain.
    let queries = [
        HairScatterLodQuery::new(0.25, 1.0, 0.25, 4.0),
        HairScatterLodQuery::new(0.0, 1.0, 0.25, 4.0),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len());
    for (idx, (q, r)) in queries.iter().zip(got.iter()).enumerate() {
        assert!(close(r.blend, 1.0), "blend should be 1, got {}", r.blend);
        assert_eq!(r.regime, 1, "regime should be Far");
        assert!(
            close(r.roughness_gain, 5.0),
            "gain should be 1 + 4 = 5, got {}",
            r.roughness_gain
        );
        pin(idx, q, r);
    }
}

#[test]
fn midband_blended() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairScatterLod::new(&ctx);
    // Width at the band midpoint: blend = (near - w) / span = 0.5, regime
    // Blended = 2, gain = 1 + 0.5 * 2 = 2.
    let query = HairScatterLodQuery::new(0.5, 1.0, 0.0, 2.0);
    let got = gpu.evaluate(&ctx, &[query]);
    assert_eq!(got.len(), 1, "one result for one query");
    assert!(
        close(got[0].blend, 0.5),
        "midband blend should be 0.5, got {}",
        got[0].blend
    );
    assert_eq!(got[0].regime, 2, "regime should be Blended");
    assert!(
        close(got[0].roughness_gain, 2.0),
        "gain should be 2, got {}",
        got[0].roughness_gain
    );
    pin(0, &query, &got[0]);
}

#[test]
fn degenerate_band_hard_step() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairScatterLod::new(&ctx);
    // near == far: there is no strictly-between band. A width above the common
    // bound is Near (blend 0); at or below it is Far (blend 1, w <= far branch).
    let queries = [
        HairScatterLodQuery::new(2.0, 1.0, 1.0, 3.0),
        HairScatterLodQuery::new(0.5, 1.0, 1.0, 3.0),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len());
    assert!(close(got[0].blend, 0.0), "above bound -> blend 0");
    assert_eq!(got[0].regime, 0, "above bound -> Near");
    assert!(close(got[1].blend, 1.0), "below bound -> blend 1");
    assert_eq!(got[1].regime, 1, "below bound -> Far");
    for (idx, (q, r)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, q, r);
    }
}

#[test]
fn nonfinite_width_sanitizes_to_far() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairScatterLod::new(&ctx);
    // A non-finite width sanitizes to 0 (the finest footprint), so it is at or
    // below the far threshold -> blend 1, regime Far.
    let queries = [
        HairScatterLodQuery::new(f32::NAN, 1.0, 0.25, 2.0),
        HairScatterLodQuery::new(f32::INFINITY, 1.0, 0.25, 2.0),
        HairScatterLodQuery::new(f32::NEG_INFINITY, 1.0, 0.25, 2.0),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len());
    for (idx, (q, r)) in queries.iter().zip(got.iter()).enumerate() {
        assert!(
            close(r.blend, 1.0),
            "non-finite width {idx} should sanitize to far (blend 1), got {}",
            r.blend
        );
        assert_eq!(r.regime, 1, "non-finite width {idx} should be Far");
        pin(idx, q, r);
    }
}

#[test]
fn nonfinite_thresholds_sanitize() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairScatterLod::new(&ctx);
    // Non-finite near/far sanitize to 0, so near = far = 0; any non-negative
    // width is >= near -> blend 0, regime Near. A negative width sanitizes to
    // 0 which is still >= near (0) -> Near as well.
    let queries = [
        HairScatterLodQuery::new(0.5, f32::NAN, f32::NAN, 2.0),
        HairScatterLodQuery::new(0.5, f32::INFINITY, f32::NEG_INFINITY, 2.0),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len());
    for (idx, (q, r)) in queries.iter().zip(got.iter()).enumerate() {
        assert!(
            close(r.blend, 0.0),
            "sanitized thresholds -> blend 0, got {}",
            r.blend
        );
        assert_eq!(r.regime, 0, "sanitized thresholds -> Near");
        pin(idx, q, r);
    }
}

#[test]
fn roughness_gain_formula() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairScatterLod::new(&ctx);
    // Blend 0.25 (w = 0.75 in a [0, 1] band) with max_gain 8: gain = 1 + 0.25 *
    // 8 = 3.
    let query = HairScatterLodQuery::new(0.75, 1.0, 0.0, 8.0);
    let got = gpu.evaluate(&ctx, &[query]);
    assert_eq!(got.len(), 1, "one result for one query");
    assert!(
        close(got[0].blend, 0.25),
        "blend should be 0.25, got {}",
        got[0].blend
    );
    assert!(
        close(got[0].roughness_gain, 3.0),
        "gain should be 1 + 0.25 * 8 = 3, got {}",
        got[0].roughness_gain
    );
    pin(0, &query, &got[0]);
}

#[test]
fn negative_max_gain_sanitizes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairScatterLod::new(&ctx);
    // A negative or non-finite max_gain sanitizes to 0, so the gain is exactly
    // 1 regardless of the blend.
    let queries = [
        HairScatterLodQuery::new(0.5, 1.0, 0.0, -4.0),
        HairScatterLodQuery::new(0.5, 1.0, 0.0, f32::NAN),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len());
    for (idx, (q, r)) in queries.iter().zip(got.iter()).enumerate() {
        assert!(
            close(r.roughness_gain, 1.0),
            "sanitized max_gain -> gain 1, got {}",
            r.roughness_gain
        );
        pin(idx, q, r);
    }
}

#[test]
fn stride_regression_two_element_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairScatterLod::new(&ctx);
    // Two deliberately distinct queries in one batch: if the host `std430` query
    // or result stride disagreed with the `WGSL` struct, the second lane would
    // decode from the wrong bytes and the pin would fail. The distinct widths,
    // thresholds and gains make such a mis-stride observable across all four
    // output lanes (two continuous, two discrete).
    let queries = [
        HairScatterLodQuery::new(0.5, 1.0, 0.0, 6.0),
        HairScatterLodQuery::new(3.0, 2.0, 0.5, 2.5),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairScatterLod::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the deterministic fixtures with many well-conditioned
    // random queries, dispatched together so the per-thread indexing and the
    // contiguous storage layout are both exercised.
    let mut queries = vec![
        HairScatterLodQuery::new(1.0, 1.0, 0.25, 4.0),
        HairScatterLodQuery::new(0.25, 1.0, 0.25, 4.0),
        HairScatterLodQuery::new(0.5, 1.0, 0.0, 2.0),
        HairScatterLodQuery::new(2.0, 1.0, 1.0, 3.0),
    ];
    for _ in 0..48 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairScatterLod::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) of well-conditioned random
    // queries pins every output across many invocations.
    let queries: Vec<HairScatterLodQuery> = (0..512).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
