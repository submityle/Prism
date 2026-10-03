//! Real-device parity for the waterline twin:
//! [`GpuWaterWaterlineWeight`](prism_volumetric_gpu::water_waterline_weight::GpuWaterWaterlineWeight)
//! must reproduce the `CPU` golden
//! [`submersion_depth`](prism_render_architecture::water::waterline::submersion_depth),
//! [`is_underwater`](prism_render_architecture::water::waterline::is_underwater),
//! [`waterline_weight`](prism_render_architecture::water::waterline::waterline_weight)
//! and
//! [`shoreline_band`](prism_render_architecture::water::waterline::shoreline_band)
//! across air/underwater samples, the soft ramp, a degenerate hard-step band,
//! the shoreline falloff and a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected values are produced by calling the golden `submersion_depth`,
//! `is_underwater`, `waterline_weight` and `shoreline_band` directly, so the
//! test pins `GPU == golden`, not merely that the shader compiles.
//!
//! # Parity criterion
//!
//! The kernel performs a signed comparison, a guarded divide and clamps, so a
//! `GPU` reciprocal may land a few units in the last place from the scalar
//! reference. Every continuous output is asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`, and the underwater flag is matched exactly. Fixtures keep
//! the submersion depth away from zero so the flag is unambiguous.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::waterline`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::waterline::{
    is_underwater, shoreline_band, submersion_depth, waterline_weight, WaterlineParams,
};
use prism_volumetric_gpu::water_waterline_weight::{
    GpuWaterWaterlineWeight, WaterWaterlineWeightQuery, WaterWaterlineWeightResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` reciprocal may land a few units in the last
/// place from the scalar reference; `1e-4` admits that legal slack while still
/// failing a wrong port.
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

/// Reconstructs the golden result in-host by calling the reference functions
/// directly under the given tuning.
fn oracle(q: &WaterWaterlineWeightQuery, params: WaterlineParams) -> WaterWaterlineWeightResult {
    WaterWaterlineWeightResult {
        submersion_depth: submersion_depth(q.sample_y, q.water_surface_y),
        is_underwater: is_underwater(q.sample_y, q.water_surface_y),
        waterline_weight: waterline_weight(q.sample_y, q.water_surface_y, params),
        shoreline_band: shoreline_band(q.sample_y, q.water_surface_y, q.water_depth, params),
    }
}

/// Pins one `GPU` result against the in-host oracle: continuous outputs within
/// tolerance, the underwater flag exact.
fn check_query(idx: usize, got: &WaterWaterlineWeightResult, want: &WaterWaterlineWeightResult) {
    assert_eq!(
        got.is_underwater, want.is_underwater,
        "query {idx} is_underwater: gpu {} vs cpu {}",
        got.is_underwater, want.is_underwater
    );
    assert!(
        close(got.submersion_depth, want.submersion_depth),
        "query {idx} submersion_depth: gpu {} vs cpu {}",
        got.submersion_depth,
        want.submersion_depth
    );
    assert!(
        close(got.waterline_weight, want.waterline_weight),
        "query {idx} waterline_weight: gpu {} vs cpu {}",
        got.waterline_weight,
        want.waterline_weight
    );
    assert!(
        close(got.shoreline_band, want.shoreline_band),
        "query {idx} shoreline_band: gpu {} vs cpu {}",
        got.shoreline_band,
        want.shoreline_band
    );
}

/// Dispatches `queries` under `params` and checks every result against the
/// oracle.
fn check(
    ctx: &GpuContext,
    gpu: &GpuWaterWaterlineWeight,
    params: WaterlineParams,
    queries: &[WaterWaterlineWeightQuery],
) {
    let got = gpu.evaluate(
        ctx,
        params.transition_half_width,
        params.shoreline_depth,
        queries,
    );
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q, params);
        check_query(idx, g, &want);
    }
}

/// A small `LCG` for the randomized sweep (host-only; the kernel is portable).
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Maps a raw `u32` to an `f32` in `[lo, hi]` without any transcendental call.
fn uniform(bits: u32, lo: f32, hi: f32) -> f32 {
    let unit = (bits as f32) / (u32::MAX as f32);
    lo + unit * (hi - lo)
}

#[test]
fn empty_batch_produces_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWaterlineWeight::new(&ctx);
    // An empty batch short-circuits on the host (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, 0.25, 1.0, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn air_and_underwater_samples_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWaterlineWeight::new(&ctx);
    let params = WaterlineParams {
        transition_half_width: 0.25,
        shoreline_depth: 1.0,
    };
    // Samples well above and well below a surface at 5.0, with depths spanning
    // dry, shallow and deep water. Submersion depth stays away from zero.
    let queries = [
        WaterWaterlineWeightQuery::new(7.0, 5.0, 0.0),
        WaterWaterlineWeightQuery::new(5.6, 5.0, 0.0),
        WaterWaterlineWeightQuery::new(4.4, 5.0, 0.3),
        WaterWaterlineWeightQuery::new(3.0, 5.0, 2.5),
    ];
    check(&ctx, &gpu, params, &queries);
}

#[test]
fn soft_ramp_midpoints_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWaterlineWeight::new(&ctx);
    let params = WaterlineParams {
        transition_half_width: 0.5,
        shoreline_depth: 1.5,
    };
    // Samples inside the +/- 0.5 band around a surface at 2.0, landing the ramp
    // weight at interior values 0.2, 0.4, 0.6, 0.8 (depths -0.3..0.3), away from
    // the clamp saturation edges.
    let queries = [
        WaterWaterlineWeightQuery::new(2.3, 2.0, 0.4),
        WaterWaterlineWeightQuery::new(2.1, 2.0, 0.4),
        WaterWaterlineWeightQuery::new(1.9, 2.0, 0.4),
        WaterWaterlineWeightQuery::new(1.7, 2.0, 0.4),
    ];
    check(&ctx, &gpu, params, &queries);
}

#[test]
fn degenerate_band_is_a_hard_step() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWaterlineWeight::new(&ctx);
    // A zero band collapses to a hard step at the surface, matching the golden
    // guard `transition_half_width <= EPS`.
    let params = WaterlineParams {
        transition_half_width: 0.0,
        shoreline_depth: 1.0,
    };
    let queries = [
        WaterWaterlineWeightQuery::new(4.9, 5.0, 0.2),
        WaterWaterlineWeightQuery::new(5.1, 5.0, 0.0),
    ];
    check(&ctx, &gpu, params, &queries);
}

#[test]
fn shoreline_band_falloff_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWaterlineWeight::new(&ctx);
    let params = WaterlineParams {
        transition_half_width: 0.25,
        shoreline_depth: 1.0,
    };
    // Submerged samples at a range of depths: shallow water rises toward 1,
    // deep water falls to 0; a dry sample never fires.
    let queries = [
        WaterWaterlineWeightQuery::new(6.0, 5.0, 0.1),
        WaterWaterlineWeightQuery::new(4.5, 5.0, 0.2),
        WaterWaterlineWeightQuery::new(4.5, 5.0, 0.6),
        WaterWaterlineWeightQuery::new(4.5, 5.0, 1.4),
    ];
    check(&ctx, &gpu, params, &queries);
}

#[test]
fn degenerate_shoreline_reach_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWaterlineWeight::new(&ctx);
    // A zero shoreline reach guards the divide and yields a zero band even for
    // very shallow submerged water, matching `shoreline_depth > EPS`.
    let params = WaterlineParams {
        transition_half_width: 0.3,
        shoreline_depth: 0.0,
    };
    let queries = [
        WaterWaterlineWeightQuery::new(4.8, 5.0, 0.05),
        WaterWaterlineWeightQuery::new(4.2, 5.0, 0.5),
    ];
    check(&ctx, &gpu, params, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWaterlineWeight::new(&ctx);
    let params = WaterlineParams {
        transition_half_width: 0.35,
        shoreline_depth: 1.25,
    };
    let mut state = 0x2f6b_9d14_3a7c_5e08_u64;
    let mut queries: Vec<WaterWaterlineWeightQuery> = Vec::new();
    // Several workgroups' worth of random samples. The submersion depth is kept
    // at least 0.05m from zero by rejection so the underwater flag is
    // unambiguous (far from the branch critical point).
    while queries.len() < 512 {
        let water_surface_y = uniform(lcg(&mut state), -4.0, 4.0);
        let sample_y = uniform(lcg(&mut state), -4.0, 4.0);
        if (water_surface_y - sample_y).abs() < 0.05 {
            continue;
        }
        let water_depth = uniform(lcg(&mut state), 0.0, 3.0);
        queries.push(WaterWaterlineWeightQuery::new(
            sample_y,
            water_surface_y,
            water_depth,
        ));
    }
    check(&ctx, &gpu, params, &queries);
}
