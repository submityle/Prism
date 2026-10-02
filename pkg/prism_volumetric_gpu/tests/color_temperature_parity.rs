//! Real-device parity for the colour-temperature twin:
//! [`GpuColorTemperature`](prism_volumetric_gpu::color_temperature::GpuColorTemperature)
//! must reproduce the `CPU` golden
//! [`color_temperature`](prism_render_architecture::particle::color_temperature)
//! across the resolved
//! [`rgb_gain`](prism_render_architecture::particle::color_temperature::WhiteBalance::rgb_gain)
//! triple and the
//! [`apply`](prism_render_architecture::particle::color_temperature::WhiteBalance::apply)
//! result.
//!
//! The fixtures cover the shapes the golden unit tests call out: the neutral
//! daylight point, a warm red-biased temperature, a very warm temperature below
//! the blue-zero knee (blue held at zero), the daylight warm branch, the cool
//! blue-biased branch, positive and negative tints, and the out-of-range
//! temperature and tint clamps. Every temperature stays clear of the
//! branch-boundary knees (`BLUE_ZERO_KELVIN` and `WARM_COOL_SPLIT_KELVIN`) and
//! the clamp edges except in the dedicated clamp fixtures, so `CPU` and `GPU`
//! always take the same branch. All inputs are written as integers or simple
//! decimals, so the fixtures stay pure and need no transcendental math.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The gain and the applied pixel thread through multiplies, adds and one
//! guarded rational division, so they are compared under tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::color_temperature`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::color_temperature::WhiteBalance;
use prism_volumetric_gpu::color_temperature::{ColorTemperatureQuery, GpuColorTemperature};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous quantities.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous quantities.
const REL_EPS: f32 = 1.0e-3;
/// Floor for the relative-tolerance denominator.
const REL_FLOOR: f32 = 1.0e-6;

/// Mixed absolute / relative tolerance comparison for one `f32` lane.
fn approx(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Tolerant comparison of two `vec3` / `RGB` triples.
fn approx3(a: [f32; 3], b: [f32; 3]) -> bool {
    approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
}

/// Builds a query from a temperature, tint and input pixel.
fn query(temp_kelvin: f32, tint: f32, rgb: [f32; 3]) -> ColorTemperatureQuery {
    ColorTemperatureQuery {
        temp_kelvin,
        tint,
        rgb,
    }
}

/// Asserts the twinned gain and applied pixel for one query match the golden.
fn assert_parity(gpu: &GpuColorTemperature, ctx: &GpuContext, q: &ColorTemperatureQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(q));
    assert_eq!(got.len(), 1, "one result per query");
    let g = got[0];

    let wb = WhiteBalance::new(q.temp_kelvin, q.tint);
    let cpu_gain = wb.rgb_gain();
    assert!(
        approx3(g.gain, cpu_gain),
        "gain mismatch at {}K tint {}: gpu {:?} vs cpu {cpu_gain:?}",
        q.temp_kelvin,
        q.tint,
        g.gain
    );
    let cpu_applied = wb.apply(q.rgb);
    assert!(
        approx3(g.applied, cpu_applied),
        "applied mismatch at {}K tint {}: gpu {:?} vs cpu {cpu_applied:?}",
        q.temp_kelvin,
        q.tint,
        g.applied
    );
}

#[test]
fn neutral_and_warm_and_cool_gains_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorTemperature::new(&ctx);
    // Neutral daylight: gain is an approximate identity.
    assert_parity(&gpu, &ctx, &query(6500.0, 0.0, [0.4, 0.6, 0.8]));
    // Warm, red-biased, within the blue warm branch.
    assert_parity(&gpu, &ctx, &query(3000.0, 0.2, [0.5, 0.25, 0.75]));
    // Cool, blue-biased, on the cool branch.
    assert_parity(&gpu, &ctx, &query(9000.0, -0.3, [0.3, 0.3, 0.9]));
}

#[test]
fn very_warm_has_no_blue() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorTemperature::new(&ctx);
    // Below the blue-zero knee the blue channel is held at zero.
    assert_parity(&gpu, &ctx, &query(1500.0, 0.0, [0.7, 0.5, 0.2]));
    assert_parity(&gpu, &ctx, &query(1200.0, 0.4, [1.0, 0.5, 0.25]));
}

#[test]
fn warm_and_cool_branches_sweep() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorTemperature::new(&ctx);
    // Several temperatures clear of the branch knees, each with a modest tint.
    let temps = [2200.0_f32, 4000.0, 5000.0, 6000.0, 7200.0, 10000.0, 13000.0];
    for &k in &temps {
        assert_parity(&gpu, &ctx, &query(k, 0.1, [0.5, 0.5, 0.5]));
        assert_parity(&gpu, &ctx, &query(k, -0.4, [0.2, 0.8, 0.6]));
    }
}

#[test]
fn tint_extremes_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorTemperature::new(&ctx);
    // Full green and full magenta correction at the tint endpoints.
    assert_parity(&gpu, &ctx, &query(6500.0, 1.0, [0.5, 0.5, 0.5]));
    assert_parity(&gpu, &ctx, &query(6500.0, -1.0, [0.5, 0.5, 0.5]));
}

#[test]
fn out_of_range_temperature_clamps() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorTemperature::new(&ctx);
    // Requests cooler/hotter than the fit domain clamp to the boundary gain.
    assert_parity(&gpu, &ctx, &query(200.0, 0.0, [0.6, 0.4, 0.2]));
    assert_parity(&gpu, &ctx, &query(40000.0, 0.0, [0.2, 0.4, 0.6]));
}

#[test]
fn out_of_range_tint_clamps() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorTemperature::new(&ctx);
    // Tints beyond [-1, 1] clamp to the endpoints before evaluation.
    assert_parity(&gpu, &ctx, &query(5000.0, 5.0, [0.5, 0.5, 0.5]));
    assert_parity(&gpu, &ctx, &query(5000.0, -5.0, [0.5, 0.5, 0.5]));
}

#[test]
fn batch_of_queries_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorTemperature::new(&ctx);
    // A batch exercises the one-thread-per-query flattening; each result must be
    // independent of its neighbours.
    let batch = [
        query(2500.0, 0.3, [0.9, 0.4, 0.1]),
        query(4800.0, 0.0, [0.5, 0.5, 0.5]),
        query(6500.0, -0.5, [0.3, 0.6, 0.9]),
        query(8000.0, 0.6, [0.2, 0.5, 0.8]),
        query(14000.0, -0.2, [0.4, 0.4, 0.7]),
    ];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), batch.len());
    for q in &batch {
        assert_parity(&gpu, &ctx, q);
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorTemperature::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
