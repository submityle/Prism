//! Real-device parity for the stylized (`NPR`) water-lighting response twin:
//! [`GpuWaterShadingNpr`](prism_volumetric_gpu::water_shading_npr::GpuWaterShadingNpr)
//! must reproduce the stateless `f32` response of the `CPU` golden
//! [`plan_npr`](prism_render_architecture::water::shading::npr::plan_npr) — the
//! per-channel ramp-quantized color (via
//! [`quantize_ramp`](prism_render_architecture::water::shading::npr::quantize_ramp)),
//! the hard toon specular block, the guarded shoreline foam-edge fade, the
//! clamped halftone caustic coverage with its scale, and the saturating flow
//! line — across the golden fixture, the toon-on/off cases, the foam shore/mid/
//! offshore and degenerate-width cases, the flow-saturating and flow-zero cases,
//! the zero-band case plus a randomized sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The reference functions are public, so they drive the oracle directly: each
//! query's five tuning fields rebuild an
//! [`NprShadingParams`](prism_render_architecture::water::shading::NprShadingParams)
//! and its seven consumed inputs rebuild a
//! [`SurfaceShadingInputs`](prism_render_architecture::water::shading::SurfaceShadingInputs)
//! (the six fields `plan_npr` never reads are fixed to representative
//! constants), then
//! [`plan_npr`](prism_render_architecture::water::shading::npr::plan_npr)
//! supplies the expected response. A passing `GPU == oracle` run is direct
//! evidence the kernel computes the same stylized response.
//!
//! # Parity criterion
//!
//! Every output is a pure `f32` map built from `clamp`, multiply, divide,
//! ordered comparison and an integer band truncation, so `CPU` and `GPU` agree
//! to within floating-point tolerance; each field is asserted with an
//! absolute-or-relative closeness check (`abs <= 1e-4` or `rel <= 1e-3`, with a
//! `REL_FLOOR` of `1e-6`). The only discrete decisions — the band truncation
//! and the toon threshold — are kept away from their exact boundaries by the
//! fixtures and by reject-sampling the random sweep, so a last-place rounding
//! difference cannot flip a band or the toon block.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::shading::npr`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::shading::npr::plan_npr;
use prism_render_architecture::water::shading::{NprShadingParams, SurfaceShadingInputs};
use prism_render_architecture::water::underwater::RgbColor;
use prism_volumetric_gpu::water_shading_npr::{
    GpuWaterShadingNpr, WaterShadingNprQuery, WaterShadingNprResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute closeness floor for the parity comparison.
const ABS: f32 = 1e-4;
/// Relative closeness bound for the parity comparison.
const REL: f32 = 1e-3;
/// Smallest denominator used in the relative comparison, guarding `0 == 0`.
const REL_FLOOR: f32 = 1e-6;

/// Absolute-or-relative closeness: `true` when `a` and `b` agree to within the
/// shared tolerance. Values that both land near zero pass through the absolute
/// bound; larger values use the relative bound against the larger magnitude.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= ABS || rel <= REL
}

/// Computes the reference response for one query by calling the golden directly.
///
/// The six [`SurfaceShadingInputs`](prism_render_architecture::water::shading::SurfaceShadingInputs)
/// fields `plan_npr` never reads (`cos_view`, `ior`, `jacobian`, `depth`,
/// `ssr_confidence`, `ray_budget`) are fixed to representative constants and do
/// not affect the stylized response.
fn oracle(q: &WaterShadingNprQuery) -> WaterShadingNprResult {
    let params = NprShadingParams {
        color_bands: q.color_bands,
        specular_threshold: q.specular_threshold,
        foam_edge_width: q.foam_edge_width,
        halftone_scale: q.halftone_scale,
        flow_line_gain: q.flow_line_gain,
    };
    let inputs = SurfaceShadingInputs {
        cos_view: 0.5,
        ior: 1.33,
        jacobian: 0.4,
        water_color: RgbColor {
            r: q.water_color_r,
            g: q.water_color_g,
            b: q.water_color_b,
        },
        specular_intensity: q.specular_intensity,
        caustic_intensity: q.caustic_intensity,
        flow_speed: q.flow_speed,
        depth: 1.0,
        dist_to_shore: q.dist_to_shore,
        ssr_confidence: 0.7,
        ray_budget: 0.5,
    };
    let r = plan_npr(params, inputs);
    WaterShadingNprResult {
        ramp_r: r.ramp_color.r,
        ramp_g: r.ramp_color.g,
        ramp_b: r.ramp_color.b,
        toon_specular: r.toon_specular,
        foam_edge: r.foam_edge,
        halftone_coverage: r.halftone_coverage,
        halftone_scale: r.halftone_scale,
        flow_line: r.flow_line,
    }
}

/// Pins one `GPU` result against the oracle: every field within tolerance.
fn check_result(idx: usize, got: &WaterShadingNprResult, want: &WaterShadingNprResult) {
    assert!(
        close(got.ramp_r, want.ramp_r),
        "query {idx} ramp_r: gpu {} vs cpu {}",
        got.ramp_r,
        want.ramp_r
    );
    assert!(
        close(got.ramp_g, want.ramp_g),
        "query {idx} ramp_g: gpu {} vs cpu {}",
        got.ramp_g,
        want.ramp_g
    );
    assert!(
        close(got.ramp_b, want.ramp_b),
        "query {idx} ramp_b: gpu {} vs cpu {}",
        got.ramp_b,
        want.ramp_b
    );
    assert!(
        close(got.toon_specular, want.toon_specular),
        "query {idx} toon_specular: gpu {} vs cpu {}",
        got.toon_specular,
        want.toon_specular
    );
    assert!(
        close(got.foam_edge, want.foam_edge),
        "query {idx} foam_edge: gpu {} vs cpu {}",
        got.foam_edge,
        want.foam_edge
    );
    assert!(
        close(got.halftone_coverage, want.halftone_coverage),
        "query {idx} halftone_coverage: gpu {} vs cpu {}",
        got.halftone_coverage,
        want.halftone_coverage
    );
    assert!(
        close(got.halftone_scale, want.halftone_scale),
        "query {idx} halftone_scale: gpu {} vs cpu {}",
        got.halftone_scale,
        want.halftone_scale
    );
    assert!(
        close(got.flow_line, want.flow_line),
        "query {idx} flow_line: gpu {} vs cpu {}",
        got.flow_line,
        want.flow_line
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuWaterShadingNpr, queries: &[WaterShadingNprQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_result(idx, result, &want);
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

/// Maps a raw `u32` into `[lo, hi)` as an `f32`, using only integer and
/// floating-point arithmetic (no transcendental), for the random sweep.
fn uniform(raw: u32, lo: f32, hi: f32) -> f32 {
    let unit = (raw as f32) / (u32::MAX as f32);
    lo + unit * (hi - lo)
}

/// Draws a color channel in `0..=1` that stays clear of every band boundary
/// `k / bands`, so a last-place difference in the integer band truncation
/// cannot flip the quantized step between `CPU` and `GPU`. The guard half-width
/// is a safe fraction of the band spacing.
fn channel_off_boundary(state: &mut u64, bands: u32) -> f32 {
    let b = bands.max(1);
    let spacing = 1.0 / (b as f32);
    let guard = spacing * 0.1;
    loop {
        let t = uniform(lcg(state), 0.02, 0.98);
        let scaled = t * (b as f32);
        let frac = scaled - (scaled as u32 as f32);
        // Keep away from both the lower edge (frac near 0) and the next edge
        // (frac near 1) of the current band.
        if frac > guard / spacing && frac < 1.0 - guard / spacing {
            return t;
        }
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_shading_npr parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWaterShadingNpr::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn golden_fixture_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterShadingNpr::new(&ctx);
    // The reference's own fixture: four bands, toon on (0.8 >= 0.7), foam fading
    // over 1.5 m with the sample 1 m from shore, caustics at 0.6, flow 1.2 * 0.5.
    let queries = [WaterShadingNprQuery::new(
        4, 0.7, 1.5, 8.0, 0.5, 0.1, 0.3, 0.5, 0.8, 0.6, 1.2, 1.0,
    )];
    check(&ctx, &gpu, &queries);
}

#[test]
fn toon_specular_blocks_on_and_off() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterShadingNpr::new(&ctx);
    // Specular well above the threshold turns the block on; well below leaves it
    // off. Both are kept clear of the exact threshold.
    let queries = [
        WaterShadingNprQuery::new(4, 0.7, 1.5, 8.0, 0.5, 0.1, 0.3, 0.5, 0.9, 0.6, 1.2, 1.0),
        WaterShadingNprQuery::new(4, 0.7, 1.5, 8.0, 0.5, 0.1, 0.3, 0.5, 0.5, 0.6, 1.2, 1.0),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert!(close(got[0].toon_specular, 1.0), "above threshold is on");
    assert!(close(got[1].toon_specular, 0.0), "below threshold is off");
    check(&ctx, &gpu, &queries);
}

#[test]
fn foam_edge_fades_and_degenerates() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterShadingNpr::new(&ctx);
    // At the shore the foam saturates to 1; mid-fade it is a clamped ratio; well
    // offshore it clamps to 0; and a collapsed (zero) edge width disables it.
    let queries = [
        WaterShadingNprQuery::new(4, 0.7, 2.0, 8.0, 0.5, 0.1, 0.3, 0.5, 0.8, 0.6, 1.2, 0.0),
        WaterShadingNprQuery::new(4, 0.7, 2.0, 8.0, 0.5, 0.1, 0.3, 0.5, 0.8, 0.6, 1.2, 0.5),
        WaterShadingNprQuery::new(4, 0.7, 2.0, 8.0, 0.5, 0.1, 0.3, 0.5, 0.8, 0.6, 1.2, 10.0),
        WaterShadingNprQuery::new(4, 0.7, 0.0, 8.0, 0.5, 0.1, 0.3, 0.5, 0.8, 0.6, 1.2, 0.5),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert!(close(got[0].foam_edge, 1.0), "at the shore foam saturates");
    assert!(close(got[1].foam_edge, 0.75), "mid-fade is a clamped ratio");
    assert!(close(got[2].foam_edge, 0.0), "offshore foam clamps to zero");
    assert!(
        close(got[3].foam_edge, 0.0),
        "a collapsed width disables foam"
    );
    check(&ctx, &gpu, &queries);
}

#[test]
fn flow_line_saturates_and_zeroes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterShadingNpr::new(&ctx);
    // A large flow speed saturates the line to 1; a zero speed yields 0.
    let queries = [
        WaterShadingNprQuery::new(4, 0.7, 1.5, 8.0, 0.5, 0.1, 0.3, 0.5, 0.8, 0.6, 100.0, 1.0),
        WaterShadingNprQuery::new(4, 0.7, 1.5, 8.0, 0.5, 0.1, 0.3, 0.5, 0.8, 0.6, 0.0, 1.0),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert!(close(got[0].flow_line, 1.0), "fast flow saturates the line");
    assert!(close(got[1].flow_line, 0.0), "still water has no flow line");
    check(&ctx, &gpu, &queries);
}

#[test]
fn halftone_coverage_clamps() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterShadingNpr::new(&ctx);
    // Caustic coverage below zero and above one both clamp into the unit range,
    // while the scale is carried through verbatim.
    let queries = [
        WaterShadingNprQuery::new(4, 0.7, 1.5, 12.0, 0.5, 0.1, 0.3, 0.5, 0.8, -0.4, 1.2, 1.0),
        WaterShadingNprQuery::new(4, 0.7, 1.5, 12.0, 0.5, 0.1, 0.3, 0.5, 0.8, 1.6, 1.2, 1.0),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert!(
        close(got[0].halftone_coverage, 0.0),
        "negative clamps to zero"
    );
    assert!(
        close(got[1].halftone_coverage, 1.0),
        "over-one clamps to one"
    );
    assert!(
        close(got[0].halftone_scale, 12.0),
        "scale is carried through"
    );
    check(&ctx, &gpu, &queries);
}

#[test]
fn zero_bands_floors_to_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterShadingNpr::new(&ctx);
    // Zero bands is floored to one, so every channel truncates to the single
    // band's lower edge (0) unless it is exactly 1 (which caps at 1). Channels
    // are kept clear of 1.0 so the one-band index stays 0.
    let queries = [WaterShadingNprQuery::new(
        0, 0.7, 1.5, 8.0, 0.5, 0.2, 0.55, 0.9, 0.8, 0.6, 1.2, 1.0,
    )];
    let got = gpu.evaluate(&ctx, &queries);
    assert!(close(got[0].ramp_r, 0.0), "one band snaps red to zero");
    assert!(close(got[0].ramp_g, 0.0), "one band snaps green to zero");
    assert!(close(got[0].ramp_b, 0.0), "one band snaps blue to zero");
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterShadingNpr::new(&ctx);
    let mut state = 0x5f1c_8a33_d209_e471_u64;
    let mut queries = Vec::new();
    // Several workgroups' worth of queries. Band counts stay small so the ramp
    // steps are coarse; color channels are reject-sampled clear of every band
    // boundary; specular is kept a margin away from its threshold; and the foam
    // width is either exactly zero (degenerate branch) or comfortably positive,
    // never near the guard epsilon.
    while queries.len() < 300 {
        let bands = 2 + (lcg(&mut state) % 7); // 2..=8
        let r = channel_off_boundary(&mut state, bands);
        let g = channel_off_boundary(&mut state, bands);
        let b = channel_off_boundary(&mut state, bands);

        let threshold = uniform(lcg(&mut state), 0.2, 0.8);
        // Push specular at least 0.05 away from the threshold, on a side chosen
        // by the generator, so the hard toon decision is unambiguous.
        let spec = if lcg(&mut state).is_multiple_of(2) {
            (threshold + 0.1).min(1.0)
        } else {
            (threshold - 0.1).max(0.0)
        };

        let foam_width = if lcg(&mut state).is_multiple_of(5) {
            0.0
        } else {
            uniform(lcg(&mut state), 0.1, 5.0)
        };
        let dist = uniform(lcg(&mut state), 0.0, 6.0);

        let caustic = uniform(lcg(&mut state), -0.3, 1.3);
        let flow_gain = uniform(lcg(&mut state), 0.0, 2.0);
        let flow_speed = uniform(lcg(&mut state), 0.0, 3.0);
        let scale = uniform(lcg(&mut state), 1.0, 16.0);

        queries.push(WaterShadingNprQuery::new(
            bands, threshold, foam_width, scale, flow_gain, r, g, b, spec, caustic, flow_speed,
            dist,
        ));
    }
    check(&ctx, &gpu, &queries);
}
