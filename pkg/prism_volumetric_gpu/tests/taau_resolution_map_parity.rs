//! Real-device parity for the temporal-upscale render <-> display pixel-mapping
//! twin: [`GpuTaauResolutionMap`](prism_volumetric_gpu::taau_resolution_map::GpuTaauResolutionMap)
//! must reproduce the per-coordinate affine maps of the `CPU` golden
//! [`UpscaleResolution`](prism_render_architecture::temporal_upscale::resolution::UpscaleResolution)
//! — [`render_to_display`](prism_render_architecture::temporal_upscale::resolution::UpscaleResolution::render_to_display),
//! [`display_to_render`](prism_render_architecture::temporal_upscale::resolution::UpscaleResolution::display_to_render),
//! [`jitter_to_clip`](prism_render_architecture::temporal_upscale::resolution::UpscaleResolution::jitter_to_clip),
//! [`effective_scale_x`](prism_render_architecture::temporal_upscale::resolution::UpscaleResolution::effective_scale_x)
//! and [`effective_scale_y`](prism_render_architecture::temporal_upscale::resolution::UpscaleResolution::effective_scale_y)
//! — across fixed fixtures, a mixed batch and a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden methods are public, so each expected value is produced by
//! constructing the reference
//! [`UpscaleResolution`](prism_render_architecture::temporal_upscale::resolution::UpscaleResolution)
//! with [`from_display`](prism_render_architecture::temporal_upscale::resolution::UpscaleResolution::from_display)
//! (the host-side integer derivation this twin does not port) and calling the
//! very same affine methods the device reproduces. The query carries the
//! already-derived integer dimensions read back through the golden's public
//! accessors, so the `GPU` and the oracle share identical inputs.
//!
//! # Parity criterion
//!
//! Every output is a finite affine combination (a divide, an add and a
//! multiply) of positive dimensions and the input coordinate, so there is no
//! discrete branch to flip: a `GPU` divide may land a few units in the last
//! place from the scalar reference, so each of the eight components is asserted
//! within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, never an `f32` `==`.
//!
//! # Conditioning
//!
//! No fixture needs tie-avoidance (the maps are branch-free), so the sweep only
//! keeps the dimensions and coordinates at sane magnitudes — display sizes in a
//! realistic pixel range and coordinates near the visible extent — so the
//! relative tolerance is never dominated by catastrophic cancellation.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::temporal_upscale::resolution`；无第三方引擎源码或衍生代码。

use prism_render_architecture::temporal_upscale::resolution::UpscaleResolution;
use prism_volumetric_gpu::taau_resolution_map::{
    GpuTaauResolutionMap, TaauResolutionMapQuery, TaauResolutionMapResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on one mapped component. A `GPU` divide may land a few
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

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws a value in `[0, 1)` at micro resolution from `state`.
fn unit(state: &mut u64) -> f32 {
    (lcg(state) % 1_000_000) as f32 / 1_000_000.0
}

/// Draws a coordinate in `[-10, span + 10]` so the sweep covers off-screen
/// samples as well as the visible extent.
fn coord(state: &mut u64, span: f32) -> f32 {
    -10.0 + unit(state) * (span + 20.0)
}

/// Draws a sub-pixel jitter offset in `[-2, 2]` render pixels.
fn jitter(state: &mut u64) -> f32 {
    -2.0 + unit(state) * 4.0
}

/// Builds a query from a reference resolution and the coordinate triple, reading
/// the integer dimensions back through the golden's public accessors so the
/// `GPU` and the oracle share identical inputs.
fn query_from(
    res: &UpscaleResolution,
    render_x: f32,
    render_y: f32,
    display_x: f32,
    display_y: f32,
    jitter_x: f32,
    jitter_y: f32,
) -> TaauResolutionMapQuery {
    TaauResolutionMapQuery::new(
        res.render_width(),
        res.render_height(),
        res.display_width(),
        res.display_height(),
        render_x,
        render_y,
        display_x,
        display_y,
        jitter_x,
        jitter_y,
    )
}

/// Pins one `GPU` result against the golden `UpscaleResolution`, asserting every
/// one of the eight affine components within tolerance.
fn check_sample(
    idx: usize,
    res: &UpscaleResolution,
    q: &TaauResolutionMapQuery,
    got: &TaauResolutionMapResult,
) {
    let rd = res.render_to_display(q.render_x, q.render_y);
    let dr = res.display_to_render(q.display_x, q.display_y);
    let jc = res.jitter_to_clip(q.jitter_x, q.jitter_y);
    let ex = res.effective_scale_x();
    let ey = res.effective_scale_y();

    assert!(
        close(got.render_to_display[0], rd[0]),
        "sample {idx} render_to_display.x: gpu {} vs cpu {}",
        got.render_to_display[0],
        rd[0]
    );
    assert!(
        close(got.render_to_display[1], rd[1]),
        "sample {idx} render_to_display.y: gpu {} vs cpu {}",
        got.render_to_display[1],
        rd[1]
    );
    assert!(
        close(got.display_to_render[0], dr[0]),
        "sample {idx} display_to_render.x: gpu {} vs cpu {}",
        got.display_to_render[0],
        dr[0]
    );
    assert!(
        close(got.display_to_render[1], dr[1]),
        "sample {idx} display_to_render.y: gpu {} vs cpu {}",
        got.display_to_render[1],
        dr[1]
    );
    assert!(
        close(got.jitter_to_clip[0], jc[0]),
        "sample {idx} jitter_to_clip.x: gpu {} vs cpu {}",
        got.jitter_to_clip[0],
        jc[0]
    );
    assert!(
        close(got.jitter_to_clip[1], jc[1]),
        "sample {idx} jitter_to_clip.y: gpu {} vs cpu {}",
        got.jitter_to_clip[1],
        jc[1]
    );
    assert!(
        close(got.effective_scale_x, ex),
        "sample {idx} effective_scale_x: gpu {} vs cpu {}",
        got.effective_scale_x,
        ex
    );
    assert!(
        close(got.effective_scale_y, ey),
        "sample {idx} effective_scale_y: gpu {} vs cpu {}",
        got.effective_scale_y,
        ey
    );
}

/// Dispatches every sample and pins each result against the golden resolution.
fn check(
    ctx: &GpuContext,
    gpu: &GpuTaauResolutionMap,
    samples: &[(UpscaleResolution, TaauResolutionMapQuery)],
) {
    let queries: Vec<TaauResolutionMapQuery> = samples.iter().map(|(_, q)| *q).collect();
    let got = gpu.evaluate(ctx, &queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, ((res, q), result)) in samples.iter().zip(got.iter()).enumerate() {
        check_sample(idx, res, q, result);
    }
}

/// The deterministic resolution fixtures: common display sizes paired with a
/// range of render scales, including native, half, and odd aspect ratios.
const FIXTURE_RESOLUTIONS: [(u32, u32, f32); 6] = [
    (1920, 1080, 0.5),
    (1280, 720, 1.0),
    (1000, 500, 0.5),
    (2560, 1440, 0.667),
    (1366, 768, 0.75),
    (3840, 2160, 0.333),
];

/// A few coordinate triples per resolution, each a divide/add/multiply away from
/// the mapped output.
const FIXTURE_COORDS: [(f32, f32, f32, f32, f32, f32); 4] = [
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (100.0, 200.0, 300.0, 150.0, 0.5, -0.25),
    (12.5, 7.25, 640.0, 360.0, -1.5, 2.0),
    (-5.0, 3.0, 1920.5, 1080.5, 1.0, -1.0),
];

/// Builds the deterministic fixture samples: every coordinate triple against
/// every fixture resolution.
fn fixture_samples() -> Vec<(UpscaleResolution, TaauResolutionMapQuery)> {
    let mut out = Vec::new();
    for &(dw, dh, scale) in &FIXTURE_RESOLUTIONS {
        let res = UpscaleResolution::from_display(dw, dh, scale);
        for &(rx, ry, dx, dy, jx, jy) in &FIXTURE_COORDS {
            out.push((res, query_from(&res, rx, ry, dx, dy, jx, jy)));
        }
    }
    out
}

#[test]
fn empty_batch_dispatches_nothing() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauResolutionMap::new(&ctx);
    // An empty batch short-circuits on the host (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn native_scale_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauResolutionMap::new(&ctx);
    let res = UpscaleResolution::from_display(1280, 720, 1.0);
    let samples: Vec<_> = FIXTURE_COORDS
        .iter()
        .map(|&(rx, ry, dx, dy, jx, jy)| (res, query_from(&res, rx, ry, dx, dy, jx, jy)))
        .collect();
    check(&ctx, &gpu, &samples);
}

#[test]
fn half_scale_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauResolutionMap::new(&ctx);
    let res = UpscaleResolution::from_display(1920, 1080, 0.5);
    let samples: Vec<_> = FIXTURE_COORDS
        .iter()
        .map(|&(rx, ry, dx, dy, jx, jy)| (res, query_from(&res, rx, ry, dx, dy, jx, jy)))
        .collect();
    check(&ctx, &gpu, &samples);
}

#[test]
fn mixed_batch_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauResolutionMap::new(&ctx);
    // Every fixture resolution dispatched together so the per-thread indexing and
    // the contiguous output slots are both exercised.
    check(&ctx, &gpu, &fixture_samples());
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauResolutionMap::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    let mut samples = fixture_samples();
    // Many random resolutions (several workgroups' worth of samples) pin the maps
    // across a wide span of display sizes and render scales.
    for _ in 0..256 {
        // Display dimensions in a realistic pixel range.
        let dw = 16 + lcg(&mut state) % 3_825;
        let dh = 16 + lcg(&mut state) % 3_825;
        // Render scale in (0, 1]; `from_display` validates and clamps.
        let scale = 0.2 + unit(&mut state) * 0.8;
        let res = UpscaleResolution::from_display(dw, dh, scale);
        let rx = coord(&mut state, res.render_width() as f32);
        let ry = coord(&mut state, res.render_height() as f32);
        let dx = coord(&mut state, res.display_width() as f32);
        let dy = coord(&mut state, res.display_height() as f32);
        let jx = jitter(&mut state);
        let jy = jitter(&mut state);
        samples.push((res, query_from(&res, rx, ry, dx, dy, jx, jy)));
    }
    check(&ctx, &gpu, &samples);
}
