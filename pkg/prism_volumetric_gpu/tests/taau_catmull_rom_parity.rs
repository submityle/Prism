//! Real-device parity for the history-reprojection twin:
//! [`GpuTaauCatmullRom`](prism_volumetric_gpu::taau_catmull_rom::GpuTaauCatmullRom)
//! must reproduce the numeric core of the `CPU` golden
//! [`reproject`](prism_render_architecture::temporal_upscale::reproject) — the
//! `catmull_rom_weights`, `reproject_pixel`, `on_screen` and `depth_disoccluded`
//! quantities — across the Catmull-Rom endpoints, the screen-bound edges, the
//! depth-tolerance disable and boundary cases, and a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The four golden functions are public, so each `GPU` field is pinned directly
//! against a host call:
//! [`catmull_rom_weights`](prism_render_architecture::temporal_upscale::reproject::catmull_rom_weights)`(t)`,
//! [`reproject_pixel`](prism_render_architecture::temporal_upscale::reproject::reproject_pixel),
//! [`on_screen`](prism_render_architecture::temporal_upscale::reproject::on_screen)
//! of the reprojected location, and
//! [`depth_disoccluded`](prism_render_architecture::temporal_upscale::reproject::depth_disoccluded).
//!
//! # Parity criterion
//!
//! The `weights` taps and the `reprojected` location are continuous `f32`
//! quantities and are asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//! `on_screen` and `depth_disoccluded` are discrete and are asserted exactly
//! with `==`.
//!
//! # Conditioning
//!
//! The randomized sweep rejects any sample whose reprojected location lands
//! within a margin of a screen edge, or whose relative depth lands within a
//! margin of the tolerance, so a last-place difference in the shared `f32`
//! arithmetic can never flip a discrete flag and the `==` assertion stays
//! faithful. The deliberate boundary fixtures use exact integer coordinates so
//! their flags are unambiguous.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::temporal_upscale::reproject`；无第三方引擎源码或衍生代码。

use prism_render_architecture::temporal_upscale::reproject::{
    catmull_rom_weights, depth_disoccluded, on_screen, reproject_pixel,
};
use prism_volumetric_gpu::taau_catmull_rom::{
    GpuTaauCatmullRom, TaauCatmullRomQuery, TaauCatmullRomResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a continuous output. A `GPU` arithmetic sequence may
/// land a few units in the last place from the scalar reference; `1e-4` admits
/// that legal slack while still failing a wrong port.
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

/// The expected result computed in-host from the four golden functions: the
/// faithful oracle the `GPU` is pinned against.
fn oracle(q: &TaauCatmullRomQuery) -> TaauCatmullRomResult {
    let reprojected = reproject_pixel(q.x, q.y, q.motion);
    TaauCatmullRomResult {
        weights: catmull_rom_weights(q.t),
        reprojected,
        on_screen: on_screen(reprojected, q.width, q.height),
        depth_disoccluded: depth_disoccluded(q.current_depth, q.history_depth, q.tolerance),
    }
}

/// Pins one `GPU` result against the in-host oracle: the continuous fields
/// within tolerance, the discrete flags exactly.
fn check_sample(idx: usize, got: &TaauCatmullRomResult, want: &TaauCatmullRomResult) {
    for tap in 0..4 {
        assert!(
            close(got.weights[tap], want.weights[tap]),
            "sample {idx} weight[{tap}]: gpu {} vs cpu {}",
            got.weights[tap],
            want.weights[tap]
        );
    }
    assert!(
        close(got.reprojected[0], want.reprojected[0]),
        "sample {idx} reprojected.x: gpu {} vs cpu {}",
        got.reprojected[0],
        want.reprojected[0]
    );
    assert!(
        close(got.reprojected[1], want.reprojected[1]),
        "sample {idx} reprojected.y: gpu {} vs cpu {}",
        got.reprojected[1],
        want.reprojected[1]
    );
    assert_eq!(
        got.on_screen, want.on_screen,
        "sample {idx} on_screen: gpu {} vs cpu {}",
        got.on_screen, want.on_screen
    );
    assert_eq!(
        got.depth_disoccluded, want.depth_disoccluded,
        "sample {idx} depth_disoccluded: gpu {} vs cpu {}",
        got.depth_disoccluded, want.depth_disoccluded
    );
}

/// Dispatches every sample and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuTaauCatmullRom, queries: &[TaauCatmullRomQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "one result per query must come back"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_sample(idx, result, &want);
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

/// Draws a fractional offset in `[0.0, 1.0]` at milli resolution from `state`.
fn frac(state: &mut u64) -> f32 {
    (lcg(state) % 1001) as f32 / 1000.0
}

/// Draws a coordinate in `[0.0, 320.0)` at centi resolution from `state`.
fn coord(state: &mut u64) -> f32 {
    (lcg(state) % 32_000) as f32 / 100.0
}

/// Draws a motion component in `[-16.0, 16.0)` at centi resolution from `state`.
fn motion_component(state: &mut u64) -> f32 {
    (lcg(state) % 3200) as f32 / 100.0 - 16.0
}

/// Draws a linear depth in `[0.0, 4.0)` at milli resolution from `state`.
fn depth(state: &mut u64) -> f32 {
    (lcg(state) % 4000) as f32 / 1000.0
}

/// Draws a positive tolerance in `[0.01, 0.5)` at milli resolution from `state`.
fn tolerance(state: &mut u64) -> f32 {
    (lcg(state) % 490) as f32 / 1000.0 + 0.01
}

/// Whether the reprojected location is clear of every screen edge by at least
/// `MARGIN`, so a last-place difference cannot flip the `on_screen` flag.
fn on_screen_well_conditioned(q: &TaauCatmullRomQuery) -> bool {
    const MARGIN: f32 = 0.05;
    let pos = reproject_pixel(q.x, q.y, q.motion);
    let w = q.width as f32;
    let h = q.height as f32;
    (pos[0] - 0.0).abs() > MARGIN
        && (pos[1] - 0.0).abs() > MARGIN
        && (pos[0] - w).abs() > MARGIN
        && (pos[1] - h).abs() > MARGIN
}

/// Whether the relative depth is clear of the tolerance by at least `MARGIN`, so
/// a last-place difference cannot flip the `depth_disoccluded` flag.
fn depth_well_conditioned(q: &TaauCatmullRomQuery) -> bool {
    const MARGIN: f32 = 0.01;
    let denom = q.current_depth.abs().max(q.history_depth.abs()).max(1.0e-6);
    let relative = (q.current_depth - q.history_depth).abs() / denom;
    (relative - q.tolerance).abs() > MARGIN
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping taau_catmull_rom parity: no wgpu adapter");
        return;
    };
    let gpu = GpuTaauCatmullRom::new(&ctx);
    // An empty batch must short-circuit on the host (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn catmull_rom_endpoints_interpolate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauCatmullRom::new(&ctx);
    // t = 0 collapses the kernel onto tap 1, t = 1 onto tap 2 (the interpolating
    // endpoints), and t = 0.5 the symmetric midpoint. On-screen reprojection and
    // a disabled depth test keep the discrete flags unambiguous.
    let base =
        |t: f32| TaauCatmullRomQuery::new(t, 100.0, 50.0, [2.0, -1.0], 256, 256, 0.5, 0.5, 0.0);
    check(&ctx, &gpu, &[base(0.0), base(0.5), base(1.0)]);
}

#[test]
fn reprojection_follows_motion() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauCatmullRom::new(&ctx);
    // Assorted motion vectors; all land well inside a 320x240 frame.
    let queries = [
        TaauCatmullRomQuery::new(0.25, 100.0, 50.0, [-4.0, 2.5], 320, 240, 0.5, 0.5, 0.0),
        TaauCatmullRomQuery::new(0.75, 10.0, 200.0, [12.0, -8.0], 320, 240, 0.5, 0.5, 0.0),
        TaauCatmullRomQuery::new(0.5, 160.0, 120.0, [0.0, 0.0], 320, 240, 0.5, 0.5, 0.0),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn on_screen_edges_and_zero_frame() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauCatmullRom::new(&ctx);
    // Exact integer coordinates with zero motion so the half-open bounds test is
    // unambiguous: (0,0) is inside, (9.5,9.5) is inside, a negative coordinate
    // and a coordinate at the width are off, and a zero-size frame is never on
    // screen.
    let queries = [
        TaauCatmullRomQuery::new(0.3, 0.0, 0.0, [0.0, 0.0], 10, 10, 0.5, 0.5, 0.0),
        TaauCatmullRomQuery::new(0.3, 9.5, 9.5, [0.0, 0.0], 10, 10, 0.5, 0.5, 0.0),
        TaauCatmullRomQuery::new(0.3, 5.0, 5.0, [-6.0, 0.0], 10, 10, 0.5, 0.5, 0.0),
        TaauCatmullRomQuery::new(0.3, 5.0, 5.0, [5.0, 0.0], 10, 10, 0.5, 0.5, 0.0),
        TaauCatmullRomQuery::new(0.3, 5.0, 5.0, [0.0, 0.0], 0, 10, 0.5, 0.5, 0.0),
        TaauCatmullRomQuery::new(0.3, 5.0, 5.0, [0.0, 0.0], 10, 0, 0.5, 0.5, 0.0),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn depth_disocclusion_cases() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauCatmullRom::new(&ctx);
    // A near-vs-far jump is disoccluded; matching depths within tolerance keep
    // history; a non-positive and a NaN tolerance both disable the test (never
    // disoccluded). All reprojections stay on screen so on_screen is a constant.
    let base = |cur: f32, hist: f32, tol: f32| {
        TaauCatmullRomQuery::new(0.4, 128.0, 128.0, [1.0, 1.0], 256, 256, cur, hist, tol)
    };
    let queries = [
        base(0.1, 0.9, 0.05),
        base(0.5, 0.505, 0.05),
        base(0.1, 0.9, 0.0),
        base(0.1, 0.9, f32::NAN),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauCatmullRom::new(&ctx);
    let mut state = 0x2718_2818_2845_9045_u64;
    let mut queries: Vec<TaauCatmullRomQuery> = Vec::new();
    // Several workgroups' worth of random samples, each kept clear of a screen
    // edge and the depth tolerance so the discrete flags cannot be flipped by a
    // last-place difference.
    let mut drawn = 0u32;
    while drawn < 256 {
        let q = TaauCatmullRomQuery::new(
            frac(&mut state),
            coord(&mut state),
            coord(&mut state),
            [motion_component(&mut state), motion_component(&mut state)],
            256,
            256,
            depth(&mut state),
            depth(&mut state),
            tolerance(&mut state),
        );
        if !on_screen_well_conditioned(&q) || !depth_well_conditioned(&q) {
            continue;
        }
        queries.push(q);
        drawn += 1;
    }
    check(&ctx, &gpu, &queries);
}
