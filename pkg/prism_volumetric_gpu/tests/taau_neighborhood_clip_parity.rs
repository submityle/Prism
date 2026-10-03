//! Real-device parity for the temporal-upscale neighborhood history-clip twin:
//! [`GpuTaauNeighborhoodClip`](prism_volumetric_gpu::taau_neighborhood_clip::GpuTaauNeighborhoodClip)
//! must reproduce the numeric core of the `CPU` golden
//! [`neighborhood`](prism_render_architecture::temporal_upscale::neighborhood) —
//! the per-pixel standard deviation, Marco Salvi variance box, and Karis
//! clip-toward-center history — across an interior point, an exterior point, a
//! `gamma = 0` clip-to-mean, a general non-degenerate box, and a randomized
//! window-plus-history sweep compared pixel-for-pixel.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden builds a
//! [`NeighborhoodStats`](prism_render_architecture::temporal_upscale::neighborhood::NeighborhoodStats)
//! from a window with
//! [`from_samples`](prism_render_architecture::temporal_upscale::neighborhood::NeighborhoodStats::from_samples),
//! then its public
//! [`variance_aabb`](prism_render_architecture::temporal_upscale::neighborhood::NeighborhoodStats::variance_aabb)
//! and
//! [`clip_to_aabb`](prism_render_architecture::temporal_upscale::neighborhood::clip_to_aabb)
//! are the oracle. The `m2` moment is private, so the test recomputes it from
//! the same window with the golden's formula (`sum(s * s) / n`) using only
//! `+ * /`, and feeds `mean`/`m2`/`min`/`max` into the `GPU` query so the kernel
//! derives the same `stddev`, box, and clipped history.
//!
//! # Parity criterion
//!
//! Every output (`lo`, `hi`, `clipped`) is continuous, threaded through a
//! `sqrt` and a reciprocal, so a `GPU` evaluation may land a few units in the
//! last place from the scalar reference; each is asserted within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Conditioning
//!
//! Every fixture uses a non-degenerate box (every channel has a strictly
//! positive extent) and keeps the history clearly inside or clearly outside the
//! box, away from the `t = 1` surface tie, so the `CPU` and `GPU` stay on the
//! same side of the keep-or-clip decision.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::temporal_upscale::neighborhood`；无第三方引擎源码或衍生代码。

use prism_render_architecture::temporal_upscale::neighborhood::{clip_to_aabb, NeighborhoodStats};
use prism_volumetric_gpu::taau_neighborhood_clip::{
    GpuTaauNeighborhoodClip, TaauNeighborhoodClipQuery, TaauNeighborhoodClipResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a continuous output. A `GPU` `sqrt`/reciprocal may
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

/// Recomputes the window mean of squares with the golden's formula
/// (`sum(s * s) / n`), since the `m2` moment has no public getter. Uses only
/// `+ * /`, never a transcendental method.
fn recompute_m2(window: &[[f32; 3]]) -> [f32; 3] {
    let mut sum_sq = [0.0f32; 3];
    for s in window {
        for c in 0..3 {
            sum_sq[c] += s[c] * s[c];
        }
    }
    let inv_n = 1.0 / window.len() as f32;
    [sum_sq[0] * inv_n, sum_sq[1] * inv_n, sum_sq[2] * inv_n]
}

/// Builds the `GPU` query and the golden oracle for a window, `gamma`, and a
/// history point, then returns both for a pixel-for-pixel comparison.
fn build(
    window: &[[f32; 3]],
    gamma: f32,
    history: [f32; 3],
) -> (TaauNeighborhoodClipQuery, TaauNeighborhoodClipResult) {
    let stats = NeighborhoodStats::from_samples(window);
    let mean = stats.mean();
    let hard_min = stats.min();
    let hard_max = stats.max();
    let m2 = recompute_m2(window);
    let (lo, hi) = stats.variance_aabb(gamma);
    let clipped = clip_to_aabb(lo, hi, history);
    let query = TaauNeighborhoodClipQuery {
        mean,
        m2,
        hard_min,
        hard_max,
        history_point: history,
        gamma,
    };
    let want = TaauNeighborhoodClipResult { lo, hi, clipped };
    (query, want)
}

/// Pins one `GPU` pixel result against the golden oracle: all three continuous
/// vectors within tolerance.
fn check_pixel(idx: usize, got: &TaauNeighborhoodClipResult, want: &TaauNeighborhoodClipResult) {
    for c in 0..3 {
        assert!(
            close(got.lo[c], want.lo[c]),
            "pixel {idx} lo[{c}]: gpu {} vs cpu {}",
            got.lo[c],
            want.lo[c]
        );
        assert!(
            close(got.hi[c], want.hi[c]),
            "pixel {idx} hi[{c}]: gpu {} vs cpu {}",
            got.hi[c],
            want.hi[c]
        );
        assert!(
            close(got.clipped[c], want.clipped[c]),
            "pixel {idx} clipped[{c}]: gpu {} vs cpu {}",
            got.clipped[c],
            want.clipped[c]
        );
    }
}

/// Dispatches every query and pins each result against its paired oracle.
fn check(
    ctx: &GpuContext,
    gpu: &GpuTaauNeighborhoodClip,
    cases: &[(TaauNeighborhoodClipQuery, TaauNeighborhoodClipResult)],
) {
    let queries: Vec<TaauNeighborhoodClipQuery> = cases.iter().map(|(q, _)| *q).collect();
    let got = gpu.evaluate(ctx, &queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (result, (_, want))) in got.iter().zip(cases.iter()).enumerate() {
        check_pixel(idx, result, want);
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

/// Draws an `f32` in `[lo, hi]` at milli resolution from `state`, never using a
/// transcendental method.
fn ranged(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let span = hi - lo;
    lo + span * (lcg(state) % 1001) as f32 / 1000.0
}

/// A deterministic, well-conditioned sample window: three channels each with a
/// strictly positive spread so every box extent is non-degenerate.
const WINDOW: [[f32; 3]; 5] = [
    [0.20, 0.52, 0.90],
    [0.34, 0.41, 0.76],
    [0.27, 0.48, 0.83],
    [0.31, 0.45, 0.88],
    [0.24, 0.50, 0.79],
];

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping taau_neighborhood_clip parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTaauNeighborhoodClip::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn interior_point_is_left_unchanged() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauNeighborhoodClip::new(&ctx);
    // The window mean is strictly inside the variance box, so the history sits
    // inside and clips to itself.
    let stats = NeighborhoodStats::from_samples(&WINDOW);
    let history = stats.mean();
    let case = build(&WINDOW, 1.0, history);
    // The oracle clip must be the untouched history for this interior fixture.
    assert!(
        close(case.1.clipped[0], history[0])
            && close(case.1.clipped[1], history[1])
            && close(case.1.clipped[2], history[2]),
        "interior fixture must clip to itself"
    );
    check(&ctx, &gpu, &[case]);
}

#[test]
fn exterior_point_is_projected_onto_the_box() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauNeighborhoodClip::new(&ctx);
    // A history far outside the box on channel 0: it clips back onto the box
    // surface toward the center.
    let history = [5.0, 0.47, 0.84];
    let case = build(&WINDOW, 1.0, history);
    let (lo, hi) = (case.1.lo, case.1.hi);
    // The oracle result lands inside the box on every axis (within tolerance).
    for c in 0..3 {
        assert!(
            case.1.clipped[c] >= lo[c] - EPS && case.1.clipped[c] <= hi[c] + EPS,
            "exterior fixture must clip inside the box on axis {c}"
        );
    }
    check(&ctx, &gpu, &[case]);
}

#[test]
fn zero_gamma_clips_to_the_mean() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauNeighborhoodClip::new(&ctx);
    // gamma = 0 collapses the variance box to the single mean point, so any
    // off-mean history clips straight to the mean (the degenerate-box path).
    let stats = NeighborhoodStats::from_samples(&WINDOW);
    let mean = stats.mean();
    let history = [mean[0] + 0.3, mean[1] - 0.25, mean[2] + 0.4];
    let case = build(&WINDOW, 0.0, history);
    assert!(
        close(case.1.clipped[0], mean[0])
            && close(case.1.clipped[1], mean[1])
            && close(case.1.clipped[2], mean[2]),
        "gamma = 0 fixture must clip to the mean"
    );
    check(&ctx, &gpu, &[case]);
}

#[test]
fn non_degenerate_box_general_case() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauNeighborhoodClip::new(&ctx);
    // A different window and a history outside on two axes but inside on one, so
    // the clip exercises a multi-axis overshoot.
    let window = [
        [0.10, 0.90, 0.30],
        [0.40, 0.60, 0.55],
        [0.25, 0.75, 0.42],
        [0.33, 0.68, 0.48],
    ];
    let history = [1.2, 0.70, -0.5];
    let case = build(&window, 1.25, history);
    check(&ctx, &gpu, &[case]);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauNeighborhoodClip::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    let mut cases = Vec::new();
    // Several workgroups' worth of pixels: random non-degenerate windows and
    // random histories pin the kernel across a wide span of inputs.
    for _ in 0..200 {
        // Build a window of samples around a per-channel base, each channel
        // forced to a strictly positive spread so the box never degenerates.
        let base = [
            ranged(&mut state, 0.1, 0.8),
            ranged(&mut state, 0.1, 0.8),
            ranged(&mut state, 0.1, 0.8),
        ];
        let mut window = [[0.0f32; 3]; 6];
        for sample in &mut window {
            for c in 0..3 {
                // Offsets span a full +/- 0.15 band, guaranteeing extent > 0.
                sample[c] = base[c] + ranged(&mut state, -0.15, 0.15);
            }
        }
        let gamma = ranged(&mut state, 0.5, 2.0);
        // History drawn from a wide band so some channels fall well outside the
        // box and some well inside, away from the t = 1 surface tie.
        let history = [
            ranged(&mut state, -0.5, 1.5),
            ranged(&mut state, -0.5, 1.5),
            ranged(&mut state, -0.5, 1.5),
        ];
        cases.push(build(&window, gamma, history));
    }
    check(&ctx, &gpu, &cases);
}
