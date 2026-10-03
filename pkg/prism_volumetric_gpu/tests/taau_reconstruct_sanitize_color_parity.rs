//! Real-device parity for the per-channel color sanitizer twin:
//! [`GpuTaauReconstructSanitizeColor`](prism_volumetric_gpu::taau_reconstruct_sanitize_color::GpuTaauReconstructSanitizeColor)
//! must reproduce the `CPU` golden `sanitize_color` from
//! [`reconstruct`](prism_render_architecture::temporal_upscale::reconstruct)
//! across pass-through positives, mixed-sign negatives that clamp to `0`,
//! non-finite channels (`NaN`, `+/- inf`) that fold to `0`, and a randomized
//! finite sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden `sanitize_color` is a private helper, but it is reached
//! deterministically through the public entry point
//! [`resolve`](prism_render_architecture::temporal_upscale::reconstruct::resolve):
//! on the reset path (here forced with `disoccluded = true` and an empty
//! neighborhood) `resolve` returns `sanitize_color(current_color)` as its output
//! color. The oracle therefore calls `resolve` and reads `.color`, grounding the
//! transitive `GPU == resolve == sanitize_color` argument through a public call.
//!
//! # Parity criterion
//!
//! The kernel performs only a magnitude comparison, a `max`, and a select, so a
//! correct port reproduces the reference branch with zero difference, including
//! the non-finite inputs that both sides fold to a finite `0`. Every continuous
//! output is still asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::temporal_upscale::reconstruct`；无第三方引擎源码或衍生代码。

use prism_render_architecture::temporal_upscale::reconstruct::{resolve, ResolveParams};
use prism_volumetric_gpu::taau_reconstruct_sanitize_color::{
    GpuTaauReconstructSanitizeColor, TaauReconstructSanitizeColorQuery,
    TaauReconstructSanitizeColorResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. The sanitizer only compares, clamps, and selects, so a
/// correct port lands exactly on the reference; `1e-4` leaves legal slack.
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

/// Reconstructs the golden result in-host: `resolve` on the forced reset path
/// (`disoccluded = true`, empty neighborhood) returns `sanitize_color(color)` as
/// its output color.
fn oracle(q: &TaauReconstructSanitizeColorQuery) -> TaauReconstructSanitizeColorResult {
    let out = resolve(ResolveParams::default(), q.color, &[], [0.0; 3], 1.0, true);
    TaauReconstructSanitizeColorResult { color: out.color }
}

/// Pins one `GPU` result against the in-host oracle: every channel within
/// tolerance.
fn check_query(
    idx: usize,
    got: &TaauReconstructSanitizeColorResult,
    want: &TaauReconstructSanitizeColorResult,
) {
    for c in 0..3 {
        assert!(
            close(got.color[c], want.color[c]),
            "query {idx} channel {c}: gpu {} vs cpu {}",
            got.color[c],
            want.color[c]
        );
    }
}

/// Dispatches `queries` and checks every result against the oracle.
fn check(
    ctx: &GpuContext,
    gpu: &GpuTaauReconstructSanitizeColor,
    queries: &[TaauReconstructSanitizeColorQuery],
) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
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

/// Maps `32` random bits to a finite channel in roughly `[-100, 1000]`, a span
/// that straddles zero so the negative clamp is exercised far from any boundary.
fn finite_channel(bits: u32) -> f32 {
    let unit = (bits as f32) / (u32::MAX as f32);
    unit * 1100.0 - 100.0
}

#[test]
fn empty_batch_produces_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauReconstructSanitizeColor::new(&ctx);
    // An empty batch short-circuits on the host (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn positive_colors_pass_through() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauReconstructSanitizeColor::new(&ctx);
    // Every channel is finite and non-negative, so the sanitizer is the
    // identity: nothing is clamped or folded.
    let queries = [
        TaauReconstructSanitizeColorQuery::new([0.25, 0.5, 0.75]),
        TaauReconstructSanitizeColorQuery::new([1.0, 2.0, 4.0]),
        TaauReconstructSanitizeColorQuery::new([12.5, 100.0, 850.0]),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn negative_channels_clamp_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauReconstructSanitizeColor::new(&ctx);
    // Finite negatives clamp to 0; the positive channels pass through unchanged,
    // so each query mixes both branches of the per-channel select.
    let queries = [
        TaauReconstructSanitizeColorQuery::new([-1.0, 2.0, -3.0]),
        TaauReconstructSanitizeColorQuery::new([-0.5, -10.0, 5.0]),
        TaauReconstructSanitizeColorQuery::new([-250.0, -0.001, -999.0]),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn non_finite_channels_fold_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauReconstructSanitizeColor::new(&ctx);
    // NaN and both infinities fold to a finite 0 on both sides, so the parity
    // comparison is 0 vs 0 and stays within tolerance. Finite neighbors in the
    // same query confirm only the non-finite lane is rewritten.
    let queries = [
        TaauReconstructSanitizeColorQuery::new([f32::NAN, 1.0, 2.0]),
        TaauReconstructSanitizeColorQuery::new([3.0, f32::INFINITY, 4.0]),
        TaauReconstructSanitizeColorQuery::new([5.0, 6.0, f32::NEG_INFINITY]),
        TaauReconstructSanitizeColorQuery::new([f32::NAN, f32::INFINITY, f32::NEG_INFINITY]),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_finite_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauReconstructSanitizeColor::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    let mut queries: Vec<TaauReconstructSanitizeColorQuery> = Vec::new();
    // Several workgroups' worth of random finite colors straddling zero pin both
    // the pass-through and the negative-clamp branch across a wide span.
    for _ in 0..512 {
        let r = finite_channel(lcg(&mut state));
        let g = finite_channel(lcg(&mut state));
        let b = finite_channel(lcg(&mut state));
        queries.push(TaauReconstructSanitizeColorQuery::new([r, g, b]));
    }
    check(&ctx, &gpu, &queries);
}
