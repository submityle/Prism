//! Real-device parity for the `chromatic`-aberration twin:
//! [`GpuChromaticAberration`](prism_volumetric_gpu::chromatic_aberration::GpuChromaticAberration)
//! must reproduce the `CPU` golden
//! [`chromatic_aberration`](prism_render_architecture::particle::chromatic_aberration)
//! across the optical center (every channel offset vanishes), a mid-field `UV`
//! (all three channels split cleanly inside the frame), several strength and
//! falloff settings, a strong-strength `UV` whose raw samples blow past both
//! frame edges (so both devices clamp to exactly `0` or `1`), a randomized batch
//! and a larger multi-workgroup sweep, all pinned element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each source `UV` is a fixed, non-reorderable sequence of multiplies, adds,
//! one divide, one `sqrt` and the bounded `pow_u32` loop, so `CPU` and `GPU`
//! evaluate the same closed form in the same order. They are not bit-exact: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few units in the last place. The
//! comparison therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on every
//! `f32` field.
//!
//! # Conditioning
//!
//! The reference clamps every sampled `UV` into `[0, 1]`, so a `UV` whose raw
//! sample lands near a frame edge is a soft branch where a few units in the last
//! place could flip whether a device clamps. The random fixtures reject any
//! `UV` whose raw per-channel sample is not comfortably inside `[0.05, 0.95]`,
//! so both devices stay off the clamp seam. The dedicated clamp fixture does the
//! opposite on purpose: it drives the raw samples far past both edges, so both
//! devices clamp to exactly `0` or `1` and agree bit for bit there. The
//! radial-intensity denominator `1 + r^2` is at least `1`, so no fixture can
//! divide by a near-zero denominator.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::chromatic_aberration`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use prism_render_architecture::particle::chromatic_aberration::ChromaParams;
use prism_volumetric_gpu::chromatic_aberration::{
    golden, ChromaticAberrationQuery, ChromaticAberrationResult, GpuChromaticAberration,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
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

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// A pseudo-random value in `[lo, hi)` drawn from `state`.
fn rand_in(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * lcg(state)
}

/// Draws a clearly-conditioned parameter set from `state`. The ranges keep the
/// `strength`, falloff and per-channel gains in the physically useful band and
/// the optical `center` near the middle of the frame, so the random `UV`s stay
/// well inside the unit square after their split offsets.
fn rand_params(state: &mut u64) -> ChromaParams {
    let strength = rand_in(state, 0.3, 1.2);
    let radial_falloff_k = rand_in(state, 0.5, 3.5);
    let r_scale = rand_in(state, 0.3, 1.0);
    let g_scale = rand_in(state, -0.2, 0.2);
    let b_scale = rand_in(state, 0.3, 1.0);
    let center = [rand_in(state, 0.4, 0.6), rand_in(state, 0.4, 0.6)];
    ChromaParams::new(
        strength,
        radial_falloff_k,
        r_scale,
        g_scale,
        b_scale,
        center,
    )
}

/// Builds a clearly-conditioned source `UV` by rejection sampling against
/// `params`: a `UV` is accepted only when every raw per-channel sample
/// (`uv + channel_offset`, before the reference clamp) lands comfortably inside
/// `[0.05, 0.95]`, so both devices skip the clamp branch and the continuous
/// tolerance applies cleanly.
fn rand_query(state: &mut u64, params: &ChromaParams) -> ChromaticAberrationQuery {
    loop {
        let uv = [rand_in(state, 0.2, 0.8), rand_in(state, 0.2, 0.8)];
        let offsets = params.channel_offsets(uv);
        let inside = offsets.iter().all(|offset| {
            let x = uv[0] + offset[0];
            let y = uv[1] + offset[1];
            (0.05..=0.95).contains(&x) && (0.05..=0.95).contains(&y)
        });
        if inside {
            return ChromaticAberrationQuery::new(uv);
        }
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: all three
/// per-channel sampled `UV`s (six scalars) and the shared radial intensity must
/// agree within bound.
fn pin(
    idx: usize,
    params: &ChromaParams,
    query: &ChromaticAberrationQuery,
    got: &ChromaticAberrationResult,
) {
    let want = golden(params, query);
    for (channel, (g, w)) in got
        .sample_uvs
        .iter()
        .zip(want.sample_uvs.iter())
        .enumerate()
    {
        assert!(
            close(g[0], w[0]),
            "query {idx} channel {channel} u: gpu {} vs cpu {}",
            g[0],
            w[0]
        );
        assert!(
            close(g[1], w[1]),
            "query {idx} channel {channel} v: gpu {} vs cpu {}",
            g[1],
            w[1]
        );
    }
    assert!(
        close(got.radial_intensity, want.radial_intensity),
        "query {idx} radial_intensity: gpu {} vs cpu {}",
        got.radial_intensity,
        want.radial_intensity
    );
}

/// Dispatches `queries` under `params` and pins every result against the golden.
fn check(
    ctx: &GpuContext,
    gpu: &GpuChromaticAberration,
    params: &ChromaParams,
    queries: &[ChromaticAberrationQuery],
) {
    let got = gpu.evaluate(ctx, params, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, params, query, result);
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuChromaticAberration::new(&ctx);
    let params = ChromaParams::new(1.0, 2.0, 0.6, 0.0, 0.4, [0.5, 0.5]);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &params, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn optical_center_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuChromaticAberration::new(&ctx);
    // At the optical center the radial vector is zero, so the radius, every
    // channel offset and the radial intensity all vanish and each sampled UV is
    // the center itself. Both devices reproduce it exactly.
    let params = ChromaParams::new(1.0, 2.0, 0.6, 0.0, 0.4, [0.5, 0.5]);
    let query = ChromaticAberrationQuery::new([0.5, 0.5]);
    check(&ctx, &gpu, &params, &[query]);
}

#[test]
fn mid_field_split_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuChromaticAberration::new(&ctx);
    // A mid-field UV with a modest strength: all three channels split cleanly
    // inside the frame, so no clamp fires and the continuous tolerance governs.
    let params = ChromaParams::new(0.8, 2.0, 0.7, 0.1, 0.6, [0.5, 0.5]);
    let query = ChromaticAberrationQuery::new([0.65, 0.6]);
    check(&ctx, &gpu, &params, &[query]);
}

#[test]
fn varied_strength_and_falloff_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuChromaticAberration::new(&ctx);
    // Sweep a handful of strength / falloff / center settings against one
    // mid-field UV, exercising the radial-intensity shaper and the per-channel
    // gains across their useful band.
    let cases = [
        ChromaParams::new(0.4, 0.5, 0.5, 0.0, 0.5, [0.5, 0.5]),
        ChromaParams::new(1.0, 3.0, 0.9, -0.15, 0.8, [0.45, 0.55]),
        ChromaParams::new(0.7, 1.5, 0.6, 0.2, 0.4, [0.55, 0.5]),
    ];
    let query = ChromaticAberrationQuery::new([0.62, 0.58]);
    for params in cases {
        check(&ctx, &gpu, &params, &[query]);
    }
}

#[test]
fn strong_strength_clamps_at_both_edges() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuChromaticAberration::new(&ctx);
    // A very large strength drives the raw red sample far beyond the +edge and
    // the raw blue sample far below the -edge, so both devices clamp to exactly
    // 1 and 0 respectively and agree bit for bit at the saturated bounds.
    let params = ChromaParams::new(50.0, 4.0, 3.0, 1.0, 2.5, [0.5, 0.5]);
    let queries = [
        ChromaticAberrationQuery::new([0.9, 0.5]),
        ChromaticAberrationQuery::new([0.1, 0.5]),
        ChromaticAberrationQuery::new([0.85, 0.2]),
    ];
    check(&ctx, &gpu, &params, &queries);
}

#[test]
fn random_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuChromaticAberration::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing deterministic fixtures with many rejection-sampled random
    // UVs under one shared parameter set, dispatched together so the per-thread
    // indexing and the contiguous storage layout are both exercised.
    let params = rand_params(&mut state);
    let mut queries = vec![
        ChromaticAberrationQuery::new(params.center),
        ChromaticAberrationQuery::new([0.6, 0.55]),
    ];
    for _ in 0..48 {
        queries.push(rand_query(&mut state, &params));
    }
    check(&ctx, &gpu, &params, &queries);
}

#[test]
fn many_queries_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuChromaticAberration::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) under fresh random parameters
    // pins the full split and the radial intensity across many random UVs.
    let params = rand_params(&mut state);
    let queries: Vec<ChromaticAberrationQuery> =
        (0..200).map(|_| rand_query(&mut state, &params)).collect();
    check(&ctx, &gpu, &params, &queries);
}
