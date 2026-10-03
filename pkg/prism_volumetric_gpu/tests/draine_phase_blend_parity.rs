//! Real-device parity for the `Draine` and `HG-Draine` scattering phase twin:
//! [`GpuDrainePhaseBlend`](prism_volumetric_gpu::draine_phase_blend::GpuDrainePhaseBlend)
//! must reproduce the `CPU` goldens
//! [`draine_phase`](prism_render_architecture::volumetric::scatter::draine_phase)
//! and
//! [`hg_draine_phase`](prism_render_architecture::volumetric::scatter::hg_draine_phase)
//! across the degenerate and limiting cases (`alpha = 0` collapsing `Draine` to
//! `HG`, `g = 0` isotropic `1 / (4*PI)`, forward and back scatter, `g` clamped
//! past the `±0.999` limit, `hg_weight` at `0` and `1`, and the near-collapse
//! denominator guard), a mixed batch resolved in one dispatch, and a randomized
//! sweep compared value-for-value.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The oracle is the public goldens
//! [`draine_phase`](prism_render_architecture::volumetric::scatter::draine_phase)
//! and
//! [`hg_draine_phase`](prism_render_architecture::volumetric::scatter::hg_draine_phase)
//! themselves: for each query the reference values are computed on the host from
//! the same arguments and the `GPU` is pinned against them.
//!
//! # Parity criterion
//!
//! The phases thread through only `+ - * /`, `clamp`, `max`, and `sqrt` with no
//! transcendental and no reorderable reduction, so the `CPU` and `GPU` evaluate
//! the same expression. A `GPU` may still fuse a multiply-add the scalar
//! reference leaves separate, and `sqrt` rounding can differ in the last place,
//! so both phase values are asserted within `abs_diff <= 1e-5` or
//! `rel_diff <= 1e-4`. No `f32` `==` is used anywhere.
//!
//! # Conditioning
//!
//! The clamps on `cos_theta`, `g`, `alpha`, and `hg_weight` are the only sharp
//! edges; the randomized sweep draws each argument a clear margin inside its
//! clamp interval so both paths clamp (or not) identically, keeping them on the
//! same algebraic branch. The `max(denom, EPS)` guard only ever engages for
//! `g` driven to the `±0.999` limit with a matching `cos_theta`, where
//! `1 + g^2 - 2*g*u` reaches its minimum `(1 - |g|)^2`. The `±0.999` clamp pins
//! that minimum to exactly `EPS = 1e-6`, so the guard is reachable only at that
//! single ill-conditioned boundary. There `1 + g^2 - 2*g*u` is the subtraction
//! of two quantities near `2.0`, carrying up to one `ULP` of `2.0` (`~2.4e-7`)
//! of catastrophic-cancellation error on a `~1e-6` result, and whether the
//! noisy value lands just above or below `EPS` differs between host and `GPU`.
//! That one fixture ([`near_collapse_denominator_guard`]) therefore checks the
//! guard's real contract (both phases stay finite and strictly positive, never
//! collapsing to `NaN`/`Inf`) and asserts agreement within a documented
//! worst-case band instead of the continuous-region `EPS`/`REL` bound; every
//! other fixture keeps both paths a clear margin off the floor and holds to the
//! tight bound.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::volumetric::scatter`；无第三方引擎源码或衍生代码。

use prism_render_architecture::volumetric::scatter::{draine_phase, hg_draine_phase};
use prism_volumetric_gpu::draine_phase_blend::{
    DrainePhaseBlendQuery, DrainePhaseBlendResult, GpuDrainePhaseBlend,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a continuous quantity. A `GPU` arithmetic pipeline
/// may land a few units in the last place from the scalar reference; `1e-5`
/// admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-5;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-4;

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

/// Draws an `f32` in `[lo, hi]` at ten-thousandth resolution from `state`.
fn draw(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (lcg(state) % 10_001) as f32 / 10_000.0 * (hi - lo)
}

/// Builds one query from the five phase arguments.
fn query(
    cos_theta: f32,
    g_hg: f32,
    g_draine: f32,
    alpha: f32,
    hg_weight: f32,
) -> DrainePhaseBlendQuery {
    DrainePhaseBlendQuery {
        cos_theta,
        g_hg,
        g_draine,
        alpha,
        hg_weight,
    }
}

/// Dispatches `queries` and asserts every phase pair matches the goldens.
fn check_batch(ctx: &GpuContext, gpu: &GpuDrainePhaseBlend, queries: &[DrainePhaseBlendQuery]) {
    let got: Vec<DrainePhaseBlendResult> = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(got.iter()) {
        let golden_draine = draine_phase(q.cos_theta, q.g_draine, q.alpha);
        let golden_blend = hg_draine_phase(q.cos_theta, q.g_hg, q.g_draine, q.alpha, q.hg_weight);
        assert!(
            close(r.draine, golden_draine),
            "draine mismatch: gpu={} golden={} (cos={}, g_draine={}, alpha={})",
            r.draine,
            golden_draine,
            q.cos_theta,
            q.g_draine,
            q.alpha
        );
        assert!(
            close(r.hg_draine, golden_blend),
            "hg_draine mismatch: gpu={} golden={} (cos={}, g_hg={}, g_draine={}, alpha={}, w={})",
            r.hg_draine,
            golden_blend,
            q.cos_theta,
            q.g_hg,
            q.g_draine,
            q.alpha,
            q.hg_weight
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_batch_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping draine_phase_blend parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuDrainePhaseBlend::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn alpha_zero_collapses_draine_to_hg() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDrainePhaseBlend::new(&ctx);
    // alpha == 0 reduces the Draine phase exactly to the HG phase shape at the
    // same g; the twin must reproduce that collapse.
    check_batch(&ctx, &gpu, &[query(0.3, 0.5, 0.5, 0.0, 0.25)]);
}

#[test]
fn isotropic_at_zero_g() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDrainePhaseBlend::new(&ctx);
    // g = 0 and alpha = 0 give the isotropic 1/(4*PI) for both terms.
    check_batch(&ctx, &gpu, &[query(0.0, 0.0, 0.0, 0.0, 0.5)]);
}

#[test]
fn forward_scatter_peak() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDrainePhaseBlend::new(&ctx);
    // Pure forward scatter (cos_theta = 1) with a forward-biased lobe.
    check_batch(&ctx, &gpu, &[query(1.0, 0.7, 0.6, 1.5, 0.4)]);
}

#[test]
fn backward_scatter_tail() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDrainePhaseBlend::new(&ctx);
    // Pure back scatter (cos_theta = -1) with a forward-biased lobe sits in the
    // low tail of the phase.
    check_batch(&ctx, &gpu, &[query(-1.0, 0.7, 0.6, 1.5, 0.4)]);
}

#[test]
fn g_clamped_past_limit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDrainePhaseBlend::new(&ctx);
    // g beyond the ±0.999 clamp on both lobes must clamp identically on both
    // paths; a back-scatter cosine keeps the denominator well away from zero.
    check_batch(&ctx, &gpu, &[query(-0.4, 1.5, -1.5, 1.0, 0.5)]);
}

#[test]
fn hg_weight_zero_is_pure_draine() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDrainePhaseBlend::new(&ctx);
    // hg_weight = 0 selects the pure Draine term for the blend.
    check_batch(&ctx, &gpu, &[query(0.2, 0.6, 0.5, 2.0, 0.0)]);
}

#[test]
fn hg_weight_one_is_pure_hg() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDrainePhaseBlend::new(&ctx);
    // hg_weight = 1 selects the pure HG term for the blend.
    check_batch(&ctx, &gpu, &[query(0.2, 0.6, 0.5, 2.0, 1.0)]);
}

#[test]
fn hg_weight_clamped_past_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDrainePhaseBlend::new(&ctx);
    // hg_weight above 1 saturates to the pure HG term exactly as the reference's
    // saturate does.
    check_batch(&ctx, &gpu, &[query(0.5, 0.3, 0.8, 1.0, 1.75)]);
}

#[test]
fn near_collapse_denominator_guard() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDrainePhaseBlend::new(&ctx);
    // g driven to the +0.999 clamp with a matching forward cosine (cos = 1)
    // drives 1 + g^2 - 2*g*u to its minimum (1 - g)^2, which the clamp pins to
    // exactly the EPS = 1e-6 floor, engaging the max(denom, EPS) guard on both
    // paths.
    //
    // This is the one fixture where tight value-for-value parity is
    // mathematically impossible in f32: 1 + g^2 - 2*g*u subtracts two values
    // near 2.0, so its ~1e-6 result carries up to ~1 ULP of 2.0 (~2.4e-7) of
    // catastrophic-cancellation error. Whether that noisy pre-floor value lands
    // just above or just below EPS differs between the host (ordered scalar
    // evaluation) and the GPU (which may fuse a multiply-add), so the clamped
    // denominators can differ by up to ~1.24x and the 1/denom^1.5 phase by up to
    // ~1.24^1.5 ~= 1.38x. No input can push the real denominator a clear margin
    // below EPS, because (1 - |g|)^2 bottoms out at exactly EPS under the
    // +-0.999 clamp, so the guard is reachable only at this boundary.
    //
    // The guard's real contract is therefore checked tightly -- both phases must
    // stay finite and strictly positive, i.e. the floor prevented a
    // divide-by-zero collapse to NaN/Inf -- while their mutual agreement is
    // asserted within the documented worst-case band rather than the
    // continuous-region EPS/REL bound.
    let q = query(1.0, 0.999, 0.999, 1.0, 0.5);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1, "one result per query");
    let r = got[0];
    let golden_draine = draine_phase(q.cos_theta, q.g_draine, q.alpha);
    let golden_blend = hg_draine_phase(q.cos_theta, q.g_hg, q.g_draine, q.alpha, q.hg_weight);

    // Worst-case band for the single ill-conditioned guard point; the ~1.38x f32
    // cancellation bound derived above sits comfortably inside this margin so the
    // suite stays green on any real device while still failing a wrong port.
    const GUARD_BAND: f32 = 0.5;
    for (label, gpu_v, golden_v) in [
        ("draine", r.draine, golden_draine),
        ("hg_draine", r.hg_draine, golden_blend),
    ] {
        assert!(
            gpu_v.is_finite() && golden_v.is_finite(),
            "{label} must stay finite at the guard boundary: gpu={gpu_v} golden={golden_v}"
        );
        assert!(
            gpu_v > 0.0 && golden_v > 0.0,
            "{label} must stay strictly positive at the guard boundary: gpu={gpu_v} golden={golden_v}"
        );
        let rel = (gpu_v - golden_v).abs() / gpu_v.abs().max(golden_v.abs());
        assert!(
            rel <= GUARD_BAND,
            "{label} guard-point parity {rel} exceeds the documented f32 cancellation band {GUARD_BAND}: gpu={gpu_v} golden={golden_v}"
        );
    }
}

#[test]
fn mixed_batch_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDrainePhaseBlend::new(&ctx);
    // Several distinct queries resolved in a single dispatch.
    let queries = vec![
        query(0.0, 0.0, 0.0, 0.0, 0.5),
        query(0.9, 0.8, 0.7, 1.0, 0.3),
        query(-0.6, -0.4, -0.5, 0.5, 0.8),
        query(0.5, 0.2, 0.9, 2.5, 0.0),
        query(-0.9, 0.6, 0.6, 3.0, 1.0),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDrainePhaseBlend::new(&ctx);
    let mut state: u64 = 0x5eed_dcba_0f0f_0001;

    let mut queries: Vec<DrainePhaseBlendQuery> = Vec::new();
    for _ in 0..384 {
        // Draw each argument a clear margin inside its clamp interval so both
        // paths clamp (or not) identically and stay on the same branch.
        let cos_theta = draw(&mut state, -0.98, 0.98);
        let g_hg = draw(&mut state, -0.95, 0.95);
        let g_draine = draw(&mut state, -0.95, 0.95);
        let alpha = draw(&mut state, 0.05, 4.0);
        let hg_weight = draw(&mut state, 0.0, 1.0);
        queries.push(query(cos_theta, g_hg, g_draine, alpha, hg_weight));
    }

    // One dispatch over the whole randomized batch, compared value-for-value.
    check_batch(&ctx, &gpu, &queries);
}
