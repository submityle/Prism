//! Real-device parity for the spectral amplitude-advance twin:
//! [`GpuWaterSpectrumAdvance`](prism_volumetric_gpu::water_spectrum_advance::GpuWaterSpectrumAdvance)
//! must reproduce the per-cell phase advance of the `CPU` golden
//! [`advance_amplitude`](prism_render_architecture::water::spectrum::advance_amplitude)
//! composed with
//! [`dispersion`](prism_render_architecture::water::spectrum::dispersion) —
//! the Hermitian `h(k,t) = h0 e^{i omega t} + conj(h0_neg) e^{-i omega t}`
//! that keeps the inverse-`FFT` height field real — across hand-chosen
//! fixtures, boundary cases, and a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The goldens `advance_amplitude` and `dispersion` are public and pure, so the
//! expected complex amplitude is built in-host by calling them directly. A
//! `GPU == oracle` pass is therefore directly a `GPU == golden` pass.
//!
//! # Parity criterion
//!
//! Each component threads through a range-reduced polynomial `sin`/`cos`, two
//! complex products and a sum, with `omega` through a `sqrt`, so a `GPU` fused
//! multiply-add may land a few units in the last place from the scalar
//! reference. Each is asserted within `abs_diff <= 1e-5` or `rel_diff <= 1e-4`,
//! tight enough to fail a wrong port — a dropped conjugate, a flipped phasor
//! sign, a missing `k <= 0` dispersion guard — yet loose enough to admit legal
//! last-place slack.
//!
//! # Conditioning
//!
//! The sweep keeps `t` modest and `k` within a band where `omega t` stays well
//! inside the polynomial's accurate range, so the phase approximation agrees
//! between `CPU` and `GPU` to the stated tolerance. The `k <= 0` guard
//! (`omega = 0`, collapsing to the `t = 0` identity) and the calm `h0 = 0`
//! cases are exercised with exact fixtures.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::spectrum`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::spectrum::{advance_amplitude, dispersion, Complex};
use prism_volumetric_gpu::water_spectrum_advance::{
    GpuWaterSpectrumAdvance, WaterSpectrumAdvanceQuery, WaterSpectrumAdvanceResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on an advanced amplitude component. A `GPU` fused
/// multiply-add may land a few units in the last place from the scalar
/// reference; `1e-5` admits that slack while still failing a wrong port.
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

/// Builds the in-host oracle for one query by calling the goldens
/// `advance_amplitude(dispersion(k))` directly. The `GPU` is pinned against this
/// exact closed form.
fn oracle(q: &WaterSpectrumAdvanceQuery) -> WaterSpectrumAdvanceResult {
    let h0 = Complex::new(q.h0_re, q.h0_im);
    let h0_neg = Complex::new(q.h0_neg_re, q.h0_neg_im);
    let out = advance_amplitude(h0, h0_neg, dispersion(q.k), q.t);
    WaterSpectrumAdvanceResult {
        re: out.re,
        im: out.im,
    }
}

/// Pins one `GPU` result against the in-host oracle, component by component.
fn check_body(idx: usize, got: &WaterSpectrumAdvanceResult, want: &WaterSpectrumAdvanceResult) {
    assert!(
        close(got.re, want.re),
        "body {idx} re: gpu {} vs cpu {}",
        got.re,
        want.re
    );
    assert!(
        close(got.im, want.im),
        "body {idx} im: gpu {} vs cpu {}",
        got.im,
        want.im
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuWaterSpectrumAdvance, queries: &[WaterSpectrumAdvanceQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_body(idx, result, &want);
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

/// Draws a scalar in `[lo, hi]` at milli resolution from `state`.
fn draw(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let span = ((hi - lo) * 1000.0) as u32;
    lo + (lcg(state) % (span + 1)) as f32 / 1000.0
}

/// Draws one random query: amplitudes in a symmetric band around zero, `k` in a
/// positive band, and `t` modest so `omega t` stays inside the polynomial's
/// accurate phase range.
fn random_query(state: &mut u64) -> WaterSpectrumAdvanceQuery {
    WaterSpectrumAdvanceQuery {
        h0_re: draw(state, -4.0, 4.0),
        h0_im: draw(state, -4.0, 4.0),
        h0_neg_re: draw(state, -4.0, 4.0),
        h0_neg_im: draw(state, -4.0, 4.0),
        k: draw(state, 0.01, 2.0),
        t: draw(state, 0.0, 3.0),
    }
}

/// The deterministic hand-chosen fixtures, each exercising a distinct branch.
fn fixture_queries() -> Vec<WaterSpectrumAdvanceQuery> {
    vec![
        // Calm: both amplitudes zero -> result exactly (0, 0) at any k, t.
        WaterSpectrumAdvanceQuery {
            h0_re: 0.0,
            h0_im: 0.0,
            h0_neg_re: 0.0,
            h0_neg_im: 0.0,
            k: 0.5,
            t: 1.25,
        },
        // t = 0: phasor is 1, result is h0 + conj(h0_neg).
        WaterSpectrumAdvanceQuery {
            h0_re: 1.5,
            h0_im: -0.75,
            h0_neg_re: 0.25,
            h0_neg_im: 2.0,
            k: 0.8,
            t: 0.0,
        },
        // k <= 0: omega = 0 (dispersion guard), collapsing to the t = 0 identity
        // even though t is non-zero.
        WaterSpectrumAdvanceQuery {
            h0_re: -1.0,
            h0_im: 0.5,
            h0_neg_re: 0.75,
            h0_neg_im: -0.25,
            k: 0.0,
            t: 2.0,
        },
        // Negative k: same omega = 0 guard, distinct amplitudes.
        WaterSpectrumAdvanceQuery {
            h0_re: 2.0,
            h0_im: 1.0,
            h0_neg_re: -1.5,
            h0_neg_im: 0.5,
            k: -0.3,
            t: 1.0,
        },
        // Small k, moderate t: slow wave advancing a modest phase.
        WaterSpectrumAdvanceQuery {
            h0_re: 0.6,
            h0_im: -1.2,
            h0_neg_re: 1.1,
            h0_neg_im: 0.4,
            k: 0.05,
            t: 2.5,
        },
        // Larger k, small t: faster wave, short time.
        WaterSpectrumAdvanceQuery {
            h0_re: -0.3,
            h0_im: 0.9,
            h0_neg_re: -0.8,
            h0_neg_im: -0.6,
            k: 1.6,
            t: 0.4,
        },
        // Only the +k amplitude populated.
        WaterSpectrumAdvanceQuery {
            h0_re: 1.0,
            h0_im: 1.0,
            h0_neg_re: 0.0,
            h0_neg_im: 0.0,
            k: 0.9,
            t: 1.5,
        },
        // Only the -k amplitude populated (exercises the conjugate path).
        WaterSpectrumAdvanceQuery {
            h0_re: 0.0,
            h0_im: 0.0,
            h0_neg_re: -1.3,
            h0_neg_im: 0.7,
            k: 0.9,
            t: 1.5,
        },
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_spectrum_advance parity: no wgpu adapter");
        return;
    };
    let gpu = GpuWaterSpectrumAdvance::new(&ctx);
    // The host short-circuits an empty batch (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn fixture_bodies_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSpectrumAdvance::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn calm_amplitudes_stay_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSpectrumAdvance::new(&ctx);
    let q = WaterSpectrumAdvanceQuery {
        h0_re: 0.0,
        h0_im: 0.0,
        h0_neg_re: 0.0,
        h0_neg_im: 0.0,
        k: 1.0,
        t: 7.0,
    };
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    assert!(got[0].re.abs() < EPS, "calm surface stays zero (re)");
    assert!(got[0].im.abs() < EPS, "calm surface stays zero (im)");
    check_body(0, &got[0], &oracle(&q));
}

#[test]
fn nonpositive_k_collapses_to_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSpectrumAdvance::new(&ctx);
    // k <= 0 means omega = 0, so the result must equal h0 + conj(h0_neg)
    // regardless of t, matching the dispersion guard.
    let q = WaterSpectrumAdvanceQuery {
        h0_re: 1.25,
        h0_im: -0.5,
        h0_neg_re: 0.75,
        h0_neg_im: 2.0,
        k: 0.0,
        t: 9.0,
    };
    // Same amplitudes and wave number, but at t = 0: with omega = 0 the phase
    // advance is a no-op, so the two queries must yield the same amplitude. We
    // assert that time-independence directly rather than a closed-form integer,
    // since cos(0) threads through the shared Taylor `cos_approx` (whose value
    // at zero is ~0.99983, not exactly 1), so both `CPU` and `GPU` land a hair
    // off the naive `h0 + conj(h0_neg)`.
    let q0 = WaterSpectrumAdvanceQuery { t: 0.0, ..q };
    let got = gpu.evaluate(&ctx, &[q, q0]);
    assert_eq!(got.len(), 2);
    // The dispersion guard makes the advance time-independent at k <= 0.
    assert!(
        close(got[0].re, got[1].re) && close(got[0].im, got[1].im),
        "omega=0 must be time-independent: t=9 {:?} vs t=0 {:?}",
        got[0],
        got[1]
    );
    // And each still matches the golden exactly within tolerance.
    check_body(0, &got[0], &oracle(&q));
    check_body(1, &got[1], &oracle(&q0));
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSpectrumAdvance::new(&ctx);
    let mut state = 0x1a2b_3c4d_5e6f_7081_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random cells pin every output across a wide
    // span of amplitudes, wave numbers and times.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
