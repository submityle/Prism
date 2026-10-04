//! Real-device parity for the pixel reconstruction filter twin:
//! [`GpuPixelFilter`](prism_volumetric_gpu::pixel_reconstruction_filter::GpuPixelFilter)
//! must reproduce the `CPU` closed form of the reference tracer
//! `prism_render_architecture::reference_pt::filter` — the box pass-through, the
//! tent inverse `CDF` `tent_offset`, and the per-axis `PixelFilter::warp` — across
//! named endpoint fixtures, an unrecognised kind and a randomized sweep compared
//! query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same closed form: `tent_offset(u)` is
//! `sqrt(2 u) - 1` on the lower half and `1 - sqrt(2 - 2 u)` on the upper half,
//! and `warp` either passes the jitter through (box) or returns
//! `0.5 + tent_offset(u)` on each axis (tent). Because the reference and this
//! oracle are both scalar `f32`, a `GPU == oracle` pass is direct evidence the
//! ported kernel computes the same warp the reference does.
//!
//! # Parity criterion
//!
//! The tent branch threads through `sqrt`, so a `GPU` result may land a few
//! units in the last place from the scalar oracle; the tent outputs are
//! asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (relative floor
//! `1e-6`). The box branch performs no arithmetic, so its pass-through is
//! asserted within `abs_diff <= 1e-6`. The discrete `valid` word is asserted
//! exactly on every query.
//!
//! # Conditioning
//!
//! The randomized sweep draws `kind` in `{0, 1}` and the jitter in `[0, 1)`,
//! rejecting any draw within `1e-4` of the tent midpoint `u = 0.5` where the
//! `sqrt` branch switches, so no query straddles the branch boundary where a
//! one-`ulp` input difference would select a different half of the inverse
//! `CDF`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::filter`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::pixel_reconstruction_filter::{
    GpuPixelFilter, PixelFilterQuery, PixelFilterResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on each continuous tent output. A `GPU` `sqrt` may land a few
/// units in the last place from the scalar oracle; `1e-4` admits that legal
/// slack while still failing a wrong port.
const ABS: f32 = 1.0e-4;

/// Relative bound on each continuous tent output, applied for larger magnitudes
/// where a few units in the last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Tight absolute bound on the box pass-through, which performs no arithmetic
/// and so must agree with the input to the last bit on any device.
const BOX_ABS: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32, abs_eps: f32, rel_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= rel_eps
}

/// Independent host reimplementation of the golden `tent_offset`: the tent's
/// inverse `CDF`, a shifted square root on each half, split at the midpoint.
fn tent_offset(u: f32) -> f32 {
    if u < 0.5 {
        (2.0 * u).sqrt() - 1.0
    } else {
        1.0 - (2.0 - 2.0 * u).sqrt()
    }
}

/// Independent host reimplementation of the golden `PixelFilter::warp` plus the
/// twin's degeneracy flag: box passes through, tent warps each axis, any other
/// kind is unrecognised and yields a zeroed position with `valid = 0`.
fn oracle(q: &PixelFilterQuery) -> PixelFilterResult {
    match q.kind {
        0 => PixelFilterResult {
            wx: q.jitter_x,
            wy: q.jitter_y,
            valid: 1,
        },
        1 => PixelFilterResult {
            wx: 0.5 + tent_offset(q.jitter_x),
            wy: 0.5 + tent_offset(q.jitter_y),
            valid: 1,
        },
        _ => PixelFilterResult {
            wx: 0.0,
            wy: 0.0,
            valid: 0,
        },
    }
}

/// Pins one `GPU` result against the oracle: `valid` exactly always; the box
/// pass-through to the tight bound; the tent warp to the continuous bound.
fn check_one(idx: usize, q: &PixelFilterQuery, got: &PixelFilterResult, want: &PixelFilterResult) {
    assert_eq!(
        got.valid, want.valid,
        "query {idx} valid: gpu {} vs cpu {}",
        got.valid, want.valid
    );
    let (abs_eps, rel_eps) = if q.kind == 0 {
        // The box filter copies the jitter through verbatim: no arithmetic, so
        // the two must agree to the last bit.
        (BOX_ABS, 0.0)
    } else {
        (ABS, REL)
    };
    assert!(
        close(got.wx, want.wx, abs_eps, rel_eps),
        "query {idx} wx: gpu {} vs cpu {}",
        got.wx,
        want.wx
    );
    assert!(
        close(got.wy, want.wy, abs_eps, rel_eps),
        "query {idx} wy: gpu {} vs cpu {}",
        got.wy,
        want.wy
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuPixelFilter, queries: &[PixelFilterQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_one(idx, q, result, &want);
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

/// Draws a `f32` in `[lo, hi)` from the generator, using only integer-to-float
/// division (no transcendental).
fn uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let u = lcg(state) as f32 * (1.0 / 4_294_967_296.0);
    lo + (hi - lo) * u
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping pixel_reconstruction_filter parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuPixelFilter::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn box_filter_is_the_identity_pass_through() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPixelFilter::new(&ctx);
    // The box filter must pass each jitter coordinate through unchanged.
    let queries = [
        PixelFilterQuery::new(0, 0.0, 0.0),
        PixelFilterQuery::new(0, 0.3, 0.8),
        PixelFilterQuery::new(0, 0.123_456, 0.654_321),
        PixelFilterQuery::new(0, 0.999_9, 0.000_1),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn tent_endpoints_and_centre_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPixelFilter::new(&ctx);
    // u = 0 -> left edge (offset -1 -> position -0.5); u -> 0.5 -> centre
    // (offset 0 -> position 0.5); u -> 1 -> right edge (offset +1 -> position
    // 1.5). The exact midpoint takes the upper branch, so probe just below and
    // just above it to pin both halves of the inverse CDF.
    let queries = [
        PixelFilterQuery::new(1, 0.0, 0.0),
        PixelFilterQuery::new(1, 0.499_9, 0.500_1),
        PixelFilterQuery::new(1, 0.999_999, 0.000_001),
        PixelFilterQuery::new(1, 0.25, 0.75),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn tent_warp_stays_within_support() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPixelFilter::new(&ctx);
    // Every tent-warped coordinate must land inside the radius-one support
    // [-0.5, 1.5], and must equal the independent oracle.
    let mut state: u64 = 0x5151_2a7b_9f3c_0011;
    let mut queries = Vec::new();
    for _ in 0..256 {
        let mut jx = uniform(&mut state, 0.0, 1.0);
        let mut jy = uniform(&mut state, 0.0, 1.0);
        if (jx - 0.5).abs() < 1.0e-4 {
            jx = 0.25;
        }
        if (jy - 0.5).abs() < 1.0e-4 {
            jy = 0.75;
        }
        queries.push(PixelFilterQuery::new(1, jx, jy));
    }
    let got = gpu.evaluate(&ctx, &queries);
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_one(idx, q, result, &want);
        assert!(
            (-0.5 - ABS..=1.5 + ABS).contains(&result.wx),
            "query {idx} wx out of tent support: {}",
            result.wx
        );
        assert!(
            (-0.5 - ABS..=1.5 + ABS).contains(&result.wy),
            "query {idx} wy out of tent support: {}",
            result.wy
        );
    }
}

#[test]
fn unrecognised_kind_is_invalid_and_zeroed() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPixelFilter::new(&ctx);
    // Any kind >= 2 is unrecognised: valid = 0 with a zeroed position.
    let queries = [
        PixelFilterQuery::new(2, 0.3, 0.7),
        PixelFilterQuery::new(7, 0.1, 0.9),
        PixelFilterQuery::new(u32::MAX, 0.5, 0.5),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_one(idx, q, result, &want);
        assert_eq!(result.valid, 0, "query {idx} must be invalid");
        assert!(
            result.wx.abs() <= BOX_ABS && result.wy.abs() <= BOX_ABS,
            "query {idx} invalid position must be zeroed"
        );
    }
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPixelFilter::new(&ctx);
    // A mixed batch pins box, tent and invalid kinds together in one dispatch.
    let queries = [
        PixelFilterQuery::new(0, 0.2, 0.7),
        PixelFilterQuery::new(1, 0.2, 0.7),
        PixelFilterQuery::new(1, 0.8, 0.3),
        PixelFilterQuery::new(0, 0.5, 0.5),
        PixelFilterQuery::new(1, 0.05, 0.95),
        PixelFilterQuery::new(3, 0.4, 0.6),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPixelFilter::new(&ctx);
    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut queries = Vec::new();
    for _ in 0..512 {
        // Draw kind in {0, 1} and jitter in [0, 1), rejecting the thin band
        // around the tent midpoint where the sqrt branch switches.
        let kind = lcg(&mut state) & 1;
        let mut jx = uniform(&mut state, 0.0, 1.0);
        let mut jy = uniform(&mut state, 0.0, 1.0);
        if (jx - 0.5).abs() < 1.0e-4 {
            jx = 0.3;
        }
        if (jy - 0.5).abs() < 1.0e-4 {
            jy = 0.7;
        }
        queries.push(PixelFilterQuery::new(kind, jx, jy));
    }
    check(&ctx, &gpu, &queries);
}
