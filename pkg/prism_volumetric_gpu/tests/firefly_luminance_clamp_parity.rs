//! Real-device parity for the firefly luminance-clamp twin:
//! [`GpuFireflyClamp`](prism_volumetric_gpu::firefly_luminance_clamp::GpuFireflyClamp)
//! must reproduce the `CPU` golden `luminance` and `FireflyClamp::apply` of
//! `prism_render_architecture::reference_pt::firefly`, which scale an
//! over-bright radiance sample down so its luminance equals a configured
//! maximum while preserving its chromaticity.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the `luminance` dot product and the `Off` / non-positive-maximum /
//! below-threshold / scaled branch of `apply` — written out directly so the
//! test never imports `prism_render_architecture`. It mirrors the reference
//! branch for branch, including the invalid-mode guard that reports
//! `valid = 0` with cleared outputs.
//!
//! The fixtures cover the branches the kernel must honor: an `Off` pass
//! through, a non-positive maximum that disables the clamp, a sample already
//! below the threshold, a sample above the threshold that is scaled down (its
//! clamped luminance landing on the maximum and its channel ratios preserved),
//! and an unknown mode that reports `valid = 0`. A sweep over random modes,
//! maxima and radiance samples follows, plus an empty batch the host
//! short-circuits with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every channel threads through multiplies, adds and one guarded division, so
//! `CPU` and `GPU` evaluate the same closed form but need not be bit-exact (a
//! `GPU` may contract a multiply-add). The continuous comparison is
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete
//! `valid` flag is compared exactly. The sweep keeps each sample's luminance
//! comfortably away from the maximum, so parity never sits on the clamp branch
//! knife edge.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::firefly`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::firefly_luminance_clamp::{FireflyClampQuery, GpuFireflyClamp};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;

/// Red luma coefficient, matching the kernel.
const LUMA_R: f32 = 0.2126;
/// Green luma coefficient, matching the kernel.
const LUMA_G: f32 = 0.7152;
/// Blue luma coefficient, matching the kernel.
const LUMA_B: f32 = 0.0722;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Returns `true` when two three-channel values agree channel-wise.
fn close3(a: [f32; 3], b: [f32; 3]) -> bool {
    close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2])
}

/// Relative luminance of a linear `RGB` radiance value.
fn luminance(v: [f32; 3]) -> f32 {
    LUMA_R * v[0] + LUMA_G * v[1] + LUMA_B * v[2]
}

/// Independent host re-implementation of `luminance` and `FireflyClamp::apply`:
/// an unknown mode is invalid; `Off` and a disabled or below-threshold clamp
/// pass the sample through; otherwise every channel is scaled by
/// `max_lum / luma`.
fn oracle(q: &FireflyClampQuery) -> ([f32; 3], u32) {
    if q.mode >= 2 {
        return ([0.0, 0.0, 0.0], 0);
    }
    if q.mode == 0 {
        return (q.value, 1);
    }
    // mode == 1 (MaxLuminance)
    let luma = luminance(q.value);
    if q.max_lum <= 0.0 || luma <= q.max_lum {
        return (q.value, 1);
    }
    let scale = q.max_lum / luma;
    (
        [q.value[0] * scale, q.value[1] * scale, q.value[2] * scale],
        1,
    )
}

/// Dispatches one query and asserts both the clamped channels and the validity
/// flag against the independent oracle.
fn assert_parity(ctx: &GpuContext, gpu: &GpuFireflyClamp, q: FireflyClampQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let (clamped, valid) = oracle(&q);
    let r = got[0];
    assert_eq!(r.valid, valid, "valid flag mismatch: query={q:?}");
    if valid == 1 {
        assert!(
            close3(r.clamped, clamped),
            "clamped mismatch: gpu={:?} cpu={clamped:?} query={q:?}",
            r.clamped
        );
    }
}

#[test]
fn off_passthrough() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFireflyClamp::new(&ctx);
    // Off ignores the maximum and passes the sample through unchanged.
    assert_parity(
        &ctx,
        &gpu,
        FireflyClampQuery::new(0, 2.0, [100.0, 2.0, 0.3]),
    );
}

#[test]
fn max_non_positive_passthrough() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFireflyClamp::new(&ctx);
    // A zero or negative maximum disables the clamp even in MaxLuminance mode.
    assert_parity(
        &ctx,
        &gpu,
        FireflyClampQuery::new(1, 0.0, [10.0, 20.0, 30.0]),
    );
    assert_parity(
        &ctx,
        &gpu,
        FireflyClampQuery::new(1, -1.0, [10.0, 20.0, 30.0]),
    );
}

#[test]
fn below_threshold_passthrough() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFireflyClamp::new(&ctx);
    // luminance([0.4,0.4,0.4]) = 0.4 <= max = 1.0, so the sample is unchanged.
    assert_parity(&ctx, &gpu, FireflyClampQuery::new(1, 1.0, [0.4, 0.4, 0.4]));
}

#[test]
fn above_threshold_scaled() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFireflyClamp::new(&ctx);
    // A bright sample whose luminance exceeds the maximum is scaled down. The
    // oracle agreement covers the clamped channels; here we additionally
    // confirm the clamped luminance lands on the maximum and the chromaticity
    // (channel ratios) is preserved.
    let value = [12.0, 4.0, 1.0];
    let max_lum = 2.0_f32;
    let q = FireflyClampQuery::new(1, max_lum, value);
    assert_parity(&ctx, &gpu, q);

    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    let clamped = got[0].clamped;
    assert!(
        close(luminance(clamped), max_lum),
        "clamped luminance should equal the maximum: {}",
        luminance(clamped)
    );
    // Chromaticity preserved: clamped == value * (max_lum / luma).
    let scale = max_lum / luminance(value);
    assert!(
        close3(
            clamped,
            [value[0] * scale, value[1] * scale, value[2] * scale]
        ),
        "chromaticity should be preserved: {clamped:?}"
    );
}

#[test]
fn mode_two_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFireflyClamp::new(&ctx);
    // An unknown mode reports valid = 0 with cleared outputs.
    let q = FireflyClampQuery::new(2, 2.0, [5.0, 5.0, 5.0]);
    assert_parity(&ctx, &gpu, q);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got[0].valid, 0, "mode 2 must be invalid");
    assert_eq!(got[0].clamped, [0.0, 0.0, 0.0], "invalid output cleared");
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFireflyClamp::new(&ctx);
    assert!(
        gpu.evaluate(&ctx, &[]).is_empty(),
        "empty batch returns an empty vector with no dispatch"
    );
}

/// A small deterministic linear-congruential generator so the sweep needs no
/// external randomness. Constants are the Numerical Recipes values.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// A `[0, 1)` fraction built from the top bits, keeping the fixture pure
    /// integer host-side with no transcendental call.
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A `[lo, hi)` fraction.
    fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_unit()
    }
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFireflyClamp::new(&ctx);
    let mut rng = Lcg::new(0x7F_3C_51_09);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let mode = rng.next_u32() % 2;
        let max_lum = rng.next_range(0.1, 50.0);
        let value = [
            rng.next_range(0.0, 100.0),
            rng.next_range(0.0, 100.0),
            rng.next_range(0.0, 100.0),
        ];
        // Reject samples whose luminance sits near the clamp threshold so the
        // ordered branch comparison agrees on both sides.
        let luma = luminance(value);
        if (luma - max_lum).abs() < 0.1 {
            continue;
        }
        queries.push(FireflyClampQuery::new(mode, max_lum, value));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (clamped, valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        if valid == 1 {
            assert!(
                close3(r.clamped, clamped),
                "sweep clamped mismatch: gpu={:?} cpu={clamped:?} query={q:?}",
                r.clamped
            );
        }
    }
}
