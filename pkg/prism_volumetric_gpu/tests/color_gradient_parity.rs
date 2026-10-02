//! Real-device parity for the multi-stop colour-gradient twin:
//! [`GpuColorGradient`](prism_volumetric_gpu::color_gradient::GpuColorGradient)
//! must reproduce the `CPU` golden
//! [`color_gradient`](prism_render_architecture::particle::color_gradient)
//! across endpoint holds, interior segment blends, hard colour steps and `HDR`
//! channels.
//!
//! The fixtures cover the shapes the golden unit tests call out: a single-stop
//! constant gradient, a two-stop linear ramp probed inside and extrapolated past
//! both ends, a many-stop ramp probed in each interior segment, a duplicated
//! position (a hard colour step), and `HDR` stop colours above `1.0`. Every
//! query `t` is chosen clear of the stop positions (a reject-sampling margin) so
//! the `CPU` and `GPU` take the same branch; a final randomized batch draws
//! stops and probes from a host-side `u64` `LCG` with the same margin.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The sampled channels thread through a subtract, a multiply, an add and one
//! guarded division, so `CPU` and `GPU` are not bit-exact and are compared under
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on
//! every channel.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::color_gradient`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::color_gradient::{ColorGradient, ColorStop, Rgba};
use prism_volumetric_gpu::color_gradient::{
    ColorGradientQuery, ColorGradientSample, GpuColorGradient,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous channels.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous channels.
const REL_EPS: f32 = 1.0e-3;
/// Floor for the relative-tolerance denominator.
const REL_FLOOR: f32 = 1.0e-6;

/// Mixed absolute / relative tolerance comparison for one `f32` channel.
fn approx(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Tolerant comparison of a `GPU` sample against a golden [`Rgba`].
fn sample_matches(got: ColorGradientSample, want: Rgba) -> bool {
    approx(got.r, want.r) && approx(got.g, want.g) && approx(got.b, want.b) && approx(got.a, want.a)
}

/// Builds the gradient from `stops`, then asserts the `GPU` sample of each `t`
/// matches the golden [`ColorGradient::sample`] within tolerance. The sorted
/// stops the gradient stores are the exact ring handed to the device.
fn assert_parity(gpu: &GpuColorGradient, ctx: &GpuContext, stops: &[ColorStop], ts: &[f32]) {
    let gradient = ColorGradient::from_stops(stops.to_vec());
    let sorted = gradient.stops();
    let queries: Vec<ColorGradientQuery> = ts.iter().map(|&t| ColorGradientQuery::new(t)).collect();
    let got = gpu.sample(ctx, sorted, &queries);
    assert_eq!(got.len(), ts.len(), "one sample per query");
    for (sample, &t) in got.iter().zip(ts.iter()) {
        let want = gradient.sample(t);
        assert!(
            sample_matches(*sample, want),
            "sample mismatch at t={t}: gpu {sample:?} vs cpu {want:?}"
        );
    }
}

/// Convenience `RGBA` builder for the fixtures.
fn rgba(r: f32, g: f32, b: f32, a: f32) -> Rgba {
    Rgba::new(r, g, b, a)
}

#[test]
fn single_stop_is_constant() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorGradient::new(&ctx);
    // A one-stop gradient holds its colour for every t, inside or outside.
    let stops = [ColorStop::new(0.5, rgba(0.25, 0.5, 0.75, 1.0))];
    assert_parity(&gpu, &ctx, &stops, &[-0.4, 0.17, 0.83, 1.4]);
}

#[test]
fn two_stop_ramp_interpolates_and_holds() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorGradient::new(&ctx);
    // A two-stop ramp: interior probes blend, probes past either end hold the
    // endpoint colour. Every t stays clear of the stop positions 0.0 and 1.0.
    let stops = [
        ColorStop::new(0.0, rgba(0.1, 0.2, 0.3, 0.4)),
        ColorStop::new(1.0, rgba(0.9, 0.8, 0.7, 0.6)),
    ];
    assert_parity(&gpu, &ctx, &stops, &[-0.35, 0.18, 0.37, 0.62, 0.81, 1.45]);
}

#[test]
fn many_stop_ramp_blends_each_segment() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorGradient::new(&ctx);
    // Five stops at 0.0, 0.25, 0.5, 0.75, 1.0; one probe sits mid-segment in
    // each interior interval, clear of every stop position.
    let stops = [
        ColorStop::new(0.0, rgba(1.0, 0.0, 0.0, 1.0)),
        ColorStop::new(0.25, rgba(0.0, 1.0, 0.0, 1.0)),
        ColorStop::new(0.5, rgba(0.0, 0.0, 1.0, 1.0)),
        ColorStop::new(0.75, rgba(1.0, 1.0, 0.0, 0.5)),
        ColorStop::new(1.0, rgba(1.0, 1.0, 1.0, 0.25)),
    ];
    assert_parity(&gpu, &ctx, &stops, &[-0.2, 0.12, 0.37, 0.63, 0.88, 1.3]);
}

#[test]
fn duplicated_position_is_a_hard_step() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorGradient::new(&ctx);
    // Two stops share position 0.5 (a hard colour step). Probes stay off 0.5 so
    // the step boundary itself is never queried, and the span guard collapses
    // the zero-width segment to the lower stop exactly as the reference does.
    let stops = [
        ColorStop::new(0.0, rgba(0.0, 0.0, 0.0, 1.0)),
        ColorStop::new(0.5, rgba(1.0, 0.0, 0.0, 1.0)),
        ColorStop::new(0.5, rgba(0.0, 0.0, 1.0, 1.0)),
        ColorStop::new(1.0, rgba(1.0, 1.0, 1.0, 1.0)),
    ];
    assert_parity(&gpu, &ctx, &stops, &[0.21, 0.37, 0.63, 0.82]);
}

#[test]
fn hdr_channels_above_one_are_preserved() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorGradient::new(&ctx);
    // HDR emissive stops with channels above 1.0 must blend and hold unclamped.
    let stops = [
        ColorStop::new(0.2, rgba(0.0, 0.0, 0.0, 1.0)),
        ColorStop::new(0.8, rgba(8.0, 4.0, 2.0, 1.0)),
    ];
    assert_parity(&gpu, &ctx, &stops, &[-0.3, 0.33, 0.5, 0.67, 1.2]);
}

#[test]
fn batch_of_queries_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorGradient::new(&ctx);
    // A batch exercises the one-thread-per-query flattening; each result must be
    // independent of its neighbours.
    let stops = [
        ColorStop::new(0.1, rgba(0.2, 0.4, 0.6, 1.0)),
        ColorStop::new(0.4, rgba(0.6, 0.2, 0.1, 0.8)),
        ColorStop::new(0.9, rgba(0.1, 0.9, 0.5, 0.3)),
    ];
    let ts = [-0.5, 0.03, 0.17, 0.26, 0.33, 0.55, 0.72, 0.85, 1.1];
    assert_parity(&gpu, &ctx, &stops, &ts);
}

#[test]
fn empty_gradient_guards_to_transparent() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorGradient::new(&ctx);
    // No stops: every query resolves to transparent black with no dispatch.
    let queries = [
        ColorGradientQuery::new(0.0),
        ColorGradientQuery::new(0.5),
        ColorGradientQuery::new(1.0),
    ];
    let got = gpu.sample(&ctx, &[], &queries);
    assert_eq!(got.len(), queries.len());
    for sample in got {
        assert!(sample_matches(sample, Rgba::TRANSPARENT));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorGradient::new(&ctx);
    // No queries: no dispatch is issued and the result vector is empty.
    let stops = [
        ColorStop::new(0.0, rgba(0.0, 0.0, 0.0, 1.0)),
        ColorStop::new(1.0, rgba(1.0, 1.0, 1.0, 1.0)),
    ];
    assert!(gpu.sample(&ctx, &stops, &[]).is_empty());
}

/// A tiny host-side `u64` linear-congruential generator (Knuth / `MMIX`
/// constants). Used only to draw fixture inputs; it emits no `f32`
/// transcendental, matching the determinism rules.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Lcg {
        Lcg { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// A uniform `f32` in `[0, 1)` from the top `24` bits, by integer division.
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32)
    }

    /// A uniform `f32` in `[lo, hi)`.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

#[test]
fn randomized_reject_sampled_batch_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorGradient::new(&ctx);
    let mut rng = Lcg::new(0x9E37_79B9_7F4A_7C15);
    // Margin keeping every probe clear of every stop position, so CPU and GPU
    // never straddle a branch boundary.
    let margin = 0.03_f32;

    for _ in 0..24 {
        // 2..=6 stops with a guaranteed gap so no two positions collide.
        let stop_count = 2 + (rng.next_u64() % 5) as usize;
        let mut stops: Vec<ColorStop> = Vec::with_capacity(stop_count);
        let mut pos = rng.range(0.02, 0.1);
        for _ in 0..stop_count {
            let color = rgba(
                rng.range(0.0, 4.0),
                rng.range(0.0, 4.0),
                rng.range(0.0, 4.0),
                rng.range(0.0, 1.0),
            );
            stops.push(ColorStop::new(pos, color));
            pos += rng.range(0.1, 0.2);
        }

        // Draw probes in [-0.2, 1.2], rejecting any within `margin` of a stop.
        let mut ts: Vec<f32> = Vec::with_capacity(8);
        let mut guard = 0;
        while ts.len() < 8 && guard < 4096 {
            guard += 1;
            let t = rng.range(-0.2, 1.2);
            if stops.iter().any(|s| (s.position - t).abs() < margin) {
                continue;
            }
            ts.push(t);
        }

        assert_parity(&gpu, &ctx, &stops, &ts);
    }
}
