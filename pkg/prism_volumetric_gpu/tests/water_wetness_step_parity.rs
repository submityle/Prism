//! Real-device parity for the full moisture-step twin:
//! [`GpuWaterWetnessStep`](prism_volumetric_gpu::water_wetness_step::GpuWaterWetnessStep)
//! must reproduce the dependency-free `CPU` golden
//! [`step_moisture`](prism_render_architecture::water::wetness::step_moisture)
//! — update both the wetness saturation and the standing puddle depth — across
//! water contact, rain-wets, pure-dry and puddle fill/drain regimes, several
//! cell counts including non-multiples of the `256`-wide workgroup, a zero
//! step, and the degenerate request.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden [`step_moisture`] is public and pure, so the expected fields are
//! built in-host, cell by cell, and compared against the `GPU` readback. A
//! `GPU == oracle` pass is therefore directly a `GPU == golden` pass.
//!
//! # Parity criterion
//!
//! Each cell performs the identical clamp, bit-replicated `exp_approx` envelope
//! and add/`max` puddle integration as the golden, in the same order, so the
//! only residual is the last-place slack of a `GPU` fused multiply-add. Each
//! updated sample is asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::wetness`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::wetness::{step_moisture, SurfaceMoisture, WetnessParams};
use prism_volumetric_gpu::water_wetness_step::{GpuWaterWetnessStep, WaterWetnessStep};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on one updated sample.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes.
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

/// Builds the oracle wetness params. Only the three rates feed a step; the
/// response-only fields are filled with arbitrary (non-trivial) values to prove
/// they never influence the result.
fn params(absorb_rate: f32, dry_rate: f32) -> WetnessParams {
    WetnessParams {
        max_capillary_height: 0.5,
        absorb_rate,
        dry_rate,
        darkening_strength: 0.4,
        puddle_threshold: 0.02,
    }
}

/// Pins one `GPU` moisture step against the `CPU` golden, cell by cell, for
/// both the wetness and puddle fields.
#[expect(clippy::too_many_arguments, reason = "mirrors the golden drivers")]
fn check(
    ctx: &GpuContext,
    gpu: &GpuWaterWetnessStep,
    wetness: &[f32],
    puddle: &[f32],
    contact: &[u32],
    rain: &[f32],
    absorb_rate: f32,
    dry_rate: f32,
    drain_rate: f32,
    dt: f32,
) {
    let p = params(absorb_rate, dry_rate);
    let mut want_wetness = Vec::with_capacity(wetness.len());
    let mut want_puddle = Vec::with_capacity(wetness.len());
    for i in 0..wetness.len() {
        let next = step_moisture(
            SurfaceMoisture {
                wetness: wetness[i],
                puddle_depth: puddle[i],
            },
            p,
            contact[i] != 0,
            rain[i],
            drain_rate,
            dt,
        );
        want_wetness.push(next.wetness);
        want_puddle.push(next.puddle_depth);
    }
    let got = gpu.evaluate(
        ctx,
        wetness,
        puddle,
        contact,
        rain,
        absorb_rate,
        dry_rate,
        drain_rate,
        dt,
    );
    let label = format!(
        "n={} absorb={absorb_rate} dry={dry_rate} drain={drain_rate} dt={dt}",
        wetness.len()
    );
    assert_eq!(
        got.wetness.len(),
        want_wetness.len(),
        "{label}: wetness len"
    );
    assert_eq!(
        got.puddle_depth.len(),
        want_puddle.len(),
        "{label}: puddle len"
    );
    for (i, (&g, &w)) in got.wetness.iter().zip(want_wetness.iter()).enumerate() {
        assert!(close(g, w), "{label}: wetness[{i}] gpu {g} vs cpu {w}");
    }
    for (i, (&g, &w)) in got.puddle_depth.iter().zip(want_puddle.iter()).enumerate() {
        assert!(close(g, w), "{label}: puddle[{i}] gpu {g} vs cpu {w}");
    }
}

/// A deterministic wetness field over `n` cells, spanning the saturation range
/// plus a couple of out-of-range inputs so the golden's clamp is exercised.
fn wetness_field(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32 / (n as f32).max(1.0);
            // A gentle ramp with two deliberate out-of-range probes.
            match i % 7 {
                0 => -0.2,
                6 => 1.3,
                _ => t,
            }
        })
        .collect()
}

/// A deterministic puddle-depth field over `n` cells, including a negative
/// probe so the golden's non-negative clamp is exercised.
fn puddle_field(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| match i % 5 {
            0 => 0.0,
            2 => -0.1,
            4 => 0.3,
            _ => 0.05 * (i as f32),
        })
        .collect()
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn degenerate_request_returns_input() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_wetness_step parity: no wgpu adapter");
        return;
    };
    let gpu = GpuWaterWetnessStep::new(&ctx);
    // An empty grid returns empty, exactly like the golden.
    assert_eq!(
        gpu.evaluate(&ctx, &[], &[], &[], &[], 2.0, 0.5, 0.1, 0.2),
        WaterWetnessStep {
            wetness: Vec::new(),
            puddle_depth: Vec::new(),
        },
        "empty grid"
    );
    // A short input is the twin's device-buffer no-op contract: it returns the
    // wetness/puddle unchanged rather than dispatching with undersized buffers.
    let w = vec![0.3_f32, 0.4];
    let p = vec![0.1_f32, 0.2];
    let c = vec![0_u32]; // deliberately shorter than wetness
    let r = vec![0.5_f32, 0.5];
    assert_eq!(
        gpu.evaluate(&ctx, &w, &p, &c, &r, 2.0, 0.5, 0.1, 0.2),
        WaterWetnessStep {
            wetness: w.clone(),
            puddle_depth: p.clone(),
        },
        "short contact input"
    );
}

#[test]
fn water_contact_soaks_at_full_absorb_rate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWetnessStep::new(&ctx);
    let n = 40usize;
    let w = wetness_field(n);
    let p = puddle_field(n);
    // Every cell in direct water contact: the wetness branch is pure absorb at
    // the full rate; rain still fills puddles.
    let c = vec![1_u32; n];
    let r = vec![0.6_f32; n];
    check(&ctx, &gpu, &w, &p, &c, &r, 2.5, 0.5, 0.2, 0.3);
}

#[test]
fn rain_wets_exposed_surfaces() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWetnessStep::new(&ctx);
    let n = 48usize;
    let w = wetness_field(n);
    let p = puddle_field(n);
    // No contact, but rain above the epsilon: the wetness branch is absorb at a
    // rain-scaled rate. A couple of cells carry rain beyond `1` so the
    // `min(rain, 1)` clamp is exercised.
    let c = vec![0_u32; n];
    let r: Vec<f32> = (0..n)
        .map(|i| match i % 4 {
            0 => 0.2,
            1 => 0.75,
            2 => 1.5,
            _ => 0.9,
        })
        .collect();
    check(&ctx, &gpu, &w, &p, &c, &r, 3.0, 0.5, 0.15, 0.4);
}

#[test]
fn rain_free_exposed_surfaces_dry() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWetnessStep::new(&ctx);
    let n = 33usize;
    let w = wetness_field(n);
    let p = puddle_field(n);
    // No contact and no rain (a mix of exact zero and negative drives, all at or
    // below the epsilon): the wetness branch is pure dry; puddles only drain.
    let c = vec![0_u32; n];
    let r: Vec<f32> = (0..n)
        .map(|i| if i % 2 == 0 { 0.0 } else { -0.3 })
        .collect();
    check(&ctx, &gpu, &w, &p, &c, &r, 2.0, 0.8, 0.25, 0.5);
}

#[test]
fn puddles_fill_and_drain() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWetnessStep::new(&ctx);
    let n = 36usize;
    let w = wetness_field(n);
    let p = puddle_field(n);
    let c = vec![0_u32; n];
    // Heavy rain against a strong drain so some cells net-fill and some
    // net-drain (and clamp at zero) within one step.
    let r = vec![0.4_f32; n];
    check(&ctx, &gpu, &w, &p, &c, &r, 2.0, 0.5, 1.2, 0.6);
}

#[test]
fn zero_step_is_identity_up_to_clamp() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWetnessStep::new(&ctx);
    let n = 25usize;
    let w = wetness_field(n);
    let p = puddle_field(n);
    let c: Vec<u32> = (0..n as u32).map(|i| i % 2).collect();
    let r = vec![0.5_f32; n];
    // A zero step: `exp_approx(0) == 1` so wetness is clamped-only and the
    // puddle integral adds nothing — the golden collapses to clamps.
    check(&ctx, &gpu, &w, &p, &c, &r, 2.0, 0.5, 0.3, 0.0);
}

#[test]
fn matches_golden_across_workgroup_tails() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWetnessStep::new(&ctx);
    // Cell counts straddling the 256-wide workgroup boundary exercise the tail
    // guard: single cell, below, exactly one group, and just over two groups.
    for &n in &[1usize, 255, 256, 257, 513] {
        let w = wetness_field(n);
        let p = puddle_field(n);
        let c: Vec<u32> = (0..n as u32).map(|i| i % 3 % 2).collect();
        let r: Vec<f32> = (0..n)
            .map(|i| match i % 4 {
                0 => 0.0,
                1 => 0.5,
                2 => 1.2,
                _ => -0.1,
            })
            .collect();
        check(&ctx, &gpu, &w, &p, &c, &r, 2.0, 0.5, 0.2, 0.35);
    }
}

#[test]
fn is_deterministic() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWetnessStep::new(&ctx);
    let n = 300usize;
    let w = wetness_field(n);
    let p = puddle_field(n);
    let c: Vec<u32> = (0..n as u32).map(|i| i % 2).collect();
    let r = vec![0.4_f32; n];
    let a = gpu.evaluate(&ctx, &w, &p, &c, &r, 2.0, 0.5, 0.2, 0.3);
    let b = gpu.evaluate(&ctx, &w, &p, &c, &r, 2.0, 0.5, 0.2, 0.3);
    assert_eq!(a, b, "the same request steps identically");
}
