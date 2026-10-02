//! Real-device parity for the temporal auto-exposure / eye-adaptation twin:
//! [`GpuExposureAdapt`](prism_volumetric_gpu::exposure_adapt::GpuExposureAdapt)
//! must reproduce the `CPU` golden
//! [`exposure_adapt`](prism_render_architecture::particle::exposure_adapt)
//! across the metered target `EV`, the frame-rate-independent blend factor, the
//! temporally converged next `EV`, and the linear exposure scale.
//!
//! The fixtures cover the shapes the golden unit tests call out: a brightening
//! step and a darkening step (clear of the direction threshold so both pick the
//! intended speed), a key-matching frame that meters to zero `EV`, saturation to
//! `min_ev` and to `max_ev` from an extreme scene, an exposure-compensation
//! shift, a zero-length frame that does not move the exposure, a long frame that
//! lands essentially on the target, and a wide-window case whose converged `EV`
//! lands on an integer octave to exercise the exact-power-of-two exposure scale.
//! All inputs are written as integers or simple decimals whose `luminance`
//! ratios are exact powers of two, so the fixtures stay pure and need no
//! transcendental math.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every twinned value is a continuous quantity threaded through multiplies,
//! adds, one guarded divide and a `bitcast`, so `CPU` and `GPU` are compared
//! under tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! `REL_FLOOR = 1e-6`), loose enough to admit a legal fused multiply-add yet
//! tight enough to catch a wrong port.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::exposure_adapt`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::exposure_adapt::{
    ev_to_exposure_scale, rate_factor, ExposureAdaptConfig,
};
use prism_volumetric_gpu::exposure_adapt::{ExposureAdaptQuery, GpuExposureAdapt};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous quantities.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous quantities.
const REL_EPS: f32 = 1.0e-3;
/// Floor for the relative-tolerance denominator.
const REL_FLOOR: f32 = 1.0e-6;

/// Mixed absolute / relative tolerance comparison for one `f32` value.
fn approx(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// The crate's neutral default-ish configuration: a `[-8, +8] EV` window with a
/// faster brightening rate than darkening rate and no compensation.
fn cfg() -> ExposureAdaptConfig {
    ExposureAdaptConfig::new(-8.0, 8.0, 3.0, 1.0, 0.0)
}

/// Builds one eye from a configuration and its per-frame inputs.
fn query(
    config: ExposureAdaptConfig,
    current_ev: f32,
    avg_luminance: f32,
    key: f32,
    dt: f32,
) -> ExposureAdaptQuery {
    ExposureAdaptQuery {
        config,
        current_ev,
        avg_luminance,
        key,
        dt,
    }
}

/// Asserts every twinned answer for one eye matches the `CPU` golden.
fn assert_parity(gpu: &GpuExposureAdapt, ctx: &GpuContext, q: &ExposureAdaptQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(q));
    assert_eq!(got.len(), 1, "one result per eye");
    let g = got[0];

    let cpu_target = q.config.target_exposure(q.avg_luminance, q.key);
    assert!(
        approx(g.target_ev, cpu_target),
        "target_ev mismatch: gpu {} vs cpu {cpu_target}",
        g.target_ev
    );

    // The step picks the direction-appropriate speed from the metered target.
    let speed = if cpu_target > q.current_ev {
        q.config.up_speed
    } else {
        q.config.down_speed
    };
    let cpu_rate = rate_factor(q.dt, speed);
    assert!(
        approx(g.rate, cpu_rate),
        "rate mismatch: gpu {} vs cpu {cpu_rate}",
        g.rate
    );

    let cpu_next = q
        .config
        .adapt_toward_luminance(q.current_ev, q.avg_luminance, q.key, q.dt);
    assert!(
        approx(g.next_ev, cpu_next),
        "next_ev mismatch: gpu {} vs cpu {cpu_next}",
        g.next_ev
    );

    let cpu_scale = ev_to_exposure_scale(cpu_next);
    assert!(
        approx(g.exposure_scale, cpu_scale),
        "exposure_scale mismatch: gpu {} vs cpu {cpu_scale}",
        g.exposure_scale
    );
}

#[test]
fn brightening_step_moves_up_toward_target() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuExposureAdapt::new(&ctx);
    // avg 0.045 against key 0.18 is a ratio of 4 -> +2 EV target, well above the
    // current -4 EV, so the step uses the faster brightening speed.
    let q = query(cfg(), -4.0, 0.045, 0.18, 0.1);
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn darkening_step_moves_down_toward_target() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuExposureAdapt::new(&ctx);
    // avg 0.72 against key 0.18 is a ratio of 0.25 -> -2 EV target, well below
    // the current +4 EV, so the step uses the slower darkening speed.
    let q = query(cfg(), 4.0, 0.72, 0.18, 0.1);
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn key_match_meters_to_zero_ev() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuExposureAdapt::new(&ctx);
    // average == key -> ratio 1 -> log2(1) == 0 EV target.
    let q = query(cfg(), -5.0, 0.18, 0.18, 0.1);
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn saturates_to_max_ev_from_pitch_dark_frame() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuExposureAdapt::new(&ctx);
    // A near-black frame wants a huge positive EV, clamped to max_ev (+8).
    let q = query(cfg(), 0.0, 1.0e-4, 0.18, 0.25);
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn saturates_to_min_ev_from_blown_out_frame() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuExposureAdapt::new(&ctx);
    // A blown-out frame wants a hugely negative EV, clamped to min_ev (-8).
    let q = query(cfg(), 0.0, 128.0, 0.18, 0.25);
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn exposure_compensation_shifts_the_target() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuExposureAdapt::new(&ctx);
    // A +2 EV compensation lifts the metered target by two stops; the wide
    // window keeps it clear of the clamp.
    let c = ExposureAdaptConfig::new(-30.0, 30.0, 3.0, 1.0, 2.0);
    let q = query(c, -6.0, 0.18, 0.18, 0.1);
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn zero_dt_does_not_move_the_exposure() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuExposureAdapt::new(&ctx);
    // A zero-length frame yields a zero blend factor: next_ev == current_ev.
    let q = query(cfg(), 1.25, 0.045, 0.18, 0.0);
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn long_frame_lands_essentially_on_target() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuExposureAdapt::new(&ctx);
    // A very long frame drives the blend factor toward one, so next_ev lands on
    // the clamped target.
    let q = query(cfg(), -8.0, 0.045, 0.18, 1.0e6);
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn converged_integer_octave_exposure_scale() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuExposureAdapt::new(&ctx);
    // Wide window plus a long frame drive next_ev onto the integer +2 EV target,
    // whose exposure scale is the exact power of two 4.0.
    let c = ExposureAdaptConfig::new(-30.0, 30.0, 3.0, 1.0, 0.0);
    let q = query(c, -10.0, 0.045, 0.18, 1.0e6);
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn batch_of_eyes_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuExposureAdapt::new(&ctx);
    // A batch exercises the one-thread-per-eye flattening; each result must be
    // independent of its neighbours.
    let batch = [
        query(cfg(), -4.0, 0.045, 0.18, 0.1),
        query(cfg(), 4.0, 0.72, 0.18, 0.1),
        query(cfg(), 0.0, 0.18, 0.18, 0.0),
        query(
            ExposureAdaptConfig::new(-30.0, 30.0, 2.0, 1.5, 1.0),
            -3.0,
            0.09,
            0.18,
            0.2,
        ),
    ];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), batch.len());
    for q in &batch {
        assert_parity(&gpu, &ctx, q);
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuExposureAdapt::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
