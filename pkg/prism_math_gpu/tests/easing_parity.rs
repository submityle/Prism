//! Real-device parity for the §24.1 scalar easing-function shader mirror.
//!
//! Each kernel dispatch evaluates one easing op over a batch of scalar
//! parameters on a real `GPU` from the single-sourced
//! [`WGSL_EASING`](prism_math::shader_mirror::WGSL_EASING) dispatcher
//! (`prism_ease`) and is diffed against the matching CPU reference in
//! [`prism_math::curve::easing`].
//!
//! The polynomial easings match to a tight FMA tolerance; the sinusoidal and
//! exponential easings are a fast-math `cos`/`sin`/`pow` versus deterministic
//! `libm`, a documented absolute+relative tolerance rather than a bit
//! contract. The branch literals (clamp, `t <= 0`/`t >= 1`/`t < 0.5`) are
//! identical on both sides, so a given input takes the same branch. The suite
//! skips gracefully when no adapter is available.

use prism_math::curve::easing;
use prism_math_gpu::{Ease, GpuContext, GpuEasing};

/// Acquires a device, or prints a skip note and returns `None` on hosts without
/// a usable adapter.
#[expect(
    clippy::print_stderr,
    reason = "test-only skip note when no GPU adapter is present"
)]
fn with_gpu() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping: no usable GPU adapter on this host");
            None
        }
    }
}

/// Absolute+relative closeness covering the fast-math transcendental segments.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    diff <= 1e-4 + 1e-4 * a.abs().max(b.abs())
}

/// The complete op table paired with its CPU reference function.
fn ops() -> [(Ease, fn(f32) -> f32); 14] {
    [
        (Ease::Smoothstep, easing::smoothstep),
        (Ease::Smootherstep, easing::smootherstep),
        (Ease::QuadIn, easing::quad_in),
        (Ease::QuadOut, easing::quad_out),
        (Ease::QuadInOut, easing::quad_in_out),
        (Ease::CubicIn, easing::cubic_in),
        (Ease::CubicOut, easing::cubic_out),
        (Ease::CubicInOut, easing::cubic_in_out),
        (Ease::SineIn, easing::sine_in),
        (Ease::SineOut, easing::sine_out),
        (Ease::SineInOut, easing::sine_in_out),
        (Ease::ExpoIn, easing::expo_in),
        (Ease::ExpoOut, easing::expo_out),
        (Ease::ExpoInOut, easing::expo_in_out),
    ]
}

/// A dense sweep across `[0, 1]`, the exact boundaries and the `0.5` branch
/// seam, plus a few out-of-range values that exercise the clamp/`<= 0`/`>= 1`
/// branches identically on both sides.
fn samples() -> Vec<f32> {
    let mut v: Vec<f32> = vec![0.0, 0.5, 1.0, -0.25, 1.25];
    for i in 0..=4099u32 {
        v.push(i as f32 / 4099.0);
    }
    v
}

#[test]
fn every_easing_matches_cpu_reference() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuEasing::new(&ctx);
    let src = samples();

    for (ease, cpu_fn) in ops() {
        let gpu = kernel.map(&ctx, ease, &src);
        assert_eq!(gpu.len(), src.len());
        for (&t, &g) in src.iter().zip(gpu.iter()) {
            let cpu = cpu_fn(t);
            assert!(close(g, cpu), "drift for {ease:?} at t={t}: gpu={g} cpu={cpu}");
        }
    }
}

#[test]
fn boundary_conditions_hold_on_device() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuEasing::new(&ctx);
    let ends = [0.0f32, 1.0];

    for (ease, _) in ops() {
        let out = kernel.map(&ctx, ease, &ends);
        // Every easing satisfies f(0) == 0 and f(1) == 1 on the device.
        assert!(close(out[0], 0.0), "{ease:?} f(0) = {} != 0", out[0]);
        assert!(close(out[1], 1.0), "{ease:?} f(1) = {} != 1", out[1]);
    }
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuEasing::new(&ctx);
    assert!(kernel.map(&ctx, Ease::Smoothstep, &[]).is_empty());
}
