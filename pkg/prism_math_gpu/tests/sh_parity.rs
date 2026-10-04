//! Real-device parity for the §24.1 order-3 spherical-harmonic evaluation
//! shader mirror.
//!
//! Each test reconstructs an order-3 (16-coefficient) real spherical harmonic
//! in a direction on the CPU ([`prism_math::Sh3::eval`]), does the same
//! evaluation on a real `GPU` from the single-sourced
//! [`WGSL_SH3_EVAL`](prism_math::shader_mirror::WGSL_SH3_EVAL) fragment, and
//! asserts the scalar agrees within a small absolute+relative tolerance. The
//! evaluation is a sum of 16 polynomial-weighted multiply-adds; Metal compiles
//! WGSL under fast-math (FMA contraction / reassociation), so the §24.1
//! contract is a *tolerance* round-trip, not a bit-exact one. The suite skips
//! gracefully when no adapter is available so it still passes on a device-less
//! CI image while running the full dispatch on a real `GPU`.

use prism_math::Sh3;
use prism_math::Vec3;
use prism_math_gpu::GpuContext;
use prism_math_gpu::GpuSh3Eval;

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

/// Absolute+relative tolerance comparison for the reconstructed scalar.
///
/// The 16-term accumulate carries more rounding than a single multiply-add, so
/// the epsilon matches the skinning suite's `5e-5`: it still rejects any real
/// algorithm/operand-order/layout drift while admitting the fast-math rounding.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs()).max(1.0);
    diff <= 5.0e-5 * scale
}

fn assert_close(cpu: f32, gpu: f32, dir: Vec3) {
    assert!(
        close(cpu, gpu),
        "sh3 eval drifted at dir=({}, {}, {}): cpu={cpu} gpu={gpu}",
        dir.x, dir.y, dir.z,
    );
}

/// A deterministic, non-trivial coefficient set exercising all four bands.
const COEFFS: [f32; 16] = [
    1.0, -0.5, 0.25, 0.75, 0.4, -0.3, 0.2, -0.6, 0.15, 0.35, -0.45, 0.55, -0.65, 0.1, -0.2, 0.3,
];

fn check(ctx: &GpuContext, kernel: &GpuSh3Eval, coeffs: &[f32; 16], dir: Vec3) {
    let d = dir.normalize();
    let cpu = Sh3::from_coeffs(*coeffs).eval(d);
    let gpu = kernel.eval(ctx, coeffs, d);
    assert_close(cpu, gpu, d);
}

#[test]
fn axis_directions_match_cpu_reference() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuSh3Eval::new(&ctx);
    check(&ctx, &kernel, &COEFFS, Vec3::new(1.0, 0.0, 0.0));
    check(&ctx, &kernel, &COEFFS, Vec3::new(0.0, 1.0, 0.0));
    check(&ctx, &kernel, &COEFFS, Vec3::new(0.0, 0.0, 1.0));
    check(&ctx, &kernel, &COEFFS, Vec3::new(0.0, 0.0, -1.0));
}

#[test]
fn oblique_directions_match_cpu_reference() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuSh3Eval::new(&ctx);
    check(&ctx, &kernel, &COEFFS, Vec3::new(1.0, 1.0, 1.0));
    check(&ctx, &kernel, &COEFFS, Vec3::new(-2.0, 1.0, 0.5));
    check(&ctx, &kernel, &COEFFS, Vec3::new(0.3, -0.7, 0.6));
    check(&ctx, &kernel, &COEFFS, Vec3::new(-1.0, -1.0, 2.0));
}

#[test]
fn single_band_sets_match_cpu_reference() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuSh3Eval::new(&ctx);
    // Band-0 only (constant ambient): result must be the DC term in every
    // direction.
    let mut dc = [0.0f32; 16];
    dc[0] = 2.0;
    check(&ctx, &kernel, &dc, Vec3::new(0.4, 0.8, -0.2));
    // Band-3 only (the highest, most oscillatory band).
    let mut band3 = [0.0f32; 16];
    for c in band3.iter_mut().take(16).skip(9) {
        *c = 0.5;
    }
    check(&ctx, &kernel, &band3, Vec3::new(1.0, -0.5, 0.3));
    check(&ctx, &kernel, &band3, Vec3::new(-0.6, 0.2, 0.9));
}
