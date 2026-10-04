//! Real-device parity for the §24.1 correlated-color-temperature -> linear-sRGB
//! shader mirror (Planckian locus).
//!
//! The forward kernel maps each Kelvin scalar to a unit-luminance linear-sRGB
//! `vec4<f32>` on a real `GPU` from the single-sourced
//! [`WGSL_TEMPERATURE`](prism_math::shader_mirror::WGSL_TEMPERATURE) fragment,
//! and the result is diffed against the CPU reference
//! [`prism_math::color::LinearRgba::from_temperature`].
//!
//! The math is a clamp, a piecewise cubic, two divides, and the XYZ->linear
//! matrix (ordinary FMA, no transcendental), and the spline branch cutoffs are
//! on the exact clamped Kelvin input, so the GPU and CPU select the same
//! segment. Parity is a tight absolute+relative tolerance that only absorbs
//! Metal fast-math last-ULP rounding. The suite skips gracefully when no
//! adapter is available.

use prism_math::color::LinearRgba;
use prism_math_gpu::GpuContext;
use prism_math_gpu::GpuTemperature;

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

/// A small deterministic linear-congruential sequence.
fn lcg(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *seed
}

/// Returns `true` if `a` and `b` agree within a combined absolute+relative
/// tolerance.
fn close(a: f32, b: f32, tol: f32) -> bool {
    let diff = (a - b).abs();
    diff <= tol + tol * a.abs().max(b.abs())
}

/// A deterministic sweep across and beyond the supported Kelvin range so the
/// clamp, both `x` cubics, and all three `y` cubics are exercised.
fn sample_kelvin() -> Vec<f32> {
    let mut src: Vec<f32> = Vec::new();
    let mut seed = 0x7e3a_1c05u32;
    for _ in 0..4096 {
        // Spread across roughly [1000, 27000] K, overshooting the clamp bounds.
        let r = (lcg(&mut seed) % 26_001) as f32;
        src.push(1000.0 + r);
    }
    // Segment boundaries and clamp corners.
    for &k in &[
        500.0, 1667.0, 1800.0, 2222.0, 2222.5, 3000.0, 4000.0, 4000.5, 5000.0, 6500.0, 10000.0,
        25000.0, 25001.0, 30000.0,
    ] {
        src.push(k);
    }
    src
}

#[test]
fn temperature_to_linear_matches_cpu_within_tolerance() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuTemperature::new(&ctx);
    let src = sample_kelvin();

    let gpu = kernel.temperature_to_linear(&ctx, &src);
    assert_eq!(gpu.len(), src.len());
    for (&k, &g) in src.iter().zip(gpu.iter()) {
        let cpu = LinearRgba::from_temperature(k);
        let cpu = [cpu.red, cpu.green, cpu.blue, cpu.alpha];
        for lane in 0..3 {
            assert!(
                close(g[lane], cpu[lane], 1.0e-4),
                "temperature->linear drift at {k} K lane {lane}: gpu {} cpu {}",
                g[lane],
                cpu[lane]
            );
        }
        assert_eq!(g[3], cpu[3], "alpha must be a constant 1.0 at {k} K");
    }
}

#[test]
fn out_of_gamut_components_are_clamped_nonnegative() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuTemperature::new(&ctx);
    // Very warm temperatures push blue negative before the clamp; the kernel
    // must never emit a negative component.
    let src: Vec<f32> = vec![1667.0, 1800.0, 2000.0, 2500.0];
    let gpu = kernel.temperature_to_linear(&ctx, &src);
    for (&k, &g) in src.iter().zip(gpu.iter()) {
        for (lane, &c) in g.iter().take(3).enumerate() {
            assert!(
                c >= 0.0,
                "component must be clamped non-negative at {k} K lane {lane}: got {c}"
            );
        }
    }
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuTemperature::new(&ctx);
    assert!(kernel.temperature_to_linear(&ctx, &[]).is_empty());
}
