//! Real-device parity for the §24.1 linear-sRGB <-> CIE 1931 XYZ (D65)
//! conversion shader mirror.
//!
//! The forward kernel converts each linear-sRGB `vec4<f32>` to CIE XYZ on a
//! real `GPU` from the single-sourced
//! [`WGSL_XYZ`](prism_math::shader_mirror::WGSL_XYZ) fragment, and the inverse
//! kernel converts back. Both are diffed against the CPU reference
//! [`prism_math::color::LinearRgba::to_xyz`] / `from_xyz`.
//!
//! Both directions are a single 3x3 matrix multiply (ordinary FMA, no
//! transcendental), so the GPU and CPU differ only by Metal fast-math last-ULP
//! rounding; the tests use a tight absolute+relative tolerance rather than
//! bit-exact equality. Alpha is carried through unchanged and must match
//! exactly. The suite skips gracefully when no adapter is available.

use prism_math::color::LinearRgba;
use prism_math_gpu::GpuContext;
use prism_math_gpu::GpuXyz;

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
/// tolerance absorbing fast-math last-ULP rounding.
fn close(a: f32, b: f32, tol: f32) -> bool {
    let diff = (a - b).abs();
    diff <= tol + tol * a.abs().max(b.abs())
}

#[test]
fn linear_to_xyz_matches_cpu_within_tolerance() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuXyz::new(&ctx);

    let mut src: Vec<[f32; 4]> = Vec::new();
    let mut seed = 0x0c1e_1931u32;
    for _ in 0..4096 {
        let c = |s: &mut u32| (lcg(s) % 1_000_001) as f32 / 1_000_000.0;
        src.push([c(&mut seed), c(&mut seed), c(&mut seed), c(&mut seed)]);
    }
    src.push([0.0, 0.0, 0.0, 1.0]);
    src.push([1.0, 1.0, 1.0, 0.25]);
    src.push([1.0, 0.0, 0.0, 0.5]);
    src.push([0.0, 1.0, 0.0, 0.75]);
    src.push([0.0, 0.0, 1.0, 0.125]);

    let gpu = kernel.linear_to_xyz(&ctx, &src);
    assert_eq!(gpu.len(), src.len());
    for (v, &g) in src.iter().zip(gpu.iter()) {
        let cpu = LinearRgba::new(v[0], v[1], v[2], v[3]).to_xyz().to_array();
        for lane in 0..3 {
            assert!(
                close(g[lane], cpu[lane], 1.0e-5),
                "linear->xyz drift at {v:?} lane {lane}: gpu {} cpu {}",
                g[lane],
                cpu[lane]
            );
        }
        assert_eq!(g[3], cpu[3], "alpha must pass through exactly at {v:?}");
    }
}

#[test]
fn xyz_to_linear_matches_cpu_within_tolerance() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuXyz::new(&ctx);

    // Feed real XYZ values produced by the CPU forward transform.
    let mut xyz: Vec<[f32; 4]> = Vec::new();
    let mut seed = 0x5eed_1931u32;
    for _ in 0..4096 {
        let c = |s: &mut u32| (lcg(s) % 1_000_001) as f32 / 1_000_000.0;
        let lin = LinearRgba::new(c(&mut seed), c(&mut seed), c(&mut seed), c(&mut seed));
        xyz.push(lin.to_xyz().to_array());
    }

    let gpu = kernel.xyz_to_linear(&ctx, &xyz);
    assert_eq!(gpu.len(), xyz.len());
    for (v, &g) in xyz.iter().zip(gpu.iter()) {
        let cpu = LinearRgba::from_xyz(prism_math::color::Xyza::from_array(*v));
        let cpu = [cpu.red, cpu.green, cpu.blue, cpu.alpha];
        for lane in 0..3 {
            assert!(
                close(g[lane], cpu[lane], 1.0e-5),
                "xyz->linear drift at {v:?} lane {lane}: gpu {} cpu {}",
                g[lane],
                cpu[lane]
            );
        }
        assert_eq!(g[3], cpu[3], "alpha must pass through exactly at {v:?}");
    }
}

#[test]
fn round_trip_linear_xyz_linear_recovers_input() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuXyz::new(&ctx);

    let mut src: Vec<[f32; 4]> = Vec::new();
    let mut seed = 0x1931_abcdu32;
    for _ in 0..4096 {
        let c = |s: &mut u32| (lcg(s) % 1_000_001) as f32 / 1_000_000.0;
        src.push([c(&mut seed), c(&mut seed), c(&mut seed), c(&mut seed)]);
    }

    let xyz = kernel.linear_to_xyz(&ctx, &src);
    let back = kernel.xyz_to_linear(&ctx, &xyz);
    assert_eq!(back.len(), src.len());
    for (v, &g) in src.iter().zip(back.iter()) {
        for lane in 0..3 {
            assert!(
                close(g[lane], v[lane], 1.0e-4),
                "round-trip drift at {v:?} lane {lane}: got {}",
                g[lane]
            );
        }
        assert_eq!(g[3], v[3], "alpha must survive the round trip at {v:?}");
    }
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuXyz::new(&ctx);
    assert!(kernel.linear_to_xyz(&ctx, &[]).is_empty());
    assert!(kernel.xyz_to_linear(&ctx, &[]).is_empty());
}
