//! Real-device parity for the §24.1 linear-sRGB <-> `OkLab` perceptual color
//! conversion shader mirror.
//!
//! The forward kernel converts each linear-sRGB `vec4<f32>` to `OkLab` on a
//! real `GPU` from the single-sourced
//! [`WGSL_OKLAB`](prism_math::shader_mirror::WGSL_OKLAB) fragment, and the
//! inverse kernel converts back. Both are diffed against the CPU reference
//! [`prism_math::color::oklab::Oklaba`].
//!
//! The matrix multiplies and the inverse cube are ordinary FMA arithmetic, but
//! the forward direction needs a cube root, which WGSL lacks as a built-in;
//! `prism_cbrt` composes it as `sign(x)·pow(|x|, 1/3)` while the CPU reference
//! uses `libm::cbrt`. The two therefore differ by a small `pow` rounding error,
//! a documented honest boundary checked with an absolute+relative tolerance
//! rather than bit-exact equality. Alpha is carried through unchanged and must
//! match exactly. The suite skips gracefully when no adapter is available.

use prism_math::color::{LinearRgba, Oklaba};
use prism_math_gpu::GpuContext;
use prism_math_gpu::GpuOklab;

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
/// tolerance; the `pow`-vs-`cbrt` honest boundary needs a little slack.
fn close(a: f32, b: f32, tol: f32) -> bool {
    let diff = (a - b).abs();
    diff <= tol + tol * a.abs().max(b.abs())
}

#[test]
fn linear_to_oklab_matches_cpu_within_tolerance() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuOklab::new(&ctx);

    let mut src: Vec<[f32; 4]> = Vec::new();
    let mut seed = 0x0c0f_fee1u32;
    for _ in 0..4096 {
        let c = |s: &mut u32| (lcg(s) % 1_000_001) as f32 / 1_000_000.0;
        src.push([c(&mut seed), c(&mut seed), c(&mut seed), c(&mut seed)]);
    }
    // A few corners: black, white, and primaries.
    src.push([0.0, 0.0, 0.0, 1.0]);
    src.push([1.0, 1.0, 1.0, 0.25]);
    src.push([1.0, 0.0, 0.0, 0.5]);
    src.push([0.0, 1.0, 0.0, 0.75]);
    src.push([0.0, 0.0, 1.0, 0.125]);

    let gpu = kernel.linear_to_oklab(&ctx, &src);
    assert_eq!(gpu.len(), src.len());
    for (v, &g) in src.iter().zip(gpu.iter()) {
        let cpu = Oklaba::from_linear(LinearRgba::new(v[0], v[1], v[2], v[3])).to_array();
        for lane in 0..3 {
            assert!(
                close(g[lane], cpu[lane], 1.0e-3),
                "linear->oklab drift at {v:?} lane {lane}: gpu {} cpu {}",
                g[lane],
                cpu[lane]
            );
        }
        assert_eq!(g[3], cpu[3], "alpha must pass through exactly at {v:?}");
    }
}

#[test]
fn oklab_to_linear_matches_cpu_within_tolerance() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuOklab::new(&ctx);

    // Feed real OkLab values produced by the CPU forward transform so the
    // inverse kernel is exercised on the manifold it will see in practice.
    let mut lab: Vec<[f32; 4]> = Vec::new();
    let mut seed = 0x5eed_0a7bu32;
    for _ in 0..4096 {
        let c = |s: &mut u32| (lcg(s) % 1_000_001) as f32 / 1_000_000.0;
        let lin = LinearRgba::new(c(&mut seed), c(&mut seed), c(&mut seed), c(&mut seed));
        lab.push(Oklaba::from_linear(lin).to_array());
    }

    let gpu = kernel.oklab_to_linear(&ctx, &lab);
    assert_eq!(gpu.len(), lab.len());
    for (v, &g) in lab.iter().zip(gpu.iter()) {
        let cpu = Oklaba::from_array(*v).to_linear();
        let cpu = [cpu.red, cpu.green, cpu.blue, cpu.alpha];
        for lane in 0..3 {
            assert!(
                close(g[lane], cpu[lane], 1.0e-3),
                "oklab->linear drift at {v:?} lane {lane}: gpu {} cpu {}",
                g[lane],
                cpu[lane]
            );
        }
        assert_eq!(g[3], cpu[3], "alpha must pass through exactly at {v:?}");
    }
}

#[test]
fn round_trip_linear_oklab_linear_recovers_input() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuOklab::new(&ctx);

    let mut src: Vec<[f32; 4]> = Vec::new();
    let mut seed = 0x1234_abcdu32;
    for _ in 0..4096 {
        let c = |s: &mut u32| (lcg(s) % 1_000_001) as f32 / 1_000_000.0;
        src.push([c(&mut seed), c(&mut seed), c(&mut seed), c(&mut seed)]);
    }

    let lab = kernel.linear_to_oklab(&ctx, &src);
    let back = kernel.oklab_to_linear(&ctx, &lab);
    assert_eq!(back.len(), src.len());
    for (v, &g) in src.iter().zip(back.iter()) {
        for lane in 0..3 {
            assert!(
                close(g[lane], v[lane], 2.0e-3),
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
    let kernel = GpuOklab::new(&ctx);
    assert!(kernel.linear_to_oklab(&ctx, &[]).is_empty());
    assert!(kernel.oklab_to_linear(&ctx, &[]).is_empty());
}
