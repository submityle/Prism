//! Real-device parity for the §24.1 non-linear sRGB <-> HSL / HSV cylindrical
//! color conversion shader mirror.
//!
//! The forward kernels convert each non-linear sRGB `vec4<f32>` to HSL/HSV on a
//! real `GPU` from the single-sourced
//! [`WGSL_HSL`](prism_math::shader_mirror::WGSL_HSL) fragment, and the inverse
//! kernels convert back. Both are diffed against the CPU reference
//! [`prism_math::color::Hsla`] / [`prism_math::color::Hsva`].
//!
//! Hue is numerically ill-conditioned near gray (chroma -> 0), so the forward
//! checks use a loose per-axis tolerance and the suite's strong invariant is
//! the sRGB -> model -> sRGB round-trip, which recovers the input tightly
//! regardless of hue stability. Alpha is carried through unchanged and must
//! match exactly. The suite skips gracefully when no adapter is available.

use prism_math::color::{Hsla, Hsva, Srgba};
use prism_math_gpu::GpuContext;
use prism_math_gpu::GpuHsl;

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

/// A deterministic batch of sRGB colors in `[0, 1]^3` plus alpha, with a few
/// saturated / gray corners appended.
fn sample_srgb() -> Vec<[f32; 4]> {
    let mut src: Vec<[f32; 4]> = Vec::new();
    let mut seed = 0x0c0f_fee1u32;
    for _ in 0..4096 {
        let c = |s: &mut u32| (lcg(s) % 1_000_001) as f32 / 1_000_000.0;
        src.push([c(&mut seed), c(&mut seed), c(&mut seed), c(&mut seed)]);
    }
    src.push([0.0, 0.0, 0.0, 1.0]);
    src.push([1.0, 1.0, 1.0, 0.25]);
    src.push([0.5, 0.5, 0.5, 0.5]);
    src.push([1.0, 0.0, 0.0, 0.5]);
    src.push([0.0, 1.0, 0.0, 0.75]);
    src.push([0.0, 0.0, 1.0, 0.125]);
    src.push([1.0, 1.0, 0.0, 1.0]);
    src.push([0.0, 1.0, 1.0, 1.0]);
    src.push([1.0, 0.0, 1.0, 1.0]);
    src
}

#[test]
fn srgb_to_hsl_matches_cpu_within_tolerance() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuHsl::new(&ctx);
    let src = sample_srgb();

    let gpu = kernel.srgb_to_hsl(&ctx, &src);
    assert_eq!(gpu.len(), src.len());
    for (v, &g) in src.iter().zip(gpu.iter()) {
        let cpu = Hsla::from_srgb(Srgba::new(v[0], v[1], v[2], v[3]));
        // Saturation/lightness are well conditioned.
        assert!(
            close(g[1], cpu.saturation, 1.0e-4),
            "sRGB->HSL saturation drift at {v:?}: gpu {} cpu {}",
            g[1],
            cpu.saturation
        );
        assert!(
            close(g[2], cpu.lightness, 1.0e-4),
            "sRGB->HSL lightness drift at {v:?}: gpu {} cpu {}",
            g[2],
            cpu.lightness
        );
        // Hue only when meaningfully chromatic (saturation guards gray).
        if cpu.saturation > 1.0e-3 {
            assert!(
                close(g[0], cpu.hue, 1.0e-2),
                "sRGB->HSL hue drift at {v:?}: gpu {} cpu {}",
                g[0],
                cpu.hue
            );
        }
        assert_eq!(g[3], cpu.alpha, "alpha must pass through exactly at {v:?}");
    }
}

#[test]
fn srgb_to_hsv_matches_cpu_within_tolerance() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuHsl::new(&ctx);
    let src = sample_srgb();

    let gpu = kernel.srgb_to_hsv(&ctx, &src);
    assert_eq!(gpu.len(), src.len());
    for (v, &g) in src.iter().zip(gpu.iter()) {
        let cpu = Hsva::from_srgb(Srgba::new(v[0], v[1], v[2], v[3]));
        assert!(
            close(g[1], cpu.saturation, 1.0e-4),
            "sRGB->HSV saturation drift at {v:?}: gpu {} cpu {}",
            g[1],
            cpu.saturation
        );
        assert!(
            close(g[2], cpu.value, 1.0e-4),
            "sRGB->HSV value drift at {v:?}: gpu {} cpu {}",
            g[2],
            cpu.value
        );
        if cpu.saturation > 1.0e-3 {
            assert!(
                close(g[0], cpu.hue, 1.0e-2),
                "sRGB->HSV hue drift at {v:?}: gpu {} cpu {}",
                g[0],
                cpu.hue
            );
        }
        assert_eq!(g[3], cpu.alpha, "alpha must pass through exactly at {v:?}");
    }
}

#[test]
fn round_trip_srgb_hsl_srgb_recovers_input() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuHsl::new(&ctx);
    let src = sample_srgb();

    let hsl = kernel.srgb_to_hsl(&ctx, &src);
    let back = kernel.hsl_to_srgb(&ctx, &hsl);
    assert_eq!(back.len(), src.len());
    for (v, &g) in src.iter().zip(back.iter()) {
        for lane in 0..3 {
            assert!(
                close(g[lane], v[lane], 1.0e-4),
                "HSL round-trip drift at {v:?} lane {lane}: got {}",
                g[lane]
            );
        }
        assert_eq!(g[3], v[3], "alpha must survive the round trip at {v:?}");
    }
}

#[test]
fn round_trip_srgb_hsv_srgb_recovers_input() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuHsl::new(&ctx);
    let src = sample_srgb();

    let hsv = kernel.srgb_to_hsv(&ctx, &src);
    let back = kernel.hsv_to_srgb(&ctx, &hsv);
    assert_eq!(back.len(), src.len());
    for (v, &g) in src.iter().zip(back.iter()) {
        for lane in 0..3 {
            assert!(
                close(g[lane], v[lane], 1.0e-4),
                "HSV round-trip drift at {v:?} lane {lane}: got {}",
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
    let kernel = GpuHsl::new(&ctx);
    assert!(kernel.srgb_to_hsl(&ctx, &[]).is_empty());
    assert!(kernel.hsl_to_srgb(&ctx, &[]).is_empty());
    assert!(kernel.srgb_to_hsv(&ctx, &[]).is_empty());
    assert!(kernel.hsv_to_srgb(&ctx, &[]).is_empty());
}
