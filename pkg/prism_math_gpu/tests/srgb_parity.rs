//! Real-device parity for the §24.1 sRGB electro-optical transfer-function
//! shader mirror.
//!
//! The decode kernel maps non-linear sRGB components to linear light on a real
//! `GPU` from the single-sourced
//! [`WGSL_SRGB`](prism_math::shader_mirror::WGSL_SRGB) fragment
//! (`prism_srgb_to_linear`) and the encode kernel maps the inverse
//! (`prism_linear_to_srgb`). Both are diffed against the CPU exact piecewise
//! reference [`prism_math::color::transfer`].
//!
//! The breakpoint literals are identical on both sides, so a given input takes
//! the same branch; the linear segment matches to tolerance and the power
//! segment is a fast-math `pow` versus deterministic `libm::powf`, a documented
//! `1e-5` tolerance rather than a bit contract. The suite skips gracefully when
//! no adapter is available.

use prism_math::color::transfer::{linear_to_srgb, srgb_to_linear};
use prism_math_gpu::GpuContext;
use prism_math_gpu::GpuSrgbTransfer;

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

/// Absolute+relative closeness for the fast-math `pow` segment.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    diff <= 1e-5 + 1e-5 * a.abs().max(b.abs())
}

/// A spread of in-gamut `[0, 1]` samples plus the two breakpoints and a few
/// slightly-out-of-range values the curves still handle monotonically.
fn samples() -> Vec<f32> {
    // 0, 1, and the two curve breakpoints up front.
    let mut v: Vec<f32> = vec![0.0, 1.0, 0.003_130_8, 0.040_448_237];
    let mut seed = 0x5eed_1234u32;
    for i in 0..=1000u32 {
        // Dense sweep across [0, 1].
        v.push(i as f32 / 1000.0);
        // Random in [0, 1.5) to exercise above-unity inputs too.
        let r = (lcg(&mut seed) % 1_500_000) as f32 / 1_000_000.0;
        v.push(r);
    }
    v
}

#[test]
fn srgb_to_linear_matches_cpu_reference() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuSrgbTransfer::new(&ctx);

    let src = samples();
    let gpu = kernel.srgb_to_linear(&ctx, &src);
    assert_eq!(gpu.len(), src.len());
    for (&c, &g) in src.iter().zip(gpu.iter()) {
        let cpu = srgb_to_linear(c);
        assert!(close(g, cpu), "decode drift at c={c}: gpu={g} cpu={cpu}");
    }
}

#[test]
fn linear_to_srgb_matches_cpu_reference() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuSrgbTransfer::new(&ctx);

    let src = samples();
    let gpu = kernel.linear_to_srgb(&ctx, &src);
    assert_eq!(gpu.len(), src.len());
    for (&c, &g) in src.iter().zip(gpu.iter()) {
        let cpu = linear_to_srgb(c);
        assert!(close(g, cpu), "encode drift at c={c}: gpu={g} cpu={cpu}");
    }
}

#[test]
fn round_trip_recovers_input() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuSrgbTransfer::new(&ctx);

    // Treat inputs as non-linear sRGB, decode to linear on the GPU, then
    // re-encode on the GPU; the result should recover the original to the
    // fast-math tolerance across the in-gamut range.
    let mut src: Vec<f32> = Vec::new();
    for i in 0..=1000u32 {
        src.push(i as f32 / 1000.0);
    }
    let linear = kernel.srgb_to_linear(&ctx, &src);
    let back = kernel.linear_to_srgb(&ctx, &linear);
    assert_eq!(back.len(), src.len());
    for (&c, &b) in src.iter().zip(back.iter()) {
        assert!(close(b, c), "round-trip drift: in={c} out={b}");
    }
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuSrgbTransfer::new(&ctx);
    assert!(kernel.srgb_to_linear(&ctx, &[]).is_empty());
    assert!(kernel.linear_to_srgb(&ctx, &[]).is_empty());
}
