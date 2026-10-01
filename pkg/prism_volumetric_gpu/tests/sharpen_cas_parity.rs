//! Real-device parity for the `CAS` sharpen twin:
//! [`GpuSharpenCas`](prism_volumetric_gpu::sharpen_cas::GpuSharpenCas) must
//! reproduce the `CPU` golden
//! [`CasParams::apply`](prism_render_architecture::particle::sharpen_cas::CasParams::apply)
//! across empty and zero-dimension images, a short buffer, a `1x1` image, flat
//! images (sharpen is the identity), a single-point impulse, a strong edge, a
//! high-frequency checkerboard, the softest and sharpest `sharpness` endpoints,
//! border `texels` whose neighbors clamp to the edge, and a batch of random
//! images at several resolutions compared `texel`-for-`texel`, per `RGB`
//! channel.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL` plus the core `sqrt` built-in, so it needs
//! no optional device feature.
//!
//! # Parity criterion
//!
//! Each output `texel` channel is a fixed, non-reorderable sequence of
//! `min`/`max`, one `clamp`, one `sqrt`, multiplies, adds and two guarded
//! divides, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, and both the hardware `sqrt` and the `reciprocal` implied by
//! the divide carry a few units in the last place of rounding slack. The
//! comparison therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` — loose
//! enough to admit that legal slack, yet tight enough to fail a genuinely wrong
//! port (a dropped tap, a swapped `min`/`max`, a wrong amplitude, a lost
//! denominator guard).
//!
//! The fixtures are built so no `texel` sits on a branch-selection tie: the
//! energy-normalizing denominator `1 + 4 w` is provably in `[0.2, 1]` for every
//! `sharpness` (the peak weight is in `[-0.2, -0.125]` and the amplitude in
//! `[0, 1]`), so the near-zero denominator guard never fires on either side, and
//! the random images carry a `+0.05` floor so their local maximum stays well
//! above the `1e-6` near-black guard, keeping both sides on the same amplitude
//! branch. The flat, impulse and zero fixtures take a branch that is identical
//! on both sides by construction.
//!
//! Provenance: standard `AMD` `FidelityFX` Contrast-Adaptive Sharpening (`CAS`);
//! mirrors the `CPU` golden `prism_render_architecture::particle::sharpen_cas`;
//! no third-party engine source or derived code.

use prism_render_architecture::particle::sharpen_cas::CasParams;
use prism_volumetric_gpu::sharpen_cas::{GpuSharpenCas, SharpenCasQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate and rounds `sqrt`/`reciprocal` to within a few units in the
/// last place; `1e-4` admits that legal slack while still failing a genuinely
/// wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
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

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Builds a `width * height` image whose pixels are pseudo-random values in
/// `[base, base + scale)` drawn from `state`. The `base` floor keeps the local
/// maximum well above the `1e-6` near-black guard so both sides stay on the same
/// amplitude branch.
fn random_image(
    width: usize,
    height: usize,
    base: f32,
    scale: f32,
    state: &mut u64,
) -> Vec<[f32; 3]> {
    let mut pixels = Vec::with_capacity(width * height);
    for _ in 0..(width * height) {
        pixels.push([
            base + lcg(state) * scale,
            base + lcg(state) * scale,
            base + lcg(state) * scale,
        ]);
    }
    pixels
}

/// Runs the `GPU` `CAS` sharpen and asserts `texel`-for-`texel`, per-channel
/// parity against the `CPU` golden [`CasParams::apply`], returning the `GPU`
/// result for any extra per-test assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuSharpenCas,
    img: &[[f32; 3]],
    width: usize,
    height: usize,
    sharpness: f32,
) -> Vec<[f32; 3]> {
    // Build the parameters once so the exact same (clamped) `sharpness` f32 is
    // fed to both the CPU reference and the GPU uniform.
    let params = CasParams::new(sharpness);
    let query = SharpenCasQuery {
        img: img.to_vec(),
        width,
        height,
        sharpness: params.sharpness,
    };
    let got = gpu.eval(ctx, &query);
    let want = params.apply(img, width, height);

    assert_eq!(
        got.len(),
        want.len(),
        "pixel count must match the reference"
    );
    for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        for channel in 0..3 {
            assert!(
                close(g[channel], w[channel]),
                "texel {idx} channel {channel}: gpu {} vs cpu {} (sharpness {sharpness}, {width}x{height})",
                g[channel],
                w[channel]
            );
        }
    }
    got
}

#[test]
fn empty_and_degenerate_images_stay_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSharpenCas::new(&ctx);
    // Zero width, zero height, and a buffer shorter than width * height all
    // short-circuit to an empty vector with no dispatch, as the reference does.
    let img = vec![[0.5_f32, 0.4, 0.3]; 4];
    assert!(check(&ctx, &gpu, &img, 0, 4, 0.5).is_empty());
    assert!(check(&ctx, &gpu, &img, 4, 0, 0.5).is_empty());
    assert!(
        check(&ctx, &gpu, &img, 4, 4, 0.5).is_empty(),
        "short buffer"
    );
    assert!(check(&ctx, &gpu, &[], 0, 0, 0.5).is_empty());
}

#[test]
fn single_pixel_is_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSharpenCas::new(&ctx);
    // Every neighbor clamps back to the lone pixel, so the neighborhood is flat
    // and the pixel is returned unchanged.
    let img = vec![[0.25_f32, 0.5, 0.75]];
    let got = check(&ctx, &gpu, &img, 1, 1, 1.0);
    assert_eq!(got.len(), 1);
    for channel in 0..3 {
        assert!(close(got[0][channel], img[0][channel]), "1x1 is identity");
    }
}

#[test]
fn flat_image_is_identity_at_all_sharpness() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSharpenCas::new(&ctx);
    // A flat image has a flat neighborhood everywhere, including the clamped
    // borders and corners, so the sharpen is the identity for any sharpness.
    let img = vec![[0.33_f32, 0.66, 0.99]; 20];
    for &sharpness in &[0.0_f32, 0.5, 1.0] {
        let got = check(&ctx, &gpu, &img, 5, 4, sharpness);
        for texel in &got {
            for channel in 0..3 {
                assert!(close(texel[channel], img[0][channel]), "flat stays flat");
            }
        }
    }
}

#[test]
fn impulse_sharpens_like_the_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSharpenCas::new(&ctx);
    // A bright texel standing above a non-black field: parity already pins every
    // texel, and we additionally confirm the center is enhanced (a local maximum
    // is pushed brighter) so the test is not vacuously constant. The field is
    // kept non-zero on purpose: `CAS`'s contrast-adaptive amplitude
    // `sqrt(clamp(min(mn, 2 - mx) / mx, 0, 1))` collapses to zero when any
    // neighbor channel is `0` (a pure impulse is intentionally *not* sharpened),
    // so a nonzero surround is required to exercise a real enhancement.
    let width = 7;
    let height = 5;
    let center = 2 * width + 3;
    let mut img = vec![[0.3_f32, 0.3, 0.3]; width * height];
    img[center] = [1.0, 0.8, 0.6];
    let got = check(&ctx, &gpu, &img, width, height, 1.0);
    assert!(got[center][0] > 1.0, "the local maximum must be enhanced");
}

#[test]
fn strong_edge_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSharpenCas::new(&ctx);
    // A hard vertical step from dark to bright exercises the amplitude falloff on
    // the high-contrast seam; parity pins every texel along the edge.
    let width = 8;
    let height = 6;
    let mut img = vec![[0.0_f32; 3]; width * height];
    for y in 0..height {
        for x in 0..width {
            let v = if x < width / 2 { 0.1 } else { 0.9 };
            img[y * width + x] = [v, v, v];
        }
    }
    check(&ctx, &gpu, &img, width, height, 0.75);
}

#[test]
fn checkerboard_high_frequency_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSharpenCas::new(&ctx);
    // The highest-frequency pattern the grid can carry: every texel is a local
    // extremum against its cross neighbors, stressing the min/max statistics and
    // the amplitude term at both the dark and bright phases.
    let width = 9;
    let height = 7;
    let mut img = vec![[0.0_f32; 3]; width * height];
    for y in 0..height {
        for x in 0..width {
            let v = if (x + y) % 2 == 0 { 0.2 } else { 0.8 };
            img[y * width + x] = [v, v * 0.5, v * 0.25];
        }
    }
    for &sharpness in &[0.0_f32, 0.5, 1.0] {
        check(&ctx, &gpu, &img, width, height, sharpness);
    }
}

#[test]
fn sharpness_endpoints_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSharpenCas::new(&ctx);
    // The softest (peak weight -1/8) and sharpest (-1/5) endpoints, plus a raw
    // out-of-range value the shader must clamp exactly as the reference clamps.
    let mut state = 0x1357_9bdf_0246_8ace_u64;
    let img = random_image(6, 5, 0.05, 0.9, &mut state);
    for &sharpness in &[0.0_f32, 1.0, -3.0, 4.0] {
        check(&ctx, &gpu, &img, 6, 5, sharpness);
    }
}

#[test]
fn borders_clamp_like_the_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSharpenCas::new(&ctx);
    // A gradient whose extremes live on the border rows and columns: the
    // clamp-to-edge gather feeds each edge texel its own replicated neighbor, so
    // matching the reference here proves the shader `clamp_coord` (including the
    // saturating decrement at coordinate 0) is correct.
    let width = 5;
    let height = 4;
    let mut img = vec![[0.0_f32; 3]; width * height];
    for y in 0..height {
        for x in 0..width {
            let v = 0.05 + 0.9 * (x as f32 / (width - 1) as f32);
            let u = 0.05 + 0.9 * (y as f32 / (height - 1) as f32);
            img[y * width + x] = [v, u, 0.5 * (v + u)];
        }
    }
    check(&ctx, &gpu, &img, width, height, 0.6);
}

#[test]
fn random_images_multiple_resolutions_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSharpenCas::new(&ctx);
    let mut state = 0x5eed_4a7d_0bad_c0de_u64;
    // Sweep a spread of resolutions (even, odd, tall, wide, prime, single row,
    // single column) and sharpness values so the index arithmetic, the
    // clamp-to-edge gather and the amplitude term are all exercised against
    // random inputs, each compared texel-for-texel per channel. The +0.05 floor
    // keeps the local maximum above the near-black guard.
    let resolutions = [
        (1_usize, 8_usize),
        (8, 1),
        (2, 2),
        (3, 7),
        (5, 5),
        (6, 10),
        (13, 11),
        (16, 9),
    ];
    for &(width, height) in &resolutions {
        for step in 0u32..=4 {
            let sharpness = step as f32 / 4.0;
            let img = random_image(width, height, 0.05, 1.5, &mut state);
            check(&ctx, &gpu, &img, width, height, sharpness);
        }
    }
}
