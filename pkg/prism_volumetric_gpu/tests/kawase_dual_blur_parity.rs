//! Real-device parity for the dual-`Kawase` blur twin:
//! [`GpuKawaseDualBlur`](prism_volumetric_gpu::kawase_dual_blur::GpuKawaseDualBlur)
//! must reproduce the `CPU` golden
//! [`dual_blur`](prism_render_architecture::particle::kawase_dual_blur::KawaseImage::dual_blur)
//! across empty and zero-dimension images, a `1x1` image, even and odd extents
//! (so the `div_ceil` pyramid halving and the clamp-to-edge sampler are both
//! exercised at a boundary), multi-level down/up round trips and a batch of
//! random images at several resolutions compared `texel`-for-`texel`.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion
//!
//! Each output `texel` is a fixed, non-reorderable sequence of bilinear lerps,
//! weighted adds and one normalizing multiply, so `CPU` and `GPU` evaluate the
//! same closed form in the same order. They are not bit-exact: a `GPU` may fuse
//! a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place, and that perturbation
//! compounds across the down/up pyramid. The comparison therefore allows
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3` — loose enough to admit a legal
//! fused multiply-add contraction summed over the pyramid, yet tight enough to
//! fail a genuinely wrong port (a swapped tap sign, a dropped diagonal, a wrong
//! normalizer, a missing edge clamp).
//!
//! Provenance: 孪生自本仓 `particle::kawase_dual_blur`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::kawase_dual_blur::KawaseImage;
use prism_volumetric_gpu::kawase_dual_blur::{
    GpuKawaseDualBlur, GpuKawaseDualBlurQuery, GpuKawaseDualBlurResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack, compounded over the pyramid, while
/// still failing a genuinely wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
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
    // Knuth multiplier / increment; the shift takes the high bits where the
    // generator mixes best.
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    // 24 usable mantissa bits mapped onto [0, 1).
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Builds a `width * height` reference image whose pixels are pseudo-random
/// `HDR` values in `[0, scale)` drawn from `state`.
fn random_image(width: usize, height: usize, scale: f32, state: &mut u64) -> KawaseImage {
    let mut pixels = Vec::with_capacity(width * height);
    for _ in 0..(width * height) {
        pixels.push([lcg(state) * scale, lcg(state) * scale, lcg(state) * scale]);
    }
    KawaseImage::new(width, height, pixels)
}

/// Builds a `width * height` constant reference image.
fn uniform(width: usize, height: usize, value: f32) -> KawaseImage {
    KawaseImage::new(width, height, vec![[value, value, value]; width * height])
}

/// Builds a twin request mirroring the reference `image` and pyramid settings.
fn query_of(image: &KawaseImage, offset: f32, passes: u32) -> GpuKawaseDualBlurQuery {
    GpuKawaseDualBlurQuery {
        width: image.width,
        height: image.height,
        pixels: image.pixels.clone(),
        offset,
        passes,
    }
}

/// Runs the `GPU` dual-`Kawase` chain and asserts `texel`-for-`texel` parity
/// against the `CPU` golden [`KawaseImage::dual_blur`], returning the `GPU`
/// result for any extra per-test assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuKawaseDualBlur,
    image: &KawaseImage,
    offset: f32,
    passes: u32,
) -> GpuKawaseDualBlurResult {
    let got = gpu.evaluate(ctx, &query_of(image, offset, passes));
    let want = image.dual_blur(passes as usize, offset);

    assert_eq!(got.width, want.width, "width must match the reference");
    assert_eq!(got.height, want.height, "height must match the reference");
    assert_eq!(
        got.pixels.len(),
        want.pixels.len(),
        "pixel count must match the reference"
    );

    for (idx, (g, w)) in got.pixels.iter().zip(want.pixels.iter()).enumerate() {
        for channel in 0..3 {
            assert!(
                close(g[channel], w[channel]),
                "texel {idx} channel {channel}: gpu {} vs cpu {} (offset {offset}, passes {passes})",
                g[channel],
                w[channel]
            );
        }
    }
    got
}

#[test]
fn empty_and_zero_dimension_images_stay_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuKawaseDualBlur::new(&ctx);
    // A zero-dimension image is degenerate; the reference returns a clone and
    // the twin must short-circuit without issuing a zero-sized dispatch.
    for &(w, h) in &[(0usize, 4usize), (4, 0), (0, 0)] {
        let image = KawaseImage::black(w, h);
        let got = gpu.evaluate(&ctx, &query_of(&image, 0.5, 3));
        assert_eq!(got.width, w);
        assert_eq!(got.height, h);
        assert!(got.pixels.is_empty(), "a degenerate image stays empty");
    }
}

#[test]
fn passes_zero_is_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuKawaseDualBlur::new(&ctx);
    let mut state = 0x1357_9bdf_0246_8ace_u64;
    let image = random_image(8, 6, 4.0, &mut state);
    // With zero passes the reference returns a clone; the twin must too, with no
    // dispatch issued.
    let got = check(&ctx, &gpu, &image, 0.5, 0);
    assert_eq!(got.pixels, image.pixels, "zero passes is the identity");
}

#[test]
fn one_by_one_round_trips() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuKawaseDualBlur::new(&ctx);
    // A 1x1 image halves to 1x1 at every level, so every tap clamps onto the
    // single texel; the chain must reproduce that constant exactly.
    let image = KawaseImage::new(1, 1, vec![[0.3, 0.7, 1.5]]);
    let got = check(&ctx, &gpu, &image, 0.5, 3);
    assert_eq!(got.width, 1);
    assert_eq!(got.height, 1);
}

#[test]
fn constant_image_survives_the_chain() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuKawaseDualBlur::new(&ctx);
    // The down and up kernels are both normalized so a constant image survives
    // unchanged; parity against the reference plus a direct constant check pins
    // the normalizers.
    let image = uniform(16, 16, 0.625);
    let got = check(&ctx, &gpu, &image, 0.5, 3);
    for pixel in &got.pixels {
        for &component in pixel {
            assert!(
                close(component, 0.625),
                "a constant image must survive the blur, got {component}"
            );
        }
    }
}

#[test]
fn even_dimensions_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuKawaseDualBlur::new(&ctx);
    let mut state = 0x2b7e_1516_28ae_d2a6_u64;
    // An even 8x8 image halves cleanly to 4x4, 2x2, 1x1 with no odd remainder.
    let image = random_image(8, 8, 3.0, &mut state);
    for passes in 1u32..=3 {
        check(&ctx, &gpu, &image, 0.5, passes);
    }
}

#[test]
fn odd_dimensions_exercise_div_ceil_and_clamp() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuKawaseDualBlur::new(&ctx);
    let mut state = 0x3243_f6a8_885a_308d_u64;
    // Odd extents force the pyramid to `div_ceil` (7 -> 4 -> 2 -> 1, 5 -> 3 ->
    // 2 -> 1) and the up chain to retrace those exact recorded extents, while
    // the fractional texel centers push several taps onto the clamp-to-edge
    // boundary.
    let image = random_image(7, 5, 2.5, &mut state);
    for passes in 1u32..=3 {
        check(&ctx, &gpu, &image, 0.5, passes);
    }
}

#[test]
fn impulse_spreads_like_the_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuKawaseDualBlur::new(&ctx);
    // A single bright texel must diffuse exactly as the reference does; the
    // parity check already pins every texel, and we additionally confirm the
    // center leaked energy to a neighbor so the test is not vacuously constant.
    let width = 8;
    let height = 8;
    let mut pixels = vec![[0.0f32; 3]; width * height];
    pixels[4 * width + 4] = [1.0, 1.0, 1.0];
    let image = KawaseImage::new(width, height, pixels);
    let got = check(&ctx, &gpu, &image, 0.5, 2);

    let center = got.pixels[4 * width + 4];
    assert!(center[0] < 1.0 - EPS, "the impulse center must darken");
    let neighbor = got.pixels[4 * width + 5];
    assert!(neighbor[0] > EPS, "energy must leak to a neighbor");
}

#[test]
fn multi_level_round_trip_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuKawaseDualBlur::new(&ctx);
    let mut state = 0xa409_3822_299f_31d0_u64;
    // A deeper pyramid (3 passes on a 24x18 image: 24 -> 12 -> 6 -> 3 and
    // 18 -> 9 -> 5 -> 3) chains six dispatches whose storage buffers feed one
    // another; parity confirms the host wired the chain in the right order.
    let image = random_image(24, 18, 5.0, &mut state);
    check(&ctx, &gpu, &image, 0.5, 3);
}

#[test]
fn random_images_multiple_resolutions_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuKawaseDualBlur::new(&ctx);
    let mut state = 0x5eed_4a7d_0bad_c0de_u64;
    // Sweep a spread of resolutions (even, odd, tall, wide, prime), pass counts
    // and offsets so the sampler, the pyramid extents and the tap offsets are
    // all exercised against random inputs, each compared texel-for-texel.
    let resolutions = [
        (2usize, 2usize),
        (3, 7),
        (5, 5),
        (6, 10),
        (9, 4),
        (13, 11),
        (16, 9),
    ];
    for &(width, height) in &resolutions {
        for passes in 1u32..=3 {
            let offset = 0.25 + lcg(&mut state) * 0.75;
            let image = random_image(width, height, 4.0, &mut state);
            check(&ctx, &gpu, &image, offset, passes);
        }
    }
}
