//! Real-device parity for the screen-space light-shaft twin:
//! [`GpuLightShaft`](prism_volumetric_gpu::light_shaft::GpuLightShaft) must
//! reproduce the `CPU` golden
//! [`light_shaft`](prism_render_architecture::particle::light_shaft) across the
//! full radial-blur march — the pixel-centre `UV`, the `num_samples == 0`
//! identity short-circuit, the per-step tap chain toward the light, the
//! half-texel-centred clamp-to-edge bilinear `mask` fetch, the decay-weighted
//! accumulation and the final `exposure` gain.
//!
//! The fixtures cover a `16x16` and an `8x8` random `mask` on power-of-two
//! grids (where the pixel-centre and marched tap coordinates are exact in
//! `f32`, so the `floor` split can never tie differently on the two devices), a
//! `1x1` single-texel image, a `7x5` non-power-of-two grid driven by a uniform
//! `mask` (whose bilinear value is constant, so the odd-size index arithmetic
//! is exercised with no `floor`-tie risk), the `num_samples == 0` identity
//! pass, an off-screen `light_uv`, an extreme `density`/`decay` above one, and a
//! zeroed `mask`. Every power-of-two fixture keeps the marched taps away from a
//! texel tie because the shared `UV` and step are bit-identical on host and
//! device.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The shaft value threads through multiplies, adds and one guarded division,
//! so it is compared under tolerance (`abs_diff <= 1e-4` or `rel_diff <=
//! 1e-3`, `REL_FLOOR = 1e-6`); the integer texel addressing of the bilinear
//! fetch is exact.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::light_shaft`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::light_shaft::LightShaftParams;
use prism_volumetric_gpu::light_shaft::{GpuLightShaft, GpuLightShaftParams, LightShaftResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the shaft values.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the shaft values.
const REL_EPS: f32 = 1.0e-3;
/// Floor for the relative-tolerance denominator.
const REL_FLOOR: f32 = 1.0e-6;

/// Mixed absolute / relative tolerance comparison for one `f32` lane.
fn approx(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Deterministic host-side `u64` LCG (numerical-recipes constants) producing a
/// repeatable stream of mask values in `[0, 1)`; no transcendental math is
/// involved and the same values feed both the `CPU` golden and the `GPU`.
struct Lcg {
    /// Current state word.
    state: u64,
}

impl Lcg {
    /// Seeds the generator.
    fn new(seed: u64) -> Lcg {
        Lcg { state: seed }
    }

    /// Advances the state and returns an `f32` in `[0, 1)` built from the high
    /// bits, so the mask values stay small and exactly shared between the `CPU`
    /// and `GPU` inputs.
    fn next_unit(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.state >> 40) as u32;
        (bits & 0x00ff_ffff) as f32 / 16_777_216.0
    }
}

/// Builds a row-major `width * height` mask of pseudo-random values in
/// `[0, 1)`.
fn build_mask(width: u32, height: u32, seed: u64) -> Vec<f32> {
    let count = (width as usize) * (height as usize);
    let mut rng = Lcg::new(seed);
    let mut mask = Vec::with_capacity(count);
    for _ in 0..count {
        mask.push(rng.next_unit());
    }
    mask
}

/// Mirrors the golden [`LightShaftParams`] field set into the `GPU` twin's
/// [`GpuLightShaftParams`].
fn gpu_params(p: &LightShaftParams) -> GpuLightShaftParams {
    GpuLightShaftParams::new(
        p.light_uv,
        p.num_samples,
        p.density,
        p.decay,
        p.weight,
        p.exposure,
    )
}

/// Dispatches the twin and asserts every shaft pixel matches the golden
/// [`LightShaftParams::radial_blur_pixel`] under tolerance.
fn assert_parity(
    gpu: &GpuLightShaft,
    ctx: &GpuContext,
    p: &LightShaftParams,
    mask: &[f32],
    width: u32,
    height: u32,
) {
    let LightShaftResult { shaft } = gpu.blur(ctx, gpu_params(p), mask, width, height);
    let w = width as usize;
    let h = height as usize;
    assert_eq!(shaft.len(), w * h, "shaft length must equal width * height");
    for py in 0..h {
        for px in 0..w {
            let expected = p.radial_blur_pixel(mask, w, h, px, py);
            let got = shaft[py * w + px];
            assert!(
                approx(got, expected),
                "pixel ({px}, {py}): gpu {got} vs cpu {expected}"
            );
        }
    }
}

#[test]
fn interior_random_mask_power_of_two() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightShaft::new(&ctx);
    // 16x16 power-of-two grid: the pixel-centre UV and every marched tap are
    // exact in f32, so the floor split is bit-identical on host and device.
    let mask = build_mask(16, 16, 0x1234_5678_9abc_def0);
    let p = LightShaftParams::new([0.5, 0.5], 8, 1.0, 0.9, 0.5, 1.0);
    assert_parity(&gpu, &ctx, &p, &mask, 16, 16);
}

#[test]
fn num_samples_zero_is_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightShaft::new(&ctx);
    // num_samples == 0: the pass returns the pixel's own bilinear mask value,
    // which at a pixel centre on a power-of-two grid is the texel itself.
    let mask = build_mask(8, 8, 0x0bad_c0de_f00d_1357);
    let p = LightShaftParams::new([0.3, 0.7], 0, 1.5, 0.8, 0.4, 2.0);
    assert_parity(&gpu, &ctx, &p, &mask, 8, 8);
}

#[test]
fn single_texel_image() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightShaft::new(&ctx);
    // 1x1 image: every bilinear fetch clamps to the lone texel; the march still
    // runs num_samples taps and the decay geometry accumulates.
    let mask = [0.625f32];
    let p = LightShaftParams::new([0.9, 0.1], 6, 1.0, 0.85, 0.5, 1.0);
    assert_parity(&gpu, &ctx, &p, &mask, 1, 1);
}

#[test]
fn odd_dimensions_uniform_mask() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightShaft::new(&ctx);
    // 7x5 non-power-of-two grid with a uniform mask: the bilinear value is the
    // same constant for any fetch, so the odd-size index arithmetic is
    // exercised while the floor split can never cause a value divergence.
    let mask = vec![0.37f32; 7 * 5];
    let p = LightShaftParams::new([0.5, 0.5], 6, 1.0, 0.85, 0.5, 1.0);
    assert_parity(&gpu, &ctx, &p, &mask, 7, 5);
}

#[test]
fn off_screen_light_uv_stays_finite() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightShaft::new(&ctx);
    // Light off-screen: taps march far past the frame and clamp to the edge.
    let mask = build_mask(16, 16, 0x00c0_ffee_1234_5678);
    let p = LightShaftParams::new([-1.5, 2.5], 8, 1.0, 0.9, 0.5, 1.0);
    assert_parity(&gpu, &ctx, &p, &mask, 16, 16);
}

#[test]
fn extreme_density_and_decay() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightShaft::new(&ctx);
    // High density and decay > 1: the accumulation grows but stays finite, and
    // the twin must track the growth exactly as the reference does.
    let mask = build_mask(8, 8, 0x5555_aaaa_3333_cccc);
    let p = LightShaftParams::new([0.5, 0.5], 8, 3.0, 2.0, 0.5, 1.0);
    assert_parity(&gpu, &ctx, &p, &mask, 8, 8);
}

#[test]
fn zero_mask_yields_zero_shaft() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightShaft::new(&ctx);
    // A zeroed mask smears to zero everywhere; it checks the march reads the
    // uploaded zeros rather than stale buffer contents.
    let mask = vec![0.0f32; 4 * 4];
    let p = LightShaftParams::new([0.5, 0.5], 8, 1.0, 0.9, 0.5, 1.0);
    assert_parity(&gpu, &ctx, &p, &mask, 4, 4);
}

#[test]
fn undersized_mask_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightShaft::new(&ctx);
    // A mask shorter than width * height returns an empty result with no
    // dispatch, mirroring the empty return of `radial_blur`.
    let mask = [0.5f32; 3];
    let p = LightShaftParams::new([0.5, 0.5], 8, 1.0, 0.9, 0.5, 1.0);
    let out = gpu.blur(&ctx, gpu_params(&p), &mask, 4, 4);
    assert!(out.shaft.is_empty());
    assert_eq!(p.radial_blur(&mask, 4, 4).len(), out.shaft.len());
}

#[test]
fn empty_image_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightShaft::new(&ctx);
    // A zero-extent image issues no dispatch and returns an empty result.
    let mask: [f32; 0] = [];
    let p = LightShaftParams::new([0.5, 0.5], 8, 1.0, 0.9, 0.5, 1.0);
    let out = gpu.blur(&ctx, gpu_params(&p), &mask, 0, 4);
    assert!(out.shaft.is_empty());
}
