//! Real-device parity for the `FXAA` resolve twin:
//! [`GpuFxaa`](prism_volumetric_gpu::fxaa::GpuFxaa) must reproduce the `CPU`
//! golden
//! [`resolve_luma_grid`](prism_render_architecture::particle::fxaa::resolve_luma_grid)
//! across empty and zero-dimension inputs, a `1x1` image, a flat (edge-free)
//! image, strong horizontal/vertical/diagonal edges, a subpixel-dominated
//! checkerboard, parameter extremes and a batch of random images at several
//! resolutions, each compared texel for texel.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each output texel is a fixed, non-reorderable sequence of `min`/`max`
//! reductions, guarded divides, a Hermite polynomial and clamps, so `CPU` and
//! `GPU` evaluate the same closed form in the same order. They are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The comparison therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` —
//! loose enough to admit a legal fused multiply-add contraction, yet tight
//! enough to fail a genuinely wrong port (a swapped neighbor, a dropped clamp,
//! a wrong threshold). Fixtures stay clear of the `edge_threshold` tie boundary
//! so a legal `ULP` wobble never flips the edge gate between `CPU` and `GPU`.
//!
//! Provenance: standard `FXAA` 3.11 luma-adaptive antialiasing (Lottes,
//! `NVIDIA`, 2011); no third-party engine source or derived code.

use prism_render_architecture::particle::fxaa::{resolve_luma_grid, FxaaParams};
use prism_volumetric_gpu::fxaa::{FxaaQuery, GpuFxaa};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
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
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Builds a `width * height` row-major `luma` image whose samples are
/// pseudo-random values in `[0, scale)` drawn from `state`.
fn random_luma(width: usize, height: usize, scale: f32, state: &mut u64) -> Vec<f32> {
    let mut luma = Vec::with_capacity(width * height);
    for _ in 0..(width * height) {
        luma.push(lcg(state) * scale);
    }
    luma
}

/// Runs the `GPU` `FXAA` resolve and asserts texel-for-texel parity against the
/// `CPU` golden [`resolve_luma_grid`], returning the `GPU` result for any extra
/// per-test assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuFxaa,
    luma: &[f32],
    width: usize,
    height: usize,
    params: &FxaaParams,
) -> Vec<f32> {
    let query = FxaaQuery {
        luma: luma.to_vec(),
        width,
        height,
        params: *params,
    };
    let got = gpu.eval(ctx, &query);
    let want = resolve_luma_grid(luma, width, height, params);

    assert_eq!(
        got.len(),
        want.len(),
        "output length must match the reference ({width}x{height})"
    );
    for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            close(*g, *w),
            "texel {idx}: gpu {g} vs cpu {w} ({width}x{height})"
        );
    }
    got
}

#[test]
fn empty_inputs_short_circuit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFxaa::new(&ctx);
    let params = FxaaParams::default_quality();
    // Zero width, zero height and a buffer shorter than width*height all return
    // an empty Vec with no dispatch issued, exactly as the reference does.
    assert!(check(&ctx, &gpu, &[0.5; 4], 0, 4, &params).is_empty());
    assert!(check(&ctx, &gpu, &[0.5; 4], 4, 0, &params).is_empty());
    assert!(check(&ctx, &gpu, &[0.5; 3], 4, 4, &params).is_empty());
}

#[test]
fn single_pixel_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFxaa::new(&ctx);
    let params = FxaaParams::default_quality();
    // A 1x1 image clamps every neighbor onto the single texel, so there is no
    // contrast and the weight is zero.
    let got = check(&ctx, &gpu, &[0.42], 1, 1, &params);
    assert_eq!(got.len(), 1);
    assert!(got[0] < EPS, "a lone texel has no edge, got {}", got[0]);
}

#[test]
fn flat_image_has_no_edges() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFxaa::new(&ctx);
    let params = FxaaParams::default_quality();
    // A constant image never reaches the edge floor, so every weight is zero.
    let luma = vec![0.6_f32; 6 * 5];
    let got = check(&ctx, &gpu, &luma, 6, 5, &params);
    for (idx, v) in got.iter().enumerate() {
        assert!(*v < EPS, "flat texel {idx} must be zero, got {v}");
    }
}

#[test]
fn strong_vertical_edge_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFxaa::new(&ctx);
    let params = FxaaParams::default_quality();
    // Left half dark, right half bright: a strong vertical edge down the
    // middle. The step (0.9) sits far above the edge floor, so no tie wobble.
    let width = 6;
    let height = 4;
    let mut luma = vec![0.05_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            if x >= width / 2 {
                luma[y * width + x] = 0.95;
            }
        }
    }
    let got = check(&ctx, &gpu, &luma, width, height, &params);
    assert!(
        got.iter().any(|v| *v > EPS),
        "a strong vertical edge must produce some positive weight"
    );
}

#[test]
fn strong_horizontal_edge_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFxaa::new(&ctx);
    let params = FxaaParams::default_quality();
    // Top half dark, bottom half bright: a strong horizontal edge.
    let width = 5;
    let height = 6;
    let mut luma = vec![0.08_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            if y >= height / 2 {
                luma[y * width + x] = 0.92;
            }
        }
    }
    let got = check(&ctx, &gpu, &luma, width, height, &params);
    assert!(
        got.iter().any(|v| *v > EPS),
        "a strong horizontal edge must produce some positive weight"
    );
}

#[test]
fn diagonal_edge_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFxaa::new(&ctx);
    let params = FxaaParams::default_quality();
    // A diagonal split (bright above the x==y line, dark below) exercises mixed
    // horizontal/vertical gradients and the full-window subpixel term.
    let width = 7;
    let height = 7;
    let mut luma = vec![0.1_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            if x >= y {
                luma[y * width + x] = 0.85;
            }
        }
    }
    let got = check(&ctx, &gpu, &luma, width, height, &params);
    assert!(
        got.iter().any(|v| *v > EPS),
        "a diagonal edge must produce some positive weight"
    );
}

#[test]
fn subpixel_dominated_checkerboard_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFxaa::new(&ctx);
    // A high `subpix_quality` on a checkerboard makes the subpixel term lead the
    // blend, so the `max(edge_blend, subpixel)` branch is exercised on its
    // subpixel side.
    let params = FxaaParams {
        edge_threshold: 0.125,
        edge_threshold_min: 0.04,
        subpix_quality: 1.0,
    };
    let width = 6;
    let height = 6;
    let mut luma = vec![0.0_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            if (x + y) % 2 == 0 {
                luma[y * width + x] = 0.8;
            }
        }
    }
    let got = check(&ctx, &gpu, &luma, width, height, &params);
    assert!(
        got.iter().any(|v| *v > EPS),
        "a checkerboard must drive a positive subpixel blend"
    );
}

#[test]
fn parameter_extremes_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFxaa::new(&ctx);
    let mut state = 0x0f0f_a5a5_1234_9876_u64;
    let width = 8;
    let height = 6;
    let luma = random_luma(width, height, 1.0, &mut state);
    // A near-zero floor with a tiny relative threshold flags almost every
    // neighborhood; a near-unit floor with a high relative threshold suppresses
    // almost all of them. Zero and full subpixel quality bracket that term.
    let presets = [
        FxaaParams {
            edge_threshold: 0.03,
            edge_threshold_min: 0.001,
            subpix_quality: 0.0,
        },
        FxaaParams {
            edge_threshold: 0.9,
            edge_threshold_min: 0.95,
            subpix_quality: 1.0,
        },
        FxaaParams {
            edge_threshold: 0.5,
            edge_threshold_min: 0.25,
            subpix_quality: 0.5,
        },
    ];
    for params in &presets {
        check(&ctx, &gpu, &luma, width, height, params);
    }
}

#[test]
fn random_images_multiple_resolutions_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFxaa::new(&ctx);
    let params = FxaaParams::default_quality();
    let mut state = 0x5eed_4a7d_0bad_c0de_u64;
    // Sweep even, odd, tall, wide and prime extents so the row-major indexing,
    // the clamp-to-edge borders and the 1-D dispatch tail are all exercised
    // against random inputs, each compared texel for texel.
    let resolutions = [
        (1usize, 1usize),
        (2, 3),
        (3, 7),
        (5, 5),
        (8, 1),
        (1, 9),
        (13, 11),
        (16, 9),
        (31, 17),
    ];
    for &(width, height) in &resolutions {
        let luma = random_luma(width, height, 2.0, &mut state);
        check(&ctx, &gpu, &luma, width, height, &params);
    }
}
