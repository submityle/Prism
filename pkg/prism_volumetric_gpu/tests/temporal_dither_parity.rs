//! Real-device parity for the temporal-dither threshold twin:
//! [`GpuTemporalDither`](prism_volumetric_gpu::temporal_dither::GpuTemporalDither)
//! must reproduce the `CPU` golden
//! [`temporal_dither`](prism_render_architecture::particle::temporal_dither)
//! across every [`DitherMode`], many pixel coordinates (including coordinates
//! beyond one tile so the modulo wrap is exercised), many frames (so the
//! golden-ratio temporal rotation and the `blue-noise` frame decorrelation are
//! both exercised) and opacities straddling the threshold on both sides (so the
//! strict keep/discard test is pinned in both directions).
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The raw rank (the `Bayer` rank or the 24-bit `blue-noise` hash) and the
//! keep/discard mask are produced by exact `u32` integer arithmetic, so they
//! are compared with `==`. The normalized threshold is a widening `u32`-to-`f32`
//! cast of a value below `2^24` followed by a division by a compile-time
//! constant and a `floor`-based wrap; `CPU` and `GPU` evaluate that identical
//! closed form, so the comparison allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` — tight enough to fail a genuinely wrong port (a swapped
//! `Bayer` base, a wrong hash constant, a dropped temporal rotation, a wrong
//! normalizer) yet loose enough to admit a few units in the last place. Every
//! opacity in these fixtures is at least `0.05` away from its threshold, far
//! beyond that slack, so the keep/discard mask is never evaluated at a tie.
//!
//! Provenance: classic ordered `Bayer` dither plus an integer `blue-noise`
//! avalanche and a golden-ratio temporal rotation (`Prism` design §16); no
//! third-party engine source or derived code.

use prism_render_architecture::particle::temporal_dither::{
    bayer4_raw, bayer8_raw, blue_noise01, dither_threshold, should_discard, DitherMode,
};
use prism_volumetric_gpu::temporal_dither::{
    DitherPixel, DitherQuery, DitherSample, GpuTemporalDither,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound for the normalized threshold.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// `2^24`, the normalization divisor used to recover the raw 24-bit
/// `blue-noise` hash from its normalized value.
const NORM_24: f32 = 16_777_216.0;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// The raw integer the twin reports for `mode`: the `Bayer` rank, or the 24-bit
/// `blue-noise` hash recovered from its normalized value (`value < 2^24` is
/// exact in an `f32`, so the multiply-back and cast are lossless).
fn expected_raw(mode: DitherMode, p: &DitherPixel) -> u32 {
    match mode {
        DitherMode::Bayer4 => bayer4_raw(p.x, p.y),
        DitherMode::Bayer8 => bayer8_raw(p.x, p.y),
        DitherMode::BlueNoise => (blue_noise01(p.x, p.y, p.frame) * NORM_24) as u32,
    }
}

/// Runs the twin over `pixels` under `mode` and asserts per-pixel parity against
/// the `CPU` golden, returning the samples for any extra per-test assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuTemporalDither,
    mode: DitherMode,
    pixels: Vec<DitherPixel>,
) -> Vec<DitherSample> {
    let query = DitherQuery {
        mode,
        pixels: pixels.clone(),
    };
    let got = gpu.eval(ctx, &query);
    assert_eq!(
        got.len(),
        pixels.len(),
        "sample count must match the input count"
    );
    for (idx, (p, s)) in pixels.iter().zip(got.iter()).enumerate() {
        let want_threshold = dither_threshold(p.x, p.y, p.frame, mode);
        let want_raw = expected_raw(mode, p);
        let want_discard = should_discard(p.alpha, want_threshold);
        assert_eq!(
            s.raw, want_raw,
            "pixel {idx} ({mode:?}) raw {} vs cpu {want_raw}",
            s.raw
        );
        assert!(
            close(s.threshold, want_threshold),
            "pixel {idx} ({mode:?}) threshold {} vs cpu {want_threshold}",
            s.threshold
        );
        assert_eq!(
            s.discard, want_discard,
            "pixel {idx} ({mode:?}) discard {} vs cpu {want_discard}",
            s.discard
        );
    }
    got
}

/// Builds pixels sweeping an `extent x extent` coordinate block (padded past one
/// tile so the modulo wrap is exercised) across `frames`, with an opacity set
/// `0.1` below or above the threshold (alternating) so the keep/discard test is
/// pinned in both directions with no tie. Opacities are clamped into `[0, 1]`.
fn sweep(mode: DitherMode, extent: u32, frames: &[u32]) -> Vec<DitherPixel> {
    let mut pixels = Vec::new();
    let mut below = false;
    for &frame in frames {
        for y in 0..extent {
            for x in 0..extent {
                let threshold = dither_threshold(x, y, frame, mode);
                // Alternate the side we test; clamp so a near-0 or near-1
                // threshold still yields a valid, tie-free opacity.
                let alpha = if below {
                    (threshold - 0.1).max(0.0)
                } else {
                    (threshold + 0.1).min(1.0)
                };
                // If clamping collapsed the margin, push the opacity firmly to
                // the opposite extreme so it never lands on a tie.
                let alpha = if (alpha - threshold).abs() < 0.05 {
                    if below {
                        0.0
                    } else {
                        1.0
                    }
                } else {
                    alpha
                };
                pixels.push(DitherPixel::new(x, y, frame, alpha));
                below = !below;
            }
        }
    }
    pixels
}

#[test]
fn empty_batch_issues_no_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalDither::new(&ctx);
    // An empty batch must short-circuit (a storage buffer may not be zero-sized)
    // and return no samples.
    let got = gpu.eval(
        &ctx,
        &DitherQuery {
            mode: DitherMode::Bayer4,
            pixels: Vec::new(),
        },
    );
    assert!(got.is_empty(), "an empty batch produces no samples");
}

#[test]
fn bayer4_matches_reference_across_tile_and_frames() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalDither::new(&ctx);
    // 8x8 coordinate block (two full 4x4 tiles) across several frames so the
    // modulo wrap and the golden-ratio temporal rotation are both exercised.
    let pixels = sweep(DitherMode::Bayer4, 8, &[0, 1, 2, 7, 42, 101]);
    let got = check(&ctx, &gpu, DitherMode::Bayer4, pixels);
    // Every 4x4 rank appears, so the raw outputs must span the full 0..16 range.
    let max_raw = got.iter().map(|s| s.raw).max().unwrap_or(0);
    assert_eq!(max_raw, 15, "the 4x4 tile must expose rank 15");
}

#[test]
fn bayer8_matches_reference_across_tile_and_frames() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalDither::new(&ctx);
    // 8x8 coordinate block (one full 8x8 tile, plus wrap at the frame sweep)
    // across several frames.
    let pixels = sweep(DitherMode::Bayer8, 8, &[0, 1, 3, 16, 255]);
    let got = check(&ctx, &gpu, DitherMode::Bayer8, pixels);
    let max_raw = got.iter().map(|s| s.raw).max().unwrap_or(0);
    assert_eq!(max_raw, 63, "the 8x8 tile must expose rank 63");
}

#[test]
fn blue_noise_matches_reference_across_coords_and_frames() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalDither::new(&ctx);
    // Blue noise folds the frame into its hash, so sweep both coordinates and
    // frames; the raw 24-bit hash is checked bit-for-bit by `check`.
    let pixels = sweep(DitherMode::BlueNoise, 6, &[0, 1, 5, 99, 1000]);
    check(&ctx, &gpu, DitherMode::BlueNoise, pixels);
}

#[test]
fn coordinates_wrap_by_modulo_for_bayer() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalDither::new(&ctx);
    // Large coordinates must reduce modulo the tile; parity against the golden
    // (which also takes the coordinates modulo the tile) pins the wrap.
    let pixels = vec![
        DitherPixel::new(0, 0, 0, 0.5),
        DitherPixel::new(4, 8, 0, 0.5),
        DitherPixel::new(400, 800, 3, 0.5),
        DitherPixel::new(1, 3, 0, 0.5),
        DitherPixel::new(5, 7, 0, 0.5),
    ];
    let got4 = check(&ctx, &gpu, DitherMode::Bayer4, pixels.clone());
    // (0,0) and (4,8) share a 4x4 cell; (1,3) and (5,7) share another.
    assert_eq!(got4[0].raw, got4[1].raw, "(0,0) and (4,8) share a 4x4 cell");
    assert_eq!(got4[3].raw, got4[4].raw, "(1,3) and (5,7) share a 4x4 cell");
    check(&ctx, &gpu, DitherMode::Bayer8, pixels);
}

#[test]
fn strict_discard_boundary_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalDither::new(&ctx);
    // A fully opaque fragment survives any in-range threshold, and a nearly
    // transparent one is discarded by any positive threshold; both directions
    // are checked against the golden `should_discard`.
    let mut pixels = Vec::new();
    for mode in [
        DitherMode::Bayer4,
        DitherMode::Bayer8,
        DitherMode::BlueNoise,
    ] {
        for frame in 0..4u32 {
            for y in 0..4u32 {
                for x in 0..4u32 {
                    let t = dither_threshold(x, y, frame, mode);
                    // Opaque fragment: never discarded (threshold < 1).
                    pixels.push((mode, DitherPixel::new(x, y, frame, 1.0)));
                    // Below the threshold by a wide margin where possible.
                    if t > 0.2 {
                        pixels.push((mode, DitherPixel::new(x, y, frame, t - 0.1)));
                    }
                }
            }
        }
    }
    for mode in [
        DitherMode::Bayer4,
        DitherMode::Bayer8,
        DitherMode::BlueNoise,
    ] {
        let batch: Vec<DitherPixel> = pixels
            .iter()
            .filter(|(m, _)| *m == mode)
            .map(|(_, p)| *p)
            .collect();
        let got = check(&ctx, &gpu, mode, batch.clone());
        // The opaque (alpha == 1.0) fragments must never be discarded.
        for (p, s) in batch.iter().zip(got.iter()) {
            if (p.alpha - 1.0).abs() < f32::EPSILON {
                assert!(!s.discard, "an opaque fragment must survive ({mode:?})");
            }
        }
    }
}

#[test]
fn single_pixel_round_trips_each_mode() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTemporalDither::new(&ctx);
    // A one-element batch exercises the dispatch tail (count not a multiple of
    // the workgroup size) for every mode.
    for mode in [
        DitherMode::Bayer4,
        DitherMode::Bayer8,
        DitherMode::BlueNoise,
    ] {
        check(&ctx, &gpu, mode, vec![DitherPixel::new(3, 5, 2, 0.37)]);
    }
}
