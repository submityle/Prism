//! Real-device parity for the `bloom` upsample-composite twin:
//! [`GpuBloomUpsample`](prism_volumetric_gpu::bloom_upsample::GpuBloomUpsample)
//! must reproduce the `CPU` golden
//! [`composite_chain`](prism_render_architecture::particle::bloom_upsample::composite_chain)
//! across the empty pyramid, a single-level pyramid, a `1x1` pyramid, two-level
//! and deep multi-level pyramids (even and odd extents so the `div_ceil`
//! halving and the clamp-to-edge `tent` are both exercised at a boundary), the
//! `radius` scatter extremes (crisp nearest at `0`, full `tent` at `1`) and a
//! batch of random pyramids at several resolutions compared `texel`-for-`texel`.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each output `texel` is a fixed, non-reorderable sequence of weighted `tent`
//! taps, one scatter lerp and one weighted add, so `CPU` and `GPU` evaluate the
//! same closed form in the same order. They are not bit-exact: a `GPU` may fuse
//! a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place, and that perturbation
//! compounds across the pyramid. The comparison therefore allows
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3` — loose enough to admit a legal
//! fused multiply-add contraction summed over the pyramid, yet tight enough to
//! fail a genuinely wrong port (a swapped `tent` tap, a dropped scatter blend, a
//! wrong normalizer, a missing edge clamp, a mis-ordered chain).
//!
//! Provenance: standard progressive dual-filter `bloom` upsample; no
//! third-party engine source or derived code.

use prism_render_architecture::particle::bloom_upsample::{composite_chain, MipImage, MipWeights};
use prism_volumetric_gpu::bloom_upsample::{BloomUpsampleQuery, GpuBloomUpsample};
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

/// Builds a `width * height` level whose pixels are pseudo-random `HDR` values
/// in `[0, scale)` drawn from `state`.
fn random_level(width: usize, height: usize, scale: f32, state: &mut u64) -> MipImage {
    let mut pixels = Vec::with_capacity(width * height);
    for _ in 0..(width * height) {
        pixels.push([lcg(state) * scale, lcg(state) * scale, lcg(state) * scale]);
    }
    MipImage::new(width, height, pixels)
}

/// Builds a finest-first random pyramid of `levels` levels starting at
/// `width x height`, halving each coarser level with `div_ceil` (floored to
/// `1`) so odd extents force the clamp-to-edge `tent` onto a boundary.
fn random_pyramid(
    width: usize,
    height: usize,
    levels: usize,
    scale: f32,
    state: &mut u64,
) -> Vec<MipImage> {
    let mut mips = Vec::with_capacity(levels);
    let (mut cw, mut ch) = (width, height);
    for _ in 0..levels {
        mips.push(random_level(cw, ch, scale, state));
        cw = cw.div_ceil(2).max(1);
        ch = ch.div_ceil(2).max(1);
    }
    mips
}

/// A `width * height` level filled with a single constant `RGB` value.
fn uniform_level(width: usize, height: usize, value: f32) -> MipImage {
    MipImage::new(width, height, vec![[value; 3]; width * height])
}

/// Runs the `GPU` composite and asserts `texel`-for-`texel` parity against the
/// `CPU` golden [`composite_chain`], returning the `GPU` result for any extra
/// per-test assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuBloomUpsample,
    mips: &[MipImage],
    weights: &MipWeights,
    radius: f32,
) -> Option<MipImage> {
    let query = BloomUpsampleQuery {
        mips: mips.to_vec(),
        weights: weights.clone(),
        radius,
    };
    let got = gpu.eval(ctx, &query);
    let want = composite_chain(mips, weights, radius);

    match (got, want) {
        (None, None) => None,
        (Some(g), Some(w)) => {
            assert_eq!(g.width, w.width, "width must match the reference");
            assert_eq!(g.height, w.height, "height must match the reference");
            assert_eq!(
                g.pixels.len(),
                w.pixels.len(),
                "pixel count must match the reference"
            );
            for (idx, (gp, wp)) in g.pixels.iter().zip(w.pixels.iter()).enumerate() {
                for channel in 0..3 {
                    assert!(
                        close(gp[channel], wp[channel]),
                        "texel {idx} channel {channel}: gpu {} vs cpu {} (radius {radius})",
                        gp[channel],
                        wp[channel]
                    );
                }
            }
            Some(g)
        }
        (g, w) => panic!(
            "Some/None disagreement: gpu is_some={} cpu is_some={}",
            g.is_some(),
            w.is_some()
        ),
    }
}

#[test]
fn empty_pyramid_is_none() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBloomUpsample::new(&ctx);
    let weights = MipWeights::geometric(1, 1, 2);
    // An empty pyramid returns `None` on both sides, with no dispatch issued.
    let got = check(&ctx, &gpu, &[], &weights, 0.5);
    assert!(got.is_none(), "an empty pyramid composites to None");
}

#[test]
fn single_level_is_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBloomUpsample::new(&ctx);
    let mut state = 0x1357_9bdf_0246_8ace_u64;
    let mips = vec![random_level(8, 6, 4.0, &mut state)];
    let weights = MipWeights::geometric(1, 1, 2);
    // The chain's base case clones the lone level unchanged.
    let got = check(&ctx, &gpu, &mips, &weights, 0.5).expect("single level composites to Some");
    assert_eq!(got, mips[0], "a single-level pyramid is the identity");
}

#[test]
fn one_by_one_multi_level() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBloomUpsample::new(&ctx);
    // Every level is 1x1, so each tap clamps onto the single texel; the chain
    // must reproduce that additive accumulation exactly.
    let mips = vec![
        MipImage::new(1, 1, vec![[0.3, 0.7, 1.5]]),
        MipImage::new(1, 1, vec![[0.2, 0.1, 0.4]]),
        MipImage::new(1, 1, vec![[0.9, 0.5, 0.6]]),
    ];
    let weights = MipWeights::geometric(3, 1, 2);
    let got = check(&ctx, &gpu, &mips, &weights, 0.5).expect("1x1 pyramid composites to Some");
    assert_eq!(got.width, 1);
    assert_eq!(got.height, 1);
}

#[test]
fn two_level_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBloomUpsample::new(&ctx);
    let mut state = 0x2b7e_1516_28ae_d2a6_u64;
    // One upsample step: a 4x3 coarse level added onto an 8x6 fine level,
    // isolating a single `upsample_add` against the reference.
    let mips = vec![
        random_level(8, 6, 3.0, &mut state),
        random_level(4, 3, 3.0, &mut state),
    ];
    let weights = MipWeights::geometric(2, 2, 3);
    for &radius in &[0.0f32, 0.3, 0.75, 1.0] {
        check(&ctx, &gpu, &mips, &weights, radius);
    }
}

#[test]
fn even_pyramid_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBloomUpsample::new(&ctx);
    let mut state = 0x3243_f6a8_885a_308d_u64;
    // An even 16x16 base halves cleanly to 8x8, 4x4, 2x2 with no odd remainder.
    let mips = random_pyramid(16, 16, 4, 3.0, &mut state);
    let weights = MipWeights::geometric(4, 1, 2);
    for &radius in &[0.25f32, 0.6, 0.95] {
        check(&ctx, &gpu, &mips, &weights, radius);
    }
}

#[test]
fn odd_pyramid_exercises_div_ceil_and_clamp() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBloomUpsample::new(&ctx);
    let mut state = 0xa409_3822_299f_31d0_u64;
    // Odd extents force the pyramid to `div_ceil` (13 -> 7 -> 4 -> 2, 11 -> 6 ->
    // 3 -> 2) and push several `tent` taps onto the clamp-to-edge boundary.
    let mips = random_pyramid(13, 11, 4, 2.5, &mut state);
    let weights = MipWeights::geometric(4, 2, 3);
    for &radius in &[0.1f32, 0.5, 0.9] {
        check(&ctx, &gpu, &mips, &weights, radius);
    }
}

#[test]
fn radius_zero_is_crisp_upsample() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBloomUpsample::new(&ctx);
    let mut state = 0x5eed_4a7d_0bad_c0de_u64;
    // With `radius == 0` the scatter blend is 0, so the upsample is the crisp
    // nearest sample (no `tent`); parity against the reference pins that branch.
    let mips = random_pyramid(12, 10, 3, 4.0, &mut state);
    let weights = MipWeights::geometric(3, 3, 4);
    check(&ctx, &gpu, &mips, &weights, 0.0);
}

#[test]
fn impulse_spreads_like_the_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBloomUpsample::new(&ctx);
    // A coarse single bright texel must diffuse into the fine level exactly as
    // the reference does; parity pins every texel and we additionally confirm
    // the glow leaked onto a neighbor so the test is not vacuously constant.
    let fine = MipImage::black(8, 8);
    let mut coarse_pixels = vec![[0.0f32; 3]; 4 * 4];
    coarse_pixels[2 * 4 + 2] = [1.0, 1.0, 1.0];
    let coarse = MipImage::new(4, 4, coarse_pixels);
    let mips = vec![fine, coarse];
    let weights = MipWeights::geometric(2, 1, 1);
    let got = check(&ctx, &gpu, &mips, &weights, 1.0).expect("impulse composites to Some");
    // The full `tent` (radius 1) spreads the coarse impulse at (2,2) over the
    // fine 2x2 block around (4,4) and its neighbors.
    assert!(
        got.pixel(4, 4)[0] > EPS,
        "the upsampled impulse must light the matching fine texel"
    );
    assert!(
        got.pixel(6, 6)[0] > EPS,
        "the tent must leak glow onto a diagonal neighbor block"
    );
}

#[test]
fn random_pyramids_multiple_resolutions_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBloomUpsample::new(&ctx);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    // Sweep a spread of base resolutions (even, odd, tall, wide, prime), depths
    // and radii so the `tent`, the pyramid extents and the scatter blend are all
    // exercised against random inputs, each compared texel-for-texel.
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
        for levels in 2usize..=4 {
            let radius = 0.1 + lcg(&mut state) * 0.9;
            let mips = random_pyramid(width, height, levels, 4.0, &mut state);
            let weights = MipWeights::geometric(levels, 1, 2);
            check(&ctx, &gpu, &mips, &weights, radius);
        }
    }
}

#[test]
fn constant_pyramid_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBloomUpsample::new(&ctx);
    // A constant pyramid: the normalized `tent` reproduces each constant level,
    // so the composite is a closed-form weighted sum; parity plus the reference
    // pins the normalizer and the additive chain together.
    let mips = vec![
        uniform_level(8, 8, 0.5),
        uniform_level(4, 4, 0.25),
        uniform_level(2, 2, 0.125),
    ];
    let weights = MipWeights::geometric(3, 1, 2);
    check(&ctx, &gpu, &mips, &weights, 0.5);
}
