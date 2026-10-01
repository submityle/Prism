//! Real-device parity for the film-grain twin:
//! [`GpuFilmGrain`](prism_volumetric_gpu::film_grain::GpuFilmGrain) must
//! reproduce the `CPU` golden
//! [`FilmGrainParams::apply`](prism_render_architecture::particle::film_grain::FilmGrainParams::apply)
//! across every [`GrainBlend`] variant, a sweep of cell sizes (including the
//! `0`-is-treated-as-`1` edge), several frame indices, exact cell-corner
//! samples and a batch of random coordinate/color fixtures compared pixel for
//! pixel, plus bit-exact agreement of the raw avalanche
//! [`hash_u32`](prism_render_architecture::particle::film_grain::hash_u32).
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion
//!
//! The integer hash is reproduced bit for bit (`WGSL` `u32` arithmetic wraps on
//! overflow, matching the `CPU` `wrapping_mul`), so [`GpuFilmGrain::hash`] is
//! asserted exactly equal. The floating noise, luminance response and blend
//! evaluate the same closed form in the same order; they are not bit-exact
//! because a `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few `ULP`. The composite comparison
//! therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` — tight enough to
//! fail a genuinely wrong port (a swapped tap, a wrong constant, a missing
//! clamp) yet loose enough to admit legal fused multiply-add contraction.
//!
//! Provenance: hand-rolled integer-hash value-noise film grain; no Unreal
//! Engine source or derived code.

use prism_render_architecture::particle::film_grain::{hash_u32, FilmGrainParams, GrainBlend};
use prism_volumetric_gpu::film_grain::{FilmGrainPixel, FilmGrainQuery, GpuFilmGrain};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound for the composited color. A `GPU` may fuse a
/// multiply-add the scalar reference leaves separate, perturbing the low
/// mantissa bits by a few units in the last place; `1e-4` admits that legal
/// slack while still failing a genuinely wrong port.
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

/// Runs the `GPU` composite and asserts pixel-for-pixel parity against the `CPU`
/// golden [`FilmGrainParams::apply`], returning the `GPU` result for any extra
/// per-test assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuFilmGrain,
    params: FilmGrainParams,
    frame: u32,
    pixels: &[FilmGrainPixel],
) -> Vec<[f32; 3]> {
    let query = FilmGrainQuery {
        params,
        frame,
        pixels: pixels.to_vec(),
    };
    let got = gpu.apply(ctx, &query);

    assert_eq!(got.len(), pixels.len(), "pixel count must match the input");
    for (idx, (g, p)) in got.iter().zip(pixels.iter()).enumerate() {
        let want = params.apply(p.color, p.x, p.y, frame);
        for channel in 0..3 {
            assert!(
                close(g[channel], want[channel]),
                "pixel {idx} channel {channel}: gpu {} vs cpu {} \
                 (frame {frame}, cell {}, blend {})",
                g[channel],
                want[channel],
                params.cell_size,
                params.blend.code(),
            );
            assert!(
                (0.0..=1.0).contains(&g[channel]),
                "pixel {idx} channel {channel} must stay clamped, got {}",
                g[channel],
            );
        }
    }
    got
}

/// Builds a batch of pseudo-random pixels at pseudo-random coordinates.
fn random_pixels(count: usize, state: &mut u64) -> Vec<FilmGrainPixel> {
    let mut pixels = Vec::with_capacity(count);
    for _ in 0..count {
        let color = [lcg(state), lcg(state), lcg(state)];
        let x = (lcg(state) * 4096.0) as u32;
        let y = (lcg(state) * 4096.0) as u32;
        pixels.push(FilmGrainPixel::new(color, x, y));
    }
    pixels
}

#[test]
fn empty_query_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFilmGrain::new(&ctx);
    let query = FilmGrainQuery {
        params: FilmGrainParams::new(0.25, 3, 1.5, 4.0, GrainBlend::Additive),
        frame: 0,
        pixels: Vec::new(),
    };
    assert!(
        gpu.apply(&ctx, &query).is_empty(),
        "empty input stays empty"
    );
    assert!(gpu.hash(&ctx, &[]).is_empty(), "empty seeds stay empty");
}

#[test]
fn hash_is_bit_exact() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFilmGrain::new(&ctx);
    // Cover zero, the all-ones word, powers of two and a dense low range so the
    // avalanche is exercised across the whole bit field.
    let mut seeds: Vec<u32> = Vec::new();
    seeds.push(0);
    seeds.push(u32::MAX);
    for shift in 0..32u32 {
        seeds.push(1u32 << shift);
    }
    for i in 0..512u32 {
        seeds.push(i.wrapping_mul(2_654_435_761));
    }
    let got = gpu.hash(&ctx, &seeds);
    assert_eq!(got.len(), seeds.len());
    for (idx, (&g, &s)) in got.iter().zip(seeds.iter()).enumerate() {
        assert_eq!(g, hash_u32(s), "hash seed {idx} (= {s}) must be bit-exact");
    }
}

#[test]
fn all_blend_variants_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFilmGrain::new(&ctx);
    let mut state = 0x0f0f_1234_5678_9abc_u64;
    let pixels = random_pixels(96, &mut state);
    for blend in [GrainBlend::Additive, GrainBlend::SoftLight] {
        let params = FilmGrainParams::new(0.6, 4, 1.5, 3.0, blend);
        check(&ctx, &gpu, params, 7, &pixels);
    }
}

#[test]
fn cell_size_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFilmGrain::new(&ctx);
    let mut state = 0x2b7e_1516_28ae_d2a6_u64;
    let pixels = random_pixels(128, &mut state);
    // Include `0` (treated as `1`), the finest per-pixel grain and several
    // coarse cells so the quotient/remainder quantization is exercised.
    for cell in [0u32, 1, 2, 3, 8, 16, 64] {
        let params = FilmGrainParams::new(0.5, cell, 1.0, 2.0, GrainBlend::Additive);
        check(&ctx, &gpu, params, 3, &pixels);
    }
}

#[test]
fn frame_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFilmGrain::new(&ctx);
    let mut state = 0x3243_f6a8_885a_308d_u64;
    let pixels = random_pixels(64, &mut state);
    let params = FilmGrainParams::new(0.75, 5, 2.0, 4.0, GrainBlend::SoftLight);
    for frame in [0u32, 1, 2, 100, 101, 65_535, u32::MAX] {
        check(&ctx, &gpu, params, frame, &pixels);
    }
}

#[test]
fn exact_cell_corners_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFilmGrain::new(&ctx);
    // At an exact cell corner the within-cell fractions are zero, so the noise
    // collapses to the raw corner hash; aligning the samples pins that anchor.
    let cell = 4u32;
    let mut pixels = Vec::new();
    for gy in 0..6u32 {
        for gx in 0..6u32 {
            pixels.push(FilmGrainPixel::new([0.3, 0.55, 0.8], gx * cell, gy * cell));
        }
    }
    let params = FilmGrainParams::new(0.4, cell, 1.0, 2.0, GrainBlend::Additive);
    check(&ctx, &gpu, params, 8, &pixels);
}

#[test]
fn zero_intensity_is_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFilmGrain::new(&ctx);
    let mut state = 0xa409_3822_299f_31d0_u64;
    let pixels = random_pixels(48, &mut state);
    for blend in [GrainBlend::Additive, GrainBlend::SoftLight] {
        let params = FilmGrainParams::new(0.0, 3, 1.5, 4.0, blend);
        let got = check(&ctx, &gpu, params, 11, &pixels);
        for (g, p) in got.iter().zip(pixels.iter()) {
            for (gc, pc) in g.iter().zip(p.color.iter()) {
                assert!(
                    close(*gc, *pc),
                    "zero intensity must return the input color unchanged",
                );
            }
        }
    }
}

#[test]
fn random_fixture_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFilmGrain::new(&ctx);
    let mut state = 0x5eed_4a7d_0bad_c0de_u64;
    // Sweep intensity, cell size, response tunables and blend together against a
    // fresh random batch each round so the whole parameter space is covered.
    for round in 0..8u32 {
        let intensity = 0.1 + lcg(&mut state) * 1.5;
        let cell = 1 + (lcg(&mut state) * 24.0) as u32;
        let shadow_boost = lcg(&mut state) * 3.0;
        let rolloff = lcg(&mut state) * 5.0;
        let blend = if round % 2 == 0 {
            GrainBlend::Additive
        } else {
            GrainBlend::SoftLight
        };
        let params = FilmGrainParams::new(intensity, cell, shadow_boost, rolloff, blend);
        let pixels = random_pixels(100, &mut state);
        check(&ctx, &gpu, params, round, &pixels);
    }
}
