//! Real-device parity for the ambient-occlusion twin: [`GpuAoSample`] must
//! reproduce the `CPU` golden
//! [`particle::ao_sample`](prism_render_architecture::particle::ao_sample) on
//! both of its surfaces — the per-sample upper-`hemisphere` kernel
//! ([`sample_kernel`](prism_render_architecture::particle::ao_sample::sample_kernel))
//! and the per-pixel occlusion fold
//! ([`AoParams::evaluate`](prism_render_architecture::particle::ao_sample::AoParams::evaluate)).
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! Both kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion
//!
//! The `hemisphere` kernel is an integer `R2` sequence mapped through the
//! trig-free elliptical-grid disk transform and a `sqrt` `hemisphere` lift; the
//! occlusion fold is a `smoothstep` range-checked mean, integer-exponent
//! contrast and intensity blend. Neither has a transcendental call or a
//! reorderable reduction, so `CPU` and `GPU` evaluate the same expression in
//! the same order. Values are asserted to within `abs_diff <= 1e-5` or
//! `rel_diff <= 1e-5` — tight enough to fail a wrong port (a swapped increment,
//! a missing `hemisphere` lift, a dropped bias or a broken range check) yet
//! loose enough to admit legal fused multiply-add contraction and the one-unit
//! `u32`-to-`f32` hash-normalization rounding.
//!
//! The covered scenarios are: the per-sample kernel across sizes and seeds
//! (also asserting upper-`hemisphere` alignment), a fully lit (no-occluder)
//! field, a fully occluded (near-occluder) field, a varied-parameter sweep, a
//! random integer-`LCG` depth field, and degenerate inputs (no depths, an empty
//! pixel set, zero intensity and a zero contrast exponent).

use prism_render_architecture::particle::ao_sample::{sample_kernel, AoParams};
use prism_volumetric_gpu::ao_sample::GpuAoSample;
use prism_volumetric_gpu::GpuContext;

/// Shared parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate or round the hash normalization by one unit in the last
/// place, so parity is asserted to a tight absolute-or-relative tolerance
/// rather than bit-exact equality.
const TOL: f32 = 1.0e-5;

/// Returns whether `got` and `want` agree to within [`TOL`] absolute or
/// relative. The `f32` comparison goes through `.abs()`, never `==`/`!=`.
fn close(got: f32, want: f32) -> bool {
    let abs_diff = (got - want).abs();
    let rel_diff = abs_diff / want.abs().max(TOL);
    abs_diff <= TOL || rel_diff <= TOL
}

/// Asserts per-pixel ambient-occlusion parity between the `GPU` result and the
/// `CPU` golden [`AoParams::evaluate`] over each pixel's depth run.
fn assert_ao_parity(
    params: AoParams,
    samples_per_pixel: u32,
    center_depths: &[f32],
    sampled_depths: &[f32],
    gpu: &[f32],
) {
    assert_eq!(gpu.len(), center_depths.len(), "one result per pixel");
    let spp = samples_per_pixel as usize;
    for (pixel, &center) in center_depths.iter().enumerate() {
        let run = if spp == 0 {
            &[][..]
        } else {
            &sampled_depths[pixel * spp..(pixel + 1) * spp]
        };
        let want = params.evaluate(run, center);
        let got = gpu[pixel];
        assert!(
            close(got, want),
            "ao mismatch at pixel {pixel}: gpu {got}, cpu {want} (center {center}, spp {spp})"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_kernel_matches_cpu_sample_kernel() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping ao_sample parity: no wgpu adapter on this host");
        return;
    };
    let twin = GpuAoSample::new(&ctx);

    // Several kernel sizes and seeds, including a large kernel.
    for (n, seed) in [(1u32, 0u32), (16, 7), (48, 3), (64, 8), (128, 0x000a_11ce)] {
        let gpu = twin.eval_kernel(&ctx, n, seed);
        let cpu = sample_kernel(n, seed);
        assert_eq!(gpu.len(), cpu.len(), "kernel length for n={n}");
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            for axis in 0..3 {
                assert!(
                    close(g[axis], c[axis]),
                    "kernel mismatch n={n} seed={seed} sample {i} axis {axis}: gpu {}, cpu {}",
                    g[axis],
                    c[axis]
                );
            }
            // Upper hemisphere: the normal-aligned axis is never negative, and
            // the radius never exceeds the unit sphere.
            assert!(g[2] >= 0.0, "sample {i} left the upper hemisphere");
            let len_sq = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
            assert!(len_sq <= 1.0 + TOL, "sample {i} exceeded the unit sphere");
        }
    }
}

#[test]
fn no_occlusion_field_is_fully_lit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuAoSample::new(&ctx);
    let params = AoParams::new(1.0, 0.05, 1.0, 1, 4);

    // Every fetched depth sits at or behind the shaded point, so nothing
    // occludes and every pixel stays fully lit (1.0).
    let samples_per_pixel = 4;
    let center_depths = [5.0f32, 2.0, 9.0];
    let sampled_depths = [
        5.0f32, 5.0, 6.0, 7.0, // pixel 0: equal / farther
        2.0, 2.5, 3.0, 10.0, // pixel 1: equal / farther
        9.0, 9.0, 9.0, 9.0, // pixel 2: all equal
    ];
    let gpu = twin.eval_ao(&ctx, params, samples_per_pixel, &center_depths, &sampled_depths);
    assert_ao_parity(params, samples_per_pixel, &center_depths, &sampled_depths, &gpu);
    for v in &gpu {
        assert!(close(*v, 1.0), "unoccluded pixel should be fully lit, got {v}");
    }
}

#[test]
fn full_occlusion_field_darkens() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuAoSample::new(&ctx);
    let params = AoParams::new(1.0, 0.0, 1.0, 1, 4);

    // Every fetched depth sits well in front of the shaded point, so the fold
    // darkens the pixel below full brightness.
    let samples_per_pixel = 4;
    let center_depths = [5.0f32, 8.0];
    let sampled_depths = [
        4.2f32, 4.3, 4.4, 4.5, // pixel 0: nearer occluders
        7.1, 7.2, 7.3, 7.4, // pixel 1: nearer occluders
    ];
    let gpu = twin.eval_ao(&ctx, params, samples_per_pixel, &center_depths, &sampled_depths);
    assert_ao_parity(params, samples_per_pixel, &center_depths, &sampled_depths, &gpu);
    for v in &gpu {
        assert!(*v >= 0.0 && *v < 1.0, "occluded pixel should darken, got {v}");
    }
}

#[test]
fn varied_parameters_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuAoSample::new(&ctx);

    // A sweep over radius, bias, intensity and contrast exponent on a mixed
    // depth field (some occluders, some not).
    let samples_per_pixel = 6;
    let center_depths = [4.0f32, 4.0, 4.0, 4.0];
    let sampled_depths = [
        3.5f32, 3.9, 4.0, 4.5, 2.0, 3.99, // pixel 0
        3.0, 3.2, 3.8, 5.0, 6.0, 3.95, // pixel 1
        3.98, 3.97, 3.96, 3.90, 3.80, 3.70, // pixel 2
        4.0, 4.1, 4.2, 4.3, 4.4, 4.5, // pixel 3 (all behind)
    ];
    for params in [
        AoParams::new(0.5, 0.0, 1.0, 1, 6),
        AoParams::new(1.5, 0.1, 0.75, 2, 6),
        AoParams::new(2.0, 0.25, 1.0, 3, 6),
        AoParams::new(0.8, 0.05, 0.5, 4, 6),
    ] {
        let gpu = twin.eval_ao(&ctx, params, samples_per_pixel, &center_depths, &sampled_depths);
        assert_ao_parity(params, samples_per_pixel, &center_depths, &sampled_depths, &gpu);
    }
}

#[test]
fn random_lcg_depth_field_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuAoSample::new(&ctx);
    let params = AoParams::new(1.25, 0.05, 0.9, 2, 8);

    // A deterministic integer linear-congruential generator fills a depth field
    // so the parity holds on arbitrary, non-degenerate data. Constants are the
    // classic Numerical Recipes LCG multiplier and increment.
    let pixel_count = 37usize;
    let samples_per_pixel = 8u32;
    let mut state = 0x1234_5678u32;
    let mut next_depth = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        // Map the high mantissa bits to a 0..8 depth with a 1/1024 quantum.
        ((state >> 16) & 0x1fff) as f32 / 1024.0
    };
    let center_depths: Vec<f32> = (0..pixel_count).map(|_| next_depth()).collect();
    let sampled_depths: Vec<f32> = (0..pixel_count * samples_per_pixel as usize)
        .map(|_| next_depth())
        .collect();

    let gpu = twin.eval_ao(&ctx, params, samples_per_pixel, &center_depths, &sampled_depths);
    assert_ao_parity(params, samples_per_pixel, &center_depths, &sampled_depths, &gpu);
}

#[test]
fn degenerate_inputs_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuAoSample::new(&ctx);

    // An empty pixel set yields no results.
    let empty = twin.eval_ao(&ctx, AoParams::new(1.0, 0.0, 1.0, 1, 4), 4, &[], &[]);
    assert!(empty.is_empty(), "an empty pixel set yields no results");

    // A zero-sample dispatch leaves every pixel fully lit (empty-set rule).
    let params = AoParams::new(1.0, 0.0, 1.0, 2, 0);
    let center_depths = [5.0f32, 1.0, 9.0];
    let gpu = twin.eval_ao(&ctx, params, 0, &center_depths, &[]);
    assert_ao_parity(params, 0, &center_depths, &[], &gpu);
    for v in &gpu {
        assert!(close(*v, 1.0), "zero-sample pixel should be fully lit, got {v}");
    }

    // Zero intensity leaves near occluders fully lit regardless of the fold.
    let unlit = AoParams::new(1.0, 0.0, 0.0, 3, 4);
    let near = [4.5f32, 4.6, 4.7, 4.8];
    let centers = [5.0f32];
    let gpu_unlit = twin.eval_ao(&ctx, unlit, 4, &centers, &near);
    assert_ao_parity(unlit, 4, &centers, &near, &gpu_unlit);
    assert!(close(gpu_unlit[0], 1.0), "zero intensity should stay fully lit");

    // A zero contrast exponent collapses the term to 1.0 before the intensity
    // blend, matching `ao_power(_, 0) == 1`.
    let flat = AoParams::new(1.0, 0.0, 1.0, 0, 4);
    let gpu_flat = twin.eval_ao(&ctx, flat, 4, &centers, &near);
    assert_ao_parity(flat, 4, &centers, &near, &gpu_flat);
    assert!(close(gpu_flat[0], 1.0), "zero exponent should stay fully lit");

    // An empty kernel is empty.
    assert!(twin.eval_kernel(&ctx, 0, 1).is_empty(), "n=0 yields an empty kernel");
}
