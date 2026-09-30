//! Real-device parity for the fractal-noise twin: [`GpuNoiseFbm`] must
//! reproduce the `CPU` goldens
//! [`fbm`](prism_render_architecture::volumetric::noise::fbm) and
//! [`worley_fbm`](prism_render_architecture::volumetric::noise::worley_fbm) for
//! every sample point across a spatial grid and a range of octave counts,
//! including the `octaves == 0` early-out.
//!
//! The [`crate::perlin_worley`](prism_volumetric_gpu::perlin_worley) twin only
//! exercises these accumulators at the fixed default octave counts; this suite
//! covers the full variable-octave surface the `CPU` contracts expose.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The integer lattice work (hash, gradient/feature selection) is bit-exact
//! because `WGSL` unsigned arithmetic wraps like Rust's `wrapping_*`; only the
//! float fBm accumulation and the Worley `sqrt` diverge, and only by legal
//! fused-multiply-add contraction. Each field is asserted to within
//! `abs_diff < 1e-5` — tight enough to fail a wrong octave loop, a swapped
//! field, or a dropped normalisation, loose enough to admit fma contraction.
//! The grid also asserts every value lands in `[0, 1]` and that the field is
//! not degenerately constant, so a stubbed kernel could not pass.
//!
//! Provenance: standard Nubis-style Perlin / inverted-Worley fractal cloud
//! noise; no Unreal Engine source or derived code.

use prism_render_architecture::volumetric::noise::{fbm, worley_fbm};
use prism_render_architecture::volumetric::Vec3;
use prism_volumetric_gpu::{GpuContext, GpuNoiseFbm, NoiseFbmQuery};

/// Asserts every GPU field matches the `CPU` golden to within the documented
/// fma tolerance and lands in `[0, 1]`.
fn assert_parity(queries: &[NoiseFbmQuery], gpu: &[prism_volumetric_gpu::NoiseFbmResult]) {
    assert_eq!(gpu.len(), queries.len(), "one result per query");
    for (i, q) in queries.iter().enumerate() {
        let point = Vec3::new(q.x, q.y, q.z);
        let cpu_fbm = fbm(point, q.seed, q.octaves);
        let cpu_worley = worley_fbm(point, q.seed, q.octaves);
        let got = gpu[i];

        let d_fbm = (got.fbm - cpu_fbm).abs();
        assert!(
            d_fbm < 1e-5,
            "fbm mismatch for query {q:?}: gpu {}, cpu {cpu_fbm} (abs {d_fbm})",
            got.fbm
        );
        let d_worley = (got.worley_fbm - cpu_worley).abs();
        assert!(
            d_worley < 1e-5,
            "worley_fbm mismatch for query {q:?}: gpu {}, cpu {cpu_worley} (abs {d_worley})",
            got.worley_fbm
        );

        assert!(
            (0.0..=1.0).contains(&got.fbm),
            "fbm {} out of [0, 1] for {q:?}",
            got.fbm
        );
        assert!(
            (0.0..=1.0).contains(&got.worley_fbm),
            "worley_fbm {} out of [0, 1] for {q:?}",
            got.worley_fbm
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_noise_fbm_matches_cpu_golden_across_grid_and_octaves() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping noise fbm parity: no wgpu adapter on this host");
        return;
    };
    let evaluator = GpuNoiseFbm::new(&ctx);

    // A spatial grid crossing several lattice cells (negative and positive
    // coordinates, non-integer offsets) at several octave counts and two seeds.
    let seeds = [0x1234_5678u32, 0x9e37_79b9u32];
    let octave_counts = [1u32, 2, 3, 5, 7];
    let mut queries = Vec::new();
    for &seed in &seeds {
        for &octaves in &octave_counts {
            for gz in 0..4 {
                for gy in 0..4 {
                    for gx in 0..4 {
                        queries.push(NoiseFbmQuery {
                            x: -1.7 + (gx as f32) * 0.93,
                            y: -0.4 + (gy as f32) * 1.31,
                            z: 0.6 + (gz as f32) * 0.77,
                            seed,
                            octaves,
                        });
                    }
                }
            }
        }
    }

    let gpu = evaluator.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // The field must not be degenerately constant: assert real spatial
    // variation across the grid for one (seed, octaves) slice.
    let slice: Vec<f32> = gpu.iter().take(16).map(|r| r.fbm).collect();
    let min = slice.iter().copied().fold(f32::INFINITY, f32::min);
    let max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max - min > 1e-3,
        "fbm field is degenerately flat across the grid (min {min}, max {max})"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_noise_fbm_zero_octaves_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping noise fbm parity: no wgpu adapter on this host");
        return;
    };
    let evaluator = GpuNoiseFbm::new(&ctx);

    // `octaves == 0` must yield exactly 0 for both fields (the divide-by-zero
    // guard), matching the CPU golden, regardless of point or seed.
    let queries = vec![
        NoiseFbmQuery {
            x: 0.3,
            y: -1.2,
            z: 2.5,
            seed: 0xABCD_0001,
            octaves: 0,
        },
        NoiseFbmQuery {
            x: -4.1,
            y: 0.0,
            z: 0.9,
            seed: 7,
            octaves: 0,
        },
    ];

    let gpu = evaluator.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);
    for r in &gpu {
        assert_eq!(r.fbm, 0.0, "zero-octave fbm must be exactly 0");
        assert_eq!(
            r.worley_fbm, 0.0,
            "zero-octave worley_fbm must be exactly 0"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_noise_fbm_handles_empty_input() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping noise fbm parity: no wgpu adapter on this host");
        return;
    };
    let evaluator = GpuNoiseFbm::new(&ctx);
    assert!(evaluator.eval(&ctx, &[]).is_empty(), "empty in, empty out");
}
