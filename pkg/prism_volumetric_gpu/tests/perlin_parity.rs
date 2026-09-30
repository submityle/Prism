//! Real-device parity for the `Perlin` gradient-noise twin: [`GpuPerlin`] must
//! reproduce the `CPU` golden
//! [`perlin_3d`](prism_render_architecture::volumetric::noise::perlin_3d)
//! across a spread of sample points and seeds.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The lattice `hash` is pure unsigned-integer work and `WGSL` unsigned
//! integers wrap on overflow exactly like Rust's `wrapping_mul` / `^` / `>>`,
//! so the `GPU` selects bit-identical gradients to the reference. Only the
//! float fade/`lerp` blend can differ, and only by a legal multiply-add
//! contraction of a few `ULP`. Values are asserted to within `abs_diff < 1e-6`
//! or `rel_diff < 1e-5` — tight enough to fail a wrong port (a swapped
//! gradient table, a dropped fade term, a mis-hashed coordinate). The scenes
//! also assert the `0..=1` range and spatial continuity, so a degenerate
//! constant kernel could not pass.
//!
//! Provenance: standard improved-`Perlin` gradient noise (`Perlin` 2002); no
//! Unreal Engine source or derived code.

use prism_render_architecture::volumetric::noise::perlin_3d;
use prism_render_architecture::volumetric::Vec3;
use prism_volumetric_gpu::{GpuContext, GpuPerlin, PerlinQuery};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays inside the unit range.
fn assert_parity(queries: &[PerlinQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one value per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = perlin_3d(q.point, q.seed);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "perlin mismatch for query {q:?}: gpu {got}, cpu {exp} (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            (0.0..=1.0).contains(&got),
            "gpu perlin must stay in the unit range: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_perlin_matches_cpu_golden_across_points_and_seeds() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping perlin parity: no wgpu adapter on this host");
        return;
    };
    let gpu_perlin = GpuPerlin::new(&ctx);

    // A deterministic lattice sweep: fractional offsets exercise the fade/lerp
    // blend, integer-crossing coordinates exercise cell boundaries, negatives
    // exercise the floor/`i32` cast, and several seeds exercise the hash mix.
    let mut queries: Vec<PerlinQuery> = Vec::new();
    let seeds = [0u32, 1, 7, 1_337, 0x9e37_79b9];
    for &seed in &seeds {
        for xi in -2..=2 {
            for yi in -2..=2 {
                for zi in -2..=2 {
                    let x = xi as f32 * 0.37 + 0.11;
                    let y = yi as f32 * 0.53 - 0.29;
                    let z = zi as f32 * 0.61 + 0.07;
                    queries.push(PerlinQuery {
                        point: Vec3::new(x, y, z),
                        seed,
                    });
                }
            }
        }
    }

    let gpu = gpu_perlin.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // The field is not constant: at least two sampled values must differ well
    // beyond the parity tolerance, proving a real gradient blend ran.
    let first = gpu[0];
    assert!(
        gpu.iter().any(|&v| (v - first).abs() > 1e-3),
        "perlin field must vary across space, not return a constant"
    );
}

#[test]
fn gpu_perlin_is_spatially_continuous() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_perlin = GpuPerlin::new(&ctx);

    // A dense walk along a line with a tiny step: neighbouring samples must be
    // close, because the quintic fade makes the field C2-continuous. A broken
    // hash or missing fade would produce discontinuous jumps at cell edges.
    let seed = 42u32;
    let step = 0.02f32;
    let queries: Vec<PerlinQuery> = (0..=200)
        .map(|k| {
            let t = k as f32 * step;
            PerlinQuery {
                point: Vec3::new(t, 0.5 * t + 0.3, 1.7 - 0.25 * t),
                seed,
            }
        })
        .collect();

    let gpu = gpu_perlin.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // Bound the per-step change. The gradient magnitude of normalised Perlin
    // noise is bounded, so a small spatial step yields a small value change;
    // 0.1 is comfortably above the true bound yet far below a cell-edge jump.
    for w in gpu.windows(2) {
        let jump = (w[1] - w[0]).abs();
        assert!(
            jump < 0.1,
            "perlin must be spatially continuous: neighbouring samples jumped {jump}"
        );
    }
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_perlin = GpuPerlin::new(&ctx);
    let out = gpu_perlin.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
