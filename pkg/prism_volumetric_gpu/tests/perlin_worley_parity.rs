//! Real-device parity for the `Perlin-Worley` base-noise twin:
//! [`GpuPerlinWorley`] must reproduce the `CPU` golden
//! [`perlin_worley`](prism_render_architecture::volumetric::noise::perlin_worley)
//! across a spread of sample points and seeds.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The gradient and feature-point selection is pure unsigned-integer work and
//! `WGSL` unsigned integers wrap on overflow exactly like Rust, so the lattice
//! work is bit-identical to the reference. Only the float `fBm` accumulation,
//! the `Worley` `sqrt` and the `remap` blend can diverge, and only by legal
//! multiply-add contraction. Values are asserted to within `abs_diff < 1e-6`
//! or `rel_diff < 1e-5` — tight enough to fail a wrong port (a wrong octave
//! count, a dropped seed decorrelation, an inverted `remap`). The scenes also
//! assert the `0..=1` range and spatial continuity, so a degenerate constant
//! kernel could not pass.
//!
//! Provenance: standard `Nubis`-style `Perlin-Worley` cloud base noise; no
//! Unreal Engine source or derived code.

use prism_render_architecture::volumetric::noise::perlin_worley;
use prism_render_architecture::volumetric::Vec3;
use prism_volumetric_gpu::{GpuContext, GpuPerlinWorley, PerlinWorleyQuery};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays inside the unit range.
fn assert_parity(queries: &[PerlinWorleyQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one value per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = perlin_worley(q.point, q.seed);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "perlin_worley mismatch for query {q:?}: gpu {got}, cpu {exp} (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            (0.0..=1.0).contains(&got),
            "gpu perlin_worley must stay in the unit range: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_perlin_worley_matches_cpu_golden_across_points_and_seeds() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping perlin_worley parity: no wgpu adapter on this host");
        return;
    };
    let gpu_pw = GpuPerlinWorley::new(&ctx);

    // A deterministic sweep exercising the stacked five-octave Perlin fBm and
    // three-octave inverted-Worley fBm plus the remap blend, across negatives,
    // cell crossings and several seeds.
    let mut queries: Vec<PerlinWorleyQuery> = Vec::new();
    let seeds = [0u32, 5, 23, 4_096, 0x846c_a68b];
    for &seed in &seeds {
        for xi in -2..=2 {
            for yi in -2..=2 {
                for zi in -2..=2 {
                    let x = xi as f32 * 0.43 + 0.17;
                    let y = yi as f32 * 0.51 - 0.23;
                    let z = zi as f32 * 0.57 + 0.05;
                    queries.push(PerlinWorleyQuery {
                        point: Vec3::new(x, y, z),
                        seed,
                    });
                }
            }
        }
    }

    let gpu = gpu_pw.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // The base shape is not constant: at least two sampled values must differ
    // well beyond the parity tolerance, proving the full fBm+remap pipeline ran.
    let first = gpu[0];
    assert!(
        gpu.iter().any(|&v| (v - first).abs() > 1e-3),
        "perlin_worley base shape must vary across space, not return a constant"
    );
}

#[test]
fn gpu_perlin_worley_is_spatially_continuous() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_pw = GpuPerlinWorley::new(&ctx);

    // A dense walk along a line with a tiny step. The base shape is a
    // composition of continuous fBm fields and a non-collapsing remap, so
    // neighbouring samples must stay close. A broken octave or hash would
    // produce discontinuous jumps.
    let seed = 7u32;
    let step = 0.02f32;
    let queries: Vec<PerlinWorleyQuery> = (0..=200)
        .map(|k| {
            let t = k as f32 * step;
            PerlinWorleyQuery {
                point: Vec3::new(0.2 + 0.4 * t, 1.1 - 0.3 * t, 0.5 * t),
                seed,
            }
        })
        .collect();

    let gpu = gpu_pw.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // The stacked-octave field has a bounded spatial gradient, so a small step
    // yields a bounded value change. 0.25 is comfortably above the true bound
    // yet far below an O(1) discontinuity from a broken port.
    for w in gpu.windows(2) {
        let jump = (w[1] - w[0]).abs();
        assert!(
            jump < 0.25,
            "perlin_worley must be spatially continuous: neighbouring samples jumped {jump}"
        );
    }
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_pw = GpuPerlinWorley::new(&ctx);
    let out = gpu_pw.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
