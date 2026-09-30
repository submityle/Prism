//! Real-device parity for the `Worley` cellular-noise twin: [`GpuWorley`] must
//! reproduce the `CPU` golden
//! [`worley_3d`](prism_render_architecture::volumetric::noise::worley_3d)
//! across a spread of sample points and seeds.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Feature-point selection is pure unsigned-integer work and `WGSL` unsigned
//! integers wrap on overflow exactly like Rust's `wrapping_mul` / `^` / `>>`,
//! so the `GPU` jitters points bit-identically to the reference. Only the
//! final `sqrt` of the summed squared distance can differ, and only by a legal
//! multiply-add contraction of a few `ULP`. Values are asserted to within
//! `abs_diff < 1e-6` or `rel_diff < 1e-5` — tight enough to fail a wrong port
//! (a swapped decorrelation seed, a wrong neighbourhood offset, a missing
//! saturation). The scenes also assert the `0..=1` range and spatial
//! continuity, so a degenerate constant kernel could not pass.
//!
//! Provenance: standard `Worley` cellular noise (`Worley` 1996); no Unreal
//! Engine source or derived code.

use prism_render_architecture::volumetric::noise::worley_3d;
use prism_render_architecture::volumetric::Vec3;
use prism_volumetric_gpu::{GpuContext, GpuWorley, WorleyQuery};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays inside the unit range.
fn assert_parity(queries: &[WorleyQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one value per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = worley_3d(q.point, q.seed);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "worley mismatch for query {q:?}: gpu {got}, cpu {exp} (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            (0.0..=1.0).contains(&got),
            "gpu worley must stay in the unit range: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_worley_matches_cpu_golden_across_points_and_seeds() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping worley parity: no wgpu adapter on this host");
        return;
    };
    let gpu_worley = GpuWorley::new(&ctx);

    // A deterministic lattice sweep: fractional offsets place the sample away
    // from feature points, integer-crossing coordinates exercise the neighbour
    // search across cells, negatives exercise the floor/`i32` cast, and several
    // seeds exercise the hash mix and the y/z decorrelation seeds.
    let mut queries: Vec<WorleyQuery> = Vec::new();
    let seeds = [0u32, 3, 19, 2_024, 0x27d4_eb2f];
    for &seed in &seeds {
        for xi in -2..=2 {
            for yi in -2..=2 {
                for zi in -2..=2 {
                    let x = xi as f32 * 0.41 + 0.13;
                    let y = yi as f32 * 0.47 - 0.31;
                    let z = zi as f32 * 0.59 + 0.09;
                    queries.push(WorleyQuery {
                        point: Vec3::new(x, y, z),
                        seed,
                    });
                }
            }
        }
    }

    let gpu = gpu_worley.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // The field is not constant: at least two sampled values must differ well
    // beyond the parity tolerance, proving a real nearest-point search ran.
    let first = gpu[0];
    assert!(
        gpu.iter().any(|&v| (v - first).abs() > 1e-3),
        "worley field must vary across space, not return a constant"
    );
}

#[test]
fn gpu_worley_is_spatially_continuous() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_worley = GpuWorley::new(&ctx);

    // A dense walk along a line with a tiny step. The Worley distance field is
    // 1-Lipschitz (the nearest-point distance changes no faster than the
    // sample moves), so neighbouring samples must stay close even where the
    // nearest feature point switches. A broken hash or wrong neighbourhood
    // would produce discontinuous jumps.
    let seed = 99u32;
    let step = 0.02f32;
    let queries: Vec<WorleyQuery> = (0..=200)
        .map(|k| {
            let t = k as f32 * step;
            WorleyQuery {
                point: Vec3::new(0.7 - 0.3 * t, t, 0.4 + 0.5 * t),
                seed,
            }
        })
        .collect();

    let gpu = gpu_worley.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // The line's per-step spatial displacement is sqrt(0.3^2+1^2+0.5^2)*step
    // ~= 0.0235; the field is 1-Lipschitz, so the value change per step is
    // bounded by that. 0.1 is a comfortable upper bound well below a spurious
    // discontinuity from a broken port.
    for w in gpu.windows(2) {
        let jump = (w[1] - w[0]).abs();
        assert!(
            jump < 0.1,
            "worley must be spatially continuous: neighbouring samples jumped {jump}"
        );
    }
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_worley = GpuWorley::new(&ctx);
    let out = gpu_worley.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
