//! Real-device parity for the god-ray sample-weight twin: [`GpuGodRayWeight`]
//! must reproduce the `CPU` golden
//! [`god_ray_weight`](prism_render_architecture::volumetric::shadow::god_ray_weight)
//! across a spread of sample indices and beam configs.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The power `decay^index` mirrors the reference's hand-rolled `pow_approx`
//! (`exp_approx(index * ln_approx(decay))`, with both the base-two `exp` and the
//! exponent-extraction / `atanh`-series `ln` mirrored bit-for-bit) rather than
//! the device-native `pow`, so `CPU` and `GPU` evaluate the same closed-form
//! algebra. Values are asserted to within `abs_diff < 1e-5` or `rel_diff < 1e-4`
//! — slightly looser than the single-`exp` kernels because the `ln`-then-`exp`
//! chain admits a few more legal multiply-add contractions, yet tight enough to
//! fail a wrong port (a dropped saturate, a swapped `decay`/`weight`, a
//! mis-scaled index). The scenes also assert the documented monotonicity
//! (weights fall with the sample index), the `[0, 1]` range and the
//! geometric-series partial-sum bound, so a degenerate kernel could not pass.
//!
//! Provenance: standard radial god-ray geometric decay; no Unreal Engine source
//! or derived code.

use prism_render_architecture::volumetric::shadow::{god_ray_weight, GodRayConfig};
use prism_volumetric_gpu::{GodRayWeightQuery, GpuContext, GpuGodRayWeight};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays inside the unit range.
fn assert_parity(queries: &[GodRayWeightQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one weight per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = god_ray_weight(q.sample_index, q.config);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-5 || rel_diff < 1e-4,
            "godray mismatch for query {i} ({q:?}): gpu {got}, cpu {exp} \
             (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            (0.0..=1.0).contains(&got),
            "gpu weight must stay in the unit range: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_godray_matches_cpu_golden_across_queries() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping godray parity: no wgpu adapter on this host");
        return;
    };
    let gpu_godray = GpuGodRayWeight::new(&ctx);

    // A deterministic spread of configs: the default beam, a fast falloff, a
    // slow falloff, full weight, and out-of-range decay/weight that must clamp.
    let configs = [
        GodRayConfig::default(),
        GodRayConfig {
            sample_count: 32,
            decay: 0.80,
            weight: 0.5,
            exposure: 1.0,
        },
        GodRayConfig {
            sample_count: 96,
            decay: 0.99,
            weight: 0.9,
            exposure: 2.0,
        },
        GodRayConfig {
            sample_count: 16,
            decay: 1.5,
            weight: 1.3,
            exposure: 1.0,
        },
        GodRayConfig {
            sample_count: 16,
            decay: -0.2,
            weight: -0.4,
            exposure: 1.0,
        },
    ];

    let mut queries: Vec<GodRayWeightQuery> = Vec::new();
    for cfg in configs {
        for k in 0..48u32 {
            queries.push(GodRayWeightQuery {
                sample_index: k,
                config: cfg,
            });
        }
    }

    let gpu = gpu_godray.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // Index 0 of the default beam is exactly the saturated weight, and a later
    // sample of the fast-falloff beam must have decayed well below its root.
    let default_root = god_ray_weight(0, GodRayConfig::default());
    assert!(
        (gpu[0] - default_root).abs() < 1e-5,
        "the beam root equals the saturated weight"
    );
    // configs[1] starts at query index 48 (decay 0.8): sample 20 is 0.8^20.
    assert!(
        gpu[48 + 20] < gpu[48] * 0.05,
        "a fast-falloff beam decays strongly by sample 20"
    );
}

#[test]
fn gpu_godray_is_monotone_non_increasing_and_sum_bounded() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_godray = GpuGodRayWeight::new(&ctx);

    // A single beam swept over its sample indices: the weight falls monotonically
    // and the running sum stays under the geometric bound `weight / (1 - decay)`.
    let cfg = GodRayConfig {
        sample_count: 128,
        decay: 0.92,
        weight: 0.7,
        exposure: 1.0,
    };
    let queries: Vec<GodRayWeightQuery> = (0..128u32)
        .map(|k| GodRayWeightQuery {
            sample_index: k,
            config: cfg,
        })
        .collect();

    let gpu = gpu_godray.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    for w in gpu.windows(2) {
        assert!(
            w[1] <= w[0] + 1e-6,
            "god-ray weight must be monotone non-increasing: {} then {}",
            w[0],
            w[1]
        );
    }
    let sum: f32 = gpu.iter().sum();
    let bound = 0.7 / (1.0 - 0.92);
    assert!(
        sum <= bound + 1e-3,
        "the partial sum {sum} must stay under the geometric bound {bound}"
    );
    assert!(
        gpu[0] > gpu[gpu.len() - 1] + 1e-3,
        "the beam root must weigh strictly more than the far sample"
    );
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_godray = GpuGodRayWeight::new(&ctx);
    let out = gpu_godray.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
