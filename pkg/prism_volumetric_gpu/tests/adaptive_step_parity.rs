//! Real-device parity for the adaptive-step twin:
//! [`GpuAdaptiveStep`] must reproduce the `CPU` golden
//! [`adaptive_step`](prism_render_architecture::volumetric::raymarch::adaptive_step)
//! across a deterministic grid of raymarch configs, sample densities and the
//! two `in_cloud` states, including the empty-space (below-threshold) branch
//! and the densest-medium clamp.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The math is pure `clamp`/`lerp`/`saturate` with no transcendental, so the
//! kernel mirrors the `CPU` golden exactly and agreement is asserted within a
//! tight absolute tolerance. The suite also checks the invariants that make the
//! kernel correct: the result always lies in `[min_step, max_step]`, empty
//! space returns `max_step`, and inside the cloud the step is monotone
//! non-increasing as density rises.
//!
//! Provenance: standard density-adaptive raymarch step selection; no Unreal
//! Engine source or derived code.

use prism_render_architecture::volumetric::raymarch::{adaptive_step, RaymarchConfig};
use prism_volumetric_gpu::{AdaptiveStepQuery, GpuAdaptiveStep, GpuContext};

/// Absolute tolerance: pure arithmetic, so the on-device result matches the
/// `CPU` golden to the last few ULPs, well under this.
const TOL: f32 = 1e-6;

/// A deterministic set of raymarch configs, including the [`Default`] and
/// custom bounds (wide, narrow, and near-degenerate).
fn configs() -> Vec<RaymarchConfig> {
    vec![
        RaymarchConfig::default(),
        RaymarchConfig {
            base_step: 4.0,
            max_step: 32.0,
            min_step: 0.5,
            density_threshold: 0.05,
            ..RaymarchConfig::default()
        },
        RaymarchConfig {
            base_step: 2.0,
            max_step: 2.0,
            min_step: 2.0,
            density_threshold: 0.1,
            ..RaymarchConfig::default()
        },
        RaymarchConfig {
            base_step: 16.0,
            max_step: 128.0,
            min_step: 2.0,
            density_threshold: 1.0e-4,
            ..RaymarchConfig::default()
        },
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_adaptive_step_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping adaptive-step parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuAdaptiveStep::new(&ctx);

    // A deterministic density sweep including below-threshold, near-threshold,
    // mid, and saturated / super-unit densities, crossed with both in_cloud
    // states and every config.
    let densities = [
        0.0_f32, 1.0e-5, 1.0e-3, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 3.0,
    ];

    let mut queries: Vec<AdaptiveStepQuery> = Vec::new();
    for cfg in configs() {
        for &in_cloud in &[false, true] {
            for &current_density in &densities {
                queries.push(AdaptiveStepQuery {
                    current_density,
                    in_cloud,
                    cfg,
                });
            }
        }
    }

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_eq!(gpu.len(), queries.len(), "one result per query");

    for (i, q) in queries.iter().enumerate() {
        let exp = adaptive_step(q.current_density, q.in_cloud, q.cfg);
        assert!(
            (gpu[i] - exp).abs() <= TOL,
            "adaptive-step mismatch for query {i} (density {}, in_cloud {}): \
             gpu {}, cpu {exp}, |diff| {}",
            q.current_density,
            q.in_cloud,
            gpu[i],
            (gpu[i] - exp).abs()
        );

        // The step always lies in `[min_step, max_step]`.
        let lo = q.cfg.min_step - TOL;
        let hi = q.cfg.max_step + TOL;
        assert!(
            gpu[i] >= lo && gpu[i] <= hi,
            "adaptive-step out of [min,max] for query {i}: gpu {} outside [{lo}, {hi}]",
            gpu[i]
        );
    }

    // Empty space (not in_cloud) always returns `max_step`, regardless of
    // density.
    for cfg in configs() {
        let empty = AdaptiveStepQuery {
            current_density: 0.9,
            in_cloud: false,
            cfg,
        };
        let out = gpu_kernel.eval(&ctx, &[empty]);
        let expected = adaptive_step(empty.current_density, empty.in_cloud, empty.cfg);
        assert!(
            (out[0] - expected).abs() <= TOL,
            "empty-space step mismatch: gpu {}, cpu {expected}",
            out[0]
        );
    }

    // Inside the cloud the step is monotone non-increasing as density rises.
    let mono_cfg = RaymarchConfig::default();
    let rising = [0.05_f32, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0];
    let mono_queries: Vec<AdaptiveStepQuery> = rising
        .iter()
        .map(|&current_density| AdaptiveStepQuery {
            current_density,
            in_cloud: true,
            cfg: mono_cfg,
        })
        .collect();
    let mono = gpu_kernel.eval(&ctx, &mono_queries);
    for pair in mono.windows(2) {
        assert!(
            pair[1] <= pair[0] + TOL,
            "in-cloud step must be monotone non-increasing in density: {} then {}",
            pair[0],
            pair[1]
        );
    }
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuAdaptiveStep::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
