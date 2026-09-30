//! Real-device parity for the analytic single-scatter twin:
//! [`GpuAnalyticSingleScatter`] must reproduce the `CPU` golden
//! [`analytic_single_scatter`](prism_render_architecture::volumetric::reference::analytic_single_scatter)
//! across a deterministic grid of extinction / scattering coefficients, phase
//! values, light radiances and distances, including zeros and negatives (which
//! must clamp) and large optical depths (where the path integral saturates).
//!
//! The test skips (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The radiance is asserted against the `CPU` golden within a tight absolute
//! tolerance. The `CPU` golden and the kernel share the same polynomial
//! `exp_approx`, so agreement is close. The test also checks that the value
//! stays non-negative, is exactly zero when any multiplicative factor is zero,
//! and grows monotonically with distance at fixed coefficients.
//!
//! Provenance: standard radiative-transfer single-scatter integral; no Unreal
//! Engine source or derived code.

use prism_render_architecture::volumetric::reference::analytic_single_scatter;
use prism_volumetric_gpu::{AnalyticSingleScatterQuery, GpuAnalyticSingleScatter, GpuContext};

/// Absolute tolerance for the radiance. The polynomial `exp_approx` is shared by
/// both sides, so agreement is close; radiance products can be a few units, so
/// the tolerance is scaled accordingly.
const TOL: f32 = 2e-5;

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_analytic_single_scatter_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping analytic-single-scatter parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuAnalyticSingleScatter::new(&ctx);

    // Coefficients / phase / radiance / distance across zero, negatives (must
    // clamp), small, moderate and large values.
    let sigma_ts = [-1.0_f32, 0.0, 0.02, 0.25, 1.0, 6.0];
    let sigma_ss = [-0.5_f32, 0.0, 0.1, 0.8];
    let phases = [-0.3_f32, 0.0, 0.0796, 0.5];
    let radiances = [-2.0_f32, 0.0, 1.0, 12.0];
    let distances = [-1.0_f32, 0.0, 0.1, 1.0, 5.0, 30.0];

    let mut queries: Vec<AnalyticSingleScatterQuery> = Vec::new();
    for &sigma_t in &sigma_ts {
        for &sigma_s in &sigma_ss {
            for &phase in &phases {
                for &light_radiance in &radiances {
                    for &distance in &distances {
                        queries.push(AnalyticSingleScatterQuery {
                            sigma_t,
                            sigma_s,
                            phase,
                            light_radiance,
                            distance,
                        });
                    }
                }
            }
        }
    }
    assert!(!queries.is_empty(), "the parity grid must be non-empty");

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_eq!(gpu.len(), queries.len());

    for (q, &got) in queries.iter().zip(gpu.iter()) {
        let want =
            analytic_single_scatter(q.sigma_t, q.sigma_s, q.phase, q.light_radiance, q.distance);
        assert!(
            (got - want).abs() <= TOL,
            "radiance mismatch for {q:?}: gpu={got} cpu={want}"
        );
        assert!(
            got >= -TOL,
            "radiance must be non-negative for {q:?}: {got}"
        );
        // Any non-positive multiplicative factor (or zero distance) zeroes it.
        if q.sigma_s <= 0.0 || q.phase <= 0.0 || q.light_radiance <= 0.0 || q.distance <= 0.0 {
            assert!(
                got.abs() <= TOL,
                "a zero factor must yield zero radiance for {q:?}: {got}"
            );
        }
    }

    // Monotonic non-decrease in distance: sweep distance at fixed positive
    // coefficients (the path integral (1 - exp(-st*d))/st grows with d).
    let ramp: Vec<AnalyticSingleScatterQuery> = [0.0_f32, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 20.0]
        .into_iter()
        .map(|distance| AnalyticSingleScatterQuery {
            sigma_t: 0.6,
            sigma_s: 0.4,
            phase: 0.12,
            light_radiance: 3.0,
            distance,
        })
        .collect();
    let ramp_gpu = gpu_kernel.eval(&ctx, &ramp);
    for pair in ramp_gpu.windows(2) {
        assert!(
            pair[1] >= pair[0] - TOL,
            "radiance must be non-decreasing in distance: {pair:?}"
        );
    }
}
