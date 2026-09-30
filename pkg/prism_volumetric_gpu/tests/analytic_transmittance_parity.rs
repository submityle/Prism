//! Real-device parity for the Beer-Lambert transmittance twin:
//! [`GpuAnalyticTransmittance`] must reproduce the `CPU` golden
//! [`analytic_transmittance`](prism_render_architecture::volumetric::reference::analytic_transmittance)
//! across a deterministic grid of extinction coefficients and distances,
//! including negatives (which must clamp) and large optical depths (which must
//! saturate toward zero).
//!
//! The test skips (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The transmittance is asserted against the `CPU` golden within a tight
//! absolute tolerance. The `CPU` golden and the kernel share the same
//! polynomial `exp_approx`, so agreement is to a few ULPs. The test also checks
//! that the value stays in `[0, 1]`, is exactly `1` at zero optical depth and is
//! non-increasing as the optical depth grows.
//!
//! Provenance: standard Beer-Lambert transmittance; no Unreal Engine source or
//! derived code.

use prism_render_architecture::volumetric::reference::analytic_transmittance;
use prism_volumetric_gpu::{AnalyticTransmittanceQuery, GpuAnalyticTransmittance, GpuContext};

/// Absolute tolerance for the transmittance. The polynomial `exp_approx` is
/// shared by both sides, so agreement is close.
const TOL: f32 = 1e-6;

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_analytic_transmittance_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping analytic-transmittance parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuAnalyticTransmittance::new(&ctx);

    // sigma_t and distance across zero, negatives (must clamp), small, moderate
    // and large values (large products drive transmittance toward zero).
    let sigma_ts = [-1.0_f32, 0.0, 0.01, 0.1, 0.5, 1.0, 4.0, 20.0];
    let distances = [-2.0_f32, 0.0, 0.05, 0.5, 1.0, 3.0, 10.0, 50.0];

    let mut queries: Vec<AnalyticTransmittanceQuery> = Vec::new();
    for &sigma_t in &sigma_ts {
        for &distance in &distances {
            queries.push(AnalyticTransmittanceQuery { sigma_t, distance });
        }
    }
    assert!(!queries.is_empty(), "the parity grid must be non-empty");

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_eq!(gpu.len(), queries.len());

    for (q, &got) in queries.iter().zip(gpu.iter()) {
        let want = analytic_transmittance(q.sigma_t, q.distance);
        assert!(
            (got - want).abs() <= TOL,
            "transmittance mismatch for {q:?}: gpu={got} cpu={want}"
        );
        assert!(
            (0.0..=1.0).contains(&got),
            "transmittance out of [0,1] for {q:?}: {got}"
        );
        // A non-positive optical depth (either factor clamped to zero) is 1.
        if q.sigma_t <= 0.0 || q.distance <= 0.0 {
            assert!(
                (got - 1.0).abs() <= TOL,
                "zero optical depth must be transmittance 1 for {q:?}: {got}"
            );
        }
    }

    // Monotonic non-increase in optical depth: sweep distance at a fixed sigma_t.
    let ramp: Vec<AnalyticTransmittanceQuery> = [0.0_f32, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        .into_iter()
        .map(|distance| AnalyticTransmittanceQuery {
            sigma_t: 0.7,
            distance,
        })
        .collect();
    let ramp_gpu = gpu_kernel.eval(&ctx, &ramp);
    for pair in ramp_gpu.windows(2) {
        assert!(
            pair[1] <= pair[0] + TOL,
            "transmittance must be non-increasing in optical depth: {pair:?}"
        );
    }
}
