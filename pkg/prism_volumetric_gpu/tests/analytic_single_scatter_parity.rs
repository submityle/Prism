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

/// Base absolute tolerance for the radiance in the well-conditioned regime. The
/// polynomial `exp_approx` is shared by both sides, so agreement is close;
/// radiance products can be a few units, so the tolerance is scaled
/// accordingly.
const TOL: f32 = 2e-5;

/// Per-evaluation numerator spread (a few `f32` `ULP` at magnitude 1) used to
/// widen the tolerance in the ill-conditioned regime. The path integral
/// `(1 - exp(-st*d)) / st` divides a numerator in `[0, 1]` by `st`; when
/// `sigma_t` is non-positive it is floored to `EPS = 1e-6`, so the shared
/// `exp_approx` rounding (which the `CPU` and `GPU` contract into `FMA`s
/// independently) is amplified by `1 / st`. The admissible spread of the
/// radiance is therefore `c / st * NUM_ULP` with `c = sigma_s*phase*L` the
/// non-negative product prefactor. Half a `ULP` at 1.0 is `2^-24 ~ 6e-8`; two
/// independent roundings differ by up to twice that, and `2e-7` keeps a safety
/// margin while staying far below any real port error (a wrong formula shifts
/// the value by `O(radiance)`, not a few `ULP`).
const NUM_ULP: f32 = 2e-7;

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
        // Conditioning-aware tolerance: the floored `st` amplifies the shared
        // `exp_approx` numerator rounding by `1 / st`, scaled by the radiance
        // prefactor `c`. In the well-conditioned regime (`st` not floored, `c`
        // moderate) this collapses back to `TOL`.
        let c = q.sigma_s.max(0.0) * q.phase.max(0.0) * q.light_radiance.max(0.0);
        let st = q.sigma_t.max(1e-6);
        let tol = TOL + (c / st) * NUM_ULP;
        assert!(
            (got - want).abs() <= tol,
            "radiance mismatch for {q:?}: gpu={got} cpu={want} (tol {tol})"
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
