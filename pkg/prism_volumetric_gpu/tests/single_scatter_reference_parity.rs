//! Real-device parity for the single-scatter reference twin:
//! [`GpuSingleScatterReference`] must reproduce the `CPU` golden
//! [`single_scatter_reference`](prism_render_architecture::volumetric::reference::single_scatter_reference)
//! for a homogeneous medium and a constant phase closure (`|_| phase`) across a
//! deterministic grid of coefficients, phase values, radiances, distances and
//! seeds.
//!
//! The test skips (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The `RNG`, `ln_approx` and arithmetic are shared bit-for-bit between the two
//! sides; the only spec-permitted divergence is floating-point division, which
//! may differ by a few `ULP` across devices and could push a rare boundary
//! walk across the `t < distance` test, changing the hit count by one. Each hit
//! contributes `sigma_s * phase * light_radiance / sigma_t` divided by the
//! sample count, so the parity criterion allows a small integer-flip margin
//! scaled by that per-sample contribution. In practice a correctly-rounding
//! device agrees exactly.
//!
//! The test also checks non-negativity, the exact-zero identity when any
//! multiplicative factor or the distance is non-positive, and convergence to
//! the analytic single-scatter radiance as `samples` grows.
//!
//! Provenance: standard free-flight importance-sampled single-scatter
//! estimator; no Unreal Engine source or derived code.

use prism_render_architecture::volumetric::reference::{
    analytic_single_scatter, single_scatter_reference,
};
use prism_volumetric_gpu::{GpuContext, GpuSingleScatterReference, SingleScatterReferenceQuery};

/// Samples per estimate for the bit-parity grid.
const PARITY_SAMPLES: u32 = 512;

/// Allowed hit-count divergence, in samples, from device division `ULP`
/// differences flipping a rare boundary walk. Converted to a radiance
/// tolerance by scaling with the per-sample contribution.
const FLIP_MARGIN: f32 = 4.0;

/// Small absolute floor added to every tolerance to absorb the final division.
const ABS_FLOOR: f32 = 1e-6;

/// `EPS` floor the golden applies to `sigma_t`.
const EPS: f32 = 1e-6;

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_single_scatter_reference_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping single-scatter reference parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuSingleScatterReference::new(&ctx);

    // sigma_t / sigma_s / phase / light_radiance / distance / seed grid,
    // including zeros and negatives (which clamp) and a thick optical regime.
    let sigma_ts = [-0.1_f32, 0.0, 0.2, 0.75, 2.0];
    let sigma_ss = [0.0_f32, 0.3, 1.0];
    let phases = [-0.2_f32, 0.0, 0.0796, 0.4];
    let radiances = [0.0_f32, 1.0, 6.0];
    let distances = [0.3_f32, 1.5, 6.0];
    let seeds = [3u32, 0x0BAD_F00D];

    let mut queries: Vec<SingleScatterReferenceQuery> = Vec::new();
    for &sigma_t in &sigma_ts {
        for &sigma_s in &sigma_ss {
            for &phase in &phases {
                for &light_radiance in &radiances {
                    for &distance in &distances {
                        for &seed in &seeds {
                            queries.push(SingleScatterReferenceQuery {
                                sigma_t,
                                sigma_s,
                                light_radiance,
                                phase,
                                distance,
                                seed,
                                samples: PARITY_SAMPLES,
                            });
                        }
                    }
                }
            }
        }
    }
    assert!(!queries.is_empty(), "the parity grid must be non-empty");

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_eq!(gpu.len(), queries.len());

    for (q, &got) in queries.iter().zip(gpu.iter()) {
        let want = single_scatter_reference(
            q.sigma_t,
            q.sigma_s,
            q.light_radiance,
            |_| q.phase,
            q.distance,
            q.seed,
            q.samples,
        );
        // Per-sample contribution magnitude sets the flip tolerance.
        let contrib =
            q.sigma_s.max(0.0) * q.phase.max(0.0) * q.light_radiance.max(0.0) / q.sigma_t.max(EPS);
        let tol = FLIP_MARGIN * contrib / PARITY_SAMPLES as f32 + ABS_FLOOR;
        assert!(
            got >= -ABS_FLOOR,
            "radiance must be non-negative for {q:?}: {got}"
        );
        assert!(
            (got - want).abs() <= tol,
            "radiance mismatch for {q:?}: gpu={got} cpu={want} tol={tol}"
        );
        // Any non-positive factor or distance zeroes the radiance.
        if q.sigma_s <= 0.0 || q.phase <= 0.0 || q.light_radiance <= 0.0 || q.distance <= 0.0 {
            assert_eq!(got, 0.0, "a zero factor must yield zero radiance for {q:?}");
        }
    }

    // Convergence: with many samples the estimator approaches the analytic
    // single-scatter radiance for a constant phase.
    let conv_samples = 32_768u32;
    let conv_cases = [
        (0.3_f32, 0.5_f32, 0.12_f32, 3.0_f32, 2.0_f32),
        (0.8, 0.6, 0.0796, 5.0, 3.0),
        (1.5, 1.0, 0.25, 2.0, 1.5),
    ];
    let conv_queries: Vec<SingleScatterReferenceQuery> = conv_cases
        .iter()
        .map(
            |&(sigma_t, sigma_s, phase, light_radiance, distance)| SingleScatterReferenceQuery {
                sigma_t,
                sigma_s,
                light_radiance,
                phase,
                distance,
                seed: 0x51CA_77E5,
                samples: conv_samples,
            },
        )
        .collect();
    let conv = gpu_kernel.eval(&ctx, &conv_queries);
    for (q, &got) in conv_queries.iter().zip(conv.iter()) {
        let analytic =
            analytic_single_scatter(q.sigma_t, q.sigma_s, q.phase, q.light_radiance, q.distance);
        // Estimator standard error ~ magnitude / sqrt(N); allow a generous
        // multiple.
        let mc_tol = 12.0 * analytic.max(EPS) / (conv_samples as f32).sqrt();
        assert!(
            (got - analytic).abs() <= mc_tol,
            "estimate must converge to analytic for {q:?}: got={got} analytic={analytic} tol={mc_tol}"
        );
    }
}
