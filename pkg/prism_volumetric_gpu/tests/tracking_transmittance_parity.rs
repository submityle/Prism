//! Real-device parity for the tracking-transmittance twin:
//! [`GpuTrackingTransmittance`] must reproduce the `CPU` goldens
//! [`delta_tracking_transmittance`](prism_render_architecture::volumetric::reference::delta_tracking_transmittance)
//! and
//! [`ratio_tracking_transmittance`](prism_render_architecture::volumetric::reference::ratio_tracking_transmittance)
//! for a homogeneous medium (a constant `|_| sigma_t` extinction closure)
//! across a deterministic grid of extinction / majorant / distance / seed
//! combinations.
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
//! may differ by a few `ULP` across devices. For ratio tracking that shows up
//! as a tiny drift in the accumulated product, checked against a tight absolute
//! tolerance. For delta tracking a `ULP` difference could push a rare boundary
//! walk across its accept threshold or past `distance`, flipping the integer
//! survival count by one; the criterion therefore allows a small integer-flip
//! margin scaled by `1 / samples`. In practice a correctly-rounding device
//! agrees exactly.
//!
//! The test also checks that both estimates stay in `[0, 1]`, that a
//! zero-length segment or empty sample budget yields exactly `1`, that both
//! estimators converge to the analytic Beer-Lambert transmittance as `samples`
//! grows, and that ratio tracking sits no further from the analytic value than
//! delta tracking (its variance is provably no higher).
//!
//! Provenance: standard delta / ratio tracking estimators (Novak et al.); no
//! Unreal Engine source or derived code.

use prism_render_architecture::volumetric::reference::{
    analytic_transmittance, delta_tracking_transmittance, ratio_tracking_transmittance,
};
use prism_volumetric_gpu::{GpuContext, GpuTrackingTransmittance, TrackingTransmittanceQuery};

/// Samples per walk for the bit-parity grid. Large enough to exercise many
/// interior events, small enough to keep the sweep fast on device.
const PARITY_SAMPLES: u32 = 512;

/// Absolute tolerance for the ratio estimate. Only floating-point division may
/// differ across devices, so the accumulated product drift is tiny.
const RATIO_TOL: f32 = 1e-4;

/// Allowed delta-count divergence, in survival counts, from device division
/// `ULP` differences flipping a rare boundary walk. Converted to a
/// transmittance tolerance by dividing by the sample count.
const DELTA_FLIP_MARGIN: f32 = 4.0;

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_tracking_transmittance_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping tracking-transmittance parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuTrackingTransmittance::new(&ctx);

    let delta_tol = DELTA_FLIP_MARGIN / PARITY_SAMPLES as f32;

    // Extinction / majorant (>= sigma_t) / distance / seed grid. Includes a
    // negative sigma_t (clamps to zero -> fully transmissive) and a thin,
    // moderate and thick optical regime.
    let sigma_ts = [-0.2_f32, 0.0, 0.15, 0.6, 1.5, 4.0];
    let majorant_scales = [1.0_f32, 1.5, 3.0];
    let distances = [0.2_f32, 1.0, 3.0, 8.0];
    let seeds = [1u32, 0x1234_5678, 0xDEAD_BEEF];

    let mut queries: Vec<TrackingTransmittanceQuery> = Vec::new();
    for &sigma_t in &sigma_ts {
        for &scale in &majorant_scales {
            // Majorant must upper-bound sigma_t; floor it so it stays positive
            // even for the non-positive sigma_t cases.
            let majorant = (sigma_t.max(0.0) * scale).max(0.05);
            for &distance in &distances {
                for &seed in &seeds {
                    queries.push(TrackingTransmittanceQuery {
                        sigma_t,
                        majorant,
                        distance,
                        seed,
                        samples: PARITY_SAMPLES,
                    });
                }
            }
        }
    }
    assert!(!queries.is_empty(), "the parity grid must be non-empty");

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_eq!(gpu.len(), queries.len());

    for (q, &got) in queries.iter().zip(gpu.iter()) {
        let want_delta =
            delta_tracking_transmittance(|_| q.sigma_t, q.majorant, q.distance, q.seed, q.samples);
        let want_ratio =
            ratio_tracking_transmittance(|_| q.sigma_t, q.majorant, q.distance, q.seed, q.samples);

        assert!(
            (0.0..=1.0).contains(&got.delta) && (0.0..=1.0).contains(&got.ratio),
            "estimates must stay in [0, 1] for {q:?}: {got:?}"
        );
        assert!(
            (got.delta - want_delta).abs() <= delta_tol,
            "delta mismatch for {q:?}: gpu={} cpu={want_delta}",
            got.delta
        );
        assert!(
            (got.ratio - want_ratio).abs() <= RATIO_TOL,
            "ratio mismatch for {q:?}: gpu={} cpu={want_ratio}",
            got.ratio
        );
    }

    // Zero distance and empty budget both yield exactly 1.0 on device.
    let edge = gpu_kernel.eval(
        &ctx,
        &[
            TrackingTransmittanceQuery {
                sigma_t: 0.8,
                majorant: 1.0,
                distance: 0.0,
                seed: 7,
                samples: PARITY_SAMPLES,
            },
            TrackingTransmittanceQuery {
                sigma_t: 0.8,
                majorant: 1.0,
                distance: 5.0,
                seed: 7,
                samples: 0,
            },
        ],
    );
    for e in &edge {
        assert_eq!(e.delta, 1.0, "zero-length / empty budget delta must be 1.0");
        assert_eq!(e.ratio, 1.0, "zero-length / empty budget ratio must be 1.0");
    }

    // Convergence: with many samples both estimators approach the analytic
    // Beer-Lambert transmittance, and ratio tracking is no further from it than
    // delta tracking (its variance is provably no higher).
    let conv_samples = 16_384u32;
    let conv_cases = [
        (0.3_f32, 1.0_f32, 2.0_f32),
        (0.7, 1.0, 3.0),
        (1.2, 2.0, 1.5),
    ];
    let conv_queries: Vec<TrackingTransmittanceQuery> = conv_cases
        .iter()
        .map(
            |&(sigma_t, majorant, distance)| TrackingTransmittanceQuery {
                sigma_t,
                majorant,
                distance,
                seed: 0xABCD_1234,
                samples: conv_samples,
            },
        )
        .collect();
    let conv = gpu_kernel.eval(&ctx, &conv_queries);
    // Monte-Carlo standard error ~ 1/sqrt(N); allow a generous multiple.
    let mc_tol = 8.0 / (conv_samples as f32).sqrt();
    for (q, e) in conv_queries.iter().zip(conv.iter()) {
        let analytic = analytic_transmittance(q.sigma_t, q.distance);
        assert!(
            (e.delta - analytic).abs() <= mc_tol,
            "delta must converge to analytic for {q:?}: got={} analytic={analytic}",
            e.delta
        );
        assert!(
            (e.ratio - analytic).abs() <= mc_tol,
            "ratio must converge to analytic for {q:?}: got={} analytic={analytic}",
            e.ratio
        );
        assert!(
            (e.ratio - analytic).abs() <= (e.delta - analytic).abs() + mc_tol,
            "ratio must be no further from analytic than delta for {q:?}: \
             ratio={} delta={} analytic={analytic}",
            e.ratio,
            e.delta
        );
    }
}
