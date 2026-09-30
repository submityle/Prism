//! Cross-kernel energy-conservation invariant (design section 9c) between the
//! two analytic reference twins on a real device:
//! [`GpuAnalyticSingleScatter`] and [`GpuAnalyticTransmittance`].
//!
//! For a homogeneous slab lit by a constant light radiance the single-scattered
//! radiance and the Beer-Lambert transmittance are not independent — they are
//! linked by energy conservation. The scattered radiance is exactly the albedo-
//! and phase-weighted fraction of the light that the medium extinguished along
//! the path:
//!
//! ```text
//! radiance = sigma_s * phase * light_radiance * (1 - T) / sigma_t
//! T        = exp(-sigma_t * distance)
//! ```
//!
//! This test dispatches both GPU kernels over one deterministic grid and checks
//! that the single-scatter kernel's output equals the value reconstructed from
//! the *other* kernel's transmittance. It is not a CPU-vs-GPU parity test (those
//! live in `analytic_single_scatter_parity.rs` and
//! `analytic_transmittance_parity.rs`); it is a GPU-vs-GPU consistency check
//! that catches drift in either kernel's polynomial `exp` independently, since
//! a per-kernel CPU parity test cannot see a shared or asymmetric error that
//! keeps each kernel individually close to its own golden.
//!
//! # Why the identity holds tightly
//!
//! Both kernels evaluate the *same* hand-rolled polynomial `exp_approx` on the
//! *same* argument `-sigma_t * distance`. As long as `sigma_t` is above the
//! single-scatter kernel's `EPS` floor (the grid uses `sigma_t >= 0.2`, far
//! above `1e-6`), the single-scatter kernel's internal `st = max(sigma_t, EPS)`
//! equals `sigma_t`, so both kernels feed an identical argument to an identical
//! polynomial and produce an identical transmittance. The transmittance
//! kernel's `saturate` is a no-op there (`exp` of a non-positive argument lies
//! in `(0, 1]`). The only spec-permitted divergence is the association of the
//! final `sigma_s * phase * light * integral` product versus the reconstructed
//! `factors * (1 - T) / sigma_t`, which differs by at most a few `ULP`.
//!
//! # Degenerate cases
//!
//! When any multiplicative factor (`sigma_s`, `phase`, `light_radiance`) or the
//! distance is non-positive the scattered radiance must be *exactly* zero; the
//! grid includes those rows and the test asserts the hard zero, so the tolerance
//! band can never launder a spurious non-zero radiance.
//!
//! The test skips (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device.
//!
//! Provenance: standard radiative-transfer single-scatter / Beer-Lambert
//! energy balance; no Unreal Engine source or derived code.

use prism_volumetric_gpu::{
    AnalyticSingleScatterQuery, AnalyticTransmittanceQuery, GpuAnalyticSingleScatter,
    GpuAnalyticTransmittance, GpuContext,
};

/// Absolute floor absorbing the final reconstruction division / product.
const ABS_TOL: f32 = 1e-6;

/// Relative tolerance for the well-conditioned identity. The two kernels share
/// the polynomial `exp` bit-for-bit on the same argument, so the only slack is
/// the association of the closing multiply/divide chain — a few `ULP`.
const REL_TOL: f32 = 1e-5;

/// Extinction values, all above the single-scatter `EPS` floor so `st` equals
/// `sigma_t` and the shared `exp` argument is identical across both kernels.
const SIGMA_T: [f32; 5] = [0.2, 0.5, 1.0, 2.5, 8.0];

/// Scattering coefficients, including the exact-zero degenerate row.
const SIGMA_S: [f32; 4] = [0.0, 0.3, 0.9, 1.7];

/// Phase-function values toward the eye, including the exact-zero row.
const PHASE: [f32; 4] = [0.0, 0.079_577_47, 0.5, 1.2];

/// Incident light radiances, including the exact-zero row.
const LIGHT: [f32; 3] = [0.0, 1.0, 4.0];

/// Path distances, including the exact-zero row and a large optical depth.
const DISTANCE: [f32; 5] = [0.0, 0.1, 0.7, 3.0, 20.0];

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice and the measured worst-case divergence must reach the test log"
)]
fn gpu_single_scatter_and_transmittance_conserve_energy() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!(
            "skipping single-scatter / transmittance energy-conservation: no wgpu adapter on this host"
        );
        return;
    };

    // Build the deterministic cartesian grid in a fixed order so both dispatches
    // and the reconstruction index the same physical configuration.
    let mut ss_queries: Vec<AnalyticSingleScatterQuery> = Vec::new();
    let mut tr_queries: Vec<AnalyticTransmittanceQuery> = Vec::new();
    for &sigma_t in &SIGMA_T {
        for &sigma_s in &SIGMA_S {
            for &phase in &PHASE {
                for &light in &LIGHT {
                    for &distance in &DISTANCE {
                        ss_queries.push(AnalyticSingleScatterQuery {
                            sigma_t,
                            sigma_s,
                            phase,
                            light_radiance: light,
                            distance,
                        });
                        tr_queries.push(AnalyticTransmittanceQuery { sigma_t, distance });
                    }
                }
            }
        }
    }
    assert_eq!(ss_queries.len(), tr_queries.len());
    assert!(!ss_queries.is_empty(), "energy-conservation grid is empty");

    let ss_kernel = GpuAnalyticSingleScatter::new(&ctx);
    let tr_kernel = GpuAnalyticTransmittance::new(&ctx);
    let radiance = ss_kernel.eval(&ctx, &ss_queries);
    let transmittance = tr_kernel.eval(&ctx, &tr_queries);
    assert_eq!(radiance.len(), ss_queries.len());
    assert_eq!(transmittance.len(), tr_queries.len());

    let mut worst_abs = 0.0_f32;
    let mut worst_rel = 0.0_f32;
    for (i, q) in ss_queries.iter().enumerate() {
        let t = transmittance[i];
        let got = radiance[i];

        // Reconstruct the single-scatter radiance from the *transmittance*
        // kernel's output, mirroring the single-scatter kernel's clamps and
        // multiply order so any divergence is purely numerical, not structural.
        let factors = q.sigma_s.max(0.0) * q.phase.max(0.0) * q.light_radiance.max(0.0);
        let integral = (1.0 - t) / q.sigma_t;
        let expected = factors * integral;

        // Every result stays physical.
        assert!(
            got.is_finite() && got >= 0.0,
            "radiance must be finite and non-negative, got {got} for {q:?}",
        );
        assert!(
            (0.0..=1.0).contains(&t),
            "transmittance must be in [0, 1], got {t} for sigma_t={} distance={}",
            q.sigma_t,
            q.distance,
        );

        // Degenerate rows: a zero multiplicative factor or a zero path must give
        // exactly zero scattered radiance, with no tolerance slack.
        if factors == 0.0 || q.distance <= 0.0 {
            assert_eq!(
                got, 0.0,
                "radiance must be exactly zero when a factor or the distance vanishes, got {got} for {q:?}",
            );
            continue;
        }

        let abs = (got - expected).abs();
        let rel = abs / expected.abs().max(f32::MIN_POSITIVE);
        worst_abs = worst_abs.max(abs);
        worst_rel = worst_rel.max(rel);
        assert!(
            abs <= ABS_TOL || rel <= REL_TOL,
            "energy-conservation identity violated for {q:?}: gpu_single_scatter={got}, \
             reconstructed_from_transmittance={expected} (T={t}), abs={abs}, rel={rel}",
        );
    }

    eprintln!(
        "energy-conservation over {} queries: worst_abs={worst_abs:e}, worst_rel={worst_rel:e}",
        ss_queries.len(),
    );
}
