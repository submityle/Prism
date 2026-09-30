//! Real-device parity for the sunset-reddening twin: [`GpuSunsetReddening`]
//! must reproduce the `CPU` golden
//! [`sunset_reddening`](prism_render_architecture::volumetric::spectral::sunset_reddening)
//! across the full range of sun altitudes, including altitudes at and below the
//! horizon and well above the reddening cutoff.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The `smoothstep` is expanded to the same closed form the CPU
//! `math::smoothstep` uses, and the kernel contains no transcendental call —
//! clamp/saturate and a multiply-add — so `CPU` and `GPU` evaluate the same
//! closed-form algebra. Values are asserted to within `abs_diff < 1e-6` or
//! `rel_diff < 1e-5` — tight enough to fail a wrong port (a dropped saturate, a
//! wrong cutoff). The scenes also assert the value stays in `0..=1`, is one at
//! and below the horizon, is zero above the cutoff, and is monotone
//! non-increasing in altitude, so a degenerate kernel could not pass.
//!
//! Provenance: standard horizon-reddening falloff; no Unreal Engine source or
//! derived code.

use prism_render_architecture::volumetric::spectral::sunset_reddening;
use prism_volumetric_gpu::{GpuContext, GpuSunsetReddening, SunsetReddeningQuery};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays in `0..=1`.
fn assert_parity(queries: &[SunsetReddeningQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one reddening value per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = sunset_reddening(q.sun_altitude);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "sunset reddening mismatch for query {i} ({q:?}): gpu {got}, cpu {exp} \
             (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            (0.0..=1.0).contains(&got),
            "gpu sunset reddening must stay in 0..=1: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_sunset_reddening_matches_cpu_golden_across_altitudes() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sunset reddening parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuSunsetReddening::new(&ctx);

    // A deterministic spread: below the horizon (full reddening), at the
    // horizon, mid-ramp, at the cutoff, and well above it (no reddening).
    let mut queries: Vec<SunsetReddeningQuery> = vec![
        SunsetReddeningQuery { sun_altitude: -0.5 },
        SunsetReddeningQuery { sun_altitude: 0.0 },
        SunsetReddeningQuery {
            sun_altitude: 0.175,
        },
        SunsetReddeningQuery { sun_altitude: 0.35 },
        SunsetReddeningQuery { sun_altitude: 1.0 },
    ];
    // A deterministic sweep from below the horizon to well above the cutoff.
    for k in 0..=120 {
        queries.push(SunsetReddeningQuery {
            sun_altitude: -0.2 + (k as f32) / 150.0,
        });
    }

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // Below the horizon and at the horizon are full reddening; above the cutoff
    // is none.
    assert!(
        (gpu[0] - 1.0).abs() < 1e-6,
        "below the horizon is full reddening: {}",
        gpu[0]
    );
    assert!(
        (gpu[1] - 1.0).abs() < 1e-6,
        "at the horizon is full reddening: {}",
        gpu[1]
    );
    assert!(
        gpu[3].abs() < 1e-6,
        "at the cutoff the reddening reaches zero: {}",
        gpu[3]
    );
    assert!(
        gpu[4].abs() < 1e-6,
        "well above the cutoff there is no reddening: {}",
        gpu[4]
    );
}

#[test]
fn gpu_sunset_reddening_is_monotone_non_increasing() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuSunsetReddening::new(&ctx);

    // A dense ascending altitude sweep; the reddening must never rise as the
    // sun climbs.
    let mut queries: Vec<SunsetReddeningQuery> = Vec::new();
    for k in 0..=200 {
        queries.push(SunsetReddeningQuery {
            sun_altitude: -0.1 + (k as f32) / 200.0 * 0.6,
        });
    }

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    let mut prev = f32::INFINITY;
    for &v in &gpu {
        assert!(
            v <= prev + 1e-6,
            "reddening must be monotone non-increasing in altitude: {prev} then {v}"
        );
        prev = v;
    }
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuSunsetReddening::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
