//! Real-device parity for the sunset in-scatter tint twin:
//! [`GpuSunsetInscatterTint`] must reproduce the `CPU` golden
//! [`sunset_inscatter_tint`](prism_render_architecture::volumetric::atmosphere::sunset_inscatter_tint)
//! across the full range of sun altitudes, including altitudes at and below the
//! horizon and above the reddening cutoff.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The reddening `smoothstep` and per-band Gaussian response use the same closed
//! forms and the same hand-rolled `exp_approx` the reference uses, so `CPU` and
//! `GPU` evaluate identical algebra. Values are asserted per channel to within
//! `abs_diff < 1e-6` or `rel_diff < 1e-5` — tight enough to fail a wrong port (a
//! dropped normalisation, a wrong twilight weight). The scenes also assert every
//! channel stays in `0..=1`, the tint is neutral one above the cutoff, warms
//! with a `red >= green >= blue` ordering, and the blue/green channels rise
//! monotonically as the sun climbs, so a degenerate kernel could not pass.
//!
//! Provenance: standard horizon-reddening falloff plus CIE-flavoured
//! spectral-to-RGB collapse; no Unreal Engine source or derived code.

use prism_render_architecture::volumetric::atmosphere::sunset_inscatter_tint;
use prism_volumetric_gpu::{
    GpuContext, GpuSunsetInscatterTint, InscatterTint, SunsetInscatterTintQuery,
};

/// Asserts every `gpu` tint matches the `CPU` golden per channel to within the
/// documented tolerance, stays in `0..=1`, and keeps a `red >= green >= blue`
/// warming ordering.
fn assert_parity(queries: &[SunsetInscatterTintQuery], gpu: &[InscatterTint]) {
    assert_eq!(gpu.len(), queries.len(), "one tint per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = sunset_inscatter_tint(q.sun_altitude);
        let got = gpu[i];
        for (channel, (g, e)) in [(got.r, exp.x), (got.g, exp.y), (got.b, exp.z)]
            .into_iter()
            .enumerate()
        {
            let abs_diff = (g - e).abs();
            let rel_diff = abs_diff / e.abs().max(1e-6);
            assert!(
                abs_diff < 1e-6 || rel_diff < 1e-5,
                "sunset in-scatter tint mismatch for query {i} ({q:?}) channel {channel}: \
                 gpu {g}, cpu {e} (abs {abs_diff}, rel {rel_diff})"
            );
            assert!(
                (0.0..=1.0 + 1e-6).contains(&g),
                "gpu tint channel {channel} of query {i} must stay in 0..=1: {g}"
            );
        }
        assert!(
            got.r + 1e-6 >= got.g && got.g + 1e-6 >= got.b,
            "the warm tint must keep red >= green >= blue for query {i}: {got:?}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_sunset_inscatter_tint_matches_cpu_golden_across_altitudes() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sunset in-scatter tint parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuSunsetInscatterTint::new(&ctx);

    // A deterministic spread: below the horizon (full reddening), at the
    // horizon, mid-ramp, at the cutoff, and well above it (neutral).
    let mut queries: Vec<SunsetInscatterTintQuery> = vec![
        SunsetInscatterTintQuery { sun_altitude: -0.5 },
        SunsetInscatterTintQuery { sun_altitude: 0.0 },
        SunsetInscatterTintQuery {
            sun_altitude: 0.175,
        },
        SunsetInscatterTintQuery { sun_altitude: 0.35 },
        SunsetInscatterTintQuery { sun_altitude: 1.0 },
    ];
    // A deterministic sweep from below the horizon to well above the cutoff.
    for k in 0..=120 {
        queries.push(SunsetInscatterTintQuery {
            sun_altitude: -0.2 + (k as f32) / 150.0,
        });
    }

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // Above the cutoff the tint is neutral (no reddening).
    let above = gpu[4];
    assert!(
        (above.r - 1.0).abs() < 1e-6
            && (above.g - 1.0).abs() < 1e-6
            && (above.b - 1.0).abs() < 1e-6,
        "well above the cutoff the tint is neutral one: {above:?}"
    );
    // Below the horizon is full reddening: the blue channel is well under one.
    assert!(
        gpu[0].b < 0.999,
        "below the horizon the blue channel is attenuated: {:?}",
        gpu[0]
    );
}

#[test]
fn gpu_sunset_inscatter_tint_warms_monotonically() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuSunsetInscatterTint::new(&ctx);

    // A dense ascending altitude sweep; as the sun climbs, reddening falls, so
    // the blue and green channels must rise monotonically toward one.
    let mut queries: Vec<SunsetInscatterTintQuery> = Vec::new();
    for k in 0..=200 {
        queries.push(SunsetInscatterTintQuery {
            sun_altitude: -0.1 + (k as f32) / 200.0 * 0.6,
        });
    }

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    let mut prev_b = f32::NEG_INFINITY;
    let mut prev_g = f32::NEG_INFINITY;
    for t in &gpu {
        assert!(
            t.b + 1e-6 >= prev_b,
            "blue must rise monotonically as the sun climbs: {prev_b} then {}",
            t.b
        );
        assert!(
            t.g + 1e-6 >= prev_g,
            "green must rise monotonically as the sun climbs: {prev_g} then {}",
            t.g
        );
        prev_b = t.b;
        prev_g = t.g;
    }
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuSunsetInscatterTint::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
