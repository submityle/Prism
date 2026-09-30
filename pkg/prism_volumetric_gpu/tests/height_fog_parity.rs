//! Real-device parity for the exponential height-fog density twin:
//! [`GpuHeightFog`] must reproduce the `CPU` golden
//! [`height_fog_density`](prism_render_architecture::volumetric::fog::height_fog_density)
//! across altitudes, layer parameters and the ceiling cull.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The kernel mirrors the reference's hand-rolled `exp_approx` (base-two range
//! reduction, a fractional polynomial times an integer power assembled from the
//! float exponent field) rather than the device-native `exp`, so `CPU` and
//! `GPU` evaluate the same closed-form algebra. The only slack is a legal
//! multiply-add contraction of a few `ULP` in the polynomial, so values are
//! asserted to within `abs_diff < 1e-6` or `rel_diff < 1e-5` — tight enough to
//! fail a wrong port (a dropped floor, a missing ceiling cull, a swapped
//! falloff). The scenes also assert the density is non-negative, never exceeds
//! the sea-level density, decays monotonically with altitude, is culled to
//! exactly zero at the ceiling and treats negative altitudes as sea level, so a
//! degenerate kernel could not pass.
//!
//! Provenance: standard exponential height-fog extinction; no Unreal Engine
//! source or derived code.

use prism_render_architecture::volumetric::fog::{height_fog_density, HeightFogParams};
use prism_volumetric_gpu::{GpuContext, GpuHeightFog, HeightFogQuery};

/// Builds the `CPU` layer parameters that mirror a query's layer fields.
fn params_of(q: &HeightFogQuery) -> HeightFogParams {
    HeightFogParams {
        density_at_sea_level: q.density_at_sea_level,
        falloff: q.falloff,
        max_height: q.max_height,
    }
}

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays non-negative and bounded by the sea-level density.
fn assert_parity(queries: &[HeightFogQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one density per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = height_fog_density(q.altitude, params_of(q));
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "height-fog mismatch for query {i} ({q:?}): gpu {got}, cpu {exp} \
             (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            got >= -1e-6,
            "gpu height-fog density must be non-negative: {got}"
        );
        assert!(
            got <= q.density_at_sea_level.max(0.0) + 1e-6,
            "gpu height-fog density must not exceed the sea-level density: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_height_fog_matches_cpu_golden_across_queries() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping height-fog parity: no wgpu adapter on this host");
        return;
    };
    let gpu_height_fog = GpuHeightFog::new(&ctx);

    // A deterministic spread: sea level, mid layer, right below the ceiling,
    // at the ceiling (culled to zero), above the ceiling (culled), a negative
    // altitude (treated as sea level), and floored negative parameters.
    let mut queries: Vec<HeightFogQuery> = vec![
        HeightFogQuery {
            altitude: 0.0,
            density_at_sea_level: 0.8,
            falloff: 0.01,
            max_height: 1000.0,
        },
        HeightFogQuery {
            altitude: 300.0,
            density_at_sea_level: 0.8,
            falloff: 0.01,
            max_height: 1000.0,
        },
        HeightFogQuery {
            altitude: 999.0,
            density_at_sea_level: 0.8,
            falloff: 0.01,
            max_height: 1000.0,
        },
        HeightFogQuery {
            altitude: 1000.0,
            density_at_sea_level: 0.8,
            falloff: 0.01,
            max_height: 1000.0,
        },
        HeightFogQuery {
            altitude: 5000.0,
            density_at_sea_level: 0.8,
            falloff: 0.01,
            max_height: 1000.0,
        },
        HeightFogQuery {
            altitude: -200.0,
            density_at_sea_level: 0.8,
            falloff: 0.01,
            max_height: 1000.0,
        },
        HeightFogQuery {
            altitude: 100.0,
            density_at_sea_level: -0.5,
            falloff: -0.02,
            max_height: 1000.0,
        },
    ];
    // A deterministic altitude ramp within the ceiling at a fixed layer.
    for k in 0..96 {
        queries.push(HeightFogQuery {
            altitude: (k as f32) * 10.0,
            density_at_sea_level: 1.0,
            falloff: 0.008,
            max_height: 1200.0,
        });
    }

    let gpu = gpu_height_fog.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // Sea level equals the sea-level density; the ceiling and above cull to
    // zero; a negative altitude equals the sea-level density; floored negative
    // parameters give a finite non-negative density.
    assert!(
        (gpu[0] - 0.8).abs() < 1e-6,
        "sea level equals the sea-level density: {}",
        gpu[0]
    );
    assert!(gpu[3].abs() < 1e-6, "the ceiling culls to zero: {}", gpu[3]);
    assert!(
        gpu[4].abs() < 1e-6,
        "above the ceiling culls to zero: {}",
        gpu[4]
    );
    assert!(
        (gpu[5] - 0.8).abs() < 1e-6,
        "a negative altitude equals the sea-level density: {}",
        gpu[5]
    );
    assert!(
        gpu[6].abs() < 1e-6,
        "a floored-negative sea-level density yields zero: {}",
        gpu[6]
    );
}

#[test]
fn gpu_height_fog_is_monotone_non_increasing_in_altitude() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_height_fog = GpuHeightFog::new(&ctx);

    // Below the ceiling the density falls monotonically from the sea-level
    // value toward zero as altitude grows.
    let queries: Vec<HeightFogQuery> = (0..=100)
        .map(|k| HeightFogQuery {
            altitude: (k as f32) * 15.0,
            density_at_sea_level: 1.0,
            falloff: 0.01,
            max_height: 5000.0,
        })
        .collect();

    let gpu = gpu_height_fog.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    for w in gpu.windows(2) {
        assert!(
            w[1] <= w[0] + 1e-6,
            "height-fog density must be monotone non-increasing in altitude: {} then {}",
            w[0],
            w[1]
        );
    }
    assert!(
        gpu[gpu.len() - 1] < gpu[0] - 1e-3,
        "the highest query must be strictly thinner than sea level"
    );
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_height_fog = GpuHeightFog::new(&ctx);
    let out = gpu_height_fog.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
