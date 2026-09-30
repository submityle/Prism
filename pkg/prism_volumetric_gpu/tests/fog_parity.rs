//! Real-device parity for the Beer-Lambert fog-transmittance twin:
//! [`GpuFogTransmittance`] must reproduce the `CPU` golden
//! [`fog_transmittance`](prism_render_architecture::volumetric::fog::fog_transmittance)
//! across a spread of distances and densities.
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
//! fail a wrong port (a dropped `max(0)` floor, a swapped optical-depth
//! product, a missing saturate). The scenes also assert the documented
//! monotone decay, the `[0, 1]` range, `distance = 0 -> 1`, `density = 0 -> 1`
//! and negative-input flooring, so a degenerate kernel could not pass.
//!
//! Provenance: standard Beer-Lambert extinction; no Unreal Engine source or
//! derived code.

use prism_render_architecture::volumetric::fog::fog_transmittance;
use prism_volumetric_gpu::{FogQuery, GpuContext, GpuFogTransmittance};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays inside the unit range.
fn assert_parity(queries: &[FogQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one fog value per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = fog_transmittance(q.distance, q.density);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "fog mismatch for query {i} ({q:?}): gpu {got}, cpu {exp} \
             (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            (0.0..=1.0).contains(&got),
            "gpu fog transmittance must stay in the unit range: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_fog_matches_cpu_golden_across_queries() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping fog parity: no wgpu adapter on this host");
        return;
    };
    let gpu_fog = GpuFogTransmittance::new(&ctx);

    // A deterministic spread: clear air (zero distance and zero density), a
    // faint slab, a moderate slab, a dense slab near full extinction, and
    // negative inputs that must floor to full transmittance.
    let mut queries: Vec<FogQuery> = vec![
        FogQuery {
            distance: 0.0,
            density: 1.0,
        },
        FogQuery {
            distance: 100.0,
            density: 0.0,
        },
        FogQuery {
            distance: 10.0,
            density: 0.02,
        },
        FogQuery {
            distance: 50.0,
            density: 0.05,
        },
        FogQuery {
            distance: 200.0,
            density: 0.5,
        },
        FogQuery {
            distance: -10.0,
            density: 0.1,
        },
        FogQuery {
            distance: 10.0,
            density: -0.1,
        },
    ];
    // A deterministic optical-depth ramp at fixed density.
    for k in 0..96 {
        queries.push(FogQuery {
            distance: (k as f32) * 2.0,
            density: 0.03,
        });
    }

    let gpu = gpu_fog.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // distance = 0 -> 1; density = 0 -> 1; a dense slab must approach zero;
    // the negative inputs must floor to full transmittance.
    assert!(
        (gpu[0] - 1.0).abs() < 1e-6,
        "zero distance yields full transmittance"
    );
    assert!(
        (gpu[1] - 1.0).abs() < 1e-6,
        "zero density yields full transmittance"
    );
    assert!(
        gpu[4] < 0.1,
        "a dense slab must approach zero transmittance: {}",
        gpu[4]
    );
    assert!(
        (gpu[5] - 1.0).abs() < 1e-6,
        "negative distance floors to full transmittance: {}",
        gpu[5]
    );
    assert!(
        (gpu[6] - 1.0).abs() < 1e-6,
        "negative density floors to full transmittance: {}",
        gpu[6]
    );
}

#[test]
fn gpu_fog_is_monotone_non_increasing_in_distance() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_fog = GpuFogTransmittance::new(&ctx);

    // At fixed density the transmittance falls monotonically from one toward
    // zero as the ray travels farther through the slab.
    let density = 0.08f32;
    let queries: Vec<FogQuery> = (0..=80)
        .map(|k| FogQuery {
            distance: (k as f32) * 3.0,
            density,
        })
        .collect();

    let gpu = gpu_fog.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    for w in gpu.windows(2) {
        assert!(
            w[1] <= w[0] + 1e-6,
            "fog transmittance must be monotone non-increasing in distance: {} then {}",
            w[0],
            w[1]
        );
    }
    assert!(
        gpu[gpu.len() - 1] < gpu[0] - 1e-3,
        "the farthest query must transmit strictly less than the clear one"
    );
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_fog = GpuFogTransmittance::new(&ctx);
    let out = gpu_fog.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
