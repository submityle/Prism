//! Real-device parity for the cloud-shadow ground-modulation twin:
//! [`GpuCloudShadowModulation`] must reproduce the `CPU` golden
//! [`cloud_shadow_modulation`](prism_render_architecture::volumetric::coupling::cloud_shadow_modulation)
//! across a spread of cloud transmittances and ground albedos.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The kernel contains no transcendental call — two saturating clamps and a
//! multiply — so `CPU` and `GPU` evaluate the same closed-form algebra. The
//! only slack is a legal multiply-add contraction of a few `ULP`, so values are
//! asserted to within `abs_diff < 1e-6` or `rel_diff < 1e-5` — tight enough to
//! fail a wrong port (a dropped saturate, a swapped factor). The scenes also
//! assert the documented `[0, 1]` range, that an opaque cloud kills the ground
//! contribution regardless of albedo, and that fully open sky returns the
//! saturated albedo, so a degenerate kernel could not pass.
//!
//! Provenance: standard cloud-shadow ground coupling; no Unreal Engine source
//! or derived code.

use prism_render_architecture::volumetric::coupling::cloud_shadow_modulation;
use prism_volumetric_gpu::{CloudShadowModulationQuery, GpuCloudShadowModulation, GpuContext};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays inside the unit range.
fn assert_parity(queries: &[CloudShadowModulationQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one modulation value per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = cloud_shadow_modulation(q.cloud_transmittance, q.ground_albedo);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "cloud shadow modulation mismatch for query {i} ({q:?}): gpu {got}, cpu {exp} \
             (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            (0.0..=1.0).contains(&got),
            "gpu cloud shadow modulation must stay in the unit range: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_cloud_shadow_modulation_matches_cpu_golden_across_queries() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cloud shadow modulation parity: no wgpu adapter on this host");
        return;
    };
    let gpu_mod = GpuCloudShadowModulation::new(&ctx);

    // A deterministic spread: opaque cloud over bright ground, open sky over
    // bright ground, mid transmittance / mid albedo, and out-of-range inputs
    // that must saturate into the unit range.
    let mut queries: Vec<CloudShadowModulationQuery> = vec![
        CloudShadowModulationQuery {
            cloud_transmittance: 0.0,
            ground_albedo: 1.0,
        },
        CloudShadowModulationQuery {
            cloud_transmittance: 1.0,
            ground_albedo: 0.8,
        },
        CloudShadowModulationQuery {
            cloud_transmittance: 0.5,
            ground_albedo: 0.5,
        },
        CloudShadowModulationQuery {
            cloud_transmittance: 0.25,
            ground_albedo: 0.4,
        },
        CloudShadowModulationQuery {
            cloud_transmittance: -0.5,
            ground_albedo: 0.9,
        },
        CloudShadowModulationQuery {
            cloud_transmittance: 2.0,
            ground_albedo: 3.0,
        },
        CloudShadowModulationQuery {
            cloud_transmittance: 0.7,
            ground_albedo: -1.0,
        },
    ];
    // A deterministic transmittance ramp at fixed albedo, plus an albedo ramp
    // at fixed transmittance.
    for k in 0..48 {
        queries.push(CloudShadowModulationQuery {
            cloud_transmittance: (k as f32) / 48.0,
            ground_albedo: 0.6,
        });
    }
    for k in 0..48 {
        queries.push(CloudShadowModulationQuery {
            cloud_transmittance: 0.6,
            ground_albedo: (k as f32) / 48.0,
        });
    }

    let gpu = gpu_mod.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // Opaque cloud kills the ground contribution; open sky returns the albedo;
    // negative inputs saturate to zero contribution.
    assert!(
        gpu[0].abs() < 1e-6,
        "opaque cloud kills the ground contribution: {}",
        gpu[0]
    );
    assert!(
        (gpu[1] - 0.8).abs() < 1e-6,
        "open sky returns the ground albedo: {}",
        gpu[1]
    );
    assert!(
        (gpu[2] - 0.25).abs() < 1e-6,
        "mid transmittance times mid albedo: {}",
        gpu[2]
    );
    assert!(
        gpu[4].abs() < 1e-6,
        "negative transmittance saturates to zero: {}",
        gpu[4]
    );
    assert!(
        gpu[6].abs() < 1e-6,
        "negative albedo saturates to zero: {}",
        gpu[6]
    );
}

#[test]
fn gpu_cloud_shadow_modulation_is_monotone_in_transmittance() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_mod = GpuCloudShadowModulation::new(&ctx);

    // At fixed albedo the lit-ground factor rises monotonically from zero
    // toward the albedo as the cloud transmittance grows.
    let albedo = 0.75f32;
    let queries: Vec<CloudShadowModulationQuery> = (0..=100)
        .map(|k| CloudShadowModulationQuery {
            cloud_transmittance: (k as f32) / 100.0,
            ground_albedo: albedo,
        })
        .collect();

    let gpu = gpu_mod.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    for w in gpu.windows(2) {
        assert!(
            w[1] >= w[0] - 1e-6,
            "modulation must be monotone non-decreasing in transmittance: {} then {}",
            w[0],
            w[1]
        );
    }
    assert!(
        (gpu[gpu.len() - 1] - albedo).abs() < 1e-6,
        "full transmittance returns the albedo"
    );
    assert!(
        gpu[0].abs() < 1e-6,
        "zero transmittance kills the contribution"
    );
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_mod = GpuCloudShadowModulation::new(&ctx);
    let out = gpu_mod.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
