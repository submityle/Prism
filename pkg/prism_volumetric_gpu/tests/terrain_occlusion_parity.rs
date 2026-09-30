//! Real-device parity for the terrain-occlusion coupling twin:
//! [`GpuTerrainOcclusion`] must reproduce the `CPU` golden
//! [`terrain_occlusion`](prism_render_architecture::volumetric::coupling::terrain_occlusion)
//! across a spread of cloud altitudes and terrain heights.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The kernel mirrors the reference's hand-rolled Hermite `smoothstep` (the
//! collapsed-edge `EPS` guard and the `t*t*(3-2t)` polynomial) rather than the
//! device-native `smoothstep`, and it contains no transcendental call, so `CPU`
//! and `GPU` evaluate the same closed-form algebra. The only slack is a legal
//! multiply-add contraction of a few `ULP`, so values are asserted to within
//! `abs_diff < 1e-6` or `rel_diff < 1e-5` — tight enough to fail a wrong port
//! (a dropped saturate, a swapped edge, a missing `1 -` complement). The scenes
//! also assert the documented `[0, 1]` range, the monotone non-increasing
//! falloff with altitude, full occlusion below the surface and full clearance
//! well above it, so a degenerate kernel could not pass.
//!
//! Provenance: standard Hermite smoothstep terrain intersection; no Unreal
//! Engine source or derived code.

use prism_render_architecture::volumetric::coupling::terrain_occlusion;
use prism_volumetric_gpu::{GpuContext, GpuTerrainOcclusion, TerrainOcclusionQuery};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays inside the unit range.
fn assert_parity(queries: &[TerrainOcclusionQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one occlusion value per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = terrain_occlusion(q.cloud_pos_y, q.terrain_height);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "terrain occlusion mismatch for query {i} ({q:?}): gpu {got}, cpu {exp} \
             (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            (0.0..=1.0).contains(&got),
            "gpu terrain occlusion must stay in the unit range: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_terrain_occlusion_matches_cpu_golden_across_queries() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping terrain occlusion parity: no wgpu adapter on this host");
        return;
    };
    let gpu_occ = GpuTerrainOcclusion::new(&ctx);

    // A deterministic spread: well below terrain (fully occluded), at the
    // surface, inside the transition band, at the top of the band (fully
    // clear), and well above it; at a couple of distinct terrain heights.
    let terrain = 120.0f32;
    let mut queries: Vec<TerrainOcclusionQuery> = vec![
        TerrainOcclusionQuery {
            cloud_pos_y: terrain - 50.0,
            terrain_height: terrain,
        },
        TerrainOcclusionQuery {
            cloud_pos_y: terrain,
            terrain_height: terrain,
        },
        TerrainOcclusionQuery {
            cloud_pos_y: terrain + 25.0,
            terrain_height: terrain,
        },
        TerrainOcclusionQuery {
            cloud_pos_y: terrain + 50.0,
            terrain_height: terrain,
        },
        TerrainOcclusionQuery {
            cloud_pos_y: terrain + 500.0,
            terrain_height: terrain,
        },
        TerrainOcclusionQuery {
            cloud_pos_y: 0.0,
            terrain_height: -30.0,
        },
        TerrainOcclusionQuery {
            cloud_pos_y: 3000.0,
            terrain_height: 2800.0,
        },
    ];
    // A deterministic altitude ramp through the transition band at fixed
    // terrain height.
    for k in 0..96 {
        queries.push(TerrainOcclusionQuery {
            cloud_pos_y: terrain - 20.0 + (k as f32),
            terrain_height: terrain,
        });
    }

    let gpu = gpu_occ.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // Below the surface is fully occluded; the top of the band and beyond is
    // fully clear; the band midpoint is strictly interior.
    assert!(
        (gpu[0] - 1.0).abs() < 1e-6,
        "well below terrain yields full occlusion: {}",
        gpu[0]
    );
    assert!(
        (gpu[1] - 1.0).abs() < 1e-6,
        "at the surface yields full occlusion: {}",
        gpu[1]
    );
    assert!(
        gpu[2] > 0.0 && gpu[2] < 1.0,
        "the band midpoint is strictly interior: {}",
        gpu[2]
    );
    assert!(
        gpu[3].abs() < 1e-6,
        "the top of the band is fully clear: {}",
        gpu[3]
    );
    assert!(
        gpu[4].abs() < 1e-6,
        "well above the band is fully clear: {}",
        gpu[4]
    );
}

#[test]
fn gpu_terrain_occlusion_is_monotone_non_increasing_in_altitude() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_occ = GpuTerrainOcclusion::new(&ctx);

    // At fixed terrain height the occlusion falls monotonically from one toward
    // zero as the cloud sample rises above the surface.
    let terrain = 80.0f32;
    let queries: Vec<TerrainOcclusionQuery> = (0..=100)
        .map(|k| TerrainOcclusionQuery {
            cloud_pos_y: terrain - 10.0 + (k as f32) * 0.8,
            terrain_height: terrain,
        })
        .collect();

    let gpu = gpu_occ.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    for w in gpu.windows(2) {
        assert!(
            w[1] <= w[0] + 1e-6,
            "terrain occlusion must be monotone non-increasing in altitude: {} then {}",
            w[0],
            w[1]
        );
    }
    assert!(
        gpu[gpu.len() - 1] < gpu[0] - 1e-3,
        "the highest sample must be strictly less occluded than the lowest"
    );
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_occ = GpuTerrainOcclusion::new(&ctx);
    let out = gpu_occ.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
