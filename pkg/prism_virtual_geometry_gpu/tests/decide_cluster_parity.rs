//! Real-device parity for the fused per-cluster decision twin:
//! [`GpuClusterDecider`] must reproduce the CPU golden
//! [`ViewCullContext::decide`](prism_render_architecture::virtual_geometry::ViewCullContext::decide)
//! for every cluster, across the cull -> LOD -> raster-path composition.
//!
//! The tests skip (with a printed notice on the first) when the host has no
//! `wgpu` adapter, so the suite stays green everywhere while still exercising
//! the full dispatch-and-readback on any real device. The kernel is portable
//! core-WGSL, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! A decision is a cull verdict, an optional pair of discrete level indices and
//! an optional raster-path index (all asserted index-for-index with no
//! tolerance) plus a single streaming priority (a single multiply-then-divide,
//! asserted within 1 ULP). The scenes place every cluster well clear of the
//! frustum, occlusion and LOD-budget boundaries, so the discrete fields are
//! stable under any legal float reassociation. The scenes span the visible /
//! frustum-culled / occlusion-culled verdicts, the empty-chain `lod == None`
//! path, and every raster path (software for tiny triangles, then mesh-shader /
//! indirect-hardware / fallback-mesh across the three capability sets).
//!
//! Provenance: standard frustum/Hi-Z culling, screen-space-error LOD selection
//! and software/hardware raster classification; no Unreal Engine source or
//! derived code.

use prism_render_architecture::gpu_scene::SceneBounds;
use prism_render_architecture::virtual_geometry::{
    ClusterRasterStats, ClusterRequest, CullVerdict, Frustum, GeometryLodPolicy, GeometryPageKey,
    GeometryPageTable, GeometryRasterPath, LodLevel, LodProjection, OcclusionProbe, Plane,
    RasterCapability, ViewCullContext,
};
use prism_virtual_geometry_gpu::{ClusterDecisionInput, GpuClusterDecider, GpuContext};

/// Default software threshold, matching the golden's own tests.
const SOFTWARE_PIXEL_THRESHOLD: f32 = 16.0;

/// The golden's own test frustum: a symmetric box in x/y with a near/far slab.
fn frustum() -> Frustum {
    Frustum::from_planes([
        Plane::new([1.0, 0.0, 0.0], 10.0),
        Plane::new([-1.0, 0.0, 0.0], 10.0),
        Plane::new([0.0, 1.0, 0.0], 10.0),
        Plane::new([0.0, -1.0, 0.0], 10.0),
        Plane::new([0.0, 0.0, 1.0], 0.0),
        Plane::new([0.0, 0.0, -1.0], 100.0),
    ])
}

/// 1000px viewport, 90 deg vertical fov (`tan(45 deg) = 1`) => focal 500px.
fn projection() -> LodProjection {
    LodProjection::from_half_fov_tan(1000.0, 1.0)
}

/// A three-level chain: level `0` finest through level `2` coarsest.
fn chain() -> [LodLevel; 3] {
    [
        LodLevel {
            level: 0,
            geometric_error: 0.01,
        },
        LodLevel {
            level: 1,
            geometric_error: 0.08,
        },
        LodLevel {
            level: 2,
            geometric_error: 0.64,
        },
    ]
}

/// Builds a view context with the given raster capability, target 2px, no
/// hysteresis, no prefetch look-ahead.
fn context(capability: RasterCapability) -> ViewCullContext {
    ViewCullContext {
        frustum: frustum(),
        projection: projection(),
        lod_policy: GeometryLodPolicy {
            target_error_pixels: 2.0,
            ..Default::default()
        },
        raster_capability: capability,
        software_pixel_threshold: SOFTWARE_PIXEL_THRESHOLD,
        frame: 7,
    }
}

/// Full raster capability (mesh shaders + hardware indirect available).
fn full_capability() -> RasterCapability {
    RasterCapability {
        mesh_shader: true,
        hardware_indirect: true,
    }
}

fn sphere_bounds(center: [f32; 3], radius: f32) -> SceneBounds {
    SceneBounds {
        center,
        radius,
        half_extents: [radius, radius, radius],
        _padding: 0.0,
    }
}

/// Whether two non-negative floats are within 1 ULP (the priority is a single
/// multiply-then-divide of correctly-rounded operands, so at most the last bit
/// may differ under a legal reassociation).
fn within_1_ulp(a: f32, b: f32) -> bool {
    if a == b {
        return true;
    }
    if a.is_nan() || b.is_nan() {
        return false;
    }
    let ua = i64::from(a.to_bits());
    let ub = i64::from(b.to_bits());
    (ua - ub).abs() <= 1
}

/// Runs the twin for `inputs` under `ctx` and asserts every returned decision
/// matches the golden `ViewCullContext::decide` (discrete fields index-for-
/// index, priority within 1 ULP). The golden's page-table mutation is thrown
/// away — the twin only reproduces the returned decision.
fn assert_parity(gpu: &GpuContext, ctx: &ViewCullContext, inputs: &[ClusterDecisionInput]) {
    let levels = chain();
    let decider = GpuClusterDecider::new(gpu);
    let got = decider.decide(gpu, ctx, &levels, inputs);
    assert_eq!(got.len(), inputs.len(), "one decision per cluster");

    for (i, input) in inputs.iter().enumerate() {
        let mut table = GeometryPageTable::new();
        let request = ClusterRequest {
            page: GeometryPageKey::new(1, i as u32),
            bounds: &input.bounds,
            lods: &levels,
            raster_stats: input.raster_stats,
            view_distance: input.view_distance,
            closing_speed: input.closing_speed,
            occlusion: input.occlusion,
            previous_lod: input.previous_lod,
        };
        let expected = ctx.decide(&request, &mut table);
        let actual = got[i];
        assert_eq!(
            actual.verdict, expected.verdict,
            "verdict mismatch for cluster {i}"
        );
        assert_eq!(actual.lod, expected.lod, "lod mismatch for cluster {i}");
        assert_eq!(
            actual.raster_path, expected.raster_path,
            "raster path mismatch for cluster {i}"
        );
        assert!(
            within_1_ulp(actual.priority, expected.priority),
            "priority mismatch for cluster {i}: gpu {}, cpu {}",
            actual.priority,
            expected.priority
        );
    }
}

/// An empty chain must still produce verdict/raster/priority, but `lod == None`
/// exactly as the golden's `select_lod` returns on an empty chain.
fn assert_parity_empty_chain(gpu: &GpuContext, ctx: &ViewCullContext, input: &ClusterDecisionInput) {
    let decider = GpuClusterDecider::new(gpu);
    let got = decider.decide(gpu, ctx, &[], std::slice::from_ref(input));
    assert_eq!(got.len(), 1, "one decision for the single cluster");

    let mut table = GeometryPageTable::new();
    let empty: [LodLevel; 0] = [];
    let request = ClusterRequest {
        page: GeometryPageKey::new(9, 9),
        bounds: &input.bounds,
        lods: &empty,
        raster_stats: input.raster_stats,
        view_distance: input.view_distance,
        closing_speed: input.closing_speed,
        occlusion: input.occlusion,
        previous_lod: input.previous_lod,
    };
    let expected = ctx.decide(&request, &mut table);
    assert_eq!(got[0].verdict, expected.verdict, "verdict mismatch");
    assert_eq!(got[0].lod, None, "empty chain must yield no LOD");
    assert_eq!(got[0].lod, expected.lod, "lod mismatch vs golden");
    assert_eq!(
        got[0].raster_path, expected.raster_path,
        "raster path mismatch"
    );
    assert!(
        within_1_ulp(got[0].priority, expected.priority),
        "priority mismatch: gpu {}, cpu {}",
        got[0].priority,
        expected.priority
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_decide_matches_cpu_golden_visible_full_decision() {
    let Some(gpu) = GpuContext::try_headless() else {
        eprintln!("skipping decide-cluster parity: no wgpu adapter on this host");
        return;
    };
    let ctx = context(full_capability());
    // A visible cluster with large triangles: verdict Visible, a LOD pair, the
    // mesh-shader raster path and a positive priority.
    let inputs = vec![ClusterDecisionInput {
        bounds: sphere_bounds([0.0, 0.0, 40.0], 1.0),
        raster_stats: ClusterRasterStats {
            max_triangle_pixels: 4096.0,
            triangle_count: 64,
        },
        view_distance: 40.0,
        closing_speed: 0.0,
        occlusion: None,
        previous_lod: None,
    }];
    assert_parity(&gpu, &ctx, &inputs);
    // Sanity: the golden really is a full visible decision for this scene.
    let mut table = GeometryPageTable::new();
    let levels = chain();
    let request = ClusterRequest {
        page: GeometryPageKey::new(1, 0),
        bounds: &inputs[0].bounds,
        lods: &levels,
        raster_stats: inputs[0].raster_stats,
        view_distance: inputs[0].view_distance,
        closing_speed: inputs[0].closing_speed,
        occlusion: inputs[0].occlusion,
        previous_lod: inputs[0].previous_lod,
    };
    let expected = ctx.decide(&request, &mut table);
    assert_eq!(expected.verdict, CullVerdict::Visible);
    assert!(expected.lod.is_some());
    assert_eq!(expected.raster_path, Some(GeometryRasterPath::MeshShader));
    assert!(expected.priority > 0.0);
}

#[test]
fn gpu_decide_matches_cpu_golden_frustum_and_occlusion_culled() {
    let Some(gpu) = GpuContext::try_headless() else {
        return;
    };
    let ctx = context(full_capability());
    let inputs = vec![
        // Far to the +x side of the box: frustum-culled, no LOD/raster/priority.
        ClusterDecisionInput {
            bounds: sphere_bounds([100.0, 0.0, 40.0], 1.0),
            raster_stats: ClusterRasterStats {
                max_triangle_pixels: 4096.0,
                triangle_count: 64,
            },
            view_distance: 40.0,
            closing_speed: 0.0,
            occlusion: None,
            previous_lod: None,
        },
        // Inside the frustum but behind a nearer occluder: occlusion-culled.
        ClusterDecisionInput {
            bounds: sphere_bounds([0.0, 0.0, 60.0], 1.0),
            raster_stats: ClusterRasterStats {
                max_triangle_pixels: 4.0,
                triangle_count: 200,
            },
            view_distance: 60.0,
            closing_speed: 0.0,
            occlusion: Some(OcclusionProbe {
                closest_depth: 60.0,
                occluder_depth: 50.0,
            }),
            previous_lod: None,
        },
    ];
    assert_parity(&gpu, &ctx, &inputs);
}

#[test]
fn gpu_decide_matches_cpu_golden_tiny_triangles_take_software() {
    let Some(gpu) = GpuContext::try_headless() else {
        return;
    };
    let ctx = context(full_capability());
    // Sub-threshold triangles must take the compute software rasterizer even
    // though the full hardware path is available.
    let inputs = vec![ClusterDecisionInput {
        bounds: sphere_bounds([0.0, 0.0, 40.0], 1.0),
        raster_stats: ClusterRasterStats {
            max_triangle_pixels: 4.0,
            triangle_count: 256,
        },
        view_distance: 40.0,
        closing_speed: 0.0,
        occlusion: None,
        previous_lod: None,
    }];
    assert_parity(&gpu, &ctx, &inputs);
    let mut table = GeometryPageTable::new();
    let levels = chain();
    let request = ClusterRequest {
        page: GeometryPageKey::new(1, 0),
        bounds: &inputs[0].bounds,
        lods: &levels,
        raster_stats: inputs[0].raster_stats,
        view_distance: inputs[0].view_distance,
        closing_speed: inputs[0].closing_speed,
        occlusion: inputs[0].occlusion,
        previous_lod: inputs[0].previous_lod,
    };
    let expected = ctx.decide(&request, &mut table);
    assert_eq!(
        expected.raster_path,
        Some(GeometryRasterPath::ComputeSoftware),
        "scene must genuinely exercise the software path"
    );
}

#[test]
fn gpu_decide_matches_cpu_golden_raster_cascade_across_capabilities() {
    let Some(gpu) = GpuContext::try_headless() else {
        return;
    };
    // Large triangles resolve to mesh-shader / indirect-hardware / fallback-mesh
    // depending on the capability set; each must match the golden cascade.
    let big = ClusterDecisionInput {
        bounds: sphere_bounds([0.0, 0.0, 40.0], 1.0),
        raster_stats: ClusterRasterStats {
            max_triangle_pixels: 4096.0,
            triangle_count: 64,
        },
        view_distance: 40.0,
        closing_speed: 0.0,
        occlusion: None,
        previous_lod: None,
    };

    let mesh = context(full_capability());
    assert_parity(&gpu, &mesh, std::slice::from_ref(&big));

    let indirect = context(RasterCapability {
        mesh_shader: false,
        hardware_indirect: true,
    });
    assert_parity(&gpu, &indirect, std::slice::from_ref(&big));

    let fallback = context(RasterCapability {
        mesh_shader: false,
        hardware_indirect: false,
    });
    assert_parity(&gpu, &fallback, std::slice::from_ref(&big));

    // Verify the three contexts genuinely differ in raster path.
    let levels = chain();
    let mut expected_paths = Vec::new();
    for ctx in [&mesh, &indirect, &fallback] {
        let mut table = GeometryPageTable::new();
        let request = ClusterRequest {
            page: GeometryPageKey::new(1, 0),
            bounds: &big.bounds,
            lods: &levels,
            raster_stats: big.raster_stats,
            view_distance: big.view_distance,
            closing_speed: big.closing_speed,
            occlusion: big.occlusion,
            previous_lod: big.previous_lod,
        };
        expected_paths.push(ctx.decide(&request, &mut table).raster_path);
    }
    assert_eq!(
        expected_paths,
        vec![
            Some(GeometryRasterPath::MeshShader),
            Some(GeometryRasterPath::IndirectHardware),
            Some(GeometryRasterPath::FallbackMesh),
        ],
        "capability scene must span the full raster cascade"
    );
}

#[test]
fn gpu_decide_matches_cpu_golden_prefetch_and_hysteresis() {
    let Some(gpu) = GpuContext::try_headless() else {
        return;
    };
    // A visible cluster with a fast-approaching camera (prefetch pulled nearer)
    // and a previously-held level: exercises the display+prefetch LOD pair.
    let ctx = ViewCullContext {
        frustum: frustum(),
        projection: projection(),
        lod_policy: GeometryLodPolicy {
            target_error_pixels: 2.0,
            hysteresis_pixels: 0.0,
            prefetch_velocity_scale: 4.0,
        },
        raster_capability: full_capability(),
        software_pixel_threshold: SOFTWARE_PIXEL_THRESHOLD,
        frame: 11,
    };
    let inputs = vec![
        ClusterDecisionInput {
            bounds: sphere_bounds([0.0, 0.0, 45.0], 1.0),
            raster_stats: ClusterRasterStats {
                max_triangle_pixels: 4096.0,
                triangle_count: 64,
            },
            view_distance: 45.0,
            closing_speed: 5.0,
            occlusion: None,
            previous_lod: Some(2),
        },
        ClusterDecisionInput {
            bounds: sphere_bounds([0.0, 0.0, 12.0], 1.0),
            raster_stats: ClusterRasterStats {
                max_triangle_pixels: 4096.0,
                triangle_count: 64,
            },
            view_distance: 12.0,
            closing_speed: 0.0,
            occlusion: None,
            previous_lod: None,
        },
    ];
    assert_parity(&gpu, &ctx, &inputs);
}

#[test]
fn gpu_decide_matches_cpu_golden_empty_chain_visible_no_lod() {
    let Some(gpu) = GpuContext::try_headless() else {
        return;
    };
    let ctx = context(full_capability());
    // Visible cluster, but no LOD chain: verdict/raster/priority stand, lod None.
    let input = ClusterDecisionInput {
        bounds: sphere_bounds([0.0, 0.0, 40.0], 1.0),
        raster_stats: ClusterRasterStats {
            max_triangle_pixels: 4096.0,
            triangle_count: 64,
        },
        view_distance: 40.0,
        closing_speed: 0.0,
        occlusion: None,
        previous_lod: None,
    };
    assert_parity_empty_chain(&gpu, &ctx, &input);
}

#[test]
fn empty_clusters_decides_nothing() {
    let Some(gpu) = GpuContext::try_headless() else {
        return;
    };
    let ctx = context(full_capability());
    let decider = GpuClusterDecider::new(&gpu);
    let levels = chain();
    let out = decider.decide(&gpu, &ctx, &levels, &[]);
    assert!(out.is_empty(), "an empty cluster slice yields no decisions");
}
