//! Real-device parity for the raster-path classify twin: [`GpuRasterClassifier`]
//! must reproduce the CPU golden
//! [`select_raster_path`](prism_render_architecture::virtual_geometry::select_raster_path)
//! for every cluster, across every backend-capability combination.
//!
//! The test skips (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The classification does no floating-point arithmetic (only a
//! `max(threshold, 0.0)` clamp and a single `<=` compare on identical
//! operands), so there is no tolerance: each emitted path index is asserted
//! bit-exact against `select_raster_path(..) as u32`. The scene deliberately
//! spans all four paths (empty -> `FallbackMesh`, tiny -> `ComputeSoftware`,
//! large under each hardware capability), the exact `<=` boundary at the
//! threshold, and a negative threshold that must clamp to zero, and it is run
//! under every capability combination so no path is left unverified.
//!
//! Provenance: standard cluster raster-path selection heuristic; no Unreal
//! Engine source or derived code.

extern crate alloc;

use alloc::collections::BTreeSet;

use prism_render_architecture::virtual_geometry::raster_path::DEFAULT_SOFTWARE_PIXEL_THRESHOLD;
use prism_render_architecture::virtual_geometry::{
    select_raster_path, ClusterRasterStats, RasterCapability,
};
use prism_virtual_geometry_gpu::{GpuContext, GpuRasterClassifier};

/// The reference path index a cluster resolves to, matching the twin's `u32`
/// output (`0` = `MeshShader`, `1` = `ComputeSoftware`, `2` = `IndirectHardware`,
/// `3` = `FallbackMesh`).
fn reference_index(
    stats: ClusterRasterStats,
    capability: RasterCapability,
    threshold: f32,
) -> u32 {
    select_raster_path(stats, capability, threshold) as u32
}

fn stat(max_triangle_pixels: f32, triangle_count: u32) -> ClusterRasterStats {
    ClusterRasterStats {
        max_triangle_pixels,
        triangle_count,
    }
}

/// A scene spanning every classification branch: an empty cluster, tiny
/// clusters at and below the threshold, and large clusters that route to the
/// best available hardware path.
fn scene() -> Vec<ClusterRasterStats> {
    vec![
        stat(1_000.0, 0),  // empty -> FallbackMesh regardless of size
        stat(4.0, 12),     // well below threshold -> ComputeSoftware
        stat(16.0, 8),     // exactly at threshold (<=) -> ComputeSoftware
        stat(64.0, 24),    // above threshold -> best hardware path
        stat(4096.0, 512), // far above threshold -> best hardware path
        stat(0.0, 3),      // zero-area but non-empty -> ComputeSoftware
    ]
}

/// The four backend-capability combinations that select distinct hardware
/// paths for above-threshold clusters.
fn capability_combos() -> [RasterCapability; 4] {
    [
        RasterCapability {
            mesh_shader: true,
            hardware_indirect: true,
        },
        RasterCapability {
            mesh_shader: true,
            hardware_indirect: false,
        },
        RasterCapability {
            mesh_shader: false,
            hardware_indirect: true,
        },
        RasterCapability {
            mesh_shader: false,
            hardware_indirect: false,
        },
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_raster_classify_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping raster-path parity: no wgpu adapter on this host");
        return;
    };
    let classifier = GpuRasterClassifier::new(&ctx);
    let stats = scene();
    let threshold = DEFAULT_SOFTWARE_PIXEL_THRESHOLD;

    // Track which path indices actually appear so a degenerate scene that only
    // ever emits one path cannot pass vacuously.
    let mut seen: BTreeSet<u32> = BTreeSet::new();

    for capability in capability_combos() {
        let gpu = classifier.classify(&ctx, &stats, capability, threshold);
        assert_eq!(gpu.len(), stats.len(), "one path per cluster");
        for (i, s) in stats.iter().enumerate() {
            let expected = reference_index(*s, capability, threshold);
            assert_eq!(
                gpu[i], expected,
                "path mismatch for cluster {s:?} under {capability:?}: gpu {}, cpu {expected}",
                gpu[i]
            );
            seen.insert(gpu[i]);
        }
    }

    // Across the capability combinations the scene must exercise all four
    // paths: FallbackMesh (empty + capability-poor), ComputeSoftware (tiny),
    // MeshShader and IndirectHardware (large under distinct capabilities).
    assert_eq!(
        seen,
        BTreeSet::from([0, 1, 2, 3]),
        "scene must exercise every raster path, saw {seen:?}"
    );
}

#[test]
fn empty_scene_classifies_to_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let classifier = GpuRasterClassifier::new(&ctx);
    let capability = RasterCapability {
        mesh_shader: true,
        hardware_indirect: true,
    };
    let out = classifier.classify(&ctx, &[], capability, DEFAULT_SOFTWARE_PIXEL_THRESHOLD);
    assert!(out.is_empty(), "no clusters classify to no paths");
}

#[test]
fn negative_threshold_clamps_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let classifier = GpuRasterClassifier::new(&ctx);
    let capability = RasterCapability {
        mesh_shader: true,
        hardware_indirect: true,
    };
    // With a negative threshold the reference clamps to 0.0, so only a
    // zero-area (<= 0) non-empty cluster takes the software path; a small
    // positive cluster now exceeds the clamped threshold and takes hardware.
    let stats = vec![
        stat(0.0, 4),   // <= max(-8, 0) == 0 -> ComputeSoftware (1)
        stat(0.5, 4),   // > 0 -> MeshShader (0) with full capability
        stat(100.0, 0), // empty -> FallbackMesh (3)
    ];
    let threshold = -8.0;
    let gpu = classifier.classify(&ctx, &stats, capability, threshold);
    let expected: Vec<u32> = stats
        .iter()
        .map(|s| reference_index(*s, capability, threshold))
        .collect();
    assert_eq!(gpu, expected, "negative threshold must clamp identically");
    assert_eq!(gpu, vec![1, 0, 3], "clamped classification is deterministic");
}
