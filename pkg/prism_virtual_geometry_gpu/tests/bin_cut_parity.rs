//! Real-device parity for the raster-bin partition twin: [`GpuCutBinner`] must
//! reproduce the CPU golden
//! [`bin_cut`](prism_render_architecture::virtual_geometry::bin_cut) for a
//! selected cluster cut, across every backend-capability combination.
//!
//! The test skips (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! three-pass dispatch-and-readback on any real device. The kernel is portable
//! core-WGSL, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The partition performs no floating-point arithmetic beyond a
//! `max(threshold, 0.0)` clamp and a single `<=` compare on identical operands,
//! and the emitted output is a discrete permutation of the input cut indices,
//! so there is no tolerance: each of the four buckets is asserted element-for-
//! element equal to `bin_cut(..)`. The scene deliberately spans all four
//! buckets (so a degenerate partition cannot pass vacuously), interleaves the
//! buckets in the cut so per-bucket order preservation is actually exercised,
//! and includes an out-of-range node that must be skipped rather than routed.
//!
//! Provenance: standard stable multi-bucket partition and cluster raster-path
//! selection heuristic; no Unreal Engine source or derived code.

extern crate alloc;

use alloc::collections::BTreeSet;

use prism_render_architecture::virtual_geometry::raster_path::DEFAULT_SOFTWARE_PIXEL_THRESHOLD;
use prism_render_architecture::virtual_geometry::{
    bin_cut, ClusterRasterStats, CutCluster, GeometryPageKey, RasterBins, RasterCapability,
};
use prism_virtual_geometry_gpu::{GpuContext, GpuCutBinner};

/// A cut cluster addressing hierarchy node `node`, with a distinct page key so
/// mismatched reattachment (wrong permutation index) is caught, not masked.
fn cluster(node: u32) -> CutCluster {
    CutCluster {
        node,
        page: GeometryPageKey::new(7, node.wrapping_mul(3).wrapping_add(1)),
    }
}

fn stat(max_triangle_pixels: f32, triangle_count: u32) -> ClusterRasterStats {
    ClusterRasterStats {
        max_triangle_pixels,
        triangle_count,
    }
}

/// Per-node statistics chosen so each node routes to a distinct, stable bucket
/// under a full-capability backend, deliberately far from the `<=` threshold
/// boundary so the assignment is unambiguous:
///   node 0 -> empty (fallback), 1 -> tiny (software),
///   2 -> large (mesh/indirect per capability), 3 -> large, 4 -> tiny.
fn scene_stats() -> Vec<ClusterRasterStats> {
    vec![
        stat(1_000.0, 0), // node 0: empty -> FallbackMesh regardless of size
        stat(1.0, 8),     // node 1: well below threshold -> ComputeSoftware
        stat(4_096.0, 64), // node 2: far above threshold -> best hardware path
        stat(2_048.0, 32), // node 3: far above threshold -> best hardware path
        stat(2.0, 4),     // node 4: below threshold -> ComputeSoftware
    ]
}

/// A cut that interleaves the buckets (so per-bucket order preservation is
/// tested), repeats nodes (so ranks advance), and ends with an out-of-range
/// node index (`99`) that must be skipped rather than routed or crash.
fn scene_cut() -> Vec<CutCluster> {
    vec![
        cluster(2),  // hardware
        cluster(1),  // software
        cluster(0),  // fallback (empty)
        cluster(3),  // hardware
        cluster(4),  // software
        cluster(2),  // hardware (repeat -> advances hardware rank)
        cluster(99), // out of range -> skipped
        cluster(0),  // fallback (repeat)
    ]
}

/// The four backend-capability combinations that route above-threshold
/// clusters to distinct hardware buckets.
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

/// Records which of the four buckets are non-empty in `bins`, so the scene can
/// be proven to exercise every bucket rather than passing vacuously.
fn nonempty_buckets(bins: &RasterBins) -> BTreeSet<u32> {
    let mut seen = BTreeSet::new();
    if !bins.mesh_shader.is_empty() {
        seen.insert(0);
    }
    if !bins.compute_software.is_empty() {
        seen.insert(1);
    }
    if !bins.indirect_hardware.is_empty() {
        seen.insert(2);
    }
    if !bins.fallback_mesh.is_empty() {
        seen.insert(3);
    }
    seen
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_bin_cut_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping bin-cut parity: no wgpu adapter on this host");
        return;
    };
    let binner = GpuCutBinner::new(&ctx);
    let stats = scene_stats();
    let cut = scene_cut();
    let threshold = DEFAULT_SOFTWARE_PIXEL_THRESHOLD;

    // Union of buckets seen across capability combinations so a degenerate
    // scene that never fills a bucket cannot pass vacuously.
    let mut seen: BTreeSet<u32> = BTreeSet::new();

    for capability in capability_combos() {
        let expected = bin_cut(&cut, &stats, capability, threshold);
        let gpu = binner.bin(&ctx, &cut, &stats, capability, threshold);

        assert_eq!(
            gpu.mesh_shader, expected.mesh_shader,
            "mesh_shader bucket mismatch under {capability:?}"
        );
        assert_eq!(
            gpu.compute_software, expected.compute_software,
            "compute_software bucket mismatch under {capability:?}"
        );
        assert_eq!(
            gpu.indirect_hardware, expected.indirect_hardware,
            "indirect_hardware bucket mismatch under {capability:?}"
        );
        assert_eq!(
            gpu.fallback_mesh, expected.fallback_mesh,
            "fallback_mesh bucket mismatch under {capability:?}"
        );
        assert_eq!(
            gpu.total(),
            expected.total(),
            "total cluster count mismatch under {capability:?} (out-of-range skip broken?)"
        );

        seen.extend(nonempty_buckets(&expected));
    }

    // Across the capability combinations the scene must fill all four buckets:
    // FallbackMesh (empty cluster + capability-poor), ComputeSoftware (tiny),
    // MeshShader and IndirectHardware (large under distinct capabilities).
    assert_eq!(
        seen,
        BTreeSet::from([0, 1, 2, 3]),
        "scene must exercise every raster bucket, saw {seen:?}"
    );
}

#[test]
fn empty_cut_bins_to_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let binner = GpuCutBinner::new(&ctx);
    let capability = RasterCapability {
        mesh_shader: true,
        hardware_indirect: true,
    };
    let out = binner.bin(&ctx, &[], &scene_stats(), capability, DEFAULT_SOFTWARE_PIXEL_THRESHOLD);
    assert!(out.is_empty());
    assert_eq!(out.total(), 0);
}

#[test]
fn all_nodes_out_of_range_bins_to_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let binner = GpuCutBinner::new(&ctx);
    let capability = RasterCapability {
        mesh_shader: true,
        hardware_indirect: true,
    };
    // Every node index is out of range for a single-entry stats slice.
    let cut = [cluster(10), cluster(20), cluster(30)];
    let stats = [stat(4_096.0, 64)];
    let gpu = binner.bin(&ctx, &cut, &stats, capability, DEFAULT_SOFTWARE_PIXEL_THRESHOLD);
    let expected = bin_cut(&cut, &stats, capability, DEFAULT_SOFTWARE_PIXEL_THRESHOLD);
    assert_eq!(gpu, expected);
    assert!(gpu.is_empty());
}
