//! Real-device parity for the cluster-cull twin: [`GpuClusterCuller`] must
//! reproduce the CPU golden
//! [`cluster_cull`](prism_render_architecture::virtual_geometry::cluster_cull)
//! for every cluster, across all three verdicts.
//!
//! The test skips (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! A cull verdict is a discrete decision derived from sign comparisons, not a
//! continuous value. The scene places every cluster far inside, far outside a
//! specific frustum face, or unambiguously in front of / behind its occluder,
//! so the integer verdict is stable under any legal float reassociation and is
//! asserted index-for-index against `cluster_cull(..) as u32` with no
//! tolerance. The scene deliberately spans all three verdicts (`Visible`,
//! `FrustumCulled` from each of the six faces, `OcclusionCulled`) and both the
//! probe-present and probe-absent (`None`) paths, so no branch is left
//! unverified.
//!
//! Provenance: standard frustum projected-radius / Hi-Z occlusion culling; no
//! Unreal Engine source or derived code.

use std::collections::BTreeSet;

use prism_render_architecture::gpu_scene::SceneBounds;
use prism_render_architecture::virtual_geometry::{
    cluster_cull, Frustum, OcclusionProbe, Plane,
};
use prism_virtual_geometry_gpu::{GpuClusterCuller, GpuContext};

/// Axis-aligned box frustum: `|x| <= 10`, `|y| <= 10`, `0 <= z <= 100`, with
/// all six planes inward-facing and unit-normal, matching the golden's own test
/// frustum.
fn box_frustum() -> Frustum {
    Frustum::from_planes([
        Plane::new([1.0, 0.0, 0.0], 10.0),
        Plane::new([-1.0, 0.0, 0.0], 10.0),
        Plane::new([0.0, 1.0, 0.0], 10.0),
        Plane::new([0.0, -1.0, 0.0], 10.0),
        Plane::new([0.0, 0.0, 1.0], 0.0),
        Plane::new([0.0, 0.0, -1.0], 100.0),
    ])
}

fn bounds(center: [f32; 3], half: [f32; 3]) -> SceneBounds {
    SceneBounds {
        center,
        radius: 0.0,
        half_extents: half,
        _padding: 0.0,
    }
}

/// A scene spanning every verdict: two clearly visible clusters (one without a
/// probe, one with a non-occluding probe), one cluster pushed far outside each
/// of the six frustum faces, and one cluster occluded by a nearer occluder.
/// Every cluster clears its boundary by a wide margin so the verdict is
/// float-reassociation stable.
fn scene() -> Vec<(SceneBounds, Option<OcclusionProbe>)> {
    let unit = [1.0, 1.0, 1.0];
    let visible_probe = OcclusionProbe {
        closest_depth: 40.0,
        occluder_depth: 50.0,
    };
    let occluded_probe = OcclusionProbe {
        closest_depth: 60.0,
        occluder_depth: 50.0,
    };
    vec![
        // Visible, no occlusion phase.
        (bounds([0.0, 0.0, 50.0], unit), None),
        // Visible, probe present but nearer than the occluder.
        (bounds([0.0, 0.0, 50.0], unit), Some(visible_probe)),
        // FrustumCulled through each of the six faces.
        (bounds([100.0, 0.0, 50.0], unit), None), // +x
        (bounds([-100.0, 0.0, 50.0], unit), None), // -x
        (bounds([0.0, 100.0, 50.0], unit), None), // +y
        (bounds([0.0, -100.0, 50.0], unit), None), // -y
        (bounds([0.0, 0.0, -50.0], unit), None),  // near z (behind camera)
        (bounds([0.0, 0.0, 200.0], unit), None),  // far z
        // Inside the frustum but behind a nearer occluder.
        (bounds([0.0, 0.0, 60.0], unit), Some(occluded_probe)),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_cluster_cull_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cluster-cull parity: no wgpu adapter on this host");
        return;
    };
    let culler = GpuClusterCuller::new(&ctx);
    let frustum = box_frustum();
    let clusters = scene();

    let gpu = culler.cull(&ctx, &frustum, &clusters);
    assert_eq!(gpu.len(), clusters.len(), "one verdict per cluster");

    // Track which verdicts actually appear so a degenerate scene that only ever
    // emits one verdict cannot pass vacuously.
    let mut seen: BTreeSet<u32> = BTreeSet::new();
    for (i, (b, probe)) in clusters.iter().enumerate() {
        let expected = cluster_cull(&frustum, b, *probe) as u32;
        assert_eq!(
            gpu[i], expected,
            "verdict mismatch for cluster {b:?} probe {probe:?}: gpu {}, cpu {expected}",
            gpu[i]
        );
        seen.insert(gpu[i]);
    }

    // The scene must exercise all three verdicts: Visible, FrustumCulled and
    // OcclusionCulled.
    assert_eq!(
        seen,
        BTreeSet::from([0, 1, 2]),
        "scene must exercise every cull verdict, saw {seen:?}"
    );
}

#[test]
fn empty_scene_culls_to_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let culler = GpuClusterCuller::new(&ctx);
    let frustum = box_frustum();
    let out = culler.cull(&ctx, &frustum, &[]);
    assert!(out.is_empty(), "no clusters cull to no verdicts");
}

#[test]
fn each_frustum_face_culls_independently() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let culler = GpuClusterCuller::new(&ctx);
    let frustum = box_frustum();
    let unit = [1.0, 1.0, 1.0];
    // One cluster pushed well past each face, so every face must reject in
    // isolation. All six are expected to be FrustumCulled (1).
    let clusters = vec![
        (bounds([100.0, 0.0, 50.0], unit), None),
        (bounds([-100.0, 0.0, 50.0], unit), None),
        (bounds([0.0, 100.0, 50.0], unit), None),
        (bounds([0.0, -100.0, 50.0], unit), None),
        (bounds([0.0, 0.0, -50.0], unit), None),
        (bounds([0.0, 0.0, 200.0], unit), None),
    ];
    let gpu = culler.cull(&ctx, &frustum, &clusters);
    let expected: Vec<u32> = clusters
        .iter()
        .map(|(b, probe)| cluster_cull(&frustum, b, *probe) as u32)
        .collect();
    assert_eq!(gpu, expected, "each face must cull identically to the golden");
    assert_eq!(gpu, vec![1, 1, 1, 1, 1, 1], "every face rejects its cluster");
}
