//! Real-device parity for the per-cluster strand cull twin:
//! [`GpuHairClusterCull`] must reproduce the `CPU` golden
//! [`cluster_cull_verdict`](prism_render_architecture::hair::cluster::cluster_cull_verdict)
//! for a batch of strand clusters, covering the frustum projected-radius test,
//! the invalid (empty) bounds short-circuit, the tangent back-facing test, the
//! degenerate-tangent "never culled" guard, the Hi-Z occlusion phase, the
//! "no probe skips occlusion" path, and a mixed multi-cluster batch where every
//! cull reason occurs at once.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The verdict is an integer (a visible flag plus a reason code), so parity is
//! asserted **bit-exact** with `assert_eq!`. The only floating-point work is the
//! plane and facing dot products, which a `GPU` may legally fuse into a
//! multiply-add the scalar reference leaves separate; every case here keeps its
//! decision well clear of the threshold (boxes sit tens of units inside or
//! outside each plane, facing dots are ±1 against a `0.5` bias) so contraction
//! cannot flip a verdict. Each test additionally asserts the specific expected
//! reason so a kernel that collapsed to a single verdict could not pass.
//!
//! Provenance: standard GPU-driven cluster culling; no Unreal Engine source or
//! derived code.

use prism_hair_gpu::cluster_cull::GpuHairClusterCull;
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::cluster::{
    cluster_cull_verdict, Aabb, ClusterCullVerdict, ClusterView, CullReason, Frustum,
    OcclusionProbe, Plane, StrandCluster,
};

/// The cosine back-facing bias shared by every parity case.
const BIAS: f32 = 0.5;

/// A box view frustum: the region `x in [-10, 10]`, `y in [-10, 10]`,
/// `z in [1, 100]`, expressed as six inward-facing (unit-normal) planes. A
/// cluster centred near `(0, 0, 10)` sits comfortably inside every plane.
fn box_frustum() -> Frustum {
    Frustum::from_planes([
        Plane::new([-1.0, 0.0, 0.0], 10.0),  // x <= 10
        Plane::new([1.0, 0.0, 0.0], 10.0),   // x >= -10
        Plane::new([0.0, -1.0, 0.0], 10.0),  // y <= 10
        Plane::new([0.0, 1.0, 0.0], 10.0),   // y >= -10
        Plane::new([0.0, 0.0, 1.0], -1.0),   // z >= 1
        Plane::new([0.0, 0.0, -1.0], 100.0), // z <= 100
    ])
}

/// A cluster with an explicit axis-aligned box and mean tangent.
fn cluster(id: u32, min: [f32; 3], max: [f32; 3], mean_tangent: [f32; 3]) -> StrandCluster {
    StrandCluster {
        id,
        bounds: Aabb { min, max },
        mean_tangent,
        strand_indices: Vec::new(),
    }
}

/// Runs `cluster_cull_verdict` per cluster and asserts the `GPU` batch matches
/// it bit-exact, returning the `GPU` verdicts.
fn assert_parity(
    ctx: &GpuContext,
    culler: &GpuHairClusterCull,
    clusters: &[StrandCluster],
    frustum: &Frustum,
    view: ClusterView,
    occlusion: &[Option<OcclusionProbe>],
) -> Vec<ClusterCullVerdict> {
    let cpu: Vec<ClusterCullVerdict> = clusters
        .iter()
        .enumerate()
        .map(|(i, c)| {
            cluster_cull_verdict(c, frustum, view, occlusion.get(i).copied().flatten(), BIAS)
        })
        .collect();
    let gpu = culler.eval(ctx, clusters, frustum, view, occlusion, BIAS);
    assert_eq!(gpu.len(), clusters.len(), "one verdict per cluster");
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert_eq!(g.visible, c.visible, "cluster {i} visible flag");
        assert_eq!(g.reason, c.reason, "cluster {i} cull reason");
    }
    gpu
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_visible_cluster_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cluster_cull parity: no wgpu adapter on this host");
        return;
    };
    let culler = GpuHairClusterCull::new(&ctx);
    let frustum = box_frustum();
    let view = ClusterView::new([0.0, 0.0, 0.0]);
    // Inside the frustum, tangent perpendicular to the view ray (facing 0), no
    // occlusion probe -> visible.
    let clusters = [cluster(
        0,
        [-1.0, -1.0, 9.0],
        [1.0, 1.0, 11.0],
        [1.0, 0.0, 0.0],
    )];
    let occlusion = [None];
    let v = assert_parity(&ctx, &culler, &clusters, &frustum, view, &occlusion);
    assert!(v[0].visible, "expected a visible verdict");
    assert_eq!(v[0].reason, CullReason::Visible);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_frustum_culled_cluster_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cluster_cull parity: no wgpu adapter on this host");
        return;
    };
    let culler = GpuHairClusterCull::new(&ctx);
    let frustum = box_frustum();
    let view = ClusterView::new([0.0, 0.0, 0.0]);
    // Centre at x = 50, far outside the x <= 10 plane -> frustum cull.
    let clusters = [cluster(
        0,
        [49.0, -1.0, 9.0],
        [51.0, 1.0, 11.0],
        [1.0, 0.0, 0.0],
    )];
    let occlusion = [None];
    let v = assert_parity(&ctx, &culler, &clusters, &frustum, view, &occlusion);
    assert!(!v[0].visible, "expected a culled verdict");
    assert_eq!(v[0].reason, CullReason::Frustum);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_empty_bounds_culls_as_frustum_like_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cluster_cull parity: no wgpu adapter on this host");
        return;
    };
    let culler = GpuHairClusterCull::new(&ctx);
    let frustum = box_frustum();
    let view = ClusterView::new([0.0, 0.0, 0.0]);
    // An un-grown (empty) box is invalid and treated as outside the frustum.
    let clusters = [StrandCluster {
        id: 0,
        bounds: Aabb::empty(),
        mean_tangent: [1.0, 0.0, 0.0],
        strand_indices: Vec::new(),
    }];
    let occlusion = [None];
    let v = assert_parity(&ctx, &culler, &clusters, &frustum, view, &occlusion);
    assert!(!v[0].visible, "expected a culled verdict");
    assert_eq!(v[0].reason, CullReason::Frustum);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_backface_culled_cluster_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cluster_cull parity: no wgpu adapter on this host");
        return;
    };
    let culler = GpuHairClusterCull::new(&ctx);
    let frustum = box_frustum();
    let view = ClusterView::new([0.0, 0.0, 0.0]);
    // Inside the frustum, but the tangent points along +Z while the to-eye
    // direction is -Z (facing = -1 <= -0.5) -> back-face cull.
    let clusters = [cluster(
        0,
        [-1.0, -1.0, 9.0],
        [1.0, 1.0, 11.0],
        [0.0, 0.0, 1.0],
    )];
    let occlusion = [None];
    let v = assert_parity(&ctx, &culler, &clusters, &frustum, view, &occlusion);
    assert!(!v[0].visible, "expected a culled verdict");
    assert_eq!(v[0].reason, CullReason::Backface);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_degenerate_tangent_is_not_culled_like_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cluster_cull parity: no wgpu adapter on this host");
        return;
    };
    let culler = GpuHairClusterCull::new(&ctx);
    let frustum = box_frustum();
    let view = ClusterView::new([0.0, 0.0, 0.0]);
    // A zero mean tangent normalises to zero, giving a facing dot of 0, which
    // never trips the positive bias -> not back-face culled.
    let clusters = [cluster(
        0,
        [-1.0, -1.0, 9.0],
        [1.0, 1.0, 11.0],
        [0.0, 0.0, 0.0],
    )];
    let occlusion = [None];
    let v = assert_parity(&ctx, &culler, &clusters, &frustum, view, &occlusion);
    assert!(v[0].visible, "expected a visible verdict");
    assert_eq!(v[0].reason, CullReason::Visible);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_occlusion_culled_cluster_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cluster_cull parity: no wgpu adapter on this host");
        return;
    };
    let culler = GpuHairClusterCull::new(&ctx);
    let frustum = box_frustum();
    let view = ClusterView::new([0.0, 0.0, 0.0]);
    // Passes frustum and facing, but the cluster's closest depth (20) is behind
    // the nearest occluder (10) -> occlusion cull.
    let clusters = [cluster(
        0,
        [-1.0, -1.0, 9.0],
        [1.0, 1.0, 11.0],
        [1.0, 0.0, 0.0],
    )];
    let occlusion = [Some(OcclusionProbe::new(20.0, 10.0))];
    let v = assert_parity(&ctx, &culler, &clusters, &frustum, view, &occlusion);
    assert!(!v[0].visible, "expected a culled verdict");
    assert_eq!(v[0].reason, CullReason::Occlusion);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_front_occluder_keeps_cluster_visible_like_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cluster_cull parity: no wgpu adapter on this host");
        return;
    };
    let culler = GpuHairClusterCull::new(&ctx);
    let frustum = box_frustum();
    let view = ClusterView::new([0.0, 0.0, 0.0]);
    // A present probe whose occluder is behind the cluster (closest 10 < 20)
    // does not occlude -> visible, distinguishing a present-but-passing probe
    // from the no-probe path.
    let clusters = [cluster(
        0,
        [-1.0, -1.0, 9.0],
        [1.0, 1.0, 11.0],
        [1.0, 0.0, 0.0],
    )];
    let occlusion = [Some(OcclusionProbe::new(10.0, 20.0))];
    let v = assert_parity(&ctx, &culler, &clusters, &frustum, view, &occlusion);
    assert!(v[0].visible, "expected a visible verdict");
    assert_eq!(v[0].reason, CullReason::Visible);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_mixed_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cluster_cull parity: no wgpu adapter on this host");
        return;
    };
    let culler = GpuHairClusterCull::new(&ctx);
    let frustum = box_frustum();
    let view = ClusterView::new([0.0, 0.0, 0.0]);
    // One of each verdict, interleaved, to confirm per-thread indexing and the
    // per-cluster occlusion slice line up with the reference.
    let clusters = [
        cluster(0, [-1.0, -1.0, 9.0], [1.0, 1.0, 11.0], [1.0, 0.0, 0.0]), // visible
        cluster(1, [49.0, -1.0, 9.0], [51.0, 1.0, 11.0], [1.0, 0.0, 0.0]), // frustum
        cluster(2, [-1.0, -1.0, 9.0], [1.0, 1.0, 11.0], [0.0, 0.0, 1.0]), // backface
        cluster(3, [-1.0, -1.0, 9.0], [1.0, 1.0, 11.0], [1.0, 0.0, 0.0]), // occlusion
        cluster(4, [-2.0, -2.0, 20.0], [2.0, 2.0, 24.0], [0.0, 1.0, 0.0]), // visible
    ];
    let occlusion = [
        None,
        None,
        None,
        Some(OcclusionProbe::new(30.0, 12.0)),
        Some(OcclusionProbe::new(5.0, 40.0)),
    ];
    let v = assert_parity(&ctx, &culler, &clusters, &frustum, view, &occlusion);
    assert_eq!(v[0].reason, CullReason::Visible);
    assert_eq!(v[1].reason, CullReason::Frustum);
    assert_eq!(v[2].reason, CullReason::Backface);
    assert_eq!(v[3].reason, CullReason::Occlusion);
    assert_eq!(v[4].reason, CullReason::Visible);
}
