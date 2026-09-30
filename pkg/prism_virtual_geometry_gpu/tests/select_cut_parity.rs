//! Real-device parity for the cut-selection twin: [`GpuCutSelector`] must
//! reproduce the CPU golden
//! [`ClusterHierarchy::select_cut`](prism_render_architecture::virtual_geometry::ClusterHierarchy::select_cut)
//! as a set of drawn clusters, across multi-level descent, mixed per-subtree
//! resolution, frustum pruning and the degenerate (empty / malformed) inputs.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernels are portable
//! core-WGSL, so they need no optional device feature.
//!
//! # Parity criterion
//!
//! A cut is a *set* of drawn nodes (one cluster per visible region); the golden
//! emits them in stack-walk order while the twin emits in node-index order, so
//! both are sorted by node index before comparison and asserted element-for-
//! element with no tolerance. Every scene places its projected errors well clear
//! of the pixel budget, so each node's discrete draw/descend verdict is stable
//! under any legal float reassociation and identical to the golden regardless of
//! fused-multiply-add contraction.
//!
//! Provenance: standard Nanite-style hierarchical screen-space-error cut
//! selection; no Unreal Engine source or derived code.

use prism_render_architecture::gpu_scene::SceneBounds;
use prism_render_architecture::virtual_geometry::{
    ClusterHierarchy, ClusterNode, CutCluster, Frustum, GeometryPageKey, LodProjection, Plane,
};
use prism_virtual_geometry_gpu::{GpuContext, GpuCutSelector};

/// A wide frustum that admits everything the scenes place in front of it,
/// matching the golden's own test frustum.
fn wide_frustum() -> Frustum {
    Frustum::from_planes([
        Plane::new([1.0, 0.0, 0.0], 1000.0),
        Plane::new([-1.0, 0.0, 0.0], 1000.0),
        Plane::new([0.0, 1.0, 0.0], 1000.0),
        Plane::new([0.0, -1.0, 0.0], 1000.0),
        Plane::new([0.0, 0.0, 1.0], 0.0),
        Plane::new([0.0, 0.0, -1.0], 10000.0),
    ])
}

/// A sphere/AABB of `radius` centred on the `+z` axis at `z`.
fn bounds_at(z: f32, radius: f32) -> SceneBounds {
    SceneBounds {
        center: [0.0, 0.0, z],
        radius,
        half_extents: [radius, radius, radius],
        _padding: 0.0,
    }
}

/// 1000px focal, the golden's own test projection.
fn projection() -> LodProjection {
    LodProjection::from_focal_length_pixels(1000.0)
}

/// A three-level hierarchy exercising mixed per-subtree resolution:
///
/// ```text
///            0 root (err 8.0 -> 80px)         interior [1,2]
///           /                    \
///   1 A (err 2.0 -> 20px)      2 B (err 0.05 -> 0.5px, leaf)
///     interior [3,4]
///    /            \
///  3 leaf         4 leaf   (err 0.5 -> 5px each)
/// ```
///
/// At distance 100 and a 4px budget: the root and `A` are too coarse and are
/// refined through; `B` fits and is drawn; the two leaves under `A` are the
/// finest available and are drawn. Expected cut = {2, 3, 4}.
fn mixed_hierarchy() -> ClusterHierarchy {
    let root =
        ClusterNode::interior(bounds_at(100.0, 8.0), 8.0, GeometryPageKey::new(0, 0), 1, 2);
    let a = ClusterNode::interior(bounds_at(100.0, 4.0), 2.0, GeometryPageKey::new(0, 1), 3, 2);
    let b = ClusterNode::leaf(bounds_at(100.0, 1.0), 0.05, GeometryPageKey::new(0, 2));
    let leaf_l = ClusterNode::leaf(bounds_at(100.0, 2.0), 0.5, GeometryPageKey::new(0, 3));
    let leaf_r = ClusterNode::leaf(bounds_at(100.0, 2.0), 0.5, GeometryPageKey::new(0, 4));
    ClusterHierarchy::new(vec![root, a, b, leaf_l, leaf_r], vec![0])
}

/// A five-node linear chain `0 -> 1 -> 2 -> 3 -> 4(leaf)`, every interior too
/// coarse for a tight budget, exercising the relax fixed point over several
/// levels. Expected cut at a tight budget = {4}.
fn deep_chain() -> ClusterHierarchy {
    let n0 = ClusterNode::interior(bounds_at(50.0, 8.0), 16.0, GeometryPageKey::new(1, 0), 1, 1);
    let n1 = ClusterNode::interior(bounds_at(50.0, 8.0), 8.0, GeometryPageKey::new(1, 1), 2, 1);
    let n2 = ClusterNode::interior(bounds_at(50.0, 8.0), 4.0, GeometryPageKey::new(1, 2), 3, 1);
    let n3 = ClusterNode::interior(bounds_at(50.0, 8.0), 2.0, GeometryPageKey::new(1, 3), 4, 1);
    let n4 = ClusterNode::leaf(bounds_at(50.0, 8.0), 0.5, GeometryPageKey::new(1, 4));
    ClusterHierarchy::new(vec![n0, n1, n2, n3, n4], vec![0])
}

/// Sorts a cut by node index so two emission orders can be compared directly.
fn sorted(mut cut: Vec<CutCluster>) -> Vec<CutCluster> {
    cut.sort_by_key(|c| c.node);
    cut
}

/// Asserts the twin's cut equals the golden's cut (as sets) for one view, and
/// returns the sorted drawn-node indices for the caller to sanity-check.
fn assert_parity(
    ctx: &GpuContext,
    view_origin: [f32; 3],
    frustum: &Frustum,
    budget: f32,
    hierarchy: &ClusterHierarchy,
) -> Vec<u32> {
    let selector = GpuCutSelector::new(ctx);
    let proj = projection();
    let gpu = sorted(selector.select_cut(ctx, view_origin, frustum, proj, budget, hierarchy));
    let cpu = sorted(hierarchy.select_cut(view_origin, frustum, proj, budget));
    assert_eq!(
        gpu, cpu,
        "twin cut must match golden cut as a set (sorted by node)"
    );
    gpu.iter().map(|c| c.node).collect()
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_cut_matches_golden_mixed_resolution() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping select-cut parity: no wgpu adapter on this host");
        return;
    };
    let h = mixed_hierarchy();
    // Tight 4px budget: root (80px) and A (20px) refine through, B (0.5px) draws,
    // the two leaves under A draw. Expect drawn nodes {2, 3, 4}.
    let nodes = assert_parity(&ctx, [0.0, 0.0, 0.0], &wide_frustum(), 4.0, &h);
    assert_eq!(
        nodes,
        vec![2, 3, 4],
        "mixed scene must draw B and both leaves under A, saw {nodes:?}"
    );
}

#[test]
fn gpu_cut_draws_coarse_root_when_budget_is_loose() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let h = mixed_hierarchy();
    // Loose 200px budget: the root (80px) already fits, so nothing refines.
    let nodes = assert_parity(&ctx, [0.0, 0.0, 0.0], &wide_frustum(), 200.0, &h);
    assert_eq!(nodes, vec![0], "loose budget must draw only the root");
}

#[test]
fn gpu_cut_relax_reaches_deep_leaf() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let h = deep_chain();
    // Tight 4px budget: every interior is too coarse, so relax must reach the
    // depth-4 leaf across four propagation passes. Expect drawn node {4}.
    let nodes = assert_parity(&ctx, [0.0, 0.0, 0.0], &wide_frustum(), 4.0, &h);
    assert_eq!(nodes, vec![4], "deep chain must refine to its single leaf");
}

#[test]
fn gpu_cut_prunes_frustum_culled_subtree() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let h = mixed_hierarchy();
    // A frustum whose interior is x >= 500 rejects every node on the z axis at
    // x = 0, so the whole hierarchy is culled and the cut is empty.
    let frustum = Frustum::from_planes([
        Plane::new([1.0, 0.0, 0.0], -500.0),
        Plane::new([-1.0, 0.0, 0.0], 1000.0),
        Plane::new([0.0, 1.0, 0.0], 1000.0),
        Plane::new([0.0, -1.0, 0.0], 1000.0),
        Plane::new([0.0, 0.0, 1.0], 0.0),
        Plane::new([0.0, 0.0, -1.0], 10000.0),
    ]);
    let nodes = assert_parity(&ctx, [0.0, 0.0, 0.0], &frustum, 4.0, &h);
    assert!(nodes.is_empty(), "a fully culled hierarchy draws nothing");
}

#[test]
fn gpu_cut_rejects_malformed_hierarchy() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // Child range runs past the end of the node array: golden and twin both
    // degrade to an empty cut rather than reading out of bounds.
    let bad =
        ClusterNode::interior(bounds_at(50.0, 8.0), 4.0, GeometryPageKey::new(0, 0), 1, 5);
    let h = ClusterHierarchy::new(vec![bad], vec![0]);
    assert!(!h.is_well_formed());
    let selector = GpuCutSelector::new(&ctx);
    let cut = selector.select_cut(&ctx, [0.0, 0.0, 0.0], &wide_frustum(), projection(), 4.0, &h);
    assert!(cut.is_empty(), "a malformed hierarchy yields an empty cut");
}

#[test]
fn gpu_cut_empty_hierarchy_draws_nothing() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let h = ClusterHierarchy::new(Vec::new(), Vec::new());
    let selector = GpuCutSelector::new(&ctx);
    let cut = selector.select_cut(&ctx, [0.0, 0.0, 0.0], &wide_frustum(), projection(), 4.0, &h);
    assert!(cut.is_empty(), "an empty hierarchy yields an empty cut");
}
