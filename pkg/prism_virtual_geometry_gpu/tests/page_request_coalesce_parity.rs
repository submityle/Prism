//! Real-device parity for the page-request coalescing twin:
//! [`GpuPageRequestCoalescer`] must reproduce the `page_requests` a CPU golden
//! [`plan_frame`](prism_render_architecture::virtual_geometry::plan_frame)
//! records for the same drawn clusters — one coalesced request per unique page
//! at the highest screen-coverage priority any referencing cluster reported.
//!
//! The tests skip (with a printed notice on the first) when the host has no
//! `wgpu` adapter, so the suite stays green everywhere while still exercising
//! the full dispatch-and-readback on any real device. The kernel is portable
//! core-WGSL, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The coalesced priority is a continuous value, so per-page priorities are
//! compared within one ULP (their float bit-patterns differ by at most one) to
//! absorb any legal fused-multiply-add contraction of the `extent^2 /
//! distance_sq` term; the number of distinct pages is asserted exactly. The
//! golden is the real public `plan_frame`, and the twin is driven with the
//! exact bounds and pages of the clusters that golden drew.
//!
//! Provenance: standard screen-coverage page-priority coalescing; no Unreal
//! Engine source or derived code.

use std::collections::BTreeSet;

use prism_render_architecture::gpu_scene::SceneBounds;
use prism_render_architecture::virtual_geometry::{
    plan_frame, ClusterHierarchy, ClusterNode, ClusterRasterStats, Frustum, FrameView,
    GeometryPageKey, LodProjection, Plane, RasterCapability, RasterConfig,
};
use prism_virtual_geometry_gpu::{GpuContext, GpuPageRequestCoalescer, PageReference};

/// Full raster capability; the raster path never affects `page_requests`, only
/// the bins, so the value is immaterial to this suite.
const FULL: RasterCapability = RasterCapability {
    mesh_shader: true,
    hardware_indirect: true,
};

/// A frustum wide enough that every test leaf placed ahead of the camera is
/// visible, so `select_cut` draws all of them.
fn wide_frustum() -> Frustum {
    Frustum::from_planes([
        Plane::new([1.0, 0.0, 0.0], 10_000.0),
        Plane::new([-1.0, 0.0, 0.0], 10_000.0),
        Plane::new([0.0, 1.0, 0.0], 10_000.0),
        Plane::new([0.0, -1.0, 0.0], 10_000.0),
        Plane::new([0.0, 0.0, 1.0], 0.0),
        Plane::new([0.0, 0.0, -1.0], 100_000.0),
    ])
}

/// The golden's own test projection scale.
fn projection() -> LodProjection {
    LodProjection::from_focal_length_pixels(1000.0)
}

fn bounds_at(center: [f32; 3], radius: f32) -> SceneBounds {
    SceneBounds {
        center,
        radius,
        half_extents: [radius, radius, radius],
        _padding: 0.0,
    }
}

/// One test leaf: bounds plus the page it references.
struct Leaf {
    center: [f32; 3],
    radius: f32,
    page: GeometryPageKey,
}

/// Builds a leaf-only hierarchy (each leaf its own root, so every visible leaf
/// is drawn), runs the golden `plan_frame`, and drives the twin with the exact
/// bounds and pages of the drawn clusters. Asserts distinct-page count and every
/// per-page priority (within one ULP).
fn assert_parity(ctx: &GpuContext, view_origin: [f32; 3], leaves: &[Leaf]) {
    let nodes: Vec<ClusterNode> = leaves
        .iter()
        .map(|l| ClusterNode::leaf(bounds_at(l.center, l.radius), 0.0, l.page))
        .collect();
    let roots: Vec<u32> = (0..leaves.len() as u32).collect();
    let hierarchy = ClusterHierarchy::new(nodes, roots);

    // Large per-cluster stats; the raster binning is irrelevant to page requests.
    let stats: Vec<ClusterRasterStats> = leaves
        .iter()
        .map(|_| ClusterRasterStats {
            max_triangle_pixels: 256.0,
            triangle_count: 12,
        })
        .collect();

    let frustum = wide_frustum();
    let plan = plan_frame(
        &hierarchy,
        FrameView {
            origin: view_origin,
            frustum: &frustum,
            projection: projection(),
            target_error_pixels: 4.0,
        },
        RasterConfig {
            stats: &stats,
            capability: FULL,
            // Immaterial to page requests; any value works (golden default is 16.0).
            software_pixel_threshold: 16.0,
        },
    );

    // Every leaf is a visible leaf, so all are drawn.
    assert_eq!(
        plan.drawn_cluster_count(),
        leaves.len(),
        "expected every visible leaf to be drawn"
    );

    // Drive the twin with the exact drawn clusters' bounds and pages.
    let hierarchy_nodes = hierarchy.nodes();
    let references: Vec<PageReference> = plan
        .cut
        .iter()
        .map(|c| {
            let bounds = hierarchy_nodes[c.node as usize].bounds;
            PageReference {
                page: c.page,
                center: bounds.center,
                radius: bounds.radius,
            }
        })
        .collect();

    let coalescer = GpuPageRequestCoalescer::new(ctx);
    let gpu_batch = coalescer.coalesce(ctx, view_origin, projection(), &references);

    assert_eq!(
        gpu_batch.len(),
        plan.page_requests.len(),
        "distinct-page count must match the golden batch"
    );

    let unique_pages: BTreeSet<GeometryPageKey> = plan.cut.iter().map(|c| c.page).collect();
    for page in unique_pages {
        let golden = plan
            .page_requests
            .priority(page)
            .expect("golden batch must hold every drawn page");
        let gpu = gpu_batch
            .priority(page)
            .expect("gpu batch must hold every drawn page");
        assert!(
            within_1_ulp(golden, gpu),
            "page {page:?}: golden priority {golden} vs gpu {gpu} exceeds 1 ULP"
        );
    }
}

/// True when two finite `f32` values are equal or their bit patterns are
/// adjacent (differ by at most one ULP).
fn within_1_ulp(a: f32, b: f32) -> bool {
    if a == b {
        return true;
    }
    if a.is_nan() || b.is_nan() || a.signum() != b.signum() {
        return false;
    }
    let ai = a.to_bits() as i64;
    let bi = b.to_bits() as i64;
    (ai - bi).abs() <= 1
}

#[test]
fn gpu_coalesce_matches_golden_distinct_pages() {
    let Some(ctx) = GpuContext::try_headless() else {
        #[expect(
            clippy::print_stderr,
            reason = "test-only notice so the suite stays green on adapterless CI"
        )]
        {
            eprintln!("skipping page-request-coalesce parity: no wgpu adapter on this host");
        }
        return;
    };
    // Each leaf references a distinct page: one request per leaf.
    let leaves = [
        Leaf {
            center: [0.0, 0.0, 50.0],
            radius: 2.0,
            page: GeometryPageKey::new(0, 0),
        },
        Leaf {
            center: [10.0, 5.0, 80.0],
            radius: 1.5,
            page: GeometryPageKey::new(0, 1),
        },
        Leaf {
            center: [-20.0, 8.0, 120.0],
            radius: 3.0,
            page: GeometryPageKey::new(1, 0),
        },
    ];
    assert_parity(&ctx, [0.0, 0.0, 0.0], &leaves);
}

#[test]
fn gpu_coalesce_matches_golden_shared_page_keeps_max() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // Three leaves all reference one page at different coverages: the coalesced
    // priority must be the maximum (the nearest / largest cluster).
    let page = GeometryPageKey::new(3, 7);
    let leaves = [
        Leaf {
            center: [0.0, 0.0, 200.0],
            radius: 1.0,
            page,
        },
        Leaf {
            center: [0.0, 0.0, 60.0],
            radius: 4.0,
            page,
        },
        Leaf {
            center: [5.0, 0.0, 130.0],
            radius: 2.0,
            page,
        },
    ];
    assert_parity(&ctx, [0.0, 0.0, 0.0], &leaves);
}

#[test]
fn gpu_coalesce_matches_golden_mixed_shared_and_distinct() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let shared = GeometryPageKey::new(2, 2);
    let leaves = [
        Leaf {
            center: [0.0, 0.0, 70.0],
            radius: 2.5,
            page: shared,
        },
        Leaf {
            center: [3.0, -4.0, 90.0],
            radius: 1.0,
            page: shared,
        },
        Leaf {
            center: [30.0, 10.0, 150.0],
            radius: 5.0,
            page: GeometryPageKey::new(2, 3),
        },
        Leaf {
            center: [-15.0, -6.0, 110.0],
            radius: 0.75,
            page: GeometryPageKey::new(0, 9),
        },
    ];
    assert_parity(&ctx, [1.0, -2.0, 3.0], &leaves);
}

#[test]
fn gpu_coalesce_matches_golden_zero_radius_page() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // A zero-radius cluster projects to zero coverage; its page must still be
    // requested (at priority 0), matching the golden's `record(key, 0.0)`.
    let leaves = [
        Leaf {
            center: [0.0, 0.0, 40.0],
            radius: 0.0,
            page: GeometryPageKey::new(5, 5),
        },
        Leaf {
            center: [8.0, 2.0, 95.0],
            radius: 2.0,
            page: GeometryPageKey::new(5, 6),
        },
    ];
    assert_parity(&ctx, [0.0, 0.0, 0.0], &leaves);
}

#[test]
fn gpu_coalesce_empty_references_is_empty_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let coalescer = GpuPageRequestCoalescer::new(&ctx);
    let batch = coalescer.coalesce(&ctx, [0.0, 0.0, 0.0], projection(), &[]);
    assert!(batch.is_empty(), "empty references must yield an empty batch");
    assert_eq!(batch.len(), 0);
}
