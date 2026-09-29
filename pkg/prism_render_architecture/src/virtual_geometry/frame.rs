//! Per-frame virtual-geometry draw-plan assembly.
//!
//! The submodules of this feature each own one decision; this module composes
//! them into the single artifact a backend consumes per view per frame. Given a
//! cluster hierarchy, the current view and the per-cluster raster statistics, it
//! runs the screen-space-error cut, coalesces the drawn clusters' page requests,
//! and fans the cut out into GPU raster buckets — producing a
//! [`GeometryFramePlan`] the backend can turn directly into streaming requests
//! and indirect draw batches. It stays GPU-independent and deterministic so the
//! whole per-frame decision can be unit-tested end to end.

use super::bins::{bin_cut, RasterBins};
use super::cull::Frustum;
use super::hierarchy::{coverage_priority, ClusterHierarchy, CutCluster};
use super::lod::LodProjection;
use super::page_request::PageRequestBatch;
use super::raster_path::{ClusterRasterStats, RasterCapability};
use alloc::vec::Vec;

/// The view parameters that drive one frame's cut selection.
#[derive(Clone, Copy, Debug)]
pub struct FrameView<'a> {
    /// Camera world-space position.
    pub origin: [f32; 3],
    /// Normalized inward frustum planes for visibility culling.
    pub frustum: &'a Frustum,
    /// Object-space-error to pixels projection for this view.
    pub projection: LodProjection,
    /// Per-frame projected-error pixel budget; the coarsest cluster within it
    /// is drawn.
    pub target_error_pixels: f32,
}

/// The raster-path configuration applied when binning a frame's cut.
#[derive(Clone, Copy, Debug)]
pub struct RasterConfig<'a> {
    /// Per-cluster raster statistics, indexed by [`CutCluster::node`].
    pub stats: &'a [ClusterRasterStats],
    /// Backend rasterization capabilities.
    pub capability: RasterCapability,
    /// Screen area below which a cluster favours the software rasterizer.
    pub software_pixel_threshold: f32,
}

/// Everything a backend needs to render one view of a virtualized-geometry
/// asset for one frame.
///
/// The three fields are consistent by construction: `bins` partitions exactly
/// `cut`, and `page_requests` holds one coalesced request per page any drawn
/// cluster references, at that page's highest screen coverage.
#[derive(Clone, Debug, Default)]
pub struct GeometryFramePlan {
    /// The selected screen-space-error cut, one cluster per visible region.
    pub cut: Vec<CutCluster>,
    /// The cut fanned out into per-path GPU raster buckets.
    pub bins: RasterBins,
    /// Coalesced page-streaming requests for the cut's pages.
    pub page_requests: PageRequestBatch,
}

impl GeometryFramePlan {
    /// Number of clusters drawn this frame.
    #[must_use]
    pub fn drawn_cluster_count(&self) -> usize {
        self.cut.len()
    }

    /// Returns `true` when nothing is drawn (empty cut).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cut.is_empty()
    }
}

/// Assembles the full per-frame virtual-geometry draw plan for one view.
///
/// Runs [`ClusterHierarchy::select_cut`] for the cut, records each drawn
/// cluster's page into a [`PageRequestBatch`] at its projected screen coverage
/// (so a page pulled by any high-coverage cluster streams in with that
/// urgency), and bins the cut with [`bin_cut`]. The page priority matches the
/// value [`ClusterHierarchy::select_cut_streaming`] records, so the two entry
/// points agree. A malformed hierarchy yields an empty plan because the cut is
/// empty.
#[must_use]
pub fn plan_frame(
    hierarchy: &ClusterHierarchy,
    view: FrameView<'_>,
    raster: RasterConfig<'_>,
) -> GeometryFramePlan {
    let cut = hierarchy.select_cut(
        view.origin,
        view.frustum,
        view.projection,
        view.target_error_pixels,
    );

    let nodes = hierarchy.nodes();
    let mut page_requests = PageRequestBatch::new();
    for cluster in &cut {
        // Cut entries always index in-range nodes, but guard anyway so a future
        // caller feeding an external cut cannot read out of bounds.
        if let Some(node) = nodes.get(cluster.node as usize) {
            let priority = coverage_priority(&node.bounds, view.origin, view.projection);
            page_requests.record(cluster.page, priority);
        }
    }

    let bins = bin_cut(
        &cut,
        raster.stats,
        raster.capability,
        raster.software_pixel_threshold,
    );

    GeometryFramePlan {
        cut,
        bins,
        page_requests,
    }
}

#[cfg(test)]
mod tests {
    use super::super::cull::Plane;
    use super::super::hierarchy::ClusterNode;
    use super::super::raster_path::DEFAULT_SOFTWARE_PIXEL_THRESHOLD;
    use super::super::GeometryPageKey;
    use super::*;
    use crate::gpu_scene::SceneBounds;
    use alloc::vec;

    const FULL: RasterCapability = RasterCapability {
        mesh_shader: true,
        hardware_indirect: true,
    };

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

    fn bounds_at(z: f32, radius: f32) -> SceneBounds {
        SceneBounds {
            center: [0.0, 0.0, z],
            radius,
            half_extents: [radius, radius, radius],
            _padding: 0.0,
        }
    }

    fn projection() -> LodProjection {
        LodProjection::from_focal_length_pixels(1000.0)
    }

    #[test]
    fn empty_hierarchy_yields_empty_plan() {
        let hierarchy = ClusterHierarchy::default();
        let frustum = wide_frustum();
        let plan = plan_frame(
            &hierarchy,
            FrameView {
                origin: [0.0, 0.0, 0.0],
                frustum: &frustum,
                projection: projection(),
                target_error_pixels: 4.0,
            },
            RasterConfig {
                stats: &[],
                capability: FULL,
                software_pixel_threshold: DEFAULT_SOFTWARE_PIXEL_THRESHOLD,
            },
        );
        assert!(plan.is_empty());
        assert_eq!(plan.drawn_cluster_count(), 0);
        assert!(plan.page_requests.is_empty());
        assert!(plan.bins.is_empty());
    }

    #[test]
    fn single_leaf_plan_is_consistent() {
        // One leaf, large triangles -> mesh-shader bucket, one page request.
        let leaf = ClusterNode::leaf(bounds_at(100.0, 1.0), 0.0, GeometryPageKey::new(0, 0));
        let hierarchy = ClusterHierarchy::new(vec![leaf], vec![0]);
        let stats = [ClusterRasterStats {
            max_triangle_pixels: 256.0,
            triangle_count: 12,
        }];
        let frustum = wide_frustum();
        let plan = plan_frame(
            &hierarchy,
            FrameView {
                origin: [0.0, 0.0, 0.0],
                frustum: &frustum,
                projection: projection(),
                target_error_pixels: 4.0,
            },
            RasterConfig {
                stats: &stats,
                capability: FULL,
                software_pixel_threshold: DEFAULT_SOFTWARE_PIXEL_THRESHOLD,
            },
        );
        assert_eq!(plan.drawn_cluster_count(), 1);
        // bins partition exactly the cut.
        assert_eq!(plan.bins.total(), plan.cut.len());
        assert_eq!(plan.bins.mesh_shader, vec![plan.cut[0]]);
        // one page request recorded for the drawn page.
        assert_eq!(plan.page_requests.len(), 1);
        assert!(plan
            .page_requests
            .priority(GeometryPageKey::new(0, 0))
            .is_some());
    }

    #[test]
    fn shared_page_across_leaves_coalesces_to_one_request() {
        // Two leaves backed by the same page must yield a single page request.
        let page = GeometryPageKey::new(7, 3);
        let a = ClusterNode::leaf(bounds_at(80.0, 1.0), 0.0, page);
        let b = ClusterNode::leaf(bounds_at(120.0, 1.0), 0.0, page);
        let hierarchy = ClusterHierarchy::new(vec![a, b], vec![0, 1]);
        let stats = [
            ClusterRasterStats {
                max_triangle_pixels: 256.0,
                triangle_count: 4,
            },
            ClusterRasterStats {
                max_triangle_pixels: 256.0,
                triangle_count: 4,
            },
        ];
        let frustum = wide_frustum();
        let plan = plan_frame(
            &hierarchy,
            FrameView {
                origin: [0.0, 0.0, 0.0],
                frustum: &frustum,
                projection: projection(),
                target_error_pixels: 4.0,
            },
            RasterConfig {
                stats: &stats,
                capability: FULL,
                software_pixel_threshold: DEFAULT_SOFTWARE_PIXEL_THRESHOLD,
            },
        );
        assert_eq!(plan.drawn_cluster_count(), 2);
        // Two clusters drawn, but they share one page -> one coalesced request.
        assert_eq!(plan.page_requests.len(), 1);
        assert_eq!(plan.bins.total(), 2);
    }
}
