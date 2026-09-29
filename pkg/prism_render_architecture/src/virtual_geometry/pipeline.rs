//! Per-cluster composition of the virtual-geometry decision layer.
//!
//! [`cull`](super::cull), [`lod`](super::lod), [`page_table`](super::page_table)
//! and [`raster_path`](super::raster_path) each own one decision; this module
//! sequences them into the single verdict a GPU-driven cluster loop consumes.
//! For one cluster the order is fixed: cull first, and only a surviving cluster
//! selects a LOD, records its page request (so streaming is driven by what is
//! actually visible), and picks a raster path. A culled cluster requests no
//! page, which is what lets the residency budget track the working set instead
//! of the whole scene.

use super::cull::{cluster_cull, CullVerdict, Frustum, OcclusionProbe};
use super::lod::{select_lod, LodLevel, LodProjection, LodSelection};
use super::page_table::GeometryPageTable;
use super::raster_path::{select_raster_path, ClusterRasterStats, RasterCapability};
use super::{GeometryLodPolicy, GeometryPageKey, GeometryRasterPath};
use crate::gpu_scene::SceneBounds;

/// Immutable per-view state shared across every cluster decision this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewCullContext {
    /// View frustum with normalized inward planes.
    pub frustum: Frustum,
    /// Error-to-pixel projection for LOD selection and page priority.
    pub projection: LodProjection,
    /// LOD refinement/prefetch tunables.
    pub lod_policy: GeometryLodPolicy,
    /// Backend raster capabilities.
    pub raster_capability: RasterCapability,
    /// Screen area below which a cluster prefers the software rasterizer.
    pub software_pixel_threshold: f32,
    /// Frame index used to stamp page requests for recency.
    pub frame: u64,
}

/// Inputs describing one cluster for a single frame's decision.
#[derive(Clone, Copy, Debug)]
pub struct ClusterRequest<'a> {
    /// Page identity used for residency bookkeeping.
    pub page: GeometryPageKey,
    /// World-space bounds for culling and screen-size priority.
    pub bounds: &'a SceneBounds,
    /// LOD chain (finest first is not required).
    pub lods: &'a [LodLevel],
    /// Cluster raster statistics for path classification.
    pub raster_stats: ClusterRasterStats,
    /// View-space distance to the cluster.
    pub view_distance: f32,
    /// Closing speed (positive when approaching) for LOD prefetch.
    pub closing_speed: f32,
    /// Optional occlusion probe over the cluster footprint.
    pub occlusion: Option<OcclusionProbe>,
    /// LOD chosen for this cluster last frame, for hysteresis.
    pub previous_lod: Option<u32>,
}

/// Outcome of deciding one cluster.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClusterDecision {
    /// Cull result; only [`CullVerdict::Visible`] carries LOD/raster data.
    pub verdict: CullVerdict,
    /// Selected LODs when visible with a non-empty chain.
    pub lod: Option<LodSelection>,
    /// Chosen raster path when visible.
    pub raster_path: Option<GeometryRasterPath>,
    /// Screen-derived streaming priority recorded for the page request.
    pub priority: f32,
}

impl ViewCullContext {
    /// Decides one cluster and, when visible, records its page request into
    /// `table` with a screen-size-derived priority.
    pub fn decide(
        &self,
        request: &ClusterRequest,
        table: &mut GeometryPageTable,
    ) -> ClusterDecision {
        let verdict = cluster_cull(&self.frustum, request.bounds, request.occlusion);
        if verdict != CullVerdict::Visible {
            return ClusterDecision {
                verdict,
                ..Default::default()
            };
        }

        let priority = self
            .projection
            .projected_error_pixels(request.bounds.radius, request.view_distance);
        table.request(request.page, priority, self.frame);

        let lod = select_lod(
            request.lods,
            self.projection,
            request.view_distance,
            request.closing_speed,
            self.lod_policy,
            request.previous_lod,
        );
        let raster_path = Some(select_raster_path(
            request.raster_stats,
            self.raster_capability,
            self.software_pixel_threshold,
        ));

        ClusterDecision {
            verdict,
            lod,
            raster_path,
            priority,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::cull::Plane;
    use super::super::page_table::PageResidency;
    use super::super::raster_path::DEFAULT_SOFTWARE_PIXEL_THRESHOLD;
    use super::*;

    fn context() -> ViewCullContext {
        ViewCullContext {
            frustum: Frustum::from_planes([
                Plane::new([1.0, 0.0, 0.0], 10.0),
                Plane::new([-1.0, 0.0, 0.0], 10.0),
                Plane::new([0.0, 1.0, 0.0], 10.0),
                Plane::new([0.0, -1.0, 0.0], 10.0),
                Plane::new([0.0, 0.0, 1.0], 0.0),
                Plane::new([0.0, 0.0, -1.0], 100.0),
            ]),
            projection: LodProjection::from_half_fov_tan(1000.0, 1.0),
            lod_policy: GeometryLodPolicy {
                target_error_pixels: 2.0,
                ..Default::default()
            },
            raster_capability: RasterCapability {
                mesh_shader: true,
                hardware_indirect: true,
            },
            software_pixel_threshold: DEFAULT_SOFTWARE_PIXEL_THRESHOLD,
            frame: 7,
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

    fn lods() -> [LodLevel; 3] {
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

    #[test]
    fn visible_cluster_selects_and_requests_page() {
        let ctx = context();
        let mut table = GeometryPageTable::new();
        let bounds = sphere_bounds([0.0, 0.0, 40.0], 1.0);
        let request = ClusterRequest {
            page: GeometryPageKey::new(1, 2),
            bounds: &bounds,
            lods: &lods(),
            raster_stats: ClusterRasterStats {
                max_triangle_pixels: 4096.0,
                triangle_count: 64,
            },
            view_distance: 40.0,
            closing_speed: 0.0,
            occlusion: None,
            previous_lod: None,
        };
        let decision = ctx.decide(&request, &mut table);
        assert_eq!(decision.verdict, CullVerdict::Visible);
        assert!(decision.lod.is_some());
        assert_eq!(decision.raster_path, Some(GeometryRasterPath::MeshShader));
        assert!(decision.priority > 0.0);
        assert_eq!(
            table.residency(GeometryPageKey::new(1, 2)),
            PageResidency::Requested
        );
    }

    #[test]
    fn frustum_culled_cluster_requests_nothing() {
        let ctx = context();
        let mut table = GeometryPageTable::new();
        let bounds = sphere_bounds([100.0, 0.0, 40.0], 1.0);
        let request = ClusterRequest {
            page: GeometryPageKey::new(3, 4),
            bounds: &bounds,
            lods: &lods(),
            raster_stats: ClusterRasterStats {
                max_triangle_pixels: 4096.0,
                triangle_count: 64,
            },
            view_distance: 40.0,
            closing_speed: 0.0,
            occlusion: None,
            previous_lod: None,
        };
        let decision = ctx.decide(&request, &mut table);
        assert_eq!(decision.verdict, CullVerdict::FrustumCulled);
        assert!(decision.lod.is_none());
        assert!(decision.raster_path.is_none());
        assert!(table.is_empty());
    }

    #[test]
    fn occluded_cluster_requests_nothing() {
        let ctx = context();
        let mut table = GeometryPageTable::new();
        let bounds = sphere_bounds([0.0, 0.0, 60.0], 1.0);
        let request = ClusterRequest {
            page: GeometryPageKey::new(5, 6),
            bounds: &bounds,
            lods: &lods(),
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
        };
        let decision = ctx.decide(&request, &mut table);
        assert_eq!(decision.verdict, CullVerdict::OcclusionCulled);
        assert!(table.is_empty());
    }
}
