//! GPU-driven raster bin assignment for a selected cluster cut.
//!
//! Once [`select_cut`](super::hierarchy::select_cut) has resolved which cluster
//! nodes to draw this frame, each drawn cluster must be routed to a concrete
//! GPU rasterization path so the backend can build one indirect batch per path.
//! Routing is a pure, per-cluster classification: a cluster's raster statistics
//! plus the backend capabilities decide its [`GeometryRasterPath`], exactly as
//! [`select_raster_path`] specifies. This module fans a whole cut out into the
//! four path buckets in a single deterministic pass, leaving the physical draw
//! submission to the backend.

use alloc::vec::Vec;

use super::hierarchy::CutCluster;
use super::raster_path::{select_raster_path, ClusterRasterStats, RasterCapability};
use super::GeometryRasterPath;

/// A selected cut partitioned by the GPU rasterization path each cluster takes.
///
/// The backend consumes one bucket at a time, emitting a single indirect batch
/// per path: mesh-shader clusters, compute software-rasterized clusters,
/// hardware indirect draws, and the plain mesh fallback. Preserving per-bucket
/// draw order matches the input cut order, which keeps submission deterministic.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RasterBins {
    /// Clusters drawn through the mesh/amplification shader pipeline.
    pub mesh_shader: Vec<CutCluster>,
    /// Small-triangle clusters drawn through the compute software rasterizer.
    pub compute_software: Vec<CutCluster>,
    /// Clusters drawn through hardware indirect draws.
    pub indirect_hardware: Vec<CutCluster>,
    /// Clusters drawn through the plain mesh fallback path.
    pub fallback_mesh: Vec<CutCluster>,
}

impl RasterBins {
    /// Total number of clusters across every bucket.
    #[must_use]
    pub fn total(&self) -> usize {
        self.mesh_shader.len()
            + self.compute_software.len()
            + self.indirect_hardware.len()
            + self.fallback_mesh.len()
    }

    /// Returns `true` when no cluster landed in any bucket.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mesh_shader.is_empty()
            && self.compute_software.is_empty()
            && self.indirect_hardware.is_empty()
            && self.fallback_mesh.is_empty()
    }

    /// Mutable handle to the bucket backing a given raster path.
    fn bucket_mut(&mut self, path: GeometryRasterPath) -> &mut Vec<CutCluster> {
        match path {
            GeometryRasterPath::MeshShader => &mut self.mesh_shader,
            GeometryRasterPath::ComputeSoftware => &mut self.compute_software,
            GeometryRasterPath::IndirectHardware => &mut self.indirect_hardware,
            GeometryRasterPath::FallbackMesh => &mut self.fallback_mesh,
        }
    }
}

/// Partitions a selected cut into per-path raster buckets.
///
/// Each [`CutCluster::node`] indexes the parallel `raster_stats` slice, which
/// carries the per-cluster raster statistics gathered during the draw-prep
/// stage. Keeping the statistics in a parallel slice keeps the hierarchy's
/// [`ClusterNode`](super::hierarchy::ClusterNode) topology lean. A cluster
/// whose node index falls outside `raster_stats` is skipped rather than
/// panicking, so a stale cut cannot crash draw submission; every in-range
/// cluster is routed via [`select_raster_path`].
#[must_use]
pub fn bin_cut(
    cut: &[CutCluster],
    raster_stats: &[ClusterRasterStats],
    capability: RasterCapability,
    software_pixel_threshold: f32,
) -> RasterBins {
    let mut bins = RasterBins::default();
    for &cluster in cut {
        let Some(&stats) = raster_stats.get(cluster.node as usize) else {
            continue;
        };
        let path = select_raster_path(stats, capability, software_pixel_threshold);
        bins.bucket_mut(path).push(cluster);
    }
    bins
}

#[cfg(test)]
mod tests {
    use super::super::raster_path::DEFAULT_SOFTWARE_PIXEL_THRESHOLD;
    use super::super::GeometryPageKey;
    use super::*;
    use alloc::vec;

    const FULL: RasterCapability = RasterCapability {
        mesh_shader: true,
        hardware_indirect: true,
    };

    fn cluster(node: u32) -> CutCluster {
        CutCluster {
            node,
            page: GeometryPageKey::new(0, node),
        }
    }

    fn stats(max_triangle_pixels: f32) -> ClusterRasterStats {
        ClusterRasterStats {
            max_triangle_pixels,
            triangle_count: 1,
        }
    }

    #[test]
    fn routes_each_cluster_to_its_path() {
        // node 0: tiny triangle -> software; node 1: large -> mesh shader.
        let cut = [cluster(0), cluster(1)];
        let raster_stats = [stats(1.0), stats(256.0)];
        let bins = bin_cut(&cut, &raster_stats, FULL, DEFAULT_SOFTWARE_PIXEL_THRESHOLD);
        assert_eq!(bins.compute_software, vec![cluster(0)]);
        assert_eq!(bins.mesh_shader, vec![cluster(1)]);
        assert!(bins.indirect_hardware.is_empty());
        assert!(bins.fallback_mesh.is_empty());
        assert_eq!(bins.total(), 2);
    }

    #[test]
    fn hardware_indirect_when_no_mesh_shader() {
        let cut = [cluster(0)];
        let raster_stats = [stats(256.0)];
        let capability = RasterCapability {
            mesh_shader: false,
            hardware_indirect: true,
        };
        let bins = bin_cut(
            &cut,
            &raster_stats,
            capability,
            DEFAULT_SOFTWARE_PIXEL_THRESHOLD,
        );
        assert_eq!(bins.indirect_hardware, vec![cluster(0)]);
    }

    #[test]
    fn preserves_cut_order_within_a_bucket() {
        let cut = [cluster(2), cluster(0), cluster(1)];
        let raster_stats = [stats(256.0), stats(256.0), stats(256.0)];
        let bins = bin_cut(&cut, &raster_stats, FULL, DEFAULT_SOFTWARE_PIXEL_THRESHOLD);
        assert_eq!(bins.mesh_shader, vec![cluster(2), cluster(0), cluster(1)]);
    }

    #[test]
    fn skips_out_of_range_nodes() {
        // node 5 has no stats entry and must be dropped, not panic.
        let cut = [cluster(0), cluster(5)];
        let raster_stats = [stats(256.0)];
        let bins = bin_cut(&cut, &raster_stats, FULL, DEFAULT_SOFTWARE_PIXEL_THRESHOLD);
        assert_eq!(bins.total(), 1);
        assert_eq!(bins.mesh_shader, vec![cluster(0)]);
    }

    #[test]
    fn empty_cut_is_empty() {
        let bins = bin_cut(&[], &[], FULL, DEFAULT_SOFTWARE_PIXEL_THRESHOLD);
        assert!(bins.is_empty());
        assert_eq!(bins.total(), 0);
    }
}
