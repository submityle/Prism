//! Software-vs-hardware rasterization path classification for clusters.
//!
//! Virtualized geometry mixes clusters whose triangles land at wildly different
//! screen sizes. Near sub-pixel triangles are cheapest through a compute
//! software rasterizer, which sidesteps fixed-function triangle setup and
//! small-triangle quad overshading; larger clusters are best served by the
//! hardware path, preferring mesh shaders where the backend exposes them, then
//! hardware indirect draws, and finally a plain mesh fallback.

use super::GeometryRasterPath;

/// Default screen area, in pixels squared, below which a cluster's triangles
/// are considered small enough to favor the software rasterizer.
pub const DEFAULT_SOFTWARE_PIXEL_THRESHOLD: f32 = 16.0;

/// Backend rasterization capabilities relevant to path selection.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RasterCapability {
    /// Mesh/amplification shader pipeline is available.
    pub mesh_shader: bool,
    /// Hardware indirect (multi-)draw is available.
    pub hardware_indirect: bool,
}

/// Per-cluster statistics driving the software/hardware decision.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClusterRasterStats {
    /// Largest projected triangle area in the cluster, in pixels squared.
    pub max_triangle_pixels: f32,
    /// Number of triangles in the cluster.
    pub triangle_count: u32,
}

/// Classifies the raster path for a cluster.
///
/// Empty clusters resolve to [`GeometryRasterPath::FallbackMesh`] as an inert
/// default. Otherwise clusters whose largest triangle is at or below
/// `software_pixel_threshold` take [`GeometryRasterPath::ComputeSoftware`]; the
/// remaining clusters take the best available hardware path.
#[must_use]
pub fn select_raster_path(
    stats: ClusterRasterStats,
    capability: RasterCapability,
    software_pixel_threshold: f32,
) -> GeometryRasterPath {
    if stats.triangle_count == 0 {
        return GeometryRasterPath::FallbackMesh;
    }
    if stats.max_triangle_pixels <= software_pixel_threshold.max(0.0) {
        return GeometryRasterPath::ComputeSoftware;
    }
    if capability.mesh_shader {
        GeometryRasterPath::MeshShader
    } else if capability.hardware_indirect {
        GeometryRasterPath::IndirectHardware
    } else {
        GeometryRasterPath::FallbackMesh
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL: RasterCapability = RasterCapability {
        mesh_shader: true,
        hardware_indirect: true,
    };

    #[test]
    fn empty_cluster_is_fallback() {
        let stats = ClusterRasterStats {
            max_triangle_pixels: 1000.0,
            triangle_count: 0,
        };
        assert_eq!(
            select_raster_path(stats, FULL, DEFAULT_SOFTWARE_PIXEL_THRESHOLD),
            GeometryRasterPath::FallbackMesh
        );
    }

    #[test]
    fn tiny_triangles_take_software_path_regardless_of_capability() {
        let stats = ClusterRasterStats {
            max_triangle_pixels: 4.0,
            triangle_count: 128,
        };
        assert_eq!(
            select_raster_path(stats, FULL, DEFAULT_SOFTWARE_PIXEL_THRESHOLD),
            GeometryRasterPath::ComputeSoftware
        );
    }

    #[test]
    fn large_triangles_prefer_mesh_shader_then_indirect_then_fallback() {
        let stats = ClusterRasterStats {
            max_triangle_pixels: 4096.0,
            triangle_count: 64,
        };
        assert_eq!(
            select_raster_path(stats, FULL, DEFAULT_SOFTWARE_PIXEL_THRESHOLD),
            GeometryRasterPath::MeshShader
        );
        let indirect = RasterCapability {
            mesh_shader: false,
            hardware_indirect: true,
        };
        assert_eq!(
            select_raster_path(stats, indirect, DEFAULT_SOFTWARE_PIXEL_THRESHOLD),
            GeometryRasterPath::IndirectHardware
        );
        let none = RasterCapability::default();
        assert_eq!(
            select_raster_path(stats, none, DEFAULT_SOFTWARE_PIXEL_THRESHOLD),
            GeometryRasterPath::FallbackMesh
        );
    }
}
