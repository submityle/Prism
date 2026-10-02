//! GPU-side clustered-light ABI.
//!
//! The Forward+ resolve pass reads three parallel resources produced by the CPU
//! golden [`assign_lights_to_clusters`](prism_render_shading::assign_lights_to_clusters):
//!
//! * a single [`GpuClusterGrid`] uniform describing the froxel grid layout plus
//!   the precomputed depth-slice constants the shader needs to locate a pixel's
//!   cluster without a transcendental round-trip,
//! * a per-cluster `[offset, count]` table (`[u32; 2]` records), and
//! * a flat `u32` light-index list the table slices into.
//!
//! The grid record is padded to a whole number of 16-byte rows so the same
//! `#[repr(C)]` layout is valid as a `std140` uniform today or a `std430`
//! storage block later.  The offset/count and index tables are plain `u32`
//! arrays, matching [`ClusterLightAssignment`](prism_render_shading::ClusterLightAssignment)
//! byte-for-byte.

use bevy_math::ops;
use bytemuck::{Pod, Zeroable};
use prism_render_shading::ClusterGrid;

/// Column-major identity matrix used as the neutral view for the fallback grid.
const IDENTITY_COLS: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
];

/// Froxel-grid description consumed by the clustered resolve shader.
///
/// `depth_scale` and `depth_bias` linearize the exponential depth slicing: a
/// view-space depth `d` (a positive distance in front of the camera) maps to
/// its slice index with `floor(depth_scale * ln(d) + depth_bias)`, which is the
/// algebraic expansion of the CPU grid's
/// `floor(dimensions.z * ln(d / near) / ln(far / near))`.  Precomputing the two
/// constants keeps the shader's per-pixel work to a single `log` and a
/// multiply-add.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuClusterGrid {
    /// Cluster counts along `x`, `y`, and `z`.
    pub dimensions: [u32; 3],
    /// `dimensions.x * dimensions.y * dimensions.z`, cached for bounds checks.
    pub cluster_count: u32,
    /// Screen-space tile footprint in pixels along `x` and `y`.
    pub tile_size: [u32; 2],
    /// Render-target resolution in pixels.
    pub screen_size: [u32; 2],
    /// Near-plane distance (`> 0`) of the first depth slice.
    pub near: f32,
    /// Far-plane distance (`> near`) of the last depth slice.
    pub far: f32,
    /// `dimensions.z / ln(far / near)`; the depth-slice `log` scale.
    pub depth_scale: f32,
    /// `-ln(near) * depth_scale`; the depth-slice `log` bias.
    pub depth_bias: f32,
    /// Hard cap on lights recorded per cluster, mirrored from the assignment
    /// config so the shader can size its inner loop conservatively.
    pub max_lights_per_cluster: u32,
    /// Padding to a 16-byte boundary; always zero.
    pub _padding: [u32; 3],
    /// World -> view matrix (column-major) the froxel grid was built against.
    ///
    /// The resolve shader transforms each fragment's world position into this
    /// view space to recover its depth slice, so the matrix must be exactly the
    /// one the CPU assignment used.  Appended after the padded 64-byte prefix so
    /// the leading layout the depth-slice constants live in is unchanged.
    pub view_from_world: [f32; 16],
}

impl Default for GpuClusterGrid {
    fn default() -> Self {
        Self {
            dimensions: [1, 1, 1],
            cluster_count: 1,
            tile_size: [1, 1],
            screen_size: [1, 1],
            near: 0.1,
            far: 100.0,
            depth_scale: 0.0,
            depth_bias: 0.0,
            max_lights_per_cluster: 0,
            _padding: [0; 3],
            view_from_world: IDENTITY_COLS,
        }
    }
}

impl GpuClusterGrid {
    /// Mirrors a CPU [`ClusterGrid`] into the GPU record, precomputing the
    /// depth-slice `log` constants.
    pub fn from_grid(
        grid: &ClusterGrid,
        max_lights_per_cluster: u32,
        view_from_world: [f32; 16],
    ) -> Self {
        let depth_range_ln = ops::ln(grid.far / grid.near);
        let depth_scale = if depth_range_ln != 0.0 {
            grid.dimensions[2] as f32 / depth_range_ln
        } else {
            0.0
        };
        let depth_bias = -ops::ln(grid.near) * depth_scale;
        Self {
            dimensions: grid.dimensions,
            cluster_count: grid.cluster_count(),
            tile_size: grid.tile_size,
            screen_size: grid.screen_size,
            near: grid.near,
            far: grid.far,
            depth_scale,
            depth_bias,
            max_lights_per_cluster,
            _padding: [0; 3],
            view_from_world,
        }
    }

    /// Reproduces the CPU grid's depth-slice index from a view-space `z`
    /// (negative in front of the camera) using the precomputed constants.
    ///
    /// This is the exact arithmetic the resolve shader performs; the CPU copy
    /// exists only so the ABI can be golden-tested against
    /// [`ClusterGrid::z_slice`] without a GPU.
    pub fn depth_slice(&self, view_z: f32) -> u32 {
        let depth = (-view_z).max(self.near);
        let raw = self.depth_scale * ops::ln(depth) + self.depth_bias;
        let clamped = raw.max(0.0);
        (clamped as u32).min(self.dimensions[2].saturating_sub(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_size_and_alignment_is_std140_safe() {
        assert_eq!(size_of::<GpuClusterGrid>(), 128);
        assert_eq!(align_of::<GpuClusterGrid>(), 4);
        assert_eq!(size_of::<GpuClusterGrid>() % 16, 0);
    }

    #[test]
    fn from_grid_mirrors_layout_and_count() {
        let grid = ClusterGrid::new([16, 9, 24], [1600, 900], 0.1, 100.0);
        let gpu = GpuClusterGrid::from_grid(&grid, 256, IDENTITY_COLS);
        assert_eq!(gpu.dimensions, grid.dimensions);
        assert_eq!(gpu.tile_size, grid.tile_size);
        assert_eq!(gpu.screen_size, grid.screen_size);
        assert_eq!(gpu.cluster_count, grid.cluster_count());
        assert_eq!(gpu.near, grid.near);
        assert_eq!(gpu.far, grid.far);
        assert_eq!(gpu.max_lights_per_cluster, 256);
        assert_eq!(gpu._padding, [0; 3]);
        assert_eq!(gpu.view_from_world, IDENTITY_COLS);
    }

    #[test]
    fn depth_slice_matches_the_cpu_grid_golden() {
        let grid = ClusterGrid::new([1, 1, 24], [16, 16], 0.5, 500.0);
        let gpu = GpuClusterGrid::from_grid(&grid, 256, IDENTITY_COLS);
        for k in 0..24u32 {
            let mid_depth = 0.5 * (grid.slice_depth(k) + grid.slice_depth(k + 1));
            assert_eq!(
                gpu.depth_slice(-mid_depth),
                grid.z_slice(-mid_depth),
                "slice {k}"
            );
        }
    }

    #[test]
    fn depth_slice_saturates_outside_the_frustum() {
        let grid = ClusterGrid::new([1, 1, 24], [16, 16], 0.5, 500.0);
        let gpu = GpuClusterGrid::from_grid(&grid, 256, IDENTITY_COLS);
        assert_eq!(gpu.depth_slice(-0.01), 0);
        assert_eq!(gpu.depth_slice(-100_000.0), 23);
    }

    #[test]
    fn defaults_are_a_neutral_single_cluster() {
        let gpu = GpuClusterGrid::default();
        assert_eq!(gpu.cluster_count, 1);
        assert_eq!(gpu.dimensions, [1, 1, 1]);
        assert_eq!(gpu.max_lights_per_cluster, 0);
    }
}
