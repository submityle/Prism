//! CPU build step turning extracted punctual lights into the clustered GPU
//! tables.
//!
//! This is the render-world adapter around the CPU golden
//! [`assign_lights_to_clusters`](prism_render_shading::assign_lights_to_clusters):
//! it constructs the froxel [`ClusterGrid`] for the active view, converts the
//! flat [`GpuPunctualLight`] records back into the reference
//! [`PunctualLight`](prism_render_shading::PunctualLight) type the assignment
//! consumes, runs the assignment, and packs the result into the three GPU
//! resources described in [`super::abi`].
//!
//! No GPU is touched here, so the whole pipeline is golden-testable on CI: the
//! packed `[offset, count]` table and index list are byte-identical to the
//! reference [`ClusterLightAssignment`](prism_render_shading::ClusterLightAssignment),
//! and the grid uniform round-trips through [`GpuClusterGrid::from_grid`].

use alloc::vec::Vec;

use bevy_ecs::prelude::Resource;
use prism_render_shading::{
    assign_lights_to_clusters, ClusterAssignmentConfig, ClusterGrid, Mat4, PunctualLight,
};

use super::abi::GpuClusterGrid;
use crate::lighting::abi::GpuPunctualLight;

/// Tunables for the per-view froxel grid and its light assignment.
#[derive(Resource, Clone, Copy, Debug)]
pub struct ClusterConfig {
    /// Target on-screen tile footprint in pixels; the `x`/`y` cluster counts
    /// are derived to cover the render target.
    pub tile_size: [u32; 2],
    /// Number of exponential depth slices along the view `z` axis.
    pub z_slices: u32,
    /// Hard cap on lights recorded per cluster.
    pub max_lights_per_cluster: u32,
    /// Illuminance threshold used to bound a range-less light's influence.
    pub intensity_cutoff: f32,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            tile_size: [64, 64],
            z_slices: 24,
            max_lights_per_cluster: 256,
            intensity_cutoff: 0.01,
        }
    }
}

impl ClusterConfig {
    /// The assignment-layer config derived from this render-layer config.
    fn assignment(&self) -> ClusterAssignmentConfig {
        ClusterAssignmentConfig {
            max_lights_per_cluster: self.max_lights_per_cluster,
            intensity_cutoff: self.intensity_cutoff,
        }
    }
}

/// The CPU-side clustered tables, ready to be streamed into GPU buffers.
///
/// `grid` is the uniform record; `offsets_and_counts` is the per-cluster
/// `[offset, count]` table indexed by
/// [`ClusterGrid::linear_index`](prism_render_shading::ClusterGrid::linear_index);
/// `light_indices` is the flat index list the table slices into.  The index
/// values address the punctual-light buffer in extraction order (points first,
/// then spots), so the resolve shader can reuse the existing punctual binding
/// unchanged.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterCpuData {
    /// Grid layout + depth-slice constants for the resolve shader.
    pub grid: GpuClusterGrid,
    /// Per-cluster `[offset, count]` into [`light_indices`](Self::light_indices).
    pub offsets_and_counts: Vec<[u32; 2]>,
    /// Flattened per-cluster light indices into the punctual buffer.
    pub light_indices: Vec<u32>,
}

impl Default for ClusterCpuData {
    fn default() -> Self {
        // A single neutral cluster with no lights, so an unlit or GPU-less path
        // still has a valid, non-empty set of tables to bind.
        Self {
            grid: GpuClusterGrid::default(),
            offsets_and_counts: alloc::vec![[0, 0]],
            light_indices: Vec::new(),
        }
    }
}

impl ClusterCpuData {
    /// Total cluster entries in the offset/count table.
    pub fn cluster_count(&self) -> usize {
        self.offsets_and_counts.len()
    }

    /// Total light references recorded across every cluster.
    pub fn index_count(&self) -> usize {
        self.light_indices.len()
    }
}

/// Builds the clustered tables for one view.
///
/// `punctuals` are the extracted point/spot lights in world space (extraction
/// order); `view_from_world` maps world into the camera's view space (looking
/// down `-z`); `projection` is the camera's clip-from-view matrix; `screen_size`
/// is the render-target resolution in pixels; `near`/`far` bound the froxel
/// depth slicing.
///
/// Returns [`None`] when `projection` is singular (the froxel bounds cannot be
/// reconstructed); callers should fall back to [`ClusterCpuData::default`].
pub fn build_cluster_data(
    punctuals: &[GpuPunctualLight],
    view_from_world: &Mat4,
    projection: &Mat4,
    screen_size: [u32; 2],
    near: f32,
    far: f32,
    config: ClusterConfig,
) -> Option<ClusterCpuData> {
    let grid =
        ClusterGrid::from_tile_size(config.tile_size, screen_size, config.z_slices, near, far);

    let lights: Vec<PunctualLight> = punctuals.iter().copied().map(PunctualLight::from).collect();

    let assignment = assign_lights_to_clusters(
        &grid,
        view_from_world,
        projection,
        &lights,
        config.assignment(),
    )?;

    Some(ClusterCpuData {
        grid: GpuClusterGrid::from_grid(&grid, config.max_lights_per_cluster, *view_from_world),
        offsets_and_counts: assignment.offsets_and_counts,
        light_indices: assignment.light_indices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_shading::{ClusterBoundsBuilder, PunctualLight};

    /// A finite right-handed perspective (wgpu clip, `z in [0, 1]`), column
    /// major, matching the reference `perspective_rh_01` golden.
    fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
        let h = 1.0 / bevy_math::ops::tan(fov_y * 0.5);
        let w = h / aspect;
        let r = far / (near - far);
        [
            w,
            0.0,
            0.0,
            0.0, //
            0.0,
            h,
            0.0,
            0.0, //
            0.0,
            0.0,
            r,
            -1.0, //
            0.0,
            0.0,
            r * near,
            0.0,
        ]
    }

    fn identity() -> Mat4 {
        [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    #[test]
    fn empty_scene_builds_a_grid_with_all_zero_counts() {
        let projection = perspective(1.0, 16.0 / 9.0, 0.1, 100.0);
        let data = build_cluster_data(
            &[],
            &identity(),
            &projection,
            [1280, 720],
            0.1,
            100.0,
            ClusterConfig::default(),
        )
        .expect("finite projection");
        assert_eq!(data.cluster_count(), data.grid.cluster_count as usize);
        assert!(data.light_indices.is_empty());
        assert!(data.offsets_and_counts.iter().all(|&[_, count]| count == 0));
    }

    #[test]
    fn singular_projection_yields_none() {
        let zero: Mat4 = [0.0; 16];
        assert!(build_cluster_data(
            &[],
            &identity(),
            &zero,
            [64, 64],
            0.1,
            100.0,
            ClusterConfig::default()
        )
        .is_none());
    }

    #[test]
    fn a_point_light_at_the_origin_touches_the_near_clusters() {
        let projection = perspective(1.0, 1.0, 0.1, 100.0);
        // A bright point light a little in front of the camera, well inside the
        // frustum, must be recorded by at least one cluster.
        let light = PunctualLight::point([0.0, 0.0, -1.0], [50.0; 3], 10.0);
        let punctuals = [GpuPunctualLight::from(light)];
        let data = build_cluster_data(
            &punctuals,
            &identity(),
            &projection,
            [256, 256],
            0.1,
            100.0,
            ClusterConfig::default(),
        )
        .expect("finite projection");
        let touched: u32 = data
            .offsets_and_counts
            .iter()
            .map(|&[_, count]| count)
            .sum();
        assert!(
            touched > 0,
            "a light inside the frustum must touch a cluster"
        );
        assert!(data.light_indices.iter().all(|&i| i == 0));
    }

    #[test]
    fn packed_tables_match_the_reference_assignment_byte_for_byte() {
        let projection = perspective(1.0, 1.0, 0.1, 100.0);
        let view = identity();
        let lights = [
            PunctualLight::point([0.0, 0.0, -2.0], [30.0; 3], 8.0),
            PunctualLight::point([1.0, 0.5, -5.0], [20.0; 3], 6.0),
        ];
        let punctuals: Vec<GpuPunctualLight> =
            lights.iter().copied().map(GpuPunctualLight::from).collect();
        let config = ClusterConfig::default();
        let data = build_cluster_data(
            &punctuals,
            &view,
            &projection,
            [256, 256],
            0.1,
            100.0,
            config,
        )
        .expect("finite projection");

        // Recompute the golden directly and compare the packed tables.
        let grid =
            ClusterGrid::from_tile_size(config.tile_size, [256, 256], config.z_slices, 0.1, 100.0);
        let reference =
            assign_lights_to_clusters(&grid, &view, &projection, &lights, config.assignment())
                .expect("finite projection");
        assert_eq!(data.offsets_and_counts, reference.offsets_and_counts);
        assert_eq!(data.light_indices, reference.light_indices);
        // The builder is used, so the froxel bounds path is exercised too.
        let _ = ClusterBoundsBuilder::new(&projection).expect("finite projection");
    }

    #[test]
    fn default_data_is_a_single_empty_cluster() {
        let data = ClusterCpuData::default();
        assert_eq!(data.cluster_count(), 1);
        assert_eq!(data.index_count(), 0);
        assert_eq!(data.offsets_and_counts[0], [0, 0]);
    }
}
