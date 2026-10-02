//! Light-to-cluster assignment: the CPU golden that fills the froxel grid's
//! per-cluster light index table.
//!
//! For every punctual light the reference transforms its position into view
//! space, derives a conservative influence radius, and tests that bounding
//! sphere against each cluster's [`ClusterAabb`].  Spot lights add a
//! sphere-vs-cone refinement (the Bart-Wronski test Unreal uses in
//! `SpotLightCulling`) so froxels outside the cone are dropped without ever
//! culling a froxel the cone truly touches.
//!
//! The output mirrors the two GPU buffers a Forward+ resolve pass reads: a flat
//! [`light_indices`](ClusterLightAssignment::light_indices) list and a per
//! cluster `[offset, count]` table
//! ([`offsets_and_counts`](ClusterLightAssignment::offsets_and_counts)).

use alloc::vec;
use alloc::vec::Vec;

use crate::shadow::math::{dot3, normalize3, sub3, transform_direction, transform_point, Mat4};
use crate::PunctualLight;

use super::bounds::{ClusterAabb, ClusterBoundsBuilder};
use super::grid::ClusterGrid;

/// Tuning knobs for [`assign_lights_to_clusters`].
#[derive(Clone, Copy, Debug)]
pub struct ClusterAssignmentConfig {
    /// Hard cap on the lights recorded per cluster; extra lights are dropped in
    /// ascending light-index order once the cap is hit.
    pub max_lights_per_cluster: u32,
    /// Illuminance threshold used to derive a finite influence radius for a
    /// light whose `range` window is disabled (`range <= 0`).
    pub intensity_cutoff: f32,
}

impl Default for ClusterAssignmentConfig {
    fn default() -> Self {
        Self {
            max_lights_per_cluster: 256,
            intensity_cutoff: 0.01,
        }
    }
}

/// The filled froxel grid: a flat light-index list plus a per-cluster
/// `[offset, count]` table indexed by [`ClusterGrid::linear_index`].
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterLightAssignment {
    /// The grid the assignment was computed for.
    pub grid: ClusterGrid,
    /// Per cluster `[offset, count]` into [`light_indices`](Self::light_indices).
    pub offsets_and_counts: Vec<[u32; 2]>,
    /// Flattened light indices, grouped by cluster and ascending within a
    /// cluster.
    pub light_indices: Vec<u32>,
}

impl ClusterLightAssignment {
    /// Returns the light indices affecting the cluster at `coords`.
    pub fn cluster_lights(&self, coords: [u32; 3]) -> &[u32] {
        self.cluster_lights_at(self.grid.linear_index(coords))
    }

    /// Returns the light indices affecting the cluster at `linear_index`.
    pub fn cluster_lights_at(&self, linear_index: u32) -> &[u32] {
        let [offset, count] = self.offsets_and_counts[linear_index as usize];
        let start = offset as usize;
        let end = start + count as usize;
        &self.light_indices[start..end]
    }
}

/// Assigns `lights` to the froxels of `grid`.
///
/// `view_from_world` maps world space into the camera's view space (looking
/// down `-z`); `projection` is the camera's clip-from-view matrix, inverted
/// once to reconstruct froxel bounds.  Returns `None` when `projection` is
/// singular.
pub fn assign_lights_to_clusters(
    grid: &ClusterGrid,
    view_from_world: &Mat4,
    projection: &Mat4,
    lights: &[PunctualLight],
    config: ClusterAssignmentConfig,
) -> Option<ClusterLightAssignment> {
    let builder = ClusterBoundsBuilder::new(projection)?;
    let cluster_count = grid.cluster_count() as usize;

    // Precompute every froxel's view-space bounds once.
    let mut aabbs = vec![
        ClusterAabb {
            min: [0.0; 3],
            max: [0.0; 3],
        };
        cluster_count
    ];
    for z in 0..grid.dimensions[2] {
        for y in 0..grid.dimensions[1] {
            for x in 0..grid.dimensions[0] {
                let linear = grid.linear_index([x, y, z]) as usize;
                aabbs[linear] = builder.cluster_aabb(grid, [x, y, z]);
            }
        }
    }

    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); cluster_count];

    for (light_index, light) in lights.iter().enumerate() {
        let radius = effective_radius(light, config.intensity_cutoff);
        if radius <= 0.0 {
            continue;
        }
        let radius_sq = radius * radius;

        let view_position = transform_to_view(view_from_world, light.position);
        let is_spot = light.spot_scale != 0.0;
        let (cone_axis, cos_half, sin_half) = if is_spot {
            let axis = normalize3(transform_direction(view_from_world, light.direction));
            let cos_half = (-light.spot_offset / light.spot_scale).clamp(-1.0, 1.0);
            let sin_half = (1.0 - cos_half * cos_half).max(0.0).sqrt();
            (axis, cos_half, sin_half)
        } else {
            ([0.0, 0.0, 0.0], 0.0, 0.0)
        };

        for (linear, aabb) in aabbs.iter().enumerate() {
            if aabb.squared_distance_to_point(view_position) > radius_sq {
                continue;
            }
            if is_spot
                && !sphere_intersects_cone(
                    view_position,
                    cone_axis,
                    cos_half,
                    sin_half,
                    radius,
                    aabb.center(),
                    aabb.bounding_sphere_radius(),
                )
            {
                continue;
            }
            let bucket = &mut buckets[linear];
            if (bucket.len() as u32) < config.max_lights_per_cluster {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "Light counts are bounded far below u32::MAX by the cluster cap."
                )]
                bucket.push(light_index as u32);
            }
        }
    }

    let total: usize = buckets.iter().map(Vec::len).sum();
    let mut offsets_and_counts = Vec::with_capacity(cluster_count);
    let mut light_indices = Vec::with_capacity(total);
    for bucket in &buckets {
        let offset = light_indices.len() as u32;
        offsets_and_counts.push([offset, bucket.len() as u32]);
        light_indices.extend_from_slice(bucket);
    }

    Some(ClusterLightAssignment {
        grid: *grid,
        offsets_and_counts,
        light_indices,
    })
}

/// Conservative influence radius: the light's `range` window when present, else
/// the inverse-square distance at which its brightest channel dims below
/// `intensity_cutoff`.
fn effective_radius(light: &PunctualLight, intensity_cutoff: f32) -> f32 {
    if light.range.is_finite() && light.range > 0.0 {
        return light.range;
    }
    let max_intensity = light.intensity.iter().copied().fold(0.0_f32, f32::max);
    if max_intensity <= 0.0 {
        return 0.0;
    }
    (max_intensity / intensity_cutoff.max(1.0e-6)).sqrt()
}

/// Transforms a world-space point into view space (with a perspective divide
/// that is a no-op for the affine view matrix but stays robust regardless).
fn transform_to_view(view_from_world: &Mat4, point: [f32; 3]) -> [f32; 3] {
    let view = transform_point(view_from_world, point);
    let inv_w = if view[3] != 0.0 { view[3].recip() } else { 1.0 };
    [view[0] * inv_w, view[1] * inv_w, view[2] * inv_w]
}

/// Tests whether a sphere intersects a finite (range-capped) cone.
///
/// `apex`/`axis` describe the cone (unit `axis`), `cos_half`/`sin_half` its
/// half-angle, and `range` its length.  Returns `true` when the sphere of
/// `radius` around `center` touches the cone volume.  Fed a froxel's bounding
/// sphere this is conservative: it never rejects a froxel the cone reaches.
fn sphere_intersects_cone(
    apex: [f32; 3],
    axis: [f32; 3],
    cos_half: f32,
    sin_half: f32,
    range: f32,
    center: [f32; 3],
    radius: f32,
) -> bool {
    let v = sub3(center, apex);
    let axial = dot3(v, axis);
    // Reject spheres beyond the cone's far cap or fully behind the apex.
    if axial > range + radius || axial < -radius {
        return false;
    }
    let v_len_sq = dot3(v, v);
    let perpendicular = (v_len_sq - axial * axial).max(0.0).sqrt();
    let distance_to_surface = cos_half * perpendicular - axial * sin_half;
    distance_to_surface <= radius
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shadow::math::perspective_rh_01;

    fn identity() -> Mat4 {
        let mut m = [0.0; 16];
        m[0] = 1.0;
        m[5] = 1.0;
        m[10] = 1.0;
        m[15] = 1.0;
        m
    }

    fn setup() -> (ClusterGrid, Mat4, Mat4) {
        let grid = ClusterGrid::new([16, 16, 24], [1024, 1024], 0.1, 100.0);
        let projection = perspective_rh_01(core::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
        (grid, identity(), projection)
    }

    fn totals(assignment: &ClusterLightAssignment) -> usize {
        assignment
            .offsets_and_counts
            .iter()
            .map(|entry| entry[1] as usize)
            .sum()
    }

    #[test]
    fn in_frustum_point_light_touches_some_clusters() {
        let (grid, view, proj) = setup();
        let light = PunctualLight::point([0.0, 0.0, -10.0], [50.0; 3], 4.0);
        let assignment = assign_lights_to_clusters(
            &grid,
            &view,
            &proj,
            &[light],
            ClusterAssignmentConfig::default(),
        )
        .unwrap();
        assert!(totals(&assignment) > 0);
    }

    #[test]
    fn light_behind_the_camera_is_culled_everywhere() {
        let (grid, view, proj) = setup();
        // Positive `z` is behind a camera looking down `-z`.
        let light = PunctualLight::point([0.0, 0.0, 12.0], [50.0; 3], 5.0);
        let assignment = assign_lights_to_clusters(
            &grid,
            &view,
            &proj,
            &[light],
            ClusterAssignmentConfig::default(),
        )
        .unwrap();
        assert_eq!(totals(&assignment), 0);
        assert!(assignment.light_indices.is_empty());
    }

    #[test]
    fn bigger_range_reaches_at_least_as_many_clusters() {
        let (grid, view, proj) = setup();
        let small = PunctualLight::point([0.0, 0.0, -10.0], [50.0; 3], 2.0);
        let large = PunctualLight::point([0.0, 0.0, -10.0], [50.0; 3], 20.0);
        let small_total = totals(
            &assign_lights_to_clusters(
                &grid,
                &view,
                &proj,
                &[small],
                ClusterAssignmentConfig::default(),
            )
            .unwrap(),
        );
        let large_total = totals(
            &assign_lights_to_clusters(
                &grid,
                &view,
                &proj,
                &[large],
                ClusterAssignmentConfig::default(),
            )
            .unwrap(),
        );
        assert!(
            large_total > small_total,
            "small {small_total} large {large_total}"
        );
    }

    #[test]
    fn spot_cone_culls_more_than_an_equivalent_point_light() {
        let (grid, view, proj) = setup();
        let position = [0.0, 0.0, -10.0];
        let range = 30.0;
        let point = PunctualLight::point(position, [200.0; 3], range);
        // Narrow cone pointing further down `-z`.
        let spot = PunctualLight::spot(position, [200.0; 3], range, [0.0, 0.0, -1.0], 0.98, 0.95);
        let point_total = totals(
            &assign_lights_to_clusters(
                &grid,
                &view,
                &proj,
                &[point],
                ClusterAssignmentConfig::default(),
            )
            .unwrap(),
        );
        let spot_total = totals(
            &assign_lights_to_clusters(
                &grid,
                &view,
                &proj,
                &[spot],
                ClusterAssignmentConfig::default(),
            )
            .unwrap(),
        );
        assert!(spot_total > 0, "spot should still light something");
        assert!(
            spot_total < point_total,
            "cone must cull froxels: spot {spot_total} point {point_total}"
        );
    }

    #[test]
    fn per_cluster_cap_is_respected() {
        let (grid, view, proj) = setup();
        // Five overlapping wide-range lights covering the whole frustum.
        let lights: Vec<PunctualLight> = (0..5)
            .map(|_| PunctualLight::point([0.0, 0.0, -10.0], [500.0; 3], 500.0))
            .collect();
        let config = ClusterAssignmentConfig {
            max_lights_per_cluster: 2,
            ..ClusterAssignmentConfig::default()
        };
        let assignment = assign_lights_to_clusters(&grid, &view, &proj, &lights, config).unwrap();
        for entry in &assignment.offsets_and_counts {
            assert!(entry[1] <= 2, "count {} exceeds cap", entry[1]);
        }
    }

    #[test]
    fn offsets_are_contiguous_and_slices_match_counts() {
        let (grid, view, proj) = setup();
        let lights = [
            PunctualLight::point([1.0, 0.0, -8.0], [80.0; 3], 6.0),
            PunctualLight::point([-2.0, 1.0, -14.0], [80.0; 3], 8.0),
        ];
        let assignment = assign_lights_to_clusters(
            &grid,
            &view,
            &proj,
            &lights,
            ClusterAssignmentConfig::default(),
        )
        .unwrap();

        let mut running = 0u32;
        for (linear, entry) in assignment.offsets_and_counts.iter().enumerate() {
            assert_eq!(entry[0], running, "cluster {linear} offset");
            let slice = assignment.cluster_lights_at(linear as u32);
            assert_eq!(slice.len() as u32, entry[1]);
            // Indices ascend within a cluster and stay in range.
            for pair in slice.windows(2) {
                assert!(pair[0] < pair[1]);
            }
            assert!(slice.iter().all(|&i| (i as usize) < lights.len()));
            running += entry[1];
        }
        assert_eq!(running as usize, assignment.light_indices.len());
    }

    #[test]
    fn singular_projection_yields_no_assignment() {
        let grid = ClusterGrid::new([4, 4, 4], [64, 64], 0.1, 100.0);
        let result = assign_lights_to_clusters(
            &grid,
            &identity(),
            &[0.0; 16],
            &[],
            ClusterAssignmentConfig::default(),
        );
        assert!(result.is_none());
    }
}
