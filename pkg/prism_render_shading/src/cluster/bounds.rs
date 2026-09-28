//! View-space axis-aligned bounds for a single froxel.
//!
//! Each cluster's bounds are reconstructed by unprojecting the four screen-tile
//! corners with the camera's *inverse projection* and intersecting the four
//! resulting camera rays with the cluster's two exponential depth-slice planes.
//! The eight intersection points are reduced to a min/max box, exactly the
//! `compute_aabb_for_cluster` construction Bevy and Unreal use for froxel light
//! culling.  Working from the inverse projection keeps the bounds correct for
//! any (perspective or off-center) projection, not just a centered frustum.

use crate::shadow::math::{invert, transform_point, Mat4};

use super::grid::ClusterGrid;

/// A view-space axis-aligned bounding box for one cluster.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClusterAabb {
    /// Minimum corner in view space.
    pub min: [f32; 3],
    /// Maximum corner in view space.
    pub max: [f32; 3],
}

impl ClusterAabb {
    /// Center of the box.
    pub fn center(&self) -> [f32; 3] {
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
            0.5 * (self.min[2] + self.max[2]),
        ]
    }

    /// Half of the box extent along each axis.
    pub fn half_extents(&self) -> [f32; 3] {
        [
            0.5 * (self.max[0] - self.min[0]),
            0.5 * (self.max[1] - self.min[1]),
            0.5 * (self.max[2] - self.min[2]),
        ]
    }

    /// Radius of the sphere that tightly bounds the box (half the diagonal).
    pub fn bounding_sphere_radius(&self) -> f32 {
        let half = self.half_extents();
        (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt()
    }

    /// Squared distance from `point` to the closest point on/in the box.
    ///
    /// Returns `0` when `point` is inside the box.  This is the standard
    /// sphere-vs-AABB primitive: a sphere of radius `r` centered at `point`
    /// overlaps the box exactly when the result is `<= r * r`.
    pub fn squared_distance_to_point(&self, point: [f32; 3]) -> f32 {
        let mut total = 0.0_f32;
        for (axis, &value) in point.iter().enumerate() {
            if value < self.min[axis] {
                let delta = self.min[axis] - value;
                total += delta * delta;
            } else if value > self.max[axis] {
                let delta = value - self.max[axis];
                total += delta * delta;
            }
        }
        total
    }
}

/// Precomputes the inverse projection once so a whole grid of clusters can be
/// bounded without re-inverting per cluster.
///
/// Construction fails (returns `None`) only when `projection` is singular.
#[derive(Clone, Copy, Debug)]
pub struct ClusterBoundsBuilder {
    inverse_projection: Mat4,
}

impl ClusterBoundsBuilder {
    /// Builds a bounds builder from the camera `projection` (clip-from-view).
    pub fn new(projection: &Mat4) -> Option<Self> {
        Some(Self {
            inverse_projection: invert(projection)?,
        })
    }

    /// Builds a bounds builder from an already-inverted projection.
    pub fn from_inverse(inverse_projection: Mat4) -> Self {
        Self { inverse_projection }
    }

    /// Unprojects a normalized-device coordinate into view space.
    fn view_from_ndc(&self, ndc: [f32; 3]) -> [f32; 3] {
        let clip = transform_point(&self.inverse_projection, ndc);
        let inv_w = clip[3].recip();
        [clip[0] * inv_w, clip[1] * inv_w, clip[2] * inv_w]
    }

    /// Intersects the camera ray through `view_point` with the plane `z == z`.
    fn ray_plane(view_point: [f32; 3], plane_z: f32) -> [f32; 3] {
        // The camera sits at the view-space origin, so the ray is simply the
        // line from the origin through `view_point`.
        let t = plane_z / view_point[2];
        [view_point[0] * t, view_point[1] * t, view_point[2] * t]
    }

    /// Computes the view-space AABB for the cluster at `coords` in `grid`.
    pub fn cluster_aabb(&self, grid: &ClusterGrid, coords: [u32; 3]) -> ClusterAabb {
        let tile = grid.tile_size;
        let screen = grid.screen_size;

        let min_px = [
            (coords[0] * tile[0]) as f32,
            (coords[1] * tile[1]) as f32,
        ];
        let max_px = [
            ((coords[0] + 1) * tile[0]).min(screen[0]) as f32,
            ((coords[1] + 1) * tile[1]).min(screen[1]) as f32,
        ];

        // Pixel corners -> NDC (`x` right, `y` up, so the top pixel maps to +1).
        let ndc_x = |px: f32| px / screen[0] as f32 * 2.0 - 1.0;
        let ndc_y = |px: f32| 1.0 - px / screen[1] as f32 * 2.0;
        let corners_ndc = [
            [ndc_x(min_px[0]), ndc_y(min_px[1])],
            [ndc_x(max_px[0]), ndc_y(min_px[1])],
            [ndc_x(min_px[0]), ndc_y(max_px[1])],
            [ndc_x(max_px[0]), ndc_y(max_px[1])],
        ];

        let plane_near = grid.slice_view_z(coords[2]);
        let plane_far = grid.slice_view_z(coords[2] + 1);

        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for corner in corners_ndc {
            // Unproject at the clip far plane (`z == 1` in wgpu `[0, 1]` clip)
            // to obtain a point on the camera ray, then slide it onto each
            // depth-slice plane.
            let ray_point = self.view_from_ndc([corner[0], corner[1], 1.0]);
            for plane_z in [plane_near, plane_far] {
                let point = Self::ray_plane(ray_point, plane_z);
                for axis in 0..3 {
                    min[axis] = min[axis].min(point[axis]);
                    max[axis] = max[axis].max(point[axis]);
                }
            }
        }

        ClusterAabb { min, max }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shadow::math::perspective_rh_01;

    fn builder() -> (ClusterGrid, ClusterBoundsBuilder) {
        let grid = ClusterGrid::new([16, 16, 24], [1024, 1024], 0.1, 100.0);
        let proj = perspective_rh_01(core::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
        (grid, ClusterBoundsBuilder::new(&proj).unwrap())
    }

    #[test]
    fn cluster_z_bounds_match_the_slice_planes() {
        let (grid, builder) = builder();
        let aabb = builder.cluster_aabb(&grid, [8, 8, 3]);
        let expected_near = grid.slice_view_z(3);
        let expected_far = grid.slice_view_z(4);
        // View `z` is negative; the far slice is the more-negative bound.
        assert!((aabb.max[2] - expected_near).abs() < 1.0e-3, "{:?}", aabb);
        assert!((aabb.min[2] - expected_far).abs() < 1.0e-3, "{:?}", aabb);
    }

    #[test]
    fn central_clusters_straddle_the_optical_axis() {
        let (grid, builder) = builder();
        // With an even `x`/`y` count the two central columns straddle `x == 0`.
        let left = builder.cluster_aabb(&grid, [7, 8, 5]);
        let right = builder.cluster_aabb(&grid, [8, 8, 5]);
        assert!(left.max[0] <= 1.0e-4, "left should end at the axis: {left:?}");
        assert!(right.min[0] >= -1.0e-4, "right should start at the axis: {right:?}");
    }

    #[test]
    fn far_clusters_are_wider_than_near_clusters() {
        let (grid, builder) = builder();
        let near = builder.cluster_aabb(&grid, [0, 8, 0]);
        let far = builder.cluster_aabb(&grid, [0, 8, 20]);
        let near_width = near.max[0] - near.min[0];
        let far_width = far.max[0] - far.min[0];
        assert!(far_width > near_width, "near {near_width} far {far_width}");
    }

    #[test]
    fn squared_distance_is_zero_inside_and_positive_outside() {
        let aabb = ClusterAabb {
            min: [-1.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        };
        assert_eq!(aabb.squared_distance_to_point([0.0, 0.0, 0.0]), 0.0);
        assert!((aabb.squared_distance_to_point([3.0, 0.0, 0.0]) - 4.0).abs() < 1.0e-6);
    }

    #[test]
    fn singular_projection_has_no_builder() {
        assert!(ClusterBoundsBuilder::new(&[0.0; 16]).is_none());
    }
}
