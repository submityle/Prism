//! The froxel (frustum-voxel) grid: screen-space tiling in `x`/`y` and an
//! exponential depth slicing in `z`.
//!
//! Clustered forward shading (Forward+) partitions the camera frustum into a
//! 3D grid of clusters ("froxels").  Each on-screen tile spans `tile_size`
//! pixels; the depth axis is split into `dimensions.z` slices whose boundaries
//! grow exponentially so near clusters stay small and far clusters stay coarse,
//! matching the DOOM-2016 / Unreal `FForwardLightingCullingSetup` scheme.
//!
//! The exponential mapping between a view-space depth `d` (a positive distance
//! in front of the camera, i.e. `-view_z`) and its slice index `k` is
//!
//! ```text
//! k(d) = floor( dimensions.z * ln(d / near) / ln(far / near) )
//! d(k) = near * (far / near) ^ (k / dimensions.z)
//! ```
//!
//! so `d(0) == near`, `d(dimensions.z) == far`, and `k` and `d` are exact
//! inverses at the slice boundaries.  All transcendental calls go through
//! [`bevy_math::ops`] so the CPU golden stays bit-reproducible across targets.

use bevy_math::ops;

/// The froxel grid covering one camera's frustum.
///
/// The grid is fully described by its cluster [`dimensions`](Self::dimensions),
/// the pixel [`tile_size`](Self::tile_size), the render-target
/// [`screen_size`](Self::screen_size), and the `near`/`far` planes used for the
/// exponential depth slicing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClusterGrid {
    /// Cluster counts along `x`, `y`, and `z`.  Every component is at least `1`.
    pub dimensions: [u32; 3],
    /// Screen-space tile footprint in pixels along `x` and `y`.
    pub tile_size: [u32; 2],
    /// Render-target resolution in pixels.
    pub screen_size: [u32; 2],
    /// Near plane distance (`> 0`) used as the first depth-slice boundary.
    pub near: f32,
    /// Far plane distance (`> near`) used as the last depth-slice boundary.
    pub far: f32,
}

impl ClusterGrid {
    /// Builds a grid from a target `tile_size` in pixels and a `z_slices` count.
    ///
    /// The `x`/`y` cluster counts are the number of `tile_size`-pixel tiles
    /// needed to cover `screen_size` (rounded up).  All counts are clamped to
    /// at least `1`, and `near`/`far` are sanitized so `0 < near < far`.
    pub fn from_tile_size(
        tile_size: [u32; 2],
        screen_size: [u32; 2],
        z_slices: u32,
        near: f32,
        far: f32,
    ) -> Self {
        let tile_x = tile_size[0].max(1);
        let tile_y = tile_size[1].max(1);
        let screen_x = screen_size[0].max(1);
        let screen_y = screen_size[1].max(1);
        let dim_x = screen_x.div_ceil(tile_x).max(1);
        let dim_y = screen_y.div_ceil(tile_y).max(1);
        let dim_z = z_slices.max(1);
        let (near, far) = sanitize_planes(near, far);
        Self {
            dimensions: [dim_x, dim_y, dim_z],
            tile_size: [tile_x, tile_y],
            screen_size: [screen_x, screen_y],
            near,
            far,
        }
    }

    /// Builds a grid from explicit cluster `dimensions`.
    ///
    /// The pixel `tile_size` is derived so `dimensions.x * tile.x` covers the
    /// screen width (and likewise for `y`).
    pub fn new(dimensions: [u32; 3], screen_size: [u32; 2], near: f32, far: f32) -> Self {
        let dim_x = dimensions[0].max(1);
        let dim_y = dimensions[1].max(1);
        let dim_z = dimensions[2].max(1);
        let screen_x = screen_size[0].max(1);
        let screen_y = screen_size[1].max(1);
        let tile_x = screen_x.div_ceil(dim_x).max(1);
        let tile_y = screen_y.div_ceil(dim_y).max(1);
        let (near, far) = sanitize_planes(near, far);
        Self {
            dimensions: [dim_x, dim_y, dim_z],
            tile_size: [tile_x, tile_y],
            screen_size: [screen_x, screen_y],
            near,
            far,
        }
    }

    /// Total number of clusters (`dimensions.x * dimensions.y * dimensions.z`).
    pub fn cluster_count(&self) -> u32 {
        self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
    }

    /// Flattens 3D cluster `coords` into a linear index.
    ///
    /// The layout is `x + dim.x * (y + dim.y * z)`, i.e. `x` varies fastest,
    /// matching the storage the GPU offset/count table indexes into.  Callers
    /// pass in-range coordinates; out-of-range components are clamped.
    pub fn linear_index(&self, coords: [u32; 3]) -> u32 {
        let x = coords[0].min(self.dimensions[0] - 1);
        let y = coords[1].min(self.dimensions[1] - 1);
        let z = coords[2].min(self.dimensions[2] - 1);
        x + self.dimensions[0] * (y + self.dimensions[1] * z)
    }

    /// Expands a linear cluster index back into 3D `coords`.
    pub fn cluster_coords(&self, index: u32) -> [u32; 3] {
        let dim_x = self.dimensions[0];
        let dim_y = self.dimensions[1];
        let x = index % dim_x;
        let y = (index / dim_x) % dim_y;
        let z = index / (dim_x * dim_y);
        [x, y, z]
    }

    /// Maps a pixel coordinate to its `x`/`y` tile, clamped into the grid.
    pub fn xy_tile(&self, pixel: [f32; 2]) -> [u32; 2] {
        let tile_x = (pixel[0] / self.tile_size[0] as f32) as i64;
        let tile_y = (pixel[1] / self.tile_size[1] as f32) as i64;
        let clamp = |value: i64, count: u32| -> u32 { value.clamp(0, count as i64 - 1) as u32 };
        [
            clamp(tile_x, self.dimensions[0]),
            clamp(tile_y, self.dimensions[1]),
        ]
    }

    /// Reciprocal of `ln(far / near)`, the shared denominator of the slicing.
    fn depth_range_ln(&self) -> f32 {
        ops::ln(self.far / self.near)
    }

    /// Maps a view-space `z` (negative in front of the camera) to its depth
    /// slice index, clamped into `[0, dimensions.z - 1]`.
    pub fn z_slice(&self, view_z: f32) -> u32 {
        let depth = (-view_z).max(self.near);
        let factor = self.dimensions[2] as f32 / self.depth_range_ln();
        let raw = ops::ln(depth / self.near) * factor;
        let clamped = raw.max(0.0);
        (clamped as u32).min(self.dimensions[2] - 1)
    }

    /// Positive view-space depth of the `k`-th slice boundary.
    ///
    /// `slice_depth(0) == near` and `slice_depth(dimensions.z) == far`.
    pub fn slice_depth(&self, k: u32) -> f32 {
        let fraction = k as f32 / self.dimensions[2] as f32;
        self.near * ops::exp(fraction * self.depth_range_ln())
    }

    /// View-space `z` (negative) of the `k`-th slice boundary plane.
    pub fn slice_view_z(&self, k: u32) -> f32 {
        -self.slice_depth(k)
    }
}

/// Clamps the near/far planes so `0 < near < far`, guarding the logarithms.
fn sanitize_planes(near: f32, far: f32) -> (f32, f32) {
    let near = if near.is_finite() && near > 0.0 {
        near
    } else {
        0.1
    };
    let far = if far.is_finite() && far > near {
        far
    } else {
        near * 1000.0
    };
    (near, far)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_sizing_rounds_up_to_cover_the_screen() {
        let grid = ClusterGrid::from_tile_size([64, 64], [1920, 1080], 24, 0.1, 100.0);
        assert_eq!(grid.dimensions, [30, 17, 24]);
        assert!(grid.dimensions[0] * grid.tile_size[0] >= grid.screen_size[0]);
        assert!(grid.dimensions[1] * grid.tile_size[1] >= grid.screen_size[1]);
    }

    #[test]
    fn explicit_dimensions_derive_tile_size() {
        let grid = ClusterGrid::new([16, 9, 24], [1600, 900], 0.1, 100.0);
        assert_eq!(grid.tile_size, [100, 100]);
        assert_eq!(grid.cluster_count(), 16 * 9 * 24);
    }

    #[test]
    fn linear_index_round_trips_through_coords() {
        let grid = ClusterGrid::new([16, 9, 24], [1600, 900], 0.1, 100.0);
        for z in [0u32, 5, 23] {
            for y in [0u32, 4, 8] {
                for x in [0u32, 7, 15] {
                    let index = grid.linear_index([x, y, z]);
                    assert_eq!(grid.cluster_coords(index), [x, y, z]);
                }
            }
        }
    }

    #[test]
    fn slice_boundaries_are_exact_at_near_and_far() {
        let grid = ClusterGrid::new([1, 1, 24], [16, 16], 0.5, 500.0);
        assert!((grid.slice_depth(0) - 0.5).abs() < 1.0e-4);
        assert!((grid.slice_depth(24) - 500.0).abs() < 1.0e-2);
    }

    #[test]
    fn z_slice_inverts_slice_depth() {
        let grid = ClusterGrid::new([1, 1, 24], [16, 16], 0.5, 500.0);
        for k in 0..24u32 {
            let mid_depth = 0.5 * (grid.slice_depth(k) + grid.slice_depth(k + 1));
            assert_eq!(grid.z_slice(-mid_depth), k, "slice {k}");
        }
    }

    #[test]
    fn z_slice_saturates_outside_the_frustum() {
        let grid = ClusterGrid::new([1, 1, 24], [16, 16], 0.5, 500.0);
        assert_eq!(grid.z_slice(-0.01), 0);
        assert_eq!(grid.z_slice(-100000.0), 23);
    }
}
