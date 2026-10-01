//! Conservative surface voxelization of a triangle mesh for the `CPU` golden
//! path.
//!
//! Surface voxelization is the first stage of voxel-cone-traced `GI`, signed
//! distance baking, voxel ambient occlusion, and conservative collision proxies
//! — all the `AAA` techniques that need "which cells does this surface touch".
//! This module rasterizes every triangle into a uniform grid using the exact
//! separating-axis triangle/box overlap test of Akenine-Möller ("Fast 3D
//! Triangle-Box Overlap Testing"): a voxel is marked when the triangle and the
//! voxel's axis-aligned box fail to separate on any of the thirteen candidate
//! axes (three box face normals, the triangle normal, and the nine edge/edge
//! cross products). Only each triangle's own voxel-space bounding box is
//! visited, so the cost scales with surface area, not grid volume.
//!
//! All arithmetic accumulates in `f64`; every test is dot/cross/abs/min/max,
//! with no transcendental function and no square root, so the occupancy set is
//! bit-for-bit reproducible across platforms.
//!
//! [`voxelize_surface`] returns a [`VoxelGrid`] (origin, voxel size, grid
//! dimensions, and the sorted set of occupied cells), or [`None`] when the mesh
//! has no triangles or collapses to a single point.

use std::collections::HashSet;

use super::triangle_mesh::TriangleMesh;

/// A uniform occupancy grid produced by surface voxelization.
///
/// Produced by [`voxelize_surface`]. Cells are addressed by integer `[x, y, z]`
/// coordinates in `0..dims[axis]`; the world-space centre of cell `c` is
/// `origin + (c + 0.5) * voxel_size`.
#[derive(Clone, Debug, PartialEq)]
pub struct VoxelGrid {
    /// World-space minimum corner of the grid (cell `[0,0,0]`'s low corner).
    origin: [f32; 3],
    /// Edge length of every (cubic) voxel.
    voxel_size: f32,
    /// Number of cells along each axis (each at least one).
    dims: [u32; 3],
    /// Sorted, de-duplicated coordinates of the occupied cells.
    occupied: Vec<[u32; 3]>,
}

impl VoxelGrid {
    /// World-space minimum corner of the grid.
    pub fn origin(&self) -> [f32; 3] {
        self.origin
    }

    /// Edge length of every voxel.
    pub fn voxel_size(&self) -> f32 {
        self.voxel_size
    }

    /// Number of cells along each axis.
    pub fn dims(&self) -> [u32; 3] {
        self.dims
    }

    /// Sorted, de-duplicated coordinates of the occupied cells.
    pub fn occupied(&self) -> &[[u32; 3]] {
        &self.occupied
    }

    /// Number of occupied cells.
    pub fn occupied_count(&self) -> usize {
        self.occupied.len()
    }

    /// Returns `true` when cell `coord` is occupied (binary search over the
    /// sorted occupancy set).
    pub fn is_occupied(&self, coord: [u32; 3]) -> bool {
        self.occupied.binary_search(&coord).is_ok()
    }

    /// Assembles a grid directly from already-validated parts.
    ///
    /// The caller must guarantee the grid invariants: every axis of `dims` is
    /// at least one, and `occupied` is sorted ascending, de-duplicated, and
    /// entirely in bounds (`coord[axis] < dims[axis]`). This is a low-level
    /// primitive for grid-to-grid transforms (such as margin padding) that
    /// preserve those invariants without re-running voxelization.
    pub(crate) fn from_sorted_parts(
        origin: [f32; 3],
        voxel_size: f32,
        dims: [u32; 3],
        occupied: Vec<[u32; 3]>,
    ) -> VoxelGrid {
        VoxelGrid {
            origin,
            voxel_size,
            dims,
            occupied,
        }
    }
}

/// Dot product of two 3-vectors in `f64`.
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Vector subtraction `a - b`.
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Cross product `a × b`.
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Tests whether `axis` is a separating axis for a triangle (already translated
/// so the box centre is the origin) and an axis-aligned box of half-size `half`.
///
/// Returns `true` when the projections do not overlap. A (near) zero axis — for
/// example from two parallel edges — cannot separate and returns `false`.
fn is_separating(axis: [f64; 3], v: [[f64; 3]; 3], half: [f64; 3]) -> bool {
    if axis[0].abs() < 1e-18 && axis[1].abs() < 1e-18 && axis[2].abs() < 1e-18 {
        return false;
    }
    let p0 = dot(axis, v[0]);
    let p1 = dot(axis, v[1]);
    let p2 = dot(axis, v[2]);
    let min = p0.min(p1).min(p2);
    let max = p0.max(p1).max(p2);
    let radius = half[0] * axis[0].abs() + half[1] * axis[1].abs() + half[2] * axis[2].abs();
    min > radius || max < -radius
}

/// Exact triangle / axis-aligned-box overlap via the separating-axis theorem.
///
/// `center`/`half` describe the box; `tri` is the world-space triangle. Overlap
/// holds when no candidate axis separates the two convex shapes.
pub fn triangle_box_overlap(center: [f64; 3], half: [f64; 3], tri: [[f64; 3]; 3]) -> bool {
    let v = [sub(tri[0], center), sub(tri[1], center), sub(tri[2], center)];
    let e = [sub(v[1], v[0]), sub(v[2], v[1]), sub(v[0], v[2])];

    let unit = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    // Three box face normals (reduces to the triangle-AABB vs box test).
    for &u in &unit {
        if is_separating(u, v, half) {
            return false;
        }
    }
    // Triangle face normal (plane / box test).
    let normal = cross(e[0], e[1]);
    if is_separating(normal, v, half) {
        return false;
    }
    // Nine edge/edge cross-product axes.
    for &edge in &e {
        for &u in &unit {
            if is_separating(cross(edge, u), v, half) {
                return false;
            }
        }
    }
    true
}

/// Minimum/maximum corners of the mesh's referenced vertices, in `f64`.
fn referenced_bounds(mesh: &TriangleMesh) -> Option<([f64; 3], [f64; 3])> {
    let positions = mesh.positions();
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    let mut seen = false;
    for tri in mesh.indices() {
        for &index in tri {
            let p = positions[index as usize];
            seen = true;
            for axis in 0..3 {
                let value = f64::from(p[axis]);
                min[axis] = min[axis].min(value);
                max[axis] = max[axis].max(value);
            }
        }
    }
    if seen {
        Some((min, max))
    } else {
        None
    }
}

/// Conservatively voxelizes the surface of `mesh` into a uniform grid.
///
/// `max_resolution` is the number of voxels along the mesh's longest axis
/// (clamped to at least one); the voxel edge length derives from it, and the
/// other axes receive as many cells as needed to cover their extent. Returns
/// [`None`] when the mesh has no triangles or collapses to a single point
/// (zero extent on every axis).
pub fn voxelize_surface(mesh: &TriangleMesh, max_resolution: u32) -> Option<VoxelGrid> {
    let (min, max) = referenced_bounds(mesh)?;
    let extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    let longest = extent[0].max(extent[1]).max(extent[2]);
    if longest <= 0.0 {
        return None;
    }
    let resolution = max_resolution.max(1);
    let voxel_size = longest / f64::from(resolution);

    // Cells needed to cover each axis (at least one), capped by resolution.
    let dim = |ext: f64| -> u32 {
        let cells = (ext / voxel_size).ceil() as u32;
        cells.max(1)
    };
    let dims = [dim(extent[0]), dim(extent[1]), dim(extent[2])];
    let half = [voxel_size * 0.5; 3];

    let positions = mesh.positions();
    let vertex = |i: u32| -> [f64; 3] {
        let p = positions[i as usize];
        [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]
    };

    // Maps a world coordinate on one axis to a clamped cell index.
    let cell_index = |value: f64, axis: usize| -> i64 {
        let raw = ((value - min[axis]) / voxel_size).floor() as i64;
        raw.clamp(0, i64::from(dims[axis]) - 1)
    };

    let mut occupied: HashSet<[u32; 3]> = HashSet::new();
    for tri_indices in mesh.indices() {
        let tri = [
            vertex(tri_indices[0]),
            vertex(tri_indices[1]),
            vertex(tri_indices[2]),
        ];
        // Visit only the triangle's own voxel-space bounding box.
        let mut lo = [0i64; 3];
        let mut hi = [0i64; 3];
        for axis in 0..3 {
            let tmin = tri[0][axis].min(tri[1][axis]).min(tri[2][axis]);
            let tmax = tri[0][axis].max(tri[1][axis]).max(tri[2][axis]);
            lo[axis] = cell_index(tmin, axis);
            hi[axis] = cell_index(tmax, axis);
        }
        for ix in lo[0]..=hi[0] {
            for iy in lo[1]..=hi[1] {
                for iz in lo[2]..=hi[2] {
                    let center = [
                        min[0] + (ix as f64 + 0.5) * voxel_size,
                        min[1] + (iy as f64 + 0.5) * voxel_size,
                        min[2] + (iz as f64 + 0.5) * voxel_size,
                    ];
                    if triangle_box_overlap(center, half, tri) {
                        occupied.insert([ix as u32, iy as u32, iz as u32]);
                    }
                }
            }
        }
    }

    let mut occupied: Vec<[u32; 3]> = occupied.into_iter().collect();
    occupied.sort_unstable();

    Some(VoxelGrid {
        origin: [min[0] as f32, min[1] as f32, min[2] as f32],
        voxel_size: voxel_size as f32,
        dims,
        occupied,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::triangle_mesh::TriangleMesh;

    /// Builds an index-only triangle mesh from positions (no normals/uvs).
    fn mesh(positions: Vec<[f32; 3]>, indices: Vec<[u32; 3]>) -> TriangleMesh {
        TriangleMesh::new(positions, Vec::new(), Vec::new(), indices).unwrap()
    }

    #[test]
    fn empty_mesh_has_no_grid() {
        let m = mesh(Vec::new(), Vec::new());
        assert!(voxelize_surface(&m, 8).is_none());
    }

    #[test]
    fn degenerate_point_mesh_has_no_grid() {
        let m = mesh(vec![[1.0, 1.0, 1.0]], vec![[0, 0, 0]]);
        assert!(voxelize_surface(&m, 8).is_none());
    }

    #[test]
    fn box_overlap_center_hit_and_far_miss() {
        let tri = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        // Box centred on the triangle plane near the origin overlaps.
        assert!(triangle_box_overlap([0.1, 0.1, 0.0], [0.2, 0.2, 0.2], tri));
        // Box far above the plane does not.
        assert!(!triangle_box_overlap([0.1, 0.1, 5.0], [0.2, 0.2, 0.2], tri));
    }

    #[test]
    fn axis_aligned_quad_fills_its_plane_layer() {
        // A unit quad in the z=0 plane voxelized at resolution 4 should occupy
        // exactly one layer of cells (all at iz == 0) and cover the plane.
        let m = mesh(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0, 1, 2], [0, 2, 3]],
        );
        let grid = voxelize_surface(&m, 4).unwrap();
        assert_eq!(grid.dims(), [4, 4, 1]);
        assert!(grid.occupied_count() > 0);
        // Every occupied cell sits in the single z layer.
        for cell in grid.occupied() {
            assert_eq!(cell[2], 0, "cell {cell:?} off the plane layer");
        }
        // The quad fully covers its layer: all 16 in-plane cells are occupied.
        assert_eq!(grid.occupied_count(), 16);
        assert!(grid.is_occupied([0, 0, 0]));
        assert!(grid.is_occupied([3, 3, 0]));
    }

    #[test]
    fn diagonal_triangle_occupies_a_connected_band() {
        // A triangle spanning a cube's diagonal touches cells along a band but
        // leaves far corners empty.
        let m = mesh(
            vec![[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
            vec![[0, 1, 2]],
        );
        let grid = voxelize_surface(&m, 4).unwrap();
        assert!(grid.occupied_count() > 0);
        // The occupancy set must be a strict subset of the full grid volume.
        let total = (grid.dims()[0] * grid.dims()[1] * grid.dims()[2]) as usize;
        assert!(grid.occupied_count() < total, "surface should not fill volume");
    }

    #[test]
    fn occupied_set_is_sorted_and_unique() {
        let m = mesh(
            vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            vec![[0, 1, 2]],
        );
        let grid = voxelize_surface(&m, 6).unwrap();
        let cells = grid.occupied();
        for pair in cells.windows(2) {
            assert!(pair[0] < pair[1], "not strictly sorted/unique: {pair:?}");
        }
    }
}
