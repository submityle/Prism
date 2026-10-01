//! Signed distance field of a watertight mesh for the `CPU` golden path.
//!
//! A signed distance field (`SDF`) is the single most reused volumetric asset
//! in `AAA` rendering: ray-marched soft shadows and ambient occlusion step
//! along it, distance-field global illumination cones read it, mesh-distance-
//! field collision queries sample it, and smooth constructive-solid-geometry
//! blends operate directly on it. The magnitude of the field is the distance
//! to the nearest surface and the sign encodes inside (negative) versus
//! outside (positive).
//!
//! This module composes the two building blocks already in the golden path:
//! the exact Euclidean distance transform
//! ([`super::mesh_voxel_distance_field::voxel_distance_field`]) supplies the
//! unsigned magnitude, and the solid classification
//! ([`super::mesh_solid_voxelization::solidify`]) supplies the inside/outside
//! sign. The combined field is stored as a *signed squared* distance in
//! integer voxel² units (negative inside, positive outside, zero on the
//! surface), so it stays bit-for-bit reproducible; a single `sqrt` recovers the
//! world-space signed distance on demand.
//!
//! The sign is only meaningful for a watertight mesh (see
//! [`super::mesh_solid_voxelization`]); an open shell reports every empty cell
//! as outside and the field degenerates to the unsigned transform.

use super::mesh_solid_voxelization::{solidify, CellClass};
use super::mesh_voxel_distance_field::voxel_distance_field;
use super::mesh_voxelize::VoxelGrid;

/// A signed distance field sampled on a uniform voxel grid.
///
/// Produced by [`signed_distance_field`]. Each cell stores the signed squared
/// Euclidean distance to the nearest surface cell (negative inside the solid,
/// positive outside, zero on the surface) in voxel² units;
/// [`SignedDistanceField::signed_distance`] converts that to a world-space
/// signed length. Cells are addressed by integer `[x, y, z]` coordinates in
/// `0..dims[axis]`, laid out row-major with `x` varying fastest.
#[derive(Clone, Debug, PartialEq)]
pub struct SignedDistanceField {
    /// Number of cells along each axis (mirrors the source grid).
    dims: [u32; 3],
    /// Edge length of every (cubic) voxel, carried from the source grid.
    voxel_size: f32,
    /// World-space minimum corner of the grid, carried from the source grid.
    origin: [f32; 3],
    /// Row-major signed squared distance (voxel² units): `<0` inside, `>0`
    /// outside, `0` on the surface.
    signed_squared: Vec<i64>,
}

impl SignedDistanceField {
    /// Number of cells along each axis.
    pub fn dims(&self) -> [u32; 3] {
        self.dims
    }

    /// Edge length of every voxel, shared with the source grid.
    pub fn voxel_size(&self) -> f32 {
        self.voxel_size
    }

    /// World-space minimum corner of the grid, shared with the source grid.
    pub fn origin(&self) -> [f32; 3] {
        self.origin
    }

    /// Row-major signed squared distances (voxel² units), `x` varying fastest.
    pub fn signed_squared_distances(&self) -> &[i64] {
        &self.signed_squared
    }

    /// Signed squared distance (voxel² units) at `coord`, which must lie inside
    /// the grid. Negative inside the solid, positive outside, zero on surface.
    pub fn signed_squared_cells(&self, coord: [u32; 3]) -> i64 {
        self.signed_squared[linear_index(coord, self.dims)]
    }

    /// World-space signed distance at `coord` (negative inside, positive
    /// outside): the single `sqrt` of the magnitude scaled by the voxel size,
    /// carrying the stored sign.
    pub fn signed_distance(&self, coord: [u32; 3]) -> f32 {
        let signed = self.signed_squared_cells(coord);
        let magnitude = (signed.unsigned_abs() as f64).sqrt() * f64::from(self.voxel_size);
        let distance = magnitude as f32;
        if signed < 0 {
            -distance
        } else {
            distance
        }
    }

    /// `true` when `coord` lies strictly inside the solid (negative sign).
    pub fn is_inside(&self, coord: [u32; 3]) -> bool {
        self.signed_squared_cells(coord) < 0
    }
}

/// Row-major linear index of a cell coordinate (`x` varying fastest).
fn linear_index(coord: [u32; 3], dims: [u32; 3]) -> usize {
    let [x, y, z] = coord;
    ((z * dims[1] + y) * dims[0] + x) as usize
}

/// Builds the signed distance field of a watertight voxelized surface by
/// pairing the exact unsigned Euclidean distance transform with the solid
/// inside/outside classification.
pub fn signed_distance_field(grid: &VoxelGrid) -> SignedDistanceField {
    let distances = voxel_distance_field(grid);
    let solid = solidify(grid);
    let signed_squared = distances
        .squared_distances()
        .iter()
        .zip(solid.classes())
        .map(|(&squared, &class)| {
            let magnitude = squared as i64;
            match class {
                CellClass::Surface => 0,
                CellClass::Outside => magnitude,
                CellClass::Interior => -magnitude,
            }
        })
        .collect();
    SignedDistanceField {
        dims: grid.dims(),
        voxel_size: grid.voxel_size(),
        origin: grid.origin(),
        signed_squared,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::mesh_voxel_distance_field::voxel_distance_field;
    use crate::ray_scene::mesh_voxelize::voxelize_surface;
    use crate::ray_scene::triangle_mesh::TriangleMesh;

    /// Builds an index-only triangle mesh from positions (no normals/uvs).
    fn mesh(positions: Vec<[f32; 3]>, indices: Vec<[u32; 3]>) -> TriangleMesh {
        TriangleMesh::new(positions, Vec::new(), Vec::new(), indices).unwrap()
    }

    /// Closed axis-aligned unit cube (12 triangles).
    fn cube() -> TriangleMesh {
        let p = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let i = vec![
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [3, 2, 6], [3, 6, 7],
            [0, 3, 7], [0, 7, 4],
            [1, 2, 6], [1, 6, 5],
        ];
        mesh(p, i)
    }

    #[test]
    fn surface_cells_are_zero() {
        let grid = voxelize_surface(&cube(), 5).unwrap();
        let field = signed_distance_field(&grid);
        for cell in grid.occupied() {
            assert_eq!(field.signed_squared_cells(*cell), 0, "cell {cell:?}");
        }
    }

    /// Flat unit quad in the z = 0 plane; a sheet encloses nothing, so every
    /// empty cell is outside.
    fn quad() -> TriangleMesh {
        mesh(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0, 1, 2], [0, 2, 3]],
        )
    }

    #[test]
    fn interior_is_negative_and_outside_positive() {
        // A sealed cube: an inner cell is interior (negative).
        let cube_grid = voxelize_surface(&cube(), 4).unwrap();
        let cube_field = signed_distance_field(&cube_grid);
        assert!(cube_field.is_inside([1, 1, 1]), "cube inner cell must be inside");
        assert!(cube_field.signed_distance([1, 1, 1]) < 0.0);

        // A flat sheet: no interior, so every empty cell is outside (>= 0).
        let quad_grid = voxelize_surface(&quad(), 6).unwrap();
        let quad_field = signed_distance_field(&quad_grid);
        for &signed in quad_field.signed_squared_distances() {
            assert!(signed >= 0, "flat sheet should have no interior cells");
        }
    }

    #[test]
    fn magnitude_matches_unsigned_transform() {
        let grid = voxelize_surface(&cube(), 6).unwrap();
        let field = signed_distance_field(&grid);
        let unsigned = voxel_distance_field(&grid);
        let dims = field.dims();
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    let coord = [x, y, z];
                    assert_eq!(
                        field.signed_squared_cells(coord).unsigned_abs(),
                        unsigned.squared_distance_cells(coord),
                        "magnitude mismatch at {coord:?}",
                    );
                }
            }
        }
    }

    #[test]
    fn signed_distance_respects_sign_and_scale() {
        let grid = voxelize_surface(&cube(), 6).unwrap();
        let field = signed_distance_field(&grid);
        let dims = field.dims();
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    let coord = [x, y, z];
                    let signed = field.signed_squared_cells(coord);
                    let magnitude =
                        (signed.unsigned_abs() as f64).sqrt() * f64::from(field.voxel_size());
                    let expected = if signed < 0 {
                        -(magnitude as f32)
                    } else {
                        magnitude as f32
                    };
                    assert_eq!(field.signed_distance(coord), expected, "at {coord:?}");
                }
            }
        }
    }

    #[test]
    fn is_inside_only_for_interior() {
        let grid = voxelize_surface(&cube(), 4).unwrap();
        let field = signed_distance_field(&grid);
        // Surface and boundary-adjacent cells are not inside.
        assert!(!field.is_inside([0, 0, 0]), "corner is on the shell");
        assert!(field.is_inside([1, 1, 1]), "inner cell is inside");
    }

    #[test]
    fn metadata_mirrors_source_grid() {
        let grid = voxelize_surface(&cube(), 4).unwrap();
        let field = signed_distance_field(&grid);
        assert_eq!(field.dims(), grid.dims());
        assert_eq!(field.voxel_size(), grid.voxel_size());
        assert_eq!(field.origin(), grid.origin());
        let expected = (grid.dims()[0] as usize)
            * (grid.dims()[1] as usize)
            * (grid.dims()[2] as usize);
        assert_eq!(field.signed_squared_distances().len(), expected);
    }
}
