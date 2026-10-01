//! Solid (inside/outside) voxelization of a watertight surface for the `CPU`
//! golden path.
//!
//! Surface voxelization ([`super::mesh_voxelize::voxelize_surface`]) tells us
//! which cells a mesh's *shell* touches; many `AAA` techniques instead need to
//! know which cells are *inside* the solid: voxel-cone-traced `GI` integrates
//! irradiance over solid occupancy, physics bakes solid collision proxies, and
//! a signed distance field needs an inside/outside sign to pair with the
//! unsigned magnitude from [`super::mesh_voxel_distance_field`].
//!
//! Because a conservative (26-separating) surface voxelization seals the shell
//! against any 6-connected path, the exterior can be found with a flood fill:
//! starting from every non-surface cell on the grid boundary, a 6-connected
//! sweep marks all reachable empty cells as [`CellClass::Outside`]. Whatever is
//! left unreached and unoccupied is enclosed by the shell and therefore
//! [`CellClass::Interior`]; occupied cells stay [`CellClass::Surface`]. The
//! classification is pure integer graph traversal — no floating-point and no
//! transcendental function — so it is bit-for-bit reproducible.
//!
//! This assumes the source mesh is watertight; an open shell lets the flood
//! fill leak inside and simply reports more [`CellClass::Outside`] cells.

use super::mesh_voxelize::VoxelGrid;

/// Inside/outside classification of a single voxel cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellClass {
    /// Empty cell reachable from the grid boundary (outside the solid).
    Outside,
    /// Occupied cell on the mesh surface shell.
    Surface,
    /// Empty cell enclosed by the shell (inside the solid).
    Interior,
}

/// A solid voxelization: every grid cell tagged inside, outside, or on surface.
///
/// Produced by [`solidify`]. Cells are addressed by integer `[x, y, z]`
/// coordinates in `0..dims[axis]`, laid out row-major with `x` varying fastest,
/// matching the source [`VoxelGrid`].
#[derive(Clone, Debug, PartialEq)]
pub struct SolidVoxelization {
    /// Number of cells along each axis (mirrors the source grid).
    dims: [u32; 3],
    /// Edge length of every (cubic) voxel, carried from the source grid.
    voxel_size: f32,
    /// World-space minimum corner of the grid, carried from the source grid.
    origin: [f32; 3],
    /// Row-major per-cell classification, `x` varying fastest.
    classes: Vec<CellClass>,
}

impl SolidVoxelization {
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

    /// Row-major per-cell classification, `x` varying fastest.
    pub fn classes(&self) -> &[CellClass] {
        &self.classes
    }

    /// Classification of the cell at `coord`, which must lie inside the grid.
    pub fn classify(&self, coord: [u32; 3]) -> CellClass {
        self.classes[linear_index(coord, self.dims)]
    }

    /// Number of cells on the surface shell.
    pub fn surface_count(&self) -> usize {
        self.count(CellClass::Surface)
    }

    /// Number of cells enclosed by the shell (inside the solid).
    pub fn interior_count(&self) -> usize {
        self.count(CellClass::Interior)
    }

    /// Number of empty cells reachable from the grid boundary.
    pub fn outside_count(&self) -> usize {
        self.count(CellClass::Outside)
    }

    /// Counts cells matching `class`.
    fn count(&self, class: CellClass) -> usize {
        self.classes.iter().filter(|&&c| c == class).count()
    }
}

/// Row-major linear index of a cell coordinate (`x` varying fastest).
fn linear_index(coord: [u32; 3], dims: [u32; 3]) -> usize {
    let [x, y, z] = coord;
    ((z * dims[1] + y) * dims[0] + x) as usize
}

/// Reconstructs the `[x, y, z]` coordinate of a row-major linear index.
fn cell_coord(index: usize, dims: [u32; 3]) -> [u32; 3] {
    let dx = dims[0] as usize;
    let dy = dims[1] as usize;
    let x = index % dx;
    let y = (index / dx) % dy;
    let z = index / (dx * dy);
    [x as u32, y as u32, z as u32]
}

/// `true` when the cell sits on the outer boundary of the grid.
fn is_boundary(coord: [u32; 3], dims: [u32; 3]) -> bool {
    (0..3).any(|axis| coord[axis] == 0 || coord[axis] + 1 == dims[axis])
}

/// Classifies every cell of a surface [`VoxelGrid`] as inside, outside, or on
/// the surface shell via a 6-connected exterior flood fill.
pub fn solidify(grid: &VoxelGrid) -> SolidVoxelization {
    let dims = grid.dims();
    let count = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);
    // Empty cells start Interior; the exterior flood fill demotes the ones it
    // can reach to Outside.
    let mut classes = vec![CellClass::Interior; count];
    for cell in grid.occupied() {
        classes[linear_index(*cell, dims)] = CellClass::Surface;
    }

    // Seed the flood fill from every empty cell on the grid boundary.
    let mut stack: Vec<usize> = Vec::new();
    for (index, class) in classes.iter_mut().enumerate() {
        if *class == CellClass::Interior && is_boundary(cell_coord(index, dims), dims) {
            *class = CellClass::Outside;
            stack.push(index);
        }
    }

    // 6-connected sweep: any empty neighbour of an outside cell is outside too.
    while let Some(index) = stack.pop() {
        let coord = cell_coord(index, dims);
        for axis in 0..3 {
            for step in [-1i64, 1] {
                let value = coord[axis] as i64 + step;
                if value < 0 || value >= i64::from(dims[axis]) {
                    continue;
                }
                let mut neighbour = coord;
                neighbour[axis] = value as u32;
                let n_index = linear_index(neighbour, dims);
                if classes[n_index] == CellClass::Interior {
                    classes[n_index] = CellClass::Outside;
                    stack.push(n_index);
                }
            }
        }
    }

    SolidVoxelization {
        dims,
        voxel_size: grid.voxel_size(),
        origin: grid.origin(),
        classes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::mesh_voxelize::voxelize_surface;
    use crate::ray_scene::triangle_mesh::TriangleMesh;

    /// Builds an index-only triangle mesh from positions (no normals/uvs).
    fn mesh(positions: Vec<[f32; 3]>, indices: Vec<[u32; 3]>) -> TriangleMesh {
        TriangleMesh::new(positions, Vec::new(), Vec::new(), indices).unwrap()
    }

    fn quad() -> TriangleMesh {
        mesh(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0, 1, 2], [0, 2, 3]],
        )
    }

    /// Closed axis-aligned unit cube (12 triangles, outward winding not needed
    /// for occupancy).
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
            [0, 1, 2], [0, 2, 3], // z = 0
            [4, 5, 6], [4, 6, 7], // z = 1
            [0, 1, 5], [0, 5, 4], // y = 0
            [3, 2, 6], [3, 6, 7], // y = 1
            [0, 3, 7], [0, 7, 4], // x = 0
            [1, 2, 6], [1, 6, 5], // x = 1
        ];
        mesh(p, i)
    }

    #[test]
    fn flat_quad_has_no_interior() {
        let grid = voxelize_surface(&quad(), 4).unwrap();
        let solid = solidify(&grid);
        assert!(solid.surface_count() > 0);
        assert_eq!(solid.interior_count(), 0, "a flat sheet encloses nothing");
    }

    #[test]
    fn closed_cube_has_an_interior() {
        let grid = voxelize_surface(&cube(), 4).unwrap();
        let solid = solidify(&grid);
        assert!(solid.interior_count() > 0, "sealed cube must enclose cells");
        // A clearly inner cell is interior; a shell corner is surface.
        assert_eq!(solid.classify([1, 1, 1]), CellClass::Interior);
        assert_eq!(solid.classify([0, 0, 0]), CellClass::Surface);
    }

    #[test]
    fn occupied_cells_are_surface() {
        let grid = voxelize_surface(&cube(), 5).unwrap();
        let solid = solidify(&grid);
        for cell in grid.occupied() {
            assert_eq!(solid.classify(*cell), CellClass::Surface, "cell {cell:?}");
        }
    }

    #[test]
    fn classes_partition_every_cell() {
        let grid = voxelize_surface(&cube(), 5).unwrap();
        let solid = solidify(&grid);
        let total = (solid.dims()[0] as usize)
            * (solid.dims()[1] as usize)
            * (solid.dims()[2] as usize);
        assert_eq!(solid.classes().len(), total);
        assert_eq!(
            solid.surface_count() + solid.interior_count() + solid.outside_count(),
            total,
        );
    }

    #[test]
    fn boundary_empty_cells_are_outside() {
        let grid = voxelize_surface(&quad(), 6).unwrap();
        let solid = solidify(&grid);
        let dims = solid.dims();
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    let coord = [x, y, z];
                    if is_boundary(coord, dims)
                        && solid.classify(coord) != CellClass::Surface
                    {
                        assert_eq!(
                            solid.classify(coord),
                            CellClass::Outside,
                            "boundary cell {coord:?} must be outside",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn metadata_mirrors_source_grid() {
        let grid = voxelize_surface(&cube(), 4).unwrap();
        let solid = solidify(&grid);
        assert_eq!(solid.dims(), grid.dims());
        assert_eq!(solid.voxel_size(), grid.voxel_size());
        assert_eq!(solid.origin(), grid.origin());
    }
}
