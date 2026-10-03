//! Solid voxelization of a triangle mesh into an occupancy grid.
//!
//! Approximate convex decomposition works on a *volumetric* proxy of the input
//! rather than its raw triangles: the mesh is rasterized into an axis-aligned
//! grid of cubic cells, and every cell whose centre lies inside the (assumed
//! watertight) surface is flagged occupied. The decomposition driver then
//! reasons purely about occupied-cell sets -- measuring volume by counting
//! cells and choosing split planes along cell boundaries -- which is exactly the
//! voxel strategy popularized by the V-HACD family of cookers.
//!
//! # Inside test
//!
//! Occupancy is decided column by column. For each `(i, j)` column the vertical
//! line through the cell centres is intersected with every triangle; the
//! crossing heights are sorted, and a cell centre is inside when an odd number
//! of crossings lie below it (the standard ray-parity rule for a closed
//! surface). Boundary cases where a column grazes a shared triangle edge are
//! resolved with a fixed *top-left* fill rule applied to a canonical
//! counter-clockwise projection, so each shared edge is owned by exactly one of
//! its two triangles and the per-column crossing count stays even. The whole
//! procedure is deterministic: it depends only on index order and the fill
//! rule, never on hashing or iteration order.
//!
//! # Provenance
//!
//! Ray-parity solid voxelization and the top-left rasterization fill rule are
//! standard computer-graphics techniques. This module contains **no Unreal
//! Engine source or derived code**.

use glam::{Vec2, Vec3};

/// An axis-aligned occupancy grid of cubic cells covering a mesh's bounds.
///
/// Cell `(i, j, k)` spans `[origin + (i,j,k) * cell, origin + (i+1,j+1,k+1) *
/// cell]`; its centre sits half a cell further in. [`VoxelGrid::occupied`] is a
/// dense row-major (`x` fastest, then `y`, then `z`) boolean mask of which
/// cells fall inside the solid.
#[derive(Clone, Debug)]
pub struct VoxelGrid {
    origin: Vec3,
    cell: f32,
    dims: [usize; 3],
    occupied: Vec<bool>,
}

impl VoxelGrid {
    /// Solid-voxelizes `(vertices, triangles)` into a grid whose longest axis
    /// holds `resolution` cells.
    ///
    /// Returns [`None`] when the mesh is empty, has no triangles, is
    /// degenerate (zero-extent bounding box), or encloses no interior cell
    /// (an open or zero-volume surface), since no solid can be sampled.
    /// `resolution` is treated as at least 1.
    #[must_use]
    pub fn voxelize(
        vertices: &[Vec3],
        triangles: &[[u32; 3]],
        resolution: u32,
    ) -> Option<VoxelGrid> {
        if vertices.is_empty() || triangles.is_empty() {
            return None;
        }
        let resolution = resolution.max(1);

        let mut min = vertices[0];
        let mut max = vertices[0];
        for &v in vertices {
            min = min.min(v);
            max = max.max(v);
        }
        let extent = max - min;
        let longest = extent.max_element();
        if longest <= 0.0 || !longest.is_finite() {
            return None;
        }
        let cell = longest / resolution as f32;
        if cell <= 0.0 || !cell.is_finite() {
            return None;
        }

        let dims = [
            ((extent.x / cell).ceil() as usize).max(1),
            ((extent.y / cell).ceil() as usize).max(1),
            ((extent.z / cell).ceil() as usize).max(1),
        ];
        let origin = min;

        let mut grid = VoxelGrid {
            origin,
            cell,
            dims,
            occupied: vec![false; dims[0] * dims[1] * dims[2]],
        };
        grid.fill_from_mesh(vertices, triangles);
        // A mesh that captured no interior cell (empty, open, or a
        // zero-volume sliver) is not a solid we can decompose.
        if grid.occupied.iter().all(|&b| !b) {
            return None;
        }
        Some(grid)
    }

    /// Fills [`Self::occupied`] via per-column ray parity along the `z` axis.
    fn fill_from_mesh(&mut self, vertices: &[Vec3], triangles: &[[u32; 3]]) {
        let [nx, ny, nz] = self.dims;
        let mut crossings: Vec<f32> = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                let cx = self.origin.x + (i as f32 + 0.5) * self.cell;
                let cy = self.origin.y + (j as f32 + 0.5) * self.cell;
                crossings.clear();
                let column = Vec2::new(cx, cy);
                for tri in triangles {
                    let a = vertices[tri[0] as usize];
                    let b = vertices[tri[1] as usize];
                    let c = vertices[tri[2] as usize];
                    if let Some(z) = column_triangle_crossing(column, a, b, c) {
                        crossings.push(z);
                    }
                }
                if crossings.len() < 2 {
                    continue;
                }
                crossings.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
                for k in 0..nz {
                    let cz = self.origin.z + (k as f32 + 0.5) * self.cell;
                    let below = crossings.iter().filter(|&&z| z < cz).count();
                    if below % 2 == 1 {
                        let idx = self.linear_index(i, j, k);
                        self.occupied[idx] = true;
                    }
                }
            }
        }
    }

    /// The cell edge length.
    #[must_use]
    pub fn cell_size(&self) -> f32 {
        self.cell
    }

    /// The grid dimensions `[nx, ny, nz]` in cells.
    #[must_use]
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// The volume of a single cell (`cell^3`).
    #[must_use]
    pub fn cell_volume(&self) -> f32 {
        self.cell * self.cell * self.cell
    }

    /// Row-major linear index of cell `(i, j, k)`.
    #[must_use]
    pub fn linear_index(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.dims[1] + j) * self.dims[0] + i
    }

    /// Decomposes a linear index back into `(i, j, k)` cell coordinates.
    #[must_use]
    pub fn cell_coord(&self, linear: usize) -> [usize; 3] {
        let i = linear % self.dims[0];
        let j = (linear / self.dims[0]) % self.dims[1];
        let k = linear / (self.dims[0] * self.dims[1]);
        [i, j, k]
    }

    /// World-space centre of cell `(i, j, k)`.
    #[must_use]
    pub fn cell_center(&self, i: usize, j: usize, k: usize) -> Vec3 {
        self.origin
            + Vec3::new(
                (i as f32 + 0.5) * self.cell,
                (j as f32 + 0.5) * self.cell,
                (k as f32 + 0.5) * self.cell,
            )
    }

    /// The eight world-space corners of cell `(i, j, k)`.
    #[must_use]
    pub fn cell_corners(&self, i: usize, j: usize, k: usize) -> [Vec3; 8] {
        let base = self.origin + Vec3::new(i as f32, j as f32, k as f32) * self.cell;
        let s = self.cell;
        [
            base,
            base + Vec3::new(s, 0.0, 0.0),
            base + Vec3::new(0.0, s, 0.0),
            base + Vec3::new(s, s, 0.0),
            base + Vec3::new(0.0, 0.0, s),
            base + Vec3::new(s, 0.0, s),
            base + Vec3::new(0.0, s, s),
            base + Vec3::new(s, s, s),
        ]
    }

    /// The linear indices of every occupied cell, in ascending order.
    #[must_use]
    pub fn occupied_indices(&self) -> Vec<usize> {
        (0..self.occupied.len())
            .filter(|&idx| self.occupied[idx])
            .collect()
    }

    /// Number of occupied cells.
    #[must_use]
    pub fn occupied_count(&self) -> usize {
        self.occupied.iter().filter(|&&b| b).count()
    }
}

/// Intersects the vertical line through `column` (an `(x, y)` point) with
/// triangle `a, b, c`, returning the `z` height of the crossing when the column
/// lies inside the triangle's `xy` projection.
///
/// A fixed top-left fill rule on a canonical counter-clockwise projection
/// decides boundary ownership so that a column grazing a shared edge is counted
/// for exactly one of the two adjacent triangles. Triangles that project to a
/// (near-)degenerate sliver contribute no crossing.
fn column_triangle_crossing(column: Vec2, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    let pa = Vec2::new(a.x, a.y);
    let pb = Vec2::new(b.x, b.y);
    let pc = Vec2::new(c.x, c.y);

    let area2 = edge(pa, pb, pc);
    if area2.abs() <= f32::EPSILON {
        return None;
    }

    // Orient the triangle counter-clockwise in the projection so the top-left
    // rule is applied consistently across the whole mesh.
    let (v0, v1, v2) = if area2 > 0.0 {
        (pa, pb, pc)
    } else {
        (pa, pc, pb)
    };
    let (w0, w1, w2) = if area2 > 0.0 {
        (a.z, b.z, c.z)
    } else {
        (a.z, c.z, b.z)
    };

    let e0 = edge(v1, v2, column);
    let e1 = edge(v2, v0, column);
    let e2 = edge(v0, v1, column);

    if !covers(e0, v1, v2) || !covers(e1, v2, v0) || !covers(e2, v0, v1) {
        return None;
    }

    let area = edge(v0, v1, v2);
    let l0 = e0 / area;
    let l1 = e1 / area;
    let l2 = e2 / area;
    Some(l0 * w0 + l1 * w1 + l2 * w2)
}

/// Twice the signed area of triangle `(a, b, c)` (positive when `c` is left of
/// the directed edge `a -> b`).
fn edge(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// Whether the point is on the interior side of directed edge `a -> b`,
/// including the edge itself only when it is a top-left edge of the
/// counter-clockwise triangle (the rasterization fill rule).
fn covers(edge_value: f32, a: Vec2, b: Vec2) -> bool {
    if edge_value > 0.0 {
        return true;
    }
    if edge_value < 0.0 {
        return false;
    }
    let d = b - a;
    (d.y == 0.0 && d.x < 0.0) || d.y < 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a closed axis-aligned box mesh spanning `[min, max]` with outward
    /// counter-clockwise winding.
    fn box_mesh(min: Vec3, max: Vec3) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let v = vec![
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(max.x, min.y, min.z),
            Vec3::new(max.x, max.y, min.z),
            Vec3::new(min.x, max.y, min.z),
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(max.x, min.y, max.z),
            Vec3::new(max.x, max.y, max.z),
            Vec3::new(min.x, max.y, max.z),
        ];
        let t = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ];
        (v, t)
    }

    #[test]
    fn voxelize_box_fills_interior() {
        let (v, t) = box_mesh(Vec3::splat(-1.0), Vec3::splat(1.0));
        let grid = VoxelGrid::voxelize(&v, &t, 16).unwrap();
        // A solid cube should fill essentially every cell of the grid.
        let total: usize = grid.dims().iter().product();
        let occ = grid.occupied_count();
        assert!(occ as f32 >= 0.95 * total as f32, "occ={occ} total={total}");
    }

    #[test]
    fn voxelized_volume_approximates_true_volume() {
        let (v, t) = box_mesh(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let grid = VoxelGrid::voxelize(&v, &t, 32).unwrap();
        let vox_volume = grid.occupied_count() as f32 * grid.cell_volume();
        let true_volume = 8.0;
        assert!((vox_volume - true_volume).abs() < 0.1 * true_volume);
    }

    #[test]
    fn empty_or_degenerate_input_is_none() {
        assert!(VoxelGrid::voxelize(&[], &[], 8).is_none());
        let flat = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        assert!(VoxelGrid::voxelize(&flat, &[[0, 1, 2]], 8).is_none());
    }

    #[test]
    fn determinism_bit_for_bit() {
        let (v, t) = box_mesh(Vec3::splat(-0.5), Vec3::splat(1.5));
        let a = VoxelGrid::voxelize(&v, &t, 20).unwrap();
        let b = VoxelGrid::voxelize(&v, &t, 20).unwrap();
        assert_eq!(a.occupied, b.occupied);
        assert_eq!(a.dims(), b.dims());
    }

    #[test]
    fn index_round_trips() {
        let (v, t) = box_mesh(Vec3::splat(-1.0), Vec3::splat(1.0));
        let grid = VoxelGrid::voxelize(&v, &t, 8).unwrap();
        let [nx, ny, nz] = grid.dims();
        for k in [0, nz / 2, nz - 1] {
            for j in [0, ny / 2, ny - 1] {
                for i in [0, nx / 2, nx - 1] {
                    let lin = grid.linear_index(i, j, k);
                    assert_eq!(grid.cell_coord(lin), [i, j, k]);
                }
            }
        }
    }
}
