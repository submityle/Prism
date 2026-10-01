//! Exact Euclidean distance transform of a voxelized surface for the `CPU`
//! golden path.
//!
//! Signed/unsigned distance fields are the backbone of `AAA` volumetric
//! techniques: voxel-cone-traced `GI` leaks less with a distance field to bias
//! cones, screen-independent soft shadows march a distance field, baked `SDF`
//! collision proxies query nearest-surface distance, and voxel ambient
//! occlusion weights occupancy by proximity. This module turns the occupancy
//! set of a [`VoxelGrid`] into the per-cell distance to the nearest occupied
//! (surface) cell.
//!
//! The transform is the separable exact Euclidean distance transform of
//! Felzenszwalb and Huttenlocher ("Distance Transforms of Sampled Functions").
//! Each axis is swept independently with a one-dimensional lower-envelope pass
//! over parabolas, so the whole grid is transformed in `O(cells)` time while
//! producing the *exact* squared Euclidean distance — not a chamfer
//! approximation. Squared distances are accumulated as integer voxel² counts,
//! so the field is bit-for-bit reproducible across platforms; the only
//! floating-point arithmetic is the parabola-intersection comparison (which
//! never feeds back into the stored integers) and the final `sqrt` used to
//! report a world-space distance.
//!
//! [`voxel_distance_field`] consumes a [`VoxelGrid`] (which always has at least
//! one occupied cell, since [`super::mesh_voxelize::voxelize_surface`] returns
//! `None` otherwise) and yields a [`VoxelDistanceField`].

use super::mesh_voxelize::VoxelGrid;

/// An exact Euclidean distance field sampled on a uniform voxel grid.
///
/// Produced by [`voxel_distance_field`]. Every cell stores the squared
/// Euclidean distance (in voxel² units) to the nearest occupied cell of the
/// source [`VoxelGrid`]; [`VoxelDistanceField::distance`] converts that to a
/// world-space length. Cells are addressed by integer `[x, y, z]` coordinates
/// in `0..dims[axis]`, laid out row-major with `x` varying fastest.
#[derive(Clone, Debug, PartialEq)]
pub struct VoxelDistanceField {
    /// Number of cells along each axis (mirrors the source grid).
    dims: [u32; 3],
    /// Edge length of every (cubic) voxel, carried for world-space conversion.
    voxel_size: f32,
    /// Row-major squared distance (voxel² units) to the nearest occupied cell.
    squared: Vec<u64>,
}

impl VoxelDistanceField {
    /// Number of cells along each axis.
    pub fn dims(&self) -> [u32; 3] {
        self.dims
    }

    /// Edge length of every voxel, shared with the source grid.
    pub fn voxel_size(&self) -> f32 {
        self.voxel_size
    }

    /// Row-major squared distances (voxel² units), `x` varying fastest.
    pub fn squared_distances(&self) -> &[u64] {
        &self.squared
    }

    /// Squared distance (voxel² units) from `coord` to the nearest occupied
    /// cell. `coord` must lie inside the grid dimensions.
    pub fn squared_distance_cells(&self, coord: [u32; 3]) -> u64 {
        self.squared[linear_index(coord, self.dims)]
    }

    /// World-space Euclidean distance from `coord` to the nearest occupied
    /// cell (the single `sqrt` scaled by [`VoxelDistanceField::voxel_size`]).
    pub fn distance(&self, coord: [u32; 3]) -> f32 {
        let squared = self.squared_distance_cells(coord) as f64;
        (squared.sqrt() * f64::from(self.voxel_size)) as f32
    }
}

/// Row-major linear index of a cell coordinate (`x` varying fastest).
fn linear_index(coord: [u32; 3], dims: [u32; 3]) -> usize {
    let [x, y, z] = coord;
    ((z * dims[1] + y) * dims[0] + x) as usize
}

/// Abscissa where the parabola rooted at `q` (height `fq`) drops below the one
/// rooted at `p` (height `fp`). `q` is always greater than `p`, so the
/// denominator is strictly positive.
fn parabola_intersection(fp: u64, p: usize, fq: u64, q: usize) -> f64 {
    let fp = fp as f64;
    let fq = fq as f64;
    let p = p as f64;
    let q = q as f64;
    ((fq + q * q) - (fp + p * p)) / (2.0 * q - 2.0 * p)
}

/// One-dimensional exact squared-distance transform (Felzenszwalb and
/// Huttenlocher): given per-sample heights `f`, returns `d[q] = min_p (f[p] +
/// (q - p)^2)`. Results are clamped to `inf` so the sentinel stays bounded
/// across the separable passes.
fn distance_transform_1d(f: &[u64], inf: u64) -> Vec<u64> {
    let n = f.len();
    let mut output = vec![0u64; n];
    // Indices of the parabolas forming the current lower envelope.
    let mut base = vec![0usize; n];
    // Boundaries between consecutive envelope parabolas (one sentinel extra).
    let mut boundary = vec![0.0f64; n + 1];
    let mut k = 0usize;
    base[0] = 0;
    boundary[0] = f64::NEG_INFINITY;
    boundary[1] = f64::INFINITY;
    for q in 1..n {
        let mut s = parabola_intersection(f[base[k]], base[k], f[q], q);
        while s <= boundary[k] {
            k -= 1;
            s = parabola_intersection(f[base[k]], base[k], f[q], q);
        }
        k += 1;
        base[k] = q;
        boundary[k] = s;
        boundary[k + 1] = f64::INFINITY;
    }
    k = 0;
    for (q, slot) in output.iter_mut().enumerate() {
        while boundary[k + 1] < q as f64 {
            k += 1;
        }
        let delta = q as i64 - base[k] as i64;
        let squared = (delta * delta) as u64;
        *slot = f[base[k]].saturating_add(squared).min(inf);
    }
    output
}

/// Sweeps the one-dimensional transform along `axis`, feeding the current grid
/// values back in place so three sequential sweeps accumulate the exact
/// three-dimensional squared Euclidean distance.
fn transform_axis(buf: &mut [u64], dims: [u32; 3], axis: usize, inf: u64) {
    let n = dims[axis] as usize;
    let (a1, a2) = match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    for i1 in 0..dims[a1] {
        for i2 in 0..dims[a2] {
            let column: Vec<u64> = (0..n)
                .map(|t| {
                    let mut coord = [0u32; 3];
                    coord[axis] = t as u32;
                    coord[a1] = i1;
                    coord[a2] = i2;
                    buf[linear_index(coord, dims)]
                })
                .collect();
            let transformed = distance_transform_1d(&column, inf);
            for (t, &value) in transformed.iter().enumerate() {
                let mut coord = [0u32; 3];
                coord[axis] = t as u32;
                coord[a1] = i1;
                coord[a2] = i2;
                buf[linear_index(coord, dims)] = value;
            }
        }
    }
}

/// Builds the exact Euclidean distance field of a voxelized surface.
///
/// Occupied cells seed the field at distance zero; every other cell receives
/// the exact squared Euclidean distance (voxel² units) to the nearest occupied
/// cell via three separable Felzenszwalb-Huttenlocher sweeps. The grid is
/// guaranteed to carry at least one occupied cell, so every cell resolves to a
/// finite distance.
pub fn voxel_distance_field(grid: &VoxelGrid) -> VoxelDistanceField {
    let dims = grid.dims();
    let dx = u64::from(dims[0]);
    let dy = u64::from(dims[1]);
    let dz = u64::from(dims[2]);
    // One past the largest achievable squared distance: a safe clamp sentinel.
    let inf = dx * dx + dy * dy + dz * dz + 1;
    let count = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);
    let mut squared = vec![inf; count];
    for cell in grid.occupied() {
        squared[linear_index(*cell, dims)] = 0;
    }
    transform_axis(&mut squared, dims, 0, inf);
    transform_axis(&mut squared, dims, 1, inf);
    transform_axis(&mut squared, dims, 2, inf);
    VoxelDistanceField {
        dims,
        voxel_size: grid.voxel_size(),
        squared,
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

    /// Brute-force exact squared distance for every cell, row-major, used as the
    /// reference oracle for the separable transform.
    fn brute_force(grid: &VoxelGrid) -> Vec<u64> {
        let dims = grid.dims();
        let occupied = grid.occupied();
        let mut out = Vec::with_capacity(
            (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize),
        );
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    let best = occupied
                        .iter()
                        .map(|o| {
                            let dx = i64::from(x) - i64::from(o[0]);
                            let dy = i64::from(y) - i64::from(o[1]);
                            let dz = i64::from(z) - i64::from(o[2]);
                            (dx * dx + dy * dy + dz * dz) as u64
                        })
                        .min()
                        .expect("grid always has at least one occupied cell");
                    out.push(best);
                }
            }
        }
        out
    }

    fn quad() -> TriangleMesh {
        mesh(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0, 1, 2], [0, 2, 3]],
        )
    }

    fn diagonal_triangle() -> TriangleMesh {
        mesh(
            vec![[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
            vec![[0, 1, 2]],
        )
    }

    fn tetrahedron() -> TriangleMesh {
        mesh(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
        )
    }

    #[test]
    fn occupied_cells_have_zero_distance() {
        let grid = voxelize_surface(&quad(), 4).unwrap();
        let field = voxel_distance_field(&grid);
        for cell in grid.occupied() {
            assert_eq!(field.squared_distance_cells(*cell), 0, "cell {cell:?}");
        }
    }

    #[test]
    fn matches_brute_force_on_quad() {
        let grid = voxelize_surface(&quad(), 4).unwrap();
        let field = voxel_distance_field(&grid);
        assert_eq!(field.squared_distances(), brute_force(&grid).as_slice());
    }

    #[test]
    fn matches_brute_force_on_diagonal_triangle() {
        let grid = voxelize_surface(&diagonal_triangle(), 4).unwrap();
        // A genuine three-dimensional occupancy exercising all three sweeps.
        assert!(grid.dims()[2] > 1, "expected a multi-layer grid");
        let field = voxel_distance_field(&grid);
        assert_eq!(field.squared_distances(), brute_force(&grid).as_slice());
    }

    #[test]
    fn matches_brute_force_on_tetrahedron() {
        let grid = voxelize_surface(&tetrahedron(), 6).unwrap();
        let field = voxel_distance_field(&grid);
        assert_eq!(field.squared_distances(), brute_force(&grid).as_slice());
    }

    #[test]
    fn distance_is_world_scaled_sqrt() {
        let grid = voxelize_surface(&tetrahedron(), 6).unwrap();
        let field = voxel_distance_field(&grid);
        let dims = field.dims();
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    let coord = [x, y, z];
                    let squared = field.squared_distance_cells(coord) as f64;
                    let expected = (squared.sqrt() * f64::from(field.voxel_size())) as f32;
                    assert_eq!(field.distance(coord), expected, "cell {coord:?}");
                }
            }
        }
    }

    #[test]
    fn dimensions_mirror_source_grid() {
        let grid = voxelize_surface(&diagonal_triangle(), 4).unwrap();
        let field = voxel_distance_field(&grid);
        assert_eq!(field.dims(), grid.dims());
        assert_eq!(field.voxel_size(), grid.voxel_size());
        let expected = (grid.dims()[0] as usize)
            * (grid.dims()[1] as usize)
            * (grid.dims()[2] as usize);
        assert_eq!(field.squared_distances().len(), expected);
    }
}
