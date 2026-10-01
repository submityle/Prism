//! Margin padding of a voxel grid to carve out an exterior shell for the
//! `CPU` golden path.
//!
//! Conservative surface voxelization ([`super::mesh_voxelize::voxelize_surface`])
//! fits the grid tightly to the mesh's axis-aligned bounding box, so the
//! occupied cells reach the very edge of the lattice and there is no empty
//! region surrounding the surface. That is a problem for every distance-field
//! consumer that queries *outside* the mesh: soft shadows march a ray away
//! from the surface toward the light, ambient occlusion samples along the
//! outward normal, and cone-traced global illumination steps outward through
//! open space. Without an exterior band the signed distance field
//! ([`super::mesh_signed_distance_field::signed_distance_field`]) has nowhere
//! to record positive (outside) distances near the boundary.
//!
//! `AAA` engines solve this by baking mesh distance fields with a fixed margin
//! of empty voxels on every side. This module provides the equivalent
//! grid-to-grid transform: [`pad_voxel_grid`] grows the lattice by `margin`
//! cells on each face, shifts the origin so the existing cells keep their
//! world-space positions, and re-indexes the occupied set into the enlarged
//! coordinate frame. The occupancy itself is unchanged — only empty exterior
//! cells are added — so a subsequent distance transform fills the new band
//! with genuine positive distances.
//!
//! The transform is purely integer coordinate arithmetic plus a single
//! floating-point origin shift, so it is exact and reproducible. Because every
//! occupied coordinate is translated by the same positive offset, the sorted,
//! de-duplicated ordering of the occupancy set is preserved and the result is
//! assembled directly with [`super::mesh_voxelize::VoxelGrid::from_sorted_parts`]
//! without re-running voxelization.

use super::mesh_voxelize::VoxelGrid;

/// Returns a copy of `grid` enlarged by `margin` empty voxel cells on every
/// face, producing an exterior shell for distance-field queries.
///
/// The returned grid shares `grid`'s voxel size and keeps every original cell
/// at its original world-space position. Its dimensions grow by `2 * margin`
/// along each axis (one `margin` band on the low side and one on the high
/// side), its origin moves to the new low corner (`origin - margin *
/// voxel_size` per axis), and each occupied coordinate is shifted by `margin`
/// so it addresses the same world-space cell in the enlarged frame.
///
/// A `margin` of zero returns a grid equal to the input (same origin,
/// dimensions, and occupancy), making the transform an identity in that case.
///
/// The occupancy is left untouched apart from the uniform translation, so no
/// previously empty cell becomes occupied; the added band is entirely exterior
/// space that a later distance transform can fill with positive distances.
pub fn pad_voxel_grid(grid: &VoxelGrid, margin: u32) -> VoxelGrid {
    let voxel_size = grid.voxel_size();
    let old_origin = grid.origin();
    let old_dims = grid.dims();

    let margin_offset = margin as f32 * voxel_size;
    let origin = [
        old_origin[0] - margin_offset,
        old_origin[1] - margin_offset,
        old_origin[2] - margin_offset,
    ];
    let dims = [
        old_dims[0] + 2 * margin,
        old_dims[1] + 2 * margin,
        old_dims[2] + 2 * margin,
    ];

    // Translating every coordinate by the same positive `margin` preserves the
    // ascending, de-duplicated ordering of the occupancy set, so the shifted
    // vector satisfies the invariants expected by `from_sorted_parts`.
    let occupied = grid
        .occupied()
        .iter()
        .map(|&[x, y, z]| [x + margin, y + margin, z + margin])
        .collect();

    VoxelGrid::from_sorted_parts(origin, voxel_size, dims, occupied)
}

#[cfg(test)]
mod tests {
    use super::pad_voxel_grid;
    use crate::ray_scene::mesh_signed_distance_field::signed_distance_field;
    use crate::ray_scene::mesh_voxelize::{voxelize_surface, VoxelGrid};
    use crate::ray_scene::triangle_mesh::TriangleMesh;

    /// Builds a triangle mesh with no normals or UVs for test fixtures.
    fn mesh(positions: Vec<[f32; 3]>, indices: Vec<[u32; 3]>) -> TriangleMesh {
        TriangleMesh::new(positions, Vec::new(), Vec::new(), indices).unwrap()
    }

    /// Unit cube (12 triangles) spanning `[0, 1]^3`.
    fn cube() -> TriangleMesh {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let indices = vec![
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 2, 6],
            [3, 6, 7],
            [0, 3, 7],
            [0, 7, 4],
            [1, 2, 6],
            [1, 6, 5],
        ];
        mesh(positions, indices)
    }

    #[test]
    fn dims_grow_by_twice_the_margin() {
        let grid = voxelize_surface(&cube(), 4).unwrap();
        let base = grid.dims();
        let padded = pad_voxel_grid(&grid, 2);
        assert_eq!(
            padded.dims(),
            [base[0] + 4, base[1] + 4, base[2] + 4],
            "each axis grows by 2 * margin",
        );
    }

    #[test]
    fn origin_shifts_by_margin_times_voxel_size() {
        let grid = voxelize_surface(&cube(), 4).unwrap();
        let voxel = grid.voxel_size();
        let base = grid.origin();
        let padded = pad_voxel_grid(&grid, 3);
        let shift = 3.0 * voxel;
        for (axis, (&moved, &start)) in padded.origin().iter().zip(base.iter()).enumerate() {
            assert!(
                (moved - (start - shift)).abs() < 1e-6,
                "origin[{axis}] moves to the new low corner",
            );
        }
    }

    #[test]
    fn occupied_cells_translate_and_count_is_preserved() {
        let grid = voxelize_surface(&cube(), 4).unwrap();
        let margin = 2u32;
        let original: Vec<[u32; 3]> = grid.occupied().to_vec();
        let padded = pad_voxel_grid(&grid, margin);
        assert_eq!(
            padded.occupied_count(),
            grid.occupied_count(),
            "padding adds only empty cells",
        );
        for cell in &original {
            let shifted = [cell[0] + margin, cell[1] + margin, cell[2] + margin];
            assert!(
                padded.is_occupied(shifted),
                "cell {cell:?} is re-indexed to {shifted:?}",
            );
        }
    }

    #[test]
    fn occupied_set_stays_sorted_and_in_bounds() {
        let grid = voxelize_surface(&cube(), 4).unwrap();
        let padded = pad_voxel_grid(&grid, 2);
        let cells = padded.occupied();
        let dims = padded.dims();
        for pair in cells.windows(2) {
            assert!(pair[0] < pair[1], "occupancy stays strictly ascending");
        }
        for cell in cells {
            for (&coord, &extent) in cell.iter().zip(dims.iter()) {
                assert!(coord < extent, "cell stays in bounds");
            }
        }
    }

    #[test]
    fn exterior_band_has_positive_distance() {
        let grid = voxelize_surface(&cube(), 4).unwrap();
        let padded = pad_voxel_grid(&grid, 2);
        let field = signed_distance_field(&padded);
        // The far corner cell lies entirely inside the added margin band, so
        // it is exterior space and must carry a positive (outside) distance.
        let corner = [0u32, 0, 0];
        assert!(
            !field.is_inside(corner),
            "the pure-margin corner is outside the mesh",
        );
        assert!(
            field.signed_distance(corner) > 0.0,
            "the exterior band records positive distances",
        );
    }

    #[test]
    fn zero_margin_is_identity() {
        let grid = voxelize_surface(&cube(), 4).unwrap();
        let padded = pad_voxel_grid(&grid, 0);
        assert_eq!(padded.dims(), grid.dims());
        assert_eq!(padded.origin(), grid.origin());
        assert_eq!(padded.occupied(), grid.occupied());
        let identity: VoxelGrid = padded;
        assert_eq!(identity, grid);
    }
}
