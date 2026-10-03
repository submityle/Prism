//! Voxelised scene: the discrete, axis-aligned occupancy grid the wave solve
//! runs on. Each cell is either air (sound propagates) or rigid (a perfectly
//! reflecting wall).
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Microsoft Project Acoustics, or Google Resonance Audio source or derived
//! code; no AI/ML.
//!
//! # Relationship
//! Provides the static geometry input for the ARD solve of design section 43.
//! The scene is the common voxel truth that both the rectangular
//! decomposition ([`crate::solver::partition`]) and the solver
//! ([`crate::solver::ard`]) consume.

use alloc::vec;
use alloc::vec::Vec;
use bevy_math::{ops, Vec3};

/// A regular voxel grid flagging each cell as air or rigid.
///
/// The origin sits at the minimum corner; cell `(x, y, z)` spans the world box
/// `[origin + (x,y,z)*dx, origin + (x+1,y+1,z+1)*dx]`. All cells start as air.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VoxelScene {
    dims: [u32; 3],
    cell_size: f32,
    origin: Vec3,
    // `true` where the cell is a rigid wall.
    solid: Vec<bool>,
}

impl VoxelScene {
    /// Builds an all-air scene of `nx * ny * nz` cells of edge `cell_size`
    /// metres, with its minimum corner at `origin`.
    ///
    /// Each dimension is clamped up to at least `1`, and `cell_size` is clamped
    /// to a small positive floor so the grid is always well formed.
    #[must_use]
    pub fn new(origin: Vec3, cell_size: f32, nx: u32, ny: u32, nz: u32) -> Self {
        let dims = [nx.max(1), ny.max(1), nz.max(1)];
        let count = dims[0] as usize * dims[1] as usize * dims[2] as usize;
        Self {
            dims,
            cell_size: cell_size.max(1.0e-4),
            origin,
            solid: vec![false; count],
        }
    }

    /// The per-axis cell counts `[nx, ny, nz]`.
    #[must_use]
    #[inline]
    pub fn dims(&self) -> [u32; 3] {
        self.dims
    }

    /// The voxel edge length in metres.
    #[must_use]
    #[inline]
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }

    /// The world-space minimum corner of the grid.
    #[must_use]
    #[inline]
    pub fn origin(&self) -> Vec3 {
        self.origin
    }

    /// Total voxel count.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.solid.len()
    }

    /// Returns `true` if the scene has no voxels. Always `false` for scenes
    /// built with [`VoxelScene::new`], which clamps every dimension to `>= 1`.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.solid.is_empty()
    }

    /// Linear index of cell `(x, y, z)` (`x` fastest), or `None` if the triplet
    /// is out of range.
    #[must_use]
    #[inline]
    pub fn index(&self, x: u32, y: u32, z: u32) -> Option<usize> {
        if x >= self.dims[0] || y >= self.dims[1] || z >= self.dims[2] {
            return None;
        }
        let nx = self.dims[0] as usize;
        let ny = self.dims[1] as usize;
        Some((z as usize * ny + y as usize) * nx + x as usize)
    }

    /// Marks cell `(x, y, z)` rigid (`true`) or air (`false`). Out-of-range
    /// triplets are ignored.
    pub fn set_solid(&mut self, x: u32, y: u32, z: u32, solid: bool) {
        if let Some(i) = self.index(x, y, z) {
            self.solid[i] = solid;
        }
    }

    /// Returns `true` if cell `(x, y, z)` is air. Out-of-range cells are
    /// treated as rigid (the world is closed at its boundary).
    #[must_use]
    #[inline]
    pub fn is_air(&self, x: u32, y: u32, z: u32) -> bool {
        match self.index(x, y, z) {
            Some(i) => !self.solid[i],
            None => false,
        }
    }

    /// World-space centre of cell `(x, y, z)`.
    #[must_use]
    pub fn cell_center(&self, x: u32, y: u32, z: u32) -> Vec3 {
        self.origin
            + Vec3::new(
                (x as f32 + 0.5) * self.cell_size,
                (y as f32 + 0.5) * self.cell_size,
                (z as f32 + 0.5) * self.cell_size,
            )
    }

    /// Maps a world position to the voxel containing it, clamped to the grid,
    /// returned as a `(x, y, z)` triplet.
    #[must_use]
    pub fn voxel_of(&self, world: Vec3) -> [u32; 3] {
        let local = (world - self.origin) / self.cell_size;
        let clamp_axis = |v: f32, n: u32| -> u32 {
            let f = ops::floor(v);
            if f < 0.0 {
                0
            } else {
                (f as u32).min(n - 1)
            }
        };
        [
            clamp_axis(local.x, self.dims[0]),
            clamp_axis(local.y, self.dims[1]),
            clamp_axis(local.z, self.dims[2]),
        ]
    }

    /// Finds the nearest air voxel to `world` within a small search radius,
    /// returning its triplet. Prefers the containing cell when it is already
    /// air. Returns `None` only when the whole scene is rigid.
    #[must_use]
    pub fn nearest_air(&self, world: Vec3) -> Option<[u32; 3]> {
        let [cx, cy, cz] = self.voxel_of(world);
        if self.is_air(cx, cy, cz) {
            return Some([cx, cy, cz]);
        }
        let max_r = self.dims[0].max(self.dims[1]).max(self.dims[2]);
        for r in 1..=max_r {
            let r = r as i32;
            for dz in -r..=r {
                for dy in -r..=r {
                    for dx in -r..=r {
                        // Only inspect the shell at Chebyshev radius `r`.
                        if dx.abs() != r && dy.abs() != r && dz.abs() != r {
                            continue;
                        }
                        let x = cx as i32 + dx;
                        let y = cy as i32 + dy;
                        let z = cz as i32 + dz;
                        if x < 0 || y < 0 || z < 0 {
                            continue;
                        }
                        let (x, y, z) = (x as u32, y as u32, z as u32);
                        if self.is_air(x, y, z) {
                            return Some([x, y, z]);
                        }
                    }
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_scene_is_all_air() {
        let scene = VoxelScene::new(Vec3::ZERO, 0.5, 4, 3, 2);
        assert_eq!(scene.len(), 24);
        assert!(!scene.is_empty());
        for z in 0..2 {
            for y in 0..3 {
                for x in 0..4 {
                    assert!(scene.is_air(x, y, z));
                }
            }
        }
    }

    #[test]
    fn out_of_range_is_rigid() {
        let scene = VoxelScene::new(Vec3::ZERO, 1.0, 2, 2, 2);
        assert!(!scene.is_air(2, 0, 0));
        assert_eq!(scene.index(5, 5, 5), None);
    }

    #[test]
    fn cell_center_and_voxel_of_round_trip() {
        let scene = VoxelScene::new(Vec3::new(1.0, 0.0, -2.0), 0.5, 6, 6, 6);
        for (x, y, z) in [(0, 0, 0), (3, 2, 5), (5, 5, 5)] {
            let c = scene.cell_center(x, y, z);
            assert_eq!(scene.voxel_of(c), [x, y, z]);
        }
    }

    #[test]
    fn nearest_air_skips_solid() {
        let mut scene = VoxelScene::new(Vec3::ZERO, 1.0, 3, 3, 3);
        scene.set_solid(1, 1, 1, true);
        let found = scene.nearest_air(scene.cell_center(1, 1, 1)).unwrap();
        assert!(scene.is_air(found[0], found[1], found[2]));
        assert_ne!(found, [1, 1, 1]);
    }
}
