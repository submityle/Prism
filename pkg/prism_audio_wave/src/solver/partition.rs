//! Rectangular decomposition: splitting the air region of a [`VoxelScene`]
//! into disjoint axis-aligned boxes. Each box becomes an independent ARD
//! partition whose interior is advanced analytically; adjacent boxes exchange
//! energy through the interface operator in [`crate::solver::ard`].
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Microsoft Project Acoustics, or Google Resonance Audio source or derived
//! code; no AI/ML.
//!
//! # Relationship
//! Implements the "rectangular subdivision" step of the ARD approach in design
//! section 43. The decomposition is a greedy maximal-box cover: simple,
//! deterministic, and guaranteed to tile every air voxel exactly once.

use alloc::vec;
use alloc::vec::Vec;

use super::scene::VoxelScene;

/// A half-open axis-aligned block of voxels, `lo <= cell < hi` per axis.
///
/// Partitions produced by [`partition_scene`] are disjoint and contain only
/// air voxels, so their interiors can be advanced independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Partition {
    /// Inclusive minimum voxel corner `[x, y, z]`.
    pub lo: [u32; 3],
    /// Exclusive maximum voxel corner `[x, y, z]`.
    pub hi: [u32; 3],
}

impl Partition {
    /// Per-axis voxel extent `[nx, ny, nz]` of the block.
    #[must_use]
    #[inline]
    pub fn dims(&self) -> [u32; 3] {
        [
            self.hi[0] - self.lo[0],
            self.hi[1] - self.lo[1],
            self.hi[2] - self.lo[2],
        ]
    }

    /// Number of voxels contained in the block.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        let d = self.dims();
        d[0] as usize * d[1] as usize * d[2] as usize
    }

    /// Returns `true` if the block has zero volume. Partitions produced by
    /// [`partition_scene`] are never empty.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.lo[0] >= self.hi[0] || self.lo[1] >= self.hi[1] || self.lo[2] >= self.hi[2]
    }

    /// Returns `true` if global voxel `(x, y, z)` lies inside the block.
    #[must_use]
    #[inline]
    pub fn contains(&self, x: u32, y: u32, z: u32) -> bool {
        x >= self.lo[0]
            && y >= self.lo[1]
            && z >= self.lo[2]
            && x < self.hi[0]
            && y < self.hi[1]
            && z < self.hi[2]
    }

    /// Local linear index (`x` fastest) of global voxel `(x, y, z)` within the
    /// block, or `None` when the voxel is outside.
    #[must_use]
    #[inline]
    pub fn local_index(&self, x: u32, y: u32, z: u32) -> Option<usize> {
        if !self.contains(x, y, z) {
            return None;
        }
        let d = self.dims();
        let lx = (x - self.lo[0]) as usize;
        let ly = (y - self.lo[1]) as usize;
        let lz = (z - self.lo[2]) as usize;
        Some((lz * d[1] as usize + ly) * d[0] as usize + lx)
    }
}

/// Decomposes the air region of `scene` into a disjoint set of rectangular
/// [`Partition`]s covering every air voxel exactly once.
///
/// The algorithm is a deterministic greedy maximal-box grower: starting from
/// the lowest uncovered air voxel it extends as far as possible in `x`, then
/// `y`, then `z`, keeping the block rectangular and air-only, marks the block
/// covered, and repeats.
///
/// # Examples
///
/// ```
/// # use bevy_math::Vec3;
/// # use prism_audio_wave::solver::{partition_scene, VoxelScene};
/// let scene = VoxelScene::new(Vec3::ZERO, 1.0, 4, 4, 1);
/// let parts = partition_scene(&scene);
/// // A fully open slab collapses to a single box.
/// assert_eq!(parts.len(), 1);
/// assert_eq!(parts[0].len(), 16);
/// ```
#[must_use]
pub fn partition_scene(scene: &VoxelScene) -> Vec<Partition> {
    let [nx, ny, nz] = scene.dims();
    let count = scene.len();
    let mut covered = vec![false; count];
    let mut out = Vec::new();

    let lin = |x: u32, y: u32, z: u32| -> usize {
        (z as usize * ny as usize + y as usize) * nx as usize + x as usize
    };

    for z0 in 0..nz {
        for y0 in 0..ny {
            for x0 in 0..nx {
                if !scene.is_air(x0, y0, z0) || covered[lin(x0, y0, z0)] {
                    continue;
                }

                // Grow along x.
                let mut x1 = x0 + 1;
                while x1 < nx && scene.is_air(x1, y0, z0) && !covered[lin(x1, y0, z0)] {
                    x1 += 1;
                }

                // Grow along y while the whole x-span stays air and uncovered.
                let mut y1 = y0 + 1;
                'grow_y: while y1 < ny {
                    for x in x0..x1 {
                        if !scene.is_air(x, y1, z0) || covered[lin(x, y1, z0)] {
                            break 'grow_y;
                        }
                    }
                    y1 += 1;
                }

                // Grow along z while the whole x-y slab stays air and uncovered.
                let mut z1 = z0 + 1;
                'grow_z: while z1 < nz {
                    for y in y0..y1 {
                        for x in x0..x1 {
                            if !scene.is_air(x, y, z1) || covered[lin(x, y, z1)] {
                                break 'grow_z;
                            }
                        }
                    }
                    z1 += 1;
                }

                for z in z0..z1 {
                    for y in y0..y1 {
                        for x in x0..x1 {
                            covered[lin(x, y, z)] = true;
                        }
                    }
                }

                out.push(Partition {
                    lo: [x0, y0, z0],
                    hi: [x1, y1, z1],
                });
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec3;

    fn total_air(scene: &VoxelScene) -> usize {
        let [nx, ny, nz] = scene.dims();
        let mut n = 0;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    if scene.is_air(x, y, z) {
                        n += 1;
                    }
                }
            }
        }
        n
    }

    #[test]
    fn open_box_is_single_partition() {
        let scene = VoxelScene::new(Vec3::ZERO, 1.0, 5, 4, 3);
        let parts = partition_scene(&scene);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].len(), 60);
    }

    #[test]
    fn partitions_tile_air_exactly_once() {
        let mut scene = VoxelScene::new(Vec3::ZERO, 1.0, 6, 6, 2);
        // Carve an interior wall to force multiple partitions.
        for z in 0..2 {
            for y in 1..5 {
                scene.set_solid(3, y, z, true);
            }
        }
        let parts = partition_scene(&scene);
        assert!(parts.len() >= 2);

        let [nx, ny, nz] = scene.dims();
        let mut hits = vec![0u32; scene.len()];
        for p in &parts {
            for z in p.lo[2]..p.hi[2] {
                for y in p.lo[1]..p.hi[1] {
                    for x in p.lo[0]..p.hi[0] {
                        assert!(scene.is_air(x, y, z));
                        hits[(z as usize * ny as usize + y as usize) * nx as usize + x as usize] +=
                            1;
                    }
                }
            }
        }
        let covered: usize = hits.iter().filter(|&&h| h == 1).count();
        assert_eq!(covered, total_air(&scene));
        assert!(hits.iter().all(|&h| h <= 1));
        let _ = nz;
    }

    #[test]
    fn local_index_is_dense() {
        let p = Partition {
            lo: [2, 1, 0],
            hi: [5, 3, 2],
        };
        assert_eq!(p.local_index(2, 1, 0), Some(0));
        assert_eq!(p.local_index(4, 2, 1), Some(p.len() - 1));
        assert_eq!(p.local_index(5, 1, 0), None);
    }
}
