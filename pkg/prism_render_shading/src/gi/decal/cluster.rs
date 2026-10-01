//! Clustered / froxel decal binning (CPU golden reference).
//!
//! A clustered deferred-decal renderer avoids testing every decal against every
//! pixel by first bucketing decals into the cells ("clusters" or "froxels") of a
//! view volume, so each pixel only consults the decals in its own cell.  This
//! module is the backend-neutral reference for that binning step, in the spirit
//! of Bartosz Wroński's clustered-decal note and the clustered-shading
//! froxel-assignment literature.
//!
//! The cluster volume is a world-space axis-aligned box ([`ClusterGrid::min`] ..
//! [`ClusterGrid::max`]) subdivided into a uniform `dims.x * dims.y * dims.z`
//! grid.  A decal is supplied as its world-space AABB ([`Aabb`]) — for an
//! oriented box decal this is the AABB of its oriented bounds.  Binning clips
//! each decal AABB to the grid, converts the clipped corners to integer cell
//! coordinates, and records the decal's index in every overlapped cell.  A decal
//! that lies entirely outside the grid is dropped.
//!
//! Cells are flattened as `x + dims.x * (y + dims.y * z)`, matching the common
//! GPU froxel layout, so the reference and its GPU twin agree on cell order.
//!
//! # Conventions
//! * Pure, deterministic functions: no RNG, IO, GPU, or `unsafe`.
//! * Collections use `alloc::vec::Vec` (the crate is `no_std`).
//! * Storage layouts mirror the GPU twin (f32 fields, explicit alignment).
//! * Binning is order-independent in input and deterministic in output: within a
//!   cell, decal indices appear in ascending input order.
//! * Defensive clamping everywhere: non-finite bounds, inverted AABBs,
//!   zero-size grids, and degenerate extents are handled so no `NaN`/`inf`
//!   escapes and no out-of-range cell index is produced.

use alloc::vec::Vec;
use bevy_math::{UVec3, Vec3};

/// A world-space axis-aligned bounding box.
///
/// Used both for a decal's world bounds and, internally, for the cluster grid's
/// extent.  Construction via [`Aabb::new`] sorts the corners so `min <= max`
/// component-wise even if the caller passes them reversed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// Lower corner (component-wise minimum).
    pub min: Vec3,
    /// Upper corner (component-wise maximum).
    pub max: Vec3,
}

impl Aabb {
    /// Builds an AABB from two corners, sorting them component-wise.
    ///
    /// Non-finite components are left as-is here; [`Aabb::sanitized`] rejects
    /// them when a well-formed box is required.
    #[inline]
    pub fn new(a: Vec3, b: Vec3) -> Self {
        Self {
            min: a.min(b),
            max: a.max(b),
        }
    }

    /// Returns the AABB with corners sorted, or `None` when any component is
    /// non-finite.
    #[inline]
    fn sanitized(self) -> Option<Self> {
        if finite(self.min) && finite(self.max) {
            Some(Self::new(self.min, self.max))
        } else {
            None
        }
    }

    /// Component-wise intersection with `other`, or `None` when the boxes are
    /// disjoint along any axis.
    #[inline]
    fn intersect(self, other: Aabb) -> Option<Aabb> {
        let min = self.min.max(other.min);
        let max = self.max.min(other.max);
        if min.x <= max.x && min.y <= max.y && min.z <= max.z {
            Some(Aabb { min, max })
        } else {
            None
        }
    }
}

/// A uniform world-space cluster (froxel) grid over an AABB.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClusterGrid {
    /// Number of cells along each axis.  Each component is treated as at least
    /// `1` during binning so a zero dimension never erases the grid.
    pub dims: UVec3,
    /// Lower corner of the grid volume (world space).
    pub min: Vec3,
    /// Upper corner of the grid volume (world space).
    pub max: Vec3,
}

impl ClusterGrid {
    /// Builds a grid over `[min, max]` with the given cell counts.
    ///
    /// The corners are sorted component-wise so a reversed `min`/`max` still
    /// yields a valid volume.
    #[inline]
    pub fn new(dims: UVec3, min: Vec3, max: Vec3) -> Self {
        Self {
            dims,
            min: min.min(max),
            max: min.max(max),
        }
    }

    /// Cell counts clamped so every axis has at least one cell.
    #[inline]
    pub fn clamped_dims(&self) -> UVec3 {
        UVec3::new(self.dims.x.max(1), self.dims.y.max(1), self.dims.z.max(1))
    }

    /// Total number of cells in the grid.
    #[inline]
    pub fn cluster_count(&self) -> usize {
        let d = self.clamped_dims();
        (d.x as usize) * (d.y as usize) * (d.z as usize)
    }

    /// Flattens a cell coordinate to a linear index
    /// (`x + dims.x * (y + dims.y * z)`).
    ///
    /// The coordinate is clamped to the valid range first, so an out-of-range
    /// coordinate maps to the nearest in-range cell rather than overflowing.
    #[inline]
    pub fn flatten(&self, coord: UVec3) -> usize {
        let d = self.clamped_dims();
        let x = coord.x.min(d.x - 1) as usize;
        let y = coord.y.min(d.y - 1) as usize;
        let z = coord.z.min(d.z - 1) as usize;
        x + (d.x as usize) * (y + (d.y as usize) * z)
    }

    /// Maps a world coordinate on one axis to a cell index in `[0, dim - 1]`.
    #[inline]
    fn axis_cell(value: f32, lo: f32, hi: f32, dim: u32) -> u32 {
        let extent = hi - lo;
        if !(extent > 0.0) {
            return 0;
        }
        let t = ((value - lo) / extent).clamp(0.0, 1.0);
        // `floor` via truncation is exact for the non-negative `t * dim`.
        let idx = (t * dim as f32) as u32;
        idx.min(dim - 1)
    }

    /// Returns the inclusive `[min, max]` cell coordinates overlapped by a
    /// world AABB already clipped to the grid.
    #[inline]
    fn cell_range(&self, clipped: Aabb) -> (UVec3, UVec3) {
        let d = self.clamped_dims();
        let lo = UVec3::new(
            Self::axis_cell(clipped.min.x, self.min.x, self.max.x, d.x),
            Self::axis_cell(clipped.min.y, self.min.y, self.max.y, d.y),
            Self::axis_cell(clipped.min.z, self.min.z, self.max.z, d.z),
        );
        let hi = UVec3::new(
            Self::axis_cell(clipped.max.x, self.min.x, self.max.x, d.x),
            Self::axis_cell(clipped.max.y, self.min.y, self.max.y, d.y),
            Self::axis_cell(clipped.max.z, self.min.z, self.max.z, d.z),
        );
        (lo, hi)
    }

    /// The grid's own extent as an [`Aabb`].
    #[inline]
    fn bounds(&self) -> Aabb {
        Aabb {
            min: self.min,
            max: self.max,
        }
    }

    /// Bins a slice of decal AABBs into per-cluster index lists.
    ///
    /// Returns a vector of length [`ClusterGrid::cluster_count`]; entry `i` holds
    /// the ascending input indices of every decal whose world AABB overlaps
    /// cluster `i`.  Decals that are non-finite or lie entirely outside the grid
    /// contribute to no cluster.
    #[inline]
    pub fn bin(&self, decals: &[Aabb]) -> Vec<Vec<u32>> {
        let count = self.cluster_count();
        let mut clusters: Vec<Vec<u32>> = Vec::with_capacity(count);
        for _ in 0..count {
            clusters.push(Vec::new());
        }

        let grid_bounds = self.bounds();
        for (index, decal) in decals.iter().enumerate() {
            let Some(aabb) = decal.sanitized() else {
                continue;
            };
            let Some(clipped) = aabb.intersect(grid_bounds) else {
                continue;
            };

            let (lo, hi) = self.cell_range(clipped);
            for z in lo.z..=hi.z {
                for y in lo.y..=hi.y {
                    for x in lo.x..=hi.x {
                        let cell = self.flatten(UVec3::new(x, y, z));
                        clusters[cell].push(index as u32);
                    }
                }
            }
        }

        clusters
    }
}

/// Returns `true` when every component of `v` is finite.
#[inline]
fn finite(v: Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `4^3` grid spanning the cube `[-1, 1]^3`.
    fn grid() -> ClusterGrid {
        ClusterGrid::new(UVec3::splat(4), Vec3::splat(-1.0), Vec3::splat(1.0))
    }

    /// A small AABB centred on `c` with half-size `h`.
    fn box_at(c: Vec3, h: f32) -> Aabb {
        Aabb::new(c - Vec3::splat(h), c + Vec3::splat(h))
    }

    #[test]
    fn centered_decal_lands_in_center_cells() {
        let g = grid();
        // A tiny box at the origin: for a 4-wide axis over [-1, 1] the origin
        // sits on the boundary between cells 1 and 2; a tiny box straddles both.
        let decals = [box_at(Vec3::ZERO, 0.01)];
        let clusters = g.bin(&decals);

        // Only the central cells (coords in {1,2} on each axis) may be hit.
        for z in 0..4u32 {
            for y in 0..4u32 {
                for x in 0..4u32 {
                    let cell = g.flatten(UVec3::new(x, y, z));
                    let central = (1..=2).contains(&x)
                        && (1..=2).contains(&y)
                        && (1..=2).contains(&z);
                    if !central {
                        assert!(
                            clusters[cell].is_empty(),
                            "non-central cell ({x},{y},{z}) got {:?}",
                            clusters[cell]
                        );
                    }
                }
            }
        }
        // The dead-centre cell is definitely covered.
        let center = g.flatten(UVec3::new(2, 2, 2));
        assert_eq!(clusters[center], [0]);
    }

    #[test]
    fn out_of_bounds_decal_is_not_binned() {
        let g = grid();
        let decals = [box_at(Vec3::splat(10.0), 0.5)];
        let clusters = g.bin(&decals);
        assert!(clusters.iter().all(|c| c.is_empty()), "nothing should bin");
    }

    #[test]
    fn large_decal_covers_every_cell() {
        let g = grid();
        // A box that fully encloses the grid overlaps every cluster.
        let decals = [box_at(Vec3::ZERO, 5.0)];
        let clusters = g.bin(&decals);
        assert_eq!(clusters.len(), g.cluster_count());
        assert!(clusters.iter().all(|c| c == &[0]), "every cell holds decal 0");
    }

    #[test]
    fn multiple_froxel_span_is_contiguous() {
        let g = grid();
        // A slab covering the lower-x half: x in [-1, 0], full y/z.
        let decals = [Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(0.0, 1.0, 1.0))];
        let clusters = g.bin(&decals);
        // Cells with x in {0,1} must hold the decal; x in {2,3} must not,
        // except the shared boundary cell x=2 edge at value 0 maps to cell 2.
        for z in 0..4u32 {
            for y in 0..4u32 {
                for x in 0..4u32 {
                    let cell = g.flatten(UVec3::new(x, y, z));
                    let hit = !clusters[cell].is_empty();
                    if x <= 1 {
                        assert!(hit, "cell ({x},{y},{z}) should be hit");
                    } else if x >= 3 {
                        assert!(!hit, "cell ({x},{y},{z}) should be empty");
                    }
                }
            }
        }
    }

    #[test]
    fn indices_are_ascending_within_a_cell() {
        let g = grid();
        let decals = [
            box_at(Vec3::ZERO, 5.0), // index 0 covers everything
            box_at(Vec3::ZERO, 5.0), // index 1 covers everything
            box_at(Vec3::ZERO, 5.0), // index 2 covers everything
        ];
        let clusters = g.bin(&decals);
        for cell in &clusters {
            assert_eq!(cell, &[0, 1, 2], "ascending input order expected");
        }
    }

    #[test]
    fn binning_is_deterministic() {
        let g = grid();
        let decals = [
            box_at(Vec3::new(-0.5, -0.5, -0.5), 0.2),
            box_at(Vec3::new(0.6, 0.1, -0.3), 0.4),
            box_at(Vec3::splat(100.0), 1.0), // out of bounds
        ];
        let a = g.bin(&decals);
        let b = g.bin(&decals);
        assert_eq!(a, b, "binning must be deterministic");
    }

    #[test]
    fn flatten_matches_froxel_layout() {
        let g = grid();
        assert_eq!(g.flatten(UVec3::new(0, 0, 0)), 0);
        assert_eq!(g.flatten(UVec3::new(1, 0, 0)), 1);
        assert_eq!(g.flatten(UVec3::new(0, 1, 0)), 4);
        assert_eq!(g.flatten(UVec3::new(0, 0, 1)), 16);
        assert_eq!(g.flatten(UVec3::new(3, 3, 3)), 63);
    }

    #[test]
    fn zero_dimension_is_treated_as_one() {
        let g = ClusterGrid::new(UVec3::new(0, 2, 0), Vec3::splat(-1.0), Vec3::splat(1.0));
        assert_eq!(g.clamped_dims(), UVec3::new(1, 2, 1));
        assert_eq!(g.cluster_count(), 2);
        let clusters = g.bin(&[box_at(Vec3::ZERO, 5.0)]);
        assert_eq!(clusters.len(), 2);
        assert!(clusters.iter().all(|c| c == &[0]));
    }

    #[test]
    fn reversed_corners_are_normalized() {
        let g = ClusterGrid::new(UVec3::splat(2), Vec3::splat(1.0), Vec3::splat(-1.0));
        assert_eq!(g.min, Vec3::splat(-1.0));
        assert_eq!(g.max, Vec3::splat(1.0));
        let decals = [Aabb::new(Vec3::splat(0.6), Vec3::splat(-0.6))];
        let clusters = g.bin(&decals);
        // A box across the centre of a 2^3 grid hits all 8 cells.
        assert!(clusters.iter().all(|c| c == &[0]));
    }

    #[test]
    fn non_finite_decal_is_skipped() {
        let g = grid();
        let decals = [Aabb {
            min: Vec3::new(f32::NAN, 0.0, 0.0),
            max: Vec3::splat(0.5),
        }];
        let clusters = g.bin(&decals);
        assert!(clusters.iter().all(|c| c.is_empty()));
    }

    #[test]
    fn degenerate_grid_extent_bins_into_single_slab() {
        // Zero extent along z: every z maps to cell 0 (dim clamped to >=1 use).
        let g = ClusterGrid::new(UVec3::new(2, 2, 2), Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 0.0));
        let clusters = g.bin(&[box_at(Vec3::ZERO, 5.0)]);
        assert_eq!(clusters.len(), 8);
        // z-extent is zero so only z=0 layer (cells 0..=3) is populated.
        for z in 0..2u32 {
            for y in 0..2u32 {
                for x in 0..2u32 {
                    let cell = g.flatten(UVec3::new(x, y, z));
                    if z == 0 {
                        assert_eq!(clusters[cell], [0]);
                    } else {
                        assert!(clusters[cell].is_empty());
                    }
                }
            }
        }
    }
}
