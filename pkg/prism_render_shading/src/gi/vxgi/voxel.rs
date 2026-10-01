//! Mip-mapped opacity / radiance voxel grid for voxel cone tracing — CPU
//! golden.
//!
//! The first half of Crassin's *Interactive Indirect Illumination Using Voxel
//! Cone Tracing* (2011) is a volumetric representation of the scene: an axis
//! aligned grid where every voxel stores how much it occludes light
//! (**opacity**, `[0, 1]`) and how much light it emits / reflects
//! (**radiance**, non-negative RGB).  Cone tracing then integrates this field
//! by reading successively coarser **mip levels** so that a widening cone can
//! be approximated by a single pre-filtered sample.  This module is the
//! backend-neutral, GPU-free reference for that structure.
//!
//! [`VoxelGrid`] owns the base (level-0) resolution, the world-space bounds the
//! grid spans, and a pyramid of [`VoxelLevel`]s.  Level 0 is the authored
//! field; [`VoxelGrid::build_mips`] produces each coarser level by averaging
//! the (up to) eight children of every parent voxel, exactly like a GPU
//! `generate_mipmaps` pass over a 3D texture.
//!
//! # Conventions
//! * **Layout.** Within a level of resolution `(nx, ny, nz)`, voxel
//!   `(x, y, z)` is stored row-major with `x` fastest:
//!   `index = x + y*nx + z*nx*ny`.  This matches the natural scan order of a 3D
//!   texture so the CPU buffer and a GPU twin share a byte layout.
//! * **World <-> voxel.** [`VoxelGrid::world_to_voxel`] maps a world point to
//!   continuous *level-0* voxel coordinates
//!   `(p - min) / (max - min) * resolution`; [`VoxelGrid::voxel_to_world`] is
//!   its exact inverse.  Voxel `i` occupies `[i, i+1)` along an axis and its
//!   centre sits at `i + 0.5`.  The world extent is clamped away from zero so
//!   the maps are always finite.
//! * **Sampling.** [`VoxelGrid::fetch_trilinear`] trilinearly blends the eight
//!   voxel *centres* around the query coordinate with clamp-to-edge, so a query
//!   exactly at a voxel centre returns that voxel's stored value bit-for-bit.
//!   [`VoxelGrid::sample_world`] returns a zero (empty-space) sample when the
//!   world point lies outside the grid bounds, otherwise it defers to
//!   [`fetch_trilinear`](VoxelGrid::fetch_trilinear).  [`VoxelGrid::sample_world_lod`]
//!   samples a fractional mip level by lerping the two bracketing integer
//!   levels, the read a cone marcher issues.
//! * **Mip construction.** Each coarser level halves every axis
//!   (`n' = max(1, n / 2)`), box-averaging the aligned `2^3` block of children,
//!   clamping child indices on odd extents.  Averaging keeps opacity in
//!   `[0, 1]` and radiance non-negative.
//! * **Defensive clamps.** Opacity is clamped to `[0, 1]` and radiance to the
//!   non-negative octant on write; every divisor is guarded and no path can
//!   produce `NaN` or infinity.
//! * **Determinism / safety.** Pure, deterministic computation: no RNG, no
//!   I/O, no GPU, no `unsafe`.

use alloc::vec;
use alloc::vec::Vec;
use bevy_math::{ops, UVec3, Vec3};

/// Smallest world extent (per axis) tolerated before the `world <-> voxel`
/// maps would divide by (near-)zero.  Keeps every mapping finite.
const MIN_EXTENT: f32 = 1.0e-6;

/// A single pre-filtered sample read out of the grid: occlusion plus emitted /
/// reflected radiance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoxelSample {
    /// Fractional occlusion in `[0, 1]`: `0` is fully transparent, `1` opaque.
    pub opacity: f32,
    /// Non-negative RGB radiance stored in the voxel.
    pub radiance: Vec3,
}

impl VoxelSample {
    /// The empty-space sample: perfectly transparent and dark.
    pub const EMPTY: Self = Self {
        opacity: 0.0,
        radiance: Vec3::ZERO,
    };
}

/// One level of the mip pyramid: a dense `nx * ny * nz` block of opacity and
/// radiance samples laid out row-major with `x` fastest.
#[derive(Clone, Debug, PartialEq)]
pub struct VoxelLevel {
    resolution: UVec3,
    opacity: Vec<f32>,
    radiance: Vec<Vec3>,
}

impl VoxelLevel {
    /// Total voxel count of this level.
    #[inline]
    fn count(resolution: UVec3) -> usize {
        resolution.x as usize * resolution.y as usize * resolution.z as usize
    }

    /// Allocates a zero-initialised level at `resolution`.
    fn zeroed(resolution: UVec3) -> Self {
        let n = Self::count(resolution);
        Self {
            resolution,
            opacity: vec![0.0; n],
            radiance: vec![Vec3::ZERO; n],
        }
    }

    /// The level's `(nx, ny, nz)` resolution.
    #[inline]
    pub fn resolution(&self) -> UVec3 {
        self.resolution
    }

    /// Row-major opacity buffer (`x` fastest).
    #[inline]
    pub fn opacity(&self) -> &[f32] {
        &self.opacity
    }

    /// Row-major radiance buffer (`x` fastest).
    #[inline]
    pub fn radiance(&self) -> &[Vec3] {
        &self.radiance
    }

    /// Linear buffer index of voxel `(x, y, z)`; the caller guarantees the
    /// coordinate is in range.
    #[inline]
    fn index(&self, x: u32, y: u32, z: u32) -> usize {
        (x as usize)
            + (y as usize) * self.resolution.x as usize
            + (z as usize) * self.resolution.x as usize * self.resolution.y as usize
    }
}

/// A mip-mapped opacity / radiance voxel grid spanning an axis-aligned world
/// box.
#[derive(Clone, Debug, PartialEq)]
pub struct VoxelGrid {
    min: Vec3,
    max: Vec3,
    levels: Vec<VoxelLevel>,
}

impl VoxelGrid {
    /// Creates a zero-initialised grid at `resolution` spanning the world box
    /// `[min, max]`.
    ///
    /// Each resolution axis is clamped to at least `1` and the box is
    /// normalised so `min <= max` component-wise with a non-degenerate extent;
    /// only level 0 is populated until [`build_mips`](Self::build_mips) runs.
    pub fn new(resolution: UVec3, min: Vec3, max: Vec3) -> Self {
        let resolution = resolution.max(UVec3::ONE);
        let (min, max) = normalise_bounds(min, max);
        Self {
            min,
            max,
            levels: vec![VoxelLevel::zeroed(resolution)],
        }
    }

    /// Builds a grid directly from authored level-0 buffers.
    ///
    /// `opacity` and `radiance` must each hold exactly
    /// `resolution.x * resolution.y * resolution.z` entries (row-major, `x`
    /// fastest); on any length mismatch a zero-initialised grid is returned so
    /// callers never observe a partially filled buffer.  Values are clamped on
    /// ingest (opacity to `[0, 1]`, radiance to the non-negative octant).
    pub fn from_level0(
        resolution: UVec3,
        min: Vec3,
        max: Vec3,
        opacity: Vec<f32>,
        radiance: Vec<Vec3>,
    ) -> Self {
        let resolution = resolution.max(UVec3::ONE);
        let expected = VoxelLevel::count(resolution);
        if opacity.len() != expected || radiance.len() != expected {
            return Self::new(resolution, min, max);
        }
        let opacity = opacity.into_iter().map(sanitize_opacity).collect();
        let radiance = radiance.into_iter().map(sanitize_radiance).collect();
        let (min, max) = normalise_bounds(min, max);
        Self {
            min,
            max,
            levels: vec![VoxelLevel {
                resolution,
                opacity,
                radiance,
            }],
        }
    }

    /// Minimum corner of the world box the grid spans.
    #[inline]
    pub fn min_bounds(&self) -> Vec3 {
        self.min
    }

    /// Maximum corner of the world box the grid spans.
    #[inline]
    pub fn max_bounds(&self) -> Vec3 {
        self.max
    }

    /// Level-0 resolution `(nx, ny, nz)`.
    #[inline]
    pub fn resolution(&self) -> UVec3 {
        self.levels[0].resolution
    }

    /// Number of mip levels currently stored (always at least `1`).
    #[inline]
    pub fn level_count(&self) -> u32 {
        self.levels.len() as u32
    }

    /// Index of the coarsest valid level.
    #[inline]
    pub fn max_level(&self) -> u32 {
        self.level_count() - 1
    }

    /// Borrows mip `level`, clamped to the available range.
    #[inline]
    pub fn level(&self, level: u32) -> &VoxelLevel {
        let idx = (level as usize).min(self.levels.len() - 1);
        &self.levels[idx]
    }

    /// Per-axis world size of a single level-0 voxel.
    #[inline]
    pub fn voxel_extent(&self) -> Vec3 {
        let res = self.resolution().as_vec3();
        (self.max - self.min) / res.max(Vec3::ONE)
    }

    /// Representative scalar voxel size (mean of the three axis extents), the
    /// finest length a cone's diameter is measured against.
    #[inline]
    pub fn voxel_size(&self) -> f32 {
        let e = self.voxel_extent();
        ((e.x + e.y + e.z) / 3.0).max(MIN_EXTENT)
    }

    /// Writes a single level-0 voxel, clamping inputs defensively.  Out-of-range
    /// coordinates are ignored.  Invalidates any previously built mip levels,
    /// which must be rebuilt via [`build_mips`](Self::build_mips).
    pub fn set_voxel(&mut self, coord: UVec3, opacity: f32, radiance: Vec3) {
        let res = self.resolution();
        if coord.x >= res.x || coord.y >= res.y || coord.z >= res.z {
            return;
        }
        let idx = self.levels[0].index(coord.x, coord.y, coord.z);
        self.levels[0].opacity[idx] = sanitize_opacity(opacity);
        self.levels[0].radiance[idx] = sanitize_radiance(radiance);
    }

    /// Maps a world point to continuous level-0 voxel coordinates.
    ///
    /// The voxel-`i` centre lies at `i + 0.5`; the inverse is
    /// [`voxel_to_world`](Self::voxel_to_world).
    #[inline]
    pub fn world_to_voxel(&self, world: Vec3) -> Vec3 {
        let extent = (self.max - self.min).max(Vec3::splat(MIN_EXTENT));
        (world - self.min) / extent * self.resolution().as_vec3()
    }

    /// Maps continuous level-0 voxel coordinates back to a world point; the
    /// exact inverse of [`world_to_voxel`](Self::world_to_voxel).
    #[inline]
    pub fn voxel_to_world(&self, voxel: Vec3) -> Vec3 {
        let extent = (self.max - self.min).max(Vec3::splat(MIN_EXTENT));
        self.min + voxel / self.resolution().as_vec3() * extent
    }

    /// Trilinearly samples `level` at continuous *level-0* voxel coordinates.
    ///
    /// The coordinate is first rescaled into the level's own resolution, then
    /// the eight surrounding voxel centres are blended with clamp-to-edge.  A
    /// query exactly at a voxel centre returns that voxel's stored value.
    pub fn fetch_trilinear(&self, level: u32, voxel_coord: Vec3) -> VoxelSample {
        let lvl = self.level(level);
        let res = lvl.resolution;

        // Rescale level-0 voxel coords into this level's grid.  Both grids span
        // the same world box, so the ratio of resolutions converts between them.
        let scale = res.as_vec3() / self.resolution().as_vec3();
        let scaled = voxel_coord * scale;

        // Shift by -0.5 so integer coordinates land on voxel centres.
        let centred = scaled - Vec3::splat(0.5);
        let base = Vec3::new(ops::floor(centred.x), ops::floor(centred.y), ops::floor(centred.z));
        let frac = centred - base;

        let bx = base.x as i32;
        let by = base.y as i32;
        let bz = base.z as i32;

        let mut opacity = 0.0;
        let mut radiance = Vec3::ZERO;
        for dz in 0..2 {
            let wz = if dz == 0 { 1.0 - frac.z } else { frac.z };
            let z = clamp_index(bz + dz, res.z);
            for dy in 0..2 {
                let wy = if dy == 0 { 1.0 - frac.y } else { frac.y };
                let y = clamp_index(by + dy, res.y);
                for dx in 0..2 {
                    let wx = if dx == 0 { 1.0 - frac.x } else { frac.x };
                    let x = clamp_index(bx + dx, res.x);
                    let w = wx * wy * wz;
                    let idx = lvl.index(x, y, z);
                    opacity += w * lvl.opacity[idx];
                    radiance += w * lvl.radiance[idx];
                }
            }
        }
        VoxelSample {
            opacity: opacity.clamp(0.0, 1.0),
            radiance: radiance.max(Vec3::ZERO),
        }
    }

    /// Samples `level` at a world point, returning [`VoxelSample::EMPTY`] when
    /// the point lies outside the grid bounds and the trilinear fetch
    /// otherwise.
    pub fn sample_world(&self, level: u32, world: Vec3) -> VoxelSample {
        if !self.contains(world) {
            return VoxelSample::EMPTY;
        }
        self.fetch_trilinear(level, self.world_to_voxel(world))
    }

    /// Samples a *fractional* mip `lod` at a world point by lerping the two
    /// bracketing integer levels; `lod` is clamped to `[0, max_level]`.  Points
    /// outside the bounds return [`VoxelSample::EMPTY`].
    pub fn sample_world_lod(&self, world: Vec3, lod: f32) -> VoxelSample {
        if !self.contains(world) {
            return VoxelSample::EMPTY;
        }
        let max_level = self.max_level() as f32;
        let lod = lod.clamp(0.0, max_level);
        let lo = ops::floor(lod);
        let frac = lod - lo;
        let voxel = self.world_to_voxel(world);
        let low = self.fetch_trilinear(lo as u32, voxel);
        if frac <= f32::MIN_POSITIVE {
            return low;
        }
        let high = self.fetch_trilinear(lo as u32 + 1, voxel);
        VoxelSample {
            opacity: (low.opacity + (high.opacity - low.opacity) * frac).clamp(0.0, 1.0),
            radiance: (low.radiance + (high.radiance - low.radiance) * frac).max(Vec3::ZERO),
        }
    }

    /// Whether a world point lies within the grid's axis-aligned bounds.
    #[inline]
    pub fn contains(&self, world: Vec3) -> bool {
        world.x >= self.min.x
            && world.y >= self.min.y
            && world.z >= self.min.z
            && world.x <= self.max.x
            && world.y <= self.max.y
            && world.z <= self.max.z
    }

    /// Rebuilds the whole mip pyramid from level 0.
    ///
    /// Starting at level 0, each coarser level halves every axis
    /// (`n' = max(1, n / 2)`) and box-averages the aligned `2^3` block of
    /// children (child indices clamped on odd extents).  Construction stops once
    /// a level reaches `1^3`.
    pub fn build_mips(&mut self) {
        self.levels.truncate(1);
        loop {
            let prev = self.levels.last().expect("level 0 always present");
            let pr = prev.resolution;
            if pr.x <= 1 && pr.y <= 1 && pr.z <= 1 {
                break;
            }
            let nr = UVec3::new((pr.x / 2).max(1), (pr.y / 2).max(1), (pr.z / 2).max(1));
            let next = Self::downsample(prev, nr);
            self.levels.push(next);
        }
    }

    /// Box-averages `prev` into a level of resolution `nr`.
    fn downsample(prev: &VoxelLevel, nr: UVec3) -> VoxelLevel {
        let pr = prev.resolution;
        let mut out = VoxelLevel::zeroed(nr);
        for z in 0..nr.z {
            for y in 0..nr.y {
                for x in 0..nr.x {
                    let mut sum_o = 0.0;
                    let mut sum_r = Vec3::ZERO;
                    let mut count = 0.0;
                    for dz in 0..2 {
                        let cz = (2 * z + dz).min(pr.z - 1);
                        for dy in 0..2 {
                            let cy = (2 * y + dy).min(pr.y - 1);
                            for dx in 0..2 {
                                let cx = (2 * x + dx).min(pr.x - 1);
                                let idx = prev.index(cx, cy, cz);
                                sum_o += prev.opacity[idx];
                                sum_r += prev.radiance[idx];
                                count += 1.0;
                            }
                        }
                    }
                    let inv = if count > 0.0 { 1.0 / count } else { 0.0 };
                    let oidx = out.index(x, y, z);
                    out.opacity[oidx] = (sum_o * inv).clamp(0.0, 1.0);
                    out.radiance[oidx] = (sum_r * inv).max(Vec3::ZERO);
                }
            }
        }
        out
    }
}

/// Clamps opacity into `[0, 1]`, mapping any `NaN` to `0`.
#[inline]
fn sanitize_opacity(o: f32) -> f32 {
    if o.is_nan() {
        0.0
    } else {
        o.clamp(0.0, 1.0)
    }
}

/// Clamps radiance to the non-negative octant, mapping `NaN` components to `0`.
#[inline]
fn sanitize_radiance(r: Vec3) -> Vec3 {
    Vec3::new(
        if r.x.is_nan() { 0.0 } else { r.x.max(0.0) },
        if r.y.is_nan() { 0.0 } else { r.y.max(0.0) },
        if r.z.is_nan() { 0.0 } else { r.z.max(0.0) },
    )
}

/// Normalises a bounding box so `min <= max` component-wise with a
/// non-degenerate extent on every axis.
#[inline]
fn normalise_bounds(min: Vec3, max: Vec3) -> (Vec3, Vec3) {
    let lo = min.min(max);
    let hi = min.max(max);
    let hi = hi.max(lo + Vec3::splat(MIN_EXTENT));
    (lo, hi)
}

/// Clamps a (possibly out-of-range) signed index to `[0, size - 1]`.
#[inline]
fn clamp_index(i: i32, size: u32) -> u32 {
    i.clamp(0, size as i32 - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_grid(res: u32) -> VoxelGrid {
        VoxelGrid::new(UVec3::splat(res), Vec3::ZERO, Vec3::splat(res as f32))
    }

    #[test]
    fn world_voxel_round_trip() {
        let grid = unit_grid(8);
        let p = Vec3::new(1.25, 6.5, 3.75);
        let v = grid.world_to_voxel(p);
        let back = grid.voxel_to_world(v);
        assert!((back - p).length() < 1.0e-5, "round trip {back:?} vs {p:?}");
    }

    #[test]
    fn voxel_centre_maps_to_integer_plus_half() {
        // With a unit-size voxel grid the centre of voxel (2,2,2) is at 2.5.
        let grid = unit_grid(8);
        let v = grid.world_to_voxel(Vec3::splat(2.5));
        assert!((v - Vec3::splat(2.5)).length() < 1.0e-5, "{v:?}");
    }

    #[test]
    fn trilinear_exact_at_voxel_centre() {
        let mut grid = unit_grid(4);
        grid.set_voxel(UVec3::new(1, 2, 3), 0.75, Vec3::new(1.0, 2.0, 3.0));
        // Query the exact centre of that voxel in level-0 voxel coords.
        let s = grid.fetch_trilinear(0, Vec3::new(1.5, 2.5, 3.5));
        assert!((s.opacity - 0.75).abs() < 1.0e-6, "opacity {}", s.opacity);
        assert!(
            (s.radiance - Vec3::new(1.0, 2.0, 3.0)).length() < 1.0e-6,
            "radiance {:?}",
            s.radiance
        );
    }

    #[test]
    fn trilinear_blends_two_neighbours() {
        let mut grid = unit_grid(4);
        grid.set_voxel(UVec3::new(0, 0, 0), 0.0, Vec3::ZERO);
        grid.set_voxel(UVec3::new(1, 0, 0), 1.0, Vec3::splat(2.0));
        // Halfway between the two voxel centres (x = 1.0) is a 50/50 blend.
        let s = grid.fetch_trilinear(0, Vec3::new(1.0, 0.5, 0.5));
        assert!((s.opacity - 0.5).abs() < 1.0e-6, "opacity {}", s.opacity);
        assert!(
            (s.radiance - Vec3::splat(1.0)).length() < 1.0e-6,
            "radiance {:?}",
            s.radiance
        );
    }

    #[test]
    fn out_of_bounds_sample_is_empty() {
        let mut grid = unit_grid(4);
        grid.set_voxel(UVec3::new(2, 2, 2), 1.0, Vec3::splat(5.0));
        let outside = grid.sample_world(0, Vec3::new(-10.0, -10.0, -10.0));
        assert_eq!(outside, VoxelSample::EMPTY);
        let far = grid.sample_world(0, Vec3::splat(100.0));
        assert_eq!(far, VoxelSample::EMPTY);
    }

    #[test]
    fn clamp_to_edge_outside_voxel_space() {
        let mut grid = unit_grid(4);
        grid.set_voxel(UVec3::new(0, 0, 0), 0.5, Vec3::splat(1.0));
        // Negative level-0 voxel coords clamp onto voxel (0,0,0).
        let s = grid.fetch_trilinear(0, Vec3::new(-5.0, -5.0, -5.0));
        assert!((s.opacity - 0.5).abs() < 1.0e-6, "opacity {}", s.opacity);
    }

    #[test]
    fn mip_dimensions_halve_each_level() {
        let mut grid = unit_grid(8);
        grid.build_mips();
        assert_eq!(grid.level_count(), 4); // 8 -> 4 -> 2 -> 1
        assert_eq!(grid.level(0).resolution(), UVec3::splat(8));
        assert_eq!(grid.level(1).resolution(), UVec3::splat(4));
        assert_eq!(grid.level(2).resolution(), UVec3::splat(2));
        assert_eq!(grid.level(3).resolution(), UVec3::splat(1));
    }

    #[test]
    fn mip_averages_children() {
        let mut grid = unit_grid(2);
        // Fill the eight level-0 voxels: four opaque, four empty.
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    let opaque = x == 0;
                    grid.set_voxel(
                        UVec3::new(x, y, z),
                        if opaque { 1.0 } else { 0.0 },
                        if opaque { Vec3::splat(4.0) } else { Vec3::ZERO },
                    );
                }
            }
        }
        grid.build_mips();
        // The single coarse voxel averages all eight: half opaque -> 0.5.
        let coarse = grid.level(1);
        assert_eq!(coarse.resolution(), UVec3::ONE);
        assert!((coarse.opacity()[0] - 0.5).abs() < 1.0e-6, "{}", coarse.opacity()[0]);
        assert!(
            (coarse.radiance()[0] - Vec3::splat(2.0)).length() < 1.0e-6,
            "{:?}",
            coarse.radiance()[0]
        );
    }

    #[test]
    fn mip_opacity_stays_in_unit_range() {
        let mut grid = unit_grid(8);
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    grid.set_voxel(UVec3::new(x, y, z), 1.0, Vec3::splat(10.0));
                }
            }
        }
        grid.build_mips();
        for lvl in 0..grid.level_count() {
            for &o in grid.level(lvl).opacity() {
                assert!((0.0..=1.0).contains(&o), "opacity {o} at level {lvl}");
            }
        }
    }

    #[test]
    fn lod_lerps_between_levels() {
        let mut grid = unit_grid(2);
        grid.set_voxel(UVec3::new(0, 0, 0), 1.0, Vec3::splat(8.0));
        grid.build_mips();
        let centre = grid.voxel_to_world(Vec3::splat(1.0));
        let fine = grid.sample_world_lod(centre, 0.0);
        let coarse = grid.sample_world_lod(centre, 1.0);
        let mid = grid.sample_world_lod(centre, 0.5);
        // The fractional LOD lies between the two integer levels.
        let lo = fine.opacity.min(coarse.opacity);
        let hi = fine.opacity.max(coarse.opacity);
        assert!(mid.opacity >= lo - 1.0e-6 && mid.opacity <= hi + 1.0e-6, "{}", mid.opacity);
    }

    #[test]
    fn degenerate_bounds_are_finite() {
        // A zero-size box must not divide by zero.
        let grid = VoxelGrid::new(UVec3::splat(4), Vec3::ZERO, Vec3::ZERO);
        let v = grid.world_to_voxel(Vec3::splat(0.0));
        assert!(v.is_finite(), "{v:?}");
        assert!(grid.voxel_size().is_finite() && grid.voxel_size() > 0.0);
    }

    #[test]
    fn from_level0_rejects_wrong_length() {
        let grid = VoxelGrid::from_level0(
            UVec3::splat(2),
            Vec3::ZERO,
            Vec3::splat(2.0),
            vec![1.0; 3],
            vec![Vec3::ZERO; 8],
        );
        // Falls back to a zeroed grid of the right size.
        assert_eq!(grid.level(0).opacity().len(), 8);
        assert!(grid.level(0).opacity().iter().all(|&o| o == 0.0));
    }
}
