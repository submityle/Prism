//! Sparse brick grid of signed distances for a global distance field — CPU
//! golden.
//!
//! Unreal Engine's *Lumen* global distance field does not store one dense
//! volume per object; it merges every mesh distance field into a single,
//! world-space clipmap of fixed-size **bricks**.  Each brick is a small cube of
//! voxels (Lumen uses `8^3`); only bricks that straddle or sit near a surface
//! are actually allocated, and everything else is "empty space" described by a
//! single conservative distance.  Sphere/cone tracing then walks this sparse
//! structure, reading a trilinearly filtered distance wherever a brick exists
//! and taking a safe conservative step where one does not.  This module is the
//! backend-neutral, GPU-free reference for that structure.
//!
//! [`BrickGrid`] owns the sparse set of [`SparseBrick`]s plus the world
//! placement (grid `origin`, `voxel_size`, and `brick_dim`).  A brick stores
//! `(brick_dim + 1)^3` signed-distance samples at its voxel *corners* — the
//! one-voxel apron on the max side lets trilinear reconstruction stay entirely
//! inside a single brick, exactly like a bordered GPU brick atlas.
//!
//! # Conventions
//! * **Layout.** Within a brick, corner `(i, j, k)` with `0 <= i,j,k <=
//!   brick_dim` is stored row-major with `i` fastest:
//!   `index = i + j*n + k*n*n` where `n = brick_dim + 1`.  This matches the
//!   natural scan order of a 3D brick texture, so the CPU buffer and a GPU twin
//!   share a byte layout.
//! * **World <-> voxel <-> brick.** `world_to_voxel` maps a world point to
//!   continuous *global* voxel coordinates (`(p - origin) / voxel_size`);
//!   `voxel_to_world` is its exact inverse.  `world_to_brick` floors the global
//!   voxel coordinate divided by `brick_dim` to the integer brick that contains
//!   the point.  `voxel_size` is clamped away from zero so these maps are always
//!   finite.
//! * **Sign.** Negative distance is inside geometry, positive outside, zero on
//!   the surface — the standard SDF / UE convention.
//! * **Sampling.** [`BrickGrid::sample_distance`] trilinearly blends the eight
//!   corners around the query point, clamping the in-brick coordinate to the
//!   brick (clamp-to-edge).  When the containing brick is *absent* it returns
//!   the grid's [`empty_distance`](BrickGrid::empty_distance): a conservative
//!   lower bound on the true distance in unallocated space, so a sphere march
//!   can take it as a safe step without ever overshooting a nearby surface.
//! * **Determinism / safety.** Every function is a pure, deterministic
//!   computation: no RNG, no I/O, no GPU, no `unsafe`.  Bricks are held in a
//!   [`BTreeMap`] keyed by brick coordinate so iteration order is stable.  Every
//!   divisor is guarded and no path can produce `NaN`.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use bevy_math::{IVec3, Vec3};

/// Default voxels-per-axis of a brick, matching Lumen's `8^3` bricks.
pub const DEFAULT_BRICK_DIM: u32 = 8;

/// Conservative "very far / empty" distance used when no better bound is known.
///
/// Large but finite so arithmetic on it never overflows to infinity or `NaN`.
pub const FAR_DISTANCE: f32 = 1.0e18;

/// A single allocated brick: a cube of signed-distance corner samples.
///
/// The sample buffer has exactly `(brick_dim + 1)^3` entries laid out as
/// described in the [module docs](self).  Construct bricks through
/// [`BrickGrid::insert_brick`], which enforces that length invariant.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseBrick {
    /// Integer brick coordinate within the grid (may be negative).
    coord: IVec3,
    /// Row-major corner distances, `i` fastest; length `(brick_dim + 1)^3`.
    distances: Vec<f32>,
}

impl SparseBrick {
    /// The integer brick coordinate this brick occupies.
    #[inline]
    pub fn coord(&self) -> IVec3 {
        self.coord
    }

    /// The raw row-major corner-distance buffer (`i` fastest).
    #[inline]
    pub fn distances(&self) -> &[f32] {
        &self.distances
    }
}

/// A sparse, world-space brick grid of signed distances.
///
/// See the [module docs](self) for the layout, mapping, and sampling
/// conventions.  Build an empty grid with [`new`](BrickGrid::new) and fill it
/// with [`insert_brick`](BrickGrid::insert_brick), or bake one from a distance
/// callback with [`from_distance_fn`](BrickGrid::from_distance_fn).
#[derive(Clone, Debug, PartialEq)]
pub struct BrickGrid {
    /// World position of global voxel coordinate `(0, 0, 0)`.
    origin: Vec3,
    /// World edge length of a single voxel; always strictly positive.
    voxel_size: f32,
    /// Voxels per brick axis; always `>= 1`.
    brick_dim: u32,
    /// Conservative distance returned for queries in unallocated space.
    empty_distance: f32,
    /// Allocated bricks keyed by `(x, y, z)` brick coordinate for stable order.
    bricks: BTreeMap<(i32, i32, i32), SparseBrick>,
}

impl BrickGrid {
    /// Smallest voxel size permitted; clamps away degenerate zero/negative
    /// spacing so world/voxel mapping never divides by zero.
    const MIN_VOXEL_SIZE: f32 = 1.0e-6;

    /// Creates an empty grid with the given placement.
    ///
    /// `voxel_size` is clamped to a tiny positive floor and `brick_dim` to at
    /// least `1`.  The grid starts with no bricks; unallocated space reports
    /// [`FAR_DISTANCE`].  Use [`with_empty_distance`](Self::with_empty_distance)
    /// to supply a tighter conservative bound for sphere tracing across gaps.
    pub fn new(origin: Vec3, voxel_size: f32, brick_dim: u32) -> Self {
        Self {
            origin,
            voxel_size: voxel_size.max(Self::MIN_VOXEL_SIZE),
            brick_dim: brick_dim.max(1),
            empty_distance: FAR_DISTANCE,
            bricks: BTreeMap::new(),
        }
    }

    /// Returns the grid with its empty-space distance replaced.
    ///
    /// The value is clamped to be non-negative.  Set it to a lower bound on the
    /// true distance of all unallocated space (for a culled sparse grid, the
    /// cull band) so a march across gaps takes safe, non-overshooting steps.
    #[inline]
    pub fn with_empty_distance(mut self, empty_distance: f32) -> Self {
        self.empty_distance = empty_distance.max(0.0);
        self
    }

    /// World position of global voxel `(0, 0, 0)`.
    #[inline]
    pub fn origin(&self) -> Vec3 {
        self.origin
    }

    /// World edge length of a single voxel (strictly positive).
    #[inline]
    pub fn voxel_size(&self) -> f32 {
        self.voxel_size
    }

    /// Voxels per brick axis (`>= 1`).
    #[inline]
    pub fn brick_dim(&self) -> u32 {
        self.brick_dim
    }

    /// Corner samples per brick axis, `brick_dim + 1`.
    #[inline]
    pub fn nodes_per_axis(&self) -> u32 {
        self.brick_dim + 1
    }

    /// Conservative distance reported for unallocated space.
    #[inline]
    pub fn empty_distance(&self) -> f32 {
        self.empty_distance
    }

    /// Number of currently allocated bricks.
    #[inline]
    pub fn brick_count(&self) -> usize {
        self.bricks.len()
    }

    /// World edge length of one brick, `brick_dim * voxel_size`.
    #[inline]
    pub fn brick_world_size(&self) -> f32 {
        self.brick_dim as f32 * self.voxel_size
    }

    /// Maps a world point to continuous *global* voxel coordinates.
    ///
    /// Not clamped, so it is the exact inverse of
    /// [`voxel_to_world`](Self::voxel_to_world); callers that need in-bounds
    /// coordinates clamp themselves.
    #[inline]
    pub fn world_to_voxel(&self, world: Vec3) -> Vec3 {
        (world - self.origin) / self.voxel_size
    }

    /// Maps continuous global voxel coordinates back to a world position; the
    /// exact inverse of [`world_to_voxel`](Self::world_to_voxel).
    #[inline]
    pub fn voxel_to_world(&self, voxel: Vec3) -> Vec3 {
        self.origin + voxel * self.voxel_size
    }

    /// Integer brick coordinate containing `world` (floor of the global voxel
    /// coordinate divided by `brick_dim`).
    #[inline]
    pub fn world_to_brick(&self, world: Vec3) -> IVec3 {
        let g = self.world_to_voxel(world);
        let bf = (g / self.brick_dim as f32).floor();
        IVec3::new(bf.x as i32, bf.y as i32, bf.z as i32)
    }

    /// World position of a brick's minimum corner (its local `(0,0,0)` node).
    #[inline]
    pub fn brick_origin_world(&self, coord: IVec3) -> Vec3 {
        let v = coord.as_vec3() * self.brick_dim as f32;
        self.voxel_to_world(v)
    }

    /// Borrows the brick at `coord`, if one is allocated there.
    #[inline]
    pub fn brick_at(&self, coord: IVec3) -> Option<&SparseBrick> {
        self.bricks.get(&(coord.x, coord.y, coord.z))
    }

    /// Flattens in-brick corner indices to a buffer offset (`i` fastest).
    #[inline]
    fn node_index(&self, i: u32, j: u32, k: u32) -> usize {
        let n = self.nodes_per_axis() as usize;
        (i as usize) + (j as usize) * n + (k as usize) * n * n
    }

    /// Inserts (or replaces) a brick, enforcing the corner-count invariant.
    ///
    /// The supplied buffer is resized to exactly `(brick_dim + 1)^3` entries,
    /// padding any shortfall with [`FAR_DISTANCE`] and dropping any excess, so
    /// the length invariant always holds and later indexing is in bounds.
    pub fn insert_brick(&mut self, coord: IVec3, mut distances: Vec<f32>) {
        let n = self.nodes_per_axis() as usize;
        distances.resize(n * n * n, FAR_DISTANCE);
        self.bricks.insert(
            (coord.x, coord.y, coord.z),
            SparseBrick { coord, distances },
        );
    }

    /// Reads a corner sample from an allocated brick, clamping the integer
    /// corner index into `[0, brick_dim]` (clamp-to-edge).
    #[inline]
    fn corner_clamped(&self, brick: &SparseBrick, i: i32, j: i32, k: i32) -> f32 {
        let dim = self.brick_dim as i32;
        let ci = i.clamp(0, dim) as u32;
        let cj = j.clamp(0, dim) as u32;
        let ck = k.clamp(0, dim) as u32;
        brick.distances[self.node_index(ci, cj, ck)]
    }

    /// Trilinearly samples the signed distance at an arbitrary world point.
    ///
    /// If the brick containing the point is allocated, the eight corners around
    /// the (clamped) in-brick coordinate are blended with the fractional
    /// position, matching a clamped GPU brick fetch.  If the brick is absent the
    /// conservative [`empty_distance`](Self::empty_distance) is returned, which
    /// a sphere march can safely use as a step length.
    pub fn sample_distance(&self, world: Vec3) -> f32 {
        let dim = self.brick_dim as f32;
        let g = self.world_to_voxel(world);
        let bf = (g / dim).floor();
        let coord = IVec3::new(bf.x as i32, bf.y as i32, bf.z as i32);

        let brick = match self.brick_at(coord) {
            Some(b) => b,
            None => return self.empty_distance,
        };

        // In-brick coordinate in [0, brick_dim]; clamp defends against
        // float round-off landing a touch outside the owned range.
        let local = (g - bf * dim).clamp(Vec3::ZERO, Vec3::splat(dim));
        let base = local.floor();
        let frac = (local - base).clamp(Vec3::ZERO, Vec3::ONE);

        let x0 = base.x as i32;
        let y0 = base.y as i32;
        let z0 = base.z as i32;
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let z1 = z0 + 1;

        let c000 = self.corner_clamped(brick, x0, y0, z0);
        let c100 = self.corner_clamped(brick, x1, y0, z0);
        let c010 = self.corner_clamped(brick, x0, y1, z0);
        let c110 = self.corner_clamped(brick, x1, y1, z0);
        let c001 = self.corner_clamped(brick, x0, y0, z1);
        let c101 = self.corner_clamped(brick, x1, y0, z1);
        let c011 = self.corner_clamped(brick, x0, y1, z1);
        let c111 = self.corner_clamped(brick, x1, y1, z1);

        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let c00 = lerp(c000, c100, frac.x);
        let c10 = lerp(c010, c110, frac.x);
        let c01 = lerp(c001, c101, frac.x);
        let c11 = lerp(c011, c111, frac.x);
        let c0 = lerp(c00, c10, frac.y);
        let c1 = lerp(c01, c11, frac.y);
        lerp(c0, c1, frac.z)
    }

    /// Central-difference gradient of the sampled field, returned as a unit
    /// vector (the surface normal for a true SDF).
    ///
    /// The stencil step is half a voxel so it stays inside one cell.  A
    /// vanishing gradient (flat region or exact symmetry point) falls back to
    /// `+Y`, so the result is never a zero vector or `NaN`.
    pub fn gradient(&self, world: Vec3) -> Vec3 {
        let h = (self.voxel_size * 0.5).max(Self::MIN_VOXEL_SIZE);
        let dx = self.sample_distance(world + Vec3::new(h, 0.0, 0.0))
            - self.sample_distance(world - Vec3::new(h, 0.0, 0.0));
        let dy = self.sample_distance(world + Vec3::new(0.0, h, 0.0))
            - self.sample_distance(world - Vec3::new(0.0, h, 0.0));
        let dz = self.sample_distance(world + Vec3::new(0.0, 0.0, h))
            - self.sample_distance(world - Vec3::new(0.0, 0.0, h));
        let grad = Vec3::new(dx, dy, dz);
        let len = grad.length();
        if len > f32::MIN_POSITIVE {
            grad / len
        } else {
            Vec3::Y
        }
    }

    /// Bakes a grid by evaluating a distance callback over a range of bricks.
    ///
    /// Every brick coordinate in the inclusive box `[brick_min, brick_max]` is
    /// considered.  For each, all `(brick_dim + 1)^3` corner distances are
    /// sampled from `f` (which receives world positions).  When `cull_band` is
    /// positive, a brick whose minimum absolute distance exceeds `cull_band` is
    /// skipped — it is far enough from any surface to be treated as empty — and
    /// the grid's [`empty_distance`](Self::empty_distance) is set to `cull_band`
    /// so marches across the culled gaps step safely.  A non-positive
    /// `cull_band` keeps every brick (a dense bake) and leaves the empty
    /// distance at [`FAR_DISTANCE`].
    ///
    /// `brick_min`/`brick_max` are ordered component-wise, so a caller may pass
    /// them swapped.  `voxel_size` and `brick_dim` are clamped exactly as in
    /// [`new`](Self::new).
    pub fn from_distance_fn(
        origin: Vec3,
        voxel_size: f32,
        brick_dim: u32,
        brick_min: IVec3,
        brick_max: IVec3,
        cull_band: f32,
        f: impl Fn(Vec3) -> f32,
    ) -> Self {
        let mut grid = Self::new(origin, voxel_size, brick_dim);
        if cull_band > 0.0 {
            grid.empty_distance = cull_band;
        }

        let lo = brick_min.min(brick_max);
        let hi = brick_min.max(brick_max);
        let n = grid.nodes_per_axis();
        let count = (n as usize) * (n as usize) * (n as usize);

        for bz in lo.z..=hi.z {
            for by in lo.y..=hi.y {
                for bx in lo.x..=hi.x {
                    let coord = IVec3::new(bx, by, bz);
                    let brick_origin = grid.brick_origin_world(coord);
                    let mut distances = Vec::with_capacity(count);
                    let mut min_abs = f32::INFINITY;
                    for k in 0..n {
                        for j in 0..n {
                            for i in 0..n {
                                let node = brick_origin
                                    + Vec3::new(i as f32, j as f32, k as f32)
                                        * grid.voxel_size;
                                let d = f(node);
                                let d = if d.is_finite() { d } else { FAR_DISTANCE };
                                let a = if d < 0.0 { -d } else { d };
                                if a < min_abs {
                                    min_abs = a;
                                }
                                distances.push(d);
                            }
                        }
                    }
                    if cull_band > 0.0 && min_abs > cull_band {
                        continue;
                    }
                    grid.insert_brick(coord, distances);
                }
            }
        }
        grid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact analytic sphere distance, used as baking ground truth.
    fn sphere(p: Vec3, center: Vec3, radius: f32) -> f32 {
        (p - center).length() - radius
    }

    fn sphere_grid() -> BrickGrid {
        // Unit sphere at the origin, voxel 0.1, 8^3 bricks, dense bake over a
        // range that comfortably brackets the sphere.
        BrickGrid::from_distance_fn(
            Vec3::ZERO,
            0.1,
            DEFAULT_BRICK_DIM,
            IVec3::splat(-3),
            IVec3::splat(2),
            0.0,
            |p| sphere(p, Vec3::ZERO, 1.0),
        )
    }

    #[test]
    fn world_voxel_roundtrip_is_exact() {
        let grid = BrickGrid::new(Vec3::new(1.0, -2.0, 0.5), 0.25, 8);
        for p in [
            Vec3::new(0.3, -0.7, 1.1),
            Vec3::new(-1.5, 0.0, 0.9),
            Vec3::ZERO,
        ] {
            let back = grid.voxel_to_world(grid.world_to_voxel(p));
            assert!((back - p).length() < 1.0e-5, "{p:?} -> {back:?}");
        }
    }

    #[test]
    fn trilinear_reproduces_corner_values() {
        // Baked from a known function; sampling exactly at a corner must return
        // the stored (and therefore analytic) value.
        let grid = sphere_grid();
        let coord = IVec3::new(0, 0, 0);
        let brick = grid.brick_at(coord).expect("brick present");
        let origin = grid.brick_origin_world(coord);
        for &(i, j, k) in &[(0u32, 0u32, 0u32), (3, 5, 2), (8, 8, 8), (8, 0, 4)] {
            let world = origin + Vec3::new(i as f32, j as f32, k as f32) * grid.voxel_size();
            let sampled = grid.sample_distance(world);
            let stored = brick.distances()[grid.node_index(i, j, k)];
            assert!(
                (sampled - stored).abs() < 1.0e-4,
                "corner ({i},{j},{k}): sampled {sampled}, stored {stored}"
            );
        }
    }

    #[test]
    fn trilinear_matches_analytic_sphere() {
        let grid = sphere_grid();
        for p in [
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.0, 0.9, 0.0),
            Vec3::new(0.3, -0.3, 0.3),
            Vec3::new(1.2, 0.0, 0.0),
        ] {
            let got = grid.sample_distance(p);
            let truth = sphere(p, Vec3::ZERO, 1.0);
            assert!(
                (got - truth).abs() < 0.02,
                "{p:?}: got {got}, truth {truth}"
            );
        }
    }

    #[test]
    fn absent_brick_reports_empty_distance() {
        let grid = sphere_grid();
        // Far outside the baked brick range.
        let far = Vec3::splat(1000.0);
        assert_eq!(grid.sample_distance(far), grid.empty_distance());
        assert!(grid.brick_at(grid.world_to_brick(far)).is_none());
    }

    #[test]
    fn culling_drops_empty_bricks_and_sets_bound() {
        // A tight cull band keeps only bricks straddling the surface.
        let band = 0.3;
        let culled = BrickGrid::from_distance_fn(
            Vec3::ZERO,
            0.1,
            DEFAULT_BRICK_DIM,
            IVec3::splat(-3),
            IVec3::splat(2),
            band,
            |p| sphere(p, Vec3::ZERO, 1.0),
        );
        let dense = sphere_grid();
        assert!(culled.brick_count() < dense.brick_count());
        assert_eq!(culled.empty_distance(), band);
        // Open space between the shell and infinity now reads the conservative
        // bound, not a bogus far value.
        assert_eq!(culled.sample_distance(Vec3::splat(5.0)), band);
    }

    #[test]
    fn insert_brick_enforces_length() {
        let mut grid = BrickGrid::new(Vec3::ZERO, 1.0, 2);
        grid.insert_brick(IVec3::ZERO, Vec::new());
        let n = grid.nodes_per_axis() as usize;
        let brick = grid.brick_at(IVec3::ZERO).unwrap();
        assert_eq!(brick.distances().len(), n * n * n);
        assert!(brick.distances().iter().all(|&d| d == FAR_DISTANCE));
    }

    #[test]
    fn degenerate_voxel_size_stays_finite() {
        let grid = BrickGrid::new(Vec3::ZERO, 0.0, 0);
        assert!(grid.voxel_size() > 0.0);
        assert_eq!(grid.brick_dim(), 1);
        let v = grid.world_to_voxel(Vec3::splat(1.0));
        assert!(v.is_finite());
    }

    #[test]
    fn gradient_points_outward_on_sphere() {
        let grid = sphere_grid();
        let p = Vec3::new(0.7, 0.0, 0.0);
        let g = grid.gradient(p);
        assert!(g.is_finite());
        assert!((g.length() - 1.0).abs() < 1.0e-3);
        // Outward normal on +X side of a sphere points roughly +X.
        assert!(g.x > 0.8, "{g:?}");
    }

    #[test]
    fn sampling_is_deterministic() {
        let a = sphere_grid();
        let b = sphere_grid();
        let p = Vec3::new(0.4, -0.2, 0.6);
        assert_eq!(a.sample_distance(p), b.sample_distance(p));
    }
}
