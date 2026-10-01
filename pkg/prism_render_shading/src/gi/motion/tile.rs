//! Tile velocity reduction for motion-blur NeighborMax — CPU golden.
//!
//! McGuire-style per-object motion blur reconstructs each pixel by scattering
//! along the dominant local motion.  To bound the gather radius it works at a
//! coarse **tile** resolution (e.g. `16x16` px): first every tile is reduced to
//! its *maximum-magnitude* velocity (`TileMax`), then a `3x3` **NeighborMax**
//! pass spreads the fastest tile velocity to its neighbours so a fast object
//! bleeds its blur into the tiles it is about to cover.
//!
//! This module is the backend-neutral reference for both passes:
//!
//! * [`VelocityTileMap::tile_max`] — reduce a full-resolution velocity field to
//!   one representative (max-magnitude) velocity per tile.
//! * [`VelocityTileMap::neighbor_max`] — `3x3` max-magnitude over tiles, each
//!   result carrying the representative velocity **and** its blur radius
//!   (magnitude).
//!
//! # Conventions
//! * Velocities are in whatever screen units the caller supplies (typically
//!   **pixels** for motion blur).  "Max" always means max *magnitude*
//!   (`length`), never per-component max, compared via squared length so no
//!   `sqrt` is needed for the selection itself.
//! * The tile grid dimension is `ceil(extent / tile_size)`; a partial edge tile
//!   covers only the real pixels inside the frame (reads are clamp-to-edge).
//! * `tile_size`, `width`, and `height` are floored to at least `1`.
//! * The NeighborMax neighbourhood is `3x3` with the centre tile at index 4 and
//!   clamp-to-edge tile addressing; ties resolve to the incumbent (scan order
//!   seeded from the centre), so a uniform field is a no-op.
//! * Every function is deterministic and never emits a `NaN`; non-finite input
//!   velocities are sanitized to zero and the reported `radius` is always a
//!   finite, non-negative length.  Output `Vec`s are the only allocations and
//!   there is no RNG/IO/GPU/unsafe.

use alloc::vec::Vec;
use bevy_math::Vec2;

/// Default motion-blur tile edge, in pixels.
pub const DEFAULT_TILE_SIZE: u32 = 16;

/// Smallest extent / tile size permitted (guards divide-by-zero and empty grids).
const MIN_DIM: u32 = 1;

/// Replaces any non-finite component of a velocity with `0.0`.
#[inline]
fn sanitize_velocity(v: Vec2) -> Vec2 {
    Vec2::new(
        if v.x.is_finite() { v.x } else { 0.0 },
        if v.y.is_finite() { v.y } else { 0.0 },
    )
}

/// Floors a dimension to at least one.
#[inline]
fn clamp_dim(n: u32) -> u32 {
    n.max(MIN_DIM)
}

/// Number of tiles spanning `extent` pixels at `tile_size` px per tile.
///
/// Computes `ceil(extent / tile_size)` on the sanitized (floored-to-one) inputs,
/// so even a `1`-pixel frame yields a single tile.
#[inline]
pub fn tile_count(extent: u32, tile_size: u32) -> u32 {
    let extent = clamp_dim(extent);
    let ts = clamp_dim(tile_size);
    extent.div_ceil(ts)
}

/// Clamps an integer pixel coordinate into `[0, extent - 1]` (clamp-to-edge).
#[inline]
fn clamp_coord(c: i64, extent: usize) -> usize {
    if c < 0 {
        0
    } else {
        let c = c as usize;
        if c >= extent { extent - 1 } else { c }
    }
}

/// Representative velocity for a tile after NeighborMax, with its blur radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileVelocity {
    /// The max-magnitude velocity selected across the `3x3` tile neighbourhood.
    pub velocity: Vec2,
    /// Blur radius = `velocity.length()`; always finite and `>= 0`.
    pub radius: f32,
}

impl TileVelocity {
    /// Builds a tile result from a (sanitized) velocity, deriving its radius.
    #[inline]
    fn from_velocity(velocity: Vec2) -> Self {
        let velocity = sanitize_velocity(velocity);
        let radius = velocity.length();
        let radius = if radius.is_finite() { radius.max(0.0) } else { 0.0 };
        Self { velocity, radius }
    }
}

/// A coarse grid of per-tile max-magnitude velocities (the `TileMax` result).
#[derive(Clone, Debug, PartialEq)]
pub struct VelocityTileMap {
    tiles_x: usize,
    tiles_y: usize,
    tile_size: u32,
    max_velocity: Vec<Vec2>,
}

impl VelocityTileMap {
    /// Reduces a full-resolution velocity field to one max-magnitude velocity
    /// per `tile_size x tile_size` tile.
    ///
    /// `read(x, y)` returns the velocity at an in-bounds pixel; it is only ever
    /// called with clamp-to-edge coordinates in `[0, width) x [0, height)`.
    /// Dimensions and `tile_size` are floored to one.
    pub fn tile_max<F>(width: u32, height: u32, tile_size: u32, read: F) -> Self
    where
        F: Fn(usize, usize) -> Vec2,
    {
        let w = clamp_dim(width) as usize;
        let h = clamp_dim(height) as usize;
        let ts = clamp_dim(tile_size);
        let tiles_x = tile_count(width, tile_size) as usize;
        let tiles_y = tile_count(height, tile_size) as usize;
        let ts_usize = ts as usize;

        let mut max_velocity = Vec::with_capacity(tiles_x * tiles_y);
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let px0 = tx * ts_usize;
                let py0 = ty * ts_usize;
                let mut best = Vec2::ZERO;
                let mut best_len2 = 0.0f32;
                for ly in 0..ts_usize {
                    let py = py0 + ly;
                    if py >= h {
                        break;
                    }
                    for lx in 0..ts_usize {
                        let px = px0 + lx;
                        if px >= w {
                            break;
                        }
                        let v = sanitize_velocity(read(px, py));
                        let len2 = v.length_squared();
                        if len2 > best_len2 {
                            best = v;
                            best_len2 = len2;
                        }
                    }
                }
                max_velocity.push(best);
            }
        }

        Self {
            tiles_x,
            tiles_y,
            tile_size: ts,
            max_velocity,
        }
    }

    /// Number of tiles along the horizontal axis (always `>= 1`).
    #[inline]
    pub fn tiles_x(&self) -> usize {
        self.tiles_x
    }

    /// Number of tiles along the vertical axis (always `>= 1`).
    #[inline]
    pub fn tiles_y(&self) -> usize {
        self.tiles_y
    }

    /// The tile edge length (pixels) this map was built with (always `>= 1`).
    #[inline]
    pub fn tile_size(&self) -> u32 {
        self.tile_size
    }

    /// The raw `TileMax` velocity at tile `(tx, ty)`, clamp-to-edge.
    #[inline]
    pub fn max_at(&self, tx: usize, ty: usize) -> Vec2 {
        let tx = tx.min(self.tiles_x - 1);
        let ty = ty.min(self.tiles_y - 1);
        self.max_velocity[ty * self.tiles_x + tx]
    }

    /// NeighborMax for a single tile: the max-magnitude velocity across its
    /// `3x3` tile neighbourhood (clamp-to-edge), plus the resulting radius.
    #[inline]
    pub fn neighbor_max_at(&self, tx: usize, ty: usize) -> TileVelocity {
        let tx = tx.min(self.tiles_x - 1);
        let ty = ty.min(self.tiles_y - 1);
        let mut best = self.max_at(tx, ty);
        let mut best_len2 = best.length_squared();
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = clamp_coord(tx as i64 + dx, self.tiles_x);
                let ny = clamp_coord(ty as i64 + dy, self.tiles_y);
                let v = self.max_at(nx, ny);
                let len2 = v.length_squared();
                if len2 > best_len2 {
                    best = v;
                    best_len2 = len2;
                }
            }
        }
        TileVelocity::from_velocity(best)
    }

    /// Full NeighborMax pass: [`neighbor_max_at`](Self::neighbor_max_at) for
    /// every tile, row-major (`tiles_x * tiles_y` entries).
    pub fn neighbor_max(&self) -> Vec<TileVelocity> {
        let mut out = Vec::with_capacity(self.tiles_x * self.tiles_y);
        for ty in 0..self.tiles_y {
            for tx in 0..self.tiles_x {
                out.push(self.neighbor_max_at(tx, ty));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_count_is_ceil_div() {
        assert_eq!(tile_count(64, 16), 4);
        assert_eq!(tile_count(65, 16), 5);
        assert_eq!(tile_count(1, 16), 1);
        // Degenerate inputs floor to one.
        assert_eq!(tile_count(0, 0), 1);
    }

    #[test]
    fn tile_max_selects_largest_magnitude() {
        // 16x16 frame, one tile. A single fast pixel must dominate.
        let fast = Vec2::new(3.0, -4.0); // magnitude 5
        let read = |x: usize, y: usize| {
            if x == 7 && y == 9 {
                fast
            } else {
                Vec2::new(0.5, 0.0)
            }
        };
        let map = VelocityTileMap::tile_max(16, 16, 16, read);
        assert_eq!(map.tiles_x(), 1);
        assert_eq!(map.tiles_y(), 1);
        assert!((map.max_at(0, 0) - fast).length() < 1e-6);
    }

    #[test]
    fn tile_max_ignores_pixels_outside_frame() {
        // 20x20 frame with tile 16 -> 2x2 tiles; the bottom/right tiles are
        // partial. Put a fast pixel only in the top-left tile.
        let read = |x: usize, y: usize| {
            if x == 1 && y == 1 {
                Vec2::new(10.0, 0.0)
            } else {
                Vec2::ZERO
            }
        };
        let map = VelocityTileMap::tile_max(20, 20, 16, read);
        assert_eq!(map.tiles_x(), 2);
        assert_eq!(map.tiles_y(), 2);
        assert!((map.max_at(0, 0) - Vec2::new(10.0, 0.0)).length() < 1e-6);
        // The partial tiles contain no fast pixel -> zero.
        assert!(map.max_at(1, 1).length() < 1e-7);
    }

    #[test]
    fn neighbor_max_spreads_fastest_tile() {
        // 48x16 -> 3x1 tiles. Fast motion only in the middle tile should also
        // appear in its left and right neighbours after NeighborMax.
        let read = |x: usize, _y: usize| {
            if (16..32).contains(&x) {
                Vec2::new(6.0, 0.0)
            } else {
                Vec2::ZERO
            }
        };
        let map = VelocityTileMap::tile_max(48, 16, 16, read);
        assert_eq!(map.tiles_x(), 3);
        let nm = map.neighbor_max();
        for t in &nm {
            assert!((t.velocity - Vec2::new(6.0, 0.0)).length() < 1e-6, "tile {:?}", t.velocity);
            assert!((t.radius - 6.0).abs() < 1e-6, "radius {:?}", t.radius);
        }
    }

    #[test]
    fn radius_is_magnitude_and_nonnegative() {
        let read = |_x: usize, _y: usize| Vec2::new(3.0, 4.0);
        let map = VelocityTileMap::tile_max(16, 16, 16, read);
        let t = map.neighbor_max_at(0, 0);
        assert!((t.radius - 5.0).abs() < 1e-6);
        assert!(t.radius >= 0.0);
    }

    #[test]
    fn clamp_edge_access_is_safe() {
        let map = VelocityTileMap::tile_max(32, 32, 16, |_x, _y| Vec2::new(1.0, 0.0));
        // Out-of-range tile indices clamp instead of panicking.
        let a = map.max_at(99, 99);
        let b = map.neighbor_max_at(99, 99);
        assert!(a.is_finite());
        assert!(b.velocity.is_finite());
    }

    #[test]
    fn degenerate_dims_single_tile() {
        let map = VelocityTileMap::tile_max(0, 0, 0, |_x, _y| Vec2::new(2.0, 0.0));
        assert_eq!(map.tiles_x(), 1);
        assert_eq!(map.tiles_y(), 1);
        assert_eq!(map.tile_size(), 1);
        assert!((map.max_at(0, 0) - Vec2::new(2.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn non_finite_velocity_sanitized() {
        let read = |x: usize, _y: usize| {
            if x == 0 {
                Vec2::new(f32::NAN, f32::INFINITY)
            } else {
                Vec2::new(0.0, 1.0)
            }
        };
        let map = VelocityTileMap::tile_max(16, 16, 16, read);
        let v = map.max_at(0, 0);
        assert!(v.is_finite());
        // The NaN pixel sanitizes to zero, so the clean (0,1) pixels win.
        assert!((v - Vec2::new(0.0, 1.0)).length() < 1e-6);
        let t = map.neighbor_max_at(0, 0);
        assert!(t.radius.is_finite());
    }
}
