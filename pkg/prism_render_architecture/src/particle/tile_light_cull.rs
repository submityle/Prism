//! `Forward+` 2D screen-space tile light culling — the deterministic `CPU`
//! gold reference for "split the framebuffer into fixed pixel tiles and bin each
//! spherical light into the tiles its screen footprint touches" (design §8.3
//! *Scene* category, consumed by the §17 lighting closures).
//!
//! # What this module is
//!
//! This is the `Forward+` (a.k.a. *tiled forward*) light-binning pass used by
//! `Frostbite`, `DOOM 2016`, and Apple's `TBDR` tiled shaders: the screen is
//! divided into an `NxN`-pixel tile grid, each tile spawns a small view-space
//! sub-frustum (its screen rectangle back-projected through the camera apex,
//! clamped by a near and far depth), and every spherical light is tested against
//! that sub-frustum. Surviving `(tile, light)` pairs are packed into a compact
//! `offset + count + indices` list a shading kernel later reads per pixel.
//!
//! # How it is *different* from `light_clustered`
//!
//! The sibling `light_clustered` module is a **3D froxel cluster** binner: it
//! slices the frustum along view `Z` into depth *slices* and owns
//! `PunctualLight` / `ClusterGrid` / `ClusterCoord` / `depth_slice` /
//! `cluster_coord` / `slice_boundaries`. This module is strictly **2D screen
//! tiles**: there is **no** depth-slice axis, no 3D cluster coordinate, and none
//! of those types are imported or reused. A single optional per-tile min/max
//! depth range only *tightens* the one near/far pair of each 2D tile frustum; it
//! never subdivides depth into froxels. The two files are alternative
//! light-binning strategies feeding one shading closure, not shared code — this
//! module defines its own minimal [`Vec3`] and [`SphereLight`] and imports only
//! `gpu_layout` for the `std430` byte-size rule.
//!
//! # Numerics
//!
//! All math is spelled out on the hand-rolled [`Vec3`]. The only irrational
//! operation is `sqrt` (via [`Vec3::length`], used to normalize the tile-frustum
//! plane normals); every intersection test is a `dot` product compared against a
//! radius, and every tile-count derivation is integer `div_ceil`. No `sin`,
//! `cos`, `tan`, `atan`, `exp`, `ln`, `powf`, `ceil`, or `round` is ever called:
//! the camera field of view enters as two precomputed half-extent *slopes*
//! (`tan(fov/2)` values the author bakes once). Degenerate inputs (zero-size
//! framebuffer, zero tile size, inverted depth range, empty light set) are
//! clamped or fall through to an empty result rather than panicking or dividing
//! by zero, so the reference stays bit-reproducible against a future `GPU`
//! tile-culling kernel.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE};

/// Shared floating-point guard tolerance for this module.
///
/// Denominators (framebuffer extents, plane normal lengths) are floored to this
/// value before a divide, and a plane normal shorter than this collapses to the
/// zero vector instead of exploding. It is intentionally small: it only rejects
/// truly degenerate geometry, never legitimate thin tiles.
const GUARD_EPS: f32 = 1e-6;

// ---------------------------------------------------------------------------
// Minimal view-space vector (self-contained; not shared with `light_clustered`).
// ---------------------------------------------------------------------------

/// A hand-rolled 3-component vector in right-handed **view space**, where the
/// camera sits at the origin looking down **+Z** and view depth increases with
/// distance.
///
/// This is deliberately a *local* type: the `Forward+` binner must not depend on
/// the froxel module's `Vec3`, so it carries its own tiny algebra. Component-wise
/// add/subtract are named [`Vec3::plus`] / [`Vec3::minus`] to avoid colliding
/// with the `core::ops` traits and to keep the arithmetic explicit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    /// View-space X (right).
    pub x: f32,
    /// View-space Y (up).
    pub y: f32,
    /// View-space Z (forward / depth).
    pub z: f32,
}

impl Vec3 {
    /// The zero vector.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Builds a vector from its components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Component-wise sum (`self + rhs`), named to avoid the `Add` trait.
    #[must_use]
    pub fn plus(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    /// Component-wise difference (`self - rhs`), named to avoid the `Sub` trait.
    #[must_use]
    pub fn minus(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }

    /// Uniform scale by a scalar.
    #[must_use]
    pub fn scale(self, factor: f32) -> Self {
        Self::new(self.x * factor, self.y * factor, self.z * factor)
    }

    /// Euclidean dot product.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Right-handed cross product (`self × rhs`).
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    /// Euclidean length (the module's only irrational operation, via `sqrt`).
    #[must_use]
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    /// Returns the unit-length version of the vector, or [`Vec3::ZERO`] when the
    /// vector is shorter than [`GUARD_EPS`] and cannot be safely normalized.
    #[must_use]
    pub fn normalize_or_zero(self) -> Self {
        let len = self.length();
        if len > GUARD_EPS {
            self.scale(1.0 / len)
        } else {
            Self::ZERO
        }
    }
}

// ---------------------------------------------------------------------------
// Spherical light (self-contained; distinct from `light_clustered::PunctualLight`).
// ---------------------------------------------------------------------------

/// One bounding-sphere light in **view space**: a center and a radius of
/// influence.
///
/// `Forward+` culling only needs each light's *conservative bounding sphere* —
/// the radius past which its contribution is clamped to zero. This is a
/// deliberately smaller type than the froxel module's punctual light (no cone
/// cosines, no color payload, no [`LightKind`]-style discriminator): binning
/// cares about geometry, and shading pulls the full material later.
///
/// [`LightKind`]: crate::particle::light_clustered::LightKind
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereLight {
    /// View-space center of the light's influence sphere.
    pub center: Vec3,
    /// Radius of influence; a non-positive radius marks an inactive light.
    pub radius: f32,
}

impl SphereLight {
    /// Builds a spherical light from a view-space center and influence radius.
    #[must_use]
    pub fn new(center: Vec3, radius: f32) -> Self {
        Self { center, radius }
    }

    /// Whether the light has a strictly positive radius and should be binned.
    #[must_use]
    pub fn is_active(self) -> bool {
        self.radius > GUARD_EPS
    }
}

// ---------------------------------------------------------------------------
// Frustum plane through the camera apex.
// ---------------------------------------------------------------------------

/// One side plane of a tile sub-frustum, passing through the camera apex (the
/// view-space origin), so its plane constant is zero and the signed distance of
/// a point is simply the `dot` of the point with the inward unit normal.
///
/// A positive [`Plane::signed_distance`] means the point lies on the *inside*
/// (frustum) half-space; a value below `-radius` means a sphere of that radius
/// is fully outside.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    normal: Vec3,
}

impl Plane {
    /// Builds a plane through the origin from a (not necessarily unit) inward
    /// normal, normalizing it so distances are true Euclidean distances.
    #[must_use]
    pub fn from_inward_normal(normal: Vec3) -> Self {
        Self {
            normal: normal.normalize_or_zero(),
        }
    }

    /// The inward-pointing unit normal.
    #[must_use]
    pub fn normal(self) -> Vec3 {
        self.normal
    }

    /// Signed distance of a point to the plane; positive is inside the frustum.
    #[must_use]
    pub fn signed_distance(self, point: Vec3) -> f32 {
        self.normal.dot(point)
    }
}

// ---------------------------------------------------------------------------
// Optional per-tile depth tightening.
// ---------------------------------------------------------------------------

/// An optional per-tile view-depth range used to *tighten* a tile frustum's
/// near/far planes (for example from a pre-pass depth buffer's per-tile min/max).
///
/// This never introduces a depth-slice axis — it only replaces the single
/// near/far pair of one 2D tile frustum, keeping this module strictly screen-tile
/// based and orthogonal to the froxel cluster binner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthRange {
    /// Nearest view depth present in the tile.
    pub min: f32,
    /// Farthest view depth present in the tile.
    pub max: f32,
}

impl DepthRange {
    /// Builds a depth range, ordering the two edges so `min <= max`.
    #[must_use]
    pub fn new(min: f32, max: f32) -> Self {
        if min <= max {
            Self { min, max }
        } else {
            Self { min: max, max: min }
        }
    }
}

// ---------------------------------------------------------------------------
// Tile grid.
// ---------------------------------------------------------------------------

/// The screen-space tile grid: a framebuffer resolution partitioned into fixed
/// square pixel tiles, plus the camera half-extent slopes needed to back-project
/// a tile's screen rectangle into a view-space sub-frustum.
///
/// The horizontal tile count is `div_ceil(width, tile_size)` and likewise for
/// the vertical axis, so the rightmost / bottommost tiles may be partial and are
/// clamped to the framebuffer edge. `slope_x` and `slope_y` are `tan(fov/2)`
/// half-extents the author bakes offline; at view depth `z` the full frustum
/// half-width is `z * slope_x`, and a normalized-device `X` in `[-1, 1]` maps to
/// view `x = ndc_x * slope_x * z`. This module never calls `tan`.
#[derive(Clone, Debug, PartialEq)]
pub struct TileGrid {
    width: u32,
    height: u32,
    tile_size: u32,
    tile_count_x: u32,
    tile_count_y: u32,
    slope_x: f32,
    slope_y: f32,
}

impl TileGrid {
    /// Builds a tile grid, clamping degenerate inputs (zero framebuffer extent,
    /// zero tile size, non-positive slopes) up to a valid minimum so the grid is
    /// always usable and always has at least one tile.
    #[must_use]
    pub fn new(width: u32, height: u32, tile_size: u32, slope_x: f32, slope_y: f32) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let tile_size = tile_size.max(1);
        Self {
            width,
            height,
            tile_size,
            tile_count_x: width.div_ceil(tile_size),
            tile_count_y: height.div_ceil(tile_size),
            slope_x: slope_x.max(GUARD_EPS),
            slope_y: slope_y.max(GUARD_EPS),
        }
    }

    /// Number of tile columns along screen `X` (`div_ceil(width, tile_size)`).
    #[must_use]
    pub fn tile_count_x(&self) -> u32 {
        self.tile_count_x
    }

    /// Number of tile rows along screen `Y` (`div_ceil(height, tile_size)`).
    #[must_use]
    pub fn tile_count_y(&self) -> u32 {
        self.tile_count_y
    }

    /// Total number of tiles (`tile_count_x * tile_count_y`).
    #[must_use]
    pub fn tile_count(&self) -> u32 {
        self.tile_count_x.saturating_mul(self.tile_count_y)
    }

    /// The pixel rectangle `(x0, y0, x1, y1)` a tile covers, clamped to the
    /// framebuffer so partial edge tiles never exceed the resolution. Tile
    /// coordinates outside the grid are clamped to the last valid tile.
    #[must_use]
    pub fn tile_pixel_rect(&self, tile_x: u32, tile_y: u32) -> (u32, u32, u32, u32) {
        let tx = tile_x.min(self.tile_count_x - 1);
        let ty = tile_y.min(self.tile_count_y - 1);
        let x0 = tx.saturating_mul(self.tile_size).min(self.width);
        let y0 = ty.saturating_mul(self.tile_size).min(self.height);
        let x1 = tx
            .saturating_add(1)
            .saturating_mul(self.tile_size)
            .min(self.width);
        let y1 = ty
            .saturating_add(1)
            .saturating_mul(self.tile_size)
            .min(self.height);
        (x0, y0, x1, y1)
    }

    /// Back-projects a tile's screen rectangle into a view-space sub-frustum
    /// with four apex-through side planes plus the supplied near/far depth
    /// planes.
    ///
    /// The near/far pair is clamped so `near >= GUARD_EPS` and `far` is strictly
    /// beyond `near`. Each side plane is built from two frustum corner rays via a
    /// cross product, then flipped (if needed) to point toward the tile center so
    /// "inside" always has a positive signed distance.
    #[must_use]
    pub fn tile_frustum(&self, tile_x: u32, tile_y: u32, near: f32, far: f32) -> TileFrustum {
        let (x0, y0, x1, y1) = self.tile_pixel_rect(tile_x, tile_y);
        let inv_w = 1.0 / f32::from(u16::try_from(self.width).unwrap_or(u16::MAX)).max(GUARD_EPS);
        let inv_h = 1.0 / f32::from(u16::try_from(self.height).unwrap_or(u16::MAX)).max(GUARD_EPS);
        // Normalized-device edges: X grows right, Y is flipped so it grows up.
        let px =
            |pixel: u32| 2.0 * (f32::from(u16::try_from(pixel).unwrap_or(u16::MAX)) * inv_w) - 1.0;
        let py =
            |pixel: u32| 1.0 - 2.0 * (f32::from(u16::try_from(pixel).unwrap_or(u16::MAX)) * inv_h);
        let nx0 = px(x0);
        let nx1 = px(x1);
        // Pixel Y grows downward, so the smaller pixel Y is the larger NDC Y.
        let ny_hi = py(y0);
        let ny_lo = py(y1);

        let ray =
            |ndc_x: f32, ndc_y: f32| Vec3::new(ndc_x * self.slope_x, ndc_y * self.slope_y, 1.0);
        let corner_ll = ray(nx0, ny_lo);
        let corner_lr = ray(nx1, ny_lo);
        let corner_ul = ray(nx0, ny_hi);
        let corner_ur = ray(nx1, ny_hi);
        let center_ray = ray((nx0 + nx1) * 0.5, (ny_lo + ny_hi) * 0.5);

        let side = |edge_a: Vec3, edge_b: Vec3| {
            let raw = edge_a.cross(edge_b);
            let oriented = if raw.dot(center_ray) < 0.0 {
                raw.scale(-1.0)
            } else {
                raw
            };
            Plane::from_inward_normal(oriented)
        };

        let sides = [
            side(corner_ll, corner_ul), // left edge (min NDC X)
            side(corner_lr, corner_ur), // right edge (max NDC X)
            side(corner_ll, corner_lr), // bottom edge (min NDC Y)
            side(corner_ul, corner_ur), // top edge (max NDC Y)
        ];

        let near = near.max(GUARD_EPS);
        let far = if far > near + GUARD_EPS {
            far
        } else {
            near + GUARD_EPS
        };
        TileFrustum { sides, near, far }
    }
}

// ---------------------------------------------------------------------------
// Tile sub-frustum and sphere intersection.
// ---------------------------------------------------------------------------

/// A single tile's view-space sub-frustum: four apex-through side planes and a
/// near/far depth pair. Sphere binning tests a light against all six half-spaces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileFrustum {
    sides: [Plane; 4],
    near: f32,
    far: f32,
}

impl TileFrustum {
    /// The four side planes (left, right, bottom, top), all inward-oriented.
    #[must_use]
    pub fn sides(&self) -> &[Plane; 4] {
        &self.sides
    }

    /// The near depth plane distance.
    #[must_use]
    pub fn near(&self) -> f32 {
        self.near
    }

    /// The far depth plane distance.
    #[must_use]
    pub fn far(&self) -> f32 {
        self.far
    }

    /// Conservative sphere/frustum test: the sphere survives when it is not
    /// entirely outside any of the four side planes or the near/far planes.
    ///
    /// This is the standard `Forward+` conservative test — a sphere straddling a
    /// frustum corner can be a false positive, which is the accepted, safe error
    /// direction (never a false *negative*).
    #[must_use]
    pub fn intersects_sphere(&self, light: SphereLight) -> bool {
        let radius = light.radius;
        let center = light.center;
        let mut i = 0usize;
        while i < self.sides.len() {
            if self.sides[i].signed_distance(center) < -radius {
                return false;
            }
            i += 1;
        }
        if center.z - self.near < -radius {
            return false;
        }
        if self.far - center.z < -radius {
            return false;
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Compact per-tile light list (`offset` + `count` + flat `indices`).
// ---------------------------------------------------------------------------

/// The packed result of a cull pass: for every tile a `(offset, count)` window
/// into one flat `indices` array, mirroring the `std430` layout a `GPU` tile
/// shader binds.
///
/// Tiles are stored row-major (`tile_y * tile_count_x + tile_x`). `offsets[t]` is
/// the start of tile `t`'s slice in `indices`, `counts[t]` its length, and the
/// offsets are exactly the exclusive prefix sum of the counts, so the whole
/// `indices` buffer is contiguous with no gaps.
#[derive(Clone, Debug, PartialEq)]
pub struct TileLightList {
    tile_count_x: u32,
    tile_count_y: u32,
    offsets: Vec<u32>,
    counts: Vec<u32>,
    indices: Vec<u32>,
}

impl TileLightList {
    /// An empty list sized for the given tile grid: every tile has count zero and
    /// an offset of zero, and the flat index array is empty.
    #[must_use]
    pub fn empty(tile_count_x: u32, tile_count_y: u32) -> Self {
        let tcx = tile_count_x.max(1);
        let tcy = tile_count_y.max(1);
        let total = (tcx as usize).saturating_mul(tcy as usize);
        let mut offsets = Vec::with_capacity(total);
        let mut counts = Vec::with_capacity(total);
        let mut i = 0usize;
        while i < total {
            offsets.push(0);
            counts.push(0);
            i += 1;
        }
        Self {
            tile_count_x: tcx,
            tile_count_y: tcy,
            offsets,
            counts,
            indices: Vec::new(),
        }
    }

    /// Number of tile columns.
    #[must_use]
    pub fn tile_count_x(&self) -> u32 {
        self.tile_count_x
    }

    /// Number of tile rows.
    #[must_use]
    pub fn tile_count_y(&self) -> u32 {
        self.tile_count_y
    }

    /// Total number of tiles.
    #[must_use]
    pub fn tile_count(&self) -> u32 {
        self.tile_count_x.saturating_mul(self.tile_count_y)
    }

    /// Total number of packed `(tile, light)` index entries across all tiles.
    #[must_use]
    pub fn total_indices(&self) -> u32 {
        u32::try_from(self.indices.len()).unwrap_or(u32::MAX)
    }

    /// The light-index slice for one tile, or an empty slice for an
    /// out-of-range tile coordinate.
    #[must_use]
    pub fn lights_for(&self, tile_x: u32, tile_y: u32) -> &[u32] {
        match self.linear(tile_x, tile_y) {
            Some(t) => {
                let start = self.offsets[t] as usize;
                let end = start + self.counts[t] as usize;
                &self.indices[start..end]
            }
            None => &[],
        }
    }

    /// The light count for one tile (zero for an out-of-range coordinate).
    #[must_use]
    pub fn count_at(&self, tile_x: u32, tile_y: u32) -> u32 {
        match self.linear(tile_x, tile_y) {
            Some(t) => self.counts[t],
            None => 0,
        }
    }

    /// The flat-index offset for one tile (zero for an out-of-range coordinate).
    #[must_use]
    pub fn offset_at(&self, tile_x: u32, tile_y: u32) -> u32 {
        match self.linear(tile_x, tile_y) {
            Some(t) => self.offsets[t],
            None => 0,
        }
    }

    /// `std430` byte size of the per-tile offset storage buffer.
    #[must_use]
    pub fn offsets_storage_bytes(&self) -> usize {
        storage_bytes(U32_STRIDE, self.offsets.len())
    }

    /// `std430` byte size of the per-tile count storage buffer.
    #[must_use]
    pub fn counts_storage_bytes(&self) -> usize {
        storage_bytes(U32_STRIDE, self.counts.len())
    }

    /// `std430` byte size of the flat light-index storage buffer.
    #[must_use]
    pub fn indices_storage_bytes(&self) -> usize {
        storage_bytes(U32_STRIDE, self.indices.len())
    }

    /// Row-major linear tile index, or `None` when the coordinate is off-grid.
    fn linear(&self, tile_x: u32, tile_y: u32) -> Option<usize> {
        if tile_x >= self.tile_count_x || tile_y >= self.tile_count_y {
            return None;
        }
        Some((tile_y as usize) * (self.tile_count_x as usize) + (tile_x as usize))
    }
}

// ---------------------------------------------------------------------------
// The cull pass.
// ---------------------------------------------------------------------------

/// Runs the `Forward+` tile-light cull: for every tile, binds the tile
/// sub-frustum (optionally tightened by a per-tile [`DepthRange`]) and appends
/// the index of every active light whose bounding sphere intersects it,
/// producing a compact [`TileLightList`].
///
/// `depth_bounds`, when supplied, must have one entry per tile in row-major
/// order; a shorter slice leaves the remaining tiles on the global `near`/`far`.
/// The output offsets are the exclusive prefix sum of the per-tile counts, so
/// the flat index buffer is gap-free.
#[must_use]
pub fn cull_tile_lights(
    grid: &TileGrid,
    lights: &[SphereLight],
    near: f32,
    far: f32,
    depth_bounds: Option<&[DepthRange]>,
) -> TileLightList {
    let tcx = grid.tile_count_x();
    let tcy = grid.tile_count_y();
    let tile_total = (tcx as usize).saturating_mul(tcy as usize);

    let mut offsets = Vec::with_capacity(tile_total);
    let mut counts = Vec::with_capacity(tile_total);
    let mut indices: Vec<u32> = Vec::new();
    let mut running: u32 = 0;

    let mut tile_y = 0u32;
    while tile_y < tcy {
        let mut tile_x = 0u32;
        while tile_x < tcx {
            let tile_index = (tile_y as usize) * (tcx as usize) + (tile_x as usize);
            let (tile_near, tile_far) = match depth_bounds {
                Some(bounds) if tile_index < bounds.len() => {
                    (bounds[tile_index].min, bounds[tile_index].max)
                }
                _ => (near, far),
            };
            let frustum = grid.tile_frustum(tile_x, tile_y, tile_near, tile_far);

            offsets.push(running);
            let mut count: u32 = 0;
            let mut li = 0usize;
            while li < lights.len() {
                let light = lights[li];
                if light.is_active() && frustum.intersects_sphere(light) {
                    indices.push(u32::try_from(li).unwrap_or(u32::MAX));
                    count = count.saturating_add(1);
                    running = running.saturating_add(1);
                }
                li += 1;
            }
            counts.push(count);

            tile_x += 1;
        }
        tile_y += 1;
    }

    TileLightList {
        tile_count_x: tcx,
        tile_count_y: tcy,
        offsets,
        counts,
        indices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Approximate-equality tolerance for the floating-point assertions; `f32`
    /// `==` is banned in this crate, so tests compare within this slack.
    const CMP_EPS: f32 = 1e-6;

    fn approx(lhs: f32, rhs: f32) -> bool {
        (lhs - rhs).abs() <= CMP_EPS
    }

    /// A 64x64 framebuffer split into 32-pixel tiles => a 2x2 tile grid with
    /// unit slopes (a 45-degree symmetric half field of view).
    fn grid_2x2() -> TileGrid {
        TileGrid::new(64, 64, 32, 1.0, 1.0)
    }

    #[test]
    fn tile_grid_div_ceil_partial_edges() {
        let grid = TileGrid::new(100, 64, 32, 1.0, 1.0);
        assert_eq!(grid.tile_count_x(), 4); // ceil(100 / 32)
        assert_eq!(grid.tile_count_y(), 2); // ceil(64 / 32)
        assert_eq!(grid.tile_count(), 8);
    }

    #[test]
    fn tile_grid_div_ceil_exact_multiple() {
        let grid = grid_2x2();
        assert_eq!(grid.tile_count_x(), 2);
        assert_eq!(grid.tile_count_y(), 2);
        assert_eq!(grid.tile_count(), 4);
    }

    #[test]
    fn degenerate_inputs_clamp_to_single_tile() {
        let grid = TileGrid::new(0, 0, 0, 0.0, 0.0);
        assert_eq!(grid.tile_count_x(), 1);
        assert_eq!(grid.tile_count_y(), 1);
        assert_eq!(grid.tile_count(), 1);
    }

    #[test]
    fn tile_pixel_rect_clamps_partial_edge() {
        let grid = TileGrid::new(100, 64, 32, 1.0, 1.0);
        // Last column tile (tile_x = 3) spans pixels [96, 100), clamped to width.
        let (x0, y0, x1, y1) = grid.tile_pixel_rect(3, 0);
        assert_eq!((x0, y0, x1, y1), (96, 0, 100, 32));
        // Interior tile is a full 32-pixel square.
        assert_eq!(grid.tile_pixel_rect(1, 1), (32, 32, 64, 64));
    }

    #[test]
    fn side_planes_point_inward() {
        let grid = grid_2x2();
        let frustum = grid.tile_frustum(0, 0, 0.1, 100.0);
        // A point on the tile's central ray must be inside every side plane.
        let inside = Vec3::new(-0.5 * 10.0, 0.5 * 10.0, 10.0);
        for plane in frustum.sides() {
            assert!(plane.signed_distance(inside) > 0.0);
        }
    }

    #[test]
    fn side_plane_normals_are_unit_length() {
        let grid = grid_2x2();
        let frustum = grid.tile_frustum(1, 1, 0.1, 100.0);
        for plane in frustum.sides() {
            assert!(approx(plane.normal().length(), 1.0));
        }
    }

    #[test]
    fn sphere_centered_in_tile_hits_only_that_tile() {
        let grid = grid_2x2();
        // NDC center (-0.5, 0.5) at depth 10 => view (-5, 5, 10): upper-left tile.
        let light = SphereLight::new(Vec3::new(-5.0, 5.0, 10.0), 1.0);
        let list = cull_tile_lights(&grid, &[light], 0.1, 100.0, None);
        assert_eq!(list.count_at(0, 0), 1);
        assert_eq!(list.lights_for(0, 0), &[0]);
        // The diagonally opposite tile (+x, -y) must not see it.
        assert_eq!(list.count_at(1, 1), 0);
    }

    #[test]
    fn sphere_far_to_the_side_is_culled() {
        let grid = grid_2x2();
        // Deep in +x while tile (0,0) covers the -x half => culled there.
        let light = SphereLight::new(Vec3::new(50.0, 0.0, 10.0), 1.0);
        let list = cull_tile_lights(&grid, &[light], 0.1, 100.0, None);
        assert_eq!(list.count_at(0, 0), 0);
    }

    #[test]
    fn sphere_radius_grazing_boundary_hits() {
        let grid = grid_2x2();
        // Tile (1,0) left plane sits at view x = 0 (unit normal +x). A center at
        // x = -0.5 is 0.5 outside; a radius of 0.6 grazes across => hit.
        let light = SphereLight::new(Vec3::new(-0.5, 5.0, 10.0), 0.6);
        let list = cull_tile_lights(&grid, &[light], 0.1, 100.0, None);
        assert_eq!(list.count_at(1, 0), 1);
    }

    #[test]
    fn sphere_radius_short_of_boundary_is_culled() {
        let grid = grid_2x2();
        // Same geometry but radius 0.4 < 0.5 gap => cannot reach tile (1,0).
        let light = SphereLight::new(Vec3::new(-0.5, 5.0, 10.0), 0.4);
        let list = cull_tile_lights(&grid, &[light], 0.1, 100.0, None);
        assert_eq!(list.count_at(1, 0), 0);
    }

    #[test]
    fn light_behind_near_plane_is_culled() {
        let grid = grid_2x2();
        // Depth well behind the near plane, radius too small to reach it.
        let light = SphereLight::new(Vec3::new(-5.0, 5.0, -5.0), 1.0);
        let list = cull_tile_lights(&grid, &[light], 0.1, 100.0, None);
        assert_eq!(list.count_at(0, 0), 0);
    }

    #[test]
    fn light_beyond_far_plane_is_culled() {
        let grid = grid_2x2();
        let light = SphereLight::new(Vec3::new(-5.0, 5.0, 200.0), 1.0);
        let list = cull_tile_lights(&grid, &[light], 0.1, 100.0, None);
        assert_eq!(list.count_at(0, 0), 0);
    }

    #[test]
    fn depth_range_tightening_culls_out_of_band_light() {
        let grid = grid_2x2();
        // Light at depth 50 is inside the global [0.1, 100] range...
        let light = SphereLight::new(Vec3::new(-5.0, 5.0, 50.0), 1.0);
        // ...but a per-tile band of [5, 15] for tile (0,0) excludes it.
        let bounds = [
            DepthRange::new(5.0, 15.0),
            DepthRange::new(0.1, 100.0),
            DepthRange::new(0.1, 100.0),
            DepthRange::new(0.1, 100.0),
        ];
        let list = cull_tile_lights(&grid, &[light], 0.1, 100.0, Some(&bounds));
        assert_eq!(list.count_at(0, 0), 0);
    }

    #[test]
    fn depth_range_tightening_keeps_in_band_light() {
        let grid = grid_2x2();
        let light = SphereLight::new(Vec3::new(-5.0, 5.0, 10.0), 1.0);
        let bounds = [
            DepthRange::new(5.0, 15.0),
            DepthRange::new(0.1, 100.0),
            DepthRange::new(0.1, 100.0),
            DepthRange::new(0.1, 100.0),
        ];
        let list = cull_tile_lights(&grid, &[light], 0.1, 100.0, Some(&bounds));
        assert_eq!(list.count_at(0, 0), 1);
    }

    #[test]
    fn offsets_are_exclusive_prefix_sum_of_counts() {
        let grid = grid_2x2();
        let lights = [
            SphereLight::new(Vec3::new(-5.0, 5.0, 10.0), 1.0), // tile (0,0)
            SphereLight::new(Vec3::new(5.0, 5.0, 10.0), 1.0),  // tile (1,0)
            SphereLight::new(Vec3::new(-5.0, -5.0, 10.0), 1.0), // tile (0,1)
            SphereLight::new(Vec3::new(5.0, -5.0, 10.0), 1.0), // tile (1,1)
        ];
        let list = cull_tile_lights(&grid, &lights, 0.1, 100.0, None);
        let mut running = 0u32;
        let mut ty = 0u32;
        while ty < list.tile_count_y() {
            let mut tx = 0u32;
            while tx < list.tile_count_x() {
                assert_eq!(list.offset_at(tx, ty), running);
                running += list.count_at(tx, ty);
                tx += 1;
            }
            ty += 1;
        }
        assert_eq!(running, list.total_indices());
    }

    #[test]
    fn lights_for_returns_the_indexed_light() {
        let grid = grid_2x2();
        let lights = [
            SphereLight::new(Vec3::new(50.0, 50.0, 10.0), 1.0), // index 0, far corner
            SphereLight::new(Vec3::new(-5.0, 5.0, 10.0), 1.0),  // index 1, tile (0,0)
        ];
        let list = cull_tile_lights(&grid, &lights, 0.1, 100.0, None);
        assert_eq!(list.lights_for(0, 0), &[1]);
    }

    #[test]
    fn multiple_lights_bin_into_multiple_tiles() {
        let grid = grid_2x2();
        let lights = [
            SphereLight::new(Vec3::new(-5.0, 5.0, 10.0), 1.0),
            SphereLight::new(Vec3::new(5.0, -5.0, 10.0), 1.0),
        ];
        let list = cull_tile_lights(&grid, &lights, 0.1, 100.0, None);
        assert_eq!(list.count_at(0, 0), 1);
        assert_eq!(list.count_at(1, 1), 1);
        // Sum of per-tile counts equals the flat index length.
        let mut total = 0u32;
        let mut ty = 0u32;
        while ty < list.tile_count_y() {
            let mut tx = 0u32;
            while tx < list.tile_count_x() {
                total += list.count_at(tx, ty);
                tx += 1;
            }
            ty += 1;
        }
        assert_eq!(total, list.total_indices());
    }

    #[test]
    fn empty_scene_produces_zero_counts() {
        let grid = grid_2x2();
        let list = cull_tile_lights(&grid, &[], 0.1, 100.0, None);
        assert_eq!(list.total_indices(), 0);
        assert_eq!(list.count_at(0, 0), 0);
        assert_eq!(list.count_at(1, 1), 0);
        assert!(list.lights_for(0, 0).is_empty());
    }

    #[test]
    fn inactive_light_is_skipped() {
        let grid = grid_2x2();
        let light = SphereLight::new(Vec3::new(-5.0, 5.0, 10.0), 0.0);
        assert!(!light.is_active());
        let list = cull_tile_lights(&grid, &[light], 0.1, 100.0, None);
        assert_eq!(list.total_indices(), 0);
    }

    #[test]
    fn empty_constructor_matches_grid_shape() {
        let list = TileLightList::empty(3, 2);
        assert_eq!(list.tile_count_x(), 3);
        assert_eq!(list.tile_count_y(), 2);
        assert_eq!(list.tile_count(), 6);
        assert_eq!(list.total_indices(), 0);
        assert_eq!(list.offset_at(2, 1), 0);
        assert_eq!(list.count_at(2, 1), 0);
    }

    #[test]
    fn std430_storage_bytes_track_element_counts() {
        let grid = grid_2x2();
        let light = SphereLight::new(Vec3::new(-5.0, 5.0, 10.0), 1.0);
        let list = cull_tile_lights(&grid, &[light], 0.1, 100.0, None);
        // 4 tiles * 4 bytes each for offsets and counts.
        assert_eq!(list.offsets_storage_bytes(), 16);
        assert_eq!(list.counts_storage_bytes(), 16);
        // One packed index => 4 bytes.
        assert_eq!(list.indices_storage_bytes(), 4);
    }

    #[test]
    fn empty_index_buffer_clamps_to_one_element() {
        let grid = grid_2x2();
        let list = cull_tile_lights(&grid, &[], 0.1, 100.0, None);
        // An empty `std430` storage binding still reserves a single element.
        assert_eq!(list.indices_storage_bytes(), U32_STRIDE);
    }

    #[test]
    fn degenerate_single_tile_sees_whole_screen_light() {
        // Framebuffer smaller than one tile => a 1x1 grid covering everything.
        let grid = TileGrid::new(16, 16, 64, 1.0, 1.0);
        assert_eq!(grid.tile_count(), 1);
        let light = SphereLight::new(Vec3::new(0.0, 0.0, 10.0), 1.0);
        let list = cull_tile_lights(&grid, &[light], 0.1, 100.0, None);
        assert_eq!(list.count_at(0, 0), 1);
        assert_eq!(list.lights_for(0, 0), &[0]);
    }

    #[test]
    fn vec3_algebra_is_consistent() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert!(approx(a.plus(b).x, 5.0));
        assert!(approx(a.minus(b).y, -3.0));
        assert!(approx(a.scale(2.0).z, 6.0));
        assert!(approx(a.dot(b), 32.0));
        // x-axis cross y-axis == z-axis.
        let cross = Vec3::new(1.0, 0.0, 0.0).cross(Vec3::new(0.0, 1.0, 0.0));
        assert!(approx(cross.z, 1.0));
        assert!(approx(Vec3::new(3.0, 4.0, 0.0).length(), 5.0));
        let unit = Vec3::new(0.0, 8.0, 0.0).normalize_or_zero();
        assert!(approx(unit.y, 1.0));
        assert_eq!(Vec3::ZERO.normalize_or_zero(), Vec3::ZERO);
    }

    #[test]
    fn depth_range_orders_its_edges() {
        let range = DepthRange::new(20.0, 5.0);
        assert!((range.min..=range.max).contains(&10.0));
        assert!(approx(range.min, 5.0));
        assert!(approx(range.max, 20.0));
    }

    #[test]
    fn tile_frustum_exposes_clamped_near_and_far() {
        let grid = grid_2x2();
        // Inverted near/far are clamped so far stays strictly beyond near.
        let frustum = grid.tile_frustum(0, 0, 10.0, 1.0);
        assert!(frustum.far() > frustum.near());
    }
}
