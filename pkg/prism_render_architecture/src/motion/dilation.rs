//! Neighborhood velocity dilation and tile-max reduction (temporal + blur).
//!
//! Two production techniques live here, both operating on the screen-space
//! velocity field produced by [`super::reproject`]:
//!
//! 1. **Closest-depth dilation** — for each pixel, the velocity of the
//!    *nearest* (smallest depth) neighbor inside a small radius is chosen.
//!    This is the classic `TAA` fix that keeps thin, fast-moving foreground
//!    silhouettes from tearing: the foreground's motion vector "bleeds" one or
//!    two pixels outward so the reprojection of edge pixels follows the
//!    occluder rather than the background it briefly covers.
//! 2. **Tile-max / neighbor-max** — `McGuire`-style reconstruction motion blur
//!    reduces the full-resolution velocity field to a coarse per-tile maximum
//!    (the largest-magnitude velocity in each tile), then takes the maximum
//!    over the 3x3 tile neighborhood. The blur pass samples this coarse field
//!    to bound how far it must gather, keeping the cost independent of the
//!    on-screen speed.
//!
//! Everything here is a deterministic `CPU` reference. The `GPU` compute
//! kernels that run these reductions per frame are out of scope and pending the
//! `GPU` backend; this module fixes the exact tie-breaking and edge-clamping
//! semantics those kernels must reproduce.

use alloc::vec::Vec;

use super::Vec2;

/// Returns whichever vector has the greater magnitude.
///
/// Ties (equal squared length) keep `current`, so reductions that fold with
/// this helper are order-stable: the first-seen maximum wins.
#[must_use]
fn keep_larger_magnitude(current: Vec2, candidate: Vec2) -> Vec2 {
    if candidate.length_squared() > current.length_squared() {
        candidate
    } else {
        current
    }
}

/// Which numeric direction is "closer to the camera" in a depth buffer.
///
/// Reversed-Z pipelines (the AAA default for precision) store the near plane at
/// the *larger* value, while a classic `[0, 1]` depth stores it at the smaller
/// value. Dilation must know which way "closest" points.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DepthOrder {
    /// Smaller stored depth is nearer (classic forward-Z, `0` = near).
    SmallerIsCloser,
    /// Larger stored depth is nearer (reversed-Z, `1` = near).
    LargerIsCloser,
}

impl DepthOrder {
    /// Returns `true` when `candidate` is strictly nearer the camera than
    /// `reference` under this ordering. Strictness makes ties keep the
    /// incumbent, which is what gives dilation its deterministic result.
    #[must_use]
    pub fn is_closer(self, candidate: f32, reference: f32) -> bool {
        match self {
            Self::SmallerIsCloser => candidate < reference,
            Self::LargerIsCloser => candidate > reference,
        }
    }
}

/// A dense, row-major screen-space velocity field in pixels.
///
/// Element `(x, y)` lives at linear index `y * width + x`, matching the pixel
/// convention used by [`super::reproject`] (origin top-left, `+x` right,
/// `+y` down). Velocities are the previous-to-current pixel displacement, i.e.
/// the vector a temporal consumer adds to a pixel to find where its surface was
/// last frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VelocityField {
    width: usize,
    height: usize,
    data: Vec<Vec2>,
}

impl VelocityField {
    /// Builds a zero-filled field of the given dimensions.
    #[must_use]
    pub fn zeroed(width: usize, height: usize) -> Self {
        let count = width * height;
        let mut data = Vec::with_capacity(count);
        for _ in 0..count {
            data.push(Vec2::ZERO);
        }
        Self {
            width,
            height,
            data,
        }
    }

    /// Wraps an existing row-major buffer, returning `None` when its length does
    /// not match `width * height`.
    #[must_use]
    pub fn from_pixels(width: usize, height: usize, data: Vec<Vec2>) -> Option<Self> {
        if data.len() == width * height {
            Some(Self {
                width,
                height,
                data,
            })
        } else {
            None
        }
    }

    /// Field width in pixels.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Field height in pixels.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Total pixel count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` when the field holds no pixels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Read-only view of the underlying row-major buffer.
    #[must_use]
    pub fn as_slice(&self) -> &[Vec2] {
        &self.data
    }

    fn linear_index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    /// Reads pixel `(x, y)`, or `None` when out of bounds.
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> Option<Vec2> {
        if x < self.width && y < self.height {
            Some(self.data[self.linear_index(x, y)])
        } else {
            None
        }
    }

    /// Writes pixel `(x, y)`, returning `false` when out of bounds.
    pub fn set(&mut self, x: usize, y: usize, value: Vec2) -> bool {
        if x < self.width && y < self.height {
            let index = self.linear_index(x, y);
            self.data[index] = value;
            true
        } else {
            false
        }
    }

    /// Samples with edge clamping; empty fields return [`Vec2::ZERO`].
    #[must_use]
    pub fn sample_clamped(&self, x: usize, y: usize) -> Vec2 {
        if self.is_empty() {
            return Vec2::ZERO;
        }
        let cx = x.min(self.width - 1);
        let cy = y.min(self.height - 1);
        self.data[self.linear_index(cx, cy)]
    }
}

/// A dense, row-major depth field paired with a [`VelocityField`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DepthField {
    width: usize,
    height: usize,
    data: Vec<f32>,
}

impl DepthField {
    /// Wraps an existing row-major depth buffer, returning `None` on a length
    /// mismatch against `width * height`.
    #[must_use]
    pub fn from_depths(width: usize, height: usize, data: Vec<f32>) -> Option<Self> {
        if data.len() == width * height {
            Some(Self {
                width,
                height,
                data,
            })
        } else {
            None
        }
    }

    /// Field width in pixels.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Field height in pixels.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns `true` when the field holds no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn linear_index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    /// Reads depth `(x, y)`, or `None` when out of bounds.
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> Option<f32> {
        if x < self.width && y < self.height {
            Some(self.data[self.linear_index(x, y)])
        } else {
            None
        }
    }
}

/// Dilates `velocity` by adopting, at each pixel, the velocity of the nearest
/// neighbor (by depth) inside a square radius.
///
/// The neighborhood is a clamped `(2 * radius + 1)` square. The center pixel
/// seeds the search, and a neighbor replaces it only when strictly closer under
/// `order`; equal depths therefore keep the incumbent, and because neighbors
/// are scanned in a fixed row-major order the result is fully deterministic.
///
/// Returns `None` when `velocity` and `depth` disagree on dimensions. A
/// `radius` of `0` copies the input (each pixel is its own only neighbor).
#[must_use]
pub fn dilate_closest_depth(
    velocity: &VelocityField,
    depth: &DepthField,
    radius: usize,
    order: DepthOrder,
) -> Option<VelocityField> {
    if velocity.width() != depth.width() || velocity.height() != depth.height() {
        return None;
    }

    let width = velocity.width();
    let height = velocity.height();
    let mut out = VelocityField::zeroed(width, height);

    for y in 0..height {
        for x in 0..width {
            // Seed with the center pixel so a fully tied neighborhood is a
            // no-op copy.
            let mut best_depth = depth.data[depth.linear_index(x, y)];
            let mut best_velocity = velocity.data[velocity.linear_index(x, y)];

            let x_lo = x.saturating_sub(radius);
            let x_hi = (x + radius).min(width - 1);
            let y_lo = y.saturating_sub(radius);
            let y_hi = (y + radius).min(height - 1);

            for ny in y_lo..=y_hi {
                for nx in x_lo..=x_hi {
                    let candidate_depth = depth.data[depth.linear_index(nx, ny)];
                    if order.is_closer(candidate_depth, best_depth) {
                        best_depth = candidate_depth;
                        best_velocity = velocity.data[velocity.linear_index(nx, ny)];
                    }
                }
            }

            out.set(x, y, best_velocity);
        }
    }

    Some(out)
}

/// A coarse, per-tile velocity field produced by [`tile_max`].
///
/// Each entry is the largest-magnitude velocity found in the corresponding
/// screen tile. The reconstruction blur pass samples this instead of the
/// full-resolution field to bound its gather radius.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TileVelocityField {
    tile_size: usize,
    tiles_x: usize,
    tiles_y: usize,
    data: Vec<Vec2>,
}

impl TileVelocityField {
    /// Tile edge length in pixels.
    #[must_use]
    pub fn tile_size(&self) -> usize {
        self.tile_size
    }

    /// Number of tile columns.
    #[must_use]
    pub fn tiles_x(&self) -> usize {
        self.tiles_x
    }

    /// Number of tile rows.
    #[must_use]
    pub fn tiles_y(&self) -> usize {
        self.tiles_y
    }

    /// Total tile count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` when there are no tiles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Read-only view of the row-major tile buffer.
    #[must_use]
    pub fn as_slice(&self) -> &[Vec2] {
        &self.data
    }

    fn linear_index(&self, tx: usize, ty: usize) -> usize {
        ty * self.tiles_x + tx
    }

    /// Reads tile `(tx, ty)`, or `None` when out of bounds.
    #[must_use]
    pub fn get(&self, tx: usize, ty: usize) -> Option<Vec2> {
        if tx < self.tiles_x && ty < self.tiles_y {
            Some(self.data[self.linear_index(tx, ty)])
        } else {
            None
        }
    }
}

/// Reduces a full-resolution velocity field to a per-tile maximum-magnitude
/// field (`McGuire`'s `TileMax`).
///
/// Tiles are `tile_size` pixels on a side; a screen dimension that is not a
/// multiple of `tile_size` produces a final partial tile that covers the
/// remainder. Returns `None` when `tile_size` is `0` or the field is empty.
#[must_use]
pub fn tile_max(velocity: &VelocityField, tile_size: usize) -> Option<TileVelocityField> {
    if tile_size == 0 || velocity.is_empty() {
        return None;
    }

    let width = velocity.width();
    let height = velocity.height();
    let tiles_x = width.div_ceil(tile_size);
    let tiles_y = height.div_ceil(tile_size);

    let mut data = Vec::with_capacity(tiles_x * tiles_y);
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let x_lo = tx * tile_size;
            let y_lo = ty * tile_size;
            let x_hi = (x_lo + tile_size).min(width);
            let y_hi = (y_lo + tile_size).min(height);

            let mut acc = Vec2::ZERO;
            for y in y_lo..y_hi {
                for x in x_lo..x_hi {
                    acc = keep_larger_magnitude(acc, velocity.data[velocity.linear_index(x, y)]);
                }
            }
            data.push(acc);
        }
    }

    Some(TileVelocityField {
        tile_size,
        tiles_x,
        tiles_y,
        data,
    })
}

/// Expands a tile-max field to its 3x3 neighborhood maximum (`McGuire`'s
/// `NeighborMax`).
///
/// Each output tile holds the largest-magnitude velocity among itself and its
/// (clamped) eight neighbors. This is the field a reconstruction blur samples:
/// a pixel can be reached by a fast mover up to one tile away, so bounding the
/// gather by the neighbor max avoids missing streaks that originate just off
/// the tile.
#[must_use]
pub fn neighbor_max(tiles: &TileVelocityField) -> TileVelocityField {
    let tiles_x = tiles.tiles_x();
    let tiles_y = tiles.tiles_y();

    let mut data = Vec::with_capacity(tiles_x * tiles_y);
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let tx_lo = tx.saturating_sub(1);
            let tx_hi = (tx + 1).min(tiles_x - 1);
            let ty_lo = ty.saturating_sub(1);
            let ty_hi = (ty + 1).min(tiles_y - 1);

            let mut acc = Vec2::ZERO;
            for ny in ty_lo..=ty_hi {
                for nx in tx_lo..=tx_hi {
                    acc = keep_larger_magnitude(acc, tiles.data[tiles.linear_index(nx, ny)]);
                }
            }
            data.push(acc);
        }
    }

    TileVelocityField {
        tile_size: tiles.tile_size(),
        tiles_x,
        tiles_y,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    fn approx(a: Vec2, b: Vec2) {
        assert!(
            (a.x - b.x).abs() <= EPS && (a.y - b.y).abs() <= EPS,
            "expected {b:?}, got {a:?}"
        );
    }

    fn field(width: usize, height: usize, values: &[(f32, f32)]) -> VelocityField {
        let data = values.iter().map(|&(x, y)| Vec2::new(x, y)).collect();
        VelocityField::from_pixels(width, height, data).expect("dimensions match")
    }

    #[test]
    fn from_pixels_rejects_length_mismatch() {
        let data = alloc::vec![Vec2::ZERO; 3];
        assert!(VelocityField::from_pixels(2, 2, data).is_none());
    }

    #[test]
    fn zeroed_field_is_all_zero() {
        let f = VelocityField::zeroed(2, 3);
        assert_eq!(f.width(), 2);
        assert_eq!(f.height(), 3);
        assert_eq!(f.len(), 6);
        assert!(!f.is_empty());
        for v in f.as_slice() {
            approx(*v, Vec2::ZERO);
        }
    }

    #[test]
    fn get_and_set_round_trip() {
        let mut f = VelocityField::zeroed(2, 2);
        assert!(f.set(1, 0, Vec2::new(3.0, -4.0)));
        approx(f.get(1, 0).expect("in bounds"), Vec2::new(3.0, -4.0));
        assert!(f.get(2, 0).is_none());
        assert!(!f.set(5, 5, Vec2::ZERO));
    }

    #[test]
    fn sample_clamped_clamps_to_edges() {
        let f = field(2, 2, &[(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]);
        approx(f.sample_clamped(9, 9), Vec2::new(4.0, 0.0));
        approx(f.sample_clamped(0, 0), Vec2::new(1.0, 0.0));
        assert_eq!(VelocityField::default().sample_clamped(0, 0), Vec2::ZERO);
    }

    #[test]
    fn depth_field_rejects_length_mismatch() {
        assert!(DepthField::from_depths(2, 2, alloc::vec![0.0; 3]).is_none());
        let d = DepthField::from_depths(1, 1, alloc::vec![0.5]).expect("matches");
        assert_eq!(d.width(), 1);
        assert_eq!(d.height(), 1);
        assert!(!d.is_empty());
        assert!(d.get(0, 0).is_some());
        assert!(d.get(1, 0).is_none());
    }

    #[test]
    fn depth_order_direction() {
        assert!(DepthOrder::SmallerIsCloser.is_closer(0.1, 0.9));
        assert!(!DepthOrder::SmallerIsCloser.is_closer(0.9, 0.1));
        assert!(DepthOrder::LargerIsCloser.is_closer(0.9, 0.1));
        // Ties are never "closer" (strict), preserving the incumbent.
        assert!(!DepthOrder::SmallerIsCloser.is_closer(0.5, 0.5));
        assert!(!DepthOrder::LargerIsCloser.is_closer(0.5, 0.5));
    }

    #[test]
    fn dilate_rejects_dimension_mismatch() {
        let vel = VelocityField::zeroed(2, 2);
        let depth = DepthField::from_depths(2, 1, alloc::vec![0.0, 0.0]).expect("matches");
        assert!(dilate_closest_depth(&vel, &depth, 1, DepthOrder::SmallerIsCloser).is_none());
    }

    #[test]
    fn dilate_radius_zero_is_identity() {
        let vel = field(2, 1, &[(5.0, 0.0), (0.0, 0.0)]);
        let depth = DepthField::from_depths(2, 1, alloc::vec![0.2, 0.8]).expect("matches");
        let out = dilate_closest_depth(&vel, &depth, 0, DepthOrder::SmallerIsCloser)
            .expect("dimensions match");
        approx(out.get(0, 0).expect("in bounds"), Vec2::new(5.0, 0.0));
        approx(out.get(1, 0).expect("in bounds"), Vec2::new(0.0, 0.0));
    }

    #[test]
    fn dilate_pulls_nearest_velocity_forward_z() {
        // Pixel 0 is the near foreground (depth 0.1) with a big velocity; pixel
        // 1 is far background. With radius 1 the background pixel adopts the
        // foreground velocity because the foreground is closer.
        let vel = field(2, 1, &[(9.0, 0.0), (0.0, 0.0)]);
        let depth = DepthField::from_depths(2, 1, alloc::vec![0.1, 0.9]).expect("matches");
        let out = dilate_closest_depth(&vel, &depth, 1, DepthOrder::SmallerIsCloser)
            .expect("dimensions match");
        approx(out.get(0, 0).expect("in bounds"), Vec2::new(9.0, 0.0));
        approx(out.get(1, 0).expect("in bounds"), Vec2::new(9.0, 0.0));
    }

    #[test]
    fn dilate_respects_reversed_z() {
        // Same layout, reversed-Z: now the *larger* depth (0.9) is the near
        // foreground, so its velocity is the one that spreads.
        let vel = field(2, 1, &[(0.0, 0.0), (7.0, 0.0)]);
        let depth = DepthField::from_depths(2, 1, alloc::vec![0.1, 0.9]).expect("matches");
        let out = dilate_closest_depth(&vel, &depth, 1, DepthOrder::LargerIsCloser)
            .expect("dimensions match");
        approx(out.get(0, 0).expect("in bounds"), Vec2::new(7.0, 0.0));
        approx(out.get(1, 0).expect("in bounds"), Vec2::new(7.0, 0.0));
    }

    #[test]
    fn tile_max_rejects_zero_tile_and_empty() {
        assert!(tile_max(&VelocityField::zeroed(4, 4), 0).is_none());
        assert!(tile_max(&VelocityField::default(), 2).is_none());
    }

    #[test]
    fn tile_max_picks_largest_magnitude() {
        // 2x2 field, one tile: the (0,3) vector has the largest magnitude.
        let vel = field(2, 2, &[(1.0, 0.0), (0.0, 2.0), (0.0, 3.0), (1.0, 1.0)]);
        let tiles = tile_max(&vel, 2).expect("non-empty");
        assert_eq!(tiles.tiles_x(), 1);
        assert_eq!(tiles.tiles_y(), 1);
        approx(tiles.get(0, 0).expect("in bounds"), Vec2::new(0.0, 3.0));
    }

    #[test]
    fn tile_max_handles_partial_edge_tiles() {
        // 3x1 field with tile_size 2 => two tiles: [x0,x1] and [x2].
        let vel = field(3, 1, &[(1.0, 0.0), (4.0, 0.0), (2.0, 0.0)]);
        let tiles = tile_max(&vel, 2).expect("non-empty");
        assert_eq!(tiles.tiles_x(), 2);
        assert_eq!(tiles.tiles_y(), 1);
        approx(tiles.get(0, 0).expect("in bounds"), Vec2::new(4.0, 0.0));
        approx(tiles.get(1, 0).expect("in bounds"), Vec2::new(2.0, 0.0));
    }

    #[test]
    fn neighbor_max_spreads_over_three_by_three() {
        // 3x1 tiles with a single fast tile in the middle: all three tiles end
        // up with the middle's velocity after neighbor-max.
        let vel = field(3, 1, &[(0.0, 0.0), (8.0, 0.0), (0.0, 0.0)]);
        let tiles = tile_max(&vel, 1).expect("non-empty");
        let expanded = neighbor_max(&tiles);
        assert_eq!(expanded.tiles_x(), 3);
        for tx in 0..3 {
            approx(expanded.get(tx, 0).expect("in bounds"), Vec2::new(8.0, 0.0));
        }
    }

    #[test]
    fn neighbor_max_is_local_beyond_radius() {
        // A fast tile at the far left must not reach a tile two away.
        let vel = field(4, 1, &[(6.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);
        let tiles = tile_max(&vel, 1).expect("non-empty");
        let expanded = neighbor_max(&tiles);
        approx(expanded.get(0, 0).expect("in bounds"), Vec2::new(6.0, 0.0));
        approx(expanded.get(1, 0).expect("in bounds"), Vec2::new(6.0, 0.0));
        approx(expanded.get(2, 0).expect("in bounds"), Vec2::ZERO);
        approx(expanded.get(3, 0).expect("in bounds"), Vec2::ZERO);
    }
}
