//! Screen-tile motion classification, `TAA` tile flags, and blur budget.
//!
//! The coarse per-tile velocity produced by [`super::dilation::tile_max`] /
//! [`super::dilation::neighbor_max`] drives two per-tile decisions that the
//! full-resolution passes read back:
//!
//! 1. **`TAA` tiling** — tiles are bucketed into static / slow / fast motion so
//!    the temporal resolve can take a cheap path where nothing moves, a
//!    standard path for gentle motion, and a conservative
//!    (dilated + neighborhood-clamped) path where fast motion risks ghosting.
//! 2. **Motion-blur budget** — the reconstruction blur derives a per-tile
//!    half-length (in pixels) from the tile velocity and the camera shutter
//!    fraction, clamped to a hard maximum so a runaway velocity cannot blow the
//!    sample budget.
//!
//! This is the deterministic `CPU` reference; the `GPU` tile-classification and
//! blur kernels are pending the `GPU` backend and must reproduce the thresholds
//! and clamps fixed here.

use alloc::vec::Vec;

use super::dilation::TileVelocityField;
use super::Vec2;

/// Motion bucket for a screen tile, ordered from least to most motion.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MotionTileClass {
    /// Sub-pixel motion; history can be reused directly.
    Static,
    /// Moderate motion; the standard temporal resolve applies.
    Slow,
    /// Large motion; dilation and neighborhood clamping are required.
    Fast,
}

/// Velocity-magnitude thresholds (in pixels) that separate the motion buckets.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileClassifierParams {
    /// At or below this magnitude a tile is [`MotionTileClass::Static`].
    pub static_max_pixels: f32,
    /// At or below this magnitude (and above `static_max_pixels`) a tile is
    /// [`MotionTileClass::Slow`]; above it the tile is
    /// [`MotionTileClass::Fast`].
    pub slow_max_pixels: f32,
}

impl Default for TileClassifierParams {
    fn default() -> Self {
        Self {
            static_max_pixels: 0.5,
            slow_max_pixels: 4.0,
        }
    }
}

impl TileClassifierParams {
    /// Builds sanitized thresholds: negatives / `NaN` become `0`, and the slow
    /// threshold is raised to at least the static one so the buckets never
    /// invert.
    #[must_use]
    pub fn new(static_max_pixels: f32, slow_max_pixels: f32) -> Self {
        let static_max_pixels = sanitize_non_negative(static_max_pixels);
        let slow_max_pixels = sanitize_non_negative(slow_max_pixels).max(static_max_pixels);
        Self {
            static_max_pixels,
            slow_max_pixels,
        }
    }
}

/// Clamps `NaN` and negatives to `0`, leaving other finite values untouched.
fn sanitize_non_negative(x: f32) -> f32 {
    if x.is_nan() || x < 0.0 {
        0.0
    } else {
        x
    }
}

/// Classifies a single tile velocity into a [`MotionTileClass`].
#[must_use]
pub fn classify_tile(velocity: Vec2, params: TileClassifierParams) -> MotionTileClass {
    let magnitude = velocity.length();
    if magnitude <= params.static_max_pixels {
        MotionTileClass::Static
    } else if magnitude <= params.slow_max_pixels {
        MotionTileClass::Slow
    } else {
        MotionTileClass::Fast
    }
}

/// Per-tile `TAA` resolve flags derived from a tile's motion class.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct TaaTileFlags(u32);

impl TaaTileFlags {
    /// No motion of note; history reuse is safe.
    pub const NONE: Self = Self(0);
    /// The tile is effectively static; the resolve may take the cheap path.
    pub const STATIC: Self = Self(1 << 0);
    /// Velocity dilation must run before resolving this tile.
    pub const NEEDS_DILATION: Self = Self(1 << 1);
    /// The neighborhood color clamp must run to suppress ghosting.
    pub const NEEDS_NEIGHBOR_CLAMP: Self = Self(1 << 2);
    /// The tile carries fast motion (drives blur and variable-rate history).
    pub const FAST_MOTION: Self = Self(1 << 3);

    /// Raw bit representation.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Returns `true` when every bit in `other` is set in `self`.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// Maps a motion class to the `TAA` tile flags it requires.
#[must_use]
pub fn taa_flags(class: MotionTileClass) -> TaaTileFlags {
    match class {
        MotionTileClass::Static => TaaTileFlags::STATIC,
        MotionTileClass::Slow => TaaTileFlags::NEEDS_NEIGHBOR_CLAMP,
        MotionTileClass::Fast => TaaTileFlags::FAST_MOTION
            .union(TaaTileFlags::NEEDS_DILATION)
            .union(TaaTileFlags::NEEDS_NEIGHBOR_CLAMP),
    }
}

/// Computes a motion-blur half-length (in pixels) for a tile velocity.
///
/// The per-frame displacement is scaled by the shutter fraction (the portion of
/// the frame interval the shutter is open, `[0, 1]`), halved to a symmetric
/// half-length, and clamped to `max_radius_pixels` so the blur sample budget
/// stays bounded. `NaN` / negative inputs sanitize to `0`.
#[must_use]
pub fn motion_blur_half_length(
    velocity: Vec2,
    shutter_fraction: f32,
    max_radius_pixels: f32,
) -> f32 {
    let shutter = sanitize_non_negative(shutter_fraction).min(1.0);
    let max_radius = sanitize_non_negative(max_radius_pixels);
    let half_length = velocity.length() * shutter * 0.5;
    half_length.min(max_radius)
}

/// A per-tile motion classification over a whole [`TileVelocityField`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TileClassification {
    tiles_x: usize,
    tiles_y: usize,
    classes: Vec<MotionTileClass>,
    static_count: usize,
    slow_count: usize,
    fast_count: usize,
}

impl TileClassification {
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
        self.classes.len()
    }

    /// Returns `true` when there are no tiles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.classes.is_empty()
    }

    /// Read-only view of the row-major class buffer.
    #[must_use]
    pub fn as_slice(&self) -> &[MotionTileClass] {
        &self.classes
    }

    /// Reads tile `(tx, ty)`, or `None` when out of bounds.
    #[must_use]
    pub fn get(&self, tx: usize, ty: usize) -> Option<MotionTileClass> {
        if tx < self.tiles_x && ty < self.tiles_y {
            Some(self.classes[ty * self.tiles_x + tx])
        } else {
            None
        }
    }

    /// Count of [`MotionTileClass::Static`] tiles.
    #[must_use]
    pub fn static_count(&self) -> usize {
        self.static_count
    }

    /// Count of [`MotionTileClass::Slow`] tiles.
    #[must_use]
    pub fn slow_count(&self) -> usize {
        self.slow_count
    }

    /// Count of [`MotionTileClass::Fast`] tiles.
    #[must_use]
    pub fn fast_count(&self) -> usize {
        self.fast_count
    }
}

/// Classifies every tile of a [`TileVelocityField`] and tallies the buckets.
#[must_use]
pub fn classify_field(
    tiles: &TileVelocityField,
    params: TileClassifierParams,
) -> TileClassification {
    let tiles_x = tiles.tiles_x();
    let tiles_y = tiles.tiles_y();

    let mut classes = Vec::with_capacity(tiles_x * tiles_y);
    let mut static_count = 0;
    let mut slow_count = 0;
    let mut fast_count = 0;

    for velocity in tiles.as_slice() {
        let class = classify_tile(*velocity, params);
        match class {
            MotionTileClass::Static => static_count += 1,
            MotionTileClass::Slow => slow_count += 1,
            MotionTileClass::Fast => fast_count += 1,
        }
        classes.push(class);
    }

    TileClassification {
        tiles_x,
        tiles_y,
        classes,
        static_count,
        slow_count,
        fast_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motion::dilation::{tile_max, VelocityField};

    const EPS: f32 = 1e-6;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() <= EPS, "expected {b}, got {a}");
    }

    #[test]
    fn params_default_thresholds() {
        let p = TileClassifierParams::default();
        approx(p.static_max_pixels, 0.5);
        approx(p.slow_max_pixels, 4.0);
    }

    #[test]
    fn params_new_prevents_inverted_buckets() {
        // Slow below static is raised up to static.
        let p = TileClassifierParams::new(3.0, 1.0);
        approx(p.static_max_pixels, 3.0);
        approx(p.slow_max_pixels, 3.0);
        // NaN / negatives collapse to zero.
        let q = TileClassifierParams::new(f32::NAN, -2.0);
        approx(q.static_max_pixels, 0.0);
        approx(q.slow_max_pixels, 0.0);
    }

    #[test]
    fn classify_tile_buckets() {
        let p = TileClassifierParams::default();
        assert_eq!(
            classify_tile(Vec2::new(0.2, 0.0), p),
            MotionTileClass::Static
        );
        assert_eq!(classify_tile(Vec2::new(2.0, 0.0), p), MotionTileClass::Slow);
        assert_eq!(classify_tile(Vec2::new(9.0, 0.0), p), MotionTileClass::Fast);
    }

    #[test]
    fn classify_tile_boundaries_are_inclusive_downward() {
        let p = TileClassifierParams::new(1.0, 2.0);
        // Exactly at the static bound => static.
        assert_eq!(
            classify_tile(Vec2::new(1.0, 0.0), p),
            MotionTileClass::Static
        );
        // Exactly at the slow bound => slow.
        assert_eq!(classify_tile(Vec2::new(2.0, 0.0), p), MotionTileClass::Slow);
        // Just above the slow bound => fast.
        assert_eq!(
            classify_tile(Vec2::new(2.0001, 0.0), p),
            MotionTileClass::Fast
        );
    }

    #[test]
    fn taa_flags_per_class() {
        assert!(taa_flags(MotionTileClass::Static).contains(TaaTileFlags::STATIC));
        let slow = taa_flags(MotionTileClass::Slow);
        assert!(slow.contains(TaaTileFlags::NEEDS_NEIGHBOR_CLAMP));
        assert!(!slow.contains(TaaTileFlags::NEEDS_DILATION));
        let fast = taa_flags(MotionTileClass::Fast);
        assert!(fast.contains(TaaTileFlags::FAST_MOTION));
        assert!(fast.contains(TaaTileFlags::NEEDS_DILATION));
        assert!(fast.contains(TaaTileFlags::NEEDS_NEIGHBOR_CLAMP));
    }

    #[test]
    fn taa_flags_none_and_bits() {
        assert_eq!(TaaTileFlags::NONE.bits(), 0);
        assert!(TaaTileFlags::NONE.contains(TaaTileFlags::NONE));
        assert_ne!(taa_flags(MotionTileClass::Fast).bits(), 0);
    }

    #[test]
    fn motion_blur_half_length_scales_and_clamps() {
        // 10px/frame, full shutter => 5px half-length.
        approx(
            motion_blur_half_length(Vec2::new(10.0, 0.0), 1.0, 100.0),
            5.0,
        );
        // Half shutter halves it again.
        approx(
            motion_blur_half_length(Vec2::new(10.0, 0.0), 0.5, 100.0),
            2.5,
        );
        // Clamped to the max radius.
        approx(
            motion_blur_half_length(Vec2::new(1000.0, 0.0), 1.0, 8.0),
            8.0,
        );
        // Sanitized inputs never go negative.
        approx(
            motion_blur_half_length(Vec2::new(10.0, 0.0), f32::NAN, -3.0),
            0.0,
        );
    }

    #[test]
    fn classify_field_tallies_buckets() {
        // 3x1 velocity field: one static, one slow, one fast under defaults.
        let data = alloc::vec![
            Vec2::new(0.1, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(20.0, 0.0),
        ];
        let vel = VelocityField::from_pixels(3, 1, data).expect("dims match");
        let tiles = tile_max(&vel, 1).expect("non-empty");
        let classification = classify_field(&tiles, TileClassifierParams::default());
        assert_eq!(classification.tiles_x(), 3);
        assert_eq!(classification.tiles_y(), 1);
        assert_eq!(classification.len(), 3);
        assert!(!classification.is_empty());
        assert_eq!(classification.static_count(), 1);
        assert_eq!(classification.slow_count(), 1);
        assert_eq!(classification.fast_count(), 1);
        assert_eq!(classification.get(0, 0), Some(MotionTileClass::Static));
        assert_eq!(classification.get(2, 0), Some(MotionTileClass::Fast));
        assert_eq!(classification.get(3, 0), None);
    }

    #[test]
    fn classify_field_empty_is_empty() {
        let vel = VelocityField::zeroed(0, 0);
        // tile_max returns None for empty; classify a real but zero field.
        let zero = VelocityField::zeroed(2, 2);
        let tiles = tile_max(&zero, 2).expect("non-empty");
        let classification = classify_field(&tiles, TileClassifierParams::default());
        assert_eq!(classification.static_count(), 1);
        assert!(vel.is_empty());
    }
}
